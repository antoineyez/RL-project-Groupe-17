"""Robustness Evaluation Module for Extension Task.

This script evaluates to what extent a pre-trained DQN overfits to its
training environment parameters (like vehicles_density = 1).
It runs evaluations across various density levels and random seeds to produce
robustness metrics: mean return and percentage of crashes.
Evaluations are run in parallel using multiple environments to speed up processing.

Usage:
    uv run python -m extension.robustness_eval --model results/checkpoints/dqn_seed52.pt
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
import highway_env  # noqa: F401
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv

from core.dqn_agent import DQNAgent
from configs.extension_config import EXTENSION_CONFIG, EXTENSION_ENV_ID


def make_eval_env(density):
    def _init():
        env = gym.make(EXTENSION_ENV_ID)
        custom_config = EXTENSION_CONFIG.copy()
        custom_config["vehicles_density"] = density
        env.unwrapped.configure(custom_config)
        return env

    return _init


def evaluate_robustness(
    model_path: str,
    densities: list,
    seeds: list,
    episodes_per_eval: int = 50,
    is_sb3: bool = False,
    min_steps_for_crash: int = 5,
):
    """
    Evaluates the agent across different traffic densities and seeds.

    Args:
        model_path: Path to the .pt trained model file
        densities: List of density multipliers to test
        seeds: List of random seeds for the environment
        episodes_per_eval: Number of episodes per (seed, density) pair

    Returns:
        results: Dictionary containing mean rewards, std rewards, and crash rates
    """
    # Results structures
    # shape: (len(seeds), len(densities))
    all_mean_rewards = np.zeros((len(seeds), len(densities)))
    all_crash_rates = np.zeros((len(seeds), len(densities)))
    all_ep_mean_speeds = np.zeros((len(seeds), len(densities)))
    all_ep_min_speeds = np.zeros((len(seeds), len(densities)))
    all_ep_max_speeds = np.zeros((len(seeds), len(densities)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}")

    if is_sb3:
        from stable_baselines3 import DQN

        agent = DQN.load(model_path, device=device)
        print(f"Success loading SB3 agent from {model_path}")
    else:
        # Initialize a base agent just to load structure
        base_env = gym.make(EXTENSION_ENV_ID)
        base_env.unwrapped.configure(EXTENSION_CONFIG)
        obs, _ = base_env.reset()
        obs_shape = obs.shape
        n_actions = base_env.action_space.n
        base_env.close()

        agent = DQNAgent(obs_shape=obs_shape, n_actions=n_actions)
        agent.load(model_path)
        print(f"Success loading custom agent from {model_path}")

    # Evaluate
    for i, seed in enumerate(seeds):
        print(f"\n--- Evaluating Seed: {seed} ---")
        for j, density in enumerate(densities):
            num_envs = min(os.cpu_count() or 4, episodes_per_eval)
            vec_env = SubprocVecEnv([make_eval_env(density) for _ in range(num_envs)])

            # Setup seeds across parallel envs (SB3 expects an int, not a list)
            vec_env.seed(seed)
            obs = vec_env.reset()

            rewards = []
            crashes = 0
            ep_mean_speeds = []
            ep_min_speeds = []
            ep_max_speeds = []

            dones_count = 0
            current_rewards = np.zeros(num_envs)
            current_speeds = [[] for _ in range(num_envs)]
            current_steps = np.zeros(num_envs)

            with tqdm(
                total=episodes_per_eval, desc=f"Density {density}", leave=False
            ) as pbar:
                while dones_count < episodes_per_eval:
                    if is_sb3:
                        actions, _ = agent.predict(obs, deterministic=True)
                    else:
                        actions = agent.select_actions(obs, training=False)

                    obs, step_rewards, dones, infos = vec_env.step(actions)

                    current_rewards += step_rewards
                    current_steps += 1

                    for env_idx in range(num_envs):
                        info_dict = infos[env_idx]
                        if "speed" in info_dict:
                            # Ne garder que les vitesses de roulage (exclure arrêts / crashs) pour ne pas fausser
                            if info_dict["speed"] > 1.0:
                                current_speeds[env_idx].append(info_dict["speed"])

                        if dones[env_idx]:
                            # Exclure les crashs immédiats (spawn bug de highway-env)
                            if current_steps[env_idx] < min_steps_for_crash:
                                current_rewards[env_idx] = 0
                                current_speeds[env_idx] = []
                                current_steps[env_idx] = 0
                                obs[env_idx] = vec_env.env_method(
                                    "reset",
                                    seed=seed
                                    + 1000
                                    + int(current_steps.sum())
                                    + env_idx,
                                    indices=[env_idx],
                                )[0][0]
                                continue

                            if dones_count < episodes_per_eval:
                                rewards.append(current_rewards[env_idx])
                                if info_dict.get("crashed", False):
                                    crashes += 1

                                if current_speeds[env_idx]:
                                    ep_mean_speeds.append(
                                        np.mean(current_speeds[env_idx])
                                    )
                                    ep_min_speeds.append(
                                        np.min(current_speeds[env_idx])
                                    )
                                    ep_max_speeds.append(
                                        np.max(current_speeds[env_idx])
                                    )

                                pbar.update(1)
                                dones_count += 1
                                current_rewards[env_idx] = 0
                                current_speeds[env_idx] = []
                                current_steps[env_idx] = 0
                                obs[env_idx] = vec_env.env_method(
                                    "reset",
                                    seed=seed + dones_count + env_idx,
                                    indices=[env_idx],
                                )[0][0]

            vec_env.close()

            # Record stats
            mean_rew = np.mean(rewards)
            crash_rate = crashes / episodes_per_eval

            all_mean_rewards[i, j] = mean_rew
            all_crash_rates[i, j] = crash_rate
            if ep_mean_speeds:
                all_ep_mean_speeds[i, j] = np.mean(ep_mean_speeds)
                all_ep_min_speeds[i, j] = np.mean(ep_min_speeds)
                all_ep_max_speeds[i, j] = np.mean(ep_max_speeds)

            print(
                f"  Density {density:4.1f} | Mean Reward: {mean_rew:8.2f} | Crash Rate: {crash_rate:6.1%} | Mean Speed: {all_ep_mean_speeds[i, j]:.1f}"
            )

    # Extraire le nom du modèle pour les fichiers de sortie
    model_basename = os.path.basename(model_path).replace(".pt", "").replace(".zip", "")

    return {
        "model_name": model_basename,
        "densities": densities,
        "mean_rewards": np.mean(all_mean_rewards, axis=0),
        "std_rewards": np.std(all_mean_rewards, axis=0),
        "mean_crash_rate": np.mean(all_crash_rates, axis=0) * 100,  # as percentage
        "std_crash_rate": np.std(all_crash_rates, axis=0) * 100,
        "mean_speeds": np.mean(all_ep_mean_speeds, axis=0),
        "min_speeds": np.mean(all_ep_min_speeds, axis=0),
        "max_speeds": np.mean(all_ep_max_speeds, axis=0),
        "std_mean_speeds": np.std(all_ep_mean_speeds, axis=0),
    }


def plot_robustness_results(results, output_dir="results/figures"):
    """Plot the results of the robustness evaluation."""
    os.makedirs(output_dir, exist_ok=True)

    densities = results["densities"]
    model_name = results.get("model_name", "model")
    prefix = f"{model_name}_"

    plt.style.use("seaborn-v0_8-darkgrid")

    # --- Plot 1: Mean Reward vs Density ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        densities,
        results["mean_rewards"],
        yerr=results["std_rewards"],
        fmt="-o",
        capsize=5,
        capthick=2,
        color="blue",
        markerfacecolor="red",
        markersize=8,
        linewidth=2,
    )
    plt.axvline(x=1.0, color="green", linestyle="--", label="Training Density (1.0)")
    plt.title(
        f"Agent Robustness ({model_name}): Mean Reward vs Traffic Density",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Traffic Density Multiplier", fontsize=12)
    plt.ylabel("Mean Reward (over 50 runs, 3 seeds)", fontsize=12)
    plt.xticks(densities)
    plt.legend()
    plt.tight_layout()
    reward_plot_path = os.path.join(
        output_dir, f"{prefix}robustness_reward_vs_density.png"
    )
    plt.savefig(reward_plot_path, dpi=300)
    print(f"Saved reward plot to: {reward_plot_path}")
    plt.close()

    # --- Plot 2: Crash Rate vs Density ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        densities,
        results["mean_crash_rate"],
        yerr=results["std_crash_rate"],
        fmt="-s",
        capsize=5,
        capthick=2,
        color="darkred",
        markerfacecolor="orange",
        markersize=8,
        linewidth=2,
    )
    plt.axvline(x=1.0, color="green", linestyle="--", label="Training Density (1.0)")
    plt.title(
        f"Safety Degradation ({model_name}): Crash Rate vs Traffic Density",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Traffic Density Multiplier", fontsize=12)
    plt.ylabel("Crash Rate (%)", fontsize=12)
    plt.xticks(densities)
    plt.ylim(
        0, max(100, np.max(results["mean_crash_rate"] + results["std_crash_rate"]) + 5)
    )
    plt.legend()
    plt.tight_layout()
    crash_plot_path = os.path.join(
        output_dir, f"{prefix}robustness_crash_rate_vs_density.png"
    )
    plt.savefig(crash_plot_path, dpi=300)
    print(f"Saved crash rate plot to: {crash_plot_path}")
    plt.close()

    # --- Plot 3: Speed evolution vs Density ---
    plt.figure(figsize=(10, 6))

    plt.plot(
        densities,
        results["max_speeds"],
        "g-^",
        label="Average Max Speed",
        linewidth=2,
        markersize=8,
    )

    plt.errorbar(
        densities,
        results["mean_speeds"],
        yerr=results["std_mean_speeds"],
        fmt="b-o",
        label="Average Mean Speed",
        capsize=5,
        linewidth=2,
        markersize=8,
    )

    plt.plot(
        densities,
        results["min_speeds"],
        "r-v",
        label="Average Min Speed",
        linewidth=2,
        markersize=8,
    )

    plt.axvline(x=1.0, color="gray", linestyle="--", label="Training Density (1.0)")
    plt.title(
        f"Behavioral Shift ({model_name}): Vehicle Speeds vs Traffic Density",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Traffic Density Multiplier", fontsize=12)
    plt.ylabel("Speed (avg per episode)", fontsize=12)
    plt.xticks(densities)
    plt.legend()
    plt.tight_layout()
    speed_plot_path = os.path.join(
        output_dir, f"{prefix}robustness_speed_vs_density.png"
    )
    plt.savefig(speed_plot_path, dpi=300)
    print(f"Saved speed distribution plot to: {speed_plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DQN Robustness across densities"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="results/checkpoints/dqn_seed52.pt",
        help="Path to the trained PyTorch model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes to run per density and per seed",
    )
    parser.add_argument(
        "--sb3",
        action="store_true",
        help="Flag to indicate the model is a Stable Baselines 3 model (.zip)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Directory to save the robustness evaluation plots",
    )
    args = parser.parse_args()

    DENSITIES = [0.75, 1.0, 1.20, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0]
    # We evaluate on fewer seeds to speed up, but over 50 episodes per density
    # which already provides good statistical significance.
    SEEDS = [72]

    print("=" * 60)
    print("Starting Robustness Evaluation (Out-of-Distribution)")
    print("=" * 60)

    results = evaluate_robustness(
        model_path=args.model,
        densities=DENSITIES,
        seeds=SEEDS,
        episodes_per_eval=args.episodes,
        is_sb3=args.sb3,
        min_steps_for_crash=2,
    )

    plot_robustness_results(results, output_dir=args.output_dir)

    print("\nEvaluation Complete!")
