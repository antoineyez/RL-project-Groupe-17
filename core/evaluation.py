"""Evaluation: 50-run stats, multi-seed comparison, video recording, failure analysis."""

import csv
import os

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib.pyplot as plt
from tqdm import tqdm

from configs.shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


def make_eval_env(render_mode=None):
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    return env


def evaluate_agent(env, select_action_fn, n_episodes: int = 50):
    """Run n_episodes and return per-episode rewards."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = truncated = False
        while not (done or truncated):
            action = select_action_fn(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.array(rewards)


def evaluate_with_failure_analysis(env, select_action_fn, n_episodes: int = 50):
    """Evaluate and collect failure data (episodes ending in crash)."""
    rewards = []
    failures = []
    for ep in tqdm(range(n_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        total_reward = 0
        done = truncated = False
        step_count = 0
        while not (done or truncated):
            action = select_action_fn(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

        crashed = info.get("crashed", False) if info else False
        rewards.append(total_reward)
        if crashed:
            failures.append({
                "episode": ep,
                "reward": total_reward,
                "steps": step_count,
                "speed": info.get("speed", None),
            })
    return np.array(rewards), failures


def print_eval_stats(rewards: np.ndarray, label: str = "Agent"):
    print(f"\n--- {label} ({len(rewards)} episodes) ---")
    print(f"  Mean reward: {rewards.mean():.2f}  Std: {rewards.std():.2f}")
    print(f"  Min: {rewards.min():.2f}  Max: {rewards.max():.2f}")


def print_comparison_table(results: dict):
    """Print a formatted comparison table for all agents and seeds.

    results: {agent_name: {seed: np.array of rewards}}
    """
    print("\n" + "=" * 70)
    print(f"{'Agent':<25} {'Seed':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)
    for agent_name, seed_results in results.items():
        if not seed_results:  # Skip if the agent wasn't evaluated
            continue
        all_rewards = []
        for seed, rewards in sorted(seed_results.items()):
            print(f"{agent_name:<25} {seed:<8} {rewards.mean():>8.2f} {rewards.std():>8.2f} "
                  f"{rewards.min():>8.2f} {rewards.max():>8.2f}")
            all_rewards.append(rewards)
        combined = np.concatenate(all_rewards)
        print(f"{agent_name + ' (all seeds)':<25} {'--':<8} {combined.mean():>8.2f} "
              f"{combined.std():>8.2f} {combined.min():>8.2f} {combined.max():>8.2f}")
        print("-" * 70)
    print("=" * 70)


def print_failure_analysis(failures: list, label: str = "Agent"):
    """Print failure case analysis."""
    print(f"\n--- Failure analysis: {label} ---")
    if not failures:
        print("  No crashes detected.")
        return
    print(f"  Total crashes: {len(failures)}")
    for f in failures[:3]:
        print(f"  Episode {f['episode']}: reward={f['reward']:.2f}, "
              f"steps={f['steps']}, speed={f['speed']}")


def record_video(select_action_fn, save_dir: str = "results/videos", name_prefix: str = "rollout"):
    """Record one episode as video using gymnasium RecordVideo wrapper."""
    os.makedirs(save_dir, exist_ok=True)
    env = gym.make(SHARED_CORE_ENV_ID, render_mode="rgb_array")
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env = gym.wrappers.RecordVideo(env, save_dir, name_prefix=name_prefix,
                                   episode_trigger=lambda _: True)
    obs, _ = env.reset()
    done = truncated = False
    total_reward = 0
    while not (done or truncated):
        action = select_action_fn(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
    env.close()
    print(f"Video saved in {save_dir}/ (reward: {total_reward:.2f})")
    return total_reward


def plot_training_curves(results_dict: dict, save_path: str = "results/figures/training_curves.png"):
    """Plot smoothed training curves for multiple agents/seeds.

    results_dict: {label: [(timestep, reward), ...]}
    """
    plt.figure(figsize=(10, 6))
    for label, results in results_dict.items():
        timesteps = [t for t, _ in results]
        rewards = [r for _, r in results]
        window = min(10, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(timesteps[:len(smoothed)], smoothed, label=label)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward (rolling mean)")
    plt.title("Training curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved: {save_path}")


def save_training_rewards_csv(agent_name: str, seed: int, results: list,
                              path: str = "results/training_rewards.csv"):
    """Append per-episode training rewards to CSV.

    results: list of (timestep, reward) tuples.
    """
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["agent", "seed", "episode", "timestep", "reward"])
        for ep, (ts, r) in enumerate(results):
            writer.writerow([agent_name, seed, ep + 1, ts, f"{r:.4f}"])
    print(f"Training rewards saved: {path} ({agent_name}, seed={seed})")


def save_eval_results_csv(agent_name: str, seed: int, rewards: np.ndarray, failures: list,
                          path: str = "results/eval_results.csv"):
    """Append per-episode eval results to CSV. Creates file with header if needed."""
    write_header = not os.path.exists(path)
    failure_eps = {f["episode"]: f for f in failures}
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["agent", "seed", "episode", "reward", "crashed", "steps", "speed"])
        for ep, r in enumerate(rewards):
            fail = failure_eps.get(ep)
            crashed = fail is not None
            steps = fail["steps"] if fail else ""
            speed = f"{fail['speed']:.2f}" if fail and fail["speed"] else ""
            writer.writerow([agent_name, seed, ep + 1, f"{r:.4f}", crashed, steps, speed])
    print(f"Eval results saved: {path} ({agent_name}, seed={seed})")


def plot_training_curves_from_csv(csv_path: str = "results/training_rewards.csv",
                                  save_path: str = "results/figures/training_curves.png"):
    """Regenerate training curves from CSV (works even with partial data)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    for (agent, seed), group in df.groupby(["agent", "seed"]):
        group = group.sort_values("timestep")
        timesteps = group["timestep"].values
        rewards = group["reward"].values
        window = min(10, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(timesteps[:len(smoothed)], smoothed, label=f"{agent} seed={seed}")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward (rolling mean)")
    plt.title("Training curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved: {save_path}")
