"""Extension task: Vanilla DQN vs Double DQN on highway-v0.

Hypothesis: vanilla DQN overestimates Q-values (max bias), leading to a less
accurate value function and potentially more crashes. Double DQN corrects this
by decoupling action selection from action evaluation.

Experimental design:
- Both agents share identical hyperparameters, architecture, and training budget.
- Trained with parallel environments (turbo mode) for speed.
- 3 seeds for statistical reliability.
- Measured: mean reward, crash rate, training curve, Q-value estimates over time.

Usage:
    python -m extension.main_extension                         # full run (3 seeds, 40k steps)
    python -m extension.main_extension --seeds 1 --timesteps 5000 --eval-episodes 10  # quick test
"""

import argparse
import os

import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from configs.shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from core.dqn_agent import DQNAgent, train_dqn_parallel, set_seed
from core.evaluation import (
    evaluate_with_failure_analysis,
    make_eval_env,
    plot_training_curves,
    print_comparison_table,
    print_failure_analysis,
    record_video,
    save_training_rewards_csv,
    save_eval_results_csv,
)
from extension.advanced_algo import DoubleDQNAgent


# ── Environment helpers ────────────────────────────────────────────────────────

def _make_single_env():
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset()
    return Monitor(env)


def make_vec_env(n_envs: int) -> SubprocVecEnv:
    return SubprocVecEnv([_make_single_env for _ in range(n_envs)])


# ── Q-value plot ───────────────────────────────────────────────────────────────

def plot_q_values(
    vanilla_q: list,
    double_q: list,
    save_path: str = "results/extension/figures/q_value_comparison.png",
):
    """Plot mean Q-value estimates over training steps for both agents.

    If vanilla DQN overestimates, its curve will sit consistently above
    Double DQN's curve. Both curves should trend upward as the agent learns,
    but vanilla's values will be inflated relative to the true optimum.
    """
    plt.figure(figsize=(10, 5))
    window = 200

    for label, data, color in [
        ("Vanilla DQN", vanilla_q, "tab:blue"),
        ("Double DQN", double_q, "tab:orange"),
    ]:
        if not data:
            continue
        smoothed = np.convolve(data, np.ones(window) / window, mode="valid")
        plt.plot(smoothed, label=label, color=color, alpha=0.85)

    plt.xlabel("Training step")
    plt.ylabel("Mean Q-value estimate")
    plt.title("Q-value overestimation: Vanilla DQN vs Double DQN")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Q-value comparison saved: {save_path}")


def plot_crash_rates(
    crash_data: dict,
    save_path: str = "results/extension/figures/crash_rates.png",
):
    """Bar chart of crash rates per agent per seed."""
    agents = list(crash_data.keys())
    seeds = sorted({s for rates in crash_data.values() for s in rates})

    x = np.arange(len(seeds))
    width = 0.35
    colors = ["tab:blue", "tab:orange"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (agent, rates) in enumerate(crash_data.items()):
        values = [rates.get(s, 0) * 100 for s in seeds]
        ax.bar(x + i * width, values, width, label=agent, color=colors[i], alpha=0.8)

    ax.set_xlabel("Seed")
    ax.set_ylabel("Crash rate (%)")
    ax.set_title("Crash rate: Vanilla DQN vs Double DQN")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Crash rate chart saved: {save_path}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extension: Vanilla DQN vs Double DQN")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=40_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Parallel envs for turbo training (default: CPU count)")
    args = parser.parse_args()

    n_envs = args.n_envs or os.cpu_count() or 4

    os.makedirs("results/extension/checkpoints", exist_ok=True)
    os.makedirs("results/extension/figures", exist_ok=True)
    os.makedirs("results/extension/videos", exist_ok=True)

    for csv_file in ["results/extension/training_rewards.csv", "results/extension/eval_results.csv"]:
        if os.path.exists(csv_file):
            os.remove(csv_file)

    # Probe env for shapes
    _probe = gym.make(SHARED_CORE_ENV_ID)
    _probe.unwrapped.configure(SHARED_CORE_CONFIG)
    obs, _ = _probe.reset()
    obs_shape = obs.shape
    n_actions = _probe.action_space.n
    _probe.close()

    seed_list = list(range(42, 42 + args.seeds))

    print(f"Extension: Vanilla DQN vs Double DQN")
    print(f"Environment: {SHARED_CORE_ENV_ID} | obs={obs_shape} | actions={n_actions}")
    print(f"Seeds: {seed_list} | Timesteps: {args.timesteps} | Parallel envs: {n_envs}")
    print(f"Evaluation: {args.eval_episodes} episodes per seed\n")

    agent_names = ["Vanilla DQN", "Double DQN"]
    all_training_curves = {}
    eval_results = {name: {} for name in agent_names}
    all_failures = {name: [] for name in agent_names}
    crash_rates = {name: {} for name in agent_names}
    all_q_values = {name: [] for name in agent_names}

    best_agents = {name: None for name in agent_names}

    shared_agent_kwargs = dict(
        obs_shape=obs_shape,
        n_actions=n_actions,
        lr=1e-3,
        gamma=0.99,
        epsilon_decay=int(args.timesteps * 0.5),
        batch_size=64,
        target_update_freq=200,
    )

    for seed in seed_list:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        for name, AgentClass in [("Vanilla DQN", DQNAgent), ("Double DQN", DoubleDQNAgent)]:
            print(f"\n[{name}] Training ({args.timesteps} timesteps, {n_envs} parallel envs)...")
            set_seed(seed)

            vec_env = make_vec_env(n_envs)
            checkpoint_path = f"results/extension/checkpoints/{name.lower().replace(' ', '_')}_seed{seed}.pt"

            agent = AgentClass(**shared_agent_kwargs)
            rewards = train_dqn_parallel(
                vec_env, agent,
                total_timesteps=args.timesteps,
                verbose=True,
                checkpoint_path=checkpoint_path,
                checkpoint_every_steps=5000,
            )
            vec_env.close()

            agent.save(checkpoint_path)
            all_training_curves[f"{name} seed={seed}"] = rewards
            all_q_values[name].extend(agent.mean_q_values)
            save_training_rewards_csv(
                name, seed, rewards,
                path="results/extension/training_rewards.csv",
            )

            print(f"[{name}] Evaluating ({args.eval_episodes} episodes)...")
            eval_env = make_eval_env()
            eval_rewards, failures = evaluate_with_failure_analysis(
                eval_env,
                lambda obs, a=agent: a.select_action(obs, training=False),
                n_episodes=args.eval_episodes,
            )
            eval_env.close()

            eval_results[name][seed] = eval_rewards
            all_failures[name].extend(failures)
            crash_rates[name][seed] = len(failures) / args.eval_episodes
            save_eval_results_csv(
                name, seed, eval_rewards, failures,
                path="results/extension/eval_results.csv",
            )

            if best_agents[name] is None or eval_rewards.mean() > max(
                r.mean() for r in eval_results[name].values()
            ) - 0.01:
                best_agents[name] = agent

    # ── Results ────────────────────────────────────────────────────────────────

    print_comparison_table(eval_results)

    for name, failures in all_failures.items():
        print_failure_analysis(failures, label=name)

    plot_training_curves(
        all_training_curves,
        save_path="results/extension/figures/training_curves.png",
    )

    plot_q_values(
        vanilla_q=all_q_values["Vanilla DQN"],
        double_q=all_q_values["Double DQN"],
    )

    plot_crash_rates(crash_rates)

    # ── Videos ────────────────────────────────────────────────────────────────
    print("\nRecording rollout videos...")
    for name, agent in best_agents.items():
        if agent:
            prefix = name.lower().replace(" ", "_")
            record_video(
                lambda obs, a=agent: a.select_action(obs, training=False),
                save_dir="results/extension/videos",
                name_prefix=prefix,
            )

    print("\nDone. Results in results/extension/")


if __name__ == "__main__":
    main()
