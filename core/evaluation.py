"""Evaluation: 50-run stats, multi-seed comparison, video recording, failure analysis."""

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


def plot_training_curves(rewards_dict: dict, save_path: str = "results/figures/training_curves.png"):
    """Plot smoothed training curves for multiple agents/seeds."""
    plt.figure(figsize=(10, 6))
    for label, rewards in rewards_dict.items():
        episodes = np.arange(1, len(rewards) + 1)
        window = min(10, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(episodes[:len(smoothed)], smoothed, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Reward (rolling mean)")
    plt.title("Training curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved: {save_path}")
