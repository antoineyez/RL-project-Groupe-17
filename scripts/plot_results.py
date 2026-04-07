"""Regenerate plots and stats from CSV files (works with partial data)."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(csv_path="results/training_rewards.csv",
                         save_path="results/figures/training_curves.png"):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    for (agent, seed), group in df.groupby(["agent", "seed"]):
        rewards = group.sort_values("episode")["reward"].values
        window = min(10, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        episodes = np.arange(1, len(smoothed) + 1)
        plt.plot(episodes, smoothed, label=f"{agent} seed={seed}")
    plt.xlabel("Episode")
    plt.ylabel("Reward (rolling mean)")
    plt.title("Training curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def print_eval_table(csv_path="results/eval_results.csv"):
    df = pd.read_csv(csv_path)
    print(f"\n{'='*70}")
    print(f"{'Agent':<25} {'Seed':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Crashes':>8}")
    print("-" * 70)
    for agent, agent_df in df.groupby("agent"):
        for seed, group in agent_df.groupby("seed"):
            rewards = group["reward"].values
            crashes = group["crashed"].sum()
            total = len(group)
            print(f"{agent:<25} {seed:<8} {rewards.mean():>8.2f} {rewards.std():>8.2f} "
                  f"{rewards.min():>8.2f} {rewards.max():>8.2f} {crashes:>4}/{total}")
        all_rewards = agent_df["reward"].values
        all_crashes = agent_df["crashed"].sum()
        all_total = len(agent_df)
        print(f"{agent + ' (all)':<25} {'--':<8} {all_rewards.mean():>8.2f} {all_rewards.std():>8.2f} "
              f"{all_rewards.min():>8.2f} {all_rewards.max():>8.2f} {all_crashes:>4}/{all_total}")
        print("-" * 70)
    print("=" * 70)


if __name__ == "__main__":
    import os
    os.makedirs("results/figures", exist_ok=True)

    if os.path.exists("results/training_rewards.csv"):
        plot_training_curves()
    else:
        print("No training_rewards.csv found")

    if os.path.exists("results/eval_results.csv"):
        print_eval_table()
    else:
        print("No eval_results.csv found")
