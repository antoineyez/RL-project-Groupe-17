"""Regenerate plots and stats from CSV files (works with partial data)."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves_per_seed(csv_path="results/training_rewards.csv",
                                  save_path="results/figures/training_curves_per_seed.png"):
    """Plot individual training curves for each agent/seed."""
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    for (agent, seed), group in df.groupby(["agent", "seed"]):
        group = group.sort_values("timestep")
        timesteps = group["timestep"].values
        rewards = group["reward"].values
        window = min(10, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(timesteps[:len(smoothed)], smoothed, label=f"{agent} seed={seed}", alpha=0.5)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward (rolling mean)")
    plt.title("Training curves (per seed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_curves(csv_path="results/training_rewards.csv",
                         save_path="results/figures/training_curves.png",
                         n_points=200, smooth_window=10):
    """Plot mean +/- std training curves, averaged across seeds.

    Interpolates each seed onto a common timestep grid, then computes
    mean and std across seeds for each agent.
    """
    df = pd.read_csv(csv_path)
    max_timestep = df["timestep"].max()
    common_ts = np.linspace(0, max_timestep, n_points)

    plt.figure(figsize=(10, 6))
    colors = {"DQN (ours)": "#1f77b4", "SB3 DQN": "#ff7f0e"}

    for agent, agent_df in df.groupby("agent"):
        seeds = agent_df["seed"].unique()
        interpolated = []

        for seed in seeds:
            seed_df = agent_df[agent_df["seed"] == seed].sort_values("timestep")
            ts = seed_df["timestep"].values
            rewards = seed_df["reward"].values

            # Smooth first, then interpolate
            window = min(smooth_window, len(rewards))
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            smoothed_ts = ts[:len(smoothed)]

            interp_rewards = np.interp(common_ts, smoothed_ts, smoothed)
            interpolated.append(interp_rewards)

        interpolated = np.array(interpolated)
        mean = interpolated.mean(axis=0)
        std = interpolated.std(axis=0)

        color = colors.get(agent, None)
        plt.plot(common_ts, mean, label=f"{agent} (n={len(seeds)} seeds)", color=color)
        plt.fill_between(common_ts, mean - std, mean + std, alpha=0.2, color=color)

    plt.xlabel("Timesteps")
    plt.ylabel("Reward (rolling mean)")
    plt.title("Training curves (mean $\\pm$ std across seeds)")
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
    import sys

    csv_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    training_csv = os.path.join(csv_dir, "training_rewards.csv")
    eval_csv = os.path.join(csv_dir, "eval_results.csv")
    fig_dir = os.path.join(csv_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if os.path.exists(training_csv):
        plot_training_curves(training_csv, os.path.join(fig_dir, "training_curves.png"))
        plot_training_curves_per_seed(training_csv, os.path.join(fig_dir, "training_curves_per_seed.png"))
    else:
        print(f"No {training_csv} found")

    if os.path.exists(eval_csv):
        print_eval_table(eval_csv)
    else:
        print(f"No {eval_csv} found")
