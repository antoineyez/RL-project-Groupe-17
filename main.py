"""
Core task pipeline: train DQN (ours) and SB3 DQN on highway-v0,
evaluate over multiple seeds with 50 runs each, compare results.

Usage:
    python main.py                          # Full pipeline (3 seeds, 200 episodes)
    python main.py --seeds 1 --episodes 50  # Quick run
    python main.py --render                 # + visual demo after training
"""

import argparse
import os

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

from configs.shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from core.dqn_agent import DQNAgent, train_dqn, set_seed
from core.sb3_training import train_sb3
from core.evaluation import (
    evaluate_with_failure_analysis,
    make_eval_env,
    plot_training_curves,
    print_comparison_table,
    print_failure_analysis,
    record_video,
)


def make_env(render_mode=None):
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    return env


def run_visual_demo(agent: DQNAgent, n_episodes: int = 3):
    env = make_env(render_mode="human")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        total_reward = 0
        while not (done or truncated):
            action = agent.select_action(obs, training=False)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            env.render()
        print(f"  Demo episode {ep + 1}: reward = {total_reward:.2f}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="RL Project - Core Task Pipeline")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes (DQN)")
    parser.add_argument("--sb3-timesteps", type=int, default=50_000, help="SB3 training timesteps")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Evaluation episodes per seed")
    parser.add_argument("--render", action="store_true", help="Visual demo after training")
    args = parser.parse_args()

    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/videos", exist_ok=True)

    seed_list = list(range(42, 42 + args.seeds))
    all_training_curves = {}
    eval_results = {"DQN (ours)": {}, "SB3 DQN": {}}
    all_failures = {"DQN (ours)": [], "SB3 DQN": []}

    best_dqn_agent = None
    best_sb3_model = None

    env = make_env()
    obs, _ = env.reset()
    obs_shape = obs.shape
    n_actions = env.action_space.n
    env.close()

    print(f"Environment: {SHARED_CORE_ENV_ID}")
    print(f"Observation: {obs_shape} | Actions: {n_actions}")
    print(f"Seeds: {seed_list} | DQN episodes: {args.episodes} | "
          f"SB3 timesteps: {args.sb3_timesteps}")
    print(f"Evaluation: {args.eval_episodes} episodes per seed\n")

    # ── Training and evaluation loop ──
    for seed in seed_list:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        # --- DQN (ours) ---
        print(f"\n[DQN] Training ({args.episodes} episodes)...")
        set_seed(seed)
        train_env = make_env()
        agent = DQNAgent(
            obs_shape=obs_shape,
            n_actions=n_actions,
            lr=1e-3,
            gamma=0.99,
            epsilon_decay=args.episodes * 15,
            batch_size=64,
            target_update_freq=200,
        )
        dqn_rewards = train_dqn(train_env, agent, n_episodes=args.episodes, verbose=True)
        train_env.close()

        agent.save(f"results/checkpoints/dqn_seed{seed}.pt")
        all_training_curves[f"DQN seed={seed}"] = dqn_rewards

        print(f"[DQN] Evaluating ({args.eval_episodes} episodes)...")
        eval_env = make_eval_env()
        dqn_eval, dqn_failures = evaluate_with_failure_analysis(
            eval_env, lambda obs, a=agent: a.select_action(obs, training=False),
            n_episodes=args.eval_episodes,
        )
        eval_env.close()
        eval_results["DQN (ours)"][seed] = dqn_eval
        all_failures["DQN (ours)"].extend(dqn_failures)

        if best_dqn_agent is None or dqn_eval.mean() > max(
            r.mean() for r in eval_results["DQN (ours)"].values()
        ) - 0.01:
            best_dqn_agent = agent

        # --- SB3 DQN ---
        print(f"\n[SB3] Training ({args.sb3_timesteps} timesteps)...")
        sb3_model, sb3_train_rewards = train_sb3(
            total_timesteps=args.sb3_timesteps,
            seed=seed,
            save_path=f"results/checkpoints/sb3_dqn_seed{seed}",
        )
        if sb3_train_rewards:
            all_training_curves[f"SB3 seed={seed}"] = sb3_train_rewards

        print(f"[SB3] Evaluating ({args.eval_episodes} episodes)...")
        eval_env = make_eval_env()
        sb3_eval, sb3_failures = evaluate_with_failure_analysis(
            eval_env, lambda obs, m=sb3_model: int(m.predict(obs, deterministic=True)[0]),
            n_episodes=args.eval_episodes,
        )
        eval_env.close()
        eval_results["SB3 DQN"][seed] = sb3_eval
        all_failures["SB3 DQN"].extend(sb3_failures)

        if best_sb3_model is None or sb3_eval.mean() > max(
            r.mean() for r in eval_results["SB3 DQN"].values()
        ) - 0.01:
            best_sb3_model = sb3_model

    # ── Comparison table ──
    print_comparison_table(eval_results)

    # ── Training curves ──
    plot_training_curves(all_training_curves)

    # ── Failure analysis ──
    for agent_name, failures in all_failures.items():
        print_failure_analysis(failures, label=agent_name)

    # ── Video recording ──
    print("\nRecording rollout videos...")
    if best_dqn_agent:
        record_video(
            lambda obs: best_dqn_agent.select_action(obs, training=False),
            name_prefix="dqn_ours",
        )
    if best_sb3_model:
        record_video(
            lambda obs: int(best_sb3_model.predict(obs, deterministic=True)[0]),
            name_prefix="sb3_dqn",
        )

    # ── Visual demo ──
    if args.render and best_dqn_agent:
        print("\nVisual demo of best DQN agent...")
        run_visual_demo(best_dqn_agent, n_episodes=3)

    print("\nDone.")


if __name__ == "__main__":
    main()
