"""
Core task pipeline: train DQN (ours) and SB3 DQN on highway-v0,
evaluate over multiple seeds with 50 runs each, compare results.

Usage:
    python main.py                                  # Full pipeline (3 seeds, 20k timesteps)
    python main.py --timesteps 5000 --seeds 1       # Quick run
    python main.py --render                         # + visual demo after training
"""

import argparse
import os

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

from configs.shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from core.dqn_agent import DQNAgent, train_dqn_parallel, set_seed
from core.sb3_training_turbo import train_sb3, make_env as sb3_make_env
from stable_baselines3.common.vec_env import SubprocVecEnv
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
    parser.add_argument("--timesteps", type=int, default=20_000, help="Training timesteps (both DQN and SB3)")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Evaluation episodes per seed")
    parser.add_argument("--render", action="store_true", help="Visual demo after training")
    parser.add_argument("--algo", type=str, default="both", choices=["dqn", "sb3", "both"], help="Algorithm to run")
    parser.add_argument("--exp-suffix", type=str, default="", help="Suffix to append to CSV files")
    args = parser.parse_args()

    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/videos", exist_ok=True)

    csv_suffix = f"_{args.exp_suffix}" if args.exp_suffix else ""
    train_csv_path = f"results/training_rewards{csv_suffix}.csv"
    eval_csv_path = f"results/eval_results{csv_suffix}.csv"

    if args.algo == "both" and not args.exp_suffix:
        for csv_file in [train_csv_path, eval_csv_path]:
            if os.path.exists(csv_file):
                os.remove(csv_file)

    seed_list = list(range(52, 52 + args.seeds))
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
    print(f"Timesteps: {args.timesteps}")
    print(f"Evaluation: {args.eval_episodes} episodes per seed\n")

    # ── Training and evaluation loop ──
    train_seed = seed_list[0]
    eval_seeds = [train_seed + 1, train_seed + 2, train_seed + 3]

    print(f"\n{'='*60}")
    print(f"  TRAINING ON SEED {train_seed}")
    print(f"  EVALUATING ON SEEDS {eval_seeds}")
    print(f"{'='*60}")

    # --- DQN (ours) ---
    if args.algo in ["both", "dqn"]:
        print(f"\n[DQN PARALLEL] Training ({args.timesteps} timesteps)...")
        set_seed(train_seed)
        num_envs = os.cpu_count() or 4
        # Création de multiples environnements pour DQNAgent
        train_env = SubprocVecEnv([sb3_make_env for _ in range(num_envs)])
        
        checkpoint_path = f"results/checkpoints/dqn_seed{train_seed}.pt"
        agent = DQNAgent(
            obs_shape=obs_shape,
            n_actions=n_actions,
            lr=1e-3,
            gamma=0.99,
            epsilon_decay=int(args.timesteps * 0.5),
            batch_size=128,
            target_update_freq=200,
        )
        
        dqn_rewards = train_dqn_parallel(
            train_env, agent, total_timesteps=args.timesteps, verbose=True,
            checkpoint_path=checkpoint_path, checkpoint_every_steps=2000,
        )
        train_env.close()

        agent.save(checkpoint_path)
        all_training_curves[f"DQN seed={train_seed}"] = dqn_rewards
        save_training_rewards_csv("DQN (ours)", train_seed, dqn_rewards, path=train_csv_path)

        for eval_seed in eval_seeds:
            print(f"[DQN] Evaluating on eval seed {eval_seed} ({args.eval_episodes} episodes)...")
            set_seed(eval_seed)
            eval_env = make_eval_env()
            dqn_eval, dqn_failures = evaluate_with_failure_analysis(
                eval_env, lambda obs, a=agent: a.select_action(obs, training=False),
                n_episodes=args.eval_episodes,
            )
            eval_env.close()
            eval_results["DQN (ours)"][eval_seed] = dqn_eval
            all_failures["DQN (ours)"].extend(dqn_failures)
            save_eval_results_csv("DQN (ours)", eval_seed, dqn_eval, dqn_failures, path=eval_csv_path)

            if best_dqn_agent is None or dqn_eval.mean() > max(
                r.mean() for r in (eval_results["DQN (ours)"].values() or [np.array([0])])
            ) - 0.01:
                best_dqn_agent = agent

    # --- SB3 DQN ---
    if args.algo in ["both", "sb3"]:
        print(f"\n[SB3] Training ({args.timesteps} timesteps)...")
        set_seed(train_seed)
        sb3_model, sb3_train_rewards = train_sb3(
            total_timesteps=args.timesteps,
            seed=train_seed,
            save_path=f"results/checkpoints/sb3_dqn_seed{train_seed}",
        )
        if sb3_train_rewards:
            all_training_curves[f"SB3 seed={train_seed}"] = sb3_train_rewards
            save_training_rewards_csv("SB3 DQN", train_seed, sb3_train_rewards, path=train_csv_path)

        for eval_seed in eval_seeds:
            print(f"[SB3] Evaluating on eval seed {eval_seed} ({args.eval_episodes} episodes)...")
            set_seed(eval_seed)
            eval_env = make_eval_env()
            sb3_eval, sb3_failures = evaluate_with_failure_analysis(
                eval_env, lambda obs, m=sb3_model: int(m.predict(obs, deterministic=True)[0]),
                n_episodes=args.eval_episodes,
            )
            eval_env.close()
            eval_results["SB3 DQN"][eval_seed] = sb3_eval
            all_failures["SB3 DQN"].extend(sb3_failures)
            save_eval_results_csv("SB3 DQN", eval_seed, sb3_eval, sb3_failures, path=eval_csv_path)

            if best_sb3_model is None or sb3_eval.mean() > max(
                r.mean() for r in (eval_results["SB3 DQN"].values() or [np.array([0])])
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
