"""
Extension task: Train a robust DQN agent using Domain Randomization.

This script trains our custom DQN agent by varying the traffic density 
randomly at each episode. This forces the agent to generalize its behavior 
rather than overfitting to a specific traffic density (e.g., density=1.0).

After training, it directly pipes into `evaluate_robustness` to test and 
plot the agent's performance across different densities.
"""

import argparse
import os
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3.common.vec_env import SubprocVecEnv

from configs.extension_config import EXTENSION_CONFIG, EXTENSION_ENV_ID
from core.dqn_agent import DQNAgent, train_dqn_parallel, set_seed
from extension.robustness_eval import evaluate_robustness, plot_robustness_results


class CyclicEpsilonDQNAgent(DQNAgent):
    """
    DQN Agent with Cyclic Epsilon (Warm Restarts).
    Specifically designed for Curriculum Learning: resets exploration back 
    to a high level every time the difficulty (density) crosses a +0.5 threshold.
    """
    def __init__(self, *args, total_timesteps: int, min_density: float, max_density: float, **kwargs):
        super().__init__(*args, **kwargs)
        
        density_range = max(0.1, max_density - min_density)
        # Nombre de paliers de 0.5 de densité
        num_cycles = max(1, density_range / 0.5)
        
        # Longueur d'un "cycle" en nombre d'étapes d'entraînement
        self.cycle_length = max(1, int(total_timesteps / num_cycles))
        
        # On définit que l'exploration redescend sur 80% de chaque palier de difficulté
        self.cycle_decay_steps = self.cycle_length * 0.8

    @property
    def epsilon(self) -> float:
        # Calcule à quelle étape de son "palier" actuel l'agent se trouve
        step_in_cycle = self.steps_done % self.cycle_length
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            max(0, 1 - step_in_cycle / self.cycle_decay_steps)


class RandomDensityWrapper(gym.Wrapper):
    """
    Gym Wrapper for Domain Randomization.
    Randomly selects a new traffic density upon each reset.
    """
    def __init__(self, env, min_density=0.75, max_density=3.0):
        super().__init__(env)
        self.min_density = min_density
        self.max_density = max_density

    def reset(self, **kwargs):
        # Sample a random density for this new episode
        new_density = np.random.uniform(self.min_density, self.max_density)
        self.env.unwrapped.configure({"vehicles_density": new_density})
        return super().reset(**kwargs)


class CurriculumDensityWrapper(gym.Wrapper):
    """
    Gym Wrapper for Curriculum Learning.
    Progressively increases traffic density from min to max based on step count.
    """
    def __init__(self, env, min_density=0.5, max_density=3.0, total_local_steps=5000):
        super().__init__(env)
        self.min_density = min_density
        self.max_density = max_density
        self.total_local_steps = total_local_steps
        self.local_step_count = 0

    def step(self, action):
        self.local_step_count += 1
        return super().step(action)

    def reset(self, **kwargs):
        # Calculate curriculum progress (0.0 to 1.0)
        progress = min(1.0, self.local_step_count / max(1, self.total_local_steps))
        current_density = self.min_density + progress * (self.max_density - self.min_density)
        print(f"[DEBUG - Curriculum] Nouvelle densité : {current_density:.2f} (Progression : {progress:.1%})")
        self.env.unwrapped.configure({"vehicles_density": current_density})
        return super().reset(**kwargs)


class MixedDensityWrapper(gym.Wrapper):
    """
    Gym Wrapper blending Curriculum and Domain Randomization.
    Progresses linearly for the first 50% of steps, then randomly
    samples densities to prevent catastrophic forgetting.
    """
    def __init__(self, env, min_density=0.75, max_density=3.0, total_local_steps=5000):
        super().__init__(env)
        self.min_density = min_density
        self.max_density = max_density
        self.total_local_steps = total_local_steps
        self.local_step_count = 0

    def step(self, action):
        self.local_step_count += 1
        return super().step(action)

    def reset(self, **kwargs):
        # Calculate curriculum progress based on reaching max density at 50% of total steps
        plateau_start_step = max(1, self.total_local_steps / 2)
        progress = min(1.0, self.local_step_count / plateau_start_step)
        
        # S'il a dépassé la moitié, il bascule en Domain Randomization
        if progress >= 1.0:
            current_density = np.random.uniform(self.min_density, self.max_density)
        else:
            # Ligne droite (min_density vers max_density) pendant les 50% premiers
            current_density = self.min_density + progress * (self.max_density - self.min_density)
            
        print(f"[DEBUG - Mixed] Nouvelle densité : {current_density:.2f}")
        self.env.unwrapped.configure({"vehicles_density": current_density})
        return super().reset(**kwargs)


def make_robust_train_env(mode="random", min_density=0.5, max_density=3.0, total_local_steps=5000):
    def _init():
        env = gym.make(EXTENSION_ENV_ID)
        env.unwrapped.configure(EXTENSION_CONFIG)
        if mode == "random":
            env = RandomDensityWrapper(env, min_density, max_density)
        elif mode == "curriculum":
            env = CurriculumDensityWrapper(env, min_density, max_density, total_local_steps)
        elif mode == "mixed":
            env = MixedDensityWrapper(env, min_density, max_density, total_local_steps)
        return env
    return _init


def run_experiment(mode, args, obs_shape, n_actions, num_envs):
    print("\n" + "=" * 60)
    print(f" EXPERIMENT: {mode.upper()} DOMAIN RANDOMIZATION ")
    print("=" * 60)

    # Estimate steps per environment instance
    total_local_steps = args.timesteps // num_envs
    
    train_env = SubprocVecEnv([make_robust_train_env(mode, args.min_dens, args.max_dens, total_local_steps) for _ in range(num_envs)])

    checkpoint_path = f"results/checkpoints/dqn_{mode}_seed{args.seed}.pt"

    # Si on est en Curriculum ou Mixed, on utilise l'Epsilon Cyclique pour survivre aux hausses de difficulté
    if mode in ["curriculum", "mixed"]:
        agent = CyclicEpsilonDQNAgent(
            obs_shape=obs_shape,
            n_actions=n_actions,
            lr=1e-3,
            gamma=0.99,
            epsilon_decay=int(args.timesteps * 0.5), # Ignoré par le CyclicEpsilonDQNAgent
            batch_size=128,
            target_update_freq=200,
            total_timesteps=args.timesteps,
            min_density=args.min_dens,
            max_density=args.max_dens
        )
    else:
        agent = DQNAgent(
            obs_shape=obs_shape,
            n_actions=n_actions,
            lr=1e-3,
            gamma=0.99,
            epsilon_decay=int(args.timesteps * 0.2), # On pousse l'exploration très loin (20% du temps)
            batch_size=128,
            target_update_freq=200,
        )

    print("\n" + "=" * 60)
    print(f" 2. TRAINING ({args.timesteps} timesteps) ")
    print("=" * 60)
    
    episode_results = train_dqn_parallel(
        train_env, agent, total_timesteps=args.timesteps, verbose=True,
        checkpoint_path=checkpoint_path, checkpoint_every_steps=2000,
    )
    train_env.close()
    agent.save(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # --- Plot Training Rewards ---
    import matplotlib.pyplot as plt
    plot_dir = f"results/figures/robust_{mode}"
    os.makedirs(plot_dir, exist_ok=True)
    
    if episode_results:
        steps, rewards = zip(*episode_results)
        plt.figure(figsize=(10, 5))
        plt.plot(steps, rewards, alpha=0.3, color='blue', label='Recompense par épisode')
        
        # Moyenne mobile sur 50 épisodes pour lisser la courbe
        window = min(50, len(rewards))
        if window > 0:
            smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(steps[window-1:], smoothed_rewards, color='red', linewidth=2, label=f'Moyenne mobile ({window} épisodes)')
            
        plt.xlabel("Pas d'entraînement (Timesteps)")
        plt.ylabel("Somme des récompenses")
        plt.title(f"Courbe d'apprentissage DQN (Mode: {mode})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        
        training_plot_path = os.path.join(plot_dir, "training_rewards.png")
        plt.savefig(training_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training rewards plot saved to: {training_plot_path}")

    print("\n" + "=" * 60)
    print(" 3. EVALUATING ROBUSTNESS")
    print("=" * 60)
    
    # Use tightly meshed densities for a detailed evaluation curve
    eval_densities = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
    
    results = evaluate_robustness(
        model_path=checkpoint_path,
        densities=eval_densities,
        seeds=[args.seed],  # Evaluate on the training seed only to save time (50 eps already averages)
        episodes_per_eval=50
    )

    print("\n  SAVING PLOTS ")
    
    # Save plots in a dedicated folder so it doesn't overwrite the original model's plots
    plot_dir = f"results/figures/robust_{mode}"
    plot_robustness_results(results, output_dir=plot_dir)
    print(f"Checkout the plots in: {plot_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train a Robust DQN using Domain Randomization")
    parser.add_argument("--timesteps", type=int, default=25_000, help="Training timesteps")
    parser.add_argument("--seed", type=int, default=52, help="Random seed")
    parser.add_argument("--min-dens", type=float, default=0.75, help="Min density for randomization/curriculum")
    parser.add_argument("--max-dens", type=float, default=3.0, help="Max density for randomization/curriculum")
    parser.add_argument("--mode", type=str, default="mixed", choices=["random", "curriculum", "mixed", "all"], help="Which method to run")
    args = parser.parse_args()

    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    set_seed(args.seed)

    # Extract observation shape and actions count from a dummy environment
    temp_env = gym.make(EXTENSION_ENV_ID)
    temp_env.unwrapped.configure(EXTENSION_CONFIG)
    obs, _ = temp_env.reset()
    obs_shape = obs.shape
    n_actions = temp_env.action_space.n
    temp_env.close()

    print(f"Observation: {obs_shape} | Actions: {n_actions}")
    print(f"Bounds: [{args.min_dens}, {args.max_dens}] | Timesteps: {args.timesteps}")

    num_envs = os.cpu_count() or 4
    modes_to_run = ["random", "curriculum", "mixed"] if args.mode == "all" else [args.mode]
    
    for mode in modes_to_run:
        run_experiment(mode, args, obs_shape, n_actions, num_envs)

    print("\n" + "=" * 60)
    print("All robust experiments generated and evaluated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

