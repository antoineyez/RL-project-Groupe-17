"""Training via Stable-Baselines3 DQN."""

import gymnasium as gym
import numpy as np
import highway_env  # noqa: F401
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from configs.shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


def make_env():
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    return env


class RewardLoggerCallback(BaseCallback):
    """Collects per-episode rewards during training."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


def train_sb3(
    total_timesteps: int = 50_000,
    seed: int = 42,
    save_path: str = "results/checkpoints/sb3_dqn",
):
    """Train a DQN agent via Stable-Baselines3. Returns (model, episode_rewards)."""
    env = DummyVecEnv([make_env])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=0,
        seed=seed,
    )

    callback = RewardLoggerCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(save_path)
    print(f"SB3 model saved: {save_path}")

    env.close()
    return model, callback.episode_rewards


if __name__ == "__main__":
    train_sb3()
