"""Training via Stable-Baselines3 DQN."""

import gymnasium as gym
import numpy as np
import highway_env  # noqa: F401
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from configs.shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID

def make_env():
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset()
    return Monitor(env)


class RewardLoggerCallback(BaseCallback):
    """Collects per-episode (timestep, reward) tuples during training."""

    def __init__(self, total_timesteps: int):
        super().__init__()
        self.episode_results = []
        self.pbar = tqdm(total=total_timesteps, desc="SB3 Training")

    def _on_step(self):
        self.pbar.update(1)
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                r = info["episode"]["r"]
                self.episode_results.append((self.num_timesteps, r))
                self.pbar.set_postfix(ep_reward=f"{r:.1f}", episodes=len(self.episode_results))
        return True

    def _on_training_end(self):
        self.pbar.close()


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
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1000,
        batch_size=128,
        gamma=0.99,
        target_update_interval=200,
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
        verbose=0,
        seed=seed,
    )

    callback = RewardLoggerCallback(total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(save_path)
    print(f"SB3 model saved: {save_path}")

    env.close()
    return model, callback.episode_results


if __name__ == "__main__":
    train_sb3()
