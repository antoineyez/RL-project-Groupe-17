"""Generate videos from saved checkpoints.

Usage:
    # Record from a DQN checkpoint
    python scripts/record_video.py results/checkpoints/dqn_seed42.pt

    # Record from an SB3 checkpoint
    python scripts/record_video.py results/checkpoints/sb3_dqn_seed42.zip

    # Record multiple episodes
    python scripts/record_video.py results/checkpoints/dqn_seed42.pt --episodes 5

    # All checkpoints in a directory
    python scripts/record_video.py results/checkpoints/
"""

import argparse
import os
import sys

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.extension_config import EXTENSION_CONFIG, EXTENSION_ENV_ID
from core.model_architecture import DQNNetwork


def load_dqn_agent(checkpoint_path: str):
    """Load a DQN checkpoint and return an action selection function."""
    import torch

    env = gym.make(EXTENSION_ENV_ID)
    env.unwrapped.configure(EXTENSION_CONFIG)
    obs, _ = env.reset()
    obs_shape = obs.shape
    n_actions = env.action_space.n
    env.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DQNNetwork(obs_shape, n_actions).to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    net.eval()

    def select_action(obs):
        with torch.no_grad():
            state = torch.tensor(obs, device=device).unsqueeze(0)
            return net(state).argmax(dim=1).item()

    return select_action


def load_sb3_agent(checkpoint_path: str):
    """Load an SB3 checkpoint and return an action selection function."""
    from stable_baselines3 import DQN
    model = DQN.load(checkpoint_path)

    def select_action(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return select_action


def record(select_action_fn, save_dir: str, name_prefix: str, n_episodes: int = 1):
    os.makedirs(save_dir, exist_ok=True)
    env = gym.make(EXTENSION_ENV_ID, render_mode="rgb_array")
    env.unwrapped.configure(EXTENSION_CONFIG)
    
    # Accélère la vidéo (par défaut highway-env la met à 2 fps ce qui donne un effet de "lag")
    env.metadata["render_fps"] = 5
    
    env = gym.wrappers.RecordVideo(env, save_dir, name_prefix=name_prefix,
                                   episode_trigger=lambda _: True)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        total_reward = 0
        while not (done or truncated):
            action = select_action_fn(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        print(f"  Episode {ep+1}: reward = {total_reward:.2f}")

    env.close()
    print(f"Videos saved in {save_dir}/")


def process_checkpoint(path: str, n_episodes: int, save_dir: str):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"\nRecording from: {path}")

    if path.endswith(".pt"):
        select_action = load_dqn_agent(path)
    elif path.endswith(".zip"):
        select_action = load_sb3_agent(path)
    else:
        print(f"  Skipping unknown format: {path}")
        return

    record(select_action, save_dir, name_prefix=name, n_episodes=n_episodes)


def main():
    parser = argparse.ArgumentParser(description="Record videos from checkpoints")
    parser.add_argument("path", help="Checkpoint file (.pt or .zip) or directory of checkpoints")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes to record per checkpoint")
    parser.add_argument("--output", default="results/videos", help="Output directory")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        checkpoints = sorted([
            os.path.join(args.path, f)
            for f in os.listdir(args.path)
            if f.endswith(".pt") or f.endswith(".zip")
        ])
        if not checkpoints:
            print(f"No .pt or .zip files found in {args.path}")
            return
        print(f"Found {len(checkpoints)} checkpoints")
        for cp in checkpoints:
            process_checkpoint(cp, args.episodes, args.output)
    else:
        process_checkpoint(args.path, args.episodes, args.output)


if __name__ == "__main__":
    main()
