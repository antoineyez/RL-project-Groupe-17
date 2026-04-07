"""Implémentation from-scratch de l'algorithme DQN."""

import random
from collections import deque

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from core.model_architecture import DQNNetwork


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class ReplayBuffer:
    """Buffer de replay pour stocker les transitions."""

    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Agent DQN maison."""

    def __init__(
        self,
        obs_shape: tuple,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        buffer_capacity: int = 50_000,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.policy_net = DQNNetwork(obs_shape, n_actions).to(self.device)
        self.target_net = DQNNetwork(obs_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.training_losses = []

    @property
    def epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            max(0, 1 - self.steps_done / self.epsilon_decay)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            state_t = torch.tensor(state, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.tensor(states, device=self.device).float()
        actions_t = torch.tensor(actions, device=self.device).long().unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device).float()
        dones_t = torch.tensor(dones, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps_done += 1
        self.training_losses.append(loss.item())

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, weights_only=True))
        self.target_net.load_state_dict(self.policy_net.state_dict())


def train_dqn(env, agent: DQNAgent, n_episodes: int = 200, verbose: bool = True):
    """Boucle d'entraînement DQN."""
    episode_rewards = []
    pbar = tqdm(range(n_episodes), desc="DQN Training", disable=not verbose)

    for episode in pbar:
        obs, info = env.reset()
        total_reward = 0
        done = truncated = False

        while not (done or truncated):
            action = agent.select_action(obs, training=True)
            next_obs, reward, done, truncated, info = env.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, done or truncated)
            agent.train_step()
            obs = next_obs
            total_reward += reward

        episode_rewards.append(total_reward)
        avg = np.mean(episode_rewards[-10:])
        pbar.set_postfix(reward=f"{total_reward:.1f}", avg10=f"{avg:.1f}", eps=f"{agent.epsilon:.3f}")

    return episode_rewards
