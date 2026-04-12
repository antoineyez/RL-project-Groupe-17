"""Double DQN — extension task implementation.

Hypothesis: vanilla DQN overestimates Q-values because it uses the same network
to both *select* the best next action and *evaluate* its value. Double DQN fixes
this by splitting the two roles across policy_net and target_net.

Vanilla DQN target:
    y = r + γ · max_a Q_target(s', a)          ← same net selects AND evaluates

Double DQN target:
    a* = argmax_a Q_policy(s', a)               ← policy_net selects
    y  = r + γ · Q_target(s', a*)              ← target_net evaluates

Everything else (replay buffer, epsilon-greedy, architecture, hyperparameters)
is identical to DQNAgent so the comparison is controlled.
"""

import torch
import torch.nn as nn

from core.dqn_agent import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """Double DQN agent.

    Identical to DQNAgent except for the target computation in train_step.
    Inherits replay buffer, epsilon-greedy exploration, network architecture,
    optimizer, checkpointing, and parallel action selection unchanged.
    """

    def train_step(self):
        """One gradient update using the Double DQN target.

        Difference from DQNAgent.train_step: next action is chosen by
        policy_net, but its Q-value is read from target_net. This decouples
        action selection from action evaluation and removes the upward bias
        introduced by taking the max over noisy estimates from a single network.
        """
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
            # Double DQN: policy_net picks the action, target_net scores it
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps_done += 1
        self.training_losses.append(loss.item())
        self.mean_q_values.append(next_q_values.mean().item())

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
