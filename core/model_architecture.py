"""Définition des réseaux de neurones (MLP) pour DQN."""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """Réseau MLP pour l'approximation de la fonction Q."""

    def __init__(self, obs_shape: tuple, n_actions: int, hidden_sizes: tuple = (256, 128)):
        super().__init__()
        input_size = 1
        for dim in obs_shape:
            input_size *= dim

        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, n_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.network(x)
