"""Définition des réseaux de neurones (MLP) pour DQN."""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """Réseau de neurones perceptron multicouche (MLP) pour l'approximation de la fonction Q (Deep Q-Network).
    
    Ce réseau prend en entrée l'état de l'environnement (aplati si nécessaire) et prédit
    la valeur Q (récompense future espérée) pour chaque action possible. L'agent choisira
    généralement l'action avec la plus haute valeur Q.
    """

    def __init__(self, obs_shape: tuple, n_actions: int, hidden_sizes: tuple = (256, 128)):
        """Initialise l'architecture du réseau de neurones.

        Args:
            obs_shape (tuple): La forme mathématique des observations en entrée (dans le cas du HighwayEnv l'observation est une matrice 10x5).
            n_actions (int): Le nombre d'actions discrètes disponibles pour l'agent en sortie (dans le cas du HighwayEnv, 5 actions).
            hidden_sizes (tuple): La taille des couches cachées successives du réseau (par défaut 256 puis 128 neurones).
        """
        super().__init__()
        # Calcul de la dimension totale d'entrée (aplatissement des dimensions si > 1D)
        input_size = 1
        for dim in obs_shape:
            input_size *= dim

        # Construction dynamique des couches (Linear -> ReLU)
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
            
        # Couche de sortie finale (sans activation pour laisser les valeurs Q libres d'être négatives/positives)
        layers.append(nn.Linear(prev_size, n_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagation avant (forward pass) dans le réseau.

        Transforme le tenseur d'entrée (état) en valeurs Q pour chaque action.
        Aplatit automatiquement les entrées si elles ont plus d'une dimension (ex: images ou matrices 2D/3D)
        tout en préservant la dimension de batch (dim=0).

        Args:
            x (torch.Tensor): Un batch d'états observés (taille: [batch_size, *obs_shape]).

        Returns:
            torch.Tensor: Les valeurs Q prédites pour chaque état du batch (taille: [batch_size, n_actions]).
        """
        x = x.float()
        # Si l'entrée n'est pas un simple vecteur 1D avec batch (ex: batch d'images ou grilles de features)
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.network(x)
