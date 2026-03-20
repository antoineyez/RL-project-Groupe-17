# Reinforcement Learning Project - CentraleSupélec (Mention IA)

Ce dépôt contient l'implémentation et l'analyse d'agents d'apprentissage par renforcement (RL) appliqués à l'environnement de conduite autonome `highway-env`.

## Présentation du Projet
L'objectif est de comparer une implémentation "maison" d'un algorithme **DQN** avec les performances d'un modèle entraîné via la bibliothèque **Stable-Baselines3** sur le benchmark `highway-v0`. Le projet se divise en deux parties : une tâche de base commune et une extension libre.

## Structure du Répertoire

```text
├── core/                         # Tâche de base (Core Task)
│   ├── dqn_agent.py              # Notre implémentation du DQN (from scratch)
│   ├── sb3_training.py           # Script d'entraînement via Stable-Baselines3
│   ├── model_architecture.py     # Définition des réseaux de neurones (MLP)
│   └── evaluation.py             # Scripts d'évaluation (moyenne/std sur 50 runs)
│
├── extension/                    # Travaux d'extension
│   ├── custom_env.py             # (Optionnel) Modifs de l'environnement
│   ├── advanced_algo.py          # (Optionnel) PPO, Rainbow, ou autre
│   └── analysis.ipynb            # Notebook d'analyse de l'extension
│
├── configs/                      # Gestion de la configuration
│   ├── shared_core_config.py     # Configuration fournie par l'instructeur
│   └── extension_config.py       # Configuration spécifique à l'extension
│
├── results/                      # Sorties et logs
│   ├── figures/                  # Courbes d'apprentissage (5 seeds) et plots
│   ├── checkpoints/              # Modèles sauvegardés (.zip ou .pt)
│   └── videos/                   # Enregistrements de l'agent en action
│
├── requirements.txt              # Dépendances du projet
└── README.md                     # Documentation du projet

Antoine Yezou, Ylias Larbi, Zacharie Boumard, Maxence Rossignol
