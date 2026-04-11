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
    """Buffer de replay pour stocker les transitions de l'agent.
    
    Utilise une queue à double entrée (deque) qui supprime automatiquement
    les plus anciennes transitions lorsque la capacité maximale est atteinte.
    """

    def __init__(self, capacity: int = 50_000):
        """Initialise le buffer.

        Args:
            capacity (int): Le nombre maximum de transitions à conserver en mémoire.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Ajoute une nouvelle transition (expérience) à la mémoire.

        Args:
            state (np.ndarray): L'état observé avant l'action.
            action (int): L'action choisie par l'agent.
            reward (float): La récompense obtenue suite à l'action.
            next_state (np.ndarray): Le nouvel état observé après l'action.
            done (bool): Indique si l'épisode s'est terminé après cette action.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Tire un échantillon aléatoire de transitions pour l'entraînement.

        Le tirage aléatoire permet de casser la corrélation temporelle entre
        les expériences consécutives, stabilisant ainsi l'apprentissage.

        Args:
            batch_size (int): Le nombre de transitions à tirer.

        Returns:
            tuple: Un tuple (states, actions, rewards, next_states, dones)
                   convertis en tableaux NumPy, prêts à être envoyés à PyTorch.
        """
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
        """Retourne le nombre actuel de transitions stockées."""
        return len(self.buffer)


class DQNAgent:
    """Agent DQN maison.
    
    Implémente l'algorithme Deep Q-Network avec un réseau principal (policy_net)
    et un réseau cible (target_net) pour stabiliser l'apprentissage.
    """

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
        """Initialise l'agent DQN.
        
        Args:
            obs_shape (tuple): La forme de l'espace d'observation (état).
            n_actions (int): Le nombre d'actions discrètes possibles.
            lr (float): Le taux d'apprentissage (learning rate) pour l'optimiseur Adam.
            gamma (float): Le discount factor de l'équation de Bellman (futur vs présent).
            epsilon_start (float): La valeur initiale de la probabilité d'exploration.
            epsilon_end (float): La valeur finale minimale de la probabilité d'exploration.
            epsilon_decay (int): Le nombre de pas sur lequel epsilon décroît linéairement.
            batch_size (int): La taille des batchs tirés du replay buffer pour l'entraînement.
            target_update_freq (int): La fréquence (en pas) de copie des poids vers le réseau cible.
            buffer_capacity (int): La taille maximale de la mémoire (replay buffer).
        """
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
        """Calcule la valeur actuelle d'epsilon (exploration) avec une décroissance linéaire.
        
        Returns:
            float: La probabilité actuelle de prendre une action aléatoire au lieu d'exploiter le réseau.
        """
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            max(0, 1 - self.steps_done / self.epsilon_decay)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Sélectionne une action en utilisant la politique d'exploration epsilon-greedy.
        Args:
            state (np.ndarray): L'état actuel de l'environnement.
            training (bool): Si True, s'autorise l'exploration aléatoire (epsilon).
                             Si False, agit toujours de manière gloutonne (la meilleure action connue).

        Returns:
            int: L'indice de l'action choisie.
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            state_t = torch.tensor(state, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def train_step(self):
        """

        Effectue une étape d'optimisation (rétropropagation) des poids du réseau :

        1. Tire un batch aléatoire de souvenirs.
        2. Calcule les valeurs Q prédites pour l'état actuel et l'action prise.
        3. Calcule la valeur Cible avec l'équation de Bellman via le Target Net.
        4. Fait une descente de gradient avec la perte (MSE).
        5. Met à jour périodiquement le réseau cible (Target Net).

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

    def select_actions(self, observations, training=True):
        """Version turbo pour traiter plusieurs environnements d'un coup de manière indépendante."""
        state = torch.FloatTensor(observations).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            actions = q_values.max(1)[1].cpu().numpy() # On récupère l'index de la meilleure action pour chaque ligne
            
        # Gestion de l'Epsilon-Greedy indépendante pour CHAQUE environnement du batch
        if training:
            # Tire un nombre aléatoire par environnement (pour ne pas explorer l'epsilon surtout ou rien)
            random_mask = np.random.rand(len(observations)) < self.epsilon
            if random_mask.any():
                # Remplace seulement ceux qui tombent sous l'epsilon par des actions aléatoires
                actions[random_mask] = np.random.randint(0, self.n_actions, size=random_mask.sum())
            
        return actions

def train_dqn(env, agent: DQNAgent, total_timesteps: int = 20_000, verbose: bool = True,
              checkpoint_path: str = None, checkpoint_every_steps: int = 2000):
    """Boucle d'entraînement principale du DQN interagissant avec l'environnement.
    
    L'agent effectue des pas dans l'environnement, sauvegarde ses expériences,
    et déclenche une phase d'entraînement à chaque action prise.

    Args:
        env (gym.Env): L'environnement avec lequel interagir (ici, HighwayEnv).
        agent (DQNAgent): L'agent contenant les réseaux de neurones (cerveau) et la politique.
        total_timesteps (int): Le nombre total d'actions/étapes d'entraînement.
        verbose (bool): Afficher ou non la barre de progression (tqdm).
        checkpoint_path (str): Le chemin de sauvegarde pour les états du réseau.
        checkpoint_every_steps (int): Sauvegarder le modèle tous les N pas de temps.

    Returns:
        list: Une liste de tuples (nombre de pas total effectué, récompense de l'épisode).
              Très utile pour tracer vos courbes d'apprentissage par la suite.
    """
    if verbose:
        print(f"\n--- Début de l'entraînement DQN sur l'appareil : {str(agent.device).upper()} ---")
        
    episode_results = []
    steps_done = 0
    last_checkpoint = 0
    pbar = tqdm(total=total_timesteps, desc="DQN Training", disable=not verbose)

    while steps_done < total_timesteps:
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
            steps_done += 1
            pbar.update(1)

            if steps_done >= total_timesteps:
                break

        episode_results.append((steps_done, total_reward))
        avg = np.mean([r for _, r in episode_results[-10:]])
        pbar.set_postfix(reward=f"{total_reward:.1f}", avg10=f"{avg:.1f}",
                         eps=f"{agent.epsilon:.3f}", ep=len(episode_results))

        if checkpoint_path and steps_done - last_checkpoint >= checkpoint_every_steps:
            agent.save(checkpoint_path)
            last_checkpoint = steps_done

    pbar.close()
    return episode_results


def train_dqn_parallel(vec_env, agent: DQNAgent, total_timesteps: int = 20_000, verbose: bool = True,
                       checkpoint_path: str = None, checkpoint_every_steps: int = 2000):
    """Boucle d'entraînement DQN TURBO utilisant les environnements vectorisés (VecEnv).
    
    Exploite la parallélisation CPU pour récolter l'expérience beaucoup plus vite, 
    et traite des batchs (GPU/CPU) via `select_actions`.
    """
    if verbose:
        print(f"\n--- Début de l'entraînement DQN PARALLÈLE sur l'appareil : {str(agent.device).upper()} ---")

    episode_results = []
    steps_done = 0
    last_checkpoint = 0
    num_envs = vec_env.num_envs
    
    pbar = tqdm(total=total_timesteps, desc="DQN Parallel Training", disable=not verbose)
    
    obs = vec_env.reset()
    
    while steps_done < total_timesteps:
        # 1. Prédiction des actions pour TOUS les environnements d'un coup (GPU batched)
        actions = agent.select_actions(obs, training=True)
        
        # 2. Exécution dans les environnements en parallèle (CPU multi-processing)
        next_obs, rewards, dones, infos = vec_env.step(actions)
        
        # 3. Stockage des N expériences dans le buffer
        for i in range(num_envs):
            # Traitement d'un détail de l'auto-reset des VecEnv: la VRAIE obs d'arrivée est dans info
            real_next_obs = infos[i]['terminal_observation'] if dones[i] and 'terminal_observation' in infos[i] else next_obs[i]
            
            agent.replay_buffer.push(obs[i], actions[i], rewards[i], real_next_obs, dones[i])
            
            # 4. Si une des voitures a fini sa partie, on la sauvegarde pour nos statistiques
            if dones[i] and "episode" in infos[i]:
                episode_results.append((steps_done, infos[i]["episode"]["r"]))
                
        # 5. On fait l'apprentissage: N pas d'environnement génèrent N expériences, 
        # on lance donc N train_steps pour équilibrer la vitesse.
        for _ in range(num_envs):
            agent.train_step()
            
        obs = next_obs
        steps_done += num_envs
        pbar.update(num_envs)
        
        # Mise à jour des stats dans la console
        if episode_results:
            avg = np.mean([r for _, r in episode_results[-10:]])
            pbar.set_postfix(avg10=f"{avg:.1f}", eps=f"{agent.epsilon:.3f}", ep=len(episode_results))

        if checkpoint_path and steps_done - last_checkpoint >= checkpoint_every_steps:
            agent.save(checkpoint_path)
            last_checkpoint = steps_done

    pbar.close()
    return episode_results


