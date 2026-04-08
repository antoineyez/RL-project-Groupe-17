# Reinforcement Learning Project - CentraleSupelec (Mention IA)

Training and analysis of RL agents on the `highway-v0` environment from [highway-env](http://highway-env.farama.org/).

**Group 17**: Antoine Yezou, Ylias Larbi, Zacharie Boumard, Maxence Rossignol

## Overview

The project has two parts:
- **Core task**: Compare a from-scratch DQN implementation against Stable-Baselines3 DQN on a shared benchmark (`highway-v0`, Kinematics observations, DiscreteMetaAction).
- **Extension task**: TBD.

## Project Structure

```
core/                       Core task implementation
  dqn_agent.py              DQN agent (replay buffer, epsilon-greedy, target network)
  model_architecture.py     MLP network (input -> 256 -> 128 -> n_actions)
  sb3_training.py           Stable-Baselines3 DQN training wrapper
  evaluation.py             Evaluation, comparison tables, failure analysis, CSV export
configs/
  shared_core_config.py     Instructor-provided environment config
  extension_config.py       Extension-specific config
extension/                  Extension task (TBD)
scripts/
  train.sh                  SLURM batch job for training on DCE
  setup_env.sh              SLURM job to create conda env on DCE
  plot_results.py           Regenerate plots from CSV (works with partial data)
results/                    Output directory (gitignored)
  training_rewards.csv      Per-episode training rewards
  eval_results.csv          Per-episode evaluation results with crash data
  figures/                  Training curves and plots
  checkpoints/              Saved models (.pt, .zip)
  videos/                   Recorded rollouts
main.py                     Full pipeline: train -> evaluate -> compare -> record
```

## Setup

### Local

```bash
uv sync
uv run python main.py --seeds 1 --episodes 50 --sb3-timesteps 1500 --eval-episodes 10  # quick test
```

### DCE (CentraleSupelec cluster)

```bash
# 1. Get an interactive session
srun -p gpu_inter -t 01:00:00 --pty bash

# 2. Create conda env (first time only)
module load anaconda3/2022.10/gcc-13.1.0
export PYTHONNOUSERSITE=1
conda create --name rl-project python=3.12 --force
source activate rl-project
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install highway-env gymnasium stable-baselines3 numpy matplotlib tqdm pandas

# 3. Submit a batch job
mkdir -p logs
sbatch scripts/train.sh                    # full run (3 seeds, 200 episodes)
SEEDS=1 EPISODES=50 sbatch scripts/train.sh  # quick test

# 4. Monitor
squeue -u $USER
tail -f $(ls -t logs/slurm-training-*.out | head -1)
```

## Usage

```bash
# Full pipeline (default: 3 seeds, 200 episodes DQN, 6000 timesteps SB3, 50 eval episodes)
uv run python main.py

# Custom parameters
uv run python main.py --seeds 5 --episodes 300 --sb3-timesteps 9000 --eval-episodes 50

# Regenerate plots from CSV (useful if a run was interrupted)
uv run python scripts/plot_results.py

# Visual demo of trained agent
uv run python main.py --seeds 1 --episodes 200 --render
```

## Results

Results are saved incrementally (per-seed) so partial runs are still usable:
- `results/training_rewards.csv` and `results/eval_results.csv` for raw data
- `results/figures/training_curves.png` for training plots
- DQN checkpoints saved every 50 episodes during training
