# Reinforcement Learning Project - CentraleSupelec (Mention IA)

Training and analysis of RL agents on the `highway-v0` environment from [highway-env](http://highway-env.farama.org/).

**Group 16**: Antoine Yezou, Ylias Larbi, Zacharie Boumard, Maxence Rossignol

## Overview

The project has two parts:
- **Core task**: Compare a from-scratch DQN implementation against Stable-Baselines3 DQN on a shared benchmark (`highway-v0`, Kinematics observations, DiscreteMetaAction).
- **Extension task**: Train a density-robust DQN agent using curriculum learning and domain randomization, with modified reward shaping (higher collision penalty, higher speed bonus).

## Project Structure

```
core/                       Core task implementation
  dqn_agent.py              DQN agent (replay buffer, epsilon-greedy, target network)
  model_architecture.py     MLP network (input -> 256 -> 256 -> n_actions)
  sb3_training.py           Stable-Baselines3 DQN training wrapper
  evaluation.py             Evaluation, comparison tables, failure analysis, CSV export
  density_wrapper.py        Gym wrapper for density curriculum/randomization
configs/
  shared_core_config.py     Instructor-provided environment config
  extension_config.py       Extension config (modified rewards, density=2.0)
extension/
  robustness_eval.py        Evaluate agent robustness across densities
  custom_env.py             Custom environment utilities
  advanced_algo.py          Advanced algorithm variants
  main_extension.py         Extension entry point
scripts/
  train.sh                  SLURM batch job for training on DCE
  setup_env.sh              SLURM job to create conda env on DCE
  plot_results.py           Regenerate plots from CSV (works with partial data)
  record_video.py           Generate videos from saved checkpoints
  test_density.py           Test crash rates at different densities
reports/                    Individual LaTeX reports
slides/                    Group presentation slides (LaTeX)
results_v3/                Core task results (matched 256x256 architectures)
main.py                     Core task pipeline: train -> evaluate -> compare
main_robust.py              Extension pipeline: robust DQN training + evaluation
```

## Setup

### Local

```bash
uv sync
uv run python main.py --seeds 1 --timesteps 1500 --eval-episodes 10  # quick test
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
sbatch scripts/train.sh                              # full run (3 seeds, 40k timesteps)
SEEDS=1 TIMESTEPS=5000 sbatch scripts/train.sh       # quick test

# 4. Monitor
squeue -u $USER
tail -f $(ls -t logs/slurm-training-*.out | head -1)
```

## Usage

```bash
# Core task (default: 3 seeds, 40k timesteps per agent, 50 eval episodes)
uv run python main.py
uv run python main.py --seeds 5 --timesteps 40000 --eval-episodes 50

# Extension: robust DQN (random/curriculum/mixed density training)
uv run python main_robust.py --mode mixed --timesteps 25000
uv run python main_robust.py --mode all                          # run all 3 modes

# Regenerate plots from CSV
uv run python scripts/plot_results.py results_v3

# Record videos from saved checkpoints
uv run python scripts/record_video.py results_v3/checkpoints/
```

## Core Task Results (v3 -- matched architectures)

Both agents use identical MLP architectures (256 -> 256) trained for 40,000 timesteps across 3 seeds (52, 53, 54).

| Agent | Mean Reward | Std | Crash Rate |
|-------|------------|-----|------------|
| DQN (ours) | 19.85 | 4.63 | 19% (29/150) |
| SB3 DQN | **20.48** | **0.05** | **0% (0/150)** |

SB3 learns a very stable, conservative policy with zero crashes. Our DQN reaches higher peak rewards (up to 24.85) but crashes in ~19% of episodes.

## Extension: Density Robustness

The extension task trains a DQN to handle varying traffic densities using three strategies:
- **Random**: uniform random density each episode
- **Curriculum**: linear progression from easy to hard density
- **Mixed**: curriculum for first 50%, then random for consolidation

Modified reward shaping: `collision_reward=-15`, `high_speed_reward=5.0`, `lane_change_reward=-0.2` to strongly penalize crashes and encourage speed.

Results are saved incrementally (per-seed) so partial runs are still usable.
