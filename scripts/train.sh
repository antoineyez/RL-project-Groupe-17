#!/bin/bash

#SBATCH --job-name=rl-training
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-training-%j.out
#SBATCH --error=logs/slurm-training-%j.err

echo "Running on $(hostname)"
echo "Start time: $(date)"

module load anaconda3/2022.10/gcc-13.1.0
export PYTHONNOUSERSITE=1
source activate rl-project

mkdir -p logs results/checkpoints results/figures results/videos

# Default values, overridable via environment variables
SEEDS=${SEEDS:-3}
TIMESTEPS=${TIMESTEPS:-40000}
EVAL_EPISODES=${EVAL_EPISODES:-50}

echo "Config: seeds=$SEEDS, timesteps=$TIMESTEPS, eval_episodes=$EVAL_EPISODES"

python main.py \
    --seeds $SEEDS \
    --timesteps $TIMESTEPS \
    --eval-episodes $EVAL_EPISODES

echo "End time: $(date)"
