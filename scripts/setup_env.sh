#!/bin/bash

#SBATCH --job-name=condaEnvSetup
#SBATCH --nodes=1
#SBATCH --partition=gpu_tp
#SBATCH --time=1:00:00

source /etc/profile
module load anaconda3/2022.10/gcc-13.1.0
export PYTHONNOUSERSITE=1

conda create --name rl-project python=3.12 --force
source activate rl-project

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install highway-env gymnasium stable-baselines3 numpy matplotlib tqdm
