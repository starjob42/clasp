#!/bin/bash -l
#SBATCH --account=vas@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --no-requeue
set -euo pipefail
umask 027
module purge
module load arch/a100
module load miniforge/24.9.0
FB_RUNTIME=/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env
conda activate "${FB_RUNTIME}"
export PATH="${FB_RUNTIME}/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${FB_SOURCE:?}/src"
cd "${FB_SOURCE}"
exec "${FB_RUNTIME}/bin/python" -B experiments/grpo_full_budget_20260918/run_chunk.py --executed-slurm "${BASH_SOURCE[0]}"
