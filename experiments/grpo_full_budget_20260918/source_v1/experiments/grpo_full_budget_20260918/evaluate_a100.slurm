#!/bin/bash -l
#SBATCH --account=vas@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --no-requeue
set -euo pipefail
umask 027
module purge
module load arch/a100
module load miniforge/24.9.0
BUDGET_EVAL_RUNTIME=/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env
conda activate "${BUDGET_EVAL_RUNTIME}"
export PATH="${BUDGET_EVAL_RUNTIME}/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${BUDGET_EVAL_SOURCE:?}/src"
cd "${BUDGET_EVAL_SOURCE}"
exec "${BUDGET_EVAL_RUNTIME}/bin/python" -B experiments/grpo_full_budget_20260918/budget_evaluation.py run \
  --contract "${BUDGET_EVAL_CONTRACT:?}" --contract-sha "${BUDGET_EVAL_CONTRACT_SHA:?}" \
  --executed-slurm "${BASH_SOURCE[0]}"
