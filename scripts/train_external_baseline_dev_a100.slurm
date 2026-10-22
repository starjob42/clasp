#!/bin/bash -l
# One declared raw-policy RLOO/external-baseline chunk; never submits or retries.
# Required EB_SOURCE, EB_SOURCE_SHA, EB_TASK, EB_ARM, EB_OUTPUT, EB_STOP.
# EB_DIAGNOSTIC=1 admits stops5/10; normal pilot admits50/100 at fixed horizon100.
# EB_RESUME, if supplied, names this arm's successful preceding checkpoint.
#SBATCH --account=vas@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --no-requeue
#SBATCH --job-name=grpo-external-dev
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
umask 027
module purge
module load arch/a100
module load miniforge/24.9.0
EB_RUNTIME=/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env
conda activate "${EB_RUNTIME}"
export PATH="${EB_RUNTIME}/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=43
unset CUDA_LAUNCH_BLOCKING
export PYTHONPATH="${EB_SOURCE:?EB_SOURCE is required}/src"
cd "${EB_SOURCE}"
exec "${EB_RUNTIME}/bin/python" -B scripts/run_external_baseline_dev_chunk.py \
    --executed-slurm "${BASH_SOURCE[0]}"
