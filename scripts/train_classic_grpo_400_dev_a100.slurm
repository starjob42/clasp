#!/bin/bash -l
# Fixed 400-update classic-GRPO control; each invocation runs one bounded chunk.
# CG_TASK, CG_OUTPUT, CG_SOURCE, CG_SOURCE_SHA, CG_STOP are required.
# CG_DIAGNOSTIC=1 selects correctness-only stops5/10; CG_RESUME is optional.
# No scheduler submissions, retry, or checkpoint deletion occurs in this file.
#SBATCH --account=vas@a100
#SBATCH --partition=gpu_p5
#SBATCH --no-requeue
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=classic-grpo400-dev
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
umask 027
module purge
module load arch/a100
module load miniforge/24.9.0
RUNTIME=/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env
conda activate "${RUNTIME}"
export PATH="${RUNTIME}/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${CG_SOURCE:?}/src"
cd "${CG_SOURCE}"
exec "${RUNTIME}/bin/python" -B scripts/run_classic_grpo_400_dev_chunk.py --executed-slurm "${BASH_SOURCE[0]}"
