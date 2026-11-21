#!/bin/bash -l
# Fixed 400-update FOARL comparison; each invocation runs one bounded chunk.
# FT_TASK, FT_METHOD, FT_OUTPUT, FT_SOURCE, FT_SOURCE_SHA, FT_STOP are required.
# FT_DIAGNOSTIC=1 selects correctness-only stops5/10; FT_RESUME is optional.
# No scheduler submissions, retry, or checkpoint deletion occurs in this file.
#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=foarl-teacher-dev
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
export PYTHONPATH="${FT_SOURCE:?}/src"
cd "${FT_SOURCE}"
exec "${RUNTIME}/bin/python" -B scripts/run_foarl_teacher_dev_chunk.py --executed-slurm "${BASH_SOURCE[0]}"
