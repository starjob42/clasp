#!/bin/bash -l
# One task/arm, terminal100 or the common SFT, existing inspected DEV100 only.
# Required: EB_SOURCE, EB_SOURCE_SHA, EB_TASK, EB_ARM, EB_CHECKPOINT, EB_EVAL_OUTPUT.
# No scheduler submission or retry occurs inside this wrapper.
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
#SBATCH --job-name=grpo-external-eval
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
export PYTHONPATH="${EB_SOURCE:?EB_SOURCE is required}/src"
cd "${EB_SOURCE}"
exec "${EB_RUNTIME}/bin/python" -B scripts/evaluate_external_baseline_dev.py run \
    --source-manifest-sha256 "${EB_SOURCE_SHA:?EB_SOURCE_SHA is required}" \
    --task "${EB_TASK:?EB_TASK is required}" --arm "${EB_ARM:?EB_ARM is required}" \
    --checkpoint "${EB_CHECKPOINT:?EB_CHECKPOINT is required}" \
    --output "${EB_EVAL_OUTPUT:?EB_EVAL_OUTPUT is required}" \
    --executed-slurm "${BASH_SOURCE[0]}"
