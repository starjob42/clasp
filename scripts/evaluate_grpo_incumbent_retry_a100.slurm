#!/bin/bash -l
#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=incumbent-raw-retry
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
umask 027
while IFS='=' read -r inherited_name _; do
    case "${inherited_name}" in
        CLASP_*|PD_*|GRPO_*|FOARL_*|GENERATION_*|WANDB_*|HF_*|TRANSFORMERS_*|TOKENIZERS_*|BITSANDBYTES_*|UNSLOTH_*|ACCELERATE_*|NCCL_*|PYTORCH_*|PYTHONPATH|PYTHONHOME|RESUME_FROM_CHECKPOINT|GPU_ARCH_MODULE|CONDA_MODULE)
            unset "${inherited_name}" ;;
    esac
done < <(env)
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"${RETRY_RUNTIME:?}/bin/python" "${RETRY_SOURCE:?}/scripts/grpo_incumbent_dev_evaluation_retry.py" worker \
    --key "${RETRY_KEY:?}" --running-wrapper "${BASH_SOURCE[0]}"
