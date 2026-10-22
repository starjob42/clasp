#!/bin/bash -l
# One predeclared raw heldout300 evaluation; all six models bind before outputs.
#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=incumbent-heldout300
set -euo pipefail
umask 027
CAMPAIGN="$(readlink -f "${HELDOUT_CAMPAIGN:?}")"
SOURCE="$(readlink -f "${HELDOUT_SOURCE:?}")"
KEY="${HELDOUT_KEY:?}"
case "$KEY" in
    cvrp/grpo_z|cvrp/grpo_verified|cvrp/foarl_f638_dev|mis/grpo_z|mis/grpo_verified|mis/foarl_f638_dev) ;;
    *) echo '[HeldoutError] undeclared task/method' >&2; exit 2 ;;
esac
while IFS='=' read -r inherited_name _; do
    case "$inherited_name" in
        CLASP_*|PD_*|GRPO_*|FOARL_*|GENERATION_*|WANDB_*|HF_*|TRANSFORMERS_*|TOKENIZERS_*|BITSANDBYTES_*|UNSLOTH_*|ACCELERATE_*|NCCL_*|PYTORCH_*|PYTHONPATH|PYTHONHOME)
            unset "$inherited_name" ;;
    esac
done < <(env)
module purge
module load arch/a100
module load miniforge/24.9.0
RUNTIME=/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env
conda activate "$RUNTIME"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec "$RUNTIME/bin/python" "$SOURCE/scripts/grpo_incumbent_heldout300_campaign.py" worker \
    --output "$CAMPAIGN" --key "$KEY" --running-wrapper "${BASH_SOURCE[0]}"
