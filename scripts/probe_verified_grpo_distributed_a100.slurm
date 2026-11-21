#!/bin/bash -l
# Model-free distributed correctness diagnostic; no benchmark inference.
#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=grpo-ddp-probe

set -euo pipefail
module purge
module load arch/a100
module load miniforge/24.9.0
conda activate "${PROBE_RUNTIME:?}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROBE_SOURCE:?}/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${PROBE_SOURCE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "${PROBE_OUTPUT:?}"
probe_status=0
for optimizer in adamw8bit adamw; do
    for inspection in 0 1; do
        if ! "${PROBE_RUNTIME}/bin/python" -u scripts/probe_verified_grpo_distributed.py \
            --device cuda --optimizer "${optimizer}" --world-size 4 --steps 4 \
            --inspect-every-step "${inspection}" \
            --output "${PROBE_OUTPUT}/${optimizer}_inspect${inspection}.json"; then
            probe_status=1
        fi
    done
done
exit "${probe_status}"
