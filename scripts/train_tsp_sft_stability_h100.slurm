#!/bin/bash -l
# Isolated H100 launcher for the TSP LR-stability recovery campaign.
#
# Submit smoke mode under qos_gpu_h100-dev (the defaults below).  Production
# mode must override QoS/time at sbatch submission and crosses the historical
# step-3,145 failure window before checkpoint-5,000 can be accepted.

#SBATCH --account=vas@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --job-name=tsp-sft-stable-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

SOURCE_ROOT="$(readlink -f "${TSP_STABILITY_SOURCE_ROOT:?TSP_STABILITY_SOURCE_ROOT is required}")"
ENVIRONMENT_PREFIX="$(readlink -f "${TSP_STABILITY_ENVIRONMENT_PREFIX:?TSP_STABILITY_ENVIRONMENT_PREFIX is required}")"
MODEL_PATH="$(readlink -f "${TSP_STABILITY_MODEL_PATH:?TSP_STABILITY_MODEL_PATH is required}")"
DATA_ROOT="$(readlink -f "${TSP_STABILITY_DATA_ROOT:?TSP_STABILITY_DATA_ROOT is required}")"
OUTPUT_ROOT="$(readlink -f "${TSP_STABILITY_OUTPUT_ROOT:?TSP_STABILITY_OUTPUT_ROOT must already exist}")"
CACHE_ROOT="$(readlink -f "${TSP_STABILITY_CACHE_ROOT:?TSP_STABILITY_CACHE_ROOT must already exist}")"
MODE="${TSP_STABILITY_MODE:-smoke}"

case "${MODE}" in
    smoke)
        TARGET_STEP=250
        ;;
    production)
        TARGET_STEP="${TSP_STABILITY_TARGET_STEP:-5000}"
        case "${TARGET_STEP}" in
            5000|10000|15000|20000|25000|30000|31250) ;;
            *) echo "[TSPStabilityLaunchError] invalid production target ${TARGET_STEP}" >&2; exit 2 ;;
        esac
        ;;
    *) echo "[TSPStabilityLaunchError] invalid mode ${MODE}" >&2; exit 2 ;;
esac

RUN_DIR="${OUTPUT_ROOT}/runs/tsp/seed42"
CHECKPOINT_ROOT="${RUN_DIR}/checkpoints"

[[ -f "${SOURCE_ROOT}/SOURCE_MANIFEST.sha256" \
   && -f "${SOURCE_ROOT}/src/clasp/entrypoints/train_sft_jiang_f638.py" \
   && -f "${SOURCE_ROOT}/src/clasp/entrypoints/train_sft_jiang_f638_tsp_stable.py" \
   && -f "${SOURCE_ROOT}/scripts/audit_sft_reload_equivalence.py" \
   && -f "${DATA_ROOT}/tsp/train/train_tsp.json" \
   && -d "${MODEL_PATH}" \
   && -d "${OUTPUT_ROOT}" \
   && -d "${CACHE_ROOT}" ]] || {
    echo "[TSPStabilityLaunchError] required frozen input is missing" >&2
    exit 2
}

(
    cd "${SOURCE_ROOT}"
    sha256sum --check --strict --quiet SOURCE_MANIFEST.sha256
) || {
    echo "[TSPStabilityLaunchError] source manifest validation failed" >&2
    exit 2
}

module purge
module load "${GPU_ARCH_MODULE:-arch/h100}"
module load "${CONDA_MODULE:-miniforge/24.9.0}"
conda activate "${ENVIRONMENT_PREFIX}"

SEALED_PYTHON="$(readlink -f "${CONDA_PREFIX:?conda activation failed}/bin/python")"
SRUN_BIN="$(readlink -f "$(command -v srun)")"
[[ -x "${SEALED_PYTHON}" && -x "${SRUN_BIN}" ]] || {
    echo "[TSPStabilityLaunchError] sealed Python or srun is unavailable" >&2
    exit 2
}

# Eliminate inherited experiment switches before installing this run's sealed
# values.  Scheduler-provided CUDA/Slurm bindings are deliberately preserved.
while IFS='=' read -r inherited_name _; do
    case "${inherited_name}" in
        CLASP_*|PD_*|GENERATION_*|GRPO_*|PYTORCH_*|HF_*|TRANSFORMERS_*|TOKENIZERS_*|BITSANDBYTES_*|UNSLOTH_*|ACCELERATE_*|WANDB_*|NCCL_*)
            unset "${inherited_name}"
            ;;
    esac
done < <(env)
unset RESUME_FROM_CHECKPOINT WANDB_RUN_ID WANDB_SWEEP_ID || true

export PATH="${CONDA_PREFIX}/bin:/usr/bin:/bin"
unset LD_PRELOAD PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS || true
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SOURCE_ROOT}/src:${SOURCE_ROOT}/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export WANDB_MODE=offline WANDB_DIR="${CACHE_ROOT}/wandb"
export WANDB_PROJECT=clasp_jiang_f638_tsp_stability_iclr
export WANDB_RUN_GROUP="tsp_lr1e4_failfast_v1-${MODE}-seed42_iclr"
export WANDB_NAME="tsp_lr1e4_failfast_v1-${MODE}-seed42-step${TARGET_STEP}-job${SLURM_JOB_ID:-local}_iclr"
export WANDB_RESUME=never

node_tmpdir="$(mktemp -d "/tmp/tsp_stable_${SLURM_JOB_ID:?}.XXXXXX")"
trap 'rm -rf -- "${node_tmpdir}"' EXIT
export TMPDIR="${node_tmpdir}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TORCH_EXTENSIONS_DIR}" \
    "${WANDB_DIR}" "${OUTPUT_ROOT}/logs"

resume_args=()
if [[ "${MODE}" == production && -n "${TSP_STABILITY_RESUME_CHECKPOINT:-}" ]]; then
    resume_checkpoint="$(readlink -f "${TSP_STABILITY_RESUME_CHECKPOINT}")"
    [[ "${resume_checkpoint}" == "${CHECKPOINT_ROOT}/checkpoint-"* ]] || {
        echo "[TSPStabilityLaunchError] resume checkpoint is outside this recovery run" >&2
        exit 2
    }
    resume_args=(--resume-from-checkpoint "${resume_checkpoint}")
fi

echo "=============================================================="
echo " TSP Jiang-f638 stability recovery"
echo " profile=tsp_lr1e4_failfast_v1 mode=${MODE} target=${TARGET_STEP}"
echo " single_change=learning_rate:2e-4->1e-4"
echo " source=${SOURCE_ROOT} run=${RUN_DIR} job=${SLURM_JOB_ID:-local}"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "${SOURCE_ROOT}"
"${SRUN_BIN}" --ntasks=1 --ntasks-per-node=1 \
    "${SEALED_PYTHON}" -m clasp.entrypoints.train_sft_jiang_f638_tsp_stable \
    --task tsp \
    --model-path "${MODEL_PATH}" \
    --tokenizer-path "${MODEL_PATH}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${RUN_DIR}" \
    --cache-dir "${HF_DATASETS_CACHE}" \
    --report-to wandb \
    --stability-mode "${MODE}" \
    --stop-after-step "${TARGET_STEP}" \
    "${resume_args[@]}"

target_checkpoint="${CHECKPOINT_ROOT}/checkpoint-${TARGET_STEP}"
[[ -f "${target_checkpoint}/checkpoint_fidelity.json" \
   && -f "${target_checkpoint}/trainer_state.json" \
   && -f "${target_checkpoint}/adapter_model.safetensors" ]] || {
    echo "[TSPStabilityLaunchError] target checkpoint is incomplete" >&2
    exit 2
}

"${SEALED_PYTHON}" scripts/audit_sft_reload_equivalence.py \
    --base-model-path "${MODEL_PATH}" \
    --adapter-path "${target_checkpoint}" \
    --runtime-dtype bfloat16 \
    --device cuda:0 \
    --output "${target_checkpoint}/sft_reload_equivalence.json"

echo "[TSPStability] completed mode=${MODE} target=${TARGET_STEP} checkpoint=${target_checkpoint}"
