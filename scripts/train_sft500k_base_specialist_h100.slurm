#!/bin/bash -l
# One certified 20-hour chunk of a Jiang-f638 Qwen2.5-7B-base specialist.
# The first chunk starts fresh; every continuation consumes only the exact
# single-H100 checkpoint certified by the preceding afterok allocation.

#SBATCH --account=vas@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --job-name=jiang-sft500k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 027

TASK="${BASE_SFT_TASK:?BASE_SFT_TASK is required}"
case "${TASK}" in tsp|cvrp|op|mis|jssp) ;; *) echo "[BaseSFT500kError] invalid task=${TASK}" >&2; exit 2 ;; esac
LAUNCH_PROFILE="${BASE_SFT_LAUNCH_PROFILE:?BASE_SFT_LAUNCH_PROFILE is required}"
[[ "${LAUNCH_PROFILE}" == "regular20h_iclr" ]] || {
    echo "[BaseSFT500kError] production worker requires regular20h_iclr" >&2
    exit 2
}
TARGET_STEP="${BASE_SFT_TARGET_STEP:?BASE_SFT_TARGET_STEP is required}"
case "${TARGET_STEP}" in
    5000|10000|15000|20000|25000|30000|31250) ;;
    *) echo "[BaseSFT500kError] invalid chunk boundary=${TARGET_STEP}" >&2; exit 2 ;;
esac

PROJECT_ROOT="$(readlink -f "${CLASP_PROJECT_ROOT:?CLASP_PROJECT_ROOT must name an immutable source snapshot}")"
IMMUTABLE_INPUT_TRUST_ROOT="$(readlink -f "${BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT:?BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT is required}")"
CAMPAIGN_CONTRACT="$(readlink -f "${BASE_SFT_CAMPAIGN_CONTRACT:?BASE_SFT_CAMPAIGN_CONTRACT is required}")"
FORBIDDEN_SOURCE_REGISTRY="$(readlink -f "${BASE_SFT_FORBIDDEN_SOURCE_REGISTRY:?BASE_SFT_FORBIDDEN_SOURCE_REGISTRY is required}")"
RESERVED_SOURCE_REGISTRY="$(readlink -f "${BASE_SFT_RESERVED_SOURCE_REGISTRY:?BASE_SFT_RESERVED_SOURCE_REGISTRY is required}")"
REMOTE_ENVIRONMENT_RECEIPT="$(readlink -f "${BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT:?BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT is required}")"
MODEL_PATH="$(readlink -f "${BASE_SFT_BASE_MODEL_PATH:?BASE_SFT_BASE_MODEL_PATH is required}")"
DATA_ROOT="$(readlink -f "${BASE_SFT_DATA_ROOT:?BASE_SFT_DATA_ROOT is required}")"
OUTPUT_ROOT="$(readlink -f "${BASE_SFT_OUTPUT_ROOT:?BASE_SFT_OUTPUT_ROOT is required and must already exist}")"
RUN_DIR="${OUTPUT_ROOT}/runs/${TASK}/seed42"
CACHE_ROOT="${BASE_SFT_CACHE_ROOT:-${OUTPUT_ROOT}/cache/${TASK}}"
[[ "${OUTPUT_ROOT}" == *_iclr* ]] || {
    echo "[BaseSFT500kError] output root must contain _iclr: ${OUTPUT_ROOT}" >&2
    exit 2
}

module purge
module load "${GPU_ARCH_MODULE:-arch/h100}"
module load "${CONDA_MODULE:-miniforge/24.9.0}"
conda activate "${BASE_SFT_CONDA_ENV:?BASE_SFT_CONDA_ENV must name the sealed environment prefix}"

SEALED_PYTHON="$(readlink -f "${CONDA_PREFIX:?conda activation did not set CONDA_PREFIX}/bin/python")"
SRUN_BIN="$(readlink -f "$(command -v srun)")"
[[ -x "${SEALED_PYTHON}" && -x "${SRUN_BIN}" ]] || {
    echo "[BaseSFT500kError] sealed Python or Slurm runtime is unavailable" >&2
    exit 2
}
export PATH="${CONDA_PREFIX}/bin:/usr/bin:/bin"
unset LD_PRELOAD PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS || true
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts"
[[ -f SOURCE_MANIFEST.sha256 \
    && -f requirements.jiang-f638-sft.txt \
    && -f scripts/validate_base_sft500k_specialist.py \
    && -f scripts/audit_lora_persistence.py \
    && -f scripts/audit_sft_reload_equivalence.py \
    && -f scripts/base_sft_single_h100_resume_fidelity.py \
    && -f scripts/seal_completed_sft_run.py \
    && -f scripts/run_jiang_f638_sft_in_sealed_environment.py \
    && -f src/clasp/entrypoints/train_sft_jiang_f638.py \
    && -f "${CAMPAIGN_CONTRACT}" \
    && -f "${FORBIDDEN_SOURCE_REGISTRY}" \
    && -f "${RESERVED_SOURCE_REGISTRY}" \
    && -f "${REMOTE_ENVIRONMENT_RECEIPT}" \
    && -d "${MODEL_PATH}" \
    && -f "${DATA_ROOT}/sft_cleanliness_audit.json" \
    && -f "${DATA_ROOT}/sft_token_length_audit.json" \
    && -d "${DATA_ROOT}/${TASK}/train" \
    && -d "${OUTPUT_ROOT}" ]] || {
    echo "[BaseSFT500kError] required frozen input is missing" >&2
    exit 2
}

# Eliminate inherited experiment switches. The dedicated entrypoint fixes every
# optimization and representation setting internally.
while IFS='=' read -r inherited_name _; do
    case "${inherited_name}" in
        CLASP_*|PD_*|GENERATION_*|GRPO_*|PYTORCH_*|HF_*|TRANSFORMERS_*|TOKENIZERS_*|BITSANDBYTES_*|UNSLOTH_*|ACCELERATE_*|WANDB_*|NCCL_*)
            unset "${inherited_name}"
            ;;
    esac
done < <(env)
unset RESUME_FROM_CHECKPOINT WANDB_RUN_ID WANDB_SWEEP_ID || true
export CLASP_PROJECT_ROOT="${PROJECT_ROOT}"

validate_live_inputs() {
    "${SEALED_PYTHON}" scripts/validate_base_sft500k_specialist.py \
        --campaign-contract "${CAMPAIGN_CONTRACT}" \
        --immutable-input-trust-root "${IMMUTABLE_INPUT_TRUST_ROOT}" \
        --source-root "${PROJECT_ROOT}" \
        --base-model-path "${MODEL_PATH}" \
        --forbidden-source-registry "${FORBIDDEN_SOURCE_REGISTRY}" \
        --reserved-source-registry "${RESERVED_SOURCE_REGISTRY}" \
        --remote-environment-receipt "${REMOTE_ENVIRONMENT_RECEIPT}" \
        --data-root "${DATA_ROOT}" \
        --output-root "${OUTPUT_ROOT}" \
        --task "${TASK}" --verify-inputs-only \
        --require-live-accelerator --require-cell-quota >/dev/null
}

seal_completed_run() {
    "${SEALED_PYTHON}" scripts/seal_completed_sft_run.py \
        --run-dir "${RUN_DIR}" \
        --base-model-path "${MODEL_PATH}" >/dev/null
}

seal_resume_checkpoint() {
    local checkpoint="$1"
    "${SEALED_PYTHON}" scripts/base_sft_single_h100_resume_fidelity.py \
        --campaign-contract "${CAMPAIGN_CONTRACT}" \
        --run-contract "${RUN_DIR}/base_sft_run_contract.json" \
        --checkpoint "${checkpoint}" \
        --seal --require-live-accelerator >/dev/null
}

validate_live_inputs

# Complete certification interrupted after the terminal adapter export without
# ever continuing optimizer updates.
if [[ -f "${RUN_DIR}/adapter_model.safetensors" \
      && -f "${RUN_DIR}/adapter_config.json" \
      && -f "${RUN_DIR}/stage_status.json" ]]; then
    seal_resume_checkpoint "${RUN_DIR}/checkpoints/checkpoint-31250"
    if [[ ! -e "${RUN_DIR}/adapter_persistence_receipt.json" \
          && ! -L "${RUN_DIR}/adapter_persistence_receipt.json" ]]; then
        "${SEALED_PYTHON}" scripts/audit_lora_persistence.py \
            --base-model-path "${MODEL_PATH}" \
            --adapter-path "${RUN_DIR}" \
            --output "${RUN_DIR}/adapter_persistence_receipt.json" >/dev/null
    fi
    if [[ ! -e "${RUN_DIR}/sft_reload_equivalence.json" \
          && ! -L "${RUN_DIR}/sft_reload_equivalence.json" ]]; then
        "${SEALED_PYTHON}" scripts/audit_sft_reload_equivalence.py \
            --base-model-path "${MODEL_PATH}" \
            --adapter-path "${RUN_DIR}" \
            --runtime-dtype bfloat16 \
            --device cuda:0 \
            --output "${RUN_DIR}/sft_reload_equivalence.json" >/dev/null
    fi
    seal_completed_run
fi

state="$("${SEALED_PYTHON}" scripts/validate_base_sft500k_specialist.py \
    --campaign-contract "${CAMPAIGN_CONTRACT}" \
    --immutable-input-trust-root "${IMMUTABLE_INPUT_TRUST_ROOT}" \
    --source-root "${PROJECT_ROOT}" \
    --base-model-path "${MODEL_PATH}" \
    --forbidden-source-registry "${FORBIDDEN_SOURCE_REGISTRY}" \
    --reserved-source-registry "${RESERVED_SOURCE_REGISTRY}" \
    --remote-environment-receipt "${REMOTE_ENVIRONMENT_RECEIPT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --task "${TASK}" --run-dir "${RUN_DIR}" \
    --require-live-accelerator \
    --initialize-run-binding)"
if [[ "${state}" == COMPLETE:* ]]; then
    echo "[BaseSFT500k] ${TASK} already complete and certified; validated no-op."
    exit 0
fi

RESUME_ARGS=()
CURRENT_STEP=0
if [[ "${state}" == "FRESH" ]]; then
    [[ "${TARGET_STEP}" == "5000" ]] || {
        echo "[BaseSFT500kError] fresh cell may only execute boundary 5000" >&2
        exit 2
    }
elif [[ "${state}" == RESUME:* ]]; then
    RESUME_CHECKPOINT="${state#RESUME:}"
    [[ "${RESUME_CHECKPOINT}" == "${RUN_DIR}/checkpoints/checkpoint-"* ]] || {
        echo "[BaseSFT500kError] validator returned checkpoint outside run: ${state}" >&2
        exit 2
    }
    CURRENT_STEP="${RESUME_CHECKPOINT##*-}"
    case "${CURRENT_STEP}:${TARGET_STEP}" in
        5000:10000|10000:15000|15000:20000|20000:25000|25000:30000|30000:31250) ;;
        5000:5000|10000:10000|15000:15000|20000:20000|25000:25000|30000:30000)
            echo "[BaseSFT500k] task=${TASK} boundary=${TARGET_STEP} already certified; no-op."
            exit 0
            ;;
        31250:31250)
            echo "[BaseSFT500kError] terminal checkpoint exists without a complete sealed run" >&2
            exit 2
            ;;
        *)
            echo "[BaseSFT500kError] nonconsecutive chunk request ${CURRENT_STEP}->${TARGET_STEP}" >&2
            exit 2
            ;;
    esac
    RESUME_ARGS=(--resume-from-checkpoint "${RESUME_CHECKPOINT}")
else
    echo "[BaseSFT500kError] unexpected validator state: ${state}" >&2
    exit 2
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16 PYTHONHASHSEED=42
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
# Keep multiprocess AF_UNIX sockets below sockaddr_un's path limit.  Only
# ephemeral IPC/compiler scratch uses node-local /tmp; checkpoints, W&B logs,
# model/data caches, and all receipts stay under the `_iclr` campaign roots.
node_tmpdir="$(mktemp -d "/tmp/csft_iclr_${SLURM_JOB_ID:?}.XXXXXX")"
trap 'rm -rf -- "${node_tmpdir}"' EXIT
export TMPDIR="${node_tmpdir}"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export WANDB_MODE=offline
export WANDB_DIR="${CACHE_ROOT}/wandb"
campaign_id="$("${SEALED_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["campaign_id"])' "${CAMPAIGN_CONTRACT}")"
[[ "${campaign_id}" == *_iclr* ]] || {
    echo "[BaseSFT500kError] campaign id must contain _iclr: ${campaign_id}" >&2
    exit 2
}
export WANDB_PROJECT=clasp_jiang_f638_qwen25_base_sft500k_iclr
export WANDB_RUN_GROUP="${campaign_id}-${LAUNCH_PROFILE}-${TASK}-seed42_iclr"
export WANDB_NAME="${campaign_id}-${LAUNCH_PROFILE}-${TASK}-seed42-step${CURRENT_STEP}-to-${TARGET_STEP}-job${SLURM_JOB_ID:-local}_iclr"
export WANDB_RESUME=never
export CLASP_LORA_BIAS=all
export CLASP_PUBLICATION_FIDELITY=1

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}" \
    "${TORCH_EXTENSIONS_DIR}" "${WANDB_DIR}" "${OUTPUT_ROOT}/logs"

echo "=============================================================="
echo " Jiang-f638 corrected specialist SFT: ${TASK}"
echo " campaign=${campaign_id} profile=${LAUNCH_PROFILE} boundary=${CURRENT_STEP}->${TARGET_STEP} job=${SLURM_JOB_ID:-local} gpus=1"
echo " source=${PROJECT_ROOT} base=${MODEL_PATH} run=${RUN_DIR}"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${SRUN_BIN}" --ntasks=1 --ntasks-per-node=1 \
    "${SEALED_PYTHON}" scripts/run_jiang_f638_sft_in_sealed_environment.py \
    --receipt "${REMOTE_ENVIRONMENT_RECEIPT}" \
    --campaign-contract "${CAMPAIGN_CONTRACT}" \
    --immutable-input-trust-root "${IMMUTABLE_INPUT_TRUST_ROOT}" \
    --source-root "${PROJECT_ROOT}" \
    --base-model-path "${MODEL_PATH}" \
    --forbidden-source-registry "${FORBIDDEN_SOURCE_REGISTRY}" \
    --reserved-source-registry "${RESERVED_SOURCE_REGISTRY}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --task "${TASK}" -- \
    -m clasp.entrypoints.train_sft_jiang_f638 \
    --task "${TASK}" \
    --model-path "${MODEL_PATH}" \
    --tokenizer-path "${MODEL_PATH}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${RUN_DIR}" \
    --cache-dir "${HF_DATASETS_CACHE}" \
    --report-to wandb \
    --post-save-reload-check \
    --stop-after-step "${TARGET_STEP}" \
    "${RESUME_ARGS[@]}"

validate_live_inputs
TARGET_CHECKPOINT="${RUN_DIR}/checkpoints/checkpoint-${TARGET_STEP}"
seal_resume_checkpoint "${TARGET_CHECKPOINT}"

if [[ "${TARGET_STEP}" != "31250" ]]; then
    partial_state="$("${SEALED_PYTHON}" scripts/validate_base_sft500k_specialist.py \
        --campaign-contract "${CAMPAIGN_CONTRACT}" \
        --immutable-input-trust-root "${IMMUTABLE_INPUT_TRUST_ROOT}" \
        --source-root "${PROJECT_ROOT}" \
        --base-model-path "${MODEL_PATH}" \
        --forbidden-source-registry "${FORBIDDEN_SOURCE_REGISTRY}" \
        --reserved-source-registry "${RESERVED_SOURCE_REGISTRY}" \
        --remote-environment-receipt "${REMOTE_ENVIRONMENT_RECEIPT}" \
        --data-root "${DATA_ROOT}" \
        --output-root "${OUTPUT_ROOT}" \
        --task "${TASK}" --run-dir "${RUN_DIR}" \
        --require-live-accelerator)"
    [[ "${partial_state}" == "RESUME:${TARGET_CHECKPOINT}" ]] || {
        echo "[BaseSFT500kError] chunk returned without certified boundary: ${partial_state}" >&2
        exit 2
    }
    echo "[BaseSFT500k] certified task=${TASK} boundary=${TARGET_STEP} checkpoint=${TARGET_CHECKPOINT}"
    exit 0
fi

"${SEALED_PYTHON}" scripts/audit_lora_persistence.py \
    --base-model-path "${MODEL_PATH}" \
    --adapter-path "${RUN_DIR}" \
    --output "${RUN_DIR}/adapter_persistence_receipt.json" >/dev/null
# The dedicated trainer writes stage_status.json immediately after the final
# save, then performs this create-once two-load audit. Reopen it through the
# final validator; only produce it here if certification was interrupted after
# the terminal save but before the reload receipt was committed.
if [[ ! -e "${RUN_DIR}/sft_reload_equivalence.json" \
      && ! -L "${RUN_DIR}/sft_reload_equivalence.json" ]]; then
    "${SEALED_PYTHON}" scripts/audit_sft_reload_equivalence.py \
        --base-model-path "${MODEL_PATH}" \
        --adapter-path "${RUN_DIR}" \
        --runtime-dtype bfloat16 \
        --device cuda:0 \
        --output "${RUN_DIR}/sft_reload_equivalence.json" >/dev/null
fi
seal_completed_run

final_state="$("${SEALED_PYTHON}" scripts/validate_base_sft500k_specialist.py \
    --campaign-contract "${CAMPAIGN_CONTRACT}" \
    --immutable-input-trust-root "${IMMUTABLE_INPUT_TRUST_ROOT}" \
    --source-root "${PROJECT_ROOT}" \
    --base-model-path "${MODEL_PATH}" \
    --forbidden-source-registry "${FORBIDDEN_SOURCE_REGISTRY}" \
    --reserved-source-registry "${RESERVED_SOURCE_REGISTRY}" \
    --remote-environment-receipt "${REMOTE_ENVIRONMENT_RECEIPT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --task "${TASK}" --run-dir "${RUN_DIR}" \
    --require-live-accelerator)"
[[ "${final_state}" == COMPLETE:* ]] || {
    echo "[BaseSFT500kError] training returned without a certified target: ${final_state}" >&2
    exit 2
}
echo "[BaseSFT500k] completed task=${TASK} adapter=${RUN_DIR}"
