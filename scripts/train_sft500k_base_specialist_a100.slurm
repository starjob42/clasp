#!/bin/bash -l
# One cell of the frozen five-task Qwen2.5-7B-base specialist-SFT campaign.
# This file is intentionally distinct from the historical Instruct worker.
# Required paths have no defaults and are verified against BASE_SFT_CAMPAIGN_CONTRACT.

#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --job-name=base-sft500k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 027

TASK="${BASE_SFT_TASK:?BASE_SFT_TASK is required}"
case "${TASK}" in tsp|cvrp|op|mis|jssp) ;; *) echo "[BaseSFT500kError] invalid task=${TASK}" >&2; exit 2 ;; esac

PROJECT_ROOT="$(readlink -f "${CLASP_PROJECT_ROOT:?CLASP_PROJECT_ROOT must name an immutable source snapshot}")"
IMMUTABLE_INPUT_TRUST_ROOT="$(readlink -f "${BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT:?BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT is required}")"
CAMPAIGN_CONTRACT="$(readlink -f "${BASE_SFT_CAMPAIGN_CONTRACT:?BASE_SFT_CAMPAIGN_CONTRACT is required}")"
FORBIDDEN_SOURCE_REGISTRY="$(readlink -f "${BASE_SFT_FORBIDDEN_SOURCE_REGISTRY:?BASE_SFT_FORBIDDEN_SOURCE_REGISTRY is required}")"
RESERVED_SOURCE_REGISTRY="$(readlink -f "${BASE_SFT_RESERVED_SOURCE_REGISTRY:?BASE_SFT_RESERVED_SOURCE_REGISTRY is required}")"
REMOTE_ENVIRONMENT_RECEIPT="$(readlink -f "${BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT:?BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT is required}")"
MODEL_PATH="$(readlink -f "${BASE_SFT_BASE_MODEL_PATH:?BASE_SFT_BASE_MODEL_PATH is required}")"
DATA_ROOT="$(readlink -f "${BASE_SFT_DATA_ROOT:?BASE_SFT_DATA_ROOT is required}")"
OUTPUT_ROOT="$(readlink -f "${BASE_SFT_OUTPUT_ROOT:?BASE_SFT_OUTPUT_ROOT is required and must already exist}")"
PREVIOUS_JOB_ID="${BASE_SFT_PREVIOUS_JOB_ID:-}"
RUN_DIR="${OUTPUT_ROOT}/runs/${TASK}/seed42"
CACHE_ROOT="${BASE_SFT_CACHE_ROOT:-${OUTPUT_ROOT}/cache/${TASK}}"
TARGET_STEP=31250

module purge
module load "${GPU_ARCH_MODULE:-arch/a100}"
module load "${CONDA_MODULE:-miniforge/24.9.0}"
conda activate "${BASE_SFT_CONDA_ENV:?BASE_SFT_CONDA_ENV must name the sealed environment prefix}"

SEALED_PYTHON="$(readlink -f "${CONDA_PREFIX:?conda activation did not set CONDA_PREFIX}/bin/python")"
SRUN_BIN="$(readlink -f "$(command -v srun)")"
SACCT_BIN="$(readlink -f "$(command -v sacct)")"
[[ -x "${SEALED_PYTHON}" && -x "${SRUN_BIN}" && -x "${SACCT_BIN}" ]] || {
    echo "[BaseSFT500kError] sealed Python or Slurm runtime is unavailable" >&2
    exit 2
}
export PATH="${CONDA_PREFIX}/bin:/usr/bin:/bin"
unset LD_PRELOAD PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS || true
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts"
[[ -f "${PROJECT_ROOT}/SOURCE_MANIFEST.sha256" \
    && -d "${IMMUTABLE_INPUT_TRUST_ROOT}" \
    && -f scripts/validate_base_sft500k_specialist.py \
    && -f scripts/build_clean_sft500k_corpus.py \
    && -f scripts/freeze_sft_reserved_source_registry.py \
    && -f scripts/audit_sft_token_lengths.py \
    && -f scripts/audit_sft_reload_equivalence.py \
    && -f scripts/run_base_sft_in_sealed_environment.py \
    && -f scripts/train_sft500k_base_specialist_a100.slurm \
    && -f "${CAMPAIGN_CONTRACT}" \
    && -f "${FORBIDDEN_SOURCE_REGISTRY}" \
    && -f "${RESERVED_SOURCE_REGISTRY}" \
    && -f "${REMOTE_ENVIRONMENT_RECEIPT}" \
    && -d "${MODEL_PATH}" \
    && -f "${DATA_ROOT}/sft_cleanliness_audit.json" \
    && -f "${DATA_ROOT}/sft_token_length_audit.json" \
    && -d "${DATA_ROOT}/${TASK}/train" \
    && -d "${OUTPUT_ROOT}" ]] || {
    echo "[BaseSFT500kError] required frozen source/model/data/contract path is missing" >&2
    exit 2
}
[[ "${SLURM_GPUS_ON_NODE:-0}" == "4" ]] || {
    echo "[BaseSFT500kError] exact campaign requires 4 GPUs; got ${SLURM_GPUS_ON_NODE:-0}" >&2
    exit 2
}

# Clear inherited research/runtime behavior before importing validation code.
# Slurm's accelerator visibility and the module-controlled native loader remain
# intact; all application-level behavior below is then set explicitly.
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

# Authorize the exact allocation before inspecting or writing any run artifact.
validate_live_inputs

# Recover safely from an authorized allocation that completed Trainer
# finalization but ended between either immutable post-save receipt write.
# Persistence is idempotently verified; reload evidence is create-once, so it
# is produced only when no filesystem entry already claims its authority path.
if [[ -f "${RUN_DIR}/adapter_model.safetensors" \
      && -f "${RUN_DIR}/adapter_config.json" \
      && -f "${RUN_DIR}/stage_status.json" ]]; then
    "${SEALED_PYTHON}" scripts/audit_lora_persistence.py \
        --base-model-path "${MODEL_PATH}" \
        --adapter-path "${RUN_DIR}" \
        --output "${RUN_DIR}/adapter_persistence_receipt.json" >/dev/null
    if [[ ! -e "${RUN_DIR}/sft_reload_equivalence.json" \
          && ! -L "${RUN_DIR}/sft_reload_equivalence.json" ]]; then
        "${SEALED_PYTHON}" scripts/audit_sft_reload_equivalence.py \
            --base-model-path "${MODEL_PATH}" \
            --adapter-path "${RUN_DIR}" \
            --runtime-dtype bfloat16 \
            --device cuda:0 \
            --output "${RUN_DIR}/sft_reload_equivalence.json" >/dev/null
    fi
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
    "${SEALED_PYTHON}" scripts/audit_lora_persistence.py \
        --base-model-path "${MODEL_PATH}" \
        --adapter-path "${RUN_DIR}" \
        --output "${RUN_DIR}/adapter_persistence_receipt.json" >/dev/null
    echo "[BaseSFT500k] ${TASK} already complete at ${state#COMPLETE:}; validated no-op."
    exit 0
fi
if [[ "${state}" == "FRESH" ]]; then
    resume_checkpoint="fresh"
elif [[ "${state}" == RESUME:* ]]; then
    resume_checkpoint="${state#RESUME:}"
else
    echo "[BaseSFT500kError] unexpected validator state: ${state}" >&2
    exit 2
fi

if [[ -n "${PREVIOUS_JOB_ID}" ]]; then
    [[ "${PREVIOUS_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || {
        echo "[BaseSFT500kError] predecessor job ID is invalid" >&2
        exit 2
    }
    previous_state="$("${SACCT_BIN}" -j "${PREVIOUS_JOB_ID}" -X -n -P -o State 2>/dev/null | awk -F'|' 'NF {print $1; exit}')"
    case "${previous_state}" in
        TIMEOUT*|PREEMPTED*|NODE_FAIL*) ;;
        COMPLETED*)
            echo "[BaseSFT500kError] predecessor ${PREVIOUS_JOB_ID} completed without finalizing ${TASK}" >&2
            exit 2
            ;;
        *)
            echo "[BaseSFT500kError] predecessor ${PREVIOUS_JOB_ID} ended ${previous_state:-unknown}; refusing continuation" >&2
            exit 2
            ;;
    esac
fi

if [[ "${resume_checkpoint}" == "fresh" ]]; then
    unset CLASP_SFT_REQUIRE_RESUME CLASP_SFT_RESUME_CHECKPOINT \
        CLASP_SFT_RESUME_FIDELITY_RECEIPT || true
    resume_wrapper_args=()
else
    export CLASP_SFT_REQUIRE_RESUME=1
    export CLASP_SFT_RESUME_CHECKPOINT="${resume_checkpoint}"
    export CLASP_SFT_RESUME_FIDELITY_RECEIPT="${resume_checkpoint}/base_sft_resume_fidelity.json"
    resume_wrapper_args=(
        --resume-checkpoint "${resume_checkpoint}"
        --resume-receipt "${CLASP_SFT_RESUME_FIDELITY_RECEIPT}"
    )
fi
export CLASP_SFT_EXPECTED_MAX_STEPS="${TARGET_STEP}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=42
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TMPDIR="${CACHE_ROOT}/tmp"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export WANDB_MODE=offline
export WANDB_DIR="${CACHE_ROOT}/wandb"
campaign_id="$("${SEALED_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["campaign_id"])' "${CAMPAIGN_CONTRACT}")"
export WANDB_PROJECT=clasp_qwen25_base_specialist_sft500k
export WANDB_RUN_GROUP="${campaign_id}-${TASK}-seed42"
export WANDB_NAME="${campaign_id}-${TASK}-seed42-job${SLURM_JOB_ID:-local}"
export WANDB_RESUME=never

export CLASP_LORA_TARGET_MODULES=jiang
# Qwen2.5-7B has exactly 84 biases, all in targeted q/k/v projections.
# PEFT 0.18/0.19 drops lora_only biases on save; `all` is scope-equivalent
# here and is the audited serialization-safe policy.
export CLASP_LORA_BIAS=all
export CLASP_PUBLICATION_FIDELITY=1
export CLASP_BASE_SFT_RESUME_FIDELITY=1
export CLASP_BASE_SFT_CAMPAIGN_CONTRACT="${CAMPAIGN_CONTRACT}"
export CLASP_BASE_SFT_RUN_CONTRACT="${RUN_DIR}/base_sft_run_contract.json"
export CLASP_SFT_REQUIRE_COMPLETION_ONLY=1
export CLASP_SFT_MEMORY_SAFE_LOSS=1
export CLASP_SFT_SUPERVISED_LOGITS_ONLY=1
export CLASP_MIS_JIANG_RESPONSE_OUTPUT=1

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}" \
    "${TORCH_EXTENSIONS_DIR}" "${WANDB_DIR}" "${OUTPUT_ROOT}/logs"

echo "=============================================================="
echo " Frozen Qwen2.5-7B-base 500k specialist SFT: ${TASK}"
echo " campaign=${campaign_id} job=${SLURM_JOB_ID:-local} gpus=${SLURM_GPUS_ON_NODE:-0}"
echo " resume=${resume_checkpoint} target_step=${TARGET_STEP}"
echo " source=${PROJECT_ROOT} base=${MODEL_PATH} run=${RUN_DIR}"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${SRUN_BIN}" --ntasks=1 --ntasks-per-node=1 \
    "${SEALED_PYTHON}" scripts/run_base_sft_in_sealed_environment.py \
    --receipt "${REMOTE_ENVIRONMENT_RECEIPT}" \
    --campaign-contract "${CAMPAIGN_CONTRACT}" \
    --immutable-input-trust-root "${IMMUTABLE_INPUT_TRUST_ROOT}" \
    --source-root "${PROJECT_ROOT}" \
    --base-model-path "${MODEL_PATH}" \
    --forbidden-source-registry "${FORBIDDEN_SOURCE_REGISTRY}" \
    --reserved-source-registry "${RESERVED_SOURCE_REGISTRY}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --task "${TASK}" \
    --run-contract "${RUN_DIR}/base_sft_run_contract.json" \
    "${resume_wrapper_args[@]}" -- \
    -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m clasp.entrypoints.train_sft \
    --tasks "${TASK}" \
    --model_path "${MODEL_PATH}" \
    --tokenizer_path "${MODEL_PATH}" \
    --data_root "${DATA_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --output_dir "${RUN_DIR}" \
    --prompt_style alpaca \
    --no_reason \
    --dtype bfloat16 \
    --lora_r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.0 \
    --seed 42 \
    --eval_ratio 0 \
    --eval-from-train-split \
    --max_samples_per_task 500000 \
    --max_eval_samples_per_task 0 \
    --min_train_samples_per_task 500000 \
    --min_eval_samples_per_task 0 \
    --no-prefer_subset_files \
    --no-deduplicate_train_rows \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --optim adamw_8bit \
    --weight_decay 0.01 \
    --warmup_steps 20 \
    --lr_scheduler_type linear \
    --num_train_epochs 1 \
    --max_steps -1 \
    --max_seq_length 20000 \
    --ddp_timeout 7200 \
    --numeric_token_loss_weight 1.0 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 64 \
    --report_to wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_group "${WANDB_RUN_GROUP}"

# Reopen the exact allocation again before any finalization-side receipt write.
validate_live_inputs

"${SEALED_PYTHON}" scripts/audit_lora_persistence.py \
    --base-model-path "${MODEL_PATH}" \
    --adapter-path "${RUN_DIR}" \
    --output "${RUN_DIR}/adapter_persistence_receipt.json" >/dev/null

# A separate post-training process must prove the serialized adapter reloads to
# the exact pre-save trainable closure and that two independent bfloat16 CUDA
# loads are bitwise identical in full state and fixed-probe distributions.
"${SEALED_PYTHON}" scripts/audit_sft_reload_equivalence.py \
    --base-model-path "${MODEL_PATH}" \
    --adapter-path "${RUN_DIR}" \
    --runtime-dtype bfloat16 \
    --device cuda:0 \
    --output "${RUN_DIR}/sft_reload_equivalence.json" >/dev/null

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
echo "[BaseSFT500k] completed task=${TASK} checkpoint=${final_state#COMPLETE:}"
