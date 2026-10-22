#!/bin/bash -l
# Jiang-matched one-epoch specialist SFT for the missing OP/MIS/JSSP tasks.
# Submit through submit_sft500k_specialists_a100.sh for timeout-safe chaining.
#
#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --job-name=sft500k-specialist
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 027

TASK="${SFT500K_TASK:?SFT500K_TASK is required}"
case "${TASK}" in op|mis|jssp) ;; *) echo "[SFT500kSpecialistError] invalid task=${TASK}" >&2; exit 2 ;; esac

module purge
module load "${GPU_ARCH_MODULE:-arch/a100}"
module load "${CONDA_MODULE:-miniforge/24.9.0}"
conda activate "${CONDA_ENV:-new_unsloth_env}"

PROJECT_ROOT="$(readlink -f "${CLASP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}")"
ASSET_ROOT="${CLASP_ASSET_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/clasp}"
OUTPUT_ROOT="${SFT500K_OUTPUT_ROOT:-${ASSET_ROOT}/experiments/jiangmatched_task_sft500k_20260713_v2}"
RUN_DIR="${OUTPUT_ROOT}/${TASK}/seed42"
CACHE_ROOT="${SFT500K_CACHE_ROOT:-${OUTPUT_ROOT}/cache/${TASK}}"
MODEL_PATH="${BASE_MODEL_PATH:-${ASSET_ROOT}/models/Qwen2.5-7B-Instruct}"
DATA_ROOT="${SFT_DATA_ROOT:-${ASSET_ROOT}/data/data_sft}"
PREVIOUS_JOB_ID="${SFT500K_PREVIOUS_JOB_ID:-}"
TARGET_STEP=31250

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

[[ -f scripts/validate_sft500k_specialist.py ]] || {
    echo "[SFT500kSpecialistError] invalid project root: ${PROJECT_ROOT}" >&2
    exit 2
}
[[ "${SLURM_GPUS_ON_NODE:-0}" == "4" ]] || {
    echo "[SFT500kSpecialistError] exact trajectory requires 4 GPUs; got ${SLURM_GPUS_ON_NODE:-0}" >&2
    exit 2
}

state="$(python scripts/validate_sft500k_specialist.py \
    --run-dir "${RUN_DIR}" --task "${TASK}" \
    --model-path "${MODEL_PATH}" --data-root "${DATA_ROOT}")"
if [[ "${state}" == COMPLETE:* ]]; then
    echo "[SFT500kSpecialist] ${TASK} already complete at ${state#COMPLETE:}; no-op."
    exit 0
fi
if [[ "${state}" == "FRESH" ]]; then
    unset CLASP_SFT_REQUIRE_RESUME CLASP_SFT_RESUME_CHECKPOINT \
        CLASP_SFT_EXPECTED_MAX_STEPS || true
    resume_checkpoint="fresh"
elif [[ "${state}" == RESUME:* ]]; then
    resume_checkpoint="${state#RESUME:}"
    export CLASP_SFT_REQUIRE_RESUME=1
    export CLASP_SFT_RESUME_CHECKPOINT="${resume_checkpoint}"
    export CLASP_SFT_EXPECTED_MAX_STEPS="${TARGET_STEP}"
else
    echo "[SFT500kSpecialistError] unexpected validator state: ${state}" >&2
    exit 2
fi

if [[ -n "${PREVIOUS_JOB_ID}" ]]; then
    previous_state="$(sacct -j "${PREVIOUS_JOB_ID}" -X -n -P -o State 2>/dev/null | awk -F'|' 'NF {print $1; exit}')"
    case "${previous_state}" in
        TIMEOUT*|PREEMPTED*|NODE_FAIL*) ;;
        COMPLETED*)
            echo "[SFT500kSpecialistError] predecessor ${PREVIOUS_JOB_ID} completed without finalizing ${TASK}." >&2
            exit 2
            ;;
        *)
            echo "[SFT500kSpecialistError] predecessor ${PREVIOUS_JOB_ID} ended ${previous_state:-unknown}; refusing automatic continuation." >&2
            exit 2
            ;;
    esac
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TMPDIR="${CACHE_ROOT}/tmp"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${CACHE_ROOT}/wandb"
export WANDB_PROJECT="${WANDB_PROJECT:-clasp_jiangmatched_tasksft_500k_20260812}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-task_sft500k_${TASK}_seed42}"
export WANDB_NAME="${WANDB_NAME:-sft500k-${TASK}-a100-${SLURM_JOB_ID:-local}}"
export WANDB_RESUME=never
unset WANDB_RUN_ID WANDB_SWEEP_ID || true
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}" \
    "${TORCH_EXTENSIONS_DIR}" "${WANDB_DIR}" "${OUTPUT_ROOT}/logs"

# Jiang et al.: all seven attention/FFN projection types, r=64, alpha=64,
# lora_only bias.  For Qwen2.5-7B this attaches LoRA to each matching module in
# all 28 transformer blocks (196 adapted projections).
export CLASP_LORA_TARGET_MODULES=jiang
export CLASP_LORA_BIAS=lora_only
export CLASP_SFT_REQUIRE_COMPLETION_ONLY=1
export CLASP_SFT_MEMORY_SAFE_LOSS=1
export CLASP_SFT_SUPERVISED_LOGITS_ONLY=1
export CLASP_MIS_JIANG_RESPONSE_OUTPUT=1

echo "=============================================================="
echo " Jiang-matched 500k specialist SFT: ${TASK}"
echo " job=${SLURM_JOB_ID:-local} host=$(hostname) gpus=${SLURM_GPUS_ON_NODE:-0}"
echo " resume=${resume_checkpoint} target_step=${TARGET_STEP}"
echo " run=${RUN_DIR}"
echo " rows=500000 epochs=1 effective_batch=4_ranks*1_local*4_accum=16"
echo " LoRA=196 projections (7 types x 28 blocks), r64/a64, bias=lora_only"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=============================================================="

srun --ntasks=1 --ntasks-per-node=1 \
    torchrun --standalone --nproc_per_node=4 \
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
    --save_total_limit 8 \
    --report_to wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_group "${WANDB_RUN_GROUP}"

final_state="$(python scripts/validate_sft500k_specialist.py \
    --run-dir "${RUN_DIR}" --task "${TASK}" \
    --model-path "${MODEL_PATH}" --data-root "${DATA_ROOT}")"
[[ "${final_state}" == COMPLETE:* ]] || {
    echo "[SFT500kSpecialistError] training returned without a finalized target: ${final_state}" >&2
    exit 2
}
echo "[SFT500kSpecialist] completed task=${TASK} checkpoint=${final_state#COMPLETE:}"
