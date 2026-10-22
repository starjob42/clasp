#!/bin/bash -l
# Continue the existing unified 100k/task attention-LoRA run from epoch 1 to
# epoch 5.  This is a strict resume of the same four-rank Trainer trajectory.
# Submit through submit_sft100k_a100_chain.sh so wall-time handoffs are chained.
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
#SBATCH --job-name=sft100k-e5-a100
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 027

module purge
module load "${GPU_ARCH_MODULE:-arch/a100}"
module load "${CONDA_MODULE:-miniforge/24.9.0}"
conda activate "${CONDA_ENV:-new_unsloth_env}"

PROJECT_ROOT="$(readlink -f "${CLASP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}")"
ASSET_ROOT="${CLASP_ASSET_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/clasp}"
RUN_DIR="${SFT100K_RUN_DIR:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp/experiments/jiangmatched_unified_oneadapter_sft100kpertask_attnlora_dev2h_chain5_20260717_v1/seed42}"
CACHE_ROOT="${SFT100K_CACHE_ROOT:-${RUN_DIR}/cache}"
MODEL_PATH="${BASE_MODEL_PATH:-${ASSET_ROOT}/models/Qwen2.5-7B-Instruct}"
DATA_ROOT="${SFT_DATA_ROOT:-${ASSET_ROOT}/data/data_sft}"
PREVIOUS_JOB_ID="${SFT100K_PREVIOUS_JOB_ID:-}"
TARGET_STEP=156250

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

[[ -f scripts/validate_sft100k_a100_continuation.py ]] || {
    echo "[SFT100kContinuationError] invalid project root: ${PROJECT_ROOT}" >&2
    exit 2
}
[[ "${SLURM_GPUS_ON_NODE:-0}" == "4" ]] || {
    echo "[SFT100kContinuationError] exact resume requires 4 GPUs; got ${SLURM_GPUS_ON_NODE:-0}" >&2
    exit 2
}

state="$(python scripts/validate_sft100k_a100_continuation.py \
    --run-dir "${RUN_DIR}" \
    --model-path "${MODEL_PATH}" \
    --data-root "${DATA_ROOT}")"
if [[ "${state}" == COMPLETE:* ]]; then
    echo "[SFT100kContinuation] five-epoch target already finalized at ${state#COMPLETE:}; no-op."
    exit 0
fi
if [[ "${state}" != RESUME:* ]]; then
    echo "[SFT100kContinuationError] unexpected validator state: ${state}" >&2
    exit 2
fi
resume_checkpoint="${state#RESUME:}"

if [[ -n "${PREVIOUS_JOB_ID}" ]]; then
    previous_state="$(sacct -j "${PREVIOUS_JOB_ID}" -X -n -P -o State 2>/dev/null | awk -F'|' 'NF {print $1; exit}')"
    case "${previous_state}" in
        TIMEOUT*|PREEMPTED*|NODE_FAIL*) ;;
        COMPLETED*)
            echo "[SFT100kContinuationError] predecessor ${PREVIOUS_JOB_ID} completed without finalizing step ${TARGET_STEP}." >&2
            exit 2
            ;;
        *)
            echo "[SFT100kContinuationError] predecessor ${PREVIOUS_JOB_ID} ended ${previous_state:-unknown}; refusing automatic continuation." >&2
            exit 2
            ;;
    esac
fi

# Preserve the immutable epoch-1 receipt before run_sft writes the continuation
# configuration to hyperparams.json.
mkdir -p "${RUN_DIR}/controller" "${PROJECT_ROOT}/logs"
cp -p --no-clobber "${RUN_DIR}/hyperparams.json" \
    "${RUN_DIR}/controller/hyperparams_epoch1_before_a100_continuation.json" || true

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${SFT100K_HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${SFT100K_HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TMPDIR="${SFT100K_TMPDIR:-${CACHE_ROOT}/tmp}"
export TORCH_EXTENSIONS_DIR="${SFT100K_TORCH_EXTENSIONS_DIR:-${CACHE_ROOT}/torch_extensions}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${SFT100K_WANDB_DIR:-${CACHE_ROOT}/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-clasp_sft100k_unified_attnlora_epoch5_a100}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-seed42_continuation_epoch1_to5}"
export WANDB_NAME="${WANDB_NAME:-sft100k-e5-a100-${SLURM_JOB_ID:-local}}"
export WANDB_RESUME=never
unset WANDB_RUN_ID WANDB_SWEEP_ID || true
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}" \
    "${TORCH_EXTENSIONS_DIR}" "${WANDB_DIR}"

export CLASP_LORA_TARGET_MODULES=default
export CLASP_LORA_BIAS=lora_only
export CLASP_SFT_REQUIRE_COMPLETION_ONLY=1
export CLASP_SFT_MEMORY_SAFE_LOSS=1
export CLASP_SFT_SUPERVISED_LOGITS_ONLY=1
export CLASP_SFT_REQUIRE_RESUME=1
export CLASP_SFT_RESUME_CHECKPOINT="${resume_checkpoint}"
# The source checkpoint records the old one-epoch horizon (31,250); later
# checkpoints record 156,250.  The pinned validator accepts only those two
# horizons, so do not apply one fixed CLASP_SFT_EXPECTED_MAX_STEPS value.
unset CLASP_SFT_EXPECTED_MAX_STEPS || true

export CLASP_MIS_JIANG_RESPONSE_OUTPUT=1
export CLASP_TSP_PROMPT_KNN=0
export CLASP_TSP_PROMPT_KNN_DISTANCES=0
export CLASP_TSP_INPUT_ORDER=""
export CLASP_TSP_RELABEL_ORDER=""
export CLASP_TSP_ROUTE_CANONICALIZE=""
export CLASP_TSP_ROUTE_ONLY_OUTPUT=0

echo "=============================================================="
echo " Unified SFT 100k/task LoRA continuation: epoch 1 -> epoch 5"
echo " job=${SLURM_JOB_ID:-local} host=$(hostname) gpus=${SLURM_GPUS_ON_NODE:-0}"
echo " resume=${resume_checkpoint} target_step=${TARGET_STEP}"
echo " run=${RUN_DIR}"
echo " batch=4_ranks*1_local*4_accum=16 seq=20000 lr=2e-4 epochs_total=5"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=============================================================="

srun --ntasks=1 --ntasks-per-node=1 \
    torchrun --standalone --nproc_per_node=4 \
    -m clasp.entrypoints.train_sft \
    --tasks tsp,cvrp,op,mis,jssp \
    --model_path "${MODEL_PATH}" \
    --tokenizer_path "${MODEL_PATH}" \
    --data_root "${DATA_ROOT}" \
    --output_root "$(dirname "${RUN_DIR}")" \
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
    --max_samples_per_task 100000 \
    --max_eval_samples_per_task 0 \
    --min_train_samples_per_task 100000 \
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
    --num_train_epochs 5 \
    --max_steps -1 \
    --max_seq_length 20000 \
    --ddp_timeout 7200 \
    --numeric_token_loss_weight 1.0 \
    --logging_steps 10 \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 8 \
    --report_to wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_group "${WANDB_RUN_GROUP}"

python scripts/validate_sft100k_a100_continuation.py \
    --run-dir "${RUN_DIR}" \
    --model-path "${MODEL_PATH}" \
    --data-root "${DATA_ROOT}"
