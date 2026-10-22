#!/bin/bash -l
# Jean Zay launcher for every CLASP training variant.
#
#   sbatch --export=ALL,VARIANT=clasp        scripts/train.slurm
#   sbatch --export=ALL,VARIANT=primal_dual  scripts/train.slurm
#   sbatch --export=ALL,VARIANT=grpo         scripts/train.slurm
#   sbatch --export=ALL,VARIANT=sft          scripts/train.slurm
#
# Every path and hyper-parameter below is an environment variable with a default, so a run
# can be retargeted without editing this file. See README.md for the full list.
#
#SBATCH --account=vas@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --job-name=clasp
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

VARIANT="${VARIANT:-clasp}"

# --- environment -----------------------------------------------------------------------
# `-l` above makes this a login shell so /etc/profile.d defines `module`; do not rely on
# `sbatch --export=ALL` propagating it as an exported shell function.
module purge
module load "${GPU_ARCH_MODULE:-arch/h100}"
module load "${CONDA_MODULE:-miniforge/24.9.0}"
conda activate "${CONDA_ENV:-new_unsloth_env}"

# Slurm executes a spooled copy of this script from /var/spool/slurmd, so BASH_SOURCE says
# nothing about where the project lives. SLURM_SUBMIT_DIR is the submission directory; fall
# back to $PWD so the script also works when run directly.
PROJECT_ROOT="${CLASP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Compute nodes have no outbound network: stay offline and keep caches on a writable root.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# --- assets ----------------------------------------------------------------------------
ASSET_ROOT="${CLASP_ASSET_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/clasp}"
STORAGE_ROOT="${CLASP_STORAGE_ROOT:-${PROJECT_ROOT}/outputs}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${ASSET_ROOT}/models/Qwen2.5-7B-Instruct}"
SFT_CHECKPOINT_PATH="${SFT_CHECKPOINT_PATH:-${ASSET_ROOT}/models/sft/checkpoint-18400}"
SFT_DATA_ROOT="${SFT_DATA_ROOT:-${ASSET_ROOT}/data/data_sft}"
GRPO_DATA_ROOT="${GRPO_DATA_ROOT:-${ASSET_ROOT}/data/data_rl}"
TASKS="${TASKS:-tsp,cvrp,op,mis,jssp}"

# --- schedule --------------------------------------------------------------------------
# TRL constraints, all validated up front by clasp.config:
#   per_device_train * world_size * grad_accum  %  num_generations == 0
#   per_device_eval  * world_size               %  num_generations == 0
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
MAX_STEPS="${MAX_STEPS:-40}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-7168}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
LEARNING_RATE="${LEARNING_RATE:-1e-06}"
# 64/64 matches the reference runs the shipped checkpoints came from.
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
SEED="${SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-20}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-2000}"
MAX_EVAL_SAMPLES_PER_TASK="${MAX_EVAL_SAMPLES_PER_TASK:-40}"
REPORT_TO="${REPORT_TO:-none}"

mkdir -p "${PROJECT_ROOT}/logs" "${STORAGE_ROOT}"

COMMON_ARGS=(
  --tasks "${TASKS}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --output_root "${STORAGE_ROOT}/${VARIANT}"
  --seed "${SEED}"
  --max_samples_per_task "${MAX_SAMPLES_PER_TASK}"
  --max_eval_samples_per_task "${MAX_EVAL_SAMPLES_PER_TASK}"
  --logging_steps "${LOGGING_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --report_to "${REPORT_TO}"
)

GRPO_ARGS=(
  --model_path "${SFT_CHECKPOINT_PATH}"
  --data_root "${GRPO_DATA_ROOT}"
  --learning_rate "${LEARNING_RATE}"
  --max_steps "${MAX_STEPS}"
  --num_generations "${NUM_GENERATIONS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
)

case "${VARIANT}" in
  sft)
    ENTRYPOINT="clasp.entrypoints.train_sft"
    STAGE_ARGS=(
      --model_path "${BASE_MODEL_PATH}"
      --data_root "${SFT_DATA_ROOT}"
      --max_steps "${MAX_STEPS}"
      --max_seq_length "${MAX_SEQ_LENGTH}"
      --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    )
    ;;
  grpo)
    ENTRYPOINT="clasp.entrypoints.train_grpo_standard"
    STAGE_ARGS=("${GRPO_ARGS[@]}")
    ;;
  primal_dual)
    ENTRYPOINT="clasp.entrypoints.train_grpo"
    STAGE_ARGS=("${GRPO_ARGS[@]}" --use_primal_dual --no-use_clasp)
    ;;
  clasp)
    ENTRYPOINT="clasp.entrypoints.train_grpo"
    STAGE_ARGS=("${GRPO_ARGS[@]}" --use_primal_dual --use_clasp)
    ;;
  *)
    echo "Unknown VARIANT '${VARIANT}'. Expected one of: sft, grpo, primal_dual, clasp." >&2
    exit 2
    ;;
esac

echo "=============================================================="
echo " CLASP  variant=${VARIANT}  entrypoint=${ENTRYPOINT}"
echo " job=${SLURM_JOB_ID:-local}  nodes=${SLURM_NNODES:-1}  gpus/node=${GPUS_PER_NODE}"
echo " project=${PROJECT_ROOT}"
echo " storage=${STORAGE_ROOT}/${VARIANT}"
echo "=============================================================="

srun --ntasks="${SLURM_NNODES:-1}" --ntasks-per-node=1 \
  torchrun \
    --nnodes="${SLURM_NNODES:-1}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n1):${MASTER_PORT:-29500}" \
    -m "${ENTRYPOINT}" "${COMMON_ARGS[@]}" "${STAGE_ARGS[@]}"
