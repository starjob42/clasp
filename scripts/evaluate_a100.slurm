#!/bin/bash -l
# Jean Zay A100 launcher for SFT-checkpoint baselines under the exact BoN=8 protocol.
#
#   CKPT_LABEL=sft50k_18400 CHECKPOINT_PATH=/path/to/checkpoint-18400 \
#     sbatch --array=0-4 scripts/evaluate_a100.slurm
# Exact Jiang SFT representation additionally requires:
#   EVALUATION_REPRESENTATION_PROFILE=jiang_f638_sft
#
# One array task per CO task. This is explicitly a replication launcher for the previously
# observed [200,300) slice, never a confirmation launcher. Every generation setting below
# is pinned to the protocol that produced the existing SFT frontier table and the Jiang
# comparison rows, so results are comparable across checkpoints. Do not let these fall back to argparse defaults --
# the repo defaults (bfloat16, T=0.8, top_p=0.95) match no published number.
#
#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=clasp-eval-a100
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

module purge
module load "${GPU_ARCH_MODULE:-arch/a100}"
module load "${CONDA_MODULE:-miniforge/24.9.0}"
conda activate "${CONDA_ENV:-new_unsloth_env}"

PROJECT_ROOT="${CLASP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

STAGE="${STAGE:-sft}"
EVALUATION_PURPOSE="${EVALUATION_PURPOSE:-replication}"
case "${EVALUATION_PURPOSE}" in
    development|replication) ;;
    *) echo "[EvalLaunchError] unsupported EVALUATION_PURPOSE=${EVALUATION_PURPOSE}" >&2; exit 2 ;;
esac

# Frozen raw-lane representation contract. Keep every field explicit, including
# empty strings, so replication jobs cannot inherit a different prompt schema.
export CLASP_TSP_PROMPT_KNN=0
export CLASP_TSP_INPUT_ORDER=""
export CLASP_TSP_RELABEL_ORDER=""
export CLASP_TSP_ROUTE_CANONICALIZE=""
export CLASP_TSP_LABEL_TTA_SCHEDULE=""
export CLASP_MIS_RELABEL_ORDER=""
EVALUATION_REPRESENTATION_PROFILE="${EVALUATION_REPRESENTATION_PROFILE:-native}"
if [[ "${EVALUATION_REPRESENTATION_PROFILE}" == "jiang_f638_sft" ]]; then
    export CLASP_TSP_PROMPT_KNN_DISTANCES=""
    export CLASP_TSP_ROUTE_ONLY_OUTPUT=""
    export CLASP_MIS_JIANG_RESPONSE_OUTPUT=0
elif [[ "${EVALUATION_REPRESENTATION_PROFILE}" == "jiang_f638_clasp" ]]; then
    # Canonical CLASP checkpoints launched by
    # train_clasp_canonical_checkpoint_dev_a100.slurm record the normalized
    # Jiang-f638 representation explicitly.  In particular, the boolean-like
    # TSP fields are "0", not the legacy empty-string representation retained
    # above for historical SFT replication checkpoints.
    export CLASP_TSP_PROMPT_KNN_DISTANCES=0
    export CLASP_TSP_ROUTE_ONLY_OUTPUT=0
    export CLASP_MIS_JIANG_RESPONSE_OUTPUT=0
elif [[ "${EVALUATION_REPRESENTATION_PROFILE}" == "native" ]]; then
    export CLASP_TSP_PROMPT_KNN_DISTANCES=0
    export CLASP_TSP_ROUTE_ONLY_OUTPUT=0
    export CLASP_MIS_JIANG_RESPONSE_OUTPUT=1
else
    echo "[EvalLaunchError] unsupported EVALUATION_REPRESENTATION_PROFILE=${EVALUATION_REPRESENTATION_PROFILE}" >&2
    exit 2
fi
export CLASP_MIS_PROMPT_NODE_DEGREES=0
export CLASP_MIS_PROMPT_LOW_DEGREE_K=32
export CLASP_OP_STRICT_BUDGET_HINT=0
export CLASP_OP_PROMPT_BUDGET_SCALE=1.0
export CLASP_OP_PROMPT_BUDGET_SCALE_MIN_SIZE=0
export CLASP_OP_PROMPT_BUDGET_SCALE_MAX_SIZE=0
export CLASP_OP_PROMPT_BUDGET_SCALE_SIZE_RANGES=""
export CLASP_OP_SIZE_BAND_TOP_P_SCHEDULE=""
export CLASP_OP_SIZE_BAND_TOP_P_MIN_SIZE=0
export CLASP_OP_SIZE_BAND_TOP_P_MAX_SIZE=0

ASSET_ROOT="${CLASP_ASSET_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/clasp}"
DATA_ROOT="${DATA_ROOT:-${ASSET_ROOT}/data/data_rl}"

: "${CHECKPOINT_PATH:?set CHECKPOINT_PATH to the checkpoint-* directory}"
: "${CKPT_LABEL:?set CKPT_LABEL to a short name for this checkpoint}"

# One task per array index; keeps a slow task from eating another task's 2h wall.
ALL_TASKS=(tsp cvrp op mis jssp)
TASK="${TASK:-${ALL_TASKS[${SLURM_ARRAY_TASK_ID:-0}]}}"

EVAL_START_INDEX="${EVAL_START_INDEX:-200}"
MAX_EVAL_SAMPLES_PER_TASK="${MAX_EVAL_SAMPLES_PER_TASK:-100}"
SEED="${SEED:-271828}"

# --- Pinned exact protocol (see docs/audited_plan_beat_jiang_20260710.md) ---
DTYPE="${DTYPE:-float16}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-8}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"   # K=8 in ONE generate call
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3000}"
# Pinned, NOT left at 0. At 0 the budget derives from each run's max_seq_length
# (5192 for the 50k family vs 17000 for the 100k family), which truncates 8 MIS
# prompts for the 50k family and makes the families incomparable.
if [[ "${EVALUATION_REPRESENTATION_PROFILE}" == "jiang_f638_sft" \
   || "${EVALUATION_REPRESENTATION_PROFILE}" == "jiang_f638_clasp" ]]; then
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-0}"
else
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-29768}"
fi

BASE_INFERENCE_ROOT="${INFERENCE_PARENT_ROOT:-${PROJECT_ROOT}/inference/sft_baseline_eval100_20260804}"
INFERENCE_ROOT="${BASE_INFERENCE_ROOT}/${CKPT_LABEL}/${TASK}"

mkdir -p "${PROJECT_ROOT}/logs" "${INFERENCE_ROOT}"

echo "=============================================================="
echo " CLASP A100 eval"
echo "   label      : ${CKPT_LABEL}"
echo "   checkpoint : ${CHECKPOINT_PATH}"
echo "   task       : ${TASK}   (array id ${SLURM_ARRAY_TASK_ID:-none})"
echo "   purpose    : ${EVALUATION_PURPOSE}"
echo "   slice      : eval[${EVAL_START_INDEX}:$((EVAL_START_INDEX + MAX_EVAL_SAMPLES_PER_TASK))]"
echo "   protocol   : ${DTYPE} BoN=${NUM_RETURN_SEQUENCES} T=${TEMPERATURE} top_p=${TOP_P}"
echo "                top_k=${TOP_K} rep=${REPETITION_PENALTY} max_new=${MAX_NEW_TOKENS}"
echo "                max_prompt=${MAX_PROMPT_LENGTH} seed=${SEED}"
echo "   inference  : ${INFERENCE_ROOT}"
echo "   job        : ${SLURM_JOB_ID:-local}  node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=============================================================="

python -m clasp.entrypoints.evaluate \
  --evaluation_purpose "${EVALUATION_PURPOSE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --stage "${STAGE}" \
  --data_root "${DATA_ROOT}" \
  --tasks "${TASK}" \
  --eval_start_index "${EVAL_START_INDEX}" \
  --max_eval_samples_per_task "${MAX_EVAL_SAMPLES_PER_TASK}" \
  --no-prefer_subset_files \
  --seed "${SEED}" \
  --dtype "${DTYPE}" \
  --num_return_sequences "${NUM_RETURN_SEQUENCES}" \
  --generation_batch_size "${GENERATION_BATCH_SIZE}" \
  --do_sample \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --repetition_penalty "${REPETITION_PENALTY}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --prompt_style auto \
  --force_no_reason_prompt \
  --save_results \
  --save_details \
  --inference_root "${INFERENCE_ROOT}" \
  --eval_name "${CKPT_LABEL}_${TASK}_eval${EVAL_START_INDEX}_n${MAX_EVAL_SAMPLES_PER_TASK}_bon${NUM_RETURN_SEQUENCES}"

echo "EVAL_EXIT=$? label=${CKPT_LABEL} task=${TASK}"
