#!/bin/bash -l
# Jean Zay launcher for checkpoint evaluation.
#
#   sbatch --export=ALL,CHECKPOINT_PATH=/path/to/checkpoint-40 scripts/evaluate.slurm
# Exact Jiang SFT representation additionally requires:
#   EVALUATION_REPRESENTATION_PROFILE=jiang_f638_sft
#
# Scores a checkpoint with the same reward and feasibility code used in training, under the
# raw best-of-N protocol (no inference-time repair or reranking). The default purpose is
# development. Confirmation is an explicit opt-in that requires caller-supplied, reviewed
# assets; this launcher intentionally has no default confirmation paths. See README.md.
# The checked-in reference contract is currently BLOCKED; do not request confirmation until
# a new source snapshot, approved references, manifest v3, and selection receipt exist.
#
#SBATCH --account=vas@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --job-name=clasp-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# `-l` above makes this a login shell so /etc/profile.d defines `module`; do not rely on
# `sbatch --export=ALL` propagating it as an exported shell function.
module purge
module load "${GPU_ARCH_MODULE:-arch/h100}"
module load "${CONDA_MODULE:-miniforge/24.9.0}"
conda activate "${CONDA_ENV:-new_unsloth_env}"

# Slurm runs a spooled copy of this script; SLURM_SUBMIT_DIR is the submission directory.
PROJECT_ROOT="${CLASP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

STAGE="${STAGE:-grpo}"

# Frozen raw-lane representation contract. Keep every field explicit, including
# empty strings: GRPO hyperparams do not currently persist this complete set and
# an inherited cluster environment must not silently change confirmation prompts.
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

DATA_ROOT_WAS_SET="${DATA_ROOT+x}"
CHECKPOINT_PATH_WAS_SET="${CHECKPOINT_PATH+x}"
INFERENCE_ROOT_WAS_SET="${INFERENCE_ROOT+x}"
TASKS_WAS_SET="${TASKS+x}"
DATA_ROOT_EXPLICIT_VALUE="${DATA_ROOT-}"
CHECKPOINT_PATH_EXPLICIT_VALUE="${CHECKPOINT_PATH-}"
INFERENCE_ROOT_EXPLICIT_VALUE="${INFERENCE_ROOT-}"
TASKS_EXPLICIT_VALUE="${TASKS-}"

ASSET_ROOT="${CLASP_ASSET_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/clasp}"
DATA_ROOT="${DATA_ROOT:-${ASSET_ROOT}/data/data_rl}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/clasp/checkpoint-40}"
INFERENCE_ROOT="${INFERENCE_ROOT:-${PROJECT_ROOT}/inference}"
TASKS="${TASKS:-tsp,cvrp,op,mis,jssp}"
EVALUATION_PURPOSE="${EVALUATION_PURPOSE:-development}"
BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST:-}"
CHECKPOINT_SELECTION_RECEIPT="${CHECKPOINT_SELECTION_RECEIPT:-}"
CAMPAIGN_MANIFEST="${CAMPAIGN_MANIFEST:-}"
MAX_EVAL_SAMPLES_PER_TASK="${MAX_EVAL_SAMPLES_PER_TASK:-300}"
EVAL_START_INDEX="${EVAL_START_INDEX:-0}"
PREFER_SUBSET_FILES="${PREFER_SUBSET_FILES:-true}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-8}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"
SEED="${SEED:-271828}"
DTYPE="${DTYPE:-float16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3000}"
if [[ "${EVALUATION_REPRESENTATION_PROFILE}" == "jiang_f638_sft" ]]; then
  MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-0}"
else
  MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-29768}"
fi

case "${PREFER_SUBSET_FILES}" in
  true) DATA_SELECTION_ARGS=(--prefer_subset_files) ;;
  false) DATA_SELECTION_ARGS=(--no-prefer_subset_files) ;;
  *) echo "[EvalLaunchError] PREFER_SUBSET_FILES must be true or false" >&2; exit 2 ;;
esac

PURPOSE_ARGS=(--evaluation_purpose "${EVALUATION_PURPOSE}")
case "${EVALUATION_PURPOSE}" in
  development|replication)
    if [[ -n "${BENCHMARK_MANIFEST}" || -n "${CHECKPOINT_SELECTION_RECEIPT}" \
          || -n "${CAMPAIGN_MANIFEST}" ]]; then
      echo "[EvalLaunchError] benchmark/selection/campaign receipts are confirmation-only" >&2
      exit 2
    fi
    ;;
  confirmation)
    [[ "${DATA_ROOT_WAS_SET}" == x && -n "${DATA_ROOT_EXPLICIT_VALUE}" ]] || {
      echo "[EvalLaunchError] confirmation requires an explicit DATA_ROOT" >&2; exit 2;
    }
    [[ "${CHECKPOINT_PATH_WAS_SET}" == x && -n "${CHECKPOINT_PATH_EXPLICIT_VALUE}" ]] || {
      echo "[EvalLaunchError] confirmation requires an explicit CHECKPOINT_PATH" >&2; exit 2;
    }
    [[ "${INFERENCE_ROOT_WAS_SET}" == x && -n "${INFERENCE_ROOT_EXPLICIT_VALUE}" ]] || {
      echo "[EvalLaunchError] confirmation requires an explicit INFERENCE_ROOT" >&2; exit 2;
    }
    [[ "${TASKS_WAS_SET}" == x && -n "${TASKS_EXPLICIT_VALUE}" ]] || {
      echo "[EvalLaunchError] confirmation requires an explicit single specialist TASKS value" >&2; exit 2;
    }
    case "${TASKS}" in
      tsp|cvrp|op|mis|jssp) ;;
      *) echo "[EvalLaunchError] confirmation TASKS must be exactly one of tsp,cvrp,op,mis,jssp" >&2; exit 2 ;;
    esac
    [[ -n "${BENCHMARK_MANIFEST}" && -n "${CHECKPOINT_SELECTION_RECEIPT}" \
       && -n "${CAMPAIGN_MANIFEST}" ]] || {
      echo "[EvalLaunchError] confirmation requires BENCHMARK_MANIFEST, CHECKPOINT_SELECTION_RECEIPT, and CAMPAIGN_MANIFEST" >&2
      exit 2
    }
    [[ "${MAX_EVAL_SAMPLES_PER_TASK}" == 0 && "${EVAL_START_INDEX}" == 0 ]] || {
      echo "[EvalLaunchError] confirmation requires all rows (cap=0, start=0)" >&2; exit 2;
    }
    [[ "${PREFER_SUBSET_FILES}" == false ]] || {
      echo "[EvalLaunchError] confirmation requires PREFER_SUBSET_FILES=false" >&2; exit 2;
    }
    [[ "${NUM_RETURN_SEQUENCES}" == 8 && "${GENERATION_BATCH_SIZE}" == 8 ]] || {
      echo "[EvalLaunchError] raw confirmation requires K=8 in one generation call" >&2; exit 2;
    }
    [[ "${STAGE}" == grpo && "${SEED}" == 271828 && "${DTYPE}" == float16 \
       && "${TEMPERATURE}" == 0.7 && "${TOP_P}" == 0.9 && "${TOP_K}" == 50 \
       && "${REPETITION_PENALTY}" == 1.0 && "${MAX_NEW_TOKENS}" == 3000 \
       && "${MAX_PROMPT_LENGTH}" == 29768 ]] || {
      echo "[EvalLaunchError] confirmation settings differ from the frozen raw contract" >&2
      exit 2
    }
    PURPOSE_ARGS+=(
      --benchmark_manifest "${BENCHMARK_MANIFEST}"
      --checkpoint_selection_receipt "${CHECKPOINT_SELECTION_RECEIPT}"
      --campaign_manifest "${CAMPAIGN_MANIFEST}"
    )
    ;;
  *) echo "[EvalLaunchError] invalid EVALUATION_PURPOSE=${EVALUATION_PURPOSE}" >&2; exit 2 ;;
esac

mkdir -p "${PROJECT_ROOT}/logs" "${INFERENCE_ROOT}"

echo "=============================================================="
echo " CLASP eval  checkpoint=${CHECKPOINT_PATH}"
echo " job=${SLURM_JOB_ID:-local}  purpose=${EVALUATION_PURPOSE}  tasks=${TASKS}  N=${NUM_RETURN_SEQUENCES}"
echo " protocol=${DTYPE} seed=${SEED} T=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K}"
echo " generation_batch=${GENERATION_BATCH_SIZE} max_new=${MAX_NEW_TOKENS} max_prompt=${MAX_PROMPT_LENGTH}"
if [[ "${EVALUATION_PURPOSE}" == confirmation ]]; then
  echo " benchmark_manifest=${BENCHMARK_MANIFEST}"
  echo " checkpoint_selection_receipt=${CHECKPOINT_SELECTION_RECEIPT}"
  echo " campaign_manifest=${CAMPAIGN_MANIFEST}"
fi
echo "=============================================================="

# No base-model flag: for a PEFT checkpoint, clasp-eval resolves the base model from
# adapter_config.json::base_model_name_or_path. `--model_path` is an alias of
# `--checkpoint_path`, not an override, and passing both with different values is an error.
python -m clasp.entrypoints.evaluate \
  "${PURPOSE_ARGS[@]}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --stage "${STAGE}" \
  --data_root "${DATA_ROOT}" \
  --tasks "${TASKS}" \
  --max_eval_samples_per_task "${MAX_EVAL_SAMPLES_PER_TASK}" \
  --eval_start_index "${EVAL_START_INDEX}" \
  "${DATA_SELECTION_ARGS[@]}" \
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
  --resume \
  --inference_root "${INFERENCE_ROOT}"
