#!/bin/bash -l
# Independent raw BoN=8 evaluation; requires the completed overlap audit.
#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=clasp-fresh-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
umask 027
SOURCE="$(readlink -f "${FRESH_SOURCE:?}")"
MANAGEMENT="$(readlink -f "${FRESH_MANAGEMENT_ROOT:?}")"
RUNTIME="$(readlink -f "${FRESH_RUNTIME:?}")"
DATA="$(readlink -f "${FRESH_DATA_ROOT:?}")"
CHECKPOINT="$(readlink -f "${FRESH_CHECKPOINT:?}")"
OUTPUT="${FRESH_OUTPUT:?}"
EVAL_TASK="${FRESH_TASK:?}"
METHOD="${FRESH_METHOD:?}"
TRAIN="${FRESH_TRAIN:-}"
case "${EVAL_TASK}" in cvrp|mis) ;; *) exit 2 ;; esac
case "${METHOD}" in
    sft) EVAL_STAGE=sft; REPRESENTATION=jiang_f638_sft ;;
    grpo_z|v3|v4) EVAL_STAGE=grpo; REPRESENTATION=jiang_f638_clasp ;;
    *) exit 2 ;;
esac
HELPER="${MANAGEMENT}/scripts/clasp_grpo_fresh_dev_artifacts.py"
ARGS=(launch --source "${SOURCE}" --management "${MANAGEMENT}" --runtime "${RUNTIME}"
      --data "${DATA}" --checkpoint "${CHECKPOINT}" --output "${OUTPUT}"
      --task "${EVAL_TASK}" --method "${METHOD}" --running-wrapper "${BASH_SOURCE[0]}")
if [[ -n "${TRAIN}" ]]; then ARGS+=(--train "${TRAIN}"); fi
while IFS='=' read -r inherited_name _; do
    case "${inherited_name}" in
        CLASP_*|PD_*|GRPO_*|GENERATION_*|WANDB_*|HF_*|TRANSFORMERS_*|PYTHONPATH|PYTHONHOME|RESUME_FROM_CHECKPOINT|GPU_ARCH_MODULE|CONDA_MODULE)
            unset "${inherited_name}" ;;
    esac
done < <(env)
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"${RUNTIME}/bin/python" "${HELPER}" "${ARGS[@]}"
export CLASP_PROJECT_ROOT="${SOURCE}" CONDA_ENV="${RUNTIME}" DATA_ROOT="${DATA}"
export CHECKPOINT_PATH="${CHECKPOINT}" CKPT_LABEL="${METHOD}_fresh_seed271828"
export TASK="${EVAL_TASK}" STAGE="${EVAL_STAGE}" EVALUATION_PURPOSE=development
export EVALUATION_REPRESENTATION_PROFILE="${REPRESENTATION}" INFERENCE_PARENT_ROOT="${OUTPUT}/eval"
export EVAL_START_INDEX=0 MAX_EVAL_SAMPLES_PER_TASK=100
export SEED=271828 DTYPE=float16 NUM_RETURN_SEQUENCES=8 GENERATION_BATCH_SIZE=8
export TEMPERATURE=0.7 TOP_P=0.9 TOP_K=50 REPETITION_PENALTY=1.0
export MAX_NEW_TOKENS=3000 MAX_PROMPT_LENGTH=29768
export GPU_ARCH_MODULE=arch/a100 CONDA_MODULE=miniforge/24.9.0
export PYTHONPATH="${SOURCE}/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
bash "${SOURCE}/scripts/evaluate_a100.slurm"
"${RUNTIME}/bin/python" "${HELPER}" index --output "${OUTPUT}"
echo "FRESH_DEV_EVAL_COMPLETE index=${OUTPUT}/result_index.json"
