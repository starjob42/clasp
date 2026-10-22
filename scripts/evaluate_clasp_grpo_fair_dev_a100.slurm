#!/bin/bash -l
# One terminal checkpoint per A100 job; every arm uses this identical evaluator.
#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=clasp-fair-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
umask 027

SOURCE="$(readlink -f "${FAIR_EVAL_SOURCE:?}")"
RUNTIME="$(readlink -f "${FAIR_EVAL_RUNTIME:?}")"
DATA="$(readlink -f "${FAIR_EVAL_DATA:?}")"
OUTPUT="${FAIR_EVAL_OUTPUT:?}"
EVAL_TASK="${FAIR_EVAL_TASK:?}"
METHOD="${FAIR_EVAL_METHOD:?}"
TRAIN="$(readlink -f "${FAIR_EVAL_TRAIN:?}")"
case "${EVAL_TASK}" in cvrp|mis) ;; *) exit 2 ;; esac
case "${METHOD}" in grpo_z|clasp_grpo_anchor_v3|clasp_grpo_anchor_v4) ;; *) exit 2 ;; esac
[[ ! -e "${OUTPUT}" ]] || { echo "[FairEvalError] output already exists" >&2; exit 2; }
"${RUNTIME}/bin/python" "${SOURCE}/scripts/clasp_grpo_fair_dev_artifacts.py" \
    validate-training "${TRAIN}" "${EVAL_TASK}" "${METHOD}" "${SOURCE}"
mkdir "${OUTPUT}"

# Remove inherited research, decoding, import, and representation overrides before
# calling the existing evaluator. Keep the Slurm allocation and module environment.
while IFS='=' read -r inherited_name _; do
    case "${inherited_name}" in
        CLASP_*|PD_*|GRPO_*|GENERATION_*|WANDB_*|HF_*|TRANSFORMERS_*|PYTHONPATH|PYTHONHOME|RESUME_FROM_CHECKPOINT|GPU_ARCH_MODULE|CONDA_MODULE)
            unset "${inherited_name}" ;;
    esac
done < <(env)
export CLASP_PROJECT_ROOT="${SOURCE}" CONDA_ENV="${RUNTIME}" DATA_ROOT="${DATA}"
export CHECKPOINT_PATH="${TRAIN}/checkpoints/checkpoint-100"
export CKPT_LABEL="${METHOD}_seed43_step100" TASK="${EVAL_TASK}" STAGE=grpo
export EVALUATION_PURPOSE=development EVALUATION_REPRESENTATION_PROFILE=jiang_f638_clasp
export INFERENCE_PARENT_ROOT="${OUTPUT}/eval" EVAL_START_INDEX=100 MAX_EVAL_SAMPLES_PER_TASK=100
export SEED=271828 DTYPE=float16 NUM_RETURN_SEQUENCES=8 GENERATION_BATCH_SIZE=8
export TEMPERATURE=0.7 TOP_P=0.9 TOP_K=50 REPETITION_PENALTY=1.0
export MAX_NEW_TOKENS=3000 MAX_PROMPT_LENGTH=29768
export GPU_ARCH_MODULE=arch/a100 CONDA_MODULE=miniforge/24.9.0
export PYTHONPATH="${SOURCE}/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
bash "${SOURCE}/scripts/evaluate_a100.slurm"
"${RUNTIME}/bin/python" "${SOURCE}/scripts/clasp_grpo_fair_dev_artifacts.py" \
    index "${OUTPUT}" "${EVAL_TASK}" "${METHOD}" "${TRAIN}"
echo "FAIR_DEV_EVAL_COMPLETE index=${OUTPUT}/result_index.json"
