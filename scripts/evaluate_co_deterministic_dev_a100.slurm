#!/bin/bash -l
# Explicit completed classic/FOARL control/teacher checkpoint100/200/400, raw K8 dev100.
# Required CO_EVAL_SOURCE, CO_EVAL_SOURCE_SHA, CO_TRAIN_ROOT, CO_TRAIN_JOB_ID,
# CO_TASK, CO_METHOD, CO_STEP, CO_EVAL_OUTPUT. No submission or retry here.
#SBATCH --account=vas@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --no-requeue
#SBATCH --job-name=co-deterministic-raw-dev
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
umask 027
EVAL_SOURCE="$(readlink -f "${CO_EVAL_SOURCE:?}")"
EVAL_OUTPUT="${CO_EVAL_OUTPUT:?}"
TRAIN_ROOT="$(readlink -f "${CO_TRAIN_ROOT:?}")"
EVAL_TASK="${CO_TASK:?}"
EVAL_METHOD="${CO_METHOD:?}"
EVAL_STEP="${CO_STEP:?}"
RUNTIME=/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env
case "${EVAL_TASK}" in cvrp|mis) ;; *) exit 2 ;; esac
case "${EVAL_METHOD}" in classic_grpo|foarl_control|foarl_teacher) ;; *) exit 2 ;; esac
case "${EVAL_STEP}" in 100|200|400) ;; *) exit 2 ;; esac
while IFS='=' read -r name _; do
    case "${name}" in
        CLASP_*|PD_*|GRPO_*|FOARL_*|GENERATION_*|WANDB_*|HF_*|TRANSFORMERS_*|TOKENIZERS_*|BITSANDBYTES_*|UNSLOTH_*|ACCELERATE_*|NCCL_*|PYTORCH_*|PYTHONPATH|PYTHONHOME|RESUME_FROM_CHECKPOINT|GPU_ARCH_MODULE|CONDA_MODULE)
            unset "${name}" ;;
    esac
done < <(env)
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"${RUNTIME}/bin/python" -B "${EVAL_SOURCE}/scripts/evaluate_co_deterministic_dev.py" launch \
    --source "${EVAL_SOURCE}" --source-sha "${CO_EVAL_SOURCE_SHA:?}" \
    --train "${TRAIN_ROOT}" --training-job-id "${CO_TRAIN_JOB_ID:?}" \
    --task "${EVAL_TASK}" --method "${EVAL_METHOD}" --step "${EVAL_STEP}" \
    --output "${EVAL_OUTPUT}" --running-wrapper "${BASH_SOURCE[0]}"
export CLASP_PROJECT_ROOT="${EVAL_SOURCE}" CONDA_ENV="${RUNTIME}"
export DATA_ROOT=/lustre/fsn1/projects/rech/vas/uag72kz/clasp_fair_development_benchmark_20260905_v1/data_v3
export CHECKPOINT_PATH="${TRAIN_ROOT}/checkpoints/checkpoint-${EVAL_STEP}"
export CKPT_LABEL="${EVAL_METHOD}_step${EVAL_STEP}_dev100_seed271828"
export TASK="${EVAL_TASK}" STAGE=grpo EVALUATION_PURPOSE=development
export EVALUATION_REPRESENTATION_PROFILE=jiang_f638_clasp INFERENCE_PARENT_ROOT="${EVAL_OUTPUT}/eval"
export EVAL_START_INDEX=0 MAX_EVAL_SAMPLES_PER_TASK=100
export SEED=271828 DTYPE=float16 NUM_RETURN_SEQUENCES=8 GENERATION_BATCH_SIZE=8
export TEMPERATURE=0.7 TOP_P=0.9 TOP_K=50 REPETITION_PENALTY=1.0
export MAX_NEW_TOKENS=3000 MAX_PROMPT_LENGTH=29768
export GPU_ARCH_MODULE=arch/a100 CONDA_MODULE=miniforge/24.9.0
export PYTHONPATH="${EVAL_SOURCE}/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
bash "${EVAL_SOURCE}/scripts/evaluate_a100.slurm"
"${RUNTIME}/bin/python" -B "${EVAL_SOURCE}/scripts/evaluate_co_deterministic_dev.py" index --output "${EVAL_OUTPUT}"
echo "CO_DETERMINISTIC_RAW_DEV_COMPLETE step=${EVAL_STEP}"
