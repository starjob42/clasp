#!/bin/bash -l
# Raw common-prompt K8 evaluation of one declared, verified 100-update endpoint.
# The campaign controller validates launch and final raw-file cross-bindings.
#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=incumbent-raw-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
umask 027
CAMPAIGN="$(readlink -f "${INCUMBENT_CAMPAIGN:?}")"
SOURCE="$(readlink -f "${INCUMBENT_SOURCE:?}")"
RUNTIME="$(readlink -f "${INCUMBENT_RUNTIME:?}")"
OUTPUT="${INCUMBENT_OUTPUT:?}"
EVAL_TASK="${INCUMBENT_TASK:?}"
METHOD="${INCUMBENT_METHOD:?}"
case "${EVAL_TASK}" in cvrp|mis) ;; *) exit 2 ;; esac
case "${METHOD}" in grpo_z|grpo_verified|foarl_f638_dev) ;; *) exit 2 ;; esac
HELPER="${SOURCE}/scripts/grpo_incumbent_dev_campaign.py"
while IFS='=' read -r inherited_name _; do
    case "${inherited_name}" in
        CLASP_*|PD_*|GRPO_*|FOARL_*|GENERATION_*|WANDB_*|HF_*|TRANSFORMERS_*|TOKENIZERS_*|BITSANDBYTES_*|UNSLOTH_*|ACCELERATE_*|NCCL_*|PYTORCH_*|PYTHONPATH|PYTHONHOME|RESUME_FROM_CHECKPOINT|GPU_ARCH_MODULE|CONDA_MODULE)
            unset "${inherited_name}" ;;
    esac
done < <(env)
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"${RUNTIME}/bin/python" "${HELPER}" eval-launch --output "${CAMPAIGN}" \
    --key "${EVAL_TASK}/${METHOD}" --running-wrapper "${BASH_SOURCE[0]}"
# Values come from the immutable declaration, never an inherited decoder knob.
readarray -t DECLARED < <("${RUNTIME}/bin/python" - "${CAMPAIGN}" "${EVAL_TASK}" "${METHOD}" <<'PY'
import json
import pathlib
import sys
root, task, method = sys.argv[1:]
protocol = json.loads((pathlib.Path(root) / "ledgers/protocol.json").read_text())
plan = next(plan for plan in protocol["planned_eval_commands"] if plan["key"] == task + "/" + method)
print(protocol["evaluation"]["data_root"])
print(plan["checkpoint"])
PY
)
[[ "${#DECLARED[@]}" == 2 ]] || { echo "[IncumbentEvalError] declaration paths missing" >&2; exit 2; }
export CLASP_PROJECT_ROOT="${SOURCE}" CONDA_ENV="${RUNTIME}" DATA_ROOT="${DECLARED[0]}"
export CHECKPOINT_PATH="${DECLARED[1]}" CKPT_LABEL="${METHOD}_incumbent_seed271828"
export TASK="${EVAL_TASK}" STAGE=grpo EVALUATION_PURPOSE=development
export EVALUATION_REPRESENTATION_PROFILE=jiang_f638_clasp INFERENCE_PARENT_ROOT="${OUTPUT}/eval"
export EVAL_START_INDEX=0 MAX_EVAL_SAMPLES_PER_TASK=100
export SEED=271828 DTYPE=float16 NUM_RETURN_SEQUENCES=8 GENERATION_BATCH_SIZE=8
export TEMPERATURE=0.7 TOP_P=0.9 TOP_K=50 REPETITION_PENALTY=1.0
export MAX_NEW_TOKENS=3000 MAX_PROMPT_LENGTH=29768
export GPU_ARCH_MODULE=arch/a100 CONDA_MODULE=miniforge/24.9.0
export PYTHONPATH="${SOURCE}/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
bash "${SOURCE}/scripts/evaluate_a100.slurm"
"${RUNTIME}/bin/python" "${HELPER}" eval-index --output "${CAMPAIGN}" --key "${EVAL_TASK}/${METHOD}"
echo "INCUMBENT_RAW_EVAL_COMPLETE index=${OUTPUT}/result_index.json"
