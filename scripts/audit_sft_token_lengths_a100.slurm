#!/bin/bash -l
# One exact all-row token-audit shard, or the create-once five-shard merge.
# Required: CLEAN_SFT_TOKEN_SOURCE_ROOT, CLEAN_SFT_PYTHON,
# CLEAN_SFT_DATA_ROOT, CLEAN_SFT_BASE_MODEL_PATH,
# CLEAN_SFT_FORBIDDEN_SOURCE_REGISTRY, CLEAN_SFT_TOKEN_SHARD_ROOT, and
# CLEAN_SFT_TOKEN_AUDIT_MODE (task or merge). Task mode also requires
# CLEAN_SFT_TOKEN_AUDIT_TASK.

#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=sft500k_token_audit_iclr

set -euo pipefail
umask 027

SOURCE_ROOT="$(readlink -f "${CLEAN_SFT_TOKEN_SOURCE_ROOT:?source root is required}")"
PYTHON_BIN="$(readlink -f "${CLEAN_SFT_PYTHON:?Python is required}")"
DATA_ROOT="$(readlink -f "${CLEAN_SFT_DATA_ROOT:?data root is required}")"
BASE_MODEL="$(readlink -f "${CLEAN_SFT_BASE_MODEL_PATH:?base model is required}")"
FORBIDDEN_REGISTRY="$(readlink -f "${CLEAN_SFT_FORBIDDEN_SOURCE_REGISTRY:?forbidden registry is required}")"
SHARD_ROOT_RAW="${CLEAN_SFT_TOKEN_SHARD_ROOT:?token shard root is required}"
MODE="${CLEAN_SFT_TOKEN_AUDIT_MODE:?token audit mode is required}"

[[ "${DATA_ROOT}" == *"_iclr"* && "${SHARD_ROOT_RAW}" == *"_iclr"* ]] || {
    echo "[SFTTokenAuditLaunchError] data and shard roots must contain _iclr" >&2
    exit 2
}
[[ -x "${PYTHON_BIN}" \
   && -f "${SOURCE_ROOT}/scripts/audit_sft_token_lengths.py" \
   && -f "${FORBIDDEN_REGISTRY}" \
   && ! -w "${FORBIDDEN_REGISTRY}" ]] || {
    echo "[SFTTokenAuditLaunchError] source, Python, or registry is invalid" >&2
    exit 2
}
case "${MODE}" in task|merge) ;; *)
    echo "[SFTTokenAuditLaunchError] mode must be task or merge" >&2
    exit 2
esac

mkdir -p "${SHARD_ROOT_RAW}"
SHARD_ROOT="$(readlink -f "${SHARD_ROOT_RAW}")"
[[ "${SHARD_ROOT}" == "${SHARD_ROOT_RAW}" ]] || {
    echo "[SFTTokenAuditLaunchError] shard root must be canonical" >&2
    exit 2
}

module purge
module load arch/a100
module load miniforge/24.9.0
export PATH="$(dirname "${PYTHON_BIN}"):/usr/bin:/bin"
unset PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS || true
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SOURCE_ROOT}/src:${SOURCE_ROOT}/scripts"
cd "${SOURCE_ROOT}"

COMMON_ARGS=(
    --data-root "${DATA_ROOT}"
    --tokenizer-path "${BASE_MODEL}"
    --forbidden-source-registry "${FORBIDDEN_REGISTRY}"
    --max-seq-length 20000
    --expected-rows-per-task 500000
)
if [[ "${MODE}" == "task" ]]; then
    TASK="${CLEAN_SFT_TOKEN_AUDIT_TASK:?task is required in task mode}"
    case "${TASK}" in cvrp|jssp|mis|op|tsp) ;; *)
        echo "[SFTTokenAuditLaunchError] invalid task=${TASK}" >&2
        exit 2
    esac
    OUTPUT="${SHARD_ROOT}/${TASK}.json"
    [[ ! -e "${OUTPUT}" ]] || {
        echo "[SFTTokenAuditLaunchError] refusing to overwrite shard: ${OUTPUT}" >&2
        exit 2
    }
    "${PYTHON_BIN}" scripts/audit_sft_token_lengths.py \
        "${COMMON_ARGS[@]}" \
        --batch-size 256 \
        --task "${TASK}" \
        --output "${OUTPUT}"
else
    OUTPUT="${DATA_ROOT}/sft_token_length_audit.json"
    [[ ! -e "${OUTPUT}" ]] || {
        echo "[SFTTokenAuditLaunchError] refusing to overwrite receipt: ${OUTPUT}" >&2
        exit 2
    }
    for task in cvrp jssp mis op tsp; do
        [[ -f "${SHARD_ROOT}/${task}.json" && ! -w "${SHARD_ROOT}/${task}.json" ]] || {
            echo "[SFTTokenAuditLaunchError] missing or writable ${task} shard" >&2
            exit 2
        }
    done
    "${PYTHON_BIN}" scripts/audit_sft_token_lengths.py \
        "${COMMON_ARGS[@]}" \
        --merge-shard-dir "${SHARD_ROOT}" \
        --output "${OUTPUT}"
fi

[[ -f "${OUTPUT}" && ! -w "${OUTPUT}" ]] || {
    echo "[SFTTokenAuditLaunchError] token audit output is missing or writable" >&2
    exit 2
}
