#!/bin/bash -l
# Merge the five clean-task receipts and freeze both source registries. Required:
# CLEAN_SFT_FINALIZE_SOURCE_ROOT, CLEAN_SFT_PYTHON,
# CLEAN_SFT_IMMUTABLE_INPUT_ROOT, CLEAN_SFT_DATA_ROOT,
# CLEAN_SFT_RESERVED_SOURCE_REGISTRY, and CLEAN_SFT_FORBIDDEN_SOURCE_REGISTRY.

#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=sft500k_finalize_iclr

set -euo pipefail
umask 027

SOURCE_ROOT="$(readlink -f "${CLEAN_SFT_FINALIZE_SOURCE_ROOT:?source root is required}")"
PYTHON_BIN="$(readlink -f "${CLEAN_SFT_PYTHON:?Python is required}")"
INPUT_ROOT="$(readlink -f "${CLEAN_SFT_IMMUTABLE_INPUT_ROOT:?input root is required}")"
DATA_ROOT="$(readlink -f "${CLEAN_SFT_DATA_ROOT:?data root is required}")"
RESERVED_REGISTRY="${CLEAN_SFT_RESERVED_SOURCE_REGISTRY:?reserved registry is required}"
FORBIDDEN_REGISTRY="${CLEAN_SFT_FORBIDDEN_SOURCE_REGISTRY:?forbidden registry is required}"

for output_path in "${DATA_ROOT}" "${RESERVED_REGISTRY}" "${FORBIDDEN_REGISTRY}"; do
    [[ "${output_path}" == *"_iclr"* ]] || {
        echo "[CleanSFTFinalizeError] output namespace must contain _iclr: ${output_path}" >&2
        exit 2
    }
done
[[ -x "${PYTHON_BIN}" \
   && -f "${SOURCE_ROOT}/scripts/build_clean_sft500k_corpus.py" \
   && -f "${SOURCE_ROOT}/scripts/freeze_sft_reserved_source_registry.py" \
   && -f "${SOURCE_ROOT}/scripts/freeze_forbidden_source_registry.py" ]] || {
    echo "[CleanSFTFinalizeError] source or Python is incomplete" >&2
    exit 2
}

RESERVED_ROOT="${INPUT_ROOT}/reserved_sources_v1"
IID_RECEIPT="${RESERVED_ROOT}/iid_confirmation/generation_receipt.json"
MERGED_RECEIPT="${DATA_ROOT}/sft_cleanliness_audit.json"

module purge
module load arch/a100
module load miniforge/24.9.0
export PATH="$(dirname "${PYTHON_BIN}"):/usr/bin:/bin"
unset PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS || true
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SOURCE_ROOT}/src:${SOURCE_ROOT}/scripts"
cd "${SOURCE_ROOT}"

if [[ ! -e "${MERGED_RECEIPT}" ]]; then
    "${PYTHON_BIN}" scripts/build_clean_sft500k_corpus.py \
        --data-root "${DATA_ROOT}" \
        --merge-task-receipt "${DATA_ROOT}/cleanliness_receipts/cvrp.json" \
        --merge-task-receipt "${DATA_ROOT}/cleanliness_receipts/jssp.json" \
        --merge-task-receipt "${DATA_ROOT}/cleanliness_receipts/mis.json" \
        --merge-task-receipt "${DATA_ROOT}/cleanliness_receipts/op.json" \
        --merge-task-receipt "${DATA_ROOT}/cleanliness_receipts/tsp.json" \
        --receipt "${MERGED_RECEIPT}"
else
    "${PYTHON_BIN}" - "${MERGED_RECEIPT}" "${DATA_ROOT}" <<'PY'
import sys
from pathlib import Path
from build_clean_sft500k_corpus import validate_merged_receipt
validate_merged_receipt(Path(sys.argv[1]), expected_data_root=Path(sys.argv[2]))
PY
fi

reserved_source_args=()
for task in cvrp jssp mis op tsp; do
    for source_class in sft_eval rl_eval rl_train iid_confirmation; do
        case "${source_class}:${task}" in
            sft_eval:cvrp) relative=sft_eval/cvrp/test.json ;;
            sft_eval:jssp) relative=sft_eval/jssp/eval.json ;;
            sft_eval:mis) relative=sft_eval/mis/test.json ;;
            sft_eval:op) relative=sft_eval/op/eval.json ;;
            sft_eval:tsp) relative=sft_eval/tsp/test.json ;;
            rl_eval:*) relative="rl_eval/${task}/test.json" ;;
            rl_train:*) relative="rl_train/${task}/train_rl.json" ;;
            iid_confirmation:*) relative="iid_confirmation/${task}/test.json" ;;
        esac
        source_path="${RESERVED_ROOT}/${relative}"
        [[ -f "${source_path}" && ! -w "${source_path}" ]] || {
            echo "[CleanSFTFinalizeError] invalid reserved source: ${source_path}" >&2
            exit 2
        }
        reserved_source_args+=(--source "${task},${source_class},${source_path}")
    done
done
if [[ ! -e "${RESERVED_REGISTRY}" ]]; then
    "${PYTHON_BIN}" scripts/freeze_sft_reserved_source_registry.py \
        --immutable-input-trust-root "${INPUT_ROOT}" \
        "${reserved_source_args[@]}" \
        --iid-generation-receipt "${IID_RECEIPT}" \
        --output "${RESERVED_REGISTRY}"
else
    "${PYTHON_BIN}" - "${RESERVED_REGISTRY}" "${INPUT_ROOT}" <<'PY'
import sys
from pathlib import Path
from freeze_sft_reserved_source_registry import validate_registry
validate_registry(Path(sys.argv[1]), expected_trust_root=Path(sys.argv[2]))
PY
fi

forbidden_source_args=()
for task in cvrp jssp mis op tsp; do
    forbidden_source_args+=(
        --source "${task},sft-train,${DATA_ROOT}/${task}/train"
        --source "${task},rl-train,${RESERVED_ROOT}/rl_train/${task}/train_rl.json"
        --source "${task},historical-eval,${RESERVED_ROOT}/rl_eval/${task}/test.json"
    )
done
if [[ ! -e "${FORBIDDEN_REGISTRY}" ]]; then
    "${PYTHON_BIN}" scripts/freeze_forbidden_source_registry.py \
        "${forbidden_source_args[@]}" \
        --base-sft-post-iid-generation-receipt "${IID_RECEIPT}" \
        --output "${FORBIDDEN_REGISTRY}"
else
    "${PYTHON_BIN}" - "${FORBIDDEN_REGISTRY}" <<'PY'
import json
import sys
from pathlib import Path
from freeze_forbidden_source_registry import validate_base_sft_source_registry
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
validate_base_sft_source_registry(
    payload,
    expected_tasks=("cvrp", "jssp", "mis", "op", "tsp"),
)
PY
fi

for sealed in "${MERGED_RECEIPT}" "${RESERVED_REGISTRY}" "${FORBIDDEN_REGISTRY}"; do
    [[ -f "${sealed}" && ! -w "${sealed}" ]] || {
        echo "[CleanSFTFinalizeError] final artifact is missing or writable: ${sealed}" >&2
        exit 2
    }
done
