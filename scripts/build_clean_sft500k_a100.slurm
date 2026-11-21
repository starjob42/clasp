#!/bin/bash -l
# Create-once publication build of one clean 500k SFT task after a successful
# audit-only replay.  Required: CLEAN_SFT_BUILD_SOURCE_ROOT, CLEAN_SFT_PYTHON,
# CLEAN_SFT_IMMUTABLE_INPUT_ROOT, CLEAN_SFT_DATA_ROOT, CLEAN_SFT_TASK, and
# CLEAN_SFT_FINAL_AUDIT_RECEIPT.  Optional: CLEAN_SFT_REPLACEMENT_ROOT, whose
# TASK/shard-* directories are consumed in lexical order.

# The CPU partition is unavailable for this project, so the smallest A100
# development cell is used even though the builder itself is CPU-only.
#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=sft500k_clean_build_iclr

set -euo pipefail
umask 027

SOURCE_ROOT="$(readlink -f "${CLEAN_SFT_BUILD_SOURCE_ROOT:?source root is required}")"
PYTHON_BIN="$(readlink -f "${CLEAN_SFT_PYTHON:?Python is required}")"
INPUT_ROOT="$(readlink -f "${CLEAN_SFT_IMMUTABLE_INPUT_ROOT:?input root is required}")"
DATA_ROOT_RAW="${CLEAN_SFT_DATA_ROOT:?data root is required}"
TASK="${CLEAN_SFT_TASK:?task is required}"
FINAL_AUDIT="$(readlink -f "${CLEAN_SFT_FINAL_AUDIT_RECEIPT:?final audit is required}")"
REPLACEMENT_ROOT_RAW="${CLEAN_SFT_REPLACEMENT_ROOT:-}"

case "${TASK}" in cvrp|jssp|mis|op|tsp) ;; *)
    echo "[CleanSFTBuildError] invalid task=${TASK}" >&2
    exit 2
esac
case "${DATA_ROOT_RAW}" in /*) ;; *)
    echo "[CleanSFTBuildError] data root must be absolute" >&2
    exit 2
esac
[[ "${DATA_ROOT_RAW}" == *"_iclr"* ]] || {
    echo "[CleanSFTBuildError] data root must contain _iclr: ${DATA_ROOT_RAW}" >&2
    exit 2
}
[[ -x "${PYTHON_BIN}" \
   && -f "${SOURCE_ROOT}/scripts/build_clean_sft500k_corpus.py" \
   && -d "${SOURCE_ROOT}/src/clasp" \
   && -f "${FINAL_AUDIT}" \
   && ! -w "${FINAL_AUDIT}" ]] || {
    echo "[CleanSFTBuildError] source, Python, or sealed final audit is invalid" >&2
    exit 2
}

RELEASED_ROOT="${INPUT_ROOT}/released_sft500k_v1"
RESERVED_ROOT="${INPUT_ROOT}/reserved_sources_v1"
case "${TASK}" in
    cvrp)
        SOURCE_JSON="${RELEASED_ROOT}/cvrp/train_cvrp.json"
        SFT_EVAL_JSON="${RESERVED_ROOT}/sft_eval/cvrp/test.json"
        OUTPUT_NAME=train_cvrp.json
        ;;
    jssp)
        SOURCE_JSON="${RELEASED_ROOT}/jssp/training_jssp.json"
        SFT_EVAL_JSON="${RESERVED_ROOT}/sft_eval/jssp/eval.json"
        OUTPUT_NAME=training_jssp.json
        ;;
    mis)
        SOURCE_JSON="${RELEASED_ROOT}/mis/mis_train.json"
        SFT_EVAL_JSON="${RESERVED_ROOT}/sft_eval/mis/test.json"
        OUTPUT_NAME=mis_train.json
        ;;
    op)
        SOURCE_JSON="${RELEASED_ROOT}/op/train_op.json"
        SFT_EVAL_JSON="${RESERVED_ROOT}/sft_eval/op/eval.json"
        OUTPUT_NAME=train_op.json
        ;;
    tsp)
        SOURCE_JSON="${RELEASED_ROOT}/tsp/train_tsp.json"
        SFT_EVAL_JSON="${RESERVED_ROOT}/sft_eval/tsp/test.json"
        OUTPUT_NAME=train_tsp.json
        ;;
esac
RL_EVAL_JSON="${RESERVED_ROOT}/rl_eval/${TASK}/test.json"
RL_TRAIN_JSON="${RESERVED_ROOT}/rl_train/${TASK}/train_rl.json"
IID_JSON="${RESERVED_ROOT}/iid_confirmation/${TASK}/test.json"
for input_file in \
    "${SOURCE_JSON}" "${SFT_EVAL_JSON}" "${RL_EVAL_JSON}" \
    "${RL_TRAIN_JSON}" "${IID_JSON}"; do
    [[ -f "${input_file}" && ! -w "${input_file}" ]] || {
        echo "[CleanSFTBuildError] missing or writable immutable input: ${input_file}" >&2
        exit 2
    }
done

mkdir -p "${DATA_ROOT_RAW}/${TASK}/train" "${DATA_ROOT_RAW}/cleanliness_receipts"
DATA_ROOT="$(readlink -f "${DATA_ROOT_RAW}")"
[[ "${DATA_ROOT}" == "${DATA_ROOT_RAW}" ]] || {
    echo "[CleanSFTBuildError] data root must be canonical: ${DATA_ROOT_RAW}" >&2
    exit 2
}
OUTPUT_JSON="${DATA_ROOT}/${TASK}/train/${OUTPUT_NAME}"
TASK_RECEIPT="${DATA_ROOT}/cleanliness_receipts/${TASK}.json"
[[ ! -e "${OUTPUT_JSON}" && ! -e "${TASK_RECEIPT}" ]] || {
    echo "[CleanSFTBuildError] refusing to overwrite task output or receipt" >&2
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
"${PYTHON_BIN}" - "${TASK}" "${FINAL_AUDIT}" "${SOURCE_JSON}" <<'PY'
import sys
from pathlib import Path

import build_clean_sft500k_corpus as clean

task, audit_name, source_name = sys.argv[1:]
audit = clean._load_self_hashed_receipt(Path(audit_name), label="final audit-only receipt")
if (
    audit.get("task") != task
    or audit.get("status") != "all_source_rows_audited"
    or audit.get("additional_replacements_required") != 0
    or audit.get("clean_rows") != 500000
    or audit.get("source") != clean.file_identity(Path(source_name))
    or audit.get("implementation") != clean.implementation_identity()
):
    raise SystemExit("[CleanSFTBuildError] final audit does not certify this exact build")
if task == "jssp":
    observed = {
        "accepted_source_rows": audit.get("accepted_source_rows"),
        "repaired_rows": audit.get("source_repairs", {}).get("repaired_rows"),
        "replacement_rows": audit.get("replacement_rows"),
    }
    expected = {
        "accepted_source_rows": 500000,
        "repaired_rows": 6633,
        "replacement_rows": 0,
    }
    if observed != expected:
        raise SystemExit(f"[CleanSFTBuildError] unexpected JSSP audit: {observed!r}")
PY

JSSP_REPAIR_ARGS=()
if [[ "${TASK}" == "jssp" ]]; then
    JSSP_REPAIR_ARGS+=(--repair-jssp-terminal-makespan)
fi
REPLACEMENT_ARGS=()
if [[ -n "${REPLACEMENT_ROOT_RAW}" ]]; then
    REPLACEMENT_ROOT="$(readlink -f "${REPLACEMENT_ROOT_RAW}")"
    [[ "${REPLACEMENT_ROOT}" == *"_iclr"* ]] || {
        echo "[CleanSFTBuildError] replacement root must contain _iclr" >&2
        exit 2
    }
    while IFS= read -r -d '' shard_dir; do
        replacement_json="${shard_dir}/replacements.json"
        replacement_receipt="${shard_dir}/replacement_receipt.json"
        [[ -f "${replacement_json}" && -f "${replacement_receipt}" ]] || {
            echo "[CleanSFTBuildError] incomplete replacement shard: ${shard_dir}" >&2
            exit 2
        }
        REPLACEMENT_ARGS+=(
            --replacement-json "${replacement_json}"
            --replacement-receipt "${replacement_receipt}"
        )
    done < <(
        find "${REPLACEMENT_ROOT}/${TASK}" -mindepth 1 -maxdepth 1 \
            -type d -name 'shard-*' -print0 | sort -z
    )
    (( ${#REPLACEMENT_ARGS[@]} > 0 )) || {
        echo "[CleanSFTBuildError] replacement root has no ${TASK} shards" >&2
        exit 2
    }
fi

"${PYTHON_BIN}" scripts/build_clean_sft500k_corpus.py \
    --task "${TASK}" \
    --source-json "${SOURCE_JSON}" \
    --reserved-source "sft_eval,${SFT_EVAL_JSON}" \
    --reserved-source "rl_eval,${RL_EVAL_JSON}" \
    --reserved-source "rl_train,${RL_TRAIN_JSON}" \
    --reserved-source "iid_confirmation,${IID_JSON}" \
    "${JSSP_REPAIR_ARGS[@]}" \
    "${REPLACEMENT_ARGS[@]}" \
    --output-json "${OUTPUT_JSON}" \
    --receipt "${TASK_RECEIPT}" \
    --expected-rows 500000

[[ ! -w "${OUTPUT_JSON}" && ! -w "${TASK_RECEIPT}" ]] || {
    echo "[CleanSFTBuildError] builder did not seal its output and receipt" >&2
    exit 2
}
