#!/bin/bash -l
# Read-only diagnostic scan of one released 500k specialist-SFT task.
# Required: CLEAN_SFT_AUDIT_ROOT, CLEAN_SFT_PYTHON, CLEAN_SFT_TASK,
# CLEAN_SFT_SOURCE_JSON,
# CLEAN_SFT_SFT_EVAL_JSON, CLEAN_SFT_RL_EVAL_JSON,
# CLEAN_SFT_RL_TRAIN_JSON, CLEAN_SFT_IID_JSON, CLEAN_SFT_AUDIT_RECEIPT.
# Optional: CLEAN_SFT_REPLACEMENT_ROOT.  When set, every sealed
# TASK/shard-*/{replacements.json,replacement_receipt.json} pair is replayed in
# lexical shard order to prove sufficiency before the create-once build.
# Set CLEAN_SFT_REQUIRE_ZERO_DEFICIT=1 for that final sufficiency gate.
# The receipt is a count diagnostic only; this job never creates a corpus.  It
# uses the same four labeled reservation classes as the publication builder and
# applies the one permitted JSSP repair, so additional_replacements_required is
# the exact publication-build deficit rather than a legacy approximation.

# The project CPU allocation is closed, so this diagnostic uses the smallest
# available A100 development cell while performing no accelerator computation.
#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=sft500k_clean_audit_iclr

set -euo pipefail
umask 027

PROJECT_ROOT="$(readlink -f "${CLEAN_SFT_AUDIT_ROOT:?CLEAN_SFT_AUDIT_ROOT is required}")"
PYTHON_BIN="$(readlink -f "${CLEAN_SFT_PYTHON:?CLEAN_SFT_PYTHON is required}")"
TASK="${CLEAN_SFT_TASK:?CLEAN_SFT_TASK is required}"
SOURCE_JSON="$(readlink -f "${CLEAN_SFT_SOURCE_JSON:?CLEAN_SFT_SOURCE_JSON is required}")"
SFT_EVAL_JSON="$(readlink -f "${CLEAN_SFT_SFT_EVAL_JSON:?CLEAN_SFT_SFT_EVAL_JSON is required}")"
RL_EVAL_JSON="$(readlink -f "${CLEAN_SFT_RL_EVAL_JSON:?CLEAN_SFT_RL_EVAL_JSON is required}")"
RL_TRAIN_JSON="$(readlink -f "${CLEAN_SFT_RL_TRAIN_JSON:?CLEAN_SFT_RL_TRAIN_JSON is required}")"
IID_JSON="$(readlink -f "${CLEAN_SFT_IID_JSON:?CLEAN_SFT_IID_JSON is required}")"
RECEIPT="${CLEAN_SFT_AUDIT_RECEIPT:?CLEAN_SFT_AUDIT_RECEIPT is required}"
REPLACEMENT_ROOT_RAW="${CLEAN_SFT_REPLACEMENT_ROOT:-}"
REQUIRE_ZERO_DEFICIT="${CLEAN_SFT_REQUIRE_ZERO_DEFICIT:-0}"

case "${TASK}" in
    tsp|cvrp|op|mis|jssp) ;;
    *) echo "[CleanSFTAuditError] invalid task=${TASK}" >&2; exit 2 ;;
esac
[[ "${REQUIRE_ZERO_DEFICIT}" == 0 || "${REQUIRE_ZERO_DEFICIT}" == 1 ]] || {
    echo "[CleanSFTAuditError] CLEAN_SFT_REQUIRE_ZERO_DEFICIT must be 0 or 1" >&2
    exit 2
}

[[ "${RECEIPT}" == *"_iclr"* ]] || {
    echo "[CleanSFTAuditError] receipt path must contain _iclr: ${RECEIPT}" >&2
    exit 2
}

[[ -f "${PROJECT_ROOT}/scripts/build_clean_sft500k_corpus.py" \
   && -d "${PROJECT_ROOT}/src/clasp" \
   && -f "${SOURCE_JSON}" \
   && -f "${SFT_EVAL_JSON}" \
   && -f "${RL_EVAL_JSON}" \
   && -f "${RL_TRAIN_JSON}" \
   && -f "${IID_JSON}" \
   && -x "${PYTHON_BIN}" ]] || {
    echo "[CleanSFTAuditError] a required source or reserved file is missing" >&2
    exit 2
}
[[ ! -e "${RECEIPT}" ]] || {
    echo "[CleanSFTAuditError] refusing to overwrite receipt: ${RECEIPT}" >&2
    exit 2
}
mkdir -p "$(dirname "${RECEIPT}")"

module purge
module load arch/a100
module load miniforge/24.9.0
export PATH="$(dirname "${PYTHON_BIN}"):/usr/bin:/bin"
unset PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS || true
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts"

cd "${PROJECT_ROOT}"
JSSP_REPAIR_ARGS=()
if [[ "${TASK}" == "jssp" ]]; then
    JSSP_REPAIR_ARGS+=(--repair-jssp-terminal-makespan)
fi
REPLACEMENT_ARGS=()
if [[ -n "${REPLACEMENT_ROOT_RAW}" ]]; then
    REPLACEMENT_ROOT="$(readlink -f "${REPLACEMENT_ROOT_RAW}")"
    [[ "${REPLACEMENT_ROOT}" == *"_iclr"* ]] || {
        echo "[CleanSFTAuditError] replacement root must contain _iclr" >&2
        exit 2
    }
    while IFS= read -r -d '' shard_dir; do
        replacement_json="${shard_dir}/replacements.json"
        replacement_receipt="${shard_dir}/replacement_receipt.json"
        [[ -f "${replacement_json}" && -f "${replacement_receipt}" ]] || {
            echo "[CleanSFTAuditError] incomplete replacement shard: ${shard_dir}" >&2
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
        echo "[CleanSFTAuditError] replacement root has no ${TASK} shards" >&2
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
    --receipt "${RECEIPT}" \
    --expected-rows 500000

"${PYTHON_BIN}" - "${TASK}" "${RECEIPT}" "${REQUIRE_ZERO_DEFICIT}" <<'PY'
import json
import sys

task, path, require_zero_text = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    receipt = json.load(handle)
observed = {
    "source_rows": receipt["source_rows"],
    "accepted_source_rows": receipt["accepted_source_rows"],
    "repaired_rows": receipt["source_repairs"]["repaired_rows"],
    "replacement_rows": receipt["replacement_rows"],
    "replacement_rows_considered": receipt["replacement_rows_considered"],
    "additional_replacements_required": receipt["additional_replacements_required"],
}
if task == "jssp":
    expected = {
        "source_rows": 500000,
        "accepted_source_rows": 500000,
        "repaired_rows": 6633,
        "replacement_rows": 0,
        "replacement_rows_considered": 0,
        "additional_replacements_required": 0,
    }
    if observed != expected:
        raise SystemExit(f"[CleanSFTAuditError] unexpected JSSP recovery: {observed!r}")
if require_zero_text == "1" and observed["additional_replacements_required"] != 0:
    raise SystemExit(f"[CleanSFTAuditError] replacement sufficiency failed: {observed!r}")
print(json.dumps({"task": task, **observed}, sort_keys=True))
PY
