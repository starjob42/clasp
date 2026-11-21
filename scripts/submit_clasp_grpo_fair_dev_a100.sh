#!/bin/bash
# Fixed, fresh-SFT GRPO/v3/v4 development comparison: 2 tasks x 3 methods.
# Set FAIR_SOURCE_SNAPSHOT and FAIR_OUTPUT_ROOT; FAIR_VALIDATE_ONLY=1 checks
# inputs without creating outputs or submitting jobs. All numerical knobs are fixed.
# Only available QoS slots are filled. Run the printed resume command later to
# submit remaining training and evaluations as preceding jobs finish.
set -euo pipefail
umask 027

SOURCE="$(readlink -f "${FAIR_SOURCE_SNAPSHOT:?FAIR_SOURCE_SNAPSHOT is required}")"
OUTPUT="${FAIR_OUTPUT_ROOT:?FAIR_OUTPUT_ROOT is required}"
RUNTIME="$(readlink -f "${FAIR_RUNTIME_ENV:-/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env}")"
DATA="$(readlink -f "${FAIR_DATA_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/primal_dual/data/data_rl}")"
SFT="$(readlink -f "${FAIR_SFT_ROOT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_jiang_f638_sft500k_sourcev17_production_output_iclr_20260828_v14}")"
VALIDATE_ONLY="${FAIR_VALIDATE_ONLY:-0}"
[[ "${VALIDATE_ONLY}" == 0 || "${VALIDATE_ONLY}" == 1 ]] || exit 2
for path in "${SOURCE}" "${OUTPUT}" "${RUNTIME}" "${DATA}" "${SFT}"; do
    [[ "${path}" == /* && "${path}" != *[[:space:],]* ]] || {
        echo "[FairDevError] absolute paths without spaces or commas are required" >&2
        exit 2
    }
done
[[ ! -e "${OUTPUT}" && -x "${RUNTIME}/bin/python" ]] || {
    echo "[FairDevError] output already exists or Python is missing" >&2
    exit 2
}

PREFLIGHT="$(mktemp /tmp/clasp_fair_preflight.XXXXXXXX.json)"
trap 'rm -f "${PREFLIGHT}"' EXIT
"${RUNTIME}/bin/python" "${SOURCE}/scripts/clasp_grpo_fair_dev_artifacts.py" \
    preflight "${SOURCE}" "${RUNTIME}" "${DATA}" "${SFT}" > "${PREFLIGHT}"
if [[ "${VALIDATE_ONLY}" == 1 ]]; then
    cat "${PREFLIGHT}"
    echo "FAIR_DEV_VALIDATED no_jobs_submitted=1"
    exit 0
fi

mkdir "${OUTPUT}"
mkdir "${OUTPUT}/train" "${OUTPUT}/evaluation" "${OUTPUT}/logs" "${OUTPUT}/ledgers"
cp "${PREFLIGHT}" "${OUTPUT}/ledgers/protocol.json"
chmod 0440 "${OUTPUT}/ledgers/protocol.json"
LEDGER="${OUTPUT}/ledgers/submission.tsv"
printf 'role\ttask\tmethod\tjob_id\tdependency\tcheckpoint_or_output\n' > "${LEDGER}"
chmod 0440 "${LEDGER}"
# All accepted/rejected actions are written as separate immutable JSON events;
# six dependent evals must not occupy the ten-slot QoS alongside six train jobs.
"${RUNTIME}/bin/python" "${SOURCE}/scripts/resume_clasp_grpo_fair_dev.py" \
    --output "${OUTPUT}" --submit --max-submit-jobs 10 --max-new-jobs 6
echo "FAIR_DEV_LEDGER=${LEDGER}"
echo "FAIR_DEV_EVENTS=${OUTPUT}/ledgers/orchestration_events"
printf 'FAIR_DEV_RESUME_COMMAND='
printf '%q ' "${RUNTIME}/bin/python" "${SOURCE}/scripts/resume_clasp_grpo_fair_dev.py" \
    --output "${OUTPUT}" --submit --max-submit-jobs 10 --max-new-jobs 6
printf '\n'
