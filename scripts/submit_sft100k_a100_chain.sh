#!/bin/bash
# Submit sequential 20-hour A100 allocations.  The account exposes A100 t3
# (not t4), so forty stages cover the measured remaining training time with
# margin.  Once step 156250 is finalized, every remaining dependency is a no-op.

set -euo pipefail
umask 027

PROJECT_ROOT="$(readlink -f "${CLASP_PROJECT_ROOT:-$PWD}")"
WORKER="${PROJECT_ROOT}/scripts/continue_sft100k_a100.slurm"
RUN_DIR="${SFT100K_RUN_DIR:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp/experiments/jiangmatched_unified_oneadapter_sft100kpertask_attnlora_dev2h_chain5_20260717_v1/seed42}"
NUM_STAGES="${SFT100K_A100_CHAIN_STAGES:-40}"
LABEL="${SFT100K_A100_CHAIN_LABEL:-epoch1_to5_a100_$(date +%Y%m%d_%H%M%S)}"

[[ -f "${WORKER}" ]] || { echo "missing worker: ${WORKER}" >&2; exit 2; }
[[ "${NUM_STAGES}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid stage count: ${NUM_STAGES}" >&2; exit 2; }
[[ "${LABEL}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]] || { echo "unsafe label: ${LABEL}" >&2; exit 2; }

mkdir -p "${RUN_DIR}/controller" "${RUN_DIR}/logs" "${PROJECT_ROOT}/logs"
ledger="${RUN_DIR}/controller/${LABEL}_submitted_jobs.tsv"
[[ ! -e "${ledger}" ]] || { echo "ledger already exists: ${ledger}" >&2; exit 2; }
printf 'stage\tjob_id\tpredecessor\tdependency\ttime_limit\tqos\tsubmitted_at\n' > "${ledger}"

previous=""
for ((stage=1; stage<=NUM_STAGES; stage++)); do
    sbatch_args=(
        --parsable
        --chdir="${PROJECT_ROOT}"
        --job-name="sft100k-e5-a100-${stage}"
        --output="${RUN_DIR}/logs/${LABEL}.stage${stage}.%j.out"
        --error="${RUN_DIR}/logs/${LABEL}.stage${stage}.%j.err"
        --export="ALL,CLASP_PROJECT_ROOT=${PROJECT_ROOT},SFT100K_RUN_DIR=${RUN_DIR},SFT100K_PREVIOUS_JOB_ID=${previous}"
    )
    dependency="none"
    if [[ -n "${previous}" ]]; then
        dependency="afterany:${previous}"
        sbatch_args+=(--dependency="${dependency}")
    fi
    response="$(sbatch "${sbatch_args[@]}" "${WORKER}")"
    job_id="${response%%;*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "unexpected sbatch response: ${response}" >&2; exit 2; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${stage}" "${job_id}" "${previous:-none}" "${dependency}" \
        "20:00:00" "qos_gpu_a100-t3" "$(date --iso-8601=seconds)" >> "${ledger}"
    echo "[SFT100kContinuation] stage=${stage}/${NUM_STAGES} job=${job_id} dependency=${dependency}"
    previous="${job_id}"
done

echo "[SFT100kContinuation] ledger=${ledger} final_job=${previous}"
