#!/bin/bash
# Submit two sequential 20-hour A100 stages for each missing 500k specialist.
# A completed first stage makes its dependent second stage a validated no-op.

set -euo pipefail
umask 027

PROJECT_ROOT="$(readlink -f "${CLASP_PROJECT_ROOT:-$PWD}")"
WORKER="${PROJECT_ROOT}/scripts/train_sft500k_specialist_a100.slurm"
ASSET_ROOT="${CLASP_ASSET_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/clasp}"
OUTPUT_ROOT="${SFT500K_OUTPUT_ROOT:-${ASSET_ROOT}/experiments/jiangmatched_task_sft500k_20260713_v2}"
TASKS_CSV="${SFT500K_TASKS:-op,mis,jssp}"
NUM_STAGES="${SFT500K_A100_CHAIN_STAGES:-2}"
LABEL="${SFT500K_CHAIN_LABEL:-missing_specialists_a100_$(date +%Y%m%d_%H%M%S)}"

[[ -f "${WORKER}" ]] || { echo "missing worker: ${WORKER}" >&2; exit 2; }
[[ "${NUM_STAGES}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid stage count: ${NUM_STAGES}" >&2; exit 2; }
[[ "${LABEL}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]] || { echo "unsafe label: ${LABEL}" >&2; exit 2; }

IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
for task in "${TASKS[@]}"; do
    case "${task}" in op|mis|jssp) ;; *) echo "invalid task: ${task}" >&2; exit 2 ;; esac
done

mkdir -p "${OUTPUT_ROOT}/controller" "${OUTPUT_ROOT}/logs" "${PROJECT_ROOT}/logs"
ledger="${OUTPUT_ROOT}/controller/${LABEL}_submitted_jobs.tsv"
[[ ! -e "${ledger}" ]] || { echo "ledger already exists: ${ledger}" >&2; exit 2; }
printf 'task\tstage\tjob_id\tpredecessor\tdependency\ttime_limit\tqos\tsubmitted_at\n' > "${ledger}"

for task in "${TASKS[@]}"; do
    previous=""
    for ((stage=1; stage<=NUM_STAGES; stage++)); do
        dependency="none"
        sbatch_args=(
            --parsable
            --chdir="${PROJECT_ROOT}"
            --job-name="sft500k-${task}-${stage}"
            --output="${OUTPUT_ROOT}/logs/${LABEL}.${task}.stage${stage}.%j.out"
            --error="${OUTPUT_ROOT}/logs/${LABEL}.${task}.stage${stage}.%j.err"
            --export="ALL,CLASP_PROJECT_ROOT=${PROJECT_ROOT},CLASP_ASSET_ROOT=${ASSET_ROOT},SFT500K_OUTPUT_ROOT=${OUTPUT_ROOT},SFT500K_TASK=${task},SFT500K_PREVIOUS_JOB_ID=${previous}"
        )
        if [[ -n "${previous}" ]]; then
            dependency="afterany:${previous}"
            sbatch_args+=(--dependency="${dependency}")
        fi
        response="$(sbatch "${sbatch_args[@]}" "${WORKER}")"
        job_id="${response%%;*}"
        [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "unexpected sbatch response: ${response}" >&2; exit 2; }
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${task}" "${stage}" "${job_id}" "${previous:-none}" "${dependency}" \
            "20:00:00" "qos_gpu_a100-t3" "$(date --iso-8601=seconds)" >> "${ledger}"
        echo "[SFT500kSpecialist] task=${task} stage=${stage}/${NUM_STAGES} job=${job_id} dependency=${dependency}"
        previous="${job_id}"
    done
done

echo "[SFT500kSpecialist] ledger=${ledger}"
