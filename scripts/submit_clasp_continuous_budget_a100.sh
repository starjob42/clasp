#!/bin/bash
# Submit a clean, fixed-horizon CLASP chunk chain on Jean Zay.
#
# Each segment keeps max_steps at the task's final FOARL-matched horizon and
# stops at DEV_STOP_STEP via GRPO_STOP_AT_GLOBAL_STEP, so the linear LR schedule
# is continuous across segment boundaries.  Re-run with START_STEP / PREV_*
# to append later segments after earlier jobs have completed.
set -euo pipefail
umask 027

SOURCE_SNAPSHOT="$(readlink -f "${CONT_SOURCE_SNAPSHOT:?CONT_SOURCE_SNAPSHOT is required}")"
RUNTIME_ENV="$(readlink -f "${CONT_RUNTIME_ENV:-/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env}")"
DATA_ROOT="$(readlink -f "${CONT_DATA_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/primal_dual/data/data_rl}")"
OUTPUT_ROOT="${CONT_OUTPUT_ROOT:?CONT_OUTPUT_ROOT is required}"
TASK="${CONT_TASK:?CONT_TASK is required}"
METHOD="${CONT_METHOD:?CONT_METHOD is required}"
FINAL_STEPS="${CONT_FINAL_STEPS:?CONT_FINAL_STEPS is required}"
CHUNK_STEPS="${CONT_CHUNK_STEPS:-100}"
SAVE_STEPS="${CONT_SAVE_STEPS:-${CHUNK_STEPS}}"
SAVE_TOTAL_LIMIT="${CONT_SAVE_TOTAL_LIMIT:-3}"
START_STEP="${CONT_START_STEP:-0}"
END_STEP="${CONT_END_STEP:-${FINAL_STEPS}}"
SEED="${CONT_SEED:-43}"
PREV_JOB_ID="${CONT_PREV_JOB_ID:-}"
PREV_CHECKPOINT="${CONT_PREV_CHECKPOINT:-}"

case "${TASK}" in cvrp|op|mis|tsp|jssp) ;; *) echo "[CONTSubmitError] invalid task=${TASK}" >&2; exit 2 ;; esac
case "${METHOD}" in clasp_v9|clasp_v10|clasp_v11_cvrp|clasp_v12_mis) ;;
    *) echo "[CONTSubmitError] invalid method=${METHOD}" >&2; exit 2 ;;
esac
for value in "${FINAL_STEPS}" "${CHUNK_STEPS}" "${SAVE_STEPS}" "${SAVE_TOTAL_LIMIT}" "${START_STEP}" "${END_STEP}" "${SEED}"; do
    [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "[CONTSubmitError] invalid integer=${value}" >&2; exit 2; }
done
(( FINAL_STEPS > 0 && CHUNK_STEPS > 0 && SAVE_STEPS > 0 && SAVE_TOTAL_LIMIT > 0 )) || {
    echo "[CONTSubmitError] horizons and limits must be positive" >&2
    exit 2
}
(( FINAL_STEPS % CHUNK_STEPS == 0 && FINAL_STEPS % SAVE_STEPS == 0
    && CHUNK_STEPS % SAVE_STEPS == 0 && START_STEP % CHUNK_STEPS == 0
    && END_STEP % CHUNK_STEPS == 0 && START_STEP < END_STEP && END_STEP <= FINAL_STEPS )) || {
    echo "[CONTSubmitError] invalid chunk bounds or divisibility" >&2
    exit 2
}
[[ -f "${SOURCE_SNAPSHOT}/SOURCE_MANIFEST.sha256" \
    && -f "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm" ]] || {
    echo "[CONTSubmitError] source snapshot is incomplete" >&2
    exit 2
}
[[ -x "${RUNTIME_ENV}/bin/python" && -d "${DATA_ROOT}/${TASK}/train" ]] || {
    echo "[CONTSubmitError] runtime or task data is missing" >&2
    exit 2
}
if (( START_STEP == 0 )); then
    SFT_CHECKPOINT="$(readlink -f "${CONT_SFT_CHECKPOINT:?CONT_SFT_CHECKPOINT is required for a fresh chain}")"
    [[ -f "${SFT_CHECKPOINT}/adapter_model.safetensors" \
        && -f "${SFT_CHECKPOINT}/adapter_config.json" \
        && -f "${SFT_CHECKPOINT}/checkpoint_fidelity.json" ]] || {
        echo "[CONTSubmitError] initial SFT checkpoint is incomplete" >&2
        exit 2
    }
else
    [[ -n "${PREV_CHECKPOINT}" ]] || {
        echo "[CONTSubmitError] CONT_PREV_CHECKPOINT is required when START_STEP>0" >&2
        exit 2
    }
    PREV_CHECKPOINT="$(readlink -f "${PREV_CHECKPOINT}")"
    [[ -f "${PREV_CHECKPOINT}/adapter_model.safetensors" \
        && -f "${PREV_CHECKPOINT}/optimizer.pt" \
        && -f "${PREV_CHECKPOINT}/scheduler.pt" \
        && -f "${PREV_CHECKPOINT}/trainer_state.json" ]] || {
        echo "[CONTSubmitError] previous chunk checkpoint is incomplete: ${PREV_CHECKPOINT}" >&2
        exit 2
    }
    SFT_CHECKPOINT="${PREV_CHECKPOINT}"
fi

mkdir -p "${OUTPUT_ROOT}/train/${TASK}" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/chain"
LEDGER="${OUTPUT_ROOT}/chain/submissions.tsv"
if [[ ! -f "${LEDGER}" ]]; then
    printf 'task\tmethod\tstart\tstop\tjob_id\tdependency\tcheckpoint\n' > "${LEDGER}"
fi

previous_job="${PREV_JOB_ID}"
previous_checkpoint="${PREV_CHECKPOINT}"
for (( stop_step = START_STEP + CHUNK_STEPS; stop_step <= END_STEP; stop_step += CHUNK_STEPS )); do
    start_step=$(( stop_step - CHUNK_STEPS ))
    label="seg${start_step}_${stop_step}"
    output_dir="${OUTPUT_ROOT}/train/${TASK}/${label}"
    [[ ! -e "${output_dir}" ]] || {
        echo "[CONTSubmitError] create-once output exists: ${output_dir}" >&2
        exit 2
    }
    if (( start_step == 0 )); then
        initialization="${SFT_CHECKPOINT}"
        resume_arg=""
    else
        initialization="${previous_checkpoint}"
        resume_arg=",DEV_RESUME_CHECKPOINT=${initialization}"
    fi
    dependency_arg=()
    dependency_text="-"
    if [[ -n "${previous_job}" ]]; then
        dependency_arg=(--dependency="afterok:${previous_job}")
        dependency_text="afterok:${previous_job}"
    fi
    export_spec="ALL,DEV_TASK=${TASK},DEV_METHOD=${METHOD},DEV_SFT_CHECKPOINT=${initialization},DEV_OUTPUT_DIR=${output_dir},DEV_PROJECT_ROOT=${SOURCE_SNAPSHOT},DEV_SEALED_ENV=${RUNTIME_ENV},DEV_DATA_ROOT=${DATA_ROOT},DEV_SEED=${SEED},DEV_FINAL_MAX_STEPS=${FINAL_STEPS},DEV_STOP_STEP=${stop_step},DEV_SAVE_STEPS=${SAVE_STEPS},DEV_SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}${resume_arg}"
    job_id="$(sbatch --parsable "${dependency_arg[@]}" \
        --job-name="${TASK}-${METHOD}-s${start_step}" \
        --output="${OUTPUT_ROOT}/logs/${TASK}-${METHOD}-s${start_step}-%j.out" \
        --error="${OUTPUT_ROOT}/logs/${TASK}-${METHOD}-s${start_step}-%j.err" \
        --export="${export_spec}" \
        "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm")"
    job_id="${job_id%%;*}"
    checkpoint="${output_dir}/checkpoints/checkpoint-${stop_step}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${TASK}" "${METHOD}" "${start_step}" "${stop_step}" "${job_id}" "${dependency_text}" "${checkpoint}" >> "${LEDGER}"
    echo "submitted task=${TASK} method=${METHOD} start=${start_step} stop=${stop_step} job=${job_id} dep=${dependency_text}"
    previous_job="${job_id}"
    previous_checkpoint="${checkpoint}"
done
chmod 0640 "${LEDGER}"
echo "CONT_CHAIN_SUBMITTED task=${TASK} method=${METHOD} last_job=${previous_job} ledger=${LEDGER}"
