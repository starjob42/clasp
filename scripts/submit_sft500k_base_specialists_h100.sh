#!/bin/bash -l
# Submit five independent seven-chunk Jiang-f638 specialist chains exactly once.

set -euo pipefail
umask 027

PROJECT_ROOT="$(readlink -f "${CLASP_PROJECT_ROOT:?CLASP_PROJECT_ROOT must name the immutable source snapshot}")"
IMMUTABLE_INPUT_TRUST_ROOT="$(readlink -f "${BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT:?BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT is required}")"
CAMPAIGN_CONTRACT="$(readlink -f "${BASE_SFT_CAMPAIGN_CONTRACT:?BASE_SFT_CAMPAIGN_CONTRACT is required}")"
FORBIDDEN_SOURCE_REGISTRY="$(readlink -f "${BASE_SFT_FORBIDDEN_SOURCE_REGISTRY:?BASE_SFT_FORBIDDEN_SOURCE_REGISTRY is required}")"
RESERVED_SOURCE_REGISTRY="$(readlink -f "${BASE_SFT_RESERVED_SOURCE_REGISTRY:?BASE_SFT_RESERVED_SOURCE_REGISTRY is required}")"
REMOTE_ENVIRONMENT_RECEIPT="$(readlink -f "${BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT:?BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT is required}")"
REQUESTED_SEALED_PYTHON="$(readlink -f "${BASE_SFT_PYTHON:?BASE_SFT_PYTHON must name the sealed environment interpreter}")"
MODEL_PATH="$(readlink -f "${BASE_SFT_BASE_MODEL_PATH:?BASE_SFT_BASE_MODEL_PATH is required}")"
DATA_ROOT="$(readlink -f "${BASE_SFT_DATA_ROOT:?BASE_SFT_DATA_ROOT is required}")"
OUTPUT_ROOT="$(readlink -m "${BASE_SFT_OUTPUT_ROOT:?BASE_SFT_OUTPUT_ROOT is required and must not exist}")"
WORKER="${PROJECT_ROOT}/scripts/train_sft500k_base_specialist_h100.slurm"
VALIDATOR="${PROJECT_ROOT}/scripts/validate_base_sft500k_specialist.py"
TOKEN_AUDIT_RECEIPT="${DATA_ROOT}/sft_token_length_audit.json"
CLEANLINESS_AUDIT_RECEIPT="${DATA_ROOT}/sft_cleanliness_audit.json"
LAUNCH_PROFILE="${BASE_SFT_LAUNCH_PROFILE:?set dev2h_iclr or regular20h_iclr}"
[[ "${LAUNCH_PROFILE}" == "regular20h_iclr" ]] || {
    echo "[BaseSFTSubmitError] production submission requires regular20h_iclr; use the separate dev smoke gate first" >&2
    exit 2
}
TASKS=(cvrp jssp mis op tsp)
BOUNDARIES=(5000 10000 15000 20000 25000 30000 31250)
WALL_TIME=20:00:00
QOS=qos_gpu_h100-t4
SBATCH_BIN="$(readlink -f "$(command -v sbatch)")"
SEALED_PREFIX="$(readlink -f "$(dirname "$(dirname "${REQUESTED_SEALED_PYTHON}")")")"

# The environment receipt binds the activation state as well as package bytes.
# Recreate the same module/conda activation on the login-side preflight instead
# of invoking the prefix's Python from an unrelated inherited shell.
module purge
module load "${GPU_ARCH_MODULE:-arch/h100}"
module load "${CONDA_MODULE:-miniforge/24.9.0}"
conda activate "${SEALED_PREFIX}"
SEALED_PYTHON="$(readlink -f "${CONDA_PREFIX:?conda activation did not set CONDA_PREFIX}/bin/python")"
[[ "${SEALED_PYTHON}" == "${REQUESTED_SEALED_PYTHON}" ]] || {
    echo "[BaseSFTSubmitError] activated environment resolved a different Python" >&2
    exit 2
}

[[ -x "${SEALED_PYTHON}" && -x "${SBATCH_BIN}" \
    && -f "${WORKER}" && -f "${VALIDATOR}" \
    && -f "${RESERVED_SOURCE_REGISTRY}" \
    && -f "${TOKEN_AUDIT_RECEIPT}" \
    && -f "${CLEANLINESS_AUDIT_RECEIPT}" \
    && -f "${PROJECT_ROOT}/requirements.jiang-f638-sft.txt" \
    && -f "${PROJECT_ROOT}/SOURCE_MANIFEST.sha256" ]] || {
    echo "[BaseSFTSubmitError] frozen Jiang-f638 source/data/environment input is incomplete" >&2
    exit 2
}
[[ ! -w "${TOKEN_AUDIT_RECEIPT}" && ! -w "${CLEANLINESS_AUDIT_RECEIPT}" ]] || {
    echo "[BaseSFTSubmitError] cleanliness and all-row token receipts must be sealed" >&2
    exit 2
}
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] || {
    echo "[BaseSFTSubmitError] output root already exists; campaign is create-once: ${OUTPUT_ROOT}" >&2
    exit 2
}
[[ "${OUTPUT_ROOT}" == *_iclr* ]] || {
    echo "[BaseSFTSubmitError] output root must contain _iclr: ${OUTPUT_ROOT}" >&2
    exit 2
}

for value in "${PROJECT_ROOT}" "${IMMUTABLE_INPUT_TRUST_ROOT}" "${CAMPAIGN_CONTRACT}" \
    "${FORBIDDEN_SOURCE_REGISTRY}" "${RESERVED_SOURCE_REGISTRY}" \
    "${REMOTE_ENVIRONMENT_RECEIPT}" "${SEALED_PYTHON}" "${MODEL_PATH}" \
    "${DATA_ROOT}" "${OUTPUT_ROOT}"; do
    [[ "${value}" != *$'\n'* && "${value}" != *,* ]] || {
        echo "[BaseSFTSubmitError] paths must not contain commas or newlines: ${value}" >&2
        exit 2
    }
done

while IFS='=' read -r inherited_name _; do
    case "${inherited_name}" in
        CLASP_*|PD_*|GENERATION_*|GRPO_*|PYTORCH_*|HF_*|TRANSFORMERS_*|TOKENIZERS_*|BITSANDBYTES_*|UNSLOTH_*|ACCELERATE_*|WANDB_*|NCCL_*)
            unset "${inherited_name}"
            ;;
    esac
done < <(env)
export PATH="${SEALED_PREFIX}/bin:/usr/bin:/bin"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts"
unset LD_PRELOAD PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS || true
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

identity="$("${SEALED_PYTHON}" "${VALIDATOR}" \
    --campaign-contract "${CAMPAIGN_CONTRACT}" \
    --immutable-input-trust-root "${IMMUTABLE_INPUT_TRUST_ROOT}" \
    --source-root "${PROJECT_ROOT}" \
    --base-model-path "${MODEL_PATH}" \
    --forbidden-source-registry "${FORBIDDEN_SOURCE_REGISTRY}" \
    --reserved-source-registry "${RESERVED_SOURCE_REGISTRY}" \
    --remote-environment-receipt "${REMOTE_ENVIRONMENT_RECEIPT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --verify-campaign-only --require-matrix-quota)"
IFS=: read -r validity campaign_id campaign_sha256 <<< "${identity}"
[[ "${validity}" == VALID && "${campaign_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ \
    && "${campaign_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "[BaseSFTSubmitError] unexpected campaign validator identity: ${identity}" >&2
    exit 2
}
[[ "${campaign_id}" == *_iclr* ]] || {
    echo "[BaseSFTSubmitError] campaign id must contain _iclr: ${campaign_id}" >&2
    exit 2
}

mkdir -p "${OUTPUT_ROOT}/controller" "${OUTPUT_ROOT}/logs"
ledger="${OUTPUT_ROOT}/controller/${campaign_id}.submitted_jobs.tsv"
set -o noclobber
exec 3> "${ledger}"
set +o noclobber
ledger_open=1
finalize_ledger() {
    if [[ "${ledger_open}" == 1 ]]; then
        exec 3>&-
        chmod 0440 "${ledger}"
        ledger_open=0
    fi
}
trap finalize_ledger EXIT
printf 'campaign_id\tcampaign_sha256\ttask\ttarget_step\tjob_id\tafterok_job_id\ttime_limit\tqos\tprofile\tsubmitted_at\n' >&3

for task in "${TASKS[@]}"; do
    previous_job=""
    for target_step in "${BOUNDARIES[@]}"; do
        dependency_args=()
        dependency_record="none"
        if [[ -n "${previous_job}" ]]; then
            dependency_args=(--dependency="afterok:${previous_job}")
            dependency_record="${previous_job}"
        fi
        response="$("${SBATCH_BIN}" --parsable \
            --chdir="${PROJECT_ROOT}" \
            --job-name="jiang-sft500k-iclr-${task}-s${target_step}" \
            --time="${WALL_TIME}" \
            --qos="${QOS}" \
            "${dependency_args[@]}" \
            --output="${OUTPUT_ROOT}/logs/${campaign_id}.${task}.step${target_step}.%j.out" \
            --error="${OUTPUT_ROOT}/logs/${campaign_id}.${task}.step${target_step}.%j.err" \
            --export="CLASP_PROJECT_ROOT=${PROJECT_ROOT},BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT=${IMMUTABLE_INPUT_TRUST_ROOT},BASE_SFT_CAMPAIGN_CONTRACT=${CAMPAIGN_CONTRACT},BASE_SFT_FORBIDDEN_SOURCE_REGISTRY=${FORBIDDEN_SOURCE_REGISTRY},BASE_SFT_RESERVED_SOURCE_REGISTRY=${RESERVED_SOURCE_REGISTRY},BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT=${REMOTE_ENVIRONMENT_RECEIPT},BASE_SFT_CONDA_ENV=${SEALED_PREFIX},BASE_SFT_BASE_MODEL_PATH=${MODEL_PATH},BASE_SFT_DATA_ROOT=${DATA_ROOT},BASE_SFT_OUTPUT_ROOT=${OUTPUT_ROOT},BASE_SFT_TASK=${task},BASE_SFT_TARGET_STEP=${target_step},BASE_SFT_LAUNCH_PROFILE=${LAUNCH_PROFILE}" \
            "${WORKER}")"
        job_id="${response%%;*}"
        [[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || {
            echo "[BaseSFTSubmitError] unexpected sbatch response: ${response}" >&2
            exit 2
        }
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${campaign_id}" "${campaign_sha256}" "${task}" "${target_step}" \
            "${job_id}" "${dependency_record}" "${WALL_TIME}" "${QOS}" \
            "${LAUNCH_PROFILE}" "$(date --iso-8601=seconds)" >&3
        echo "[BaseSFTSubmit] task=${task} target_step=${target_step} job=${job_id} afterok=${dependency_record}"
        previous_job="${job_id}"
    done
done

finalize_ledger
trap - EXIT
echo "[BaseSFTSubmit] campaign=${campaign_id} sha256=${campaign_sha256}"
echo "[BaseSFTSubmit] immutable submission ledger=${ledger}"
