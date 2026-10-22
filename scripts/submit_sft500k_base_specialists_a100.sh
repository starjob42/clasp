#!/bin/bash
# Submit the complete frozen Qwen2.5-7B-base five-task SFT matrix.
# Historical Instruct jobs and their output root are intentionally untouched.

set -euo pipefail
umask 027

PROJECT_ROOT="$(readlink -f "${CLASP_PROJECT_ROOT:?CLASP_PROJECT_ROOT must name the new immutable source snapshot}")"
IMMUTABLE_INPUT_TRUST_ROOT="$(readlink -f "${BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT:?BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT is required}")"
CAMPAIGN_CONTRACT="$(readlink -f "${BASE_SFT_CAMPAIGN_CONTRACT:?BASE_SFT_CAMPAIGN_CONTRACT is required}")"
FORBIDDEN_SOURCE_REGISTRY="$(readlink -f "${BASE_SFT_FORBIDDEN_SOURCE_REGISTRY:?BASE_SFT_FORBIDDEN_SOURCE_REGISTRY is required}")"
RESERVED_SOURCE_REGISTRY="$(readlink -f "${BASE_SFT_RESERVED_SOURCE_REGISTRY:?BASE_SFT_RESERVED_SOURCE_REGISTRY is required}")"
REMOTE_ENVIRONMENT_RECEIPT="$(readlink -f "${BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT:?BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT is required}")"
SEALED_PYTHON="$(readlink -f "${BASE_SFT_PYTHON:?BASE_SFT_PYTHON must name the sealed environment interpreter}")"
MODEL_PATH="$(readlink -f "${BASE_SFT_BASE_MODEL_PATH:?BASE_SFT_BASE_MODEL_PATH is required}")"
DATA_ROOT="$(readlink -f "${BASE_SFT_DATA_ROOT:?BASE_SFT_DATA_ROOT is required}")"
OUTPUT_ROOT="$(readlink -m "${BASE_SFT_OUTPUT_ROOT:?BASE_SFT_OUTPUT_ROOT is required and must not exist}")"
WORKER="${PROJECT_ROOT}/scripts/train_sft500k_base_specialist_a100.slurm"
VALIDATOR="${PROJECT_ROOT}/scripts/validate_base_sft500k_specialist.py"
TOKEN_AUDITOR="${PROJECT_ROOT}/scripts/audit_sft_token_lengths.py"
TOKEN_AUDIT_RECEIPT="${DATA_ROOT}/sft_token_length_audit.json"
CLEANLINESS_AUDITOR="${PROJECT_ROOT}/scripts/build_clean_sft500k_corpus.py"
CLEANLINESS_AUDIT_RECEIPT="${DATA_ROOT}/sft_cleanliness_audit.json"
TASKS=(cvrp jssp mis op tsp)
NUM_STAGES=2
SBATCH_BIN="$(readlink -f "$(command -v sbatch)")"
SEALED_PREFIX="$(readlink -f "$(dirname "$(dirname "${SEALED_PYTHON}")")")"
[[ -x "${SEALED_PYTHON}" && -x "${SBATCH_BIN}" ]] || {
    echo "[BaseSFTSubmitError] sealed Python or sbatch is unavailable" >&2
    exit 2
}

for value in "${PROJECT_ROOT}" "${IMMUTABLE_INPUT_TRUST_ROOT}" "${CAMPAIGN_CONTRACT}" "${FORBIDDEN_SOURCE_REGISTRY}" \
    "${RESERVED_SOURCE_REGISTRY}" \
    "${REMOTE_ENVIRONMENT_RECEIPT}" \
    "${SEALED_PYTHON}" "${MODEL_PATH}" "${DATA_ROOT}" "${OUTPUT_ROOT}"; do
    [[ "${value}" != *$'\n'* && "${value}" != *,* ]] || {
        echo "[BaseSFTSubmitError] paths must not contain commas or newlines: ${value}" >&2
        exit 2
    }
done
[[ -f "${WORKER}" && -f "${VALIDATOR}" && -f "${TOKEN_AUDITOR}" \
    && -f "${CLEANLINESS_AUDITOR}" \
    && -f "${RESERVED_SOURCE_REGISTRY}" \
    && -f "${TOKEN_AUDIT_RECEIPT}" && -f "${CLEANLINESS_AUDIT_RECEIPT}" \
    && -f "${PROJECT_ROOT}/SOURCE_MANIFEST.sha256" ]] || {
    echo "[BaseSFTSubmitError] immutable source/data snapshot lacks worker, validator, cleanliness audit, or all-row token audit" >&2
    exit 2
}
[[ ! -w "${TOKEN_AUDIT_RECEIPT}" ]] || {
    echo "[BaseSFTSubmitError] all-row token audit must be sealed before submission: ${TOKEN_AUDIT_RECEIPT}" >&2
    exit 2
}
[[ ! -w "${CLEANLINESS_AUDIT_RECEIPT}" ]] || {
    echo "[BaseSFTSubmitError] cleanliness audit must be sealed before submission: ${CLEANLINESS_AUDIT_RECEIPT}" >&2
    exit 2
}
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] || {
    echo "[BaseSFTSubmitError] output root already exists; publication campaign is create-once: ${OUTPUT_ROOT}" >&2
    exit 2
}

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
# The cleanliness and token-length audits are create-once before campaign
# freeze. This validator call reopens both, binds every clean output hash to the
# registered 500k source, and checks all tokenizer lengths before any scheduler
# mutation.
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
printf 'campaign_id\tcampaign_sha256\ttask\tstage\tjob_id\tpredecessor\tdependency\ttime_limit\tqos\tsubmitted_at\n' >&3

for task in "${TASKS[@]}"; do
    previous=""
    for ((stage=1; stage<=NUM_STAGES; stage++)); do
        dependency="none"
        sbatch_args=(
            --parsable
            --chdir="${PROJECT_ROOT}"
            --job-name="base-sft500k-${task}-${stage}"
            --output="${OUTPUT_ROOT}/logs/${campaign_id}.${task}.stage${stage}.%j.out"
            --error="${OUTPUT_ROOT}/logs/${campaign_id}.${task}.stage${stage}.%j.err"
            --export="CLASP_PROJECT_ROOT=${PROJECT_ROOT},BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT=${IMMUTABLE_INPUT_TRUST_ROOT},BASE_SFT_CAMPAIGN_CONTRACT=${CAMPAIGN_CONTRACT},BASE_SFT_FORBIDDEN_SOURCE_REGISTRY=${FORBIDDEN_SOURCE_REGISTRY},BASE_SFT_RESERVED_SOURCE_REGISTRY=${RESERVED_SOURCE_REGISTRY},BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT=${REMOTE_ENVIRONMENT_RECEIPT},BASE_SFT_CONDA_ENV=${SEALED_PREFIX},BASE_SFT_BASE_MODEL_PATH=${MODEL_PATH},BASE_SFT_DATA_ROOT=${DATA_ROOT},BASE_SFT_OUTPUT_ROOT=${OUTPUT_ROOT},BASE_SFT_TASK=${task},BASE_SFT_PREVIOUS_JOB_ID=${previous}"
        )
        if [[ -n "${previous}" ]]; then
            dependency="afterany:${previous}"
            sbatch_args+=(--dependency="${dependency}")
        fi
        response="$("${SBATCH_BIN}" "${sbatch_args[@]}" "${WORKER}")"
        job_id="${response%%;*}"
        [[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || {
            echo "[BaseSFTSubmitError] unexpected sbatch response: ${response}" >&2
            exit 2
        }
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${campaign_id}" "${campaign_sha256}" "${task}" "${stage}" "${job_id}" \
            "${previous:-none}" "${dependency}" "20:00:00" "qos_gpu_a100-t3" \
            "$(date --iso-8601=seconds)" >&3
        echo "[BaseSFTSubmit] task=${task} stage=${stage}/${NUM_STAGES} job=${job_id} dependency=${dependency}"
        previous="${job_id}"
    done
done

finalize_ledger
trap - EXIT
echo "[BaseSFTSubmit] campaign=${campaign_id} sha256=${campaign_sha256}"
echo "[BaseSFTSubmit] immutable submission ledger=${ledger}"
