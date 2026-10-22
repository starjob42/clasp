#!/bin/bash
# Submit one matched 100-step CVRP classic-GRPO versus best-supported-CLASP
# development showdown, followed by a single paired SFT/GRPO/CLASP evaluation.

set -euo pipefail
umask 027

LEDGER=""
seal_partial_ledger() {
    if [[ -n "${LEDGER}" && -f "${LEDGER}" ]]; then chmod 0440 "${LEDGER}"; fi
}
trap seal_partial_ledger EXIT

SOURCE_SNAPSHOT="$(readlink -f "${SHOWDOWN_SOURCE_SNAPSHOT:?SHOWDOWN_SOURCE_SNAPSHOT is required}")"
RUNTIME_ENV="$(readlink -f "${SHOWDOWN_RUNTIME_ENV:-/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env}")"
DATA_ROOT="$(readlink -f "${SHOWDOWN_DATA_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/primal_dual/data/data_rl}")"
OUTPUT_ROOT="${SHOWDOWN_OUTPUT_ROOT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_grpo_clasp_showdown_20260903_v1}"
SFT_CHECKPOINT="$(readlink -f "${SHOWDOWN_SFT_CHECKPOINT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_jiang_f638_sft500k_sourcev17_production_output_iclr_20260828_v14/runs/cvrp/seed42/checkpoints/checkpoint-31250}")"
MAX_STEPS="${SHOWDOWN_MAX_STEPS:-100}"
EVAL_START_INDEX="${SHOWDOWN_EVAL_START_INDEX:-100}"
EVAL_SAMPLES="${SHOWDOWN_EVAL_SAMPLES:-100}"

[[ -f "${SOURCE_SNAPSHOT}/SOURCE_MANIFEST.sha256" \
    && -f "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm" \
    && -f "${SOURCE_SNAPSHOT}/scripts/evaluate_grpo_clasp_showdown_a100.slurm" ]] || {
    echo "[ShowdownSubmitError] source snapshot is incomplete" >&2
    exit 2
}
[[ -x "${RUNTIME_ENV}/bin/python" && -d "${DATA_ROOT}/cvrp/train" \
    && -d "${DATA_ROOT}/cvrp/eval" ]] || {
    echo "[ShowdownSubmitError] runtime environment or CVRP data is missing" >&2
    exit 2
}
[[ -f "${SFT_CHECKPOINT}/adapter_model.safetensors" \
    && -f "${SFT_CHECKPOINT}/adapter_config.json" \
    && -f "${SFT_CHECKPOINT}/checkpoint_fidelity.json" ]] || {
    echo "[ShowdownSubmitError] final SFT checkpoint is incomplete" >&2
    exit 2
}
SFT_RUN_ROOT="$(dirname "$(dirname "${SFT_CHECKPOINT}")")"
for receipt in stage_status.json completed_sft_run_seal.json sft_reload_equivalence.json adapter_persistence_receipt.json; do
    [[ -f "${SFT_RUN_ROOT}/${receipt}" ]] || {
        echo "[ShowdownSubmitError] missing completed-SFT receipt: ${SFT_RUN_ROOT}/${receipt}" >&2
        exit 2
    }
done
for value in "${MAX_STEPS}" "${EVAL_START_INDEX}" "${EVAL_SAMPLES}"; do
    [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo "[ShowdownSubmitError] invalid integer=${value}" >&2
        exit 2
    }
done
(( MAX_STEPS > 0 && EVAL_SAMPLES > 0 )) || {
    echo "[ShowdownSubmitError] train/evaluation horizons must be positive" >&2
    exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
    echo "[ShowdownSubmitError] create-once output root exists: ${OUTPUT_ROOT}" >&2
    exit 2
}

mkdir "${OUTPUT_ROOT}"
mkdir "${OUTPUT_ROOT}/train" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/ledgers"
LEDGER="${OUTPUT_ROOT}/ledgers/submission.tsv"
printf 'role\tmethod\tjob_id\tdependency\tcheckpoint_or_output\n' > "${LEDGER}"

submit_train() {
    local method="$1"
    local label="$2"
    local output="${OUTPUT_ROOT}/train/${label}"
    local job_id
    job_id="$(sbatch --parsable --job-name="cvrp-${method}" \
        --output="${OUTPUT_ROOT}/logs/train_${method}_%j.out" \
        --error="${OUTPUT_ROOT}/logs/train_${method}_%j.err" \
        --export="ALL,DEV_TASK=cvrp,DEV_METHOD=${method},DEV_SFT_CHECKPOINT=${SFT_CHECKPOINT},DEV_OUTPUT_DIR=${output},DEV_PROJECT_ROOT=${SOURCE_SNAPSHOT},DEV_SEALED_ENV=${RUNTIME_ENV},DEV_DATA_ROOT=${DATA_ROOT},DEV_SEED=42,DEV_MAX_STEPS=${MAX_STEPS},DEV_SAVE_STEPS=${MAX_STEPS},DEV_SAVE_TOTAL_LIMIT=1" \
        "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm")"
    job_id="${job_id%%;*}"
    printf 'train\t%s\t%s\t-\t%s\n' "${method}" "${job_id}" "${output}/checkpoints/checkpoint-${MAX_STEPS}" >> "${LEDGER}"
    SUBMITTED_JOB_ID="${job_id}"
}

GRPO_LABEL="cvrp_grpo_z_seed42_step${MAX_STEPS}"
CLASP_LABEL="cvrp_clasp_sequence_a05_seed42_step${MAX_STEPS}"
SUBMITTED_JOB_ID=""
submit_train grpo_z "${GRPO_LABEL}"
GRPO_JOB_ID="${SUBMITTED_JOB_ID}"
submit_train clasp_sequence_a05 "${CLASP_LABEL}"
CLASP_JOB_ID="${SUBMITTED_JOB_ID}"
GRPO_CHECKPOINT="${OUTPUT_ROOT}/train/${GRPO_LABEL}/checkpoints/checkpoint-${MAX_STEPS}"
CLASP_CHECKPOINT="${OUTPUT_ROOT}/train/${CLASP_LABEL}/checkpoints/checkpoint-${MAX_STEPS}"
DEPENDENCY="afterok:${GRPO_JOB_ID}:${CLASP_JOB_ID}"

EVAL_JOB_ID="$(sbatch --parsable --dependency="${DEPENDENCY}" --job-name=cvrp-showdown-eval \
    --output="${OUTPUT_ROOT}/logs/eval_%j.out" --error="${OUTPUT_ROOT}/logs/eval_%j.err" \
    --export="ALL,SHOWDOWN_PROJECT_ROOT=${SOURCE_SNAPSHOT},SHOWDOWN_RUNTIME_ENV=${RUNTIME_ENV},SHOWDOWN_DATA_ROOT=${DATA_ROOT},SHOWDOWN_OUTPUT_ROOT=${OUTPUT_ROOT},SHOWDOWN_SFT_CHECKPOINT=${SFT_CHECKPOINT},SHOWDOWN_GRPO_CHECKPOINT=${GRPO_CHECKPOINT},SHOWDOWN_CLASP_CHECKPOINT=${CLASP_CHECKPOINT},SHOWDOWN_EVAL_START_INDEX=${EVAL_START_INDEX},SHOWDOWN_EVAL_SAMPLES=${EVAL_SAMPLES}" \
    "${SOURCE_SNAPSHOT}/scripts/evaluate_grpo_clasp_showdown_a100.slurm")"
EVAL_JOB_ID="${EVAL_JOB_ID%%;*}"
printf 'eval\tpaired_sft_grpo_clasp\t%s\t%s\t%s\n' "${EVAL_JOB_ID}" "${DEPENDENCY}" "${OUTPUT_ROOT}/showdown_result_index.json" >> "${LEDGER}"

chmod 0440 "${LEDGER}"
echo "GRPO_CLASP_SHOWDOWN_SUBMITTED grpo_job=${GRPO_JOB_ID} clasp_job=${CLASP_JOB_ID} eval_job=${EVAL_JOB_ID} ledger=${LEDGER}"
