#!/bin/bash
# Recover a showdown whose CLASP arm completed but whose classic-GRPO arm
# failed before training. Submit one replacement GRPO trajectory and a fresh
# paired evaluator bound to the already completed CLASP checkpoint.

set -euo pipefail
umask 027

SOURCE_SNAPSHOT="$(readlink -f "${SHOWDOWN_SOURCE_SNAPSHOT:?SHOWDOWN_SOURCE_SNAPSHOT is required}")"
RUNTIME_ENV="$(readlink -f "${SHOWDOWN_RUNTIME_ENV:-/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env}")"
DATA_ROOT="$(readlink -f "${SHOWDOWN_DATA_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/primal_dual/data/data_rl}")"
OUTPUT_ROOT="${SHOWDOWN_OUTPUT_ROOT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_grpo_clasp_showdown_20260903_v2}"
SFT_CHECKPOINT="$(readlink -f "${SHOWDOWN_SFT_CHECKPOINT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_jiang_f638_sft500k_sourcev17_production_output_iclr_20260828_v14/runs/cvrp/seed42/checkpoints/checkpoint-31250}")"
CLASP_CHECKPOINT="$(readlink -f "${SHOWDOWN_EXISTING_CLASP_CHECKPOINT:?SHOWDOWN_EXISTING_CLASP_CHECKPOINT is required}")"
MAX_STEPS="${SHOWDOWN_MAX_STEPS:-100}"
EVAL_START_INDEX="${SHOWDOWN_EVAL_START_INDEX:-100}"
EVAL_SAMPLES="${SHOWDOWN_EVAL_SAMPLES:-100}"

[[ -f "${SOURCE_SNAPSHOT}/SOURCE_MANIFEST.sha256" \
    && -f "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm" \
    && -f "${SOURCE_SNAPSHOT}/scripts/evaluate_grpo_clasp_showdown_a100.slurm" ]] || {
    echo "[ShowdownRecoveryError] immutable source snapshot is incomplete" >&2
    exit 2
}
for checkpoint in "${SFT_CHECKPOINT}" "${CLASP_CHECKPOINT}"; do
    [[ -f "${checkpoint}/adapter_model.safetensors" \
        && -f "${checkpoint}/adapter_config.json" \
        && -f "${checkpoint}/checkpoint_fidelity.json" ]] || {
        echo "[ShowdownRecoveryError] required checkpoint is incomplete: ${checkpoint}" >&2
        exit 2
    }
done
[[ -x "${RUNTIME_ENV}/bin/python" && -d "${DATA_ROOT}/cvrp/train" \
    && -d "${DATA_ROOT}/cvrp/eval" ]] || {
    echo "[ShowdownRecoveryError] runtime environment or CVRP data is missing" >&2
    exit 2
}
for value in "${MAX_STEPS}" "${EVAL_START_INDEX}" "${EVAL_SAMPLES}"; do
    [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo "[ShowdownRecoveryError] invalid integer=${value}" >&2
        exit 2
    }
done
(( MAX_STEPS > 0 && EVAL_SAMPLES > 0 )) || {
    echo "[ShowdownRecoveryError] train/evaluation horizons must be positive" >&2
    exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
    echo "[ShowdownRecoveryError] create-once output root exists: ${OUTPUT_ROOT}" >&2
    exit 2
}

mkdir "${OUTPUT_ROOT}"
mkdir "${OUTPUT_ROOT}/train" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/ledgers"
LEDGER="${OUTPUT_ROOT}/ledgers/recovery_submission.tsv"
printf 'role\tmethod\tjob_id\tdependency\tcheckpoint_or_output\n' > "${LEDGER}"
trap 'if [[ -f "${LEDGER}" ]]; then chmod 0440 "${LEDGER}"; fi' EXIT

GRPO_LABEL="cvrp_grpo_z_seed42_step${MAX_STEPS}_recovery"
GRPO_OUTPUT="${OUTPUT_ROOT}/train/${GRPO_LABEL}"
GRPO_JOB_ID="$(sbatch --parsable --job-name=cvrp-grpo-z-fix \
    --output="${OUTPUT_ROOT}/logs/train_grpo_z_%j.out" \
    --error="${OUTPUT_ROOT}/logs/train_grpo_z_%j.err" \
    --export="ALL,DEV_TASK=cvrp,DEV_METHOD=grpo_z,DEV_SFT_CHECKPOINT=${SFT_CHECKPOINT},DEV_OUTPUT_DIR=${GRPO_OUTPUT},DEV_PROJECT_ROOT=${SOURCE_SNAPSHOT},DEV_SEALED_ENV=${RUNTIME_ENV},DEV_DATA_ROOT=${DATA_ROOT},DEV_SEED=42,DEV_MAX_STEPS=${MAX_STEPS},DEV_SAVE_STEPS=${MAX_STEPS},DEV_SAVE_TOTAL_LIMIT=1" \
    "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm")"
GRPO_JOB_ID="${GRPO_JOB_ID%%;*}"
GRPO_CHECKPOINT="${GRPO_OUTPUT}/checkpoints/checkpoint-${MAX_STEPS}"
printf 'train\tgrpo_z\t%s\t-\t%s\n' "${GRPO_JOB_ID}" "${GRPO_CHECKPOINT}" >> "${LEDGER}"

EVAL_JOB_ID="$(sbatch --parsable --dependency="afterok:${GRPO_JOB_ID}" \
    --job-name=cvrp-showdown-eval-fix \
    --output="${OUTPUT_ROOT}/logs/eval_%j.out" \
    --error="${OUTPUT_ROOT}/logs/eval_%j.err" \
    --export="ALL,SHOWDOWN_PROJECT_ROOT=${SOURCE_SNAPSHOT},SHOWDOWN_RUNTIME_ENV=${RUNTIME_ENV},SHOWDOWN_DATA_ROOT=${DATA_ROOT},SHOWDOWN_OUTPUT_ROOT=${OUTPUT_ROOT},SHOWDOWN_SFT_CHECKPOINT=${SFT_CHECKPOINT},SHOWDOWN_GRPO_CHECKPOINT=${GRPO_CHECKPOINT},SHOWDOWN_CLASP_CHECKPOINT=${CLASP_CHECKPOINT},SHOWDOWN_EVAL_START_INDEX=${EVAL_START_INDEX},SHOWDOWN_EVAL_SAMPLES=${EVAL_SAMPLES}" \
    "${SOURCE_SNAPSHOT}/scripts/evaluate_grpo_clasp_showdown_a100.slurm")"
EVAL_JOB_ID="${EVAL_JOB_ID%%;*}"
printf 'eval\tpaired_sft_grpo_clasp\t%s\tafterok:%s\t%s\n' \
    "${EVAL_JOB_ID}" "${GRPO_JOB_ID}" "${OUTPUT_ROOT}/showdown_result_index.json" >> "${LEDGER}"

chmod 0440 "${LEDGER}"
trap - EXIT
echo "GRPO_SHOWDOWN_RECOVERY_SUBMITTED grpo_job=${GRPO_JOB_ID} eval_job=${EVAL_JOB_ID} ledger=${LEDGER}"
