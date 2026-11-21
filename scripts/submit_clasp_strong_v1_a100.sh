#!/bin/bash
# Submit the create-once 2h CVRP development run for dense hybrid CLASP v1.

set -euo pipefail
umask 027

SOURCE_SNAPSHOT="$(readlink -f "${STRONG_CLASP_SOURCE_SNAPSHOT:?STRONG_CLASP_SOURCE_SNAPSHOT is required}")"
RUNTIME_ENV="$(readlink -f "${STRONG_CLASP_RUNTIME_ENV:-/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env}")"
DATA_ROOT="$(readlink -f "${STRONG_CLASP_DATA_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/primal_dual/data/data_rl}")"
OUTPUT_ROOT="${STRONG_CLASP_OUTPUT_ROOT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_strong_v1_cvrp_dev_20260904_v1}"
SFT_CHECKPOINT="$(readlink -f "${STRONG_CLASP_SFT_CHECKPOINT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_jiang_f638_sft500k_sourcev17_production_output_iclr_20260828_v14/runs/cvrp/seed42/checkpoints/checkpoint-20000}")"
MAX_STEPS="${STRONG_CLASP_MAX_STEPS:-100}"
SAVE_STEPS="${STRONG_CLASP_SAVE_STEPS:-100}"

[[ -f "${SOURCE_SNAPSHOT}/SOURCE_MANIFEST.sha256" \
    && -f "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm" ]] || {
    echo "[StrongCLASPSubmitError] source snapshot is incomplete" >&2
    exit 2
}
[[ -x "${RUNTIME_ENV}/bin/python" && -d "${DATA_ROOT}/cvrp/train" ]] || {
    echo "[StrongCLASPSubmitError] runtime environment or CVRP data is missing" >&2
    exit 2
}
[[ -f "${SFT_CHECKPOINT}/adapter_model.safetensors" \
    && -f "${SFT_CHECKPOINT}/adapter_config.json" \
    && -f "${SFT_CHECKPOINT}/checkpoint_fidelity.json" ]] || {
    echo "[StrongCLASPSubmitError] certified SFT checkpoint is incomplete" >&2
    exit 2
}
for value in "${MAX_STEPS}" "${SAVE_STEPS}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "[StrongCLASPSubmitError] invalid positive integer=${value}" >&2
        exit 2
    }
done
(( MAX_STEPS % SAVE_STEPS == 0 )) || {
    echo "[StrongCLASPSubmitError] max steps must be divisible by save steps" >&2
    exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
    echo "[StrongCLASPSubmitError] create-once output exists: ${OUTPUT_ROOT}" >&2
    exit 2
}

mkdir "${OUTPUT_ROOT}"
mkdir "${OUTPUT_ROOT}/train" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/ledgers"
OUTPUT_DIR="${OUTPUT_ROOT}/train/cvrp_clasp_strong_v1_seed42_step${MAX_STEPS}"
JOB_ID="$(sbatch --parsable --job-name=cvrp-clasp-strong-v1 \
    --output="${OUTPUT_ROOT}/logs/train_%j.out" \
    --error="${OUTPUT_ROOT}/logs/train_%j.err" \
    --export="ALL,DEV_TASK=cvrp,DEV_METHOD=clasp_strong_v1,DEV_SFT_CHECKPOINT=${SFT_CHECKPOINT},DEV_OUTPUT_DIR=${OUTPUT_DIR},DEV_PROJECT_ROOT=${SOURCE_SNAPSHOT},DEV_SEALED_ENV=${RUNTIME_ENV},DEV_DATA_ROOT=${DATA_ROOT},DEV_SEED=42,DEV_MAX_STEPS=${MAX_STEPS},DEV_SAVE_STEPS=${SAVE_STEPS},DEV_SAVE_TOTAL_LIMIT=1" \
    "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm")"
JOB_ID="${JOB_ID%%;*}"

LEDGER="${OUTPUT_ROOT}/ledgers/submission.tsv"
printf 'role\tmethod\tjob_id\tcheckpoint_or_output\n' > "${LEDGER}"
printf 'train\tclasp_strong_v1\t%s\t%s\n' \
    "${JOB_ID}" "${OUTPUT_DIR}/checkpoints/checkpoint-${MAX_STEPS}" >> "${LEDGER}"
chmod 0440 "${LEDGER}"

echo "CLASP_STRONG_V1_SUBMITTED job=${JOB_ID} output=${OUTPUT_DIR} ledger=${LEDGER}"
