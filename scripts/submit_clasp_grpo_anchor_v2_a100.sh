#!/bin/bash
# Submit matched classic-GRPO and GRPO-anchored CLASP v2 development runs.

set -euo pipefail
umask 027

SOURCE_SNAPSHOT="$(readlink -f "${ANCHOR_SOURCE_SNAPSHOT:?ANCHOR_SOURCE_SNAPSHOT is required}")"
RUNTIME_ENV="$(readlink -f "${ANCHOR_RUNTIME_ENV:-/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env}")"
DATA_ROOT="$(readlink -f "${ANCHOR_DATA_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/primal_dual/data/data_rl}")"
OUTPUT_ROOT="${ANCHOR_OUTPUT_ROOT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_grpo_anchor_v2_matched_cvrp_mis_dev_20260904_v1}"
SFT_ROOT="$(readlink -f "${ANCHOR_SFT_ROOT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_jiang_f638_sft500k_sourcev17_production_output_iclr_20260828_v14}")"
MAX_STEPS="${ANCHOR_MAX_STEPS:-100}"
SAVE_STEPS="${ANCHOR_SAVE_STEPS:-100}"
TASKS=("cvrp" "mis")
METHODS=("grpo_z" "clasp_grpo_anchor_v2")

[[ -f "${SOURCE_SNAPSHOT}/SOURCE_MANIFEST.sha256" \
    && -f "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm" ]] || {
    echo "[AnchorV2SubmitError] source snapshot is incomplete" >&2
    exit 2
}
[[ -x "${RUNTIME_ENV}/bin/python" ]] || {
    echo "[AnchorV2SubmitError] runtime environment is missing" >&2
    exit 2
}
for value in "${MAX_STEPS}" "${SAVE_STEPS}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "[AnchorV2SubmitError] invalid positive integer=${value}" >&2
        exit 2
    }
done
(( MAX_STEPS % SAVE_STEPS == 0 )) || {
    echo "[AnchorV2SubmitError] max steps must be divisible by save steps" >&2
    exit 2
}
for task in "${TASKS[@]}"; do
    checkpoint="${SFT_ROOT}/runs/${task}/seed42/checkpoints/checkpoint-31250"
    [[ -d "${DATA_ROOT}/${task}/train" \
        && -f "${checkpoint}/adapter_model.safetensors" \
        && -f "${checkpoint}/adapter_config.json" \
        && -f "${checkpoint}/checkpoint_fidelity.json" ]] || {
        echo "[AnchorV2SubmitError] certified inputs are incomplete for task=${task}" >&2
        exit 2
    }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || {
    echo "[AnchorV2SubmitError] create-once output exists: ${OUTPUT_ROOT}" >&2
    exit 2
}

mkdir "${OUTPUT_ROOT}"
mkdir "${OUTPUT_ROOT}/train" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/ledgers"
LEDGER="${OUTPUT_ROOT}/ledgers/submission.tsv"
printf 'role\ttask\tmethod\tjob_id\tcheckpoint_or_output\n' > "${LEDGER}"

for task in "${TASKS[@]}"; do
    checkpoint="${SFT_ROOT}/runs/${task}/seed42/checkpoints/checkpoint-31250"
    for method in "${METHODS[@]}"; do
        output_dir="${OUTPUT_ROOT}/train/${task}_${method}_seed42_step${MAX_STEPS}"
        job_id="$(sbatch --parsable --job-name="${task}-${method}" \
            --output="${OUTPUT_ROOT}/logs/${task}_${method}_%j.out" \
            --error="${OUTPUT_ROOT}/logs/${task}_${method}_%j.err" \
            --export="ALL,DEV_TASK=${task},DEV_METHOD=${method},DEV_SFT_CHECKPOINT=${checkpoint},DEV_OUTPUT_DIR=${output_dir},DEV_PROJECT_ROOT=${SOURCE_SNAPSHOT},DEV_SEALED_ENV=${RUNTIME_ENV},DEV_DATA_ROOT=${DATA_ROOT},DEV_SEED=42,DEV_MAX_STEPS=${MAX_STEPS},DEV_SAVE_STEPS=${SAVE_STEPS},DEV_SAVE_TOTAL_LIMIT=1" \
            "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm")"
        job_id="${job_id%%;*}"
        printf 'train\t%s\t%s\t%s\t%s\n' \
            "${task}" "${method}" "${job_id}" \
            "${output_dir}/checkpoints/checkpoint-${MAX_STEPS}" >> "${LEDGER}"
        echo "ANCHOR_V2_SUBMITTED task=${task} method=${method} job=${job_id} output=${output_dir}"
    done
done

chmod 0440 "${LEDGER}"
echo "ANCHOR_V2_LEDGER=${LEDGER}"
