#!/bin/bash
# Submit one create-once paired GRPO-vs-CLASP CVRP validation evaluation.

set -euo pipefail
umask 027

SOURCE_SNAPSHOT="$(readlink -f "${PAIR_EVAL_SOURCE_SNAPSHOT:?PAIR_EVAL_SOURCE_SNAPSHOT is required}")"
RUNTIME_ENV="$(readlink -f "${PAIR_EVAL_RUNTIME_ENV:-/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env}")"
DATA_ROOT="$(readlink -f "${PAIR_EVAL_DATA_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/primal_dual/data/data_rl}")"
OUTPUT_ROOT="${PAIR_EVAL_OUTPUT_ROOT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_grpo_clasp_pair_validation_20260903_v1}"
GRPO_CHECKPOINT="$(readlink -f "${PAIR_EVAL_GRPO_CHECKPOINT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_grpo_clasp_showdown_20260903_v2/train/cvrp_grpo_z_seed42_step100_recovery/checkpoints/checkpoint-100}")"
CLASP_CHECKPOINT="$(readlink -f "${PAIR_EVAL_CLASP_CHECKPOINT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_grpo_clasp_showdown_20260903_v1/train/cvrp_clasp_sequence_a05_seed42_step100/checkpoints/checkpoint-100}")"
EVAL_START_INDEX="${PAIR_EVAL_START_INDEX:-100}"
EVAL_SAMPLES="${PAIR_EVAL_SAMPLES:-100}"

[[ -f "${SOURCE_SNAPSHOT}/SOURCE_MANIFEST.sha256" \
    && -f "${SOURCE_SNAPSHOT}/scripts/evaluate_grpo_clasp_pair_a100.slurm" ]] || {
    echo "[PairEvalSubmitError] immutable source snapshot is incomplete" >&2
    exit 2
}
for checkpoint in "${GRPO_CHECKPOINT}" "${CLASP_CHECKPOINT}"; do
    [[ -f "${checkpoint}/adapter_model.safetensors" \
        && -f "${checkpoint}/checkpoint_fidelity.json" ]] || {
        echo "[PairEvalSubmitError] checkpoint is incomplete: ${checkpoint}" >&2
        exit 2
    }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || {
    echo "[PairEvalSubmitError] create-once output exists: ${OUTPUT_ROOT}" >&2
    exit 2
}

mkdir "${OUTPUT_ROOT}"
mkdir "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/ledgers"
JOB_ID="$(sbatch --parsable --job-name=cvrp-grpo-clasp-val \
    --output="${OUTPUT_ROOT}/logs/eval_%j.out" \
    --error="${OUTPUT_ROOT}/logs/eval_%j.err" \
    --export="ALL,PAIR_EVAL_PROJECT_ROOT=${SOURCE_SNAPSHOT},PAIR_EVAL_RUNTIME_ENV=${RUNTIME_ENV},PAIR_EVAL_DATA_ROOT=${DATA_ROOT},PAIR_EVAL_OUTPUT_ROOT=${OUTPUT_ROOT}/results,PAIR_EVAL_GRPO_CHECKPOINT=${GRPO_CHECKPOINT},PAIR_EVAL_CLASP_CHECKPOINT=${CLASP_CHECKPOINT},PAIR_EVAL_START_INDEX=${EVAL_START_INDEX},PAIR_EVAL_SAMPLES=${EVAL_SAMPLES}" \
    "${SOURCE_SNAPSHOT}/scripts/evaluate_grpo_clasp_pair_a100.slurm")"
JOB_ID="${JOB_ID%%;*}"
LEDGER="${OUTPUT_ROOT}/ledgers/submission.tsv"
printf 'job_id\ttask\tstart\tcount\tgrpo_checkpoint\tclasp_checkpoint\tresult_index\n' > "${LEDGER}"
printf '%s\tcvrp\t%s\t%s\t%s\t%s\t%s\n' "${JOB_ID}" "${EVAL_START_INDEX}" "${EVAL_SAMPLES}" \
    "${GRPO_CHECKPOINT}" "${CLASP_CHECKPOINT}" "${OUTPUT_ROOT}/results/pair_validation_result_index.json" >> "${LEDGER}"
chmod 0440 "${LEDGER}"
echo "GRPO_CLASP_PAIR_VALIDATION_SUBMITTED job=${JOB_ID} ledger=${LEDGER}"
