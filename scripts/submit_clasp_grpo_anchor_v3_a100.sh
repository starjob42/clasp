#!/bin/bash
# Branch norm-matched anchored CLASP v3 from the shared v2 step-100 policy,
# continue to step 200, then evaluate classic/v2/v3 from one frozen evaluator.

set -euo pipefail
umask 027

SOURCE_SNAPSHOT="$(readlink -f "${V3_SOURCE_SNAPSHOT:?V3_SOURCE_SNAPSHOT is required}")"
RUNTIME_ENV="$(readlink -f "${V3_RUNTIME_ENV:-/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env}")"
DATA_ROOT="$(readlink -f "${V3_DATA_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/primal_dual/data/data_rl}")"
OUTPUT_ROOT="${V3_OUTPUT_ROOT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_grpo_anchor_v3_cvrp_branch_step200_dev_20260905_v1}"
V3_RESUME="$(readlink -f "${V3_RESUME_CHECKPOINT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_grpo_anchor_v2_matched_cvrp_mis_dev_20260904_v1/train/cvrp_clasp_grpo_anchor_v2_seed42_step100/checkpoints/checkpoint-100}")"
GRPO_CHECKPOINT="$(readlink -f "${V3_GRPO_CHECKPOINT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_grpo_anchor_v2_cvrp_continue_step200_dev_20260905_v1/train/cvrp_grpo_z_seed42_step200/checkpoints/checkpoint-200}")"
V2_CHECKPOINT="$(readlink -f "${V3_V2_CHECKPOINT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_grpo_anchor_v2_cvrp_continue_step200_dev_20260905_v1/train/cvrp_clasp_grpo_anchor_v2_seed42_step200/checkpoints/checkpoint-200}")"
ADDITIONAL_STEPS="${V3_ADDITIONAL_STEPS:-100}"
SAVE_STEPS="${V3_SAVE_STEPS:-100}"
EVAL_START_INDEX="${V3_EVAL_START_INDEX:-100}"
EVAL_SAMPLES="${V3_EVAL_SAMPLES:-100}"

[[ -f "${SOURCE_SNAPSHOT}/SOURCE_MANIFEST.sha256" \
    && -f "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm" \
    && -f "${SOURCE_SNAPSHOT}/scripts/evaluate_grpo_clasp_v3_triplet_a100.slurm" ]] || {
    echo "[AnchorV3SubmitError] source snapshot is incomplete" >&2
    exit 2
}
[[ -x "${RUNTIME_ENV}/bin/python" && -d "${DATA_ROOT}/cvrp/train" \
    && -d "${DATA_ROOT}/cvrp/eval" ]] || {
    echo "[AnchorV3SubmitError] runtime environment or CVRP data is missing" >&2
    exit 2
}
for value in "${ADDITIONAL_STEPS}" "${SAVE_STEPS}" "${EVAL_START_INDEX}" "${EVAL_SAMPLES}"; do
    [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo "[AnchorV3SubmitError] invalid integer=${value}" >&2
        exit 2
    }
done
(( ADDITIONAL_STEPS > 0 && SAVE_STEPS > 0 && EVAL_SAMPLES > 0 )) || {
    echo "[AnchorV3SubmitError] step and evaluation counts must be positive" >&2
    exit 2
}

RESUME_STEP="${V3_RESUME##*/checkpoint-}"
[[ "${RESUME_STEP}" =~ ^[1-9][0-9]*$ \
    && "${V3_RESUME}" == */checkpoint-"${RESUME_STEP}" ]] || {
    echo "[AnchorV3SubmitError] invalid v3 resume checkpoint path" >&2
    exit 2
}
for checkpoint in "${V3_RESUME}" "${GRPO_CHECKPOINT}" "${V2_CHECKPOINT}"; do
    for required in adapter_model.safetensors adapter_config.json checkpoint_fidelity.json; do
        [[ -f "${checkpoint}/${required}" ]] || {
            echo "[AnchorV3SubmitError] ${checkpoint} lacks ${required}" >&2
            exit 2
        }
    done
done
for required in optimizer.pt scheduler.pt trainer_state.json \
    multitask_primal_dual_state.json rng_state_0.pth rng_state_1.pth \
    rng_state_2.pth rng_state_3.pth; do
    [[ -f "${V3_RESUME}/${required}" ]] || {
        echo "[AnchorV3SubmitError] resume checkpoint lacks ${required}" >&2
        exit 2
    }
done

FINAL_STEP=$((RESUME_STEP + ADDITIONAL_STEPS))
(( FINAL_STEP % SAVE_STEPS == 0 )) || {
    echo "[AnchorV3SubmitError] final step must be divisible by save interval" >&2
    exit 2
}
[[ "${GRPO_CHECKPOINT}" == */checkpoint-"${FINAL_STEP}" \
    && "${V2_CHECKPOINT}" == */checkpoint-"${FINAL_STEP}" ]] || {
    echo "[AnchorV3SubmitError] comparator checkpoints must match final step" >&2
    exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
    echo "[AnchorV3SubmitError] create-once output exists: ${OUTPUT_ROOT}" >&2
    exit 2
}

mkdir "${OUTPUT_ROOT}"
mkdir "${OUTPUT_ROOT}/train" "${OUTPUT_ROOT}/evaluation" \
    "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/ledgers"
V3_OUTPUT="${OUTPUT_ROOT}/train/cvrp_clasp_grpo_anchor_v3_seed42_step${FINAL_STEP}"
V3_TERMINAL="${V3_OUTPUT}/checkpoints/checkpoint-${FINAL_STEP}"

TRAIN_JOB="$(sbatch --parsable --job-name=cvrp-anchor-v3-cont \
    --output="${OUTPUT_ROOT}/logs/train_v3_%j.out" \
    --error="${OUTPUT_ROOT}/logs/train_v3_%j.err" \
    --export="ALL,DEV_TASK=cvrp,DEV_METHOD=clasp_grpo_anchor_v3,DEV_SFT_CHECKPOINT=${V3_RESUME},DEV_RESUME_CHECKPOINT=${V3_RESUME},DEV_OUTPUT_DIR=${V3_OUTPUT},DEV_PROJECT_ROOT=${SOURCE_SNAPSHOT},DEV_SEALED_ENV=${RUNTIME_ENV},DEV_DATA_ROOT=${DATA_ROOT},DEV_SEED=42,DEV_MAX_STEPS=${FINAL_STEP},DEV_SAVE_STEPS=${SAVE_STEPS},DEV_SAVE_TOTAL_LIMIT=1" \
    "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm")"
TRAIN_JOB="${TRAIN_JOB%%;*}"

DEPENDENCY="afterok:${TRAIN_JOB}"
EVAL_JOB="$(sbatch --parsable --dependency="${DEPENDENCY}" \
    --job-name=cvrp-anchor-v3-triplet \
    --output="${OUTPUT_ROOT}/logs/eval_%j.out" \
    --error="${OUTPUT_ROOT}/logs/eval_%j.err" \
    --export="ALL,V3_EVAL_PROJECT_ROOT=${SOURCE_SNAPSHOT},V3_EVAL_RUNTIME_ENV=${RUNTIME_ENV},V3_EVAL_DATA_ROOT=${DATA_ROOT},V3_EVAL_OUTPUT_ROOT=${OUTPUT_ROOT}/evaluation/results,V3_EVAL_GRPO_CHECKPOINT=${GRPO_CHECKPOINT},V3_EVAL_V2_CHECKPOINT=${V2_CHECKPOINT},V3_EVAL_V3_CHECKPOINT=${V3_TERMINAL},V3_EVAL_START_INDEX=${EVAL_START_INDEX},V3_EVAL_SAMPLES=${EVAL_SAMPLES}" \
    "${SOURCE_SNAPSHOT}/scripts/evaluate_grpo_clasp_v3_triplet_a100.slurm")"
EVAL_JOB="${EVAL_JOB%%;*}"

LEDGER="${OUTPUT_ROOT}/ledgers/submission.tsv"
printf 'role\tjob_id\tdependency\tcheckpoint_or_result\n' > "${LEDGER}"
printf 'train_v3\t%s\t-\t%s\n' "${TRAIN_JOB}" "${V3_TERMINAL}" >> "${LEDGER}"
printf 'eval_triplet\t%s\t%s\t%s\n' "${EVAL_JOB}" "${DEPENDENCY}" \
    "${OUTPUT_ROOT}/evaluation/results/triplet_validation_result_index.json" >> "${LEDGER}"
chmod 0440 "${LEDGER}"

echo "ANCHOR_V3_SUBMITTED resume_step=${RESUME_STEP} additional_steps=${ADDITIONAL_STEPS} final_step=${FINAL_STEP}"
echo "ANCHOR_V3_TRAIN job=${TRAIN_JOB} checkpoint=${V3_TERMINAL}"
echo "ANCHOR_V3_EVAL job=${EVAL_JOB} dependency=${DEPENDENCY}"
echo "ANCHOR_V3_LEDGER=${LEDGER}"
