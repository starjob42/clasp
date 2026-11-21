#!/bin/bash
# Continue matched CVRP classic-GRPO and anchored-CLASP-v2 checkpoints, then
# evaluate both terminal policies on the same held-out exact BoN=8 slice.

set -euo pipefail
umask 027

SOURCE_SNAPSHOT="$(readlink -f "${CONT_SOURCE_SNAPSHOT:?CONT_SOURCE_SNAPSHOT is required}")"
RUNTIME_ENV="$(readlink -f "${CONT_RUNTIME_ENV:-/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env}")"
DATA_ROOT="$(readlink -f "${CONT_DATA_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/primal_dual/data/data_rl}")"
OUTPUT_ROOT="${CONT_OUTPUT_ROOT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_grpo_anchor_v2_cvrp_continue_step200_dev_20260905_v1}"
GRPO_RESUME="$(readlink -f "${CONT_GRPO_RESUME:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_grpo_clasp_showdown_20260903_v2/train/cvrp_grpo_z_seed42_step100_recovery/checkpoints/checkpoint-100}")"
CLASP_RESUME="$(readlink -f "${CONT_CLASP_RESUME:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_grpo_anchor_v2_matched_cvrp_mis_dev_20260904_v1/train/cvrp_clasp_grpo_anchor_v2_seed42_step100/checkpoints/checkpoint-100}")"
ADDITIONAL_STEPS="${CONT_ADDITIONAL_STEPS:-100}"
SAVE_STEPS="${CONT_SAVE_STEPS:-100}"
EVAL_START_INDEX="${CONT_EVAL_START_INDEX:-100}"
EVAL_SAMPLES="${CONT_EVAL_SAMPLES:-100}"

[[ -f "${SOURCE_SNAPSHOT}/SOURCE_MANIFEST.sha256" \
    && -f "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm" \
    && -f "${SOURCE_SNAPSHOT}/scripts/evaluate_grpo_clasp_pair_a100.slurm" ]] || {
    echo "[AnchorV2ContinuationSubmitError] source snapshot is incomplete" >&2
    exit 2
}
[[ -x "${RUNTIME_ENV}/bin/python" && -d "${DATA_ROOT}/cvrp/train" \
    && -d "${DATA_ROOT}/cvrp/eval" ]] || {
    echo "[AnchorV2ContinuationSubmitError] runtime environment or CVRP data is missing" >&2
    exit 2
}
for value in "${ADDITIONAL_STEPS}" "${SAVE_STEPS}" "${EVAL_START_INDEX}" "${EVAL_SAMPLES}"; do
    [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo "[AnchorV2ContinuationSubmitError] invalid nonnegative integer=${value}" >&2
        exit 2
    }
done
(( ADDITIONAL_STEPS > 0 && SAVE_STEPS > 0 && EVAL_SAMPLES > 0 )) || {
    echo "[AnchorV2ContinuationSubmitError] steps, save interval, and eval count must be positive" >&2
    exit 2
}

resume_step=""
for checkpoint in "${GRPO_RESUME}" "${CLASP_RESUME}"; do
    step="${checkpoint##*/checkpoint-}"
    [[ "${step}" =~ ^[1-9][0-9]*$ && "${checkpoint}" == */checkpoint-"${step}" ]] || {
        echo "[AnchorV2ContinuationSubmitError] invalid checkpoint path=${checkpoint}" >&2
        exit 2
    }
    for required in adapter_model.safetensors adapter_config.json checkpoint_fidelity.json \
        optimizer.pt scheduler.pt trainer_state.json multitask_primal_dual_state.json \
        rng_state_0.pth rng_state_1.pth rng_state_2.pth rng_state_3.pth; do
        [[ -f "${checkpoint}/${required}" ]] || {
            echo "[AnchorV2ContinuationSubmitError] ${checkpoint} lacks ${required}" >&2
            exit 2
        }
    done
    if [[ -z "${resume_step}" ]]; then
        resume_step="${step}"
    elif [[ "${step}" != "${resume_step}" ]]; then
        echo "[AnchorV2ContinuationSubmitError] matched arms must resume at the same step" >&2
        exit 2
    fi
done

FINAL_STEP=$((resume_step + ADDITIONAL_STEPS))
(( FINAL_STEP % SAVE_STEPS == 0 )) || {
    echo "[AnchorV2ContinuationSubmitError] final step must be divisible by save interval" >&2
    exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
    echo "[AnchorV2ContinuationSubmitError] create-once output exists: ${OUTPUT_ROOT}" >&2
    exit 2
}

mkdir "${OUTPUT_ROOT}"
mkdir "${OUTPUT_ROOT}/train" "${OUTPUT_ROOT}/evaluation" \
    "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/ledgers"
LEDGER="${OUTPUT_ROOT}/ledgers/submission.tsv"
printf 'role\tmethod\tjob_id\tdependency\tcheckpoint_or_result\n' > "${LEDGER}"

submit_train() {
    local method="$1"
    local checkpoint="$2"
    local output_dir="${OUTPUT_ROOT}/train/cvrp_${method}_seed42_step${FINAL_STEP}"
    local job_id
    job_id="$(sbatch --parsable --job-name="cvrp-${method}-cont" \
        --output="${OUTPUT_ROOT}/logs/train_${method}_%j.out" \
        --error="${OUTPUT_ROOT}/logs/train_${method}_%j.err" \
        --export="ALL,DEV_TASK=cvrp,DEV_METHOD=${method},DEV_SFT_CHECKPOINT=${checkpoint},DEV_RESUME_CHECKPOINT=${checkpoint},DEV_OUTPUT_DIR=${output_dir},DEV_PROJECT_ROOT=${SOURCE_SNAPSHOT},DEV_SEALED_ENV=${RUNTIME_ENV},DEV_DATA_ROOT=${DATA_ROOT},DEV_SEED=42,DEV_MAX_STEPS=${FINAL_STEP},DEV_SAVE_STEPS=${SAVE_STEPS},DEV_SAVE_TOTAL_LIMIT=1" \
        "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm")"
    job_id="${job_id%%;*}"
    printf 'train\t%s\t%s\t-\t%s\n' "${method}" "${job_id}" \
        "${output_dir}/checkpoints/checkpoint-${FINAL_STEP}" >> "${LEDGER}"
    printf '%s\n' "${job_id}"
}

GRPO_JOB="$(submit_train grpo_z "${GRPO_RESUME}")"
CLASP_JOB="$(submit_train clasp_grpo_anchor_v2 "${CLASP_RESUME}")"
GRPO_TERMINAL="${OUTPUT_ROOT}/train/cvrp_grpo_z_seed42_step${FINAL_STEP}/checkpoints/checkpoint-${FINAL_STEP}"
CLASP_TERMINAL="${OUTPUT_ROOT}/train/cvrp_clasp_grpo_anchor_v2_seed42_step${FINAL_STEP}/checkpoints/checkpoint-${FINAL_STEP}"
DEPENDENCY="afterok:${GRPO_JOB}:${CLASP_JOB}"

EVAL_JOB="$(sbatch --parsable --dependency="${DEPENDENCY}" \
    --job-name=cvrp-anchor-v2-step${FINAL_STEP}-eval \
    --output="${OUTPUT_ROOT}/logs/eval_%j.out" \
    --error="${OUTPUT_ROOT}/logs/eval_%j.err" \
    --export="ALL,PAIR_EVAL_PROJECT_ROOT=${SOURCE_SNAPSHOT},PAIR_EVAL_RUNTIME_ENV=${RUNTIME_ENV},PAIR_EVAL_DATA_ROOT=${DATA_ROOT},PAIR_EVAL_OUTPUT_ROOT=${OUTPUT_ROOT}/evaluation/results,PAIR_EVAL_GRPO_CHECKPOINT=${GRPO_TERMINAL},PAIR_EVAL_CLASP_CHECKPOINT=${CLASP_TERMINAL},PAIR_EVAL_GRPO_LABEL=grpo_z_step${FINAL_STEP},PAIR_EVAL_CLASP_LABEL=clasp_grpo_anchor_v2_step${FINAL_STEP},PAIR_EVAL_START_INDEX=${EVAL_START_INDEX},PAIR_EVAL_SAMPLES=${EVAL_SAMPLES}" \
    "${SOURCE_SNAPSHOT}/scripts/evaluate_grpo_clasp_pair_a100.slurm")"
EVAL_JOB="${EVAL_JOB%%;*}"
RESULT_INDEX="${OUTPUT_ROOT}/evaluation/results/pair_validation_result_index.json"
printf 'eval\tpaired_exact_bon8\t%s\t%s\t%s\n' "${EVAL_JOB}" "${DEPENDENCY}" \
    "${RESULT_INDEX}" >> "${LEDGER}"
chmod 0440 "${LEDGER}"

echo "ANCHOR_V2_CONTINUATION_SUBMITTED resume_step=${resume_step} additional_steps=${ADDITIONAL_STEPS} final_step=${FINAL_STEP}"
echo "ANCHOR_V2_CONTINUATION_TRAIN classic_job=${GRPO_JOB} clasp_job=${CLASP_JOB}"
echo "ANCHOR_V2_CONTINUATION_EVAL job=${EVAL_JOB} dependency=${DEPENDENCY}"
echo "ANCHOR_V2_CONTINUATION_LEDGER=${LEDGER}"
