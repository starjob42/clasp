#!/bin/bash
# Submit the development-only CLASP vNext length-estimator screen on Jean Zay.
#
# Required:
#   PILOT_SOURCE_SNAPSHOT  immutable source snapshot containing this change
# Optional paths have the current Jiang-F638 development defaults below.
# The default grid is CVRP/MIS x {sequence alpha 0, .5, 1}, one train seed,
# 40 optimizer steps, followed by exact held-out BoN=8 evaluation.

set -euo pipefail
umask 027

LEDGER=""
seal_partial_ledger() {
    if [[ -n "${LEDGER}" && -f "${LEDGER}" ]]; then
        chmod 0440 "${LEDGER}"
    fi
}
trap seal_partial_ledger EXIT

SOURCE_SNAPSHOT="$(readlink -f "${PILOT_SOURCE_SNAPSHOT:?PILOT_SOURCE_SNAPSHOT is required}")"
SEALED_ENV="$(readlink -f "${PILOT_SEALED_ENV:-/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env}")"
DATA_ROOT="$(readlink -f "${PILOT_DATA_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/primal_dual/data/data_rl}")"
OUTPUT_ROOT="${PILOT_OUTPUT_ROOT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_vnext_length_pilot_20260901_v1}"
TASK_CSV="${PILOT_TASKS:-cvrp,mis}"
ARM_CSV="${PILOT_ARMS:-sequence_a0,sequence_a05,sequence_a1}"
SEED_CSV="${PILOT_SEEDS:-42}"
MAX_STEPS="${PILOT_MAX_STEPS:-40}"
SAVE_STEPS="${PILOT_SAVE_STEPS:-10}"
SAVE_TOTAL_LIMIT="${PILOT_SAVE_TOTAL_LIMIT:-5}"
EVAL_START_INDEX="${PILOT_EVAL_START_INDEX:-200}"
EVAL_SAMPLES="${PILOT_EVAL_SAMPLES:-100}"

[[ -f "${SOURCE_SNAPSHOT}/SOURCE_MANIFEST.sha256" \
    && -f "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm" \
    && -f "${SOURCE_SNAPSHOT}/scripts/evaluate_a100.slurm" ]] || {
    echo "[CLASPNextPilotError] source snapshot is incomplete" >&2
    exit 2
}
[[ -x "${SEALED_ENV}/bin/python" ]] || {
    echo "[CLASPNextPilotError] runtime environment is incomplete" >&2
    exit 2
}
[[ -d "${DATA_ROOT}" ]] || {
    echo "[CLASPNextPilotError] data root is missing" >&2
    exit 2
}
for numeric_value in "${MAX_STEPS}" "${SAVE_STEPS}" "${SAVE_TOTAL_LIMIT}" \
    "${EVAL_START_INDEX}" "${EVAL_SAMPLES}"; do
    [[ "${numeric_value}" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo "[CLASPNextPilotError] invalid numeric setting=${numeric_value}" >&2
        exit 2
    }
done
(( MAX_STEPS > 0 && SAVE_STEPS > 0 && SAVE_TOTAL_LIMIT > 0 && EVAL_SAMPLES > 0 )) || {
    echo "[CLASPNextPilotError] horizons and limits must be positive" >&2
    exit 2
}
(( MAX_STEPS % SAVE_STEPS == 0 )) || {
    echo "[CLASPNextPilotError] max steps must be divisible by save steps" >&2
    exit 2
}

CVRP_CHECKPOINT="${PILOT_CVRP_SFT_CHECKPOINT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_jiang_f638_sft500k_sourcev17_production_output_iclr_20260828_v14/runs/cvrp/seed42/checkpoints/checkpoint-20000}"
MIS_CHECKPOINT="${PILOT_MIS_SFT_CHECKPOINT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_jiang_f638_sft500k_sourcev17_production_output_iclr_20260828_v14/runs/mis/seed42/checkpoints/checkpoint-20000}"
TSP_CHECKPOINT="${PILOT_TSP_SFT_CHECKPOINT:-}"
OP_CHECKPOINT="${PILOT_OP_SFT_CHECKPOINT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_jiang_f638_sft500k_sourcev17_production_output_iclr_20260828_v14/runs/op/seed42/checkpoints/checkpoint-25000}"
JSSP_CHECKPOINT="${PILOT_JSSP_SFT_CHECKPOINT:-/lustre/fsn1/projects/rech/vas/uag72kz/clasp_jiang_f638_sft500k_sourcev17_production_output_iclr_20260828_v14/runs/jssp/seed42/checkpoints/checkpoint-25000}"

checkpoint_for_task() {
    case "$1" in
        cvrp) printf '%s\n' "${CVRP_CHECKPOINT}" ;;
        mis) printf '%s\n' "${MIS_CHECKPOINT}" ;;
        tsp) printf '%s\n' "${TSP_CHECKPOINT}" ;;
        op) printf '%s\n' "${OP_CHECKPOINT}" ;;
        jssp) printf '%s\n' "${JSSP_CHECKPOINT}" ;;
        *) return 2 ;;
    esac
}

IFS=',' read -r -a TASKS <<< "${TASK_CSV}"
IFS=',' read -r -a ARMS <<< "${ARM_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEED_CSV}"

mkdir -p "${OUTPUT_ROOT}/train" "${OUTPUT_ROOT}/eval" "${OUTPUT_ROOT}/ledgers"
LEDGER="${OUTPUT_ROOT}/ledgers/submission_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
printf 'task\tarm\tseed\ttrain_job_id\teval_job_id\tcheckpoint\ttrain_output\n' > "${LEDGER}"

for task in "${TASKS[@]}"; do
    checkpoint="$(checkpoint_for_task "${task}")" || {
        echo "[CLASPNextPilotError] unsupported task=${task}" >&2
        exit 2
    }
    [[ -n "${checkpoint}" && -f "${checkpoint}/adapter_model.safetensors" ]] || {
        echo "[CLASPNextPilotError] missing SFT checkpoint for task=${task}: ${checkpoint}" >&2
        exit 2
    }
    [[ -d "${DATA_ROOT}/${task}/train" ]] || {
        echo "[CLASPNextPilotError] missing training data for task=${task}" >&2
        exit 2
    }
    for arm in "${ARMS[@]}"; do
        case "${arm}" in
            token_a0_control|sequence_a0|sequence_a05|sequence_a1|sequence_a0_dualrec1|sequence_a05_dualrec1|sequence_a1_dualrec1) ;;
            *) echo "[CLASPNextPilotError] unsupported arm=${arm}" >&2; exit 2 ;;
        esac
        for seed in "${SEEDS[@]}"; do
            [[ "${seed}" =~ ^(0|[1-9][0-9]*)$ ]] || {
                echo "[CLASPNextPilotError] invalid seed=${seed}" >&2
                exit 2
            }
            label="${task}_${arm}_seed${seed}_step${MAX_STEPS}"
            train_output="${OUTPUT_ROOT}/train/${label}"
            [[ ! -e "${train_output}" ]] || {
                echo "[CLASPNextPilotError] create-once output exists: ${train_output}" >&2
                exit 2
            }

            train_job_id="$(sbatch --parsable \
                --export="ALL,DEV_TASK=${task},DEV_SFT_CHECKPOINT=${checkpoint},DEV_OUTPUT_DIR=${train_output},DEV_PROJECT_ROOT=${SOURCE_SNAPSHOT},DEV_SEALED_ENV=${SEALED_ENV},DEV_DATA_ROOT=${DATA_ROOT},DEV_SEED=${seed},DEV_MAX_STEPS=${MAX_STEPS},DEV_SAVE_STEPS=${SAVE_STEPS},DEV_SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT},DEV_ESTIMATOR_ARM=${arm}" \
                "${SOURCE_SNAPSHOT}/scripts/train_clasp_canonical_checkpoint_dev_a100.slurm")"
            train_job_id="${train_job_id%%;*}"
            terminal_checkpoint="${train_output}/checkpoints/checkpoint-${MAX_STEPS}"
            eval_job_id="$(sbatch --parsable --dependency="afterok:${train_job_id}" \
                --export="ALL,CLASP_PROJECT_ROOT=${SOURCE_SNAPSHOT},CONDA_ENV=${SEALED_ENV},DATA_ROOT=${DATA_ROOT},CHECKPOINT_PATH=${terminal_checkpoint},CKPT_LABEL=${label},TASK=${task},STAGE=grpo,EVALUATION_REPRESENTATION_PROFILE=jiang_f638_clasp,INFERENCE_PARENT_ROOT=${OUTPUT_ROOT}/eval,EVAL_START_INDEX=${EVAL_START_INDEX},MAX_EVAL_SAMPLES_PER_TASK=${EVAL_SAMPLES}" \
                "${SOURCE_SNAPSHOT}/scripts/evaluate_a100.slurm")"
            eval_job_id="${eval_job_id%%;*}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${task}" "${arm}" "${seed}" "${train_job_id}" "${eval_job_id}" \
                "${terminal_checkpoint}" "${train_output}" >> "${LEDGER}"
            echo "submitted task=${task} arm=${arm} seed=${seed} train=${train_job_id} eval=${eval_job_id}"
        done
    done
done

chmod 0440 "${LEDGER}"
echo "CLASP_VNEXT_PILOT_SUBMITTED ledger=${LEDGER}"
