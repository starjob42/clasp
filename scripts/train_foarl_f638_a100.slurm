#!/bin/bash -l
# One corrected-FOARL publication trajectory.  This launcher accepts exactly
# foarl_f638_native or foarl_f638_matched_b9 and exactly one specialist task.
# It never submits another job; re-submit the same immutable tuple to resume.
# The matched-B9 lane is a common-attribution port, never a reproduction claim.
#
#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --job-name=foarl-f638
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 027

TASK="${FOARL_TASK:?FOARL_TASK is required (one of tsp/cvrp/op/mis/jssp)}"
METHOD="${FOARL_METHOD:?FOARL_METHOD is required (foarl_f638_native or foarl_f638_matched_b9)}"
TRAIN_SEED="${FOARL_TRAIN_SEED:?FOARL_TRAIN_SEED is required}"
CAMPAIGN_ID="${FOARL_CAMPAIGN_ID:?FOARL_CAMPAIGN_ID is required and must be unique}"
case "${TASK}" in tsp|cvrp|op|mis|jssp) ;; *) echo "[FOARLLaunchError] invalid task=${TASK}" >&2; exit 2 ;; esac
case "${METHOD}" in foarl_f638_native|foarl_f638_matched_b9) ;; *) echo "[FOARLLaunchError] invalid method=${METHOD}" >&2; exit 2 ;; esac
[[ "${TRAIN_SEED}" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "[FOARLLaunchError] seed must be a canonical nonnegative decimal" >&2; exit 2; }
[[ "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "[FOARLLaunchError] unsafe campaign ID" >&2; exit 2; }

module purge
module load arch/a100
module load miniforge/24.9.0
SEALED_ENV="$(readlink -f "${FOARL_SEALED_ENV:?FOARL_SEALED_ENV is required}")"
REMOTE_ENVIRONMENT_RECEIPT="$(readlink -f "${FOARL_REMOTE_ENVIRONMENT_RECEIPT:?FOARL_REMOTE_ENVIRONMENT_RECEIPT is required}")"
[[ -d "${SEALED_ENV}" && -f "${REMOTE_ENVIRONMENT_RECEIPT}" ]] || { echo "[FOARLLaunchError] sealed environment/receipt is missing" >&2; exit 2; }
[[ ! -w "${SEALED_ENV}" && ! -w "$(dirname "${SEALED_ENV}")" ]] || { echo "[FOARLLaunchError] sealed environment and parent must be read-only" >&2; exit 2; }
conda activate "${SEALED_ENV}"
SEALED_PYTHON="$(readlink -f "${SEALED_ENV}/bin/python")"
[[ "$(readlink -f "$(command -v python)")" == "${SEALED_PYTHON}" ]] || { echo "[FOARLLaunchError] activated Python is not the sealed interpreter" >&2; exit 2; }
export PATH="${SEALED_ENV}/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

# Resolve operator-supplied artifact paths before sealing the research/runtime
# environment.  Every scientific path is subsequently rebound by a receipt.
PROJECT_ROOT="$(readlink -f "${CLASP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}")"
STORAGE_ROOT="${CLASP_STORAGE_ROOT:?CLASP_STORAGE_ROOT is required and must be a unique campaign root}"
SFT_MODEL_PATH="$(readlink -f "${FOARL_SFT_MODEL_PATH:?FOARL_SFT_MODEL_PATH must name the task specialist corrected SFT}")"
BASE_MODEL_PATH="$(readlink -f "${FOARL_BASE_MODEL_PATH:?FOARL_BASE_MODEL_PATH is required}")"
SFT_DATA_ROOT="$(readlink -f "${FOARL_SFT_DATA_ROOT:?FOARL_SFT_DATA_ROOT is required}")"
TASK_DATA_ROOT="$(readlink -f "${FOARL_RL_DATA_ROOT:?FOARL_RL_DATA_ROOT is required}")"
FORBIDDEN_SOURCE_REGISTRY="$(readlink -f "${FOARL_FORBIDDEN_SOURCE_REGISTRY:?FOARL_FORBIDDEN_SOURCE_REGISTRY is required}")"
MODEL_INITIALIZATION_REGISTRY="$(readlink -f "${FOARL_MODEL_INITIALIZATION_REGISTRY:?FOARL_MODEL_INITIALIZATION_REGISTRY is required}")"
ADVANCEMENT_POLICY="$(readlink -f "${FOARL_ADVANCEMENT_POLICY:?FOARL_ADVANCEMENT_POLICY is required}")"
PRIOR_STAGE_RECEIPTS_JSON="${FOARL_PRIOR_STAGE_RECEIPTS_JSON:-[]}"
KL_REFERENCE_MODEL_PATH="$(readlink -f "${FOARL_KL_REFERENCE_MODEL_PATH:?FOARL_KL_REFERENCE_MODEL_PATH is required}")"
KL_REFERENCE_RECEIPT="$(readlink -f "${FOARL_KL_REFERENCE_RECEIPT:?FOARL_KL_REFERENCE_RECEIPT is required}")"
RUN_DIR="${STORAGE_ROOT}/runs/${TASK}/${METHOD}/seed${TRAIN_SEED}"
CACHE_ROOT="${FOARL_CACHE_ROOT:-${STORAGE_ROOT}/cache/${TASK}/${METHOD}/seed${TRAIN_SEED}}"
WANDB_PROJECT_NAME="${FOARL_WANDB_PROJECT:-foarl_f638_publication}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src"
[[ -f scripts/prepare_sota_run.py && -f scripts/audit_sota_advancement.py \
    && -f scripts/preflight_foarl_f638_campaign.py \
    && -f scripts/run_grpo_in_sealed_environment.py \
    && -f scripts/train_foarl_f638_a100.slurm \
    && -f SOURCE_MANIFEST.sha256 ]] || { echo "[FOARLLaunchError] project root is not a frozen FOARL source snapshot" >&2; exit 2; }
[[ "${SLURM_GPUS_ON_NODE:-0}" == "4" ]] || { echo "[FOARLLaunchError] sealed node must expose exactly four A100 GPUs" >&2; exit 2; }
[[ -d "${SFT_MODEL_PATH}" && -d "${BASE_MODEL_PATH}" \
    && -d "${SFT_DATA_ROOT}/${TASK}/train" && -d "${TASK_DATA_ROOT}/${TASK}/train" \
    && -f "${FORBIDDEN_SOURCE_REGISTRY}" && -f "${MODEL_INITIALIZATION_REGISTRY}" \
    && -f "${ADVANCEMENT_POLICY}" ]] || { echo "[FOARLLaunchError] required model/data/registry/policy artifact is missing" >&2; exit 2; }
[[ "${KL_REFERENCE_MODEL_PATH}" == "${SFT_MODEL_PATH}" ]] || { echo "[FOARLLaunchError] fixed reference must equal the corrected task-specialist SFT" >&2; exit 2; }
[[ "${KL_REFERENCE_RECEIPT}" == "${SFT_MODEL_PATH}/checkpoint_fidelity.json" ]] || { echo "[FOARLLaunchError] fixed reference receipt must be the specialist checkpoint_fidelity.json" >&2; exit 2; }

case "${METHOD}" in
    foarl_f638_native)
        [[ "${TRAIN_SEED}" == 42 ]] || { echo "[FOARLLaunchError] native lane requires released seed=42" >&2; exit 2; }
        WORLD_SIZE=1
        NUM_GENERATIONS=8
        GENERATION_BATCH_SIZE=8
        PER_DEVICE_TRAIN_BATCH_SIZE=8
        PER_DEVICE_EVAL_BATCH_SIZE=8
        GRADIENT_ACCUMULATION_STEPS=8
        OPTIMIZER=adamw_torch
        WEIGHT_DECAY=0.02
        WARMUP_RATIO=0.0
        EVAL_FROM_TRAIN_SPLIT=false
        MAX_EVAL_SAMPLES=300
        MIN_EVAL_SAMPLES=300
        MIN_TRAIN_SAMPLES=1
        SOURCE_ROWS=0
        case "${TASK}" in
            cvrp) FINAL_STEP=250; SOURCE_ROWS=2000 ;;
            jssp|mis|tsp) FINAL_STEP=400; SOURCE_ROWS=3200 ;;
            op) FINAL_STEP=625; SOURCE_ROWS=5000 ;;
        esac
        # OP's row-derived terminal step is not a multiple of released save=50.
        # A 25-step operational checkpoint cadence preserves the optimization
        # trajectory while making every clean stop and terminal state resumable.
        if [[ "${TASK}" == op ]]; then SAVE_STEPS=25; else SAVE_STEPS=50; fi
        EVAL_STEPS="${SAVE_STEPS}"
        CHUNK_STEPS="${SAVE_STEPS}"
        ATTRIBUTION=corrected_f638_native_hyperparameter_port
        ARM_SCHEMA=foarl-f638-native-corrected-a100-v1
        ;;
    foarl_f638_matched_b9)
        WORLD_SIZE=4
        NUM_GENERATIONS=9
        GENERATION_BATCH_SIZE=36
        PER_DEVICE_TRAIN_BATCH_SIZE=3
        PER_DEVICE_EVAL_BATCH_SIZE=9
        GRADIENT_ACCUMULATION_STEPS=3
        OPTIMIZER=adamw_8bit
        WEIGHT_DECAY=0.01
        WARMUP_RATIO=0.03
        EVAL_FROM_TRAIN_SPLIT=true
        MAX_EVAL_SAMPLES=100
        MIN_EVAL_SAMPLES=100
        MIN_TRAIN_SAMPLES=1900
        SOURCE_ROWS=registry_bound
        FINAL_STEP=3200
        SAVE_STEPS=400
        EVAL_STEPS=100
        CHUNK_STEPS=400
        ATTRIBUTION=common_attribution_port_never_reproduction
        ARM_SCHEMA=foarl-f638-matched-b9-common-attribution-a100-v1
        ;;
esac
SAVE_TOTAL_LIMIT=$((FINAL_STEP / SAVE_STEPS))
(( FINAL_STEP % SAVE_STEPS == 0 && FINAL_STEP % EVAL_STEPS == 0 \
    && CHUNK_STEPS % SAVE_STEPS == 0 && CHUNK_STEPS % EVAL_STEPS == 0 )) || { echo "[FOARLLaunchError] immutable checkpoint/eval cadence is inconsistent" >&2; exit 2; }

PRIOR_STAGE_RECEIPTS_TEXT="$(
    "${SEALED_PYTHON}" -c 'import json, os, sys; value=json.loads(sys.argv[1]); assert isinstance(value, list) and all(isinstance(x, str) and x and "\n" not in x and os.path.isabs(x) and os.path.realpath(x)==x for x in value); print("\n".join(value))' "${PRIOR_STAGE_RECEIPTS_JSON}"
)"
PRIOR_STAGE_RECEIPTS=()
if [[ -n "${PRIOR_STAGE_RECEIPTS_TEXT}" ]]; then
    mapfile -t PRIOR_STAGE_RECEIPTS <<<"${PRIOR_STAGE_RECEIPTS_TEXT}"
fi
PREFLIGHT_ARGS=(
    --method "${METHOD}" --task "${TASK}" --train-seed "${TRAIN_SEED}"
    --final-step "${FINAL_STEP}" --world-size "${WORLD_SIZE}"
    --source-root "${PROJECT_ROOT}" --source-manifest "${PROJECT_ROOT}/SOURCE_MANIFEST.sha256"
    --forbidden-source-registry "${FORBIDDEN_SOURCE_REGISTRY}"
    --model-initialization-registry "${MODEL_INITIALIZATION_REGISTRY}"
    --advancement-policy "${ADVANCEMENT_POLICY}" --campaign-id "${CAMPAIGN_ID}"
)
for receipt in "${PRIOR_STAGE_RECEIPTS[@]}"; do PREFLIGHT_ARGS+=(--prior-stage-receipt "${receipt}"); done
PREFLIGHT_JSON="$("${SEALED_PYTHON}" scripts/preflight_foarl_f638_campaign.py "${PREFLIGHT_ARGS[@]}")"
read -r ADVANCEMENT_POLICY_SHA256 ADVANCEMENT_STAGE_ID ADVANCEMENT_STAGE_INDEX ADVANCEMENT_PRIOR_CHAIN_SHA256 < <(
    "${SEALED_PYTHON}" -c 'import json, sys; p=json.loads(sys.argv[1]); assert p["status"]=="PASS" and p["side_effects"]=={"writes":0,"scheduler_actions":0}; print(p["advancement_policy_sha256"], p["advancement_stage_id"], p["advancement_stage_index"], p["prior_stage_receipt_chain_sha256"])' "${PREFLIGHT_JSON}"
)

# Seal behavior-affecting inherited state after resolving the permitted paths.
while IFS='=' read -r inherited_name _; do
    case "${inherited_name}" in
        CLASP_*|PD_*|GENERATION_*|GRPO_*|PYTORCH_*|HF_*|TRANSFORMERS_*|TOKENIZERS_*|BITSANDBYTES_*|UNSLOTH_*|ACCELERATE_*|WANDB_*|NCCL_*) unset "${inherited_name}" ;;
    esac
done < <(env)
unset RESUME_FROM_CHECKPOINT || true

export CLASP_PROJECT_ROOT="${PROJECT_ROOT}"
export CLASP_STORAGE_ROOT="${STORAGE_ROOT}"
export CLASP_PUBLICATION_FIDELITY=1
export CLASP_LORA_TARGET_MODULES=jiang CLASP_LORA_BIAS=all
export CLASP_TSP_PROMPT_KNN=0 CLASP_TSP_PROMPT_KNN_DISTANCES=0
export CLASP_TSP_INPUT_ORDER=original CLASP_TSP_RELABEL_ORDER=original
export CLASP_TSP_ROUTE_CANONICALIZE=original CLASP_TSP_ROUTE_ONLY_OUTPUT=0
export CLASP_TSP_LABEL_TTA_SCHEDULE=""
export CLASP_MIS_RELABEL_ORDER=original CLASP_MIS_PROMPT_NODE_DEGREES=0
export CLASP_MIS_PROMPT_LOW_DEGREE_K=32
if [[ "${TASK}" == mis ]]; then export CLASP_MIS_JIANG_RESPONSE_OUTPUT=1; else export CLASP_MIS_JIANG_RESPONSE_OUTPUT=0; fi
export CLASP_AUX_GROUPING_STRICT=1
export PD_IMPORTANCE_SAMPLING_LEVEL=token PD_ALLOW_TOKEN_IMPORTANCE_ABLATION=1
export GRPO_DATA_SEED="${TRAIN_SEED}" GRPO_STRICT_OPTIMIZER_RESUME=1
export CLASP_KL_REFERENCE_MODEL_PATH="${KL_REFERENCE_MODEL_PATH}"
export CLASP_KL_REFERENCE_RECEIPT="${KL_REFERENCE_RECEIPT}"
export CLASP_TRAINING_RUN_CONTRACT_PATH="${RUN_DIR}/run_contract.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED="${TRAIN_SEED}"
export HF_HOME="${CACHE_ROOT}/huggingface" HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export TMPDIR="${CACHE_ROOT}/tmp" TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export WANDB_MODE=offline WANDB_DIR="${CACHE_ROOT}/wandb" WANDB_PROJECT="${WANDB_PROJECT_NAME}"
export WANDB_RUN_GROUP="${CAMPAIGN_ID}_${TASK}_${METHOD}"
export WANDB_RUN_NAME_OVERRIDE="${CAMPAIGN_ID}-${TASK}-${METHOD}-seed${TRAIN_SEED}"
export WANDB_NAME="${WANDB_RUN_NAME_OVERRIDE}" WANDB_RUN_ID="${WANDB_RUN_NAME_OVERRIDE}"
export WANDB_RESUME=allow
unset WANDB_SWEEP_ID || true
mkdir -p "${CACHE_ROOT}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}" \
    "${TORCH_EXTENSIONS_DIR}" "${WANDB_DIR}" "${PROJECT_ROOT}/logs" "${STORAGE_ROOT}/logs"

read -r PYTHON_VERSION TORCH_VERSION TRANSFORMERS_VERSION TRL_VERSION PEFT_VERSION CUDA_VERSION < <(
    "${SEALED_PYTHON}" -c 'import platform, peft, torch, transformers, trl; print(platform.python_version(), torch.__version__, transformers.__version__, trl.__version__, peft.__version__, torch.version.cuda or "none")'
)
[[ "${TRL_VERSION}" == 0.24.* ]] || { echo "[FOARLLaunchError] exact loss validator requires TRL 0.24.x" >&2; exit 2; }
"${SEALED_PYTHON}" -c 'from clasp.baselines.llmcosolver_f638_training import validate_trl_foarl_implementation; from trl import GRPOTrainer; assert validate_trl_foarl_implementation(GRPOTrainer)["status"] == "PASS"'

ALLOWED_ENV=(
    CLASP_PROJECT_ROOT CLASP_STORAGE_ROOT CLASP_PUBLICATION_FIDELITY
    CLASP_LORA_TARGET_MODULES CLASP_LORA_BIAS CLASP_TSP_PROMPT_KNN
    CLASP_TSP_PROMPT_KNN_DISTANCES CLASP_TSP_INPUT_ORDER CLASP_TSP_RELABEL_ORDER
    CLASP_TSP_ROUTE_CANONICALIZE CLASP_TSP_ROUTE_ONLY_OUTPUT CLASP_TSP_LABEL_TTA_SCHEDULE
    CLASP_MIS_RELABEL_ORDER CLASP_MIS_PROMPT_NODE_DEGREES CLASP_MIS_PROMPT_LOW_DEGREE_K
    CLASP_MIS_JIANG_RESPONSE_OUTPUT CLASP_AUX_GROUPING_STRICT
    CLASP_KL_REFERENCE_MODEL_PATH CLASP_KL_REFERENCE_RECEIPT
    CLASP_TRAINING_RUN_CONTRACT_PATH PD_IMPORTANCE_SAMPLING_LEVEL
    PD_ALLOW_TOKEN_IMPORTANCE_ABLATION GRPO_DATA_SEED GRPO_STRICT_OPTIMIZER_RESUME
)
CONTRACT_ARGS=(
    --run-dir "${RUN_DIR}" --method "${METHOD}" --task "${TASK}"
    --train-seed "${TRAIN_SEED}" --sft-seed 42 --final-step "${FINAL_STEP}" --world-size "${WORLD_SIZE}"
    --model-path "${SFT_MODEL_PATH}" --base-model-path "${BASE_MODEL_PATH}"
    --data-path "${TASK_DATA_ROOT}/${TASK}/train" --sft-data-path "${SFT_DATA_ROOT}/${TASK}/train"
    --forbidden-source-registry "${FORBIDDEN_SOURCE_REGISTRY}"
    --model-initialization-registry "${MODEL_INITIALIZATION_REGISTRY}"
    --remote-environment-receipt "${REMOTE_ENVIRONMENT_RECEIPT}"
    --fixed-kl-reference-model-path "${KL_REFERENCE_MODEL_PATH}"
    --fixed-kl-reference-receipt-path "${KL_REFERENCE_RECEIPT}"
    --source-root "${PROJECT_ROOT}" --source-manifest "${PROJECT_ROOT}/SOURCE_MANIFEST.sha256"
    --generation-config "num_generations=${NUM_GENERATIONS}"
    --generation-config "generation_batch_size=${GENERATION_BATCH_SIZE}"
    --generation-config temperature=0.7 --generation-config top_p=0.9
    --generation-config top_k=50 --generation-config repetition_penalty=1.0
    --generation-config max_prompt_length=20000 --generation-config max_completion_length=1000
    --config entrypoint=clasp.entrypoints.train_foarl_f638_campaign
    --config "arm_contract_schema=${ARM_SCHEMA}" --config "attribution=${ATTRIBUTION}"
    --config reproduction_claim=not_an_official_reproduction
    --config prompt_style=jiang_f638 --config prompt_template_sha256=8e90f30020283d017f2b7a9c3109d10fde262488a8f36fd16f0da17ba984a606
    --config no_reason=false --config reward_heads=optimality,feasibility
    --config reward_weights=1.0,1.0 --config reward_aggregation=sum
    --config loss_type=grpo --config token_aggregation=completion_token_mean_then_sequence_mean
    --config scale_rewards=group --config std_estimator=sample_unbiased
    --config advantage_epsilon=0.0001 --config num_iterations=1 --config sync_ref_model=false
    --config importance_sampling_level=token --config beta=0.05 --config epsilon=0.2 --config epsilon_high=0.2
    --config reference_policy=fixed_corrected_sft_step_zero --config use_primal_dual=false --config use_clasp=false
    --config dtype=bfloat16 --config lora_r=64 --config lora_alpha=64 --config lora_dropout=0.0 --config lora_bias=all
    --config "per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --config "per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE}"
    --config "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
    --config learning_rate=0.000001 --config "optim=${OPTIMIZER}" --config "weight_decay=${WEIGHT_DECAY}"
    --config "warmup_ratio=${WARMUP_RATIO}" --config lr_scheduler_type=linear
    --config "eval_from_train_split=${EVAL_FROM_TRAIN_SPLIT}" --config max_samples_per_task=0
    --config "max_eval_samples_per_task=${MAX_EVAL_SAMPLES}" --config "min_train_samples_per_task=${MIN_TRAIN_SAMPLES}"
    --config "min_eval_samples_per_task=${MIN_EVAL_SAMPLES}" --config eval_ratio=0.05
    --config prefer_subset_files=false --config train_size_bucket=all
    --config "source_rows=${SOURCE_ROWS}" --config data_order=registered_rows_seeded_shuffle --config "data_seed=${TRAIN_SEED}"
    --config "hardware_contract=$([[ "${WORLD_SIZE}" == 1 ]] && echo sealed_a100_single_process || echo sealed_4xa100_clasp_matched)"
    --config token_budget_contract=max_prompt_20000_max_completion_1000
    --config checkpoint_fidelity=publication_exact_required
    --config "campaign_id=${CAMPAIGN_ID}" --config "wandb_project=${WANDB_PROJECT_NAME}"
    --config "advancement_policy_sha256=${ADVANCEMENT_POLICY_SHA256}"
    --config "advancement_stage_id=${ADVANCEMENT_STAGE_ID}" --config "advancement_stage_index=${ADVANCEMENT_STAGE_INDEX}"
    --config "advancement_prior_stage_receipt_chain_sha256=${ADVANCEMENT_PRIOR_CHAIN_SHA256}"
    --config "logging_steps=1" --config "save_steps=${SAVE_STEPS}" --config "eval_steps=${EVAL_STEPS}"
    --config "save_total_limit=${SAVE_TOTAL_LIMIT}" --config "final_max_steps=${FINAL_STEP}"
    --config "python_version=${PYTHON_VERSION}" --config "torch_version=${TORCH_VERSION}"
    --config "transformers_version=${TRANSFORMERS_VERSION}" --config "trl_version=${TRL_VERSION}"
    --config "peft_version=${PEFT_VERSION}" --config "cuda_version=${CUDA_VERSION}"
)
for allowed_name in "${ALLOWED_ENV[@]}"; do CONTRACT_ARGS+=(--allowed-env "${allowed_name}"); done

state="$("${SEALED_PYTHON}" scripts/prepare_sota_run.py "${CONTRACT_ARGS[@]}")"
case "${state}" in
    COMPLETE:*) echo "[FOARLLaunch] already complete: ${state#COMPLETE:}"; exit 0 ;;
    FRESH) current_step=0; unset RESUME_FROM_CHECKPOINT || true ;;
    RESUME:*)
        resume_checkpoint="${state#RESUME:}"
        current_step="${resume_checkpoint##*-}"
        [[ "${current_step}" =~ ^[1-9][0-9]*$ ]] || { echo "[FOARLLaunchError] invalid resume step" >&2; exit 2; }
        export RESUME_FROM_CHECKPOINT="${resume_checkpoint}"
        export WANDB_RESUME=must
        ;;
    *) echo "[FOARLLaunchError] unexpected contract state: ${state}" >&2; exit 2 ;;
esac
exec 9>"${RUN_DIR}/.active_training.lock"
flock -n 9 || { echo "[FOARLLaunchError] another job owns this immutable run tuple" >&2; exit 2; }
stop_step=$(( ((current_step / CHUNK_STEPS) + 1) * CHUNK_STEPS ))
(( stop_step > FINAL_STEP )) && stop_step="${FINAL_STEP}"
(( stop_step > current_step && stop_step <= FINAL_STEP && stop_step % SAVE_STEPS == 0 \
    && stop_step % EVAL_STEPS == 0 )) || { echo "[FOARLLaunchError] invalid clean-stop boundary" >&2; exit 2; }
export GRPO_STOP_AT_GLOBAL_STEP="${stop_step}"

COMMON_ARGS=(
    --foarl_campaign_lane "${METHOD}" --tasks "${TASK}"
    --sft_checkpoint_path "${SFT_MODEL_PATH}" --tokenizer_path "${SFT_MODEL_PATH}"
    --data_root "${TASK_DATA_ROOT}" --output_root "${STORAGE_ROOT}" --output_dir "${RUN_DIR}"
    --prompt_style jiang_f638 --no-no_reason --dtype bfloat16
    --no-use_primal_dual --no-use_clasp --no-verify_gt_rewards
    --lora_r 64 --lora_alpha 64 --lora_dropout 0.0 --seed "${TRAIN_SEED}"
    --max_samples_per_task 0 --max_eval_samples_per_task "${MAX_EVAL_SAMPLES}"
    --min_train_samples_per_task "${MIN_TRAIN_SAMPLES}" --min_eval_samples_per_task "${MIN_EVAL_SAMPLES}"
    --no-prefer_subset_files --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate 0.000001 --optim "${OPTIMIZER}" --weight_decay "${WEIGHT_DECAY}"
    --warmup_ratio "${WARMUP_RATIO}" --lr_scheduler_type linear --num_train_epochs 1
    --max_steps "${FINAL_STEP}" --max_prompt_length 20000 --max_completion_length 1000
    --num_generations "${NUM_GENERATIONS}" --generation_batch_size "${GENERATION_BATCH_SIZE}"
    --generation_temperature 0.7 --generation_top_p 0.9 --generation_top_k 50
    --generation_repetition_penalty 1.0 --grpo_beta 0.05 --grpo_epsilon 0.2
    --grpo_epsilon_high 0.2 --grpo_loss_type grpo --foarl_advantage_epsilon 0.0001
    --foarl_scale_rewards group --foarl_num_iterations 1
    --foarl_token_aggregation completion_token_mean_then_sequence_mean
    --no-foarl_sync_ref_model --logging_steps 1 --save_steps "${SAVE_STEPS}"
    --eval_steps "${EVAL_STEPS}" --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --report_to wandb --wandb_project "${WANDB_PROJECT_NAME}" --wandb_group "${WANDB_RUN_GROUP}"
)
if [[ "${EVAL_FROM_TRAIN_SPLIT}" == true ]]; then COMMON_ARGS+=(--eval-from-train-split --eval_ratio 0.05); else COMMON_ARGS+=(--no-eval-from-train-split --eval_ratio 0.05); fi

echo "[FOARLLaunch] campaign=${CAMPAIGN_ID} task=${TASK} method=${METHOD} seed=${TRAIN_SEED}"
echo "[FOARLLaunch] classification=${ATTRIBUTION}; reproduction_claim=not_an_official_reproduction"
echo "[FOARLLaunch] state=${state} current=${current_step} stop=${stop_step} final=${FINAL_STEP} world=${WORLD_SIZE} B=${NUM_GENERATIONS} prompt=20000 completion=1000"
srun --ntasks=1 --ntasks-per-node=1 \
    "${SEALED_PYTHON}" scripts/run_grpo_in_sealed_environment.py \
    --receipt "${REMOTE_ENVIRONMENT_RECEIPT}" -- \
    -m torch.distributed.run --standalone --nproc_per_node="${WORLD_SIZE}" \
    -m clasp.entrypoints.train_foarl_f638_campaign "${COMMON_ARGS[@]}"

final_state="$("${SEALED_PYTHON}" scripts/prepare_sota_run.py "${CONTRACT_ARGS[@]}")"
case "${final_state}" in
    COMPLETE:*) echo "[FOARLLaunch] final horizon complete: ${final_state#COMPLETE:}" ;;
    RESUME:*) echo "[FOARLLaunch] clean continuation boundary saved: ${final_state#RESUME:}" ;;
    *) echo "[FOARLLaunchError] training returned without strict resumable state: ${final_state}" >&2; exit 2 ;;
esac
