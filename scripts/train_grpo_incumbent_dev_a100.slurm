#!/bin/bash -l
# Fresh matched development screen: classic GRPO, feasibility-incumbent
# teachers with optimizer-window normalization, and the corrected FOARL port.
# All arms use B9/global36, 100 updates, seed43; FOARL retains its own rewards,
# fixed-SFT beta=.05 reference. All arms use the same no-reason prompt.
# This common-budget FOARL port is not an official reproduction.
# Required: DEV_TASK, DEV_METHOD, DEV_SFT_CHECKPOINT, DEV_OUTPUT_DIR,
# DEV_PROJECT_ROOT, DEV_SEALED_ENV, DEV_DATA_ROOT, DEV_SOURCE_MANIFEST_SHA256.
# Inherited research knobs are cleared. No submission or automatic retry occurs.
# DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES=5 explicitly enables a five-update
# correctness-only run with gradient and policy consensus checked every step.
# Such runs are excluded by the 100-update comparison controller.

#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=grpo-incumbent-dev
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 027

TASK="${DEV_TASK:?DEV_TASK is required}"
SFT_CHECKPOINT="$(readlink -f "${DEV_SFT_CHECKPOINT:?DEV_SFT_CHECKPOINT is required}")"
OUTPUT_DIR="${DEV_OUTPUT_DIR:?DEV_OUTPUT_DIR is required}"
PROJECT_ROOT="$(readlink -f "${DEV_PROJECT_ROOT:?DEV_PROJECT_ROOT is required}")"
SEALED_ENV="$(readlink -f "${DEV_SEALED_ENV:?DEV_SEALED_ENV is required}")"
DATA_ROOT="$(readlink -f "${DEV_DATA_ROOT:?DEV_DATA_ROOT is required}")"
TRAIN_SEED="${DEV_SEED:-43}"
MAX_STEPS="${DEV_MAX_STEPS:-100}"
SAVE_STEPS="${DEV_SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${DEV_SAVE_TOTAL_LIMIT:-1}"
ESTIMATOR_ARM="${DEV_ESTIMATOR_ARM:-token_a0_control}"
METHOD="${DEV_METHOD:?DEV_METHOD is required}"
case "${METHOD}" in
    grpo_z|grpo_verified|foarl_f638_dev) ;;
    *) echo "[GRPOIncumbentDevError] unsupported method" >&2; exit 2 ;;
esac
SOURCE_MANIFEST_SHA256="${DEV_SOURCE_MANIFEST_SHA256:?DEV_SOURCE_MANIFEST_SHA256 is required}"
[[ "${SOURCE_MANIFEST_SHA256}" =~ ^[a-f0-9]{64}$ ]] || {
    echo "[GRPOIncumbentDevError] invalid expected source-manifest digest" >&2; exit 2;
}
TEACHER_SELECTION="${DEV_TEACHER_SELECTION:-}"
if [[ "${METHOD}" == grpo_verified ]]; then
    TEACHER_SELECTION="${TEACHER_SELECTION:-feasibility_incumbent}"
    case "${TEACHER_SELECTION}" in
        feasibility_incumbent|best_per_prompt) ;;
        *) echo "[GRPOIncumbentDevError] modified arm requires an approved incumbent selector" >&2; exit 2 ;;
    esac
else
    TEACHER_SELECTION="${TEACHER_SELECTION:-per_completion}"
    [[ "${TEACHER_SELECTION}" == per_completion ]] || {
        echo "[GRPOIncumbentDevError] baseline arms cannot select auxiliary teachers" >&2; exit 2;
    }
fi
TEACHER_NORMALIZATION="${DEV_TEACHER_NORMALIZATION:-}"
if [[ "${METHOD}" == grpo_verified ]]; then
    TEACHER_NORMALIZATION="${TEACHER_NORMALIZATION:-optimizer_window}"
    REQUIRED_NORMALIZATION=optimizer_window
else
    TEACHER_NORMALIZATION="${TEACHER_NORMALIZATION:-microbatch}"
    REQUIRED_NORMALIZATION=microbatch
fi
[[ "${TEACHER_NORMALIZATION}" == "${REQUIRED_NORMALIZATION}" ]] || {
    echo "[GRPOIncumbentDevError] teacher normalization differs from declared method" >&2; exit 2;
}
DIAGNOSTIC_UPDATES="${DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES:-0}"
case "${DIAGNOSTIC_UPDATES}" in
    0) REQUIRED_UPDATES=100 ;;
    5) REQUIRED_UPDATES=5 ;;
    *) echo "[GRPOIncumbentDevError] diagnostic updates must be 0 or 5" >&2; exit 2 ;;
esac
# Both plain GRPO arms use the stock sample standard deviation.
GROUP_STD_CORRECTION="${DEV_GROUP_STD_CORRECTION:-1}"
case "${GROUP_STD_CORRECTION}" in
    0|1) ;;
    *) echo "[CLASPDevError] DEV_GROUP_STD_CORRECTION must be 0 or 1" >&2; exit 2 ;;
esac
CANONICAL_QUALITY_ANCHOR_RATIO="0.0"
RESUME_CHECKPOINT=""
if [[ -n "${DEV_RESUME_CHECKPOINT:-}" ]]; then
    RESUME_CHECKPOINT="$(readlink -f "${DEV_RESUME_CHECKPOINT}")"
fi

[[ "${TASK}" == cvrp || "${TASK}" == mis ]] || { echo "[GRPOIncumbentDevError] pilot supports CVRP and MIS" >&2; exit 2; }
[[ "${TRAIN_SEED}" == 43 && "${MAX_STEPS}" == "${REQUIRED_UPDATES}" && "${SAVE_STEPS}" == "${REQUIRED_UPDATES}" && "${SAVE_TOTAL_LIMIT}" == 1 ]] || {
    echo "[GRPOIncumbentDevError] fixed pilot requires seed43, ${REQUIRED_UPDATES} steps, one terminal checkpoint" >&2; exit 2;
}
[[ -z "${DEV_RESUME_CHECKPOINT:-}" && -z "${DEV_ROLLOUT_BATCH_SIZE:-}" && "${GROUP_STD_CORRECTION}" == 1 ]] || {
    echo "[GRPOIncumbentDevError] fresh B9 pilot requires sample std1 and forbids resume or rollout override" >&2; exit 2;
}
ESTIMATOR_ARM="plain_group_z_b9"
POLICY_LENGTH_ALPHA="0.0"
IMPORTANCE_SAMPLING_LEVEL="token"
ALLOW_TOKEN_IMPORTANCE_ABLATION="1"
DUAL_RECOVERY_COUPLING="0.0"
CANONICAL_SET="0"
USE_CLASP="0"
BON_DUAL_LR="0.0"
CANONICAL_RECOVERY_SCALE="0.0"
ROLLOUT_BATCH_SIZE="9"
CANONICAL_MAX_SUBSETS="9"
CANONICAL_EXACT_SCALE="1.0"
CANONICAL_GRPO_ANCHOR_SCALE="0.0"
DENSE_QUALITY_SCALE="0.0"
DENSE_FEASIBILITY_SCALE="0.0"
DENSE_VIOLATION_SCALE="0.0"
DENSE_RECOVERY_SCALE="0.0"
COMPONENT_DUAL_RECOVERY_SCALE="0.0"
FEASIBILITY_GATED="0"
POLICY_CREDIT_FORMULA="group_z_verified_feasible_only_quality_v1"
GRADED_REWARD="0"
VERIFIED_IMPROVEMENT="0"
IMPROVEMENT_WEIGHT="0.0"
if [[ "${METHOD}" == grpo_verified ]]; then
    GRADED_REWARD="1"
    VERIFIED_IMPROVEMENT="1"
    IMPROVEMENT_WEIGHT="0.05"
    POLICY_CREDIT_FORMULA="group_z_graded_feasibility_incumbent_teacher_window_nll005_v1"
    if [[ "${TEACHER_SELECTION}" == best_per_prompt ]]; then
        POLICY_CREDIT_FORMULA="group_z_graded_best_per_prompt_teacher_window_nll005_ablation_v1"
    fi
fi
GRPO_BETA="0.0"
TRAIN_ENTRYPOINT="clasp.entrypoints.train_grpo"
if [[ "${METHOD}" == foarl_f638_dev ]]; then
    GRPO_BETA="0.05"
    TRAIN_ENTRYPOINT="clasp.entrypoints.train_foarl_f638_dev_20260907"
    POLICY_CREDIT_FORMULA="foarl_f638_released_two_reward_heads_group_std_beta005_matched_dev_v1"
    ESTIMATOR_ARM="foarl_f638_group_z_b9_matched_dev"
fi

case "${TASK}" in tsp|cvrp|op|mis|jssp) ;; *) echo "[CLASPDevError] invalid task=${TASK}" >&2; exit 2 ;; esac
[[ "${TRAIN_SEED}" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "[CLASPDevError] invalid seed" >&2; exit 2; }
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "[CLASPDevError] invalid max steps" >&2; exit 2; }
[[ "${SAVE_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "[CLASPDevError] invalid save steps" >&2; exit 2; }
[[ "${SAVE_TOTAL_LIMIT}" =~ ^[1-9][0-9]*$ ]] || { echo "[CLASPDevError] invalid save limit" >&2; exit 2; }
(( MAX_STEPS % SAVE_STEPS == 0 )) || {
    echo "[CLASPDevError] DEV_MAX_STEPS must be divisible by DEV_SAVE_STEPS" >&2
    exit 2
}
[[ -d "${SFT_CHECKPOINT}" && -f "${SFT_CHECKPOINT}/adapter_model.safetensors" \
    && -f "${SFT_CHECKPOINT}/adapter_config.json" \
    && -f "${SFT_CHECKPOINT}/checkpoint_fidelity.json" ]] || {
    echo "[CLASPDevError] initialization is not a certified adapter checkpoint" >&2
    exit 2
}
RESUME_STEP="0"
[[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/src/clasp/set_grpo/canonical.py" \
    && -f "${PROJECT_ROOT}/SOURCE_MANIFEST.sha256" ]] || {
    echo "[CLASPDevError] DEV_PROJECT_ROOT is not a frozen canonical-CLASP source snapshot" >&2
    exit 2
}
[[ -x "${SEALED_ENV}/bin/python" && -d "${DATA_ROOT}/${TASK}/train" ]] || {
    echo "[CLASPDevError] sealed environment or task training data is missing" >&2
    exit 2
}
[[ ! -e "${OUTPUT_DIR}" ]] || {
    echo "[CLASPDevError] create-once output already exists: ${OUTPUT_DIR}" >&2
    exit 2
}

module purge
module load arch/a100
module load miniforge/24.9.0
conda activate "${SEALED_ENV}"
SEALED_PYTHON="$(readlink -f "${SEALED_ENV}/bin/python")"
[[ "$(readlink -f "$(command -v python)")" == "${SEALED_PYTHON}" ]] || {
    echo "[CLASPDevError] active Python is not DEV_SEALED_ENV/bin/python" >&2
    exit 2
}
export PATH="${SEALED_ENV}/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src"
"${SEALED_PYTHON}" - "${PROJECT_ROOT}" "${SOURCE_MANIFEST_SHA256}" "${BASH_SOURCE[0]}" <<'PY_SOURCE'
import hashlib
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
manifest = root / "SOURCE_MANIFEST.sha256"
expected = sys.argv[2]
if manifest.is_symlink() or hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
    raise SystemExit("[GRPOIncumbentDevError] frozen source-manifest digest mismatch")
seen = set()
for line in manifest.read_text().splitlines():
    match = re.fullmatch(r"([a-f0-9]{64})  ([^\n]+)", line)
    if match is None:
        raise SystemExit("[GRPOIncumbentDevError] malformed source manifest")
    digest, name = match.groups()
    relative = pathlib.PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name or name in seen:
        raise SystemExit("[GRPOIncumbentDevError] unsafe or repeated source path")
    seen.add(name)
    path = root / name
    if not path.is_file() or path.is_symlink() or path.resolve().relative_to(root).as_posix() != name:
        raise SystemExit("[GRPOIncumbentDevError] invalid frozen source file")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"[GRPOIncumbentDevError] frozen source mismatch: {name}")
observed = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
if observed != seen | {"SOURCE_MANIFEST.sha256"}:
    raise SystemExit("[GRPOIncumbentDevError] source file inventory differs from manifest")
launcher = "scripts/train_grpo_incumbent_dev_a100.slurm"
if launcher not in seen:
    raise SystemExit("[GRPOIncumbentDevError] launcher absent from source manifest")
if pathlib.Path(sys.argv[3]).read_bytes() != (root / launcher).read_bytes():
    raise SystemExit("[GRPOIncumbentDevError] executed Slurm script differs from frozen launcher")
print(f"[GRPOIncumbentDevSource] verified {len(seen)} files and executed launcher bytes")
PY_SOURCE

[[ "${SLURM_GPUS_ON_NODE:-0}" == "4" ]] || {
    echo "[CLASPDevError] canonical compute contract requires exactly four GPUs" >&2
    exit 2
}
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" == 4 ]] || { echo "[CLASPDevError] expected four visible GPUs" >&2; exit 2; }
for gpu_name in "${GPU_NAMES[@]}"; do
    [[ "${gpu_name}" == *A100* ]] || {
        echo "[CLASPDevError] non-A100 GPU rejected: ${gpu_name}" >&2
        exit 2
    }
done

"${SEALED_PYTHON}" - <<'PY'
import inspect
import json
import platform

import peft
import torch
import transformers
import trl
from trl import GRPOConfig

parameters = inspect.signature(GRPOConfig.__init__).parameters
required = {
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "beta",
    "epsilon",
    "epsilon_high",
    "loss_type",
    "optim",
    "weight_decay",
    "warmup_ratio",
    "lr_scheduler_type",
    "scale_rewards",
    "importance_sampling_level",
}
missing = sorted(required - set(parameters))
if missing:
    raise SystemExit(
        "[CLASPDevError] runtime GRPOConfig cannot honor the canonical objective: "
        + ",".join(missing)
    )
print(
    "[CLASPDevRuntime] "
    + json.dumps(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        sort_keys=True,
    ),
    flush=True,
)
PY

# The saved adapter must point to the same immutable base model that will be
# loaded by PEFT.  Record the resolved base for the immutable development receipt.
BASE_MODEL_PATH="$("${SEALED_PYTHON}" - "${SFT_CHECKPOINT}" <<'PY'
import json
import os
import sys

with open(os.path.join(sys.argv[1], "adapter_config.json"), encoding="utf-8") as handle:
    config = json.load(handle)
base = os.path.realpath(str(config.get("base_model_name_or_path", "")))
if not base or not os.path.isdir(base):
    raise SystemExit("[CLASPDevError] adapter base_model_name_or_path is missing")
print(base)
PY
)"

# Reject inherited research knobs.  All behavior-affecting values are restated
# below so a submission shell cannot silently change the method under test.
while IFS='=' read -r inherited_name _; do
    case "${inherited_name}" in
        CLASP_*|PD_*|GENERATION_*|GRPO_*|FOARL_*|WANDB_*|NCCL_*|PYTORCH_*|HF_*|TRANSFORMERS_*|TOKENIZERS_*|BITSANDBYTES_*|UNSLOTH_*|ACCELERATE_*) unset "${inherited_name}" ;;
    esac
done < <(env)
unset RESUME_FROM_CHECKPOINT || true

CACHE_ROOT="${OUTPUT_DIR}.cache"
mkdir -p "$(dirname "${OUTPUT_DIR}")" "${CACHE_ROOT}/huggingface/datasets" \
    "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/torch_extensions" "${CACHE_ROOT}/wandb"
mkdir "${OUTPUT_DIR}"

export CLASP_LORA_TARGET_MODULES=jiang
export CLASP_LORA_BIAS=all
export CLASP_DEVELOPMENT_FIDELITY=1
export CLASP_TRAINING_RUN_CONTRACT_PATH="${OUTPUT_DIR}/local_development_run_contract.json"

export CLASP_MIS_JIANG_RESPONSE_OUTPUT=0
export CLASP_MIS_RELABEL_ORDER=original
export CLASP_MIS_PROMPT_NODE_DEGREES=0
export CLASP_MIS_PROMPT_LOW_DEGREE_K=32
if [[ "${METHOD}" == foarl_f638_dev ]]; then
    export CLASP_KL_REFERENCE_MODEL_PATH="${SFT_CHECKPOINT}"
    export CLASP_KL_REFERENCE_RECEIPT="${SFT_CHECKPOINT}/checkpoint_fidelity.json"
fi
export CLASP_TSP_PROMPT_KNN=0
export CLASP_TSP_PROMPT_KNN_DISTANCES=0
export CLASP_TSP_INPUT_ORDER=original
export CLASP_TSP_RELABEL_ORDER=original
export CLASP_TSP_ROUTE_CANONICALIZE=original
export CLASP_TSP_LABEL_TTA_SCHEDULE=""
export CLASP_TSP_ROUTE_ONLY_OUTPUT=0

export CLASP_OBJECTIVE_QUALITY_MODE=exp
export CLASP_OBJECTIVE_GAP_SLOPE=3.0
export CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT=0.0
export CLASP_CANONICAL_SET="${CANONICAL_SET}"
export CLASP_DEPLOYMENT_SET_SIZE=8
export CLASP_CANONICAL_RECOVERY_SCALE="${CANONICAL_RECOVERY_SCALE}"
export CLASP_CANONICAL_DUAL_RECOVERY_COUPLING="${DUAL_RECOVERY_COUPLING}"
export CLASP_CANONICAL_RECOVERY_TEMPERATURE=0.15
export CLASP_CANONICAL_MAX_SUBSETS="${CANONICAL_MAX_SUBSETS}"
export CLASP_CANONICAL_EXACT_SCALE="${CANONICAL_EXACT_SCALE}"
export CLASP_CANONICAL_GRPO_ANCHOR_SCALE="${CANONICAL_GRPO_ANCHOR_SCALE}"
export CLASP_CANONICAL_QUALITY_ANCHOR_RATIO="${CANONICAL_QUALITY_ANCHOR_RATIO}"
export CLASP_CANONICAL_DENSE_QUALITY_SCALE="${DENSE_QUALITY_SCALE}"
export CLASP_CANONICAL_DENSE_FEASIBILITY_SCALE="${DENSE_FEASIBILITY_SCALE}"
export CLASP_CANONICAL_DENSE_VIOLATION_SCALE="${DENSE_VIOLATION_SCALE}"
export CLASP_CANONICAL_DENSE_RECOVERY_SCALE="${DENSE_RECOVERY_SCALE}"
export CLASP_POLICY_LENGTH_ALPHA="${POLICY_LENGTH_ALPHA}"
export CLASP_BON_ALIGNED="${USE_CLASP}"
export CLASP_FRONTIER_CDF=0
export CLASP_BON_TARGET_FAILURE_RATE=0.005
export CLASP_BON_TARGET_FAILURE_RATES="${TASK}:0.005"
export CLASP_BON_DUAL_LR="${BON_DUAL_LR}"
export CLASP_BON_DUAL_EMA_BETA=0.9
export CLASP_BON_LAMBDA_INIT=0.0
export CLASP_BON_LAMBDA_MAX=100.0
export CLASP_BON_QUALITY_TEMPERATURE=0.0
export CLASP_BON_DENSE_QUALITY_SCALE=1.0
export CLASP_BON_DENSE_FEASIBILITY_SCALE=1.0
export CLASP_BON_DENSE_VIOLATION_MIX=0.5
export CLASP_BON_RECOVERY_SCALE=0.0
export CLASP_COMPONENT_DUAL_RECOVERY_SCALE="${COMPONENT_DUAL_RECOVERY_SCALE}"
export CLASP_MIXED_ADAPTIVE_LENGTH_MODE=error
export CLASP_AUX_GROUPING_STRICT=1
export CLASP_PAIRWISE_LOSS_WEIGHT=0.0
export CLASP_ELITE_IMITATION_LOSS_WEIGHT=0.0
export CLASP_MULTI_ELITE_IMITATION_LOSS_WEIGHT=0.0
export CLASP_REFERENCE_IMITATION_LOSS_WEIGHT=0.0
export CLASP_REFERENCE_CONTRASTIVE_LOSS_WEIGHT=0.0
export CLASP_TSP_TWO_OPT_DISTILL_LOSS_WEIGHT=0.0
export CLASP_CVRP_LOCAL_SEARCH_DISTILL_LOSS_WEIGHT=0.0
export CLASP_JSSP_ACTIVE_SCHEDULE_DISTILL_LOSS_WEIGHT=0.0
export CLASP_MIS_MAXIMALIZE_DISTILL_LOSS_WEIGHT=0.0
export CLASP_HARD_GROUP_WEIGHT_SCALE=0.0
export CLASP_FEASIBILITY_GATED="${FEASIBILITY_GATED}"
export CLASP_LEXICOGRAPHIC_GATE=0
export CLASP_FEASIBILITY_WARMUP_STEPS=0
export CLASP_FEASIBILITY_WARMUP_TARGETS="${TASK}:0.995"
export CLASP_FEASIBILITY_WARMUP_HOLD_UNTIL_EVAL=1
export CLASP_SELF_REPLAY_ENABLED=0
export CLASP_SELF_REPLAY_IMITATION_LOSS_WEIGHT=0.0
export CLASP_SELF_REPLAY_CONTRASTIVE_LOSS_WEIGHT=0.0
export CLASP_VALIDATION_GATE_CHECKPOINTS=0
export PD_IMPORTANCE_SAMPLING_LEVEL="${IMPORTANCE_SAMPLING_LEVEL}"
export PD_ALLOW_TOKEN_IMPORTANCE_ABLATION="${ALLOW_TOKEN_IMPORTANCE_ABLATION}"
export PD_GROUP_STD_CORRECTION="${GROUP_STD_CORRECTION}"
# These values are derived from DEV_METHOD after inherited GRPO_* variables
# have been removed. External submission shells cannot silently change the arm.
export GRPO_GRADED_REWARD="${GRADED_REWARD}"
export GRPO_VERIFIED_IMPROVEMENT="${VERIFIED_IMPROVEMENT}"
export GRPO_VERIFIED_IMPROVEMENT_WEIGHT="${IMPROVEMENT_WEIGHT}"
export GRPO_VERIFIED_IMPROVEMENT_MAX_TOKENS=3000
export GRPO_VERIFIED_IMPROVEMENT_MAX_PASSES=2
export GRPO_VERIFIED_TEACHER_SELECTION="${TEACHER_SELECTION}"
export GRPO_VERIFIED_TEACHER_NORMALIZATION="${TEACHER_NORMALIZATION}"
export CLASP_POLICY_CONSISTENCY_CHECKS=1
export CLASP_POLICY_CONSISTENCY_CHECK_INTERVAL=10
export CLASP_GRADIENT_CONSISTENCY_CHECKS=0
if [[ "${DIAGNOSTIC_UPDATES}" == 5 ]]; then
    export CLASP_POLICY_CONSISTENCY_CHECK_INTERVAL=1
    export CLASP_GRADIENT_CONSISTENCY_CHECKS=1
fi
export GRPO_DATA_SEED="${TRAIN_SEED}"
export GRPO_STRICT_OPTIMIZER_RESUME=1

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED="${TRAIN_SEED}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TMPDIR="${CACHE_ROOT}/tmp"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export WANDB_MODE=offline WANDB_DIR="${CACHE_ROOT}/wandb"

COMMON_ARGS=(
    --tasks "${TASK}" --model_path "${SFT_CHECKPOINT}" --tokenizer_path "${SFT_CHECKPOINT}"
    --data_root "${DATA_ROOT}" --output_root "$(dirname "${OUTPUT_DIR}")" --output_dir "${OUTPUT_DIR}"
    --prompt_style jiang_f638 --dtype bfloat16 --lora_r 64 --lora_alpha 64
    --lora_dropout 0.0 --seed "${TRAIN_SEED}" --eval_ratio 0 --eval-from-train-split
    --max_samples_per_task 0 --max_eval_samples_per_task 0
    --min_train_samples_per_task 1900 --min_eval_samples_per_task 0 --no-prefer_subset_files
    --per_device_train_batch_size 3 --per_device_eval_batch_size 9
    --gradient_accumulation_steps 3 --learning_rate 1e-6 --optim adamw_8bit
    --weight_decay 0.01 --warmup_ratio 0.03 --lr_scheduler_type linear
    --num_train_epochs 1 --max_steps "${MAX_STEPS}"
    --max_prompt_length 29768 --max_completion_length 3000
    --num_generations "${ROLLOUT_BATCH_SIZE}" --generation_batch_size 36 --generation_temperature 0.7
    --generation_top_p 0.9 --generation_top_k 50 --generation_repetition_penalty 1.0
    --grpo_beta "${GRPO_BETA}" --grpo_epsilon 0.2 --grpo_epsilon_high 0.2 --grpo_loss_type grpo
    --reward_check_samples_per_task 5 --train_size_bucket all
    --logging_steps 1 --save_steps "${SAVE_STEPS}" --eval_steps 1
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --report_to none
)
PD_ARGS=(
    --pd_constraint_target 1.0 --pd_cost_limit 0.0 --pd_lambda_init 0.0
    --pd_dual_lr 0.0 --pd_dual_lr_up 0.0 --pd_dual_lr_down 0.0
    --pd_dual_update_interval 1 --pd_violation_ema_beta 0.1 --pd_lambda_max 100.0
    --pd_violation_tolerance 0.0 --pd_objective_weight 1.0 --pd_structure_weight 0.0
    --pd_sync_across_ranks --pd_require_state_on_resume --pd_strict_constraint_range
    --pd_force_scale_rewards_none --no-pd_soft_objective_for_infeasible
    --no-pd_normalize_primal_advantage --pd_primal_advantage_scale 1.0
    --pd_penalty_scale 1.0 --pd_positive_constraint_penalty
    --no-pd_lexicographic_advantage --pd_component_profile co_feasibility_lock_all
    --no-pd_enable_schedule
)
DISABLED_CLASP_ARGS=(
    --no-use_clasp --no-clasp_canonical_set
    --clasp_feasibility_warmup_steps 0
    --clasp_feasibility_warmup_targets "${TASK}:0.995"
    --no-clasp_raw_feasibility_targeting --no-clasp_self_replay
    --no-clasp_validation_gate_checkpoints --no-clasp_pareto_anchor
    --no-clasp_proper_dual --no-clasp_adaptive_length
    --pd_group_scalarization --no-pd_separate_z "${PD_ARGS[@]}"
)

METHOD_ARGS=(--no_reason --verify_gt_rewards --use_primal_dual "${DISABLED_CLASP_ARGS[@]}")
if [[ "${METHOD}" == foarl_f638_dev ]]; then
    METHOD_ARGS=(
        "${DISABLED_CLASP_ARGS[@]}"
        --sft_checkpoint_path "${SFT_CHECKPOINT}"
        --foarl_dev_source_manifest "${PROJECT_ROOT}/SOURCE_MANIFEST.sha256"
        --foarl_dev_source_manifest_sha256 "${SOURCE_MANIFEST_SHA256}"
        --no_reason --no-verify_gt_rewards --no-use_primal_dual
        --foarl_advantage_epsilon 0.0001 --foarl_scale_rewards group
        --foarl_num_iterations 1
        --foarl_token_aggregation completion_token_mean_then_sequence_mean
        --no-foarl_sync_ref_model
    )
fi
TRAIN_COMMAND=(
    srun --ntasks=1 --ntasks-per-node=1
    "${SEALED_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4
    -m "${TRAIN_ENTRYPOINT}" "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}"
)

# The fidelity verifier only requires stage and the exact fixed horizon, but the
# remainder documents every deliberate development-only departure and input.
"${SEALED_PYTHON}" - "${OUTPUT_DIR}" "${TASK}" "${TRAIN_SEED}" "${MAX_STEPS}" \
    "${SFT_CHECKPOINT}" "${BASE_MODEL_PATH}" "${PROJECT_ROOT}" "${DATA_ROOT}" \
    "$(readlink -f "${BASH_SOURCE[0]}")" "${SEALED_ENV}" "${ESTIMATOR_ARM}" \
    "${POLICY_LENGTH_ALPHA}" "${IMPORTANCE_SAMPLING_LEVEL}" "${SAVE_STEPS}" \
    "${SAVE_TOTAL_LIMIT}" "${DUAL_RECOVERY_COUPLING}" "${METHOD}" \
    "${POLICY_CREDIT_FORMULA}" "${CANONICAL_SET}" "${USE_CLASP}" \
    "${BON_DUAL_LR}" "${CANONICAL_RECOVERY_SCALE}" "${ROLLOUT_BATCH_SIZE}" \
    "${CANONICAL_MAX_SUBSETS}" "${CANONICAL_EXACT_SCALE}" \
    "${CANONICAL_GRPO_ANCHOR_SCALE}" "${CANONICAL_QUALITY_ANCHOR_RATIO}" \
    "${DENSE_QUALITY_SCALE}" "${DENSE_FEASIBILITY_SCALE}" \
    "${DENSE_VIOLATION_SCALE}" "${DENSE_RECOVERY_SCALE}" \
    "${COMPONENT_DUAL_RECOVERY_SCALE}" --launch-command "${TRAIN_COMMAND[@]}" <<'PY'
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
import pathlib
import platform
import sys

(
    out,
    task,
    seed,
    steps,
    checkpoint,
    base,
    source,
    data,
    launcher,
    runtime_env,
    estimator_arm,
    policy_length_alpha,
    importance_sampling_level,
    save_steps,
    save_total_limit,
    dual_recovery_coupling,
    method,
    policy_credit_formula,
    canonical_set,
    use_clasp,
    bon_dual_lr,
    canonical_recovery_scale,
    rollout_batch_size,
    canonical_max_subsets,
    canonical_exact_scale,
    canonical_grpo_anchor_scale,
    canonical_quality_anchor_ratio,
    dense_quality_scale,
    dense_feasibility_scale,
    dense_violation_scale,
    dense_recovery_scale,
    component_dual_recovery_scale,
) = sys.argv[1:sys.argv.index("--launch-command")]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "schema_version": 2,
    "protocol": "grpo-incumbent-foarl-matched-development-20260907-v1",
    "stage": "grpo",
    "distributed_correctness_diagnostic": int(steps) == 5,
    "distributed_checks": {
        "policy": os.environ["CLASP_POLICY_CONSISTENCY_CHECKS"] == "1",
        "gradients": os.environ["CLASP_GRADIENT_CONSISTENCY_CHECKS"] == "1",
        "interval": int(os.environ["CLASP_POLICY_CONSISTENCY_CHECK_INTERVAL"]),
    },
    "status": "development_only_not_publication_eligible",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "fixed_horizon": {"max_steps": int(steps), "world_size": 4},
    "task": task,
    "method": method,
    "execution": {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "command": sys.argv[sys.argv.index("--launch-command") + 1:],
        "source_manifest_expected_sha256": os.environ["DEV_SOURCE_MANIFEST_SHA256"],
    },
    "policy_credit_formula": policy_credit_formula,
    "seed": int(seed),
    "initialization": {
        "checkpoint": os.path.realpath(checkpoint),
        "adapter_model_sha256": sha256(os.path.join(checkpoint, "adapter_model.safetensors")),
        "adapter_config_sha256": sha256(os.path.join(checkpoint, "adapter_config.json")),
        "checkpoint_fidelity_sha256": sha256(os.path.join(checkpoint, "checkpoint_fidelity.json")),
        "base_model": os.path.realpath(base),
    },
    "continuation": {
        "resume_checkpoint": (
            os.path.realpath(os.environ["DEV_RESUME_CHECKPOINT"])
            if os.environ.get("DEV_RESUME_CHECKPOINT")
            else None
        ),
        "resume_global_step": int(
            os.path.basename(os.path.realpath(os.environ["DEV_RESUME_CHECKPOINT"]))
            .removeprefix("checkpoint-")
        ) if os.environ.get("DEV_RESUME_CHECKPOINT") else 0,
        "strict_optimizer_resume": bool(os.environ.get("DEV_RESUME_CHECKPOINT")),
    },
    "source": {
        "root": os.path.realpath(source),
        "manifest_sha256": sha256(os.path.join(source, "SOURCE_MANIFEST.sha256")),
    },
    "launcher": {
        "path": os.path.realpath(os.path.join(source, "scripts/train_grpo_incumbent_dev_a100.slurm")),
        "execution_path": os.path.realpath(launcher),
        "sha256": sha256(launcher),
    },
    "runtime": {
        "environment": os.path.realpath(runtime_env),
        "python": platform.python_version(),
        "torch": metadata.version("torch"),
        "transformers": metadata.version("transformers"),
        "trl": metadata.version("trl"),
        "peft": metadata.version("peft"),
        "bitsandbytes": metadata.version("bitsandbytes"),
        "environment_is_publication_sealed": False,
    },
    "data_root": os.path.realpath(data),
    "compute": {"accelerator": "A100", "gpu_count": 4, "qos": "qos_gpu_a100-dev", "walltime": "02:00:00"},
    "production_preserved": {
        "rollout_batch_b": int(rollout_batch_size),
        "deployment_set_k": 8,
        "global_generation_batch_size": 36,
        "per_device_train_batch_size": 3,
        "gradient_accumulation_steps": 3,
        "max_prompt_length": 29768,
        "max_completion_length": 3000,
        "learning_rate": 1e-6,
        "beta": 0.05 if method == "foarl_f638_dev" else 0.0,
        "target_failure_at_k": 0.005,
        "dual_learning_rate": float(bon_dual_lr),
        "recovery_scale": float(canonical_recovery_scale),
        "recovery_temperature": 0.15,
        "prompt_style": "jiang_f638",
        "no_reason": True,
        "mis_jiang_response_output": False,
    },
    "plain_grpo_variant": {
        "graded_reward": bool(int(os.environ["GRPO_GRADED_REWARD"])),
        "verified_improvement": bool(int(os.environ["GRPO_VERIFIED_IMPROVEMENT"])),
        "improvement_weight": float(os.environ["GRPO_VERIFIED_IMPROVEMENT_WEIGHT"]),
        "improvement_max_tokens": int(os.environ["GRPO_VERIFIED_IMPROVEMENT_MAX_TOKENS"]),
        "improvement_max_passes": int(os.environ["GRPO_VERIFIED_IMPROVEMENT_MAX_PASSES"]),
        "teacher_selection": os.environ["GRPO_VERIFIED_TEACHER_SELECTION"],
        "teacher_normalization": os.environ["GRPO_VERIFIED_TEACHER_NORMALIZATION"],
    },
    "foarl_matched_port": {
        "enabled": method == "foarl_f638_dev",
        "claim": "corrected_f638_algorithm_common_budget_development_port_not_official_reproduction",
        "released_reward_heads": ["optimality", "feasibility"] if method == "foarl_f638_dev" else [],
        "frozen_reference_checkpoint": os.path.realpath(checkpoint) if method == "foarl_f638_dev" else None,
        "frozen_reference_receipt_sha256": sha256(os.path.join(checkpoint, "checkpoint_fidelity.json")) if method == "foarl_f638_dev" else None,
        "deliberate_common_budget_changes": ["B9_global36_four_A100", "100_updates_seed43", "common_no_reason_prompt_and_MIS_output0", "prompt29768_completion3000", "adamw_8bit_weight_decay001_warmup003", "no_internal_evaluation"],
    },
    "budget_accounting": {
        "onpolicy_completions": int(steps) * 36,
        "prompt_groups": int(steps) * 4,
        "actual_tokens_are_matched": False,
        "teacher_supervised_tokens_and_search_are_additional": method == "grpo_verified",
        "fixed_reference_forward_is_additional": method == "foarl_f638_dev",
    },
    "policy_estimator": {
        "arm": estimator_arm,
        "group_std_correction": int(os.environ["PD_GROUP_STD_CORRECTION"]),
        "importance_sampling_level": importance_sampling_level,
        "policy_length_alpha": float(policy_length_alpha),
        "dual_recovery_coupling": float(dual_recovery_coupling),
        "canonical_max_subsets": int(canonical_max_subsets),
        "canonical_exact_scale": float(canonical_exact_scale),
        "canonical_grpo_anchor_scale": float(canonical_grpo_anchor_scale),
        "canonical_quality_anchor_ratio": float(canonical_quality_anchor_ratio),
        "canonical_dense_quality_scale": float(dense_quality_scale),
        "canonical_dense_feasibility_scale": float(dense_feasibility_scale),
        "canonical_dense_violation_scale": float(dense_violation_scale),
        "canonical_dense_recovery_scale": float(dense_recovery_scale),
        "component_dual_recovery_scale": float(component_dual_recovery_scale),
        "canonical_set": bool(int(canonical_set)),
        "use_clasp": bool(int(use_clasp)),
        "fixed_completion_horizon": 3000,
        "kl_reduction": "per_response_token_mean",
    },
    "development_differences": {
        "max_steps": int(steps),
        "save_steps": int(save_steps),
        "save_total_limit": int(save_total_limit),
        "internal_evaluation": False,
        "report_to": "none",
        "publication_eligible": False,
    },
}
raw = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
path = pathlib.Path(out) / "local_development_run_contract.json"
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(raw)
    handle.flush()
    os.fsync(handle.fileno())
PY

echo "=============================================================="
echo " Matched incumbent-teacher and FOARL pilot (development only)"
echo " graded_reward=${GRPO_GRADED_REWARD} verified_improvement=${GRPO_VERIFIED_IMPROVEMENT} improvement_weight=${GRPO_VERIFIED_IMPROVEMENT_WEIGHT} improvement_max_tokens=${GRPO_VERIFIED_IMPROVEMENT_MAX_TOKENS} improvement_max_passes=${GRPO_VERIFIED_IMPROVEMENT_MAX_PASSES}"
echo " teacher_selection=${GRPO_VERIFIED_TEACHER_SELECTION} normalization=${GRPO_VERIFIED_TEACHER_NORMALIZATION}"
echo " task=${TASK} method=${METHOD} seed=${TRAIN_SEED} steps=${MAX_STEPS}"
echo " save_steps=${SAVE_STEPS} save_total_limit=${SAVE_TOTAL_LIMIT}"
echo " estimator_arm=${ESTIMATOR_ARM} importance=${IMPORTANCE_SAMPLING_LEVEL} alpha=${POLICY_LENGTH_ALPHA} dual_recovery_coupling=${DUAL_RECOVERY_COUPLING}"
echo " initialization=${SFT_CHECKPOINT}"
echo " resume_checkpoint=${RESUME_CHECKPOINT:-none} resume_step=${RESUME_STEP} additional_steps=$((MAX_STEPS - RESUME_STEP))"
echo " output=${OUTPUT_DIR}"
echo " rollout_B=${ROLLOUT_BATCH_SIZE} deployment_K=8 generation_batch=36 effective_batch=36 subsets=${CANONICAL_MAX_SUBSETS}"
echo " canonical_set=${CANONICAL_SET} use_clasp=${USE_CLASP} canonical_recovery=${CANONICAL_RECOVERY_SCALE} exact_scale=${CANONICAL_EXACT_SCALE} grpo_anchor_scale=${CANONICAL_GRPO_ANCHOR_SCALE} quality_anchor_ratio=${CANONICAL_QUALITY_ANCHOR_RATIO} dense_q/f/v/r=${DENSE_QUALITY_SCALE}/${DENSE_FEASIBILITY_SCALE}/${DENSE_VIOLATION_SCALE}/${DENSE_RECOVERY_SCALE} target_failure@8=0.005 dual_lr=${BON_DUAL_LR} beta=${GRPO_BETA}"
printf ' gpu=%s\n' "${GPU_NAMES[@]}"
echo "=============================================================="


"${TRAIN_COMMAND[@]}"

[[ -f "${OUTPUT_DIR}/checkpoints/checkpoint-${MAX_STEPS}/adapter_model.safetensors" \
    && -f "${OUTPUT_DIR}/checkpoints/checkpoint-${MAX_STEPS}/checkpoint_fidelity.json" ]] || {
    echo "[CLASPDevError] terminal certified checkpoint was not produced" >&2
    exit 2
}
echo "GRPO_INCUMBENT_DEV_COMPLETE task=${TASK} method=${METHOD} checkpoint=${OUTPUT_DIR}/checkpoints/checkpoint-${MAX_STEPS}"
