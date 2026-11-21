#!/bin/bash -l
# Fixed fresh-SFT plain GRPO pilot: baseline versus graded reward and verified
# improvement supervision. Both arms use B9/global36, 100 steps, seed43.
# Required: DEV_TASK, DEV_METHOD, DEV_SFT_CHECKPOINT, DEV_OUTPUT_DIR,
# DEV_PROJECT_ROOT, DEV_SEALED_ENV, DEV_DATA_ROOT. Inherited GRPO knobs are cleared.
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
#SBATCH --job-name=clasp-dev
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
TEACHER_SELECTION="${DEV_TEACHER_SELECTION:-per_completion}"
case "${TEACHER_SELECTION}" in
    per_completion|best_per_prompt) ;;
    *) echo "[GRPOVerifiedDevError] unknown teacher selection" >&2; exit 2 ;;
esac
[[ "${METHOD}" != grpo_z || "${TEACHER_SELECTION}" == per_completion ]] || {
    echo "[GRPOVerifiedDevError] classic control cannot select teachers" >&2; exit 2;
}
DIAGNOSTIC_UPDATES="${DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES:-0}"
case "${DIAGNOSTIC_UPDATES}" in
    0) REQUIRED_UPDATES=100 ;;
    5) REQUIRED_UPDATES=5 ;;
    *) echo "[GRPOVerifiedDevError] diagnostic updates must be 0 or 5" >&2; exit 2 ;;
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

case "${METHOD}" in
    grpo_z|grpo_verified) ;;
    *) echo "[GRPOVerifiedDevError] DEV_METHOD must be grpo_z or grpo_verified" >&2; exit 2 ;;
esac
[[ "${TASK}" == cvrp || "${TASK}" == mis ]] || { echo "[GRPOVerifiedDevError] pilot supports CVRP and MIS" >&2; exit 2; }
[[ "${TRAIN_SEED}" == 43 && "${MAX_STEPS}" == "${REQUIRED_UPDATES}" && "${SAVE_STEPS}" == "${REQUIRED_UPDATES}" && "${SAVE_TOTAL_LIMIT}" == 1 ]] || {
    echo "[GRPOVerifiedDevError] fixed pilot requires seed43, ${REQUIRED_UPDATES} steps, one terminal checkpoint" >&2; exit 2;
}
[[ -z "${DEV_RESUME_CHECKPOINT:-}" && -z "${DEV_ROLLOUT_BATCH_SIZE:-}" && "${GROUP_STD_CORRECTION}" == 1 ]] || {
    echo "[GRPOVerifiedDevError] fresh B9 pilot requires sample std1 and forbids resume or rollout override" >&2; exit 2;
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
    POLICY_CREDIT_FORMULA="group_z_graded_violations_plus_verified_improvement_nll005_v1"
    if [[ "${TEACHER_SELECTION}" == best_per_prompt ]]; then
        POLICY_CREDIT_FORMULA="group_z_graded_violations_plus_incumbent_filtered_teacher_nll005_v2"
    fi
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
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    [[ "${RESUME_CHECKPOINT}" == "${SFT_CHECKPOINT}" ]] || {
        echo "[CLASPDevError] continuation must load and resume the same checkpoint" >&2
        exit 2
    }
    RESUME_STEP="${RESUME_CHECKPOINT##*/checkpoint-}"
    [[ "${RESUME_STEP}" =~ ^[1-9][0-9]*$ && "${RESUME_CHECKPOINT}" == */checkpoint-"${RESUME_STEP}" ]] || {
        echo "[CLASPDevError] DEV_RESUME_CHECKPOINT must end in checkpoint-<positive-step>" >&2
        exit 2
    }
    (( RESUME_STEP < MAX_STEPS )) || {
        echo "[CLASPDevError] resume step must be below DEV_MAX_STEPS" >&2
        exit 2
    }
    for resume_file in optimizer.pt scheduler.pt trainer_state.json \
        multitask_primal_dual_state.json rng_state_0.pth rng_state_1.pth \
        rng_state_2.pth rng_state_3.pth; do
        [[ -f "${RESUME_CHECKPOINT}/${resume_file}" ]] || {
            echo "[CLASPDevError] continuation checkpoint lacks ${resume_file}" >&2
            exit 2
        }
    done
else
    RESUME_STEP="0"
fi
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

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    "${SEALED_PYTHON}" - "${RESUME_CHECKPOINT}/trainer_state.json" "${RESUME_STEP}" <<'PY'
import json
import sys

state_path, expected_step = sys.argv[1], int(sys.argv[2])
with open(state_path, encoding="utf-8") as handle:
    state = json.load(handle)
observed_step = state.get("global_step")
if observed_step != expected_step:
    raise SystemExit(
        "[CLASPDevError] checkpoint dirname/trainer state mismatch: "
        f"dirname={expected_step}, trainer_state={observed_step!r}"
    )
print(f"[CLASPDevResume] verified complete trainer state at global_step={observed_step}")
PY
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src"

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
        CLASP_*|PD_*|GENERATION_*|GRPO_*|WANDB_*|NCCL_*|PYTORCH_CUDA_ALLOC_CONF) unset "${inherited_name}" ;;
    esac
done < <(env)
unset RESUME_FROM_CHECKPOINT || true
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    export RESUME_FROM_CHECKPOINT="${RESUME_CHECKPOINT}"
fi

CACHE_ROOT="${OUTPUT_DIR}.cache"
mkdir -p "$(dirname "${OUTPUT_DIR}")" "${CACHE_ROOT}/huggingface/datasets" \
    "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/torch_extensions" "${CACHE_ROOT}/wandb"
mkdir "${OUTPUT_DIR}"

export CLASP_LORA_TARGET_MODULES=jiang
export CLASP_LORA_BIAS=all
export CLASP_DEVELOPMENT_FIDELITY=1
export CLASP_TRAINING_RUN_CONTRACT_PATH="${OUTPUT_DIR}/local_development_run_contract.json"

export CLASP_MIS_JIANG_RESPONSE_OUTPUT=0
export CLASP_TSP_PROMPT_KNN=0
export CLASP_TSP_PROMPT_KNN_DISTANCES=0
export CLASP_TSP_INPUT_ORDER=""
export CLASP_TSP_RELABEL_ORDER=""
export CLASP_TSP_ROUTE_CANONICALIZE=""
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
    "${COMPONENT_DUAL_RECOVERY_SCALE}" <<'PY'
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
) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "schema_version": 1,
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
        "path": os.path.realpath(launcher),
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
        "beta": 0.0,
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
echo " Matched plain GRPO verified-improvement pilot (development only)"
echo " graded_reward=${GRPO_GRADED_REWARD} verified_improvement=${GRPO_VERIFIED_IMPROVEMENT} improvement_weight=${GRPO_VERIFIED_IMPROVEMENT_WEIGHT} improvement_max_tokens=${GRPO_VERIFIED_IMPROVEMENT_MAX_TOKENS} improvement_max_passes=${GRPO_VERIFIED_IMPROVEMENT_MAX_PASSES}"
echo " teacher_selection=${GRPO_VERIFIED_TEACHER_SELECTION}"
echo " task=${TASK} method=${METHOD} seed=${TRAIN_SEED} steps=${MAX_STEPS}"
echo " save_steps=${SAVE_STEPS} save_total_limit=${SAVE_TOTAL_LIMIT}"
echo " estimator_arm=${ESTIMATOR_ARM} importance=${IMPORTANCE_SAMPLING_LEVEL} alpha=${POLICY_LENGTH_ALPHA} dual_recovery_coupling=${DUAL_RECOVERY_COUPLING}"
echo " initialization=${SFT_CHECKPOINT}"
echo " resume_checkpoint=${RESUME_CHECKPOINT:-none} resume_step=${RESUME_STEP} additional_steps=$((MAX_STEPS - RESUME_STEP))"
echo " output=${OUTPUT_DIR}"
echo " rollout_B=${ROLLOUT_BATCH_SIZE} deployment_K=8 generation_batch=36 effective_batch=36 subsets=${CANONICAL_MAX_SUBSETS}"
echo " canonical_set=${CANONICAL_SET} use_clasp=${USE_CLASP} canonical_recovery=${CANONICAL_RECOVERY_SCALE} exact_scale=${CANONICAL_EXACT_SCALE} grpo_anchor_scale=${CANONICAL_GRPO_ANCHOR_SCALE} quality_anchor_ratio=${CANONICAL_QUALITY_ANCHOR_RATIO} dense_q/f/v/r=${DENSE_QUALITY_SCALE}/${DENSE_FEASIBILITY_SCALE}/${DENSE_VIOLATION_SCALE}/${DENSE_RECOVERY_SCALE} target_failure@8=0.005 dual_lr=${BON_DUAL_LR} beta=0"
printf ' gpu=%s\n' "${GPU_NAMES[@]}"
echo "=============================================================="

COMMON_ARGS=(
    --tasks "${TASK}" --model_path "${SFT_CHECKPOINT}" --tokenizer_path "${SFT_CHECKPOINT}"
    --data_root "${DATA_ROOT}" --output_root "$(dirname "${OUTPUT_DIR}")" --output_dir "${OUTPUT_DIR}"
    --prompt_style jiang_f638 --no_reason --dtype bfloat16 --lora_r 64 --lora_alpha 64
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
    --grpo_beta 0.0 --grpo_epsilon 0.2 --grpo_epsilon_high 0.2 --grpo_loss_type grpo
    --verify_gt_rewards --reward_check_samples_per_task 5 --train_size_bucket all
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
METHOD_ARGS=(
    --use_primal_dual --no-use_clasp --no-clasp_canonical_set
    --clasp_feasibility_warmup_steps 0
    --clasp_feasibility_warmup_targets "${TASK}:0.995"
    --no-clasp_raw_feasibility_targeting --no-clasp_self_replay
    --no-clasp_validation_gate_checkpoints --no-clasp_pareto_anchor
    --no-clasp_proper_dual --no-clasp_adaptive_length
    --pd_group_scalarization --no-pd_separate_z "${PD_ARGS[@]}"
)

srun --ntasks=1 --ntasks-per-node=1 \
    "${SEALED_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m clasp.entrypoints.train_grpo "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}"

[[ -f "${OUTPUT_DIR}/checkpoints/checkpoint-${MAX_STEPS}/adapter_model.safetensors" \
    && -f "${OUTPUT_DIR}/checkpoints/checkpoint-${MAX_STEPS}/checkpoint_fidelity.json" ]] || {
    echo "[CLASPDevError] terminal certified checkpoint was not produced" >&2
    exit 2
}
echo "GRPO_VERIFIED_DEV_COMPLETE task=${TASK} method=${METHOD} checkpoint=${OUTPUT_DIR}/checkpoints/checkpoint-${MAX_STEPS}"
