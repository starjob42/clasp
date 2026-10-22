#!/bin/bash -l
# One publication-contract canonical-CLASP trajectory on 4xA100.
# Required: SOTA_CAMPAIGN_ID, SOTA_TASK, SOTA_METHOD, SOTA_TRAIN_SEED,
# CLASP_STORAGE_ROOT, SOTA_FORBIDDEN_SOURCE_REGISTRY,
# SOTA_MODEL_INITIALIZATION_REGISTRY, SOTA_BASE_MODEL_PATH,
# SOTA_REMOTE_ENVIRONMENT_RECEIPT, and SOTA_SEALED_ENV.
# Runnable methods:
#   grpo_z, dual_scalar_z, dual_separate_z, clasp_canonical,
#   clasp_canonical_fixed_lambda, clasp_canonical_no_recovery,
#   clasp_canonical_foarl_matched_b9, set_null.
# The historical grpo/dual_grpo/clasp_dense/clasp_exact identifiers are legacy
# and deliberately rejected by this source snapshot.  A same-utility set_null
# arm requires a separately frozen construction receipt; this launcher never
# chooses or synthesizes the scientific null candidate.
# Re-submit the same tuple to continue; max_steps always remains the final 3200.
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
#SBATCH --job-name=clasp-sota
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 027

TASK="${SOTA_TASK:?SOTA_TASK is required (tsp/cvrp/op/mis/jssp)}"
METHOD="${SOTA_METHOD:?SOTA_METHOD is required (grpo_z/dual_scalar_z/dual_separate_z/clasp_canonical/clasp_canonical_fixed_lambda/clasp_canonical_no_recovery/clasp_canonical_foarl_matched_b9/set_null)}"
TRAIN_SEED="${SOTA_TRAIN_SEED:?SOTA_TRAIN_SEED is required}"
CAMPAIGN_ID="${SOTA_CAMPAIGN_ID:?SOTA_CAMPAIGN_ID is required and must be unique per artifact campaign}"
case "${TASK}" in tsp|cvrp|op|mis|jssp) ;; *) echo "[SOTALaunchError] invalid task=${TASK}" >&2; exit 2 ;; esac
case "${METHOD}" in
    grpo_z|dual_scalar_z|dual_separate_z|clasp_canonical|clasp_canonical_fixed_lambda|clasp_canonical_no_recovery|clasp_canonical_foarl_matched_b9|set_null) ;;
    clasp_set_null|clasp_fixed_null)
        echo "[SOTALaunchError] invalid fixed-null alias; use the exact method ID set_null" >&2
        exit 2
        ;;
    dual_separate)
        echo "[SOTALaunchError] method=dual_separate is not a valid alias; use the exact method ID dual_separate_z" >&2
        exit 2
        ;;
    grpo|dual_grpo|clasp_dense|clasp_exact)
        echo "[SOTALaunchError] method=${METHOD} is a retired legacy arm and is not runnable in the canonical publication campaign" >&2
        exit 2
        ;;
    *) echo "[SOTALaunchError] invalid method=${METHOD}" >&2; exit 2 ;;
esac
[[ "${TRAIN_SEED}" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "[SOTALaunchError] seed must be canonical nonnegative decimal" >&2; exit 2; }
[[ "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "[SOTALaunchError] campaign ID contains unsafe characters" >&2; exit 2; }

module purge
module load arch/a100
module load miniforge/24.9.0
SEALED_ENV="$(readlink -f "${SOTA_SEALED_ENV:?SOTA_SEALED_ENV is required and must name the frozen publication environment}")"
REMOTE_ENVIRONMENT_RECEIPT="$(readlink -f "${SOTA_REMOTE_ENVIRONMENT_RECEIPT:?SOTA_REMOTE_ENVIRONMENT_RECEIPT is required}")"
FIXED_NULL_RECEIPT="${SOTA_FIXED_NULL_RECEIPT:-}"
if [[ "${METHOD}" == "set_null" ]]; then
    FIXED_NULL_RECEIPT="$(readlink -f "${FIXED_NULL_RECEIPT:?SOTA_FIXED_NULL_RECEIPT is required for set_null}")"
    [[ -f "${FIXED_NULL_RECEIPT}" ]] || {
        echo "[SOTALaunchError] fixed-null construction receipt is missing" >&2
        exit 2
    }
elif [[ -n "${FIXED_NULL_RECEIPT}" ]]; then
    echo "[SOTALaunchError] SOTA_FIXED_NULL_RECEIPT is valid only for set_null" >&2
    exit 2
fi
[[ -d "${SEALED_ENV}" && -f "${REMOTE_ENVIRONMENT_RECEIPT}" ]] || {
    echo "[SOTALaunchError] sealed environment or its receipt is missing" >&2
    exit 2
}
[[ ! -w "${SEALED_ENV}" && ! -w "$(dirname "${SEALED_ENV}")" ]] || {
    echo "[SOTALaunchError] sealed environment and its parent must be non-writable" >&2
    exit 2
}
conda activate "${SEALED_ENV}"
SEALED_PYTHON="$(readlink -f "${SEALED_ENV}/bin/python")"
[[ "$(readlink -f "$(command -v python)")" == "${SEALED_PYTHON}" ]] || {
    echo "[SOTALaunchError] activated Python does not equal SOTA_SEALED_ENV/bin/python" >&2
    exit 2
}
export PATH="${SEALED_ENV}/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

# Resolve user-selectable paths before clearing inherited research knobs.
PROJECT_ROOT="$(readlink -f "${CLASP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}")"
ASSET_ROOT="${CLASP_ASSET_ROOT:-/lustre/fswork/projects/rech/vas/uag72kz/clasp}"
STORAGE_ROOT="${CLASP_STORAGE_ROOT:?CLASP_STORAGE_ROOT is required; use a new campaign root, never a historical root}"
ADVANCEMENT_POLICY="${SOTA_ADVANCEMENT_POLICY:?SOTA_ADVANCEMENT_POLICY is required and must be frozen before results}"
PRIOR_STAGE_RECEIPTS_JSON="${SOTA_PRIOR_STAGE_RECEIPTS_JSON:-[]}"
SFT500K_ROOT="${SOTA_SFT500K_ROOT:-${ASSET_ROOT}/experiments/jiangmatched_task_sft500k_20260713_v2}"
SFT_MODEL_PATH="${SOTA_SFT_MODEL_PATH:-${SFT500K_ROOT}/${TASK}/seed42}"
BASE_MODEL_PATH="${SOTA_BASE_MODEL_PATH:?SOTA_BASE_MODEL_PATH is required and must be the approved non-instruction Qwen2.5-7B artifact}"
KL_REFERENCE_MODEL_PATH="${CLASP_KL_REFERENCE_MODEL_PATH:-}"
KL_REFERENCE_RECEIPT="${CLASP_KL_REFERENCE_RECEIPT:-}"
SFT_DATA_ROOT="${SFT_DATA_ROOT:-${ASSET_ROOT}/data/data_sft}"
TASK_DATA_ROOT="${GRPO_DATA_ROOT:-${ASSET_ROOT}/data/data_rl}"
FORBIDDEN_SOURCE_REGISTRY="${SOTA_FORBIDDEN_SOURCE_REGISTRY:?SOTA_FORBIDDEN_SOURCE_REGISTRY is required and must be frozen before IID generation}"
MODEL_INITIALIZATION_REGISTRY="${SOTA_MODEL_INITIALIZATION_REGISTRY:?SOTA_MODEL_INITIALIZATION_REGISTRY is required and must be frozen before publication training}"
RUN_DIR="${STORAGE_ROOT}/runs/${TASK}/${METHOD}/seed${TRAIN_SEED}"
CACHE_ROOT="${SOTA_CACHE_ROOT:-${STORAGE_ROOT}/cache/${TASK}/${METHOD}/seed${TRAIN_SEED}}"
WANDB_PROJECT_NAME="${SOTA_WANDB_PROJECT:-clasp_sota_matched_20260814}"

cd "${PROJECT_ROOT}"
[[ -f "${ADVANCEMENT_POLICY}" ]] || {
    echo "[SOTALaunchError] advancement policy is missing: ${ADVANCEMENT_POLICY}" >&2
    exit 2
}
ADVANCEMENT_POLICY="$(readlink -f "${ADVANCEMENT_POLICY}")"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
[[ -f scripts/prepare_sota_run.py \
    && -f scripts/freeze_remote_environment.py \
    && -f scripts/run_grpo_in_sealed_environment.py \
    && -f scripts/train_clasp_sota_a100.slurm ]] || {
    echo "[SOTALaunchError] invalid project root: ${PROJECT_ROOT}" >&2
    exit 2
}
[[ -f "${PROJECT_ROOT}/SOURCE_MANIFEST.sha256" ]] || {
    echo "[SOTALaunchError] project root is not a frozen source snapshot (missing SOURCE_MANIFEST.sha256)" >&2
    exit 2
}
[[ "${SLURM_GPUS_ON_NODE:-0}" == "4" ]] || {
    echo "[SOTALaunchError] fixed trajectory requires 4 GPUs; got ${SLURM_GPUS_ON_NODE:-0}" >&2
    exit 2
}
[[ -d "${SFT_MODEL_PATH}" && -d "${BASE_MODEL_PATH}" \
    && -d "${SFT_DATA_ROOT}/${TASK}/train" \
    && -d "${TASK_DATA_ROOT}/${TASK}/train" \
    && -f "${FORBIDDEN_SOURCE_REGISTRY}" \
    && -f "${MODEL_INITIALIZATION_REGISTRY}" ]] || {
    echo "[SOTALaunchError] missing SFT/base/data or required frozen registry" >&2
    exit 2
}
"${SEALED_PYTHON}" - "${SFT_MODEL_PATH}" "${BASE_MODEL_PATH}" <<'PY'
import json
import os
import sys

adapter_path = os.path.join(sys.argv[1], "adapter_config.json")
with open(adapter_path, encoding="utf-8") as handle:
    adapter = json.load(handle)
actual = os.path.realpath(str(adapter.get("base_model_name_or_path", "")))
contracted = os.path.realpath(sys.argv[2])
if not actual or actual != contracted:
    raise SystemExit(
        "[SOTALaunchError] adapter base_model_name_or_path does not equal the "
        f"contracted base model: actual={actual or '<missing>'}, contracted={contracted}"
    )
PY
"${SEALED_PYTHON}" -c 'import inspect; from trl import GRPOConfig; required={"temperature","top_p","top_k","repetition_penalty","beta","epsilon","epsilon_high","loss_type","optim","weight_decay","warmup_ratio","lr_scheduler_type","importance_sampling_level"}; missing=sorted(required-set(inspect.signature(GRPOConfig.__init__).parameters)); assert not missing, f"GRPOConfig cannot honor publication contract: {missing}"'
read -r PYTHON_VERSION TORCH_VERSION TRANSFORMERS_VERSION TRL_VERSION PEFT_VERSION CUDA_VERSION < <(
    "${SEALED_PYTHON}" -c 'import platform, peft, torch, transformers, trl; print(platform.python_version(), torch.__version__, transformers.__version__, trl.__version__, peft.__version__, torch.version.cuda or "none")'
)

# sbatch --export=ALL must not inject an old experimental reward knob.  Preserve
# resolved paths above, clear every research-behaviour namespace, then define the
# complete publication environment below.  Runtime/package namespaces are also
# scrubbed so a submit shell cannot alter the sealed environment's behavior.
while IFS='=' read -r inherited_name _; do
    case "${inherited_name}" in
        CLASP_*|PD_*|GENERATION_*|GRPO_*|PYTORCH_*|HF_*|TRANSFORMERS_*|TOKENIZERS_*|BITSANDBYTES_*|UNSLOTH_*|ACCELERATE_*|WANDB_*|NCCL_*) unset "${inherited_name}" ;;
    esac
done < <(env)
unset RESUME_FROM_CHECKPOINT || true

FINAL_STEP="${SOTA_FINAL_STEP:-3200}"
CHUNK_STEPS="${SOTA_CHUNK_STEPS:-400}"
SAVE_STEPS="${SOTA_SAVE_STEPS:-400}"
EVAL_STEPS="${SOTA_EVAL_STEPS:-100}"
NUM_GENERATIONS=9
DEPLOYMENT_SET_SIZE=8
PER_DEVICE_TRAIN_BATCH_SIZE=3
PER_DEVICE_EVAL_BATCH_SIZE=9
GRADIENT_ACCUMULATION_STEPS=3
ROLLOUT_BATCH_SIZE=36
if [[ "${METHOD}" == clasp_canonical_foarl_matched_b9 ]]; then
    MAX_PROMPT_LENGTH=20000
    MAX_COMPLETION_LENGTH=1000
    ARM_CONTRACT_SCHEMA=clasp-canonical-foarl-matched-b9-20k1k-a100-v1
    TOKEN_BUDGET_CONTRACT=max_prompt_20000_max_completion_1000
    BETA="${SOTA_BETA:-0.05}"
else
    MAX_PROMPT_LENGTH=29768
    MAX_COMPLETION_LENGTH=3000
    ARM_CONTRACT_SCHEMA=clasp-canonical-matched-a100-v1
    TOKEN_BUDGET_CONTRACT=max_prompt_29768_max_completion_3000
    BETA="${SOTA_BETA:-0.0}"
fi
LEARNING_RATE=1e-6
[[ "${FINAL_STEP}" =~ ^[1-9][0-9]*$ && "${CHUNK_STEPS}" =~ ^[1-9][0-9]*$ \
    && "${SAVE_STEPS}" =~ ^[1-9][0-9]*$ && "${EVAL_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[SOTALaunchError] final/chunk/save/eval steps must be positive integers" >&2
    exit 2
}
(( CHUNK_STEPS % SAVE_STEPS == 0 && FINAL_STEP % SAVE_STEPS == 0 )) || {
    echo "[SOTALaunchError] final and chunk steps must be divisible by save_steps=${SAVE_STEPS}" >&2
    exit 2
}
(( CHUNK_STEPS % EVAL_STEPS == 0 && FINAL_STEP % EVAL_STEPS == 0 )) || {
    echo "[SOTALaunchError] final and chunk steps must be divisible by eval_steps=${EVAL_STEPS}" >&2
    exit 2
}
SAVE_TOTAL_LIMIT="${SOTA_SAVE_TOTAL_LIMIT:-$((FINAL_STEP / SAVE_STEPS))}"
[[ "${SAVE_TOTAL_LIMIT}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[SOTALaunchError] save_total_limit must be a positive integer" >&2
    exit 2
}
(( SAVE_TOTAL_LIMIT >= FINAL_STEP / SAVE_STEPS )) || {
    echo "[SOTALaunchError] save_total_limit=${SAVE_TOTAL_LIMIT} would delete a prespecified checkpoint; require at least $((FINAL_STEP / SAVE_STEPS))" >&2
    exit 2
}
BETA_IS_POSITIVE="$("${SEALED_PYTHON}" - "${BETA}" <<'PY'
import math
import sys

try:
    beta = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit("[SOTALaunchError] SOTA_BETA must be a finite nonnegative number") from exc
if not math.isfinite(beta) or beta < 0.0:
    raise SystemExit("[SOTALaunchError] SOTA_BETA must be a finite nonnegative number")
print("1" if beta > 0.0 else "0")
PY
)"
if [[ "${METHOD}" == clasp_canonical_foarl_matched_b9 ]]; then
    "${SEALED_PYTHON}" - "${BETA}" <<'PY'
import math
import sys

try:
    beta = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit("[SOTALaunchError] matched-B9 CLASP comparator requires beta=0.05") from exc
if not math.isfinite(beta) or beta != 0.05:
    raise SystemExit("[SOTALaunchError] matched-B9 CLASP comparator requires beta=0.05")
PY
fi
if [[ "${BETA_IS_POSITIVE}" == "1" ]]; then
    [[ -n "${KL_REFERENCE_MODEL_PATH}" && -n "${KL_REFERENCE_RECEIPT}" \
        && -d "${KL_REFERENCE_MODEL_PATH}" && -f "${KL_REFERENCE_RECEIPT}" ]] || {
        echo "[SOTALaunchError] beta>0 requires CLASP_KL_REFERENCE_MODEL_PATH and CLASP_KL_REFERENCE_RECEIPT" >&2
        exit 2
    }
    KL_REFERENCE_MODEL_PATH="$(readlink -f "${KL_REFERENCE_MODEL_PATH}")"
    KL_REFERENCE_RECEIPT="$(readlink -f "${KL_REFERENCE_RECEIPT}")"
    SFT_MODEL_PATH="$(readlink -f "${SFT_MODEL_PATH}")"
    [[ "${KL_REFERENCE_MODEL_PATH}" == "${SFT_MODEL_PATH}" ]] || {
        echo "[SOTALaunchError] beta>0 exact PEFT KL reference must be the same specialist SFT adapter used to initialize the policy" >&2
        exit 2
    }
    [[ "${KL_REFERENCE_RECEIPT}" == "${SFT_MODEL_PATH}/checkpoint_fidelity.json" ]] || {
        echo "[SOTALaunchError] beta>0 KL receipt must be the specialist adapter's authoritative checkpoint_fidelity.json" >&2
        exit 2
    }
fi
TRAINING_MODEL_PATH="${SFT_MODEL_PATH}"
read -r ADVANCEMENT_POLICY_SHA256 ADVANCEMENT_STAGE_ID ADVANCEMENT_STAGE_INDEX ADVANCEMENT_PRIOR_STAGE_RECEIPT_CHAIN_SHA256 < <(
    "${SEALED_PYTHON}" - "${ADVANCEMENT_POLICY}" "${CAMPAIGN_ID}" "${TASK}" "${METHOD}" \
        "0.005" "${SOTA_CANONICAL_FIXED_LAMBDA:-}" \
        "${FIXED_NULL_RECEIPT}" \
        "${TRAIN_SEED}" "${FINAL_STEP}" "${SAVE_STEPS}" "${EVAL_STEPS}" \
        "${CHUNK_STEPS}" "${SAVE_TOTAL_LIMIT}" \
        "${PROJECT_ROOT}" "${STORAGE_ROOT}" \
        "${MODEL_INITIALIZATION_REGISTRY}" "${PRIOR_STAGE_RECEIPTS_JSON}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

(
    policy_path,
    campaign_id,
    task,
    method,
    target_failure_raw,
    fixed_lambda_raw,
    fixed_null_receipt_raw,
    seed_raw,
    final_raw,
    save_raw,
    eval_raw,
    chunk_raw,
    save_total_limit_raw,
    source_root,
    storage_root,
    model_initialization_registry_path,
    prior_stage_receipts_json,
) = sys.argv[1:]
try:
    target_failure = float(target_failure_raw)
except ValueError as exc:
    raise SystemExit("[SOTALaunchError] invalid frozen failure target") from exc
if target_failure != 0.005:
    raise SystemExit("[SOTALaunchError] canonical publication failure target must be 0.005")
with open(policy_path, encoding="utf-8") as handle:
    policy = json.load(handle)
if not isinstance(policy, dict):
    raise SystemExit("[SOTALaunchError] advancement policy is not a JSON object")
stored_hash = policy.get("policy_sha256")
unsigned = dict(policy)
unsigned.pop("policy_sha256", None)
observed_hash = hashlib.sha256(
    json.dumps(
        unsigned,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
if stored_hash != observed_hash:
    raise SystemExit("[SOTALaunchError] advancement policy self-hash mismatch")
sys.path.insert(0, os.path.join(source_root, "scripts"))
try:
    import audit_sota_advancement as advancement_audit
except ImportError as exc:
    raise SystemExit(
        "[SOTALaunchError] cannot import the frozen advancement auditor"
    ) from exc
try:
    validated_policy = advancement_audit._validate_policy(pathlib.Path(policy_path))
except (OSError, TypeError, ValueError) as exc:
    raise SystemExit(f"[SOTALaunchError] advancement policy is invalid: {exc}") from exc
if validated_policy != policy:
    raise SystemExit("[SOTALaunchError] advancement policy changed during validation")
try:
    prior_stage_receipts = json.loads(prior_stage_receipts_json)
except json.JSONDecodeError as exc:
    raise SystemExit(
        "[SOTALaunchError] SOTA_PRIOR_STAGE_RECEIPTS_JSON is not valid JSON"
    ) from exc
if not isinstance(prior_stage_receipts, list) or any(
    not isinstance(path, str)
    or not path
    or not os.path.isabs(path)
    or os.path.realpath(path) != path
    for path in prior_stage_receipts
):
    raise SystemExit(
        "[SOTALaunchError] SOTA_PRIOR_STAGE_RECEIPTS_JSON must be an ordered JSON "
        "array of canonical absolute paths"
    )
prior_authorization = advancement_audit._prior_stage_authorization_audit(
    policy,
    [pathlib.Path(path) for path in prior_stage_receipts],
)
if prior_authorization["errors"]:
    raise SystemExit(
        "[SOTALaunchError] prior-stage authorization failed: "
        + "; ".join(str(error) for error in prior_authorization["errors"][:8])
    )
with open(model_initialization_registry_path, encoding="utf-8") as handle:
    model_registry = json.load(handle)
if not isinstance(model_registry, dict):
    raise SystemExit("[SOTALaunchError] model-initialization registry is not an object")
model_registry_unsigned = dict(model_registry)
model_registry_stored_hash = model_registry_unsigned.pop("registry_sha256", None)
model_registry_observed_hash = hashlib.sha256(
    json.dumps(
        model_registry_unsigned,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
if (
    model_registry_stored_hash != model_registry_observed_hash
    or model_registry.get("schema_version") != 2
    or model_registry.get("protocol") != "clasp-model-initialization-registry-v2"
    or model_registry.get("status") != "frozen_before_training"
    or model_registry.get("model_family") != "Qwen2.5-7B"
    or sorted((model_registry.get("specialist_sft_by_task") or {}).keys())
    != ["cvrp", "jssp", "mis", "op", "tsp"]
):
    raise SystemExit("[SOTALaunchError] model-initialization registry is invalid")
requirements = policy.get("contract_requirements")
expected_requirements = {
    "world_size": 4,
    "sft_seed": 42,
    "num_generations": 9,
    "logging_steps": 1,
    "save_steps": int(save_raw),
    "eval_steps": int(eval_raw),
}
checks = {
    "schema_version": policy.get("schema_version") == 1,
    "status": policy.get("status") == "frozen_before_results",
    "campaign_id": policy.get("campaign_id") == campaign_id,
    "task": task in (policy.get("expected_tasks") or []),
    "method": method in (policy.get("expected_methods") or []),
    "canonical_target_failure_rate": policy.get(
        "canonical_target_failure_rate"
    )
    == target_failure,
    "seed": int(seed_raw) in (policy.get("train_seeds") or []),
    "target_step": policy.get("target_step") == int(final_raw),
    "chunk_steps": policy.get("chunk_steps") == int(chunk_raw),
    "save_total_limit": policy.get("save_total_limit")
    == int(save_total_limit_raw),
    "requirements": requirements == expected_requirements,
    "source_snapshot_id": policy.get("source_snapshot_id")
    == pathlib.Path(source_root).name,
    "artifact_root": os.path.realpath(str(policy.get("artifact_root", "")))
    == os.path.realpath(storage_root),
    "model_initialization_registry_sha256": policy.get(
        "model_initialization_registry_sha256"
    )
    == model_registry_observed_hash,
}
fixed_by_task = policy.get("canonical_fixed_lambda_by_task")
if method == "clasp_canonical_fixed_lambda":
    try:
        supplied_fixed = float(fixed_lambda_raw)
        registered_fixed = float(
            fixed_by_task.get(task) if isinstance(fixed_by_task, dict) else None
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "[SOTALaunchError] fixed-lambda arm is missing its preregistered task value"
        ) from exc
    checks["canonical_fixed_lambda_by_task"] = (
        supplied_fixed >= 0.0
        and supplied_fixed == registered_fixed
    )
elif fixed_lambda_raw:
    checks["unexpected_fixed_lambda"] = False
fixed_null_by_task = policy.get("fixed_null_receipt_sha256_by_task")
if method == "set_null":
    try:
        from clasp.set_grpo.fixed_null_receipt import load_fixed_null_score_receipt

        fixed_null_identity = load_fixed_null_score_receipt(
            fixed_null_receipt_raw
        ).identity
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"[SOTALaunchError] fixed-null construction receipt is invalid: {exc}"
        ) from exc
    registered_receipt_sha256 = (
        fixed_null_by_task.get(task)
        if isinstance(fixed_null_by_task, dict)
        else None
    )
    checks["fixed_null_receipt_sha256_by_task"] = (
        fixed_null_identity["receipt_sha256"] == registered_receipt_sha256
        and task in fixed_null_identity["tasks"]
    )
elif fixed_null_receipt_raw:
    checks["unexpected_fixed_null_receipt"] = False
failed = sorted(name for name, passed in checks.items() if not passed)
stage_id = str(policy.get("stage_id", "") or "")
stage_index = policy.get("stage_index")
if failed or not stage_id or not stage_id.replace("-", "").replace("_", "").replace(".", "").isalnum():
    raise SystemExit(
        "[SOTALaunchError] advancement policy does not authorize this launch: "
        + ", ".join(failed or ["stage_id"])
    )
print(
    observed_hash,
    stage_id,
    stage_index,
    prior_authorization["chain_sha256"],
)
PY
)

export CLASP_PROJECT_ROOT="${PROJECT_ROOT}"
export CLASP_ASSET_ROOT="${ASSET_ROOT}"
export CLASP_STORAGE_ROOT="${STORAGE_ROOT}"
export CLASP_LORA_TARGET_MODULES=jiang
export CLASP_LORA_BIAS=all
export CLASP_PUBLICATION_FIDELITY=1
if [[ "${BETA_IS_POSITIVE}" == "1" ]]; then
    export CLASP_KL_REFERENCE_MODEL_PATH="${KL_REFERENCE_MODEL_PATH}"
    export CLASP_KL_REFERENCE_RECEIPT="${KL_REFERENCE_RECEIPT}"
fi
if [[ "${METHOD}" == clasp_canonical_foarl_matched_b9 ]]; then
    if [[ "${TASK}" == mis ]]; then export CLASP_MIS_JIANG_RESPONSE_OUTPUT=1; else export CLASP_MIS_JIANG_RESPONSE_OUTPUT=0; fi
else
    export CLASP_MIS_JIANG_RESPONSE_OUTPUT=1
fi
export CLASP_TSP_PROMPT_KNN=0
export CLASP_TSP_PROMPT_KNN_DISTANCES=0
if [[ "${METHOD}" == clasp_canonical_foarl_matched_b9 ]]; then
    export CLASP_TSP_INPUT_ORDER=original
    export CLASP_TSP_RELABEL_ORDER=original
    export CLASP_TSP_ROUTE_CANONICALIZE=original
    export CLASP_TSP_LABEL_TTA_SCHEDULE=""
    export CLASP_MIS_RELABEL_ORDER=original
    export CLASP_MIS_PROMPT_NODE_DEGREES=0
    export CLASP_MIS_PROMPT_LOW_DEGREE_K=32
else
    export CLASP_TSP_INPUT_ORDER=""
    export CLASP_TSP_RELABEL_ORDER=""
    export CLASP_TSP_ROUTE_CANONICALIZE=""
fi
export CLASP_TSP_ROUTE_ONLY_OUTPUT=0
export CLASP_OBJECTIVE_QUALITY_MODE=exp
export CLASP_OBJECTIVE_GAP_SLOPE=3.0
export CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT=0.0

# Method profiles are deliberately closed: every difference below is named in
# the immutable run contract.  Supplying a fixed lambda to another arm is an
# error instead of an ignored research knob.
CANONICAL_SET=0
FIXED_NULL_SET=0
USE_CLASP=0
PD_GROUP_SCALARIZATION=1
PD_SEPARATE_Z=0
COMPONENT_DUAL_LR=0.0
BON_DUAL_LR=0.0
BON_LAMBDA_INIT=0.0
CANONICAL_RECOVERY_SCALE=0.0
POLICY_CREDIT_FORMULA=group_z_verified_quality_v1
COMPONENT_CONTROLLER=frozen_zero
SET_CONTROLLER=off
if [[ "${METHOD}" != "clasp_canonical_fixed_lambda" \
      && -n "${SOTA_CANONICAL_FIXED_LAMBDA:-}" ]]; then
    echo "[SOTALaunchError] SOTA_CANONICAL_FIXED_LAMBDA is valid only for clasp_canonical_fixed_lambda" >&2
    exit 2
fi
case "${METHOD}" in
    grpo_z)
        ;;
    dual_scalar_z)
        COMPONENT_DUAL_LR=0.05
        POLICY_CREDIT_FORMULA=group_z_scalarized_positive_component_penalty_v1
        COMPONENT_CONTROLLER=legacy_projected_ema_component_gap
        ;;
    dual_separate_z)
        PD_GROUP_SCALARIZATION=0
        PD_SEPARATE_Z=1
        COMPONENT_DUAL_LR=0.05
        POLICY_CREDIT_FORMULA=group_z_reward_minus_group_z_positive_component_penalty_v1
        COMPONENT_CONTROLLER=legacy_projected_ema_component_gap
        ;;
    clasp_canonical)
        CANONICAL_SET=1
        USE_CLASP=1
        PD_GROUP_SCALARIZATION=0
        BON_DUAL_LR=0.05
        CANONICAL_RECOVERY_SCALE=1.0
        POLICY_CREDIT_FORMULA=canonical_k8_leave_out_set_utility_v1
        SET_CONTROLLER=signed_failure_at_k_ema
        ;;
    clasp_canonical_foarl_matched_b9)
        CANONICAL_SET=1
        USE_CLASP=1
        PD_GROUP_SCALARIZATION=0
        BON_DUAL_LR=0.05
        CANONICAL_RECOVERY_SCALE=1.0
        POLICY_CREDIT_FORMULA=canonical_k8_leave_out_set_utility_v1
        SET_CONTROLLER=signed_failure_at_k_ema
        ;;
    clasp_canonical_fixed_lambda)
        CANONICAL_SET=1
        USE_CLASP=1
        PD_GROUP_SCALARIZATION=0
        CANONICAL_RECOVERY_SCALE=1.0
        POLICY_CREDIT_FORMULA=canonical_k8_leave_out_set_utility_v1
        SET_CONTROLLER=frozen
        BON_LAMBDA_INIT="${SOTA_CANONICAL_FIXED_LAMBDA:?SOTA_CANONICAL_FIXED_LAMBDA must be preregistered for the fixed-lambda arm}"
        BON_LAMBDA_INIT="$("${SEALED_PYTHON}" - "${BON_LAMBDA_INIT}" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit("[SOTALaunchError] fixed lambda must be a finite nonnegative number") from exc
if not math.isfinite(value) or value < 0.0:
    raise SystemExit("[SOTALaunchError] fixed lambda must be a finite nonnegative number")
print(format(value, ".17g"))
PY
)"
        ;;
    clasp_canonical_no_recovery)
        CANONICAL_SET=1
        USE_CLASP=1
        PD_GROUP_SCALARIZATION=0
        BON_DUAL_LR=0.05
        POLICY_CREDIT_FORMULA=canonical_k8_leave_out_set_utility_v1
        SET_CONTROLLER=signed_failure_at_k_ema
        ;;
    set_null)
        FIXED_NULL_SET=1
        USE_CLASP=1
        PD_GROUP_SCALARIZATION=0
        BON_DUAL_LR=0.05
        CANONICAL_RECOVERY_SCALE=1.0
        POLICY_CREDIT_FORMULA=fixed_null_k8_same_set_utility_v1
        SET_CONTROLLER=signed_failure_at_k_ema
        ;;
esac

export CLASP_CANONICAL_SET="${CANONICAL_SET}"
export CLASP_FIXED_NULL_SET="${FIXED_NULL_SET}"
if [[ "${FIXED_NULL_SET}" == 1 ]]; then
    export CLASP_FIXED_NULL_RECEIPT_PATH="${FIXED_NULL_RECEIPT}"
else
    unset CLASP_FIXED_NULL_RECEIPT_PATH || true
fi
export CLASP_DEPLOYMENT_SET_SIZE="${DEPLOYMENT_SET_SIZE}"
export CLASP_CANONICAL_RECOVERY_SCALE="${CANONICAL_RECOVERY_SCALE}"
export CLASP_CANONICAL_RECOVERY_TEMPERATURE=0.15
export CLASP_CANONICAL_MAX_SUBSETS=9
export CLASP_BON_ALIGNED="${USE_CLASP}"
export CLASP_FRONTIER_CDF=0
export CLASP_BON_TARGET_FAILURE_RATE=0.005
export CLASP_BON_TARGET_FAILURE_RATES="${TASK}:0.005"
export CLASP_BON_DUAL_LR="${BON_DUAL_LR}"
export CLASP_BON_DUAL_EMA_BETA=0.9
export CLASP_BON_LAMBDA_INIT="${BON_LAMBDA_INIT}"
export CLASP_BON_LAMBDA_MAX=100.0
export CLASP_BON_QUALITY_TEMPERATURE=0.0
export CLASP_BON_DENSE_QUALITY_SCALE=1.0
export CLASP_BON_DENSE_FEASIBILITY_SCALE=1.0
export CLASP_BON_DENSE_VIOLATION_MIX=0.5
export CLASP_BON_RECOVERY_SCALE=0.0
export CLASP_COMPONENT_DUAL_RECOVERY_SCALE=0.0
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
export CLASP_FEASIBILITY_GATED="${USE_CLASP}"
export CLASP_LEXICOGRAPHIC_GATE=0
export CLASP_FEASIBILITY_WARMUP_STEPS=0
export CLASP_FEASIBILITY_WARMUP_TARGETS="${TASK}:0.995"
export CLASP_FEASIBILITY_WARMUP_HOLD_UNTIL_EVAL=1
export CLASP_SELF_REPLAY_ENABLED=0
export CLASP_SELF_REPLAY_IMITATION_LOSS_WEIGHT=0.0
export CLASP_SELF_REPLAY_CONTRASTIVE_LOSS_WEIGHT=0.0
export CLASP_SELF_REPLAY_SAMPLE_RATE=0.5
export CLASP_SELF_REPLAY_MAX_TOKENS=512
export CLASP_SELF_REPLAY_MAX_KEYS=20000
export CLASP_SELF_REPLAY_MAX_PER_KEY=4
export CLASP_SELF_REPLAY_MIN_BUFFER_SIZE=8
export CLASP_VALIDATION_GATE_CHECKPOINTS=0
export PD_IMPORTANCE_SAMPLING_LEVEL=token
export PD_ALLOW_TOKEN_IMPORTANCE_ABLATION=1
export GRPO_DATA_SEED="${TRAIN_SEED}"
export GRPO_STRICT_OPTIMIZER_RESUME=1

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED="${TRAIN_SEED}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TMPDIR="${CACHE_ROOT}/tmp"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export WANDB_MODE=offline
export WANDB_DIR="${CACHE_ROOT}/wandb"
export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
export WANDB_RUN_GROUP="${CAMPAIGN_ID}_${TASK}_${METHOD}"
export WANDB_RUN_NAME_OVERRIDE="${CAMPAIGN_ID}-${TASK}-${METHOD}-seed${TRAIN_SEED}"
export WANDB_NAME="${WANDB_RUN_NAME_OVERRIDE}"
export WANDB_RESUME=allow
export WANDB_RUN_ID="${WANDB_RUN_NAME_OVERRIDE}"
unset WANDB_SWEEP_ID || true
mkdir -p "${CACHE_ROOT}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}" \
    "${TORCH_EXTENSIONS_DIR}" "${WANDB_DIR}" "${PROJECT_ROOT}/logs" "${STORAGE_ROOT}/logs"

ALLOWED_ENV=(
    CLASP_PROJECT_ROOT CLASP_ASSET_ROOT CLASP_STORAGE_ROOT CLASP_LORA_TARGET_MODULES
    CLASP_LORA_BIAS CLASP_PUBLICATION_FIDELITY CLASP_MIS_JIANG_RESPONSE_OUTPUT
    CLASP_TSP_PROMPT_KNN
    CLASP_TSP_PROMPT_KNN_DISTANCES CLASP_TSP_INPUT_ORDER CLASP_TSP_RELABEL_ORDER
    CLASP_TSP_ROUTE_CANONICALIZE CLASP_TSP_ROUTE_ONLY_OUTPUT CLASP_TSP_LABEL_TTA_SCHEDULE
    CLASP_MIS_RELABEL_ORDER CLASP_MIS_PROMPT_NODE_DEGREES CLASP_MIS_PROMPT_LOW_DEGREE_K
    CLASP_OBJECTIVE_QUALITY_MODE
    CLASP_OBJECTIVE_GAP_SLOPE CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT
    CLASP_CANONICAL_SET CLASP_FIXED_NULL_SET CLASP_FIXED_NULL_RECEIPT_PATH
    CLASP_DEPLOYMENT_SET_SIZE CLASP_CANONICAL_RECOVERY_SCALE
    CLASP_CANONICAL_RECOVERY_TEMPERATURE CLASP_CANONICAL_MAX_SUBSETS CLASP_BON_ALIGNED
    CLASP_FRONTIER_CDF CLASP_BON_TARGET_FAILURE_RATE CLASP_BON_TARGET_FAILURE_RATES
    CLASP_BON_DUAL_LR CLASP_BON_DUAL_EMA_BETA CLASP_BON_LAMBDA_INIT
    CLASP_BON_LAMBDA_MAX CLASP_BON_QUALITY_TEMPERATURE
    CLASP_BON_DENSE_QUALITY_SCALE CLASP_BON_DENSE_FEASIBILITY_SCALE
    CLASP_BON_DENSE_VIOLATION_MIX CLASP_BON_RECOVERY_SCALE
    CLASP_COMPONENT_DUAL_RECOVERY_SCALE
    CLASP_MIXED_ADAPTIVE_LENGTH_MODE CLASP_AUX_GROUPING_STRICT
    CLASP_PAIRWISE_LOSS_WEIGHT CLASP_ELITE_IMITATION_LOSS_WEIGHT
    CLASP_MULTI_ELITE_IMITATION_LOSS_WEIGHT CLASP_REFERENCE_IMITATION_LOSS_WEIGHT
    CLASP_REFERENCE_CONTRASTIVE_LOSS_WEIGHT CLASP_TSP_TWO_OPT_DISTILL_LOSS_WEIGHT
    CLASP_CVRP_LOCAL_SEARCH_DISTILL_LOSS_WEIGHT
    CLASP_JSSP_ACTIVE_SCHEDULE_DISTILL_LOSS_WEIGHT
    CLASP_MIS_MAXIMALIZE_DISTILL_LOSS_WEIGHT CLASP_HARD_GROUP_WEIGHT_SCALE
    CLASP_FEASIBILITY_GATED CLASP_LEXICOGRAPHIC_GATE CLASP_FEASIBILITY_WARMUP_STEPS
    CLASP_FEASIBILITY_WARMUP_TARGETS CLASP_FEASIBILITY_WARMUP_HOLD_UNTIL_EVAL
    CLASP_SELF_REPLAY_ENABLED CLASP_SELF_REPLAY_IMITATION_LOSS_WEIGHT
    CLASP_SELF_REPLAY_CONTRASTIVE_LOSS_WEIGHT CLASP_SELF_REPLAY_SAMPLE_RATE
    CLASP_SELF_REPLAY_MAX_TOKENS CLASP_SELF_REPLAY_MAX_KEYS
    CLASP_SELF_REPLAY_MAX_PER_KEY CLASP_SELF_REPLAY_MIN_BUFFER_SIZE
    CLASP_VALIDATION_GATE_CHECKPOINTS PD_IMPORTANCE_SAMPLING_LEVEL
    PD_ALLOW_TOKEN_IMPORTANCE_ABLATION GRPO_DATA_SEED GRPO_STRICT_OPTIMIZER_RESUME
)
if [[ "${BETA_IS_POSITIVE}" == "1" ]]; then
    ALLOWED_ENV+=(CLASP_KL_REFERENCE_MODEL_PATH CLASP_KL_REFERENCE_RECEIPT)
fi
CONTRACT_ARGS=(
    --run-dir "${RUN_DIR}" --method "${METHOD}" --task "${TASK}"
    --train-seed "${TRAIN_SEED}" --sft-seed 42 --final-step "${FINAL_STEP}" --world-size 4
    --model-path "${SFT_MODEL_PATH}" --base-model-path "${BASE_MODEL_PATH}"
    --data-path "${TASK_DATA_ROOT}/${TASK}/train"
    --sft-data-path "${SFT_DATA_ROOT}/${TASK}/train"
    --forbidden-source-registry "${FORBIDDEN_SOURCE_REGISTRY}"
    --model-initialization-registry "${MODEL_INITIALIZATION_REGISTRY}"
    --remote-environment-receipt "${REMOTE_ENVIRONMENT_RECEIPT}"
    --source-root "${PROJECT_ROOT}"
    --source-manifest "${PROJECT_ROOT}/SOURCE_MANIFEST.sha256"
    --generation-config "num_generations=${NUM_GENERATIONS}"
    --generation-config "generation_batch_size=${ROLLOUT_BATCH_SIZE}"
    --generation-config "temperature=0.7" --generation-config "top_p=0.9"
    --generation-config "top_k=50" --generation-config "repetition_penalty=1.0"
    --generation-config "max_prompt_length=${MAX_PROMPT_LENGTH}"
    --generation-config "max_completion_length=${MAX_COMPLETION_LENGTH}"
    --config "entrypoint=clasp.entrypoints.train_grpo"
    --config "arm_contract_schema=${ARM_CONTRACT_SCHEMA}"
    --config "policy_surrogate=clipped_token_level_practical"
    --config "policy_credit_formula=${POLICY_CREDIT_FORMULA}"
    --config "component_controller=${COMPONENT_CONTROLLER}"
    --config "set_controller=${SET_CONTROLLER}"
    --config "deployment_budget_k=${DEPLOYMENT_SET_SIZE}"
    --config "verifier_gate=strict_recomputed_feasible_only_quality"
    --config "intentional_method_diff=policy_credit_formula,component_controller,set_controller,use_clasp,clasp_canonical_set,clasp_fixed_null_set,pd_group_scalarization,pd_separate_z,pd_dual_lr,pd_dual_lr_up,pd_dual_lr_down,clasp_bon_dual_lr,clasp_bon_lambda_init,clasp_canonical_recovery_scale"
    --config "campaign_id=${CAMPAIGN_ID}" --config "wandb_project=${WANDB_PROJECT_NAME}"
    --config "advancement_policy_sha256=${ADVANCEMENT_POLICY_SHA256}"
    --config "advancement_stage_id=${ADVANCEMENT_STAGE_ID}"
    --config "advancement_stage_index=${ADVANCEMENT_STAGE_INDEX}"
    --config "advancement_prior_stage_receipt_chain_sha256=${ADVANCEMENT_PRIOR_STAGE_RECEIPT_CHAIN_SHA256}"
    --config "prompt_style=alpaca" --config "no_reason=true" --config "dtype=bfloat16"
    --config "python_version=${PYTHON_VERSION}" --config "torch_version=${TORCH_VERSION}"
    --config "transformers_version=${TRANSFORMERS_VERSION}" --config "trl_version=${TRL_VERSION}"
    --config "peft_version=${PEFT_VERSION}" --config "cuda_version=${CUDA_VERSION}"
    --config "lora_r=64" --config "lora_alpha=64" --config "lora_dropout=0.0"
    --config "lora_bias=all" --config "lora_bias_persistence_audit=required"
    --config "checkpoint_fidelity=publication_exact_required"
    --config "per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --config "per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE}"
    --config "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
    --config "learning_rate=${LEARNING_RATE}" --config "optim=adamw_8bit"
    --config "weight_decay=0.01" --config "warmup_ratio=0.03"
    --config "lr_scheduler_type=linear" --config "loss_type=grpo"
    --config "beta=${BETA}" --config "epsilon=0.2" --config "epsilon_high=0.2"
    --config "eval_ratio=0.05" --config "eval_from_train_split=true"
    --config "max_samples_per_task=0" --config "max_eval_samples_per_task=100"
    --config "min_train_samples_per_task=1900" --config "min_eval_samples_per_task=100"
    --config "save_steps=${SAVE_STEPS}" --config "eval_steps=${EVAL_STEPS}"
    --config "logging_steps=1" --config "save_total_limit=${SAVE_TOTAL_LIMIT}"
    --config "final_max_steps=${FINAL_STEP}"
    --config "use_primal_dual=true"
    --config "use_clasp=$([[ "${USE_CLASP}" == 1 ]] && echo true || echo false)"
    --config "clasp_canonical_set=$([[ "${CANONICAL_SET}" == 1 ]] && echo true || echo false)"
    --config "clasp_fixed_null_set=$([[ "${FIXED_NULL_SET}" == 1 ]] && echo true || echo false)"
    --config "clasp_deployment_set_size=${DEPLOYMENT_SET_SIZE}"
    --config "clasp_canonical_recovery_scale=${CANONICAL_RECOVERY_SCALE}"
    --config "clasp_canonical_recovery_temperature=0.15"
    --config "clasp_canonical_max_subsets=9"
    --config "pd_constraint_target=1.0" --config "pd_cost_limit=0.0"
    --config "pd_lambda_init=0.0" --config "pd_dual_lr=${COMPONENT_DUAL_LR}"
    --config "pd_dual_lr_up=${COMPONENT_DUAL_LR}" --config "pd_dual_lr_down=${COMPONENT_DUAL_LR}"
    --config "pd_dual_update_interval=1" --config "pd_violation_ema_beta=0.1"
    --config "pd_lambda_max=100.0" --config "pd_violation_tolerance=0.0"
    --config "pd_positive_constraint_penalty=true"
    --config "pd_objective_weight=1.0" --config "pd_structure_weight=0.0"
    --config "pd_soft_objective_for_infeasible=false"
    --config "pd_group_scalarization=$([[ "${PD_GROUP_SCALARIZATION}" == 1 ]] && echo true || echo false)"
    --config "pd_separate_z=$([[ "${PD_SEPARATE_Z}" == 1 ]] && echo true || echo false)"
    --config "pd_component_profile=co_feasibility_lock_all"
    --config "clasp_bon_dual_lr=${BON_DUAL_LR}"
    --config "clasp_bon_lambda_init=${BON_LAMBDA_INIT}"
    --config "clasp_bon_target_failure_rate=0.005"
    --config "clasp_component_dual_recovery_scale=0.0"
    --config "clasp_strict_defaults=true" --config "clasp_raw_feasibility_targeting=false"
    --config "clasp_feasibility_warmup_steps=0" --config "clasp_self_replay=false"
    --config "clasp_pairwise_loss_weight=0.0" --config "clasp_elite_loss_weight=0.0"
    --config "clasp_proper_dual=false" --config "clasp_dual_update_interval=1"
    --config "clasp_adaptive_length=false"
    --config "clasp_mixed_adaptive_length_mode=error"
    --config "importance_sampling_level=token"
)
if [[ "${METHOD}" == clasp_canonical_foarl_matched_b9 ]]; then
    CONTRACT_ARGS+=(
        --config "comparison_protocol=foarl-f638-clasp-shared-compute-data-token-design-v1"
        --config "comparison_baseline=foarl_f638_matched_b9"
        --config "attribution=common_compute_data_token_schedule_only"
        --config "reproduction_claim=not_an_official_jiang_reproduction"
        --config "prefer_subset_files=false"
        --config "train_size_bucket=all"
        --config "source_rows=registry_bound"
        --config "data_order=registered_rows_seeded_shuffle"
        --config "data_seed=${TRAIN_SEED}"
        --config "hardware_contract=sealed_4xa100_clasp_matched"
        --config "token_budget_contract=${TOKEN_BUDGET_CONTRACT}"
    )
fi
if [[ "${BETA_IS_POSITIVE}" == "1" ]]; then
    CONTRACT_ARGS+=(
        --fixed-kl-reference-model-path "${KL_REFERENCE_MODEL_PATH}"
        --fixed-kl-reference-receipt-path "${KL_REFERENCE_RECEIPT}"
    )
fi
if [[ "${FIXED_NULL_SET}" == 1 ]]; then
    CONTRACT_ARGS+=(--fixed-null-receipt "${FIXED_NULL_RECEIPT}")
fi
for allowed_name in "${ALLOWED_ENV[@]}"; do CONTRACT_ARGS+=(--allowed-env "${allowed_name}"); done

# The absolute path is only an operational pointer.  The pre/post-step fidelity
# receipts reopen and bind the immutable run-contract bytes; prepare_sota_run
# deliberately excludes this path itself from the relocatable contract payload.
export CLASP_TRAINING_RUN_CONTRACT_PATH="${RUN_DIR}/run_contract.json"
state="$("${SEALED_PYTHON}" scripts/prepare_sota_run.py "${CONTRACT_ARGS[@]}")"
case "${state}" in
    COMPLETE:*) echo "[SOTALaunch] already complete: ${state#COMPLETE:}"; exit 0 ;;
    FRESH) current_step=0; unset RESUME_FROM_CHECKPOINT || true ;;
    RESUME:*)
        resume_checkpoint="${state#RESUME:}"
        current_step="${resume_checkpoint##*-}"
        [[ "${current_step}" =~ ^[1-9][0-9]*$ ]] || { echo "[SOTALaunchError] invalid resume step" >&2; exit 2; }
        export RESUME_FROM_CHECKPOINT="${resume_checkpoint}"
        ;;
    *) echo "[SOTALaunchError] unexpected contract state: ${state}" >&2; exit 2 ;;
esac
exec 9>"${RUN_DIR}/.active_training.lock"
flock -n 9 || {
    echo "[SOTALaunchError] another job already owns this task/method/seed run" >&2
    exit 2
}

stop_step=$(( ((current_step / CHUNK_STEPS) + 1) * CHUNK_STEPS ))
(( stop_step > FINAL_STEP )) && stop_step="${FINAL_STEP}"
[[ "${stop_step}" =~ ^[1-9][0-9]*$ ]] || { echo "[SOTALaunchError] stop step is invalid" >&2; exit 2; }
(( stop_step > current_step && stop_step <= FINAL_STEP && stop_step % SAVE_STEPS == 0 && stop_step % EVAL_STEPS == 0 )) || {
    echo "[SOTALaunchError] stop=${stop_step} must be > current=${current_step}, <= final=${FINAL_STEP}, and divisible by save=${SAVE_STEPS} and eval=${EVAL_STEPS}" >&2
    exit 2
}
export GRPO_STOP_AT_GLOBAL_STEP="${stop_step}"

COMMON_ARGS=(
    --tasks "${TASK}" --model_path "${TRAINING_MODEL_PATH}" --tokenizer_path "${TRAINING_MODEL_PATH}"
    --data_root "${TASK_DATA_ROOT}" --output_root "${STORAGE_ROOT}" --output_dir "${RUN_DIR}"
    --prompt_style alpaca --no_reason --dtype bfloat16 --lora_r 64 --lora_alpha 64
    --lora_dropout 0.0 --seed "${TRAIN_SEED}" --eval_ratio 0.05 --eval-from-train-split
    --max_samples_per_task 0 --max_eval_samples_per_task 100
    --min_train_samples_per_task 1900 --min_eval_samples_per_task 100 --no-prefer_subset_files
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate "${LEARNING_RATE}" --optim adamw_8bit --weight_decay 0.01
    --warmup_ratio 0.03 --lr_scheduler_type linear --num_train_epochs 1
    --max_steps "${FINAL_STEP}" --max_prompt_length "${MAX_PROMPT_LENGTH}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}" --num_generations "${NUM_GENERATIONS}"
    --generation_batch_size "${ROLLOUT_BATCH_SIZE}" --generation_temperature 0.7
    --generation_top_p 0.9 --generation_top_k 50 --generation_repetition_penalty 1.0
    --grpo_beta "${BETA}" --grpo_epsilon 0.2 --grpo_epsilon_high 0.2 --grpo_loss_type grpo
    --verify_gt_rewards --reward_check_samples_per_task 5 --train_size_bucket all
    --logging_steps 1 --save_steps "${SAVE_STEPS}" --eval_steps "${EVAL_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}" --report_to wandb --wandb_project "${WANDB_PROJECT}"
    --wandb_group "${WANDB_RUN_GROUP}"
)
PD_ARGS=(
    --pd_constraint_target 1.0 --pd_cost_limit 0.0 --pd_lambda_init 0.0
    --pd_dual_lr "${COMPONENT_DUAL_LR}" --pd_dual_lr_up "${COMPONENT_DUAL_LR}"
    --pd_dual_lr_down "${COMPONENT_DUAL_LR}"
    --pd_dual_update_interval 1 --pd_violation_ema_beta 0.1 --pd_lambda_max 100.0
    --pd_violation_tolerance 0.0 --pd_objective_weight 1.0 --pd_structure_weight 0.0
    --pd_sync_across_ranks --pd_require_state_on_resume --pd_strict_constraint_range
    --pd_force_scale_rewards_none --no-pd_soft_objective_for_infeasible
    --no-pd_normalize_primal_advantage --pd_primal_advantage_scale 1.0
    --pd_penalty_scale 1.0 --pd_positive_constraint_penalty
    --no-pd_lexicographic_advantage
    --pd_component_profile co_feasibility_lock_all --no-pd_enable_schedule
)
CLASP_ARGS=(
    --use_clasp
    --clasp_deployment_set_size "${DEPLOYMENT_SET_SIZE}"
    --clasp_canonical_recovery_scale "${CANONICAL_RECOVERY_SCALE}"
    --clasp_canonical_recovery_temperature 0.15 --clasp_canonical_max_subsets 9
    --clasp_strict_defaults --no-clasp_raw_feasibility_targeting
    --clasp_component_dual_recovery_scale 0.0
    --clasp_feasibility_warmup_steps 0 --clasp_feasibility_warmup_targets "${TASK}:0.995"
    --clasp_feasibility_warmup_hold_until_eval --no-clasp_self_replay
    --no-clasp_validation_gate_checkpoints --no-clasp_pareto_anchor
    --no-clasp_proper_dual --clasp_dual_update_interval 1
    --no-clasp_adaptive_length
    --clasp_mixed_adaptive_length_mode error --clasp_adaptive_length_quantile 0.99
    --clasp_adaptive_length_floor 128 --clasp_adaptive_length_window 1000
    --clasp_adaptive_length_refresh_steps 50 --clasp_adaptive_length_margin 32
)

ENTRYPOINT=clasp.entrypoints.train_grpo
if [[ "${METHOD}" == grpo_z || "${METHOD}" == dual_scalar_z ]]; then
    METHOD_ARGS=(
        --use_primal_dual --no-use_clasp --no-clasp_canonical_set
        --pd_group_scalarization --no-pd_separate_z "${PD_ARGS[@]}"
    )
elif [[ "${METHOD}" == dual_separate_z ]]; then
    METHOD_ARGS=(
        --use_primal_dual --no-use_clasp --no-clasp_canonical_set
        --no-pd_group_scalarization --pd_separate_z "${PD_ARGS[@]}"
    )
elif [[ "${METHOD}" == set_null ]]; then
    METHOD_ARGS=(
        --use_primal_dual --no-pd_group_scalarization --no-pd_separate_z
        --no-clasp_canonical_set --clasp_fixed_null_set
        --clasp_fixed_null_receipt_path "${FIXED_NULL_RECEIPT}"
        "${PD_ARGS[@]}" "${CLASP_ARGS[@]}"
    )
else
    METHOD_ARGS=(
        --use_primal_dual --no-pd_group_scalarization --no-pd_separate_z
        --clasp_canonical_set --no-clasp_fixed_null_set
        "${PD_ARGS[@]}" "${CLASP_ARGS[@]}"
    )
fi

echo "=============================================================="
echo " SOTA contract campaign=${CAMPAIGN_ID} task=${TASK} method=${METHOD} train_seed=${TRAIN_SEED} sft_seed=42"
echo " job=${SLURM_JOB_ID:-local} run=${RUN_DIR}"
echo " state=${state} current=${current_step} clean_stop=${stop_step} fixed_final=${FINAL_STEP}"
echo " checkpoint_every=${SAVE_STEPS} eval_every=${EVAL_STEPS} retained=${SAVE_TOTAL_LIMIT}"
echo " advancement_policy=${ADVANCEMENT_POLICY_SHA256} stage=${ADVANCEMENT_STAGE_ID}"
echo " rollout_B=9 deployment_K=8 global_generation_batch=36 effective_batch=36 prompt=${MAX_PROMPT_LENGTH} completion=${MAX_COMPLETION_LENGTH}"
if [[ "${METHOD}" == clasp_canonical_foarl_matched_b9 ]]; then
    echo " comparison=foarl_f638_matched_b9 protocol=shared_compute_data_token_schedule_only reproduction_claim=not_an_official_jiang_reproduction"
fi
echo " rollout=temp0.7/top_p0.9/top_k50/repetition1 optimizer=adamw8bit/lr1e-6"
echo " stack=python${PYTHON_VERSION}/torch${TORCH_VERSION}/transformers${TRANSFORMERS_VERSION}/trl${TRL_VERSION}/peft${PEFT_VERSION}/cuda${CUDA_VERSION}"
echo " surrogate=clipped_token_level_practical reward_gate=strict_feasible_only pairwise=0 elite=0 self_replay=0 adaptive_length=0"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=============================================================="

srun --ntasks=1 --ntasks-per-node=1 \
    "${SEALED_PYTHON}" scripts/run_grpo_in_sealed_environment.py \
    --receipt "${REMOTE_ENVIRONMENT_RECEIPT}" -- \
    -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m "${ENTRYPOINT}" "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}"

final_state="$("${SEALED_PYTHON}" scripts/prepare_sota_run.py "${CONTRACT_ARGS[@]}")"
case "${final_state}" in
    COMPLETE:*) echo "[SOTALaunch] final horizon complete: ${final_state#COMPLETE:}" ;;
    RESUME:*) echo "[SOTALaunch] clean continuation boundary saved: ${final_state#RESUME:}" ;;
    *) echo "[SOTALaunchError] training returned without resumable state: ${final_state}" >&2; exit 2 ;;
esac
