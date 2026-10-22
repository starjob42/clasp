#!/bin/bash -l
# Two-hour, create-once development gate for the exact Jiang-f638 H100 stack.
# This is isolated from every production campaign output. It performs one real
# optimizer update, persists the complete trainable policy, and independently
# reloads the result before declaring the environment safe.

#SBATCH --account=vas@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --job-name=jiang-f638-sft-dev2h_iclr

set -euo pipefail
set -o noclobber
umask 027

project_root="$(readlink -f "${CLASP_PROJECT_ROOT:?CLASP_PROJECT_ROOT is required}")"
environment_prefix="$(readlink -f "${BASE_SFT_CONDA_ENV:?BASE_SFT_CONDA_ENV is required}")"
base_model="$(readlink -f "${BASE_SFT_BASE_MODEL_PATH:?BASE_SFT_BASE_MODEL_PATH is required}")"
output_root="$(readlink -m "${BASE_SFT_SMOKE_OUTPUT_ROOT:?BASE_SFT_SMOKE_OUTPUT_ROOT is required}")"
cache_root="$(readlink -m "${BASE_SFT_SMOKE_CACHE_ROOT:?BASE_SFT_SMOKE_CACHE_ROOT is required}")"
requirements_lock="${project_root}/requirements.jiang-f638-sft.txt"
runtime_closure_lock="${project_root}/requirements.jiang-f638-runtime-closure.txt"
job_id="${SLURM_JOB_ID:?the development gate must run as a Slurm job}"

[[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "[JiangF638DevGateError] invalid Slurm job id: ${job_id}" >&2
    exit 2
}
for safe_root in "${output_root}" "${cache_root}"; do
    [[ "${safe_root}" == /* && "${safe_root}" == *_iclr* ]] || {
        echo "[JiangF638DevGateError] explicit output/cache roots must be absolute and contain _iclr: ${safe_root}" >&2
        exit 2
    }
done
[[ "${output_root}" != "${cache_root}" ]] || {
    echo "[JiangF638DevGateError] output and cache roots must be distinct" >&2
    exit 2
}

path_is_within() {
    local candidate="$1"
    local protected="$2"
    case "${candidate}/" in
        "${protected}/"*) return 0 ;;
        *) return 1 ;;
    esac
}

if path_is_within "${output_root}" "${cache_root}" \
    || path_is_within "${cache_root}" "${output_root}"; then
    echo "[JiangF638DevGateError] output and cache roots must not overlap" >&2
    exit 2
fi

# Neither development root may contain, or be contained by, an immutable
# source, model, or environment root.
for safe_root in "${output_root}" "${cache_root}"; do
    for protected_root in "${project_root}" "${environment_prefix}" "${base_model}"; do
        if path_is_within "${safe_root}" "${protected_root}" \
            || path_is_within "${protected_root}" "${safe_root}"; then
            echo "[JiangF638DevGateError] development and protected paths overlap: ${safe_root} ${protected_root}" >&2
            exit 2
        fi
    done
done

mkdir -p "${output_root}" "${cache_root}"
for safe_root in "${output_root}" "${cache_root}"; do
    [[ ! -L "${safe_root}" && -d "${safe_root}" \
        && "$(readlink -f "${safe_root}")" == "${safe_root}" ]] || {
        echo "[JiangF638DevGateError] unsafe output/cache root: ${safe_root}" >&2
        exit 2
    }
done

# Exclusive creation makes accidental reuse fail closed.
run_dir="${output_root}/jiang_f638_sft_dev2h_iclr_job${job_id}"
job_cache="${cache_root}/jiang_f638_sft_dev2h_iclr_job${job_id}"
mkdir "${run_dir}"
mkdir "${job_cache}"
adapter_dir="${run_dir}/adapter_iclr"
wandb_dir="${run_dir}/wandb_logs_iclr"
mkdir "${adapter_dir}" "${wandb_dir}"

module purge
module load "${GPU_ARCH_MODULE:-arch/h100}"
module load "${CONDA_MODULE:-miniforge/24.9.0}"
conda activate "${environment_prefix}"

python_bin="$(readlink -f "${CONDA_PREFIX:?conda activation did not set CONDA_PREFIX}/bin/python")"
[[ "${CONDA_PREFIX}" == "${environment_prefix}" && -x "${python_bin}" \
    && -f "${requirements_lock}" && -f "${runtime_closure_lock}" \
    && -d "${base_model}" ]] || {
    echo "[JiangF638DevGateError] candidate environment, base, or dependency lock is unavailable" >&2
    exit 2
}

# Eliminate inherited experiment switches before constructing the gate runtime.
while IFS='=' read -r inherited_name _; do
    case "${inherited_name}" in
        CLASP_*|PD_*|GENERATION_*|GRPO_*|PYTORCH_*|HF_*|TRANSFORMERS_*|TOKENIZERS_*|BITSANDBYTES_*|UNSLOTH_*|ACCELERATE_*|WANDB_*|NCCL_*)
            unset "${inherited_name}"
            ;;
    esac
done < <(env)
unset RESUME_FROM_CHECKPOINT WANDB_RUN_ID WANDB_SWEEP_ID PYTHONOPTIMIZE || true

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${job_cache}/huggingface_iclr"
export HF_DATASETS_CACHE="${HF_HOME}/datasets_iclr"
# `datasets`/`multiprocess` creates an AF_UNIX SyncManager socket beneath
# TMPDIR.  Lustre campaign paths are long enough to exceed sockaddr_un's path
# limit before tokenization starts, so keep only ephemeral IPC state in a
# short node-local directory.  Durable caches and every user-facing artifact
# remain in the explicit `_iclr` roots above.
node_tmpdir="$(mktemp -d "/tmp/c_iclr_${job_id}.XXXXXX")"
trap 'rm -rf -- "${node_tmpdir}"' EXIT
export TMPDIR="${node_tmpdir}"
export TORCH_EXTENSIONS_DIR="${job_cache}/torch_extensions_iclr"
export WANDB_MODE=offline
export WANDB_DIR="${wandb_dir}"
export WANDB_PROJECT="clasp_jiang_f638_sft_dev2h_iclr"
export WANDB_RUN_GROUP="jiang_f638_sft_dev2h_iclr_job${job_id}"
export WANDB_NAME="jiang_f638_sft_dev2h_iclr_job${job_id}"
export WANDB_RESUME=never
export WANDB_DISABLE_CODE=true
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}"

cd "${project_root}"
export PYTHONPATH="${project_root}/src:${project_root}/scripts"

echo "[JiangF638DevGate] output_dir=${run_dir}"
echo "[JiangF638DevGate] wandb_dir=${WANDB_DIR}"
echo "[JiangF638DevGate] wandb_project=${WANDB_PROJECT}"
echo "[JiangF638DevGate] wandb_group=${WANDB_RUN_GROUP}"
echo "[JiangF638DevGate] wandb_name=${WANDB_NAME}"

"${python_bin}" - \
    "${environment_prefix}" \
    "${base_model}" \
    "${requirements_lock}" \
    "${runtime_closure_lock}" \
    "${adapter_dir}" \
    "${wandb_dir}" <<'PY'
import gc
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

import torch
import unsloth  # Must precede Transformers/TRL, exactly as in Jiang f638.
import wandb
from datasets import Dataset
from trl import SFTTrainer
from unsloth import FastLanguageModel
from transformers import AutoConfig, AutoTokenizer, TrainingArguments

from clasp.data.prompts import build_jiang_f638_sft_text
from clasp.entrypoints.train_sft_jiang_f638 import (
    audit_effective_sft_config,
    build_strict_response_collator,
)
from clasp.models.checkpoint_fidelity import (
    assert_qwen25_7b_publication_invariant,
    audit_and_write_checkpoint_fidelity,
    audit_trainable_state_closure,
    enable_qwen25_targeted_bias_training,
)

(
    expected_prefix,
    base_model_raw,
    lock_raw,
    runtime_lock_raw,
    adapter_raw,
    wandb_raw,
) = sys.argv[1:]
base_model = Path(base_model_raw)
requirements_lock = Path(lock_raw)
runtime_closure_lock = Path(runtime_lock_raw)
adapter_dir = Path(adapter_raw)
wandb_dir = Path(wandb_raw)

assert sys.prefix == expected_prefix, (sys.prefix, expected_prefix)
assert sys.flags.optimize == 0, "development-gate assertions must not be disabled"
assert int(os.environ.get("WORLD_SIZE", "1") or "1") == 1
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
properties = torch.cuda.get_device_properties(0)
assert "H100" in properties.name.upper(), properties.name
assert (properties.major, properties.minor) == (9, 0)

# The released freeze, the eight runtime-closure packages proven by the
# historical reproduction receipt, and one pip bootstrap pin are the complete
# allowed visible distribution inventory. Extras and duplicates are fatal.
expected_versions = {}
with requirements_lock.open(encoding="utf-8") as handle:
    for line_number, raw_line in enumerate(handle, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert line.count("==") == 1, (line_number, line)
        name, version = line.split("==", 1)
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        assert normalized and version and normalized not in expected_versions
        expected_versions[normalized] = version
assert len(expected_versions) == 177, len(expected_versions)
expected_runtime_closure_versions = {
    "annotated-types": "0.7.0",
    "cffi": "1.17.1",
    "distro": "1.9.0",
    "platformdirs": "4.3.8",
    "pycparser": "2.22",
    "pydantic": "2.11.7",
    "pydantic-core": "2.33.2",
    "typing-inspection": "0.4.1",
}
runtime_closure_versions = {}
with runtime_closure_lock.open(encoding="utf-8") as handle:
    for line_number, raw_line in enumerate(handle, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert line.count("==") == 1, (line_number, line)
        name, version = line.split("==", 1)
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        assert normalized and version and normalized not in runtime_closure_versions
        runtime_closure_versions[normalized] = version
assert runtime_closure_versions == expected_runtime_closure_versions
assert not set(expected_versions) & set(runtime_closure_versions)
expected_versions.update(runtime_closure_versions)
assert "pip" not in expected_versions
expected_versions["pip"] = "25.1.1"
assert len(expected_versions) == 186

observed_versions = {}
for distribution in metadata.distributions():
    raw_name = str(distribution.metadata.get("Name", ""))
    assert raw_name
    normalized = re.sub(r"[-_.]+", "-", raw_name).lower()
    assert normalized not in observed_versions, f"duplicate distribution: {normalized}"
    observed_versions[normalized] = distribution.version
assert observed_versions == expected_versions, {
    "missing": sorted(set(expected_versions) - set(observed_versions)),
    "unexpected": sorted(set(observed_versions) - set(expected_versions)),
    "version_drift": {
        name: {"expected": expected_versions[name], "observed": observed_versions.get(name)}
        for name in sorted(expected_versions)
        if observed_versions.get(name) != expected_versions[name]
    },
}

config = AutoConfig.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(
    base_model, trust_remote_code=True, local_files_only=True
)
assert config.model_type == "qwen2"
assert tokenizer.is_fast and tokenizer.eos_token_id is not None

model, loaded_tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(base_model),
    max_seq_length=20_000,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    local_files_only=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=64,
    lora_dropout=0.0,
    bias="all",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)
bias_trainability_restore = enable_qwen25_targeted_bias_training(model)
pre_update_closure = audit_trainable_state_closure(model)
publication_invariant = assert_qwen25_7b_publication_invariant(model, pre_update_closure)
assert publication_invariant["lora_tensor_count"] == 392
assert publication_invariant["bias_tensor_count"] == 84
assert publication_invariant["trainable_tensor_count"] == 476
assert loaded_tokenizer.eos_token_id == tokenizer.eos_token_id
assert isinstance(loaded_tokenizer.pad_token, str)
assert loaded_tokenizer.pad_token_id is not None

# Thirty-two rows preserve batch=4, accumulation=4, and one epoch while
# yielding two optimizer steps. The first released warmup step has zero LR;
# the second is therefore the first value-changing optimizer update.
smoke_text = build_jiang_f638_sft_text(
    "Solve the tiny development-gate instance.",
    "Node 0; Node 1",
    "Route: [0, 1, 0], Objective: 2",
    eos_token=loaded_tokenizer.eos_token,
)
smoke_dataset = Dataset.from_dict({"text": [smoke_text] * 32})
smoke_training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=20,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    output_dir=str(adapter_dir / "trainer_checkpoints_iclr"),
    report_to="wandb",
    evaluation_strategy="no",
    metric_for_best_model="eval_loss",
    save_total_limit=50,
    save_steps=5_000,
)
smoke_trainer = SFTTrainer(
    model=model,
    tokenizer=loaded_tokenizer,
    train_dataset=smoke_dataset,
    dataset_text_field="text",
    max_seq_length=20_000,
    dataset_num_proc=16,
    packing=False,
    data_collator=build_strict_response_collator(loaded_tokenizer),
    args=smoke_training_args,
)
effective_sft_config = audit_effective_sft_config(smoke_trainer.args)
tokenized_row = smoke_trainer.train_dataset[0]
# SFTTrainer retains the source `text` column in its in-memory dataset and
# removes it through Trainer's normal column-pruning path when constructing the
# dataloader.  This direct preflight bypasses that path, so give the collator
# only model inputs rather than asking tokenizer.pad to tensorize a string.
collator_feature = {
    key: tokenized_row[key]
    for key in ("input_ids", "attention_mask")
    if key in tokenized_row
}
assert "input_ids" in collator_feature
smoke_batch = smoke_trainer.data_collator([collator_feature])
smoke_labels = smoke_batch["labels"][0].tolist()
assert any(label == -100 for label in smoke_labels)
assert any(label != -100 for label in smoke_labels)
first_supervised = next(i for i, label in enumerate(smoke_labels) if label != -100)
assert all(label == -100 for label in smoke_labels[:first_supervised])
assert all(label != -100 for label in smoke_labels[first_supervised:])

post_constructor_closure = audit_trainable_state_closure(smoke_trainer.model)
post_constructor_invariant = assert_qwen25_7b_publication_invariant(
    smoke_trainer.model, post_constructor_closure
)
assert post_constructor_closure == pre_update_closure
assert post_constructor_invariant == publication_invariant

smoke_trainer.create_optimizer()
optimizer_parameter_ids = {
    id(parameter)
    for group in smoke_trainer.optimizer.param_groups
    for parameter in group["params"]
}
trainable_parameter_ids = {
    id(parameter)
    for parameter in smoke_trainer.model.parameters()
    if parameter.requires_grad
}
bias_parameter_ids = {
    id(parameter)
    for name, parameter in smoke_trainer.model.named_parameters()
    if name.endswith(".base_layer.bias") and parameter.requires_grad
}
assert optimizer_parameter_ids == trainable_parameter_ids
assert len(bias_parameter_ids) == 84
assert bias_parameter_ids <= optimizer_parameter_ids

for key in ("WANDB_PROJECT", "WANDB_RUN_GROUP", "WANDB_NAME"):
    assert "_iclr" in os.environ[key]
assert "_iclr" in str(wandb_dir)
wandb_run = wandb.init(
    project=os.environ["WANDB_PROJECT"],
    group=os.environ["WANDB_RUN_GROUP"],
    name=os.environ["WANDB_NAME"],
    dir=str(wandb_dir),
    mode="offline",
    reinit=True,
    config={
        "gate": "jiang_f638_sft_dev2h_iclr",
        "optimizer_steps": 2,
        "bias_policy": "all_iclr",
    },
)
assert wandb_run is not None
assert wandb_run.project == os.environ["WANDB_PROJECT"]
assert wandb_run.group == os.environ["WANDB_RUN_GROUP"]
assert wandb_run.name == os.environ["WANDB_NAME"]
assert "_iclr" in str(Path(wandb_run.dir).resolve())

train_result = smoke_trainer.train()
assert int(smoke_trainer.state.global_step) == 2
assert int(smoke_trainer.state.max_steps) == 2
assert math.isfinite(float(train_result.training_loss))

live_model = smoke_trainer.accelerator.unwrap_model(smoke_trainer.model)
post_update_closure = audit_trainable_state_closure(live_model)
post_update_invariant = assert_qwen25_7b_publication_invariant(
    live_model, post_update_closure
)
assert post_update_invariant == publication_invariant
pre_records = {record["key"]: record for record in pre_update_closure["tensors"]}
post_records = {record["key"]: record for record in post_update_closure["tensors"]}
assert pre_records.keys() == post_records.keys()
for key in pre_records:
    for field in ("key", "shape", "dtype", "numel"):
        assert pre_records[key][field] == post_records[key][field]
changed_keys = sorted(
    key for key in pre_records
    if pre_records[key]["value_sha256"] != post_records[key]["value_sha256"]
)
changed_bias_keys = [key for key in changed_keys if key.endswith(".base_layer.bias")]
changed_lora_keys = [key for key in changed_keys if ".lora_" in key]
assert changed_keys, "the claimed optimizer update changed no trainable tensor"
assert changed_bias_keys, "the update did not exercise trained Q/K/V biases"
assert changed_lora_keys, "the update did not exercise LoRA weights"

smoke_trainer.save_model(str(adapter_dir))
loaded_tokenizer.save_pretrained(str(adapter_dir))
checkpoint_fidelity = audit_and_write_checkpoint_fidelity(
    live_model, adapter_dir, require_qwen25_7b_publication=True
)
assert checkpoint_fidelity["status"] == "exact_trainable_policy_persisted"

wandb_run.log({
    "dev_gate_iclr/optimizer_steps": 2,
    "dev_gate_iclr/value_changing_optimizer_updates": 1,
    "dev_gate_iclr/changed_trainable_tensors": len(changed_keys),
    "dev_gate_iclr/changed_bias_tensors": len(changed_bias_keys),
    "dev_gate_iclr/changed_lora_tensors": len(changed_lora_keys),
    "dev_gate_iclr/save_fidelity": 1,
})
wandb_runtime_dir = str(Path(wandb_run.dir).resolve())
wandb.finish(exit_code=0)

evidence = {
    "schema_version": 1,
    "protocol": "clasp-jiang-f638-sft-dev2h-iclr-update-v1",
    "status": "one_exact_optimizer_update_and_save_verified",
    "python": sys.executable,
    "prefix": sys.prefix,
    "gpu": properties.name,
    "compute_capability": [properties.major, properties.minor],
    "cuda_runtime": torch.version.cuda,
    "model_type": config.model_type,
    "tokenizer_class": type(tokenizer).__name__,
    "eos_token_id": tokenizer.eos_token_id,
    "visible_distribution_count": len(observed_versions),
    "runtime_closure_lock_sha256": hashlib.sha256(
        runtime_closure_lock.read_bytes()
    ).hexdigest(),
    "visible_versions_sha256": hashlib.sha256(
        json.dumps(observed_versions, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest(),
    "bias_trainability_restore": bias_trainability_restore,
    "publication_invariant": publication_invariant,
    "effective_sft_config": effective_sft_config,
    "optimizer_trainable_tensor_count": len(optimizer_parameter_ids),
    "optimizer_bias_tensor_count": len(bias_parameter_ids),
    "optimizer_steps": 2,
    "value_changing_optimizer_updates": 1,
    "training_loss": float(train_result.training_loss),
    "changed_trainable_tensor_count": len(changed_keys),
    "changed_trainable_keys_sha256": hashlib.sha256(
        json.dumps(changed_keys, separators=(",", ":")).encode()
    ).hexdigest(),
    "changed_bias_tensor_count": len(changed_bias_keys),
    "changed_lora_tensor_count": len(changed_lora_keys),
    "collator_first_supervised_token_index": first_supervised,
    "checkpoint_fidelity_receipt_sha256": checkpoint_fidelity["receipt_sha256"],
    "wandb": {
        "mode": "offline",
        "directory": str(wandb_dir.resolve()),
        "runtime_directory": wandb_runtime_dir,
        "project": os.environ["WANDB_PROJECT"],
        "group": os.environ["WANDB_RUN_GROUP"],
        "name": os.environ["WANDB_NAME"],
    },
}
evidence["receipt_sha256"] = hashlib.sha256(
    json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
evidence_path = adapter_dir / "optimizer_update_evidence_iclr.json"
with evidence_path.open("x", encoding="utf-8") as handle:
    json.dump(evidence, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(evidence_path, stat.S_IRUSR | stat.S_IRGRP)

del smoke_trainer, live_model, model, loaded_tokenizer
gc.collect()
torch.cuda.empty_cache()
print(json.dumps(evidence, sort_keys=True))
PY

# Independent processes audit on-disk bias persistence, then perform two fresh
# local base+adapter loads and exact state/fixed-probe equality checks.
"${python_bin}" scripts/audit_lora_persistence.py \
    --base-model-path "${base_model}" \
    --adapter-path "${adapter_dir}" \
    --output "${adapter_dir}/adapter_persistence_receipt.json" \
    > "${run_dir}/persistence_audit_stdout_iclr.jsonl"

"${python_bin}" scripts/audit_sft_reload_equivalence.py \
    --base-model-path "${base_model}" \
    --adapter-path "${adapter_dir}" \
    --runtime-dtype bfloat16 \
    --device cuda:0 \
    --output "${adapter_dir}/sft_reload_equivalence.json" \
    > "${run_dir}/reload_audit_stdout_iclr.jsonl"

"${python_bin}" - \
    "${base_model}" \
    "${adapter_dir}" \
    "${wandb_dir}" \
    "${run_dir}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from audit_lora_persistence import validate_audit_receipt
from audit_sft_reload_equivalence import validate_reload_receipt
from clasp.models.checkpoint_fidelity import verify_checkpoint_fidelity

base_model, adapter_dir, wandb_dir, run_dir = map(Path, sys.argv[1:])
for path in (adapter_dir, wandb_dir, run_dir):
    assert "_iclr" in str(path.resolve())

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

fidelity = verify_checkpoint_fidelity(adapter_dir)
persistence_path = adapter_dir / "adapter_persistence_receipt.json"
reload_path = adapter_dir / "sft_reload_equivalence.json"
evidence_path = adapter_dir / "optimizer_update_evidence_iclr.json"
validate_audit_receipt(
    persistence_path, base_model_path=base_model, adapter_path=adapter_dir
)
reload_receipt = validate_reload_receipt(
    reload_path, base_model_path=base_model, adapter_path=adapter_dir
)

evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
unsigned_evidence = dict(evidence)
evidence_digest = unsigned_evidence.pop("receipt_sha256")
assert evidence_digest == hashlib.sha256(
    json.dumps(unsigned_evidence, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
assert evidence["optimizer_steps"] == 2
assert evidence["value_changing_optimizer_updates"] >= 1
assert evidence["changed_trainable_tensor_count"] > 0
assert evidence["changed_bias_tensor_count"] > 0
assert evidence["changed_lora_tensor_count"] > 0
assert evidence["checkpoint_fidelity_receipt_sha256"] == fidelity["receipt_sha256"]
assert all(
    "_iclr" in evidence["wandb"][field]
    for field in ("directory", "runtime_directory", "project", "group", "name")
)

wandb_run_files = sorted(
    path for path in wandb_dir.rglob("*.wandb")
    if path.is_file() and not path.is_symlink() and path.stat().st_size > 0
)
assert wandb_run_files, f"offline W&B run data is missing below {wandb_dir}"

required_adapter_files = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "checkpoint_fidelity.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "adapter_persistence_receipt.json",
    "sft_reload_equivalence.json",
    "optimizer_update_evidence_iclr.json",
)
for name in required_adapter_files:
    path = adapter_dir / name
    assert path.is_file() and not path.is_symlink() and path.stat().st_size > 0, path

payload = {
    "schema_version": 1,
    "protocol": "clasp-jiang-f638-sft-dev2h-iclr-gate-v1",
    "status": "passed",
    "output_dir": str(run_dir.resolve()),
    "adapter_dir": str(adapter_dir.resolve()),
    "wandb": evidence["wandb"],
    "wandb_run_file_count": len(wandb_run_files),
    "wandb_run_files_sha256": hashlib.sha256(
        json.dumps([
            {
                "relative_path": str(path.relative_to(wandb_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in wandb_run_files
        ], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest(),
    "optimizer_update_evidence_sha256": sha256_file(evidence_path),
    "checkpoint_fidelity_receipt_sha256": fidelity["receipt_sha256"],
    "adapter_persistence_receipt_file_sha256": sha256_file(persistence_path),
    "reload_equivalence_receipt_sha256": reload_receipt["receipt_sha256"],
    "reload_equivalence_receipt_file_sha256": sha256_file(reload_path),
}
payload["receipt_sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
receipt_path = run_dir / "dev_gate_success_iclr.json"
with receipt_path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(receipt_path, stat.S_IRUSR | stat.S_IRGRP)
assert stat.S_IMODE(receipt_path.stat(follow_symlinks=False).st_mode) & 0o222 == 0
print(json.dumps(payload, sort_keys=True))
PY

echo "[JiangF638DevGate] PASSED receipt=${run_dir}/dev_gate_success_iclr.json"
exit 0
