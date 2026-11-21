#!/usr/bin/env python3
"""Create and enforce an immutable CLASP SOTA training contract.

This helper deliberately has no third-party dependencies so it can run before a
distributed job starts.  On success it prints exactly one machine-readable line:

* ``FRESH`` when the contracted run has no training state yet;
* ``RESUME:<checkpoint>`` for the newest complete optimizer checkpoint; or
* ``COMPLETE:<checkpoint>`` only when the fixed final horizon is finalized.

Existing contracts are never rewritten.  Any contract drift or ambiguous state
is an error rather than an invitation to silently restart a scientific run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from audit_lora_persistence import (  # type: ignore[import-not-found]
    EXPECTED_BIAS_PARAMETERS,
    EXPECTED_BIAS_TENSORS,
    EXPECTED_LORA_PARAMETERS,
    EXPECTED_LORA_TENSORS,
    LORA_BIAS_POLICY,
    audit_adapter_persistence,
    audit_qwen25_base_bias_scope,
    validate_audit_receipt,
)
from audit_sft_reload_equivalence import (  # type: ignore[import-not-found]
    RECEIPT_FILENAME as SFT_RELOAD_EQUIVALENCE_FILENAME,
)
from audit_sft_reload_equivalence import (
    reload_equivalence_identity,
    validate_reload_equivalence_identity,
    validate_reload_receipt,
)
from freeze_remote_environment import environment_identity  # type: ignore[import-not-found]
from freeze_forbidden_source_registry import (  # type: ignore[import-not-found]
    BASE_SFT_PROTOCOL,
    validate_base_sft_source_registry,
)

from clasp.baselines.foarl_f638_campaign import (
    FOARL_F638_CAMPAIGN_METHODS,
    FOARL_F638_NATIVE,
    get_foarl_f638_campaign_profile,
    validate_foarl_f638_campaign_contract,
)
from clasp.baselines.foarl_f638_comparison import (
    CLASP_FOARL_MATCHED_B9,
    CLASP_FOARL_MATCHED_B9_ARM_SCHEMA,
    FOARL_MATCHED_B9_COMPARISON_PROTOCOL,
    validate_clasp_foarl_matched_b9_method_contract,
)
from clasp.data.benchmark import validate_forbidden_source_registry
from clasp.models import exact_peft_kl_post_step
from clasp.models.checkpoint_fidelity import (
    PROTOCOL as CHECKPOINT_FIDELITY_PROTOCOL,
)
from clasp.models.checkpoint_fidelity import (
    QWEN_PUBLICATION_PROTOCOL,
)
from clasp.models.checkpoint_fidelity import (
    RECEIPT_FILENAME as CHECKPOINT_FIDELITY_FILENAME,
)
from clasp.models.exact_peft_kl_reference import (
    RECEIPT_FILENAME as EXACT_PEFT_KL_REFERENCE_FILENAME,
)
from clasp.models.jiang_f638_base import (
    audit_jiang_f638_base_snapshot,
    validate_jiang_f638_base_receipt,
)
from clasp.models.publication_fidelity import (
    IDENTITY_PROTOCOL as CHECKPOINT_FIDELITY_IDENTITY_PROTOCOL,
)
from clasp.models.publication_fidelity import (
    exact_peft_kl_post_step_lineage_identity,
    exact_peft_kl_runtime_identity,
    fixed_kl_reference_identity,
    publication_fidelity_identity,
    validate_fixed_kl_reference_identity,
    validate_publication_fidelity_identity,
)
from clasp.set_grpo.fixed_null_receipt import load_fixed_null_score_receipt

SCHEMA_VERSION = 2
CONTRACT_FILENAME = "run_contract.json"
SOURCE_MANIFEST_FILENAME = "SOURCE_MANIFEST.sha256"
CHECKPOINT_RE = re.compile(r"^checkpoint-([1-9][0-9]*)$")
LEGACY_METHODS = {"grpo", "dual_grpo", "clasp_dense", "clasp_exact"}
CANONICAL_PUBLICATION_METHODS = {
    "grpo_z",
    "dual_scalar_z",
    "dual_separate_z",
    "clasp_canonical",
    "clasp_canonical_fixed_lambda",
    "clasp_canonical_no_recovery",
    CLASP_FOARL_MATCHED_B9,
    "set_null",
}
FOARL_PUBLICATION_METHODS = set(FOARL_F638_CAMPAIGN_METHODS)
PUBLICATION_METHODS = CANONICAL_PUBLICATION_METHODS | FOARL_PUBLICATION_METHODS
METHODS = LEGACY_METHODS | PUBLICATION_METHODS
TASKS = {"tsp", "cvrp", "op", "mis", "jssp"}
MODEL_ARTIFACTS = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
REQUIRED_CHECKPOINT_FILES = (
    "adapter_config.json",
    CHECKPOINT_FIDELITY_FILENAME,
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
)
PD_TASK_KEYS = {
    "cost_names",
    "cost_limits",
    "cost_weights",
    "dual_lambdas",
    "controller_state",
}
BON_STATE_KEYS = {"lambdas", "failure_ema", "observed_groups"}
SFT_ROWS = 500_000
SFT_FINAL_STEP = 31_250
MODEL_INITIALIZATION_REGISTRY_SCHEMA_VERSION = 2
MODEL_INITIALIZATION_REGISTRY_PROTOCOL = "clasp-model-initialization-registry-v2"
MODEL_INITIALIZATION_REGISTRY_STATUS = "frozen_before_training"
MODEL_FAMILY = "Qwen2.5-7B"
JIANG_LORA_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
TRACKED_ENV_PREFIXES = ("CLASP_", "PD_", "GENERATION_", "GRPO_")
# The stop boundary intentionally advances between continuation jobs.  The
# run-contract path is an operational pointer whose *bytes* are bound by the
# fidelity receipts.  Neither belongs in the immutable optimizer/scheduler
# trajectory (and embedding the absolute pointer would make it non-portable).
MUTABLE_ENV_NAMES = {
    "CLASP_TRAINING_RUN_CONTRACT_PATH",
    "GRPO_STOP_AT_GLOBAL_STEP",
}

# A continuation chunk is represented only by its retained checkpoint plus the
# create-once root reference lineage.  These files certify terminal root state
# and must never be left behind by an early GRPO_STOP_AT_GLOBAL_STEP boundary.
TERMINAL_ROOT_ARTIFACTS = frozenset(
    {
        *MODEL_ARTIFACTS,
        "README.md",
        "adapter_config.json",
        "added_tokens.json",
        "chat_template.jinja",
        "clasp_bon_controller_state.json",
        "clasp_component_dual_state.json",
        CHECKPOINT_FIDELITY_FILENAME,
        SFT_RELOAD_EQUIVALENCE_FILENAME,
        "component_dual_state.json",
        "config.json",
        "exact_peft_kl_reference_post_step.json",
        "generation_config.json",
        "grpo_log_history.json",
        "grpo_trainer_state.json",
        "merges.txt",
        "multitask_primal_dual_state.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "stage_status.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "training_args.bin",
        "vocab.json",
    }
)
TERMINAL_ROOT_WEIGHT_PREFIXES = ("adapter_model", "model", "pytorch_model")

MATCHED_SFT_STAGE_HPARAMETERS = {
    "recipe": "jiang-llmcosolver-f638-corrected-v1",
    "jiang_source_commit": "f6388991d243c0aa91d9ee7e2fa18ed1c9e72477",
    "model_loader": "unsloth.FastLanguageModel",
    "load_in_4bit": False,
    "full_finetune": False,
    "lora_r": 64,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "lora_bias": LORA_BIAS_POLICY,
    "explicit_bias_trainability_restore_after_unsloth": True,
    "lora_target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "use_gradient_checkpointing": "unsloth",
    "use_rslora": False,
    "loftq_config": None,
    "train_lm_head": False,
    "train_embed_tokens": False,
    "seed": 42,
    "dataset_shuffle_seed": 42,
    "dataset_shuffle_passes": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "effective_train_batch_size": 16,
    "learning_rate": 0.0002,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "warmup_steps": 20,
    "lr_scheduler_type": "linear",
    "num_train_epochs": 1.0,
    "max_steps": -1,
    "max_seq_length": 20_000,
    "packing": False,
    "min_train_samples_per_task": SFT_ROWS,
    "deduplicate_train_rows": False,
    "no_reason": True,
    "prompt_style": "jiang_f638",
    "released_response_template": "### Response:",
    "response_boundary_prefix": "    ### Response",
    "response_boundary_policy": "supervise_colon_output_crossing_token",
    "append_eos_token": True,
    "completion_only_loss": True,
    "memory_safe_loss": False,
    "supervised_logits_only": False,
    "evaluation_max_new_tokens": 3_000,
    "evaluation_max_prompt_length": 128_072,
    "evaluation_requires_zero_prompt_truncation": True,
    "evaluation_no_reason": True,
    "clasp_tsp_prompt_knn": "0",
    "clasp_tsp_prompt_knn_distances": "",
    "clasp_tsp_input_order": "",
    "clasp_tsp_relabel_order": "",
    "clasp_tsp_route_canonicalize": "",
    "clasp_tsp_route_only_output": "",
    "clasp_tsp_label_tta_schedule": "",
    "clasp_mis_relabel_order": "",
    "clasp_mis_jiang_response_output": "0",
    "clasp_mis_prompt_node_degrees": "0",
    "clasp_mis_prompt_low_degree_k": "32",
}
MATCHED_SFT_CONTRACT = {
    "stage": "sft",
    "specialist_task_count": 1,
    "train_rows": SFT_ROWS,
    "eval_rows": 0,
    "world_size": 1,
    "final_step": SFT_FINAL_STEP,
    "released_requirements_sha256": "b8c309502184dbb01f01c3d016c70d5d69d04ce03ec770ae7be720424a9a22b6",
    "prompt_template_sha256": "0e4ff1b30b6b81b3ba4d84fd68c791f7a600dd470a2aae4dfe129fca8ffe6883",
    "tokenizer_json_sha256": "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa",
    "base_model": {
        "repository": "unsloth/Qwen2.5-7B",
        "resolved_revision": "86e6e1eccea424c7fb77d59f7d2a9125c0673bfd",
        "paper_reported_revision": None,
    },
    "hardware": {
        "accelerator_count": 1,
        "accelerator_family": "NVIDIA H100",
        "compute_capability": [9, 0],
    },
    "deliberate_deviations": {
        "cleaned_decontaminated_data": True,
        "bias_persistence_policy": LORA_BIAS_POLICY,
        "fresh_run_resume_bug_fixed": True,
        "explicit_final_save": True,
        "fail_closed_full_context_audit": True,
        "save_reload_equality_required": True,
        "released_full_example_fallback_disabled": True,
    },
    "stage_hyperparameters": MATCHED_SFT_STAGE_HPARAMETERS,
    "adapter_persistence": {
        "bias_policy": LORA_BIAS_POLICY,
        "architecture_scope": "all_base_biases_equal_targeted_qkv_biases",
        "expected_bias_tensors": EXPECTED_BIAS_TENSORS,
        "expected_bias_parameters": EXPECTED_BIAS_PARAMETERS,
        "expected_lora_tensors": EXPECTED_LORA_TENSORS,
        "expected_lora_parameters": EXPECTED_LORA_PARAMETERS,
        "requires_exact_safetensors_inventory": True,
    },
    "checkpoint_fidelity": {
        "receipt_filename": CHECKPOINT_FIDELITY_FILENAME,
        "receipt_protocol": CHECKPOINT_FIDELITY_PROTOCOL,
        "identity_protocol": CHECKPOINT_FIDELITY_IDENTITY_PROTOCOL,
        "publication_invariant_protocol": QWEN_PUBLICATION_PROTOCOL,
        "requires_exact_trainable_state_closure": True,
        "requires_read_only_self_hashed_receipt": True,
    },
}
QWEN25_7B_BASE_SIGNATURE = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
    "max_position_embeddings": 131072,
}


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ValueError(f"cannot fingerprint {path}: {exc}") from exc
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"artifact changed while it was fingerprinted: {path}")
    return digest.hexdigest()


def _tree_fingerprint(
    root: Path,
    *,
    include: Callable[[str], bool] | None = None,
    label: str,
) -> dict[str, Any]:
    raw_root = root.expanduser().absolute()
    if raw_root.is_symlink():
        raise ValueError(f"{label} root must not be a symbolic link: {raw_root}")
    root = raw_root.resolve()
    if not root.exists():
        raise ValueError(f"{label} path does not exist: {root}")

    if root.is_file():
        candidates = [(root.name, root)]
    elif root.is_dir():
        candidates = []
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise ValueError(f"{label} tree contains a symbolic link: {candidate}")
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(root).as_posix()
            if include is None or include(relative):
                candidates.append((relative, candidate))
    else:
        raise ValueError(f"{label} path is neither a regular file nor directory: {root}")

    candidates.sort(key=lambda item: item[0])
    if not candidates:
        raise ValueError(f"{label} fingerprint selected no files under {root}")

    aggregate = hashlib.sha256()
    total_size = 0
    file_receipts: list[dict[str, Any]] = []
    for relative, path in candidates:
        try:
            before = path.stat()
        except OSError as exc:
            raise ValueError(f"cannot stat {path}: {exc}") from exc
        file_digest = _sha256_file(path)
        try:
            after = path.stat()
        except OSError as exc:
            raise ValueError(f"cannot restat {path}: {exc}") from exc
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"artifact changed while its tree was fingerprinted: {path}")
        size = int(after.st_size)
        total_size += size
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(file_digest.encode("ascii"))
        aggregate.update(b"\n")
        file_receipts.append({"path": relative, "size_bytes": size, "sha256": file_digest})

    return {
        "canonical_path": str(root),
        "sha256": aggregate.hexdigest(),
        "file_count": len(candidates),
        "size_bytes": total_size,
        "files": file_receipts,
    }


def _registered_source_binding(
    *,
    registry_identity: dict[str, object],
    task: str,
    source_label: str,
    source_path: Path,
) -> dict[str, Any]:
    """Verify an actual data tree against one exact frozen registry entry."""

    sources_by_task = registry_identity.get("sources_by_task_label")
    registered = (
        sources_by_task.get(task, {}).get(source_label)
        if isinstance(sources_by_task, dict) and isinstance(sources_by_task.get(task), dict)
        else None
    )
    if not isinstance(registered, dict):
        raise ValueError(f"forbidden-source registry has no {task}/{source_label} binding")
    raw_root = source_path.expanduser().absolute()
    if raw_root.is_symlink():
        raise ValueError(f"registered {source_label} source root must not be a symbolic link")
    root = raw_root.resolve()
    if root.is_file():
        if root.suffix != ".json":
            raise ValueError(f"registered {source_label} file must end in .json")
        candidates = [(root.name, root)]
    elif root.is_dir():
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise ValueError(
                    f"registered {source_label} tree contains a symbolic link: {candidate}"
                )
        candidates = sorted(
            (
                (candidate.relative_to(root).as_posix(), candidate)
                for candidate in root.rglob("*.json")
                if candidate.is_file()
            ),
            key=lambda item: item[0],
        )
    else:
        raise ValueError(f"registered {source_label} path does not exist: {root}")
    if not candidates:
        raise ValueError(f"registered {source_label} path contains no JSON files")

    expected_files = registered.get("files")
    if not isinstance(expected_files, list):
        raise ValueError(f"registry {task}/{source_label} has no file manifest")
    expected_by_path = {
        str(record.get("relative_path")): record
        for record in expected_files
        if isinstance(record, dict)
    }
    if [relative for relative, _path in candidates] != list(expected_by_path):
        raise ValueError(
            f"actual {task}/{source_label} data tree does not exactly match "
            "registered relative paths"
        )
    verified_files: list[dict[str, Any]] = []
    for relative, path in candidates:
        expected = expected_by_path[relative]
        before = path.stat()
        digest = _sha256_file(path)
        after = path.stat()
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise ValueError(f"registered data file changed during audit: {path}")
        if digest != expected.get("sha256") or after.st_size != expected.get("size_bytes"):
            raise ValueError(f"actual {task}/{source_label} file differs from registry: {relative}")
        verified_files.append(
            {
                "relative_path": relative,
                "sha256": digest,
                "size_bytes": int(after.st_size),
                "rows": expected.get("rows"),
            }
        )
    if root.is_dir():
        observed_paths = sorted(
            candidate.relative_to(root).as_posix()
            for candidate in root.rglob("*.json")
            if candidate.is_file()
        )
        if observed_paths != [relative for relative, _path in candidates]:
            raise ValueError(f"actual {task}/{source_label} tree changed during registry audit")
    if verified_files != expected_files:
        raise ValueError(f"actual {task}/{source_label} file manifest differs from registry")
    return {
        "task": task,
        "label": source_label,
        "canonical_path": str(root),
        "registry_sha256": registry_identity["registry_sha256"],
        "source_tree_manifest_sha256": registry_identity["source_tree_manifest_sha256"],
        "source_identity_sha256": registered["source_identity_sha256"],
        "file_manifest_sha256": registered["file_manifest_sha256"],
        "rows": registered["rows"],
        "unique_instance_hashes": registered["unique_instance_hashes"],
        "files": verified_files,
    }


def _model_file(relative: str) -> bool:
    name = Path(relative).name
    if "/" in relative:
        return False
    if name in {
        *MODEL_ARTIFACTS,
        "adapter_config.json",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "chat_template.jinja",
        "hyperparams.json",
        "stage_status.json",
        "base_sft_run_contract.json",
        "adapter_persistence_receipt.json",
        "completed_sft_run_seal.json",
        SFT_RELOAD_EQUIVALENCE_FILENAME,
        CHECKPOINT_FIDELITY_FILENAME,
    }:
        return True
    return (name.startswith("model-") and name.endswith(".safetensors")) or (
        name.startswith("pytorch_model-") and name.endswith(".bin")
    )


def _source_file(relative: str) -> bool:
    path = Path(relative)
    return relative != SOURCE_MANIFEST_FILENAME and not (path.parts and path.parts[0] == "logs")


def _source_snapshot_paths(root: Path) -> tuple[list[str], list[str]]:
    """Return frozen files/directories while rejecting ambiguous filesystem entries."""

    files: list[str] = []
    directories: list[str] = []
    try:
        candidates = sorted(root.rglob("*"), key=lambda path: path.as_posix())
    except OSError as exc:
        raise ValueError(f"cannot enumerate source snapshot {root}: {exc}") from exc
    for candidate in candidates:
        relative = candidate.relative_to(root).as_posix()
        path = Path(relative)
        if path.parts and path.parts[0] == "logs":
            continue
        if candidate.is_symlink():
            raise ValueError(f"source snapshot contains a symbolic link: {relative}")
        try:
            mode = candidate.stat().st_mode
        except OSError as exc:
            raise ValueError(f"cannot stat source snapshot entry {candidate}: {exc}") from exc
        if stat.S_ISDIR(mode):
            directories.append(relative)
        elif stat.S_ISREG(mode):
            if relative != SOURCE_MANIFEST_FILENAME:
                files.append(relative)
        else:
            raise ValueError(f"source snapshot contains a special entry: {relative}")
    return files, directories


def validate_source_snapshot(source_root: Path, manifest_path: Path) -> None:
    """Require an exact, read-only, hash-verified source snapshot.

    ``logs/`` is the sole writable exception because Slurm opens its output files
    before the job body runs.  It is excluded from both the source manifest and
    the immutable training fingerprint.
    """

    raw_root = source_root.expanduser().absolute()
    if raw_root.is_symlink():
        raise ValueError(f"source snapshot root must not be a symbolic link: {raw_root}")
    root = raw_root.resolve()
    manifest = manifest_path.expanduser().resolve()
    expected_manifest = root / SOURCE_MANIFEST_FILENAME
    if manifest != expected_manifest:
        raise ValueError(f"source manifest must be {expected_manifest}, got {manifest}")
    if not root.is_dir():
        raise ValueError(f"source snapshot root is not a regular directory: {root}")
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError(f"source snapshot has no regular manifest: {manifest}")
    root_mode = stat.S_IMODE(root.stat().st_mode)
    manifest_mode = stat.S_IMODE(manifest.stat().st_mode)
    if root_mode & 0o222 or manifest_mode & 0o222:
        raise ValueError("source snapshot root and manifest must be non-writable")

    files, directories = _source_snapshot_paths(root)
    for relative in directories:
        mode = stat.S_IMODE((root / relative).stat().st_mode)
        if mode & 0o222:
            raise ValueError(f"source snapshot directory is writable (mode {mode:04o}): {relative}")
    for relative in files:
        mode = stat.S_IMODE((root / relative).stat().st_mode)
        if mode & 0o222:
            raise ValueError(f"source snapshot file is writable (mode {mode:04o}): {relative}")

    try:
        manifest_bytes = manifest.read_bytes()
        if not manifest_bytes or not manifest_bytes.endswith(b"\n"):
            raise ValueError("source manifest must be nonempty and newline-terminated")
        lines = manifest_bytes.decode("utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read source snapshot manifest {manifest}: {exc}") from exc
    entries: list[tuple[str, str]] = []
    for line_number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\\\r\n]+)", line)
        if match is None:
            raise ValueError(
                f"invalid source manifest line {line_number}: expected SHA256<two spaces>path"
            )
        digest, relative = match.groups()
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or relative in {"", ".", SOURCE_MANIFEST_FILENAME}
            or ".." in relative_path.parts
            or (relative_path.parts and relative_path.parts[0] == "logs")
        ):
            raise ValueError(f"unsafe source manifest path: {relative!r}")
        entries.append((relative, digest))
    manifest_files = [relative for relative, _ in entries]
    if manifest_files != sorted(manifest_files) or len(manifest_files) != len(set(manifest_files)):
        raise ValueError("source manifest paths must be sorted and unique")
    if manifest_files != files:
        missing = sorted(set(files) - set(manifest_files))
        extra = sorted(set(manifest_files) - set(files))
        raise ValueError(
            "source manifest does not exactly cover the frozen tree: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for relative, expected_digest in entries:
        observed_digest = _sha256_file(root / relative)
        if observed_digest != expected_digest:
            raise ValueError(
                f"source manifest hash mismatch for {relative}: "
                f"expected={expected_digest}, observed={observed_digest}"
            )

    logs = root / "logs"
    if logs.is_symlink() or not logs.is_dir():
        raise ValueError("source snapshot requires a regular writable logs/ directory")
    if not os.access(logs, os.W_OK | os.X_OK):
        raise ValueError("source snapshot logs/ directory is not writable/searchable")


def fingerprint_model(path: Path, *, label: str) -> dict[str, Any]:
    receipt = _tree_fingerprint(path, include=_model_file, label=label)
    selected = {item["path"] for item in receipt["files"]}
    if not any(name in selected for name in MODEL_ARTIFACTS):
        raise ValueError(f"{label} has no model weight artifact at its root: {path}")
    return receipt


def validate_qwen25_7b_base(path: Path) -> dict[str, Any]:
    """Require the non-instruction Qwen2.5-7B architecture used by the claim."""

    root = path.expanduser().resolve()
    config = _load_object(root / "config.json")
    mismatches = {
        field: {"actual": config.get(field), "expected": expected}
        for field, expected in QWEN25_7B_BASE_SIGNATURE.items()
        if config.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"base model is not the frozen Qwen2.5-7B family: {mismatches}")
    tokenizer_config = _load_object(root / "tokenizer_config.json")
    if (
        str(tokenizer_config.get("chat_template", "") or "").strip()
        or (root / "chat_template.jinja").exists()
    ):
        raise ValueError(
            "base model carries an instruction/chat template; the publication contract "
            "requires non-instruction Qwen2.5-7B"
        )
    return dict(QWEN25_7B_BASE_SIGNATURE)


def validate_matched_sft(
    path: Path,
    *,
    task: str,
    sft_seed: int,
    sft_data_path: Path,
) -> dict[str, Any]:
    """Require the semantic receipt for the fixed 500k specialist initialization."""
    path = path.expanduser().resolve()
    payload = _load_object(path / "hyperparams.json")
    if sft_seed != 42:
        raise ValueError("matched 500k specialist SFT seed must be exactly 42")
    if payload.get("stage") != "sft" or payload.get("tasks") != [task]:
        raise ValueError(f"initial model is not a {task} specialist SFT run: {path}")
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("initial SFT hyperparams has no dataset object")
    if (
        dataset.get("train_size") != SFT_ROWS
        or dataset.get("eval_size") != 0
        or dataset.get("train_task_counts") != {task: SFT_ROWS}
    ):
        raise ValueError("initial SFT is not the matched 500k specialist dataset")
    args = payload.get("args")
    if not isinstance(args, dict) or not str(args.get("data_root", "")).strip():
        raise ValueError("initial SFT hyperparams has no recorded data_root lineage")
    recorded_sft_data = Path(str(args["data_root"])).expanduser().resolve() / task / "train"
    if recorded_sft_data != sft_data_path.expanduser().resolve():
        raise ValueError(
            "initial SFT recorded data_root does not equal registered sft-train tree: "
            f"recorded={recorded_sft_data}, registered={sft_data_path.resolve()}"
        )

    stage = payload.get("stage_hyperparameters")
    if not isinstance(stage, dict):
        raise ValueError("initial SFT hyperparams has no stage_hyperparameters object")
    for key, value in MATCHED_SFT_STAGE_HPARAMETERS.items():
        if stage.get(key) != value:
            raise ValueError(
                f"initial SFT stage_hyperparameters.{key}={stage.get(key)!r}, expected {value!r}"
            )
    if set(stage.get("lora_target_modules") or []) != JIANG_LORA_MODULES:
        raise ValueError("initial SFT does not use the seven Jiang LoRA target modules")

    distributed = payload.get("distributed")
    if not isinstance(distributed, dict) or distributed.get("world_size") != 1:
        raise ValueError("initial SFT is not the Jiang-matched single-H100 specialist run")

    adapter = _load_object(path / "adapter_config.json")
    expected_adapter = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 64,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "bias": LORA_BIAS_POLICY,
    }
    for key, value in expected_adapter.items():
        if adapter.get(key) != value:
            raise ValueError(
                f"initial SFT adapter_config.{key}={adapter.get(key)!r}, expected {value!r}"
            )
    if set(adapter.get("target_modules") or []) != JIANG_LORA_MODULES:
        raise ValueError("initial SFT adapter does not use the seven Jiang LoRA target modules")

    status = _load_object(path / "stage_status.json")
    if not (
        status.get("stage") == "sft"
        and status.get("completed") is True
        and int(status.get("global_step", -1)) == SFT_FINAL_STEP
        and int(status.get("max_steps", -1)) == SFT_FINAL_STEP
        and status.get("has_saved_model") is True
    ):
        raise ValueError("initial SFT has no completed step-31250 finalization receipt")
    return json.loads(json.dumps(MATCHED_SFT_CONTRACT))


def validate_adapter_base_model(model_path: Path, base_model_path: Path) -> dict[str, Any]:
    """Require the PEFT loader's actual base to equal the contracted base artifact."""

    adapter_path = model_path.expanduser().resolve() / "adapter_config.json"
    adapter = _load_object(adapter_path)
    raw_reference = str(adapter.get("base_model_name_or_path", "") or "").strip()
    if not raw_reference:
        raise ValueError("initial SFT adapter_config has no base_model_name_or_path")
    actual = Path(raw_reference).expanduser().resolve()
    expected = base_model_path.expanduser().resolve()
    if actual != expected:
        raise ValueError(
            "initial SFT adapter base does not equal contracted base model: "
            f"actual={actual}, contracted={expected}"
        )
    audit_qwen25_base_bias_scope(expected)
    audit_adapter_persistence(model_path)
    validate_audit_receipt(
        model_path / "adapter_persistence_receipt.json",
        base_model_path=expected,
        adapter_path=model_path,
    )
    validate_reload_receipt(
        model_path / SFT_RELOAD_EQUIVALENCE_FILENAME,
        base_model_path=expected,
        adapter_path=model_path,
    )
    return publication_fidelity_identity(model_path)


def _validate_tree_receipt(receipt: object, *, label: str) -> dict[str, Any]:
    expected_keys = {
        "canonical_path",
        "sha256",
        "file_count",
        "size_bytes",
        "files",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise ValueError(f"{label} has an invalid tree-receipt schema")
    canonical_path = receipt.get("canonical_path")
    files = receipt.get("files")
    if (
        not isinstance(canonical_path, str)
        or not os.path.isabs(canonical_path)
        or not isinstance(files, list)
        or not files
        or isinstance(receipt.get("file_count"), bool)
        or not isinstance(receipt.get("file_count"), int)
        or receipt.get("file_count") != len(files)
        or isinstance(receipt.get("size_bytes"), bool)
        or not isinstance(receipt.get("size_bytes"), int)
        or int(receipt["size_bytes"]) <= 0
        or re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("sha256", ""))) is None
    ):
        raise ValueError(f"{label} has invalid tree-receipt identity fields")
    aggregate = hashlib.sha256()
    observed_size = 0
    observed_paths: list[str] = []
    for index, record in enumerate(files):
        if not isinstance(record, dict) or set(record) != {"path", "size_bytes", "sha256"}:
            raise ValueError(f"{label}.files[{index}] has an invalid schema")
        relative = record.get("path")
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(digest or "")) is None
        ):
            raise ValueError(f"{label}.files[{index}] has invalid identity fields")
        observed_paths.append(relative)
        observed_size += size
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(digest).encode("ascii"))
        aggregate.update(b"\n")
    if observed_paths != sorted(observed_paths) or len(observed_paths) != len(set(observed_paths)):
        raise ValueError(f"{label} file paths are not sorted and unique")
    if observed_size != receipt["size_bytes"] or aggregate.hexdigest() != receipt["sha256"]:
        raise ValueError(f"{label} tree receipt is internally inconsistent")
    return receipt


def _portable_tree_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sha256": receipt["sha256"],
        "file_count": receipt["file_count"],
        "size_bytes": receipt["size_bytes"],
        "files": receipt["files"],
    }


def validate_model_initialization_registry(payload: object) -> dict[str, Any]:
    """Validate a full pre-training registry and return its portable compact identity."""

    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "frozen_at_utc",
        "model_family",
        "base_model_signature",
        "forbidden_source_registry",
        "base_model",
        "matched_sft_contract",
        "specialist_sft_by_task",
        "registry_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("model-initialization registry has an invalid schema")
    unsigned = dict(payload)
    stored_hash = unsigned.pop("registry_sha256", None)
    observed_hash = _sha256_json(unsigned)
    if stored_hash != observed_hash:
        raise ValueError(
            "model-initialization registry self-hash mismatch: "
            f"stored={stored_hash!r}, observed={observed_hash}"
        )
    if (
        payload.get("schema_version") != MODEL_INITIALIZATION_REGISTRY_SCHEMA_VERSION
        or payload.get("protocol") != MODEL_INITIALIZATION_REGISTRY_PROTOCOL
        or payload.get("status") != MODEL_INITIALIZATION_REGISTRY_STATUS
        or payload.get("model_family") != MODEL_FAMILY
        or payload.get("base_model_signature") != QWEN25_7B_BASE_SIGNATURE
        or payload.get("matched_sft_contract") != MATCHED_SFT_CONTRACT
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            str(payload.get("frozen_at_utc", "")),
        )
        is None
    ):
        raise ValueError("model-initialization registry is not the frozen Qwen2.5-7B design")
    forbidden = payload.get("forbidden_source_registry")
    if (
        not isinstance(forbidden, dict)
        or set(forbidden) != {"registry_sha256", "source_tree_manifest_sha256"}
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(forbidden.get(field, ""))) is None
            for field in ("registry_sha256", "source_tree_manifest_sha256")
        )
    ):
        raise ValueError("model-initialization registry has invalid training-data lineage")
    base_model = _validate_tree_receipt(payload.get("base_model"), label="approved base model")
    validate_jiang_f638_base_receipt(base_model)
    specialists = payload.get("specialist_sft_by_task")
    if not isinstance(specialists, dict) or set(specialists) != TASKS:
        raise ValueError("model-initialization registry must cover all five specialist tasks")
    matched_contract_sha256 = _sha256_json(MATCHED_SFT_CONTRACT)
    compact_specialists: dict[str, dict[str, Any]] = {}
    for task in sorted(TASKS):
        specialist = specialists.get(task)
        expected_specialist_keys = {
            "task",
            "sft_seed",
            "sft_train_source_identity_sha256",
            "sft_train_file_manifest_sha256",
            "initial_model",
            "matched_sft_contract_sha256",
            "checkpoint_fidelity",
            "reload_equivalence",
        }
        if (
            not isinstance(specialist, dict)
            or set(specialist) != expected_specialist_keys
            or specialist.get("task") != task
            or specialist.get("sft_seed") != 42
            or specialist.get("matched_sft_contract_sha256") != matched_contract_sha256
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(specialist.get(field, ""))) is None
                for field in (
                    "sft_train_source_identity_sha256",
                    "sft_train_file_manifest_sha256",
                )
            )
        ):
            raise ValueError(f"model-initialization specialist entry is invalid for {task}")
        initial_model = _validate_tree_receipt(
            specialist.get("initial_model"), label=f"approved {task} specialist SFT"
        )
        try:
            checkpoint_fidelity = validate_publication_fidelity_identity(
                specialist.get("checkpoint_fidelity")
            )
        except ValueError as exc:
            raise ValueError(
                f"model-initialization specialist fidelity is invalid for {task}: {exc}"
            ) from exc
        try:
            reload_equivalence = validate_reload_equivalence_identity(
                specialist.get("reload_equivalence")
            )
        except ValueError as exc:
            raise ValueError(
                f"model-initialization specialist reload equivalence is invalid for {task}: {exc}"
            ) from exc
        if (
            reload_equivalence["checkpoint_fidelity_identity_sha256"]
            != checkpoint_fidelity["identity_sha256"]
        ):
            raise ValueError(
                f"model-initialization specialist reload evidence targets another checkpoint for {task}"
            )
        compact_specialists[task] = {
            "initial_model_tree_sha256": initial_model["sha256"],
            "sft_train_source_identity_sha256": specialist["sft_train_source_identity_sha256"],
            "sft_seed": specialist["sft_seed"],
            "matched_sft_contract_sha256": matched_contract_sha256,
            "checkpoint_fidelity": checkpoint_fidelity,
            "reload_equivalence": reload_equivalence,
        }
    return {
        "schema_version": MODEL_INITIALIZATION_REGISTRY_SCHEMA_VERSION,
        "protocol": MODEL_INITIALIZATION_REGISTRY_PROTOCOL,
        "status": MODEL_INITIALIZATION_REGISTRY_STATUS,
        "registry_sha256": stored_hash,
        "model_family": MODEL_FAMILY,
        "base_model_tree_sha256": base_model["sha256"],
        "forbidden_source_registry_sha256": forbidden["registry_sha256"],
        "forbidden_source_tree_manifest_sha256": forbidden["source_tree_manifest_sha256"],
        "specialist_sft_by_task": compact_specialists,
    }


def build_model_initialization_registry(
    *,
    base_model_path: Path,
    specialists: Mapping[str, Mapping[str, Any]],
    forbidden_source_registry: Path,
    frozen_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the complete five-task registry before any publication RL trajectory."""

    if set(specialists) != TASKS:
        raise ValueError("specialist specifications must exactly cover all five tasks")
    registry_payload = _load_object(forbidden_source_registry.expanduser().resolve())
    if registry_payload.get("protocol") == BASE_SFT_PROTOCOL:
        registry_identity = validate_base_sft_source_registry(
            registry_payload,
            expected_tasks=sorted(TASKS),
        )
    else:
        # Retain read compatibility for historical initialization-registry
        # fixtures. The operational freezer separately requires a v7 base-SFT
        # campaign, whose runtime gate admits only the post-IID protocol.
        registry_identity = validate_forbidden_source_registry(
            registry_payload,
            expected_tasks=sorted(TASKS),
        )
    base_signature = validate_qwen25_7b_base(base_model_path)
    base_fingerprint = audit_jiang_f638_base_snapshot(base_model_path)
    specialist_receipts: dict[str, dict[str, Any]] = {}
    for task in sorted(TASKS):
        spec = specialists[task]
        if set(spec) != {"model_path", "sft_data_path", "sft_seed"}:
            raise ValueError(f"specialist specification has an invalid schema for {task}")
        model_path = Path(spec["model_path"])
        sft_data_path = Path(spec["sft_data_path"])
        raw_seed = spec["sft_seed"]
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise ValueError(f"specialist SFT seed is invalid for {task}")
        validate_matched_sft(
            model_path,
            task=task,
            sft_seed=raw_seed,
            sft_data_path=sft_data_path,
        )
        checkpoint_fidelity = validate_adapter_base_model(model_path, base_model_path)
        reload_equivalence = reload_equivalence_identity(
            model_path,
            base_model_path=base_model_path,
        )
        source_binding = _registered_source_binding(
            registry_identity=registry_identity,
            task=task,
            source_label="sft-train",
            source_path=sft_data_path,
        )
        specialist_receipts[task] = {
            "task": task,
            "sft_seed": raw_seed,
            "sft_train_source_identity_sha256": source_binding["source_identity_sha256"],
            "sft_train_file_manifest_sha256": source_binding["file_manifest_sha256"],
            "initial_model": fingerprint_model(model_path, label=f"approved {task} specialist SFT"),
            "matched_sft_contract_sha256": _sha256_json(MATCHED_SFT_CONTRACT),
            "checkpoint_fidelity": checkpoint_fidelity,
            "reload_equivalence": reload_equivalence,
        }
    timestamp = frozen_at_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    core: dict[str, Any] = {
        "schema_version": MODEL_INITIALIZATION_REGISTRY_SCHEMA_VERSION,
        "protocol": MODEL_INITIALIZATION_REGISTRY_PROTOCOL,
        "status": MODEL_INITIALIZATION_REGISTRY_STATUS,
        "frozen_at_utc": timestamp,
        "model_family": MODEL_FAMILY,
        "base_model_signature": base_signature,
        "forbidden_source_registry": {
            "registry_sha256": registry_identity["registry_sha256"],
            "source_tree_manifest_sha256": registry_identity["source_tree_manifest_sha256"],
        },
        "base_model": base_fingerprint,
        "matched_sft_contract": json.loads(json.dumps(MATCHED_SFT_CONTRACT)),
        "specialist_sft_by_task": specialist_receipts,
    }
    core["registry_sha256"] = _sha256_json(core)
    validate_model_initialization_registry(core)
    return core


def verify_model_initialization_registry(
    *,
    registry_path: Path,
    task: str,
    model_path: Path,
    base_model_path: Path,
    sft_data_path: Path,
    sft_seed: int,
    forbidden_registry_identity: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify actual training inputs against the exact artifact registry."""

    payload = _load_object(registry_path.expanduser().resolve())
    compact = validate_model_initialization_registry(payload)
    if compact["forbidden_source_registry_sha256"] != forbidden_registry_identity.get(
        "registry_sha256"
    ) or compact["forbidden_source_tree_manifest_sha256"] != forbidden_registry_identity.get(
        "source_tree_manifest_sha256"
    ):
        raise ValueError(
            "model-initialization registry belongs to another forbidden-source registry"
        )
    validate_qwen25_7b_base(base_model_path)
    validate_matched_sft(
        model_path,
        task=task,
        sft_seed=sft_seed,
        sft_data_path=sft_data_path,
    )
    actual_checkpoint_fidelity = validate_adapter_base_model(model_path, base_model_path)
    actual_reload_equivalence = reload_equivalence_identity(
        model_path,
        base_model_path=base_model_path,
    )
    actual_base = audit_jiang_f638_base_snapshot(base_model_path)
    approved_base = payload["base_model"]
    if _portable_tree_receipt(actual_base) != _portable_tree_receipt(approved_base):
        raise ValueError("actual base model bytes differ from the approved initialization registry")
    specialist = payload["specialist_sft_by_task"][task]
    if actual_checkpoint_fidelity != specialist["checkpoint_fidelity"]:
        raise ValueError(
            f"actual {task} specialist checkpoint fidelity differs from the approved registry"
        )
    if actual_reload_equivalence != specialist["reload_equivalence"]:
        raise ValueError(
            f"actual {task} specialist reload equivalence differs from the approved registry"
        )
    actual_initial = fingerprint_model(model_path, label=f"actual {task} specialist SFT")
    if _portable_tree_receipt(actual_initial) != _portable_tree_receipt(
        specialist["initial_model"]
    ):
        raise ValueError(
            f"actual {task} specialist SFT bytes differ from the approved initialization registry"
        )
    task_sources = forbidden_registry_identity.get("sources_by_task_label")
    registered_sft = (
        task_sources.get(task, {}).get("sft-train")
        if isinstance(task_sources, dict) and isinstance(task_sources.get(task), dict)
        else None
    )
    if (
        not isinstance(registered_sft, dict)
        or specialist["sft_train_source_identity_sha256"]
        != registered_sft.get("source_identity_sha256")
        or specialist["sft_train_file_manifest_sha256"]
        != registered_sft.get("file_manifest_sha256")
        or specialist["sft_seed"] != sft_seed
    ):
        raise ValueError(f"approved {task} specialist SFT lineage/settings disagree with this run")
    return compact, actual_initial, actual_base


def _parse_key_values(items: Iterable[str], *, label: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{label} entry must use KEY=VALUE syntax, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", key):
            raise ValueError(f"invalid {label} key {key!r}")
        if key in parsed:
            raise ValueError(f"duplicate {label} key {key!r}")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        if isinstance(value, (dict, list)):
            raise ValueError(f"{label} values must be scalar, got {item!r}")
        parsed[key] = value
    return dict(sorted(parsed.items()))


def tracked_environment(
    environ: dict[str, str] | os._Environ[str],
    *,
    allowed_names: set[str] | None = None,
) -> dict[str, str]:
    tracked = {
        key: value
        for key, value in environ.items()
        if key not in MUTABLE_ENV_NAMES and key.startswith(TRACKED_ENV_PREFIXES)
    }
    if allowed_names is not None:
        unexpected = sorted(set(tracked) - allowed_names)
        if unexpected:
            raise ValueError(
                "unexpected tracked environment variables would contaminate the run: "
                + ", ".join(unexpected)
            )
    return dict(sorted(tracked.items()))


def _require_exact_fields(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    differences = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if differences:
        raise ValueError(f"{label} differs from canonical matched contract: {differences}")


def validate_canonical_publication_arm(
    *,
    method: str,
    task: str,
    train_seed: int,
    config: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    environment: Mapping[str, str],
) -> None:
    """Fail closed on any trajectory-critical drift in the new A100 matrix."""

    if method not in CANONICAL_PUBLICATION_METHODS:
        return
    expected_generation = {
            "num_generations": 9,
            "generation_batch_size": 36,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "max_prompt_length": (
                20_000 if method == CLASP_FOARL_MATCHED_B9 else 29_768
            ),
            "max_completion_length": (
                1_000 if method == CLASP_FOARL_MATCHED_B9 else 3_000
            ),
        }
    _require_exact_fields(
        generation_config,
        expected_generation,
        label="generation configuration",
    )
    expected_arm_schema = (
        CLASP_FOARL_MATCHED_B9_ARM_SCHEMA
        if method == CLASP_FOARL_MATCHED_B9
        else "clasp-canonical-matched-a100-v1"
    )
    _require_exact_fields(
        config,
        {
            "entrypoint": "clasp.entrypoints.train_grpo",
            "arm_contract_schema": expected_arm_schema,
            "policy_surrogate": "clipped_token_level_practical",
            "deployment_budget_k": 8,
            "verifier_gate": "strict_recomputed_feasible_only_quality",
            "intentional_method_diff": (
                "policy_credit_formula,component_controller,set_controller,use_clasp,"
                "clasp_canonical_set,clasp_fixed_null_set,pd_group_scalarization,pd_separate_z,"
                "pd_dual_lr,pd_dual_lr_up,"
                "pd_dual_lr_down,clasp_bon_dual_lr,clasp_bon_lambda_init,"
                "clasp_canonical_recovery_scale"
            ),
            "loss_type": "grpo",
            "dtype": "bfloat16",
            "per_device_train_batch_size": 3,
            "per_device_eval_batch_size": 9,
            "gradient_accumulation_steps": 3,
            "use_primal_dual": True,
            "pd_objective_weight": 1.0,
            "pd_structure_weight": 0.0,
            "pd_lambda_init": 0.0,
            "pd_cost_limit": 0.0,
            "pd_dual_update_interval": 1,
            "pd_violation_ema_beta": 0.1,
            "pd_positive_constraint_penalty": True,
            "pd_separate_z": method == "dual_separate_z",
            "pd_soft_objective_for_infeasible": False,
            "pd_component_profile": "co_feasibility_lock_all",
            "clasp_deployment_set_size": 8,
            "clasp_canonical_recovery_temperature": 0.15,
            "clasp_canonical_max_subsets": 9,
            "clasp_component_dual_recovery_scale": 0.0,
            "clasp_bon_target_failure_rate": 0.005,
            "clasp_feasibility_warmup_steps": 0,
            "clasp_self_replay": False,
            "clasp_proper_dual": False,
            "clasp_adaptive_length": False,
            "importance_sampling_level": "token",
        },
        label="training configuration",
    )
    _require_exact_fields(
        environment,
        {
            "CLASP_FRONTIER_CDF": "0",
            "CLASP_BON_TARGET_FAILURE_RATE": "0.005",
            "CLASP_BON_TARGET_FAILURE_RATES": f"{task}:0.005",
            "CLASP_BON_RECOVERY_SCALE": "0.0",
            "CLASP_COMPONENT_DUAL_RECOVERY_SCALE": "0.0",
            "CLASP_FEASIBILITY_WARMUP_STEPS": "0",
            "CLASP_SELF_REPLAY_ENABLED": "0",
            "CLASP_VALIDATION_GATE_CHECKPOINTS": "0",
            "CLASP_PAIRWISE_LOSS_WEIGHT": "0.0",
            "CLASP_ELITE_IMITATION_LOSS_WEIGHT": "0.0",
            "CLASP_MULTI_ELITE_IMITATION_LOSS_WEIGHT": "0.0",
            "CLASP_REFERENCE_IMITATION_LOSS_WEIGHT": "0.0",
            "CLASP_REFERENCE_CONTRASTIVE_LOSS_WEIGHT": "0.0",
            "CLASP_HARD_GROUP_WEIGHT_SCALE": "0.0",
            "PD_IMPORTANCE_SAMPLING_LEVEL": "token",
            "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION": "1",
            "GRPO_DATA_SEED": str(train_seed),
        },
        label="tracked environment",
    )

    canonical = method.startswith("clasp_canonical")
    fixed_null = method == "set_null"
    exact_set_credit = canonical or fixed_null
    policy_formula = (
        "group_z_verified_quality_v1"
        if method == "grpo_z"
        else "group_z_scalarized_positive_component_penalty_v1"
        if method == "dual_scalar_z"
        else "group_z_reward_minus_group_z_positive_component_penalty_v1"
        if method == "dual_separate_z"
        else "fixed_null_k8_same_set_utility_v1"
        if fixed_null
        else "canonical_k8_leave_out_set_utility_v1"
    )
    component_controller = (
        "legacy_projected_ema_component_gap"
        if method in {"dual_scalar_z", "dual_separate_z"}
        else "frozen_zero"
    )
    set_controller = (
        "frozen"
        if method == "clasp_canonical_fixed_lambda"
        else "signed_failure_at_k_ema"
        if exact_set_credit
        else "off"
    )
    method_expected: dict[str, Any] = {
        "policy_credit_formula": policy_formula,
        "component_controller": component_controller,
        "set_controller": set_controller,
        "use_clasp": exact_set_credit,
        "clasp_canonical_set": canonical,
        "clasp_fixed_null_set": fixed_null,
        "pd_group_scalarization": method in {"grpo_z", "dual_scalar_z"},
        "pd_separate_z": method == "dual_separate_z",
        "pd_dual_lr": (
            0.05 if method in {"dual_scalar_z", "dual_separate_z"} else 0.0
        ),
        "clasp_bon_dual_lr": (
            0.0
            if method
            in {
                "grpo_z",
                "dual_scalar_z",
                "dual_separate_z",
                "clasp_canonical_fixed_lambda",
            }
            else 0.05
        ),
        "clasp_canonical_recovery_scale": (
            1.0
            if method
            in {
                "clasp_canonical",
                "clasp_canonical_fixed_lambda",
                CLASP_FOARL_MATCHED_B9,
                "set_null",
            }
            else 0.0
        ),
    }
    _require_exact_fields(config, method_expected, label=f"{method} method profile")
    _require_exact_fields(
        config,
        {
            "pd_dual_lr_up": method_expected["pd_dual_lr"],
            "pd_dual_lr_down": method_expected["pd_dual_lr"],
        },
        label=f"{method} component-controller profile",
    )
    _require_exact_fields(
        environment,
        {
            "CLASP_CANONICAL_SET": "1" if canonical else "0",
            "CLASP_FIXED_NULL_SET": "1" if fixed_null else "0",
            "CLASP_BON_ALIGNED": "1" if exact_set_credit else "0",
            "CLASP_FEASIBILITY_GATED": "1" if exact_set_credit else "0",
            "CLASP_CANONICAL_RECOVERY_SCALE": str(
                method_expected["clasp_canonical_recovery_scale"]
            ),
            "CLASP_BON_DUAL_LR": str(method_expected["clasp_bon_dual_lr"]),
        },
        label=f"{method} environment profile",
    )
    receipt_environment_path = str(
        environment.get("CLASP_FIXED_NULL_RECEIPT_PATH", "") or ""
    ).strip()
    if fixed_null and not receipt_environment_path:
        raise ValueError(
            "set_null requires CLASP_FIXED_NULL_RECEIPT_PATH in the tracked environment"
        )
    if not fixed_null and receipt_environment_path:
        raise ValueError(
            "non-set_null methods reject CLASP_FIXED_NULL_RECEIPT_PATH"
        )
    try:
        fixed_lambda = float(config.get("clasp_bon_lambda_init"))
        environment_lambda = float(environment.get("CLASP_BON_LAMBDA_INIT", "nan"))
    except (TypeError, ValueError) as exc:
        raise ValueError("canonical matched contract has a nonnumeric set lambda") from exc
    if (
        not math.isfinite(fixed_lambda)
        or fixed_lambda < 0.0
        or environment_lambda != fixed_lambda
        or (method != "clasp_canonical_fixed_lambda" and fixed_lambda != 0.0)
    ):
        raise ValueError(
            "set lambda differs from the explicit controller/fixed-lambda arm contract"
        )
    if method == CLASP_FOARL_MATCHED_B9:
        _require_exact_fields(
            config,
            {
                "comparison_protocol": FOARL_MATCHED_B9_COMPARISON_PROTOCOL,
                "comparison_baseline": "foarl_f638_matched_b9",
                "attribution": "common_compute_data_token_schedule_only",
                "reproduction_claim": "not_an_official_jiang_reproduction",
                "beta": 0.05,
            },
            label="CLASP/FOARL matched-B9 comparison identity",
        )


def build_contract(
    *,
    method: str,
    task: str,
    train_seed: int,
    sft_seed: int,
    final_step: int,
    world_size: int,
    model_path: Path,
    base_model_path: Path,
    data_path: Path,
    sft_data_path: Path,
    forbidden_source_registry: Path,
    model_initialization_registry: Path,
    source_root: Path,
    source_manifest: Path,
    remote_environment_receipt: Path | None,
    fixed_kl_reference_model_path: Path | None,
    fixed_kl_reference_receipt_path: Path | None,
    fixed_null_receipt_path: Path | None,
    config: dict[str, Any],
    generation_config: dict[str, Any],
    environment: dict[str, str],
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}, got {method!r}")
    if task not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}, got {task!r}")
    if train_seed < 0 or sft_seed < 0:
        raise ValueError("train and SFT seeds must be nonnegative")
    if final_step <= 0:
        raise ValueError("final_step must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not generation_config:
        raise ValueError("generation_config must not be empty")
    if environment.get("CLASP_PUBLICATION_FIDELITY") != "1":
        raise ValueError("publication run contract requires CLASP_PUBLICATION_FIDELITY=1")
    if config.get("checkpoint_fidelity") != "publication_exact_required":
        raise ValueError(
            "publication run contract requires checkpoint_fidelity=publication_exact_required"
        )
    beta = config.get("beta")
    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not math.isfinite(float(beta))
        or float(beta) < 0.0
    ):
        raise ValueError("publication run contract requires a finite nonnegative beta")

    validate_canonical_publication_arm(
        method=method,
        task=task,
        train_seed=train_seed,
        config=config,
        generation_config=generation_config,
        environment=environment,
    )
    if method == CLASP_FOARL_MATCHED_B9:
        validate_clasp_foarl_matched_b9_method_contract(
            task=task,
            train_seed=train_seed,
            fixed_horizon={"max_steps": final_step, "world_size": world_size},
            generation=generation_config,
            training_config=config,
            tracked_environment=environment,
        )
    if method in FOARL_PUBLICATION_METHODS:
        validate_foarl_f638_campaign_contract(
            method=method,
            task=task,
            train_seed=train_seed,
            final_step=final_step,
            world_size=world_size,
            config=config,
            generation_config=generation_config,
            environment=environment,
        )
    config = dict(config)
    if method in PUBLICATION_METHODS:
        if remote_environment_receipt is None:
            raise ValueError(
                "publication arms require --remote-environment-receipt"
            )
        config["remote_environment"] = environment_identity(
            remote_environment_receipt,
            require_accelerator=True,
        )
    elif remote_environment_receipt is not None:
        raise ValueError(
            "legacy run contracts cannot be relabeled with the canonical campaign's "
            "remote-environment receipt"
        )

    fixed_null_receipt = None
    fixed_null_environment_path = str(
        environment.get("CLASP_FIXED_NULL_RECEIPT_PATH", "") or ""
    ).strip()
    if method == "set_null":
        if fixed_null_receipt_path is None:
            raise ValueError("set_null requires --fixed-null-receipt")
        canonical_receipt_path = fixed_null_receipt_path.expanduser().resolve()
        if fixed_null_environment_path != str(canonical_receipt_path):
            raise ValueError(
                "fixed-null CLI receipt path disagrees with the tracked environment"
            )
        # This import is intentionally lazy: historical/non-SET preparation
        # remains dependency-light, while SET-NULL must load the active reward
        # stack and replay every frozen source row before any model allocation.
        from clasp.set_grpo.fixed_null_replay import verify_fixed_null_receipt

        verify_fixed_null_receipt(
            receipt_path=canonical_receipt_path,
            training_rows_path=data_path,
            expected_task=task,
            expected_prompt_style=str(config.get("prompt_style", "")),
            expected_no_reason=bool(config.get("no_reason")),
            expected_config=config,
            expected_environment=environment,
        )
        receipt = load_fixed_null_score_receipt(canonical_receipt_path)
        if task not in receipt.identity["tasks"]:
            raise ValueError(
                f"fixed-null receipt has no evaluator records for task {task!r}"
            )
        fixed_null_receipt = dict(receipt.identity)
    elif fixed_null_receipt_path is not None or fixed_null_environment_path:
        raise ValueError("only set_null may bind a fixed-null construction receipt")

    registry = _load_object(forbidden_source_registry.expanduser().resolve())
    if method in PUBLICATION_METHODS:
        registry_identity = validate_base_sft_source_registry(
            registry,
            expected_tasks=sorted(TASKS),
        )
    else:
        registry_identity = validate_forbidden_source_registry(registry)
    sft_binding = _registered_source_binding(
        registry_identity=registry_identity,
        task=task,
        source_label="sft-train",
        source_path=sft_data_path,
    )
    rl_binding = _registered_source_binding(
        registry_identity=registry_identity,
        task=task,
        source_label="rl-train",
        source_path=data_path,
    )
    if method == FOARL_F638_NATIVE:
        expected_rows = get_foarl_f638_campaign_profile(method).source_rows(task)
        if rl_binding.get("rows") != expected_rows:
            raise ValueError(
                f"{method}/{task} requires exactly {expected_rows} registered source rows, "
                f"got {rl_binding.get('rows')!r}"
            )

    validate_source_snapshot(source_root, source_manifest)
    validate_matched_sft(
        model_path,
        task=task,
        sft_seed=sft_seed,
        sft_data_path=sft_data_path,
    )
    validate_adapter_base_model(model_path, base_model_path)
    (
        initialization_identity,
        initial_model_fingerprint,
        base_model_fingerprint,
    ) = verify_model_initialization_registry(
        registry_path=model_initialization_registry,
        task=task,
        model_path=model_path,
        base_model_path=base_model_path,
        sft_data_path=sft_data_path,
        sft_seed=sft_seed,
        forbidden_registry_identity=registry_identity,
    )

    kl_model_env = environment.get("CLASP_KL_REFERENCE_MODEL_PATH")
    kl_receipt_env = environment.get("CLASP_KL_REFERENCE_RECEIPT")
    if float(beta) == 0.0:
        if (
            fixed_kl_reference_model_path is not None
            or fixed_kl_reference_receipt_path is not None
            or kl_model_env is not None
            or kl_receipt_env is not None
        ):
            raise ValueError("beta=0 publication contracts forbid fixed-KL reference inputs")
        fixed_kl_reference = None
    else:
        if fixed_kl_reference_model_path is None or fixed_kl_reference_receipt_path is None:
            raise ValueError(
                "beta>0 publication contracts require explicit fixed-KL model and receipt paths"
            )
        canonical_kl_model = fixed_kl_reference_model_path.expanduser().resolve()
        canonical_kl_receipt = fixed_kl_reference_receipt_path.expanduser().resolve()
        if kl_model_env != str(canonical_kl_model) or kl_receipt_env != str(canonical_kl_receipt):
            raise ValueError("fixed-KL CLI paths disagree with the tracked publication environment")
        canonical_policy = model_path.expanduser().resolve()
        if canonical_kl_model != canonical_policy:
            raise ValueError(
                "beta>0 exact PEFT KL reference must be the same specialist SFT adapter "
                "used to initialize the policy"
            )
        expected_receipt = canonical_policy / CHECKPOINT_FIDELITY_FILENAME
        if canonical_kl_receipt != expected_receipt:
            raise ValueError(
                "beta>0 KL receipt must be the specialist adapter's authoritative "
                f"{CHECKPOINT_FIDELITY_FILENAME}"
            )
        fixed_kl_reference = fixed_kl_reference_identity(
            canonical_kl_model,
            canonical_kl_receipt,
        )
        expected_source = initialization_identity["specialist_sft_by_task"][task][
            "checkpoint_fidelity"
        ]
        if fixed_kl_reference["source_checkpoint_fidelity"] != expected_source:
            raise ValueError(
                "fixed-KL source policy differs from the approved specialist initialization"
            )
        validate_fixed_kl_reference_identity(fixed_kl_reference)

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "grpo",
        "method": method,
        "task": task,
        "seeds": {"train": train_seed, "data": train_seed, "sft": sft_seed},
        "fixed_horizon": {"max_steps": final_step, "world_size": world_size},
        "data_lineage": {
            "forbidden_source_registry": registry_identity,
            "sft_train": sft_binding,
            "rl_train": rl_binding,
        },
        "model_initialization_registry": initialization_identity,
        "fixed_kl_reference": fixed_kl_reference,
        "fixed_null_receipt": fixed_null_receipt,
        "fingerprints": {
            "initial_model": initial_model_fingerprint,
            "base_model": base_model_fingerprint,
            "data": _tree_fingerprint(data_path, label="task data"),
            "source": _tree_fingerprint(
                source_root,
                include=_source_file,
                label="training source",
            ),
        },
        "generation": generation_config,
        "training_config": config,
        "tracked_environment": environment,
    }


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_contract_once(run_dir: Path, payload: dict[str, Any]) -> None:
    """Atomically create a read-only contract without replacing an existing one."""
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / CONTRACT_FILENAME
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{CONTRACT_FILENAME}.", dir=run_dir)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, stat.S_IRUSR | stat.S_IRGRP)
        try:
            os.link(temporary_path, contract_path)
        except FileExistsError:
            return
        _fsync_directory(run_dir)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _diff_values(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            label = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                differences.append(f"{label}: unexpected={actual[key]!r}")
            elif key not in actual:
                differences.append(f"{label}: missing (expected {expected[key]!r})")
            else:
                differences.extend(_diff_values(expected[key], actual[key], label))
        return differences
    if expected != actual:
        return [f"{prefix}: actual={actual!r}, expected={expected!r}"]
    return []


def enforce_contract(run_dir: Path, expected: dict[str, Any]) -> None:
    contract_path = run_dir / CONTRACT_FILENAME
    if not contract_path.exists():
        existing = [path.name for path in run_dir.iterdir()] if run_dir.exists() else []
        if existing:
            raise ValueError(
                f"run directory already contains artifacts but has no {CONTRACT_FILENAME}: "
                f"{run_dir} ({', '.join(sorted(existing)[:8])})"
            )
        create_contract_once(run_dir, expected)

    actual = _load_object(contract_path)
    differences = _diff_values(expected, actual)
    if differences:
        preview = "\n  - ".join(differences[:20])
        remainder = len(differences) - min(20, len(differences))
        suffix = f"\n  - ... and {remainder} more difference(s)" if remainder else ""
        raise ValueError(
            f"immutable run contract mismatch at {contract_path}:\n  - {preview}{suffix}"
        )


def _checkpoint_model_exists(checkpoint: Path) -> bool:
    return any((checkpoint / name).is_file() for name in MODEL_ARTIFACTS)


def _validate_pd_state(path: Path, *, task: str, require_bon_state: bool) -> None:
    state = _load_object(path)
    task_state = state.get(task)
    if not isinstance(task_state, dict):
        raise ValueError(f"primal-dual state has no object for task {task!r}")
    missing = sorted(PD_TASK_KEYS - set(task_state))
    if missing:
        raise ValueError(f"primal-dual task state is missing keys: {', '.join(missing)}")
    cost_names = task_state.get("cost_names")
    for key in ("cost_limits", "cost_weights", "dual_lambdas"):
        value = task_state.get(key)
        if not isinstance(cost_names, list) or not isinstance(value, list):
            raise ValueError(f"primal-dual {key} and cost_names must be lists")
        if len(value) != len(cost_names):
            raise ValueError(f"primal-dual {key} length does not match cost_names")
    if require_bon_state:
        bon_state = state.get("__clasp_bon__")
        if not isinstance(bon_state, dict):
            raise ValueError("CLASP checkpoint has no __clasp_bon__ controller state")
        bon_missing = sorted(BON_STATE_KEYS - set(bon_state))
        if bon_missing:
            raise ValueError(
                "CLASP BoN controller state is missing keys: " + ", ".join(bon_missing)
            )


def strict_checkpoint_step(
    checkpoint: Path,
    *,
    task: str,
    final_step: int,
    world_size: int,
    require_pd_state: bool,
    require_bon_state: bool,
) -> int:
    match = CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None or not checkpoint.is_dir():
        raise ValueError(f"invalid checkpoint directory name: {checkpoint}")
    step = int(match.group(1))
    if step > final_step:
        raise ValueError(f"checkpoint step {step} exceeds fixed horizon {final_step}")
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"checkpoint {checkpoint} is missing: {', '.join(missing)}")
    if not _checkpoint_model_exists(checkpoint):
        raise ValueError(f"checkpoint {checkpoint} has no model weight artifact")
    missing_rng = [
        f"rng_state_{rank}.pth"
        for rank in range(world_size)
        if not (checkpoint / f"rng_state_{rank}.pth").is_file()
    ]
    if missing_rng:
        raise ValueError(f"checkpoint {checkpoint} is missing: {', '.join(missing_rng)}")

    trainer_state = _load_object(checkpoint / "trainer_state.json")
    if int(trainer_state.get("global_step", -1)) != step:
        raise ValueError(f"checkpoint {checkpoint} trainer_state.global_step != {step}")
    if int(trainer_state.get("max_steps", -1)) != final_step:
        raise ValueError(
            f"checkpoint {checkpoint} changed scheduler horizon: "
            f"{trainer_state.get('max_steps')!r} != {final_step}"
        )

    if require_pd_state:
        pd_path = checkpoint / "multitask_primal_dual_state.json"
        if not pd_path.is_file():
            raise ValueError(f"checkpoint {checkpoint} has no primal-dual state")
        _validate_pd_state(pd_path, task=task, require_bon_state=require_bon_state)
    publication_fidelity_identity(checkpoint)
    return step


def _is_finalized(run_dir: Path, checkpoint: Path, *, final_step: int) -> bool:
    if not any((run_dir / name).is_file() for name in MODEL_ARTIFACTS):
        return False
    try:
        status = _load_object(run_dir / "stage_status.json")
    except ValueError:
        return False
    latest = str(status.get("latest_checkpoint_dir", "") or "")
    status_complete = (
        status.get("stage") == "grpo"
        and status.get("completed") is True
        and int(status.get("global_step", -1)) == final_step
        and int(status.get("max_steps", -1)) == final_step
        and bool(latest)
        and Path(latest).resolve() == checkpoint.resolve()
    )
    if not status_complete:
        return False
    root_fidelity = publication_fidelity_identity(run_dir)
    terminal_fidelity = publication_fidelity_identity(checkpoint)
    if root_fidelity != terminal_fidelity:
        raise ValueError("final run adapter differs from terminal checkpoint fidelity identity")
    return True


def _validate_continuation_exact_kl(
    run_dir: Path,
    checkpoint: Path,
    *,
    contract: dict[str, Any],
    contract_file_sha256: str,
    step: int,
) -> dict[str, Any] | None:
    """Require exact pre/post reference lineage on every resumable checkpoint."""

    training = contract.get("training_config")
    beta = training.get("beta") if isinstance(training, dict) else None
    roots = (run_dir, checkpoint)
    names = (
        EXACT_PEFT_KL_REFERENCE_FILENAME,
        exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME,
        exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME,
    )
    if not isinstance(beta, (int, float)) or isinstance(beta, bool) or float(beta) < 0.0:
        raise ValueError("run contract has invalid beta for exact-KL continuation")
    if float(beta) == 0.0:
        stale = [
            str(root / name) for root in roots for name in names if os.path.lexists(root / name)
        ]
        if stale:
            raise ValueError(f"beta=0 continuation carries stale exact-KL sidecars: {stale}")
        return None
    fixed = validate_fixed_kl_reference_identity(contract.get("fixed_kl_reference"))
    runtime_identities = [
        exact_peft_kl_runtime_identity(
            root / EXACT_PEFT_KL_REFERENCE_FILENAME,
            expected_fixed_kl_reference=fixed,
        )
        for root in roots
    ]
    if runtime_identities[0] != runtime_identities[1]:
        raise ValueError("continuation root/checkpoint exact step-zero receipts differ")
    checkpoint_fidelity = publication_fidelity_identity(checkpoint)
    root_step1 = exact_peft_kl_post_step.post_step_reference_identity(
        run_dir / exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME,
        claim_scope="publication",
        expected_global_step=1,
        expected_artifact_role="step1",
        rehash_policy_artifact=False,
    )
    checkpoint_step1 = exact_peft_kl_post_step.post_step_reference_identity(
        checkpoint / exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME,
        claim_scope="publication",
        expected_global_step=1,
        expected_artifact_role="step1",
        rehash_policy_artifact=False,
    )
    post = exact_peft_kl_post_step.post_step_reference_identity(
        checkpoint / exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME,
        claim_scope="publication",
        expected_global_step=step,
        expected_artifact_role="checkpoint",
        expected_policy_trainable_tensor_inventory_sha256=checkpoint_fidelity[
            "trainable_tensor_inventory_sha256"
        ],
    )
    runtime = runtime_identities[0]
    horizon = contract["fixed_horizon"]
    if (
        root_step1 != checkpoint_step1
        or post["invariant"] != root_step1["invariant"]
        or post["pre_step_receipt_sha256"] != runtime["receipt_sha256"]
        or post["pre_step_receipt_file_sha256"] != runtime["receipt_file_sha256"]
        or post["training_run_contract_file_sha256"] != contract_file_sha256
        or post["training_run_contract_max_steps"] != horizon["max_steps"]
        or post["training_run_contract_world_size"] != horizon["world_size"]
        or post["verifier_implementation_sha256"]
        != fixed["post_step_verifier_implementation_sha256"]
    ):
        raise ValueError("continuation exact-KL post-step lineage differs from sealed run")
    return {
        "runtime": runtime,
        "root_step1": root_step1,
        "checkpoint_post_step": post,
    }


def resolve_state(
    run_dir: Path,
    *,
    task: str,
    method: str,
    final_step: int,
    world_size: int,
    contract: dict[str, Any] | None = None,
) -> str:
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.exists():
        state_evidence = {
            "hyperparams.json",
            "stage_status.json",
            "grpo_trainer_state.json",
            EXACT_PEFT_KL_REFERENCE_FILENAME,
            exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME,
            exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME,
            *MODEL_ARTIFACTS,
        }
        present = sorted(name for name in state_evidence if (run_dir / name).exists())
        if present:
            raise ValueError(
                "run has training artifacts but no checkpoint directory: " + ", ".join(present)
            )
        return "FRESH"
    if not checkpoint_root.is_dir():
        raise ValueError(f"checkpoint path is not a directory: {checkpoint_root}")

    entries = sorted(checkpoint_root.iterdir(), key=lambda path: path.name)
    checkpoint_dirs = [path for path in entries if path.name.startswith("checkpoint-")]
    unexpected = [
        path.name for path in entries if path.name.startswith("checkpoint-") and not path.is_dir()
    ]
    if unexpected:
        raise ValueError("checkpoint entries are not directories: " + ", ".join(unexpected))
    if not checkpoint_dirs:
        if entries:
            raise ValueError(
                f"checkpoint directory has no valid checkpoint-* state: {checkpoint_root}"
            )
        return "FRESH"

    require_pd_state, require_bon_state = method_state_requirements(method)
    contract_file_sha256 = (
        _sha256_file(run_dir / CONTRACT_FILENAME) if contract is not None else None
    )
    candidates: list[tuple[int, Path]] = []
    errors: list[str] = []
    for checkpoint in checkpoint_dirs:
        try:
            step = strict_checkpoint_step(
                checkpoint,
                task=task,
                final_step=final_step,
                world_size=world_size,
                require_pd_state=require_pd_state,
                require_bon_state=require_bon_state,
            )
            if contract is not None and contract_file_sha256 is not None:
                _validate_continuation_exact_kl(
                    run_dir,
                    checkpoint,
                    contract=contract,
                    contract_file_sha256=contract_file_sha256,
                    step=step,
                )
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        else:
            candidates.append((step, checkpoint.resolve()))
    if errors:
        raise ValueError("malformed checkpoint state; refusing fallback: " + " | ".join(errors))
    if not candidates:
        raise ValueError("no optimizer-resumable checkpoint satisfies the run contract")

    step, checkpoint = max(candidates)
    if step == final_step:
        if _is_finalized(run_dir, checkpoint, final_step=final_step):
            if contract is not None and float(contract["training_config"]["beta"]) > 0.0:
                runtime = exact_peft_kl_runtime_identity(
                    run_dir / EXACT_PEFT_KL_REFERENCE_FILENAME,
                    expected_fixed_kl_reference=contract["fixed_kl_reference"],
                )
                exact_peft_kl_post_step_lineage_identity(
                    run_dir,
                    checkpoint,
                    expected_fixed_kl_reference=contract["fixed_kl_reference"],
                    expected_runtime_identity=runtime,
                    training_run_contract_file_sha256=str(contract_file_sha256),
                    terminal_global_step=final_step,
                )
            return f"COMPLETE:{checkpoint}"
        raise ValueError("target checkpoint exists but final model/status receipt is incomplete")
    premature_root = sorted(
        path.name
        for path in run_dir.iterdir()
        if path.name in TERMINAL_ROOT_ARTIFACTS
        or (
            path.name.startswith(TERMINAL_ROOT_WEIGHT_PREFIXES)
            and path.name.endswith((".bin", ".safetensors", ".index.json"))
        )
    )
    if premature_root:
        raise ValueError(
            "nonterminal continuation checkpoint is contaminated by terminal root "
            "artifacts: " + ", ".join(premature_root)
        )
    return f"RESUME:{checkpoint}"


def method_state_requirements(method: str) -> tuple[bool, bool]:
    """Return immutable checkpoint-sidecar requirements for one method."""

    if method not in METHODS:
        raise ValueError(f"unsupported training method {method!r}")
    require_pd_state = method not in {"grpo", *FOARL_PUBLICATION_METHODS}
    require_bon_state = method.startswith("clasp_") or method == "set_null"
    return require_pd_state, require_bon_state


def prepare(args: argparse.Namespace, *, environ: dict[str, str] | None = None) -> str:
    environ = dict(os.environ if environ is None else environ)
    config = _parse_key_values(args.config, label="config")
    generation = _parse_key_values(args.generation_config, label="generation-config")
    allowed = set(args.allowed_env) if args.allowed_env else None
    environment = tracked_environment(environ, allowed_names=allowed)
    expected = build_contract(
        method=args.method,
        task=args.task,
        train_seed=args.train_seed,
        sft_seed=args.sft_seed,
        final_step=args.final_step,
        world_size=args.world_size,
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        data_path=args.data_path,
        sft_data_path=args.sft_data_path,
        forbidden_source_registry=args.forbidden_source_registry,
        model_initialization_registry=args.model_initialization_registry,
        source_root=args.source_root,
        source_manifest=args.source_manifest,
        remote_environment_receipt=getattr(args, "remote_environment_receipt", None),
        fixed_kl_reference_model_path=args.fixed_kl_reference_model_path,
        fixed_kl_reference_receipt_path=args.fixed_kl_reference_receipt_path,
        fixed_null_receipt_path=getattr(args, "fixed_null_receipt", None),
        config=config,
        generation_config=generation,
        environment=environment,
    )
    run_dir = args.run_dir.expanduser().resolve()
    enforce_contract(run_dir, expected)
    return resolve_state(
        run_dir,
        task=args.task,
        method=args.method,
        final_step=args.final_step,
        world_size=args.world_size,
        contract=expected,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--sft-seed", type=int, default=42)
    parser.add_argument("--final-step", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--sft-data-path", type=Path, required=True)
    parser.add_argument(
        "--forbidden-source-registry",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model-initialization-registry",
        type=Path,
        required=True,
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--remote-environment-receipt", type=Path)
    parser.add_argument("--fixed-kl-reference-model-path", type=Path)
    parser.add_argument("--fixed-kl-reference-receipt-path", type=Path)
    parser.add_argument("--fixed-null-receipt", type=Path)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--generation-config", action="append", default=[])
    parser.add_argument("--allowed-env", action="append", default=[])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        print(prepare(args))
    except ValueError as exc:
        print(f"[SOTARunContractError] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
