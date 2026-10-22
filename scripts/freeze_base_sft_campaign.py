#!/usr/bin/env python3
"""Freeze the five-task Qwen2.5-7B-base specialist-SFT campaign.

This is deliberately separate from the historical Instruct-model specialist
launchers.  The receipt is created before any training output directory exists
and binds the immutable source snapshot, non-instruction base-model bytes, all
five registered SFT data trees, the exact training design, and the fresh output
root.  Workers reopen and revalidate those identities before every fresh or
resumed allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_lora_persistence import (  # type: ignore[import-not-found]
    EXPECTED_BIAS_PARAMETERS,
    EXPECTED_BIAS_TENSORS,
    LORA_BIAS_POLICY,
    audit_qwen25_base_bias_scope,
)
from audit_sft_token_lengths import (  # type: ignore[import-not-found]
    RECEIPT_FILENAME as TOKEN_LENGTH_RECEIPT_FILENAME,
)
from audit_sft_token_lengths import (
    token_length_audit_identity,
    validate_token_length_audit_identity,
    validate_token_length_receipt,
)
from build_clean_sft500k_corpus import (  # type: ignore[import-not-found]
    MERGED_RECEIPT_FILENAME as CLEANLINESS_RECEIPT_FILENAME,
)
from build_clean_sft500k_corpus import (
    implementation_identity as cleanliness_implementation_identity,
)
from build_clean_sft500k_corpus import (
    validate_merged_identity as validate_cleanliness_identity,
)
from build_clean_sft500k_corpus import validate_merged_receipt
from freeze_remote_environment import (  # type: ignore[import-not-found]
    CLAIM_SCOPE as ENVIRONMENT_CLAIM_SCOPE,
)
from freeze_remote_environment import (
    EXPECTED_GPU_COUNT as ENVIRONMENT_EXPECTED_GPU_COUNT,
)
from freeze_remote_environment import PROTOCOL as ENVIRONMENT_PROTOCOL
from freeze_remote_environment import SCHEMA_VERSION as ENVIRONMENT_SCHEMA_VERSION
from freeze_remote_environment import STATUS as ENVIRONMENT_STATUS
from freeze_remote_environment import environment_identity
from freeze_forbidden_source_registry import (  # type: ignore[import-not-found]
    BASE_SFT_PRE_IID_PROVENANCE_REASON,
    BASE_SFT_PROTOCOL as BASE_SFT_SOURCE_REGISTRY_PROTOCOL,
    BASE_SFT_SCHEMA_VERSION as BASE_SFT_SOURCE_REGISTRY_SCHEMA_VERSION,
    BASE_SFT_STATUS as BASE_SFT_SOURCE_REGISTRY_STATUS,
    BASE_SFT_TEMPORAL_CLAIM as BASE_SFT_SOURCE_REGISTRY_TEMPORAL_CLAIM,
    validate_base_sft_source_registry as validate_forbidden_source_registry,
)
from freeze_sft_reserved_source_registry import (  # type: ignore[import-not-found]
    LEGACY_IID_PROVENANCE_CLAIM,
    RESERVED_ROOT_NAME,
    SOURCE_CLASSES as RESERVED_SOURCE_CLASSES,
)
from freeze_sft_reserved_source_registry import (
    sources_by_task_class as reserved_sources_by_task_class,
)
from freeze_sft_reserved_source_registry import (
    validate_registry as validate_reserved_source_registry,
)
from freeze_sft_reserved_source_registry import (
    validate_registry_identity as validate_reserved_source_registry_identity,
)
from freeze_source_snapshot import (  # type: ignore[import-not-found]
    _directory_open_flags,
    _ExpectedDirectory,
    _ExpectedFile,
    _file_open_flags,
    _open_absolute_directory,
    _rename_noreplace_at,
    _require_canonical_directory_identity,
    _stat_identity,
    _verify_canonical_published_tree,
    _verify_sealed_tree,
)
from prepare_sota_run import (  # type: ignore[import-not-found]
    MATCHED_SFT_CONTRACT,
    QWEN25_7B_BASE_SIGNATURE,
    _registered_source_binding,
    _sha256_file,
    _source_file,
    _tree_fingerprint,
    _validate_tree_receipt,
    validate_qwen25_7b_base,
    validate_source_snapshot,
)

from clasp.models.jiang_f638_base import (
    audit_jiang_f638_base_snapshot,
    validate_jiang_f638_base_receipt,
)

SCHEMA_VERSION = 9
PROTOCOL = "clasp-jiang-f638-base-specialist-sft-campaign-v9"
STATUS = "frozen_before_training"
SEALED_PATH_PROTOCOL = "clasp-fd-relative-sealed-input-path-v1"
TASKS = ("cvrp", "jssp", "mis", "op", "tsp")
CAMPAIGN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
RUN_RELATIVE_TEMPLATE = "runs/{task}/seed42"
RUN_CONTRACT_PROTOCOL = "clasp-jiang-f638-base-specialist-sft-run-v9"
RUN_CONTRACT_STATUS = "created_before_training"
RUN_CONTRACT_AUTHORITY_PROTOCOL = "clasp-base-sft-run-contract-authority-v1"
RUN_CONTRACT_AUTHORITY_RECEIPT = "run_contract_authority.json"
MINIMUM_MATRIX_FREE_BYTES = 1_099_511_627_776
MINIMUM_CELL_FREE_BYTES = 214_748_364_800
REQUIRED_SOURCE_FILES = {
    "scripts/audit_lora_persistence.py",
    "scripts/audit_sft_reload_equivalence.py",
    "scripts/audit_sft_token_lengths.py",
    "scripts/base_sft_resume_fidelity.py",
    "scripts/base_sft_single_h100_resume_fidelity.py",
    "scripts/build_clean_sft500k_corpus.py",
    "scripts/freeze_base_sft_campaign.py",
    "scripts/freeze_benchmark.py",
    "scripts/freeze_forbidden_source_registry.py",
    "scripts/freeze_remote_environment.py",
    "scripts/freeze_sft_reserved_source_registry.py",
    "scripts/freeze_source_snapshot.py",
    "scripts/generate_clean_sft_replacements.py",
    "scripts/generate_clean_sft_replacements_a100.slurm",
    "scripts/generate_iid_benchmark.py",
    "scripts/run_jiang_f638_sft_in_sealed_environment.py",
    "scripts/seal_completed_sft_run.py",
    "scripts/run_reference_evidence.py",
    "scripts/smoke_jiang_f638_sft_environment_h100.slurm",
    "scripts/validate_base_sft500k_specialist.py",
    "scripts/train_sft500k_base_specialist_h100.slurm",
    "scripts/submit_sft500k_base_specialists_h100.sh",
    "requirements.jiang-f638-runtime-closure.txt",
    "requirements.jiang-f638-sft.txt",
    "src/clasp/data/benchmark.py",
    "src/clasp/data/loaders.py",
    "src/clasp/data/prompts.py",
    "src/clasp/entrypoints/evaluate.py",
    "src/clasp/entrypoints/train_sft_jiang_f638.py",
    "src/clasp/models/checkpoint_fidelity.py",
    "src/clasp/models/exact_peft_kl_reference.py",
    "src/clasp/models/jiang_f638_base.py",
    "src/clasp/models/loading.py",
    "src/clasp/models/publication_fidelity.py",
    "src/clasp/rewards/co_tasks.py",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _is_timestamp(value: object) -> bool:
    if not isinstance(value, str) or TIMESTAMP_RE.fullmatch(value) is None:
        return False
    try:
        return (
            datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%dT%H:%M:%SZ") == value
        )
    except ValueError:
        return False


def _canonical_path(path: Path, *, label: str, must_exist: bool) -> Path:
    raw = path.expanduser().absolute()
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{label} must be an absolute normalized path")
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {raw}")
    resolved = raw.resolve(strict=must_exist)
    if resolved != raw:
        raise ValueError(f"{label} must be canonical: supplied={raw}, resolved={resolved}")
    if must_exist and not resolved.exists():
        raise ValueError(f"{label} does not exist: {resolved}")
    return resolved


def _require_read_only_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a regular non-symlink file: {path}")
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise ValueError(f"{label} must be non-writable before training: {path}")
    if path.stat(follow_symlinks=False).st_nlink != 1:
        raise ValueError(f"{label} must be a single-link file: {path}")


def _require_read_only_tree(root: Path, *, label: str) -> None:
    raw = root.expanduser().absolute()
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError(f"{label} root is not a regular directory: {raw}")
    for candidate in [raw, *sorted(raw.rglob("*"), key=lambda item: item.as_posix())]:
        if candidate.is_symlink():
            raise ValueError(f"{label} contains a symbolic link: {candidate}")
        mode = candidate.stat(follow_symlinks=False).st_mode
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError(f"{label} contains a special entry: {candidate}")
        if stat.S_IMODE(mode) & 0o222:
            raise ValueError(f"{label} entry is writable: {candidate}")


def _stat_record(*, relative_path: str, value: os.stat_result) -> dict[str, Any]:
    mode = stat.S_IMODE(value.st_mode)
    if stat.S_ISDIR(value.st_mode):
        entry_type = "directory"
    elif stat.S_ISREG(value.st_mode):
        entry_type = "file"
    else:
        entry_type = "special"
    return {
        "relative_path": relative_path,
        "entry_type": entry_type,
        "mode": f"{mode:04o}",
        "device": value.st_dev,
        "inode": value.st_ino,
        "uid": value.st_uid,
        "gid": value.st_gid,
        "nlink": value.st_nlink,
    }


def sealed_input_path_identity(
    path: Path,
    *,
    trust_root: Path,
    label: str,
    expect_directory: bool,
) -> dict[str, Any]:
    """Bind a live input through a non-writable, fd-relative path chain.

    The explicitly supplied trust root is the declared mode-sealed boundary;
    this does not claim administrator-owned immutability. Every component from
    that root through the leaf is opened with ``O_NOFOLLOW`` and must be
    non-writable. Device/inode/mode evidence makes a later path swap visible
    when workers reopen the campaign immediately before use.
    """

    root = _canonical_path(trust_root, label="immutable input trust root", must_exist=True)
    target = _canonical_path(path, label=label, must_exist=True)
    if root == target or not target.is_relative_to(root):
        raise ValueError(f"{label} must be a strict descendant of immutable input trust root")
    relative = target.relative_to(root)
    if not relative.parts:
        raise ValueError(f"{label} has an empty trust-root-relative path")

    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root, os.O_RDONLY | directory_flag | nofollow_flag)
    opened = [root_fd]
    try:
        root_stat = os.fstat(root_fd)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError("immutable input trust root is not a directory")
        chain = [_stat_record(relative_path=".", value=root_stat)]
        if stat.S_IMODE(root_stat.st_mode) & 0o222:
            raise ValueError("immutable input trust root must be non-writable")
        parent_fd = root_fd
        for index, component in enumerate(relative.parts):
            is_leaf = index == len(relative.parts) - 1
            flags = os.O_RDONLY | nofollow_flag
            if not is_leaf or expect_directory:
                flags |= directory_flag
            elif is_leaf:
                flags |= getattr(os, "O_NONBLOCK", 0)
            try:
                child_fd = os.open(component, flags, dir_fd=parent_fd)
            except OSError as exc:
                raise ValueError(f"{label} path chain cannot be opened safely: {exc}") from exc
            opened.append(child_fd)
            child_stat = os.fstat(child_fd)
            if (
                (is_leaf and expect_directory and not stat.S_ISDIR(child_stat.st_mode))
                or (is_leaf and not expect_directory and not stat.S_ISREG(child_stat.st_mode))
                or (not is_leaf and not stat.S_ISDIR(child_stat.st_mode))
            ):
                raise ValueError(f"{label} path chain contains an unexpected entry type")
            if stat.S_IMODE(child_stat.st_mode) & 0o222:
                raise ValueError(f"{label} path chain contains a writable component: {component}")
            chain.append(
                _stat_record(
                    relative_path=Path(*relative.parts[: index + 1]).as_posix(),
                    value=child_stat,
                )
            )
            parent_fd = child_fd
        identity: dict[str, Any] = {
            "protocol": SEALED_PATH_PROTOCOL,
            "trust_root": str(root),
            "canonical_path": str(target),
            "relative_path": relative.as_posix(),
            "entry_type": "directory" if expect_directory else "file",
            "chain": chain,
        }
        identity["identity_sha256"] = sha256_json(identity)
        return identity
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def _validate_sealed_path_identity(value: object, *, label: str) -> dict[str, Any]:
    keys = {
        "protocol",
        "trust_root",
        "canonical_path",
        "relative_path",
        "entry_type",
        "chain",
        "identity_sha256",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} has an invalid sealed-path schema")
    unsigned = dict(value)
    stored_hash = unsigned.pop("identity_sha256", None)
    if stored_hash != sha256_json(unsigned):
        raise ValueError(f"{label} sealed-path self-hash mismatch")
    root = Path(str(value.get("trust_root", "")))
    target = Path(str(value.get("canonical_path", "")))
    relative = value.get("relative_path")
    entry_type = value.get("entry_type")
    chain = value.get("chain")
    if (
        value.get("protocol") != SEALED_PATH_PROTOCOL
        or not root.is_absolute()
        or not target.is_absolute()
        or not isinstance(relative, str)
        or relative in {"", "."}
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or target != root / relative
        or entry_type not in {"directory", "file"}
        or not isinstance(chain, list)
        or len(chain) != len(Path(relative).parts) + 1
    ):
        raise ValueError(f"{label} has invalid sealed-path values")
    record_keys = {
        "relative_path",
        "entry_type",
        "mode",
        "device",
        "inode",
        "uid",
        "gid",
        "nlink",
    }
    expected_paths = ["."] + [
        Path(*Path(relative).parts[:index]).as_posix()
        for index in range(1, len(Path(relative).parts) + 1)
    ]
    for index, record in enumerate(chain):
        if (
            not isinstance(record, dict)
            or set(record) != record_keys
            or record.get("relative_path") != expected_paths[index]
            or record.get("entry_type") != (entry_type if index == len(chain) - 1 else "directory")
            or not isinstance(record.get("mode"), str)
            or re.fullmatch(r"[0-7]{4}", record["mode"]) is None
            or int(record["mode"], 8) & 0o222
            or any(
                type(record.get(field)) is not int or record[field] < 0
                for field in ("device", "inode", "uid", "gid", "nlink")
            )
        ):
            raise ValueError(f"{label} has invalid sealed-path chain evidence")
    return value


def _portable_fingerprint(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "canonical_path": receipt["canonical_path"],
        "sha256": receipt["sha256"],
        "file_count": receipt["file_count"],
        "size_bytes": receipt["size_bytes"],
        "files": receipt["files"],
    }


def _source_snapshot_receipt(root: Path) -> dict[str, Any]:
    manifest = root / "SOURCE_MANIFEST.sha256"
    validate_source_snapshot(root, manifest)
    for relative in REQUIRED_SOURCE_FILES:
        if not (root / relative).is_file():
            raise ValueError(f"source snapshot lacks required base-SFT file: {relative}")
    tree = _tree_fingerprint(root, include=_source_file, label="base-SFT source snapshot")
    return {
        "canonical_path": str(root),
        "manifest_sha256": _sha256_file(manifest),
        "tree_sha256": tree["sha256"],
        "file_count": tree["file_count"],
        "size_bytes": tree["size_bytes"],
    }


def _compact_source_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "canonical_path": binding["canonical_path"],
        "source_identity_sha256": binding["source_identity_sha256"],
        "file_manifest_sha256": binding["file_manifest_sha256"],
        "rows": binding["rows"],
        "unique_instance_hashes": binding["unique_instance_hashes"],
    }


def _compact_base_sft_source_registry(
    *,
    registry_path: Path,
    registry_identity: Mapping[str, Any],
) -> dict[str, Any]:
    generation = registry_identity.get("iid_generation_receipt")
    if not isinstance(generation, Mapping):
        raise ValueError("base-SFT source registry lacks IID generation provenance")
    return {
        "canonical_path": str(registry_path),
        "file_sha256": _sha256_file(registry_path),
        "schema_version": registry_identity["schema_version"],
        "protocol": registry_identity["protocol"],
        "status": registry_identity["status"],
        "registry_sha256": registry_identity["registry_sha256"],
        "source_tree_manifest_sha256": registry_identity[
            "source_tree_manifest_sha256"
        ],
        "registry_identity_sha256": registry_identity["identity_sha256"],
        "temporal_claim": registry_identity["temporal_claim"],
        "iid_generation_receipt_sha256": generation["receipt_sha256"],
        "iid_generation_receipt_file_sha256": generation["sha256"],
        "iid_provenance_claim": generation["provenance_claim"],
        "pre_iid_forbidden_registry_identity": registry_identity[
            "pre_iid_forbidden_registry_identity"
        ],
        "pre_iid_provenance_reason": registry_identity[
            "pre_iid_provenance_reason"
        ],
    }


def _require_clean_reserved_source_equality(
    *,
    cleanliness_identity: Mapping[str, Any],
    reserved_registry_identity: Mapping[str, Any],
    tasks: Sequence[str] = TASKS,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Require the clean audit to use the authority registry class-for-class."""

    reserved = reserved_sources_by_task_class(reserved_registry_identity)
    clean_by_task = cleanliness_identity.get("task_reserved_source_manifest")
    if not isinstance(clean_by_task, Mapping):
        raise ValueError("SFT cleanliness audit has no reserved-source manifests")
    for task in tasks:
        clean_manifest = clean_by_task.get(task)
        if not isinstance(clean_manifest, list):
            raise ValueError(f"SFT cleanliness audit has no {task} reserved manifest")
        authoritative = []
        for source_class in RESERVED_SOURCE_CLASSES:
            record = reserved[task][source_class]
            authoritative.append(
                {
                    "source_class": source_class,
                    "canonical_path": record["canonical_path"],
                    "size_bytes": record["size_bytes"],
                    "sha256": record["sha256"],
                    "rows": record["rows"],
                }
            )
        if clean_manifest != authoritative:
            raise ValueError(
                f"{task} cleanliness reservation differs from the authoritative registry"
            )
    return reserved


def _require_registered_rl_reserved_sources(
    *,
    reserved: Mapping[str, Mapping[str, Mapping[str, Any]]],
    forbidden_registry_identity: dict[str, object],
    tasks: Sequence[str] = TASKS,
) -> None:
    """Cross-bind RL reservation classes to the existing pre-results registry."""

    label_by_class = {"rl_train": "rl-train", "rl_eval": "historical-eval"}
    for task in tasks:
        for source_class, source_label in label_by_class.items():
            record = reserved[task][source_class]
            binding = _registered_source_binding(
                registry_identity=forbidden_registry_identity,
                task=task,
                source_label=source_label,
                source_path=Path(str(record["canonical_path"])),
            )
            files = binding.get("files")
            if (
                not isinstance(files, list)
                or len(files) != 1
                or files[0].get("sha256") != record["sha256"]
                or files[0].get("size_bytes") != record["size_bytes"]
                or files[0].get("rows") != record["rows"]
            ):
                raise ValueError(
                    f"{task}/{source_class} does not equal registered {source_label} bytes"
                )


def training_design() -> dict[str, Any]:
    design = json.loads(json.dumps(MATCHED_SFT_CONTRACT))
    stage = design["stage_hyperparameters"]
    stage.update(
        {
            "dtype": "bfloat16",
            "per_device_eval_batch_size": 8,
            "min_eval_samples_per_task": 0,
            "numeric_token_loss_weight": 1.0,
            "checkpoint_fidelity_mode": "publication",
            "logging_steps": 1,
            "save_steps": 5_000,
            "eval_steps": None,
            "evaluation_strategy": "no",
            "save_total_limit": 50,
        }
    )
    design.update(
        {
            "base_model_requirement": {
                "model_family": "Qwen2.5-7B",
                "non_instruction": True,
                "upstream_repository": "unsloth/Qwen2.5-7B",
                "resolved_revision": "86e6e1eccea424c7fb77d59f7d2a9125c0673bfd",
                "revision_provenance": "explicit_reproducibility_pin_not_reported_by_paper",
                "signature": dict(QWEN25_7B_BASE_SIGNATURE),
            },
            "controller": {
                "chain_stages": 7,
                "stage_time_limit": "20:00:00",
                "gpus_per_job": 1,
                "dependency": "afterok_within_each_task",
                "chunk_stop_steps": [5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 31_250],
            },
            "run_relative_template": RUN_RELATIVE_TEMPLATE,
            "environment_exact": {
                "CLASP_PUBLICATION_FIDELITY": "1",
                "receipt_protocol": ENVIRONMENT_PROTOCOL,
                "receipt_claim_scope": ENVIRONMENT_CLAIM_SCOPE,
                "accelerator_count": ENVIRONMENT_EXPECTED_GPU_COUNT,
                "accelerator_family": "NVIDIA H100 compute capability 9.0",
                "requirements_lock": "requirements.jiang-f638-sft.txt",
                "requirements_lock_sha256": (
                    "b8c309502184dbb01f01c3d016c70d5d69d04ce03ec770ae7be720424a9a22b6"
                ),
                "full_python_prefix_tree_sha256": True,
                "requires_nonwritable_prefix_and_parent": True,
                "requires_live_revalidation_every_allocation": True,
            },
            "checkpoint_retention": {
                "save_total_limit": 50,
                "trainer_checkpoint_count": 7,
                "forced_terminal_checkpoint_step": 31_250,
                "resume_supported": True,
                "fresh_run_only": False,
                "resume_world_size": 1,
                "resume_boundaries": [
                    5_000,
                    10_000,
                    15_000,
                    20_000,
                    25_000,
                    30_000,
                    31_250,
                ],
                "resume_dependency": "afterok",
                "resume_receipt": "base_sft_single_h100_resume_fidelity.json",
                "resume_receipt_protocol": (
                    "clasp-publication-base-sft-single-h100-resume-state-v1"
                ),
                "exact_resume_receipt_every_checkpoint": True,
                "completed_run_seal_receipt": "completed_sft_run_seal.json",
                "requires_recursive_exact_modes_for_complete": True,
                "checkpoint_file_mode": "0440",
                "checkpoint_directory_mode": "0550",
                "minimum_matrix_free_bytes_before_submission": MINIMUM_MATRIX_FREE_BYTES,
                "minimum_cell_free_bytes_before_allocation": MINIMUM_CELL_FREE_BYTES,
            },
            "clean_corpus_audit": {
                "receipt_filename": CLEANLINESS_RECEIPT_FILENAME,
                "all_five_tasks": True,
                "post_iid_base_sft_source_registry": True,
                "source_registry_temporal_claim": BASE_SFT_SOURCE_REGISTRY_TEMPORAL_CLAIM,
                "legacy_iid_without_pre_iid_registry_binding": True,
                "legacy_iid_provenance_reason": BASE_SFT_PRE_IID_PROVENANCE_REASON,
                "authoritative_reserved_source_registry": True,
                "reserved_source_classes": list(RESERVED_SOURCE_CLASSES),
                "requires_exact_task_class_manifest_equality": True,
                "requires_rl_registry_cross_binding": True,
                "requires_iid_generation_receipt": True,
                "rows_per_task": 500_000,
                "requires_unique_demonstrations": True,
                "requires_zero_reserved_evaluation_overlap": True,
                "requires_strictly_valid_labels": True,
                "requires_registered_output_hash_equality": True,
            },
            "full_context_audit": {
                "receipt_filename": TOKEN_LENGTH_RECEIPT_FILENAME,
                "all_five_tasks": True,
                "rows_per_task": 500_000,
                "max_seq_length": 20_000,
                "prompt_style": "jiang_f638",
                "exact_prompt_bytes": True,
                "prompt_template_sha256": (
                    "0e4ff1b30b6b81b3ba4d84fd68c791f7a600dd470a2aae4dfe129fca8ffe6883"
                ),
                "tokenizer_json_sha256": (
                    "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa"
                ),
                "dataset_shuffle_passes": 1,
                "streaming_all_rows": True,
                "requires_zero_truncated_rows": True,
                "requires_zero_prompt_completion_alignment_mismatches": True,
                "requires_one_corrected_contextual_boundary_per_row": True,
                "response_boundary_policy": "supervise_colon_output_crossing_token",
                "requires_nonempty_supervised_completion": True,
            },
            "post_save_reload_equivalence": {
                "receipt_filename": "sft_reload_equivalence.json",
                "requires_pre_save_to_reloaded_trainable_tensor_equality": True,
                "requires_two_independent_full_state_loads": True,
                "requires_bitwise_fixed_probe_distributions": True,
            },
        }
    )
    return design


BASE_SFT_TRAINING_DESIGN = training_design()
BASE_SFT_TRAINING_DESIGN_SHA256 = sha256_json(BASE_SFT_TRAINING_DESIGN)


def build_campaign(
    *,
    campaign_id: str,
    run_contract_authority_root: Path,
    immutable_input_trust_root: Path,
    source_root: Path,
    base_model_path: Path,
    forbidden_source_registry: Path,
    reserved_source_registry: Path,
    remote_environment_receipt: Path,
    sft_data_root: Path,
    output_root: Path,
    frozen_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build and validate one pre-results five-task base-SFT receipt."""

    if CAMPAIGN_ID_RE.fullmatch(campaign_id) is None:
        raise ValueError("campaign_id must be a safe nonempty token")
    authority_root = _canonical_path(
        run_contract_authority_root,
        label="run-contract authority root",
        must_exist=False,
    )
    if authority_root.exists() or authority_root.is_symlink():
        raise ValueError("run-contract authority root must not exist before freezing")
    trust_root = _canonical_path(
        immutable_input_trust_root,
        label="immutable input trust root",
        must_exist=True,
    )
    source = _canonical_path(source_root, label="source root", must_exist=True)
    base = _canonical_path(base_model_path, label="base model", must_exist=True)
    registry_path = _canonical_path(
        forbidden_source_registry,
        label="forbidden-source registry",
        must_exist=True,
    )
    reserved_registry_path = _canonical_path(
        reserved_source_registry,
        label="SFT reserved-source registry",
        must_exist=True,
    )
    environment_path = _canonical_path(
        remote_environment_receipt,
        label="sealed remote-environment receipt",
        must_exist=True,
    )
    data_root = _canonical_path(sft_data_root, label="SFT data root", must_exist=True)
    output = _canonical_path(output_root, label="base-SFT output root", must_exist=False)
    if output.exists() or output.is_symlink():
        raise ValueError(f"base-SFT output root must not exist before freezing: {output}")
    if output == trust_root or output.is_relative_to(trust_root):
        raise ValueError("base-SFT output root must live outside the immutable input trust root")
    if authority_root == output or authority_root.is_relative_to(output):
        raise ValueError("run-contract authority must live outside the mutable output root")
    forbidden_targets = {Path("/"), Path.home().resolve(), source, base, data_root}
    if output in forbidden_targets:
        raise ValueError(f"unsafe or aliased base-SFT output root: {output}")
    for label, forbidden_root in (
        ("source snapshot", source),
        ("base model", base),
        ("SFT data", data_root),
    ):
        if environment_path == forbidden_root or environment_path.is_relative_to(forbidden_root):
            raise ValueError(
                f"sealed remote-environment receipt must live outside the {label} tree"
            )

    sealed_paths = {
        "source_snapshot": sealed_input_path_identity(
            source,
            trust_root=trust_root,
            label="source root",
            expect_directory=True,
        ),
        "base_model": sealed_input_path_identity(
            base,
            trust_root=trust_root,
            label="base model",
            expect_directory=True,
        ),
        "forbidden_source_registry": sealed_input_path_identity(
            registry_path,
            trust_root=trust_root,
            label="forbidden-source registry",
            expect_directory=False,
        ),
        "sft_reserved_source_registry": sealed_input_path_identity(
            reserved_registry_path,
            trust_root=trust_root,
            label="SFT reserved-source registry",
            expect_directory=False,
        ),
        "sft_data_root": sealed_input_path_identity(
            data_root,
            trust_root=trust_root,
            label="SFT data root",
            expect_directory=True,
        ),
    }
    source_receipt = _source_snapshot_receipt(source)
    environment_receipt = environment_identity(
        environment_path,
        require_accelerator=False,
    )
    _require_read_only_tree(base, label="base model")
    base_signature = validate_qwen25_7b_base(base)
    base_bias_scope = audit_qwen25_base_bias_scope(base)
    base_receipt = audit_jiang_f638_base_snapshot(base)

    _require_read_only_file(registry_path, label="forbidden-source registry")
    _require_read_only_tree(data_root, label="SFT data")
    _require_read_only_tree(
        trust_root / RESERVED_ROOT_NAME,
        label="SFT reserved-source tree",
    )
    try:
        registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read forbidden-source registry: {exc}") from exc
    registry_identity = validate_forbidden_source_registry(
        registry_payload,
        expected_tasks=list(TASKS),
    )
    reserved_registry_identity = validate_reserved_source_registry(
        reserved_registry_path,
        expected_trust_root=trust_root,
    )
    if reserved_registry_identity["iid_generation_receipt"] is None:
        raise ValueError("publication SFT reserved registry requires an IID generation receipt")
    if (
        registry_identity.get("iid_generation_receipt")
        != reserved_registry_identity["iid_generation_receipt"]
    ):
        raise ValueError(
            "post-IID base-SFT registry and reservation authority bind different IID provenance"
        )
    task_sources: dict[str, dict[str, Any]] = {}
    task_source_files: dict[str, list[dict[str, Any]]] = {}
    for task in TASKS:
        binding = _registered_source_binding(
            registry_identity=registry_identity,
            task=task,
            source_label="sft-train",
            source_path=data_root / task / "train",
        )
        if binding["rows"] != 500_000:
            raise ValueError(
                f"registered {task} SFT source has {binding['rows']} rows, expected 500000"
            )
        task_sources[task] = _compact_source_binding(binding)
        task_source_files[task] = list(binding["files"])

    cleanliness_receipt_path = data_root / CLEANLINESS_RECEIPT_FILENAME
    cleanliness_identity = validate_merged_receipt(
        cleanliness_receipt_path,
        expected_data_root=data_root,
    )
    live_cleanliness_implementation = cleanliness_implementation_identity()[
        "implementation_manifest_sha256"
    ]
    if (
        cleanliness_identity["implementation_manifest_sha256"]
        != live_cleanliness_implementation
    ):
        raise ValueError("SFT cleanliness audit used a different verifier implementation")
    reserved_by_task = _require_clean_reserved_source_equality(
        cleanliness_identity=cleanliness_identity,
        reserved_registry_identity=reserved_registry_identity,
    )
    _require_registered_rl_reserved_sources(
        reserved=reserved_by_task,
        forbidden_registry_identity=registry_identity,
    )
    for task in TASKS:
        files = task_source_files[task]
        if (
            len(files) != 1
            or files[0].get("rows") != 500_000
            or files[0].get("sha256")
            != cleanliness_identity["task_output_sha256"][task]
        ):
            raise ValueError(
                f"registered {task} SFT source is not the exact audited clean output"
            )

    token_receipt_path = data_root / TOKEN_LENGTH_RECEIPT_FILENAME
    token_receipt = validate_token_length_receipt(
        token_receipt_path,
        data_root=data_root,
        tokenizer_path=base,
        expected_sources=task_sources,
        expected_registry_identity=registry_identity,
    )
    token_length_identity = token_length_audit_identity(
        token_receipt,
        receipt_path=token_receipt_path,
    )

    timestamp = frozen_at_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not _is_timestamp(timestamp):
        raise ValueError("frozen_at_utc must use canonical YYYY-MM-DDTHH:MM:SSZ")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
        "campaign_id": campaign_id,
        "frozen_at_utc": timestamp,
        "tasks": list(TASKS),
        "run_contract_authority": {
            "protocol": RUN_CONTRACT_AUTHORITY_PROTOCOL,
            "canonical_path": str(authority_root),
            "campaign_filename": "base_sft_campaign.json",
            "receipt_filename": RUN_CONTRACT_AUTHORITY_RECEIPT,
            "contracts": {task: f"{task}.seed42.run_contract.json" for task in TASKS},
            "file_mode": "0440",
            "directory_mode": "0550",
        },
        "immutable_input_trust_root": str(trust_root),
        "sealed_input_paths": sealed_paths,
        "source_snapshot": source_receipt,
        "base_model_signature": base_signature,
        "base_bias_scope": base_bias_scope,
        "base_model": base_receipt,
        "forbidden_source_registry": _compact_base_sft_source_registry(
            registry_path=registry_path,
            registry_identity=registry_identity,
        ),
        "sft_reserved_source_registry": reserved_registry_identity,
        "remote_environment": environment_receipt,
        "sft_data_root": str(data_root),
        "sft_train_by_task": task_sources,
        "sft_cleanliness_audit": cleanliness_identity,
        "sft_token_length_audit": token_length_identity,
        "output_root": str(output),
        "run_relative_template": RUN_RELATIVE_TEMPLATE,
        "training_design": BASE_SFT_TRAINING_DESIGN,
        "training_design_sha256": BASE_SFT_TRAINING_DESIGN_SHA256,
    }
    payload["campaign_sha256"] = sha256_json(payload)
    validate_campaign(payload)
    return payload


def validate_campaign(payload: object) -> dict[str, Any]:
    """Validate the portable schema and self-hash without touching live paths."""

    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "campaign_id",
        "frozen_at_utc",
        "tasks",
        "run_contract_authority",
        "immutable_input_trust_root",
        "sealed_input_paths",
        "source_snapshot",
        "base_model_signature",
        "base_bias_scope",
        "base_model",
        "forbidden_source_registry",
        "sft_reserved_source_registry",
        "remote_environment",
        "sft_data_root",
        "sft_train_by_task",
        "sft_cleanliness_audit",
        "sft_token_length_audit",
        "output_root",
        "run_relative_template",
        "training_design",
        "training_design_sha256",
        "campaign_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("base-SFT campaign has an invalid top-level schema")
    unsigned = dict(payload)
    stored_hash = unsigned.pop("campaign_sha256", None)
    observed_hash = sha256_json(unsigned)
    if stored_hash != observed_hash:
        raise ValueError(
            "base-SFT campaign self-hash mismatch: "
            f"stored={stored_hash!r}, observed={observed_hash}"
        )
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("protocol") != PROTOCOL
        or payload.get("status") != STATUS
        or CAMPAIGN_ID_RE.fullmatch(str(payload.get("campaign_id", ""))) is None
        or not _is_timestamp(payload.get("frozen_at_utc"))
        or payload.get("tasks") != list(TASKS)
        or payload.get("run_relative_template") != RUN_RELATIVE_TEMPLATE
        or payload.get("base_model_signature") != QWEN25_7B_BASE_SIGNATURE
        or payload.get("training_design") != BASE_SFT_TRAINING_DESIGN
        or payload.get("training_design_sha256") != BASE_SFT_TRAINING_DESIGN_SHA256
    ):
        raise ValueError("base-SFT campaign disagrees with the frozen design")

    authority = payload.get("run_contract_authority")
    if (
        not isinstance(authority, dict)
        or set(authority)
        != {
            "protocol",
            "canonical_path",
            "campaign_filename",
            "receipt_filename",
            "contracts",
            "file_mode",
            "directory_mode",
        }
        or authority.get("protocol") != RUN_CONTRACT_AUTHORITY_PROTOCOL
        or not Path(str(authority.get("canonical_path", ""))).is_absolute()
        or authority.get("campaign_filename") != "base_sft_campaign.json"
        or authority.get("receipt_filename") != RUN_CONTRACT_AUTHORITY_RECEIPT
        or authority.get("contracts")
        != {task: f"{task}.seed42.run_contract.json" for task in TASKS}
        or authority.get("file_mode") != "0440"
        or authority.get("directory_mode") != "0550"
    ):
        raise ValueError("base-SFT campaign has an invalid run-contract authority")

    source = payload.get("source_snapshot")
    source_keys = {
        "canonical_path",
        "manifest_sha256",
        "tree_sha256",
        "file_count",
        "size_bytes",
    }
    if (
        not isinstance(source, dict)
        or set(source) != source_keys
        or not Path(str(source.get("canonical_path", ""))).is_absolute()
        or not _is_sha256(source.get("manifest_sha256"))
        or not _is_sha256(source.get("tree_sha256"))
        or isinstance(source.get("file_count"), bool)
        or not isinstance(source.get("file_count"), int)
        or source["file_count"] <= 0
        or isinstance(source.get("size_bytes"), bool)
        or not isinstance(source.get("size_bytes"), int)
        or source["size_bytes"] <= 0
    ):
        raise ValueError("base-SFT campaign has an invalid source-snapshot receipt")

    trust_root = payload.get("immutable_input_trust_root")
    sealed_paths = payload.get("sealed_input_paths")
    if (
        not isinstance(trust_root, str)
        or not Path(trust_root).is_absolute()
        or ".." in Path(trust_root).parts
        or not isinstance(sealed_paths, dict)
        or set(sealed_paths)
        != {
            "source_snapshot",
            "base_model",
            "forbidden_source_registry",
            "sft_reserved_source_registry",
            "sft_data_root",
        }
    ):
        raise ValueError("base-SFT campaign has an invalid immutable-input trust boundary")
    expected_path_values = {
        "source_snapshot": (source["canonical_path"], "directory"),
        "base_model": (
            payload.get("base_model", {}).get("canonical_path")
            if isinstance(payload.get("base_model"), dict)
            else None,
            "directory",
        ),
        "forbidden_source_registry": (
            payload.get("forbidden_source_registry", {}).get("canonical_path")
            if isinstance(payload.get("forbidden_source_registry"), dict)
            else None,
            "file",
        ),
        "sft_reserved_source_registry": (
            payload.get("sft_reserved_source_registry", {}).get("canonical_path")
            if isinstance(payload.get("sft_reserved_source_registry"), dict)
            else None,
            "file",
        ),
        "sft_data_root": (payload["sft_data_root"], "directory"),
    }
    for role, (expected_path, expected_type) in expected_path_values.items():
        identity = _validate_sealed_path_identity(
            sealed_paths[role],
            label=f"campaign {role}",
        )
        if (
            identity["trust_root"] != trust_root
            or identity["canonical_path"] != expected_path
            or identity["entry_type"] != expected_type
        ):
            raise ValueError(f"campaign {role} sealed path disagrees with bound input")

    base = _validate_tree_receipt(payload.get("base_model"), label="campaign base model")
    validate_jiang_f638_base_receipt(base)
    if not Path(str(base.get("canonical_path", ""))).is_absolute():
        raise ValueError("campaign base-model path is not absolute")

    base_bias_scope = payload.get("base_bias_scope")
    expected_bias_scope_keys = {
        "policy",
        "scope_equivalence",
        "bias_tensor_count",
        "bias_parameter_count",
        "bias_keys_sha256",
        "base_weight_index_sha256",
    }
    if (
        not isinstance(base_bias_scope, dict)
        or set(base_bias_scope) != expected_bias_scope_keys
        or base_bias_scope.get("policy") != LORA_BIAS_POLICY
        or base_bias_scope.get("scope_equivalence") != "all_base_biases_equal_targeted_qkv_biases"
        or base_bias_scope.get("bias_tensor_count") != EXPECTED_BIAS_TENSORS
        or base_bias_scope.get("bias_parameter_count") != EXPECTED_BIAS_PARAMETERS
        or not _is_sha256(base_bias_scope.get("bias_keys_sha256"))
        or not _is_sha256(base_bias_scope.get("base_weight_index_sha256"))
    ):
        raise ValueError("base-SFT campaign has an invalid base-bias scope receipt")

    registry = payload.get("forbidden_source_registry")
    registry_keys = {
        "canonical_path",
        "file_sha256",
        "schema_version",
        "protocol",
        "status",
        "registry_sha256",
        "source_tree_manifest_sha256",
        "registry_identity_sha256",
        "temporal_claim",
        "iid_generation_receipt_sha256",
        "iid_generation_receipt_file_sha256",
        "iid_provenance_claim",
        "pre_iid_forbidden_registry_identity",
        "pre_iid_provenance_reason",
    }
    if (
        not isinstance(registry, dict)
        or set(registry) != registry_keys
        or not Path(str(registry.get("canonical_path", ""))).is_absolute()
        or any(
            not _is_sha256(registry.get(field))
            for field in (
                "file_sha256",
                "registry_sha256",
                "source_tree_manifest_sha256",
                "registry_identity_sha256",
                "iid_generation_receipt_sha256",
                "iid_generation_receipt_file_sha256",
            )
        )
        or registry.get("schema_version") != BASE_SFT_SOURCE_REGISTRY_SCHEMA_VERSION
        or registry.get("protocol") != BASE_SFT_SOURCE_REGISTRY_PROTOCOL
        or registry.get("status") != BASE_SFT_SOURCE_REGISTRY_STATUS
        or registry.get("temporal_claim") != BASE_SFT_SOURCE_REGISTRY_TEMPORAL_CLAIM
        or registry.get("iid_provenance_claim") != LEGACY_IID_PROVENANCE_CLAIM
        or registry.get("pre_iid_forbidden_registry_identity") is not None
        or registry.get("pre_iid_provenance_reason")
        != BASE_SFT_PRE_IID_PROVENANCE_REASON
    ):
        raise ValueError("base-SFT campaign has an invalid data-registry receipt")
    try:
        reserved_registry_identity = validate_reserved_source_registry_identity(
            payload.get("sft_reserved_source_registry")
        )
    except ValueError as exc:
        raise ValueError(
            f"base-SFT campaign has an invalid reserved-source registry: {exc}"
        ) from exc
    if reserved_registry_identity["immutable_input_trust_root"] != trust_root:
        raise ValueError("reserved-source registry belongs to another trust root")
    if reserved_registry_identity["iid_generation_receipt"] is None:
        raise ValueError("base-SFT campaign lacks IID generation-receipt provenance")
    generation = reserved_registry_identity["iid_generation_receipt"]
    if (
        registry["iid_generation_receipt_sha256"] != generation["receipt_sha256"]
        or registry["iid_generation_receipt_file_sha256"] != generation["sha256"]
        or registry["iid_provenance_claim"] != generation["provenance_claim"]
    ):
        raise ValueError("base-SFT registries disagree on retained IID provenance")

    environment = payload.get("remote_environment")
    environment_keys = {
        "schema_version",
        "protocol",
        "status",
        "claim_scope",
        "canonical_path",
        "receipt_sha256",
        "receipt_file_sha256",
        "producer_implementation_sha256",
        "python_sha256",
        "packages_sha256",
        "runtime_sha256",
        "accelerator_sha256",
        "identity_sha256",
    }
    if (
        not isinstance(environment, dict)
        or set(environment) != environment_keys
        or type(environment.get("schema_version")) is not int
        or environment.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION
        or environment.get("protocol") != ENVIRONMENT_PROTOCOL
        or environment.get("status") != ENVIRONMENT_STATUS
        or environment.get("claim_scope") != ENVIRONMENT_CLAIM_SCOPE
        or not Path(str(environment.get("canonical_path", ""))).is_absolute()
        or any(
            not _is_sha256(environment.get(field))
            for field in environment_keys
            - {
                "schema_version",
                "protocol",
                "status",
                "claim_scope",
                "canonical_path",
            }
        )
    ):
        raise ValueError("base-SFT campaign has an invalid remote-environment identity")
    unsigned_environment = dict(environment)
    stored_environment_identity = unsigned_environment.pop("identity_sha256")
    if stored_environment_identity != sha256_json(unsigned_environment):
        raise ValueError("base-SFT campaign remote-environment identity self-hash mismatch")

    for field in ("sft_data_root", "output_root"):
        value = payload.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError(f"base-SFT campaign {field} is not an absolute normalized path")
    sources = payload.get("sft_train_by_task")
    if not isinstance(sources, dict) or tuple(sources) != TASKS:
        raise ValueError("base-SFT campaign must bind all five sorted SFT sources")
    source_keys = {
        "canonical_path",
        "source_identity_sha256",
        "file_manifest_sha256",
        "rows",
        "unique_instance_hashes",
    }
    for task in TASKS:
        binding = sources.get(task)
        expected_path = Path(payload["sft_data_root"]) / task / "train"
        if (
            not isinstance(binding, dict)
            or set(binding) != source_keys
            or binding.get("canonical_path") != str(expected_path)
            or not _is_sha256(binding.get("source_identity_sha256"))
            or not _is_sha256(binding.get("file_manifest_sha256"))
            or binding.get("rows") != 500_000
            or isinstance(binding.get("unique_instance_hashes"), bool)
            or not isinstance(binding.get("unique_instance_hashes"), int)
            or binding["unique_instance_hashes"] <= 0
        ):
            raise ValueError(f"base-SFT campaign has an invalid {task} source binding")
    try:
        cleanliness_identity = validate_cleanliness_identity(
            payload.get("sft_cleanliness_audit")
        )
    except ValueError as exc:
        raise ValueError(f"base-SFT campaign has an invalid cleanliness audit: {exc}") from exc
    if (
        cleanliness_identity["data_root"] != payload["sft_data_root"]
        or cleanliness_identity["rows_per_task"] != 500_000
        or cleanliness_identity["total_rows"] != 2_500_000
        or any(
            cleanliness_identity["task_unique_demonstrations"][task] != 500_000
            for task in TASKS
        )
    ):
        raise ValueError("base-SFT campaign cleanliness audit disagrees with training")
    _require_clean_reserved_source_equality(
        cleanliness_identity=cleanliness_identity,
        reserved_registry_identity=reserved_registry_identity,
    )
    try:
        token_length_identity = validate_token_length_audit_identity(
            payload.get("sft_token_length_audit")
        )
    except ValueError as exc:
        raise ValueError(f"base-SFT campaign has an invalid token-length audit: {exc}") from exc
    if (
        token_length_identity["rows_per_task"] != 500_000
        or token_length_identity["total_rows"] != 2_500_000
        or token_length_identity["max_seq_length"]
        != BASE_SFT_TRAINING_DESIGN["stage_hyperparameters"]["max_seq_length"]
    ):
        raise ValueError("base-SFT campaign token-length audit disagrees with training")
    return payload


def expected_run_contract(
    *,
    campaign: Mapping[str, Any],
    campaign_file_sha256: str,
    task: str,
) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"invalid base-SFT task for run contract: {task}")
    run_dir = Path(campaign["output_root"]) / campaign["run_relative_template"].format(task=task)
    authority_path = (
        Path(campaign["run_contract_authority"]["canonical_path"])
        / campaign["run_contract_authority"]["contracts"][task]
    )
    source = campaign["sft_train_by_task"][task]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": RUN_CONTRACT_PROTOCOL,
        "status": RUN_CONTRACT_STATUS,
        "created_at_utc": campaign["frozen_at_utc"],
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": campaign["campaign_sha256"],
        "campaign_file_sha256": campaign_file_sha256,
        "training_design_sha256": BASE_SFT_TRAINING_DESIGN_SHA256,
        "task": task,
        "sft_seed": 42,
        "run_dir": str(run_dir),
        "authoritative_path": str(authority_path),
        "immutable_input_trust_root": campaign["immutable_input_trust_root"],
        "sealed_input_path_identity_sha256": {
            role: identity["identity_sha256"]
            for role, identity in campaign["sealed_input_paths"].items()
        },
        "source_snapshot_tree_sha256": campaign["source_snapshot"]["tree_sha256"],
        "base_model_tree_sha256": campaign["base_model"]["sha256"],
        "sft_cleanliness_audit_identity_sha256": campaign["sft_cleanliness_audit"][
            "identity_sha256"
        ],
        "sft_token_length_audit_identity_sha256": campaign["sft_token_length_audit"][
            "identity_sha256"
        ],
        "forbidden_source_registry_sha256": campaign["forbidden_source_registry"][
            "registry_sha256"
        ],
        "sft_reserved_source_registry_identity_sha256": campaign[
            "sft_reserved_source_registry"
        ]["identity_sha256"],
        "remote_environment_identity_sha256": campaign["remote_environment"]["identity_sha256"],
        "remote_environment_receipt_sha256": campaign["remote_environment"]["receipt_sha256"],
        "remote_environment_receipt_file_sha256": campaign["remote_environment"][
            "receipt_file_sha256"
        ],
        "sft_train_source_identity_sha256": source["source_identity_sha256"],
        "sft_train_file_manifest_sha256": source["file_manifest_sha256"],
    }
    payload["run_contract_sha256"] = sha256_json(payload)
    return payload


def _authority_receipt(
    *,
    campaign: Mapping[str, Any],
    campaign_file_sha256: str,
    contracts: Mapping[str, Mapping[str, Any]],
    parent_stat: os.stat_result,
    root_stat: os.stat_result,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "protocol": RUN_CONTRACT_AUTHORITY_PROTOCOL,
        "status": "sealed_before_submission",
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": campaign["campaign_sha256"],
        "campaign_file_sha256": campaign_file_sha256,
        "publication_identity": {
            "parent_device": parent_stat.st_dev,
            "parent_inode": parent_stat.st_ino,
            "root_device": root_stat.st_dev,
            "root_inode": root_stat.st_ino,
        },
        "contracts": {
            task: {
                "filename": campaign["run_contract_authority"]["contracts"][task],
                "run_contract_sha256": contracts[task]["run_contract_sha256"],
                "file_sha256": hashlib.sha256(
                    json.dumps(contracts[task], indent=2, sort_keys=True).encode("utf-8") + b"\n"
                ).hexdigest(),
            }
            for task in TASKS
        },
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    return receipt


def load_run_contract_authority(
    campaign: Mapping[str, Any],
    *,
    campaign_file_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    authority = campaign["run_contract_authority"]
    root = _canonical_path(
        Path(authority["canonical_path"]),
        label="run-contract authority root",
        must_exist=True,
    )
    expected_names = {
        authority["campaign_filename"],
        authority["receipt_filename"],
        *authority["contracts"].values(),
    }
    parent_fd = _open_absolute_directory(root.parent)
    try:
        parent_stat = os.fstat(parent_fd)
        root_fd = os.open(root.name, _directory_open_flags(), dir_fd=parent_fd)
        try:
            root_stat = os.fstat(root_fd)
            if stat.S_IMODE(root_stat.st_mode) != 0o550:
                raise ValueError("run-contract authority root must have exact mode 0550")
            with os.scandir(root_fd) as iterator:
                names = {entry.name for entry in iterator}
            if names != expected_names:
                raise ValueError(
                    "run-contract authority entry set mismatch: "
                    f"missing={sorted(expected_names - names)}, "
                    f"extras={sorted(names - expected_names)}"
                )
            contents: dict[str, bytes] = {}
            expected_files: dict[str, _ExpectedFile] = {}
            for name in sorted(expected_names):
                descriptor = os.open(name, _file_open_flags(), dir_fd=root_fd)
                try:
                    before = os.fstat(descriptor)
                    if (
                        not stat.S_ISREG(before.st_mode)
                        or stat.S_IMODE(before.st_mode) != 0o440
                        or before.st_nlink != 1
                    ):
                        raise ValueError(f"authority entry is not exact 0440 regular file: {name}")
                    blocks: list[bytes] = []
                    while True:
                        block = os.read(descriptor, 1024 * 1024)
                        if not block:
                            break
                        blocks.append(block)
                    after = os.fstat(descriptor)
                    stable_fields = (
                        "st_dev",
                        "st_ino",
                        "st_mode",
                        "st_nlink",
                        "st_size",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                    if any(
                        getattr(before, field) != getattr(after, field) for field in stable_fields
                    ):
                        raise ValueError(f"authority entry changed while read: {name}")
                    content = b"".join(blocks)
                    contents[name] = content
                    expected_files[name] = _ExpectedFile(
                        sha256=hashlib.sha256(content).hexdigest(),
                        size_bytes=after.st_size,
                        device=after.st_dev,
                        inode=after.st_ino,
                        mtime_ns=after.st_mtime_ns,
                    )
                finally:
                    os.close(descriptor)
            directories = {
                "": _ExpectedDirectory(
                    device=root_stat.st_dev,
                    inode=root_stat.st_ino,
                    mode=0o550,
                )
            }
            _verify_sealed_tree(
                root_descriptor=root_fd,
                files=expected_files,
                directories=directories,
            )
        finally:
            os.close(root_fd)
    finally:
        os.close(parent_fd)
    _verify_canonical_published_tree(
        final=root,
        parent_identity=_stat_identity(parent_stat),
        root_identity=_stat_identity(root_stat),
        files=expected_files,
        directories=directories,
    )
    campaign_bytes = contents[authority["campaign_filename"]]
    if hashlib.sha256(campaign_bytes).hexdigest() != campaign_file_sha256:
        raise ValueError("authority campaign bytes differ from requested campaign")
    try:
        authority_campaign = json.loads(campaign_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse authority campaign: {exc}") from exc
    if authority_campaign != campaign:
        raise ValueError("authority campaign payload differs from live campaign")
    contracts: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        try:
            contract = json.loads(contents[authority["contracts"][task]].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read {task} authoritative run contract: {exc}") from exc
        expected = expected_run_contract(
            campaign=campaign,
            campaign_file_sha256=campaign_file_sha256,
            task=task,
        )
        if contract != expected:
            raise ValueError(f"{task} authoritative run contract differs from campaign")
        contracts[task] = contract
    try:
        receipt = json.loads(contents[authority["receipt_filename"]].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read run-contract authority receipt: {exc}") from exc
    expected_receipt = _authority_receipt(
        campaign=campaign,
        campaign_file_sha256=campaign_file_sha256,
        contracts=contracts,
        parent_stat=parent_stat,
        root_stat=root_stat,
    )
    if receipt != expected_receipt:
        raise ValueError("run-contract authority receipt differs from sealed contracts")
    return contracts, receipt


def load_campaign(path: Path) -> tuple[dict[str, Any], str]:
    contract = _canonical_path(path, label="base-SFT campaign contract", must_exist=True)
    _require_read_only_file(contract, label="base-SFT campaign contract")
    try:
        payload = json.loads(contract.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read base-SFT campaign contract: {exc}") from exc
    validated = validate_campaign(payload)
    file_sha256 = _sha256_file(contract)
    expected_root = Path(validated["run_contract_authority"]["canonical_path"])
    if contract.parent != expected_root or contract.name != "base_sft_campaign.json":
        raise ValueError("base-SFT campaign is outside its sealed run-contract authority")
    load_run_contract_authority(validated, campaign_file_sha256=file_sha256)
    return validated, file_sha256


def verify_campaign_runtime(
    *,
    campaign_path: Path,
    immutable_input_trust_root: Path | None = None,
    source_root: Path,
    base_model_path: Path,
    forbidden_source_registry: Path,
    reserved_source_registry: Path | None = None,
    remote_environment_receipt: Path | None = None,
    sft_data_root: Path,
    output_root: Path,
    tasks: Sequence[str] = TASKS,
    require_output_absent: bool,
    require_accelerator: bool = False,
    minimum_free_bytes: int = 0,
) -> tuple[dict[str, Any], str]:
    """Reopen every live input and compare it with the frozen campaign."""

    payload, campaign_file_sha256 = load_campaign(campaign_path)
    requested_tasks = tuple(tasks)
    if not requested_tasks or any(task not in TASKS for task in requested_tasks):
        raise ValueError("runtime task selection is invalid")
    if len(requested_tasks) != len(set(requested_tasks)):
        raise ValueError("runtime task selection contains duplicates")

    trust_root = _canonical_path(
        immutable_input_trust_root
        if immutable_input_trust_root is not None
        else Path(payload["immutable_input_trust_root"]),
        label="runtime immutable input trust root",
        must_exist=True,
    )
    source = _canonical_path(source_root, label="runtime source root", must_exist=True)
    base = _canonical_path(base_model_path, label="runtime base model", must_exist=True)
    registry_path = _canonical_path(
        forbidden_source_registry,
        label="runtime forbidden-source registry",
        must_exist=True,
    )
    reserved_registry_path = _canonical_path(
        reserved_source_registry
        if reserved_source_registry is not None
        else Path(payload["sft_reserved_source_registry"]["canonical_path"]),
        label="runtime SFT reserved-source registry",
        must_exist=True,
    )
    environment_path = _canonical_path(
        remote_environment_receipt
        if remote_environment_receipt is not None
        else Path(payload["remote_environment"]["canonical_path"]),
        label="runtime sealed remote-environment receipt",
        must_exist=True,
    )
    data_root = _canonical_path(sft_data_root, label="runtime SFT data root", must_exist=True)
    output = _canonical_path(output_root, label="runtime base-SFT output root", must_exist=False)
    exact_paths = {
        "immutable input trust root": (
            str(trust_root),
            payload["immutable_input_trust_root"],
        ),
        "source root": (str(source), payload["source_snapshot"]["canonical_path"]),
        "base model": (str(base), payload["base_model"]["canonical_path"]),
        "forbidden-source registry": (
            str(registry_path),
            payload["forbidden_source_registry"]["canonical_path"],
        ),
        "SFT reserved-source registry": (
            str(reserved_registry_path),
            payload["sft_reserved_source_registry"]["canonical_path"],
        ),
        "sealed remote environment": (
            str(environment_path),
            payload["remote_environment"]["canonical_path"],
        ),
        "SFT data root": (str(data_root), payload["sft_data_root"]),
        "output root": (str(output), payload["output_root"]),
    }
    mismatches = [
        f"{label}: actual={actual}, frozen={expected}"
        for label, (actual, expected) in exact_paths.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError("runtime paths disagree with base-SFT campaign: " + "; ".join(mismatches))

    actual_sealed_paths = {
        "source_snapshot": sealed_input_path_identity(
            source,
            trust_root=trust_root,
            label="runtime source root",
            expect_directory=True,
        ),
        "base_model": sealed_input_path_identity(
            base,
            trust_root=trust_root,
            label="runtime base model",
            expect_directory=True,
        ),
        "forbidden_source_registry": sealed_input_path_identity(
            registry_path,
            trust_root=trust_root,
            label="runtime forbidden-source registry",
            expect_directory=False,
        ),
        "sft_reserved_source_registry": sealed_input_path_identity(
            reserved_registry_path,
            trust_root=trust_root,
            label="runtime SFT reserved-source registry",
            expect_directory=False,
        ),
        "sft_data_root": sealed_input_path_identity(
            data_root,
            trust_root=trust_root,
            label="runtime SFT data root",
            expect_directory=True,
        ),
    }
    if actual_sealed_paths != payload["sealed_input_paths"]:
        raise ValueError("runtime sealed input path identity differs from the campaign")

    actual_environment = environment_identity(
        environment_path,
        require_accelerator=require_accelerator,
    )
    if actual_environment != payload["remote_environment"]:
        raise ValueError("runtime sealed remote environment differs from the campaign")

    source_receipt = _source_snapshot_receipt(source)
    if source_receipt != payload["source_snapshot"]:
        raise ValueError("runtime source snapshot differs from the base-SFT campaign")
    _require_read_only_tree(base, label="runtime base model")
    if validate_qwen25_7b_base(base) != payload["base_model_signature"]:
        raise ValueError("runtime base-model signature differs from the campaign")
    if audit_qwen25_base_bias_scope(base) != payload["base_bias_scope"]:
        raise ValueError("runtime base-model bias scope differs from the campaign")
    actual_base = audit_jiang_f638_base_snapshot(base)
    if _portable_fingerprint(actual_base) != _portable_fingerprint(payload["base_model"]):
        raise ValueError("runtime base-model bytes differ from the base-SFT campaign")

    _require_read_only_file(registry_path, label="runtime forbidden-source registry")
    _require_read_only_tree(data_root, label="runtime SFT data")
    _require_read_only_tree(
        trust_root / RESERVED_ROOT_NAME,
        label="runtime SFT reserved-source tree",
    )
    if _sha256_file(registry_path) != payload["forbidden_source_registry"]["file_sha256"]:
        raise ValueError("runtime forbidden-source registry bytes differ from the campaign")
    try:
        registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read runtime forbidden-source registry: {exc}") from exc
    registry_identity = validate_forbidden_source_registry(
        registry_payload,
        expected_tasks=list(TASKS),
    )
    if _compact_base_sft_source_registry(
        registry_path=registry_path,
        registry_identity=registry_identity,
    ) != payload["forbidden_source_registry"]:
        raise ValueError("runtime forbidden-source registry identity differs from the campaign")
    live_reserved_registry_identity = validate_reserved_source_registry(
        reserved_registry_path,
        expected_trust_root=trust_root,
    )
    if live_reserved_registry_identity != payload["sft_reserved_source_registry"]:
        raise ValueError("runtime SFT reserved-source registry differs from the campaign")
    if (
        registry_identity.get("iid_generation_receipt")
        != live_reserved_registry_identity["iid_generation_receipt"]
    ):
        raise ValueError("runtime base-SFT registries disagree on IID provenance")
    runtime_task_source_files: dict[str, list[dict[str, Any]]] = {}
    for task in requested_tasks:
        actual_binding = _registered_source_binding(
            registry_identity=registry_identity,
            task=task,
            source_label="sft-train",
            source_path=data_root / task / "train",
        )
        if _compact_source_binding(actual_binding) != payload["sft_train_by_task"][task]:
            raise ValueError(f"runtime {task} SFT source differs from the campaign")
        runtime_task_source_files[task] = list(actual_binding["files"])

    live_cleanliness_receipt_path = data_root / CLEANLINESS_RECEIPT_FILENAME
    live_cleanliness_identity = validate_merged_receipt(
        live_cleanliness_receipt_path,
        expected_data_root=data_root,
    )
    if live_cleanliness_identity != payload["sft_cleanliness_audit"]:
        raise ValueError("runtime SFT cleanliness audit differs from the campaign")
    if (
        live_cleanliness_identity["implementation_manifest_sha256"]
        != cleanliness_implementation_identity()["implementation_manifest_sha256"]
    ):
        raise ValueError("runtime SFT cleanliness verifier differs from the campaign")
    live_reserved_by_task = _require_clean_reserved_source_equality(
        cleanliness_identity=live_cleanliness_identity,
        reserved_registry_identity=live_reserved_registry_identity,
        tasks=requested_tasks,
    )
    _require_registered_rl_reserved_sources(
        reserved=live_reserved_by_task,
        forbidden_registry_identity=registry_identity,
        tasks=requested_tasks,
    )
    for task in requested_tasks:
        files = runtime_task_source_files[task]
        if (
            len(files) != 1
            or files[0].get("rows") != 500_000
            or files[0].get("sha256")
            != live_cleanliness_identity["task_output_sha256"][task]
        ):
            raise ValueError(f"runtime {task} SFT source is not the audited clean output")

    live_token_receipt_path = data_root / TOKEN_LENGTH_RECEIPT_FILENAME
    live_token_receipt = validate_token_length_receipt(
        live_token_receipt_path,
        data_root=data_root,
        tokenizer_path=base,
        expected_sources=payload["sft_train_by_task"],
        expected_registry_identity=payload["forbidden_source_registry"],
    )
    live_token_identity = token_length_audit_identity(
        live_token_receipt,
        receipt_path=live_token_receipt_path,
    )
    if live_token_identity != payload["sft_token_length_audit"]:
        raise ValueError("runtime all-row token-length audit differs from the campaign")

    if require_output_absent:
        if output.exists() or output.is_symlink():
            raise ValueError(f"fresh base-SFT output root already exists: {output}")
    elif output.is_symlink() or not output.is_dir():
        raise ValueError(f"base-SFT output root is not a regular directory: {output}")
    if type(minimum_free_bytes) is not int or minimum_free_bytes < 0:
        raise ValueError("minimum_free_bytes must be a nonnegative plain integer")
    if minimum_free_bytes:
        probe = output
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        if probe.is_symlink() or not probe.is_dir():
            raise ValueError("cannot identify a safe output filesystem for quota preflight")
        filesystem = os.statvfs(probe)
        available = int(filesystem.f_bavail) * int(filesystem.f_frsize)
        if available < minimum_free_bytes:
            raise ValueError(
                "base-SFT output filesystem lacks retained-checkpoint capacity: "
                f"available={available}, required={minimum_free_bytes}"
            )
    return payload, campaign_file_sha256


def _write_create_once(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    authority_root = Path(str(payload["run_contract_authority"]["canonical_path"]))
    if destination != authority_root / "base_sft_campaign.json":
        raise ValueError("campaign output must be the bound authority campaign path")
    parent_fd = _open_absolute_directory(authority_root.parent)
    staging_name = f".{authority_root.name}.{secrets.token_hex(12)}"
    published = False
    staging_identity: tuple[int, int] | None = None
    try:
        parent_stat = os.fstat(parent_fd)
        try:
            os.stat(authority_root.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite run-contract authority: {authority_root}")
        os.mkdir(staging_name, mode=0o750, dir_fd=parent_fd)
        staging_fd = os.open(staging_name, _directory_open_flags(), dir_fd=parent_fd)
        root_stat = os.fstat(staging_fd)
        staging_identity = _stat_identity(root_stat)
        campaign_bytes = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        campaign_file_sha256 = hashlib.sha256(campaign_bytes).hexdigest()
        contracts = {
            task: expected_run_contract(
                campaign=payload,
                campaign_file_sha256=campaign_file_sha256,
                task=task,
            )
            for task in TASKS
        }
        files: dict[str, bytes] = {
            "base_sft_campaign.json": campaign_bytes,
            **{
                payload["run_contract_authority"]["contracts"][task]: (
                    json.dumps(contracts[task], indent=2, sort_keys=True).encode("utf-8") + b"\n"
                )
                for task in TASKS
            },
        }
        receipt = _authority_receipt(
            campaign=payload,
            campaign_file_sha256=campaign_file_sha256,
            contracts=contracts,
            parent_stat=parent_stat,
            root_stat=root_stat,
        )
        files[RUN_CONTRACT_AUTHORITY_RECEIPT] = (
            json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        expected_files: dict[str, _ExpectedFile] = {}
        try:
            for name, content in sorted(files.items()):
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o640,
                    dir_fd=staging_fd,
                )
                try:
                    offset = 0
                    while offset < len(content):
                        offset += os.write(descriptor, content[offset:])
                    os.fsync(descriptor)
                    os.fchmod(descriptor, 0o440)
                    os.fsync(descriptor)
                    observed = os.fstat(descriptor)
                    expected_files[name] = _ExpectedFile(
                        sha256=hashlib.sha256(content).hexdigest(),
                        size_bytes=observed.st_size,
                        device=observed.st_dev,
                        inode=observed.st_ino,
                        mtime_ns=observed.st_mtime_ns,
                    )
                finally:
                    os.close(descriptor)
            os.fsync(staging_fd)
            os.fchmod(staging_fd, 0o550)
            os.fsync(staging_fd)
            sealed_root_stat = os.fstat(staging_fd)
            directories = {
                "": _ExpectedDirectory(
                    device=sealed_root_stat.st_dev,
                    inode=sealed_root_stat.st_ino,
                    mode=0o550,
                )
            }
            _verify_sealed_tree(
                root_descriptor=staging_fd,
                files=expected_files,
                directories=directories,
            )
            _verify_sealed_tree(
                root_descriptor=staging_fd,
                files=expected_files,
                directories=directories,
            )
            _require_canonical_directory_identity(
                authority_root.parent,
                _stat_identity(parent_stat),
                label="run-contract authority parent",
            )
            _rename_noreplace_at(
                parent_descriptor=parent_fd,
                source_name=staging_name,
                destination_name=authority_root.name,
            )
            published = True
            _verify_canonical_published_tree(
                final=authority_root,
                parent_identity=_stat_identity(parent_stat),
                root_identity=_stat_identity(sealed_root_stat),
                files=expected_files,
                directories=directories,
            )
        finally:
            os.close(staging_fd)
        os.fsync(parent_fd)
        load_campaign(destination)
    except Exception:
        if not published and staging_identity is not None:
            try:
                cleanup_fd = os.open(staging_name, _directory_open_flags(), dir_fd=parent_fd)
            except FileNotFoundError:
                cleanup_fd = None
            if cleanup_fd is not None:
                try:
                    if _stat_identity(os.fstat(cleanup_fd)) == staging_identity:
                        os.fchmod(cleanup_fd, 0o750)
                        with os.scandir(cleanup_fd) as iterator:
                            names = [entry.name for entry in iterator]
                        for name in names:
                            file_fd = os.open(
                                name,
                                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                                dir_fd=cleanup_fd,
                            )
                            try:
                                os.fchmod(file_fd, 0o640)
                            finally:
                                os.close(file_fd)
                            os.unlink(name, dir_fd=cleanup_fd)
                finally:
                    os.close(cleanup_fd)
                os.rmdir(staging_name, dir_fd=parent_fd)
        raise
    finally:
        os.close(parent_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--run-contract-authority-root", type=Path, required=True)
    parser.add_argument("--immutable-input-trust-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--forbidden-source-registry", type=Path, required=True)
    parser.add_argument("--reserved-source-registry", type=Path, required=True)
    parser.add_argument("--remote-environment-receipt", type=Path, required=True)
    parser.add_argument("--sft-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-at-utc")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_campaign(
        campaign_id=args.campaign_id,
        run_contract_authority_root=args.run_contract_authority_root,
        immutable_input_trust_root=args.immutable_input_trust_root,
        source_root=args.source_root,
        base_model_path=args.base_model_path,
        forbidden_source_registry=args.forbidden_source_registry,
        reserved_source_registry=args.reserved_source_registry,
        remote_environment_receipt=args.remote_environment_receipt,
        sft_data_root=args.sft_data_root,
        output_root=args.output_root,
        frozen_at_utc=args.frozen_at_utc,
    )
    _write_create_once(args.output, payload)
    print(f"frozen base-SFT campaign: {args.output.expanduser().absolute()}")
    print(f"campaign_sha256: {payload['campaign_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
