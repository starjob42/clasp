#!/usr/bin/env python3
"""Fail-closed advancement audit for staged CLASP training campaigns.

The auditor is intentionally read-only with respect to run directories.  It
does not invoke a scheduler or training entrypoint.  Its only write is an
atomic JSON receipt at ``--output``.  A successful receipt authorizes moving
to the *next development stage*; it never authorizes a publication/SOTA claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import posixpath
import re
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from clasp.baselines.foarl_f638_campaign import (
    FOARL_F638_CAMPAIGN_METHODS,
    get_foarl_f638_campaign_profile,
    validate_foarl_f638_campaign_contract,
)
from clasp.baselines.foarl_f638_comparison import (
    CLASP_FOARL_MATCHED_B9,
    CLASP_FOARL_MATCHED_B9_ARM_SCHEMA,
    FOARL_MATCHED_B9_COMPARISON_PROTOCOL,
    validate_clasp_foarl_matched_b9_method_contract,
)
from clasp.models import exact_peft_kl_post_step
from clasp.models.checkpoint_fidelity import RECEIPT_FILENAME as CHECKPOINT_FIDELITY_FILENAME
from clasp.models.exact_peft_kl_reference import (
    RECEIPT_FILENAME as EXACT_PEFT_KL_REFERENCE_FILENAME,
)
from clasp.models.publication_fidelity import (
    exact_peft_kl_runtime_identity,
    fixed_kl_reference_identity,
    publication_fidelity_identity,
    validate_fixed_kl_reference_identity,
    validate_publication_fidelity_identity,
)
from audit_sft_reload_equivalence import validate_reload_equivalence_identity

SCHEMA_VERSION = 2
POLICY_SCHEMA_VERSION = 1
SUPPORTED_METHODS = ("grpo", "dual_grpo", "clasp_dense", "clasp_exact")
CANONICAL_PUBLICATION_METHODS = (
    "grpo_z",
    "dual_scalar_z",
    "dual_separate_z",
    "clasp_canonical",
    "clasp_canonical_fixed_lambda",
    "clasp_canonical_no_recovery",
    CLASP_FOARL_MATCHED_B9,
    "set_null",
)
FOARL_PUBLICATION_METHODS = FOARL_F638_CAMPAIGN_METHODS
ALL_SUPPORTED_METHODS = (
    SUPPORTED_METHODS + CANONICAL_PUBLICATION_METHODS + FOARL_PUBLICATION_METHODS
)
SUPPORTED_TASKS = ("cvrp", "jssp", "mis", "op", "tsp")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHECKPOINT_RE = re.compile(r"^checkpoint-([1-9][0-9]*)$")
MODEL_ARTIFACTS = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
CHECKPOINT_REQUIRED = (
    "adapter_config.json",
    CHECKPOINT_FIDELITY_FILENAME,
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
)
PD_REQUIRED = {
    "cost_names",
    "cost_limits",
    "cost_weights",
    "dual_lambdas",
    "controller_state",
}
BON_REQUIRED = {"lambdas", "failure_ema", "observed_groups"}
HEALTH_GATE_KEYS = {
    "all_numeric_values_finite",
    "complete_training_step_coverage",
    "require_target_evaluation",
    "require_nonzero_gradient",
    "require_nonzero_advantage",
    "require_nonzero_objective_credit",
    "forbid_objective_fallback",
    "forbid_group_auxiliary_skips",
}
RESUME_GATE_KEYS = {
    "validate_all_retained_checkpoints",
    "require_optimizer",
    "require_scheduler",
    "require_all_rank_rng",
    "require_method_state",
    "require_root_checkpoint_log_continuity",
}
RESUME_DEFINITION = "all_retained_checkpoints_optimizer_resumable_and_terminal_state_consistent"
METHOD_CONTRACT = {
    "grpo": {
        "training_config": {"use_primal_dual": False, "use_clasp": False},
        "tracked_environment": {
            "CLASP_BON_ALIGNED": "0",
            "CLASP_FEASIBILITY_GATED": "0",
            "CLASP_PUBLICATION_FIDELITY": "1",
        },
    },
    "dual_grpo": {
        "training_config": {"use_primal_dual": True, "use_clasp": False},
        "tracked_environment": {
            "CLASP_BON_ALIGNED": "0",
            "CLASP_FEASIBILITY_GATED": "0",
            "CLASP_PUBLICATION_FIDELITY": "1",
        },
    },
    "clasp_dense": {
        "training_config": {
            "use_primal_dual": True,
            "use_clasp": True,
            "clasp_bon_success_credit": "dense_rank",
        },
        "tracked_environment": {
            "CLASP_BON_ALIGNED": "1",
            "CLASP_BON_SUCCESS_CREDIT": "dense_rank",
            "CLASP_FEASIBILITY_GATED": "1",
            "CLASP_PUBLICATION_FIDELITY": "1",
        },
    },
    "clasp_exact": {
        "training_config": {
            "use_primal_dual": True,
            "use_clasp": True,
            "clasp_bon_success_credit": "exact_loo",
        },
        "tracked_environment": {
            "CLASP_BON_ALIGNED": "1",
            "CLASP_BON_SUCCESS_CREDIT": "exact_loo",
            "CLASP_FEASIBILITY_GATED": "1",
            "CLASP_PUBLICATION_FIDELITY": "1",
        },
    },
    "grpo_z": {
        "training_config": {
            "entrypoint": "clasp.entrypoints.train_grpo",
            "arm_contract_schema": "clasp-canonical-matched-a100-v1",
            "use_primal_dual": True,
            "use_clasp": False,
            "clasp_canonical_set": False,
            "pd_group_scalarization": True,
            "pd_dual_lr": 0.0,
            "clasp_bon_dual_lr": 0.0,
            "clasp_bon_lambda_init": 0.0,
            "clasp_canonical_recovery_scale": 0.0,
        },
        "tracked_environment": {
            "CLASP_CANONICAL_SET": "0",
            "CLASP_BON_ALIGNED": "0",
            "CLASP_FRONTIER_CDF": "0",
            "CLASP_FEASIBILITY_GATED": "0",
            "PD_IMPORTANCE_SAMPLING_LEVEL": "token",
            "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION": "1",
            "CLASP_PUBLICATION_FIDELITY": "1",
        },
    },
    "dual_scalar_z": {
        "training_config": {
            "entrypoint": "clasp.entrypoints.train_grpo",
            "arm_contract_schema": "clasp-canonical-matched-a100-v1",
            "use_primal_dual": True,
            "use_clasp": False,
            "clasp_canonical_set": False,
            "pd_group_scalarization": True,
            "pd_dual_lr": 0.05,
            "clasp_bon_dual_lr": 0.0,
            "clasp_bon_lambda_init": 0.0,
            "clasp_canonical_recovery_scale": 0.0,
        },
        "tracked_environment": {
            "CLASP_CANONICAL_SET": "0",
            "CLASP_BON_ALIGNED": "0",
            "CLASP_FRONTIER_CDF": "0",
            "CLASP_FEASIBILITY_GATED": "0",
            "PD_IMPORTANCE_SAMPLING_LEVEL": "token",
            "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION": "1",
            "CLASP_PUBLICATION_FIDELITY": "1",
        },
    },
    "clasp_canonical": {
        "training_config": {
            "entrypoint": "clasp.entrypoints.train_grpo",
            "arm_contract_schema": "clasp-canonical-matched-a100-v1",
            "use_primal_dual": True,
            "use_clasp": True,
            "clasp_canonical_set": True,
            "pd_group_scalarization": False,
            "pd_dual_lr": 0.0,
            "clasp_bon_dual_lr": 0.05,
            "clasp_bon_lambda_init": 0.0,
            "clasp_canonical_recovery_scale": 1.0,
        },
        "tracked_environment": {
            "CLASP_CANONICAL_SET": "1",
            "CLASP_BON_ALIGNED": "1",
            "CLASP_FRONTIER_CDF": "0",
            "CLASP_FEASIBILITY_GATED": "1",
            "PD_IMPORTANCE_SAMPLING_LEVEL": "token",
            "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION": "1",
            "CLASP_PUBLICATION_FIDELITY": "1",
        },
    },
    "clasp_canonical_fixed_lambda": {
        "training_config": {
            "entrypoint": "clasp.entrypoints.train_grpo",
            "arm_contract_schema": "clasp-canonical-matched-a100-v1",
            "use_primal_dual": True,
            "use_clasp": True,
            "clasp_canonical_set": True,
            "pd_group_scalarization": False,
            "pd_dual_lr": 0.0,
            "clasp_bon_dual_lr": 0.0,
            "clasp_canonical_recovery_scale": 1.0,
        },
        "tracked_environment": {
            "CLASP_CANONICAL_SET": "1",
            "CLASP_BON_ALIGNED": "1",
            "CLASP_FRONTIER_CDF": "0",
            "CLASP_FEASIBILITY_GATED": "1",
            "PD_IMPORTANCE_SAMPLING_LEVEL": "token",
            "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION": "1",
            "CLASP_PUBLICATION_FIDELITY": "1",
        },
    },
    "clasp_canonical_no_recovery": {
        "training_config": {
            "entrypoint": "clasp.entrypoints.train_grpo",
            "arm_contract_schema": "clasp-canonical-matched-a100-v1",
            "use_primal_dual": True,
            "use_clasp": True,
            "clasp_canonical_set": True,
            "pd_group_scalarization": False,
            "pd_dual_lr": 0.0,
            "clasp_bon_dual_lr": 0.05,
            "clasp_bon_lambda_init": 0.0,
            "clasp_canonical_recovery_scale": 0.0,
        },
        "tracked_environment": {
            "CLASP_CANONICAL_SET": "1",
            "CLASP_BON_ALIGNED": "1",
            "CLASP_FRONTIER_CDF": "0",
            "CLASP_FEASIBILITY_GATED": "1",
            "PD_IMPORTANCE_SAMPLING_LEVEL": "token",
            "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION": "1",
            "CLASP_PUBLICATION_FIDELITY": "1",
        },
    },
}
METHOD_CONTRACT["dual_separate_z"] = copy.deepcopy(METHOD_CONTRACT["dual_scalar_z"])
METHOD_CONTRACT["dual_separate_z"]["training_config"].update(
    {
        "pd_group_scalarization": False,
        "pd_separate_z": True,
    }
)
METHOD_CONTRACT["set_null"] = copy.deepcopy(METHOD_CONTRACT["clasp_canonical"])
METHOD_CONTRACT["set_null"]["training_config"].update(
    {
        "clasp_canonical_set": False,
        "clasp_fixed_null_set": True,
    }
)
METHOD_CONTRACT["set_null"]["tracked_environment"].update(
    {
        "CLASP_CANONICAL_SET": "0",
        "CLASP_FIXED_NULL_SET": "1",
    }
)
METHOD_CONTRACT[CLASP_FOARL_MATCHED_B9] = copy.deepcopy(
    METHOD_CONTRACT["clasp_canonical"]
)
METHOD_CONTRACT[CLASP_FOARL_MATCHED_B9]["training_config"].update(
    {
        "arm_contract_schema": CLASP_FOARL_MATCHED_B9_ARM_SCHEMA,
        "comparison_protocol": FOARL_MATCHED_B9_COMPARISON_PROTOCOL,
        "comparison_baseline": "foarl_f638_matched_b9",
        "attribution": "common_compute_data_token_schedule_only",
        "reproduction_claim": "not_an_official_jiang_reproduction",
        "beta": 0.05,
    }
)
for _canonical_method in CANONICAL_PUBLICATION_METHODS:
    _is_canonical_set = _canonical_method.startswith("clasp_canonical")
    _is_fixed_null_set = _canonical_method == "set_null"
    _is_exact_set = _is_canonical_set or _is_fixed_null_set
    METHOD_CONTRACT[_canonical_method]["training_config"].update(
        {
            "clasp_bon_target_failure_rate": 0.005,
            "pd_lambda_init": 0.0,
            "pd_cost_limit": 0.0,
            "pd_dual_update_interval": 1,
            "pd_violation_ema_beta": 0.1,
            "pd_positive_constraint_penalty": True,
            "pd_separate_z": _canonical_method == "dual_separate_z",
            "clasp_fixed_null_set": _is_fixed_null_set,
            "policy_credit_formula": (
                "group_z_verified_quality_v1"
                if _canonical_method == "grpo_z"
                else "group_z_scalarized_positive_component_penalty_v1"
                if _canonical_method == "dual_scalar_z"
                else "group_z_reward_minus_group_z_positive_component_penalty_v1"
                if _canonical_method == "dual_separate_z"
                else "fixed_null_k8_same_set_utility_v1"
                if _is_fixed_null_set
                else "canonical_k8_leave_out_set_utility_v1"
            ),
            "component_controller": (
                "legacy_projected_ema_component_gap"
                if _canonical_method in {"dual_scalar_z", "dual_separate_z"}
                else "frozen_zero"
            ),
            "set_controller": (
                "frozen"
                if _canonical_method == "clasp_canonical_fixed_lambda"
                else "signed_failure_at_k_ema"
                if _is_exact_set
                else "off"
            ),
        }
    )
    METHOD_CONTRACT[_canonical_method]["tracked_environment"].update(
        {
            "CLASP_BON_TARGET_FAILURE_RATE": "0.005",
            "CLASP_FIXED_NULL_SET": "1" if _is_fixed_null_set else "0",
        }
    )
for _foarl_method in FOARL_PUBLICATION_METHODS:
    _foarl_profile = get_foarl_f638_campaign_profile(_foarl_method)
    METHOD_CONTRACT[_foarl_method] = {
        "training_config": {
            "entrypoint": "clasp.entrypoints.train_foarl_f638_campaign",
            "arm_contract_schema": _foarl_profile.arm_contract_schema,
            "attribution": _foarl_profile.attribution,
            "reproduction_claim": "not_an_official_reproduction",
            "use_primal_dual": False,
            "use_clasp": False,
            "prompt_style": "jiang_f638",
            "no_reason": False,
            "loss_type": "grpo",
            "beta": 0.05,
            "reference_policy": "fixed_corrected_sft_step_zero",
        },
        "tracked_environment": {
            "CLASP_PUBLICATION_FIDELITY": "1",
            "CLASP_LORA_TARGET_MODULES": "jiang",
            "CLASP_LORA_BIAS": "all",
            "GRPO_STRICT_OPTIMIZER_RESUME": "1",
        },
    }
COMPARABILITY_MASK = {
    "training_config": {
        "entrypoint",
        "policy_credit_formula",
        "component_controller",
        "set_controller",
        "use_primal_dual",
        "use_clasp",
        "clasp_canonical_set",
        "clasp_fixed_null_set",
        "pd_group_scalarization",
        "pd_separate_z",
        "pd_dual_lr",
        "pd_dual_lr_up",
        "pd_dual_lr_down",
        "clasp_bon_dual_lr",
        "clasp_bon_lambda_init",
        "clasp_canonical_recovery_scale",
        "clasp_bon_success_credit",
    },
    "tracked_environment": {
        "CLASP_BON_ALIGNED",
        "CLASP_BON_SUCCESS_CREDIT",
        "CLASP_FEASIBILITY_GATED",
        "CLASP_CANONICAL_SET",
        "CLASP_FIXED_NULL_SET",
        "CLASP_BON_DUAL_LR",
        "CLASP_BON_LAMBDA_INIT",
        "CLASP_CANONICAL_RECOVERY_SCALE",
    },
}
RUN_CONTRACT_KEYS = {
    "schema_version",
    "stage",
    "method",
    "task",
    "seeds",
    "fixed_horizon",
    "data_lineage",
    "model_initialization_registry",
    "fixed_kl_reference",
    "fixed_null_receipt",
    "fingerprints",
    "generation",
    "training_config",
    "tracked_environment",
}
_FIXED_KL_LIVE_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
DATA_LINEAGE_KEYS = {
    "forbidden_source_registry",
    "sft_train",
    "rl_train",
}
FORBIDDEN_SOURCE_REGISTRY_KEYS = {
    "schema_version",
    "protocol",
    "status",
    "registry_sha256",
    "source_tree_manifest_sha256",
    "tasks",
    "required_labels",
    "source_count",
    "sources_by_task_label",
}
FORBIDDEN_SOURCE_KEYS = {
    "source_identity_sha256",
    "file_manifest_sha256",
    "rows",
    "unique_instance_hashes",
    "files",
}
FORBIDDEN_SOURCE_FILE_KEYS = {"relative_path", "sha256", "size_bytes", "rows"}
LINEAGE_BINDING_KEYS = {
    "task",
    "label",
    "canonical_path",
    "registry_sha256",
    "source_tree_manifest_sha256",
    *FORBIDDEN_SOURCE_KEYS,
}
REQUIRED_FORBIDDEN_SOURCE_LABELS = (
    "historical-eval",
    "rl-train",
    "sft-train",
)
FORBIDDEN_SOURCE_REGISTRY_PROTOCOL = "clasp-forbidden-source-registry-v1"
FORBIDDEN_SOURCE_REGISTRY_STATUS = "frozen_before_confirmation_generation"
MODEL_INITIALIZATION_REGISTRY_KEYS = {
    "schema_version",
    "protocol",
    "status",
    "registry_sha256",
    "model_family",
    "base_model_tree_sha256",
    "forbidden_source_registry_sha256",
    "forbidden_source_tree_manifest_sha256",
    "specialist_sft_by_task",
}
SPECIALIST_SFT_KEYS = {
    "initial_model_tree_sha256",
    "sft_train_source_identity_sha256",
    "sft_seed",
    "matched_sft_contract_sha256",
    "checkpoint_fidelity",
    "reload_equivalence",
}
MODEL_INITIALIZATION_REGISTRY_PROTOCOL = "clasp-model-initialization-registry-v2"
MODEL_INITIALIZATION_REGISTRY_STATUS = "frozen_before_training"
MODEL_INITIALIZATION_FAMILY = "Qwen2.5-7B"
MODEL_INITIALIZATION_SFT_SEED = 42
MATCHED_SFT_CONTRACT_SHA256 = "dd294083e1643d08e833bca35be1ad275544b1bc59e4c0ddac44deb874f5bf48"
PRIOR_STAGE_RECEIPT_IDENTITY_KEYS = {
    "campaign_id",
    "stage_id",
    "policy_sha256",
    "receipt_sha256",
    "receipt_file_sha256",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, expected: type | tuple[type, ...]) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, expected):
        names = (
            ", ".join(item.__name__ for item in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise ValueError(f"expected {names} in {path}, got {type(value).__name__}")
    return value


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_json_object_snapshot(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], str, int, int]:
    """Read and hash one immutable JSON object through a stable descriptor."""

    try:
        path_before = path.lstat()
        if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
            raise ValueError(f"{label} is not a regular non-symlink file")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError(f"cannot open stable {label}: {exc}") from exc
    try:
        descriptor_before = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_before.st_mode):
            raise ValueError(f"{label} descriptor is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
        descriptor_after = os.fstat(descriptor)
        path_after = path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot read stable {label}: {exc}") from exc
    finally:
        os.close(descriptor)
    identities = {
        _stat_identity(path_before),
        _stat_identity(descriptor_before),
        _stat_identity(descriptor_after),
        _stat_identity(path_after),
    }
    if len(identities) != 1:
        raise ValueError(f"{label} changed or its pathname was replaced during audit")
    content = b"".join(chunks)
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {label}, got {type(value).__name__}")
    return (
        value,
        hashlib.sha256(content).hexdigest(),
        len(content),
        stat.S_IMODE(descriptor_after.st_mode),
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _finite_errors(value: object, *, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return errors
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            errors.append(f"non-finite numeric value at {path}: {value!r}")
        return errors
    if isinstance(value, dict):
        for key, child in value.items():
            errors.extend(_finite_errors(child, path=f"{path}.{key}"))
        return errors
    if isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_finite_errors(child, path=f"{path}[{index}]"))
    return errors


def _artifact_receipt(path: Path, *, relative_to: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required artifact is not a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"required artifact is empty: {path}")
    try:
        relative = path.resolve().relative_to(relative_to.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact escaped run directory: {path}") from exc
    return {"path": relative, "size_bytes": size, "sha256": _sha256_file(path)}


def _validate_fingerprint(value: object, *, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} is not an object"]
    files = value.get("files")
    if not isinstance(files, list) or not files:
        return [f"{label}.files is missing or empty"]
    errors: list[str] = []
    aggregate = hashlib.sha256()
    total_size = 0
    paths: list[str] = []
    for index, item in enumerate(files):
        item_label = f"{label}.files[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_label} is not an object")
            continue
        relative = item.get("path")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            errors.append(f"{item_label}.path is not a safe relative path")
            continue
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            errors.append(f"{item_label}.size_bytes is invalid")
            continue
        if not _is_sha256(digest):
            errors.append(f"{item_label}.sha256 is invalid")
            continue
        paths.append(relative)
        total_size += size
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        errors.append(f"{label}.files must be uniquely sorted by path")
    if value.get("file_count") != len(files):
        errors.append(f"{label}.file_count disagrees with files")
    if value.get("size_bytes") != total_size:
        errors.append(f"{label}.size_bytes disagrees with files")
    if value.get("sha256") != aggregate.hexdigest():
        errors.append(f"{label}.sha256 disagrees with its file receipts")
    canonical_path = value.get("canonical_path")
    if not isinstance(canonical_path, str) or not Path(canonical_path).is_absolute():
        errors.append(f"{label}.canonical_path is not absolute")
    return errors


def _key_set_errors(
    value: dict[str, Any],
    expected: set[str],
    *,
    label: str,
) -> list[str]:
    observed = set(value)
    if observed == expected:
        return []
    return [
        f"{label} has an unexpected key set: "
        f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
    ]


def _validate_compact_forbidden_source(
    value: object,
    *,
    task: str,
    source_label: str,
    label: str,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} is not an object"]
    errors = _key_set_errors(value, FORBIDDEN_SOURCE_KEYS, label=label)
    files = value.get("files")
    if not isinstance(files, list) or not files:
        return [*errors, f"{label}.files is missing or empty"]
    relative_paths: list[str] = []
    rows = 0
    for index, item in enumerate(files):
        item_label = f"{label}.files[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_label} is not an object")
            continue
        errors.extend(_key_set_errors(item, FORBIDDEN_SOURCE_FILE_KEYS, label=item_label))
        relative = item.get("relative_path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative != posixpath.normpath(relative)
            or relative.startswith("/")
            or relative == ".."
            or relative.startswith("../")
            or "\\" in relative
        ):
            errors.append(f"{item_label}.relative_path is not a safe canonical path")
        else:
            relative_paths.append(relative)
        if not _is_sha256(item.get("sha256")):
            errors.append(f"{item_label}.sha256 is invalid")
        size = item.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            errors.append(f"{item_label}.size_bytes is invalid")
        item_rows = item.get("rows")
        if isinstance(item_rows, bool) or not isinstance(item_rows, int) or item_rows <= 0:
            errors.append(f"{item_label}.rows is invalid")
        else:
            rows += item_rows
    if relative_paths != sorted(relative_paths) or len(relative_paths) != len(set(relative_paths)):
        errors.append(f"{label}.files must be uniquely sorted by relative_path")
    if (
        value.get("file_manifest_sha256")
        != hashlib.sha256(_canonical_json_bytes(files)).hexdigest()
    ):
        errors.append(f"{label}.file_manifest_sha256 disagrees with files")
    declared_rows = value.get("rows")
    unique_rows = value.get("unique_instance_hashes")
    if (
        isinstance(declared_rows, bool)
        or not isinstance(declared_rows, int)
        or declared_rows <= 0
        or declared_rows != rows
    ):
        errors.append(f"{label}.rows disagrees with files")
    if (
        isinstance(unique_rows, bool)
        or not isinstance(unique_rows, int)
        or unique_rows <= 0
        or not isinstance(declared_rows, int)
        or unique_rows > declared_rows
    ):
        errors.append(f"{label}.unique_instance_hashes is invalid")
    source_identity = {
        "task": task,
        "label": source_label,
        "files": files,
        "file_manifest_sha256": value.get("file_manifest_sha256"),
        "rows": value.get("rows"),
        "unique_instance_hashes": value.get("unique_instance_hashes"),
    }
    expected_identity = hashlib.sha256(_canonical_json_bytes(source_identity)).hexdigest()
    if value.get("source_identity_sha256") != expected_identity:
        errors.append(f"{label}.source_identity_sha256 disagrees with source identity")
    return errors


def _validate_compact_forbidden_registry(
    value: object,
    *,
    task: str,
    label: str,
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(value, dict):
        return {}, [f"{label} is not an object"]
    registry = value
    errors = _key_set_errors(registry, FORBIDDEN_SOURCE_REGISTRY_KEYS, label=label)
    errors.extend(
        _require_equal(
            registry,
            {
                "schema_version": 1,
                "protocol": FORBIDDEN_SOURCE_REGISTRY_PROTOCOL,
                "status": FORBIDDEN_SOURCE_REGISTRY_STATUS,
            },
            label=label,
        )
    )
    for field in ("registry_sha256", "source_tree_manifest_sha256"):
        if not _is_sha256(registry.get(field)):
            errors.append(f"{label}.{field} is invalid")
    tasks = registry.get("tasks")
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(not isinstance(item, str) or not item for item in tasks)
        or tasks != sorted(tasks)
        or len(tasks) != len(set(tasks))
    ):
        errors.append(f"{label}.tasks must be a sorted unique nonempty string list")
        tasks = []
    elif tasks != list(SUPPORTED_TASKS):
        errors.append(f"{label}.tasks does not exactly cover all five official tasks")
    if task not in tasks:
        errors.append(f"{label}.tasks does not include {task!r}")
    required_labels = registry.get("required_labels")
    if required_labels != list(REQUIRED_FORBIDDEN_SOURCE_LABELS):
        errors.append(f"{label}.required_labels does not match the official registry")
    expected_source_count = len(tasks) * len(REQUIRED_FORBIDDEN_SOURCE_LABELS)
    source_count = registry.get("source_count")
    if (
        isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count != expected_source_count
    ):
        errors.append(f"{label}.source_count disagrees with tasks and required_labels")
    sources = registry.get("sources_by_task_label")
    if not isinstance(sources, dict):
        errors.append(f"{label}.sources_by_task_label is not an object")
        return registry, errors
    if set(sources) != set(tasks):
        errors.append(f"{label}.sources_by_task_label does not exactly cover tasks")
    source_projections: list[dict[str, Any]] = []
    for registered_task in tasks:
        task_sources = sources.get(registered_task)
        task_label = f"{label}.sources_by_task_label.{registered_task}"
        if not isinstance(task_sources, dict):
            errors.append(f"{task_label} is not an object")
            continue
        if set(task_sources) != set(REQUIRED_FORBIDDEN_SOURCE_LABELS):
            errors.append(f"{task_label} does not exactly cover required_labels")
        for source_label in REQUIRED_FORBIDDEN_SOURCE_LABELS:
            source = task_sources.get(source_label)
            errors.extend(
                _validate_compact_forbidden_source(
                    source,
                    task=registered_task,
                    source_label=source_label,
                    label=f"{task_label}.{source_label}",
                )
            )
            if isinstance(source, dict):
                source_projections.append(
                    {"task": registered_task, "label": source_label, **source}
                )
    expected_tree_hash = hashlib.sha256(_canonical_json_bytes(source_projections)).hexdigest()
    if registry.get("source_tree_manifest_sha256") != expected_tree_hash:
        errors.append(f"{label}.source_tree_manifest_sha256 disagrees with sources")
    return registry, errors


def _validate_lineage_binding(
    value: object,
    *,
    registry: dict[str, Any],
    task: str,
    source_label: str,
    label: str,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} is not an object"]
    binding = value
    errors = _key_set_errors(binding, LINEAGE_BINDING_KEYS, label=label)
    errors.extend(
        _require_equal(
            binding,
            {
                "task": task,
                "label": source_label,
                "registry_sha256": registry.get("registry_sha256"),
                "source_tree_manifest_sha256": registry.get("source_tree_manifest_sha256"),
            },
            label=label,
        )
    )
    canonical_path = binding.get("canonical_path")
    if not isinstance(canonical_path, str) or not Path(canonical_path).is_absolute():
        errors.append(f"{label}.canonical_path is not absolute")
    sources = registry.get("sources_by_task_label")
    registered = (
        sources.get(task, {}).get(source_label)
        if isinstance(sources, dict) and isinstance(sources.get(task), dict)
        else None
    )
    if not isinstance(registered, dict):
        errors.append(f"{label} has no matching compact registry source")
    else:
        for field in sorted(FORBIDDEN_SOURCE_KEYS):
            if binding.get(field) != registered.get(field):
                errors.append(f"{label}.{field} disagrees with compact registry source")
    return errors


def _validate_data_lineage(
    value: object,
    *,
    task: str,
    data_fingerprint: object,
) -> list[str]:
    label = "run_contract.data_lineage"
    if not isinstance(value, dict):
        return [f"{label} is not an object"]
    lineage = value
    errors = _key_set_errors(lineage, DATA_LINEAGE_KEYS, label=label)
    registry, registry_errors = _validate_compact_forbidden_registry(
        lineage.get("forbidden_source_registry"),
        task=task,
        label=f"{label}.forbidden_source_registry",
    )
    errors.extend(registry_errors)
    for field, source_label in (("sft_train", "sft-train"), ("rl_train", "rl-train")):
        errors.extend(
            _validate_lineage_binding(
                lineage.get(field),
                registry=registry,
                task=task,
                source_label=source_label,
                label=f"{label}.{field}",
            )
        )
    fingerprint_files = (
        data_fingerprint.get("files") if isinstance(data_fingerprint, dict) else None
    )
    rl_binding = lineage.get("rl_train")
    registered_files = rl_binding.get("files") if isinstance(rl_binding, dict) else None
    expected_fingerprint_files = [
        {
            "path": row.get("relative_path"),
            "size_bytes": row.get("size_bytes"),
            "sha256": row.get("sha256"),
        }
        for row in (registered_files or [])
        if isinstance(row, dict)
    ]
    if fingerprint_files != expected_fingerprint_files:
        errors.append("run_contract.fingerprints.data differs from registered RL data")
    return errors


def _validate_model_initialization_registry(
    value: object,
    *,
    task: str,
    seeds: object,
    fingerprints: object,
    data_lineage: object,
    expected_registry_sha256: str,
) -> list[str]:
    label = "run_contract.model_initialization_registry"
    if not isinstance(value, dict):
        return [f"{label} is not an object"]
    registry = value
    errors = _key_set_errors(registry, MODEL_INITIALIZATION_REGISTRY_KEYS, label=label)
    errors.extend(
        _require_equal(
            registry,
            {
                "schema_version": 2,
                "protocol": MODEL_INITIALIZATION_REGISTRY_PROTOCOL,
                "status": MODEL_INITIALIZATION_REGISTRY_STATUS,
            },
            label=label,
        )
    )
    for field in (
        "registry_sha256",
        "base_model_tree_sha256",
        "forbidden_source_registry_sha256",
        "forbidden_source_tree_manifest_sha256",
    ):
        if not _is_sha256(registry.get(field)):
            errors.append(f"{label}.{field} is invalid")
    if registry.get("registry_sha256") != expected_registry_sha256:
        errors.append(f"{label}.registry_sha256 disagrees with advancement policy")
    if registry.get("model_family") != MODEL_INITIALIZATION_FAMILY:
        errors.append(f"{label}.model_family is not the approved model family")

    lineage_registry = (
        data_lineage.get("forbidden_source_registry") if isinstance(data_lineage, dict) else None
    )
    if not isinstance(lineage_registry, dict):
        lineage_registry = {}
    if registry.get("forbidden_source_registry_sha256") != lineage_registry.get("registry_sha256"):
        errors.append(f"{label} disagrees with the forbidden-source registry hash")
    if registry.get("forbidden_source_tree_manifest_sha256") != lineage_registry.get(
        "source_tree_manifest_sha256"
    ):
        errors.append(f"{label} disagrees with the forbidden-source tree hash")

    specialists = registry.get("specialist_sft_by_task")
    lineage_tasks = lineage_registry.get("tasks")
    if not isinstance(specialists, dict) or not specialists:
        errors.append(f"{label}.specialist_sft_by_task is not a nonempty object")
        specialists = {}
    elif set(specialists) != set(SUPPORTED_TASKS):
        errors.append(f"{label}.specialist_sft_by_task does not cover all five tasks")
    if (
        isinstance(lineage_tasks, list)
        and all(isinstance(item, str) for item in lineage_tasks)
        and set(specialists) != set(lineage_tasks)
    ):
        errors.append(f"{label}.specialist_sft_by_task does not exactly cover lineage tasks")
    lineage_sources = lineage_registry.get("sources_by_task_label")
    for specialist_task, specialist in specialists.items():
        specialist_label = f"{label}.specialist_sft_by_task.{specialist_task}"
        if not isinstance(specialist_task, str) or not specialist_task:
            errors.append(f"{specialist_label} has an invalid task key")
        if not isinstance(specialist, dict):
            errors.append(f"{specialist_label} is not an object")
            continue
        errors.extend(_key_set_errors(specialist, SPECIALIST_SFT_KEYS, label=specialist_label))
        for field in (
            "initial_model_tree_sha256",
            "sft_train_source_identity_sha256",
        ):
            if not _is_sha256(specialist.get(field)):
                errors.append(f"{specialist_label}.{field} is invalid")
        sft_seed = specialist.get("sft_seed")
        if sft_seed != MODEL_INITIALIZATION_SFT_SEED:
            errors.append(f"{specialist_label}.sft_seed is not the matched SFT seed")
        if specialist.get("matched_sft_contract_sha256") != MATCHED_SFT_CONTRACT_SHA256:
            errors.append(
                f"{specialist_label}.matched_sft_contract_sha256 is not the official design"
            )
        try:
            checkpoint_fidelity = validate_publication_fidelity_identity(
                specialist.get("checkpoint_fidelity")
            )
        except ValueError as exc:
            errors.append(f"{specialist_label}.checkpoint_fidelity is invalid: {exc}")
            checkpoint_fidelity = None
        try:
            reload_equivalence = validate_reload_equivalence_identity(
                specialist.get("reload_equivalence")
            )
        except ValueError as exc:
            errors.append(f"{specialist_label}.reload_equivalence is invalid: {exc}")
            reload_equivalence = None
        if (
            checkpoint_fidelity is not None
            and reload_equivalence is not None
            and reload_equivalence["checkpoint_fidelity_identity_sha256"]
            != checkpoint_fidelity["identity_sha256"]
        ):
            errors.append(
                f"{specialist_label}.reload_equivalence targets another checkpoint"
            )
        registered_sft = (
            lineage_sources.get(specialist_task, {}).get("sft-train")
            if isinstance(lineage_sources, dict)
            and isinstance(lineage_sources.get(specialist_task), dict)
            else None
        )
        if not isinstance(registered_sft, dict) or specialist.get(
            "sft_train_source_identity_sha256"
        ) != registered_sft.get("source_identity_sha256"):
            errors.append(f"{specialist_label} disagrees with registered SFT lineage")

    current = specialists.get(task)
    if not isinstance(current, dict):
        errors.append(f"{label} has no specialist SFT initialization for {task!r}")
        current = {}
    fingerprint_map = fingerprints if isinstance(fingerprints, dict) else {}
    base_fingerprint = fingerprint_map.get("base_model")
    initial_fingerprint = fingerprint_map.get("initial_model")
    base_sha256 = base_fingerprint.get("sha256") if isinstance(base_fingerprint, dict) else None
    initial_sha256 = (
        initial_fingerprint.get("sha256") if isinstance(initial_fingerprint, dict) else None
    )
    sft_seed = seeds.get("sft") if isinstance(seeds, dict) else None
    sft_binding = data_lineage.get("sft_train") if isinstance(data_lineage, dict) else None
    sft_source_sha256 = (
        sft_binding.get("source_identity_sha256") if isinstance(sft_binding, dict) else None
    )
    if registry.get("base_model_tree_sha256") != base_sha256:
        errors.append(f"{label}.base_model_tree_sha256 disagrees with base fingerprint")
    if current.get("initial_model_tree_sha256") != initial_sha256:
        errors.append(f"{label} specialist initial model disagrees with initial fingerprint")
    if current.get("sft_seed") != sft_seed:
        errors.append(f"{label} specialist SFT seed disagrees with run seeds")
    if current.get("sft_train_source_identity_sha256") != sft_source_sha256:
        errors.append(f"{label} specialist SFT source disagrees with run lineage")
    return errors


def _validate_policy(policy_path: Path) -> dict[str, Any]:
    policy = _load_json(policy_path, expected=dict)
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("unsupported advancement-policy schema_version")
    stored_hash = policy.get("policy_sha256")
    unsigned = dict(policy)
    unsigned.pop("policy_sha256", None)
    observed_hash = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if stored_hash != observed_hash:
        raise ValueError(
            "advancement policy self-hash mismatch: "
            f"stored={stored_hash!r}, observed={observed_hash}"
        )
    if policy.get("status") != "frozen_before_results":
        raise ValueError("advancement policy is not frozen_before_results")
    for key in ("campaign_id", "stage_id", "frozen_at_utc", "run_path_template"):
        if not isinstance(policy.get(key), str) or not policy[key]:
            raise ValueError(f"advancement policy has no valid {key}")
    stage_index = policy.get("stage_index")
    if isinstance(stage_index, bool) or not isinstance(stage_index, int) or stage_index < 0:
        raise ValueError("advancement policy stage_index must be a nonnegative integer")
    prior_chain = policy.get("prior_stage_receipt_chain")
    if not isinstance(prior_chain, list) or len(prior_chain) != stage_index:
        raise ValueError(
            "advancement policy prior_stage_receipt_chain length must equal stage_index"
        )
    prior_stage_ids: list[str] = []
    prior_receipt_hashes: list[str] = []
    for index, entry in enumerate(prior_chain):
        label = f"advancement policy prior_stage_receipt_chain[{index}]"
        if not isinstance(entry, dict) or set(entry) != PRIOR_STAGE_RECEIPT_IDENTITY_KEYS:
            raise ValueError(f"{label} does not use the exact receipt-identity schema")
        if entry.get("campaign_id") != policy["campaign_id"]:
            raise ValueError(f"{label} belongs to another campaign")
        prior_stage_id = entry.get("stage_id")
        if not isinstance(prior_stage_id, str) or not prior_stage_id:
            raise ValueError(f"{label} has no valid stage_id")
        prior_stage_ids.append(prior_stage_id)
        for field in (
            "policy_sha256",
            "receipt_sha256",
            "receipt_file_sha256",
        ):
            if not _is_sha256(entry.get(field)):
                raise ValueError(f"{label}.{field} is invalid")
        prior_receipt_hashes.append(str(entry["receipt_sha256"]))
    if len(prior_stage_ids) != len(set(prior_stage_ids)):
        raise ValueError("advancement policy prior stage IDs must be unique")
    if policy["stage_id"] in prior_stage_ids:
        raise ValueError("advancement policy current stage appears in its prior chain")
    if len(prior_receipt_hashes) != len(set(prior_receipt_hashes)):
        raise ValueError("advancement policy prior receipt hashes must be unique")
    if not _is_sha256(policy.get("model_initialization_registry_sha256")):
        raise ValueError("advancement policy has no valid model_initialization_registry_sha256")
    target_step = policy.get("target_step")
    if isinstance(target_step, bool) or not isinstance(target_step, int) or target_step <= 0:
        raise ValueError("advancement policy target_step must be positive")
    for key in ("expected_tasks", "expected_methods", "train_seeds"):
        values = policy.get(key)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"advancement policy {key} must be a nonempty unique list")
    methods = policy["expected_methods"]
    if any(method not in ALL_SUPPORTED_METHODS for method in methods):
        raise ValueError("advancement policy contains an unsupported method")
    foarl_methods = [method for method in methods if method in FOARL_PUBLICATION_METHODS]
    if foarl_methods:
        if len(methods) != 1 or len(foarl_methods) != 1:
            raise ValueError(
                "each FOARL advancement policy must bind exactly one lane; "
                "native and matched-B9 cannot share a policy"
            )
        if len(policy["expected_tasks"]) != 1:
            raise ValueError("each FOARL advancement policy must bind exactly one task")
    if any(method in CANONICAL_PUBLICATION_METHODS for method in methods):
        target = policy.get("canonical_target_failure_rate")
        if (
            isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not math.isfinite(float(target))
            or float(target) != 0.005
        ):
            raise ValueError(
                "canonical advancement policy must preregister target failure@8=0.005"
            )
        fixed_by_task = policy.get("canonical_fixed_lambda_by_task")
        if "clasp_canonical_fixed_lambda" in methods:
            expected_tasks = set(policy["expected_tasks"])
            if not isinstance(fixed_by_task, dict) or set(fixed_by_task) != expected_tasks:
                raise ValueError(
                    "canonical fixed-lambda policy must bind exactly one value per task"
                )
            for task, value in fixed_by_task.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(
                        f"canonical fixed lambda is invalid for task {task!r}"
                    )
        fixed_null_by_task = policy.get("fixed_null_receipt_sha256_by_task")
        if "set_null" in methods:
            expected_tasks = set(policy["expected_tasks"])
            if (
                not isinstance(fixed_null_by_task, dict)
                or set(fixed_null_by_task) != expected_tasks
            ):
                raise ValueError(
                    "set_null policy must bind exactly one fixed-null receipt "
                    "SHA-256 per task"
                )
            invalid_tasks = sorted(
                task
                for task, digest in fixed_null_by_task.items()
                if not _is_sha256(digest)
            )
            if invalid_tasks:
                raise ValueError(
                    "set_null policy has invalid fixed-null receipt SHA-256 for: "
                    + ", ".join(invalid_tasks)
                )
        elif fixed_null_by_task is not None:
            raise ValueError(
                "fixed_null_receipt_sha256_by_task is valid only when set_null "
                "is an expected method"
            )
    if set(policy.get("health_gates", {})) != HEALTH_GATE_KEYS or not all(
        policy["health_gates"].values()
    ):
        raise ValueError("all fixed health gates must be present and enabled")
    resume = policy.get("resume_gate")
    if not isinstance(resume, dict) or resume.get("definition") != RESUME_DEFINITION:
        raise ValueError("resume gate has an unsupported definition")
    resume_flags = {key: value for key, value in resume.items() if key != "definition"}
    if set(resume_flags) != RESUME_GATE_KEYS or not all(resume_flags.values()):
        raise ValueError("all fixed resume gates must be present and enabled")
    requirements = policy.get("contract_requirements")
    if not isinstance(requirements, dict):
        raise ValueError("advancement policy has no contract_requirements object")
    required_positive = {
        "world_size",
        "sft_seed",
        "num_generations",
        "logging_steps",
        "save_steps",
        "eval_steps",
    }
    if set(requirements) != required_positive:
        raise ValueError("contract_requirements keys do not match the fixed schema")
    if any(
        isinstance(requirements[key], bool)
        or not isinstance(requirements[key], int)
        or requirements[key] <= 0
        for key in required_positive
    ):
        raise ValueError("contract_requirements must be positive integers")
    if target_step % requirements["save_steps"] != 0:
        raise ValueError("target_step must be divisible by save_steps")
    if target_step % requirements["eval_steps"] != 0:
        raise ValueError("target_step must be divisible by eval_steps")
    if foarl_methods:
        method = foarl_methods[0]
        task = policy["expected_tasks"][0]
        profile = get_foarl_f638_campaign_profile(method)
        if target_step != profile.final_step(task):
            raise ValueError(
                f"FOARL policy target_step must equal {profile.final_step(task)} "
                f"for {method}/{task}"
            )
        if requirements["world_size"] != profile.world_size:
            raise ValueError(
                f"FOARL policy world_size must equal {profile.world_size} for {method}"
            )
        if requirements["num_generations"] != profile.semantics.num_generations:
            raise ValueError(
                "FOARL policy num_generations disagrees with its immutable lane"
            )
        if profile.required_train_seed is not None and policy["train_seeds"] != [
            profile.required_train_seed
        ]:
            raise ValueError(
                f"{method} advancement policy must bind only seed "
                f"{profile.required_train_seed}"
            )
    return policy


def _prior_stage_authorization_audit(
    policy: dict[str, Any],
    receipt_paths: Sequence[Path],
) -> dict[str, object]:
    expected_chain = policy["prior_stage_receipt_chain"]
    errors: list[str] = []
    receipts: list[dict[str, object]] = []
    if len(receipt_paths) != len(expected_chain):
        errors.append(
            "supplied prior-stage receipt count does not equal the complete policy chain: "
            f"supplied={len(receipt_paths)}, expected={len(expected_chain)}"
        )
    resolved_paths = [path.expanduser().resolve() for path in receipt_paths]
    if len(resolved_paths) != len(set(resolved_paths)):
        errors.append("the same prior-stage receipt path was supplied more than once")
    for index, expected in enumerate(expected_chain):
        entry_errors: list[str] = []
        entry_result: dict[str, object] = {
            "chain_index": index,
            "expected_identity": expected,
            "status": "FAIL",
            "errors": entry_errors,
        }
        if index >= len(receipt_paths):
            entry_errors.append("required prior-stage receipt was not supplied")
            receipts.append(entry_result)
            errors.extend(f"prior-stage receipt {index}: {error}" for error in entry_errors)
            continue
        raw_path = receipt_paths[index].expanduser()
        entry_result["path"] = str(raw_path.resolve())
        try:
            receipt, file_sha256, size_bytes, mode = _stable_json_object_snapshot(
                raw_path,
                label=f"prior-stage receipt {index}",
            )
            if mode & 0o222:
                entry_errors.append(f"prior-stage receipt is writable (mode {mode:04o})")
            entry_errors.extend(_finite_errors(receipt, path=f"$.prior_stage_receipts[{index}]"))
            stored_receipt_sha256 = receipt.get("receipt_sha256")
            unsigned_receipt = dict(receipt)
            unsigned_receipt.pop("receipt_sha256", None)
            observed_receipt_sha256 = hashlib.sha256(
                _canonical_json_bytes(unsigned_receipt)
            ).hexdigest()
            if (
                not _is_sha256(stored_receipt_sha256)
                or stored_receipt_sha256 != observed_receipt_sha256
            ):
                entry_errors.append("prior-stage receipt self-hash mismatch")
            receipt_policy = receipt.get("policy")
            observed_identity = {
                "campaign_id": receipt.get("campaign_id"),
                "stage_id": receipt.get("stage_id"),
                "policy_sha256": (
                    receipt_policy.get("policy_sha256")
                    if isinstance(receipt_policy, dict)
                    else None
                ),
                "receipt_sha256": stored_receipt_sha256,
                "receipt_file_sha256": file_sha256,
            }
            entry_result.update(
                {
                    "mode": f"{mode:04o}",
                    "size_bytes": size_bytes,
                    "file_sha256": file_sha256,
                    "observed_identity": observed_identity,
                }
            )
            if observed_identity != expected:
                entry_errors.append("prior-stage receipt identity disagrees with policy chain")
            if receipt.get("schema_version") != SCHEMA_VERSION:
                entry_errors.append("prior-stage receipt has an unsupported schema")
            if receipt.get("auditor") != "clasp_sota_advancement":
                entry_errors.append("prior-stage receipt was not produced by this auditor")
            if receipt.get("decision") != "ADVANCE":
                entry_errors.append("prior-stage receipt decision is not ADVANCE")
            if receipt.get("publication_claim_authorized") is not False:
                entry_errors.append("prior-stage receipt has invalid publication authorization")
            if receipt.get("campaign_id") != policy["campaign_id"]:
                entry_errors.append("prior-stage receipt belongs to another campaign")
            if receipt.get("stage_index") != index:
                entry_errors.append("prior-stage receipt stage_index breaks chain order")
            if receipt.get("prior_stage_receipt_chain") != expected_chain[:index]:
                entry_errors.append("prior-stage receipt does not embed the complete chain prefix")
            expected_runs = receipt.get("expected_run_count")
            passed_runs = receipt.get("passed_run_count")
            run_results = receipt.get("runs")
            if (
                isinstance(expected_runs, bool)
                or not isinstance(expected_runs, int)
                or expected_runs <= 0
                or passed_runs != expected_runs
                or not isinstance(run_results, dict)
                or len(run_results) != expected_runs
                or any(
                    not isinstance(result, dict) or result.get("status") != "PASS"
                    for result in run_results.values()
                )
            ):
                entry_errors.append("prior-stage receipt does not prove every scheduled run PASS")
            if receipt.get("errors") != []:
                entry_errors.append("prior-stage receipt contains audit errors")
            comparability = receipt.get("matched_contract_comparability")
            if not isinstance(comparability, dict) or comparability.get("status") != "PASS":
                entry_errors.append("prior-stage receipt did not pass comparability")
            performance = receipt.get("development_performance_gate")
            if not isinstance(performance, dict) or performance.get("status") not in {
                "PASS",
                "NOT_APPLICABLE",
            }:
                entry_errors.append("prior-stage receipt did not pass its performance gate")
        except (OSError, TypeError, ValueError) as exc:
            entry_errors.append(str(exc))
        entry_result["status"] = "PASS" if not entry_errors else "FAIL"
        receipts.append(entry_result)
        errors.extend(f"prior-stage receipt {index}: {error}" for error in entry_errors)
    return {
        "status": "PASS" if not errors else "FAIL",
        "stage_index": policy["stage_index"],
        "initial_stage": policy["stage_index"] == 0,
        "expected_receipt_count": len(expected_chain),
        "validated_receipt_count": sum(receipt["status"] == "PASS" for receipt in receipts),
        "chain_sha256": hashlib.sha256(_canonical_json_bytes(expected_chain)).hexdigest(),
        "receipts": receipts,
        "errors": errors,
    }


def _expected_run_ids(policy: dict[str, Any]) -> list[tuple[str, str, int]]:
    return [
        (str(task), str(method), int(seed))
        for task in policy["expected_tasks"]
        for method in policy["expected_methods"]
        for seed in policy["train_seeds"]
    ]


def _resolve_run_dir(
    run_root: Path,
    template: str,
    *,
    task: str,
    method: str,
    seed: int,
) -> Path:
    try:
        rendered = template.format(task=task, method=method, seed=seed)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid run_path_template: {exc}") from exc
    relative = Path(rendered)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("run_path_template rendered an unsafe path")
    resolved_root = run_root.expanduser().resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("run path escaped --run-root") from exc
    return resolved


def _require_equal(
    container: dict[str, Any],
    expected: dict[str, object],
    *,
    label: str,
) -> list[str]:
    return [
        f"{label}.{key}={container.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if container.get(key) != value
    ]


_FIXED_NULL_RECEIPT_IDENTITY_KEYS = {
    "schema_version",
    "protocol",
    "status",
    "receipt_sha256",
    "receipt_file_sha256",
    "null_candidate_sha256",
    "evaluator_identity_sha256",
    "record_count",
    "tasks",
}


def _validate_fixed_null_receipt_identity(
    value: object,
    *,
    task: str,
    expected_receipt_sha256: str | None,
) -> list[str]:
    if expected_receipt_sha256 is None:
        return [] if value is None else [
            "non-set_null run contract carries a fixed-null receipt identity"
        ]
    if not isinstance(value, dict):
        return ["set_null run contract has no fixed-null receipt identity"]
    errors = _key_set_errors(
        value,
        _FIXED_NULL_RECEIPT_IDENTITY_KEYS,
        label="run_contract.fixed_null_receipt",
    )
    errors.extend(
        _require_equal(
            value,
            {
                "schema_version": 1,
                "protocol": "clasp-fixed-null-construction-v1",
                "status": "frozen_before_training",
                "receipt_sha256": expected_receipt_sha256,
            },
            label="run_contract.fixed_null_receipt",
        )
    )
    for field in (
        "receipt_sha256",
        "receipt_file_sha256",
        "null_candidate_sha256",
        "evaluator_identity_sha256",
    ):
        if not _is_sha256(value.get(field)):
            errors.append(f"run_contract.fixed_null_receipt.{field} is invalid")
    record_count = value.get("record_count")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count <= 0
    ):
        errors.append("run_contract.fixed_null_receipt.record_count must be positive")
    tasks = value.get("tasks")
    if (
        not isinstance(tasks, list)
        or not tasks
        or len(tasks) != len(set(tasks))
        or task not in tasks
    ):
        errors.append(
            "run_contract.fixed_null_receipt.tasks does not cover the run task"
        )
    return errors


def _validate_contract(
    path: Path,
    *,
    task: str,
    method: str,
    seed: int,
    target_step: int,
    requirements: dict[str, int],
    model_initialization_registry_sha256: str,
    campaign_id: str,
    stage_id: str,
    stage_index: int,
    advancement_policy_sha256: str,
    prior_stage_receipt_chain_sha256: str,
    expected_fixed_null_receipt_sha256: str | None,
) -> tuple[dict[str, Any], list[str], dict[str, object]]:
    errors: list[str] = []
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular immutable run contract: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o222:
        errors.append(f"run contract is writable (mode {mode:04o})")
    contract = _load_json(path, expected=dict)
    errors.extend(_finite_errors(contract, path="$.run_contract"))
    errors.extend(_key_set_errors(contract, RUN_CONTRACT_KEYS, label="run_contract"))
    errors.extend(
        _require_equal(
            contract,
            {"schema_version": 2, "stage": "grpo", "method": method, "task": task},
            label="run_contract",
        )
    )
    seeds = contract.get("seeds")
    if not isinstance(seeds, dict):
        errors.append("run_contract.seeds is not an object")
    else:
        errors.extend(
            _key_set_errors(
                seeds,
                {"train", "data", "sft"},
                label="run_contract.seeds",
            )
        )
        errors.extend(
            _require_equal(
                seeds,
                {"train": seed, "data": seed, "sft": requirements["sft_seed"]},
                label="run_contract.seeds",
            )
        )
    fixed = contract.get("fixed_horizon")
    if not isinstance(fixed, dict):
        errors.append("run_contract.fixed_horizon is not an object")
    else:
        errors.extend(
            _key_set_errors(
                fixed,
                {"max_steps", "world_size"},
                label="run_contract.fixed_horizon",
            )
        )
        errors.extend(
            _require_equal(
                fixed,
                {
                    "max_steps": target_step,
                    "world_size": requirements["world_size"],
                },
                label="run_contract.fixed_horizon",
            )
        )
    generation = contract.get("generation")
    if not isinstance(generation, dict):
        errors.append("run_contract.generation is not an object")
    elif generation.get("num_generations") != requirements["num_generations"]:
        errors.append("run_contract.generation.num_generations is not prespecified K")
    training = contract.get("training_config")
    if not isinstance(training, dict):
        errors.append("run_contract.training_config is not an object")
        training = {}
    else:
        errors.extend(
            _require_equal(
                training,
                {
                    "logging_steps": requirements["logging_steps"],
                    "save_steps": requirements["save_steps"],
                    "eval_steps": requirements["eval_steps"],
                    "final_max_steps": target_step,
                    "checkpoint_fidelity": "publication_exact_required",
                    "campaign_id": campaign_id,
                    "advancement_stage_id": stage_id,
                    "advancement_stage_index": stage_index,
                    "advancement_policy_sha256": advancement_policy_sha256,
                    "advancement_prior_stage_receipt_chain_sha256": (
                        prior_stage_receipt_chain_sha256
                    ),
                },
                label="run_contract.training_config",
            )
        )
    environment = contract.get("tracked_environment")
    if not isinstance(environment, dict):
        errors.append("run_contract.tracked_environment is not an object")
        environment = {}
    else:
        errors.extend(
            _require_equal(
                environment,
                {
                    "GRPO_STRICT_OPTIMIZER_RESUME": "1",
                    "CLASP_AUX_GROUPING_STRICT": "1",
                },
                label="run_contract.tracked_environment",
            )
        )
    method_contract = METHOD_CONTRACT[method]
    errors.extend(
        _require_equal(
            training,
            method_contract["training_config"],
            label="run_contract.training_config",
        )
    )
    errors.extend(
        _require_equal(
            environment,
            method_contract["tracked_environment"],
            label="run_contract.tracked_environment",
        )
    )
    if method in CANONICAL_PUBLICATION_METHODS:
        errors.extend(
            _require_equal(
                environment,
                {"CLASP_BON_TARGET_FAILURE_RATES": f"{task}:0.005"},
                label="run_contract.tracked_environment",
            )
        )
    if method in FOARL_PUBLICATION_METHODS:
        try:
            validate_foarl_f638_campaign_contract(
                method=method,
                task=task,
                train_seed=seed,
                final_step=target_step,
                world_size=requirements["world_size"],
                config=training,
                generation_config=generation,
                environment=environment,
            )
        except ValueError as exc:
            errors.append(f"FOARL immutable lane validation failed: {exc}")
    if method == CLASP_FOARL_MATCHED_B9:
        try:
            validate_clasp_foarl_matched_b9_method_contract(
                task=task,
                train_seed=seed,
                fixed_horizon=fixed if isinstance(fixed, dict) else {},
                generation=generation if isinstance(generation, dict) else {},
                training_config=training,
                tracked_environment=environment,
            )
        except ValueError as exc:
            errors.append(f"CLASP/FOARL matched-B9 lane validation failed: {exc}")
    fingerprints = contract.get("fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != {
        "initial_model",
        "base_model",
        "data",
        "source",
    }:
        errors.append("run_contract.fingerprints has an unexpected key set")
    else:
        for name, fingerprint in fingerprints.items():
            errors.extend(
                _validate_fingerprint(
                    fingerprint,
                    label=f"run_contract.fingerprints.{name}",
                )
            )
    data_fingerprint = fingerprints.get("data") if isinstance(fingerprints, dict) else None
    errors.extend(
        _validate_data_lineage(
            contract.get("data_lineage"),
            task=task,
            data_fingerprint=data_fingerprint,
        )
    )
    errors.extend(
        _validate_model_initialization_registry(
            contract.get("model_initialization_registry"),
            task=task,
            seeds=seeds,
            fingerprints=fingerprints,
            data_lineage=contract.get("data_lineage"),
            expected_registry_sha256=model_initialization_registry_sha256,
        )
    )
    errors.extend(
        _validate_fixed_null_receipt_identity(
            contract.get("fixed_null_receipt"),
            task=task,
            expected_receipt_sha256=expected_fixed_null_receipt_sha256,
        )
    )
    fixed_null_environment_path = environment.get("CLASP_FIXED_NULL_RECEIPT_PATH")
    if expected_fixed_null_receipt_sha256 is None:
        if fixed_null_environment_path is not None:
            errors.append(
                "non-set_null run contract carries CLASP_FIXED_NULL_RECEIPT_PATH"
            )
    elif (
        not isinstance(fixed_null_environment_path, str)
        or not os.path.isabs(fixed_null_environment_path)
        or os.path.realpath(fixed_null_environment_path)
        != fixed_null_environment_path
    ):
        errors.append("set_null run contract has no canonical fixed-null receipt path")
    beta = training.get("beta")
    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not math.isfinite(float(beta))
        or float(beta) < 0.0
    ):
        errors.append("run_contract.training_config.beta is not finite and nonnegative")
    else:
        fixed_kl = contract.get("fixed_kl_reference")
        kl_model_path = environment.get("CLASP_KL_REFERENCE_MODEL_PATH")
        kl_receipt_path = environment.get("CLASP_KL_REFERENCE_RECEIPT")
        if float(beta) == 0.0:
            if fixed_kl is not None or kl_model_path is not None or kl_receipt_path is not None:
                errors.append("beta=0 run contract must not bind a fixed-KL reference")
        else:
            try:
                fixed_identity = validate_fixed_kl_reference_identity(fixed_kl)
            except ValueError as exc:
                errors.append(f"beta>0 fixed-KL identity is invalid: {exc}")
            else:
                if (
                    not isinstance(kl_model_path, str)
                    or not os.path.isabs(kl_model_path)
                    or os.path.realpath(kl_model_path) != kl_model_path
                    or not isinstance(kl_receipt_path, str)
                    or not os.path.isabs(kl_receipt_path)
                    or os.path.realpath(kl_receipt_path) != kl_receipt_path
                ):
                    errors.append("beta>0 run contract has no canonical fixed-KL paths")
                else:
                    initialization = contract.get("model_initialization_registry")
                    initial_model = (
                        fingerprints.get("initial_model")
                        if isinstance(fingerprints, dict)
                        else None
                    )
                    expected_model_path = (
                        initial_model.get("canonical_path")
                        if isinstance(initial_model, dict)
                        else None
                    )
                    if (
                        not isinstance(expected_model_path, str)
                        or os.path.realpath(expected_model_path) != kl_model_path
                        or kl_receipt_path
                        != os.path.join(kl_model_path, CHECKPOINT_FIDELITY_FILENAME)
                    ):
                        errors.append("exact KL source is not the policy initialization adapter")
                    specialists = (
                        initialization.get("specialist_sft_by_task")
                        if isinstance(initialization, dict)
                        else None
                    )
                    specialist = specialists.get(task) if isinstance(specialists, dict) else None
                    if not isinstance(specialist, dict) or fixed_identity[
                        "source_checkpoint_fidelity"
                    ] != specialist.get("checkpoint_fidelity"):
                        errors.append(
                            "fixed-KL source differs from approved specialist initialization"
                        )
                    cache_key = (
                        kl_model_path,
                        kl_receipt_path,
                        str(fixed_identity["identity_sha256"]),
                    )
                    try:
                        live_identity = _FIXED_KL_LIVE_CACHE.get(cache_key)
                        if live_identity is None:
                            live_identity = fixed_kl_reference_identity(
                                kl_model_path,
                                kl_receipt_path,
                            )
                            _FIXED_KL_LIVE_CACHE[cache_key] = live_identity
                        if live_identity != fixed_identity:
                            errors.append(
                                "live fixed-KL reference differs from run contract identity"
                            )
                    except (OSError, TypeError, ValueError) as exc:
                        errors.append(f"live fixed-KL verification failed: {exc}")
    receipt = {
        "path": path.name,
        "mode": f"{mode:04o}",
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    return contract, errors, receipt


def _validate_pd_state(path: Path, *, task: str, require_bon: bool) -> list[str]:
    state = _load_json(path, expected=dict)
    errors = _finite_errors(state, path=f"$.{path.name}")
    task_state = state.get(task)
    if not isinstance(task_state, dict):
        return [*errors, f"primal-dual state has no {task!r} object"]
    missing = sorted(PD_REQUIRED - set(task_state))
    if missing:
        errors.append("primal-dual task state is missing: " + ", ".join(missing))
    cost_names = task_state.get("cost_names")
    if not isinstance(cost_names, list) or not cost_names:
        errors.append("primal-dual cost_names is missing or empty")
    else:
        for name in ("cost_limits", "cost_weights", "dual_lambdas"):
            values = task_state.get(name)
            if not isinstance(values, list) or len(values) != len(cost_names):
                errors.append(f"primal-dual {name} does not align with cost_names")
    if require_bon:
        bon = state.get("__clasp_bon__")
        if not isinstance(bon, dict):
            errors.append("CLASP checkpoint has no __clasp_bon__ object")
        else:
            missing_bon = sorted(BON_REQUIRED - set(bon))
            if missing_bon:
                errors.append("CLASP BoN state is missing: " + ", ".join(missing_bon))
            for name in BON_REQUIRED:
                values = bon.get(name)
                if not isinstance(values, dict) or task not in values:
                    errors.append(f"CLASP BoN {name} has no {task!r} value")
    return errors


def method_state_requirements(method: str) -> tuple[bool, bool]:
    """Return the exact checkpoint state sidecars required by a method."""

    if method not in ALL_SUPPORTED_METHODS:
        raise ValueError(f"unsupported training method {method!r}")
    require_pd_state = method not in {"grpo", *FOARL_PUBLICATION_METHODS}
    require_bon_state = method.startswith("clasp_") or method == "set_null"
    return require_pd_state, require_bon_state


def _validate_checkpoint(
    checkpoint: Path,
    *,
    run_dir: Path,
    task: str,
    method: str,
    target_step: int,
    world_size: int,
) -> tuple[
    dict[str, Any],
    list[str],
    list[dict[str, object]],
    dict[str, Any] | None,
]:
    errors: list[str] = []
    match = CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None or checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError(f"invalid checkpoint directory: {checkpoint}")
    step = int(match.group(1))
    if step > target_step:
        errors.append(f"checkpoint step {step} exceeds target {target_step}")
    required = [checkpoint / name for name in CHECKPOINT_REQUIRED]
    model_paths = [checkpoint / name for name in MODEL_ARTIFACTS if (checkpoint / name).is_file()]
    if not model_paths:
        errors.append("checkpoint has no model weight artifact")
    required.extend(model_paths)
    rng_paths = [checkpoint / f"rng_state_{rank}.pth" for rank in range(world_size)]
    required.extend(rng_paths)
    require_pd_state, require_bon_state = method_state_requirements(method)
    if require_pd_state:
        required.append(checkpoint / "multitask_primal_dual_state.json")
    receipts: list[dict[str, object]] = []
    for path in required:
        try:
            receipts.append(_artifact_receipt(path, relative_to=run_dir))
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    checkpoint_fidelity: dict[str, Any] | None = None
    try:
        checkpoint_fidelity = publication_fidelity_identity(checkpoint)
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"checkpoint fidelity is not publication-exact: {exc}")
    trainer_state: dict[str, Any] = {}
    trainer_path = checkpoint / "trainer_state.json"
    if trainer_path.is_file() and trainer_path.stat().st_size > 0:
        trainer_state = _load_json(trainer_path, expected=dict)
        errors.extend(_finite_errors(trainer_state, path=f"$.{checkpoint.name}.trainer_state"))
        if trainer_state.get("global_step") != step:
            errors.append("checkpoint trainer_state.global_step disagrees with its name")
        if trainer_state.get("max_steps") != target_step:
            errors.append("checkpoint trainer_state.max_steps disagrees with target")
        if not isinstance(trainer_state.get("log_history"), list):
            errors.append("checkpoint trainer_state.log_history is missing")
    pd_path = checkpoint / "multitask_primal_dual_state.json"
    if require_pd_state and pd_path.is_file() and pd_path.stat().st_size > 0:
        errors.extend(
            _validate_pd_state(
                pd_path,
                task=task,
                require_bon=require_bon_state,
            )
        )
    return trainer_state, errors, receipts, checkpoint_fidelity


def _validate_exact_runtime_artifact(
    artifact_dir: Path,
    *,
    contract: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Live-verify the post-Trainer receipt required exactly for beta-positive GRPO."""

    errors: list[str] = []
    training = contract.get("training_config")
    horizon = contract.get("fixed_horizon")
    beta = training.get("beta") if isinstance(training, dict) else None
    receipt_path = artifact_dir / EXACT_PEFT_KL_REFERENCE_FILENAME
    post_paths = (
        artifact_dir / exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME,
        artifact_dir / exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME,
    )
    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not math.isfinite(float(beta))
        or float(beta) < 0.0
    ):
        return None, ["cannot project exact PEFT KL runtime from invalid contract beta"]
    if float(beta) == 0.0:
        stale = [path.name for path in (receipt_path, *post_paths) if os.path.lexists(path)]
        if stale:
            errors.append(f"beta=0 artifact carries unexpected exact PEFT KL receipts: {stale}")
        return None, errors
    try:
        runtime = exact_peft_kl_runtime_identity(
            receipt_path,
            expected_fixed_kl_reference=contract.get("fixed_kl_reference"),
        )
    except (OSError, TypeError, ValueError) as exc:
        return None, [f"exact PEFT KL runtime is not publication-exact: {exc}"]
    if (
        runtime["training_beta"] != float(beta)
        or runtime["training_dtype"] != training.get("dtype")
        or not isinstance(horizon, dict)
        or runtime["distributed_world_size"] != horizon.get("world_size")
    ):
        errors.append("exact PEFT KL runtime settings disagree with the run contract")
    return runtime, errors


def _validate_post_step_artifact(
    artifact_dir: Path,
    *,
    contract: dict[str, Any],
    contract_file_sha256: str,
    runtime: dict[str, Any] | None,
    checkpoint_fidelity: dict[str, Any] | None,
    artifact_role: str,
    global_step: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Live-verify one root/checkpoint post-step receipt pair."""

    errors: list[str] = []
    training = contract.get("training_config")
    horizon = contract.get("fixed_horizon")
    beta = training.get("beta") if isinstance(training, dict) else None
    if not isinstance(beta, (int, float)) or isinstance(beta, bool) or float(beta) <= 0.0:
        return None, errors
    if runtime is None or checkpoint_fidelity is None or not isinstance(horizon, dict):
        return None, ["beta>0 post-step audit lacks runtime/fidelity/contract evidence"]
    try:
        step1 = exact_peft_kl_post_step.post_step_reference_identity(
            artifact_dir / exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME,
            claim_scope="publication",
            expected_global_step=1,
            expected_artifact_role="step1",
            rehash_policy_artifact=False,
        )
        post = exact_peft_kl_post_step.post_step_reference_identity(
            artifact_dir / exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME,
            claim_scope="publication",
            expected_global_step=global_step,
            expected_artifact_role=artifact_role,
            expected_policy_trainable_tensor_inventory_sha256=checkpoint_fidelity[
                "trainable_tensor_inventory_sha256"
            ],
        )
    except (OSError, TypeError, ValueError) as exc:
        return None, [f"post-step exact PEFT KL evidence is invalid: {exc}"]
    fixed = contract.get("fixed_kl_reference")
    if not isinstance(fixed, dict):
        errors.append("post-step exact PEFT KL evidence lacks fixed source identity")
        fixed = {}
    identities = (step1, post)
    if (
        step1["invariant"] != post["invariant"]
        or step1["invariant_sha256"] != post["invariant_sha256"]
    ):
        errors.append("step1/post-step within-run invariant differs")
    if any(
        identity["pre_step_receipt_sha256"] != runtime["receipt_sha256"]
        or identity["pre_step_receipt_file_sha256"] != runtime["receipt_file_sha256"]
        or identity["training_run_contract_file_sha256"] != contract_file_sha256
        or identity["training_run_contract_max_steps"] != horizon.get("max_steps")
        or identity["training_run_contract_world_size"] != horizon.get("world_size")
        or identity["world_size"] != runtime["distributed_world_size"]
        or identity["verifier_implementation_sha256"]
        != fixed.get("post_step_verifier_implementation_sha256")
        for identity in identities
    ):
        errors.append("post-step exact PEFT KL lineage differs from runtime/contract/source")
    if (
        post["policy_trainable_tensor_inventory_sha256"]
        != checkpoint_fidelity["trainable_tensor_inventory_sha256"]
    ):
        errors.append("post-step exact PEFT KL policy closure differs from checkpoint fidelity")
    return {"step1": step1, "post_step": post}, errors


def _metric_values(rows: Iterable[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            values.append(float(value))
    return values


def _health_check(
    history: list[object],
    *,
    method: str,
    target_step: int,
    logging_steps: int,
    eval_steps: int,
) -> tuple[list[str], dict[str, object]]:
    errors = _finite_errors(history, path="$.grpo_log_history")
    rows = [row for row in history if isinstance(row, dict)]
    if len(rows) != len(history):
        errors.append("grpo_log_history contains non-object rows")
    train_rows = [row for row in rows if "loss" in row and "grad_norm" in row]
    train_steps = [row.get("step") for row in train_rows]
    expected_steps = list(range(logging_steps, target_step + 1, logging_steps))
    if not expected_steps or expected_steps[-1] != target_step:
        expected_steps.append(target_step)
    if train_steps != expected_steps:
        errors.append(
            "training log step coverage mismatch: "
            f"observed={train_steps}, expected={expected_steps}"
        )
    eval_rows = [row for row in rows if "eval_loss" in row]
    eval_observed = [row.get("step") for row in eval_rows]
    eval_expected = list(range(eval_steps, target_step + 1, eval_steps))
    if eval_observed != eval_expected:
        errors.append(
            "evaluation log step coverage mismatch: "
            f"observed={eval_observed}, expected={eval_expected}"
        )
    losses = _metric_values(train_rows, "loss")
    gradients = _metric_values(train_rows, "grad_norm")
    if len(losses) != len(train_rows):
        errors.append("one or more training loss values is missing or non-finite")
    if len(gradients) != len(train_rows):
        errors.append("one or more grad_norm values is missing or non-finite")
    if not gradients or max(abs(value) for value in gradients) <= 0.0:
        errors.append("gradient gate failed: no nonzero finite grad_norm")

    if method == "grpo" or method in FOARL_PUBLICATION_METHODS:
        advantage_metric = "reward_std"
        objective_metric = "reward_std"
    elif method in {"dual_grpo", "grpo_z", "dual_scalar_z", "dual_separate_z"}:
        advantage_metric = "constrained_grpo/std_reward_advantage"
        objective_metric = "constrained_grpo/mean_objective_quality"
    elif method.startswith("clasp_canonical"):
        advantage_metric = "constrained_grpo/clasp/canonical_coefficient_abs_mean"
        objective_metric = (
            "constrained_grpo/clasp/canonical_quality_coefficient_abs_mean"
        )
    elif method == "set_null":
        advantage_metric = "constrained_grpo/clasp/fixed_null_coefficient_abs_mean"
        objective_metric = (
            "constrained_grpo/clasp/fixed_null_quality_coefficient_abs_mean"
        )
    elif method == "clasp_dense":
        advantage_metric = "constrained_grpo/clasp/advantage_abs_mean"
        objective_metric = "constrained_grpo/clasp/bon_dense_quality_nonzero_rate"
    else:
        advantage_metric = "constrained_grpo/clasp/advantage_abs_mean"
        objective_metric = "constrained_grpo/clasp/bon_success_credit_nonzero_rate"
    advantages = _metric_values(train_rows, advantage_metric)
    objective_credit = _metric_values(train_rows, objective_metric)
    if not advantages or max(abs(value) for value in advantages) <= 0.0:
        errors.append(f"advantage gate failed: {advantage_metric} was never nonzero")
    if not objective_credit or max(abs(value) for value in objective_credit) <= 0.0:
        errors.append(f"objective-credit gate failed: {objective_metric} was never nonzero")

    skip_metric = "constrained_grpo/clasp_group_aux_skipped"
    skipped = _metric_values(rows, skip_metric)
    if any(value != 0.0 for value in skipped):
        errors.append("group/auxiliary skip gate failed")
    fallback = _metric_values(rows, "constrained_grpo/objective_quality_fallback")
    if any(value != 0.0 for value in fallback):
        errors.append("objective quality fallback was used")
    for metric in (
        "constrained_grpo/clasp/adaptive_budget_mixed_unequal",
        "constrained_grpo/clasp/adaptive_budget_global_max_ablation",
    ):
        if any(value != 0.0 for value in _metric_values(rows, metric)):
            errors.append(f"forbidden publication-path metric was active: {metric}")
    if method.startswith("clasp_canonical"):
        exact_metrics = {
            "constrained_grpo/clasp/canonical_set": 1.0,
            "constrained_grpo/clasp/canonical_rollout_batch_size": 9.0,
            "constrained_grpo/clasp/canonical_deployment_set_size": 8.0,
            "constrained_grpo/clasp/canonical_subset_count": 9.0,
            "constrained_grpo/clasp/canonical_token_importance_sampling_ablation": 1.0,
            "constrained_grpo/component_dual_recovery_active": 0.0,
            "constrained_grpo/clasp/bon_controller_commit": 1.0,
        }
        for metric, expected_value in exact_metrics.items():
            values = _metric_values(train_rows, metric)
            if len(values) != len(train_rows) or any(
                value != expected_value for value in values
            ):
                errors.append(
                    f"canonical health metric must equal {expected_value:g} on "
                    f"every train row: {metric}"
                )
        for metric in (
            "constrained_grpo/clasp/canonical_advantage_centering_residual_abs_max",
            "constrained_grpo/clasp/canonical_component_decomposition_residual_abs_max",
        ):
            values = _metric_values(train_rows, metric)
            if len(values) != len(train_rows) or any(value > 1e-5 for value in values):
                errors.append(f"canonical numerical identity gate failed: {metric}")
        recovery_values = _metric_values(
            train_rows,
            "constrained_grpo/clasp/canonical_recovery_scale",
        )
        expected_recovery = 0.0 if method == "clasp_canonical_no_recovery" else 1.0
        if len(recovery_values) != len(train_rows) or any(
            value != expected_recovery for value in recovery_values
        ):
            errors.append("canonical recovery-scale health gate failed")
    elif method == "set_null":
        exact_metrics = {
            "constrained_grpo/clasp/fixed_null_set": 1.0,
            "constrained_grpo/clasp/canonical_set": 0.0,
            "constrained_grpo/clasp/fixed_null_rollout_batch_size": 9.0,
            "constrained_grpo/clasp/fixed_null_deployment_set_size": 8.0,
            "constrained_grpo/clasp/fixed_null_subset_count": 9.0,
            "constrained_grpo/clasp/fixed_null_token_importance_sampling_ablation": 1.0,
            "constrained_grpo/clasp/fixed_null_recovery_scale": 1.0,
            "constrained_grpo/component_dual_recovery_active": 0.0,
            "constrained_grpo/clasp/bon_controller_commit": 1.0,
        }
        for metric, expected_value in exact_metrics.items():
            values = _metric_values(train_rows, metric)
            if len(values) != len(train_rows) or any(
                value != expected_value for value in values
            ):
                errors.append(
                    f"fixed-null health metric must equal {expected_value:g} on "
                    f"every train row: {metric}"
                )
        decomposition = _metric_values(
            train_rows,
            "constrained_grpo/clasp/fixed_null_component_decomposition_residual_abs_max",
        )
        if len(decomposition) != len(train_rows) or any(
            value > 1e-5 for value in decomposition
        ):
            errors.append("fixed-null component decomposition identity gate failed")
    elif method.startswith("clasp_"):
        for metric in (
            "constrained_grpo/component_dual_recovery_active",
            "constrained_grpo/component_dual_recovery_api_available",
            "constrained_grpo/clasp/bon_controller_commit",
            "constrained_grpo/clasp/bon_controller_pressure_observed",
        ):
            values = _metric_values(train_rows, metric)
            if len(values) != len(train_rows) or any(value != 1.0 for value in values):
                errors.append(f"CLASP health metric must equal 1 on every train row: {metric}")
    summary = {
        "history_rows": len(rows),
        "training_rows": len(train_rows),
        "training_steps": train_steps,
        "evaluation_steps": eval_observed,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "max_abs_gradient": max((abs(value) for value in gradients), default=None),
        "advantage_metric": advantage_metric,
        "max_abs_advantage_signal": max((abs(value) for value in advantages), default=None),
        "objective_credit_metric": objective_metric,
        "max_abs_objective_credit": max((abs(value) for value in objective_credit), default=None),
        "max_group_auxiliary_skipped": max(skipped, default=0.0),
    }
    return errors, summary


def _audit_run(
    run_dir: Path,
    *,
    task: str,
    method: str,
    seed: int,
    policy: dict[str, Any],
) -> tuple[dict[str, object], dict[str, Any] | None, bool]:
    run_id = f"{task}/{method}/seed{seed}"
    result: dict[str, object] = {
        "run_id": run_id,
        "path": str(run_dir),
        "status": "FAIL",
        "errors": [],
    }
    errors: list[str] = []
    receipts: list[dict[str, object]] = []
    contract: dict[str, Any] | None = None
    contract_valid = False
    try:
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise ValueError(f"run directory is missing or is a symlink: {run_dir}")
        contract, contract_errors, contract_receipt = _validate_contract(
            run_dir / "run_contract.json",
            task=task,
            method=method,
            seed=seed,
            target_step=policy["target_step"],
            requirements=policy["contract_requirements"],
            model_initialization_registry_sha256=policy["model_initialization_registry_sha256"],
            campaign_id=policy["campaign_id"],
            stage_id=policy["stage_id"],
            stage_index=policy["stage_index"],
            advancement_policy_sha256=policy["policy_sha256"],
            prior_stage_receipt_chain_sha256=hashlib.sha256(
                _canonical_json_bytes(policy["prior_stage_receipt_chain"])
            ).hexdigest(),
            expected_fixed_null_receipt_sha256=(
                policy["fixed_null_receipt_sha256_by_task"][task]
                if method == "set_null"
                else None
            ),
        )
        errors.extend(contract_errors)
        contract_valid = not contract_errors
        result["contract"] = contract_receipt
        result["contract_validation_status"] = "PASS" if contract_valid else "FAIL"

        status_path = run_dir / "stage_status.json"
        status_payload = _load_json(status_path, expected=dict)
        errors.extend(_finite_errors(status_payload, path="$.stage_status"))
        errors.extend(
            _require_equal(
                status_payload,
                {
                    "stage": "grpo",
                    "completed": True,
                    "global_step": policy["target_step"],
                    "max_steps": policy["target_step"],
                    "has_saved_model": True,
                },
                label="stage_status",
            )
        )
        receipts.append(_artifact_receipt(status_path, relative_to=run_dir))
        root_models = [run_dir / name for name in MODEL_ARTIFACTS if (run_dir / name).is_file()]
        if not root_models:
            errors.append("final run directory has no model weight artifact")
        for path in root_models:
            receipts.append(_artifact_receipt(path, relative_to=run_dir))
        for path in (
            run_dir / "adapter_config.json",
            run_dir / CHECKPOINT_FIDELITY_FILENAME,
        ):
            try:
                receipts.append(_artifact_receipt(path, relative_to=run_dir))
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
        root_checkpoint_fidelity: dict[str, Any] | None = None
        try:
            root_checkpoint_fidelity = publication_fidelity_identity(run_dir)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"final root fidelity is not publication-exact: {exc}")
        root_exact_runtime, root_runtime_errors = _validate_exact_runtime_artifact(
            run_dir,
            contract=contract,
        )
        errors.extend(f"final root {error}" for error in root_runtime_errors)
        root_post_step, root_post_errors = _validate_post_step_artifact(
            run_dir,
            contract=contract,
            contract_file_sha256=str(contract_receipt.get("sha256", "")),
            runtime=root_exact_runtime,
            checkpoint_fidelity=root_checkpoint_fidelity,
            artifact_role="final",
            global_step=int(policy["target_step"]),
        )
        errors.extend(f"final root {error}" for error in root_post_errors)
        if root_exact_runtime is not None:
            try:
                receipts.append(
                    _artifact_receipt(
                        run_dir / EXACT_PEFT_KL_REFERENCE_FILENAME,
                        relative_to=run_dir,
                    )
                )
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
            for name in (
                exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME,
                exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME,
            ):
                try:
                    receipts.append(_artifact_receipt(run_dir / name, relative_to=run_dir))
                except (OSError, ValueError) as exc:
                    errors.append(str(exc))

        checkpoint_root = run_dir / "checkpoints"
        if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
            raise ValueError("run has no regular checkpoints directory")
        checkpoint_dirs: list[tuple[int, Path]] = []
        for path in checkpoint_root.iterdir():
            if not path.name.startswith("checkpoint-"):
                continue
            match = CHECKPOINT_RE.fullmatch(path.name)
            if match is None or path.is_symlink() or not path.is_dir():
                errors.append(f"malformed checkpoint entry: {path.name}")
                continue
            checkpoint_dirs.append((int(match.group(1)), path))
        checkpoint_dirs.sort()
        if not checkpoint_dirs:
            raise ValueError("run has no checkpoint-* directories")
        checkpoint_states: dict[int, dict[str, Any]] = {}
        checkpoint_receipts: dict[str, list[dict[str, object]]] = {}
        checkpoint_fidelity_identities: dict[str, dict[str, Any]] = {}
        checkpoint_exact_runtime_identities: dict[str, dict[str, Any]] = {}
        checkpoint_post_step_identities: dict[str, dict[str, Any]] = {}
        for step, checkpoint in checkpoint_dirs:
            state, checkpoint_errors, artifacts, checkpoint_fidelity = _validate_checkpoint(
                checkpoint,
                run_dir=run_dir,
                task=task,
                method=method,
                target_step=policy["target_step"],
                world_size=policy["contract_requirements"]["world_size"],
            )
            checkpoint_states[step] = state
            checkpoint_receipts[str(step)] = artifacts
            if checkpoint_fidelity is not None:
                checkpoint_fidelity_identities[str(step)] = checkpoint_fidelity
            checkpoint_exact_runtime, runtime_errors = _validate_exact_runtime_artifact(
                checkpoint,
                contract=contract,
            )
            if checkpoint_exact_runtime is not None:
                checkpoint_exact_runtime_identities[str(step)] = checkpoint_exact_runtime
                try:
                    artifacts.append(
                        _artifact_receipt(
                            checkpoint / EXACT_PEFT_KL_REFERENCE_FILENAME,
                            relative_to=run_dir,
                        )
                    )
                except (OSError, ValueError) as exc:
                    runtime_errors.append(str(exc))
                if (
                    root_exact_runtime is not None
                    and checkpoint_exact_runtime != root_exact_runtime
                ):
                    runtime_errors.append(
                        "checkpoint exact PEFT KL runtime differs from the root receipt"
                    )
            checkpoint_post_step, post_errors = _validate_post_step_artifact(
                checkpoint,
                contract=contract,
                contract_file_sha256=str(contract_receipt.get("sha256", "")),
                runtime=checkpoint_exact_runtime,
                checkpoint_fidelity=checkpoint_fidelity,
                artifact_role="checkpoint",
                global_step=step,
            )
            if checkpoint_post_step is not None:
                checkpoint_post_step_identities[str(step)] = checkpoint_post_step
                for name in (
                    exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME,
                    exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME,
                ):
                    try:
                        artifacts.append(_artifact_receipt(checkpoint / name, relative_to=run_dir))
                    except (OSError, ValueError) as exc:
                        post_errors.append(str(exc))
                if root_post_step is not None and (
                    checkpoint_post_step["step1"] != root_post_step["step1"]
                    or checkpoint_post_step["post_step"]["invariant"]
                    != root_post_step["post_step"]["invariant"]
                ):
                    post_errors.append(
                        "checkpoint post-step lineage differs from root step1/run invariant"
                    )
            errors.extend(f"checkpoint-{step}: {error}" for error in checkpoint_errors)
            errors.extend(f"checkpoint-{step}: {error}" for error in runtime_errors)
            errors.extend(f"checkpoint-{step}: {error}" for error in post_errors)
        target = policy["target_step"]
        target_checkpoint = run_dir / "checkpoints" / f"checkpoint-{target}"
        if target not in checkpoint_states:
            errors.append(f"terminal checkpoint-{target} is missing")
        terminal_checkpoint_fidelity = checkpoint_fidelity_identities.get(str(target))
        if (
            root_checkpoint_fidelity is not None
            and terminal_checkpoint_fidelity is not None
            and root_checkpoint_fidelity != terminal_checkpoint_fidelity
        ):
            errors.append("final root policy differs from terminal checkpoint fidelity identity")
        terminal_post_step = checkpoint_post_step_identities.get(str(target))
        if (
            root_post_step is not None
            and terminal_post_step is not None
            and (
                root_post_step["post_step"]["policy_trainable_tensor_inventory_sha256"]
                != terminal_post_step["post_step"]["policy_trainable_tensor_inventory_sha256"]
            )
        ):
            errors.append("final root policy differs from terminal checkpoint post-step identity")
        latest = status_payload.get("latest_checkpoint_dir")
        if not isinstance(latest, str) or not latest:
            errors.append("stage_status.latest_checkpoint_dir is missing")
        elif Path(latest).expanduser().resolve() != target_checkpoint.resolve():
            errors.append("stage_status.latest_checkpoint_dir is not the terminal checkpoint")

        root_state_path = run_dir / "grpo_trainer_state.json"
        root_history_path = run_dir / "grpo_log_history.json"
        root_state = _load_json(root_state_path, expected=dict)
        root_history = _load_json(root_history_path, expected=list)
        receipts.append(_artifact_receipt(root_state_path, relative_to=run_dir))
        receipts.append(_artifact_receipt(root_history_path, relative_to=run_dir))
        errors.extend(_finite_errors(root_state, path="$.grpo_trainer_state"))
        if root_state.get("global_step") != target or root_state.get("max_steps") != target:
            errors.append("root trainer state does not end at the target step/horizon")
        if root_state.get("log_history") != root_history:
            errors.append("grpo_trainer_state.log_history differs from grpo_log_history.json")
        target_state = checkpoint_states.get(target, {})
        checkpoint_history = target_state.get("log_history")
        if not isinstance(checkpoint_history, list):
            errors.append("terminal checkpoint has no log history for continuity audit")
        elif root_history[: len(checkpoint_history)] != checkpoint_history:
            errors.append("terminal checkpoint log history is not a prefix of final root history")
        health_errors, health_summary = _health_check(
            root_history,
            method=method,
            target_step=target,
            logging_steps=policy["contract_requirements"]["logging_steps"],
            eval_steps=policy["contract_requirements"]["eval_steps"],
        )
        errors.extend(health_errors)
        result["health"] = health_summary
        result["completion"] = {
            "target_step": target,
            "retained_checkpoint_steps": [step for step, _ in checkpoint_dirs],
            "resume_ready_checkpoint_count": len(checkpoint_dirs),
            "terminal_checkpoint": str(target_checkpoint),
        }
        result["artifacts"] = {
            "run_level": sorted(receipts, key=lambda item: str(item["path"])),
            "checkpoints": checkpoint_receipts,
        }
        result["checkpoint_fidelity"] = {
            "root": root_checkpoint_fidelity,
            "checkpoints": checkpoint_fidelity_identities,
            "terminal_root_equals_checkpoint": (
                root_checkpoint_fidelity is not None
                and root_checkpoint_fidelity == terminal_checkpoint_fidelity
            ),
        }
        result["exact_peft_kl_runtime"] = {
            "root": root_exact_runtime,
            "checkpoints": checkpoint_exact_runtime_identities,
            "all_retained_equal_root": (
                root_exact_runtime is None
                and not checkpoint_exact_runtime_identities
                or root_exact_runtime is not None
                and len(checkpoint_exact_runtime_identities) == len(checkpoint_dirs)
                and all(
                    identity == root_exact_runtime
                    for identity in checkpoint_exact_runtime_identities.values()
                )
            ),
        }
        result["exact_peft_kl_post_step"] = {
            "root": root_post_step,
            "checkpoints": checkpoint_post_step_identities,
            "all_retained_bound_to_root_invariant": (
                root_post_step is None
                and not checkpoint_post_step_identities
                or root_post_step is not None
                and len(checkpoint_post_step_identities) == len(checkpoint_dirs)
                and all(
                    identity["step1"] == root_post_step["step1"]
                    and identity["post_step"]["invariant"]
                    == root_post_step["post_step"]["invariant"]
                    for identity in checkpoint_post_step_identities.values()
                )
            ),
        }
    except (OSError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    result["errors"] = errors
    result["status"] = "PASS" if not errors else "FAIL"
    return result, contract, contract_valid


def _masked_contract(contract: dict[str, Any]) -> dict[str, Any]:
    masked = copy.deepcopy(contract)
    masked.pop("method", None)
    masked.pop("fixed_null_receipt", None)
    for section, keys in COMPARABILITY_MASK.items():
        values = masked.get(section)
        if isinstance(values, dict):
            for key in keys:
                values.pop(key, None)
    return masked


def _comparability_audit(
    policy: dict[str, Any],
    contracts: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, object]:
    groups: dict[str, object] = {}
    errors: list[str] = []
    for task in policy["expected_tasks"]:
        for seed in policy["train_seeds"]:
            group_id = f"{task}/seed{seed}"
            methods = policy["expected_methods"]
            missing = [method for method in methods if (task, method, seed) not in contracts]
            group_errors: list[str] = []
            if missing:
                group_errors.append("missing valid contracts: " + ", ".join(missing))
            else:
                reference_method = methods[0]
                reference = _masked_contract(contracts[(task, reference_method, seed)])
                reference_hash = hashlib.sha256(_canonical_json_bytes(reference)).hexdigest()
                hashes = {reference_method: reference_hash}
                for method in methods[1:]:
                    candidate = _masked_contract(contracts[(task, method, seed)])
                    candidate_hash = hashlib.sha256(_canonical_json_bytes(candidate)).hexdigest()
                    hashes[method] = candidate_hash
                    if candidate != reference:
                        group_errors.append(
                            f"{method} differs from {reference_method} outside the method allowlist"
                        )
                groups[group_id] = {
                    "status": "PASS" if not group_errors else "FAIL",
                    "masked_contract_sha256": hashes,
                    "errors": group_errors,
                }
            if missing:
                groups[group_id] = {
                    "status": "FAIL",
                    "masked_contract_sha256": {},
                    "errors": group_errors,
                }
            errors.extend(f"{group_id}: {error}" for error in group_errors)
    return {"status": "PASS" if not errors else "FAIL", "groups": groups, "errors": errors}


def _development_performance_audit(
    policy: dict[str, Any],
    aggregate_path: Path | None,
) -> dict[str, object]:
    gate = policy.get("development_performance_gate")
    if gate is None:
        return {"status": "NOT_APPLICABLE", "errors": []}
    errors: list[str] = []
    if not isinstance(gate, dict):
        return {"status": "FAIL", "errors": ["performance gate is not an object"]}
    expected_keys = {
        "baseline_method",
        "challenger_method",
        "max_pass_at_k_regression",
        "min_tasks_improved",
        "required_tasks",
    }
    if set(gate) != expected_keys:
        errors.append("development_performance_gate has an unexpected key set")
    if aggregate_path is None:
        errors.append("--development-aggregate is required by this policy")
        return {"status": "FAIL", "errors": errors}
    try:
        aggregate = _load_json(aggregate_path, expected=dict)
    except ValueError as exc:
        return {"status": "FAIL", "errors": [*errors, str(exc)]}
    errors.extend(_finite_errors(aggregate, path="$.development_aggregate"))
    if aggregate.get("schema_version") != 3:
        errors.append("development aggregate has an unsupported schema")
    stored_receipt = aggregate.get("aggregate_receipt_sha256")
    unsigned_aggregate = dict(aggregate)
    unsigned_aggregate.pop("aggregate_receipt_sha256", None)
    observed_receipt = hashlib.sha256(_canonical_json_bytes(unsigned_aggregate)).hexdigest()
    if not _is_sha256(stored_receipt) or stored_receipt != observed_receipt:
        errors.append("development aggregate receipt self-hash mismatch")
    if aggregate.get("evidence_scope") != "development_only":
        errors.append("pilot performance gate requires development_only evidence")
    if (
        aggregate.get("expected_candidate_budget")
        != policy["contract_requirements"]["num_generations"]
    ):
        errors.append("development aggregate candidate budget disagrees with training policy")
    comparison = aggregate.get("comparison")
    baseline = gate.get("baseline_method")
    challenger = gate.get("challenger_method")
    if (
        not isinstance(comparison, dict)
        or comparison.get("baseline") != baseline
        or comparison.get("challenger") != challenger
    ):
        errors.append("development aggregate does not contain the prespecified comparison")
    summaries = aggregate.get("summaries")
    task_rows: dict[str, object] = {}
    improved = 0
    if not isinstance(summaries, dict):
        errors.append("development aggregate has no summaries")
    else:
        for task in gate.get("required_tasks", []):
            task_errors: list[str] = []
            try:
                baseline_seed = summaries[baseline][task]["per_seed"]
                challenger_seed = summaries[challenger][task]["per_seed"]
            except (KeyError, TypeError):
                errors.append(f"development aggregate is missing {task} summaries")
                continue
            expected_seeds = {str(seed) for seed in policy["train_seeds"]}
            if set(baseline_seed) != expected_seeds or set(challenger_seed) != expected_seeds:
                task_errors.append("per-seed denominator does not match the scheduled seeds")
            seed_results: dict[str, object] = {}
            task_improved = True
            for seed in sorted(expected_seeds):
                base = baseline_seed.get(seed, {})
                chal = challenger_seed.get(seed, {})
                base_pass = base.get("pass_at_k") if isinstance(base, dict) else None
                chal_pass = chal.get("pass_at_k") if isinstance(chal, dict) else None
                base_gap = base.get("conditional_gap_mean") if isinstance(base, dict) else None
                chal_gap = chal.get("conditional_gap_mean") if isinstance(chal, dict) else None
                values = (base_pass, chal_pass)
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in values
                ):
                    task_errors.append(f"seed{seed} has invalid pass@K metrics")
                    task_improved = False
                    continue
                pass_delta = float(chal_pass) - float(base_pass)
                if pass_delta < -float(gate.get("max_pass_at_k_regression", -1.0)):
                    task_errors.append(
                        f"seed{seed} pass@K regression {pass_delta:.6f} exceeds gate"
                    )
                gap_improved = bool(
                    isinstance(base_gap, (int, float))
                    and not isinstance(base_gap, bool)
                    and isinstance(chal_gap, (int, float))
                    and not isinstance(chal_gap, bool)
                    and math.isfinite(float(base_gap))
                    and math.isfinite(float(chal_gap))
                    and float(chal_gap) < float(base_gap)
                )
                seed_improved = pass_delta > 0.0 or gap_improved
                task_improved = task_improved and seed_improved
                seed_results[seed] = {
                    "challenger_minus_baseline_pass_at_k": pass_delta,
                    "challenger_minus_baseline_conditional_gap": (
                        float(chal_gap) - float(base_gap) if gap_improved else None
                    ),
                    "improved": seed_improved,
                }
            if task_improved and not task_errors:
                improved += 1
            task_rows[str(task)] = {
                "status": "PASS" if not task_errors else "FAIL",
                "improved_for_all_scheduled_seeds": task_improved,
                "seeds": seed_results,
                "errors": task_errors,
            }
            errors.extend(f"{task}: {error}" for error in task_errors)
    minimum = gate.get("min_tasks_improved")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
        errors.append("min_tasks_improved is invalid")
    elif improved < minimum:
        errors.append(f"only {improved} tasks improved; policy requires at least {minimum}")
    artifact = None
    if aggregate_path.is_file():
        artifact = {
            "path": str(aggregate_path.resolve()),
            "size_bytes": aggregate_path.stat().st_size,
            "sha256": _sha256_file(aggregate_path),
        }
    return {
        "status": "PASS" if not errors else "FAIL",
        "artifact": artifact,
        "improved_task_count": improved,
        "tasks": task_rows,
        "errors": errors,
    }


def audit_campaign(
    *,
    policy_path: Path,
    run_root: Path,
    development_aggregate: Path | None = None,
    prior_stage_receipts: Sequence[Path] = (),
) -> dict[str, object]:
    policy = _validate_policy(policy_path.expanduser().resolve())
    run_root = run_root.expanduser().resolve()
    run_results: dict[str, object] = {}
    contracts: dict[tuple[str, str, int], dict[str, Any]] = {}
    prior_authorization = _prior_stage_authorization_audit(
        policy,
        prior_stage_receipts,
    )
    errors = [str(error) for error in prior_authorization["errors"]]
    for task, method, seed in _expected_run_ids(policy):
        run_id = f"{task}/{method}/seed{seed}"
        try:
            run_dir = _resolve_run_dir(
                run_root,
                policy["run_path_template"],
                task=task,
                method=method,
                seed=seed,
            )
            result, contract, contract_valid = _audit_run(
                run_dir,
                task=task,
                method=method,
                seed=seed,
                policy=policy,
            )
        except (OSError, TypeError, ValueError) as exc:
            result = {
                "run_id": run_id,
                "path": None,
                "status": "FAIL",
                "errors": [str(exc)],
            }
            contract = None
            contract_valid = False
        run_results[run_id] = result
        if contract is not None and contract_valid:
            contracts[(task, method, seed)] = contract
        if result["status"] != "PASS":
            errors.extend(f"{run_id}: {error}" for error in result["errors"])
    comparability = _comparability_audit(policy, contracts)
    errors.extend(str(error) for error in comparability["errors"])
    performance = _development_performance_audit(policy, development_aggregate)
    errors.extend(str(error) for error in performance["errors"])
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "auditor": "clasp_sota_advancement",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "campaign_id": policy["campaign_id"],
        "stage_id": policy["stage_id"],
        "stage_index": policy["stage_index"],
        "prior_stage_receipt_chain": policy["prior_stage_receipt_chain"],
        "decision": "ADVANCE" if not errors else "HOLD",
        "publication_claim_authorized": False,
        "policy": {
            "path": str(policy_path.expanduser().resolve()),
            "file_sha256": _sha256_file(policy_path.expanduser().resolve()),
            "policy_sha256": policy["policy_sha256"],
            "frozen_at_utc": policy["frozen_at_utc"],
            "model_initialization_registry_sha256": policy["model_initialization_registry_sha256"],
            "prior_stage_receipt_chain_sha256": prior_authorization["chain_sha256"],
        },
        "run_root": str(run_root),
        "expected_run_count": len(_expected_run_ids(policy)),
        "passed_run_count": sum(
            result["status"] == "PASS"
            for result in run_results.values()  # type: ignore[index]
        ),
        "runs": run_results,
        "matched_contract_comparability": comparability,
        "development_performance_gate": performance,
        "prior_stage_authorization": prior_authorization,
        "errors": errors,
        "side_effects": {
            "run_directories_modified": False,
            "scheduler_actions": [],
            "jobs_submitted": 0,
            "jobs_cancelled": 0,
        },
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def _atomic_json(path: Path, payload: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing audit receipt: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to replace existing audit receipt: {path}") from exc
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--development-aggregate", type=Path)
    parser.add_argument(
        "--prior-stage-receipt",
        action="append",
        default=[],
        type=Path,
        help="immutable prior ADVANCE receipt, repeated in chronological chain order",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = audit_campaign(
            policy_path=args.policy,
            run_root=args.run_root,
            development_aggregate=args.development_aggregate,
            prior_stage_receipts=args.prior_stage_receipt,
        )
    except (OSError, TypeError, ValueError) as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "auditor": "clasp_sota_advancement",
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "decision": "HOLD",
            "publication_claim_authorized": False,
            "errors": [str(exc)],
            "side_effects": {
                "run_directories_modified": False,
                "scheduler_actions": [],
                "jobs_submitted": 0,
                "jobs_cancelled": 0,
            },
        }
        payload["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    try:
        _atomic_json(args.output, payload)
    except FileExistsError as exc:
        print(f"HOLD: {exc}", file=sys.stderr)
        return 2
    print(f"{payload['decision']}: wrote {args.output}")
    if payload["decision"] != "ADVANCE":
        for error in payload.get("errors", [])[:20]:
            print(f"  - {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
