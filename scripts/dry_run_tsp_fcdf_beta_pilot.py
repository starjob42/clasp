#!/usr/bin/env python3
"""Audit and render the frozen TSP Frontier-CDF x beta pilot lifecycle.

There is intentionally no general execute flag or scheduler integration.  The
only child-process surface is the claim-bound one-update memory-gate supervisor;
matrix training and evaluation remain exact, preauthorized exec transitions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import shlex
import signal
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import freeze_tsp_fcdf_beta_pilot as contractlib

BASE_ENV = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "3",
    "CLASP_AUX_GROUPING_STRICT": "1",
    "CLASP_BON_ALIGNED": "1",
    "CLASP_BON_DENSE_FEASIBILITY_SCALE": "1.0",
    "CLASP_BON_DENSE_QUALITY_SCALE": "1.0",
    "CLASP_BON_DENSE_VIOLATION_MIX": "0.5",
    "CLASP_BON_RECOVERY_SCALE": "1.0",
    "CLASP_BON_SUCCESS_CREDIT": "dense_rank",
    "CLASP_BON_TARGET_FAILURE_RATE": "0.005",
    "CLASP_BON_TARGET_FAILURE_RATES": "tsp:0.005",
    "CLASP_COMPONENT_DUAL_RECOVERY_SCALE": "1.0",
    "CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT": "0.0",
    "CLASP_DEVELOPMENT_FIDELITY": "1",
    "CLASP_FRONTIER_CDF_FEASIBILITY_VIOLATION_TOLERANCE": "1e-8",
    "CLASP_FRONTIER_CDF_GAP_SLOPE": "3.0",
    "CLASP_FRONTIER_CDF_GAP_TARGETS": "tsp:0.0107",
    "CLASP_FRONTIER_CDF_OBJECTIVE_SCORE_SCALE": "2.0",
    "CLASP_FRONTIER_CDF_QUALITY_ANNEAL_FRACTION": "0.6",
    "CLASP_FRONTIER_CDF_QUALITY_TEMPERATURE_FINAL": "0.15",
    "CLASP_FRONTIER_CDF_QUALITY_TEMPERATURE_INITIAL": "0.5",
    "CLASP_FRONTIER_CDF_RECOVERY_SCALE": "1.0",
    "CLASP_FRONTIER_CDF_RELATIVE_RECOVERY_SCALE": "1.0",
    "CLASP_FRONTIER_CDF_VIOLATION_TEMPERATURE": "0.15",
    "CLASP_LORA_BIAS": "all",
    "CLASP_LORA_TARGET_MODULES": "jiang",
    "CLASP_OBJECTIVE_GAP_SLOPE": "3.0",
    "CLASP_OBJECTIVE_QUALITY_MODE": "exp",
    "CLASP_PROPER_DUAL_GRADIENT_MODE": "signed",
    "CLASP_PUBLICATION_FIDELITY": "0",
    "CLASP_TSP_PROMPT_KNN": "0",
    "CLASP_TSP_PROMPT_KNN_DISTANCES": "0",
    "CLASP_TSP_ROUTE_ONLY_OUTPUT": "0",
    "CLASP_TSP_EDGE_OVERLAP_MIX": "0",
    "CLASP_TSP_TARGET_GAP_BONUS": "0",
    "CLASP_TSP_TARGET_GAP_EXCESS_PENALTY": "0",
    "CLASP_TSP_TWO_OPT_PENALTY": "0",
    "GRPO_STRICT_OPTIMIZER_RESUME": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "HOME": "/home/lamsade/habgaryan",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LOGNAME": "habgaryan",
    "PATH": contractlib.CLOSED_ENVIRONMENT_PATH,
    "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION": "0",
    "PD_IMPORTANCE_SAMPLING_LEVEL": "sequence",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "USER": "habgaryan",
    "WANDB_MODE": "disabled",
}

FRONTIER_ENV = {
    "CLASP_FRONTIER_CDF": "1",
    "CLASP_FRONTIER_CDF_RELATIVE_FAILED_RECOVERY": "1",
}

DENSE_ENV = {
    "CLASP_FRONTIER_CDF": "0",
    "CLASP_FRONTIER_CDF_RELATIVE_FAILED_RECOVERY": "0",
}

ALWAYS_UNSET = [
    "RESUME_FROM_CHECKPOINT",
    "GRPO_STOP_AT_GLOBAL_STEP",
    "GRPO_REBASE_LOADED_SCHEDULER_AT_STEP",
    "GRPO_EXTENSION_BASE_STEP",
    "GRPO_EXTENSION_FINAL_STEP",
    "GRPO_EXTENSION_PEAK_LR",
    "GRPO_EXTENSION_START_LR",
    "GRPO_EXTENSION_WARMUP_STEPS",
]

KL_ENV_NAMES = [
    "CLASP_KL_REFERENCE_MODEL_PATH",
    "CLASP_KL_REFERENCE_RECEIPT",
]
TRAINING_RUN_CONTRACT_ENV = "CLASP_TRAINING_RUN_CONTRACT_PATH"
MEMORY_GATE_DIRNAME = contractlib.MEMORY_GATE_DIRNAME
MEMORY_GATE_PASS_FILENAME = contractlib.MEMORY_GATE_PASS_FILENAME
MATRIX_AUTHORIZATION_FILENAME = contractlib.MATRIX_AUTHORIZATION_FILENAME
MEMORY_GATE_PASS_PROTOCOL = contractlib.MEMORY_GATE_PASS_PROTOCOL
MATRIX_AUTHORIZATION_PROTOCOL = contractlib.MATRIX_AUTHORIZATION_PROTOCOL
MEMORY_GATE_PASS_STATUS = "development_memory_gate_passed"
MATRIX_AUTHORIZATION_STATUS = "development_full_matrix_authorized_pre_outcome"
MEMORY_GATE_ATTEMPT_FILENAME = "memory_gate_attempt_started.json"
MEMORY_GATE_ATTEMPT_PROTOCOL = "clasp-tsp-fcdf-beta-memory-gate-attempt-v2"
MEMORY_GATE_TRANSCRIPT_FILENAME = "memory_gate_execution.transcript"
MEMORY_GATE_OUTCOME_FILENAME = "memory_gate_execution_outcome.json"
MEMORY_GATE_ANCHOR_GUARD_DIRNAME = "_generation_guards"
MEMORY_GATE_OUTCOME_PROTOCOL = "clasp-tsp-fcdf-beta-memory-gate-execution-v1"
MEMORY_GATE_TRANSCRIPT_PROTOCOL = "clasp-tsp-fcdf-beta-memory-gate-transcript-v1"
MEMORY_GATE_SUCCESS_STATUS = "development_memory_gate_child_succeeded"
MEMORY_GATE_FAILURE_STATUS = "development_memory_gate_child_failed"
MEMORY_GATE_SUPERVISOR_FAILURE_STATUS = "development_memory_gate_supervisor_failed"
MEMORY_GATE_UNKNOWN_STATUS = "development_memory_gate_exit_unknown"
CELL_AUTHORIZATION_FILENAME = "matrix_cell_authorization.json"
CELL_AUTHORIZATION_PROTOCOL = "clasp-tsp-fcdf-beta-cell-reservation-v1"
EVALUATION_AUTHORIZATION_FILENAME = "matrix_evaluation_authorization.json"
EVALUATION_AUTHORIZATION_PROTOCOL = "clasp-tsp-fcdf-beta-evaluation-reservation-v1"
RESERVATION_CLOSURE_FILENAME = "tsp_fcdf_beta_matrix_reservation_closure.json"
RESERVATION_CLOSURE_PROTOCOL = "clasp-tsp-fcdf-beta-matrix-reservation-closure-v1"
RESERVATION_CLOSURE_STATUS = "development_full_matrix_reserved_pre_outcome"
TRAINING_ATTEMPT_FILENAME = "matrix_training_attempt_started.json"
TRAINING_ATTEMPT_PROTOCOL = "clasp-tsp-fcdf-beta-training-attempt-v1"
TRAINING_RESUME_PREFIX = "matrix_training_resume_checkpoint_"
EVALUATION_ATTEMPT_FILENAME = "matrix_evaluation_attempt_started.json"
EVALUATION_ATTEMPT_PROTOCOL = "clasp-tsp-fcdf-beta-evaluation-attempt-v1"
TERMINAL_AUDIT_FILENAME = "matrix_cell_terminal_audit.json"
TERMINAL_AUDIT_PROTOCOL = "clasp-tsp-fcdf-beta-cell-terminal-audit-v1"
TERMINAL_AUDIT_STATUS = "development_terminal_policy_fidelity_verified"
CLAIM_ANCHOR_DIRNAME = "_claim_anchors"
MIN_WORKSPACE_FREE_BYTES = contractlib.MIN_WORKSPACE_FREE_BYTES


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _self_hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_sha256"] = contractlib.sha256_json(result)
    return result


def _verify_self_hash(payload: Mapping[str, Any], *, label: str) -> None:
    claimed = payload.get("receipt_sha256")
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    if claimed != contractlib.sha256_json(unsigned):
        contractlib.fail(f"{label} self-hash mismatch")


def _file_identity(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "protocol": payload.get("protocol"),
        "status": payload.get("status"),
        "receipt_sha256": payload.get("receipt_sha256"),
        "file_sha256": contractlib.sha256_file(path),
        "payload_sha256": contractlib.sha256_json(payload),
        "size_bytes": path.stat().st_size,
    }


def _path_stat_identity(path: Path) -> dict[str, Any]:
    """Bind one local reservation inode so delete/recreate cannot pass by byte copy."""

    if path.is_symlink() or not path.exists():
        contractlib.fail(f"cannot bind missing or symlinked reservation path: {path}")
    observed = path.lstat()
    return {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file" if path.is_file() else "other",
        "device": observed.st_dev,
        "inode": observed.st_ino,
    }


def _path_directory_change_identity(path: Path) -> dict[str, Any]:
    """Bind directory metadata that child create/delete/rename operations change."""

    stable = _path_stat_identity(path)
    if stable["kind"] != "directory":
        contractlib.fail(f"claim-anchor path is not a directory: {path}")
    observed = path.lstat()
    return {
        **stable,
        "mode": stat.S_IMODE(observed.st_mode),
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
    }


def _claim_anchor_names(kind: str) -> tuple[str, ...]:
    if kind == "training":
        return (
            TRAINING_ATTEMPT_FILENAME,
            *(f"{TRAINING_RESUME_PREFIX}{step}.json" for step in range(1, 400)),
        )
    if kind == "evaluation":
        return (EVALUATION_ATTEMPT_FILENAME,)
    contractlib.fail(f"unsupported claim-anchor kind: {kind}")


def _claim_anchor_path(reserved_path: Path, name: str) -> Path:
    if not name or Path(name).name != name:
        contractlib.fail(f"invalid claim-anchor filename: {name!r}")
    return reserved_path / CLAIM_ANCHOR_DIRNAME / name


def _create_empty_claim_anchor(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o660)
    try:
        os.fchmod(descriptor, 0o660)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_claim_anchor_tree(reserved_path: Path, *, kind: str) -> None:
    """Precreate every single-use claim inode before reservation closure."""

    root = reserved_path / CLAIM_ANCHOR_DIRNAME
    if root.exists() or root.is_symlink():
        if root.is_symlink() or not root.is_dir():
            contractlib.fail(f"claim-anchor root is not a regular directory: {root}")
        mode = stat.S_IMODE(root.lstat().st_mode)
        if mode not in {0o700, 0o550}:
            contractlib.fail(f"claim-anchor root has unsafe mode {oct(mode)}: {root}")
    else:
        root.mkdir(mode=0o700)
        mode = 0o700
    expected = set(_claim_anchor_names(kind))
    observed = {path.name for path in root.iterdir()}
    if observed - expected:
        contractlib.fail(f"claim-anchor root contains unknown entries: {root}")
    if mode == 0o550 and observed != expected:
        contractlib.fail(f"sealed claim-anchor root is incomplete: {root}")
    for name in sorted(expected - observed):
        _create_empty_claim_anchor(root / name)
    for name in sorted(expected):
        path = root / name
        if path.is_symlink() or not path.is_file():
            contractlib.fail(f"claim anchor is not a regular file: {path}")
        observed_stat = path.lstat()
        if observed_stat.st_size != 0 or stat.S_IMODE(observed_stat.st_mode) != 0o660:
            contractlib.fail(f"pre-outcome claim anchor is already consumed: {path}")
    root.chmod(0o550)
    directory_descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _partial_claim_anchor_tree_is_pristine(reserved_path: Path, *, kind: str) -> None:
    root = reserved_path / CLAIM_ANCHOR_DIRNAME
    if not root.exists() and not root.is_symlink():
        return
    if root.is_symlink() or not root.is_dir():
        contractlib.fail(f"partial claim-anchor root is invalid: {root}")
    expected = set(_claim_anchor_names(kind))
    observed = {path.name for path in root.iterdir()}
    if observed - expected:
        contractlib.fail(f"partial claim-anchor root contains unknown entries: {root}")
    mode = stat.S_IMODE(root.lstat().st_mode)
    if mode == 0o550 and observed != expected:
        contractlib.fail(f"sealed partial claim-anchor root is incomplete: {root}")
    if mode not in {0o700, 0o550}:
        contractlib.fail(f"partial claim-anchor root has unsafe mode: {root}")
    for path in root.iterdir():
        observed_stat = path.lstat()
        if (
            path.is_symlink()
            or not path.is_file()
            or observed_stat.st_size != 0
            or stat.S_IMODE(observed_stat.st_mode) != 0o660
        ):
            contractlib.fail(f"partial claim anchor contains outcome bytes: {path}")


def _initial_claim_anchor_manifest(reserved_path: Path, *, kind: str) -> dict[str, Any]:
    root = reserved_path / CLAIM_ANCHOR_DIRNAME
    expected = _claim_anchor_names(kind)
    if {path.name for path in root.iterdir()} != set(expected):
        contractlib.fail(f"claim-anchor inventory is incomplete: {root}")
    records = []
    for name in expected:
        path = root / name
        observed = path.lstat()
        if (
            path.is_symlink()
            or not path.is_file()
            or observed.st_size != 0
            or stat.S_IMODE(observed.st_mode) != 0o660
        ):
            contractlib.fail(f"claim anchor is not pristine before closure: {path}")
        records.append(
            {
                **_path_stat_identity(path),
                "initial_mode": 0o660,
                "initial_size_bytes": 0,
            }
        )
    directory = _path_directory_change_identity(root)
    if directory["mode"] != 0o550:
        contractlib.fail(f"claim-anchor directory is not sealed: {root}")
    return {
        "kind": kind,
        "directory": directory,
        "anchor_count": len(records),
        "anchors": records,
    }


def _inspect_claim_anchor_manifest(
    reserved_path: Path,
    *,
    kind: str,
    frozen: object,
) -> dict[str, Any]:
    if not isinstance(frozen, dict):
        contractlib.fail("matrix reservation closure lacks claim-anchor evidence")
    expected_names = _claim_anchor_names(kind)
    root = reserved_path / CLAIM_ANCHOR_DIRNAME
    if (
        frozen.get("kind") != kind
        or frozen.get("anchor_count") != len(expected_names)
        or frozen.get("directory") != _path_directory_change_identity(root)
    ):
        contractlib.fail(f"claim-anchor directory was deleted/recreated/altered: {root}")
    records = frozen.get("anchors")
    if not isinstance(records, list) or len(records) != len(expected_names):
        contractlib.fail(f"claim-anchor record count differs from closure: {root}")
    observed_names = {path.name for path in root.iterdir()}
    if observed_names != set(expected_names):
        contractlib.fail(f"claim-anchor inventory was deleted/recreated: {root}")
    for name, record in zip(expected_names, records, strict=True):
        if not isinstance(record, dict):
            contractlib.fail(f"claim-anchor closure record is malformed: {root}")
        path = root / name
        stable = _path_stat_identity(path)
        expected_stable = {key: record.get(key) for key in ("path", "kind", "device", "inode")}
        if (
            stable != expected_stable
            or record.get("initial_mode") != 0o660
            or record.get("initial_size_bytes") != 0
        ):
            contractlib.fail(f"claim-anchor inode was deleted/recreated: {path}")
        observed = path.lstat()
        mode = stat.S_IMODE(observed.st_mode)
        if observed.st_size == 0:
            if mode != 0o660:
                contractlib.fail(f"unused claim anchor has unsafe mode: {path}")
        elif mode == 0o440:
            payload = contractlib.load_json_object(path, label="consumed claim anchor")
            _verify_self_hash(payload, label="consumed claim anchor")
        else:
            contractlib.fail(f"claim anchor is partially consumed or writable: {path}")
    return dict(frozen)


def _contract_identity(
    contract: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    expected = Path(str(contract["paths"]["frozen_contract_path"]))
    observed = contractlib.canonical_contract_path(
        str(contract_path),
        require_absent=False,
    )
    if observed != expected:
        contractlib.fail("training run contract path differs from frozen preregistration")
    contractlib.require_read_only(observed, label="frozen pilot contract")
    reopened = contractlib.load_json_object(observed, label="frozen pilot contract")
    if reopened != dict(contract):
        contractlib.fail("frozen pilot contract bytes decode to a different payload")
    return _file_identity(observed, reopened)


def _top_level_names(root: Path) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        contractlib.fail(f"pilot output_root is not a regular directory: {root}")
    names: set[str] = set()
    for child in root.iterdir():
        if child.is_symlink():
            contractlib.fail(f"pilot output_root contains a symlink: {child}")
        names.add(child.name)
    return names


def _require_top_level(root: Path, expected: set[str], *, phase: str) -> None:
    observed = _top_level_names(root)
    if observed != expected:
        contractlib.fail(
            f"{phase} output-root contamination: observed={sorted(observed)} "
            f"expected={sorted(expected)}"
        )


def _seal_tree_read_only(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        contractlib.fail(f"cannot seal non-directory memory-gate tree: {root}")
    entries = sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True)
    for path in entries:
        if path.is_symlink():
            contractlib.fail(f"memory-gate tree contains a symlink: {path}")
        desired_mode = 0o550 if path.is_dir() else 0o440
        if stat.S_IMODE(path.lstat().st_mode) != desired_mode:
            path.chmod(desired_mode)
    if stat.S_IMODE(root.lstat().st_mode) != 0o550:
        root.chmod(0o550)
    contractlib.require_read_only(root, label="memory-gate tree", recursive=True)


def _stable_gate_manifest(root: Path) -> dict[str, Any]:
    first = contractlib.tree_manifest(root, include_size_total=True)
    second = contractlib.tree_manifest(root, include_size_total=True)
    if first != second:
        contractlib.fail("memory-gate tree changed while its full manifest was computed")
    return first


def _tree_manifest_excluding_exact_paths(
    root: Path,
    excluded: Sequence[Path],
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        contractlib.fail(f"manifest root must be a regular directory: {root}")
    excluded_paths = set(excluded)
    if not excluded_paths:
        contractlib.fail("failure manifest requires at least one exact excluded path")
    for path in excluded_paths:
        try:
            path.relative_to(root)
        except ValueError:
            contractlib.fail(f"excluded manifest path is outside its root: {path}")
    files = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            contractlib.fail(f"manifest root contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            contractlib.fail(f"manifest root contains a non-regular entry: {path}")
        if path in excluded_paths:
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": contractlib.sha256_file(path),
            }
        )
    if not files:
        contractlib.fail("manifest root contains no evidence outside exact exclusions")
    return {
        "file_count": len(files),
        "size_bytes": sum(record["size_bytes"] for record in files),
        "tree_sha256": contractlib.sha256_json(files),
        "files": files,
    }


def _tree_manifest_excluding_exact_path(root: Path, excluded: Path) -> dict[str, Any]:
    return _tree_manifest_excluding_exact_paths(root, [excluded])


def _stable_tree_manifest_excluding_exact_path(
    root: Path,
    excluded: Path,
) -> dict[str, Any]:
    first = _tree_manifest_excluding_exact_path(root, excluded)
    second = _tree_manifest_excluding_exact_path(root, excluded)
    if first != second:
        contractlib.fail("memory-gate failure tree changed while its manifest was computed")
    return first


def _stable_failure_tree_manifest(
    root: Path,
    *,
    outcome_path: Path,
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    anchors = claim.get("execution_anchors")
    frozen_outcome = anchors.get("outcome") if isinstance(anchors, dict) else None
    guard = frozen_outcome.get("generation_guard") if isinstance(frozen_outcome, dict) else None
    guard_value = guard.get("path") if isinstance(guard, dict) else None
    if not isinstance(guard_value, str) or not guard_value:
        contractlib.fail("outcome hard-link generation guard is absent from the claim")
    guard_path = Path(guard_value)
    primary_stat = outcome_path.lstat()
    guard_stat = guard_path.lstat()
    if (
        outcome_path.is_symlink()
        or guard_path.is_symlink()
        or not outcome_path.is_file()
        or not guard_path.is_file()
        or (primary_stat.st_dev, primary_stat.st_ino) != (guard_stat.st_dev, guard_stat.st_ino)
        or primary_stat.st_nlink != 2
        or guard_stat.st_nlink != 2
    ):
        contractlib.fail("outcome hard-link generation guard differs before failure manifest")
    excluded_paths = [outcome_path, guard_path]
    first = _tree_manifest_excluding_exact_paths(root, excluded_paths)
    second = _tree_manifest_excluding_exact_paths(root, excluded_paths)
    if first != second:
        contractlib.fail("memory-gate failure tree changed while its manifest was computed")
    return {
        "manifest": first,
        "excluded_outcome_hardlinks": [
            {
                "path": str(path),
                "device": primary_stat.st_dev,
                "inode": primary_stat.st_ino,
                "expected_link_count": 2,
            }
            for path in excluded_paths
        ],
    }


def _validate_required_numeric_metrics(
    record: Mapping[str, Any],
    *,
    required: Sequence[str],
    label: str,
) -> None:
    """Require real finite scalars and semantic nonnegativity (loss may be signed)."""

    missing = sorted(set(required) - set(record))
    if missing:
        contractlib.fail(f"{label} telemetry is incomplete: {missing}")
    invalid: dict[str, Any] = {}
    negative: dict[str, Any] = {}
    for key in required:
        value = record[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            invalid[key] = value
        elif key != "loss" and float(value) < 0.0:
            negative[key] = value
    if invalid:
        contractlib.fail(f"{label} required telemetry is not finite numeric data: {invalid}")
    if negative:
        contractlib.fail(f"{label} nonnegative telemetry is negative: {negative}")


def _audit_step1_trainer_state(path: Path) -> dict[str, Any]:
    state = contractlib.load_json_object(path, label="memory-gate trainer state")
    if (
        state.get("global_step") != 1
        or isinstance(state.get("global_step"), bool)
        or not isinstance(state.get("global_step"), int)
    ):
        contractlib.fail("memory-gate trainer_state global_step must be exactly 1")
    history = state.get("log_history")
    if not isinstance(history, list) or not history:
        contractlib.fail("memory-gate trainer_state lacks optimizer-step telemetry")
    candidates = [
        record
        for record in history
        if isinstance(record, dict)
        and record.get("step") == 1
        and not isinstance(record.get("step"), bool)
        and isinstance(record.get("step"), int)
        and "loss" in record
        and not any(str(key).startswith("eval_") for key in record)
    ]
    if len(candidates) != 1:
        contractlib.fail(
            "memory gate requires exactly one non-eval step-1 training telemetry record"
        )
    record = candidates[0]
    required = {
        "loss",
        "learning_rate",
        "grad_norm",
        "kl",
        "constrained_grpo/candidate_feasibility_rate",
        "constrained_grpo/clasp/advantage_l2",
        "constrained_grpo/clasp/advantage_abs_mean",
    }
    _validate_required_numeric_metrics(
        record,
        required=sorted(required),
        label="memory-gate step-1",
    )
    nonfinite = {
        key: value
        for key, value in record.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isfinite(float(value))
    }
    if nonfinite:
        contractlib.fail(f"memory-gate step-1 telemetry is non-finite: {nonfinite}")
    if (
        not isinstance(record.get("learning_rate"), (int, float))
        or float(record["learning_rate"]) <= 0.0
    ):
        contractlib.fail("memory-gate optimizer update did not use a positive learning rate")
    return {
        "path": str(path),
        "file_sha256": contractlib.sha256_file(path),
        "global_step": 1,
        "record_sha256": contractlib.sha256_json(record),
        "learning_rate": float(record["learning_rate"]),
        "all_numeric_values_finite": True,
        "required_keys": sorted(required),
    }


def _audit_memory_gate_root_claim(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    gate_dir: Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    expected_root_names = {
        CLAIM_ANCHOR_DIRNAME,
        "_runtime_tmp",
        "checkpoints",
        "hyperparams.json",
        contractlib.EXACT_PEFT_RUNTIME_RECEIPT,
        contractlib.EXACT_PEFT_STEP1_RECEIPT,
    }
    observed_root_names = {path.name for path in gate_dir.iterdir()}
    if observed_root_names != expected_root_names:
        contractlib.fail(
            "memory-gate nonterminal root layout differs from the exact allowlist: "
            f"observed={sorted(observed_root_names)} "
            f"expected={sorted(expected_root_names)}"
        )
    anchor_root = gate_dir / CLAIM_ANCHOR_DIRNAME
    if (
        anchor_root.is_symlink()
        or not anchor_root.is_dir()
        or stat.S_IMODE(anchor_root.lstat().st_mode) & 0o222
        or {path.name for path in anchor_root.iterdir()}
        != {
            MEMORY_GATE_ATTEMPT_FILENAME,
            MEMORY_GATE_TRANSCRIPT_FILENAME,
            MEMORY_GATE_OUTCOME_FILENAME,
            MEMORY_GATE_ANCHOR_GUARD_DIRNAME,
        }
    ):
        contractlib.fail("memory-gate attempt anchor is absent, writable, or contaminated")
    gate_command = build_plan(contract, contract_path=contract_path)["memory_gate_command"]
    gate_attempt_path = anchor_root / MEMORY_GATE_ATTEMPT_FILENAME
    gate_attempt = _inspect_attempt_claim(
        gate_attempt_path,
        expected_protocol=MEMORY_GATE_ATTEMPT_PROTOCOL,
        expected_status="development_memory_gate_attempt_started",
        expected_kind="memory_gate",
        trajectory_id="engineering-gate-D5-seed42",
        ledger_identity=None,
        reservation_identity=None,
        command=gate_command,
        resume_step=None,
    )
    _, outcome_path, outcome = _inspect_memory_gate_execution_anchors(gate_attempt)
    if outcome is None:
        contractlib.fail("memory-gate executor outcome is absent")
    outcome = inspect_memory_gate_execution_outcome(
        outcome_path,
        contract,
        contract_path=contract_path,
        command=gate_command,
        claim=gate_attempt,
    )
    if outcome.get("status") != MEMORY_GATE_SUCCESS_STATUS:
        contractlib.fail("memory-gate executor did not persist an exact zero exit")
    return gate_attempt_path, gate_attempt, outcome_path, outcome


def audit_memory_gate_checkpoint(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    gate_dir: Path,
) -> dict[str, Any]:
    """Reopen the sole checkpoint-1 artifact and prove the engineering gate."""

    from clasp.models.checkpoint_fidelity import verify_checkpoint_fidelity
    from clasp.models.exact_peft_kl_post_step import (
        POST_STEP_RECEIPT_FILENAME,
        STEP1_RECEIPT_FILENAME,
        inspect_post_step_receipt,
        post_step_reference_identity,
        training_run_contract_identity,
    )
    from clasp.models.exact_peft_kl_reference import (
        RECEIPT_FILENAME as PRE_STEP_RECEIPT_FILENAME,
    )
    from clasp.models.exact_peft_kl_reference import (
        inspect_runtime_receipt,
    )

    expected_gate = Path(str(contract["paths"]["output_root"])) / MEMORY_GATE_DIRNAME
    if gate_dir != expected_gate or gate_dir.is_symlink() or not gate_dir.is_dir():
        contractlib.fail("memory-gate output path differs from preregistration")
    gate_attempt_path, gate_attempt, gate_outcome_path, gate_outcome = (
        _audit_memory_gate_root_claim(
            contract,
            contract_path=contract_path,
            gate_dir=gate_dir,
        )
    )
    checkpoint_root = gate_dir / "checkpoints"
    checkpoint = checkpoint_root / "checkpoint-1"
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        contractlib.fail("memory gate lacks checkpoint-1")
    checkpoint_children = {path.name for path in checkpoint_root.iterdir()}
    if checkpoint_children != {"checkpoint-1"}:
        contractlib.fail(
            "memory gate checkpoint root differs from the exact allowlist: "
            f"{sorted(checkpoint_children)}"
        )

    previous_contract = os.environ.get(TRAINING_RUN_CONTRACT_ENV)
    os.environ[TRAINING_RUN_CONTRACT_ENV] = str(contract_path)
    try:
        fidelity = verify_checkpoint_fidelity(checkpoint)
        if (
            fidelity.get("protocol") != "clasp-checkpoint-fidelity-v1"
            or fidelity.get("status") != "exact_trainable_policy_persisted"
            or fidelity.get("publication_invariant") is not None
        ):
            contractlib.fail("checkpoint-1 fidelity claim boundary mismatch")
        closure = fidelity.get("trainable_closure")
        if not isinstance(closure, dict):
            contractlib.fail("checkpoint-1 fidelity lacks trainable closure")
        closure_digest = closure.get("tensor_inventory_sha256")
        if not isinstance(closure_digest, str):
            contractlib.fail("checkpoint-1 fidelity lacks trainable inventory digest")

        root_pre_path = gate_dir / PRE_STEP_RECEIPT_FILENAME
        root_step1_path = gate_dir / STEP1_RECEIPT_FILENAME
        pre_path = checkpoint / PRE_STEP_RECEIPT_FILENAME
        step1_path = checkpoint / STEP1_RECEIPT_FILENAME
        post_path = checkpoint / POST_STEP_RECEIPT_FILENAME
        root_pre = inspect_runtime_receipt(root_pre_path, claim_scope="development")
        pre = inspect_runtime_receipt(pre_path, claim_scope="development")
        root_step1 = inspect_post_step_receipt(
            root_step1_path,
            claim_scope="development",
            expected_global_step=1,
            expected_artifact_role="step1",
            rehash_sources=True,
            rehash_policy_artifact=False,
        )
        inspect_post_step_receipt(
            step1_path,
            claim_scope="development",
            expected_global_step=1,
            expected_artifact_role="step1",
            rehash_sources=True,
            rehash_policy_artifact=False,
        )
        if (
            root_pre != pre
            or contractlib.sha256_file(root_pre_path) != contractlib.sha256_file(pre_path)
            or root_step1.get("receipt_sha256")
            != inspect_post_step_receipt(
                step1_path,
                claim_scope="development",
                expected_global_step=1,
                expected_artifact_role="step1",
                rehash_sources=True,
                rehash_policy_artifact=False,
            ).get("receipt_sha256")
            or contractlib.sha256_file(root_step1_path) != contractlib.sha256_file(step1_path)
        ):
            contractlib.fail("root/checkpoint-1 pre-step or step1 receipts are not identical")
        post = inspect_post_step_receipt(
            post_path,
            claim_scope="development",
            expected_global_step=1,
            expected_artifact_role="checkpoint",
            rehash_sources=True,
            rehash_policy_artifact=True,
            expected_policy_trainable_tensor_inventory_sha256=closure_digest,
        )
        run_contract = training_run_contract_identity(
            contract_path,
            claim_scope="development",
        )
        if post.get("training_run_contract") != run_contract:
            contractlib.fail("checkpoint-1 post-step receipt binds a different run contract")
        policy = post.get("policy_context")
        if (
            not isinstance(policy, dict)
            or policy.get("changed_from_pre_step") is not True
            or not isinstance(policy.get("changed_trainable_tensor_count"), int)
            or isinstance(policy.get("changed_trainable_tensor_count"), bool)
            or policy["changed_trainable_tensor_count"] <= 0
            or policy.get("policy_trainable_tensor_inventory_sha256") != closure_digest
        ):
            contractlib.fail("memory gate does not prove a real trainable-policy update")
        reference = post.get("reference_audit")
        if (
            not isinstance(reference, dict)
            or reference.get("reference_trainable_tensor_count") != 0
            or reference.get("reference_eval_mode") is not True
            or reference.get("live_equals_pre_step_bitwise") is not True
            or reference.get("policy_reference_storage_independent") is not True
            or reference.get("optimizer_excludes_reference_storage") is not True
            or reference.get("optimizer_matches_policy_trainables") is not True
        ):
            contractlib.fail("memory gate live reference/optimizer invariants are incomplete")
        distributed = post.get("distributed")
        memory = distributed.get("rank_cuda_memory") if isinstance(distributed, dict) else None
        if (
            not isinstance(distributed, dict)
            or distributed.get("world_size") != 1
            or not isinstance(memory, list)
            or len(memory) != 1
            or not isinstance(memory[0], dict)
        ):
            contractlib.fail("memory gate lacks single-rank in-process CUDA evidence")
        cuda = memory[0]
        expected_hardware = contractlib.EXPECTED_LOCAL_HARDWARE
        observed_uuid = str(cuda.get("device_uuid") or "")
        normalized_uuid = observed_uuid.removeprefix("GPU-")
        expected_uuid = str(expected_hardware["uuid"]).removeprefix("GPU-")
        if (
            cuda.get("rank") != 0
            or cuda.get("cuda_available") is not True
            or cuda.get("device_index") != 0
            or cuda.get("physical_device_index") != expected_hardware["physical_index"]
            or str(cuda.get("visible_device_token")) != "3"
            or normalized_uuid != expected_uuid
            or cuda.get("device_name") != expected_hardware["name"]
            or str(cuda.get("driver_version")) != expected_hardware["driver_version"]
            or cuda.get("cuda_runtime_version") != contractlib.EXPECTED_RUNTIME["cuda"]
            or cuda.get("total_memory_bytes") != expected_hardware["torch_total_memory_bytes"]
            or cuda.get("nvidia_smi_total_memory_mib") != expected_hardware["memory_total_mib"]
            or not isinstance(cuda.get("peak_allocated_bytes"), int)
            or isinstance(cuda.get("peak_allocated_bytes"), bool)
            or cuda["peak_allocated_bytes"] <= 0
            or not isinstance(cuda.get("peak_reserved_bytes"), int)
            or isinstance(cuda.get("peak_reserved_bytes"), bool)
            or cuda["peak_reserved_bytes"] < cuda["peak_allocated_bytes"]
            or cuda["peak_reserved_bytes"] >= cuda["total_memory_bytes"]
        ):
            contractlib.fail(f"memory gate CUDA identity/peak evidence mismatch: {cuda}")

        trainer_state = _audit_step1_trainer_state(checkpoint / "trainer_state.json")
        pre_identity = {
            "path": str(pre_path),
            "protocol": pre["protocol"],
            "status": pre["status"],
            "receipt_sha256": pre["receipt_sha256"],
            "receipt_file_sha256": contractlib.sha256_file(pre_path),
            "runtime_tensor_inventory_sha256": pre["runtime_pair"]["tensor_inventory_sha256"],
        }
        root_pre_identity = {
            **pre_identity,
            "path": str(root_pre_path),
            "receipt_file_sha256": contractlib.sha256_file(root_pre_path),
        }
        root_step1_identity = post_step_reference_identity(
            root_step1_path,
            claim_scope="development",
            expected_global_step=1,
            expected_artifact_role="step1",
            rehash_sources=True,
            rehash_policy_artifact=False,
        )
        step1_identity = post_step_reference_identity(
            step1_path,
            claim_scope="development",
            expected_global_step=1,
            expected_artifact_role="step1",
            rehash_sources=True,
            rehash_policy_artifact=False,
        )
        post_identity = post_step_reference_identity(
            post_path,
            claim_scope="development",
            expected_global_step=1,
            expected_artifact_role="checkpoint",
            rehash_sources=True,
            rehash_policy_artifact=True,
            expected_policy_trainable_tensor_inventory_sha256=closure_digest,
        )
    finally:
        if previous_contract is None:
            os.environ.pop(TRAINING_RUN_CONTRACT_ENV, None)
        else:
            os.environ[TRAINING_RUN_CONTRACT_ENV] = previous_contract

    return {
        "checkpoint_fidelity": {
            "path": str(checkpoint / "checkpoint_fidelity.json"),
            "receipt_sha256": fidelity["receipt_sha256"],
            "receipt_file_sha256": contractlib.sha256_file(checkpoint / "checkpoint_fidelity.json"),
            "trainable_tensor_inventory_sha256": closure_digest,
            "publication_invariant": None,
        },
        "attempt_claim": _file_identity(gate_attempt_path, gate_attempt),
        "executor_zero_exit": _file_identity(gate_outcome_path, gate_outcome),
        "pre_step_reference": pre_identity,
        "root_pre_step_reference": root_pre_identity,
        "root_step1_reference": root_step1_identity,
        "step1_reference": step1_identity,
        "checkpoint_post_step_reference": post_identity,
        "training_run_contract": run_contract,
        "policy_update": {
            "changed_from_pre_step": True,
            "changed_trainable_tensor_count": policy["changed_trainable_tensor_count"],
            "trainable_tensor_inventory_sha256": closure_digest,
        },
        "reference_immutability": {
            "live_equals_pre_step_bitwise": True,
            "reference_trainable_tensor_count": 0,
            "reference_eval_mode": True,
            "policy_reference_storage_independent": True,
            "optimizer_excludes_reference_storage": True,
            "optimizer_matches_policy_trainables": True,
        },
        "cuda": dict(cuda),
        "trainer_state": trainer_state,
        "nonterminal_root_finalization_absent": True,
    }


def _authorized_command(command: Mapping[str, Any], *, ledger_path: Path) -> dict[str, Any]:
    result = json.loads(contractlib.canonical_json_bytes(command).decode("utf-8"))
    result["status"] = "AUTHORIZED_PRE_OUTCOME_BY_FULL_MATRIX_LEDGER"
    result["launch_authorized"] = True
    result["authorization_ledger_path"] = str(ledger_path)
    return result


def build_matrix_authorization(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    memory_gate_pass_path: Path,
    memory_gate_pass: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the one pre-outcome authorization for the complete 12-cell matrix."""

    _inspect_memory_gate_zero_outcome(contract, contract_path=contract_path)
    output_root = Path(str(contract["paths"]["output_root"]))
    ledger_path = output_root / MATRIX_AUTHORIZATION_FILENAME
    candidate = build_plan(
        contract,
        contract_path=contract_path,
        include_cell_commands=True,
    )
    training = [
        _authorized_command(command, ledger_path=ledger_path)
        for command in candidate["training_commands"]
    ]
    audits = [
        _authorized_command(command, ledger_path=ledger_path)
        for command in candidate["fidelity_audit_commands"]
    ]
    evaluations = [
        _authorized_command(command, ledger_path=ledger_path)
        for command in candidate["evaluation_commands"]
    ]
    trajectory_ids = [str(command["trajectory_id"]) for command in training]
    if len(trajectory_ids) != 12 or len(set(trajectory_ids)) != 12:
        contractlib.fail("matrix authorization does not contain 12 unique trajectories")
    expected_ids = [str(cell["trajectory_id"]) for cell in contract["design"]["cells"]]
    if trajectory_ids != expected_ids:
        contractlib.fail("matrix authorization trajectory order differs from preregistration")
    training_dirs = [str(command["output_dir"]) for command in training]
    if len(set(training_dirs)) != 12:
        contractlib.fail("matrix authorization training output directories are not unique")
    evaluation_attempts = [
        {
            "trajectory_id": command["trajectory_id"],
            "inference_root": _argument(command["argv"], "--inference_root"),
            "eval_name": _argument(command["argv"], "--eval_name"),
        }
        for command in evaluations
    ]
    attempt_keys = [
        (attempt["inference_root"], attempt["eval_name"]) for attempt in evaluation_attempts
    ]
    if len(attempt_keys) != 12 or len(set(attempt_keys)) != 12:
        contractlib.fail("matrix authorization evaluation attempts are not unique")
    pass_identity = _file_identity(memory_gate_pass_path, memory_gate_pass)
    payload = {
        "schema_version": 1,
        "protocol": MATRIX_AUTHORIZATION_PROTOCOL,
        "status": MATRIX_AUTHORIZATION_STATUS,
        "stage": "grpo",
        "fixed_horizon": {"max_steps": 400, "world_size": 1},
        "development_only": True,
        "publication_eligible": False,
        "sota_eligible": False,
        "created_at_utc": _utc_now(),
        "contract": _contract_identity(contract, contract_path),
        "memory_gate_pass": pass_identity,
        "ledger_path": str(ledger_path),
        "chronology": {
            "created_before_any_scientific_training_or_evaluation_attempt": True,
            "regeneration_after_any_outcome_forbidden": True,
            "single_authorization_for_all_trajectories": True,
        },
        "matrix_rule": {
            "required_trajectory_ids": trajectory_ids,
            "trajectory_count": 12,
            "all_trajectories_required": True,
            "missing_failed_or_invalid_remain_in_denominator": True,
            "replacement_or_rerun_forbidden": True,
        },
        "reserved_training_output_dirs": training_dirs,
        "reserved_evaluation_attempts": evaluation_attempts,
        "training_commands": training,
        "terminal_audit_commands": audits,
        "evaluation_commands": evaluations,
    }
    return _self_hashed(payload)


def _argument(argv: Sequence[str], name: str) -> str:
    try:
        position = list(argv).index(name)
        return str(argv[position + 1])
    except (ValueError, IndexError) as exc:
        contractlib.fail(f"authorized command lacks required argument {name}")
        raise AssertionError from exc


def _reservation_payload(
    *,
    kind: str,
    trajectory_id: str,
    reserved_path: Path,
    command: Mapping[str, Any],
    ledger_path: Path,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    if kind not in {"training", "evaluation"}:
        contractlib.fail(f"unsupported matrix reservation kind: {kind}")
    return _self_hashed(
        {
            "schema_version": 1,
            "protocol": (
                CELL_AUTHORIZATION_PROTOCOL
                if kind == "training"
                else EVALUATION_AUTHORIZATION_PROTOCOL
            ),
            "status": "development_matrix_slot_reserved_pre_outcome",
            "development_only": True,
            "publication_eligible": False,
            "sota_eligible": False,
            "kind": kind,
            "trajectory_id": trajectory_id,
            "reserved_path": str(reserved_path),
            "workspace_bound_tmpdir": str(reserved_path / "_runtime_tmp"),
            "claim_anchor_directory": str(reserved_path / CLAIM_ANCHOR_DIRNAME),
            "claim_anchor_count": len(_claim_anchor_names(kind)),
            "claim_anchor_semantics": (
                "all attempt inodes are precreated before closure inside a read-only "
                "directory; each inode permits one empty-to-read-only receipt transition"
            ),
            "ledger": _file_identity(ledger_path, ledger),
            "authorized_command_sha256": contractlib.sha256_json(command),
        }
    )


def reserve_full_matrix(
    ledger_path: Path,
    ledger: Mapping[str, Any],
) -> None:
    """Materialize all pre-outcome run/evaluation slots or fail without authorization."""

    output_root = ledger_path.parent
    training = ledger.get("training_commands")
    evaluations = ledger.get("evaluation_commands")
    if not isinstance(training, list) or not isinstance(evaluations, list):
        contractlib.fail("matrix ledger lacks command arrays")

    reservations: list[tuple[Path, str, Mapping[str, Any], str]] = []
    for command in training:
        if not isinstance(command, dict):
            contractlib.fail("matrix ledger contains an invalid training command")
        reservations.append(
            (
                Path(str(command["output_dir"])),
                CELL_AUTHORIZATION_FILENAME,
                command,
                "training",
            )
        )
    for command in evaluations:
        if not isinstance(command, dict):
            contractlib.fail("matrix ledger contains an invalid evaluation command")
        reservations.append(
            (
                Path(_argument(command["argv"], "--inference_root")),
                EVALUATION_AUTHORIZATION_FILENAME,
                command,
                "evaluation",
            )
        )

    for reserved_path, filename, command, kind in reservations:
        if output_root not in reserved_path.parents:
            contractlib.fail(f"matrix reservation escapes output_root: {reserved_path}")
        reserved_path.mkdir(mode=0o750, parents=True, exist_ok=True)
        marker_path = reserved_path / filename
        tmpdir = reserved_path / "_runtime_tmp"
        expected = _reservation_payload(
            kind=kind,
            trajectory_id=str(command["trajectory_id"]),
            reserved_path=reserved_path,
            command=command,
            ledger_path=ledger_path,
            ledger=ledger,
        )
        allowed_names = {filename, "_runtime_tmp", CLAIM_ANCHOR_DIRNAME}
        existing_names = {value.name for value in reserved_path.iterdir()}
        if existing_names - allowed_names:
            contractlib.fail(
                f"matrix slot contains outcome bytes before authorization: {reserved_path}"
            )
        if not tmpdir.exists() and not tmpdir.is_symlink():
            tmpdir.mkdir(mode=0o700)
        if marker_path.exists() or marker_path.is_symlink():
            observed = contractlib.load_json_object(marker_path, label="matrix reservation")
            contractlib.require_read_only(marker_path, label="matrix reservation")
            if observed != expected:
                contractlib.fail(f"matrix reservation differs from ledger: {marker_path}")
        else:
            contractlib.write_contract_create_once(marker_path, expected)
        _ensure_claim_anchor_tree(reserved_path, kind=kind)
        if {value.name for value in reserved_path.iterdir()} != allowed_names:
            contractlib.fail(f"matrix reservation is incomplete: {reserved_path}")
        if tmpdir.is_symlink() or not tmpdir.is_dir() or any(tmpdir.iterdir()):
            contractlib.fail(f"matrix reservation TMPDIR is absent or contaminated: {tmpdir}")


def _reservation_specifications(
    ledger: Mapping[str, Any],
) -> list[tuple[Path, str, Mapping[str, Any], str]]:
    result: list[tuple[Path, str, Mapping[str, Any], str]] = []
    for command in ledger.get("training_commands", []):
        if not isinstance(command, dict):
            contractlib.fail("matrix ledger contains an invalid training command")
        result.append(
            (
                Path(str(command["output_dir"])),
                CELL_AUTHORIZATION_FILENAME,
                command,
                "training",
            )
        )
    for command in ledger.get("evaluation_commands", []):
        if not isinstance(command, dict):
            contractlib.fail("matrix ledger contains an invalid evaluation command")
        result.append(
            (
                Path(_argument(command["argv"], "--inference_root")),
                EVALUATION_AUTHORIZATION_FILENAME,
                command,
                "evaluation",
            )
        )
    if len(result) != 24:
        contractlib.fail("full matrix reservation requires exactly 24 train/eval slots")
    return result


def build_reservation_closure(
    ledger_path: Path,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Close the two-phase reservation before any trajectory may claim a slot."""

    records: list[dict[str, Any]] = []
    for reserved_path, filename, command, kind in _reservation_specifications(ledger):
        marker_path = reserved_path / filename
        marker = contractlib.load_json_object(marker_path, label="matrix reservation")
        contractlib.require_read_only(marker_path, label="matrix reservation")
        expected = _reservation_payload(
            kind=kind,
            trajectory_id=str(command["trajectory_id"]),
            reserved_path=reserved_path,
            command=command,
            ledger_path=ledger_path,
            ledger=ledger,
        )
        if marker != expected:
            contractlib.fail(f"matrix reservation differs before closure: {marker_path}")
        tmpdir = reserved_path / "_runtime_tmp"
        if tmpdir.is_symlink() or not tmpdir.is_dir() or any(tmpdir.iterdir()):
            contractlib.fail(f"reservation TMPDIR is contaminated before closure: {tmpdir}")
        records.append(
            {
                "kind": kind,
                "trajectory_id": str(command["trajectory_id"]),
                "reserved_path": _path_stat_identity(reserved_path),
                "marker": {
                    **_file_identity(marker_path, marker),
                    "stat": _path_stat_identity(marker_path),
                },
                "tmpdir": _path_stat_identity(tmpdir),
                "claim_anchors": _initial_claim_anchor_manifest(
                    reserved_path,
                    kind=kind,
                ),
            }
        )
    return _self_hashed(
        {
            "schema_version": 1,
            "protocol": RESERVATION_CLOSURE_PROTOCOL,
            "status": RESERVATION_CLOSURE_STATUS,
            "stage": "grpo",
            "development_only": True,
            "publication_eligible": False,
            "sota_eligible": False,
            "created_at_utc": _utc_now(),
            "ledger": _file_identity(ledger_path, ledger),
            "reservation_count": 24,
            "all_twelve_training_and_evaluation_slots_required": True,
            "missing_failed_invalid_or_deleted_slot_remains_in_denominator": True,
            "records": records,
        }
    )


def inspect_reservation_closure(
    path: Path,
    ledger_path: Path,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen the pre-outcome closure without recreating a missing reservation."""

    contractlib.require_read_only(path, label="matrix reservation closure")
    closure = contractlib.load_json_object(path, label="matrix reservation closure")
    _verify_self_hash(closure, label="matrix reservation closure")
    if (
        closure.get("schema_version") != 1
        or closure.get("protocol") != RESERVATION_CLOSURE_PROTOCOL
        or closure.get("status") != RESERVATION_CLOSURE_STATUS
        or closure.get("stage") != "grpo"
        or closure.get("development_only") is not True
        or closure.get("publication_eligible") is not False
        or closure.get("sota_eligible") is not False
        or closure.get("reservation_count") != 24
        or closure.get("ledger") != _file_identity(ledger_path, ledger)
    ):
        contractlib.fail("matrix reservation closure identity/claim mismatch")
    expected_specs = _reservation_specifications(ledger)
    records = closure.get("records")
    if not isinstance(records, list) or len(records) != len(expected_specs):
        contractlib.fail("matrix reservation closure record count mismatch")
    for record, (reserved_path, filename, command, kind) in zip(
        records, expected_specs, strict=True
    ):
        if not isinstance(record, dict):
            contractlib.fail("matrix reservation closure has a malformed record")
        marker_path = reserved_path / filename
        marker = contractlib.load_json_object(marker_path, label="matrix reservation")
        contractlib.require_read_only(marker_path, label="matrix reservation")
        expected_marker = _reservation_payload(
            kind=kind,
            trajectory_id=str(command["trajectory_id"]),
            reserved_path=reserved_path,
            command=command,
            ledger_path=ledger_path,
            ledger=ledger,
        )
        expected_record = {
            "kind": kind,
            "trajectory_id": str(command["trajectory_id"]),
            "reserved_path": _path_stat_identity(reserved_path),
            "marker": {
                **_file_identity(marker_path, marker),
                "stat": _path_stat_identity(marker_path),
            },
            "tmpdir": _path_stat_identity(reserved_path / "_runtime_tmp"),
        }
        observed_projection = {key: record.get(key) for key in expected_record}
        if marker != expected_marker or observed_projection != expected_record:
            contractlib.fail(
                f"matrix slot was deleted, recreated, or altered after closure: {reserved_path}"
            )
        _inspect_claim_anchor_manifest(
            reserved_path,
            kind=kind,
            frozen=record.get("claim_anchors"),
        )
    return closure


def verify_partial_reservations_have_no_outcomes(
    ledger_path: Path,
    ledger: Mapping[str, Any],
) -> None:
    """Validate every already-created slot before completing a partial reservation."""

    specifications: list[tuple[Path, str, Mapping[str, Any], str]] = []
    for command in ledger["training_commands"]:
        specifications.append(
            (
                Path(str(command["output_dir"])),
                CELL_AUTHORIZATION_FILENAME,
                command,
                "training",
            )
        )
    evaluation_roots = {
        Path(_argument(command["argv"], "--inference_root")).name
        for command in ledger["evaluation_commands"]
    }
    evaluation_parent = ledger_path.parent / "terminal_development_evaluation"
    if evaluation_parent.exists():
        observed_children = {path.name for path in evaluation_parent.iterdir()}
        if observed_children - evaluation_roots:
            contractlib.fail("unknown evaluation outcome path exists before reservation closes")
    for command in ledger["evaluation_commands"]:
        specifications.append(
            (
                Path(_argument(command["argv"], "--inference_root")),
                EVALUATION_AUTHORIZATION_FILENAME,
                command,
                "evaluation",
            )
        )
    for reserved_path, filename, command, kind in specifications:
        if not reserved_path.exists():
            continue
        if reserved_path.is_symlink() or not reserved_path.is_dir():
            contractlib.fail(f"matrix reservation path is not a directory: {reserved_path}")
        names = {path.name for path in reserved_path.iterdir()}
        if not names <= {filename, "_runtime_tmp", CLAIM_ANCHOR_DIRNAME}:
            contractlib.fail(f"outcome bytes exist before matrix reservation: {reserved_path}")
        marker = reserved_path / filename
        if marker.exists():
            expected = _reservation_payload(
                kind=kind,
                trajectory_id=str(command["trajectory_id"]),
                reserved_path=reserved_path,
                command=command,
                ledger_path=ledger_path,
                ledger=ledger,
            )
            observed = contractlib.load_json_object(marker, label="matrix reservation")
            contractlib.require_read_only(marker, label="matrix reservation")
            if observed != expected:
                contractlib.fail(f"partial matrix reservation differs: {marker}")
        tmpdir = reserved_path / "_runtime_tmp"
        if tmpdir.exists() and (
            tmpdir.is_symlink() or not tmpdir.is_dir() or any(tmpdir.iterdir())
        ):
            contractlib.fail(f"partial reservation TMPDIR contains outcome bytes: {tmpdir}")
        _partial_claim_anchor_tree_is_pristine(reserved_path, kind=kind)


def verify_pristine_matrix_reservations(
    ledger_path: Path,
    ledger: Mapping[str, Any],
    closure_path: Path,
    closure: Mapping[str, Any],
) -> None:
    """Require every reserved slot and no observed scientific outcome bytes."""

    output_root = ledger_path.parent
    expected_top = {
        MEMORY_GATE_DIRNAME,
        MEMORY_GATE_PASS_FILENAME,
        MATRIX_AUTHORIZATION_FILENAME,
        RESERVATION_CLOSURE_FILENAME,
        "terminal_development_evaluation",
        *(str(command["trajectory_id"]) for command in ledger["training_commands"]),
    }
    _require_top_level(output_root, expected_top, phase="matrix authorization")
    evaluation_root = output_root / "terminal_development_evaluation"
    expected_evaluation_children = {
        Path(_argument(command["argv"], "--inference_root")).name
        for command in ledger["evaluation_commands"]
    }
    _require_top_level(
        evaluation_root,
        expected_evaluation_children,
        phase="evaluation-attempt reservation",
    )
    observed_closure = inspect_reservation_closure(closure_path, ledger_path, ledger)
    if observed_closure != dict(closure):
        contractlib.fail("matrix reservation closure changed after creation")
    for reserved_path, filename, _, _ in _reservation_specifications(ledger):
        names = {item.name for item in reserved_path.iterdir()}
        if names != {filename, "_runtime_tmp", CLAIM_ANCHOR_DIRNAME}:
            contractlib.fail(
                f"scientific outcome bytes appeared before matrix release: {reserved_path}"
            )
        if any((reserved_path / "_runtime_tmp").iterdir()):
            contractlib.fail(f"scientific TMPDIR changed before matrix release: {reserved_path}")
        manifest = next(
            record["claim_anchors"]
            for record in closure["records"]
            if record["kind"]
            == ("training" if filename == CELL_AUTHORIZATION_FILENAME else "evaluation")
            and record["reserved_path"]["path"] == str(reserved_path)
        )
        _inspect_claim_anchor_manifest(
            reserved_path,
            kind=("training" if filename == CELL_AUTHORIZATION_FILENAME else "evaluation"),
            frozen=manifest,
        )
        if any(path.stat().st_size for path in (reserved_path / CLAIM_ANCHOR_DIRNAME).iterdir()):
            contractlib.fail(f"claim anchor was consumed before matrix release: {reserved_path}")


def build_memory_gate_pass(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    gate_dir: Path,
    gate_evidence: Mapping[str, Any],
    gate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    outcome_path, outcome = _inspect_memory_gate_zero_outcome(
        contract,
        contract_path=contract_path,
    )
    if gate_evidence.get("executor_zero_exit") != _file_identity(outcome_path, outcome):
        contractlib.fail("memory-gate PASS evidence lacks the exact executor-zero receipt")
    return _self_hashed(
        {
            "schema_version": 1,
            "protocol": MEMORY_GATE_PASS_PROTOCOL,
            "status": MEMORY_GATE_PASS_STATUS,
            "development_only": True,
            "publication_eligible": False,
            "sota_eligible": False,
            "scientific_result_eligible": False,
            "created_at_utc": _utc_now(),
            "contract": _contract_identity(contract, contract_path),
            "gate": {
                "path": str(gate_dir),
                "checkpoint_path": str(gate_dir / "checkpoints" / "checkpoint-1"),
                "scheduler_horizon_max_steps": 400,
                "hard_stop_global_step": 1,
                "never_resume_or_extend": True,
                "root_finalization_forbidden": True,
            },
            "gate_evidence": dict(gate_evidence),
            "sealed_gate_tree": dict(gate_manifest),
            "claim_limit": (
                "engineering memory/reference-integrity evidence only; never a scientific, "
                "publication, benchmark, or SOTA result"
            ),
        }
    )


def inspect_memory_gate_pass(
    path: Path,
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    reverify_gate: bool,
) -> dict[str, Any]:
    receipt_path = path.expanduser().absolute()
    contractlib.require_read_only(receipt_path, label="memory-gate PASS receipt")
    receipt = contractlib.load_json_object(receipt_path, label="memory-gate PASS receipt")
    _verify_self_hash(receipt, label="memory-gate PASS receipt")
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "development_only",
        "publication_eligible",
        "sota_eligible",
        "scientific_result_eligible",
        "created_at_utc",
        "contract",
        "gate",
        "gate_evidence",
        "sealed_gate_tree",
        "claim_limit",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        contractlib.fail("memory-gate PASS top-level schema mismatch")
    expected_claims = {
        "schema_version": 1,
        "protocol": MEMORY_GATE_PASS_PROTOCOL,
        "status": MEMORY_GATE_PASS_STATUS,
        "development_only": True,
        "publication_eligible": False,
        "sota_eligible": False,
        "scientific_result_eligible": False,
    }
    if any(receipt.get(key) != value for key, value in expected_claims.items()):
        contractlib.fail("memory-gate PASS claim boundary mismatch")
    if receipt.get("contract") != _contract_identity(contract, contract_path):
        contractlib.fail("memory-gate PASS is not bound to the live frozen contract")
    gate_dir = Path(str(receipt.get("gate", {}).get("path", "")))
    expected_gate = Path(str(contract["paths"]["output_root"])) / MEMORY_GATE_DIRNAME
    if gate_dir != expected_gate:
        contractlib.fail("memory-gate PASS points to the wrong gate subtree")
    outcome_path, outcome = _inspect_memory_gate_zero_outcome(
        contract,
        contract_path=contract_path,
    )
    if receipt.get("gate_evidence", {}).get("executor_zero_exit") != _file_identity(
        outcome_path,
        outcome,
    ):
        contractlib.fail("memory-gate PASS does not bind the live zero-exit outcome")
    contractlib.require_read_only(gate_dir, label="memory-gate tree", recursive=True)
    if _stable_gate_manifest(gate_dir) != receipt.get("sealed_gate_tree"):
        contractlib.fail("sealed memory-gate subtree differs from PASS")
    if reverify_gate:
        observed = audit_memory_gate_checkpoint(
            contract,
            contract_path=contract_path,
            gate_dir=gate_dir,
        )
        if observed != receipt.get("gate_evidence"):
            contractlib.fail("live checkpoint-1 evidence differs from memory-gate PASS")
    return receipt


def inspect_matrix_authorization(
    path: Path,
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    memory_gate_pass_path: Path,
    memory_gate_pass: Mapping[str, Any],
) -> dict[str, Any]:
    _inspect_memory_gate_zero_outcome(contract, contract_path=contract_path)
    ledger_path = path.expanduser().absolute()
    contractlib.require_read_only(ledger_path, label="matrix authorization ledger")
    ledger = contractlib.load_json_object(ledger_path, label="matrix authorization ledger")
    _verify_self_hash(ledger, label="matrix authorization ledger")
    if (
        ledger.get("schema_version") != 1
        or ledger.get("protocol") != MATRIX_AUTHORIZATION_PROTOCOL
        or ledger.get("status") != MATRIX_AUTHORIZATION_STATUS
        or ledger.get("stage") != "grpo"
        or ledger.get("fixed_horizon") != {"max_steps": 400, "world_size": 1}
        or ledger.get("development_only") is not True
        or ledger.get("publication_eligible") is not False
        or ledger.get("sota_eligible") is not False
        or ledger.get("ledger_path") != str(ledger_path)
    ):
        contractlib.fail("matrix authorization identity/claim mismatch")
    if ledger.get("contract") != _contract_identity(contract, contract_path):
        contractlib.fail("matrix ledger does not bind the frozen pilot contract")
    if ledger.get("memory_gate_pass") != _file_identity(
        memory_gate_pass_path,
        memory_gate_pass,
    ):
        contractlib.fail("matrix ledger does not bind the live memory-gate PASS")
    candidate = build_plan(
        contract,
        contract_path=contract_path,
        include_cell_commands=True,
    )
    expected_groups = {
        "training_commands": candidate["training_commands"],
        "terminal_audit_commands": candidate["fidelity_audit_commands"],
        "evaluation_commands": candidate["evaluation_commands"],
    }
    for key, commands in expected_groups.items():
        expected_commands = [
            _authorized_command(command, ledger_path=ledger_path) for command in commands
        ]
        if ledger.get(key) != expected_commands:
            contractlib.fail(f"matrix ledger {key} differ from frozen command construction")
    matrix_rule = ledger.get("matrix_rule")
    expected_ids = [str(cell["trajectory_id"]) for cell in contract["design"]["cells"]]
    if (
        not isinstance(matrix_rule, dict)
        or matrix_rule.get("required_trajectory_ids") != expected_ids
        or matrix_rule.get("trajectory_count") != 12
        or matrix_rule.get("all_trajectories_required") is not True
        or matrix_rule.get("missing_failed_or_invalid_remain_in_denominator") is not True
        or matrix_rule.get("replacement_or_rerun_forbidden") is not True
    ):
        contractlib.fail("matrix ledger weakens complete-matrix accounting")
    return ledger


def record_memory_gate_pass_and_matrix(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Audit checkpoint-1, seal it, and create the sole full-matrix authority."""

    output_root = Path(str(contract["paths"]["output_root"]))
    gate_dir = output_root / MEMORY_GATE_DIRNAME
    pass_path = output_root / MEMORY_GATE_PASS_FILENAME
    ledger_path = output_root / MATRIX_AUTHORIZATION_FILENAME
    closure_path = output_root / RESERVATION_CLOSURE_FILENAME
    observed_top = _top_level_names(output_root)
    base_names = {MEMORY_GATE_DIRNAME, MEMORY_GATE_PASS_FILENAME}
    maximum_names = {
        *base_names,
        MATRIX_AUTHORIZATION_FILENAME,
        RESERVATION_CLOSURE_FILENAME,
        "terminal_development_evaluation",
        *(str(cell["trajectory_id"]) for cell in contract["design"]["cells"]),
    }
    if MEMORY_GATE_DIRNAME not in observed_top or not observed_top <= maximum_names:
        contractlib.fail(
            "cannot create/regenerate matrix authorization after reservations or outcomes exist"
        )
    if not ledger_path.exists() and observed_top not in (
        {MEMORY_GATE_DIRNAME},
        base_names,
    ):
        contractlib.fail("scientific reservation paths exist before the matrix ledger")

    if pass_path.exists():
        memory_pass = inspect_memory_gate_pass(
            pass_path,
            contract,
            contract_path=contract_path,
            reverify_gate=True,
        )
    else:
        # A terminal executor failure must be diagnosed before any checkpoint/root
        # audit so malformed or incomplete child output cannot obscure FAILED.
        _inspect_memory_gate_zero_outcome(contract, contract_path=contract_path)
        evidence = audit_memory_gate_checkpoint(
            contract,
            contract_path=contract_path,
            gate_dir=gate_dir,
        )
        _seal_tree_read_only(gate_dir)
        manifest = _stable_gate_manifest(gate_dir)
        memory_pass = build_memory_gate_pass(
            contract,
            contract_path=contract_path,
            gate_dir=gate_dir,
            gate_evidence=evidence,
            gate_manifest=manifest,
        )
        contractlib.write_contract_create_once(pass_path, memory_pass)

    if ledger_path.exists():
        ledger = inspect_matrix_authorization(
            ledger_path,
            contract,
            contract_path=contract_path,
            memory_gate_pass_path=pass_path,
            memory_gate_pass=memory_pass,
        )
    else:
        ledger = build_matrix_authorization(
            contract,
            contract_path=contract_path,
            memory_gate_pass_path=pass_path,
            memory_gate_pass=memory_pass,
        )
        contractlib.write_contract_create_once(ledger_path, ledger)
    if closure_path.exists():
        closure = inspect_reservation_closure(closure_path, ledger_path, ledger)
    else:
        # A ledger is not executable authority until this two-phase closure
        # exists. Recovery from a crash between ledger creation and closure is
        # permitted only while no attempt/outcome byte exists anywhere.
        verify_partial_reservations_have_no_outcomes(ledger_path, ledger)
        reserve_full_matrix(ledger_path, ledger)
        closure = build_reservation_closure(ledger_path, ledger)
        contractlib.write_contract_create_once(closure_path, closure)
    verify_pristine_matrix_reservations(
        ledger_path,
        ledger,
        closure_path,
        closure,
    )
    return memory_pass, ledger, closure


def _load_matrix_lifecycle(
    contract_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the immutable post-gate authority without creating missing bytes."""

    contract_path = contract_path.expanduser().absolute()
    contractlib.require_read_only(contract_path, label="frozen pilot contract")
    contract = contractlib.load_json_object(contract_path, label="frozen pilot contract")
    contractlib.verify_contract(
        contract,
        require_fresh_output=False,
        contract_path=contract_path,
    )
    output_root = Path(str(contract["paths"]["output_root"]))
    pass_path = output_root / MEMORY_GATE_PASS_FILENAME
    ledger_path = output_root / MATRIX_AUTHORIZATION_FILENAME
    closure_path = output_root / RESERVATION_CLOSURE_FILENAME
    memory_pass = inspect_memory_gate_pass(
        pass_path,
        contract,
        contract_path=contract_path,
        reverify_gate=True,
    )
    ledger = inspect_matrix_authorization(
        ledger_path,
        contract,
        contract_path=contract_path,
        memory_gate_pass_path=pass_path,
        memory_gate_pass=memory_pass,
    )
    closure = inspect_reservation_closure(closure_path, ledger_path, ledger)
    return contract, memory_pass, ledger, closure


def _ledger_command(
    ledger: Mapping[str, Any],
    *,
    group: str,
    trajectory_id: str,
) -> dict[str, Any]:
    commands = ledger.get(group)
    if not isinstance(commands, list):
        contractlib.fail(f"matrix ledger lacks {group}")
    matches = [
        command
        for command in commands
        if isinstance(command, dict) and command.get("trajectory_id") == trajectory_id
    ]
    if len(matches) != 1:
        contractlib.fail(
            f"matrix ledger does not contain exactly one {group} command for {trajectory_id}"
        )
    return dict(matches[0])


def _attempt_payload(
    *,
    protocol: str,
    status: str,
    kind: str,
    trajectory_id: str,
    attempt_path: Path,
    ledger_path: Path | None,
    ledger: Mapping[str, Any] | None,
    reservation_path: Path | None,
    reservation: Mapping[str, Any] | None,
    command: Mapping[str, Any] | None,
    resume_checkpoint: Mapping[str, Any] | None = None,
    execution_anchors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "protocol": protocol,
        "status": status,
        "stage": "grpo",
        "development_only": True,
        "publication_eligible": False,
        "sota_eligible": False,
        "kind": kind,
        "trajectory_id": trajectory_id,
        "created_at_utc": _utc_now(),
        "attempt_path": str(attempt_path),
        "ledger": (
            _file_identity(ledger_path, ledger)
            if ledger_path is not None and ledger is not None
            else None
        ),
        "reservation": (
            _file_identity(reservation_path, reservation)
            if reservation_path is not None and reservation is not None
            else None
        ),
        "authorized_command_sha256": (
            contractlib.sha256_json(command) if command is not None else None
        ),
        "resume_checkpoint": (dict(resume_checkpoint) if resume_checkpoint is not None else None),
        "claim_semantics": (
            "create_once_before_exec; a failed or interrupted attempt remains in the "
            "complete-matrix denominator and cannot be silently replaced"
        ),
    }
    if execution_anchors is not None:
        payload["execution_anchors"] = dict(execution_anchors)
    return _self_hashed(payload)


def _inspect_attempt(path: Path, expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    contractlib.require_read_only(path, label=label)
    observed = contractlib.load_json_object(path, label=label)
    _verify_self_hash(observed, label=label)
    if observed != dict(expected):
        contractlib.fail(f"{label} differs from its frozen authority")
    return observed


def _validate_utc_timestamp(value: object, *, label: str) -> dt.datetime:
    if not isinstance(value, str):
        contractlib.fail(f"{label} timestamp is not a string")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        contractlib.fail(f"{label} timestamp is invalid: {exc}")
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        contractlib.fail(f"{label} timestamp must be timezone-aware UTC")
    return parsed


def _authorized_inner_payload(command: Mapping[str, Any]) -> list[str]:
    argv = command.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(value, str) for value in argv):
        contractlib.fail("authorized attempt command has an invalid argv")
    module_positions = [
        index
        for index, value in enumerate(argv)
        if value in {"clasp.entrypoints.train_grpo", "clasp.entrypoints.evaluate"}
    ]
    if len(module_positions) != 1:
        contractlib.fail("authorized attempt command lacks one scientific module payload")
    module_index = module_positions[0]
    start = module_index - 2
    if start < 0 or argv[start + 1] != "-m":
        contractlib.fail("authorized attempt command has malformed module invocation")
    return list(argv[start:])


def _assert_exact_attempt_material(
    command: Mapping[str, Any],
    *,
    actual_payload_argv: Sequence[str],
    actual_environment: Mapping[str, str],
    consumed_environment_names: Sequence[str] = (),
    resume_checkpoint_path: str | None = None,
) -> None:
    if isinstance(actual_payload_argv, (str, bytes)) or not all(
        isinstance(value, str) for value in actual_payload_argv
    ):
        contractlib.fail("live attempt payload argv is malformed")
    expected_payload = _authorized_inner_payload(command)
    observed_payload = list(actual_payload_argv)
    if observed_payload != expected_payload:
        contractlib.fail(
            "live attempt payload argv differs from frozen authority: "
            f"expected_sha256={contractlib.sha256_json(expected_payload)} "
            f"observed_sha256={contractlib.sha256_json(observed_payload)}"
        )
    expected_environment = command.get("env")
    if not isinstance(expected_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in expected_environment.items()
    ):
        contractlib.fail("authorized attempt command has an invalid environment")
    expected_live = dict(expected_environment)
    for name in consumed_environment_names:
        expected_live.pop(name, None)
    if resume_checkpoint_path is not None:
        expected_live["RESUME_FROM_CHECKPOINT"] = resume_checkpoint_path
    observed_environment = dict(actual_environment)
    if observed_environment != expected_live:
        contractlib.fail(
            "live attempt environment differs from frozen authority: "
            f"expected_sha256={contractlib.sha256_json(expected_live)} "
            f"observed_sha256={contractlib.sha256_json(observed_environment)}"
        )


def _claim_anchor_record(
    closure: Mapping[str, Any],
    *,
    kind: str,
    trajectory_id: str,
    name: str,
) -> tuple[Path, dict[str, Any]]:
    records = closure.get("records")
    if not isinstance(records, list):
        contractlib.fail("matrix closure lacks reservation records")
    matches = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("kind") == kind
        and record.get("trajectory_id") == trajectory_id
    ]
    if len(matches) != 1:
        contractlib.fail("matrix closure lacks one matching claim-anchor reservation")
    reservation = matches[0]
    frozen = reservation.get("claim_anchors")
    reserved_path = Path(str(reservation.get("reserved_path", {}).get("path", "")))
    _inspect_claim_anchor_manifest(reserved_path, kind=kind, frozen=frozen)
    if not isinstance(frozen, dict) or not isinstance(frozen.get("anchors"), list):
        contractlib.fail("matrix closure claim-anchor inventory is malformed")
    anchor_path = _claim_anchor_path(reserved_path, name)
    anchor_matches = [
        record
        for record in frozen["anchors"]
        if isinstance(record, dict) and record.get("path") == str(anchor_path)
    ]
    if len(anchor_matches) != 1:
        contractlib.fail(f"matrix closure lacks claim anchor {anchor_path}")
    return anchor_path, dict(anchor_matches[0])


def _consume_claim_anchor_once(
    path: Path,
    payload: Mapping[str, Any],
    *,
    frozen_inode: Mapping[str, Any],
) -> dict[str, Any]:
    """Consume one closure-bound inode; partial writes permanently fail closed."""

    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    )
    flags = os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            contractlib.fail(f"claim anchor is concurrently being consumed: {path}")
        observed = os.fstat(descriptor)
        expected = {
            "path": str(path),
            "kind": "file",
            "device": observed.st_dev,
            "inode": observed.st_ino,
        }
        frozen_stable = {key: frozen_inode.get(key) for key in ("path", "kind", "device", "inode")}
        if expected != frozen_stable:
            contractlib.fail(f"claim-anchor inode differs from reservation closure: {path}")
        if observed.st_size != 0 or stat.S_IMODE(observed.st_mode) != 0o660:
            contractlib.fail(f"claim anchor was already consumed or corrupted: {path}")
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write while consuming claim anchor: {path}")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o440)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    observed_payload = contractlib.load_json_object(path, label="consumed claim anchor")
    contractlib.require_read_only(path, label="consumed claim anchor")
    _verify_self_hash(observed_payload, label="consumed claim anchor")
    if observed_payload != dict(payload):
        contractlib.fail(f"consumed claim-anchor payload differs after write: {path}")
    return observed_payload


def _execution_anchor_record(
    path: Path,
    *,
    guard_path: Path,
    initial_mode: int,
) -> dict[str, Any]:
    observed = path.lstat()
    guard_observed = guard_path.lstat()
    if (
        path.is_symlink()
        or not path.is_file()
        or guard_path.is_symlink()
        or not guard_path.is_file()
        or observed.st_size != 0
        or stat.S_IMODE(observed.st_mode) != initial_mode
        or (observed.st_dev, observed.st_ino) != (guard_observed.st_dev, guard_observed.st_ino)
        or observed.st_nlink != 2
        or guard_observed.st_nlink != 2
    ):
        contractlib.fail(f"memory-gate execution anchor is not pristine: {path}")
    return {
        **_path_stat_identity(path),
        "generation_guard": _path_stat_identity(guard_path),
        "expected_link_count": 2,
        "initial_mode": initial_mode,
        "initial_size_bytes": 0,
    }


def _create_memory_gate_execution_anchors(anchor_root: Path) -> tuple[dict[str, Any], int]:
    guard_root = anchor_root / MEMORY_GATE_ANCHOR_GUARD_DIRNAME
    guard_root.mkdir(mode=0o700)
    attempt_path = anchor_root / MEMORY_GATE_ATTEMPT_FILENAME
    _create_empty_claim_anchor(attempt_path)
    transcript_path = anchor_root / MEMORY_GATE_TRANSCRIPT_FILENAME
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    transcript_descriptor = os.open(transcript_path, flags, 0o440)
    try:
        os.fchmod(transcript_descriptor, 0o440)
        fcntl.flock(transcript_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.fsync(transcript_descriptor)
        outcome_path = anchor_root / MEMORY_GATE_OUTCOME_FILENAME
        _create_empty_claim_anchor(outcome_path)
        for path in (attempt_path, transcript_path, outcome_path):
            os.link(
                path,
                guard_root / path.name,
                follow_symlinks=False,
            )
        guard_root.chmod(0o550)
        anchors = {
            "guard_directory": _path_directory_change_identity(guard_root),
            "attempt": _execution_anchor_record(
                attempt_path,
                guard_path=guard_root / attempt_path.name,
                initial_mode=0o660,
            ),
            "transcript": _execution_anchor_record(
                transcript_path,
                guard_path=guard_root / transcript_path.name,
                initial_mode=0o440,
            ),
            "outcome": _execution_anchor_record(
                outcome_path,
                guard_path=guard_root / outcome_path.name,
                initial_mode=0o660,
            ),
        }
    except Exception:
        os.close(transcript_descriptor)
        raise
    return anchors, transcript_descriptor


def _inspect_memory_gate_execution_anchors(
    claim: Mapping[str, Any],
    *,
    require_pristine_container: bool = False,
) -> tuple[Path, Path, dict[str, Any] | None]:
    attempt_path = Path(str(claim.get("attempt_path", "")))
    anchor_root = attempt_path.parent
    expected_names = {
        MEMORY_GATE_ATTEMPT_FILENAME,
        MEMORY_GATE_TRANSCRIPT_FILENAME,
        MEMORY_GATE_OUTCOME_FILENAME,
        MEMORY_GATE_ANCHOR_GUARD_DIRNAME,
    }
    if (
        attempt_path.name != MEMORY_GATE_ATTEMPT_FILENAME
        or anchor_root.name != CLAIM_ANCHOR_DIRNAME
        or anchor_root.is_symlink()
        or not anchor_root.is_dir()
        or stat.S_IMODE(anchor_root.lstat().st_mode) != 0o550
        or {path.name for path in anchor_root.iterdir()} != expected_names
    ):
        contractlib.fail("memory-gate execution-anchor directory differs from its claim")
    anchors = claim.get("execution_anchors")
    if not isinstance(anchors, dict) or set(anchors) != {
        "directory",
        "guard_directory",
        "container_directory",
        "output_directory",
        "attempt",
        "transcript",
        "outcome",
    }:
        contractlib.fail("memory-gate claim lacks exact execution anchors")
    if anchors.get("directory") != _path_directory_change_identity(anchor_root):
        contractlib.fail("memory-gate execution-anchor directory was deleted/recreated/altered")
    guard_root = anchor_root / MEMORY_GATE_ANCHOR_GUARD_DIRNAME
    expected_guard_names = {
        MEMORY_GATE_ATTEMPT_FILENAME,
        MEMORY_GATE_TRANSCRIPT_FILENAME,
        MEMORY_GATE_OUTCOME_FILENAME,
    }
    if (
        guard_root.is_symlink()
        or not guard_root.is_dir()
        or stat.S_IMODE(guard_root.lstat().st_mode) != 0o550
        or {path.name for path in guard_root.iterdir()} != expected_guard_names
        or anchors.get("guard_directory") != _path_directory_change_identity(guard_root)
    ):
        contractlib.fail("memory-gate generation-guard directory differs from its claim")
    frozen_container = anchors.get("container_directory")
    live_container = _path_directory_change_identity(anchor_root.parent)
    if not isinstance(frozen_container, dict):
        contractlib.fail("memory-gate execution-anchor container identity is malformed")
    frozen_container_stable = {
        key: frozen_container.get(key) for key in ("path", "kind", "device", "inode")
    }
    live_container_stable = {
        key: live_container.get(key) for key in ("path", "kind", "device", "inode")
    }
    if (
        live_container_stable != frozen_container_stable
        or live_container.get("mode") not in {frozen_container.get("mode"), 0o550}
        or (require_pristine_container and live_container != frozen_container)
    ):
        contractlib.fail("memory-gate containing directory was deleted/recreated/altered")
    frozen_output = anchors.get("output_directory")
    if not isinstance(frozen_output, dict):
        contractlib.fail("memory-gate output-container identity is malformed")

    transcript_path = anchor_root / MEMORY_GATE_TRANSCRIPT_FILENAME
    outcome_path = anchor_root / MEMORY_GATE_OUTCOME_FILENAME
    for name, path, initial_mode in (
        ("attempt", attempt_path, 0o660),
        ("transcript", transcript_path, 0o440),
        ("outcome", outcome_path, 0o660),
    ):
        frozen = anchors.get(name)
        expected_frozen_keys = {
            "path",
            "kind",
            "device",
            "inode",
            "generation_guard",
            "expected_link_count",
            "initial_mode",
            "initial_size_bytes",
        }
        if not isinstance(frozen, dict) or set(frozen) != expected_frozen_keys:
            contractlib.fail(f"memory-gate {name} anchor identity is malformed")
        guard_path = guard_root / path.name
        stable = _path_stat_identity(path)
        expected_stable = {key: frozen.get(key) for key in ("path", "kind", "device", "inode")}
        frozen_guard = frozen.get("generation_guard")
        guard_stable = _path_stat_identity(guard_path)
        observed = path.lstat()
        guard_observed = guard_path.lstat()
        if (
            stable != expected_stable
            or not isinstance(frozen_guard, dict)
            or set(frozen_guard) != {"path", "kind", "device", "inode"}
            or guard_stable != frozen_guard
            or (observed.st_dev, observed.st_ino) != (guard_observed.st_dev, guard_observed.st_ino)
            or frozen.get("expected_link_count") != 2
            or observed.st_nlink != 2
            or guard_observed.st_nlink != 2
            or frozen.get("initial_mode") != initial_mode
            or frozen.get("initial_size_bytes") != 0
        ):
            contractlib.fail(
                f"memory-gate {name} anchor or generation guard was deleted, "
                "recreated, or multiply linked"
            )

    transcript_stat = transcript_path.lstat()
    if stat.S_IMODE(transcript_stat.st_mode) != 0o440:
        contractlib.fail("memory-gate transcript is writable or partially created")
    outcome_stat = outcome_path.lstat()
    outcome_mode = stat.S_IMODE(outcome_stat.st_mode)
    if outcome_stat.st_size == 0:
        if outcome_mode != 0o660:
            contractlib.fail("unused memory-gate outcome anchor has unsafe mode")
        outcome = None
    elif outcome_mode == 0o440:
        outcome = contractlib.load_json_object(
            outcome_path,
            label="memory-gate execution outcome",
        )
        _verify_self_hash(outcome, label="memory-gate execution outcome")
    else:
        contractlib.fail("memory-gate outcome anchor is partially consumed or writable")
    live_output = _path_directory_change_identity(anchor_root.parent.parent)
    frozen_output_stable = {
        key: frozen_output.get(key) for key in ("path", "kind", "device", "inode")
    }
    live_output_stable = {key: live_output.get(key) for key in ("path", "kind", "device", "inode")}
    if (
        live_output_stable != frozen_output_stable
        or live_output.get("mode") not in {frozen_output.get("mode"), 0o550}
        or (outcome is None and live_output != frozen_output)
    ):
        contractlib.fail("memory-gate output directory entry was deleted/recreated/altered")
    return transcript_path, outcome_path, outcome


def _inspect_attempt_claim(
    path: Path,
    *,
    expected_protocol: str,
    expected_status: str,
    expected_kind: str,
    trajectory_id: str,
    ledger_identity: Mapping[str, Any] | None,
    reservation_identity: Mapping[str, Any] | None,
    command: Mapping[str, Any],
    resume_step: int | None,
) -> dict[str, Any]:
    contractlib.require_read_only(path, label="attempt claim")
    payload = contractlib.load_json_object(path, label="attempt claim")
    _verify_self_hash(payload, label="attempt claim")
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "stage",
        "development_only",
        "publication_eligible",
        "sota_eligible",
        "kind",
        "trajectory_id",
        "created_at_utc",
        "attempt_path",
        "ledger",
        "reservation",
        "authorized_command_sha256",
        "resume_checkpoint",
        "claim_semantics",
        "receipt_sha256",
    }
    if expected_kind == "memory_gate":
        expected_keys.add("execution_anchors")
    resume = payload.get("resume_checkpoint")
    if resume_step is None:
        resume_valid = resume is None
    else:
        resume_valid = isinstance(resume, dict) and resume.get("step") == resume_step
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != 1
        or payload.get("protocol") != expected_protocol
        or payload.get("status") != expected_status
        or payload.get("stage") != "grpo"
        or payload.get("development_only") is not True
        or payload.get("publication_eligible") is not False
        or payload.get("sota_eligible") is not False
        or payload.get("kind") != expected_kind
        or payload.get("trajectory_id") != trajectory_id
        or payload.get("attempt_path") != str(path)
        or payload.get("ledger") != ledger_identity
        or payload.get("reservation") != reservation_identity
        or payload.get("authorized_command_sha256") != contractlib.sha256_json(command)
        or not resume_valid
        or (
            expected_kind == "memory_gate"
            and not isinstance(payload.get("execution_anchors"), dict)
        )
    ):
        contractlib.fail(f"attempt claim differs from frozen authority: {path}")
    _validate_utc_timestamp(payload.get("created_at_utc"), label="attempt claim")
    return payload


def claim_memory_gate_attempt(
    contract_path: Path,
    *,
    actual_payload_argv: Sequence[str],
    actual_environment: Mapping[str, str],
) -> tuple[dict[str, Any], int]:
    """Atomically consume the gate attempt and retain its locked transcript fd."""

    contract_path = contract_path.expanduser().absolute()
    contract = contractlib.load_json_object(contract_path, label="frozen pilot contract")
    contractlib.verify_contract(
        contract,
        require_fresh_output=False,
        contract_path=contract_path,
    )
    command = build_plan(contract, contract_path=contract_path)["memory_gate_command"]
    _assert_exact_attempt_material(
        command,
        actual_payload_argv=actual_payload_argv,
        actual_environment=actual_environment,
        consumed_environment_names=("CLASP_PILOT_CREATE_TMPDIR",),
    )
    output_root = Path(str(contract["paths"]["output_root"]))
    gate_dir = output_root / MEMORY_GATE_DIRNAME
    if gate_dir.is_symlink() or not gate_dir.is_dir():
        contractlib.fail("runtime guard did not create the preregistered memory-gate directory")
    expected_top = {MEMORY_GATE_DIRNAME}
    _require_top_level(output_root, expected_top, phase="memory-gate first attempt")
    names = {item.name for item in gate_dir.iterdir()}
    if names != {"_runtime_tmp"}:
        contractlib.fail(f"memory-gate first-attempt directory is contaminated: {names}")
    anchor_root = gate_dir / CLAIM_ANCHOR_DIRNAME
    if anchor_root.exists() or anchor_root.is_symlink():
        contractlib.fail("memory-gate claim-anchor root already exists")
    anchor_root.mkdir(mode=0o700)
    execution_anchors, transcript_descriptor = _create_memory_gate_execution_anchors(anchor_root)
    try:
        attempt_path = anchor_root / MEMORY_GATE_ATTEMPT_FILENAME
        anchor_root.chmod(0o550)
        execution_anchors["directory"] = _path_directory_change_identity(anchor_root)
        execution_anchors["container_directory"] = _path_directory_change_identity(gate_dir)
        execution_anchors["output_directory"] = _path_directory_change_identity(output_root)
        payload = _attempt_payload(
            protocol=MEMORY_GATE_ATTEMPT_PROTOCOL,
            status="development_memory_gate_attempt_started",
            kind="memory_gate",
            trajectory_id="engineering-gate-D5-seed42",
            attempt_path=attempt_path,
            ledger_path=None,
            ledger=None,
            reservation_path=None,
            reservation=None,
            command=command,
            execution_anchors=execution_anchors,
        )
        _consume_claim_anchor_once(
            attempt_path,
            payload,
            frozen_inode=execution_anchors["attempt"],
        )
        claim = _inspect_attempt_claim(
            attempt_path,
            expected_protocol=MEMORY_GATE_ATTEMPT_PROTOCOL,
            expected_status="development_memory_gate_attempt_started",
            expected_kind="memory_gate",
            trajectory_id="engineering-gate-D5-seed42",
            ledger_identity=None,
            reservation_identity=None,
            command=command,
            resume_step=None,
        )
        _inspect_memory_gate_execution_anchors(claim, require_pristine_container=True)
    except Exception:
        os.close(transcript_descriptor)
        raise
    return claim, transcript_descriptor


def _write_descriptor_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write to memory-gate execution evidence")
        view = view[written:]


def _write_transcript_event(descriptor: int, event: Mapping[str, Any]) -> dict[str, Any]:
    sealed_event = _self_hashed(event)
    encoded = contractlib.canonical_json_bytes(sealed_event) + b"\n"
    _write_descriptor_all(descriptor, encoded)
    os.fsync(descriptor)
    return sealed_event


def _memory_gate_transcript_identity(
    path: Path,
    *,
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    anchors = claim.get("execution_anchors")
    frozen = anchors.get("transcript") if isinstance(anchors, dict) else None
    if not isinstance(frozen, dict) or frozen.get("path") != str(path):
        contractlib.fail("memory-gate transcript path differs from its attempt claim")
    before = path.lstat()
    digest = contractlib.sha256_file(path)
    after = path.lstat()
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        contractlib.fail("memory-gate transcript changed while it was hashed")
    if stat.S_IMODE(after.st_mode) != 0o440:
        contractlib.fail("memory-gate transcript must remain mode 0440")
    frozen_stable = {key: frozen.get(key) for key in ("path", "kind", "device", "inode")}
    live_stable = {
        "path": str(path),
        "kind": "file",
        "device": after.st_dev,
        "inode": after.st_ino,
    }
    if live_stable != frozen_stable:
        contractlib.fail("memory-gate transcript inode differs from its attempt claim")
    return {
        "path": str(path),
        "device": after.st_dev,
        "inode": after.st_ino,
        "mode": 0o440,
        "size_bytes": after.st_size,
        "sha256": digest,
        "stream_semantics": "combined_child_stdout_stderr_in_observed_byte_order",
    }


def _memory_gate_execution_anchor_closure(claim: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the sealed anchor and its containing gate after child execution."""

    attempt_path = Path(str(claim.get("attempt_path", "")))
    anchor_root = attempt_path.parent
    anchors = claim.get("execution_anchors")
    frozen_anchor = anchors.get("directory") if isinstance(anchors, dict) else None
    live_anchor = _path_directory_change_identity(anchor_root)
    if live_anchor != frozen_anchor:
        contractlib.fail("memory-gate execution-anchor directory changed before outcome closure")
    return {
        "anchor_directory": live_anchor,
        "generation_guard_directory": _path_directory_change_identity(
            anchor_root / MEMORY_GATE_ANCHOR_GUARD_DIRNAME
        ),
        "containing_gate_directory": _path_directory_change_identity(anchor_root.parent),
    }


def _memory_gate_live_environment(command: Mapping[str, Any]) -> dict[str, str]:
    environment = command.get("env")
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in environment.items()
    ):
        contractlib.fail("memory-gate command environment is malformed")
    result = dict(environment)
    result.pop("CLASP_PILOT_CREATE_TMPDIR", None)
    return result


def _build_memory_gate_execution_outcome(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    command: Mapping[str, Any],
    claim: Mapping[str, Any],
    outcome_path: Path,
    payload_argv: Sequence[str],
    environment: Mapping[str, str],
    child: Mapping[str, Any],
    status: str,
    operator_finalized: bool,
    post_execution_absence: Mapping[str, Any],
    operator_finalization_evidence: Mapping[str, Any] | None,
    sealed_failure_tree: Mapping[str, Any] | None,
) -> dict[str, Any]:
    transcript_path = Path(str(claim["execution_anchors"]["transcript"]["path"]))
    return _self_hashed(
        {
            "schema_version": 1,
            "protocol": MEMORY_GATE_OUTCOME_PROTOCOL,
            "status": status,
            "stage": "grpo",
            "development_only": True,
            "publication_eligible": False,
            "sota_eligible": False,
            "scientific_result_eligible": False,
            "created_at_utc": _utc_now(),
            "outcome_path": str(outcome_path),
            "contract": _contract_identity(contract, contract_path),
            "attempt_claim": _file_identity(Path(str(claim["attempt_path"])), claim),
            "authorized_command_sha256": contractlib.sha256_json(command),
            "cwd": str(command.get("cwd", "")),
            "execution_authority": {
                "payload_argv": list(payload_argv),
                "payload_argv_sha256": contractlib.sha256_json(list(payload_argv)),
                "environment": dict(environment),
                "environment_sha256": contractlib.sha256_json(dict(environment)),
            },
            "transcript": _memory_gate_transcript_identity(
                transcript_path,
                claim=claim,
            ),
            "execution_anchor_closure": _memory_gate_execution_anchor_closure(claim),
            "child": dict(child),
            "operator_finalized": operator_finalized,
            "post_execution_absence": dict(post_execution_absence),
            "operator_finalization_evidence": (
                dict(operator_finalization_evidence)
                if operator_finalization_evidence is not None
                else None
            ),
            "sealed_failure_tree_excluding_outcome": (
                dict(sealed_failure_tree) if sealed_failure_tree is not None else None
            ),
            "retry_allowed": False,
            "matrix_authorized": False,
            "claim_limit": (
                "terminal engineering-gate execution evidence only; never scientific, "
                "publication, benchmark, or SOTA evidence"
            ),
        }
    )


def _persist_memory_gate_execution_outcome(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    command: Mapping[str, Any],
    claim: Mapping[str, Any],
    payload_argv: Sequence[str],
    environment: Mapping[str, str],
    child: Mapping[str, Any],
    status: str,
    operator_finalized: bool,
    post_execution_absence: Mapping[str, Any],
    operator_finalization_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(str(contract["paths"]["output_root"]))
    _, outcome_path, existing = _inspect_memory_gate_execution_anchors(claim)
    if existing is not None:
        contractlib.fail("memory-gate execution outcome was already consumed")
    frozen = claim["execution_anchors"]["outcome"]
    flags = os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(outcome_path, flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = os.fstat(descriptor)
        frozen_stable = {key: frozen.get(key) for key in ("path", "kind", "device", "inode")}
        live_stable = {
            "path": str(outcome_path),
            "kind": "file",
            "device": observed.st_dev,
            "inode": observed.st_ino,
        }
        if (
            live_stable != frozen_stable
            or observed.st_size != 0
            or stat.S_IMODE(observed.st_mode) != 0o660
        ):
            contractlib.fail("memory-gate outcome inode was consumed, replaced, or corrupted")
        os.fchmod(descriptor, 0o440)
        is_failure = status != MEMORY_GATE_SUCCESS_STATUS
        if is_failure:
            _seal_tree_read_only(output_root)
            sealed_tree = _stable_failure_tree_manifest(
                output_root,
                outcome_path=outcome_path,
                claim=claim,
            )
        else:
            # A zero exit is necessary but not sufficient for PASS. Seal the gate
            # immediately so the later checkpoint audit observes exactly the tree
            # whose post-execution directory identity the zero receipt binds.
            _seal_tree_read_only(outcome_path.parents[1])
            sealed_tree = None
        outcome = _build_memory_gate_execution_outcome(
            contract,
            contract_path=contract_path,
            command=command,
            claim=claim,
            outcome_path=outcome_path,
            payload_argv=payload_argv,
            environment=environment,
            child=child,
            status=status,
            operator_finalized=operator_finalized,
            post_execution_absence=post_execution_absence,
            operator_finalization_evidence=operator_finalization_evidence,
            sealed_failure_tree=sealed_tree,
        )
        encoded = (
            json.dumps(outcome, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
        )
        _write_descriptor_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return inspect_memory_gate_execution_outcome(
        outcome_path,
        contract,
        contract_path=contract_path,
        command=command,
        claim=claim,
    )


def inspect_memory_gate_execution_outcome(
    path: Path,
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    command: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    outcome_path = path.expanduser().absolute()
    contractlib.require_read_only(outcome_path, label="memory-gate execution outcome")
    outcome = contractlib.load_json_object(outcome_path, label="memory-gate execution outcome")
    _verify_self_hash(outcome, label="memory-gate execution outcome")
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "stage",
        "development_only",
        "publication_eligible",
        "sota_eligible",
        "scientific_result_eligible",
        "created_at_utc",
        "outcome_path",
        "contract",
        "attempt_claim",
        "authorized_command_sha256",
        "cwd",
        "execution_authority",
        "transcript",
        "execution_anchor_closure",
        "child",
        "operator_finalized",
        "post_execution_absence",
        "operator_finalization_evidence",
        "sealed_failure_tree_excluding_outcome",
        "retry_allowed",
        "matrix_authorized",
        "claim_limit",
        "receipt_sha256",
    }
    statuses = {
        MEMORY_GATE_SUCCESS_STATUS,
        MEMORY_GATE_FAILURE_STATUS,
        MEMORY_GATE_SUPERVISOR_FAILURE_STATUS,
        MEMORY_GATE_UNKNOWN_STATUS,
    }
    if (
        set(outcome) != expected_keys
        or outcome.get("schema_version") != 1
        or outcome.get("protocol") != MEMORY_GATE_OUTCOME_PROTOCOL
        or outcome.get("status") not in statuses
        or outcome.get("stage") != "grpo"
        or outcome.get("development_only") is not True
        or outcome.get("publication_eligible") is not False
        or outcome.get("sota_eligible") is not False
        or outcome.get("scientific_result_eligible") is not False
        or outcome.get("outcome_path") != str(outcome_path)
        or outcome.get("contract") != _contract_identity(contract, contract_path)
        or outcome.get("attempt_claim") != _file_identity(Path(str(claim["attempt_path"])), claim)
        or outcome.get("authorized_command_sha256") != contractlib.sha256_json(command)
        or outcome.get("cwd") != str(command.get("cwd", ""))
        or outcome.get("retry_allowed") is not False
        or outcome.get("matrix_authorized") is not False
        or outcome.get("claim_limit")
        != (
            "terminal engineering-gate execution evidence only; never scientific, "
            "publication, benchmark, or SOTA evidence"
        )
    ):
        contractlib.fail("memory-gate execution outcome identity/claim mismatch")
    _validate_utc_timestamp(outcome.get("created_at_utc"), label="memory-gate outcome")
    expected_payload = _authorized_inner_payload(command)
    expected_environment = _memory_gate_live_environment(command)
    authority = outcome.get("execution_authority")
    if authority != {
        "payload_argv": expected_payload,
        "payload_argv_sha256": contractlib.sha256_json(expected_payload),
        "environment": expected_environment,
        "environment_sha256": contractlib.sha256_json(expected_environment),
    }:
        contractlib.fail("memory-gate outcome does not bind exact argv/environment")
    transcript_path, _, _ = _inspect_memory_gate_execution_anchors(claim)
    if outcome.get("transcript") != _memory_gate_transcript_identity(
        transcript_path,
        claim=claim,
    ):
        contractlib.fail("memory-gate transcript differs from its execution outcome")
    if outcome.get("execution_anchor_closure") != _memory_gate_execution_anchor_closure(claim):
        contractlib.fail("memory-gate execution-anchor closure differs from its outcome")

    child = outcome.get("child")
    if not isinstance(child, dict) or set(child) != {
        "child_started",
        "exit_known",
        "returncode",
        "exit_code",
        "signal",
        "classification",
        "error_type",
        "error_message",
        "process_identity",
    }:
        contractlib.fail("memory-gate child outcome schema mismatch")
    status = outcome["status"]
    returncode = child.get("returncode")
    exit_code = child.get("exit_code")
    signal_number = child.get("signal")
    process_identity = child.get("process_identity")
    if process_identity is not None:
        process_identity = _validate_memory_gate_process_identity(process_identity)
    if status == MEMORY_GATE_SUCCESS_STATUS:
        valid_child = (
            child.get("child_started") is True
            and child.get("exit_known") is True
            and returncode == 0
            and not isinstance(returncode, bool)
            and exit_code == 0
            and not isinstance(exit_code, bool)
            and signal_number is None
            and child.get("classification") == "exited_zero"
            and child.get("error_type") is None
            and child.get("error_message") is None
            and outcome.get("operator_finalized") is False
            and outcome.get("operator_finalization_evidence") is None
            and outcome.get("sealed_failure_tree_excluding_outcome") is None
            and process_identity is not None
        )
    elif status == MEMORY_GATE_FAILURE_STATUS:
        valid_child = (
            child.get("child_started") is True
            and child.get("exit_known") is True
            and isinstance(returncode, int)
            and not isinstance(returncode, bool)
            and returncode != 0
            and (
                (
                    returncode > 0
                    and returncode <= 255
                    and exit_code == returncode
                    and not isinstance(exit_code, bool)
                    and signal_number is None
                    and child.get("classification") == "nonzero_exit"
                )
                or (
                    returncode < 0
                    and exit_code is None
                    and signal_number == -returncode
                    and isinstance(signal_number, int)
                    and not isinstance(signal_number, bool)
                    and 0 < signal_number < signal.NSIG
                    and child.get("classification") == "signal"
                )
            )
            and child.get("error_type") is None
            and child.get("error_message") is None
            and outcome.get("operator_finalized") is False
            and outcome.get("operator_finalization_evidence") is None
            and process_identity is not None
        )
    elif status == MEMORY_GATE_SUPERVISOR_FAILURE_STATUS:
        valid_child = (
            isinstance(child.get("child_started"), bool)
            and child.get("exit_known") is False
            and returncode is None
            and exit_code is None
            and signal_number is None
            and child.get("classification") in {"spawn_error", "supervisor_error"}
            and isinstance(child.get("error_type"), str)
            and bool(child["error_type"])
            and isinstance(child.get("error_message"), str)
            and outcome.get("operator_finalized") is False
            and outcome.get("operator_finalization_evidence") is None
            and (child.get("child_started") is True or process_identity is None)
            and (
                child.get("classification") == "supervisor_error"
                or (child.get("child_started") is False and process_identity is None)
            )
        )
    else:
        valid_child = (
            child.get("child_started") in {None, True}
            and child.get("exit_known") is False
            and returncode is None
            and exit_code is None
            and signal_number is None
            and child.get("classification") == "abrupt_supervisor_death_unknown"
            and child.get("error_type") == "abrupt_supervisor_death"
            and child.get("error_message") is None
            and outcome.get("operator_finalized") is True
            and isinstance(outcome.get("operator_finalization_evidence"), dict)
            and (
                (child.get("child_started") is None and process_identity is None)
                or (child.get("child_started") is True and process_identity is not None)
            )
        )
    if not valid_child:
        contractlib.fail("memory-gate child exit/signal evidence is invalid")

    post_absence = outcome.get("post_execution_absence")
    process_absence = (
        post_absence.get("process_absence") if isinstance(post_absence, dict) else None
    )
    gpu_absence = post_absence.get("gpu_absence") if isinstance(post_absence, dict) else None
    process_absence_keys = {
        "protocol",
        "procfs_root",
        "double_inspection_exact_equal",
        "spawned_process_identity",
        "original_process_status",
        "nonzombie_process_group_members",
        "authorized_payload_process_matches",
        "supervisor_wrapper_process_matches",
    }
    gpu_absence_keys = {
        "protocol",
        "double_inspection_exact_equal",
        "normalized_probe_executable",
        "query",
        "stdout_sha256",
        "target_gpu_uuid",
        "target_gpu_processes",
    }
    gpu_stdout_sha = gpu_absence.get("stdout_sha256") if isinstance(gpu_absence, dict) else None
    if (
        not isinstance(post_absence, dict)
        or set(post_absence) != {"process_absence", "gpu_absence"}
        or not isinstance(process_absence, dict)
        or set(process_absence) != process_absence_keys
        or process_absence.get("protocol") != "clasp-memory-gate-operator-process-absence-v1"
        or process_absence.get("procfs_root") != "/proc"
        or process_absence.get("spawned_process_identity") != process_identity
        or process_absence.get("original_process_status")
        not in {
            "no_spawn_marker",
            "absent",
            "pid_reused_original_absent",
            "terminated_zombie",
        }
        or process_absence.get("nonzombie_process_group_members") != []
        or process_absence.get("authorized_payload_process_matches") != []
        or process_absence.get("supervisor_wrapper_process_matches") != []
        or process_absence.get("double_inspection_exact_equal") is not True
        or not isinstance(gpu_absence, dict)
        or set(gpu_absence) != gpu_absence_keys
        or gpu_absence.get("protocol") != "clasp-memory-gate-operator-gpu-absence-v1"
        or gpu_absence.get("normalized_probe_executable")
        != contractlib.FROZEN_NVIDIA_SMI_NORMALIZED_IDENTITY
        or gpu_absence.get("query")
        != ["--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"]
        or not isinstance(gpu_stdout_sha, str)
        or len(gpu_stdout_sha) != 64
        or any(character not in "0123456789abcdef" for character in gpu_stdout_sha)
        or gpu_absence.get("target_gpu_uuid") != str(contractlib.EXPECTED_LOCAL_HARDWARE["uuid"])
        or gpu_absence.get("target_gpu_processes") != []
        or gpu_absence.get("double_inspection_exact_equal") is not True
    ):
        contractlib.fail("memory-gate post-execution process/GPU absence evidence is invalid")

    if status == MEMORY_GATE_UNKNOWN_STATUS:
        finalization = outcome["operator_finalization_evidence"]
        transcript_descriptor = os.open(transcript_path, os.O_RDONLY)
        try:
            expected_prefix = _inspect_supervisor_transcript_prefix(transcript_descriptor)
        finally:
            os.close(transcript_descriptor)
        if (
            set(finalization) != {"transcript_prefix"}
            or finalization.get("transcript_prefix") != expected_prefix
        ):
            contractlib.fail("operator memory-gate absence evidence is invalid")

    if status != MEMORY_GATE_SUCCESS_STATUS:
        output_root = Path(str(contract["paths"]["output_root"]))
        contractlib.require_read_only(
            output_root,
            label="failed memory-gate output tree",
            recursive=True,
        )
        manifest = _stable_failure_tree_manifest(
            output_root,
            outcome_path=outcome_path,
            claim=claim,
        )
        if outcome.get("sealed_failure_tree_excluding_outcome") != manifest:
            contractlib.fail("sealed failed memory-gate tree differs from its outcome")
    return outcome


def _inspect_memory_gate_zero_outcome(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
) -> tuple[Path, dict[str, Any]]:
    gate_dir = Path(str(contract["paths"]["output_root"])) / MEMORY_GATE_DIRNAME
    command = build_plan(contract, contract_path=contract_path)["memory_gate_command"]
    attempt_path = gate_dir / CLAIM_ANCHOR_DIRNAME / MEMORY_GATE_ATTEMPT_FILENAME
    claim = _inspect_attempt_claim(
        attempt_path,
        expected_protocol=MEMORY_GATE_ATTEMPT_PROTOCOL,
        expected_status="development_memory_gate_attempt_started",
        expected_kind="memory_gate",
        trajectory_id="engineering-gate-D5-seed42",
        ledger_identity=None,
        reservation_identity=None,
        command=command,
        resume_step=None,
    )
    _, outcome_path, outcome = _inspect_memory_gate_execution_anchors(claim)
    if outcome is None:
        contractlib.fail("memory-gate zero-exit outcome is absent")
    outcome = inspect_memory_gate_execution_outcome(
        outcome_path,
        contract,
        contract_path=contract_path,
        command=command,
        claim=claim,
    )
    if outcome.get("status") != MEMORY_GATE_SUCCESS_STATUS:
        contractlib.fail("memory-gate is terminal FAILED and cannot authorize PASS or matrix")
    return outcome_path, outcome


def _parse_linux_process_stat(pid: int, raw: str) -> dict[str, Any]:
    closing = raw.rfind(")")
    if closing < 0 or not raw.startswith(f"{pid} ("):
        contractlib.fail(f"cannot parse Linux process identity for pid {pid}")
    fields = raw[closing + 2 :].split()
    if len(fields) < 20:
        contractlib.fail(f"Linux process identity is truncated for pid {pid}")
    try:
        return {
            "pid": pid,
            "state": fields[0],
            "parent_pid": int(fields[1]),
            "process_group_id": int(fields[2]),
            "session_id": int(fields[3]),
            "start_time_ticks": int(fields[19]),
        }
    except ValueError as exc:
        contractlib.fail(f"Linux process identity is malformed for pid {pid}: {exc}")


def _read_linux_process_identity(pid: int) -> dict[str, Any] | None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        contractlib.fail(f"invalid process id in memory-gate evidence: {pid!r}")
    path = Path("/proc") / str(pid) / "stat"
    try:
        before = path.read_text(encoding="utf-8")
        after = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        contractlib.fail(f"cannot inspect memory-gate process {pid}: {exc}")
    before_identity = _parse_linux_process_stat(pid, before)
    after_identity = _parse_linux_process_stat(pid, after)
    stable_fields = (
        "pid",
        "parent_pid",
        "process_group_id",
        "session_id",
        "start_time_ticks",
    )
    if any(before_identity[field] != after_identity[field] for field in stable_fields):
        contractlib.fail(f"memory-gate process identity changed while read: {pid}")
    return after_identity


def _validate_memory_gate_process_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "pid",
        "state",
        "parent_pid",
        "process_group_id",
        "session_id",
        "start_time_ticks",
    }:
        contractlib.fail("memory-gate child process identity schema mismatch")
    for name in (
        "pid",
        "parent_pid",
        "process_group_id",
        "session_id",
        "start_time_ticks",
    ):
        observed = value.get(name)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
            contractlib.fail(f"memory-gate child process identity has invalid {name}")
    if not isinstance(value.get("state"), str) or len(value["state"]) != 1:
        contractlib.fail("memory-gate child process identity has invalid state")
    if value["process_group_id"] != value["pid"] or value["session_id"] != value["pid"]:
        contractlib.fail("memory-gate child does not own its isolated process group/session")
    return dict(value)


def _memory_gate_exec_wrapper_argv(
    payload_argv: Sequence[str],
    *,
    expected_supervisor_pid: int,
) -> list[str]:
    """Build the audited Linux parent-death/process-group wrapper."""

    if (
        not payload_argv
        or isinstance(expected_supervisor_pid, bool)
        or expected_supervisor_pid <= 0
    ):
        contractlib.fail("memory-gate exec wrapper received invalid authority")
    code = (
        "import ctypes,os,signal,sys\n"
        "parent=int(sys.argv[1])\n"
        "payload=sys.argv[2:]\n"
        "libc=ctypes.CDLL(None,use_errno=True)\n"
        "if libc.prctl(1,signal.SIGTERM,0,0,0)!=0: os._exit(125)\n"
        "def terminate_group(signum,frame):\n"
        "    signal.signal(signum,signal.SIG_DFL)\n"
        "    os.killpg(os.getpgrp(),signum)\n"
        "for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):\n"
        "    signal.signal(signum,terminate_group)\n"
        "if os.getppid()!=parent: os.killpg(os.getpgrp(),signal.SIGKILL)\n"
        "try:\n"
        "    child=os.fork()\n"
        "except BaseException:\n"
        "    os._exit(125)\n"
        "if child==0:\n"
        "    try:\n"
        "        os.execve(payload[0],payload,dict(os.environ))\n"
        "    except BaseException:\n"
        "        os._exit(126)\n"
        "while True:\n"
        "    try:\n"
        "        _,status=os.waitpid(child,0)\n"
        "        break\n"
        "    except InterruptedError:\n"
        "        continue\n"
        "if os.WIFEXITED(status): os._exit(os.WEXITSTATUS(status))\n"
        "if os.WIFSIGNALED(status):\n"
        "    signum=os.WTERMSIG(status)\n"
        "    signal.signal(signum,signal.SIG_DFL)\n"
        "    os.kill(os.getpid(),signum)\n"
        "os._exit(125)\n"
    )
    return [
        str(payload_argv[0]),
        "-c",
        code,
        str(expected_supervisor_pid),
        *list(payload_argv),
    ]


def _signal_memory_gate_process_group(process: subprocess.Popen, signum: int) -> None:
    if process.poll() is not None:
        return
    try:
        observed_group = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    if observed_group != process.pid:
        contractlib.fail("memory-gate child escaped its isolated process group")
    os.killpg(observed_group, signum)


def _inspect_supervisor_transcript_prefix(descriptor: int) -> dict[str, Any]:
    observed = os.fstat(descriptor)
    prefix = os.pread(descriptor, min(observed.st_size, 65_536), 0)
    complete_lines = prefix.split(b"\n")[:-1]
    expected_events = ("supervisor_claimed", "child_spawn_pending", "child_spawned")
    events: list[dict[str, Any]] = []
    consumed_bytes = 0
    for index, encoded in enumerate(complete_lines[: len(expected_events)]):
        try:
            event = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            contractlib.fail(f"memory-gate supervisor transcript prefix is malformed: {exc}")
        if not isinstance(event, dict):
            contractlib.fail("memory-gate supervisor transcript event is not an object")
        _verify_self_hash(event, label="memory-gate supervisor transcript event")
        if (
            event.get("schema_version") != 1
            or event.get("protocol") != MEMORY_GATE_TRANSCRIPT_PROTOCOL
            or event.get("sequence") != index + 1
            or event.get("event") != expected_events[index]
        ):
            contractlib.fail("memory-gate supervisor transcript event order/identity mismatch")
        events.append(event)
        consumed_bytes += len(encoded) + 1
        if event["event"] == "child_spawn_pending" and (
            isinstance(event.get("supervisor_pid"), bool)
            or not isinstance(event.get("supervisor_pid"), int)
            or event["supervisor_pid"] <= 0
        ):
            contractlib.fail("memory-gate spawn-pending event has invalid supervisor pid")
        if event["event"] == "child_spawned":
            if event.get("process_role") != "parent_death_process_group_wrapper":
                contractlib.fail("memory-gate spawn event has invalid process role")
            _validate_memory_gate_process_identity(event.get("process_identity"))
            break
    if len(complete_lines) > len(events) and len(events) < len(expected_events):
        contractlib.fail("memory-gate transcript contains bytes before the spawn marker sequence")
    spawned = next((event for event in events if event["event"] == "child_spawned"), None)
    pending = next(
        (event for event in events if event["event"] == "child_spawn_pending"),
        None,
    )
    last = events[-1] if events else None
    return {
        "complete_event_count": len(events),
        "last_complete_event": (
            {
                "sequence": last["sequence"],
                "event": last["event"],
                "receipt_sha256": last["receipt_sha256"],
            }
            if last is not None
            else None
        ),
        "complete_prefix_size_bytes": consumed_bytes,
        "complete_prefix_sha256": hashlib.sha256(prefix[:consumed_bytes]).hexdigest(),
        "spawned_process_identity": (
            _validate_memory_gate_process_identity(spawned["process_identity"])
            if spawned is not None
            else None
        ),
        "pending_supervisor_pid": pending.get("supervisor_pid") if pending else None,
    }


def _relevant_linux_process_snapshot(
    *,
    spawned_identity: Mapping[str, Any] | None,
    payload_argv: Sequence[str],
    wrapper_argv: Sequence[str] | None,
) -> dict[str, Any]:
    target_group = (
        int(spawned_identity["process_group_id"]) if spawned_identity is not None else None
    )
    target_pid = int(spawned_identity["pid"]) if spawned_identity is not None else None
    target_start = (
        int(spawned_identity["start_time_ticks"]) if spawned_identity is not None else None
    )
    group_members = []
    payload_matches = []
    wrapper_matches = []
    original = "no_spawn_marker"
    current_uid = os.getuid()
    for path in sorted(Path("/proc").iterdir(), key=lambda item: item.name):
        if not path.name.isdigit():
            continue
        pid = int(path.name)
        try:
            if path.stat().st_uid != current_uid:
                continue
            record = _parse_linux_process_stat(
                pid,
                (path / "stat").read_text(encoding="utf-8"),
            )
            raw_argv = (path / "cmdline").read_bytes()
        except FileNotFoundError:
            continue
        except OSError as exc:
            contractlib.fail(f"cannot complete operator process guard for pid {pid}: {exc}")
        argv = [os.fsdecode(value) for value in raw_argv.split(b"\0") if value]
        if record["state"] != "Z" and argv == list(payload_argv):
            payload_matches.append(record)
        if record["state"] != "Z" and wrapper_argv is not None and argv == list(wrapper_argv):
            wrapper_matches.append(record)
        if target_group is not None and record["process_group_id"] == target_group:
            group_members.append(record)
        if pid == target_pid:
            if record["start_time_ticks"] == target_start:
                original = "terminated_zombie" if record["state"] == "Z" else "still_running"
            else:
                original = "pid_reused_original_absent"
    if spawned_identity is not None and original == "no_spawn_marker":
        original = "absent"
    return {
        "original_process_status": original,
        "nonzombie_process_group_members": [
            record for record in group_members if record["state"] != "Z"
        ],
        "authorized_payload_process_matches": payload_matches,
        "supervisor_wrapper_process_matches": wrapper_matches,
    }


def _audit_memory_gate_processes_absent(
    *,
    spawned_identity: Mapping[str, Any] | None,
    payload_argv: Sequence[str],
    wrapper_argv: Sequence[str] | None,
) -> dict[str, Any]:
    if spawned_identity is not None:
        spawned_identity = _validate_memory_gate_process_identity(spawned_identity)
    first = _relevant_linux_process_snapshot(
        spawned_identity=spawned_identity,
        payload_argv=payload_argv,
        wrapper_argv=wrapper_argv,
    )
    second = _relevant_linux_process_snapshot(
        spawned_identity=spawned_identity,
        payload_argv=payload_argv,
        wrapper_argv=wrapper_argv,
    )
    if first != second:
        contractlib.fail("operator process guard changed during its double inspection")
    if first["original_process_status"] == "still_running":
        contractlib.fail("claimed memory-gate child process is still running")
    if first["nonzombie_process_group_members"]:
        contractlib.fail("claimed memory-gate child process group still has live members")
    if first["authorized_payload_process_matches"]:
        contractlib.fail("an authorized memory-gate payload process is still running")
    if first["supervisor_wrapper_process_matches"]:
        contractlib.fail("a memory-gate parent-death wrapper process is still running")
    return {
        "protocol": "clasp-memory-gate-operator-process-absence-v1",
        "procfs_root": "/proc",
        "double_inspection_exact_equal": True,
        "spawned_process_identity": (
            dict(spawned_identity) if spawned_identity is not None else None
        ),
        **first,
    }


def _memory_gate_gpu_process_snapshot() -> dict[str, Any]:
    executable = contractlib.resolve_executable_identity(
        contractlib.NVIDIA_SMI_ENTRYPOINT,
        label="operator memory-gate GPU-process guard",
        expected_resolved_path=contractlib.FROZEN_NVIDIA_SMI_RESOLVED_PATH,
        expected_sha256=contractlib.FROZEN_NVIDIA_SMI_RESOLVED_SHA256,
        expected_size_bytes=contractlib.FROZEN_NVIDIA_SMI_RESOLVED_SIZE_BYTES,
        expected_mode=contractlib.FROZEN_NVIDIA_SMI_RESOLVED_MODE,
        accepted_owner_ids=contractlib.FROZEN_NVIDIA_SMI_ACCEPTED_OWNER_IDS,
    )
    if (
        contractlib.normalized_executable_identity(executable)
        != contractlib.FROZEN_NVIDIA_SMI_NORMALIZED_IDENTITY
    ):
        contractlib.fail("operator GPU-process probe executable identity drifted")
    command = [
        str(executable["resolved_path"]),
        "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={"PATH": contractlib.CLOSED_ENVIRONMENT_PATH, "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        contractlib.fail(f"cannot execute operator GPU-process guard: {exc}")
    if completed.returncode != 0:
        contractlib.fail(f"operator GPU-process guard failed: {completed.stderr.strip()}")
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    parsed = []
    for row in rows:
        fields = [value.strip() for value in row.split(",")]
        if len(fields) != 2 or not fields[0].isdigit():
            contractlib.fail(f"operator GPU-process guard returned malformed row: {row!r}")
        parsed.append({"pid": int(fields[0]), "gpu_uuid": fields[1]})
    expected_uuid = str(contractlib.EXPECTED_LOCAL_HARDWARE["uuid"])
    target = [record for record in parsed if record["gpu_uuid"] == expected_uuid]
    return {
        "normalized_probe_executable": contractlib.normalized_executable_identity(executable),
        "query": command[1:],
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "target_gpu_uuid": expected_uuid,
        "target_gpu_processes": target,
    }


def _audit_memory_gate_gpu_processes_absent() -> dict[str, Any]:
    first = _memory_gate_gpu_process_snapshot()
    second = _memory_gate_gpu_process_snapshot()
    if first["target_gpu_processes"] or second["target_gpu_processes"]:
        contractlib.fail("memory-gate GPU still has a live compute process")
    if first != second:
        contractlib.fail("operator GPU-process guard changed during its double inspection")
    return {
        "protocol": "clasp-memory-gate-operator-gpu-absence-v1",
        "double_inspection_exact_equal": True,
        **first,
    }


def _run_memory_gate_child(
    payload_argv: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: Path,
    transcript_descriptor: int,
) -> dict[str, Any]:
    process = None
    process_identity = None
    previous_handlers = {}
    try:
        _write_transcript_event(
            transcript_descriptor,
            {
                "schema_version": 1,
                "protocol": MEMORY_GATE_TRANSCRIPT_PROTOCOL,
                "sequence": 2,
                "event": "child_spawn_pending",
                "supervisor_pid": os.getpid(),
                "payload_argv_sha256": contractlib.sha256_json(list(payload_argv)),
                "isolation": {
                    "start_new_session": True,
                    "linux_parent_death_signal": "SIGTERM",
                    "parent_death_wrapper_kills_entire_process_group": True,
                },
            },
        )
        supervisor_pid = os.getpid()
        wrapper_argv = _memory_gate_exec_wrapper_argv(
            payload_argv,
            expected_supervisor_pid=supervisor_pid,
        )
        process = subprocess.Popen(
            wrapper_argv,
            shell=False,
            cwd=str(cwd),
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        process_identity = _read_linux_process_identity(process.pid)
        if process_identity is None:
            raise RuntimeError("memory-gate child vanished before process identity binding")
        process_identity = _validate_memory_gate_process_identity(process_identity)
        _write_transcript_event(
            transcript_descriptor,
            {
                "schema_version": 1,
                "protocol": MEMORY_GATE_TRANSCRIPT_PROTOCOL,
                "sequence": 3,
                "event": "child_spawned",
                "payload_argv_sha256": contractlib.sha256_json(list(payload_argv)),
                "process_role": "parent_death_process_group_wrapper",
                "process_identity": process_identity,
            },
        )

        def forward_signal(signum, _frame):
            if process is not None:
                _signal_memory_gate_process_group(process, signum)

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, forward_signal)
        if process.stdout is None:
            raise RuntimeError("memory-gate child has no combined output pipe")
        while True:
            chunk = process.stdout.read(65_536)
            if not chunk:
                break
            _write_descriptor_all(transcript_descriptor, chunk)
            try:
                _write_descriptor_all(sys.stdout.fileno(), chunk)
            except (AttributeError, OSError):
                pass
        returncode = process.wait()
        if returncode == 0:
            return {
                "child_started": True,
                "exit_known": True,
                "returncode": 0,
                "exit_code": 0,
                "signal": None,
                "classification": "exited_zero",
                "error_type": None,
                "error_message": None,
                "process_identity": process_identity,
            }
        if returncode > 0:
            return {
                "child_started": True,
                "exit_known": True,
                "returncode": returncode,
                "exit_code": returncode,
                "signal": None,
                "classification": "nonzero_exit",
                "error_type": None,
                "error_message": None,
                "process_identity": process_identity,
            }
        return {
            "child_started": True,
            "exit_known": True,
            "returncode": returncode,
            "exit_code": None,
            "signal": -returncode,
            "classification": "signal",
            "error_type": None,
            "error_message": None,
            "process_identity": process_identity,
        }
    except BaseException as exc:
        if process is not None and process.poll() is None:
            try:
                _signal_memory_gate_process_group(process, signal.SIGTERM)
            except (OSError, contractlib.ContractError):
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    _signal_memory_gate_process_group(process, signal.SIGKILL)
                except (OSError, contractlib.ContractError):
                    process.kill()
                process.wait(timeout=30)
        return {
            "child_started": process is not None,
            "exit_known": False,
            "returncode": None,
            "exit_code": None,
            "signal": None,
            "classification": "spawn_error" if process is None else "supervisor_error",
            "error_type": f"{exc.__class__.__module__}.{exc.__class__.__qualname__}",
            "error_message": str(exc),
            "process_identity": process_identity,
        }
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def supervise_memory_gate_attempt(
    contract_path: Path,
    *,
    trajectory_id: str,
    actual_payload_argv: Sequence[str],
    actual_environment: Mapping[str, str],
) -> int:
    if trajectory_id != "engineering-gate-D5-seed42":
        contractlib.fail("memory-gate supervisor received the wrong trajectory")
    claim, transcript_descriptor = claim_memory_gate_attempt(
        contract_path,
        actual_payload_argv=actual_payload_argv,
        actual_environment=actual_environment,
    )
    contract = contractlib.load_json_object(contract_path, label="frozen pilot contract")
    command = build_plan(contract, contract_path=contract_path)["memory_gate_command"]
    transcript_closed = False
    try:
        _write_transcript_event(
            transcript_descriptor,
            {
                "schema_version": 1,
                "protocol": MEMORY_GATE_TRANSCRIPT_PROTOCOL,
                "sequence": 1,
                "event": "supervisor_claimed",
                "supervisor_pid": os.getpid(),
                "attempt_receipt_sha256": claim["receipt_sha256"],
                "payload_argv_sha256": contractlib.sha256_json(list(actual_payload_argv)),
                "environment_sha256": contractlib.sha256_json(dict(actual_environment)),
            },
        )
        try:
            verify_attempt_execution_authority(
                contract_path,
                trajectory_id=trajectory_id,
                kind="memory_gate",
                claim=claim,
                actual_payload_argv=actual_payload_argv,
                actual_environment=actual_environment,
            )
            expected_cwd = Path(str(command.get("cwd", ""))).resolve(strict=True)
            if Path.cwd().resolve(strict=True) != expected_cwd:
                contractlib.fail("memory-gate supervisor cwd differs from frozen authority")
            child = _run_memory_gate_child(
                actual_payload_argv,
                environment=actual_environment,
                cwd=expected_cwd,
                transcript_descriptor=transcript_descriptor,
            )
        except BaseException as exc:
            child = {
                "child_started": False,
                "exit_known": False,
                "returncode": None,
                "exit_code": None,
                "signal": None,
                "classification": "supervisor_error",
                "error_type": f"{exc.__class__.__module__}.{exc.__class__.__qualname__}",
                "error_message": str(exc),
                "process_identity": None,
            }
        wrapper_argv = _memory_gate_exec_wrapper_argv(
            actual_payload_argv,
            expected_supervisor_pid=os.getpid(),
        )
        process_absence = _audit_memory_gate_processes_absent(
            spawned_identity=child.get("process_identity"),
            payload_argv=actual_payload_argv,
            wrapper_argv=wrapper_argv,
        )
        gpu_absence = _audit_memory_gate_gpu_processes_absent()
        _write_transcript_event(
            transcript_descriptor,
            {
                "schema_version": 1,
                "protocol": MEMORY_GATE_TRANSCRIPT_PROTOCOL,
                "sequence": 4,
                "event": "supervisor_observed_child_outcome",
                "child": child,
            },
        )
        os.fsync(transcript_descriptor)
        status = (
            MEMORY_GATE_SUCCESS_STATUS
            if child["classification"] == "exited_zero"
            else MEMORY_GATE_FAILURE_STATUS
            if child["classification"] in {"nonzero_exit", "signal"}
            else MEMORY_GATE_SUPERVISOR_FAILURE_STATUS
        )
        _persist_memory_gate_execution_outcome(
            contract,
            contract_path=contract_path,
            command=command,
            claim=claim,
            payload_argv=actual_payload_argv,
            environment=actual_environment,
            child=child,
            status=status,
            operator_finalized=False,
            post_execution_absence={
                "process_absence": process_absence,
                "gpu_absence": gpu_absence,
            },
        )
        if status == MEMORY_GATE_SUCCESS_STATUS:
            return 0
        returncode = child.get("returncode")
        if isinstance(returncode, int) and not isinstance(returncode, bool):
            return returncode if returncode > 0 else 128 + (-returncode)
        return 2
    finally:
        if not transcript_closed:
            os.close(transcript_descriptor)


def finalize_interrupted_memory_gate(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
) -> dict[str, Any]:
    output_root = Path(str(contract["paths"]["output_root"]))
    gate_dir = output_root / MEMORY_GATE_DIRNAME
    _require_top_level(
        output_root,
        {MEMORY_GATE_DIRNAME},
        phase="interrupted memory-gate finalization",
    )
    command = build_plan(contract, contract_path=contract_path)["memory_gate_command"]
    attempt_path = gate_dir / CLAIM_ANCHOR_DIRNAME / MEMORY_GATE_ATTEMPT_FILENAME
    claim = _inspect_attempt_claim(
        attempt_path,
        expected_protocol=MEMORY_GATE_ATTEMPT_PROTOCOL,
        expected_status="development_memory_gate_attempt_started",
        expected_kind="memory_gate",
        trajectory_id="engineering-gate-D5-seed42",
        ledger_identity=None,
        reservation_identity=None,
        command=command,
        resume_step=None,
    )
    transcript_path, _, existing = _inspect_memory_gate_execution_anchors(claim)
    if existing is not None:
        contractlib.fail("operator finalization requires the unused execution-outcome slot")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    transcript_descriptor = os.open(transcript_path, flags)
    try:
        try:
            fcntl.flock(transcript_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            contractlib.fail("memory-gate supervisor still holds the transcript lock")
        transcript_prefix = _inspect_supervisor_transcript_prefix(transcript_descriptor)
        spawned_identity = transcript_prefix["spawned_process_identity"]
        payload_argv = _authorized_inner_payload(command)
        pending_supervisor_pid = transcript_prefix["pending_supervisor_pid"]
        wrapper_argv = (
            _memory_gate_exec_wrapper_argv(
                payload_argv,
                expected_supervisor_pid=pending_supervisor_pid,
            )
            if pending_supervisor_pid is not None
            else None
        )
        process_absence = _audit_memory_gate_processes_absent(
            spawned_identity=spawned_identity,
            payload_argv=payload_argv,
            wrapper_argv=wrapper_argv,
        )
        gpu_absence = _audit_memory_gate_gpu_processes_absent()
        child = {
            "child_started": True if spawned_identity is not None else None,
            "exit_known": False,
            "returncode": None,
            "exit_code": None,
            "signal": None,
            "classification": "abrupt_supervisor_death_unknown",
            "error_type": "abrupt_supervisor_death",
            "error_message": None,
            "process_identity": spawned_identity,
        }
        return _persist_memory_gate_execution_outcome(
            contract,
            contract_path=contract_path,
            command=command,
            claim=claim,
            payload_argv=payload_argv,
            environment=_memory_gate_live_environment(command),
            child=child,
            status=MEMORY_GATE_UNKNOWN_STATUS,
            operator_finalized=True,
            post_execution_absence={
                "process_absence": process_absence,
                "gpu_absence": gpu_absence,
            },
            operator_finalization_evidence={
                "transcript_prefix": transcript_prefix,
            },
        )
    finally:
        os.close(transcript_descriptor)


def _checkpoint_steps(run_dir: Path) -> list[int]:
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.exists():
        return []
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        contractlib.fail(f"training checkpoint root is invalid: {checkpoint_root}")
    result: list[int] = []
    for path in checkpoint_root.iterdir():
        if not path.name.startswith("checkpoint-"):
            continue
        if path.is_symlink() or not path.is_dir():
            contractlib.fail(f"checkpoint-like entry is not a regular directory: {path}")
        suffix = path.name.removeprefix("checkpoint-")
        if not suffix.isdigit() or int(suffix) <= 0:
            contractlib.fail(f"malformed checkpoint directory: {path}")
        result.append(int(suffix))
    return sorted(result)


def _resume_checkpoint_identity(
    run_dir: Path,
    step: int,
    *,
    beta: float,
    ledger_path: Path,
) -> dict[str, Any]:
    from clasp.models.checkpoint_fidelity import verify_checkpoint_fidelity

    checkpoint = run_dir / "checkpoints" / f"checkpoint-{step}"
    fidelity = verify_checkpoint_fidelity(checkpoint)
    if (
        fidelity.get("protocol") != contractlib.CHECKPOINT_FIDELITY_PROTOCOL
        or fidelity.get("status") != "exact_trainable_policy_persisted"
        or fidelity.get("publication_invariant") is not None
    ):
        contractlib.fail("resume checkpoint fidelity boundary mismatch")
    trainer = contractlib.load_json_object(
        checkpoint / "trainer_state.json",
        label="resume checkpoint trainer state",
    )
    if trainer.get("global_step") != step:
        contractlib.fail("resume checkpoint trainer-state step mismatch")
    evidence: dict[str, Any] = {
        "path": str(checkpoint),
        "step": step,
        "checkpoint_fidelity_receipt_sha256": fidelity["receipt_sha256"],
        "checkpoint_fidelity_file_sha256": contractlib.sha256_file(
            checkpoint / "checkpoint_fidelity.json"
        ),
        "trainable_closure_sha256": contractlib.sha256_json(fidelity["trainable_closure"]),
        "trainer_state_file_sha256": contractlib.sha256_file(checkpoint / "trainer_state.json"),
    }
    if beta > 0.0:
        from clasp.models.exact_peft_kl_post_step import (
            POST_STEP_RECEIPT_FILENAME,
            inspect_post_step_receipt,
            training_run_contract_identity,
        )

        previous = os.environ.get(TRAINING_RUN_CONTRACT_ENV)
        os.environ[TRAINING_RUN_CONTRACT_ENV] = str(ledger_path)
        try:
            post = inspect_post_step_receipt(
                checkpoint / POST_STEP_RECEIPT_FILENAME,
                claim_scope="development",
                expected_global_step=step,
                expected_artifact_role="checkpoint",
                rehash_sources=True,
                rehash_policy_artifact=True,
                expected_policy_trainable_tensor_inventory_sha256=fidelity["trainable_closure"][
                    "tensor_inventory_sha256"
                ],
            )
            if post.get("training_run_contract") != training_run_contract_identity(
                ledger_path,
                claim_scope="development",
            ):
                contractlib.fail("resume checkpoint binds a different matrix ledger")
        finally:
            if previous is None:
                os.environ.pop(TRAINING_RUN_CONTRACT_ENV, None)
            else:
                os.environ[TRAINING_RUN_CONTRACT_ENV] = previous
        evidence["post_step_receipt_sha256"] = post["receipt_sha256"]
    return evidence


def claim_matrix_attempt(
    contract_path: Path,
    *,
    trajectory_id: str,
    kind: str,
    actual_payload_argv: Sequence[str],
    actual_environment: Mapping[str, str],
) -> dict[str, Any]:
    """Claim one preauthorized train/eval slot immediately before ``exec``."""

    if kind not in {"training", "evaluation"}:
        contractlib.fail(f"unsupported matrix attempt kind: {kind}")
    contract, _, ledger, closure = _load_matrix_lifecycle(contract_path)
    output_root = Path(str(contract["paths"]["output_root"]))
    ledger_path = output_root / MATRIX_AUTHORIZATION_FILENAME
    closure_path = output_root / RESERVATION_CLOSURE_FILENAME
    if os.environ.get(TRAINING_RUN_CONTRACT_ENV) != str(ledger_path):
        contractlib.fail("matrix attempt environment does not point at the immutable ledger")
    if inspect_reservation_closure(closure_path, ledger_path, ledger) != closure:
        contractlib.fail("matrix reservation closure drifted before attempt claim")
    group = "training_commands" if kind == "training" else "evaluation_commands"
    command = _ledger_command(ledger, group=group, trajectory_id=trajectory_id)
    _assert_exact_attempt_material(
        command,
        actual_payload_argv=actual_payload_argv,
        actual_environment=actual_environment,
    )
    reserved_path = (
        Path(str(command["output_dir"]))
        if kind == "training"
        else Path(_argument(command["argv"], "--inference_root"))
    )
    marker_name = (
        CELL_AUTHORIZATION_FILENAME if kind == "training" else EVALUATION_AUTHORIZATION_FILENAME
    )
    marker_path = reserved_path / marker_name
    marker = contractlib.load_json_object(marker_path, label="matrix reservation")
    expected_marker = _reservation_payload(
        kind=kind,
        trajectory_id=trajectory_id,
        reserved_path=reserved_path,
        command=command,
        ledger_path=ledger_path,
        ledger=ledger,
    )
    if marker != expected_marker:
        contractlib.fail("matrix attempt reservation differs from ledger")
    ledger_identity = _file_identity(ledger_path, ledger)
    reservation_identity = _file_identity(marker_path, marker)

    if kind == "evaluation":
        inspect_cell_terminal_audit(
            contract,
            contract_path=contract_path,
            ledger=ledger,
            trajectory_id=trajectory_id,
            reverify=True,
        )
        attempt_path, anchor_inode = _claim_anchor_record(
            closure,
            kind=kind,
            trajectory_id=trajectory_id,
            name=EVALUATION_ATTEMPT_FILENAME,
        )
        if attempt_path.stat().st_size != 0:
            contractlib.fail(
                "evaluation slot was already attempted; rerun/replacement is forbidden"
            )
        allowed = {marker_name, "_runtime_tmp", CLAIM_ANCHOR_DIRNAME}
        names = {item.name for item in reserved_path.iterdir()}
        if names != allowed:
            contractlib.fail(f"evaluation slot contains pre-attempt outcome bytes: {names}")
        payload = _attempt_payload(
            protocol=EVALUATION_ATTEMPT_PROTOCOL,
            status="development_evaluation_attempt_started",
            kind=kind,
            trajectory_id=trajectory_id,
            attempt_path=attempt_path,
            ledger_path=ledger_path,
            ledger=ledger,
            reservation_path=marker_path,
            reservation=marker,
            command=command,
        )
        return _consume_claim_anchor_once(
            attempt_path,
            payload,
            frozen_inode=anchor_inode,
        )

    run_dir = reserved_path
    terminal_audit = run_dir / TERMINAL_AUDIT_FILENAME
    if terminal_audit.exists() or terminal_audit.is_symlink():
        contractlib.fail("terminally audited training output cannot be started again")
    attempt_path, initial_anchor_inode = _claim_anchor_record(
        closure,
        kind=kind,
        trajectory_id=trajectory_id,
        name=TRAINING_ATTEMPT_FILENAME,
    )
    if attempt_path.stat().st_size == 0:
        allowed = {marker_name, "_runtime_tmp", CLAIM_ANCHOR_DIRNAME}
        names = {item.name for item in run_dir.iterdir()}
        if names != allowed:
            contractlib.fail(f"training slot contains pre-attempt outcome bytes: {names}")
        payload = _attempt_payload(
            protocol=TRAINING_ATTEMPT_PROTOCOL,
            status="development_training_attempt_started",
            kind=kind,
            trajectory_id=trajectory_id,
            attempt_path=attempt_path,
            ledger_path=ledger_path,
            ledger=ledger,
            reservation_path=marker_path,
            reservation=marker,
            command=command,
        )
        return _consume_claim_anchor_once(
            attempt_path,
            payload,
            frozen_inode=initial_anchor_inode,
        )

    _inspect_attempt_claim(
        attempt_path,
        expected_protocol=TRAINING_ATTEMPT_PROTOCOL,
        expected_status="development_training_attempt_started",
        expected_kind=kind,
        trajectory_id=trajectory_id,
        ledger_identity=ledger_identity,
        reservation_identity=reservation_identity,
        command=command,
        resume_step=None,
    )
    steps = _checkpoint_steps(run_dir)
    if not steps or steps[-1] >= 400:
        contractlib.fail(
            "a repeated training invocation requires one surviving nonterminal receipted "
            "checkpoint; fresh retry and terminal rerun are forbidden"
        )
    newest = steps[-1]
    consumed_resume_steps = [
        int(path.name.removeprefix(TRAINING_RESUME_PREFIX).removesuffix(".json"))
        for path in (run_dir / CLAIM_ANCHOR_DIRNAME).glob(f"{TRAINING_RESUME_PREFIX}*.json")
        if path.stat().st_size > 0
    ]
    if consumed_resume_steps and newest <= max(consumed_resume_steps):
        contractlib.fail("strict resume must advance beyond every previously claimed checkpoint")
    beta = float(_argument(command["argv"], "--grpo_beta"))
    checkpoint = _resume_checkpoint_identity(
        run_dir,
        newest,
        beta=beta,
        ledger_path=ledger_path,
    )
    resume_name = f"{TRAINING_RESUME_PREFIX}{newest}.json"
    resume_path, resume_anchor_inode = _claim_anchor_record(
        closure,
        kind=kind,
        trajectory_id=trajectory_id,
        name=resume_name,
    )
    if resume_path.stat().st_size != 0:
        contractlib.fail(
            "this exact surviving checkpoint already consumed its sole strict-resume claim"
        )
    payload = _attempt_payload(
        protocol=TRAINING_ATTEMPT_PROTOCOL,
        status="development_training_strict_resume_attempt_started",
        kind=kind,
        trajectory_id=trajectory_id,
        attempt_path=resume_path,
        ledger_path=ledger_path,
        ledger=ledger,
        reservation_path=marker_path,
        reservation=marker,
        command=command,
        resume_checkpoint=checkpoint,
    )
    return _consume_claim_anchor_once(
        resume_path,
        payload,
        frozen_inode=resume_anchor_inode,
    )


def verify_attempt_execution_authority(
    contract_path: Path,
    *,
    trajectory_id: str,
    kind: str,
    claim: Mapping[str, Any],
    actual_payload_argv: Sequence[str],
    actual_environment: Mapping[str, str],
) -> None:
    """Rehash authority and compare live material again immediately before exec."""

    contract_path = contract_path.expanduser().absolute()
    if kind == "memory_gate":
        contractlib.require_read_only(contract_path, label="frozen pilot contract")
        contract = contractlib.load_json_object(contract_path, label="frozen pilot contract")
        contractlib.verify_contract(
            contract,
            require_fresh_output=False,
            contract_path=contract_path,
        )
        command = build_plan(contract, contract_path=contract_path)["memory_gate_command"]
        consumed = ("CLASP_PILOT_CREATE_TMPDIR",)
    elif kind in {"training", "evaluation"}:
        _, _, ledger, _ = _load_matrix_lifecycle(contract_path)
        command = _ledger_command(
            ledger,
            group="training_commands" if kind == "training" else "evaluation_commands",
            trajectory_id=trajectory_id,
        )
        consumed = ()
    else:
        contractlib.fail(f"unsupported attempt-guard kind: {kind}")
    if claim.get("trajectory_id") != trajectory_id or claim.get("kind") != kind:
        contractlib.fail("live attempt claim identity differs before exec")
    if claim.get("authorized_command_sha256") != contractlib.sha256_json(command):
        contractlib.fail("live attempt claim binds a different authorized command")
    claim_path = Path(str(claim.get("attempt_path", "")))
    reopened = contractlib.load_json_object(claim_path, label="live attempt claim")
    contractlib.require_read_only(claim_path, label="live attempt claim")
    _verify_self_hash(reopened, label="live attempt claim")
    if reopened != dict(claim):
        contractlib.fail("attempt claim changed between atomic claim and exec")
    if kind == "memory_gate":
        _, _, outcome = _inspect_memory_gate_execution_anchors(
            claim,
            require_pristine_container=True,
        )
        if outcome is not None:
            contractlib.fail("memory-gate outcome existed before child spawn")
    resume = claim.get("resume_checkpoint")
    resume_path = str(resume.get("path")) if isinstance(resume, dict) else None
    _assert_exact_attempt_material(
        command,
        actual_payload_argv=actual_payload_argv,
        actual_environment=actual_environment,
        consumed_environment_names=consumed,
        resume_checkpoint_path=resume_path,
    )


def apply_claimed_resume_environment(
    claim: Mapping[str, Any],
    environment: dict[str, str],
) -> str | None:
    """Apply only the exact closure-claimed checkpoint to the training environment."""

    resume = claim.get("resume_checkpoint")
    if resume is None:
        environment.pop("RESUME_FROM_CHECKPOINT", None)
        return None
    if not isinstance(resume, dict):
        contractlib.fail("strict-resume claim has a malformed checkpoint identity")
    step = resume.get("step")
    path_value = resume.get("path")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 1 <= step < 400
        or not isinstance(path_value, str)
    ):
        contractlib.fail("strict-resume claim has an invalid checkpoint step/path")
    path = Path(path_value)
    if (
        not path.is_absolute()
        or path.name != f"checkpoint-{step}"
        or path.is_symlink()
        or not path.is_dir()
    ):
        contractlib.fail("strict-resume claim does not point at its exact live checkpoint")
    environment["RESUME_FROM_CHECKPOINT"] = str(path)
    return str(path)


def _validate_training_attempt_lineage(
    run_dir: Path,
    *,
    trajectory_id: str,
    ledger_path: Path,
    ledger: Mapping[str, Any],
) -> list[dict[str, Any]]:
    command = _ledger_command(
        ledger,
        group="training_commands",
        trajectory_id=trajectory_id,
    )
    marker_path = run_dir / CELL_AUTHORIZATION_FILENAME
    marker = contractlib.load_json_object(marker_path, label="matrix reservation")
    ledger_identity = _file_identity(ledger_path, ledger)
    reservation_identity = _file_identity(marker_path, marker)
    anchor_root = run_dir / CLAIM_ANCHOR_DIRNAME
    attempt_path = anchor_root / TRAINING_ATTEMPT_FILENAME
    attempt = _inspect_attempt_claim(
        attempt_path,
        expected_protocol=TRAINING_ATTEMPT_PROTOCOL,
        expected_status="development_training_attempt_started",
        expected_kind="training",
        trajectory_id=trajectory_id,
        ledger_identity=ledger_identity,
        reservation_identity=reservation_identity,
        command=command,
        resume_step=None,
    )
    identities = [_file_identity(attempt_path, attempt)]
    resume_paths = sorted(
        (
            path
            for path in anchor_root.glob(f"{TRAINING_RESUME_PREFIX}*.json")
            if path.stat().st_size > 0
        ),
        key=lambda path: int(path.name.removeprefix(TRAINING_RESUME_PREFIX).removesuffix(".json")),
    )
    observed_steps: list[int] = []
    observed_times = [
        _validate_utc_timestamp(attempt.get("created_at_utc"), label="training attempt")
    ]
    for path in resume_paths:
        suffix = path.name.removeprefix(TRAINING_RESUME_PREFIX).removesuffix(".json")
        if not suffix.isdigit():
            contractlib.fail(f"malformed strict-resume claim filename: {path}")
        step = int(suffix)
        if step <= 0 or step >= 400 or step in observed_steps:
            contractlib.fail(f"invalid strict-resume checkpoint step: {path}")
        observed_steps.append(step)
        receipt = _inspect_attempt_claim(
            path,
            expected_protocol=TRAINING_ATTEMPT_PROTOCOL,
            expected_status="development_training_strict_resume_attempt_started",
            expected_kind="training",
            trajectory_id=trajectory_id,
            ledger_identity=ledger_identity,
            reservation_identity=reservation_identity,
            command=command,
            resume_step=step,
        )
        observed_times.append(
            _validate_utc_timestamp(
                receipt.get("created_at_utc"),
                label="strict-resume attempt",
            )
        )
        identities.append(_file_identity(path, receipt))
    if (
        observed_steps != sorted(observed_steps)
        or len(set(observed_steps)) != len(observed_steps)
        or any(later <= earlier for earlier, later in zip(observed_times, observed_times[1:]))
    ):
        contractlib.fail("strict-resume attempt chronology is invalid")
    return identities


def _audit_terminal_telemetry(
    trainer_state_path: Path,
    *,
    required: Sequence[str],
) -> dict[str, Any]:
    state = contractlib.load_json_object(trainer_state_path, label="terminal trainer state")
    if state.get("global_step") != 400:
        contractlib.fail("terminal trainer state global_step must be exactly 400")
    history = state.get("log_history")
    if not isinstance(history, list):
        contractlib.fail("terminal trainer state lacks log_history")
    records: dict[int, Mapping[str, Any]] = {}
    for value in history:
        if not isinstance(value, dict) or "loss" not in value:
            continue
        step = value.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or not 1 <= step <= 400:
            continue
        if step in records:
            contractlib.fail(f"duplicate terminal training telemetry at step {step}")
        records[step] = value
    if sorted(records) != list(range(1, 401)):
        missing = sorted(set(range(1, 401)) - set(records))
        contractlib.fail(f"terminal telemetry does not cover every optimizer step: {missing[:8]}")
    missing_by_step = {
        step: sorted(set(required) - set(record))
        for step, record in records.items()
        if set(required) - set(record)
    }
    if missing_by_step:
        first = min(missing_by_step)
        contractlib.fail(
            f"terminal telemetry is incomplete at step {first}: {missing_by_step[first]}"
        )
    for step, record in records.items():
        _validate_required_numeric_metrics(
            record,
            required=required,
            label=f"terminal step {step}",
        )
        nonfinite = {
            key: value
            for key, value in record.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not math.isfinite(float(value))
        }
        if nonfinite:
            contractlib.fail(f"terminal telemetry is non-finite at step {step}: {nonfinite}")
    return {
        "path": str(trainer_state_path),
        "file_sha256": contractlib.sha256_file(trainer_state_path),
        "global_step": 400,
        "complete_optimizer_steps": 400,
        "required_keys": sorted(required),
        "step_record_digests_sha256": contractlib.sha256_json(
            [
                {"step": step, "record_sha256": contractlib.sha256_json(records[step])}
                for step in range(1, 401)
            ]
        ),
        "all_numeric_values_finite": True,
    }


def _validate_terminal_hyperparams(
    path: Path,
    *,
    command: Mapping[str, Any],
    ledger_path: Path,
    method: str,
    beta: float,
    seed: int,
) -> dict[str, Any]:
    from clasp.models.exact_peft_kl_post_step import training_run_contract_identity

    payload = contractlib.load_json_object(path, label="terminal hyperparameters")
    stage = payload.get("stage_hyperparameters")
    args = payload.get("args")
    if not isinstance(stage, dict) or not isinstance(args, dict):
        contractlib.fail("terminal hyperparameters lack args/stage_hyperparameters")
    argv = command["argv"]
    expected_top = {
        "stage": "grpo",
        "model_path": _argument(argv, "--model_path"),
        "output_dir": str(command["output_dir"]),
        "tasks": ["tsp"],
    }
    if any(payload.get(key) != value for key, value in expected_top.items()):
        contractlib.fail("terminal hyperparameter top-level provenance differs from ledger")
    expected_stage = {
        "dtype": "bfloat16",
        "lora_r": 64,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "lora_bias": "all",
        "seed": seed,
        "data_seed": seed,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 32,
        "effective_train_batch_size": 32,
        "learning_rate": 1e-6,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "linear",
        "max_steps": 400,
        "num_generations": 8,
        "generation_batch_size": 8,
        "grpo_beta": beta,
        "grpo_loss_type": "dr_grpo",
        "training_run_contract_path": str(ledger_path),
        "checkpoint_fidelity_mode": "development",
        "initialization_fidelity_protocol": contractlib.RELOCATION_PROTOCOL,
        "initialization_fidelity_status": contractlib.RELOCATION_STATUS,
        "initialization_fidelity_receipt_sha256": (contractlib.FROZEN_RELOCATION_RECEIPT_SHA256),
        "clasp_frontier_cdf": "1" if method.startswith("frontier_cdf") else "0",
        "clasp_frontier_cdf_relative_failed_recovery": (
            "1" if method.startswith("frontier_cdf") else "0"
        ),
        "clasp_frontier_cdf_gap_targets": "tsp:0.0107",
        "clasp_bon_success_credit": "dense_rank",
    }
    expected_stage["kl_reference_mode"] = (
        "explicit_frozen_exact_peft_step_zero" if beta > 0.0 else "none_beta_zero"
    )
    expected_stage["kl_reference_model_path"] = (
        _argument(argv, "--model_path") if beta > 0.0 else None
    )
    expected_stage["kl_reference_receipt_path"] = (
        command["env"].get("CLASP_KL_REFERENCE_RECEIPT") if beta > 0.0 else None
    )
    mismatches = {
        key: {"expected": value, "observed": stage.get(key)}
        for key, value in expected_stage.items()
        if stage.get(key) != value
    }
    if mismatches:
        contractlib.fail(f"terminal stage-hyperparameter drift: {mismatches}")
    run_identity = training_run_contract_identity(
        ledger_path,
        claim_scope="development",
    )
    if stage.get("training_run_contract_identity") != run_identity:
        contractlib.fail("beta-zero/beta-positive run provenance does not bind the ledger")
    expected_args = {
        "model_path": _argument(argv, "--model_path"),
        "tokenizer_path": _argument(argv, "--tokenizer_path"),
        "data_root": _argument(argv, "--data_root"),
        "output_root": _argument(argv, "--output_root"),
        "output_dir": _argument(argv, "--output_dir"),
        "seed": seed,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "learning_rate": 1e-6,
        "max_steps": 400,
        "num_generations": 8,
        "generation_batch_size": 8,
        "grpo_beta": beta,
    }
    arg_mismatches = {
        key: {"expected": value, "observed": args.get(key)}
        for key, value in expected_args.items()
        if args.get(key) != value
    }
    if arg_mismatches:
        contractlib.fail(f"terminal parsed-argument drift: {arg_mismatches}")
    return {
        "path": str(path),
        "file_sha256": contractlib.sha256_file(path),
        "payload_sha256": contractlib.sha256_json(payload),
        "training_run_contract": run_identity,
        "critical_stage_hyperparameters": expected_stage,
        "critical_parsed_arguments": expected_args,
    }


def audit_terminal_cell(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    ledger: Mapping[str, Any],
    trajectory_id: str,
) -> dict[str, Any]:
    """Build exact terminal evidence for one preauthorized scientific trajectory."""

    from clasp.models.checkpoint_fidelity import verify_checkpoint_fidelity
    from clasp.models.exact_peft_kl_post_step import (
        POST_STEP_RECEIPT_FILENAME,
        STEP1_RECEIPT_FILENAME,
        inspect_post_step_receipt,
        post_step_reference_identity,
        training_run_contract_identity,
    )
    from clasp.models.exact_peft_kl_reference import RECEIPT_FILENAME as PRE_FILENAME

    output_root = Path(str(contract["paths"]["output_root"]))
    ledger_path = output_root / MATRIX_AUTHORIZATION_FILENAME
    command = _ledger_command(
        ledger,
        group="training_commands",
        trajectory_id=trajectory_id,
    )
    run_dir = Path(str(command["output_dir"]))
    checkpoint = run_dir / "checkpoints" / "checkpoint-400"
    if _checkpoint_steps(run_dir) != [400]:
        contractlib.fail("terminal cell must retain exactly checkpoint-400")
    attempts = _validate_training_attempt_lineage(
        run_dir,
        trajectory_id=trajectory_id,
        ledger_path=ledger_path,
        ledger=ledger,
    )
    root_fidelity = verify_checkpoint_fidelity(run_dir)
    checkpoint_fidelity = verify_checkpoint_fidelity(checkpoint)
    for receipt in (root_fidelity, checkpoint_fidelity):
        if (
            receipt.get("protocol") != contractlib.CHECKPOINT_FIDELITY_PROTOCOL
            or receipt.get("status") != "exact_trainable_policy_persisted"
            or receipt.get("publication_invariant") is not None
        ):
            contractlib.fail("terminal checkpoint-fidelity claim boundary mismatch")
    if root_fidelity.get("trainable_closure") != checkpoint_fidelity.get("trainable_closure"):
        contractlib.fail("root and checkpoint-400 trainable closures are not exact-equal")
    closure_digest = root_fidelity["trainable_closure"]["tensor_inventory_sha256"]

    cell = next(
        (value for value in contract["design"]["cells"] if value["trajectory_id"] == trajectory_id),
        None,
    )
    if not isinstance(cell, dict):
        contractlib.fail("trajectory is absent from frozen design")
    method = str(cell["method"])
    beta = float(cell["grpo_beta"])
    seed = int(cell["seed"])
    hyperparams = _validate_terminal_hyperparams(
        run_dir / "hyperparams.json",
        command=command,
        ledger_path=ledger_path,
        method=method,
        beta=beta,
        seed=seed,
    )
    required = list(contract["required_telemetry"]["common"])
    if beta > 0.0:
        required.extend(contract["required_telemetry"]["beta_positive_cells"])
    if method.startswith("frontier_cdf"):
        required.extend(contract["required_telemetry"]["frontier_cells"])
    telemetry = _audit_terminal_telemetry(
        checkpoint / "trainer_state.json",
        required=required,
    )

    previous = os.environ.get(TRAINING_RUN_CONTRACT_ENV)
    os.environ[TRAINING_RUN_CONTRACT_ENV] = str(ledger_path)
    try:
        run_identity = training_run_contract_identity(
            ledger_path,
            claim_scope="development",
        )
        exact_reference: dict[str, Any] | None
        sidecar_names = {PRE_FILENAME, STEP1_RECEIPT_FILENAME, POST_STEP_RECEIPT_FILENAME}
        if beta == 0.0:
            forbidden = [
                str(root / name)
                for root in (run_dir, checkpoint)
                for name in sidecar_names
                if (root / name).exists() or (root / name).is_symlink()
            ]
            if forbidden:
                contractlib.fail(f"beta-zero trajectory emitted exact-KL sidecars: {forbidden}")
            exact_reference = None
        else:
            root_post = inspect_post_step_receipt(
                run_dir / POST_STEP_RECEIPT_FILENAME,
                claim_scope="development",
                expected_global_step=400,
                expected_artifact_role="final",
                rehash_sources=True,
                rehash_policy_artifact=True,
                expected_policy_trainable_tensor_inventory_sha256=closure_digest,
            )
            checkpoint_post = inspect_post_step_receipt(
                checkpoint / POST_STEP_RECEIPT_FILENAME,
                claim_scope="development",
                expected_global_step=400,
                expected_artifact_role="checkpoint",
                rehash_sources=True,
                rehash_policy_artifact=True,
                expected_policy_trainable_tensor_inventory_sha256=closure_digest,
            )
            if (
                root_post.get("training_run_contract") != run_identity
                or checkpoint_post.get("training_run_contract") != run_identity
                or root_post["reference_audit"]["live_inventory"]
                != checkpoint_post["reference_audit"]["live_inventory"]
                or root_post["reference_audit"]["reference_trainable_tensor_count"] != 0
                or checkpoint_post["reference_audit"]["reference_trainable_tensor_count"] != 0
            ):
                contractlib.fail("terminal exact-KL root/checkpoint invariants diverge")
            exact_reference = {
                "root": post_step_reference_identity(
                    run_dir / POST_STEP_RECEIPT_FILENAME,
                    claim_scope="development",
                    expected_global_step=400,
                    expected_artifact_role="final",
                    rehash_sources=True,
                    rehash_policy_artifact=True,
                    expected_policy_trainable_tensor_inventory_sha256=closure_digest,
                ),
                "checkpoint_400": post_step_reference_identity(
                    checkpoint / POST_STEP_RECEIPT_FILENAME,
                    claim_scope="development",
                    expected_global_step=400,
                    expected_artifact_role="checkpoint",
                    rehash_sources=True,
                    rehash_policy_artifact=True,
                    expected_policy_trainable_tensor_inventory_sha256=closure_digest,
                ),
            }
    finally:
        if previous is None:
            os.environ.pop(TRAINING_RUN_CONTRACT_ENV, None)
        else:
            os.environ[TRAINING_RUN_CONTRACT_ENV] = previous

    eval_command = _ledger_command(
        ledger,
        group="evaluation_commands",
        trajectory_id=trajectory_id,
    )
    eval_root = Path(_argument(eval_command["argv"], "--inference_root"))
    evaluation_attempt = _claim_anchor_path(eval_root, EVALUATION_ATTEMPT_FILENAME)
    if evaluation_attempt.stat().st_size != 0:
        contractlib.fail("terminal audit must precede the sole evaluation attempt")
    return _self_hashed(
        {
            "schema_version": 1,
            "protocol": TERMINAL_AUDIT_PROTOCOL,
            "status": TERMINAL_AUDIT_STATUS,
            "stage": "grpo",
            "development_only": True,
            "publication_eligible": False,
            "sota_eligible": False,
            "trajectory_id": trajectory_id,
            "cell": str(cell["cell"]),
            "method": method,
            "grpo_beta": beta,
            "seed": seed,
            "created_at_utc": _utc_now(),
            "ledger": _file_identity(ledger_path, ledger),
            "authorized_training_command_sha256": contractlib.sha256_json(command),
            "attempt_lineage": attempts,
            "hyperparameters": hyperparams,
            "training_run_contract": run_identity,
            "root_checkpoint_fidelity": {
                "receipt_sha256": root_fidelity["receipt_sha256"],
                "file_sha256": contractlib.sha256_file(run_dir / "checkpoint_fidelity.json"),
            },
            "checkpoint_400_fidelity": {
                "receipt_sha256": checkpoint_fidelity["receipt_sha256"],
                "file_sha256": contractlib.sha256_file(checkpoint / "checkpoint_fidelity.json"),
            },
            "root_checkpoint_trainable_closure_exact_equal": True,
            "trainable_closure_sha256": contractlib.sha256_json(root_fidelity["trainable_closure"]),
            "telemetry": telemetry,
            "exact_peft_kl_reference": exact_reference,
            "evaluation_attempt_absent_at_audit": True,
            "claim_limit": "development-only terminal integrity; never publication or SOTA",
        }
    )


def inspect_cell_terminal_audit(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    ledger: Mapping[str, Any],
    trajectory_id: str,
    reverify: bool,
) -> dict[str, Any]:
    command = _ledger_command(
        ledger,
        group="training_commands",
        trajectory_id=trajectory_id,
    )
    path = Path(str(command["output_dir"])) / TERMINAL_AUDIT_FILENAME
    contractlib.require_read_only(path, label="terminal cell audit")
    receipt = contractlib.load_json_object(path, label="terminal cell audit")
    _verify_self_hash(receipt, label="terminal cell audit")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol") != TERMINAL_AUDIT_PROTOCOL
        or receipt.get("status") != TERMINAL_AUDIT_STATUS
        or receipt.get("trajectory_id") != trajectory_id
        or receipt.get("development_only") is not True
        or receipt.get("publication_eligible") is not False
        or receipt.get("sota_eligible") is not False
    ):
        contractlib.fail("terminal cell audit identity/claim mismatch")
    if reverify:
        # Evaluation attempt creation changes only its separately reserved
        # directory, so rebuilding terminal policy evidence stays exact.
        expected = audit_terminal_cell(
            contract,
            contract_path=contract_path,
            ledger=ledger,
            trajectory_id=trajectory_id,
        )
        expected["created_at_utc"] = receipt.get("created_at_utc")
        unsigned = dict(expected)
        unsigned.pop("receipt_sha256", None)
        expected["receipt_sha256"] = contractlib.sha256_json(unsigned)
        if receipt != expected:
            contractlib.fail("terminal policy evidence drifted after audit")
    return receipt


def record_terminal_cell_audit(
    contract_path: Path,
    *,
    trajectory_id: str,
) -> dict[str, Any]:
    contract, _, ledger, _ = _load_matrix_lifecycle(contract_path)
    command = _ledger_command(
        ledger,
        group="training_commands",
        trajectory_id=trajectory_id,
    )
    path = Path(str(command["output_dir"])) / TERMINAL_AUDIT_FILENAME
    if path.exists() or path.is_symlink():
        return inspect_cell_terminal_audit(
            contract,
            contract_path=contract_path,
            ledger=ledger,
            trajectory_id=trajectory_id,
            reverify=True,
        )
    receipt = audit_terminal_cell(
        contract,
        contract_path=contract_path,
        ledger=ledger,
        trajectory_id=trajectory_id,
    )
    contractlib.write_contract_create_once(path, receipt)
    return receipt


def _cell_code(method: str, beta: float) -> str:
    prefix = "F" if method == "frontier_cdf_relative_failed_recovery" else "D"
    return prefix + ("0" if beta == 0.0 else "5")


def _run_directory(output_root: Path, method: str, beta: float, seed: int) -> Path:
    return output_root / f"{_cell_code(method, beta)}-seed{seed}"


def build_environment(
    *,
    method: str,
    beta: float,
    seed: int,
    relocation: Mapping[str, Any],
    training_run_contract_path: Path,
    project_root: Path,
    tmpdir: Path,
    create_tmpdir: bool,
) -> tuple[dict[str, str], list[str]]:
    environment = dict(BASE_ENV)
    environment.update(FRONTIER_ENV if method.startswith("frontier_cdf") else DENSE_ENV)
    environment["GRPO_DATA_SEED"] = str(seed)
    environment["PYTHONHASHSEED"] = str(seed)
    environment["PYTHONPATH"] = str(project_root / "src")
    environment[TRAINING_RUN_CONTRACT_ENV] = str(training_run_contract_path)
    environment["TMPDIR"] = str(tmpdir)
    environment["TMP"] = str(tmpdir)
    environment["TEMP"] = str(tmpdir)
    environment["XDG_CACHE_HOME"] = str(tmpdir / "xdg")
    environment["HF_HOME"] = str(tmpdir / "huggingface")
    environment["HF_DATASETS_CACHE"] = str(tmpdir / "huggingface" / "datasets")
    environment["TORCH_HOME"] = str(tmpdir / "torch")
    environment["HOME"] = str(tmpdir / "home")
    environment["TRITON_CACHE_DIR"] = str(tmpdir / "triton")
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(tmpdir / "torchinductor")
    environment["CUDA_CACHE_PATH"] = str(tmpdir / "cuda")
    if create_tmpdir:
        environment["CLASP_PILOT_CREATE_TMPDIR"] = "1"
    unset = list(ALWAYS_UNSET)
    if beta > 0.0:
        environment["CLASP_KL_REFERENCE_MODEL_PATH"] = str(relocation["adapter_path"])
        environment["CLASP_KL_REFERENCE_RECEIPT"] = str(relocation["receipt_path"])
    else:
        unset.extend(KL_ENV_NAMES)
    return environment, unset


def build_train_argv(
    *,
    python_executable: Path,
    initialization: Path,
    tokenizer_path: Path,
    train_data_root: Path,
    output_root: Path,
    run_dir: Path,
    beta: float,
    seed: int,
    max_steps: int = 400,
    warmup_ratio: float = 0.03,
) -> list[str]:
    return [
        str(python_executable),
        "-m",
        "clasp.entrypoints.train_grpo",
        "--tasks",
        "tsp",
        "--model_path",
        str(initialization),
        "--tokenizer_path",
        str(tokenizer_path),
        "--data_root",
        str(train_data_root),
        "--output_root",
        str(output_root),
        "--output_dir",
        str(run_dir),
        "--prompt_style",
        "alpaca",
        "--no_reason",
        "--dtype",
        contractlib.TRAINING_LOAD_DTYPE,
        "--lora_r",
        "64",
        "--lora_alpha",
        "64",
        "--lora_dropout",
        "0",
        "--seed",
        str(seed),
        "--eval_ratio",
        "0.05",
        "--eval-from-train-split",
        "--max_samples_per_task",
        "0",
        "--max_eval_samples_per_task",
        "100",
        "--min_train_samples_per_task",
        "1900",
        "--min_eval_samples_per_task",
        "100",
        "--no-prefer_subset_files",
        "--per_device_train_batch_size",
        "1",
        "--per_device_eval_batch_size",
        "8",
        "--gradient_accumulation_steps",
        "32",
        "--learning_rate",
        "1e-6",
        "--optim",
        "adamw_8bit",
        "--weight_decay",
        "0.01",
        "--warmup_ratio",
        format(warmup_ratio, "g"),
        "--lr_scheduler_type",
        "linear",
        "--num_train_epochs",
        "1",
        "--max_steps",
        str(max_steps),
        "--max_prompt_length",
        "29768",
        "--max_completion_length",
        "3000",
        "--num_generations",
        "8",
        "--generation_batch_size",
        "8",
        "--generation_temperature",
        "0.7",
        "--generation_top_p",
        "0.9",
        "--generation_top_k",
        "50",
        "--generation_repetition_penalty",
        "1",
        "--grpo_beta",
        format(beta, "g"),
        "--grpo_epsilon",
        "0.2",
        "--grpo_epsilon_high",
        "0.2",
        "--grpo_loss_type",
        "dr_grpo",
        "--verify_gt_rewards",
        "--reward_check_samples_per_task",
        "5",
        "--logging_steps",
        "1",
        "--save_steps",
        str(max_steps),
        "--eval_steps",
        str(max_steps),
        "--save_total_limit",
        "1",
        "--report_to",
        "none",
        "--use_primal_dual",
        "--pd_constraint_target",
        "1",
        "--pd_cost_limit",
        "0",
        "--pd_lambda_init",
        "0",
        "--pd_dual_lr",
        "0.05",
        "--pd_dual_lr_up",
        "0.05",
        "--pd_dual_lr_down",
        "0.05",
        "--pd_dual_update_interval",
        "1",
        "--pd_violation_ema_beta",
        "0.1",
        "--pd_lambda_max",
        "100",
        "--pd_violation_tolerance",
        "0",
        "--pd_objective_weight",
        "1",
        "--pd_structure_weight",
        "0.2",
        "--pd_sync_across_ranks",
        "--pd_require_state_on_resume",
        "--pd_strict_constraint_range",
        "--pd_force_scale_rewards_none",
        "--no-pd_soft_objective_for_infeasible",
        "--no-pd_normalize_primal_advantage",
        "--pd_primal_advantage_scale",
        "1",
        "--pd_penalty_scale",
        "1",
        "--pd_positive_constraint_penalty",
        "--pd_group_scalarization",
        "--pd_component_profile",
        "co_feasibility_lock_all",
        "--no-pd_enable_schedule",
        "--use_clasp",
        "--clasp_strict_defaults",
        "--no-clasp_raw_feasibility_targeting",
        "--clasp_bon_success_credit",
        "dense_rank",
        "--clasp_component_dual_recovery_scale",
        "1",
        "--clasp_feasibility_warmup_steps",
        "0",
        "--clasp_feasibility_warmup_targets",
        "tsp:0.995",
        "--no-clasp_self_replay",
        "--no-clasp_validation_gate_checkpoints",
        "--no-clasp_pareto_anchor",
        "--clasp_proper_dual",
        "--clasp_dual_update_interval",
        "4",
        "--clasp_proper_dual_gradient_mode",
        "signed",
        "--no-clasp_adaptive_length",
        "--clasp_mixed_adaptive_length_mode",
        "error",
    ]


def build_runtime_guarded_argv(
    *,
    python_executable: Path,
    workspace_root: Path,
    payload_argv: Sequence[str],
) -> list[str]:
    """Verify the frozen local runtime/GPU in-process, then exec the payload."""

    if not payload_argv or str(payload_argv[0]) != str(python_executable):
        contractlib.fail("runtime guard payload must use the audited Python executable")
    expected_runtime = json.dumps(contractlib.EXPECTED_RUNTIME, sort_keys=True)
    expected_hardware = json.dumps(contractlib.EXPECTED_LOCAL_HARDWARE, sort_keys=True)
    expected_smi_owners = json.dumps(
        [list(value) for value in sorted(contractlib.FROZEN_NVIDIA_SMI_ACCEPTED_OWNER_IDS)]
    )
    expected_smi_identity = json.dumps(
        contractlib.FROZEN_NVIDIA_SMI_NORMALIZED_IDENTITY,
        sort_keys=True,
    )
    code = (
        "import hashlib,json,os,platform,stat,subprocess,sys,tempfile;"
        "from pathlib import Path;"
        "tmp=Path(os.environ['TMPDIR']);"
        "create=os.environ.pop('CLASP_PILOT_CREATE_TMPDIR','0')=='1';"
        "tmp.mkdir(mode=0o700,parents=True,exist_ok=False) if create else None;"
        "assert tmp.is_dir() and not tmp.is_symlink();"
        "tmp=tmp.resolve(strict=True);workspace=Path(sys.argv[1]).resolve(strict=True);"
        "tmp.relative_to(workspace);"
        "assert tempfile.gettempdir()==str(tmp),(tempfile.gettempdir(),tmp);"
        f"assert os.statvfs(tmp).f_bavail*os.statvfs(tmp).f_frsize>={MIN_WORKSPACE_FREE_BYTES};"
        "assert os.environ.get('TMP')==str(tmp) and os.environ.get('TEMP')==str(tmp);"
        "cache_vars=('HOME','XDG_CACHE_HOME','HF_HOME','HF_DATASETS_CACHE','TORCH_HOME',"
        "'TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH');"
        "[Path(os.environ[k]).resolve(strict=False).relative_to(tmp) for k in cache_vars];"
        "import torch,transformers,trl,peft,datasets,bitsandbytes;"
        "runtime={'python':platform.python_version(),'torch':torch.__version__,"
        "'transformers':transformers.__version__,'trl':trl.__version__,"
        "'peft':peft.__version__,'datasets':datasets.__version__,"
        "'bitsandbytes':bitsandbytes.__version__,'cuda':torch.version.cuda or 'none'};"
        f"assert runtime==json.loads({expected_runtime!r}),runtime;"
        f"smi_entry=Path({str(contractlib.NVIDIA_SMI_ENTRYPOINT)!r});"
        "assert smi_entry.is_symlink(),smi_entry;"
        "smi=smi_entry.resolve(strict=True);smi_before=smi.lstat();"
        f"assert str(smi)=={str(contractlib.FROZEN_NVIDIA_SMI_RESOLVED_PATH)!r},smi;"
        "assert stat.S_ISREG(smi_before.st_mode) and os.access(smi,os.X_OK);"
        "assert not os.access(smi,os.W_OK);"
        f"assert stat.S_IMODE(smi_before.st_mode)=={contractlib.FROZEN_NVIDIA_SMI_RESOLVED_MODE};"
        f"assert smi_before.st_size=={contractlib.FROZEN_NVIDIA_SMI_RESOLVED_SIZE_BYTES};"
        f"assert [smi_before.st_uid,smi_before.st_gid] in json.loads({expected_smi_owners!r});"
        "smi_sha=hashlib.sha256(smi.read_bytes()).hexdigest();"
        f"assert smi_sha=={contractlib.FROZEN_NVIDIA_SMI_RESOLVED_SHA256!r},smi_sha;"
        "smi_identity={'entrypoint_path':str(smi_entry),'resolved_path':str(smi),"
        "'resolved_sha256':smi_sha,'resolved_size_bytes':smi_before.st_size,"
        "'resolved_mode':stat.S_IMODE(smi_before.st_mode),"
        f"'resolved_owner_identity':{contractlib.TRUSTED_SYSTEM_OWNER_IDENTITY!r}}};"
        f"assert smi_identity==json.loads({expected_smi_identity!r}),smi_identity;"
        "raw=subprocess.run([str(smi),"
        "'--query-gpu=index,uuid,name,memory.total,driver_version',"
        "'--format=csv,noheader,nounits','-i','3'],check=True,capture_output=True,text=True).stdout;"
        "smi_after=smi.lstat();"
        "smi_keys=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns');"
        "assert all(getattr(smi_before,k)==getattr(smi_after,k) for k in smi_keys);"
        "assert smi_entry.resolve(strict=True)==smi;"
        "assert hashlib.sha256(smi.read_bytes()).hexdigest()==smi_sha;"
        "rows=[x.strip() for x in raw.splitlines() if x.strip()];assert len(rows)==1,rows;"
        "f=[x.strip() for x in rows[0].split(',')];assert len(f)==5,f;"
        "p=torch.cuda.get_device_properties(0);"
        "hardware={'physical_index':int(f[0]),'uuid':f[1],'name':f[2],"
        "'memory_total_mib':int(f[3]),'torch_total_memory_bytes':int(p.total_memory),"
        "'driver_version':f[4],'visible_device_count':torch.cuda.device_count(),"
        "'visible_runtime_index':0};"
        f"assert hardware==json.loads({expected_hardware!r}),hardware;"
        "assert p.name==hardware['name'];"
        "os.execv(sys.argv[2],sys.argv[2:])"
    )
    return [
        str(python_executable),
        "-c",
        code,
        str(workspace_root),
        str(python_executable),
        *[str(value) for value in payload_argv[1:]],
    ]


def build_attempt_guarded_argv(
    *,
    python_executable: Path,
    project_root: Path,
    contract_path: Path,
    kind: str,
    trajectory_id: str,
    payload_argv: Sequence[str],
) -> list[str]:
    """Atomically consume the preregistered attempt immediately before payload exec."""

    if kind not in {"memory_gate", "training", "evaluation"}:
        contractlib.fail(f"unsupported attempt-guard kind: {kind}")
    if not payload_argv or str(payload_argv[0]) != str(python_executable):
        contractlib.fail("attempt guard payload must use the audited Python executable")
    code = (
        "import os,sys\n"
        "sys.path.insert(0,sys.argv[1])\n"
        "import dry_run_tsp_fcdf_beta_pilot as d\n"
        "from pathlib import Path\n"
        "kind=sys.argv[3]\n"
        "payload=sys.argv[5:]\n"
        "if kind=='memory_gate':\n"
        "    raise SystemExit(d.supervise_memory_gate_attempt(Path(sys.argv[2]),"
        "trajectory_id=sys.argv[4],actual_payload_argv=payload,"
        "actual_environment=dict(os.environ)))\n"
        "claim=d.claim_matrix_attempt(Path(sys.argv[2]),trajectory_id=sys.argv[4],"
        "kind=kind,actual_payload_argv=payload,actual_environment=dict(os.environ))\n"
        "d.apply_claimed_resume_environment(claim,os.environ)\n"
        "d.verify_attempt_execution_authority(Path(sys.argv[2]),trajectory_id=sys.argv[4],"
        "kind=kind,claim=claim,actual_payload_argv=payload,"
        "actual_environment=dict(os.environ))\n"
        "os.execv(sys.argv[5],payload)"
    )
    return [
        str(python_executable),
        "-c",
        code,
        str(project_root / "scripts"),
        str(contract_path),
        kind,
        trajectory_id,
        str(python_executable),
        *[str(value) for value in payload_argv[1:]],
    ]


def build_fidelity_audit_argv(
    *,
    python_executable: Path,
    project_root: Path,
    contract_path: Path,
    trajectory_id: str,
) -> list[str]:
    return [
        str(python_executable),
        str(project_root / "scripts" / "dry_run_tsp_fcdf_beta_pilot.py"),
        "--contract",
        str(contract_path),
        "--audit-cell",
        trajectory_id,
        "--format",
        "json",
    ]


def build_relocation_authorization_argv(
    *,
    python_executable: Path,
    project_root: Path,
    relocation: Mapping[str, Any],
    converted_source: Mapping[str, Any],
) -> list[str]:
    return [
        str(python_executable),
        str(project_root / "scripts/relocate_development_adapter_base.py"),
        "--source-adapter-path",
        str(converted_source["checkpoint_path"]),
        "--frozen-base-model-path",
        str(relocation["frozen_base_path"]),
        "--output-dir",
        str(relocation["adapter_path"]),
        "--verify-only",
    ]


def build_eval_argv(
    *,
    python_executable: Path,
    tokenizer_path: Path,
    development_data_root: Path,
    output_root: Path,
    run_dir: Path,
    trajectory_id: str,
) -> list[str]:
    return [
        str(python_executable),
        "-m",
        "clasp.entrypoints.evaluate",
        "--tasks",
        "tsp",
        "--stage",
        "grpo",
        "--model_path",
        str(run_dir / "checkpoints" / "checkpoint-400"),
        "--tokenizer_path",
        str(tokenizer_path),
        "--data_root",
        str(development_data_root),
        "--split",
        "eval",
        "--evaluation_purpose",
        "development",
        "--prompt_style",
        "alpaca",
        "--force_no_reason_prompt",
        "--dtype",
        "bfloat16",
        "--seed",
        "271828",
        "--no-prefer_subset_files",
        "--max_eval_samples_per_task",
        "200",
        "--eval_start_index",
        "0",
        "--size_bucket",
        "all",
        "--max_prompt_length",
        "29768",
        "--max_new_tokens",
        "3000",
        "--num_return_sequences",
        "8",
        "--generation_batch_size",
        "8",
        "--do_sample",
        "--temperature",
        "0.7",
        "--top_p",
        "0.9",
        "--top_k",
        "50",
        "--repetition_penalty",
        "1",
        "--inference_root",
        str(output_root / "terminal_development_evaluation" / trajectory_id),
        "--eval_name",
        "checkpoint400_eval200_k8_seed271828",
        "--no-update_best_run",
        "--save_details",
        "--save_results",
        "--no-resume",
    ]


def render_command(command: Mapping[str, Any]) -> str:
    unset = " ".join(f"-u {shlex.quote(name)}" for name in command.get("unset_env", []))
    environment = " ".join(
        f"{name}={shlex.quote(value)}" for name, value in sorted(command.get("env", {}).items())
    )
    argv = " ".join(shlex.quote(part) for part in command["argv"])
    prefix = "env -i"
    if unset:
        prefix += " " + unset
    if environment:
        prefix += " " + environment
    return f"{prefix} {argv}"


def build_plan(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    include_cell_commands: bool = False,
) -> dict[str, Any]:
    paths = contract["paths"]
    policy_initialization = contract["policy_initialization"]
    relocation = contract["relocated_saved_policy"]
    converted_source = contract["converted_source_lineage"]
    data = contract["data"]
    python_executable = Path(paths["python_executable"])
    project_root = Path(paths["project_root"])
    output_root = Path(paths["output_root"])
    expected_contract_path = Path(paths["frozen_contract_path"])
    contract_path = contract_path.expanduser().absolute()
    if contract_path != expected_contract_path:
        contractlib.fail("dry-run contract path differs from preregistered frozen_contract_path")
    expected_precision = {
        "relocation_verification_dtype": contractlib.RELOCATION_PROOF_DTYPE,
        "training_load_dtype": contractlib.TRAINING_LOAD_DTYPE,
        "common_bfloat16_load_not_exact_fp32_source_policy_equivalence": True,
        "exact_source_policy_equivalence_claim_allowed": False,
    }
    mismatches = {
        key: {"actual": policy_initialization.get(key), "expected": expected}
        for key, expected in expected_precision.items()
        if policy_initialization.get(key) != expected
    }
    if mismatches:
        contractlib.fail(f"policy initialization precision boundary mismatch: {mismatches}")
    # Every cell directly continues the same relocated bias=all adapter in
    # BF16. No merge and no fresh RL LoRA are involved. The nonzero SFT LoRA
    # A/B tensors and saved base-layer biases remain the optimization geometry.
    init_path = Path(policy_initialization["model_path"])
    tokenizer_path = Path(policy_initialization["tokenizer_path"])
    train_path = Path(data["train"]["path"])
    development_path = Path(data["development"]["source_path"])
    train_root = train_path.parents[2]
    development_root = development_path.parents[2]
    matrix_authorization_path = output_root / MATRIX_AUTHORIZATION_FILENAME

    authorization = {
        "phase": "strict_relocation_source_reauthorization",
        "status": "DRY_RUN_ONLY_NOT_EXECUTED",
        "cwd": str(project_root),
        "env": {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HOME": "/home/lamsade/habgaryan",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "LOGNAME": "habgaryan",
            "PATH": contractlib.CLOSED_ENVIRONMENT_PATH,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(project_root / "src"),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "USER": "habgaryan",
        },
        "unset_env": [
            "CLASP_DEVELOPMENT_FIDELITY",
            "CLASP_PUBLICATION_FIDELITY",
            TRAINING_RUN_CONTRACT_ENV,
            *KL_ENV_NAMES,
        ],
        "argv": build_relocation_authorization_argv(
            python_executable=python_executable,
            project_root=project_root,
            relocation=relocation,
            converted_source=converted_source,
        ),
        "authorizes_memory_gate_source_only": True,
        "receipt_only_digests_are_not_authorization": True,
    }
    authorization["rendered"] = render_command(authorization)

    gate_dir = output_root / MEMORY_GATE_DIRNAME
    gate_environment, gate_unset = build_environment(
        method="dense_rank",
        beta=0.05,
        seed=42,
        relocation=relocation,
        training_run_contract_path=contract_path,
        project_root=project_root,
        tmpdir=gate_dir / "_runtime_tmp",
        create_tmpdir=True,
    )
    gate_environment["GRPO_STOP_AT_GLOBAL_STEP"] = "1"
    gate_unset = [name for name in gate_unset if name != "GRPO_STOP_AT_GLOBAL_STEP"]
    memory_gate = {
        "phase": "one_step_beta_0.05_memory_and_exactness_gate",
        "status": "AUTHORIZED_ENGINEERING_PREREQUISITE_BY_FROZEN_CONTRACT",
        "launch_authorized": True,
        "scientific_result_eligible": False,
        "cwd": str(project_root),
        "env": gate_environment,
        "unset_env": gate_unset,
        "argv": build_runtime_guarded_argv(
            python_executable=python_executable,
            workspace_root=project_root,
            payload_argv=build_attempt_guarded_argv(
                python_executable=python_executable,
                project_root=project_root,
                contract_path=contract_path,
                kind="memory_gate",
                trajectory_id="engineering-gate-D5-seed42",
                payload_argv=build_train_argv(
                    python_executable=python_executable,
                    initialization=init_path,
                    tokenizer_path=tokenizer_path,
                    train_data_root=train_root,
                    output_root=output_root,
                    run_dir=gate_dir,
                    beta=0.05,
                    seed=42,
                    max_steps=400,
                    warmup_ratio=0.0,
                ),
            ),
        ),
        "output_dir": str(gate_dir),
        "requires_successful_phase": "strict_relocation_source_reauthorization",
        "required_evidence": contract["memory_gate"]["success_evidence"],
        "scheduler_contract": {
            "max_steps_horizon": 400,
            "hard_stop_global_step": 1,
            "warmup_ratio": 0.0,
            "reason": (
                "the scientific pilot warmup_ratio=0.03 has zero LR on update one; the "
                "engineering-only gate fixes warmup_ratio=0 so policy-change evidence is real"
            ),
            "resume_or_extension_forbidden": True,
        },
        "available_core_evidence": [
            "pre_optimization_exact_pair_runtime_receipt",
            "zero_initial_reference_trainables",
            "immutable_source_binding",
            "checkpoint_1_trainable_policy_fidelity",
            "step1_live_reference_immutability_receipt",
            "checkpoint_1_live_reference_immutability_receipt",
            "per_rank_in_process_cuda_identity_and_peak_memory",
        ],
        "missing_fail_closed_evidence": None,
        "note": (
            "This is a single create-once engineering attempt. It can unlock the immutable "
            "full-matrix ledger only after checkpoint-1 passes the external changed-policy, "
            "live-reference, fidelity, finite-telemetry, and CUDA-memory audit."
        ),
    }
    memory_gate["rendered"] = render_command(memory_gate)

    training = []
    audits = []
    evaluations = []
    for cell in contract["design"]["cells"]:
        method = str(cell["method"])
        beta = float(cell["grpo_beta"])
        seed = int(cell["seed"])
        trajectory_id = str(cell["trajectory_id"])
        run_dir = _run_directory(output_root, method, beta, seed)
        environment, unset = build_environment(
            method=method,
            beta=beta,
            seed=seed,
            relocation=relocation,
            training_run_contract_path=matrix_authorization_path,
            project_root=project_root,
            tmpdir=run_dir / "_runtime_tmp",
            create_tmpdir=False,
        )
        training_command = {
            "trajectory_id": trajectory_id,
            "cell": cell["cell"],
            "phase": "train_to_fixed_terminal_step",
            "status": "DRY_RUN_ONLY_NOT_EXECUTED",
            "cwd": str(project_root),
            "env": environment,
            "unset_env": unset,
            "argv": build_runtime_guarded_argv(
                python_executable=python_executable,
                workspace_root=project_root,
                payload_argv=build_attempt_guarded_argv(
                    python_executable=python_executable,
                    project_root=project_root,
                    contract_path=contract_path,
                    kind="training",
                    trajectory_id=trajectory_id,
                    payload_argv=build_train_argv(
                        python_executable=python_executable,
                        initialization=init_path,
                        tokenizer_path=tokenizer_path,
                        train_data_root=train_root,
                        output_root=output_root,
                        run_dir=run_dir,
                        beta=beta,
                        seed=seed,
                    ),
                ),
            ),
            "output_dir": str(run_dir),
            "requires_successful_phase": (
                "one_step_beta_memory_gate_with_post_step_reference_audit"
            ),
            "launch_authorized": False,
        }
        training_command["rendered"] = render_command(training_command)
        training.append(training_command)

        audit_command = {
            "trajectory_id": trajectory_id,
            "phase": "fail_closed_terminal_fidelity_audit",
            "status": "DRY_RUN_ONLY_NOT_EXECUTED",
            "cwd": str(project_root),
            "env": {
                "CUDA_CACHE_PATH": str(run_dir / "_runtime_tmp" / "cuda"),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "3",
                "HF_DATASETS_OFFLINE": "1",
                "HF_DATASETS_CACHE": str(run_dir / "_runtime_tmp" / "huggingface" / "datasets"),
                "HF_HOME": str(run_dir / "_runtime_tmp" / "huggingface"),
                "HF_HUB_OFFLINE": "1",
                "HOME": str(run_dir / "_runtime_tmp" / "home"),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "LOGNAME": "habgaryan",
                "PATH": contractlib.CLOSED_ENVIRONMENT_PATH,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(project_root / "src"),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TMPDIR": str(run_dir / "_runtime_tmp"),
                "TMP": str(run_dir / "_runtime_tmp"),
                "TEMP": str(run_dir / "_runtime_tmp"),
                "TOKENIZERS_PARALLELISM": "false",
                "TORCHINDUCTOR_CACHE_DIR": str(run_dir / "_runtime_tmp" / "torchinductor"),
                "TORCH_HOME": str(run_dir / "_runtime_tmp" / "torch"),
                "TRANSFORMERS_OFFLINE": "1",
                "TRITON_CACHE_DIR": str(run_dir / "_runtime_tmp" / "triton"),
                "USER": "habgaryan",
                "XDG_CACHE_HOME": str(run_dir / "_runtime_tmp" / "xdg"),
                TRAINING_RUN_CONTRACT_ENV: str(matrix_authorization_path),
            },
            "unset_env": [],
            "launch_authorized": False,
            "argv": build_runtime_guarded_argv(
                python_executable=python_executable,
                workspace_root=project_root,
                payload_argv=build_fidelity_audit_argv(
                    python_executable=python_executable,
                    project_root=project_root,
                    contract_path=contract_path,
                    trajectory_id=trajectory_id,
                ),
            ),
        }
        audit_command["rendered"] = render_command(audit_command)
        audits.append(audit_command)

        evaluation_attempt_root = output_root / "terminal_development_evaluation" / trajectory_id
        evaluation_tmpdir = evaluation_attempt_root / "_runtime_tmp"
        eval_command = {
            "trajectory_id": trajectory_id,
            "phase": "terminal_frozen_development_evaluation",
            "status": "DRY_RUN_ONLY_NOT_EXECUTED",
            "cwd": str(project_root),
            "env": {
                "CUDA_CACHE_PATH": str(evaluation_tmpdir / "cuda"),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "3",
                "CLASP_TSP_PROMPT_KNN": "0",
                "CLASP_TSP_PROMPT_KNN_DISTANCES": "0",
                "CLASP_TSP_ROUTE_ONLY_OUTPUT": "0",
                "HF_DATASETS_OFFLINE": "1",
                "HF_DATASETS_CACHE": str(evaluation_tmpdir / "huggingface" / "datasets"),
                "HF_HOME": str(evaluation_tmpdir / "huggingface"),
                "HF_HUB_OFFLINE": "1",
                "HOME": str(evaluation_tmpdir / "home"),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "LOGNAME": "habgaryan",
                "PATH": contractlib.CLOSED_ENVIRONMENT_PATH,
                "PYTHONHASHSEED": "271828",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(project_root / "src"),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TOKENIZERS_PARALLELISM": "false",
                "TMPDIR": str(evaluation_tmpdir),
                "TMP": str(evaluation_tmpdir),
                "TEMP": str(evaluation_tmpdir),
                "TORCHINDUCTOR_CACHE_DIR": str(evaluation_tmpdir / "torchinductor"),
                "TORCH_HOME": str(evaluation_tmpdir / "torch"),
                "TRANSFORMERS_OFFLINE": "1",
                "TRITON_CACHE_DIR": str(evaluation_tmpdir / "triton"),
                "USER": "habgaryan",
                "XDG_CACHE_HOME": str(evaluation_tmpdir / "xdg"),
                TRAINING_RUN_CONTRACT_ENV: str(matrix_authorization_path),
            },
            "unset_env": [
                "RESUME_FROM_CHECKPOINT",
                "CLASP_DEVELOPMENT_FIDELITY",
                "CLASP_PUBLICATION_FIDELITY",
                *KL_ENV_NAMES,
            ],
            "launch_authorized": False,
            "argv": build_runtime_guarded_argv(
                python_executable=python_executable,
                workspace_root=project_root,
                payload_argv=build_attempt_guarded_argv(
                    python_executable=python_executable,
                    project_root=project_root,
                    contract_path=contract_path,
                    kind="evaluation",
                    trajectory_id=trajectory_id,
                    payload_argv=build_eval_argv(
                        python_executable=python_executable,
                        tokenizer_path=tokenizer_path,
                        development_data_root=development_root,
                        output_root=output_root,
                        run_dir=run_dir,
                        trajectory_id=trajectory_id,
                    ),
                ),
            ),
            "requires_successful_phase": "fail_closed_terminal_fidelity_audit",
        }
        eval_command["rendered"] = render_command(eval_command)
        evaluations.append(eval_command)

    visible_training = training if include_cell_commands else []
    visible_audits = audits if include_cell_commands else []
    visible_evaluations = evaluations if include_cell_commands else []
    return {
        "schema_version": 3,
        "protocol": "clasp-tsp-fcdf-beta-development-dry-run-v3",
        "status": "READY_FOR_UNRUN_ONE_UPDATE_ENGINEERING_GATE",
        "development_only": True,
        "publication_eligible": False,
        "sota_eligible": False,
        "contract_sha256": contract["contract_sha256"],
        "training_or_evaluation_started": False,
        "precision_claim_boundary": {
            **expected_precision,
            "continued_lora_optimization_geometry": True,
            "fresh_rl_lora_attached": False,
            "merged_reference_or_policy_used": False,
            "historical_terminal_sft_policy_claim_allowed": False,
            "publication_claim_allowed": False,
            "sota_claim_allowed": False,
        },
        "runtime_preconditions": {
            "world_size": 1,
            "exactly_one_visible_gpu": True,
            "deepspeed": False,
            "fsdp": False,
            "fresh_output_root": str(output_root),
            "physical_gpu_index": 3,
            "gpu_uuid": contractlib.EXPECTED_LOCAL_HARDWARE["uuid"],
            "gpu_name": contractlib.EXPECTED_LOCAL_HARDWARE["name"],
            "driver_version": contractlib.EXPECTED_LOCAL_HARDWARE["driver_version"],
            "note": (
                "the transient idle check is never authority; the same-process env-i guard "
                "rechecks exact runtime, physical GPU identity, contract, inputs, and source"
            ),
        },
        "environment_policy": {
            "inheritance": "fail_closed_allowlist_only",
            "forbidden_inherited_prefixes": [
                "CLASP_",
                "PD_",
                "GRPO_",
                "GENERATION_",
                "WANDB_",
            ],
            "scheduler_owned_exceptions": [],
            "rendered_commands_use_env_i_and_are_closed_environment_executables": True,
        },
        "counts": {
            "relocation_authorization_commands": 1,
            "memory_gate_training_commands": 1,
            "post_step_reference_audit_commands": 1,
            "training_commands": len(visible_training),
            "fidelity_audit_commands": len(visible_audits),
            "evaluation_commands": len(visible_evaluations),
        },
        "relocation_authorization_command": authorization,
        "memory_gate_command": memory_gate,
        "global_launch_authorized": False,
        "hold_reason": (
            "the required one-update beta-positive engineering gate has not run or passed; "
            "all 12 scientific commands remain withheld"
        ),
        "cell_commands_withheld_until_memory_gate_pass": not include_cell_commands,
        "training_commands": visible_training,
        "fidelity_audit_commands": visible_audits,
        "evaluation_commands": visible_evaluations,
        "analysis": contract["analysis"],
    }


def _observed_matrix_outcome_paths(ledger: Mapping[str, Any]) -> list[str]:
    observed: list[str] = []
    for reserved_path, filename, _, _ in _reservation_specifications(ledger):
        allowed = {filename, "_runtime_tmp", CLAIM_ANCHOR_DIRNAME}
        for child in reserved_path.iterdir():
            if child.name not in allowed:
                observed.append(str(child))
        tmpdir = reserved_path / "_runtime_tmp"
        if any(tmpdir.iterdir()):
            observed.append(str(tmpdir))
        anchor_root = reserved_path / CLAIM_ANCHOR_DIRNAME
        observed.extend(str(path) for path in anchor_root.iterdir() if path.stat().st_size > 0)
    return sorted(observed)


def inspect_lifecycle(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
) -> dict[str, Any]:
    """Describe the current immutable phase without creating or reauthorizing bytes."""

    output_root = Path(str(contract["paths"]["output_root"]))
    if not output_root.exists():
        return build_plan(contract, contract_path=contract_path)
    observed_top = _top_level_names(output_root)
    gate_dir = output_root / MEMORY_GATE_DIRNAME
    pass_path = output_root / MEMORY_GATE_PASS_FILENAME
    ledger_path = output_root / MATRIX_AUTHORIZATION_FILENAME
    closure_path = output_root / RESERVATION_CLOSURE_FILENAME
    if not pass_path.exists():
        if observed_top != {MEMORY_GATE_DIRNAME} or not gate_dir.is_dir():
            contractlib.fail("pre-PASS output root contains non-gate bytes")
        attempt_path = gate_dir / CLAIM_ANCHOR_DIRNAME / MEMORY_GATE_ATTEMPT_FILENAME
        if attempt_path.exists():
            command = build_plan(contract, contract_path=contract_path)["memory_gate_command"]
            claim = _inspect_attempt_claim(
                attempt_path,
                expected_protocol=MEMORY_GATE_ATTEMPT_PROTOCOL,
                expected_status="development_memory_gate_attempt_started",
                expected_kind="memory_gate",
                trajectory_id="engineering-gate-D5-seed42",
                ledger_identity=None,
                reservation_identity=None,
                command=command,
                resume_step=None,
            )
            _, outcome_path, outcome = _inspect_memory_gate_execution_anchors(claim)
            if outcome is None:
                return {
                    "schema_version": 1,
                    "protocol": "clasp-tsp-fcdf-beta-development-lifecycle-v1",
                    "status": "MEMORY_GATE_SUPERVISOR_OUTCOME_ABSENT",
                    "development_only": True,
                    "publication_eligible": False,
                    "sota_eligible": False,
                    "global_launch_authorized": False,
                    "retry_allowed": False,
                    "matrix_authorized": False,
                    "cell_commands_reemitted": False,
                    "next_action": (
                        "use --finalize-memory-gate-supervisor-death only after proving "
                        "the transcript lock is released"
                    ),
                }
            if outcome is not None:
                outcome = inspect_memory_gate_execution_outcome(
                    outcome_path,
                    contract,
                    contract_path=contract_path,
                    command=command,
                    claim=claim,
                )
                if outcome.get("status") != MEMORY_GATE_SUCCESS_STATUS:
                    return {
                        "schema_version": 1,
                        "protocol": "clasp-tsp-fcdf-beta-development-lifecycle-v1",
                        "status": "MEMORY_GATE_FAILED_TERMINAL",
                        "development_only": True,
                        "publication_eligible": False,
                        "sota_eligible": False,
                        "global_launch_authorized": False,
                        "retry_allowed": False,
                        "matrix_authorized": False,
                        "execution_outcome": _file_identity(outcome_path, outcome),
                        "cell_commands_reemitted": False,
                        "next_action": None,
                    }
        return {
            "schema_version": 1,
            "protocol": "clasp-tsp-fcdf-beta-development-lifecycle-v1",
            "status": "MEMORY_GATE_OUTPUT_AWAITS_FAIL_CLOSED_ACCEPTANCE",
            "development_only": True,
            "publication_eligible": False,
            "sota_eligible": False,
            "global_launch_authorized": False,
            "next_action": "run --record-memory-gate-pass only after checkpoint-1 completes",
            "cell_commands_reemitted": False,
        }
    memory_pass = inspect_memory_gate_pass(
        pass_path,
        contract,
        contract_path=contract_path,
        reverify_gate=True,
    )
    if not ledger_path.exists() or not closure_path.exists():
        return {
            "schema_version": 1,
            "protocol": "clasp-tsp-fcdf-beta-development-lifecycle-v1",
            "status": "INCOMPLETE_PRE_OUTCOME_MATRIX_CLOSURE",
            "development_only": True,
            "publication_eligible": False,
            "sota_eligible": False,
            "global_launch_authorized": False,
            "memory_gate_pass": _file_identity(pass_path, memory_pass),
            "next_action": (
                "run --record-memory-gate-pass to complete the interrupted two-phase "
                "reservation only if no attempt/outcome byte exists"
            ),
            "cell_commands_reemitted": False,
        }
    ledger = inspect_matrix_authorization(
        ledger_path,
        contract,
        contract_path=contract_path,
        memory_gate_pass_path=pass_path,
        memory_gate_pass=memory_pass,
    )
    closure = inspect_reservation_closure(closure_path, ledger_path, ledger)
    outcomes = _observed_matrix_outcome_paths(ledger)
    if outcomes:
        return {
            "schema_version": 1,
            "protocol": "clasp-tsp-fcdf-beta-development-lifecycle-v1",
            "status": "MATRIX_AUTHORITY_FROZEN_ATTEMPT_OR_OUTCOME_PRESENT",
            "development_only": True,
            "publication_eligible": False,
            "sota_eligible": False,
            "global_launch_authorized": False,
            "ledger": _file_identity(ledger_path, ledger),
            "reservation_closure": _file_identity(closure_path, closure),
            "observed_attempt_or_outcome_count": len(outcomes),
            "observed_attempt_or_outcome_paths": outcomes,
            "cell_commands_reemitted": False,
            "note": (
                "the original immutable ledger remains sole authority; commands are not "
                "regenerated after outcomes, and missing/failed cells remain denominator"
            ),
        }
    verify_pristine_matrix_reservations(
        ledger_path,
        ledger,
        closure_path,
        closure,
    )
    return {
        "schema_version": 1,
        "protocol": "clasp-tsp-fcdf-beta-development-lifecycle-v1",
        "status": "FULL_MATRIX_RESERVED_PRE_OUTCOME",
        "development_only": True,
        "publication_eligible": False,
        "sota_eligible": False,
        "global_launch_authorized": True,
        "ledger": _file_identity(ledger_path, ledger),
        "reservation_closure": _file_identity(closure_path, closure),
        "all_twelve_required": True,
        "cell_commands_reemitted": True,
        "training_commands": ledger["training_commands"],
        "terminal_audit_commands": ledger["terminal_audit_commands"],
        "evaluation_commands": ledger["evaluation_commands"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and print the development-only TSP 2x2 pilot command plan. "
            "It never launches scientific work; its only write modes seal audited evidence."
        )
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--record-memory-gate-pass",
        action="store_true",
        help=(
            "audit checkpoint-1, seal the gate tree, then create the one immutable "
            "full-matrix ledger and reservation closure; never starts a process"
        ),
    )
    action.add_argument(
        "--finalize-memory-gate-supervisor-death",
        action="store_true",
        help=(
            "consume the precreated outcome slot as exit-unknown only after the claimed "
            "memory-gate supervisor is demonstrably gone; never authorizes retry or matrix"
        ),
    )
    action.add_argument(
        "--audit-cell",
        metavar="TRAJECTORY_ID",
        help=(
            "create or reverify one terminal integrity receipt before its sole "
            "preauthorized development evaluation"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        contract_path = args.contract.expanduser().absolute()
        contract = contractlib.load_json_object(contract_path, label="pilot contract")
        if stat.S_IMODE(contract_path.lstat().st_mode) & 0o222:
            contractlib.fail(f"pilot contract must be read-only: {contract_path}")
        output_root = Path(str(contract.get("paths", {}).get("output_root", "")))
        contractlib.verify_contract(
            contract,
            require_fresh_output=not output_root.exists(),
            contract_path=contract_path,
        )
        if args.finalize_memory_gate_supervisor_death:
            outcome = finalize_interrupted_memory_gate(
                contract,
                contract_path=contract_path,
            )
            plan = {
                "schema_version": 1,
                "protocol": "clasp-tsp-fcdf-beta-memory-gate-failure-finalization-v1",
                "status": "MEMORY_GATE_FAILED_TERMINAL",
                "development_only": True,
                "publication_eligible": False,
                "sota_eligible": False,
                "global_launch_authorized": False,
                "retry_allowed": False,
                "matrix_authorized": False,
                "execution_outcome": _file_identity(
                    Path(str(outcome["outcome_path"])),
                    outcome,
                ),
            }
        elif args.record_memory_gate_pass:
            memory_pass, ledger, closure = record_memory_gate_pass_and_matrix(
                contract,
                contract_path=contract_path,
            )
            plan = {
                "schema_version": 1,
                "protocol": "clasp-tsp-fcdf-beta-development-matrix-release-v1",
                "status": "FULL_MATRIX_RESERVED_PRE_OUTCOME",
                "development_only": True,
                "publication_eligible": False,
                "sota_eligible": False,
                "global_launch_authorized": True,
                "memory_gate_pass": _file_identity(
                    output_root / MEMORY_GATE_PASS_FILENAME,
                    memory_pass,
                ),
                "ledger": _file_identity(
                    output_root / MATRIX_AUTHORIZATION_FILENAME,
                    ledger,
                ),
                "reservation_closure": _file_identity(
                    output_root / RESERVATION_CLOSURE_FILENAME,
                    closure,
                ),
                "all_twelve_required": True,
                "training_commands": ledger["training_commands"],
                "terminal_audit_commands": ledger["terminal_audit_commands"],
                "evaluation_commands": ledger["evaluation_commands"],
            }
        elif args.audit_cell:
            plan = record_terminal_cell_audit(
                contract_path,
                trajectory_id=args.audit_cell,
            )
        else:
            plan = inspect_lifecycle(contract, contract_path=contract_path)
    except (contractlib.ContractError, OSError, KeyError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(f"# {plan.get('status', 'DEVELOPMENT_ONLY')} — no process was started.")
        if "relocation_authorization_command" in plan:
            print("# Strict relocation reauthorization:")
            print(plan["relocation_authorization_command"]["rendered"])
        if "memory_gate_command" in plan:
            print("# One-update beta memory gate:")
            print(plan["memory_gate_command"]["rendered"])
        print("# 400-step cells from the immutable pre-outcome ledger:")
        for command in plan.get("training_commands", []):
            print(f"# {command['trajectory_id']}")
            print(command["rendered"])
        print("# Create-once terminal audits:")
        for command in plan.get(
            "terminal_audit_commands",
            plan.get("fidelity_audit_commands", []),
        ):
            print(command["rendered"])
        print("# Single-attempt frozen terminal development evaluations:")
        for command in plan.get("evaluation_commands", []):
            print(command["rendered"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
