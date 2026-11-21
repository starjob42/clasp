#!/usr/bin/env python3
"""Run a preregistered, development-only OP exact-solver scalability sweep.

This harness is intentionally separate from publication reference generation.
It samples a frozen, unlabeled, non-confirmation IID-like OP draft by explicit
node-count strata and calls the exact OP model's diagnostic surface.  Native
resource-limit states are retained as diagnostic evidence but are never
relabeled as exact references.  A successful development screen is not
authorization to solve or inspect confirmation data.

The checked-in contract is a HOLD template.  Running requires a distinct,
read-only, self-hashed contract whose source, strata, limits, solver identity,
runner source, verifier identity, and operational budget were frozen before
the sweep.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from clasp.data.benchmark import (
    REFERENCE_OBJECTIVE_SEMANTICS,
    benchmark_row_sha256,
    semantic_instance_sha256,
    sha256_json,
)

try:  # Direct script execution places ``scripts`` on sys.path.
    import run_reference_evidence as reference_runner
except ModuleNotFoundError:  # Importing as ``scripts.run_op_scalability_sweep``.
    from scripts import run_reference_evidence as reference_runner  # type: ignore


SCHEMA_VERSION = 1
CONTRACT_PROTOCOL = "clasp-op-reference-scalability-sweep-v1"
CONTRACT_STATUS = "frozen_before_development_sweep"
PENDING_REMOTE_STATUS = (
    "frozen_before_development_sweep_pending_read_only_artifact"
)
HOLD_STATUS = "HOLD_UNRESOLVED_DO_NOT_RUN"
PREFLIGHT_HOLD_REMOTE = "HOLD_REMOTE_READ_ONLY_ARTIFACT_FINALIZATION_REQUIRED"
RECEIPT_PROTOCOL = "clasp-op-reference-scalability-sweep-receipt-v1"
FINALIZATION_PROTOCOL = "clasp-op-scalability-remote-finalization-v1"
SOURCE_ROLE = "development_iid_like_nonconfirmation"
ADAPTER = reference_runner.OP_EXACT_ADAPTER
OBJECTIVE_SEMANTICS = REFERENCE_OBJECTIVE_SEMANTICS["op"]
TARGET_PRODUCTION_ROWS = 1_000
EXACT_CLASSIFICATION = "OPTIMAL_CERTIFIED"
NONEXACT_CLASSIFICATION = "NOT_CERTIFIED"
PASS_SCREEN = "PASS_DEVELOPMENT_SCREEN_NOT_PRODUCTION_AUTHORIZATION"
HOLD_NONOPTIMAL = "HOLD_NONOPTIMAL_OBSERVED"
HOLD_RUNTIME = "HOLD_PROJECTED_RUNTIME_ABOVE_PREREGISTERED_BUDGET"


class DiagnosticOPBackend(Protocol):
    def probe_identity(self) -> dict[str, str]: ...

    def solve_diagnostic(
        self,
        row: dict[str, object],
        *,
        parameters: dict[str, object],
    ) -> reference_runner.NativeSolveResult: ...


@dataclass(frozen=True)
class SelectedRow:
    stratum: dict[str, object]
    source_index: int
    sample_rank: int
    row: dict[str, object]


@dataclass(frozen=True)
class PreparedSweep:
    contract_path: Path
    contract: dict[str, object]
    contract_file_identity: dict[str, object]
    source_rows: list[dict[str, object]]
    selected_rows: list[SelectedRow]
    source_rows_by_stratum: dict[str, int]
    evaluator: Callable[..., dict]
    verifier_identity: dict[str, object]


@dataclass(frozen=True)
class PendingSweepPreflight:
    """Validated local plan which remains deliberately non-executable."""

    contract_path: Path
    contract: dict[str, object]
    source_rows: list[dict[str, object]]
    selected_rows: list[SelectedRow]
    source_rows_by_stratum: dict[str, int]
    verifier_identity: dict[str, object]
    finalization_spec: dict[str, object]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("OP scalability values must be finite canonical JSON") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )


def _strict_file_identity(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    if metadata.st_nlink != 1:
        raise ValueError(f"{label} must be singly linked: {path}")
    mode = stat.S_IMODE(metadata.st_mode)
    if mode & 0o222:
        raise ValueError(f"{label} must be frozen read-only, mode is {mode:04o}")
    return {
        "canonical_path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size_bytes": int(metadata.st_size),
        "mode": f"{mode:04o}",
    }


def _load_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verify_self_hash(value: Mapping[str, object], field: str, *, label: str) -> None:
    digest = value.get(field)
    if not _is_sha256(digest):
        raise ValueError(f"{label} has no valid {field}")
    unsigned = dict(value)
    unsigned.pop(field, None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if observed != digest:
        raise ValueError(f"{label} {field} does not match")


def seal_contract(value: Mapping[str, object]) -> dict[str, object]:
    """Return a contract copy with its canonical self-hash populated."""

    sealed = copy.deepcopy(dict(value))
    sealed.pop("contract_sha256", None)
    sealed["contract_sha256"] = hashlib.sha256(
        _canonical_json_bytes(sealed)
    ).hexdigest()
    return sealed


def _scientific_material(contract: Mapping[str, object]) -> dict[str, object]:
    """Project a pending contract onto choices remote finalization may not alter."""

    source = contract.get("source")
    solver = contract.get("solver")
    if not isinstance(source, dict) or not isinstance(solver, dict):
        raise ValueError("pending OP contract has no scientific source or solver material")
    source_core = {
        key: copy.deepcopy(source[key])
        for key in (
            "role",
            "confirmation_data_used",
            "draft_root",
            "expected_rows",
            "generation_receipt_sha256",
            "op_dataset_relative_path",
            "op_dataset_sha256",
            "generation_contract_relative_path",
            "generation_contract_file_sha256",
            "generation_contract_sha256",
        )
        if key in source
    }
    return {
        "purpose": copy.deepcopy(contract.get("purpose")),
        "source": source_core,
        "selection": copy.deepcopy(contract.get("selection")),
        "solver": {
            "adapter": copy.deepcopy(solver.get("adapter")),
            "objective_semantics": copy.deepcopy(solver.get("objective_semantics")),
            "runner_code_sha256": copy.deepcopy(solver.get("runner_code_sha256")),
            "license_class": copy.deepcopy(solver.get("license_class")),
            "backend_identity": copy.deepcopy(solver.get("backend_identity")),
            "base_parameters": copy.deepcopy(solver.get("base_parameters")),
        },
        "strict_verifier_identity_sha256": copy.deepcopy(
            contract.get("strict_verifier_identity_sha256")
        ),
        "production_screen": copy.deepcopy(contract.get("production_screen")),
    }


def _require_exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} keys differ; missing={missing}, extra={extra}")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _positive_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _optional_limit(
    value: object,
    *,
    label: str,
    integer: bool,
) -> int | float | None:
    if value is None:
        return None
    if integer:
        return _positive_int(value, label=label)
    return _positive_number(value, label=label)


def _op_node_count(row: Mapping[str, object]) -> int:
    instance = row.get("instance_json")
    if not isinstance(instance, list) or len(instance) < 3:
        raise ValueError("OP sweep source row has no structured OP instance")
    coordinates, prizes, _route_limit = instance[:3]
    if not isinstance(coordinates, list) or not isinstance(prizes, list):
        raise ValueError("OP sweep source row has malformed coordinates or prizes")
    count = len(coordinates)
    if count < 3 or len(prizes) != count:
        raise ValueError("OP sweep source row has inconsistent node dimensions")
    try:
        declared = int(str(row.get("num_nodes", "")))
    except ValueError as exc:
        raise ValueError("OP sweep source row has no exact num_nodes") from exc
    if declared != count:
        raise ValueError("OP sweep source row num_nodes disagrees with instance_json")
    return count


def _validate_strata(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError("selection.strata must be a non-empty list")
    expected = {
        "stratum_id",
        "min_nodes",
        "max_nodes",
        "sample_count",
        "time_limit_seconds",
        "work_limit",
        "node_limit",
    }
    strata: list[dict[str, object]] = []
    identifiers: set[str] = set()
    ranges: list[tuple[int, int, str]] = []
    for index, raw in enumerate(value):
        stratum = _require_exact_keys(raw, expected, label=f"selection.strata[{index}]")
        identifier = str(stratum["stratum_id"]).strip()
        if not identifier or identifier in identifiers:
            raise ValueError("selection strata require unique non-empty stratum_id values")
        identifiers.add(identifier)
        minimum = _positive_int(stratum["min_nodes"], label=f"{identifier}.min_nodes")
        maximum = _positive_int(stratum["max_nodes"], label=f"{identifier}.max_nodes")
        if maximum < minimum:
            raise ValueError(f"{identifier} max_nodes must be >= min_nodes")
        sample_count = _positive_int(
            stratum["sample_count"], label=f"{identifier}.sample_count"
        )
        time_limit = _positive_number(
            stratum["time_limit_seconds"],
            label=f"{identifier}.time_limit_seconds",
        )
        work_limit = _optional_limit(
            stratum["work_limit"], label=f"{identifier}.work_limit", integer=False
        )
        node_limit = _optional_limit(
            stratum["node_limit"], label=f"{identifier}.node_limit", integer=True
        )
        for left, right, other in ranges:
            if max(left, minimum) <= min(right, maximum):
                raise ValueError(f"size strata {identifier!r} and {other!r} overlap")
        ranges.append((minimum, maximum, identifier))
        strata.append(
            {
                "stratum_id": identifier,
                "min_nodes": minimum,
                "max_nodes": maximum,
                "sample_count": sample_count,
                "time_limit_seconds": time_limit,
                "work_limit": work_limit,
                "node_limit": node_limit,
            }
        )
    return strata


def _validate_base_parameters(value: object) -> dict[str, object]:
    expected = dict(reference_runner.OP_EXACT_REQUIRED_PARAMETERS)
    parameters = _require_exact_keys(
        value,
        set(expected),
        label="solver.base_parameters",
    )
    for key, expected_value in expected.items():
        observed = parameters[key]
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise ValueError(f"solver.base_parameters.{key} must be numeric")
        if float(observed) != float(expected_value):
            raise ValueError(
                f"solver.base_parameters.{key} must equal exact-adapter value "
                f"{expected_value!r}"
            )
    return copy.deepcopy(parameters)


def _validate_contract_shape(contract: dict[str, object]) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    if contract.get("status") == HOLD_STATUS:
        reasons = contract.get("hold_reasons")
        detail = "; ".join(str(value) for value in reasons) if isinstance(reasons, list) else ""
        raise ValueError(f"OP scalability sweep contract remains HOLD: {detail}")
    expected_top = {
        "schema_version",
        "protocol",
        "status",
        "frozen_at_utc",
        "contract_sha256",
        "purpose",
        "source",
        "selection",
        "solver",
        "strict_verifier_identity_sha256",
        "production_screen",
    }
    _require_exact_keys(contract, expected_top, label="OP scalability contract")
    if contract["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"OP scalability schema_version must equal {SCHEMA_VERSION}")
    if contract["protocol"] != CONTRACT_PROTOCOL:
        raise ValueError(f"OP scalability protocol must equal {CONTRACT_PROTOCOL!r}")
    if contract["status"] != CONTRACT_STATUS:
        raise ValueError(f"OP scalability status must equal {CONTRACT_STATUS!r}")
    frozen_at = contract["frozen_at_utc"]
    if not isinstance(frozen_at, str) or not frozen_at.endswith("Z"):
        raise ValueError("OP scalability frozen_at_utc must be an explicit UTC timestamp")
    if contract["purpose"] != "development_only_production_feasibility_screen":
        raise ValueError("OP scalability purpose is not development-only")
    if not _is_sha256(contract["strict_verifier_identity_sha256"]):
        raise ValueError("OP scalability strict verifier identity is invalid")

    source = _require_exact_keys(
        contract["source"],
        {
            "role",
            "confirmation_data_used",
            "draft_root",
            "expected_rows",
            "generation_receipt_sha256",
            "op_dataset_relative_path",
            "op_dataset_sha256",
        },
        label="source",
    )
    if source["role"] != SOURCE_ROLE or source["confirmation_data_used"] is not False:
        raise ValueError("OP sweep source must be development IID-like non-confirmation data")
    root = Path(str(source["draft_root"]))
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError("source.draft_root must be an absolute canonical path")
    _positive_int(source["expected_rows"], label="source.expected_rows")
    if not _is_sha256(source["generation_receipt_sha256"]):
        raise ValueError("source generation receipt SHA-256 is invalid")
    if not _is_sha256(source["op_dataset_sha256"]):
        raise ValueError("source OP dataset SHA-256 is invalid")
    relative = source["op_dataset_relative_path"]
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("source.op_dataset_relative_path must stay within draft_root")

    selection = _require_exact_keys(
        contract["selection"],
        {"selection_seed", "strata"},
        label="selection",
    )
    _nonnegative_int(selection["selection_seed"], label="selection.selection_seed")
    strata = _validate_strata(selection["strata"])

    solver = _require_exact_keys(
        contract["solver"],
        {
            "adapter",
            "objective_semantics",
            "runner_code_sha256",
            "license_class",
            "artifact",
            "backend_identity",
            "base_parameters",
        },
        label="solver",
    )
    if solver["adapter"] != ADAPTER:
        raise ValueError(f"OP scalability adapter must equal {ADAPTER!r}")
    if solver["objective_semantics"] != OBJECTIVE_SEMANTICS:
        raise ValueError("OP scalability objective semantics are not exact raw Euclidean OP")
    if solver["license_class"] != "gurobi_academic_site_license_nonredistributable":
        raise ValueError("OP scalability contract must retain the licensed solver class")
    if not _is_sha256(solver["runner_code_sha256"]):
        raise ValueError("solver.runner_code_sha256 is invalid")
    artifact = _require_exact_keys(
        solver["artifact"], {"path", "sha256"}, label="solver.artifact"
    )
    artifact_path = Path(str(artifact["path"]))
    if not artifact_path.is_absolute() or artifact_path.resolve() != artifact_path:
        raise ValueError("solver artifact path must be absolute and canonical")
    if not _is_sha256(artifact["sha256"]):
        raise ValueError("solver artifact SHA-256 is invalid")
    backend_identity = _require_exact_keys(
        solver["backend_identity"],
        {
            "backend_name",
            "native_version",
            "distribution_name",
            "module_origin",
            "installed_distribution_manifest_sha256",
        },
        label="solver.backend_identity",
    )
    for field, value in backend_identity.items():
        if not isinstance(value, str) or not value or "UNRESOLVED" in value.upper():
            raise ValueError(f"solver.backend_identity.{field} is unresolved")
    _validate_base_parameters(solver["base_parameters"])

    screen = _require_exact_keys(
        contract["production_screen"],
        {
            "target_rows",
            "require_all_sampled_optimal",
            "max_projected_native_runtime_seconds",
        },
        label="production_screen",
    )
    if screen["target_rows"] != TARGET_PRODUCTION_ROWS:
        raise ValueError(f"production_screen.target_rows must equal {TARGET_PRODUCTION_ROWS}")
    if screen["require_all_sampled_optimal"] is not True:
        raise ValueError("exact OP production screen must require every sampled row OPTIMAL")
    _positive_number(
        screen["max_projected_native_runtime_seconds"],
        label="production_screen.max_projected_native_runtime_seconds",
    )
    return source, strata, solver, screen


def _validate_pending_contract_shape(contract: dict[str, object]) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    """Validate a fully frozen scientific plan whose artifact gate is still HOLD.

    The pending form is intentionally not accepted by :func:`prepare_sweep` or
    :func:`run_sweep`.  It permits an exact, already observed remote backend
    identity, but records that the audited wheel is still owner-writable and
    therefore cannot satisfy the executable contract's immutable-artifact
    requirement.
    """

    expected_top = {
        "schema_version",
        "protocol",
        "status",
        "frozen_at_utc",
        "contract_sha256",
        "purpose",
        "source",
        "selection",
        "solver",
        "strict_verifier_identity_sha256",
        "production_screen",
        "execution_policy",
        "remote_identity_resolution",
    }
    _require_exact_keys(contract, expected_top, label="pending OP scalability contract")
    if contract["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"pending OP schema_version must equal {SCHEMA_VERSION}")
    if contract["protocol"] != CONTRACT_PROTOCOL:
        raise ValueError("pending OP scalability protocol is invalid")
    if contract["status"] != PENDING_REMOTE_STATUS:
        raise ValueError(f"pending OP status must equal {PENDING_REMOTE_STATUS!r}")
    frozen_at = contract["frozen_at_utc"]
    if not isinstance(frozen_at, str) or not frozen_at.endswith("Z"):
        raise ValueError("pending OP frozen_at_utc must be an explicit UTC timestamp")
    if contract["purpose"] != "development_only_production_feasibility_screen":
        raise ValueError("pending OP purpose is not development-only")
    if not _is_sha256(contract["strict_verifier_identity_sha256"]):
        raise ValueError("pending OP strict verifier identity is invalid")

    source = _require_exact_keys(
        contract["source"],
        {
            "role",
            "confirmation_data_used",
            "draft_root",
            "expected_rows",
            "generation_receipt_sha256",
            "op_dataset_relative_path",
            "op_dataset_sha256",
            "generation_contract_relative_path",
            "generation_contract_file_sha256",
            "generation_contract_sha256",
        },
        label="pending source",
    )
    if source["role"] != SOURCE_ROLE or source["confirmation_data_used"] is not False:
        raise ValueError("pending OP source must be non-confirmation development data")
    root = Path(str(source["draft_root"]))
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError("pending source.draft_root must be absolute and canonical")
    _positive_int(source["expected_rows"], label="pending source.expected_rows")
    for field in (
        "generation_receipt_sha256",
        "op_dataset_sha256",
        "generation_contract_file_sha256",
        "generation_contract_sha256",
    ):
        if not _is_sha256(source[field]):
            raise ValueError(f"pending source.{field} is invalid")
    for field in ("op_dataset_relative_path", "generation_contract_relative_path"):
        relative = source[field]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError(f"pending source.{field} must stay within draft_root")

    selection = _require_exact_keys(
        contract["selection"], {"selection_seed", "strata"}, label="pending selection"
    )
    _nonnegative_int(selection["selection_seed"], label="selection.selection_seed")
    strata = _validate_strata(selection["strata"])

    solver = _require_exact_keys(
        contract["solver"],
        {
            "adapter",
            "objective_semantics",
            "runner_code_sha256",
            "license_class",
            "artifact",
            "backend_identity",
            "base_parameters",
        },
        label="pending solver",
    )
    if solver["adapter"] != ADAPTER or solver["objective_semantics"] != OBJECTIVE_SEMANTICS:
        raise ValueError("pending OP exact adapter or objective semantics is invalid")
    if solver["license_class"] != "gurobi_academic_site_license_nonredistributable":
        raise ValueError("pending OP solver license class is invalid")
    if not _is_sha256(solver["runner_code_sha256"]):
        raise ValueError("pending solver.runner_code_sha256 is invalid")
    artifact = _require_exact_keys(
        solver["artifact"],
        {"path", "sha256", "size_bytes", "mode", "nlink", "immutable_gate_passed"},
        label="pending solver.artifact",
    )
    artifact_path = Path(str(artifact["path"]))
    if not artifact_path.is_absolute() or artifact_path.resolve() != artifact_path:
        raise ValueError("pending observed artifact path must be absolute and canonical")
    if not _is_sha256(artifact["sha256"]):
        raise ValueError("pending observed artifact SHA-256 is invalid")
    _positive_int(artifact["size_bytes"], label="pending artifact.size_bytes")
    if artifact["mode"] != "0644" or artifact["nlink"] != 1:
        raise ValueError("pending artifact must record the exact observed 0644/single-link gate")
    if artifact["immutable_gate_passed"] is not False:
        raise ValueError("pending artifact must fail the immutable-artifact gate")
    backend_identity = _require_exact_keys(
        solver["backend_identity"],
        {
            "backend_name",
            "native_version",
            "distribution_name",
            "module_origin",
            "installed_distribution_manifest_sha256",
        },
        label="pending solver.backend_identity",
    )
    for field, value in backend_identity.items():
        if not isinstance(value, str) or not value or "UNRESOLVED" in value.upper():
            raise ValueError(f"pending solver.backend_identity.{field} is unresolved")
    _validate_base_parameters(solver["base_parameters"])

    screen = _require_exact_keys(
        contract["production_screen"],
        {
            "target_rows",
            "require_all_sampled_optimal",
            "max_projected_native_runtime_seconds",
        },
        label="pending production_screen",
    )
    if screen["target_rows"] != TARGET_PRODUCTION_ROWS:
        raise ValueError("pending production screen must project exactly 1,000 rows")
    if screen["require_all_sampled_optimal"] is not True:
        raise ValueError("pending production screen must require every sampled row OPTIMAL")
    _positive_number(
        screen["max_projected_native_runtime_seconds"],
        label="pending production_screen.max_projected_native_runtime_seconds",
    )

    policy = _require_exact_keys(
        contract["execution_policy"],
        {
            "development_only",
            "confirmation_data_access_authorized",
            "confirmation_solving_authorized",
            "development_sweep_authorized",
            "production_authorization",
        },
        label="pending execution_policy",
    )
    if policy != {
        "development_only": True,
        "confirmation_data_access_authorized": False,
        "confirmation_solving_authorized": False,
        "development_sweep_authorized": False,
        "production_authorization": False,
    }:
        raise ValueError("pending OP execution policy is not fail-closed")

    resolution = _require_exact_keys(
        contract["remote_identity_resolution"],
        {
            "status",
            "observed_at_utc",
            "blocker",
            "finalization_spec_relative_path",
            "finalization_spec_file_sha256",
            "finalization_spec_sha256",
            "scientific_material_sha256",
        },
        label="pending remote_identity_resolution",
    )
    if resolution["status"] != PREFLIGHT_HOLD_REMOTE:
        raise ValueError("pending OP remote finalization status is not HOLD")
    if resolution["blocker"] != "audited_artifact_is_owner_writable_mode_0644":
        raise ValueError("pending OP immutable-artifact blocker changed")
    observed_at = resolution["observed_at_utc"]
    if not isinstance(observed_at, str) or not observed_at.endswith("Z"):
        raise ValueError("pending remote observation time is invalid")
    for field in (
        "finalization_spec_file_sha256",
        "finalization_spec_sha256",
        "scientific_material_sha256",
    ):
        if not _is_sha256(resolution[field]):
            raise ValueError(f"pending remote_identity_resolution.{field} is invalid")
    spec_relative = resolution["finalization_spec_relative_path"]
    if (
        not isinstance(spec_relative, str)
        or not spec_relative
        or Path(spec_relative).is_absolute()
        or ".." in Path(spec_relative).parts
    ):
        raise ValueError("pending remote finalization spec path escapes draft_root")
    if sha256_json(_scientific_material(contract)) != resolution["scientific_material_sha256"]:
        raise ValueError("pending OP scientific material hash mismatch")
    return source, strata, solver, screen


def _select_rows(
    rows: list[dict[str, object]],
    *,
    strata: list[dict[str, object]],
    seed: int,
) -> tuple[list[SelectedRow], dict[str, int]]:
    grouped: dict[str, list[tuple[int, dict[str, object]]]] = {
        str(stratum["stratum_id"]): [] for stratum in strata
    }
    for source_index, row in enumerate(rows):
        node_count = _op_node_count(row)
        matches = [
            stratum
            for stratum in strata
            if int(stratum["min_nodes"]) <= node_count <= int(stratum["max_nodes"])
        ]
        if len(matches) != 1:
            raise ValueError(
                "Every frozen source row must belong to exactly one preregistered "
                f"size stratum; source index {source_index} has {len(matches)} matches"
            )
        grouped[str(matches[0]["stratum_id"])].append((source_index, row))

    selected: list[SelectedRow] = []
    counts: dict[str, int] = {}
    for stratum in strata:
        identifier = str(stratum["stratum_id"])
        eligible = grouped[identifier]
        counts[identifier] = len(eligible)
        sample_count = int(stratum["sample_count"])
        if len(eligible) < sample_count:
            raise ValueError(
                f"stratum {identifier!r} requests {sample_count} rows but only "
                f"{len(eligible)} are frozen"
            )

        def selection_key(item: tuple[int, dict[str, object]]) -> tuple[str, int]:
            index, row = item
            row_hash = benchmark_row_sha256(row)
            material = f"{seed}:{identifier}:{row_hash}".encode("utf-8")
            return hashlib.sha256(material).hexdigest(), index

        for rank, (source_index, row) in enumerate(
            sorted(eligible, key=selection_key)[:sample_count]
        ):
            selected.append(
                SelectedRow(
                    stratum=copy.deepcopy(stratum),
                    source_index=source_index,
                    sample_rank=rank,
                    row=copy.deepcopy(row),
                )
            )
    return selected, counts


def preflight_pending_sweep(
    contract_path: str | Path,
    *,
    verifier_bundle: tuple[dict[str, Callable[..., dict]], dict[str, object]] | None = None,
) -> PendingSweepPreflight:
    """Validate a frozen development plan without importing or invoking Gurobi."""

    supplied = Path(contract_path).expanduser()
    contract = _load_object(supplied, label="pending OP scalability contract")
    _strict_file_identity(supplied, label="pending OP scalability contract")
    source, strata, solver, _screen = _validate_pending_contract_shape(contract)
    _verify_self_hash(contract, "contract_sha256", label="pending OP scalability contract")

    runner_identity = reference_runner._file_identity(reference_runner.__file__)
    if solver["runner_code_sha256"] != runner_identity["sha256"]:
        raise ValueError("pending OP contract does not bind the active exact runner")

    evaluators, verifier_identity = verifier_bundle or reference_runner._default_verifier_bundle()
    if "op" not in evaluators:
        raise ValueError("strict verifier bundle has no OP evaluator")
    if sha256_json(verifier_identity) != contract["strict_verifier_identity_sha256"]:
        raise ValueError("pending OP strict verifier identity mismatch")

    root = Path(str(source["draft_root"]))
    generation_contract_path = (
        root / str(source["generation_contract_relative_path"])
    ).resolve()
    try:
        generation_contract_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("development generation contract escapes draft_root") from exc
    generation_contract_identity = _strict_file_identity(
        generation_contract_path, label="frozen development generation contract"
    )
    if generation_contract_identity["sha256"] != source["generation_contract_file_sha256"]:
        raise ValueError("development generation contract file hash mismatch")
    generation_contract = _load_object(
        generation_contract_path, label="frozen development generation contract"
    )
    _verify_self_hash(
        generation_contract,
        "contract_sha256",
        label="frozen development generation contract",
    )
    if generation_contract["contract_sha256"] != source["generation_contract_sha256"]:
        raise ValueError("development generation contract self-hash mismatch")
    if generation_contract.get("confirmation_data_used") is not False:
        raise ValueError("development generation contract is not confirmation-disjoint")

    receipt_path = root / "generation_receipt.json"
    receipt_identity = _strict_file_identity(
        receipt_path, label="frozen development generation receipt"
    )
    if receipt_identity["sha256"] != source["generation_receipt_sha256"]:
        raise ValueError("development generation receipt file hash mismatch")
    dataset_path = (root / str(source["op_dataset_relative_path"])).resolve()
    try:
        dataset_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("OP development dataset escapes draft_root") from exc
    dataset_identity = _strict_file_identity(
        dataset_path, label="frozen development OP dataset"
    )
    if dataset_identity["sha256"] != source["op_dataset_sha256"]:
        raise ValueError("development OP dataset file hash mismatch")
    _root, rows, generation_receipt = reference_runner._load_and_validate_draft(
        root, "op", expected_instances=int(source["expected_rows"])
    )
    task_receipt = generation_receipt["tasks"]["op"]
    if task_receipt["relative_path"] != source["op_dataset_relative_path"]:
        raise ValueError("development OP dataset path disagrees with generation receipt")
    if task_receipt["file_sha256"] != source["op_dataset_sha256"]:
        raise ValueError("development OP dataset identity disagrees with generation receipt")

    resolution = contract["remote_identity_resolution"]
    assert isinstance(resolution, dict)
    spec_path = (root / str(resolution["finalization_spec_relative_path"])).resolve()
    try:
        spec_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("remote finalization spec escapes draft_root") from exc
    spec_identity = _strict_file_identity(spec_path, label="remote finalization spec")
    if spec_identity["sha256"] != resolution["finalization_spec_file_sha256"]:
        raise ValueError("remote finalization spec file hash mismatch")
    spec = _load_object(spec_path, label="remote finalization spec")
    _verify_self_hash(spec, "spec_sha256", label="remote finalization spec")
    if spec.get("spec_sha256") != resolution["finalization_spec_sha256"]:
        raise ValueError("remote finalization spec self-hash mismatch")
    if spec.get("protocol") != FINALIZATION_PROTOCOL:
        raise ValueError("remote finalization spec protocol mismatch")
    if spec.get("status") != PREFLIGHT_HOLD_REMOTE:
        raise ValueError("remote finalization spec is not fail-closed")
    if spec.get("scientific_material_sha256") != resolution["scientific_material_sha256"]:
        raise ValueError("remote finalization spec scientific material mismatch")
    confirmation_policy = spec.get("confirmation_policy")
    if confirmation_policy != {
        "access_confirmation_data": False,
        "solve_confirmation_data": False,
        "authorize_confirmation_reference_generation": False,
    }:
        raise ValueError("remote finalization spec does not forbid confirmation solving")

    selection = contract["selection"]
    assert isinstance(selection, dict)
    selected, counts = _select_rows(
        rows, strata=strata, seed=int(selection["selection_seed"])
    )
    return PendingSweepPreflight(
        contract_path=supplied.resolve(),
        contract=contract,
        source_rows=rows,
        selected_rows=selected,
        source_rows_by_stratum=counts,
        verifier_identity=verifier_identity,
        finalization_spec=spec,
    )


def preflight_sweep(
    contract_path: str | Path,
    *,
    verifier_bundle: tuple[dict[str, Callable[..., dict]], dict[str, object]] | None = None,
) -> dict[str, object]:
    """Return a non-solving readiness record for pending or executable contracts."""

    contract = _load_object(Path(contract_path).expanduser(), label="OP scalability contract")
    if contract.get("status") == PENDING_REMOTE_STATUS:
        pending = preflight_pending_sweep(contract_path, verifier_bundle=verifier_bundle)
        return {
            "status": PREFLIGHT_HOLD_REMOTE,
            "contract_sha256": pending.contract["contract_sha256"],
            "source_rows": len(pending.source_rows),
            "sample_rows": len(pending.selected_rows),
            "source_rows_by_stratum": pending.source_rows_by_stratum,
            "development_sweep_authorized": False,
            "confirmation_data_access_authorized": False,
            "confirmation_solving_authorized": False,
            "production_authorization": False,
            "required_next_step": "create_once_read_only_audited_artifact_then_finalize_contract",
        }
    prepared = prepare_sweep(contract_path, verifier_bundle=verifier_bundle)
    return {
        "status": "READY_FROZEN_DEVELOPMENT_SWEEP",
        "contract_sha256": prepared.contract["contract_sha256"],
        "source_rows": len(prepared.source_rows),
        "sample_rows": len(prepared.selected_rows),
        "source_rows_by_stratum": prepared.source_rows_by_stratum,
        "development_sweep_authorized": True,
        "confirmation_data_access_authorized": False,
        "confirmation_solving_authorized": False,
        "production_authorization": False,
    }


def prepare_sweep(
    contract_path: str | Path,
    *,
    verifier_bundle: tuple[dict[str, Callable[..., dict]], dict[str, object]] | None = None,
) -> PreparedSweep:
    supplied = Path(contract_path).expanduser()
    contract = _load_object(supplied, label="OP scalability contract")
    # The checked-in template must report its scientific HOLD reasons before
    # any immutability requirement.  No backend or source is touched on this
    # path.  Runnable contracts are then required to be frozen read-only.
    if contract.get("status") == HOLD_STATUS:
        _validate_contract_shape(contract)
        raise AssertionError("unreachable HOLD contract validation")
    if contract.get("status") == PENDING_REMOTE_STATUS:
        raise ValueError(
            "OP scalability contract passed local preflight but remains HOLD pending "
            "a create-once read-only audited solver artifact; use --preflight-only"
        )
    contract_identity = _strict_file_identity(supplied, label="OP scalability contract")
    canonical_path = supplied.resolve()
    source, strata, solver, _screen = _validate_contract_shape(contract)
    _verify_self_hash(contract, "contract_sha256", label="OP scalability contract")

    runner_identity = reference_runner._file_identity(reference_runner.__file__)
    if solver["runner_code_sha256"] != runner_identity["sha256"]:
        raise ValueError("OP scalability contract does not bind the active exact runner")

    artifact = solver["artifact"]
    assert isinstance(artifact, dict)
    artifact_identity = _strict_file_identity(
        Path(str(artifact["path"])), label="audited gurobipy artifact"
    )
    if artifact_identity["sha256"] != artifact["sha256"]:
        raise ValueError("audited gurobipy artifact hash does not match the contract")
    if int(artifact_identity["size_bytes"]) <= 0:
        raise ValueError("audited gurobipy artifact is empty")

    evaluators, verifier_identity = verifier_bundle or reference_runner._default_verifier_bundle()
    if "op" not in evaluators:
        raise ValueError("strict verifier bundle has no OP evaluator")
    if sha256_json(verifier_identity) != contract["strict_verifier_identity_sha256"]:
        raise ValueError("OP scalability contract strict verifier identity mismatch")

    root = Path(str(source["draft_root"]))
    receipt_path = root / "generation_receipt.json"
    receipt_identity = _strict_file_identity(
        receipt_path, label="frozen development generation receipt"
    )
    if receipt_identity["sha256"] != source["generation_receipt_sha256"]:
        raise ValueError("development generation receipt file hash mismatch")
    dataset_path = (root / str(source["op_dataset_relative_path"])).resolve()
    try:
        dataset_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("OP development dataset escapes draft_root") from exc
    dataset_identity = _strict_file_identity(
        dataset_path, label="frozen development OP dataset"
    )
    if dataset_identity["sha256"] != source["op_dataset_sha256"]:
        raise ValueError("development OP dataset file hash mismatch")

    _draft_root, rows, generation_receipt = reference_runner._load_and_validate_draft(
        root,
        "op",
        expected_instances=int(source["expected_rows"]),
    )
    task_receipt = generation_receipt["tasks"]["op"]
    if task_receipt["relative_path"] != source["op_dataset_relative_path"]:
        raise ValueError("development OP dataset path disagrees with generation receipt")
    if task_receipt["file_sha256"] != source["op_dataset_sha256"]:
        raise ValueError("development OP dataset identity disagrees with generation receipt")
    selection = contract["selection"]
    assert isinstance(selection, dict)
    selected, counts = _select_rows(
        rows,
        strata=strata,
        seed=int(selection["selection_seed"]),
    )
    return PreparedSweep(
        contract_path=canonical_path,
        contract=contract,
        contract_file_identity=contract_identity,
        source_rows=rows,
        selected_rows=selected,
        source_rows_by_stratum=counts,
        evaluator=evaluators["op"],
        verifier_identity=verifier_identity,
    )


def _verify_backend_identity(
    prepared: PreparedSweep,
    backend: DiagnosticOPBackend,
) -> dict[str, str]:
    observed = backend.probe_identity()
    if not isinstance(observed, dict):
        raise ValueError("OP diagnostic backend identity probe did not return an object")
    solver = prepared.contract["solver"]
    assert isinstance(solver, dict)
    expected = solver["backend_identity"]
    assert isinstance(expected, dict)
    for field, expected_value in expected.items():
        if observed.get(field) != expected_value:
            raise ValueError(
                f"OP diagnostic backend {field} mismatch: expected "
                f"{expected_value!r}, observed {observed.get(field)!r}"
            )
    return {field: str(observed[field]) for field in expected}


def _same_optional_number(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return math.isfinite(float(left)) and math.isfinite(float(right)) and float(left) == float(right)


def _validate_result(
    result: object,
    *,
    row: dict[str, object],
    evaluator: Callable[..., dict],
) -> tuple[reference_runner.NativeSolveResult, bool]:
    if not isinstance(result, reference_runner.NativeSolveResult):
        raise TypeError("OP diagnostic backend must return NativeSolveResult")
    if not isinstance(result.native_status, str) or not result.native_status:
        raise ValueError("OP diagnostic result has no native status")
    if not isinstance(result.raw_log, bytes) or not result.raw_log:
        raise ValueError("OP diagnostic result must retain a nonempty native solver log")
    runtime = result.native_wall_time_seconds
    if (
        isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or not math.isfinite(float(runtime))
        or float(runtime) < 0.0
    ):
        raise ValueError("OP diagnostic result has invalid native runtime")
    for field, value in (("incumbent", result.incumbent), ("best_bound", result.best_bound)):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"OP diagnostic result has invalid {field}")
    if result.gap is not None and (
        isinstance(result.gap, bool)
        or not isinstance(result.gap, (int, float))
        or not math.isfinite(float(result.gap))
        or float(result.gap) < 0.0
    ):
        raise ValueError("OP diagnostic result has invalid native gap")
    metadata = result.native_metadata
    if not isinstance(metadata, dict):
        raise ValueError("OP diagnostic result has no native metadata")
    solution_count = metadata.get("solution_count")
    if isinstance(solution_count, bool) or not isinstance(solution_count, int) or solution_count < 0:
        raise ValueError("OP diagnostic result has invalid native solution count")
    if metadata.get("gurobi_status") != result.native_status:
        raise ValueError("OP diagnostic metadata status disagrees with native status")
    if metadata.get("gurobi_status_code") != result.native_status_code:
        raise ValueError("OP diagnostic metadata status code disagrees with native status code")
    if not _same_optional_number(metadata.get("gurobi_best_bound"), result.best_bound):
        raise ValueError("OP diagnostic metadata bound disagrees with native bound")
    if not _same_optional_number(metadata.get("gurobi_gap"), result.gap):
        raise ValueError("OP diagnostic metadata gap disagrees with native gap")
    for field in ("node_count", "deterministic_work"):
        value = metadata.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"OP diagnostic metadata {field} is invalid")

    if solution_count == 0:
        if result.incumbent is not None or result.output.strip():
            raise ValueError("OP diagnostic result without a solution must not expose an incumbent")
    else:
        if result.incumbent is None or not result.output.strip():
            raise ValueError("OP diagnostic result with a solution must retain its incumbent output")
        metrics = evaluator(
            str(row.get("instruction", "")),
            str(row.get("input", "")),
            result.output,
            row.get("instance_json"),
            result.output,
        )
        if not isinstance(metrics, dict) or not metrics.get("is_feasible"):
            raise ValueError("strict OP verifier rejects the diagnostic incumbent")
        verified = metrics.get("objective_value")
        reference_runner._objective_matches(
            "op", verified, result.incumbent, label="OP diagnostic incumbent"
        )

    certified = result.native_status == "OPTIMAL"
    if certified:
        if result.native_status_code != 2:
            raise ValueError("OP diagnostic OPTIMAL status lacks Gurobi numeric code 2")
        if solution_count <= 0 or result.incumbent is None or result.best_bound is None:
            raise ValueError("OP diagnostic OPTIMAL status lacks incumbent or bound")
        reference_runner._objective_matches(
            "op", result.best_bound, result.incumbent, label="OP diagnostic OPTIMAL bound"
        )
        if result.gap != 0.0:
            raise ValueError("OP diagnostic OPTIMAL status has a nonzero native gap")
    return result, certified


def _parameters_for(prepared: PreparedSweep, selected: SelectedRow) -> dict[str, object]:
    solver = prepared.contract["solver"]
    assert isinstance(solver, dict)
    parameters = _validate_base_parameters(solver["base_parameters"])
    parameters["time_limit_seconds"] = float(selected.stratum["time_limit_seconds"])
    if selected.stratum["work_limit"] is not None:
        parameters["work_limit"] = float(selected.stratum["work_limit"])
    if selected.stratum["node_limit"] is not None:
        parameters["node_limit"] = int(selected.stratum["node_limit"])
    return parameters


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty runtime sample")
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return float(ordered[index])


def _runtime_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "sum_seconds": float(sum(values)),
        "mean_seconds": float(sum(values) / len(values)),
        "min_seconds": float(min(values)),
        "median_seconds": _quantile(values, 0.5),
        "p90_seconds": _quantile(values, 0.9),
        "p95_seconds": _quantile(values, 0.95),
        "max_seconds": float(max(values)),
    }


def _file_descriptor(path: Path, *, relative_to: Path) -> dict[str, object]:
    metadata = path.stat()
    return {
        "relative_path": path.relative_to(relative_to).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": int(metadata.st_size),
    }


def _write_json_new(path: Path, value: object) -> None:
    reference_runner._write_json_new(path, value)


def _write_bytes_new(path: Path, value: bytes) -> None:
    reference_runner._write_bytes_new(path, value)


def run_sweep(
    *,
    contract_path: str | Path,
    output_dir: str | Path,
    backend: DiagnosticOPBackend | None = None,
    verifier_bundle: tuple[dict[str, Callable[..., dict]], dict[str, object]] | None = None,
) -> dict[str, object]:
    """Run one immutable-plan development sweep into a new evidence directory."""

    prepared = prepare_sweep(contract_path, verifier_bundle=verifier_bundle)
    selected_backend = backend or reference_runner.GurobiOPBackend()
    backend_identity = _verify_backend_identity(prepared, selected_backend)
    completed: dict[str, object] = {}

    def write(destination: Path) -> None:
        nonlocal completed
        _write_bytes_new(destination / "sweep_contract.json", prepared.contract_path.read_bytes())
        records: list[dict[str, object]] = []
        for selected in prepared.selected_rows:
            parameters = _parameters_for(prepared, selected)
            measured_started = time.monotonic()
            raw_result = selected_backend.solve_diagnostic(
                copy.deepcopy(selected.row), parameters=copy.deepcopy(parameters)
            )
            measured_wall = time.monotonic() - measured_started
            if not math.isfinite(measured_wall) or measured_wall < 0.0:
                raise ValueError("measured OP diagnostic wall time is invalid")
            result, certified = _validate_result(
                raw_result,
                row=selected.row,
                evaluator=prepared.evaluator,
            )
            row_id = str(selected.row["benchmark_row_id"])
            log_path = destination / "native_logs" / f"{row_id}.log"
            _write_bytes_new(log_path, result.raw_log)
            metadata = result.native_metadata
            assert isinstance(metadata, dict)
            record: dict[str, object] = {
                "stratum_id": selected.stratum["stratum_id"],
                "source_index": selected.source_index,
                "sample_rank": selected.sample_rank,
                "benchmark_row_id": row_id,
                "node_count": _op_node_count(selected.row),
                "semantic_instance_sha256": semantic_instance_sha256(
                    "op", selected.row, require_structured=True
                ),
                "benchmark_row_sha256": benchmark_row_sha256(selected.row),
                "solver_parameters": parameters,
                "solver_parameters_sha256": sha256_json(parameters),
                "native_status": result.native_status,
                "native_status_code": result.native_status_code,
                "native_solution_count": metadata["solution_count"],
                "native_incumbent": result.incumbent,
                "native_best_bound": result.best_bound,
                "native_gap": result.gap,
                "native_output": result.output,
                "native_output_sha256": hashlib.sha256(
                    result.output.encode("utf-8")
                ).hexdigest(),
                "native_runtime_seconds": float(result.native_wall_time_seconds),
                "measured_wall_time_seconds": float(measured_wall),
                "native_deterministic_work": float(metadata["deterministic_work"]),
                "native_search_nodes": float(metadata["node_count"]),
                "native_metadata": copy.deepcopy(metadata),
                "native_log": _file_descriptor(log_path, relative_to=destination),
                "classification": (
                    EXACT_CLASSIFICATION if certified else NONEXACT_CLASSIFICATION
                ),
                "exact_reference_usable": bool(certified),
            }
            record["record_sha256"] = hashlib.sha256(
                _canonical_json_bytes(record)
            ).hexdigest()
            records.append(record)

        _write_json_new(destination / "details.json", records)
        strata_contract = prepared.contract["selection"]
        assert isinstance(strata_contract, dict)
        strata = strata_contract["strata"]
        assert isinstance(strata, list)
        by_stratum: dict[str, dict[str, object]] = {}
        for raw_stratum in strata:
            assert isinstance(raw_stratum, dict)
            identifier = str(raw_stratum["stratum_id"])
            subset = [row for row in records if row["stratum_id"] == identifier]
            runtimes = [float(row["native_runtime_seconds"]) for row in subset]
            by_stratum[identifier] = {
                "source_rows": prepared.source_rows_by_stratum[identifier],
                "sample_rows": len(subset),
                "native_status_counts": dict(
                    sorted(Counter(str(row["native_status"]) for row in subset).items())
                ),
                "certified_optimal_rows": sum(
                    bool(row["exact_reference_usable"]) for row in subset
                ),
                "runtime": _runtime_summary(runtimes),
            }

        source_total = len(prepared.source_rows)
        weighted_mean = sum(
            prepared.source_rows_by_stratum[identifier]
            * float(summary["runtime"]["mean_seconds"])
            for identifier, summary in by_stratum.items()
        ) / source_total
        projected_seconds = weighted_mean * TARGET_PRODUCTION_ROWS
        certified_count = sum(bool(row["exact_reference_usable"]) for row in records)
        all_optimal = certified_count == len(records)
        screen_contract = prepared.contract["production_screen"]
        assert isinstance(screen_contract, dict)
        runtime_budget = float(screen_contract["max_projected_native_runtime_seconds"])
        if not all_optimal:
            screen_status = HOLD_NONOPTIMAL
        elif projected_seconds > runtime_budget:
            screen_status = HOLD_RUNTIME
        else:
            screen_status = PASS_SCREEN

        details_path = destination / "details.json"
        contract_snapshot_path = destination / "sweep_contract.json"
        receipt: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "protocol": RECEIPT_PROTOCOL,
            "status": "complete_development_scalability_evidence",
            "publishable": False,
            "confirmation_data_used": False,
            "production_authorization": False,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "contract_sha256": prepared.contract["contract_sha256"],
            "contract_file_identity": prepared.contract_file_identity,
            "contract_snapshot": _file_descriptor(
                contract_snapshot_path, relative_to=destination
            ),
            "source": copy.deepcopy(prepared.contract["source"]),
            "solver": copy.deepcopy(prepared.contract["solver"]),
            "observed_backend_identity": backend_identity,
            "strict_verifier_identity": prepared.verifier_identity,
            "strict_verifier_identity_sha256": sha256_json(
                prepared.verifier_identity
            ),
            "selection": copy.deepcopy(prepared.contract["selection"]),
            "sample_rows": len(records),
            "certified_optimal_rows": certified_count,
            "noncertified_rows": len(records) - certified_count,
            "native_status_counts": dict(
                sorted(Counter(str(row["native_status"]) for row in records).items())
            ),
            "runtime": _runtime_summary(
                [float(row["native_runtime_seconds"]) for row in records]
            ),
            "strata": by_stratum,
            "production_projection": {
                "target_rows": TARGET_PRODUCTION_ROWS,
                "method": "source_stratum_frequency_weighted_sample_means",
                "descriptive_only_no_confidence_claim": True,
                "projected_native_runtime_seconds": projected_seconds,
                "projected_native_runtime_hours": projected_seconds / 3600.0,
                "preregistered_max_projected_native_runtime_seconds": runtime_budget,
            },
            "production_screen": {
                "status": screen_status,
                "all_sampled_rows_native_optimal": all_optimal,
                "production_authorization": False,
                "required_next_step": (
                    "author_review_then_new_frozen_production_contract"
                    if screen_status == PASS_SCREEN
                    else "keep_op_exact_production_on_hold"
                ),
            },
            "details_artifact": _file_descriptor(details_path, relative_to=destination),
            "ordered_record_sha256": [row["record_sha256"] for row in records],
        }
        receipt["records_sha256"] = sha256_json(receipt["ordered_record_sha256"])
        receipt["receipt_sha256"] = hashlib.sha256(
            _canonical_json_bytes(receipt)
        ).hexdigest()
        _write_json_new(destination / "receipt.json", receipt)
        completed = receipt

    reference_runner._atomic_directory(output_dir, write)
    return completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New output directory; required unless --validate-only/--preflight-only is used.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the frozen contract and source without importing Gurobi or solving.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate either a runnable contract or a fail-closed pending remote-"
            "artifact plan without solving."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.preflight_only:
            print(json.dumps(preflight_sweep(args.contract), sort_keys=True))
            return 0
        prepared = prepare_sweep(args.contract)
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "status": "VALID_FROZEN_DEVELOPMENT_SWEEP",
                        "contract_sha256": prepared.contract["contract_sha256"],
                        "sample_rows": len(prepared.selected_rows),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.output_dir is None:
            raise ValueError("--output-dir is required for an actual development sweep")
        receipt = run_sweep(contract_path=args.contract, output_dir=args.output_dir)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    except (FileExistsError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[OPScalabilitySweepError] {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
