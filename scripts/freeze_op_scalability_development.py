#!/usr/bin/env python3
"""Create the immutable, OP-only development scalability preregistration.

This utility deliberately generates no reference labels and never imports
Gurobi.  It calls the released-code-compatible ``_op_row`` implementation with
one frozen Python/NumPy seed, rejects semantic duplicates, and atomically
creates the generation contract, draft, receipt, remote-finalization spec, and
fail-closed sweep contract.  Existing destinations are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np

from clasp.data.benchmark import (
    benchmark_row_sha256,
    semantic_instance_sha256,
    sha256_json,
)

try:  # Direct execution places ``scripts`` on sys.path.
    import generate_iid_benchmark as iid_generator
    import run_op_scalability_sweep as sweep
    import run_reference_evidence as reference_runner
except ModuleNotFoundError:  # Import via ``scripts.freeze_op_scalability_development``.
    from scripts import generate_iid_benchmark as iid_generator  # type: ignore
    from scripts import run_op_scalability_sweep as sweep  # type: ignore
    from scripts import run_reference_evidence as reference_runner  # type: ignore


SCHEMA_VERSION = 1
GENERATION_PROTOCOL = "clasp-op-scalability-development-generation-v1"
GENERATION_STATUS = "frozen_op_only_unlabeled_development_generation"
COUNT = 256
SEED = 2_026_082_601
SELECTION_SEED = SEED
DATASET_RELATIVE_PATH = "op/eval/test.json"
GENERATION_CONTRACT_RELATIVE_PATH = "generation_contract.json"
FINALIZATION_SPEC_RELATIVE_PATH = "remote_finalization_spec.json"
REQUIRED_NUMPY_VERSION = "2.3.2"
REQUIRED_NETWORKX_VERSION = "3.5"
PRODUCTION_RUNTIME_CEILING_SECONDS = 172_800.0
REMOTE_OBSERVED_ARTIFACT = {
    "path": (
        "/lustre/fswork/projects/rech/vas/uag72kz/clasp/reference_solvers/"
        "gurobi_13.0.1_audit/wheels/"
        "gurobipy-13.0.1-cp310-cp310-manylinux2014_x86_64."
        "manylinux_2_17_x86_64.whl"
    ),
    "sha256": "7fc1b5ff46530aee391ab0f616620a1399abea0778b25e374b37f7ed222d5dfb",
    "size_bytes": 15_012_230,
    "mode": "0644",
    "nlink": 1,
    "immutable_gate_passed": False,
}
REMOTE_BACKEND_IDENTITY = {
    "backend_name": "gurobi",
    "native_version": "13.0.1",
    "distribution_name": "gurobipy",
    "module_origin": (
        "/lustre/fswork/projects/rech/vas/uag72kz/clasp/reference_solvers/"
        "gurobi_13.0.1_audit/venv/lib/python3.10/site-packages/"
        "gurobipy/__init__.py"
    ),
    "installed_distribution_manifest_sha256": (
        "3d81f9826721aff63aa8ceb7ce97d8ff25df6bd2c7de40490bee48d09eb529cb"
    ),
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _seal(value: dict[str, object], field: str) -> dict[str, object]:
    sealed = dict(value)
    sealed.pop(field, None)
    sealed[field] = hashlib.sha256(_canonical_json_bytes(sealed)).hexdigest()
    return sealed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_descriptor(path: Path) -> dict[str, object]:
    return {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_pretty_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _strict_verifier_identity(repository_root: Path) -> dict[str, object]:
    implementations: list[dict[str, object]] = []
    for relative in ("rewards/co_tasks.py", "entrypoints/evaluate.py"):
        path = repository_root / "src" / "clasp" / relative
        identity = _file_descriptor(path)
        implementations.append(
            {
                "path": relative,
                "sha256": identity["sha256"],
                "size": identity["size_bytes"],
            }
        )
    return {
        "name": "clasp_strict_shared_verifier",
        "schema_version": 1,
        "implementation_manifest_sha256": hashlib.sha256(
            _canonical_json_bytes(implementations)
        ).hexdigest(),
        "implementations": implementations,
        "tasks": ["cvrp", "jssp", "mis", "op", "tsp"],
    }


def _generation_contract(
    *,
    repository_root: Path,
    frozen_at_utc: str,
) -> dict[str, object]:
    generator_path = repository_root / "scripts" / "generate_iid_benchmark.py"
    builder_path = Path(__file__).resolve()
    return _seal(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": GENERATION_PROTOCOL,
            "status": GENERATION_STATUS,
            "frozen_at_utc": frozen_at_utc,
            "purpose": "development_only_op_exact_solver_scalability",
            "confirmation_data_used": False,
            "contains_references": False,
            "publishable": False,
            "task": "op",
            "instance_count": COUNT,
            "seed": SEED,
            "uniqueness": "semantic_instance_sha256_rejection_sampling",
            "generator": {
                "callable": "scripts.generate_iid_benchmark._op_row",
                "compatibility_claim": (
                    "released-code-compatible deterministic interpretation; not a "
                    "byte-identical historical reconstruction"
                ),
                "source_sha256": _sha256_file(generator_path),
                "construction_code_sha256": _sha256_file(builder_path),
                "op_semantics": {
                    "node_count": "DiscreteUniform[10,100]_inclusive",
                    "coordinate_distribution": (
                        "uniform_choice[uniform,gaussian_mixture_2_5,"
                        "gaussian_mixture_3_10]"
                    ),
                    "prize": "DiscreteUniform[1,10]_depot_overwritten_zero",
                    "route_limit": (
                        "round_1dp(Uniform[0.5*deterministic_NN_cycle,"
                        "0.7*deterministic_NN_cycle])"
                    ),
                },
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "networkx": nx.__version__,
                "required_numpy": REQUIRED_NUMPY_VERSION,
                "required_networkx": REQUIRED_NETWORKX_VERSION,
            },
            "output": {
                "relative_path": DATASET_RELATIVE_PATH,
                "reference_status": "pending",
                "canonical_row_ids": "op-%06d",
                "frozen_read_only_mode": "0440",
            },
        },
        "contract_sha256",
    )


def _generate_rows() -> tuple[list[dict[str, object]], int, dict[str, int]]:
    py_rng = random.Random(SEED)
    np_rng = np.random.RandomState(SEED)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    distributions: dict[str, int] = {}
    attempts = 0
    while len(rows) < COUNT:
        attempts += 1
        row, distribution = iid_generator._op_row(py_rng, np_rng)
        digest = semantic_instance_sha256("op", row, require_structured=True)
        if digest in seen:
            continue
        seen.add(digest)
        index = len(rows)
        row["benchmark_row_id"] = f"op-{index:06d}"
        row["instance_sha256"] = digest
        row["semantic_instance_sha256"] = digest
        row["reference_status"] = "pending"
        row["generator_distribution"] = distribution
        rows.append(row)
        distributions[distribution] = distributions.get(distribution, 0) + 1
    return rows, attempts, distributions


def _strata() -> list[dict[str, object]]:
    return [
        {
            "stratum_id": "nodes_10_30",
            "min_nodes": 10,
            "max_nodes": 30,
            "sample_count": 8,
            "time_limit_seconds": 60.0,
            "work_limit": None,
            "node_limit": None,
        },
        {
            "stratum_id": "nodes_31_60",
            "min_nodes": 31,
            "max_nodes": 60,
            "sample_count": 8,
            "time_limit_seconds": 300.0,
            "work_limit": None,
            "node_limit": None,
        },
        {
            "stratum_id": "nodes_61_80",
            "min_nodes": 61,
            "max_nodes": 80,
            "sample_count": 5,
            "time_limit_seconds": 900.0,
            "work_limit": None,
            "node_limit": None,
        },
        {
            "stratum_id": "nodes_81_100",
            "min_nodes": 81,
            "max_nodes": 100,
            "sample_count": 3,
            "time_limit_seconds": 1800.0,
            "work_limit": None,
            "node_limit": None,
        },
    ]


def freeze_development_plan(
    *,
    output_dir: str | Path,
    frozen_at_utc: str,
    remote_observed_at_utc: str,
    after_dataset_write: Callable[[Path], None] | None = None,
) -> dict[str, object]:
    """Atomically create the exact development draft and fail-closed plan."""

    if not frozen_at_utc.endswith("Z") or not remote_observed_at_utc.endswith("Z"):
        raise ValueError("freeze and observation times must be explicit UTC timestamps")
    if np.__version__ != REQUIRED_NUMPY_VERSION or nx.__version__ != REQUIRED_NETWORKX_VERSION:
        raise RuntimeError(
            "OP development generation requires numpy==2.3.2 and networkx==3.5; "
            f"observed numpy=={np.__version__}, networkx=={nx.__version__}"
        )
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite frozen OP development plan: {target}")
    repository_root = Path(__file__).resolve().parents[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    completed: dict[str, object] = {}
    try:
        generation_contract = _generation_contract(
            repository_root=repository_root, frozen_at_utc=frozen_at_utc
        )
        generation_contract_path = temporary / GENERATION_CONTRACT_RELATIVE_PATH
        _write_new(generation_contract_path, generation_contract)

        rows, attempts, distributions = _generate_rows()
        dataset_path = temporary / DATASET_RELATIVE_PATH
        _write_new(dataset_path, rows)
        if after_dataset_write is not None:
            after_dataset_write(dataset_path)
        semantic_hashes = [str(row["semantic_instance_sha256"]) for row in rows]
        row_hashes = [benchmark_row_sha256(row) for row in rows]
        generation_receipt = _seal(
            {
                "schema_version": 2,
                "status": "instances_generated",
                "created_at_utc": frozen_at_utc,
                "purpose": "development_only_op_exact_solver_scalability",
                "confirmation_data_used": False,
                "generator_contract_sha256": generation_contract["contract_sha256"],
                "generator_contract_file_sha256": _sha256_file(generation_contract_path),
                "generator_code_sha256": _sha256_file(
                    repository_root / "scripts" / "generate_iid_benchmark.py"
                ),
                "generation_builder_code_sha256": _sha256_file(Path(__file__).resolve()),
                "runtime_audit": generation_contract["runtime"],
                "tasks": {
                    "op": {
                        "rows": len(rows),
                        "attempts": attempts,
                        "seed": SEED,
                        "distribution_counts": distributions,
                        "relative_path": DATASET_RELATIVE_PATH,
                        "file_sha256": _sha256_file(dataset_path),
                        "ordered_instance_sha256": semantic_hashes,
                        "ordered_instance_hashes_sha256": sha256_json(semantic_hashes),
                        "ordered_semantic_instance_sha256": semantic_hashes,
                        "semantic_instance_hashes_sha256": sha256_json(semantic_hashes),
                        "ordered_benchmark_row_sha256": row_hashes,
                        "benchmark_row_hashes_sha256": sha256_json(row_hashes),
                    }
                },
                "contains_references": False,
                "publishable": False,
            },
            "receipt_sha256",
        )
        generation_receipt_path = temporary / "generation_receipt.json"
        _write_new(generation_receipt_path, generation_receipt)

        verifier_identity = _strict_verifier_identity(repository_root)
        source = {
            "role": sweep.SOURCE_ROLE,
            "confirmation_data_used": False,
            "draft_root": str(target),
            "expected_rows": COUNT,
            "generation_receipt_sha256": _sha256_file(generation_receipt_path),
            "op_dataset_relative_path": DATASET_RELATIVE_PATH,
            "op_dataset_sha256": _sha256_file(dataset_path),
            "generation_contract_relative_path": GENERATION_CONTRACT_RELATIVE_PATH,
            "generation_contract_file_sha256": _sha256_file(generation_contract_path),
            "generation_contract_sha256": generation_contract["contract_sha256"],
        }
        pending_base: dict[str, object] = {
            "schema_version": sweep.SCHEMA_VERSION,
            "protocol": sweep.CONTRACT_PROTOCOL,
            "status": sweep.PENDING_REMOTE_STATUS,
            "frozen_at_utc": frozen_at_utc,
            "purpose": "development_only_production_feasibility_screen",
            "source": source,
            "selection": {"selection_seed": SELECTION_SEED, "strata": _strata()},
            "solver": {
                "adapter": reference_runner.OP_EXACT_ADAPTER,
                "objective_semantics": reference_runner.OBJECTIVE_SEMANTICS["op"],
                "runner_code_sha256": _sha256_file(
                    repository_root / "scripts" / "run_reference_evidence.py"
                ),
                "license_class": "gurobi_academic_site_license_nonredistributable",
                "artifact": dict(REMOTE_OBSERVED_ARTIFACT),
                "backend_identity": dict(REMOTE_BACKEND_IDENTITY),
                "base_parameters": dict(reference_runner.OP_EXACT_REQUIRED_PARAMETERS),
            },
            "strict_verifier_identity_sha256": sha256_json(verifier_identity),
            "production_screen": {
                "target_rows": 1_000,
                "require_all_sampled_optimal": True,
                "max_projected_native_runtime_seconds": (
                    PRODUCTION_RUNTIME_CEILING_SECONDS
                ),
            },
            "execution_policy": {
                "development_only": True,
                "confirmation_data_access_authorized": False,
                "confirmation_solving_authorized": False,
                "development_sweep_authorized": False,
                "production_authorization": False,
            },
        }
        scientific_material_sha256 = sha256_json(sweep._scientific_material(pending_base))
        finalization_spec = _seal(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": sweep.FINALIZATION_PROTOCOL,
                "status": sweep.PREFLIGHT_HOLD_REMOTE,
                "frozen_at_utc": frozen_at_utc,
                "scientific_material_sha256": scientific_material_sha256,
                "scope": "development_op_scalability_only",
                "confirmation_policy": {
                    "access_confirmation_data": False,
                    "solve_confirmation_data": False,
                    "authorize_confirmation_reference_generation": False,
                },
                "observed_remote_identity": {
                    "host": "jean-zay.idris.fr",
                    "project_group": "vas",
                    "observed_at_utc": remote_observed_at_utc,
                    "probe_mode": "read_only_no_remote_files_created",
                    "artifact": dict(REMOTE_OBSERVED_ARTIFACT),
                    "backend_identity": dict(REMOTE_BACKEND_IDENTITY),
                    "installed_manifest_entries": 33,
                    "missing_distribution_files": [],
                },
                "blocking_conditions": [
                    "observed audited wheel is owner-writable mode 0644",
                    "executable sweep contracts require a regular singly-linked read-only artifact",
                    "no development solve is authorized until a finalized contract passes prepare_sweep",
                ],
                "required_remote_finalization": {
                    "stage_artifact_create_once": True,
                    "source_artifact_path": REMOTE_OBSERVED_ARTIFACT["path"],
                    "required_artifact_sha256": REMOTE_OBSERVED_ARTIFACT["sha256"],
                    "required_artifact_size_bytes": REMOTE_OBSERVED_ARTIFACT["size_bytes"],
                    "required_final_mode": "0440",
                    "required_final_nlink": 1,
                    "reprobe_backend_identity_exact_match": True,
                    "final_contract_create_once": True,
                    "final_contract_status": sweep.CONTRACT_STATUS,
                    "rerun_local_source_and_verifier_preflight": True,
                    "require_prepare_sweep_before_any_solver_call": True,
                },
                "allowed_finalization_changes": [
                    "status_to_frozen_before_development_sweep",
                    "artifact_path_to_new_read_only_exact_hash_copy",
                    "artifact_object_to_exact_runnable_schema_path_and_sha256",
                    "drop_pending_only_source_generation_contract_fields",
                    "drop_execution_policy_and_remote_identity_resolution",
                    "recompute_contract_sha256",
                ],
                "forbidden_changes": [
                    "source_rows_or_hashes",
                    "selection_seed_or_strata",
                    "sample_counts_or_resource_limits",
                    "solver_backend_identity_or_base_parameters",
                    "runner_or_verifier_hashes",
                    "production_projection_ceiling",
                    "any_confirmation_data_access_or_solving",
                ],
            },
            "spec_sha256",
        )
        finalization_spec_path = temporary / FINALIZATION_SPEC_RELATIVE_PATH
        _write_new(finalization_spec_path, finalization_spec)

        pending_base["remote_identity_resolution"] = {
            "status": sweep.PREFLIGHT_HOLD_REMOTE,
            "observed_at_utc": remote_observed_at_utc,
            "blocker": "audited_artifact_is_owner_writable_mode_0644",
            "finalization_spec_relative_path": FINALIZATION_SPEC_RELATIVE_PATH,
            "finalization_spec_file_sha256": _sha256_file(finalization_spec_path),
            "finalization_spec_sha256": finalization_spec["spec_sha256"],
            "scientific_material_sha256": scientific_material_sha256,
        }
        pending_contract = sweep.seal_contract(pending_base)
        pending_contract_path = temporary / "sweep_contract.pending_remote_artifact.json"
        _write_new(pending_contract_path, pending_contract)

        readme = {
            "status": sweep.PREFLIGHT_HOLD_REMOTE,
            "development_only": True,
            "confirmation_data_used": False,
            "confirmation_solving_authorized": False,
            "draft_rows": COUNT,
            "sample_rows": sum(int(row["sample_count"]) for row in _strata()),
            "next_step": (
                "stage the exact audited wheel as a create-once mode-0440 file, "
                "finalize a new runnable contract, and rerun preflight"
            ),
        }
        _write_new(temporary / "STATUS.json", readme)

        for path in temporary.rglob("*"):
            if path.is_file():
                path.chmod(0o440)
        for path in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            path.chmod(0o550)
        temporary.chmod(0o550)
        os.replace(temporary, target)
        completed = {
            "status": sweep.PREFLIGHT_HOLD_REMOTE,
            "output_dir": str(target),
            "generation_contract_sha256": generation_contract["contract_sha256"],
            "generation_receipt_sha256": generation_receipt["receipt_sha256"],
            "sweep_contract_sha256": pending_contract["contract_sha256"],
            "source_rows": COUNT,
            "sample_rows": sum(int(row["sample_count"]) for row in _strata()),
            "confirmation_solving_authorized": False,
        }
    except BaseException:
        if temporary.exists():
            for path in [temporary, *temporary.rglob("*")]:
                try:
                    path.chmod(0o700 if path.is_dir() else 0o600)
                except OSError:
                    pass
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-at-utc", required=True)
    parser.add_argument("--remote-observed-at-utc", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = freeze_development_plan(
            output_dir=args.output_dir,
            frozen_at_utc=args.frozen_at_utc,
            remote_observed_at_utc=args.remote_observed_at_utc,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"[OPScalabilityFreezeError] {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
