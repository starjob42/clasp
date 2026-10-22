from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from clasp.data.benchmark import (
    benchmark_row_sha256,
    semantic_instance_sha256,
    sha256_json,
)
from clasp.entrypoints.evaluate import TASK_EVALUATORS


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_reference_evidence", SCRIPTS / "run_reference_evidence.py"
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)

SWEEP_SPEC = importlib.util.spec_from_file_location(
    "run_op_scalability_sweep", SCRIPTS / "run_op_scalability_sweep.py"
)
assert SWEEP_SPEC is not None and SWEEP_SPEC.loader is not None
sweep = importlib.util.module_from_spec(SWEEP_SPEC)
sys.modules[SWEEP_SPEC.name] = sweep
SWEEP_SPEC.loader.exec_module(sweep)

FREEZER_SPEC = importlib.util.spec_from_file_location(
    "freeze_op_scalability_development",
    SCRIPTS / "freeze_op_scalability_development.py",
)
assert FREEZER_SPEC is not None and FREEZER_SPEC.loader is not None
freezer = importlib.util.module_from_spec(FREEZER_SPEC)
sys.modules[FREEZER_SPEC.name] = freezer
FREEZER_SPEC.loader.exec_module(freezer)

VERIFIER_IDENTITY = {"name": "test-op-scalability-verifier", "schema_version": 1}
VERIFIER_BUNDLE = (dict(TASK_EVALUATORS), VERIFIER_IDENTITY)
BACKEND_IDENTITY = {
    "backend_name": "gurobi",
    "native_version": "13.0.1",
    "distribution_name": "gurobipy",
    "module_origin": "/frozen/test/gurobipy/__init__.py",
    "installed_distribution_manifest_sha256": "7" * 64,
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row(index: int, node_count: int) -> dict[str, object]:
    coordinates = [[node, 0] for node in range(node_count)]
    prizes = [0, 1, *([0] * (node_count - 2))]
    row: dict[str, object] = {
        "benchmark_row_id": f"op-{index:06d}",
        "num_nodes": str(node_count),
        "instruction": "Solve this open Orienteering Problem from depot 0.",
        "input": "Nodes lie on a line; only node 1 has positive prize.",
        "instance_json": [coordinates, prizes, 100.0],
        "reference_status": "pending",
        "generator_distribution": "test_iid_like",
    }
    digest = semantic_instance_sha256("op", row, require_structured=True)
    row["instance_sha256"] = digest
    row["semantic_instance_sha256"] = digest
    return row


def _draft(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = (tmp_path / "development-op-draft").resolve()
    dataset = root / "op" / "eval" / "test.json"
    rows = [_row(index, size) for index, size in enumerate((3, 4, 5, 6))]
    _write_json(dataset, rows)
    semantic = [str(row["semantic_instance_sha256"]) for row in rows]
    row_hashes = [benchmark_row_sha256(row) for row in rows]
    receipt = runner._self_hash(
        {
            "schema_version": 2,
            "status": "instances_generated",
            "generator_contract_sha256": "a" * 64,
            "generator_code_sha256": "b" * 64,
            "runtime_audit": {"test": True},
            "tasks": {
                "op": {
                    "rows": len(rows),
                    "relative_path": "op/eval/test.json",
                    "file_sha256": runner._file_identity(dataset)["sha256"],
                    "ordered_instance_sha256": semantic,
                    "ordered_instance_hashes_sha256": sha256_json(semantic),
                    "ordered_semantic_instance_sha256": semantic,
                    "semantic_instance_hashes_sha256": sha256_json(semantic),
                    "ordered_benchmark_row_sha256": row_hashes,
                    "benchmark_row_hashes_sha256": sha256_json(row_hashes),
                }
            },
            "contains_references": False,
            "publishable": False,
        },
        "receipt_sha256",
    )
    generation_receipt = root / "generation_receipt.json"
    _write_json(generation_receipt, receipt)
    dataset.chmod(0o440)
    generation_receipt.chmod(0o440)
    return root, dataset, generation_receipt


def _contract(tmp_path: Path) -> Path:
    draft_root, dataset, generation_receipt = _draft(tmp_path)
    artifact = (tmp_path / "gurobipy-audited.whl").resolve()
    artifact.write_bytes(b"test-only audited gurobipy artifact")
    artifact.chmod(0o440)
    contract = sweep.seal_contract(
        {
            "schema_version": 1,
            "protocol": sweep.CONTRACT_PROTOCOL,
            "status": sweep.CONTRACT_STATUS,
            "frozen_at_utc": "2026-08-26T10:00:00Z",
            "purpose": "development_only_production_feasibility_screen",
            "source": {
                "role": sweep.SOURCE_ROLE,
                "confirmation_data_used": False,
                "draft_root": str(draft_root),
                "expected_rows": 4,
                "generation_receipt_sha256": runner._file_identity(generation_receipt)[
                    "sha256"
                ],
                "op_dataset_relative_path": "op/eval/test.json",
                "op_dataset_sha256": runner._file_identity(dataset)["sha256"],
            },
            "selection": {
                "selection_seed": 271828,
                "strata": [
                    {
                        "stratum_id": "small",
                        "min_nodes": 3,
                        "max_nodes": 4,
                        "sample_count": 1,
                        "time_limit_seconds": 10.0,
                        "work_limit": None,
                        "node_limit": None,
                    },
                    {
                        "stratum_id": "large",
                        "min_nodes": 5,
                        "max_nodes": 6,
                        "sample_count": 1,
                        "time_limit_seconds": 20.0,
                        "work_limit": 30.0,
                        "node_limit": 1000,
                    },
                ],
            },
            "solver": {
                "adapter": runner.OP_EXACT_ADAPTER,
                "objective_semantics": runner.OBJECTIVE_SEMANTICS["op"],
                "runner_code_sha256": runner._file_identity(
                    SCRIPTS / "run_reference_evidence.py"
                )["sha256"],
                "license_class": "gurobi_academic_site_license_nonredistributable",
                "artifact": {
                    "path": str(artifact),
                    "sha256": runner._file_identity(artifact)["sha256"],
                },
                "backend_identity": dict(BACKEND_IDENTITY),
                "base_parameters": dict(runner.OP_EXACT_REQUIRED_PARAMETERS),
            },
            "strict_verifier_identity_sha256": sha256_json(VERIFIER_IDENTITY),
            "production_screen": {
                "target_rows": 1000,
                "require_all_sampled_optimal": True,
                "max_projected_native_runtime_seconds": 2000.0,
            },
        }
    )
    path = (tmp_path / "op-scalability-contract.json").resolve()
    _write_json(path, contract)
    path.chmod(0o440)
    return path


def _rewrite_contract(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    value = sweep.seal_contract(value)
    path.chmod(0o640)
    _write_json(path, value)
    path.chmod(0o440)


def _result(
    status: str,
    *,
    runtime: float,
    status_code: int | None = None,
    incumbent: float | None = 1.0,
    bound: float | None = 1.0,
    gap: float | None = 0.0,
) -> object:
    if status_code is None:
        status_code = 2 if status == "OPTIMAL" else 9
    solution_count = 0 if incumbent is None else 1
    return runner.NativeSolveResult(
        output=("" if incumbent is None else "Route: [0, 1], Objective: 1"),
        native_status=status,
        native_status_code=status_code,
        incumbent=incumbent,
        best_bound=bound,
        gap=gap,
        raw_log=f"Gurobi status {status}\n".encode(),
        objective_semantics=runner.OBJECTIVE_SEMANTICS["op"],
        native_wall_time_seconds=runtime,
        native_metadata={
            "solution_count": solution_count,
            "node_count": 12.0,
            "deterministic_work": runtime / 2.0,
            "gurobi_status": status,
            "gurobi_status_code": status_code,
            "gurobi_best_bound": bound,
            "gurobi_gap": gap,
        },
    )


class FakeDiagnosticBackend:
    def __init__(self, results: list[object]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, object]] = []

    def probe_identity(self) -> dict[str, str]:
        return dict(BACKEND_IDENTITY)

    def solve_diagnostic(
        self, row: dict[str, object], *, parameters: dict[str, object]
    ) -> object:
        self.calls.append(dict(parameters))
        return self.results.pop(0)


def test_checked_in_template_is_an_explicit_hold() -> None:
    template = (
        ROOT
        / "experiments"
        / "sota_20260814"
        / "op_reference_scalability_sweep.template.json"
    )
    with pytest.raises(ValueError, match="remains HOLD.*strata"):
        sweep.prepare_sweep(template, verifier_bundle=VERIFIER_BUNDLE)


def test_contract_requires_explicit_positive_time_limits(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    _rewrite_contract(
        contract,
        lambda value: value["selection"]["strata"][0].__setitem__(
            "time_limit_seconds", None
        ),
    )
    with pytest.raises(ValueError, match="time_limit_seconds.*numeric"):
        sweep.prepare_sweep(contract, verifier_bundle=VERIFIER_BUNDLE)


def test_selection_is_deterministic_and_covers_each_frozen_size_stratum(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    first = sweep.prepare_sweep(contract, verifier_bundle=VERIFIER_BUNDLE)
    second = sweep.prepare_sweep(contract, verifier_bundle=VERIFIER_BUNDLE)
    assert [row.source_index for row in first.selected_rows] == [
        row.source_index for row in second.selected_rows
    ]
    assert first.source_rows_by_stratum == {"small": 2, "large": 2}
    assert len(first.selected_rows) == 2


def test_nonoptimal_native_state_is_retained_but_never_labeled_exact(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    backend = FakeDiagnosticBackend(
        [
            _result("OPTIMAL", runtime=1.0),
            _result("TIME_LIMIT", runtime=2.0, bound=2.0, gap=0.5),
        ]
    )
    output = tmp_path / "sweep-output"
    receipt = sweep.run_sweep(
        contract_path=contract,
        output_dir=output,
        backend=backend,
        verifier_bundle=VERIFIER_BUNDLE,
    )
    details = json.loads((output / "details.json").read_text(encoding="utf-8"))
    limited = next(row for row in details if row["native_status"] == "TIME_LIMIT")
    assert limited["classification"] == sweep.NONEXACT_CLASSIFICATION
    assert limited["exact_reference_usable"] is False
    assert limited["native_best_bound"] == 2.0
    assert limited["native_runtime_seconds"] == 2.0
    assert (output / limited["native_log"]["relative_path"]).read_bytes()
    assert receipt["native_status_counts"] == {"OPTIMAL": 1, "TIME_LIMIT": 1}
    assert receipt["production_screen"]["status"] == sweep.HOLD_NONOPTIMAL
    assert receipt["production_authorization"] is False
    assert backend.calls[0]["time_limit_seconds"] == 10.0
    assert backend.calls[1]["time_limit_seconds"] == 20.0
    assert "work_limit" not in backend.calls[0]
    assert backend.calls[1]["work_limit"] == 30.0


def test_all_optimal_sweep_emits_self_hashed_non_authorizing_pass_receipt(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    output = tmp_path / "all-optimal-output"
    receipt = sweep.run_sweep(
        contract_path=contract,
        output_dir=output,
        backend=FakeDiagnosticBackend(
            [_result("OPTIMAL", runtime=1.0), _result("OPTIMAL", runtime=2.0)]
        ),
        verifier_bundle=VERIFIER_BUNDLE,
    )
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_sha256")
    assert hashlib.sha256(sweep._canonical_json_bytes(unsigned)).hexdigest() == digest
    assert receipt["certified_optimal_rows"] == 2
    assert receipt["production_projection"]["projected_native_runtime_seconds"] == 1500.0
    assert receipt["production_screen"]["status"] == sweep.PASS_SCREEN
    assert receipt["production_screen"]["production_authorization"] is False


def test_invalid_optimal_certificate_fails_atomically(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    output = tmp_path / "invalid-output"
    backend = FakeDiagnosticBackend(
        [_result("OPTIMAL", runtime=1.0, bound=2.0), _result("OPTIMAL", runtime=2.0)]
    )
    with pytest.raises(ValueError, match="OPTIMAL bound"):
        sweep.run_sweep(
            contract_path=contract,
            output_dir=output,
            backend=backend,
            verifier_bundle=VERIFIER_BUNDLE,
        )
    assert not output.exists()


def test_publication_op_solve_remains_fail_closed_on_time_limit() -> None:
    backend = object.__new__(runner.GurobiOPBackend)
    backend.solve_diagnostic = lambda row, parameters: _result(  # type: ignore[attr-defined]
        "TIME_LIMIT", runtime=10.0, bound=2.0, gap=0.5
    )
    with pytest.raises(RuntimeError, match="refusing a best-known fallback"):
        backend.solve({}, parameters={})


def test_op_only_development_freeze_is_deterministic_read_only_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The publication freeze is generated in .venv_iid.  Unit tests exercise
    # the same code under the repository test environment without weakening
    # the runtime gate used by the actual CLI invocation.
    monkeypatch.setattr(freezer.np, "__version__", freezer.REQUIRED_NUMPY_VERSION)
    monkeypatch.setattr(freezer.nx, "__version__", freezer.REQUIRED_NETWORKX_VERSION)
    output = tmp_path / "op_scalability_development_v1"
    result = freezer.freeze_development_plan(
        output_dir=output,
        frozen_at_utc="2026-08-26T09:24:14Z",
        remote_observed_at_utc="2026-08-26T09:24:14Z",
    )
    assert result["status"] == sweep.PREFLIGHT_HOLD_REMOTE
    assert result["source_rows"] == 256
    assert result["sample_rows"] == 24
    assert result["confirmation_solving_authorized"] is False

    dataset = output / "op" / "eval" / "test.json"
    receipt_path = output / "generation_receipt.json"
    generation_contract_path = output / "generation_contract.json"
    pending_path = output / "sweep_contract.pending_remote_artifact.json"
    for path in (
        dataset,
        receipt_path,
        generation_contract_path,
        pending_path,
        output / "remote_finalization_spec.json",
    ):
        assert path.stat().st_mode & 0o777 == 0o440

    rows = json.loads(dataset.read_text(encoding="utf-8"))
    assert len(rows) == 256
    assert [row["benchmark_row_id"] for row in rows] == [
        f"op-{index:06d}" for index in range(256)
    ]
    assert all(row["reference_status"] == "pending" for row in rows)
    assert all("output" not in row for row in rows)
    assert len({row["semantic_instance_sha256"] for row in rows}) == 256
    assert all(10 <= int(row["num_nodes"]) <= 100 for row in rows)

    generation_contract = json.loads(generation_contract_path.read_text(encoding="utf-8"))
    unsigned_generation = dict(generation_contract)
    generation_hash = unsigned_generation.pop("contract_sha256")
    assert hashlib.sha256(sweep._canonical_json_bytes(unsigned_generation)).hexdigest() == generation_hash
    assert generation_contract["seed"] == 2026082601
    assert generation_contract["confirmation_data_used"] is False

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    unsigned_receipt = dict(receipt)
    receipt_hash = unsigned_receipt.pop("receipt_sha256")
    assert hashlib.sha256(sweep._canonical_json_bytes(unsigned_receipt)).hexdigest() == receipt_hash
    assert receipt["contains_references"] is False
    assert receipt["publishable"] is False
    assert receipt["tasks"]["op"]["file_sha256"] == runner._file_identity(dataset)[
        "sha256"
    ]

    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    assert pending["selection"]["strata"] == [
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
    assert pending["solver"]["base_parameters"] == runner.OP_EXACT_REQUIRED_PARAMETERS
    assert pending["production_screen"] == {
        "target_rows": 1000,
        "require_all_sampled_optimal": True,
        "max_projected_native_runtime_seconds": 172800.0,
    }
    assert pending["solver"]["artifact"]["immutable_gate_passed"] is False

    verifier_identity = freezer._strict_verifier_identity(ROOT)
    preflight = sweep.preflight_sweep(
        pending_path,
        verifier_bundle=(dict(TASK_EVALUATORS), verifier_identity),
    )
    assert preflight["status"] == sweep.PREFLIGHT_HOLD_REMOTE
    assert preflight["source_rows"] == 256
    assert preflight["sample_rows"] == 24
    assert preflight["confirmation_solving_authorized"] is False
    with pytest.raises(ValueError, match="remains HOLD pending"):
        sweep.prepare_sweep(
            pending_path,
            verifier_bundle=(dict(TASK_EVALUATORS), verifier_identity),
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freezer.freeze_development_plan(
            output_dir=output,
            frozen_at_utc="2026-08-26T09:24:14Z",
            remote_observed_at_utc="2026-08-26T09:24:14Z",
        )


def test_checked_in_op_development_plan_passes_only_the_non_solving_preflight() -> None:
    contract = (
        ROOT
        / "experiments"
        / "sota_20260814"
        / "op_scalability_development_v1"
        / "sweep_contract.pending_remote_artifact.json"
    )
    verifier_identity = freezer._strict_verifier_identity(ROOT)
    result = sweep.preflight_sweep(
        contract,
        verifier_bundle=(dict(TASK_EVALUATORS), verifier_identity),
    )
    assert result == {
        "status": sweep.PREFLIGHT_HOLD_REMOTE,
        "contract_sha256": "4431861b3c3af2614b4dad66149fe92269186501686c2d4c444b54e3d0dfdc4a",
        "source_rows": 256,
        "sample_rows": 24,
        "source_rows_by_stratum": {
            "nodes_10_30": 66,
            "nodes_31_60": 80,
            "nodes_61_80": 51,
            "nodes_81_100": 59,
        },
        "development_sweep_authorized": False,
        "confirmation_data_access_authorized": False,
        "confirmation_solving_authorized": False,
        "production_authorization": False,
        "required_next_step": (
            "create_once_read_only_audited_artifact_then_finalize_contract"
        ),
    }
