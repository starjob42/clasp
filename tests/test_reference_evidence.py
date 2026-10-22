from __future__ import annotations

import importlib.util
import itertools
import json
import math
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
SPEC = importlib.util.spec_from_file_location(
    "run_reference_evidence", ROOT / "scripts" / "run_reference_evidence.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "generate_iid_reference_evidence",
    ROOT / "scripts" / "generate_iid_benchmark.py",
)
assert GENERATOR_SPEC is not None and GENERATOR_SPEC.loader is not None
generator = importlib.util.module_from_spec(GENERATOR_SPEC)
sys.modules[GENERATOR_SPEC.name] = generator
GENERATOR_SPEC.loader.exec_module(generator)

FREEZE_SPEC = importlib.util.spec_from_file_location(
    "freeze_reference_evidence", ROOT / "scripts" / "freeze_benchmark.py"
)
assert FREEZE_SPEC is not None and FREEZE_SPEC.loader is not None
freeze = importlib.util.module_from_spec(FREEZE_SPEC)
sys.modules[FREEZE_SPEC.name] = freeze
FREEZE_SPEC.loader.exec_module(freeze)

VERIFIER_IDENTITY = {"name": "test-strict-clasp-verifier", "version": 1}
VERIFIER_BUNDLE = (dict(TASK_EVALUATORS), VERIFIER_IDENTITY)


def _row(task: str) -> dict[str, object]:
    if task == "tsp":
        row = {
            "instruction": "Solve the TSP and return a closed route.",
            "input": "Three nodes.",
            "instance_json": [[0, 0], [1, 0], [0, 1]],
        }
    elif task == "cvrp":
        row = {
            "instruction": "Solve CVRP with capacity 2.",
            "input": "Depot and two unit-demand customers.",
            "instance_json": [[[0, 0], [1, 0], [0, 1]], [0, 1, 1], 2],
        }
    elif task == "mis":
        row = {
            "instruction": "Find a maximum independent set on 3 nodes.",
            "input": "Edges: [(0,1),(1,2)]",
            "instance_json": {
                "num_nodes": 3,
                "edges": [[0, 1], [1, 2]],
                "distribution": "test",
            },
        }
    elif task == "jssp":
        row = {
            "instruction": "Solve this two-job two-machine JSSP.",
            "input": "Job 0: [(0,1),(1,1)]; Job 1: [(1,1),(0,1)]",
            "instance_json": [[0, 1, 1, 1], [1, 1, 0, 1]],
        }
    else:
        row = {
            "instruction": "Solve OP.",
            "input": "Two nodes.",
            "instance_json": [[[0, 0], [1, 0]], [0, 1], 2.0],
        }
    row["benchmark_row_id"] = f"{task}-000000"
    digest = semantic_instance_sha256(task, row, require_structured=True)
    row["instance_sha256"] = digest
    row["semantic_instance_sha256"] = digest
    row["reference_status"] = "pending"
    row["generator_distribution"] = "test"
    return row


def _draft(tmp_path: Path, task: str) -> Path:
    root = tmp_path / f"draft-{task}"
    dataset = root / task / "eval" / "test.json"
    dataset.parent.mkdir(parents=True)
    row = _row(task)
    dataset.write_text(json.dumps([row], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    semantic = [row["semantic_instance_sha256"]]
    row_hashes = [benchmark_row_sha256(row)]
    receipt = runner._self_hash(
        {
            "schema_version": 2,
            "status": "instances_generated",
            "generator_contract_sha256": "a" * 64,
            "generator_code_sha256": "b" * 64,
            "runtime_audit": {"test": True},
            "tasks": {
                task: {
                    "rows": 1,
                    "relative_path": f"{task}/eval/test.json",
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
    (root / "generation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def _task_settings(task: str) -> tuple[dict[str, object], dict[str, object]]:
    if task == "tsp":
        return (
            {"starts": "all", "max_2opt_passes": 20},
            {
                "adapter": "raw_euclidean_tsp_heuristic_v1",
                "backend_name": "clasp_raw_tsp_heuristic",
                "native_version": "1",
            },
        )
    if task == "cvrp":
        return (
            {"max_2opt_passes": 20},
            {
                "adapter": "raw_euclidean_cvrp_heuristic_v1",
                "backend_name": "clasp_raw_cvrp_heuristic",
                "native_version": "1",
            },
        )
    if task == "op":
        return (
            {
                "time_limit_seconds": 300,
                "threads": 1,
                "seed": 0,
                "method": 1,
                "mip_gap": 0.0,
                "mip_gap_abs": 0.0,
                "work_limit": 60.0,
                "node_limit": 10000,
                "feasibility_tolerance": 1e-9,
                "integrality_tolerance": 1e-9,
                "numeric_focus": 3,
                "output_flag": 1,
                "log_to_console": 0,
            },
            {
                "adapter": "gurobi_op_exact_v2",
                "backend_name": "gurobi",
                "native_version": "13.0.1",
            },
        )
    if task == "mis":
        return (
            {
                "time_limit_seconds": 300,
                "threads": 1,
                "seed": 0,
                "method": 1,
                "mip_gap": 0.0,
                "mip_gap_abs": 0.0,
                "feasibility_tolerance": 1e-9,
                "integrality_tolerance": 1e-9,
                "numeric_focus": 3,
                "output_flag": 1,
                "log_to_console": 0,
            },
            {"adapter": "gurobi_mis_v1", "backend_name": "gurobi", "native_version": "13.0.1"},
        )
    if task == "jssp":
        return (
            {
                "max_time_seconds": 300,
                "num_search_workers": 1,
                "random_seed": 1,
                "absolute_gap_limit": 0.0,
                "relative_gap_limit": 0.0,
                "log_search_progress": True,
            },
            {
                "adapter": "ortools_jssp_v1",
                "backend_name": "ortools_cp_sat",
                "native_version": "9.15.6755",
            },
        )
    raise AssertionError(f"unknown task: {task}")


def _contract(
    tmp_path: Path,
    task: str,
    *,
    status: str = "approved_for_reference_evidence",
) -> tuple[Path, Path]:
    artifact = tmp_path / f"{task}-solver.artifact"
    artifact.write_bytes(f"pinned {task} solver artifact".encode())
    parameters, adapter = _task_settings(task)
    task_contract = {
        "backend": adapter["backend_name"],
        "version": adapter["native_version"],
        "license": "test-only fake backend",
        "parameters": parameters,
        "accepted_statuses": [runner.PUBLIC_STATUS.get(task, "BLOCKED")],
        "quality_semantics": runner.REFERENCE_QUALITY.get(task, "blocked"),
        "evidence_runner": {
            "approval": "approved",
            "adapter": adapter["adapter"],
            "objective_semantics": runner.OBJECTIVE_SEMANTICS[task],
            "reference_quality": runner.REFERENCE_QUALITY.get(task, "blocked"),
            "seed_parameter": (
                "random_seed"
                if task == "jssp"
                else ("seed" if task in {"op", "mis"} else None)
            ),
            "accepted_native_statuses": [runner.REQUIRED_NATIVE_STATUS.get(task, "BLOCKED")],
            "solver_identity": {
                "backend_name": adapter["backend_name"],
                "native_version": adapter["native_version"],
                "artifact_kind": "test_fixture",
                "artifact_path": str(artifact),
                "artifact_sha256": runner._file_identity(artifact)["sha256"],
            },
        },
    }
    contract = runner._self_hash(
        {
            "schema_version": 2,
            "status": status,
            "reference_evidence_runner": {
                "schema_version": 1,
                "protocol": runner.RUNNER_PROTOCOL,
                "expected_instances_per_task": 1,
                "strict_verifier_identity_sha256": sha256_json(VERIFIER_IDENTITY),
                "runner_code_sha256": runner._file_identity(
                    ROOT / "scripts" / "run_reference_evidence.py"
                )["sha256"],
            },
            "tasks": {task: task_contract},
        },
        "contract_sha256",
    )
    path = tmp_path / f"{task}-contract.json"
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, artifact


class FakeBackend:
    def __init__(self, identity: dict[str, str], result: runner.NativeSolveResult):
        self.identity = identity
        self.result = result
        self.probes = 0
        self.calls = 0

    def probe_identity(self) -> dict[str, str]:
        self.probes += 1
        return self.identity

    def solve(self, _row, *, parameters):
        self.calls += 1
        assert isinstance(parameters, dict)
        return self.result


def _result(task: str, **overrides) -> runner.NativeSolveResult:
    values: dict[str, object]
    if task == "tsp":
        objective = 2.0 + math.sqrt(2.0)
        values = {
            "output": f"Route: [0, 1, 2, 0], Objective: {objective:.17g}",
            "native_status": "HEURISTIC_COMPLETE",
            "native_status_code": "HEURISTIC_COMPLETE",
            "incumbent": objective,
            "best_bound": None,
            "gap": None,
            "raw_log": b"raw tsp native log\n",
            "objective_semantics": runner.OBJECTIVE_SEMANTICS[task],
            "native_metadata": {"starts": 3},
        }
    elif task == "cvrp":
        objective = 2.0 + math.sqrt(2.0)
        values = {
            "output": f"Routes: [[0, 1, 2, 0]], Objective: {objective:.17g}",
            "native_status": "HEURISTIC_COMPLETE",
            "native_status_code": "HEURISTIC_COMPLETE",
            "incumbent": objective,
            "best_bound": None,
            "gap": None,
            "raw_log": b"raw cvrp native log\n",
            "objective_semantics": runner.OBJECTIVE_SEMANTICS[task],
            "native_metadata": {"routes": 1},
        }
    elif task == "op":
        values = {
            "output": "Route: [0, 1], Objective: 1",
            "native_status": "OPTIMAL",
            "native_status_code": 2,
            "incumbent": 1.0,
            "best_bound": 1.0,
            "gap": 0.0,
            "raw_log": b"Gurobi Optimizer version 13.0.1\nOptimal solution found\n",
            "objective_semantics": runner.OBJECTIVE_SEMANTICS[task],
            "native_metadata": {
                "solution_count": 1,
                "node_count": 1,
                "gurobi_status": "OPTIMAL",
                "gurobi_status_code": 2,
                "gurobi_best_bound": 1.0,
                "gurobi_gap": 0.0,
            },
        }
    elif task == "mis":
        values = {
            "output": "Set: [0, 2], Objective: 2",
            "native_status": "OPTIMAL",
            "native_status_code": 2,
            "incumbent": 2.0,
            "best_bound": 2.0,
            "gap": 0.0,
            "raw_log": b"Gurobi Optimizer version 13.0.1\nOptimal solution found\n",
            "objective_semantics": runner.OBJECTIVE_SEMANTICS[task],
            "native_metadata": {"solution_count": 1, "node_count": 1},
        }
    else:
        raw_log = (
            b"CpSolverResponse summary:\n"
            b"status: OPTIMAL\n"
            b"objective: 2\n"
            b"best_bound: 2\n"
        )
        values = {
            "output": "Schedule: [[0, 1], [1, 0]], Makespan: 2",
            "native_status": "OPTIMAL",
            "native_status_code": 4,
            "incumbent": 2.0,
            "best_bound": 2.0,
            "gap": 0.0,
            "raw_log": raw_log,
            "objective_semantics": runner.OBJECTIVE_SEMANTICS[task],
            "native_metadata": {
                "response_stats_sha256": runner.hashlib.sha256(raw_log).hexdigest()
            },
        }
    values.update(overrides)
    return runner.NativeSolveResult(**values)


def _fake_backend(task: str, result: runner.NativeSolveResult | None = None) -> FakeBackend:
    _parameters, adapter = _task_settings(task)
    return FakeBackend(
        {"backend_name": adapter["backend_name"], "native_version": adapter["native_version"]},
        result or _result(task),
    )


def _solve(tmp_path: Path, task: str, result=None):
    draft = _draft(tmp_path, task)
    contract, _artifact = _contract(tmp_path, task)
    evidence = tmp_path / f"evidence-{task}"
    backend = _fake_backend(task, result)
    summary = runner.solve_task(
        draft_root=draft,
        solver_contract_path=contract,
        task=task,
        output_dir=evidence,
        backend=backend,
        verifier_bundle=VERIFIER_BUNDLE,
    )
    return draft, contract, evidence, backend, summary


def test_routing_evidence_and_export_are_best_known_never_optimal(tmp_path: Path) -> None:
    draft, contract, evidence, backend, summary = _solve(tmp_path, "tsp")
    assert backend.calls == 1
    assert summary["reference_quality"] == "best_known_heuristic"
    records = json.loads((evidence / "details.json").read_text())
    record = records[0]
    assert record["solver_status"] == "BEST_KNOWN_HEURISTIC_VERIFIED"
    assert record["optimality_certificate"] is False
    assert record["native_best_bound"] is None
    assert record["native_gap"] is None
    assert record["semantic_instance_sha256"] == _row("tsp")["semantic_instance_sha256"]
    runner._verify_self_hash(record, "reference_evidence_record_sha256", label="record")

    exported_dir = tmp_path / "references"
    exported_summary = runner.export_references(
        draft_root=draft,
        solver_contract_path=contract,
        evidence_dirs={"tsp": evidence},
        output_dir=exported_dir,
        verifier_bundle=VERIFIER_BUNDLE,
    )
    exported = json.loads((exported_dir / "tsp.json").read_text())[0]
    assert exported["reference_evidence"]["native_status"] == "HEURISTIC_COMPLETE"
    assert exported["reference_evidence"]["raw_native_log"]["sha256"]
    assert exported["reference_evidence"]["strict_verifier_result"]["is_feasible"] is True
    assert exported["reference_evidence"]["reference_evidence_summary_sha256"] == summary[
        "reference_evidence_summary_sha256"
    ]
    runner._verify_self_hash(exported, "reference_export_record_sha256", label="export")
    runner._verify_self_hash(
        exported_summary, "reference_export_summary_sha256", label="export summary"
    )


@pytest.mark.parametrize("task", ["tsp", "cvrp"])
def test_builtin_raw_routing_backends_match_strict_objective_semantics(
    tmp_path: Path, task: str
) -> None:
    draft = _draft(tmp_path, task)
    contract_path, artifact = _contract(tmp_path, task)
    contract = json.loads(contract_path.read_text())
    identity = contract["tasks"][task]["evidence_runner"]["solver_identity"]
    identity["artifact_path"] = str(ROOT / "scripts" / "run_reference_evidence.py")
    identity["artifact_sha256"] = runner._file_identity(identity["artifact_path"])["sha256"]
    contract = runner._self_hash(contract, "contract_sha256")
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    artifact.unlink()
    summary = runner.solve_task(
        draft_root=draft,
        solver_contract_path=contract_path,
        task=task,
        output_dir=tmp_path / f"builtin-{task}",
        verifier_bundle=VERIFIER_BUNDLE,
    )
    assert summary["native_status_counts"] == {"HEURISTIC_COMPLETE": 1}


def test_op_exact_instance_and_mip_start_match_open_route_semantics() -> None:
    row = {
        "instance_json": [
            [[0, 0], [1, 0], [2, 0], [100, 0]],
            [0, 2, 3, 100],
            2.0,
        ]
    }
    instance = runner._validated_op_instance(row)
    route = runner._deterministic_op_start(instance)
    assert route == [0, 1, 2]
    assert runner._op_route_distance(instance, route) == 2.0
    assert sum(instance.prizes[node] for node in route) == 5


def test_op_budget_incompatibility_cuts_preserve_all_small_feasible_routes() -> None:
    instance = runner._validated_op_instance(
        {
            "instance_json": [
                [[0, 0], [1, 0], [0, 1], [5, 0]],
                [0, 1, 1, 1],
                2.1,
            ]
        }
    )
    incompatible = set(runner._op_budget_incompatible_pairs(instance))
    assert (1, 2) in incompatible
    for size in range(4):
        for suffix in itertools.permutations(range(1, 4), size):
            route = (0, *suffix)
            if runner._op_route_distance(instance, route) <= instance.route_limit:
                selected = set(route)
                assert all(
                    left not in selected or right not in selected
                    for left, right in incompatible
                )


@pytest.mark.parametrize(
    "instance,match",
    [
        ([[[0, 0]], [True], 0.0], "must be integral"),
        ([[[0, 0]], [-1], 0.0], "nonnegative"),
        ([[[0, 0]], [0], -1.0], "nonnegative"),
        ([[[float("nan"), 0]], [0], 0.0], "non-finite coordinate"),
        ([[[0, 0], [1, 0]], [0], 1.0], "inconsistent"),
    ],
)
def test_op_exact_instance_rejects_adversarial_schema(instance, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        runner._validated_op_instance({"instance_json": instance})


@pytest.mark.parametrize(
    "field,value",
    [
        ("threads", 2),
        ("seed", 1),
        ("method", 0),
        ("mip_gap", 0.01),
        ("feasibility_tolerance", 1e-6),
    ],
)
def test_op_exact_contract_rejects_unfrozen_solver_parameters(
    tmp_path: Path, field: str, value: object
) -> None:
    draft = _draft(tmp_path, "op")
    contract_path, _artifact = _contract(tmp_path, "op")
    contract = json.loads(contract_path.read_text())
    contract["tasks"]["op"]["parameters"][field] = value
    contract = runner._self_hash(contract, "contract_sha256")
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    backend = _fake_backend("op")
    with pytest.raises(ValueError, match=rf"op {field} must be fixed"):
        runner.solve_task(
            draft_root=draft,
            solver_contract_path=contract_path,
            task="op",
            output_dir=tmp_path / "bad-op-contract",
            backend=backend,
            verifier_bundle=VERIFIER_BUNDLE,
        )
    assert backend.probes == backend.calls == 0


def test_historical_op_adapter_cannot_enter_exact_lane(tmp_path: Path) -> None:
    draft = _draft(tmp_path, "op")
    contract_path, _artifact = _contract(tmp_path, "op")
    contract = json.loads(contract_path.read_text())
    contract["tasks"]["op"]["evidence_runner"]["adapter"] = "gurobi_op_v1"
    contract = runner._self_hash(contract, "contract_sha256")
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    backend = _fake_backend("op")
    with pytest.raises(ValueError, match="machine policy is invalid for op"):
        runner.solve_task(
            draft_root=draft,
            solver_contract_path=contract_path,
            task="op",
            output_dir=tmp_path / "historical-op-adapter",
            backend=backend,
            verifier_bundle=VERIFIER_BUNDLE,
        )
    assert backend.probes == backend.calls == 0


def test_op_production_contract_requires_audited_gurobi_13_wheel(
    tmp_path: Path,
) -> None:
    contract_path, _artifact = _contract(tmp_path, "op")
    contract = json.loads(contract_path.read_text())
    with pytest.raises(ValueError, match="artifact_kind must equal 'python_wheel'"):
        runner.validate_reference_solver_task_policy(
            "op",
            contract["tasks"]["op"],
            runner_code_sha256=contract["reference_evidence_runner"]["runner_code_sha256"],
            require_production_identity=True,
        )


def test_checked_in_op_exact_template_remains_production_blocked() -> None:
    template = json.loads(
        (
            ROOT
            / "experiments"
            / "sota_20260814"
            / "reference_solver_contract.template.json"
        ).read_text()
    )
    op = template["tasks"]["op"]
    assert template["status"] == "BLOCKED_UNRESOLVED_DO_NOT_USE"
    assert op["accepted_statuses"] == []
    assert op["evidence_runner"]["approval"].startswith("blocked_")
    assert op["evidence_runner"]["adapter"] == runner.OP_EXACT_ADAPTER
    assert op["evidence_runner"]["reference_quality"] == "certified_optimum"
    assert any("all 1,000 OPTIMAL" in blocker for blocker in op["blockers"])


def test_checked_in_solver_template_tracks_current_contract_surface() -> None:
    template = json.loads(
        (
            ROOT
            / "experiments"
            / "sota_20260814"
            / "reference_solver_contract.template.json"
        ).read_text()
    )
    protocol = template["reference_evidence_runner"]
    assert protocol["runner_code_sha256"] == runner._file_identity(
        ROOT / "scripts" / "run_reference_evidence.py"
    )["sha256"]
    assert "approved_for_reference_evidence" in template["approval_rule"]
    assert "separate all-five evidence approval receipt" in template["approval_rule"]

    for task in ("tsp", "cvrp"):
        task_policy = template["tasks"][task]
        assert task_policy["evidence_runner"]["seed_parameter"] is None
        assert "seed" not in task_policy["parameters"]

    for task in ("op", "mis"):
        parameters = template["tasks"][task]["parameters"]
        assert parameters["feasibility_tolerance"] == 1e-9
        assert parameters["integrality_tolerance"] == 1e-9
        assert parameters["numeric_focus"] == 3

    assert template["tasks"]["jssp"]["parameters"]["log_search_progress"] is True
    for task in ("op", "mis", "jssp"):
        identity = template["tasks"][task]["evidence_runner"]["solver_identity"]
        assert identity["distribution_name"] in {"gurobipy", "ortools"}
        assert identity["module_origin"]
        assert identity["installed_distribution_manifest_sha256"]

    candidate = json.loads(json.dumps(template))
    verifier_sha256 = "b" * 64
    candidate["status"] = "approved_for_reference_evidence"
    candidate["reference_evidence_runner"][
        "strict_verifier_identity_sha256"
    ] = verifier_sha256
    for task, task_policy in candidate["tasks"].items():
        task_policy["accepted_statuses"] = [runner.PUBLIC_STATUS[task]]
        evidence_runner = task_policy["evidence_runner"]
        evidence_runner["approval"] = "approved"
        evidence_runner["accepted_native_statuses"] = [runner.REQUIRED_NATIVE_STATUS[task]]
        identity = evidence_runner["solver_identity"]
        identity["artifact_path"] = f"/frozen/reference-solvers/{task}.artifact"
        if task in {"tsp", "cvrp"}:
            identity["artifact_sha256"] = protocol["runner_code_sha256"]
        else:
            identity["module_origin"] = f"/frozen/site-packages/{task}/__init__.py"
            identity["installed_distribution_manifest_sha256"] = "c" * 64
    candidate = runner._self_hash(candidate, "contract_sha256")
    runner.validate_reference_solver_contract_policy(
        candidate,
        verifier_identity_sha256=verifier_sha256,
        expected_instances_per_task=1000,
        require_production_identity=True,
    )


def test_contract_and_artifact_gates_run_before_backend_invocation(tmp_path: Path) -> None:
    draft = _draft(tmp_path, "mis")
    contract_path, artifact = _contract(
        tmp_path, "mis", status="BLOCKED_UNRESOLVED_DO_NOT_USE"
    )
    backend = _fake_backend("mis")
    with pytest.raises(ValueError, match="not approved"):
        runner.solve_task(
            draft_root=draft,
            solver_contract_path=contract_path,
            task="mis",
            output_dir=tmp_path / "blocked",
            backend=backend,
            verifier_bundle=VERIFIER_BUNDLE,
        )
    assert backend.probes == backend.calls == 0

    contract_path, artifact = _contract(tmp_path, "mis")
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        runner.solve_task(
            draft_root=draft,
            solver_contract_path=contract_path,
            task="mis",
            output_dir=tmp_path / "bad-artifact",
            backend=backend,
            verifier_bundle=VERIFIER_BUNDLE,
        )
    assert backend.probes == backend.calls == 0


def test_op_exact_mip_reference_is_certified_verified_and_exportable(
    tmp_path: Path,
) -> None:
    draft, contract, evidence, backend, summary = _solve(tmp_path, "op")
    assert backend.calls == 1
    assert summary["reference_quality"] == "certified_optimum"
    record = json.loads((evidence / "details.json").read_text())[0]
    assert record["solver_status"] == "OPTIMAL_CERTIFIED"
    assert record["optimality_certificate"] is True
    assert record["native_best_bound"] == 1.0
    assert record["native_gap"] == 0.0
    assert record["native_metadata"]["gurobi_best_bound"] == 1.0
    assert record["strict_verifier_result"]["is_feasible"] is True

    exported_dir = tmp_path / "references"
    exported = runner.export_references(
        draft_root=draft,
        solver_contract_path=contract,
        evidence_dirs={"op": evidence},
        output_dir=exported_dir,
        verifier_bundle=VERIFIER_BUNDLE,
    )
    assert exported["tasks"]["op"]["rows"] == 1


def test_op_exact_export_survives_assembly_and_freezer_certificate_replay(
    tmp_path: Path,
) -> None:
    draft, contract_path, evidence, _backend, _summary = _solve(tmp_path, "op")
    references = tmp_path / "references"
    export_summary = runner.export_references(
        draft_root=draft,
        solver_contract_path=contract_path,
        evidence_dirs={"op": evidence},
        output_dir=references,
        verifier_bundle=VERIFIER_BUNDLE,
    )
    reference = json.loads((references / "op.json").read_text())[0]
    row = json.loads((draft / "op" / "eval" / "test.json").read_text())[0]
    solver_task = json.loads(contract_path.read_text())["tasks"]["op"]
    output = reference["output"]
    metrics = TASK_EVALUATORS["op"](
        row["instruction"], row["input"], output, row["instance_json"], output
    )
    evidence_payload = generator._validate_reference_evidence(
        task="op",
        reference=reference,
        solver_task=solver_task,
        expected_summary_sha256=export_summary["tasks"]["op"][
            "reference_evidence_summary_sha256"
        ],
        verifier_identity=VERIFIER_IDENTITY,
        strict_metrics=metrics,
        reference_root=references,
        expected_runner_code_sha256=json.loads(contract_path.read_text())[
            "reference_evidence_runner"
        ]["runner_code_sha256"],
    )
    assert evidence_payload["optimality_certificate"] is True

    provenance = {
        "solver_status": reference["solver_status"],
        "solver_runtime_seconds": reference["solver_runtime_seconds"],
        "solver_contract_task_sha256": sha256_json(solver_task),
        "quality_semantics": solver_task["quality_semantics"],
        "reference_export_record_sha256": reference[
            "reference_export_record_sha256"
        ],
        "reference_evidence_approval_receipt_sha256": "a" * 64,
        "reference_evidence": evidence_payload,
    }
    frozen_row = dict(row)
    frozen_row.pop("reference_status")
    frozen_row["output"] = output
    frozen_row["reference_provenance"] = provenance
    dataset = tmp_path / "assembled-op.json"
    dataset.write_text(json.dumps([frozen_row], indent=2, sort_keys=True) + "\n")
    manifest, _hashes = freeze._audit_test_task(
        freeze.DatasetSpec("op", "iid", str(dataset)),
        minimum_instances=1,
        evaluator=TASK_EVALUATORS["op"],
        solver_contract_task=solver_task,
        verifier_identity=VERIFIER_IDENTITY,
        require_reference_evidence=True,
        runner_code_sha256=json.loads(contract_path.read_text())[
            "reference_evidence_runner"
        ]["runner_code_sha256"],
        evidence_root=references,
        approval_receipt_sha256="a" * 64,
    )
    assert manifest["reference_audit"]["strictly_feasible"] == 1
    assert manifest["reference_evidence_summary_sha256"] == export_summary["tasks"][
        "op"
    ]["reference_evidence_summary_sha256"]

    downgraded = deepcopy_json(reference)
    downgraded["solver_status"] = "BEST_KNOWN_HEURISTIC_VERIFIED"
    downgraded = runner._self_hash(downgraded, "reference_export_record_sha256")
    with pytest.raises(ValueError, match="invalid public solver status"):
        generator._validate_reference_evidence(
            task="op",
            reference=downgraded,
            solver_task=solver_task,
            expected_summary_sha256=export_summary["tasks"]["op"][
                "reference_evidence_summary_sha256"
            ],
            verifier_identity=VERIFIER_IDENTITY,
            strict_metrics=metrics,
            reference_root=references,
            expected_runner_code_sha256=json.loads(contract_path.read_text())[
                "reference_evidence_runner"
            ]["runner_code_sha256"],
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"native_status": "TIME_LIMIT"}, "native status"),
        ({"native_status_code": 9}, "numeric status 2"),
        ({"best_bound": 1.0}, "does not equal"),
        ({"gap": 0.01}, "exactly zero"),
        ({"incumbent": 2.5}, "must be integral"),
        ({"native_metadata": {"solution_count": 0}}, "SolCount > 0"),
    ],
)
def test_mis_requires_full_gurobi_optimal_certificate(
    tmp_path: Path, overrides: dict, match: str
) -> None:
    result = _result("mis", **overrides)
    with pytest.raises(ValueError, match=match):
        _solve(tmp_path, "mis", result)
    assert not (tmp_path / "evidence-mis").exists()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"native_status": "WORK_LIMIT"}, "native status"),
        ({"native_status": "TIME_LIMIT"}, "native status"),
        ({"native_status_code": 9}, "numeric status 2"),
        ({"best_bound": 2.0}, "does not equal"),
        ({"gap": 0.01}, "exactly zero"),
        ({"incumbent": 1.5}, "must be integral"),
        ({"native_metadata": {"solution_count": 0}}, "SolCount > 0"),
        (
            {
                "native_metadata": {
                    "solution_count": 1,
                    "gurobi_status": "WORK_LIMIT",
                    "gurobi_status_code": 16,
                    "gurobi_best_bound": 1.0,
                    "gurobi_gap": 0.0,
                }
            },
            "metadata must retain Gurobi status OPTIMAL",
        ),
        (
            {
                "native_metadata": {
                    "solution_count": 1,
                    "gurobi_status": "OPTIMAL",
                    "gurobi_status_code": 2,
                    "gurobi_best_bound": 2.0,
                    "gurobi_gap": 0.0,
                }
            },
            "retained Gurobi bound disagrees",
        ),
    ],
)
def test_op_requires_full_gurobi_optimal_certificate(
    tmp_path: Path, overrides: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _solve(tmp_path, "op", _result("op", **overrides))
    assert not (tmp_path / "evidence-op").exists()


def test_op_exact_certificate_still_requires_strict_route_replay(tmp_path: Path) -> None:
    result = _result("op", output="Route: [1], Objective: 1")
    with pytest.raises(ValueError, match="strict verifier rejected"):
        _solve(tmp_path, "op", result)
    assert not (tmp_path / "evidence-op").exists()


def test_exact_records_require_incumbent_equal_bound(
    tmp_path: Path,
) -> None:
    for task in ("op", "mis", "jssp"):
        task_root = tmp_path / task
        task_root.mkdir()
        _draft_path, _contract_path, evidence, _backend, summary = _solve(
            task_root, task
        )
        record = json.loads((evidence / "details.json").read_text())[0]
        assert record["solver_status"] == "OPTIMAL_CERTIFIED"
        assert record["optimality_certificate"] is True
        assert record["native_incumbent"] == record["native_best_bound"]
        assert record["native_gap"] == 0.0
        assert summary["reference_quality"] == "certified_optimum"


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"native_status": "FEASIBLE"}, "native status"),
        ({"native_status_code": 2}, "numeric status 4"),
        ({"best_bound": 1.0}, "does not equal"),
        ({"gap": 0.1}, "exactly zero"),
    ],
)
def test_jssp_rejects_feasible_or_unmatched_bound(
    tmp_path: Path, overrides: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _solve(tmp_path, "jssp", _result("jssp", **overrides))


def test_heuristic_cannot_smuggle_bound_or_wrong_objective_semantics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not manufacture"):
        _solve(tmp_path, "tsp", _result("tsp", best_bound=3.0, gap=0.0))
    second = tmp_path / "second"
    second.mkdir()
    with pytest.raises(ValueError, match="objective semantics different"):
        _solve(
            second,
            "tsp",
            _result("tsp", objective_semantics="rounded_tsplib_euc_2d"),
        )


def test_backend_identity_and_strict_objective_mismatch_are_atomic(tmp_path: Path) -> None:
    draft = _draft(tmp_path, "tsp")
    contract, _artifact = _contract(tmp_path, "tsp")
    bad_identity = _fake_backend("tsp")
    bad_identity.identity["native_version"] = "different"
    output = tmp_path / "bad-identity"
    with pytest.raises(ValueError, match="native_version mismatch"):
        runner.solve_task(
            draft_root=draft,
            solver_contract_path=contract,
            task="tsp",
            output_dir=output,
            backend=bad_identity,
            verifier_bundle=VERIFIER_BUNDLE,
        )
    assert not output.exists()
    assert bad_identity.calls == 0

    bad_objective = _fake_backend("tsp", _result("tsp", incumbent=99.0))
    with pytest.raises(ValueError, match="raw Euclidean"):
        runner.solve_task(
            draft_root=draft,
            solver_contract_path=contract,
            task="tsp",
            output_dir=tmp_path / "bad-objective",
            backend=bad_objective,
            verifier_bundle=VERIFIER_BUNDLE,
        )
    assert not (tmp_path / "bad-objective").exists()


def test_tampered_native_log_or_details_is_rejected(tmp_path: Path) -> None:
    draft, contract, evidence, _backend, _summary = _solve(tmp_path, "mis")
    log = evidence / "native_logs" / "mis-000000.log"
    log.write_bytes(b"altered log")
    with pytest.raises(ValueError, match="native log identity mismatch"):
        runner.validate_evidence_directory(
            evidence_dir=evidence,
            draft_root=draft,
            solver_contract_path=contract,
            verifier_bundle=VERIFIER_BUNDLE,
        )

    second = tmp_path / "second"
    second.mkdir()
    draft, contract, evidence, _backend, _summary = _solve(second, "mis")
    records = json.loads((evidence / "details.json").read_text())
    records[0]["native_status"] = "TIME_LIMIT"
    (evidence / "details.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = json.loads((evidence / "summary.json").read_text())
    identity = runner._file_identity(evidence / "details.json")
    summary["details_artifact"].update(identity)
    summary = runner._self_hash(summary, "reference_evidence_summary_sha256")
    (evidence / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="record 0 self-hash mismatch"):
        runner.validate_evidence_directory(
            evidence_dir=evidence,
            draft_root=draft,
            solver_contract_path=contract,
            verifier_bundle=VERIFIER_BUNDLE,
        )


def test_stale_draft_semantic_identity_and_verifier_identity_fail_closed(
    tmp_path: Path,
) -> None:
    draft = _draft(tmp_path, "tsp")
    contract, _artifact = _contract(tmp_path, "tsp")
    rows_path = draft / "tsp" / "eval" / "test.json"
    rows = json.loads(rows_path.read_text())
    rows[0]["instance_json"][1] = [9, 9]
    rows_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    backend = _fake_backend("tsp")
    with pytest.raises(ValueError, match="draft file hash"):
        runner.solve_task(
            draft_root=draft,
            solver_contract_path=contract,
            task="tsp",
            output_dir=tmp_path / "stale-draft",
            backend=backend,
            verifier_bundle=VERIFIER_BUNDLE,
        )
    assert backend.calls == 0

    clean = tmp_path / "clean"
    clean.mkdir()
    draft = _draft(clean, "tsp")
    contract, _artifact = _contract(clean, "tsp")
    with pytest.raises(ValueError, match="strict-verifier identity"):
        runner.solve_task(
            draft_root=draft,
            solver_contract_path=contract,
            task="tsp",
            output_dir=clean / "wrong-verifier",
            backend=_fake_backend("tsp"),
            verifier_bundle=(dict(TASK_EVALUATORS), {"name": "different"}),
        )


def test_evidence_record_binds_exact_parameters_seed_log_and_verifier(tmp_path: Path) -> None:
    _draft_path, _contract_path, evidence, _backend, _summary = _solve(tmp_path, "jssp")
    record = json.loads((evidence / "details.json").read_text())[0]
    assert record["solver_seed"] == 1
    assert record["solver_parameters"]["num_search_workers"] == 1
    assert record["solver_parameters_sha256"] == sha256_json(record["solver_parameters"])
    assert record["solver_identity"]["artifact_sha256"]
    assert record["solver_identity"]["native_version"] == "9.15.6755"
    assert record["wall_time_seconds"] >= 0
    assert record["raw_native_log"]["size_bytes"] > 0
    assert record["strict_verifier_result_sha256"] == sha256_json(
        record["strict_verifier_result"]
    )
    assert record["benchmark_row_sha256"] == benchmark_row_sha256(_row("jssp"))


def test_assembly_and_freeze_validation_reject_dropped_or_unbound_evidence(
    tmp_path: Path,
) -> None:
    draft, contract_path, evidence, _backend, _summary = _solve(tmp_path, "tsp")
    references = tmp_path / "references"
    export_summary = runner.export_references(
        draft_root=draft,
        solver_contract_path=contract_path,
        evidence_dirs={"tsp": evidence},
        output_dir=references,
        verifier_bundle=VERIFIER_BUNDLE,
    )
    reference = json.loads((references / "tsp.json").read_text())[0]
    row = json.loads((draft / "tsp" / "eval" / "test.json").read_text())[0]
    contract = json.loads(contract_path.read_text())
    solver_task = contract["tasks"]["tsp"]
    output = reference["output"]
    metrics = TASK_EVALUATORS["tsp"](
        row["instruction"], row["input"], output, row["instance_json"], output
    )
    evidence_payload = generator._validate_reference_evidence(
        task="tsp",
        reference=reference,
        solver_task=solver_task,
        expected_summary_sha256=export_summary["tasks"]["tsp"][
            "reference_evidence_summary_sha256"
        ],
        verifier_identity=VERIFIER_IDENTITY,
        strict_metrics=metrics,
        reference_root=references,
        expected_runner_code_sha256=contract["reference_evidence_runner"][
            "runner_code_sha256"
        ],
    )
    provenance = {
        "solver_status": reference["solver_status"],
        "solver_runtime_seconds": reference["solver_runtime_seconds"],
        "solver_contract_task_sha256": sha256_json(solver_task),
        "quality_semantics": solver_task["quality_semantics"],
        "reference_export_record_sha256": reference["reference_export_record_sha256"],
        "reference_evidence_approval_receipt_sha256": "a" * 64,
        "reference_evidence": evidence_payload,
    }
    frozen_row = dict(row)
    frozen_row.pop("reference_status")
    frozen_row["output"] = output
    frozen_row["reference_provenance"] = provenance
    dataset = tmp_path / "assembled-tsp.json"
    dataset.write_text(json.dumps([frozen_row], indent=2, sort_keys=True) + "\n")
    manifest, _hashes = freeze._audit_test_task(
        freeze.DatasetSpec("tsp", "iid", str(dataset)),
        minimum_instances=1,
        evaluator=TASK_EVALUATORS["tsp"],
        solver_contract_task=solver_task,
        verifier_identity=VERIFIER_IDENTITY,
        require_reference_evidence=True,
        runner_code_sha256=contract["reference_evidence_runner"]["runner_code_sha256"],
        evidence_root=references,
        approval_receipt_sha256="a" * 64,
    )
    assert manifest["ordered_reference_evidence_record_sha256"] == [
        evidence_payload["reference_evidence_record_sha256"]
    ]

    dropped = deepcopy_json(frozen_row)
    dropped["reference_provenance"].pop("reference_evidence")
    dataset.write_text(json.dumps([dropped], indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="missing bound reference evidence"):
        freeze._audit_test_task(
            freeze.DatasetSpec("tsp", "iid", str(dataset)),
            minimum_instances=1,
            evaluator=TASK_EVALUATORS["tsp"],
            solver_contract_task=solver_task,
            verifier_identity=VERIFIER_IDENTITY,
            require_reference_evidence=True,
            runner_code_sha256=contract["reference_evidence_runner"][
                "runner_code_sha256"
            ],
            evidence_root=references,
            approval_receipt_sha256="a" * 64,
        )


def deepcopy_json(value):
    return json.loads(json.dumps(value))
