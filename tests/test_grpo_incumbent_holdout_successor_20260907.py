"""Model-free checks of the separate two-task holdout's admission boundaries."""
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/build_grpo_incumbent_holdout_successor_20260907.py"
    spec = importlib.util.spec_from_file_location("incumbent_successor_holdout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


holdout = _load()


def plan():
    return {
        "plan_id": holdout.PLAN_ID, "instances_per_task": 300,
        "task_seeds": {"cvrp": 2026090711, "mis": 2026090712}, "workers": 4,
        "runtime_versions": {"numpy": "2.3.2", "networkx": "3.5", "ortools": "9.15.6755"},
        "reference_parameters": {
            "cvrp_seconds": 10.0, "mis_deterministic_seconds": 5.0, "mis_wall_seconds": 30.0,
            "mis_solver_seed": 20260905, "mis_refine_deterministic_seconds": 180.0,
            "mis_refine_wall_seconds": 300.0, "mis_max_refinement_rows": 20,
            "lkh_seeds": [101, 202], "lkh_seconds": 10.0, "lkh_external_grace_seconds": 5.0,
            "lkh_integer_distance_scale": 1000, "lkh_precision": 100,
        },
        "forbidden_sources": [
            {"task": task, "label": label, "path": f"/{task}/{label}.json", "sha256": "a" * 64,
             "size_bytes": 100, "expected_rows": 500000 if label == "sft_train" else 300
             if label == "public_eval" else 100 if label == "inspected_dev100" else 2000 if task == "cvrp" else 3200}
            for task in holdout.TASKS for label in holdout.LABELS
        ],
    }


def cvrp_row(offset=0):
    return {"instance_json": [[[0, 0], [1 + offset, 0], [1, 1]], [0, 1, 1], 2]}


def mis_row(edge=(0, 1)):
    return {"instance_json": {"num_nodes": 5, "edges": [list(edge)]}}


def mis_native(objective=3, bound=3, status="OPTIMAL"):
    return {"objective": objective, "best_bound": bound, "native_status": status,
            "native_status_code": 4 if status == "OPTIMAL" else 2,
            "optimality_certified": status == "OPTIMAL",
            "native_response_stats": f"CpSolverResponse summary:\nstatus: {status}\nobjective: {objective}\nbest_bound: {bound}\n"}


def test_declared_three_hundred_row_budget_and_eight_source_classes():
    holdout.validate_plan(plan())


@pytest.mark.parametrize("field,value", [("instances_per_task", 100), ("workers", 8),
                                         ("task_seeds", {"cvrp": 2026090501, "mis": 2026090502})])
def test_changed_generation_contract_fails(field, value):
    value_plan = plan()
    value_plan[field] = value
    with pytest.raises(ValueError):
        holdout.validate_plan(value_plan)


def test_stale_mis_rl_count_and_missing_development_denominator_fail():
    value = plan()
    next(row for row in value["forbidden_sources"] if row["task"] == "mis" and row["label"] == "rl_train")["expected_rows"] = 2000
    with pytest.raises(ValueError, match="row count"):
        holdout.validate_plan(value)
    value = plan()
    value["forbidden_sources"] = [row for row in value["forbidden_sources"] if row["label"] != "inspected_dev100"]
    with pytest.raises(ValueError, match="eight"):
        holdout.validate_plan(value)


def test_cvrp_reference_uses_raw_float64_round_trip_and_never_claims_optimum():
    value = 2 + math.sqrt(2)
    text, proof = holdout.canonical_reference("cvrp", cvrp_row(), "Routes: [[0, 1, 2, 0]], Objective: 0",
                                             {"objective": value, "optimality_certified": True})
    assert float(text.split("Objective: ")[1]) == value
    assert proof["objective"] == value
    assert proof["reference_quality"] == "verified_heuristic_reference"
    assert proof["optimality_certified"] is False


@pytest.mark.parametrize("routes", ["[[0, 1, 0]]", "[[0, 1, 1, 2, 0]]", "[[0, 1, 9, 0]]", "[[1, 2, 0]]"])
def test_invalid_cvrp_reference_is_rejected(routes):
    with pytest.raises(ValueError, match="infeasible CVRP"):
        holdout.canonical_reference("cvrp", cvrp_row(), f"Routes: {routes}", {"objective": 1.0})


@pytest.mark.parametrize("change", [
    {"best_bound": 4}, {"best_bound": float("nan")}, {"native_status_code": 2},
    {"native_response_stats": "unretained"}, {"native_response_stats": "status: OPTIMAL\nobjective: 3\nbest_bound: 4\n"},
])
def test_mis_certification_requires_matching_native_status_bound_and_response(change):
    with pytest.raises(ValueError, match="optimum assertion"):
        holdout.canonical_reference("mis", mis_row(), "Set: [0, 2, 3]", {**mis_native(), **change})


def test_mis_feasible_incumbent_is_retained_without_optimum_claim():
    text, proof = holdout.canonical_reference("mis", mis_row(), "Set: [0, 2, 3]", mis_native(3, 4, "FEASIBLE"))
    assert text.endswith("Objective: 3")
    assert not proof["optimality_certified"]
    assert proof["reference_quality"] == "verified_feasible_reference"


def test_refinement_keeps_better_incumbent_and_rejects_contradictory_certificate():
    previous = ("Set: [0, 2, 3]", {"strict_verifier_feasible": True, "objective": 3, "optimality_certified": False})
    worse = ("Set: [0, 2]", {"strict_verifier_feasible": True, "objective": 2, "optimality_certified": False})
    assert holdout.choose_mis_reference(previous, worse) == previous
    failure = (None, {"strict_verifier_feasible": False})
    assert holdout.choose_mis_reference(previous, failure) == previous
    with pytest.raises(ValueError, match="contradicts"):
        holdout.choose_mis_reference(previous, (worse[0], {**worse[1], "optimality_certified": True}))


def test_solver_exception_returns_unresolved_row_instead_of_filtering_sample(monkeypatch):
    def fail(*args):
        raise RuntimeError("no solver incumbent")

    monkeypatch.setattr(holdout, "helper", lambda name: SimpleNamespace(solve_mis=fail))
    text, proof = holdout.solve_job(("mis", mis_row(), False))
    assert text is None
    assert proof["reference_quality"] == "unresolved"
    assert not proof["optimality_certified"]
    assert "no solver incumbent" in proof["error"]


def _audit_fixture(tmp_path, monkeypatch, *, overlap=False):
    sources = []
    candidate = {task: holdout.semantic_instance_sha256(task, cvrp_row(50) if task == "cvrp" else mis_row((3, 4)))
                 for task in holdout.TASKS}
    for task in holdout.TASKS:
        for index, label in enumerate(holdout.LABELS):
            row = cvrp_row(index) if task == "cvrp" else mis_row((0, index + 1))
            if overlap and task == "cvrp" and label == "inspected_dev100":
                row = cvrp_row(50)
            path = tmp_path / f"{task}_{label}.json"
            path.write_text(json.dumps([row]))
            sources.append({"task": task, "label": label, "path": str(path), "expected_rows": 1, **holdout.identity(path)})
    small_plan = {"forbidden_sources": sources, "plan_sha256": "a" * 64,
                  "source_files": {name: holdout.identity(holdout.REPO / name) for name in holdout.SOURCES}}
    manifest = {"manifest_sha256": "b" * 64, "tasks": {
        task: {"ordered_semantic_sha256": [candidate[task]]} for task in holdout.TASKS}}
    monkeypatch.setattr(holdout, "validate_build", lambda root: (small_plan, manifest))
    monkeypatch.setattr(holdout, "helper", lambda name: SimpleNamespace(
        iter_json_array=lambda path: enumerate(json.loads(path.read_text()))))
    return SimpleNamespace(output_root=tmp_path, forbidden_sources=None), sources


def test_overlap_audit_scans_all_eight_sources_with_replica_byte_binding(tmp_path, monkeypatch):
    args, sources = _audit_fixture(tmp_path, monkeypatch)
    replicas = copy.deepcopy(sources)
    for index, source in enumerate(replicas):
        target = tmp_path / f"replica{index}.json"
        target.write_bytes(Path(source["path"]).read_bytes())
        source["path"] = str(target)
    args.forbidden_sources = tmp_path / "replicas.json"
    args.forbidden_sources.write_text(json.dumps(replicas))
    holdout.run_audit(args)
    receipt = holdout.checked(tmp_path / "overlap_audit.json", "audit_sha256")
    assert receipt["status"] == "passed_zero_semantic_overlap"
    assert len(receipt["sources"]) == 8
    assert {row["label"] for row in receipt["sources"]} == set(holdout.LABELS)
    with pytest.raises(ValueError, match="create-only"):
        holdout.run_audit(args)


def test_old_dev_overlap_is_retained_in_failed_receipt(tmp_path, monkeypatch):
    args, _ = _audit_fixture(tmp_path, monkeypatch, overlap=True)
    with pytest.raises(ValueError, match="overlaps a forbidden"):
        holdout.run_audit(args)
    receipt = holdout.checked(tmp_path / "overlap_audit.json", "audit_sha256")
    assert receipt["status"] == "failed_semantic_overlap"
    assert receipt["total_overlaps"] == 1
    assert next(row for row in receipt["sources"] if row["overlaps"])["label"] == "inspected_dev100"


def test_missing_source_and_changed_replica_bytes_cannot_pass(tmp_path, monkeypatch):
    args, sources = _audit_fixture(tmp_path, monkeypatch)
    args.forbidden_sources = tmp_path / "registry.json"
    args.forbidden_sources.write_text(json.dumps(sources[:-1]))
    with pytest.raises(ValueError, match="eight"):
        holdout.run_audit(args)
    args.forbidden_sources.write_text(json.dumps(sources))
    Path(sources[0]["path"]).write_text("[]")
    with pytest.raises(ValueError, match="identity differs"):
        holdout.run_audit(args)
    assert not (tmp_path / "overlap_audit.json").exists()


def test_fresh_source_change_during_scan_fails(tmp_path, monkeypatch):
    args, _ = _audit_fixture(tmp_path, monkeypatch)

    def change_during_scan(path):
        yield from enumerate(json.loads(path.read_text()))
        path.write_text("[]")

    monkeypatch.setattr(holdout, "helper", lambda name: SimpleNamespace(iter_json_array=change_during_scan))
    with pytest.raises(ValueError, match="changed while scanned"):
        holdout.run_audit(args)


def test_live_helper_change_is_rejected_even_when_retained_copy_matches(tmp_path, monkeypatch):
    value = plan()
    value["source_files"] = {name: holdout.identity(holdout.REPO / name) for name in holdout.SOURCES}
    holdout.write_new(tmp_path / "plan.json", holdout.seal(value, "plan_sha256"))
    helper_name = "src/clasp/rewards/co_tasks.py"
    live = holdout.REPO / helper_name
    retained = tmp_path / "implementation_sources" / helper_name
    retained.parent.mkdir(parents=True)
    retained.write_bytes(live.read_bytes())
    observed_identity = holdout.identity

    def changed_live_identity(path):
        actual = observed_identity(path)
        return {**actual, "sha256": "0" * 64} if Path(path) == live else actual

    monkeypatch.setattr(holdout, "identity", changed_live_identity)
    assert holdout.identity(retained) == value["source_files"][helper_name]
    with pytest.raises(ValueError, match="executing helper source differs"):
        holdout.validate_build(tmp_path)


def test_audit_rechecks_live_helper_bytes_after_long_source_scan(tmp_path, monkeypatch):
    args, _ = _audit_fixture(tmp_path, monkeypatch)
    observed_identity = holdout.identity
    changed = False
    live = holdout.REPO / "scripts/build_clean_sft500k_corpus.py"

    def identity_after_change(path):
        actual = observed_identity(path)
        return {**actual, "sha256": "0" * 64} if changed and Path(path) == live else actual

    def scan_and_change_live_helper(path):
        nonlocal changed
        yield from enumerate(json.loads(path.read_text()))
        changed = True

    monkeypatch.setattr(holdout, "identity", identity_after_change)
    monkeypatch.setattr(holdout, "helper", lambda name: SimpleNamespace(iter_json_array=scan_and_change_live_helper))
    with pytest.raises(ValueError, match="executing helper source differs"):
        holdout.run_audit(args)
    assert not (tmp_path / "overlap_audit.json").exists()


def test_write_new_and_self_hash_do_not_allow_overwrite_or_silent_tampering(tmp_path):
    path = tmp_path / "receipt.json"
    holdout.write_new(path, holdout.seal({"status": "unadmitted"}, "receipt_sha256"))
    with pytest.raises(FileExistsError):
        holdout.write_new(path, {})
    value = json.loads(path.read_text())
    value["status"] = "passed"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="invalid receipt_sha256"):
        holdout.checked(path, "receipt_sha256")


def test_synthetic_build_freeze_round_trip_and_all_failed_optional_lkh_attempts_are_retained(tmp_path, monkeypatch):
    """Exercise every new phase except streaming with explicit synthetic solvers.

    No real generator or benchmark instance is sampled by this test. The eight
    source-scan receipts are synthetic here; run_audit is tested independently.
    """
    import networkx
    import numpy

    monkeypatch.setattr(numpy, "__version__", "2.3.2")
    monkeypatch.setattr(networkx, "__version__", "3.5")
    monkeypatch.setitem(sys.modules, "ortools", SimpleNamespace(__version__="9.15.6755"))

    class ImmediatePool:
        def __init__(self, *, max_workers):
            assert max_workers == 4

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @staticmethod
        def map(function, values):
            return map(function, values)

    monkeypatch.setattr(holdout, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(holdout, "ThreadPoolExecutor", ImmediatePool)
    counters = {"cvrp": 0, "mis": 0}

    def fake_cvrp(*args):
        index = counters["cvrp"]
        counters["cvrp"] += 1
        return {**cvrp_row(index), "instruction": "synthetic test CVRP", "input": str(index)}, "synthetic"

    def fake_mis(*args):
        index = counters["mis"]
        counters["mis"] += 1
        return {"instance_json": {"num_nodes": 10, "edges": [[0, i + 1] for i in range(9) if index & (1 << i)]},
                "instruction": "synthetic test MIS", "input": str(index)}, "synthetic"

    real_helper = holdout.helper
    monkeypatch.setattr(holdout, "helper", lambda name: SimpleNamespace(_cvrp_row=fake_cvrp, _mis_row=fake_mis)
                        if name == "generate_iid_benchmark" else real_helper(name))

    def fake_solve(job):
        task, row, _ = job
        if task == "cvrp":
            objective = holdout._cvrp_total_distance(row["instance_json"][0], [[0, 1, 2, 0]])
            return holdout.canonical_reference(task, row, "Routes: [[0, 1, 2, 0]]", {"objective": objective})
        vertices = list(range(1, 10)) if row["instance_json"]["edges"] else list(range(10))
        return holdout.canonical_reference(task, row, f"Set: {vertices}", mis_native(len(vertices), len(vertices)))

    monkeypatch.setattr(holdout, "solve_job", fake_solve)

    def fake_lkh(job):
        _binary, parameter, _solution, _log, _instance, seed = job
        return {"seed": seed, "parameter": {"path": str(parameter), **holdout.identity(parameter)},
                "problem": {"path": str(parameter.parent / "problem.vrp"), **holdout.identity(parameter.parent / "problem.vrp")},
                "raw_log": None, "solution": None, "strict_verifier_feasible": False,
                "returncode": 1, "external_timeout": False, "error": "synthetic optional candidate failure"}

    monkeypatch.setattr(holdout, "lkh_job", fake_lkh)
    value = plan()
    value["source_files"] = {name: holdout.identity(holdout.REPO / name) for name in holdout.SOURCES}
    binary = tmp_path / "synthetic_binary"
    binary.write_text("not executable; solver is mocked")
    binary.chmod(0o700)
    value["lkh_binary_identity"] = holdout.identity(binary)
    plan_path = tmp_path / "plan.json"
    holdout.write_new(plan_path, holdout.seal(value, "plan_sha256"))
    args = SimpleNamespace(plan=plan_path, output_root=tmp_path / "heldout", lkh_binary=binary)
    holdout.run_build(args)
    assert counters == {"cvrp": 300, "mis": 300}
    _, manifest = holdout.validate_build(args.output_root)
    assert len(json.loads((args.output_root / "cvrp/lkh_attempts.json").read_text())) == 600
    assert manifest["tasks"]["mis"]["reference_quality_counts"] == {"certified_optimum": 300}
    assert manifest["tasks"]["cvrp"]["reference_quality_counts"] == {"verified_heuristic_reference": 300}
    with pytest.raises(ValueError, match="new directory"):
        holdout.run_build(args)
    stored_plan = holdout.checked(plan_path, "plan_sha256")
    holdout.write_new(args.output_root / "overlap_audit.json", holdout.seal({
        "plan_sha256": stored_plan["plan_sha256"], "build_manifest_sha256": manifest["manifest_sha256"],
        "status": "passed_zero_semantic_overlap", "total_overlaps": 0,
        "audit_source": holdout.identity(Path(holdout.__file__)),
        "sources": [{**row, "rows": row["expected_rows"], "overlaps": []} for row in value["forbidden_sources"]],
    }, "audit_sha256"))
    # A source changed while reference replay runs must be caught before the
    # create-only freeze receipt is written, even if the starting check passed.
    observed_identity = holdout.identity
    original_replay = holdout.canonical_reference
    changed = False
    live = holdout.REPO / "src/clasp/rewards/co_tasks.py"

    def identity_after_replay(path):
        actual = observed_identity(path)
        return {**actual, "sha256": "0" * 64} if changed and Path(path) == live else actual

    def replay_and_change(*args):
        nonlocal changed
        result = original_replay(*args)
        changed = True
        return result

    with monkeypatch.context() as during_freeze:
        during_freeze.setattr(holdout, "identity", identity_after_replay)
        during_freeze.setattr(holdout, "canonical_reference", replay_and_change)
        with pytest.raises(ValueError, match="executing helper source differs"):
            holdout.run_freeze(args)
        assert not (args.output_root / "heldout_manifest.json").exists()
    holdout.run_freeze(args)
    frozen = holdout.checked(args.output_root / "heldout_manifest.json", "heldout_manifest_sha256")
    assert frozen["claim_eligibility"] == "not_confirmation_or_sota"
    assert frozen["model_evaluation_allowed_only_after_separate_preoutput_binding"]
    with pytest.raises(ValueError, match="create-only"):
        holdout.run_freeze(args)


def test_successor_builder_changes_only_plan_self_identity_and_fixed_seed_guards():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    original = (root / "scripts/build_grpo_incumbent_holdout_20260907.py").read_bytes()
    assert hashlib.sha256(original).hexdigest() == "182a624c6b9da3bfb89373d54cfbea46749531e5f3aeaca7f47c57b56db95f31"
    expected = original.decode()
    for previous, current in (
        ("scripts/build_grpo_incumbent_holdout_20260907.py", "scripts/build_grpo_incumbent_holdout_successor_20260907.py"),
        ("grpo-incumbent-two-task300-holdout-20260907-v1", "grpo-incumbent-two-task300-holdout-20260907-v2"),
        ("2026090701", "2026090711"), ("2026090702", "2026090712"),
    ):
        expected = expected.replace(previous, current)
    assert Path(holdout.__file__).read_text() == expected


def test_successor_rejects_previous_generation_identity_and_seeds():
    old_identity = plan()
    old_identity["plan_id"] = "grpo-incumbent-two-task300-holdout-20260907-v1"
    with pytest.raises(ValueError, match="plan/count"):
        holdout.validate_plan(old_identity)
    old_seed = plan()
    old_seed["task_seeds"] = {"cvrp": 2026090701, "mis": 2026090702}
    with pytest.raises(ValueError, match="task seeds"):
        holdout.validate_plan(old_seed)
