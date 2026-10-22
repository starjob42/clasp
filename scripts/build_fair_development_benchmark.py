#!/usr/bin/env python3
"""Build a fresh CVRP/MIS development split and audit semantic training overlap.

This has no confirmation/publication side effects. Generation settings are
written before any instance or solution is produced. References and native
solver evidence are retained even when a MIS optimum is not certified.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from clasp.data.benchmark import semantic_instance_sha256, sha256_json
from clasp.rewards.co_tasks import (
    _check_cvrp_constraints,
    _check_mis_constraints,
    _cvrp_total_distance,
    _parse_cvrp_solution,
)


def import_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def write_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def solve_cvrp(row: dict, seconds: float, heuristic_backend) -> tuple[str, dict]:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    coords, demands, capacity = row["instance_json"]
    count = len(coords)
    if max(demands) > capacity:
        raise ValueError("generated instance has an individually unsatisfiable demand")
    heuristic = heuristic_backend.solve(row, parameters={"max_2opt_passes": 100})
    initial_routes = _parse_cvrp_solution(heuristic.output)["routes"]
    best_routes = initial_routes
    best_objective = float(heuristic.incumbent)
    scale = 1_000_000
    distances = [[int(round(math.dist(a, b) * scale)) for b in coords] for a in coords]
    manager = pywrapcp.RoutingIndexManager(count, count - 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    distance_callback = routing.RegisterTransitCallback(
        lambda left, right: distances[manager.IndexToNode(left)][manager.IndexToNode(right)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)
    demand_callback = routing.RegisterUnaryTransitCallback(lambda index: int(demands[manager.IndexToNode(index)]))
    routing.AddDimensionWithVehicleCapacity(demand_callback, 0, [int(capacity)] * (count - 1), True, "Capacity")
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.time_limit.FromMilliseconds(int(seconds * 1000))
    parameters.log_search = False
    started = time.monotonic()
    assignment = routing.ReadAssignmentFromRoutes([r[1:-1] for r in initial_routes], True)
    solution = (routing.SolveFromAssignmentWithParameters(assignment, parameters)
                if assignment is not None else routing.SolveWithParameters(parameters))
    ortools_objective = None
    if solution is not None:
        routes = []
        for vehicle in range(count - 1):
            index = routing.Start(vehicle)
            route = [0]
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))
            if len(route) > 2:
                routes.append(route)
        assert _check_cvrp_constraints(routes, coords, demands, capacity)[2]
        ortools_objective = float(_cvrp_total_distance(coords, routes))
        if ortools_objective < best_objective:
            best_routes, best_objective = routes, ortools_objective
    assert _check_cvrp_constraints(best_routes, coords, demands, capacity)[2]
    best_objective = float(_cvrp_total_distance(coords, best_routes))
    evidence = {
        "task": "cvrp", "reference_quality": "verified_heuristic_reference",
        "optimality_certified": False, "objective": best_objective, "best_bound": None,
        "native_status_code": int(routing.status()), "ortools_candidate_raw_objective": ortools_objective,
        "clarke_wright_raw_objective": float(heuristic.incumbent),
        "objective_semantics": "raw_float64_euclidean_distance",
        "search_objective_semantics": "Euclidean distances rounded after multiplying by 1000000",
        "integer_distance_scale": scale, "search_wall_limit_seconds": seconds,
        "elapsed_seconds": time.monotonic() - started,
        "reference_routes": best_routes,
        "strict_verifier_feasible": True,
    }
    return f"Routes: {best_routes}, Objective: {best_objective:.12f}", evidence


def solve_mis(row: dict, deterministic_seconds: float, wall_seconds: float, seed: int) -> tuple[str, dict]:
    from ortools.sat.python import cp_model

    instance = row["instance_json"]
    count = int(instance["num_nodes"])
    edges = {tuple(sorted(edge)) for edge in instance["edges"]}
    model = cp_model.CpModel()
    variables = [model.new_bool_var(f"vertex_{i}") for i in range(count)]
    for left, right in edges:
        model.add(variables[left] + variables[right] <= 1)
    model.maximize(sum(variables))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    solver.parameters.max_deterministic_time = deterministic_seconds
    solver.parameters.max_time_in_seconds = wall_seconds
    solver.parameters.relative_gap_limit = 0.0
    solver.parameters.absolute_gap_limit = 0.0
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"MIS solver returned no incumbent: {solver.status_name(status)}")
    chosen = [i for i, variable in enumerate(variables) if solver.value(variable)]
    assert _check_mis_constraints(chosen, count, edges)[2]
    objective = len(chosen)
    bound = float(solver.best_objective_bound)
    certified = status == cp_model.OPTIMAL and objective == round(bound) and abs(bound - round(bound)) < 1e-8
    evidence = {
        "task": "mis", "reference_quality": "certified_optimum" if certified else "verified_feasible_reference",
        "optimality_certified": certified, "objective": objective, "best_bound": bound,
        "native_status": solver.status_name(status), "native_status_code": int(status),
        "native_response_stats": solver.response_stats(), "reference_set": chosen,
        "objective_semantics": "integer_independent_set_cardinality",
        "strict_verifier_feasible": True,
    }
    return f"Set: {chosen}, Objective: {objective}", evidence


def solve_reference_job(job: tuple) -> tuple[str, dict]:
    task, row, cvrp_seconds, mis_deterministic_seconds, mis_wall_seconds = job
    if task == "cvrp":
        references = import_script("run_reference_evidence")
        return solve_cvrp(row, cvrp_seconds, references.RawCVRPHeuristicBackend())
    return solve_mis(row, mis_deterministic_seconds, mis_wall_seconds, 20260905)


def build(args) -> None:
    if args.ortools_site_packages:
        sys.path.append(str(Path(args.ortools_site_packages).resolve()))
    import numpy as np
    import networkx
    import ortools

    generator = import_script("generate_iid_benchmark")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema_version": 1, "purpose": "fresh_development_only",
        "claim_eligibility": "not_confirmation_or_sota",
        "generated_before_model_outputs": True,
        "instances_per_task": args.instances, "seeds": {"cvrp": args.cvrp_seed, "mis": args.mis_seed},
        "reference_worker_processes": args.workers,
        "distributions": {"cvrp": "10..100 customers uniform; equal uniform/2-mode/3-mode coordinate mixtures; released capacity/demand process",
                          "mis": "10..100 nodes uniform; equal ER/BA mixture; ER p~U(.1,.4), BA attachment~Uniform{1,2,3,4}"},
        "generator": file_identity(REPO / "scripts/generate_iid_benchmark.py"),
        "builder": file_identity(Path(__file__)),
        "strict_verifier": file_identity(REPO / "src/clasp/rewards/co_tasks.py"),
        "reference_helper": file_identity(REPO / "scripts/run_reference_evidence.py"),
        "semantic_identity": file_identity(REPO / "src/clasp/data/benchmark.py"),
        "runtime": {"python": sys.version, "numpy": np.__version__, "networkx": networkx.__version__, "ortools": ortools.__version__},
        "cvrp_solver": {"wall_seconds_per_instance": args.cvrp_seconds, "integer_distance_scale": 1_000_000,
                        "initialization": "deterministic Clarke-Wright and raw-distance 2-opt", "selection": "best raw-distance strictly feasible reference"},
        "mis_solver": {"workers": 1, "seed": 20260905, "deterministic_time_limit": args.mis_deterministic_seconds,
                       "wall_seconds_limit": args.mis_wall_seconds, "require_certified_for_optimum_label": True,
                       "retain_uncertified_instances": True},
        "model_evaluation": {"candidate_budget": 8, "seed": 271828, "dtype": "float16", "temperature": .7,
                             "top_p": .9, "top_k": 50, "repetition_penalty": 1., "prompt_style": "jiang_f638",
                             "no_reason": True, "max_new_tokens": 3000, "max_prompt_length": 29768},
    }
    plan["plan_sha256"] = sha256_json(plan)
    write_new(root / "generation_plan.json", plan)
    manifest = {"schema_version": 1, "purpose": "fresh_development_only", "plan_sha256": plan["plan_sha256"],
                "status": "generated_pending_remote_overlap_audit", "tasks": {}}
    for task in ("cvrp", "mis"):
        seed = plan["seeds"][task]
        py_rng, np_rng = random.Random(seed), np.random.RandomState(seed)
        rows, evidence, seen, pending = [], [], set(), []
        for index in range(args.instances):
            row, distribution = getattr(generator, f"_{task}_row")(py_rng, np_rng)
            identity = semantic_instance_sha256(task, row, require_structured=True)
            if identity in seen:
                raise RuntimeError("generated duplicate instance; refusing silent replacement")
            seen.add(identity)
            pending.append((row, identity, distribution))
        jobs = [(task, row, args.cvrp_seconds, args.mis_deterministic_seconds, args.mis_wall_seconds)
                for row, _identity, _distribution in pending]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, ((row, identity, distribution), (output, record)) in enumerate(
                zip(pending, executor.map(solve_reference_job, jobs), strict=True)
            ):
                row["output"] = output
                row["development_instance_id"] = f"fresh_{task}_{seed}_{index}"
                record.update({"index": index, "semantic_instance_sha256": identity, "distribution": distribution})
                record["evidence_sha256"] = sha256_json(record)
                rows.append(row)
                evidence.append(record)
                if (index + 1) % 10 == 0:
                    print(f"{task}: solved {index + 1}/{args.instances}", flush=True)
        output_path = root / task / "eval/test.json"
        write_new(output_path, rows)
        evidence_path = root / task / "reference_evidence.json"
        write_new(evidence_path, evidence)
        manifest["tasks"][task] = {"instances": len(rows), "dataset": file_identity(output_path),
            "reference_evidence": file_identity(evidence_path), "unique_semantic_instances": len(seen),
            "semantic_instance_sha256": sorted(seen), "reference_quality_counts": dict(Counter(r["reference_quality"] for r in evidence)),
            "distribution_counts": dict(Counter(r["distribution"] for r in evidence)),
            "strictly_feasible_references": len(evidence)}
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_new(root / "development_manifest.json", manifest)
    print(f"Generated development dataset: {root}; semantic overlap audit remains required.", flush=True)


def audit(args) -> None:
    root = args.output_root.resolve()
    manifest = json.loads((root / "development_manifest.json").read_text())
    expected = manifest.pop("manifest_sha256")
    assert sha256_json(manifest) == expected
    sources = json.loads(args.forbidden_sources.read_text())
    stream = import_script("build_clean_sft500k_corpus").iter_json_array
    records = []
    for task in ("cvrp", "mis"):
        candidates = set(manifest["tasks"][task]["semantic_instance_sha256"])
        dataset = root / task / "eval/test.json"
        assert file_identity(dataset)["sha256"] == manifest["tasks"][task]["dataset"]["sha256"]
        entries = sources[task]
        if {entry["label"] for entry in entries} != {"sft_train", "rl_train", "public_eval"}:
            raise ValueError(f"{task} audit requires SFT train, RL train, and public eval sources")
        for source in entries:
            path = Path(source["path"])
            expected_rows = source.get("expected_rows")
            if type(expected_rows) is not int or expected_rows <= 0:
                raise ValueError("each forbidden source needs a positive expected_rows")
            before = file_identity(path)
            count, overlaps, parsed_digest = 0, [], hashlib.sha256()
            for index, row in stream(path):
                identity = semantic_instance_sha256(task, row)
                parsed_digest.update((identity + "\n").encode())
                if identity in candidates:
                    overlaps.append({"row_index": index, "semantic_instance_sha256": identity})
                count += 1
                if count % 50_000 == 0:
                    print(f"overlap audit {task}/{source['label']}: {count} rows", flush=True)
            after = file_identity(path)
            assert before == after, "forbidden source changed while it was being audited"
            if count != expected_rows:
                raise ValueError(f"{task}/{source['label']} has {count} rows, expected {expected_rows}")
            records.append({"task": task, "label": source["label"], "source": before, "rows": count,
                            "ordered_semantic_digest": parsed_digest.hexdigest(), "overlaps": overlaps})
            print(f"Audited {task}/{source['label']}: rows={count}, overlaps={len(overlaps)}", flush=True)
    result = {"purpose": "fresh_development_only", "development_manifest_sha256": expected,
              "audit_implementation": file_identity(Path(__file__)),
              "semantic_identity_implementation": file_identity(REPO / "src/clasp/data/benchmark.py"),
              "source_spec": file_identity(args.forbidden_sources), "sources": records,
              "status": "passed_zero_semantic_overlap" if all(not r["overlaps"] for r in records) else "failed_semantic_overlap",
              "total_overlaps": sum(len(r["overlaps"]) for r in records)}
    result["audit_sha256"] = sha256_json(result)
    write_new(root / "training_overlap_audit.json", result)
    if result["total_overlaps"]:
        raise RuntimeError("Fresh benchmark overlaps declared training/development data")
    print("Fresh benchmark passed exact semantic overlap checks.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--cvrp-seed", type=int, default=2026090501)
    parser.add_argument("--mis-seed", type=int, default=2026090502)
    parser.add_argument("--cvrp-seconds", type=float, default=10.)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mis-deterministic-seconds", type=float, default=5.)
    parser.add_argument("--mis-wall-seconds", type=float, default=30.)
    parser.add_argument("--ortools-site-packages", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--forbidden-sources", type=Path)
    args = parser.parse_args()
    if args.audit_only:
        if args.forbidden_sources is None:
            parser.error("--audit-only requires --forbidden-sources")
        audit(args)
    else:
        if args.instances <= 0 or args.workers <= 0 or min(args.cvrp_seconds, args.mis_deterministic_seconds, args.mis_wall_seconds) <= 0:
            parser.error("instance count and solver budgets must be positive")
        build(args)


if __name__ == "__main__":
    main()
