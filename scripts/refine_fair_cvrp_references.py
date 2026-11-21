#!/usr/bin/env python3
"""Bounded LKH reference refinement; preserve all prompts and parent versions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from clasp.data.benchmark import semantic_instance_sha256, sha256_json  # noqa: E402
from clasp.rewards.co_tasks import (  # noqa: E402
    _check_cvrp_constraints,
    _cvrp_total_distance,
    _parse_cvrp_solution,
)

SEEDS = (101, 202)
SECONDS = 10.0
WORKERS = 4
DISTANCE_SCALE = 1000
PRECISION = 100


def file_identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def write_new(path: Path, payload: object) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def write_text_new(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(text)


def serialize_cvrp(instance: list, *, name: str) -> str:
    coords, demands, capacity = instance
    count = len(coords)
    if count < 3 or len(demands) != count:
        raise ValueError("CVRP needs at least two customers and matching demands")
    if any(len(coord) != 2 or not all(math.isfinite(float(v)) for v in coord) for coord in coords):
        raise ValueError("CVRP coordinates must be finite pairs")
    if (not math.isfinite(float(capacity)) or int(capacity) != capacity or capacity <= 0
            or any(not math.isfinite(float(d)) or int(d) != d or d < 0 or d > capacity for d in demands)
            or demands[0] != 0):
        raise ValueError("LKH requires integral, feasible demands and capacity with zero depot demand")
    matrix = [[int(round(math.dist(a, b) * DISTANCE_SCALE)) for b in coords] for a in coords]
    # LKH stores scaled edges and intermediate move costs as signed C ints.
    # Leave room for four-edge arithmetic at the requested internal precision.
    if max(map(max, matrix)) * PRECISION > (2**31 - 1) // 4:
        raise ValueError("LKH scaled edge costs exceed the conservative signed-int range")
    lines = [
        f"NAME : {name}", "TYPE : CVRP", f"DIMENSION : {count}",
        f"CAPACITY : {int(capacity)}", f"VEHICLES : {count - 1}",
        "EDGE_WEIGHT_TYPE : EXPLICIT", "EDGE_WEIGHT_FORMAT : FULL_MATRIX", "EDGE_WEIGHT_SECTION",
        *(" ".join(map(str, row)) for row in matrix), "DEMAND_SECTION",
        *(f"{index + 1} {int(demand)}" for index, demand in enumerate(demands)),
        "DEPOT_SECTION", "1", "-1", "EOF", "",
    ]
    return "\n".join(lines)


def parameter_text(problem: Path, solution: Path, *, seed: int, seconds: float = SECONDS) -> str:
    return "\n".join([
        f"PROBLEM_FILE = {problem.resolve()}", f"MTSP_SOLUTION_FILE = {solution.resolve()}",
        f"SEED = {seed}", f"PRECISION = {PRECISION}", "MTSP_MIN_SIZE = 0",
        "RUNS = 1", "MAX_TRIALS = 1000000", "INITIAL_TOUR_ALGORITHM = CVRP",
        f"TIME_LIMIT = {seconds}", f"TOTAL_TIME_LIMIT = {seconds}", "TRACE_LEVEL = 1", "",
    ])


def parse_lkh_routes(text: str, *, node_count: int) -> list[list[int]]:
    lines = text.strip().splitlines()
    if len(lines) < 3 or re.search(r", Cost:\s*0_\d+\s*$", lines[0]) is None:
        raise ValueError("LKH solution does not report a zero-penalty feasible incumbent")
    match = re.fullmatch(r"The tours traveled by the (\d+) salesmen are:", lines[1].strip())
    if match is None or int(match[1]) != node_count - 1:
        raise ValueError("LKH vehicle count does not match the unrestricted reference protocol")
    routes = []
    parsed_count = 0
    for line in lines[2:]:
        match = re.fullmatch(r"\s*([\d\s]+)\s*\(#(\d+)\)\s+Cost:\s*\d+\s*", line)
        if match is None:
            raise ValueError("malformed LKH route line")
        route = [int(value) - 1 for value in match[1].split()]
        if (len(route) < 2 or route[0] != 0 or route[-1] != 0
                or len(route) - 2 != int(match[2])
                or any(value < 0 or value >= node_count for value in route)):
            raise ValueError("LKH route boundary, size, or node index mismatch")
        parsed_count += 1
        if len(route) > 2:
            routes.append(route)
    if parsed_count != node_count - 1:
        raise ValueError("LKH output omits one or more vehicle routes")
    return routes


def solve_candidate(binary: Path, parameter: Path, solution: Path, log: Path,
                    instance: list, *, seconds: float = SECONDS) -> dict:
    started = time.monotonic()
    timed_out = False
    error = None
    returncode = None
    try:
        with log.open("xb") as handle:
            try:
                process = subprocess.run(
                    [str(binary.resolve()), str(parameter.resolve())], stdout=handle,
                    stderr=subprocess.STDOUT, timeout=seconds + 5.0, check=False,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                    cwd=parameter.parent,
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
    except OSError as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    candidate = {
        "returncode": returncode, "external_timeout": timed_out,
        "elapsed_seconds": elapsed, "parameter": file_identity(parameter),
        "raw_log": file_identity(log) if log.exists() else None,
        "solution": file_identity(solution) if solution.exists() else None,
        "strict_verifier_feasible": False, "raw_euclidean_objective": None,
        "reference_routes": None, "error": error,
    }
    if solution.exists() and not timed_out and returncode == 0:
        try:
            routes = parse_lkh_routes(solution.read_text(), node_count=len(instance[0]))
            if not _check_cvrp_constraints(routes, *instance)[2]:
                raise ValueError("LKH incumbent failed the repository's strict CVRP verifier")
            objective = float(_cvrp_total_distance(instance[0], routes))
            if not math.isfinite(objective):
                raise ValueError("LKH raw objective is nonfinite")
            candidate.update(strict_verifier_feasible=True, raw_euclidean_objective=objective,
                             reference_routes=routes)
        except (ValueError, TypeError, IndexError) as exc:
            candidate["error"] = f"{type(exc).__name__}: {exc}"
    elif error is None:
        candidate["error"] = (
            "LKH exceeded its external timeout" if timed_out else
            "LKH exited unsuccessfully" if returncode != 0 else
            "LKH did not produce an incumbent solution file"
        )
    for path in (log, solution):
        if path.exists():
            path.chmod(0o440)
    return candidate


def choose_reference(row: dict, parent: dict, candidates: list[dict]) -> tuple[dict, dict]:
    instance = row["instance_json"]
    original_routes = _parse_cvrp_solution(row["output"])["routes"]
    if not _check_cvrp_constraints(original_routes, *instance)[2]:
        raise ValueError("parent CVRP reference is infeasible")
    original_objective = float(_cvrp_total_distance(instance[0], original_routes))
    if not math.isclose(original_objective, parent["objective"], rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("parent objective does not match strict raw recomputation")
    best_objective, best_routes, chosen = original_objective, original_routes, "parent"
    for candidate in candidates:
        if candidate["strict_verifier_feasible"] and candidate["raw_euclidean_objective"] < best_objective:
            best_objective = candidate["raw_euclidean_objective"]
            best_routes, chosen = candidate["reference_routes"], f"lkh_seed{candidate['seed']}"
    updated = dict(row)
    if chosen != "parent":
        updated["output"] = f"Routes: {best_routes}, Objective: {best_objective:.17g}"
    evidence = dict(parent)
    evidence.pop("evidence_sha256", None)
    evidence.update(
        objective=best_objective, reference_routes=best_routes, strict_verifier_feasible=True,
        optimality_certified=False, reference_quality="verified_heuristic_reference",
        objective_semantics="raw_float64_euclidean_distance",
        parent_reference_objective=original_objective, reference_selected_from=chosen,
        refinement_improvement=original_objective - best_objective, lkh_candidates=candidates,
    )
    return updated, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--lkh-binary", type=Path, required=True)
    args = parser.parse_args()
    parent, output, evidence_root = (p.resolve() for p in (args.parent_root, args.output_root, args.evidence_root))
    binary = args.lkh_binary.resolve(strict=True)
    if output.exists() or evidence_root.exists():
        raise FileExistsError("refinement outputs must be new paths")
    if not os.access(binary, os.X_OK):
        raise ValueError("LKH binary is not executable")
    parent_manifest = json.loads((parent / "development_manifest.json").read_text())
    parent_hash = parent_manifest.pop("manifest_sha256")
    if sha256_json(parent_manifest) != parent_hash:
        raise ValueError("invalid parent manifest hash")
    rows = json.loads((parent / "cvrp/eval/test.json").read_text())
    parent_evidence = json.loads((parent / "cvrp/reference_evidence.json").read_text())
    if len(rows) != len(parent_evidence) or len(rows) != 100:
        raise ValueError("the fixed development refinement requires all 100 CVRP rows")
    evidence_root.mkdir(parents=True, exist_ok=False)
    (evidence_root / "runs").mkdir()
    source_snapshot = evidence_root / "sources"
    source_snapshot.mkdir()
    for source in (Path(__file__), REPO / "src/clasp/rewards/co_tasks.py", binary,
                   *(binary.parent / "SRC" / name for name in
                     ("ReadProblem.c", "ReadParameters.c", "Penalty_CVRP.c", "MTSP_WriteSolution.c"))):
        copied = source_snapshot / source.name
        shutil.copyfile(source, copied)
        copied.chmod(0o440)
    workload = []
    # Serialize and identity every solver input before any benchmark solver runs.
    for index, (row, old) in enumerate(zip(rows, parent_evidence)):
        instance_hash = semantic_instance_sha256("cvrp", row, require_structured=True)
        if old["index"] != index or old["semantic_instance_sha256"] != instance_hash:
            raise ValueError("parent evidence/row identity mismatch")
        directory = evidence_root / "runs" / f"instance_{index:03d}"
        directory.mkdir()
        problem = directory / "problem.vrp"
        write_text_new(problem, serialize_cvrp(row["instance_json"], name=f"fair_cvrp_{index:03d}"))
        trials = []
        for seed in SEEDS:
            parameter, solution, log = (directory / f"seed{seed}.{suffix}" for suffix in ("par", "sol", "log"))
            write_text_new(parameter, parameter_text(problem, solution, seed=seed))
            trials.append({"seed": seed, "parameter": file_identity(parameter),
                           "solution_path": str(solution), "raw_log_path": str(log)})
        workload.append({"index": index, "semantic_instance_sha256": instance_hash,
                         "problem": file_identity(problem), "trials": trials})
    plan = {
        "schema_version": 1, "purpose": "development_reference_refinement_before_model_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "parent_manifest_sha256": parent_hash,
        "parent_dataset": file_identity(parent / "cvrp/eval/test.json"),
        "parent_reference_evidence": file_identity(parent / "cvrp/reference_evidence.json"),
        "lkh_binary": file_identity(binary), "refinement_script": file_identity(Path(__file__)),
        "verifier_source": file_identity(REPO / "src/clasp/rewards/co_tasks.py"),
        "source_files": [file_identity(binary.parent / "SRC" / name) for name in
                         ("ReadProblem.c", "ReadParameters.c", "Penalty_CVRP.c", "MTSP_WriteSolution.c")],
        "immutable_source_copies": [file_identity(path) for path in sorted(source_snapshot.iterdir())],
        "parameters": {"seeds": list(SEEDS), "time_seconds_per_seed": SECONDS,
                       "external_timeout_grace_seconds": 5.0, "cpu_workers": WORKERS,
                       "vehicles": "number_of_customers", "mtsp_min_size": 0,
                       "integer_distance_scale": DISTANCE_SCALE, "lkh_precision": PRECISION,
                       "runs_per_seed": 1, "max_trials": 1000000},
        "selection": "minimum raw Euclidean cost among parent and all strictly verified LKH candidates",
        "failure_policy": "retain parent, record solver errors, never drop an instance",
        "changes_allowed": ["CVRP reference output", "CVRP reference evidence", "derived manifest and overlap audit"],
        "model_outputs_examined": False, "workload": workload,
    }
    plan["plan_sha256"] = sha256_json(plan)
    write_new(evidence_root / "refinement_plan.json", plan)
    print(json.dumps({"status": "plan_frozen", "plan_sha256": plan["plan_sha256"],
                      "instances": len(rows), "trials": len(rows) * len(SEEDS)}), flush=True)

    def process_row(index: int) -> tuple[int, dict, dict]:
        item, candidates = workload[index], []
        for trial in item["trials"]:
            result = solve_candidate(binary, Path(trial["parameter"]["path"]),
                                     Path(trial["solution_path"]), Path(trial["raw_log_path"]),
                                     rows[index]["instance_json"])
            result["seed"] = trial["seed"]
            candidates.append(result)
        row, evidence = choose_reference(rows[index], parent_evidence[index], candidates)
        evidence["refinement_plan_sha256"] = plan["plan_sha256"]
        evidence["evidence_sha256"] = sha256_json(evidence)
        write_new(evidence_root / "runs" / f"instance_{index:03d}" / "result.json", evidence)
        return index, row, evidence

    refined_rows, refined_evidence = [None] * len(rows), [None] * len(rows)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(process_row, index) for index in range(len(rows))]
        for completed, future in enumerate(as_completed(futures), 1):
            index, row, evidence = future.result()
            refined_rows[index], refined_evidence[index] = row, evidence
            print(json.dumps({"completed": completed, "index": index,
                              "reference_selected_from": evidence["reference_selected_from"],
                              "raw_improvement": evidence["refinement_improvement"]}), flush=True)
    for before, after in zip(rows, refined_rows):
        if {k: v for k, v in before.items() if k != "output"} != {k: v for k, v in after.items() if k != "output"}:
            raise RuntimeError("a frozen instance or prompt field changed")
    # The evidence agent will create the final v3 manifest and overlap binding.
    shutil.copytree(parent, output, ignore=shutil.ignore_patterns(
        "development_manifest.json", "training_overlap_audit.json", "README.md"))
    for path, payload in ((output / "cvrp/eval/test.json", refined_rows),
                          (output / "cvrp/reference_evidence.json", refined_evidence)):
        path.unlink()
        write_new(path, payload)
    all_candidates = [candidate for evidence in refined_evidence for candidate in evidence["lkh_candidates"]]
    summary = {
        "schema_version": 1, "status": "reference_refinement_complete_pending_manifest_and_overlap_rebind",
        "parent_manifest_sha256": parent_hash, "plan_sha256": plan["plan_sha256"],
    }
    summary.update(
        instance_count=len(rows), improved_instances=sum(e["refinement_improvement"] > 0 for e in refined_evidence),
        parent_retained_instances=sum(e["reference_selected_from"] == "parent" for e in refined_evidence),
        strictly_verified_lkh_candidates=sum(c["strict_verifier_feasible"] for c in all_candidates),
        failed_candidates=sum(not c["strict_verifier_feasible"] for c in all_candidates),
        external_timeouts=sum(c["external_timeout"] for c in all_candidates),
        mean_relative_reference_improvement=sum(e["refinement_improvement"] / e["parent_reference_objective"]
                                                for e in refined_evidence) / len(rows),
        dataset=file_identity(output / "cvrp/eval/test.json"),
        reference_evidence=file_identity(output / "cvrp/reference_evidence.json"),
        all_instance_and_prompt_fields_unchanged=True, mis_evidence_unchanged=True,
        output_root=str(output), evidence_root=str(evidence_root),
    )
    summary["receipt_sha256"] = sha256_json(summary)
    write_new(evidence_root / "refinement_result.json", summary)
    write_new(output / "cvrp_reference_refinement_plan.json", plan)
    write_new(output / "cvrp_reference_refinement_result.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
