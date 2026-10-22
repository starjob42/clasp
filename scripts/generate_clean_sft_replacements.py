#!/usr/bin/env python3
"""Generate solver-backed, strict-valid replacement rows for clean SFT data.

CVRP uses the audited LKH-3.0.8 binary with a fixed nonzero seed, private
per-row files, and raw-Euclidean strict replay.  OP uses the repository's
raw-Euclidean Gurobi open-path model but accepts the best retained incumbent
for SFT supervision (never as an optimality certificate).  MIS accepts only a
Gurobi-certified optimum.  Every native input, output, log, status, and strict
replay is retained under a create-once shard directory.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import math
import os
import random
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent


def _load_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import script module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


clean = _load_script_module(
    "build_clean_sft500k_corpus", SCRIPT_DIR / "build_clean_sft500k_corpus.py"
)
_GENERATOR_MODULE: Any | None = None
_REFERENCE_MODULE: Any | None = None


def _generator_module() -> Any:
    global _GENERATOR_MODULE
    if _GENERATOR_MODULE is None:
        _GENERATOR_MODULE = _load_script_module(
            "generate_iid_benchmark_for_sft_replacements",
            SCRIPT_DIR / "generate_iid_benchmark.py",
        )
    return _GENERATOR_MODULE


def _reference_module() -> Any:
    global _REFERENCE_MODULE
    if _REFERENCE_MODULE is None:
        _REFERENCE_MODULE = _load_script_module(
            "run_reference_evidence_for_sft_replacements",
            SCRIPT_DIR / "run_reference_evidence.py",
        )
    return _REFERENCE_MODULE


SCHEMA_VERSION = 1
PROTOCOL = "clasp-solver-backed-sft-replacement-shard-v1"
TASKS = ("cvrp", "mis", "op")
PINNED_LKH_SHA256 = "87016cd10c3739d80fd4aa7e2517b2bc3ddeeb673a23de0791bab6ecd9dbafc4"


class ReplacementGenerationError(RuntimeError):
    """Raised when a solver-backed replacement cannot be certified."""


def _emit_phase_event(event: str, **fields: object) -> None:
    payload = {
        "event": event,
        "observed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "monotonic_seconds": time.monotonic(),
        **fields,
    }
    print(
        "[ReplacementPhase] "
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        file=sys.stderr,
        flush=True,
    )


def _validated_op_local_temp_root(
    path: Path | None,
    *,
    require_empty: bool,
) -> Path:
    """Return a private, canonical node-local directory for Gurobi's live log."""

    if path is None:
        raise ReplacementGenerationError("OP requires an explicit node-local temp root")
    if not path.is_absolute():
        raise ReplacementGenerationError("OP local temp root must be absolute")
    try:
        canonical = path.resolve(strict=True)
        system_tmp = Path("/tmp").resolve(strict=True)
    except OSError as exc:
        raise ReplacementGenerationError(f"cannot resolve OP local temp root: {exc}") from exc
    if canonical != path:
        raise ReplacementGenerationError("OP local temp root must be canonical and normalized")
    try:
        relative = canonical.relative_to(system_tmp)
    except ValueError as exc:
        raise ReplacementGenerationError("OP local temp root must be below /tmp") from exc
    if not relative.parts:
        raise ReplacementGenerationError("OP local temp root cannot be /tmp itself")
    observed = canonical.stat(follow_symlinks=False)
    if not stat.S_ISDIR(observed.st_mode) or path.is_symlink():
        raise ReplacementGenerationError("OP local temp root must be a regular directory")
    if observed.st_uid != os.geteuid():
        raise ReplacementGenerationError("OP local temp root must be owned by the launch user")
    if stat.S_IMODE(observed.st_mode) != 0o700:
        raise ReplacementGenerationError("OP local temp root mode must be exactly 0700")
    if not os.access(canonical, os.W_OK | os.X_OK):
        raise ReplacementGenerationError("OP local temp root is not writable and searchable")
    if require_empty:
        with os.scandir(canonical) as entries:
            if next(entries, None) is not None:
                raise ReplacementGenerationError("OP local temp root must initially be empty")
    return canonical


def _require_op_local_capacity(
    root: Path,
    *,
    count: int,
    max_attempt_multiplier: int,
) -> dict[str, int | None]:
    observed = os.statvfs(root)
    fragment_size = int(observed.f_frsize or observed.f_bsize)
    available_bytes = int(observed.f_bavail) * fragment_size
    maximum_attempts = int(count) * int(max_attempt_multiplier)
    required_bytes = max(256 * 1024 * 1024, maximum_attempts * 2 * 1024 * 1024)
    if available_bytes < required_bytes:
        raise ReplacementGenerationError(
            "OP node-local temp root has insufficient free byte capacity"
        )
    required_inodes = maximum_attempts * 8 + 128
    reported_free_inodes = int(observed.f_favail)
    if reported_free_inodes >= 0 and reported_free_inodes < required_inodes:
        raise ReplacementGenerationError(
            "OP node-local temp root has insufficient free inode capacity"
        )
    return {
        "required_available_bytes": required_bytes,
        "observed_available_bytes": available_bytes,
        "required_free_inodes": required_inodes,
        "maximum_attempts_budgeted": maximum_attempts,
        "observed_free_inodes": (
            reported_free_inodes if reported_free_inodes >= 0 else None
        ),
    }


def _write_bytes_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_new(path: Path, value: object) -> None:
    _write_bytes_new(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n",
    )


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "size_bytes": path.stat(follow_symlinks=False).st_size,
        "sha256": clean.sha256_file(path),
    }
    if relative_to is not None:
        record["relative_path"] = path.relative_to(relative_to).as_posix()
    else:
        record["canonical_path"] = str(path.resolve(strict=True))
    return record


def _candidate_seed(base_seed: int, task: str, candidate_number: int) -> int:
    payload = f"{base_seed}:{task}:{candidate_number}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def generate_candidate(task: str, *, base_seed: int, candidate_number: int) -> dict[str, Any]:
    import numpy as np

    seed = _candidate_seed(base_seed, task, candidate_number)
    py_rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    row, distribution = _generator_module().GENERATORS[task](py_rng, np_rng)
    row["replacement_generation"] = {
        "protocol": PROTOCOL,
        "base_seed": base_seed,
        "candidate_number": candidate_number,
        "derived_seed": seed,
        "generator_distribution": distribution,
    }
    return row


def _lkh_write_problem(path: Path, row: Mapping[str, Any]) -> None:
    instance = row.get("instance_json")
    if not isinstance(instance, list) or len(instance) < 3:
        raise ReplacementGenerationError("CVRP replacement has no structured instance")
    coords, demands, capacity = instance[:3]
    if not isinstance(coords, list) or not isinstance(demands, list):
        raise ReplacementGenerationError("CVRP replacement has malformed coordinates/demands")
    if len(coords) != len(demands) or len(coords) < 2:
        raise ReplacementGenerationError("CVRP replacement dimensions are inconsistent")
    lines = [
        "NAME : clasp_sft_replacement",
        "TYPE : CVRP",
        f"DIMENSION : {len(coords)}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        f"CAPACITY : {int(capacity)}",
        "NODE_COORD_SECTION",
    ]
    lines.extend(
        f"{index + 1} {int(point[0])} {int(point[1])}"
        for index, point in enumerate(coords)
    )
    lines.append("DEMAND_SECTION")
    lines.extend(f"{index + 1} {int(value)}" for index, value in enumerate(demands))
    lines.extend(("DEPOT_SECTION", "1", "-1", "EOF"))
    _write_bytes_new(path, ("\n".join(lines) + "\n").encode("ascii"))


def _lkh_parse_routes(path: Path, *, customers: int) -> list[list[int]]:
    dimension: int | None = None
    sequence: list[int] = []
    started = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("DIMENSION"):
            dimension = int(line.replace(":", " ").split()[-1])
        elif line == "TOUR_SECTION":
            started = True
        elif started:
            value = int(line)
            if value == -1:
                break
            sequence.append(value - 1)
    if dimension is None or len(sequence) != dimension:
        raise ReplacementGenerationError(
            f"LKH tour length mismatch: dimension={dimension}, rows={len(sequence)}"
        )
    # IDs above the real customer range are LKH's virtual depot copies.
    mapped = [0 if value > customers else value for value in sequence]
    if not mapped or mapped[0] != 0:
        raise ReplacementGenerationError("LKH tour does not begin at the depot")
    routes: list[list[int]] = []
    current = [0]
    for value in mapped[1:]:
        current.append(value)
        if value == 0:
            if len(current) > 2:
                routes.append(current)
            current = [0]
    if len(current) > 1:
        current.append(0)
        routes.append(current)
    if not routes:
        raise ReplacementGenerationError("LKH returned no CVRP routes")
    return routes


def _cvrp_raw_objective(row: Mapping[str, Any], routes: Sequence[Sequence[int]]) -> float:
    instance = row["instance_json"]
    assert isinstance(instance, list)
    coords = instance[0]
    assert isinstance(coords, list)
    return sum(
        math.dist(coords[left], coords[right])
        for route in routes
        for left, right in zip(route, route[1:])
    )


def solve_cvrp_lkh(
    row: dict[str, Any],
    *,
    row_dir: Path,
    executable: Path,
    timeout_seconds: float,
) -> tuple[str, dict[str, Any]]:
    problem = row_dir / "problem.vrp"
    params = row_dir / "parameters.par"
    tour = row_dir / "solution.tour"
    stdout_path = row_dir / "native_stdout.log"
    stderr_path = row_dir / "native_stderr.log"
    _lkh_write_problem(problem, row)
    _write_bytes_new(
        params,
        (
            f"PROBLEM_FILE = {problem}\n"
            f"OUTPUT_TOUR_FILE = {tour}\n"
            "SPECIAL\nRUNS = 10\nTRACE_LEVEL = 1\nSEED = 1\n"
        ).encode("utf-8"),
    )
    started = time.monotonic()
    result = subprocess.run(
        [str(executable), str(params)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_seconds,
    )
    elapsed = time.monotonic() - started
    _write_bytes_new(stdout_path, result.stdout)
    _write_bytes_new(stderr_path, result.stderr)
    if result.returncode != 0 or not tour.is_file():
        raise ReplacementGenerationError(
            f"LKH failed with return code {result.returncode}"
        )
    instance = row["instance_json"]
    assert isinstance(instance, list) and isinstance(instance[0], list)
    routes = _lkh_parse_routes(tour, customers=len(instance[0]) - 1)
    objective = _cvrp_raw_objective(row, routes)
    output = f"Routes: {routes}, Objective: {objective:.17g}"
    return output, {
        "native_status": "HEURISTIC_COMPLETE",
        "native_status_code": result.returncode,
        "native_wall_time_seconds": elapsed,
        "native_incumbent_raw_euclidean": objective,
        "native_best_bound": None,
        "native_gap": None,
        "search_objective_semantics": "TSPLIB_EUC_2D_rounded",
        "training_objective_semantics": "clasp_raw_euclidean_float64",
        "fixed_seed": 1,
        "runs": 10,
        "artifacts": {
            path.name: _file_record(path, relative_to=row_dir.parent.parent)
            for path in (problem, params, tour, stdout_path, stderr_path)
        },
    }


def solve_mis_gurobi(
    row: dict[str, Any], *, row_dir: Path, timeout_seconds: float
) -> tuple[str, dict[str, Any]]:
    backend = _reference_module().GurobiMISBackend()
    result = backend.solve(
        row,
        parameters={
            "time_limit_seconds": timeout_seconds,
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
    )
    log_path = row_dir / "gurobi.log"
    _write_bytes_new(log_path, result.raw_log)
    if result.native_status != "OPTIMAL" or result.native_status_code != 2:
        raise ReplacementGenerationError(
            f"MIS did not certify optimality: {result.native_status}"
        )
    if result.incumbent is None or result.best_bound is None:
        raise ReplacementGenerationError("MIS optimum has no incumbent/bound")
    if round(float(result.incumbent)) != round(float(result.best_bound)):
        raise ReplacementGenerationError("MIS optimum incumbent does not equal bound")
    return result.output, {
        "native_status": result.native_status,
        "native_status_code": result.native_status_code,
        "native_wall_time_seconds": result.native_wall_time_seconds,
        "native_incumbent": result.incumbent,
        "native_best_bound": result.best_bound,
        "native_gap": result.gap,
        "native_metadata": result.native_metadata,
        "artifacts": {"gurobi.log": _file_record(log_path, relative_to=row_dir.parent.parent)},
    }


def solve_op_gurobi_incumbent(
    row: dict[str, Any],
    *,
    row_dir: Path,
    local_temp_root: Path,
    timeout_seconds: float,
    work_limit: float,
    node_limit: int,
) -> tuple[str, dict[str, Any]]:
    """Run the exact OP model but retain any strict-feasible incumbent for SFT."""

    import gurobipy as gp

    local_temp_root = _validated_op_local_temp_root(
        local_temp_root,
        require_empty=False,
    )
    reference = _reference_module()
    instance = reference._validated_op_instance(row)
    count = len(instance.coordinates)
    depot, sink = 0, count
    customers = list(range(1, count))
    arcs = [(depot, sink)]
    arcs.extend((depot, node) for node in customers)
    arcs.extend((node, sink) for node in customers)
    arcs.extend((left, right) for left in customers for right in customers if left != right)
    evidence_log_path = row_dir / "gurobi.log"
    local_log_descriptor, local_log_name = tempfile.mkstemp(
        prefix=f"{row_dir.name}.",
        suffix=".gurobi.log",
        dir=local_temp_root,
    )
    os.close(local_log_descriptor)
    local_log_path = Path(local_log_name)
    model: Any | None = None
    output: str | None = None
    native: dict[str, Any] | None = None
    try:
        model = gp.Model("clasp_sft_op_best_known")
        model.Params.OutputFlag = 1
        model.Params.LogToConsole = 0
        model.Params.LogFile = str(local_log_path)
        model.Params.TimeLimit = float(timeout_seconds)
        model.Params.WorkLimit = float(work_limit)
        model.Params.NodeLimit = int(node_limit)
        model.Params.Threads = 1
        model.Params.Seed = 0
        model.Params.Method = 1
        model.Params.MIPGap = 0.0
        model.Params.MIPGapAbs = 0.0
        model.Params.FeasibilityTol = 1e-9
        model.Params.IntFeasTol = 1e-9
        model.Params.NumericFocus = 3
        used = model.addVars(range(count), vtype=gp.GRB.BINARY, name="used")
        edge = model.addVars(arcs, vtype=gp.GRB.BINARY, name="edge")
        model.addConstr(used[depot] == 1)
        model.addConstr(
            gp.quicksum(edge[depot, target] for target in [*customers, sink]) == 1
        )
        model.addConstr(
            gp.quicksum(edge[source, sink] for source in [depot, *customers]) == 1
        )
        for node in customers:
            model.addConstr(
                gp.quicksum(
                    edge[source, node]
                    for source in [depot, *customers]
                    if source != node
                )
                == used[node]
            )
        arrival = model.addVars(
            customers,
            lb=0.0,
            ub=instance.route_limit,
            vtype=gp.GRB.CONTINUOUS,
            name="arrival_distance",
        )
        for node in customers:
            model.addConstr(arrival[node] <= instance.route_limit * used[node])
            model.addConstr(
                arrival[node]
                >= reference._op_pair_distance(instance, depot, node) * used[node]
            )
            model.addGenConstrIndicator(
                edge[depot, node],
                True,
                arrival[node] >= reference._op_pair_distance(instance, depot, node),
            )
        for left in customers:
            for right in customers:
                if left == right:
                    continue
                model.addGenConstrIndicator(
                    edge[left, right],
                    True,
                    arrival[right]
                    >= arrival[left] + reference._op_pair_distance(instance, left, right),
                )
            model.addConstr(
                gp.quicksum(
                    edge[left, target]
                    for target in [*customers, sink]
                    if target != left
                )
                == used[left]
            )
        flow_arcs = [(left, right) for left, right in arcs if right != sink]
        flow = model.addVars(
            flow_arcs,
            lb=0.0,
            ub=float(len(customers)),
            vtype=gp.GRB.CONTINUOUS,
            name="connectivity_flow",
        )
        for arc in flow_arcs:
            model.addConstr(flow[arc] <= len(customers) * edge[arc])
        model.addConstr(
            gp.quicksum(flow[depot, node] for node in customers)
            == gp.quicksum(used[node] for node in customers)
        )
        for node in customers:
            model.addConstr(
                gp.quicksum(
                    flow[source, node]
                    for source in [depot, *customers]
                    if source != node
                )
                - gp.quicksum(
                    flow[node, target] for target in customers if target != node
                )
                == used[node]
            )
        for left, right in reference._op_budget_incompatible_pairs(instance):
            model.addConstr(used[left] + used[right] <= 1)
        model.addConstr(
            gp.quicksum(
                reference._op_pair_distance(instance, left, right) * edge[left, right]
                for left, right in arcs
                if right != sink
            )
            <= instance.route_limit
        )
        model.setObjective(
            gp.quicksum(instance.prizes[node] * used[node] for node in range(count)),
            gp.GRB.MAXIMIZE,
        )
        start_route = reference._deterministic_op_start(instance)
        start_edges = set(zip(start_route, [*start_route[1:], sink]))
        start_nodes = set(start_route)
        for node in range(count):
            used[node].Start = float(node in start_nodes)
        for arc in arcs:
            edge[arc].Start = float(arc in start_edges)
        for arc in flow_arcs:
            flow[arc].Start = 0.0
        selected_customers = len(start_route) - 1
        for position, arc in enumerate(zip(start_route, start_route[1:])):
            flow[arc].Start = float(selected_customers - position)
        for node in customers:
            arrival[node].Start = 0.0
        cumulative_distance = 0.0
        for left, right in zip(start_route, start_route[1:]):
            cumulative_distance += reference._op_pair_distance(instance, left, right)
            arrival[right].Start = cumulative_distance
        started = time.monotonic()
        model.optimize()
        measured = time.monotonic() - started
        status_code = int(model.Status)
        status_names = {
            int(gp.GRB.OPTIMAL): "OPTIMAL",
            int(gp.GRB.TIME_LIMIT): "TIME_LIMIT",
            int(gp.GRB.WORK_LIMIT): "WORK_LIMIT",
            int(gp.GRB.NODE_LIMIT): "NODE_LIMIT",
            int(gp.GRB.INTERRUPTED): "INTERRUPTED",
        }
        status = status_names.get(status_code, f"STATUS_{status_code}")
        solution_count = int(model.SolCount)
        if solution_count <= 0:
            raise ReplacementGenerationError(f"OP returned no incumbent: {status}")
        successor = {
            left: right for left, right in arcs if float(edge[left, right].X) > 0.5
        }
        route = [depot]
        current = depot
        while True:
            if current not in successor:
                raise ReplacementGenerationError("OP incumbent has no path successor")
            current = successor[current]
            if current == sink:
                break
            if current in route:
                raise ReplacementGenerationError("OP incumbent contains a cycle")
            route.append(current)
        selected_nodes = {
            node for node in range(count) if float(used[node].X) > 0.5
        }
        if selected_nodes != set(route):
            raise ReplacementGenerationError(
                "OP incumbent selected nodes do not equal the extracted path"
            )
        recomputed_objective = sum(int(instance.prizes[node]) for node in route)
        if not math.isclose(
            float(model.ObjVal),
            float(recomputed_objective),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ReplacementGenerationError(
                "OP incumbent objective does not equal the extracted path prize"
            )
        objective = int(recomputed_objective)
        bound = float(model.ObjBound) if math.isfinite(float(model.ObjBound)) else None
        gap = float(model.MIPGap) if math.isfinite(float(model.MIPGap)) else None
        native_runtime = float(model.Runtime)
        search_nodes = float(model.NodeCount)
        deterministic_work = float(model.Work)
        output = f"Route: {route}, Objective: {objective}"
        native = {
            "native_status": status,
            "native_status_code": status_code,
            "native_wall_time_seconds": native_runtime,
            "measured_wall_time_seconds": measured,
            "native_incumbent": objective,
            "native_best_bound": bound,
            "native_gap": gap,
            "solution_count": solution_count,
            "node_count": search_nodes,
            "deterministic_work": deterministic_work,
            "mip_start_route": start_route,
            "mip_start_objective": sum(instance.prizes[node] for node in start_route),
            "quality_semantics": "best_known_gurobi_incumbent_not_optimality_certificate",
        }
    finally:
        try:
            if model is not None:
                model.dispose()
        finally:
            if local_log_path.is_file():
                _write_bytes_new(
                    evidence_log_path,
                    local_log_path.read_bytes(),
                )
                local_log_path.unlink()
    if output is None or native is None:  # pragma: no cover - protected by control flow
        raise ReplacementGenerationError("OP solve produced no result")
    native["artifacts"] = {
        "gurobi.log": _file_record(
            evidence_log_path,
            relative_to=row_dir.parent.parent,
        )
    }
    return output, native


def _solver_identity(task: str, args: argparse.Namespace) -> dict[str, object]:
    if task == "cvrp":
        executable = args.lkh_executable.resolve(strict=True)
        observed = clean.sha256_file(executable)
        if observed != args.lkh_sha256:
            raise ReplacementGenerationError(
                f"LKH hash mismatch: expected={args.lkh_sha256}, observed={observed}"
            )
        return {
            "backend": "LKH-3.0.8",
            "artifact": _file_record(executable),
            "license_class": "research_use_non_osi",
            "quality_semantics": "rounded_EUC_2D_heuristic_raw_distance_replayed",
        }
    import networkx as nx
    import numpy as np

    backend = (
        _reference_module().GurobiMISBackend()
        if task == "mis"
        else _reference_module().GurobiOPBackend()
    )
    return {
        **backend.probe_identity(),
        "python": _file_record(Path(sys.executable).resolve(strict=True)),
        "support_packages": {
            "numpy": {
                "version": np.__version__,
                "module": _file_record(Path(np.__file__).resolve(strict=True)),
            },
            "networkx": {
                "version": nx.__version__,
                "module": _file_record(Path(nx.__file__).resolve(strict=True)),
            },
        },
        "license_class": "gurobi_academic_site_license_nonredistributable",
        "quality_semantics": (
            "certified_optimum" if task == "mis" else "best_known_incumbent"
        ),
    }


def _solver_parameters(task: str, args: argparse.Namespace) -> dict[str, object]:
    if task == "cvrp":
        return {
            "runs": 10,
            "seed": 1,
            "trace_level": 1,
            "special": True,
            "timeout_seconds": args.timeout_seconds,
            "search_distance": "TSPLIB_EUC_2D_rounded",
            "replay_distance": "raw_euclidean_float64",
        }
    common = {
        "time_limit_seconds": args.timeout_seconds,
        "threads": 1,
        "seed": 0,
        "method": 1,
        "mip_gap": 0.0,
        "mip_gap_abs": 0.0,
        "feasibility_tolerance": 1e-9,
        "integrality_tolerance": 1e-9,
        "numeric_focus": 3,
    }
    if task == "op":
        common.update({"work_limit": args.work_limit, "node_limit": args.node_limit})
    return common


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise ReplacementGenerationError(f"refusing to seal symlink: {path}")
        os.chmod(path, 0o550 if path.is_dir() else 0o440)
    os.chmod(root, 0o550)


def _read_regular_file_pinned(
    path: Path,
    observed: os.stat_result,
    *,
    capture_payload: bool,
) -> tuple[str, bytes | None]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise ReplacementGenerationError("platform cannot safely inspect a sealed tree")
    descriptor = os.open(path, flags | os.O_NOFOLLOW)
    payload = bytearray() if capture_payload else None
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or _stat_identity(before) != _stat_identity(observed)
            or before.st_size != observed.st_size
            or before.st_nlink != 1
        ):
            raise ReplacementGenerationError(f"sealed-tree file changed before read: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if payload is not None:
                payload.extend(chunk)
        after = os.fstat(descriptor)
        if (
            _stat_identity(after) != _stat_identity(before)
            or after.st_size != before.st_size
            or total != before.st_size
        ):
            raise ReplacementGenerationError(f"sealed-tree file changed during read: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), bytes(payload) if payload is not None else None


def _sealed_tree_inventory(
    root: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    root_stat = root.stat(follow_symlinks=False)
    if not stat.S_ISDIR(root_stat.st_mode) or root.is_symlink():
        raise ReplacementGenerationError("sealed shard root is not a regular directory")
    inventory: dict[str, dict[str, object]] = {
        ".": {
            "type": "directory",
            "mode": stat.S_IMODE(root_stat.st_mode),
        }
    }
    json_payloads: dict[str, object] = {}

    def visit(directory: Path, relative: PurePosixPath) -> None:
        with os.scandir(directory) as scanned:
            entries = sorted(scanned, key=lambda entry: entry.name)
        for entry in entries:
            if entry.name in {"", ".", ".."}:
                raise ReplacementGenerationError("invalid entry in sealed shard tree")
            child_relative = relative / entry.name
            relative_text = child_relative.as_posix()
            child = directory / entry.name
            observed = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(observed.st_mode):
                inventory[relative_text] = {
                    "type": "directory",
                    "mode": stat.S_IMODE(observed.st_mode),
                }
                visit(child, child_relative)
            elif stat.S_ISREG(observed.st_mode):
                if observed.st_nlink != 1:
                    raise ReplacementGenerationError(
                        f"sealed shard contains a multiply-linked file: {relative_text}"
                    )
                digest, payload = _read_regular_file_pinned(
                    child,
                    observed,
                    capture_payload=child.suffix == ".json",
                )
                inventory[relative_text] = {
                    "type": "file",
                    "mode": stat.S_IMODE(observed.st_mode),
                    "size_bytes": observed.st_size,
                    "sha256": digest,
                }
                if payload is not None:
                    try:
                        json_payloads[relative_text] = json.loads(payload)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ReplacementGenerationError(
                            f"invalid JSON in sealed shard tree: {relative_text}"
                        ) from exc
            else:
                raise ReplacementGenerationError(
                    f"sealed shard contains a symlink or special entry: {relative_text}"
                )

    visit(root, PurePosixPath())
    return dict(sorted(inventory.items())), dict(sorted(json_payloads.items()))


def _copy_regular_file_from_inventory(
    source: Path,
    destination: Path,
    expected: Mapping[str, object],
) -> None:
    observed = source.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != expected["mode"]
        or observed.st_size != expected["size_bytes"]
        or observed.st_nlink != 1
    ):
        raise ReplacementGenerationError(f"local sealed file changed before copy: {source}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise ReplacementGenerationError("platform cannot safely copy a sealed tree")
    source_descriptor = os.open(source, flags | os.O_NOFOLLOW)
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    destination_descriptor: int | None = None
    digest = hashlib.sha256()
    try:
        pinned = os.fstat(source_descriptor)
        if _stat_identity(pinned) != _stat_identity(observed):
            raise ReplacementGenerationError(f"local sealed file changed before copy: {source}")
        destination_descriptor = os.open(destination, destination_flags, 0o600)
        with os.fdopen(destination_descriptor, "wb", closefd=False) as output:
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
            output.flush()
        after = os.fstat(source_descriptor)
        if (
            _stat_identity(after) != _stat_identity(pinned)
            or after.st_size != pinned.st_size
            or digest.hexdigest() != expected["sha256"]
        ):
            raise ReplacementGenerationError(f"local sealed file changed during copy: {source}")
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        os.close(source_descriptor)
    os.chmod(destination, int(expected["mode"]), follow_symlinks=False)


def _copy_sealed_tree_once(
    source: Path,
    destination: Path,
    inventory: Mapping[str, Mapping[str, object]],
) -> None:
    directories = [
        relative
        for relative, record in inventory.items()
        if relative != "." and record["type"] == "directory"
    ]
    for relative in sorted(directories, key=lambda value: (value.count("/"), value)):
        (destination / relative).mkdir(mode=0o700)
    for relative, record in sorted(inventory.items()):
        if record["type"] != "file":
            continue
        _copy_regular_file_from_inventory(
            source / relative,
            destination / relative,
            record,
        )
    for relative in sorted(
        directories,
        key=lambda value: (value.count("/"), value),
        reverse=True,
    ):
        os.chmod(destination / relative, int(inventory[relative]["mode"]))
    os.chmod(destination, int(inventory["."]["mode"]))


_STRICT_VERIFIER_IMPLEMENTATION_RELATIVES = (
    "scripts/build_clean_sft500k_corpus.py",
    "src/clasp/data/benchmark.py",
    "src/clasp/data/loaders.py",
    "src/clasp/entrypoints/evaluate.py",
    "src/clasp/rewards/co_tasks.py",
)


def _is_sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_safe_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ReplacementGenerationError(f"{label} has a non-string relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != value
    ):
        raise ReplacementGenerationError(f"{label} has an unsafe relative path")
    return value


def _validated_canonical_path(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ReplacementGenerationError(f"{label} has a non-string canonical path")
    canonical = PurePosixPath(value)
    if (
        not canonical.is_absolute()
        or any(part in {"", ".", ".."} for part in canonical.parts[1:])
        or canonical.as_posix() != value
    ):
        raise ReplacementGenerationError(f"{label} has a noncanonical absolute path")
    return value


def _validate_canonical_file_record(
    value: object,
    *,
    label: str,
    expected_relative_path: str | None = None,
) -> str:
    expected_keys = {"canonical_path", "size_bytes", "sha256"}
    if expected_relative_path is not None:
        expected_keys.add("relative_path")
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ReplacementGenerationError(f"{label} has an invalid canonical file schema")
    canonical_path = _validated_canonical_path(value["canonical_path"], label=label)
    size = value["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ReplacementGenerationError(f"{label} has an invalid file size")
    if not _is_sha256_text(value["sha256"]):
        raise ReplacementGenerationError(f"{label} has an invalid SHA256")
    if expected_relative_path is not None:
        relative_path = _validated_safe_relative_path(
            value["relative_path"],
            label=label,
        )
        if relative_path != expected_relative_path:
            raise ReplacementGenerationError(f"{label} has the wrong relative path")
        if not canonical_path.endswith(f"/{relative_path}"):
            raise ReplacementGenerationError(
                f"{label} canonical and relative paths are inconsistent"
            )
    return canonical_path


def _validate_receipt_implementation(value: object) -> None:
    expected_roles = {"producer", "generator", "reference_runner", "strict_verifier"}
    if not isinstance(value, Mapping) or set(value) != expected_roles:
        raise ReplacementGenerationError("replacement implementation has an invalid schema")
    role_relatives = {
        "producer": "scripts/generate_clean_sft_replacements.py",
        "generator": "scripts/generate_iid_benchmark.py",
        "reference_runner": "scripts/run_reference_evidence.py",
    }
    repository_roots: set[str] = set()
    for role, relative_path in role_relatives.items():
        canonical_path = _validate_canonical_file_record(
            value[role],
            label=f"replacement implementation {role}",
        )
        suffix = f"/{relative_path}"
        if not canonical_path.endswith(suffix):
            raise ReplacementGenerationError(
                f"replacement implementation {role} has the wrong source path"
            )
        repository_roots.add(canonical_path[: -len(suffix)])
    strict = value["strict_verifier"]
    if not isinstance(strict, Mapping) or set(strict) != {
        "files",
        "implementation_manifest_sha256",
    }:
        raise ReplacementGenerationError(
            "replacement strict-verifier implementation has an invalid schema"
        )
    files = strict["files"]
    if not isinstance(files, list) or len(files) != len(
        _STRICT_VERIFIER_IMPLEMENTATION_RELATIVES
    ):
        raise ReplacementGenerationError(
            "replacement strict-verifier implementation has the wrong file count"
        )
    compact: list[dict[str, object]] = []
    for record, relative_path in zip(
        files,
        _STRICT_VERIFIER_IMPLEMENTATION_RELATIVES,
        strict=True,
    ):
        canonical_path = _validate_canonical_file_record(
            record,
            label=f"replacement strict-verifier source {relative_path}",
            expected_relative_path=relative_path,
        )
        repository_roots.add(canonical_path[: -(len(relative_path) + 1)])
        assert isinstance(record, Mapping)
        compact.append(
            {
                "relative_path": record["relative_path"],
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }
        )
    if len(repository_roots) != 1:
        raise ReplacementGenerationError(
            "replacement implementation source records have inconsistent roots"
        )
    if strict["implementation_manifest_sha256"] != clean.sha256_json(compact):
        raise ReplacementGenerationError(
            "replacement strict-verifier implementation manifest mismatch"
        )


def _validate_receipt_solver_identity(value: object) -> None:
    expected_keys = {
        "backend_name",
        "native_version",
        "distribution_name",
        "module_origin",
        "installed_distribution_manifest_sha256",
        "python",
        "support_packages",
        "license_class",
        "quality_semantics",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ReplacementGenerationError("OP solver identity has an invalid schema")
    expected_text = {
        "backend_name": "gurobi",
        "native_version": "13.0.1",
        "distribution_name": "gurobipy",
        "license_class": "gurobi_academic_site_license_nonredistributable",
        "quality_semantics": "best_known_incumbent",
    }
    if any(value.get(field) != expected for field, expected in expected_text.items()):
        raise ReplacementGenerationError("OP solver identity has an unexpected value")
    _validated_canonical_path(value["module_origin"], label="Gurobi module origin")
    if not _is_sha256_text(value["installed_distribution_manifest_sha256"]):
        raise ReplacementGenerationError("OP solver identity has an invalid manifest SHA256")
    _validate_canonical_file_record(value["python"], label="OP Python identity")
    support = value["support_packages"]
    if not isinstance(support, Mapping) or set(support) != {"numpy", "networkx"}:
        raise ReplacementGenerationError("OP solver support-package identity is invalid")
    for package in ("numpy", "networkx"):
        package_record = support[package]
        if not isinstance(package_record, Mapping) or set(package_record) != {
            "version",
            "module",
        }:
            raise ReplacementGenerationError(
                f"OP solver {package} identity has an invalid schema"
            )
        if not isinstance(package_record["version"], str) or not package_record[
            "version"
        ]:
            raise ReplacementGenerationError(f"OP solver {package} version is invalid")
        _validate_canonical_file_record(
            package_record["module"],
            label=f"OP solver {package} module identity",
        )


def _validate_relative_file_records(
    value: object,
    inventory: Mapping[str, Mapping[str, object]],
) -> None:
    if isinstance(value, Mapping):
        keys = set(value)
        shard_record_keys = {"relative_path", "size_bytes", "sha256"}
        if "canonical_path" in keys:
            raise ReplacementGenerationError(
                "canonical file record appears outside an approved provenance context"
            )
        if keys & shard_record_keys:
            if keys != shard_record_keys:
                raise ReplacementGenerationError(
                    "shard-relative file record has an invalid schema"
                )
            raw_relative = _validated_safe_relative_path(
                value["relative_path"],
                label="shard file record",
            )
            observed = inventory.get(raw_relative)
            if (
                observed is None
                or observed.get("type") != "file"
                or observed.get("size_bytes") != value["size_bytes"]
                or observed.get("sha256") != value["sha256"]
            ):
                raise ReplacementGenerationError(
                    f"shard file record does not match copied file: {raw_relative}"
                )
        for child in value.values():
            _validate_relative_file_records(child, inventory)
    elif isinstance(value, list):
        for child in value:
            _validate_relative_file_records(child, inventory)


def _deep_validate_copied_op_shard(
    root: Path,
    expected_inventory: Mapping[str, Mapping[str, object]],
) -> None:
    observed_inventory, json_payloads = _sealed_tree_inventory(root)
    if observed_inventory != expected_inventory:
        raise ReplacementGenerationError("copied OP shard inventory does not match local tree")
    required_json = {
        "replacements.json",
        "evidence_manifest.json",
        "replacement_receipt.json",
    }
    if not required_json <= set(json_payloads):
        raise ReplacementGenerationError("copied OP shard is missing final JSON artifacts")
    receipt = json_payloads["replacement_receipt.json"]
    replacements = json_payloads["replacements.json"]
    evidence_manifest = json_payloads["evidence_manifest.json"]
    if not isinstance(receipt, dict):
        raise ReplacementGenerationError("copied OP shard receipt is not an object")
    receipt_body = dict(receipt)
    stored_receipt_hash = receipt_body.pop("receipt_sha256", None)
    if stored_receipt_hash != clean.sha256_json(receipt_body):
        raise ReplacementGenerationError("copied OP shard receipt self-hash mismatch")
    rows = receipt.get("rows")
    attempts = receipt.get("attempts")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("protocol") != PROTOCOL
        or receipt.get("status")
        != "strict_solver_backed_replacement_shard_complete"
        or receipt.get("task") != "op"
        or not isinstance(rows, int)
        or isinstance(rows, bool)
        or rows <= 0
        or not isinstance(attempts, int)
        or isinstance(attempts, bool)
        or attempts < rows
        or receipt.get("unique_semantic_instances") != rows
        or not isinstance(replacements, list)
        or not isinstance(evidence_manifest, list)
        or rows != len(replacements)
        or rows != len(evidence_manifest)
    ):
        raise ReplacementGenerationError("copied OP shard final counts/status are invalid")
    output_record = receipt.get("output")
    evidence_record = receipt.get("evidence_manifest")
    if (
        not isinstance(output_record, Mapping)
        or output_record.get("relative_path") != "replacements.json"
        or not isinstance(evidence_record, Mapping)
        or evidence_record.get("relative_path") != "evidence_manifest.json"
    ):
        raise ReplacementGenerationError("copied OP shard final file bindings are invalid")
    for expected_index, row_evidence in enumerate(evidence_manifest):
        if (
            not isinstance(row_evidence, Mapping)
            or row_evidence.get("row_index") != expected_index
            or not isinstance(row_evidence.get("semantic_instance_sha256"), str)
            or len(row_evidence["semantic_instance_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in row_evidence["semantic_instance_sha256"]
            )
            or not isinstance(row_evidence.get("demonstration_sha256"), str)
            or len(row_evidence["demonstration_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in row_evidence["demonstration_sha256"]
            )
            or not isinstance(row_evidence.get("native_status"), str)
            or not row_evidence["native_status"]
            or not isinstance(row_evidence.get("strict_replay"), Mapping)
        ):
            raise ReplacementGenerationError("copied OP shard evidence row is malformed")
    phase_evidence = receipt.get("phase_evidence")
    if phase_evidence is not None and (
        not isinstance(phase_evidence, Mapping)
        or phase_evidence.get("protocol")
        != "clasp-op-local-stage-then-atomic-remote-publication-v1"
        or phase_evidence.get("remote_publication_started_at_receipt_creation") is not False
    ):
        raise ReplacementGenerationError("copied OP shard phase evidence is invalid")
    _validate_receipt_implementation(receipt.get("implementation"))
    _validate_receipt_solver_identity(receipt.get("solver_identity"))
    local_receipt = dict(receipt)
    local_receipt.pop("implementation")
    local_receipt.pop("solver_identity")
    _validate_relative_file_records(local_receipt, observed_inventory)
    for relative_path, payload in json_payloads.items():
        if relative_path != "replacement_receipt.json":
            _validate_relative_file_records(payload, observed_inventory)


def _fsync_remote_authoritative_artifacts(root: Path) -> None:
    for name in (
        "replacements.json",
        "evidence_manifest.json",
        "replacement_receipt.json",
    ):
        path = root / name
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
        try:
            observed = os.fstat(descriptor)
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise ReplacementGenerationError(
                    f"authoritative copied artifact is not a regular file: {name}"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    root_descriptor = os.open(root, _directory_open_flags())
    try:
        os.fsync(root_descriptor)
    finally:
        os.close(root_descriptor)


def _require_distinct_stage_filesystems(local_stage: Path, destination_parent: Path) -> None:
    local_observed = local_stage.stat(follow_symlinks=False)
    destination_observed = destination_parent.stat(follow_symlinks=False)
    if not stat.S_ISDIR(local_observed.st_mode) or not stat.S_ISDIR(
        destination_observed.st_mode
    ):
        raise ReplacementGenerationError("OP staging endpoints must be directories")
    if local_observed.st_dev == destination_observed.st_dev:
        raise ReplacementGenerationError(
            "OP local staging and destination must be on different filesystems"
        )


def _validate_safe_publication_parent(observed: os.stat_result) -> None:
    if not stat.S_ISDIR(observed.st_mode):
        raise ReplacementGenerationError("OP publication parent is not a directory")
    if observed.st_uid != os.geteuid():
        raise ReplacementGenerationError("OP publication parent has the wrong owner")
    if stat.S_IMODE(observed.st_mode) & 0o022:
        raise ReplacementGenerationError(
            "OP publication parent must not be group/other writable"
        )


def _create_hidden_sibling_stage(
    destination: Path,
) -> tuple[Path, tuple[int, int]]:
    parent_descriptor = os.open(destination.parent, _directory_open_flags())
    try:
        parent_observed = os.fstat(parent_descriptor)
        _validate_safe_publication_parent(parent_observed)
        parent_identity = _stat_identity(parent_observed)
        _require_directory_identity(
            destination.parent,
            parent_identity,
            label="OP publication parent",
        )
        try:
            os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite replacement shard: {destination}")
        for _attempt in range(32):
            hidden_name = f".{destination.name}.publish.{secrets.token_hex(12)}"
            try:
                os.mkdir(hidden_name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            hidden_identity: tuple[int, int] | None = None
            hidden_descriptor: int | None = None
            try:
                created_named = os.stat(
                    hidden_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                hidden_identity = _stat_identity(created_named)
                if not stat.S_ISDIR(created_named.st_mode):
                    raise ReplacementGenerationError(
                        "hidden OP publication reservation is not a directory"
                    )
                hidden_descriptor = os.open(
                    hidden_name,
                    _directory_open_flags(),
                    dir_fd=parent_descriptor,
                )
                os.fchmod(hidden_descriptor, 0o700)
                if _stat_identity(os.fstat(hidden_descriptor)) != hidden_identity:
                    raise ReplacementGenerationError(
                        "hidden OP publication stage changed while being pinned"
                    )
                hidden_named = os.stat(
                    hidden_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(hidden_named.st_mode)
                    or _stat_identity(hidden_named) != hidden_identity
                    or stat.S_IMODE(hidden_named.st_mode) != 0o700
                ):
                    raise ReplacementGenerationError(
                        "hidden OP publication stage was replaced during creation"
                    )
                _require_directory_identity(
                    destination.parent,
                    parent_identity,
                    label="OP publication parent after hidden-stage creation",
                )
                _validate_safe_publication_parent(os.fstat(parent_descriptor))
                return destination.parent / hidden_name, hidden_identity
            except Exception:
                if hidden_descriptor is not None:
                    os.close(hidden_descriptor)
                    hidden_descriptor = None
                if hidden_identity is None:
                    try:
                        created_after_failure = os.stat(
                            hidden_name,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                    except OSError:
                        created_after_failure = None
                    if (
                        created_after_failure is not None
                        and stat.S_ISDIR(created_after_failure.st_mode)
                        and created_after_failure.st_uid == os.geteuid()
                    ):
                        hidden_identity = _stat_identity(created_after_failure)
                if hidden_identity is not None:
                    _remove_publication_reservation_at(
                        parent_descriptor=parent_descriptor,
                        destination_name=hidden_name,
                        reservation_identity=hidden_identity,
                    )
                raise
            finally:
                if hidden_descriptor is not None:
                    os.close(hidden_descriptor)
        raise ReplacementGenerationError("could not reserve a unique OP publication stage")
    finally:
        os.close(parent_descriptor)


def _quarantine_postrename_tree(
    destination: Path,
    *,
    expected_identity: tuple[int, int],
) -> Path:
    parent_descriptor = os.open(destination.parent, _directory_open_flags())
    try:
        observed = os.stat(
            destination.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(observed.st_mode)
            or _stat_identity(observed) != expected_identity
        ):
            raise ReplacementGenerationError(
                "refusing to quarantine a destination not owned by this publication"
            )
        for _attempt in range(32):
            quarantine_name = (
                f".{destination.name}.quarantine.{secrets.token_hex(12)}"
            )
            try:
                os.stat(
                    quarantine_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                continue
            os.rename(
                destination.name,
                quarantine_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.fsync(parent_descriptor)
            return destination.parent / quarantine_name
        raise ReplacementGenerationError("could not reserve an OP quarantine name")
    finally:
        os.close(parent_descriptor)


def _remove_private_tree(
    root: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    try:
        observed = root.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISDIR(observed.st_mode)
        or root.is_symlink()
        or _stat_identity(observed) != expected_identity
    ):
        raise ReplacementGenerationError(f"refusing to clean a replaced private tree: {root}")
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise ReplacementGenerationError("platform cannot safely clean private shard trees")
    for current, directories, _files in os.walk(root, topdown=True, followlinks=False):
        os.chmod(current, 0o700, follow_symlinks=False)
        for name in directories:
            candidate = Path(current) / name
            candidate_stat = candidate.stat(follow_symlinks=False)
            if stat.S_ISDIR(candidate_stat.st_mode) and not candidate.is_symlink():
                os.chmod(candidate, 0o700, follow_symlinks=False)
    shutil.rmtree(root)


def _cleanup_local_op_stage_after_generation(
    stage: Path,
    *,
    expected_identity: tuple[int, int],
    completed: bool,
) -> None:
    try:
        _remove_private_tree(stage, expected_identity=expected_identity)
    except Exception as exc:
        if not completed:
            raise
        _emit_phase_event(
            "local_cleanup_deferred_to_launcher",
            task="op",
            error_type=type(exc).__name__,
        )


def _publish_completed_local_op_shard(
    local_stage: Path,
    destination: Path,
    *,
    attempts: int,
    rows: int,
) -> None:
    inventory, _local_json = _sealed_tree_inventory(local_stage)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite replacement shard: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_distinct_stage_filesystems(local_stage, destination.parent)
    remote_stage, remote_identity = _create_hidden_sibling_stage(destination)
    published = False
    publication_started = time.monotonic()
    try:
        _emit_phase_event(
            "remote_publication_begin",
            task="op",
            attempts=attempts,
            rows=rows,
        )
        _copy_sealed_tree_once(local_stage, remote_stage, inventory)
        _emit_phase_event(
            "remote_copy_complete",
            task="op",
            attempts=attempts,
            rows=rows,
            publication_elapsed_seconds=time.monotonic() - publication_started,
        )
        _seal_tree(remote_stage)
        _emit_phase_event(
            "remote_seal_complete",
            task="op",
            attempts=attempts,
            rows=rows,
            publication_elapsed_seconds=time.monotonic() - publication_started,
        )
        _deep_validate_copied_op_shard(remote_stage, inventory)
        _emit_phase_event(
            "remote_postseal_validation_complete",
            task="op",
            attempts=attempts,
            rows=rows,
            publication_elapsed_seconds=time.monotonic() - publication_started,
        )
        _fsync_remote_authoritative_artifacts(remote_stage)
        _emit_phase_event(
            "remote_authoritative_fsync_complete",
            task="op",
            attempts=attempts,
            rows=rows,
            publication_elapsed_seconds=time.monotonic() - publication_started,
        )
        _emit_phase_event(
            "remote_atomic_rename_begin",
            task="op",
            attempts=attempts,
            rows=rows,
            publication_elapsed_seconds=time.monotonic() - publication_started,
        )
        _rename_directory_noreplace(remote_stage, destination)
        published = True
        _emit_phase_event(
            "remote_publication_complete",
            task="op",
            attempts=attempts,
            rows=rows,
            publication_elapsed_seconds=time.monotonic() - publication_started,
        )
    except Exception as exc:
        if not remote_stage.exists():
            try:
                destination_observed = destination.stat(follow_symlinks=False)
            except FileNotFoundError:
                destination_observed = None
            if (
                destination_observed is not None
                and stat.S_ISDIR(destination_observed.st_mode)
                and _stat_identity(destination_observed) == remote_identity
            ):
                _quarantine_postrename_tree(
                    destination,
                    expected_identity=remote_identity,
                )
                _emit_phase_event(
                    "remote_publication_quarantined",
                    task="op",
                    attempts=attempts,
                    rows=rows,
                    error_type=type(exc).__name__,
                )
        _emit_phase_event(
            "remote_publication_failed",
            task="op",
            attempts=attempts,
            rows=rows,
            error_type=type(exc).__name__,
        )
        raise
    finally:
        if not published:
            _remove_private_tree(remote_stage, expected_identity=remote_identity)


def _stat_identity(observed: os.stat_result) -> tuple[int, int]:
    return int(observed.st_dev), int(observed.st_ino)


def _directory_open_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    missing = [name for name in required if not hasattr(os, name)]
    if missing:
        raise ReplacementGenerationError(
            "platform cannot perform no-follow shard publication; missing: "
            + ", ".join(missing)
        )
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _require_directory_identity(
    path: Path,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> None:
    try:
        observed = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ReplacementGenerationError(f"cannot stat {label} {path}: {exc}") from exc
    if not stat.S_ISDIR(observed.st_mode) or _stat_identity(observed) != expected_identity:
        raise ReplacementGenerationError(
            f"{label} no longer names the pinned directory: {path}"
        )


def _renameat2_noreplace_errno(
    *,
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> int | None:
    """Return ``None`` on renameat2 success, otherwise its positive errno."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return errno.ENOSYS
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    rename_noreplace = 1
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        rename_noreplace,
    )
    return None if result == 0 else ctypes.get_errno()


def _remove_publication_reservation_at(
    *,
    parent_descriptor: int,
    destination_name: str,
    reservation_identity: tuple[int, int],
) -> None:
    """Remove only the still-empty reservation inode made by this process."""

    try:
        observed = os.stat(
            destination_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ReplacementGenerationError(
            f"cannot inspect failed shard-publication reservation "
            f"{destination_name}: {exc}"
        ) from exc
    if not stat.S_ISDIR(observed.st_mode) or _stat_identity(observed) != reservation_identity:
        raise ReplacementGenerationError(
            "refusing to clean a replaced shard-publication reservation: "
            f"{destination_name}"
        )
    try:
        os.rmdir(destination_name, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ReplacementGenerationError(
            f"cannot remove failed shard-publication reservation "
            f"{destination_name}: {exc}"
        ) from exc
    os.fsync(parent_descriptor)


def _rename_with_empty_directory_reservation_at(
    *,
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    source_identity: tuple[int, int],
    destination: Path,
) -> None:
    """Publish after an exclusive empty-directory reservation.

    Some Lustre deployments reject ``renameat2(RENAME_NOREPLACE)`` with
    ``EINVAL``.  ``mkdirat`` supplies the create-once decision in that case.
    The reservation is pinned and checked immediately before ordinary rename,
    and failure cleanup removes it only if the name still refers to the inode
    created by this invocation.  The containing directory is already pinned
    and is controlled by the shard owner.
    """

    reservation_descriptor: int | None = None
    reservation_identity: tuple[int, int] | None = None
    published = False
    try:
        try:
            os.mkdir(destination_name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite replacement shard: {destination}"
            ) from exc
        reservation_stat = os.stat(
            destination_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(reservation_stat.st_mode):  # pragma: no cover - mkdir contract
            raise ReplacementGenerationError(
                "shard-publication reservation is not a directory"
            )
        reservation_identity = _stat_identity(reservation_stat)
        reservation_descriptor = os.open(
            destination_name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
        if _stat_identity(os.fstat(reservation_descriptor)) != reservation_identity:
            raise ReplacementGenerationError(
                "shard-publication reservation changed while it was pinned"
            )
        with os.scandir(reservation_descriptor) as entries:
            if next(entries, None) is not None:  # pragma: no cover - fresh mkdir contract
                raise ReplacementGenerationError(
                    "shard-publication reservation is not empty"
                )
        os.fsync(parent_descriptor)

        source_before = os.stat(
            source_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        immediately_before = os.stat(
            destination_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(source_before.st_mode)
            or _stat_identity(source_before) != source_identity
        ):
            raise ReplacementGenerationError(
                "staged replacement shard changed before publication"
            )
        if (
            not stat.S_ISDIR(immediately_before.st_mode)
            or _stat_identity(immediately_before) != reservation_identity
        ):
            raise ReplacementGenerationError(
                "shard-publication reservation changed before rename"
            )
        try:
            os.rename(
                source_name,
                destination_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise ReplacementGenerationError(
                f"cannot publish replacement shard {source_name} -> "
                f"{destination_name}: {exc}"
            ) from exc
        published = True
        os.fsync(parent_descriptor)
        published_stat = os.stat(
            destination_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(published_stat.st_mode) or _stat_identity(
            published_stat
        ) != source_identity:
            raise ReplacementGenerationError(
                "portable shard publication did not install the pinned staging directory"
            )
    finally:
        if not published and reservation_identity is not None:
            _remove_publication_reservation_at(
                parent_descriptor=parent_descriptor,
                destination_name=destination_name,
                reservation_identity=reservation_identity,
            )
        if reservation_descriptor is not None:
            os.close(reservation_descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Publish a sibling directory without replacing any existing target."""

    if source.parent != destination.parent:
        raise ReplacementGenerationError(
            "staged and published replacement shard must be siblings"
        )
    if source == destination or source.name in {"", ".", ".."} or destination.name in {
        "",
        ".",
        "..",
    }:
        raise ReplacementGenerationError("invalid replacement shard publication paths")

    directory_flags = _directory_open_flags()
    parent_descriptor = os.open(source.parent, directory_flags)
    source_descriptor: int | None = None
    try:
        parent_identity = _stat_identity(os.fstat(parent_descriptor))
        _require_directory_identity(
            source.parent,
            parent_identity,
            label="replacement shard parent",
        )
        source_descriptor = os.open(
            source.name,
            directory_flags,
            dir_fd=parent_descriptor,
        )
        source_identity = _stat_identity(os.fstat(source_descriptor))
        source_named_stat = os.stat(
            source.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(source_named_stat.st_mode)
            or _stat_identity(source_named_stat) != source_identity
        ):
            raise ReplacementGenerationError(
                "staged replacement shard changed while it was pinned"
            )

        error_number = _renameat2_noreplace_errno(
            parent_descriptor=parent_descriptor,
            source_name=source.name,
            destination_name=destination.name,
        )
        if error_number == errno.EEXIST:
            raise FileExistsError(
                f"refusing to overwrite replacement shard: {destination}"
            )
        unsupported = {
            errno.ENOSYS,
            errno.EINVAL,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
        }
        if error_number in unsupported:
            _rename_with_empty_directory_reservation_at(
                parent_descriptor=parent_descriptor,
                source_name=source.name,
                destination_name=destination.name,
                source_identity=source_identity,
                destination=destination,
            )
        elif error_number is not None:
            raise OSError(error_number, os.strerror(error_number), str(destination))
        else:
            os.fsync(parent_descriptor)

        _require_directory_identity(
            source.parent,
            parent_identity,
            label="replacement shard parent after publication",
        )
        destination_descriptor = os.open(
            destination.name,
            directory_flags,
            dir_fd=parent_descriptor,
        )
        try:
            destination_identity = _stat_identity(os.fstat(destination_descriptor))
            destination_named_stat = os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                destination_identity != source_identity
                or _stat_identity(destination_named_stat) != source_identity
            ):
                raise ReplacementGenerationError(
                    "published replacement shard is not the pinned staging directory"
                )
        finally:
            os.close(destination_descriptor)
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        os.close(parent_descriptor)


def generate_shard(args: argparse.Namespace) -> dict[str, object]:
    generation_started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    generation_started_monotonic = time.monotonic()
    op_local_temp_root: Path | None = None
    op_local_capacity: dict[str, int | None] | None = None
    if args.task == "op":
        op_local_temp_root = _validated_op_local_temp_root(
            getattr(args, "op_local_temp_root", None),
            require_empty=True,
        )
        op_local_capacity = _require_op_local_capacity(
            op_local_temp_root,
            count=args.count,
            max_attempt_multiplier=args.max_attempt_multiplier,
        )
    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite replacement shard: {destination}")
    if args.task == "op":
        assert op_local_temp_root is not None
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.local-stage.",
                dir=op_local_temp_root,
            )
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    stage_identity = _stat_identity(stage.stat(follow_symlinks=False))
    completed = False
    try:
        native_root = stage / "native_evidence"
        native_root.mkdir()
        solver_identity = _solver_identity(args.task, args)
        solver_parameters = _solver_parameters(args.task, args)
        rows: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        seen_semantic: set[str] = set()
        candidate_number = args.candidate_offset
        maximum_attempts = args.count * args.max_attempt_multiplier
        attempts = 0
        with clean.fixed_representation_environment():
            while len(rows) < args.count and attempts < maximum_attempts:
                attempts += 1
                row = generate_candidate(
                    args.task,
                    base_seed=args.base_seed,
                    candidate_number=candidate_number,
                )
                current_candidate = candidate_number
                candidate_number += 1
                transformed_unlabeled = clean.default_forbidden_transform(args.task, row)
                semantic_hash = clean.default_semantic_hash(args.task, transformed_unlabeled)
                if semantic_hash in seen_semantic:
                    continue
                row_dir = native_root / (
                    f"attempt-{attempts:06d}-candidate-{current_candidate:09d}"
                )
                row_dir.mkdir()
                _write_json_new(
                    row_dir / "structured_input.json",
                    row,
                )
                try:
                    if args.task == "cvrp":
                        output, native = solve_cvrp_lkh(
                            row,
                            row_dir=row_dir,
                            executable=args.lkh_executable.resolve(strict=True),
                            timeout_seconds=args.timeout_seconds,
                        )
                    elif args.task == "mis":
                        output, native = solve_mis_gurobi(
                            row,
                            row_dir=row_dir,
                            timeout_seconds=args.timeout_seconds,
                        )
                    else:
                        assert op_local_temp_root is not None
                        output, native = solve_op_gurobi_incumbent(
                            row,
                            row_dir=row_dir,
                            local_temp_root=op_local_temp_root,
                            timeout_seconds=args.timeout_seconds,
                            work_limit=args.work_limit,
                            node_limit=args.node_limit,
                        )
                    row["output"] = output
                    transformed = clean.default_transform(args.task, row)
                    strict = clean.default_strict_verify(args.task, transformed)
                    observed_hash = clean.default_semantic_hash(args.task, transformed)
                    if observed_hash != semantic_hash:
                        raise ReplacementGenerationError(
                            "semantic instance hash changed after attaching a label"
                        )
                    _write_json_new(
                        row_dir / "strict_replay.json",
                        {
                            "semantic_instance_sha256": semantic_hash,
                            "demonstration_sha256": clean.demonstration_sha256(transformed),
                            "strict_result": strict,
                            "native_result": native,
                        },
                    )
                except Exception as exc:
                    _write_json_new(
                        row_dir / "rejected.json",
                        {"error_type": type(exc).__name__, "error": str(exc)},
                    )
                    continue
                seen_semantic.add(semantic_hash)
                rows.append(row)
                records.append(
                    {
                        "row_index": len(rows) - 1,
                        "candidate_number": current_candidate,
                        "derived_seed": row["replacement_generation"]["derived_seed"],
                        "semantic_instance_sha256": semantic_hash,
                        "demonstration_sha256": clean.demonstration_sha256(transformed),
                        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                        "native_status": native["native_status"],
                        "native_status_code": native["native_status_code"],
                        "native_incumbent": native.get(
                            "native_incumbent", native.get("native_incumbent_raw_euclidean")
                        ),
                        "native_best_bound": native.get("native_best_bound"),
                        "native_gap": native.get("native_gap"),
                        "native_wall_time_seconds": native.get("native_wall_time_seconds"),
                        "strict_replay": _file_record(
                            row_dir / "strict_replay.json", relative_to=stage
                        ),
                    }
                )
        if len(rows) != args.count:
            raise ReplacementGenerationError(
                f"generated only {len(rows)}/{args.count} strict replacements in "
                f"{attempts} attempts"
            )
        rows_completed_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rows_completed_elapsed = time.monotonic() - generation_started_monotonic
        if args.task == "op":
            _emit_phase_event(
                "all_local_rows_complete",
                task="op",
                attempts=attempts,
                rows=len(rows),
                elapsed_seconds=rows_completed_elapsed,
            )
        output_path = stage / "replacements.json"
        _write_json_new(output_path, rows)
        evidence_manifest = stage / "evidence_manifest.json"
        _write_json_new(evidence_manifest, records)
        receipt: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "strict_solver_backed_replacement_shard_complete",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "task": args.task,
            "base_seed": args.base_seed,
            "candidate_offset": args.candidate_offset,
            "attempts": attempts,
            "rows": len(rows),
            "unique_semantic_instances": len(seen_semantic),
            "output": _file_record(output_path, relative_to=stage),
            "evidence_manifest": _file_record(evidence_manifest, relative_to=stage),
            "ordered_semantic_instance_sha256": clean.sha256_json(
                [record["semantic_instance_sha256"] for record in records]
            ),
            "ordered_demonstration_sha256": clean.sha256_json(
                [record["demonstration_sha256"] for record in records]
            ),
            "solver_identity": solver_identity,
            "solver_parameters": solver_parameters,
            "implementation": {
                "producer": _file_record(Path(__file__).resolve(strict=True)),
                "generator": _file_record(SCRIPT_DIR / "generate_iid_benchmark.py"),
                "reference_runner": _file_record(SCRIPT_DIR / "run_reference_evidence.py"),
                "strict_verifier": clean.implementation_identity(),
            },
        }
        if args.task == "op":
            assert op_local_capacity is not None
            receipt["phase_evidence"] = {
                "protocol": "clasp-op-local-stage-then-atomic-remote-publication-v1",
                "local_generation_started_at_utc": generation_started_at_utc,
                "all_local_rows_completed_at_utc": rows_completed_at_utc,
                "all_local_rows_elapsed_seconds": rows_completed_elapsed,
                "attempts_completed_before_publication": attempts,
                "strict_rows_completed_before_publication": len(rows),
                "remote_publication_started_at_receipt_creation": False,
                "node_local_capacity_preflight": op_local_capacity,
                "publication_strategy": (
                    "sealed_local_tree_to_hidden_remote_copy_deep_validate_"
                    "atomic_noreplace"
                ),
            }
        receipt["receipt_sha256"] = clean.sha256_json(receipt)
        _write_json_new(stage / "replacement_receipt.json", receipt)
        _seal_tree(stage)
        if args.task == "op":
            _emit_phase_event(
                "local_shard_sealed",
                task="op",
                attempts=attempts,
                rows=len(rows),
            )
            _publish_completed_local_op_shard(
                stage,
                destination,
                attempts=attempts,
                rows=len(rows),
            )
        else:
            _rename_directory_noreplace(stage, destination)
        completed = True
        return receipt
    finally:
        if args.task == "op" and stage.exists():
            _cleanup_local_op_stage_after_generation(
                stage,
                expected_identity=stage_identity,
                completed=completed,
            )
        elif not completed and stage.exists():
            # Keep a failed shard out of the trusted namespace.  The temporary
            # path is private to this invocation and has never been published.
            shutil.rmtree(stage, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=2_026_082_100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--work-limit", type=float, default=60.0)
    parser.add_argument("--node-limit", type=int, default=10_000)
    parser.add_argument("--max-attempt-multiplier", type=int, default=10)
    parser.add_argument("--lkh-executable", type=Path)
    parser.add_argument("--lkh-sha256", default=PINNED_LKH_SHA256)
    parser.add_argument("--op-local-temp-root", type=Path)
    args = parser.parse_args(argv)
    if args.count <= 0 or args.candidate_offset < 0:
        parser.error("--count must be positive and --candidate-offset nonnegative")
    if args.timeout_seconds <= 0 or args.work_limit <= 0 or args.node_limit <= 0:
        parser.error("solver resource limits must be positive")
    if args.max_attempt_multiplier < 1:
        parser.error("--max-attempt-multiplier must be positive")
    if args.task == "cvrp" and args.lkh_executable is None:
        parser.error("CVRP replacement generation requires --lkh-executable")
    if args.task == "op":
        try:
            args.op_local_temp_root = _validated_op_local_temp_root(
                args.op_local_temp_root,
                require_empty=True,
            )
        except ReplacementGenerationError as exc:
            parser.error(str(exc))
    elif args.op_local_temp_root is not None:
        parser.error("--op-local-temp-root is valid only for OP replacement generation")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = generate_shard(args)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
