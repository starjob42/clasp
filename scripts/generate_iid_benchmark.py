#!/usr/bin/env python3
"""Generate, reference, and assemble a leakage-auditable CLASP IID benchmark.

The workflow is intentionally two phase. ``generate`` writes instances with no
labels, so they cannot accidentally be evaluated.  An independently executed
solver writes hash-bound references. ``assemble`` attaches those references
only after the shared strict verifier accepts every one.  The resulting receipt
is consumed by :mod:`scripts.freeze_benchmark` before any model inference.

This is a documented, deterministic interpretation of the released
LLMCoSolver generators; it is not claimed to be byte-identical to an
undocumented historical invocation.  ``smoke-references`` exists solely to
exercise the five-task pipeline and is rejected by a publication freeze unless
the explicit smoke override is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import networkx as nx
import numpy as np

from clasp.data.benchmark import (
    PUBLICATION_INSTANCES_PER_TASK,
    REFERENCE_EVIDENCE_CONTRACT_STATUS,
    SUPPORTED_BENCHMARK_TASKS,
    benchmark_row_sha256,
    instance_sha256,
    reference_output_sha256,
    sha256_json,
    validate_reference_evidence_approval_receipt,
    validate_reference_solver_contract_policy,
    validate_forbidden_source_registry,
)

TASKS = SUPPORTED_BENCHMARK_TASKS


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    return {
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _load_object(path: str | Path, *, label: str) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty JSON object: {path}")
    return value


def _load_rows(path: str | Path, *, label: str) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{label} must be a JSON list of objects: {path}")
    return value


def _self_hashed(value: dict[str, object], field: str) -> dict[str, object]:
    unsigned = dict(value)
    unsigned.pop(field, None)
    unsigned[field] = sha256_json(unsigned)
    return unsigned


def _verify_self_hash(value: dict[str, object], field: str, *, label: str) -> None:
    stored = str(value.get(field, ""))
    unsigned = dict(value)
    unsigned.pop(field, None)
    observed = sha256_json(unsigned)
    if not stored or stored != observed:
        raise ValueError(
            f"{label} hash mismatch: stored={stored or '<missing>'}, observed={observed}"
        )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_directory(destination: Path, writer: Callable[[Path], None]) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite benchmark directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        writer(temporary)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "networkx": nx.__version__,
    }
    return versions


def _validate_contract(
    contract: dict[str, object],
    *,
    count: int,
    smoke: bool,
    official_source_root: str | None,
) -> dict[str, object]:
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported generator contract schema")
    if (
        contract.get("status")
        != "frozen_candidate_distribution_pending_reference_solver_resolution"
    ):
        raise ValueError("generator contract is not the frozen candidate distribution")
    if contract.get("benchmark_tasks") != list(TASKS):
        raise ValueError("generator contract must list all five tasks in canonical order")
    expected_count = int(contract.get("instances_per_task", -1))
    if smoke:
        if count <= 0 or count > 5:
            raise ValueError("smoke generation count must be in [1,5]")
    elif count != expected_count or count < 1_000:
        raise ValueError(
            f"confirmation generation requires the frozen count {expected_count} (>=1000)"
        )

    seeds = contract.get("task_seeds")
    if not isinstance(seeds, dict) or set(seeds) != set(TASKS):
        raise ValueError("generator contract must freeze one seed for every task")
    if len({int(value) for value in seeds.values()}) != len(TASKS):
        raise ValueError("task seeds must be distinct")

    source = contract.get("official_source")
    if not isinstance(source, dict):
        raise ValueError("generator contract has no official_source identity")
    if source.get("license_spdx") != "MIT":
        raise ValueError("generator source license must be explicitly acknowledged as MIT")
    expected_files = source.get("files_sha256")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("generator contract has no official source file hashes")
    for relative, digest in expected_files.items():
        if not isinstance(relative, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("invalid official source hash entry in generator contract")

    source_audit: dict[str, object] = {"verified_locally": False}
    if official_source_root:
        root = Path(official_source_root).resolve()
        observed: dict[str, str] = {}
        for relative, expected in expected_files.items():
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"missing official source file: {path}")
            observed[relative] = _sha256_file(path)
            if observed[relative] != expected:
                raise ValueError(f"official source hash mismatch: {relative}")
        source_audit = {
            "verified_locally": True,
            "root": str(root),
            "observed_files_sha256": observed,
        }
    elif not smoke:
        raise ValueError("--official-source-root is required for confirmation generation")

    required = contract.get("required_runtime_versions")
    if not isinstance(required, dict):
        raise ValueError("generator contract has no required runtime versions")
    runtime = _runtime_versions()
    mismatches = {
        package: {"required": str(version), "observed": runtime.get(package)}
        for package, version in required.items()
        if runtime.get(package) != str(version)
    }
    if mismatches and not smoke:
        raise RuntimeError(f"generator runtime version mismatch: {mismatches}")
    return {
        "versions": runtime,
        "version_mismatches": mismatches,
        "source_audit": source_audit,
    }


def _minmax_scale(points: np.ndarray) -> np.ndarray:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = maximum - minimum
    if np.any(span == 0):
        raise ValueError("degenerate Gaussian-mixture coordinate dimension")
    return (points - minimum) / span


def _coordinates(
    distribution: str, count: int, np_rng: np.random.RandomState
) -> list[list[int]]:
    if distribution == "uniform":
        points = np_rng.uniform(0, 1, size=(count, 2))
    else:
        modes, center_scale = {
            "gaussian_mixture_2_5": (2, 5),
            "gaussian_mixture_3_10": (3, 10),
        }[distribution]
        cluster_counts = np_rng.multinomial(count, np.ones(modes) / modes)
        pieces = []
        for cluster_count in cluster_counts:
            center = np_rng.uniform(0, center_scale, size=(1, 2)).squeeze()
            pieces.append(
                np_rng.multivariate_normal(
                    mean=center, cov=np.eye(2, 2), size=(int(cluster_count),)
                )
            )
        points = _minmax_scale(np.concatenate(pieces, axis=0))
    return (points * 1000).astype(int).tolist()


def _top_two(coords: list[list[int]]) -> list[list[tuple[int, float]]]:
    result: list[list[tuple[int, float]]] = []
    for i, (x1, y1) in enumerate(coords):
        candidates = []
        for j, (x2, y2) in enumerate(coords):
            if i != j:
                candidates.append((j, math.hypot(x1 - x2, y1 - y2)))
        candidates.sort(key=lambda item: (item[1], item[0]))
        result.append(candidates[:2])
    return result


def _routing_input(
    coords: list[list[int]], *, values: list[int] | None, value_label: str | None
) -> str:
    neighbors = _top_two(coords)
    pieces = []
    for index, coordinate in enumerate(coords):
        nearest = [f"{node}: {distance:.1f}" for node, distance in neighbors[index]]
        value = "" if values is None else f", {value_label}: {int(values[index])}"
        pieces.append(
            f"Node {index}, coordinates: {coordinate}{value}, neighbors: {nearest};"
        )
    return ".".join("".join(pieces).rsplit(";", 1))


def _tsp_row(
    py_rng: random.Random, np_rng: np.random.RandomState
) -> tuple[dict[str, object], str]:
    sizes = list(range(10, 101))
    size = py_rng.choices(sizes, weights=sizes, k=1)[0]
    distribution = py_rng.choice(
        ["uniform", "gaussian_mixture_2_5", "gaussian_mixture_3_10"]
    )
    coords = _coordinates(distribution, size, np_rng)
    instruction = (
        f"Solve the Traveling Salesman Problem (TSP) for the given list of {size} cities. "
        "Each city is represented as a node with coordinates (x, y). "
        "Identify the shortest route that visits every city exactly once and returns to "
        "the starting city. The input includes city coordinates, the 2 nearest neighbors "
        "for each city, and their respective distances. Provide the solution in the "
        "following format:\n\n1. Route: List the nodes in the order they are visited.\n"
        "2. Objective: The objective value (total travel distance)."
    )
    return {
        "num_nodes": str(size),
        "instruction": instruction,
        "input": _routing_input(coords, values=None, value_label=None),
        "instance_json": coords,
    }, distribution


def _cvrp_row(
    py_rng: random.Random, np_rng: np.random.RandomState
) -> tuple[dict[str, object], str]:
    customers = py_rng.randint(10, 100)
    distribution = py_rng.choice(
        ["uniform", "gaussian_mixture_2_5", "gaussian_mixture_3_10"]
    )
    coords = _coordinates(distribution, customers + 1, np_rng)
    demands = np_rng.randint(1, 11, size=customers + 1).tolist()
    demands[0] = 0
    multiplier = float(np_rng.triangular(3, 6, 25))
    capacity = int(multiplier * float(np.mean(demands[1:])))
    instruction = (
        f"Solve the Capacitated Vehicle Routing Problem (CVRP) with {customers} "
        "customers and 1 depot (node 0). Each customer node has a demand. "
        f"All vehicles have the same capacity {capacity}. You must assign each customer "
        "to exactly one route and ensure that the sum of demands on each route does not "
        "exceed the vehicle capacity. Minimize the total distance traveled.\n\n"
        "The input includes city coordinates, the 2 nearest neighbors for each city, "
        "and their respective distances. Provide the solution in the following format:\n"
        "1. A list of routes, each route as an ordered list of visited nodes "
        "(start/end at the depot).\n2. Objective: The total distance of all routes."
    )
    return {
        "num_nodes": str(customers + 1),
        "vehicle_capacity": f"{capacity:.1f}",
        "instruction": instruction,
        "input": _routing_input(coords, values=demands, value_label="demand"),
        "instance_json": [coords, demands, capacity],
    }, distribution


def _nearest_neighbor_cycle_length(coords: list[list[int]]) -> float:
    remaining = set(range(1, len(coords)))
    current = 0
    total = 0.0
    while remaining:
        next_node = min(
            remaining,
            key=lambda node: (
                math.dist(coords[current], coords[node]),
                node,
            ),
        )
        total += math.dist(coords[current], coords[next_node])
        current = next_node
        remaining.remove(next_node)
    return total + math.dist(coords[current], coords[0])


def _op_row(
    py_rng: random.Random, np_rng: np.random.RandomState
) -> tuple[dict[str, object], str]:
    size = py_rng.randint(10, 100)
    distribution = py_rng.choice(
        ["uniform", "gaussian_mixture_2_5", "gaussian_mixture_3_10"]
    )
    coords = _coordinates(distribution, size, np_rng)
    prizes = np_rng.randint(1, 11, size=size).tolist()
    prizes[0] = 0
    tsp_length = _nearest_neighbor_cycle_length(coords)
    # The solver and verifier must receive the same budget that the prompt
    # exposes.  Keeping extra hidden precision makes a displayed-feasible route
    # fail (or vice versa) at the one-decimal boundary.
    route_limit = round(
        float(np_rng.uniform(0.5 * tsp_length, 0.7 * tsp_length)), 1
    )
    instruction = (
        f"Solve the Orienteering Problem with {size} nodes. Each node has (x, y) "
        "coordinates and a prize for visiting it. You must plan a route that starts at "
        f"depot 0, collecting the maximum total prize possible, subject to a maximum "
        f"route length T = {route_limit:.1f}. You may visit a subset of nodes, but the "
        "total distance traveled must not exceed T.\n\nThe input includes city coordinates, "
        "the 2 nearest neighbors for each city, and their respective distances. Provide "
        "the solution in the following format:\n1. Route: The ordered list of visited "
        "nodes.\n2. Objective: The objective value (summation of the collecting prizes)."
    )
    return {
        "num_nodes": str(size),
        "max_route_length": f"{route_limit:.1f}",
        "start_node": "0",
        "instruction": instruction,
        "input": _routing_input(coords, values=prizes, value_label="prize"),
        "instance_json": [coords, prizes, route_limit],
    }, distribution


def _mis_row(
    py_rng: random.Random, np_rng: np.random.RandomState
) -> tuple[dict[str, object], str]:
    size = py_rng.randint(10, 100)
    distribution = py_rng.choice(["er", "ba"])
    if distribution == "er":
        probability = float(np_rng.uniform(0.1, 0.4))
        edges = [
            [u, v]
            for u in range(size)
            for v in range(u + 1, size)
            if py_rng.random() < probability
        ]
    else:
        attachment = py_rng.randint(1, 4)
        graph = nx.barabasi_albert_graph(size, attachment, seed=py_rng)
        edges = [[int(u), int(v)] for u, v in graph.edges()]
    adjacency = {node: [] for node in range(size)}
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    degrees = {node: len(neighbors) for node, neighbors in adjacency.items()}
    neighbor_entries = []
    for node in range(size):
        neighbors = sorted(
            adjacency[node], key=lambda value: degrees[value], reverse=True
        )[:2]
        values = neighbors + [f"#{degrees[value]}" for value in neighbors]
        neighbor_entries.append(f"N{node}:[{','.join(map(str, values))}]")
    compact_edges = str([tuple(edge) for edge in edges]).replace(" ", "")
    input_text = f"Edges: {compact_edges}\n\n" + "; ".join(neighbor_entries)
    instruction = (
        f"Given an undirected graph with {size} nodes (0..{size - 1}) and edges "
        "specified below. For each node, we also provide up to 2 neighbors connected "
        "to it. Find a maximum independent set: the largest set of vertices where no "
        "two vertices share an edge.\n\nThe input includes the edges of the graph and "
        "the top-2 neighbors for each node in format N[a,b,#c,#d], where a and b are "
        "the top-2 neighbors, #c is degree of a, and #d is degree of b. Format:\n"
        "1. Set: The list of vertices in the maximum independent set.\n"
        "2. Objective: The size of that set."
    )
    return {
        "num_nodes": str(size),
        "instruction": instruction,
        "input": input_text,
        "instance_json": {
            "num_nodes": size,
            "edges": edges,
            "distribution": distribution,
        },
    }, distribution


def _jssp_row(
    py_rng: random.Random, np_rng: np.random.RandomState
) -> tuple[dict[str, object], str]:
    jobs_count = int(np_rng.randint(5, 21))
    machines_count = int(np_rng.randint(5, 21))
    processing = np_rng.randint(1, 100, size=(jobs_count, machines_count))
    jobs: list[list[int]] = []
    descriptions = []
    for job_index in range(jobs_count):
        machines = list(range(machines_count))
        py_rng.shuffle(machines)
        flat: list[int] = []
        operations = []
        for operation_index, machine in enumerate(machines):
            duration = int(processing[job_index, operation_index])
            flat.extend([machine, duration])
            operations.append((machine, duration))
        jobs.append(flat)
        shortest = sorted(
            enumerate(operations), key=lambda item: item[1][1]
        )[:2]
        shortest_text = [
            f"{operation}: ({machine}, {duration})"
            for operation, (machine, duration) in shortest
        ]
        descriptions.append(
            f"Job {job_index}, machines and processing times for operations: "
            f"{operations}, operators with lowest processing time: {shortest_text}; "
        )
    input_text = ".".join("".join(descriptions).replace("'", "").rsplit(";", 1))
    instruction = (
        f"Solve the Job Shop Scheduling Problem (JSSP) with {jobs_count} jobs and "
        f"{machines_count} machines. Each job consists of {machines_count} operations "
        "which need to be sequentially processed on specific machines. Each machine "
        "can process only one job at a time and each job can be processed by only one "
        "machine at a time. Identify the schedule that minimizes the maximum completion "
        "time (makespan). The input includes the information of operations for each job, "
        "including their specific machine and processing time, as well as the operators "
        "with lowest processing time and their respective machines and processing times. "
        "Provide the solution in the following format:\n\n1. Schedule: List the order "
        "that jobs are processed on each machine.\n2. Makespan: The makespan of the schedule.\n"
    )
    return {
        "n": str(jobs_count),
        "m": str(machines_count),
        "instruction": instruction,
        "input": input_text,
        "instance_json": jobs,
    }, "uniform_jobs_machines_processing"


GENERATORS = {
    "tsp": _tsp_row,
    "cvrp": _cvrp_row,
    "op": _op_row,
    "mis": _mis_row,
    "jssp": _jssp_row,
}


def generate_instances(
    *,
    contract_path: str,
    output_dir: str,
    count: int,
    smoke: bool,
    official_source_root: str | None,
    forbidden_source_registry_path: str | None = None,
) -> dict[str, object]:
    contract = _load_object(contract_path, label="generator contract")
    runtime_audit = _validate_contract(
        contract,
        count=count,
        smoke=smoke,
        official_source_root=official_source_root,
    )
    forbidden_registry_identity: dict[str, object] | None = None
    if forbidden_source_registry_path:
        forbidden_registry = _load_object(
            forbidden_source_registry_path,
            label="forbidden-source registry",
        )
        forbidden_registry_identity = validate_forbidden_source_registry(
            forbidden_registry,
            expected_tasks=TASKS,
        )
    elif not smoke:
        raise ValueError(
            "--forbidden-source-registry is required before confirmation generation"
        )
    contract_sha = sha256_json(contract)
    code_sha = _sha256_file(__file__)
    seeds = contract["task_seeds"]
    receipt_tasks: dict[str, object] = {}

    def write(destination: Path) -> None:
        if not smoke:
            assert reference_export_summary is not None
            assert reference_evidence_approval is not None
            _write_json(
                destination / "reference_export_summary.json",
                reference_export_summary,
            )
            _write_json(
                destination / "reference_evidence_approval.json",
                reference_evidence_approval,
            )
            for task in TASKS:
                tree = reference_evidence_approval["tasks"][task]["evidence_tree"]
                for entry in tree["files"]:
                    relative = Path("evidence") / task / entry["relative_path"]
                    source = reference_root / relative
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with open(source, "rb") as source_handle, open(target, "xb") as target_handle:
                        shutil.copyfileobj(source_handle, target_handle)
                        target_handle.flush()
                        os.fsync(target_handle.fileno())
                    if _file_identity(target) != {
                        "sha256": entry["sha256"],
                        "size_bytes": entry["size_bytes"],
                    }:
                        raise ValueError(f"assembled evidence copy changed for {task}")
        for task in TASKS:
            py_rng = random.Random(int(seeds[task]))
            np_rng = np.random.RandomState(int(seeds[task]))
            rows: list[dict[str, object]] = []
            hashes: list[str] = []
            seen: set[str] = set()
            attempts = 0
            distribution_counts: dict[str, int] = {}
            while len(rows) < count:
                attempts += 1
                row, distribution = GENERATORS[task](py_rng, np_rng)
                digest = instance_sha256(task, row, require_structured=True)
                if digest in seen:
                    continue
                seen.add(digest)
                index = len(rows)
                row["benchmark_row_id"] = f"{task}-{index:06d}"
                row["instance_sha256"] = digest
                row["semantic_instance_sha256"] = digest
                row["reference_status"] = "pending"
                row["generator_distribution"] = distribution
                rows.append(row)
                hashes.append(digest)
                distribution_counts[distribution] = (
                    distribution_counts.get(distribution, 0) + 1
                )
            relative = Path(task) / "eval" / "test.json"
            _write_json(destination / relative, rows)
            row_hashes = [benchmark_row_sha256(row) for row in rows]
            receipt_tasks[task] = {
                "rows": len(rows),
                "attempts": attempts,
                "seed": int(seeds[task]),
                "distribution_counts": distribution_counts,
                "relative_path": str(relative),
                "file_sha256": _sha256_file(destination / relative),
                "ordered_instance_sha256": hashes,
                "ordered_instance_hashes_sha256": sha256_json(hashes),
                "ordered_semantic_instance_sha256": hashes,
                "semantic_instance_hashes_sha256": sha256_json(hashes),
                "ordered_benchmark_row_sha256": row_hashes,
                "benchmark_row_hashes_sha256": sha256_json(row_hashes),
            }
        receipt = _self_hashed(
            {
                "schema_version": 2,
                "status": "smoke_instances_generated" if smoke else "instances_generated",
                "created_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "generator_contract_sha256": contract_sha,
                "generator_code_sha256": code_sha,
                "runtime_audit": runtime_audit,
                "forbidden_source_registry_identity": forbidden_registry_identity,
                "tasks": receipt_tasks,
                "contains_references": False,
                "publishable": False,
            },
            "receipt_sha256",
        )
        _write_json(destination / "generation_receipt.json", receipt)

    _atomic_directory(Path(output_dir), write)
    return _load_object(
        Path(output_dir) / "generation_receipt.json", label="generation receipt"
    )


def _validate_solver_contract(
    contract: dict[str, object],
    *,
    smoke: bool,
    verifier_identity_sha256: str | None = None,
) -> None:
    if not smoke:
        if verifier_identity_sha256 is None:
            raise ValueError("production solver contract validation requires verifier identity")
        validate_reference_solver_contract_policy(
            contract,
            verifier_identity_sha256=verifier_identity_sha256,
            expected_tasks=TASKS,
            expected_instances_per_task=PUBLICATION_INSTANCES_PER_TASK,
            require_production_identity=True,
        )
        return
    expected_status = "smoke_only"
    if contract.get("status") != expected_status:
        raise ValueError(
            f"solver contract status must be {expected_status!r}, got {contract.get('status')!r}"
        )
    tasks = contract.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(TASKS):
        raise ValueError("solver contract must define all five tasks")
    required_fields = (
        "backend",
        "version",
        "license",
        "parameters",
        "accepted_statuses",
        "quality_semantics",
    )
    for task in TASKS:
        record = tasks[task]
        if not isinstance(record, dict):
            raise ValueError(f"invalid solver contract record for {task}")
        for field in required_fields:
            value = record.get(field)
            if value in (None, "", [], {}) or "UNRESOLVED" in str(value).upper():
                raise ValueError(f"solver contract {task}.{field} is unresolved")
        if not isinstance(record["accepted_statuses"], list):
            raise ValueError(f"solver contract {task}.accepted_statuses must be a list")


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _objective_matches(task: str, left: object, right: object) -> bool:
    try:
        first = float(left)
        second = float(right)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(first) and math.isfinite(second)):
        return False
    if task in {"op", "mis", "jssp"}:
        return (
            abs(first - round(first)) <= 1e-9
            and abs(second - round(second)) <= 1e-9
            and round(first) == round(second)
        )
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-9)


def _validate_reference_evidence(
    *,
    task: str,
    reference: dict[str, object],
    solver_task: dict[str, object],
    expected_summary_sha256: str,
    verifier_identity: dict[str, object],
    strict_metrics: dict[str, object],
    reference_root: Path,
    expected_runner_code_sha256: str,
) -> dict[str, object]:
    """Validate the compact evidence exported by run_reference_evidence.py."""

    _verify_self_hash(
        reference,
        "reference_export_record_sha256",
        label=f"{task} reference export record",
    )
    evidence = reference.get("reference_evidence")
    if not isinstance(evidence, dict) or evidence.get("schema_version") != 1:
        raise ValueError(f"missing schema-v1 reference evidence for {task}")
    for field in (
        "reference_evidence_record_sha256",
        "reference_evidence_summary_sha256",
        "runner_code_sha256",
        "solver_parameters_sha256",
        "strict_verifier_identity_sha256",
        "strict_verifier_result_sha256",
    ):
        if not _is_sha256(evidence.get(field)):
            raise ValueError(f"invalid {task} reference evidence hash field {field}")
    if evidence["reference_evidence_summary_sha256"] != expected_summary_sha256:
        raise ValueError(f"{task} reference evidence summary hash mismatch")
    if evidence.get("runner_protocol") != "clasp-iid-reference-evidence-v1":
        raise ValueError(f"{task} reference evidence runner protocol mismatch")
    if evidence.get("runner_code_sha256") != expected_runner_code_sha256:
        raise ValueError(f"{task} reference evidence runner source is not contract-bound")
    parameters = solver_task.get("parameters")
    if evidence.get("solver_parameters") != parameters:
        raise ValueError(f"{task} reference evidence solver parameters mismatch")
    if evidence.get("solver_parameters_sha256") != sha256_json(parameters):
        raise ValueError(f"{task} reference evidence parameter hash mismatch")
    runner_contract = solver_task.get("evidence_runner")
    if not isinstance(runner_contract, dict):
        raise ValueError(f"{task} solver contract has no evidence runner")
    seed_parameter = runner_contract.get("seed_parameter")
    observed_seed = evidence.get("solver_seed")
    if seed_parameter is None:
        if task not in {"tsp", "cvrp"} or observed_seed is not None:
            raise ValueError(f"{task} deterministic reference evidence seed is invalid")
    else:
        if not isinstance(seed_parameter, str) or not isinstance(parameters, dict):
            raise ValueError(f"{task} solver contract has no evidence seed parameter")
        try:
            expected_seed = parameters[seed_parameter]
        except KeyError as exc:
            raise ValueError(f"{task} reference evidence seed is invalid") from exc
        if (
            isinstance(expected_seed, bool)
            or not isinstance(expected_seed, int)
            or isinstance(observed_seed, bool)
            or not isinstance(observed_seed, int)
        ):
            raise ValueError(f"{task} reference evidence seed is invalid")
        if observed_seed != expected_seed:
            raise ValueError(f"{task} reference evidence seed mismatch")
    expected_semantics = {
        "tsp": "clasp_raw_euclidean_float64",
        "cvrp": "clasp_raw_euclidean_float64",
        "op": "clasp_raw_euclidean_float64_integer_prizes",
        "mis": "clasp_integer_mis_cardinality",
        "jssp": "clasp_integer_jssp_makespan",
    }[task]
    expected_quality = (
        "best_known_heuristic" if task in {"tsp", "cvrp"} else "certified_optimum"
    )
    expected_native = "HEURISTIC_COMPLETE" if task in {"tsp", "cvrp"} else "OPTIMAL"
    expected_public = (
        "BEST_KNOWN_HEURISTIC_VERIFIED"
        if task in {"tsp", "cvrp"}
        else "OPTIMAL_CERTIFIED"
    )
    if (
        runner_contract.get("objective_semantics") != expected_semantics
        or evidence.get("objective_semantics") != expected_semantics
    ):
        raise ValueError(f"{task} reference evidence objective semantics mismatch")
    if (
        runner_contract.get("reference_quality") != expected_quality
        or evidence.get("reference_quality") != expected_quality
    ):
        raise ValueError(f"{task} reference evidence quality classification mismatch")
    if reference.get("solver_status") != expected_public:
        raise ValueError(f"{task} reference export has an invalid public solver status")
    if evidence.get("native_status") != expected_native:
        raise ValueError(f"{task} reference evidence has an invalid native status")
    solver_identity = evidence.get("solver_identity")
    if not isinstance(solver_identity, dict):
        raise ValueError(f"{task} reference evidence solver identity is missing")
    for field in (
        "backend_name",
        "native_version",
        "artifact_kind",
        "artifact_sha256",
        "size_bytes",
    ):
        if solver_identity.get(field) in (None, ""):
            raise ValueError(f"{task} reference evidence solver identity lacks {field}")
    if not _is_sha256(solver_identity.get("artifact_sha256")):
        raise ValueError(f"{task} reference evidence solver artifact hash is invalid")
    contract_identity = runner_contract.get("solver_identity")
    if not isinstance(contract_identity, dict):
        raise ValueError(f"{task} solver contract identity is missing")
    for field, expected in contract_identity.items():
        if solver_identity.get(field) != expected:
            raise ValueError(
                f"{task} reference evidence solver identity field {field} mismatch"
            )
    try:
        artifact_size = int(solver_identity["size_bytes"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{task} reference evidence solver artifact size is invalid") from exc
    if artifact_size <= 0:
        raise ValueError(f"{task} reference evidence solver artifact is empty")
    raw_log = evidence.get("raw_native_log")
    if not isinstance(raw_log, dict) or not _is_sha256(raw_log.get("sha256")):
        raise ValueError(f"{task} reference evidence raw native log identity is invalid")
    try:
        log_size = int(raw_log.get("size_bytes", 0))
    except (TypeError, ValueError):
        log_size = 0
    if log_size <= 0:
        raise ValueError(f"{task} reference evidence raw native log is empty")
    relative_log = raw_log.get("relative_path")
    if not isinstance(relative_log, str) or not relative_log:
        raise ValueError(f"{task} reference evidence native log path is missing")
    live_log_path = (reference_root / relative_log).resolve()
    try:
        live_log_path.relative_to(reference_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{task} reference evidence native log escapes the export") from exc
    live_log_identity = _file_identity(live_log_path)
    if (
        live_log_identity["sha256"] != raw_log.get("sha256")
        or live_log_identity["size_bytes"] != raw_log.get("size_bytes")
    ):
        raise ValueError(f"{task} reference evidence native log bytes changed")
    try:
        runtime = float(evidence.get("wall_time_seconds", -1))
    except (TypeError, ValueError):
        runtime = -1.0
    if not math.isfinite(runtime) or runtime < 0:
        raise ValueError(f"{task} reference evidence wall time is invalid")
    if not _objective_matches(
        task, evidence.get("native_incumbent"), strict_metrics.get("objective_value")
    ):
        raise ValueError(f"{task} native incumbent disagrees with the strict verifier")
    if task in {"tsp", "cvrp"}:
        if evidence.get("optimality_certificate") is not False:
            raise ValueError(f"{task} heuristic reference cannot claim optimality")
        if evidence.get("native_best_bound") is not None or evidence.get("native_gap") is not None:
            raise ValueError(f"{task} heuristic reference cannot carry a bound or gap")
    else:
        expected_code = 2 if task in {"op", "mis"} else 4
        if evidence.get("optimality_certificate") is not True:
            raise ValueError(f"{task} exact reference lacks an optimality certificate")
        if evidence.get("native_status_code") != expected_code:
            raise ValueError(f"{task} exact reference has the wrong native status code")
        if not _objective_matches(
            task, evidence.get("native_incumbent"), evidence.get("native_best_bound")
        ):
            raise ValueError(f"{task} incumbent and best bound disagree")
        try:
            native_gap = float(evidence.get("native_gap"))
        except (TypeError, ValueError):
            native_gap = math.inf
        if native_gap != 0.0:
            raise ValueError(f"{task} exact reference does not have a zero native gap")
        if task in {"op", "mis"}:
            metadata = evidence.get("native_metadata")
            if not isinstance(metadata, dict) or int(metadata.get("solution_count", 0)) <= 0:
                raise ValueError(f"{task.upper()} exact reference does not have SolCount > 0")
        else:
            metadata = evidence.get("native_metadata")
            if not isinstance(metadata, dict) or metadata.get(
                "response_stats_sha256"
            ) != live_log_identity["sha256"]:
                raise ValueError("JSSP response statistics do not match the retained log")
            parsed_stats: dict[str, str] = {}
            for line in live_log_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip() and key.strip() not in parsed_stats:
                    parsed_stats[key.strip()] = value.strip()
            if parsed_stats.get("status", "").upper() != "OPTIMAL":
                raise ValueError("JSSP retained response statistics are not OPTIMAL")
            if not _objective_matches(
                task, parsed_stats.get("objective"), evidence.get("native_incumbent")
            ) or not _objective_matches(
                task, parsed_stats.get("best_bound"), evidence.get("native_best_bound")
            ):
                raise ValueError("JSSP retained response statistics objective/bound mismatch")
    if evidence.get("strict_verifier_identity") != verifier_identity:
        raise ValueError(f"{task} reference evidence strict-verifier identity mismatch")
    if evidence.get("strict_verifier_identity_sha256") != sha256_json(verifier_identity):
        raise ValueError(f"{task} reference evidence verifier hash mismatch")
    if evidence.get("strict_verifier_result") != strict_metrics:
        raise ValueError(f"{task} stored strict-verifier result is stale")
    if evidence.get("strict_verifier_result_sha256") != sha256_json(strict_metrics):
        raise ValueError(f"{task} stored strict-verifier result hash is invalid")
    if not _objective_matches(
        task,
        evidence.get("strict_verified_objective"),
        strict_metrics.get("objective_value"),
    ):
        raise ValueError(f"{task} stored strict objective is stale")
    return copy.deepcopy(evidence)


def _validate_live_exported_evidence_tree(
    *,
    reference_root: Path,
    task: str,
    export_task: dict[str, object],
    approval_task: dict[str, object],
) -> None:
    relative_root = export_task.get("evidence_relative_root")
    if relative_root != f"evidence/{task}":
        raise ValueError(f"reference export evidence root is invalid for {task}")
    evidence_root = (reference_root / str(relative_root)).resolve()
    try:
        evidence_root.relative_to(reference_root.resolve())
    except ValueError as exc:
        raise ValueError(f"reference export evidence root escapes for {task}") from exc
    expected_tree = approval_task.get("evidence_tree")
    if export_task.get("evidence_tree") != expected_tree or not isinstance(
        expected_tree, dict
    ):
        raise ValueError(f"reference export evidence tree receipt mismatch for {task}")
    expected_files = expected_tree.get("files")
    if not isinstance(expected_files, list):
        raise ValueError(f"reference export evidence tree has no files for {task}")
    observed: list[dict[str, object]] = []
    for path in sorted(evidence_root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise ValueError(f"reference export evidence contains a symlink for {task}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"reference export evidence contains a special file for {task}")
        observed.append(
            {
                "relative_path": path.relative_to(evidence_root).as_posix(),
                **_file_identity(path),
            }
        )
    if observed != expected_files:
        raise ValueError(f"reference export evidence live bytes changed for {task}")
    if (
        expected_tree.get("file_count") != len(observed)
        or expected_tree.get("total_size_bytes")
        != sum(int(entry["size_bytes"]) for entry in observed)
        or expected_tree.get("manifest_sha256") != sha256_json(observed)
    ):
        raise ValueError(f"reference export evidence tree identity is invalid for {task}")


def assemble_references(
    *,
    draft_root: str,
    references_root: str,
    generator_contract_path: str,
    solver_contract_path: str,
    output_dir: str,
    smoke: bool,
    reference_evidence_approval_path: str | None = None,
) -> dict[str, object]:
    from clasp.entrypoints.evaluate import TASK_EVALUATORS, _strict_verifier_identity

    draft = Path(draft_root).resolve()
    receipt = _load_object(
        draft / "generation_receipt.json", label="generation receipt"
    )
    _verify_self_hash(receipt, "receipt_sha256", label="generation receipt")
    if receipt.get("schema_version") != 2:
        raise ValueError("draft generation receipt must use schema version 2")
    expected_draft_status = (
        "smoke_instances_generated" if smoke else "instances_generated"
    )
    if receipt.get("status") != expected_draft_status:
        raise ValueError(f"draft receipt status must be {expected_draft_status!r}")
    forbidden_registry_identity = receipt.get(
        "forbidden_source_registry_identity"
    )
    if not smoke and not isinstance(forbidden_registry_identity, dict):
        raise ValueError(
            "production draft receipt is not bound to a forbidden-source registry"
        )
    if smoke and forbidden_registry_identity is not None and not isinstance(
        forbidden_registry_identity, dict
    ):
        raise ValueError("draft forbidden-source registry identity is invalid")
    generator_contract = _load_object(
        generator_contract_path, label="generator contract"
    )
    if receipt.get("generator_contract_sha256") != sha256_json(generator_contract):
        raise ValueError("draft and supplied generator contract differ")
    solver_contract = _load_object(solver_contract_path, label="solver contract")
    verifier_identity = _strict_verifier_identity()
    _validate_solver_contract(
        solver_contract,
        smoke=smoke,
        verifier_identity_sha256=sha256_json(verifier_identity),
    )
    solver_contract_sha = sha256_json(solver_contract)
    reference_root = Path(references_root).resolve()
    reference_export_summary: dict[str, object] | None = None
    reference_evidence_approval: dict[str, object] | None = None
    if not smoke:
        if set(receipt.get("tasks", {})) != set(TASKS):
            raise ValueError("production draft must contain exactly all five tasks")
        if reference_evidence_approval_path is None:
            raise ValueError("production assembly requires a reference evidence approval receipt")
        reference_export_summary = _load_object(
            reference_root / "reference_export_summary.json",
            label="reference evidence export summary",
        )
        _verify_self_hash(
            reference_export_summary,
            "reference_export_summary_sha256",
            label="reference evidence export summary",
        )
        if (
            reference_export_summary.get("schema_version") != 1
            or reference_export_summary.get("status")
            != "reference_evidence_export_complete"
        ):
            raise ValueError("reference evidence export summary is not assembly-eligible")
        if reference_export_summary.get("runner_protocol") != (
            "clasp-iid-reference-evidence-v1"
        ):
            raise ValueError("reference evidence export used an unsupported runner")
        if reference_export_summary.get("source_generation_receipt_sha256") != receipt[
            "receipt_sha256"
        ]:
            raise ValueError("reference evidence export targets a different draft")
        if reference_export_summary.get("solver_contract_sha256") != solver_contract[
            "contract_sha256"
        ]:
            raise ValueError("reference evidence export targets a different solver contract")
        export_tasks = reference_export_summary.get("tasks")
        if not isinstance(export_tasks, dict) or set(export_tasks) != set(TASKS):
            raise ValueError(
                "production reference evidence export must contain all five tasks"
            )
        reference_evidence_approval = _load_object(
            reference_evidence_approval_path,
            label="reference evidence approval receipt",
        )
        snapshot_descriptor = reference_export_summary.get(
            "reference_evidence_approval_snapshot"
        )
        if not isinstance(snapshot_descriptor, dict):
            raise ValueError("reference export has no approval-receipt snapshot")
        snapshot_path = reference_root / str(snapshot_descriptor.get("relative_path", ""))
        snapshot_identity = _file_identity(snapshot_path)
        if (
            snapshot_descriptor.get("sha256") != snapshot_identity["sha256"]
            or snapshot_descriptor.get("size_bytes") != snapshot_identity["size_bytes"]
            or _load_object(snapshot_path, label="reference approval snapshot")
            != reference_evidence_approval
        ):
            raise ValueError("reference evidence approval snapshot is stale or altered")
        approval_sha = reference_evidence_approval.get(
            "reference_evidence_approval_receipt_sha256"
        )
        if (
            reference_export_summary.get(
                "reference_evidence_approval_receipt_sha256"
            )
            != approval_sha
        ):
            raise ValueError("reference export targets a different approval receipt")
        validate_reference_evidence_approval_receipt(
            reference_evidence_approval,
            solver_contract=solver_contract,
            source_generation_receipt_sha256=str(receipt["receipt_sha256"]),
            verifier_identity_sha256=sha256_json(verifier_identity),
            task_evidence_summary_sha256={
                task: str(export_tasks[task]["reference_evidence_summary_sha256"])
                for task in TASKS
            },
            expected_tasks=TASKS,
            expected_instances_per_task=PUBLICATION_INSTANCES_PER_TASK,
        )
    output_tasks: dict[str, object] = {}

    def write(destination: Path) -> None:
        for task in TASKS:
            task_receipt = receipt["tasks"][task]
            draft_path = draft / task_receipt["relative_path"]
            if _sha256_file(draft_path) != task_receipt["file_sha256"]:
                raise ValueError(f"draft file hash mismatch for {task}")
            rows = _load_rows(draft_path, label=f"{task} draft")
            references = _load_rows(
                reference_root / f"{task}.json", label=f"{task} references"
            )
            export_task: dict[str, object] | None = None
            if reference_export_summary is not None:
                candidate = reference_export_summary["tasks"].get(task)
                if not isinstance(candidate, dict):
                    raise ValueError(f"reference evidence export has no task record for {task}")
                export_task = candidate
                reference_path = reference_root / f"{task}.json"
                reference_identity = _file_identity(reference_path)
                if (
                    candidate.get("relative_path") != f"{task}.json"
                    or candidate.get("sha256") != reference_identity["sha256"]
                    or candidate.get("size_bytes") != reference_identity["size_bytes"]
                    or candidate.get("rows") != len(references)
                ):
                    raise ValueError(f"reference evidence export file identity mismatch for {task}")
                assert reference_evidence_approval is not None
                approval_task = reference_evidence_approval["tasks"].get(task)
                if not isinstance(approval_task, dict):
                    raise ValueError(f"reference approval has no task record for {task}")
                _validate_live_exported_evidence_tree(
                    reference_root=reference_root,
                    task=task,
                    export_task=candidate,
                    approval_task=approval_task,
                )
            if len(rows) != len(references) or len(rows) != task_receipt["rows"]:
                raise ValueError(f"reference count does not match draft for {task}")
            expected_order = task_receipt["ordered_instance_sha256"]
            expected_semantic_order = task_receipt.get(
                "ordered_semantic_instance_sha256"
            )
            if expected_semantic_order != expected_order:
                raise ValueError(
                    f"draft semantic instance hashes disagree with legacy hashes for {task}"
                )
            observed_draft_row_hashes = [benchmark_row_sha256(row) for row in rows]
            if task_receipt.get("ordered_benchmark_row_sha256") != observed_draft_row_hashes:
                raise ValueError(f"draft ordered benchmark row hashes mismatch for {task}")
            if task_receipt.get("benchmark_row_hashes_sha256") != sha256_json(
                observed_draft_row_hashes
            ):
                raise ValueError(f"draft aggregate benchmark row hash mismatch for {task}")
            solver_task = solver_contract["tasks"][task]
            solver_task_sha = sha256_json(solver_task)
            assembled = []
            observed_hashes = []
            objectives = []
            evidence_record_hashes: list[str] = []
            export_record_hashes: list[str] = []
            for index, (row, reference) in enumerate(zip(rows, references)):
                expected_id = f"{task}-{index:06d}"
                digest = instance_sha256(task, row, require_structured=True)
                if (
                    row.get("benchmark_row_id") != expected_id
                    or reference.get("benchmark_row_id") != expected_id
                    or row.get("instance_sha256") != digest
                    or row.get("semantic_instance_sha256") != digest
                    or reference.get("instance_sha256") != digest
                    or (
                        not smoke
                        and reference.get("semantic_instance_sha256") != digest
                    )
                    or digest != expected_order[index]
                ):
                    raise ValueError(f"reference identity mismatch for {task} row {index}")
                source_row_sha = benchmark_row_sha256(row)
                if not smoke and reference.get("benchmark_row_sha256") != source_row_sha:
                    raise ValueError(
                        f"reference benchmark-row identity mismatch for {task} row {index}"
                    )
                status = reference.get("solver_status")
                if status not in solver_task["accepted_statuses"]:
                    raise ValueError(
                        f"unaccepted solver status for {task} row {index}: {status!r}"
                    )
                if reference.get("solver_contract_task_sha256") != solver_task_sha:
                    raise ValueError(
                        f"solver provenance hash mismatch for {task} row {index}"
                    )
                output = str(reference.get("output", ""))
                if not output.strip():
                    raise ValueError(f"empty reference output for {task} row {index}")
                runtime = float(reference.get("solver_runtime_seconds", -1))
                if not math.isfinite(runtime) or runtime < 0:
                    raise ValueError(f"invalid solver runtime for {task} row {index}")
                instance = row["instance_json"]
                metrics = TASK_EVALUATORS[task](
                    str(row["instruction"]),
                    str(row["input"]),
                    output,
                    instance,
                    output,
                )
                if not metrics.get("is_feasible"):
                    raise ValueError(
                        f"shared strict verifier rejects {task} reference row {index}"
                    )
                objective = float(metrics["objective_value"])
                declared = metrics.get("declared_objective")
                if declared is None or not math.isfinite(float(declared)):
                    raise ValueError(f"reference has no declared objective: {task} row {index}")
                tolerance = max(0.051, 5e-6 * max(1.0, abs(objective)))
                if abs(float(declared) - objective) > tolerance:
                    raise ValueError(
                        f"declared objective disagrees with verifier for {task} row {index}"
                    )
                evidence: dict[str, object] | None = None
                if not smoke:
                    assert export_task is not None
                    if reference.get("reference_output_sha256") != reference_output_sha256(
                        output
                    ):
                        raise ValueError(
                            f"reference output identity mismatch for {task} row {index}"
                        )
                    summary_sha = export_task.get("reference_evidence_summary_sha256")
                    if not _is_sha256(summary_sha):
                        raise ValueError(f"invalid evidence summary hash for {task}")
                    evidence = _validate_reference_evidence(
                        task=task,
                        reference=reference,
                        solver_task=solver_task,
                        expected_summary_sha256=str(summary_sha),
                        verifier_identity=_strict_verifier_identity(),
                        strict_metrics=metrics,
                        reference_root=reference_root,
                        expected_runner_code_sha256=str(
                            solver_contract["reference_evidence_runner"][
                                "runner_code_sha256"
                            ]
                        ),
                    )
                    evidence_record_hashes.append(
                        str(evidence["reference_evidence_record_sha256"])
                    )
                    export_record_hashes.append(
                        str(reference["reference_export_record_sha256"])
                    )
                clean = dict(row)
                clean.pop("reference_status", None)
                clean["output"] = output
                clean["reference_provenance"] = {
                    "solver_status": status,
                    "solver_runtime_seconds": runtime,
                    "solver_contract_task_sha256": solver_task_sha,
                    "quality_semantics": solver_task["quality_semantics"],
                }
                if evidence is not None:
                    clean["reference_provenance"]["reference_export_record_sha256"] = (
                        reference["reference_export_record_sha256"]
                    )
                    clean["reference_provenance"]["reference_evidence"] = evidence
                    assert reference_evidence_approval is not None
                    clean["reference_provenance"][
                        "reference_evidence_approval_receipt_sha256"
                    ] = reference_evidence_approval[
                        "reference_evidence_approval_receipt_sha256"
                    ]
                assembled.append(clean)
                observed_hashes.append(digest)
                objectives.append(objective)
            relative = Path(task) / "eval" / "test.json"
            _write_json(destination / relative, assembled)
            assembled_row_hashes = [benchmark_row_sha256(row) for row in assembled]
            reference_hashes = [
                reference_output_sha256(row["output"]) for row in assembled
            ]
            output_tasks[task] = {
                "rows": len(assembled),
                "relative_path": str(relative),
                "file_sha256": _sha256_file(destination / relative),
                "ordered_instance_sha256": observed_hashes,
                "ordered_instance_hashes_sha256": sha256_json(observed_hashes),
                "ordered_semantic_instance_sha256": observed_hashes,
                "semantic_instance_hashes_sha256": sha256_json(observed_hashes),
                "ordered_benchmark_row_sha256": assembled_row_hashes,
                "benchmark_row_hashes_sha256": sha256_json(assembled_row_hashes),
                "ordered_reference_output_sha256": reference_hashes,
                "reference_output_hashes_sha256": sha256_json(reference_hashes),
                "strict_reference_objective_min": min(objectives),
                "strict_reference_objective_max": max(objectives),
            }
            if not smoke:
                assert export_task is not None
                if export_task.get(
                    "ordered_reference_evidence_record_sha256"
                ) != evidence_record_hashes:
                    raise ValueError(f"ordered evidence record hashes mismatch for {task}")
                if export_task.get(
                    "ordered_reference_export_record_sha256"
                ) != export_record_hashes:
                    raise ValueError(f"ordered export record hashes mismatch for {task}")
                if export_task.get("reference_export_records_sha256") != sha256_json(
                    export_record_hashes
                ):
                    raise ValueError(f"aggregate export record hash mismatch for {task}")
                output_tasks[task].update(
                    {
                        "reference_evidence_summary_sha256": export_task[
                            "reference_evidence_summary_sha256"
                        ],
                        "ordered_reference_evidence_record_sha256": evidence_record_hashes,
                        "reference_evidence_records_sha256": sha256_json(
                            evidence_record_hashes
                        ),
                        "ordered_reference_export_record_sha256": export_record_hashes,
                        "reference_export_records_sha256": sha256_json(
                            export_record_hashes
                        ),
                        "reference_evidence_tree_manifest_sha256": export_task[
                            "evidence_tree"
                        ]["manifest_sha256"],
                    }
                )
        assembled_receipt = _self_hashed(
            {
                "schema_version": 2,
                "status": (
                    "smoke_references_attached" if smoke else "references_attached"
                ),
                "created_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "generator_contract_sha256": receipt[
                    "generator_contract_sha256"
                ],
                "generator_code_sha256": receipt["generator_code_sha256"],
                "source_generation_receipt_sha256": receipt["receipt_sha256"],
                "forbidden_source_registry_identity": forbidden_registry_identity,
                "reference_solver_contract_sha256": solver_contract_sha,
                "strict_verifier": _strict_verifier_identity(),
                "tasks": output_tasks,
                "contains_references": True,
                "publishable": not smoke,
                **(
                    {
                        "reference_evidence_schema_version": 1,
                        "reference_export_summary_sha256": reference_export_summary[
                            "reference_export_summary_sha256"
                        ],
                        "reference_evidence_approval_schema_version": 1,
                        "reference_evidence_approval_receipt_sha256": (
                            reference_evidence_approval[
                                "reference_evidence_approval_receipt_sha256"
                            ]
                        ),
                        "reference_evidence_approval": reference_evidence_approval,
                    }
                    if reference_export_summary is not None
                    else {}
                ),
            },
            "receipt_sha256",
        )
        _write_json(destination / "generation_receipt.json", assembled_receipt)

    _atomic_directory(Path(output_dir), write)
    return _load_object(
        Path(output_dir) / "generation_receipt.json", label="assembled receipt"
    )


def _distance(coords: list[list[int]], route: list[int]) -> float:
    return sum(math.dist(coords[a], coords[b]) for a, b in zip(route, route[1:]))


def _smoke_output(task: str, row: dict[str, object]) -> str:
    instance = row["instance_json"]
    if task == "tsp":
        route = list(range(len(instance))) + [0]
        return f"Route: {route}, Objective: {_distance(instance, route):.6f}"
    if task == "cvrp":
        coords, demands, capacity = instance
        if max(demands) > capacity:
            raise ValueError("smoke CVRP instance has a demand above vehicle capacity")
        routes = [[0, customer, 0] for customer in range(1, len(coords))]
        objective = sum(_distance(coords, route) for route in routes)
        return f"Routes: {routes}, Objective: {objective:.6f}"
    if task == "op":
        coords, prizes, route_limit = instance
        candidates = sorted(
            range(1, len(coords)), key=lambda node: (math.dist(coords[0], coords[node]), node)
        )
        route = [0, candidates[0]]
        if _distance(coords, route) > route_limit:
            raise ValueError("smoke OP instance has no one-hop feasible route")
        objective = prizes[route[1]]
        return f"Route: {route}, Objective: {objective:.6f}"
    if task == "mis":
        count = int(instance["num_nodes"])
        edges = {tuple(edge) for edge in instance["edges"]}
        chosen = []
        for node in range(count):
            if all((min(node, other), max(node, other)) not in edges for other in chosen):
                chosen.append(node)
        return f"Set: {chosen}, Objective: {len(chosen)}"
    jobs_count = len(instance)
    machines_count = len(instance[0]) // 2
    schedule = [list(range(jobs_count)) for _ in range(machines_count)]
    from clasp.rewards.co_tasks import _jssp_jobs, _schedule_makespan

    jobs = _jssp_jobs(instance, "")
    makespan = _schedule_makespan(jobs, schedule)
    if makespan is None:
        raise ValueError("smoke JSSP schedule unexpectedly deadlocks")
    return f"Schedule: {schedule}, Makespan: {makespan:.0f}"


def make_smoke_references(
    *, draft_root: str, references_dir: str, solver_contract_output: str
) -> None:
    draft = Path(draft_root).resolve()
    receipt = _load_object(
        draft / "generation_receipt.json", label="generation receipt"
    )
    _verify_self_hash(receipt, "receipt_sha256", label="generation receipt")
    if receipt.get("status") != "smoke_instances_generated":
        raise ValueError("smoke references require a smoke-generated draft")
    solver_tasks = {
        task: {
            "backend": "deterministic_feasible_constructive_smoke",
            "version": "1",
            "license": "repository-internal-test-code",
            "parameters": {"quality_optimization": False},
            "accepted_statuses": ["HEURISTIC_SMOKE"],
            "quality_semantics": "feasibility-only; never a publication reference",
        }
        for task in TASKS
    }
    solver_contract = {
        "schema_version": 1,
        "status": "smoke_only",
        "tasks": solver_tasks,
    }
    contract_path = Path(solver_contract_output)
    if contract_path.exists():
        raise FileExistsError(f"refusing to overwrite solver contract: {contract_path}")

    def write_references(destination: Path) -> None:
        for task in TASKS:
            task_receipt = receipt["tasks"][task]
            rows = _load_rows(
                draft / task_receipt["relative_path"], label=f"{task} smoke draft"
            )
            task_sha = sha256_json(solver_tasks[task])
            references = [
                {
                    "benchmark_row_id": row["benchmark_row_id"],
                    "instance_sha256": row["instance_sha256"],
                    "output": _smoke_output(task, row),
                    "solver_status": "HEURISTIC_SMOKE",
                    "solver_runtime_seconds": 0.0,
                    "solver_contract_task_sha256": task_sha,
                }
                for row in rows
            ]
            _write_json(destination / f"{task}.json", references)

    _atomic_directory(Path(references_dir), write_references)
    try:
        _write_json(contract_path, solver_contract)
    except BaseException:
        shutil.rmtree(Path(references_dir), ignore_errors=True)
        raise


def _default_contract() -> str:
    return str(
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "sota_20260814"
        / "iid_generator_contract.json"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate unlabeled instances")
    generate.add_argument("--generator-contract", default=_default_contract())
    generate.add_argument("--official-source-root")
    generate.add_argument(
        "--forbidden-source-registry",
        help="Required for production and frozen before IID generation.",
    )
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--count", type=int, default=1_000)
    generate.add_argument("--smoke", action="store_true")

    smoke_refs = subparsers.add_parser(
        "smoke-references", help="create explicitly non-publishable test references"
    )
    smoke_refs.add_argument("--draft-root", required=True)
    smoke_refs.add_argument("--references-dir", required=True)
    smoke_refs.add_argument("--solver-contract-output", required=True)

    assemble = subparsers.add_parser(
        "assemble", help="attach solver references after strict verification"
    )
    assemble.add_argument("--draft-root", required=True)
    assemble.add_argument("--references-root", required=True)
    assemble.add_argument("--generator-contract", default=_default_contract())
    assemble.add_argument("--solver-contract", required=True)
    assemble.add_argument(
        "--reference-evidence-approval",
        help="Required for production; create once after all five evidence trees pass.",
    )
    assemble.add_argument("--output-dir", required=True)
    assemble.add_argument("--smoke", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "generate":
        receipt = generate_instances(
            contract_path=args.generator_contract,
            output_dir=args.output_dir,
            count=args.count,
            smoke=args.smoke,
            official_source_root=args.official_source_root,
            forbidden_source_registry_path=args.forbidden_source_registry,
        )
        print(f"generated: {Path(args.output_dir).resolve()}")
        print(f"receipt_sha256: {receipt['receipt_sha256']}")
        return 0
    if args.command == "smoke-references":
        make_smoke_references(
            draft_root=args.draft_root,
            references_dir=args.references_dir,
            solver_contract_output=args.solver_contract_output,
        )
        print(f"smoke references: {Path(args.references_dir).resolve()}")
        return 0
    receipt = assemble_references(
        draft_root=args.draft_root,
        references_root=args.references_root,
        generator_contract_path=args.generator_contract,
        solver_contract_path=args.solver_contract,
        output_dir=args.output_dir,
        smoke=args.smoke,
        reference_evidence_approval_path=args.reference_evidence_approval,
    )
    print(f"assembled: {Path(args.output_dir).resolve()}")
    print(f"receipt_sha256: {receipt['receipt_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
