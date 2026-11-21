#!/usr/bin/env python3
"""Build and independently audit a new two-task holdout; never claim confirmation.

Only ``build`` generates instances. ``audit`` scans all eight forbidden sources;
``freeze`` reopens the resulting evidence and requires all MIS optima certified.
Every phase is create-only and preserves all 300 rows per task on solver failure.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from clasp.data.benchmark import (  # noqa: E402
    benchmark_row_sha256,
    semantic_instance_sha256,
    sha256_json,
)
from clasp.rewards.co_tasks import (  # noqa: E402
    _check_cvrp_constraints,
    _check_mis_constraints,
    _cvrp_total_distance,
    _parse_cvrp_solution,
    _parse_mis_solution,
)

TASKS = ("cvrp", "mis")
LABELS = ("sft_train", "rl_train", "public_eval", "inspected_dev100")
PLAN_ID = "grpo-incumbent-two-task300-holdout-20260907-v1"
SOURCES = (
    "scripts/build_grpo_incumbent_holdout_20260907.py",
    "scripts/build_fair_development_benchmark.py",
    "scripts/generate_iid_benchmark.py",
    "scripts/refine_fair_cvrp_references.py",
    "scripts/run_reference_evidence.py",
    "scripts/build_clean_sft500k_corpus.py",
    "src/clasp/data/benchmark.py",
    "src/clasp/rewards/co_tasks.py",
)


def helper(name):
    key = "holdout_20260907_" + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO / "scripts" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def seal(payload, field):
    require(field not in payload, "cannot reseal an existing receipt")
    return {**payload, field: sha256_json(payload)}


def checked(path, field):
    payload = json.loads(Path(path).read_text())
    require(payload.get(field) == sha256_json({k: v for k, v in payload.items() if k != field}),
            f"invalid {field}: {path}")
    return payload


def verify_live_sources(plan):
    require({name: identity(REPO / name) for name in SOURCES} == plan["source_files"],
            "executing helper source differs from predeclared bytes")


def validate_plan(plan):
    require(plan.get("plan_id") == PLAN_ID and plan.get("instances_per_task") == 300, "wrong holdout plan/count")
    require(plan.get("task_seeds") == {"cvrp": 2026090701, "mis": 2026090702}, "wrong task seeds")
    require(plan.get("workers") == 4, "exactly four reference workers required")
    require(plan.get("runtime_versions") == {"numpy": "2.3.2", "networkx": "3.5", "ortools": "9.15.6755"},
            "wrong reference runtime pins")
    require(plan.get("reference_parameters") == {
        "cvrp_seconds": 10.0, "mis_deterministic_seconds": 5.0, "mis_wall_seconds": 30.0,
        "mis_solver_seed": 20260905, "mis_refine_deterministic_seconds": 180.0,
        "mis_refine_wall_seconds": 300.0, "mis_max_refinement_rows": 20,
        "lkh_seeds": [101, 202], "lkh_seconds": 10.0, "lkh_external_grace_seconds": 5.0,
        "lkh_integer_distance_scale": 1000, "lkh_precision": 100,
    }, "reference procedure differs from the predeclaration")
    entries = plan.get("forbidden_sources", [])
    require(len(entries) == 8 and {(r["task"], r["label"]) for r in entries}
            == {(task, label) for task in TASKS for label in LABELS}, "all eight forbidden sources required")
    for item in entries:
        expected = 500000 if item["label"] == "sft_train" else 300 if item["label"] == "public_eval" else (
            100 if item["label"] == "inspected_dev100" else 2000 if item["task"] == "cvrp" else 3200)
        require(item.get("expected_rows") == expected, "incorrect forbidden-source row count")
        require(isinstance(item.get("sha256"), str) and len(item["sha256"]) == 64
                and type(item.get("size_bytes")) is int and item["size_bytes"] > 0, "missing source byte identity")


def canonical_reference(task, row, text, native):
    """Recompute the reference under the evaluator's metric and retain solver evidence."""
    instance = row["instance_json"]
    if task == "cvrp":
        solution = _parse_cvrp_solution(text)["routes"]
        require(_check_cvrp_constraints(solution, *instance)[2], "infeasible CVRP reference")
        objective = float(_cvrp_total_distance(instance[0], solution))
        require(math.isfinite(objective) and objective > 0, "invalid raw CVRP distance")
        require(math.isclose(objective, native["objective"], rel_tol=1e-12, abs_tol=1e-9),
                "CVRP native objective differs from raw route replay")
        output = f"Routes: {solution}, Objective: {objective:.17g}"
        certified = False
    else:
        solution = _parse_mis_solution(text)["node_set"]
        require(_check_mis_constraints(solution, instance["num_nodes"], instance["edges"])[2],
                "infeasible MIS reference")
        objective = len(solution)
        require(native["objective"] == objective, "MIS objective differs from verified set cardinality")
        bound = native.get("best_bound")
        response = str(native.get("native_response_stats", ""))
        response_fields = {}
        for name in ("status", "objective", "best_bound"):
            found = re.findall(rf"^{name}:\s*(\S+)\s*$", response, re.MULTILINE)
            if len(found) == 1:
                response_fields[name] = found[0]
        response_matches = response_fields.get("status") == "OPTIMAL"
        try:
            response_matches = (response_matches and float(response_fields["objective"]) == objective
                                and float(response_fields["best_bound"]) == objective)
        except (KeyError, ValueError):
            response_matches = False
        certified = (native.get("native_status") == "OPTIMAL" and native.get("native_status_code") == 4
                     and type(native.get("native_status_code")) is int and isinstance(bound, (int, float))
                     and not isinstance(bound, bool) and math.isfinite(bound) and bound == objective
                     and response_matches)
        if native.get("native_status") == "OPTIMAL" or native.get("optimality_certified"):
            require(certified, "MIS optimum assertion lacks matching native status/bound evidence")
        output = f"Set: {solution}, Objective: {objective}"
    return output, {
        "objective": objective, "solution": solution, "strict_verifier_feasible": True,
        "optimality_certified": certified,
        "reference_quality": "certified_optimum" if certified else (
            "verified_heuristic_reference" if task == "cvrp" else "verified_feasible_reference"),
        "objective_semantics": "raw_float64_euclidean_distance" if task == "cvrp"
        else "integer_independent_set_cardinality", "native_evidence": native,
    }


def solve_job(job):
    task, row, refine = job
    try:
        builder = helper("build_fair_development_benchmark")
        if task == "mis":
            text, native = builder.solve_mis(row, 180.0 if refine else 5.0, 300.0 if refine else 30.0, 20260905)
        else:
            backend = helper("run_reference_evidence").RawCVRPHeuristicBackend()
            text, native = builder.solve_cvrp(row, 10.0, backend)
        return canonical_reference(task, row, text, native)
    except Exception as exc:
        # Keep this row and its failed attempt; admission later rejects missing
        # references. Solver failures must never filter the sampled distribution.
        return None, {"strict_verifier_feasible": False, "optimality_certified": False,
                      "reference_quality": "unresolved", "error": f"{type(exc).__name__}: {exc}"}


def lkh_job(job):
    binary, parameter, solution, log, instance, seed = job
    result = helper("refine_fair_cvrp_references").solve_candidate(binary, parameter, solution, log, instance)
    problem = parameter.parent / "problem.vrp"
    return {**result, "seed": seed, "problem": {"path": str(problem), **identity(problem)}}


def choose_mis_reference(previous, refined):
    old_text, old = previous
    new_text, new = refined
    if not new["strict_verifier_feasible"]:
        return previous
    if old["strict_verifier_feasible"]:
        require(not new["optimality_certified"] or new["objective"] >= old["objective"],
                "certified MIS refinement contradicts a verified larger incumbent")
        if new["objective"] < old["objective"]:
            return previous
    return new_text, new


def run_build(args):
    plan = checked(args.plan, "plan_sha256")
    validate_plan(plan)
    require(not args.output_root.exists(), "holdout output must be a new directory")
    import networkx
    import numpy as np
    import ortools

    versions = {"numpy": np.__version__, "networkx": networkx.__version__, "ortools": ortools.__version__}
    require(versions == plan["runtime_versions"], f"reference runtime mismatch: {versions}")
    require(identity(args.lkh_binary) == plan["lkh_binary_identity"], "LKH binary differs from declared data-v3 binary")
    require(os.access(args.lkh_binary, os.X_OK), "declared LKH binary must be executable")
    observed_sources = {name: identity(REPO / name) for name in SOURCES}
    require(observed_sources == plan["source_files"], "holdout source differs from predeclared bytes")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "plan.json", plan)
    for name in SOURCES:
        target = root / "implementation_sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
        require(identity(target) == observed_sources[name], "source changed while copied")
    declaration = seal({
        "purpose": "new_two_task_holdout_reference_construction",
        "claim_eligibility": "not_confirmation_or_sota", "model_outputs_examined": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan["plan_sha256"],
        "source_files": observed_sources, "runtime_versions": versions,
        "python": sys.version, "python_executable": str(Path(sys.executable).resolve()),
        "output_root_at_build": str(root),
        "lkh_binary": {"path": str(args.lkh_binary.resolve()), **identity(args.lkh_binary)},
    }, "declaration_sha256")
    write_new(root / "build_declaration.json", declaration)
    generator = helper("generate_iid_benchmark")
    task_rows = {}
    for task in TASKS:
        rng, np_rng = random.Random(plan["task_seeds"][task]), np.random.RandomState(plan["task_seeds"][task])
        rows, seen = [], set()
        for index in range(300):
            row, distribution = getattr(generator, f"_{task}_row")(rng, np_rng)
            semantic = semantic_instance_sha256(task, row, require_structured=True)
            require(semantic not in seen, "generated duplicate; refusing silent resampling")
            seen.add(semantic)
            rows.append({**row, "benchmark_row_id": f"{PLAN_ID}/{task}/{index:03d}",
                         "generation_seed": plan["task_seeds"][task], "generation_index": index,
                         "distribution_family": distribution, "semantic_instance_sha256": semantic})
        task_rows[task] = rows
        write_new(root / task / "instances.json", rows)
    # All 600 immutable unlabeled rows exist before the first reference solve.
    tasks = {}
    for task in TASKS:
        rows = task_rows[task]
        with ProcessPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(solve_job, [(task, row, False) for row in rows]))
        write_new(root / task / "initial_reference_attempts.json", results)
        if task == "mis":
            pending = [i for i, (_, proof) in enumerate(results) if not proof["optimality_certified"]]
            # Exceeding the declared allowance retains every initial row
            # and bound, but makes the all-optimum admission gate fail.
            attempts = []
            if len(pending) <= 20:
                with ProcessPoolExecutor(max_workers=4) as pool:
                    refined = list(pool.map(solve_job, [(task, rows[i], True) for i in pending]))
                for index, result in zip(pending, refined, strict=True):
                    attempts.append({"index": index, "result": result})
                    results[index] = choose_mis_reference(results[index], result)
            write_new(root / task / "refinement_attempts.json", {
                "uncertified_initial_indices": pending, "allowed_rows": 20,
                "status": "attempted_all_uncertified" if len(pending) <= 20 else "blocked_refinement_row_budget",
                "attempts": attempts,
            })
        else:
            refiner = helper("refine_fair_cvrp_references")
            jobs = []
            for index, row in enumerate(rows):
                directory = root / task / "lkh" / f"instance_{index:03d}"
                directory.mkdir(parents=True)
                problem = directory / "problem.vrp"
                refiner.write_text_new(problem, refiner.serialize_cvrp(row["instance_json"], name=f"holdout_{index:03d}"))
                for seed in (101, 202):
                    parameter, solution, log = [directory / f"seed{seed}.{suffix}" for suffix in ("par", "sol", "log")]
                    refiner.write_text_new(parameter, refiner.parameter_text(problem, solution, seed=seed))
                    jobs.append((args.lkh_binary, parameter, solution, log, row["instance_json"], seed))
            with ThreadPoolExecutor(max_workers=4) as threads:
                candidates = list(threads.map(lkh_job, jobs))
            for candidate in candidates:
                for field in ("problem", "parameter", "raw_log", "solution"):
                    if candidate.get(field) is not None:
                        candidate[field]["path"] = str(Path(candidate[field]["path"]).relative_to(root))
            write_new(root / task / "lkh_attempts.json", candidates)
            for index, row in enumerate(rows):
                for candidate in candidates[2 * index:2 * index + 2]:
                    if not candidate["strict_verifier_feasible"]:
                        continue
                    text = f"Routes: {candidate['reference_routes']}"
                    native = {"objective": candidate["raw_euclidean_objective"], "selected_lkh_candidate": candidate}
                    proposed = canonical_reference(task, row, text, native)
                    if (not results[index][1]["strict_verifier_feasible"]
                            or proposed[1]["objective"] < results[index][1]["objective"]):
                        results[index] = proposed
        final_rows, proofs = [], []
        for index, (row, (text, proof)) in enumerate(zip(rows, results, strict=True)):
            final_rows.append({**row, **({"output": text} if text is not None else {})})
            proofs.append(seal({**proof, "index": index, "semantic_instance_sha256": row["semantic_instance_sha256"]},
                               "evidence_sha256"))
        write_new(root / task / "eval/test.json", final_rows)
        write_new(root / task / "reference_evidence.json", proofs)
        files = {relative: identity(root / relative) for relative in (
            f"{task}/instances.json", f"{task}/eval/test.json", f"{task}/reference_evidence.json",
            f"{task}/initial_reference_attempts.json",
            f"{task}/{'lkh_attempts.json' if task == 'cvrp' else 'refinement_attempts.json'}",
        )}
        tasks[task] = {"instances": 300, "files": files,
                       "ordered_semantic_sha256": [row["semantic_instance_sha256"] for row in rows],
                       "ordered_benchmark_row_sha256": [benchmark_row_sha256(row) for row in final_rows],
                       "reference_quality_counts": dict(Counter(proof["reference_quality"] for proof in proofs))}
    require({name: identity(REPO / name) for name in SOURCES} == observed_sources, "source changed during building")
    require(identity(args.lkh_binary) == plan["lkh_binary_identity"], "LKH binary changed during building")
    write_new(root / "build_manifest.json", seal({
        "status": "built_pending_independent_audit_and_freeze", "claim_eligibility": "not_confirmation_or_sota",
        "plan_sha256": plan["plan_sha256"], "declaration_sha256": declaration["declaration_sha256"], "tasks": tasks,
    }, "manifest_sha256"))


def validate_build(root):
    plan = checked(root / "plan.json", "plan_sha256")
    validate_plan(plan)
    verify_live_sources(plan)
    declaration = checked(root / "build_declaration.json", "declaration_sha256")
    manifest = checked(root / "build_manifest.json", "manifest_sha256")
    require(manifest["plan_sha256"] == declaration["plan_sha256"] == plan["plan_sha256"], "plan binding differs")
    require(manifest["declaration_sha256"] == declaration["declaration_sha256"], "build declaration binding differs")
    require(declaration["source_files"] == plan["source_files"]
            and declaration["runtime_versions"] == plan["runtime_versions"], "runtime/source declaration differs")
    require(set(plan["source_files"]) == set(SOURCES), "source inventory differs")
    require(set(manifest["tasks"]) == set(TASKS), "holdout task inventory differs")
    require(identity(Path(__file__)) == plan["source_files"]["scripts/build_grpo_incumbent_holdout_20260907.py"],
            "audit/freeze implementation differs from predeclared builder")
    for name, expected in plan["source_files"].items():
        require(identity(root / "implementation_sources" / name) == expected, "retained implementation source differs")
    for task in TASKS:
        entry = manifest["tasks"][task]
        require(entry["instances"] == 300, "missing holdout rows")
        expected_files = {f"{task}/{suffix}" for suffix in (
            "instances.json", "eval/test.json", "reference_evidence.json", "initial_reference_attempts.json",
            "lkh_attempts.json" if task == "cvrp" else "refinement_attempts.json",
        )}
        require(set(entry["files"]) == expected_files, "required reference artifact inventory differs")
        for relative, expected in entry["files"].items():
            require(identity(root / relative) == expected, f"built artifact differs: {relative}")
        rows = json.loads((root / task / "eval/test.json").read_text())
        original = json.loads((root / task / "instances.json").read_text())
        require(len(rows) == len(original) == 300, "holdout row count changed")
        require([{k: v for k, v in row.items() if k != "output"} for row in rows] == original,
                "instance or prompt changed during reference construction")
        semantic = [semantic_instance_sha256(task, row, require_structured=True) for row in rows]
        require(len(set(semantic)) == 300 and semantic == entry["ordered_semantic_sha256"], "semantic order differs")
        require([benchmark_row_sha256(row) for row in rows] == entry["ordered_benchmark_row_sha256"], "row identity differs")
    return plan, manifest


def run_audit(args):
    root = args.output_root.resolve()
    require(not (root / "overlap_audit.json").exists(), "overlap audit is create-only")
    plan, manifest = validate_build(root)
    sources = json.loads(args.forbidden_sources.read_text()) if args.forbidden_sources else plan["forbidden_sources"]
    declared = {(row["task"], row["label"]): row for row in plan["forbidden_sources"]}
    require(len(sources) == 8 and {(row["task"], row["label"]) for row in sources} == set(declared),
            "audit requires exactly eight declared task/source pairs")
    records = []
    for source in sources:
        task, label = source["task"], source["label"]
        expected = declared[task, label]
        require({k: v for k, v in source.items() if k != "path"}
                == {k: v for k, v in expected.items() if k != "path"}, "only source replica paths may change")
        path = Path(source["path"])
        before = identity(path)
        require(before == {key: expected[key] for key in ("sha256", "size_bytes")}, "forbidden source identity differs")
        candidates = set(manifest["tasks"][task]["ordered_semantic_sha256"])
        count, overlap, digest = 0, [], hashlib.sha256()
        for index, row in helper("build_clean_sft500k_corpus").iter_json_array(path):
            semantic = semantic_instance_sha256(task, row)
            digest.update((semantic + "\n").encode())
            if semantic in candidates:
                overlap.append({"row_index": index, "semantic_instance_sha256": semantic})
            count += 1
        require(identity(path) == before, "forbidden source changed while scanned")
        require(count == expected["expected_rows"], "forbidden source row count differs")
        records.append({"task": task, "label": label, "path": str(path.resolve()), **before,
                        "rows": count, "ordered_semantic_digest": digest.hexdigest(), "overlaps": overlap})
        print(f"Audited {task}/{label}: {count} rows, {len(overlap)} overlaps", flush=True)
    total = sum(len(record["overlaps"]) for record in records)
    verify_live_sources(plan)
    write_new(root / "overlap_audit.json", seal({
        "status": "passed_zero_semantic_overlap" if total == 0 else "failed_semantic_overlap",
        "total_overlaps": total, "sources": records, "plan_sha256": plan["plan_sha256"],
        "build_manifest_sha256": manifest["manifest_sha256"], "audit_source": identity(Path(__file__)),
    }, "audit_sha256"))
    require(total == 0, "holdout overlaps a forbidden source; no replacement or row dropping permitted")


def run_freeze(args):
    root = args.output_root.resolve()
    require(not (root / "heldout_manifest.json").exists(), "holdout freeze is create-only")
    plan, manifest = validate_build(root)
    declaration = checked(root / "build_declaration.json", "declaration_sha256")
    audit = checked(root / "overlap_audit.json", "audit_sha256")
    require(audit["plan_sha256"] == plan["plan_sha256"] and audit["build_manifest_sha256"] == manifest["manifest_sha256"],
            "overlap audit belongs to different data")
    require(audit["status"] == "passed_zero_semantic_overlap" and audit["total_overlaps"] == 0, "overlap audit failed")
    require(audit["audit_source"] == identity(Path(__file__)), "overlap auditor differs from declared implementation")
    declared = {(entry["task"], entry["label"]): entry for entry in plan["forbidden_sources"]}
    require(len(audit["sources"]) == 8 and {(entry["task"], entry["label"]) for entry in audit["sources"]} == set(declared),
            "overlap audit does not cover all eight sources")
    for entry in audit["sources"]:
        expected = declared[entry["task"], entry["label"]]
        require(not entry["overlaps"] and entry["rows"] == expected["expected_rows"]
                and all(entry[key] == expected[key] for key in ("sha256", "size_bytes")), "source audit binding differs")
    for task in TASKS:
        rows = json.loads((root / task / "eval/test.json").read_text())
        proofs = json.loads((root / task / "reference_evidence.json").read_text())
        initial = json.loads((root / task / "initial_reference_attempts.json").read_text())
        require(len(initial) == 300, "initial reference attempt count differs")
        if task == "cvrp":
            candidates = json.loads((root / task / "lkh_attempts.json").read_text())
            require(len(candidates) == 600, "both declared LKH seeds must be attempted on every row")
        else:
            refinements = json.loads((root / task / "refinement_attempts.json").read_text())
            pending = [i for i, (_, proof) in enumerate(initial) if not proof["optimality_certified"]]
            require(refinements["uncertified_initial_indices"] == pending and refinements["allowed_rows"] == 20,
                    "MIS refinement selection/budget differs")
            require(len(pending) <= 20, "MIS refinement row budget exceeded; preserve rows without admission")
            require(refinements["status"] == "attempted_all_uncertified"
                    and [attempt["index"] for attempt in refinements["attempts"]] == pending,
                    "MIS refinement must cover exactly all initially uncertified rows")
        require(len(proofs) == 300, "reference evidence row count differs")
        for index, (row, proof) in enumerate(zip(rows, proofs, strict=True)):
            require(proof["evidence_sha256"] == sha256_json({k: v for k, v in proof.items() if k != "evidence_sha256"}),
                    "reference evidence hash differs")
            require(proof["index"] == index and proof["semantic_instance_sha256"] == row["semantic_instance_sha256"],
                    "reference evidence identifies another row")
            require(proof["strict_verifier_feasible"] and "output" in row, "missing verified reference")
            text, observed = canonical_reference(task, row, row["output"], proof["native_evidence"])
            require(text == row["output"] and all(observed[key] == proof[key] for key in observed),
                    "reference failed independent native/solution replay")
            require(task != "mis" or observed["optimality_certified"], "all 300 MIS optima must be certified")
            available = []
            old_text, old = initial[index]
            if old["strict_verifier_feasible"]:
                original = canonical_reference(task, row, old_text, old["native_evidence"])
                require(original == (old_text, old), "initial reference failed replay")
                available.append(original)
            if task == "cvrp":
                refiner = helper("refine_fair_cvrp_references")
                for seed, candidate in zip((101, 202), candidates[2 * index:2 * index + 2], strict=True):
                    require(candidate["seed"] == seed, "LKH seed order differs")
                    for field in ("problem", "parameter", "raw_log", "solution"):
                        if candidate.get(field) is not None:
                            artifact = candidate[field]
                            relative = Path(artifact["path"])
                            require(not relative.is_absolute() and ".." not in relative.parts, "nonrelative LKH artifact")
                            require(identity(root / relative) == {k: artifact[k] for k in ("sha256", "size_bytes")},
                                    "native LKH artifact differs")
                    problem = root / candidate["problem"]["path"]
                    require(problem.read_text() == refiner.serialize_cvrp(row["instance_json"], name=f"holdout_{index:03d}"),
                            "native LKH problem is not the declared optimization instance")
                    original_dir = Path(declaration["output_root_at_build"]) / task / "lkh" / f"instance_{index:03d}"
                    expected_parameter = refiner.parameter_text(original_dir / "problem.vrp",
                                                               original_dir / f"seed{seed}.sol", seed=seed)
                    require((root / candidate["parameter"]["path"]).read_text() == expected_parameter,
                            "native LKH parameters differ from the fixed search budget")
                    if candidate["strict_verifier_feasible"]:
                        require(candidate["returncode"] == 0 and not candidate["external_timeout"], "failed LKH candidate admitted")
                        routes = refiner.parse_lkh_routes((root / candidate["solution"]["path"]).read_text(),
                                                         node_count=len(row["instance_json"][0]))
                        require(routes == candidate["reference_routes"], "LKH routes differ from native solution")
                        available.append(canonical_reference(task, row, f"Routes: {routes}", {
                            "objective": candidate["raw_euclidean_objective"], "selected_lkh_candidate": candidate,
                        }))
                expected = min(available, key=lambda result: result[1]["objective"])
            else:
                expected = initial[index]
                for attempt in refinements["attempts"]:
                    if attempt["index"] == index:
                        ref_text, ref_proof = attempt["result"]
                        if ref_proof["strict_verifier_feasible"]:
                            replay = canonical_reference(task, row, ref_text, ref_proof["native_evidence"])
                            require(replay == (ref_text, ref_proof), "MIS refinement failed replay")
                        expected = choose_mis_reference(expected, attempt["result"])
            require(tuple(expected) == (text, observed), "final reference is not the declared best verified selection")
    verify_live_sources(plan)
    write_new(root / "heldout_manifest.json", seal({
        "status": "frozen_new_two_task_holdout_requires_model_predeclaration",
        "claim_eligibility": "not_confirmation_or_sota", "model_outputs_examined": False,
        "plan_sha256": plan["plan_sha256"], "build_manifest_sha256": manifest["manifest_sha256"],
        "overlap_audit_sha256": audit["audit_sha256"], "tasks": manifest["tasks"],
        "scope": "300 new instances per task; CVRP heuristic gap-to-reference and certified MIS optima",
        "model_evaluation_allowed_only_after_separate_preoutput_binding": True,
    }, "heldout_manifest_sha256"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "audit", "freeze"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--lkh-binary", type=Path)
    parser.add_argument("--forbidden-sources", type=Path)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    if args.phase == "build" and (args.plan is None or args.lkh_binary is None):
        parser.error("build requires --plan and --lkh-binary")
    {"build": run_build, "audit": run_audit, "freeze": run_freeze}[args.phase](args)


if __name__ == "__main__":
    main()
