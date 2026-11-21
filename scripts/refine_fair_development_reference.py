#!/usr/bin/env python3
"""Refine one MIS development reference without changing its parent dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from clasp.data.benchmark import semantic_instance_sha256, sha256_json
from build_fair_development_benchmark import file_identity, solve_mis, write_new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--row-index", type=int, default=84)
    parser.add_argument("--deterministic-seconds", type=float, default=180.)
    parser.add_argument("--wall-seconds", type=float, default=300.)
    parser.add_argument("--ortools-site-packages", required=True, type=Path)
    args = parser.parse_args()
    sys.path.append(str(args.ortools_site_packages.resolve()))
    import ortools

    parent, output, evidence_root = args.parent_root.resolve(), args.output_root.resolve(), args.evidence_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    evidence_root.mkdir(parents=True, exist_ok=False)
    parent_manifest = json.loads((parent / "development_manifest.json").read_text())
    parent_hash = parent_manifest.pop("manifest_sha256")
    assert sha256_json(parent_manifest) == parent_hash
    rows = json.loads((parent / "mis/eval/test.json").read_text())
    old_evidence = json.loads((parent / "mis/reference_evidence.json").read_text())
    row, old = rows[args.row_index], old_evidence[args.row_index]
    assert old["index"] == args.row_index
    identity = semantic_instance_sha256("mis", row, require_structured=True)
    assert identity == old["semantic_instance_sha256"]
    plan = {
        "purpose": "development_reference_only_refinement_before_policy_outputs",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_manifest_sha256": parent_hash, "parent_dataset": file_identity(parent / "mis/eval/test.json"),
        "parent_evidence": file_identity(parent / "mis/reference_evidence.json"),
        "row_index": args.row_index, "semantic_instance_sha256": identity,
        "initial_incumbent": old["objective"], "initial_bound": old["best_bound"],
        "ortools_version": ortools.__version__, "solver_seed": 20260905,
        "solver_workers": 1, "deterministic_time_limit": args.deterministic_seconds,
        "wall_time_limit_seconds": args.wall_seconds,
        "refinement_script": file_identity(Path(__file__)),
        "solver_implementation": file_identity(REPO / "scripts/build_fair_development_benchmark.py"),
        "instance_selection_reason": "sole uncertified reference in the frozen v1 dataset; no model outputs inspected",
        "output_changes_allowed": ["MIS row reference output", "MIS row reference solver evidence", "derived manifests and audit bindings"],
    }
    plan["plan_sha256"] = sha256_json(plan)
    write_new(evidence_root / "refinement_plan.json", plan)
    new_output, result = solve_mis(row, args.deterministic_seconds, args.wall_seconds, 20260905)
    result.update({"index": args.row_index, "semantic_instance_sha256": identity,
                   "distribution": old["distribution"], "refinement_plan_sha256": plan["plan_sha256"]})
    result["evidence_sha256"] = sha256_json(result)
    write_new(evidence_root / "refinement_result.json", result)
    if result["objective"] < old["objective"]:
        raise RuntimeError("Refinement has a worse incumbent; retain the parent benchmark")
    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("development_manifest.json", "training_overlap_audit.json", "README.md"))
    rows[args.row_index] = {**row, "output": new_output}
    old_evidence[args.row_index] = result
    # These copied files belong solely to this new, not-yet-exported version.
    (output / "mis/eval/test.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    (output / "mis/reference_evidence.json").write_text(json.dumps(old_evidence, indent=2, sort_keys=True) + "\n")
    new_manifest = dict(parent_manifest)
    new_manifest["reference_refinement"] = {
        "parent_manifest_sha256": parent_hash, "plan_sha256": plan["plan_sha256"],
        "task": "mis", "row_index": args.row_index,
        "previous_incumbent": old["objective"], "new_incumbent": result["objective"],
        "previous_best_bound": old["best_bound"], "new_best_bound": result["best_bound"],
        "semantic_instance_set_unchanged": True,
    }
    for task in ("cvrp", "mis"):
        task_data = output / task / "eval/test.json"
        observed_rows = json.loads(task_data.read_text())
        observed = sorted(semantic_instance_sha256(task, item, require_structured=True) for item in observed_rows)
        assert observed == parent_manifest["tasks"][task]["semantic_instance_sha256"]
        new_manifest["tasks"][task]["dataset"] = file_identity(task_data)
        new_manifest["tasks"][task]["reference_evidence"] = file_identity(output / task / "reference_evidence.json")
    new_manifest["tasks"]["mis"]["reference_quality_counts"] = dict(Counter(e["reference_quality"] for e in old_evidence))
    new_manifest["manifest_sha256"] = sha256_json(new_manifest)
    write_new(output / "development_manifest.json", new_manifest)
    write_new(output / "reference_refinement_plan.json", plan)
    write_new(output / "reference_refinement_result.json", result)
    print(json.dumps({"output_root": str(output), "manifest_sha256": new_manifest["manifest_sha256"],
                      "refinement": new_manifest["reference_refinement"],
                      "mis_reference_quality": new_manifest["tasks"]["mis"]["reference_quality_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
