#!/usr/bin/env python3
"""Freeze a reference-only CVRP revision after independently checking its evidence.

This creates a new development manifest. It never edits the parent benchmark or
the training data, and it leaves overlap re-binding to the separate audit tool.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from clasp.data.benchmark import semantic_instance_sha256, sha256_json
from build_fair_development_benchmark import file_identity, write_new
from clasp_grpo_fresh_dev_artifacts import validate_reference
from refine_fair_cvrp_references import parse_lkh_routes


def checked(path: Path, field: str) -> dict:
    value = json.loads(path.read_text())
    if sha256_json({key: item for key, item in value.items() if key != field}) != value[field]:
        raise ValueError(f"invalid {field}: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finalize(parent: Path, output: Path) -> dict:
    parent, output = parent.resolve(strict=True), output.resolve(strict=True)
    require(parent != output, "reference revisions need a different output root")
    require(not (output / "development_manifest.json").exists(), "child manifest already exists")
    require(not (output / "training_overlap_audit.json").exists(), "child audit already exists")
    previous = checked(parent / "development_manifest.json", "manifest_sha256")
    plan_path = output / "cvrp_reference_refinement_plan.json"
    receipt_path = output / "cvrp_reference_refinement_result.json"
    plan, receipt = checked(plan_path, "plan_sha256"), checked(receipt_path, "receipt_sha256")
    require(plan["parent_manifest_sha256"] == receipt["parent_manifest_sha256"] == previous["manifest_sha256"],
            "refinement parent mismatch")
    require(receipt["plan_sha256"] == plan["plan_sha256"], "refinement plan/result mismatch")
    require(plan["model_outputs_examined"] is False, "reference revision must precede policy evaluation")
    require(plan["parameters"]["seeds"] == [101, 202] and plan["parameters"]["time_seconds_per_seed"] == 10.0,
            "reference refinement differs from the frozen solver budget")
    require(receipt["instance_count"] == len(plan["workload"]) == 100, "expected 100 CVRP rows")
    for snapshot in [plan["lkh_binary"], plan["refinement_script"], plan["verifier_source"], *plan["source_files"]]:
        require(file_identity(Path(snapshot["path"])) == snapshot, "frozen solver implementation changed")
    manifest = deepcopy(previous)
    manifest.pop("manifest_sha256")
    changed, improvements = [], []
    for task in ("cvrp", "mis"):
        parent_data, new_data = (root / task / "eval/test.json" for root in (parent, output))
        parent_proof, new_proof = (root / task / "reference_evidence.json" for root in (parent, output))
        for kind, path in (("dataset", parent_data), ("reference_evidence", parent_proof)):
            require(file_identity(path)["sha256"] == previous["tasks"][task][kind]["sha256"],
                    f"parent {task} {kind} changed")
        before, after, old_proofs, proofs = [json.loads(path.read_text()) for path in
                                           (parent_data, new_data, parent_proof, new_proof)]
        require(len(before) == len(after) == len(old_proofs) == len(proofs) == 100, f"{task} row count changed")
        identities = []
        for index, (old, new, old_proof, proof) in enumerate(zip(before, after, old_proofs, proofs, strict=True)):
            instance_id = semantic_instance_sha256(task, new, require_structured=True)
            require(instance_id == semantic_instance_sha256(task, old, require_structured=True), "instance changed")
            require({key: value for key, value in old.items() if key != "output"}
                    == {key: value for key, value in new.items() if key != "output"}, "non-reference field changed")
            require(proof["index"] == index and proof["semantic_instance_sha256"] == instance_id, "proof identity mismatch")
            validate_reference(task, old, old_proof)
            validate_reference(task, new, proof)
            identities.append(instance_id)
            if task == "cvrp":
                require(proof["refinement_plan_sha256"] == plan["plan_sha256"], "row refinement plan mismatch")
                require(proof["parent_reference_objective"] == old_proof["objective"], "row parent objective mismatch")
                require(proof["objective"] <= old_proof["objective"], "reference became weaker")
                improvement = old_proof["objective"] - proof["objective"]
                require(abs(improvement - proof["refinement_improvement"]) <= 1e-9, "reference improvement mismatch")
                improvements.append(improvement / old_proof["objective"])
                workload = plan["workload"][index]
                require(workload["index"] == index and workload["semantic_instance_sha256"] == instance_id,
                        "frozen solver workload identity mismatch")
                require(file_identity(Path(workload["problem"]["path"])) == workload["problem"],
                        "frozen solver problem file changed")
                candidates = proof["lkh_candidates"]
                require([candidate["seed"] for candidate in candidates] == [101, 202], "candidate seeds changed")
                objectives = [old_proof["objective"]]
                for candidate, trial in zip(candidates, workload["trials"], strict=True):
                    require(candidate["parameter"] == trial["parameter"], "candidate parameter identity changed")
                    for field in ("parameter", "raw_log", "solution"):
                        identity = candidate[field]
                        if identity is not None:
                            require(file_identity(Path(identity["path"])) == identity, "candidate artifact identity mismatch")
                    if candidate["strict_verifier_feasible"]:
                        require(parse_lkh_routes(Path(candidate["solution"]["path"]).read_text(),
                                                 node_count=len(new["instance_json"][0])) == candidate["reference_routes"],
                                "raw LKH incumbent differs from candidate evidence")
                        candidate_proof = {**proof, "objective": candidate["raw_euclidean_objective"],
                                           "reference_routes": candidate["reference_routes"]}
                        candidate_proof.pop("evidence_sha256")
                        candidate_proof["evidence_sha256"] = sha256_json(candidate_proof)
                        candidate_row = {**new, "output": f"Routes: {candidate['reference_routes']}, Objective: {candidate['raw_euclidean_objective']:.17g}"}
                        validate_reference(task, candidate_row, candidate_proof)
                        objectives.append(candidate["raw_euclidean_objective"])
                require(proof["objective"] == min(objectives), "selected reference is not the best verified candidate")
                if old["output"] != new["output"]:
                    changed.append(index)
        require(sorted(identities) == previous["tasks"][task]["semantic_instance_sha256"], "semantic set mismatch")
        if task == "mis":
            require(parent_data.read_bytes() == new_data.read_bytes(), "MIS dataset bytes changed")
            require(parent_proof.read_bytes() == new_proof.read_bytes(), "MIS certification evidence bytes changed")
        manifest["tasks"][task]["dataset"] = file_identity(new_data)
        manifest["tasks"][task]["reference_evidence"] = file_identity(new_proof)
        manifest["tasks"][task]["reference_quality_counts"] = dict(Counter(proof["reference_quality"] for proof in proofs))
    require(receipt["dataset"] == manifest["tasks"]["cvrp"]["dataset"], "result dataset mismatch")
    require(receipt["reference_evidence"] == manifest["tasks"]["cvrp"]["reference_evidence"], "result evidence mismatch")
    require(receipt["improved_instances"] == sum(value > 0 for value in improvements), "improvement count mismatch")
    require(abs(receipt["mean_relative_reference_improvement"] - sum(improvements) / 100) <= 1e-12,
            "mean reference improvement mismatch")
    history = list(manifest.get("prior_reference_refinements", []))
    if "reference_refinement" in manifest:
        history.append(manifest["reference_refinement"])
    manifest["prior_reference_refinements"] = history
    manifest["reference_refinement"] = {
        "parent_manifest_sha256": previous["manifest_sha256"], "plan_sha256": plan["plan_sha256"],
        "task": "cvrp", "plan": file_identity(plan_path), "result": file_identity(receipt_path),
        "changed_reference_output_indices": changed, "improved_instances": receipt["improved_instances"],
        "mean_relative_reference_improvement": sum(improvements) / 100,
        "semantic_instance_set_unchanged": True, "all_nonreference_fields_unchanged": True,
        "mis_dataset_and_certification_evidence_byte_identical_to_parent": True,
        "selection": plan["selection"], "model_outputs_examined": False,
        "finalization_implementation": file_identity(Path(__file__)),
        "independent_reference_validation_implementation": file_identity(REPO / "scripts/clasp_grpo_fresh_dev_artifacts.py"),
        "reference_semantics": "gap_to_verified_heuristic_reference_not_certified_optimum",
        "reference_evidence_schema_note": (
            "CVRP evidence retains top-level ORtools search metadata from the parent: "
            "integer_distance_scale, native_status_code, elapsed_seconds, "
            "search_objective_semantics, search_wall_limit_seconds, and ortools_candidate_raw_objective. "
            "These describe the parent search, not the selected LKH candidate. "
            "reference_selected_from identifies the selected source; lkh_candidates and "
            "the refinement plan describe the new LKH searches (distance scale 1000, precision 100). "
            "objective and reference_routes always describe the final selected raw Euclidean reference."
        ),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_new(output / "development_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    manifest = finalize(args.parent_root, args.output_root)
    print(json.dumps({"manifest_sha256": manifest["manifest_sha256"],
                      "reference_refinement": manifest["reference_refinement"]}, indent=2))


if __name__ == "__main__":
    main()
