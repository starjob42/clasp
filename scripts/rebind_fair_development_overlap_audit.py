#!/usr/bin/env python3
"""Reuse a semantic overlap scan after a reference-only development revision.

This proves equality of every optimization instance and every non-reference
dataset field. It reuses the exact source byte identities certified by the
parent scan; it does not assert a new scan of possibly changed training files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from clasp.data.benchmark import semantic_instance_sha256, sha256_json
from build_fair_development_benchmark import file_identity, write_new


def verified_json(path: Path, field: str) -> tuple[dict, str]:
    value = json.loads(path.read_text())
    expected = value.pop(field)
    assert sha256_json(value) == expected, path
    return value, expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", required=True, type=Path)
    parser.add_argument("--child-root", required=True, type=Path)
    args = parser.parse_args()
    parent_root, child_root = args.parent_root.resolve(), args.child_root.resolve()
    parent, parent_hash = verified_json(parent_root / "development_manifest.json", "manifest_sha256")
    child, child_hash = verified_json(child_root / "development_manifest.json", "manifest_sha256")
    audit_path = parent_root / "training_overlap_audit.json"
    audit, audit_hash = verified_json(audit_path, "audit_sha256")
    assert audit["status"] == "passed_zero_semantic_overlap" and audit["total_overlaps"] == 0
    assert audit["development_manifest_sha256"] == parent_hash
    assert child["reference_refinement"]["parent_manifest_sha256"] == parent_hash
    source_pairs = {(s["task"], s["label"]) for s in audit["sources"]}
    assert source_pairs == {(task, label) for task in ("cvrp", "mis") for label in ("sft_train", "rl_train", "public_eval")}
    assert len(audit["sources"]) == 6
    assert all(not s["overlaps"] and s["rows"] > 0 for s in audit["sources"])
    comparisons = {}
    changes = []
    for task in ("cvrp", "mis"):
        paths = [base / task / "eval/test.json" for base in (parent_root, child_root)]
        for manifest, path in zip((parent, child), paths, strict=True):
            assert file_identity(path)["sha256"] == manifest["tasks"][task]["dataset"]["sha256"]
        old_rows, new_rows = [json.loads(path.read_text()) for path in paths]
        assert len(old_rows) == len(new_rows) == child["tasks"][task]["instances"]
        identities = []
        for index, (old, new) in enumerate(zip(old_rows, new_rows, strict=True)):
            old_id = semantic_instance_sha256(task, old, require_structured=True)
            new_id = semantic_instance_sha256(task, new, require_structured=True)
            assert old_id == new_id
            identities.append(new_id)
            assert {k: v for k, v in old.items() if k != "output"} == {k: v for k, v in new.items() if k != "output"}
            if old["output"] != new["output"]:
                changes.append({"task": task, "index": index, "semantic_instance_sha256": new_id})
        assert sorted(identities) == parent["tasks"][task]["semantic_instance_sha256"] == child["tasks"][task]["semantic_instance_sha256"]
        comparisons[task] = {"instances": len(identities), "ordered_semantic_sha256": sha256_json(identities),
                             "identical_semantics_and_all_nonreference_fields": True}
    result = {
        "purpose": "fresh_development_only", "status": "passed_zero_semantic_overlap",
        "development_manifest_sha256": child_hash, "total_overlaps": 0,
        "validation_method": "reuse_exact_parent_source_scan_after_proven_reference_only_revision",
        "parent_development_manifest_sha256": parent_hash, "parent_audit_sha256": audit_hash,
        "parent_audit_file": file_identity(audit_path), "source_spec": audit["source_spec"],
        "sources": audit["sources"], "semantic_equivalence": comparisons,
        "changed_reference_outputs": changes, "rebind_implementation": file_identity(Path(__file__)),
        "limitation": "Source identities are those scanned in the parent receipt; no new claim about subsequently modified training sources is made.",
    }
    result["audit_sha256"] = sha256_json(result)
    write_new(child_root / "training_overlap_audit.json", result)
    print(json.dumps({"status": result["status"], "child_manifest_sha256": child_hash,
                      "audit_sha256": result["audit_sha256"], "changed_reference_outputs": changes}, indent=2))


if __name__ == "__main__":
    main()
