from __future__ import annotations

from copy import deepcopy

import pytest

from clasp.data.benchmark import (
    forbidden_source_manifest_identity,
    forbidden_source_registry_identity,
    sha256_json,
    validate_frozen_benchmark_manifest,
)


def _seal(value: dict[str, object], field: str) -> dict[str, object]:
    sealed = deepcopy(value)
    sealed.pop(field, None)
    sealed[field] = sha256_json(sealed)
    return sealed


def _publication_op_manifest(*, exact: bool) -> dict[str, object]:
    """Build the smallest fully closed publication manifest for one OP row."""

    verifier = {"name": "fixture-strict-verifier", "version": 1}
    verifier_sha = sha256_json(verifier)
    generator = {"name": "fixture-generator", "schema_version": 1}
    public_status = "OPTIMAL_CERTIFIED" if exact else "BEST_KNOWN_HEURISTIC_VERIFIED"
    native_status = "OPTIMAL" if exact else "HEURISTIC_COMPLETE"
    quality = "certified_optimum" if exact else "best_known_heuristic"
    solver = _seal(
        {
            "schema_version": 2,
            "status": "approved_for_confirmation",
            "reference_evidence_runner": {
                "schema_version": 1,
                "protocol": "clasp-iid-reference-evidence-v1",
                "expected_instances_per_task": 1,
                "strict_verifier_identity_sha256": verifier_sha,
            },
            "tasks": {
                "op": {
                    "accepted_statuses": [public_status],
                    "parameters": {"seed": 0},
                    "evidence_runner": {
                        "approval": "approved",
                        "adapter": (
                            "gurobi_op_exact_v2" if exact else "fixture_op_heuristic_v1"
                        ),
                        "accepted_native_statuses": [native_status],
                        "objective_semantics": (
                            "clasp_raw_euclidean_float64_integer_prizes"
                        ),
                        "reference_quality": quality,
                        "seed_parameter": "seed",
                        "solver_identity": {
                            "backend_name": "fixture-op",
                            "native_version": "1",
                            "artifact_kind": "test_fixture",
                            "artifact_path": "/frozen/fixture-op",
                            "artifact_sha256": "a" * 64,
                        },
                    },
                }
            },
        },
        "contract_sha256",
    )
    solver_sha = sha256_json(solver)

    files = [
        {
            "relative_path": "op/eval/test.json",
            "sha256": "b" * 64,
            "size_bytes": 1,
            "rows": 1,
        }
    ]
    forbidden_sources: list[dict[str, object]] = []
    for label in sorted(("sft-train", "rl-train", "historical-eval")):
        source: dict[str, object] = {
            "task": "op",
            "label": label,
            "source_locator": f"forbidden/{label}.json",
            "files": files,
            "file_manifest_sha256": sha256_json(files),
            "rows": 1,
            "unique_instance_hashes": 1,
        }
        source["source_identity_sha256"] = forbidden_source_manifest_identity(source)[
            "source_identity_sha256"
        ]
        forbidden_sources.append(source)

    registry_sources = [deepcopy(source) for source in forbidden_sources]
    registry = _seal(
        {
            "schema_version": 1,
            "protocol": "clasp-forbidden-source-registry-v1",
            "status": "frozen_before_confirmation_generation",
            "created_at_utc": "2026-08-14T00:00:00Z",
            "tasks": ["op"],
            "required_labels": ["historical-eval", "rl-train", "sft-train"],
            "sources": registry_sources,
            "source_tree_manifest_sha256": sha256_json(
                [forbidden_source_manifest_identity(source) for source in registry_sources]
            ),
        },
        "registry_sha256",
    )
    registry_identity = forbidden_source_registry_identity(registry)

    semantic_hashes = ["c" * 64]
    row_hashes = ["d" * 64]
    reference_hashes = ["e" * 64]
    evidence_hashes = ["f" * 64]
    export_hashes = ["1" * 64]
    task_manifest: dict[str, object] = {
        "rows": 1,
        "source_locator": "op/eval/test.json",
        "files": files,
        "file_manifest_sha256": sha256_json(files),
        "ordered_instance_sha256": semantic_hashes,
        "ordered_semantic_instance_sha256": semantic_hashes,
        "instance_hashes_sha256": sha256_json(semantic_hashes),
        "ordered_benchmark_row_sha256": row_hashes,
        "row_hashes_sha256": sha256_json(row_hashes),
        "ordered_reference_output_sha256": reference_hashes,
        "reference_hashes_sha256": sha256_json(reference_hashes),
        "reference_audit": {
            "strictly_feasible": 1,
            "zero_gap": 1,
            "objective_min": 1,
            "objective_max": 1,
        },
        "reference_evidence_schema_version": 1,
        "reference_evidence_summary_sha256": "2" * 64,
        "ordered_reference_evidence_record_sha256": evidence_hashes,
        "reference_evidence_records_sha256": sha256_json(evidence_hashes),
        "ordered_reference_export_record_sha256": export_hashes,
        "reference_export_records_sha256": sha256_json(export_hashes),
    }
    receipt_task = {
        "rows": 1,
        "relative_path": "op/eval/test.json",
        "file_sha256": files[0]["sha256"],
        "ordered_instance_sha256": semantic_hashes,
        "ordered_instance_hashes_sha256": sha256_json(semantic_hashes),
        "ordered_semantic_instance_sha256": semantic_hashes,
        "semantic_instance_hashes_sha256": sha256_json(semantic_hashes),
        "ordered_benchmark_row_sha256": row_hashes,
        "benchmark_row_hashes_sha256": sha256_json(row_hashes),
        "ordered_reference_output_sha256": reference_hashes,
        "reference_output_hashes_sha256": sha256_json(reference_hashes),
        "strict_reference_objective_min": 1,
        "strict_reference_objective_max": 1,
        "reference_evidence_summary_sha256": task_manifest[
            "reference_evidence_summary_sha256"
        ],
        "ordered_reference_evidence_record_sha256": evidence_hashes,
        "reference_evidence_records_sha256": sha256_json(evidence_hashes),
        "ordered_reference_export_record_sha256": export_hashes,
        "reference_export_records_sha256": sha256_json(export_hashes),
    }
    receipt = _seal(
        {
            "schema_version": 2,
            "status": "references_attached",
            "created_at_utc": "2026-08-14T00:00:00Z",
            "generator_contract_sha256": sha256_json(generator),
            "generator_code_sha256": "3" * 64,
            "source_generation_receipt_sha256": "4" * 64,
            "reference_solver_contract_sha256": solver_sha,
            "strict_verifier": verifier,
            "forbidden_source_registry_identity": registry_identity,
            "tasks": {"op": receipt_task},
            "contains_references": True,
            "publishable": True,
            "reference_evidence_schema_version": 1,
            "reference_export_summary_sha256": "5" * 64,
        },
        "receipt_sha256",
    )
    return _seal(
        {
            "schema_version": 3,
            "status": "frozen_before_inference",
            "benchmark_mode": "publication",
            "minimum_instances_per_task": 1,
            "tasks": ["op"],
            "test_sets": {"op": task_manifest},
            "strict_verifier": verifier,
            "contracts": {
                "generator": generator,
                "generator_contract_sha256": sha256_json(generator),
                "reference_solver": solver,
                "reference_solver_contract_sha256": solver_sha,
            },
            "generation_receipt": receipt,
            "forbidden_sources": forbidden_sources,
            "forbidden_source_registry": registry,
            "forbidden_source_registry_identity": registry_identity,
            "overlap_audit": {
                "passed": True,
                "collisions": 0,
                "tested_forbidden_sources": len(forbidden_sources),
            },
        },
        "manifest_sha256",
    )


def test_publication_manifest_rejects_heuristic_op_reference_contract() -> None:
    with pytest.raises(ValueError, match="invalid accepted status for op"):
        validate_frozen_benchmark_manifest(_publication_op_manifest(exact=False))


def test_publication_manifest_accepts_certified_exact_op_reference_contract() -> None:
    identity = validate_frozen_benchmark_manifest(_publication_op_manifest(exact=True))

    assert identity["tasks"] == ["op"]
    assert identity["reference_evidence_schema_version"] == 1
