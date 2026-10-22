from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "aggregate_sota", ROOT / "scripts" / "aggregate_sota.py"
)
assert SPEC is not None and SPEC.loader is not None
aggregate_sota = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = aggregate_sota
SPEC.loader.exec_module(aggregate_sota)


def _digest(value: object) -> str:
    return hashlib.sha256(aggregate_sota._canonical_json_bytes(value)).hexdigest()


def _attach_forbidden_source_registry(manifest: dict) -> None:
    manifest["forbidden_sources"].sort(key=lambda row: (row["task"], row["label"]))
    sources = []
    for source in manifest["forbidden_sources"]:
        projection = {
            key: deepcopy(source[key])
            for key in (
                "task",
                "label",
                "files",
                "file_manifest_sha256",
                "rows",
                "unique_instance_hashes",
            )
        }
        projection["source_identity_sha256"] = _digest(projection)
        source["source_identity_sha256"] = projection["source_identity_sha256"]
        sources.append(projection)
    sources.sort(key=lambda row: (row["task"], row["label"]))
    registry = {
        "schema_version": 1,
        "protocol": "clasp-forbidden-source-registry-v1",
        "status": "frozen_before_confirmation_generation",
        "created_at_utc": "2026-08-14T00:00:00Z",
        "tasks": sorted(manifest["tasks"]),
        "required_labels": ["historical-eval", "rl-train", "sft-train"],
        "sources": sources,
        "source_tree_manifest_sha256": _digest(sources),
    }
    registry["registry_sha256"] = _digest(registry)
    manifest["benchmark_mode"] = "publication"
    manifest["forbidden_source_registry"] = registry
    manifest["forbidden_source_registry_identity"] = (
        aggregate_sota.forbidden_source_registry_identity(registry)
    )
    prior_receipt = manifest["generation_receipt"]
    receipt_tasks = {}
    for task, task_manifest in manifest["test_sets"].items():
        receipt_tasks[task] = {
            "rows": task_manifest["rows"],
            "relative_path": task_manifest["files"][0]["relative_path"],
            "file_sha256": task_manifest["files"][0]["sha256"],
            "ordered_instance_sha256": task_manifest["ordered_instance_sha256"],
            "ordered_instance_hashes_sha256": _digest(task_manifest["ordered_instance_sha256"]),
            "ordered_semantic_instance_sha256": task_manifest["ordered_semantic_instance_sha256"],
            "semantic_instance_hashes_sha256": _digest(
                task_manifest["ordered_semantic_instance_sha256"]
            ),
            "ordered_benchmark_row_sha256": task_manifest["ordered_benchmark_row_sha256"],
            "benchmark_row_hashes_sha256": _digest(task_manifest["ordered_benchmark_row_sha256"]),
            "ordered_reference_output_sha256": task_manifest["ordered_reference_output_sha256"],
            "reference_output_hashes_sha256": _digest(
                task_manifest["ordered_reference_output_sha256"]
            ),
            "strict_reference_objective_min": task_manifest["reference_audit"]["objective_min"],
            "strict_reference_objective_max": task_manifest["reference_audit"]["objective_max"],
            "reference_evidence_summary_sha256": task_manifest["reference_evidence_summary_sha256"],
            "ordered_reference_evidence_record_sha256": task_manifest[
                "ordered_reference_evidence_record_sha256"
            ],
            "reference_evidence_records_sha256": task_manifest["reference_evidence_records_sha256"],
            "ordered_reference_export_record_sha256": task_manifest[
                "ordered_reference_export_record_sha256"
            ],
            "reference_export_records_sha256": task_manifest["reference_export_records_sha256"],
        }
    receipt = {
        "schema_version": 2,
        "status": "references_attached",
        "created_at_utc": "2026-08-14T00:00:00Z",
        "generator_contract_sha256": manifest["contracts"]["generator_contract_sha256"],
        "generator_code_sha256": prior_receipt["generator_code_sha256"],
        "source_generation_receipt_sha256": "6" * 64,
        "forbidden_source_registry_identity": deepcopy(
            manifest["forbidden_source_registry_identity"]
        ),
        "reference_solver_contract_sha256": manifest["contracts"][
            "reference_solver_contract_sha256"
        ],
        "strict_verifier": deepcopy(manifest["strict_verifier"]),
        "tasks": receipt_tasks,
        "contains_references": True,
        "publishable": True,
        "reference_evidence_schema_version": 1,
        "reference_export_summary_sha256": prior_receipt["reference_export_summary_sha256"],
    }
    receipt["receipt_sha256"] = _digest(receipt)
    manifest["generation_receipt"] = receipt


def _record(index: int, feasible: bool, gap: float | None) -> dict:
    return {
        "task": "tsp",
        "split": "eval",
        "example_index": index,
        "num_return_sequences": 8,
        "best_of_n_feasible": feasible,
        "best_feasible_gap": gap,
        "candidates": [{"is_feasible": feasible and candidate == 0} for candidate in range(8)],
    }


def test_summary_keeps_conditional_gap_denominator_explicit() -> None:
    summary = aggregate_sota.summarize_records(
        [_record(0, True, 0.01), _record(1, False, None)], target_gap=0.02
    )
    assert summary["pass_at_k"] == 0.5
    assert summary["conditional_gap_mean"] == 0.01
    assert summary["conditional_gap_n"] == 1
    assert summary["success_at_target_gap"] == 0.5
    assert summary["candidate_feasibility"] == pytest.approx(1 / 16)


def test_normalized_feasible_gap_auc_has_frozen_boundaries_and_infeasible_zero() -> None:
    records = [
        _record(0, False, None),
        _record(1, True, 0.0),
        _record(2, True, 0.05),
        _record(3, True, 0.10),
        _record(4, True, 0.20),
    ]
    assert [aggregate_sota.normalized_feasible_gap_auc_score(row) for row in records] == [
        0.0,
        1.0,
        0.5,
        0.0,
        0.0,
    ]
    summary = aggregate_sota.summarize_records(records)
    assert summary["normalized_feasible_gap_auc"] == pytest.approx(0.3)
    assert summary["normalized_feasible_gap_auc_range"] == [0.0, 0.10]


def test_load_run_fails_on_duplicate_instance(tmp_path: Path) -> None:
    details = tmp_path / "run_details__x.jsonl"
    record = _record(0, True, 0.01)
    details.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    spec = aggregate_sota.RunSpec("method", 42, str(details))
    with pytest.raises(ValueError, match="duplicate detail key"):
        aggregate_sota.load_run(spec, expected_k=8, allow_legacy_provenance=True)


def _fingerprinted_record(index: int = 0) -> dict:
    record = _record(index, True, 0.01)
    for candidate in record["candidates"]:
        candidate["completion_tokens"] = 3
    instance = {"nodes": [[0, 0], [1, 1]]}
    instance_hash = hashlib.sha256(aggregate_sota._canonical_json_bytes(instance)).hexdigest()
    record.update(
        {
            "instance_json": instance,
            "instance_sha256": instance_hash,
            "semantic_instance_sha256": "e" * 64,
            "benchmark_row_sha256": "f" * 64,
            "reference_output_sha256": "1" * 64,
            "generated_tokens": 24,
            "generation_time_sec": 1.0,
            "peak_gpu_memory_allocated_bytes": 100,
            "peak_gpu_memory_reserved_bytes": 120,
            "experiment_fingerprint": "a" * 64,
            "protocol_fingerprint": "b" * 64,
            "strict_verifier": {
                "name": "clasp_strict_shared_verifier",
                "schema_version": 1,
                "implementation_manifest_sha256": "c" * 64,
            },
            "source_row_locator": {
                "data_manifest_sha256": "d" * 64,
                "task": "tsp",
                "split": "eval",
                "merged_dataset_index": index,
            },
        }
    )
    return record


def _write_complete_native_run(tmp_path: Path, records: list[dict]) -> Path:
    details = tmp_path / "run_details__paper.jsonl"
    details.write_text("".join(json.dumps(record) + "\n" for record in records))
    config = {
        "run_status": "complete",
        "experiment_fingerprint": "a" * 64,
        "protocol_fingerprint": "b" * 64,
        "verifier_identity": records[0]["strict_verifier"],
    }
    (tmp_path / "run_config__paper.json").write_text(json.dumps(config))
    return details


def test_load_run_requires_complete_matching_provenance(tmp_path: Path) -> None:
    details = _write_complete_native_run(tmp_path, [_fingerprinted_record()])
    spec = aggregate_sota.RunSpec("method", 42, str(details))
    loaded = aggregate_sota.load_run(spec, expected_k=8)
    assert list(loaded) == [("tsp", "eval", 0)]

    config_path = tmp_path / "run_config__paper.json"
    config = json.loads(config_path.read_text())
    config["run_status"] = "in_progress"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="not complete"):
        aggregate_sota.load_run(spec, expected_k=8)


def test_load_run_rejects_corrupt_embedded_instance(tmp_path: Path) -> None:
    record = _fingerprinted_record()
    record["instance_json"] = {"nodes": []}
    details = _write_complete_native_run(tmp_path, [record])
    with pytest.raises(ValueError, match="instance_json hash mismatch"):
        aggregate_sota.load_run(aggregate_sota.RunSpec("method", 42, str(details)), expected_k=8)


def test_paired_provenance_requires_same_protocol_and_instance() -> None:
    baseline_record = _fingerprinted_record()
    challenger_record = _fingerprinted_record()
    key = (42, ("tsp", "eval", 0))
    challenger_record["protocol_fingerprint"] = "e" * 64
    with pytest.raises(ValueError, match="paired protocol mismatch"):
        aggregate_sota._validate_paired_provenance({key: baseline_record}, {key: challenger_record})


def test_frozen_benchmark_requires_every_ordered_instance() -> None:
    record = _fingerprinted_record()
    run = {("method", 42): {("tsp", "eval", 0): record}}
    semantic = [record["semantic_instance_sha256"]]
    rows = [record["benchmark_row_sha256"]]
    references = [record["reference_output_sha256"]]
    manifest = {
        "strict_verifier": record["strict_verifier"],
        "test_sets": {
            "tsp": {
                "ordered_instance_sha256": list(semantic),
                "ordered_semantic_instance_sha256": list(semantic),
                "ordered_benchmark_row_sha256": list(rows),
                "ordered_reference_output_sha256": list(references),
                "instance_hashes_sha256": hashlib.sha256(
                    aggregate_sota._canonical_json_bytes(semantic)
                ).hexdigest(),
                "row_hashes_sha256": hashlib.sha256(
                    aggregate_sota._canonical_json_bytes(rows)
                ).hexdigest(),
                "reference_hashes_sha256": hashlib.sha256(
                    aggregate_sota._canonical_json_bytes(references)
                ).hexdigest(),
            }
        },
    }
    aggregate_sota._validate_frozen_benchmark_records(run, manifest)
    assert record["instance_sha256"] != record["semantic_instance_sha256"]

    longer = deepcopy(manifest)
    task_manifest = longer["test_sets"]["tsp"]
    for field, digest in (
        ("ordered_instance_sha256", "2" * 64),
        ("ordered_semantic_instance_sha256", "2" * 64),
        ("ordered_benchmark_row_sha256", "3" * 64),
        ("ordered_reference_output_sha256", "4" * 64),
    ):
        task_manifest[field].append(digest)
    for ordered_field, aggregate_field in (
        ("ordered_semantic_instance_sha256", "instance_hashes_sha256"),
        ("ordered_benchmark_row_sha256", "row_hashes_sha256"),
        ("ordered_reference_output_sha256", "reference_hashes_sha256"),
    ):
        task_manifest[aggregate_field] = hashlib.sha256(
            aggregate_sota._canonical_json_bytes(task_manifest[ordered_field])
        ).hexdigest()
    with pytest.raises(ValueError, match="incomplete or reordered"):
        aggregate_sota._validate_frozen_benchmark_records(run, longer)

    for field, message in (
        ("semantic_instance_sha256", "semantic instance"),
        ("benchmark_row_sha256", "source row"),
        ("reference_output_sha256", "reference output"),
    ):
        tampered_run = deepcopy(run)
        tampered_run[("method", 42)][("tsp", "eval", 0)][field] = "0" * 64
        with pytest.raises(ValueError, match=message):
            aggregate_sota._validate_frozen_benchmark_records(tampered_run, manifest)

    wrong_verifier_run = deepcopy(run)
    wrong_verifier_run[("method", 42)][("tsp", "eval", 0)]["strict_verifier"] = {"name": "other"}
    with pytest.raises(ValueError, match="record verifier"):
        aggregate_sota._validate_frozen_benchmark_records(wrong_verifier_run, manifest)


def test_frozen_benchmark_load_binds_solver_and_verifier(tmp_path: Path) -> None:
    verifier = {"name": "strict-verifier", "implementation": "a" * 64}
    verifier_sha256 = hashlib.sha256(aggregate_sota._canonical_json_bytes(verifier)).hexdigest()
    solver = {
        "schema_version": 2,
        "status": "approved_for_confirmation",
        "reference_evidence_runner": {
            "schema_version": 1,
            "protocol": "clasp-iid-reference-evidence-v1",
            "expected_instances_per_task": 1,
            "strict_verifier_identity_sha256": verifier_sha256,
        },
        "tasks": {
            "tsp": {
                "accepted_statuses": ["BEST_KNOWN_HEURISTIC_VERIFIED"],
                "parameters": {"seed": 1},
                "evidence_runner": {
                    "approval": "approved",
                    "adapter": "test_tsp_v1",
                    "accepted_native_statuses": ["HEURISTIC_COMPLETE"],
                    "objective_semantics": "clasp_raw_euclidean_float64",
                    "reference_quality": "best_known_heuristic",
                    "seed_parameter": "seed",
                    "solver_identity": {
                        "backend_name": "test-tsp",
                        "native_version": "1",
                        "artifact_kind": "test_fixture",
                        "artifact_path": "/frozen/test-tsp.artifact",
                        "artifact_sha256": "a" * 64,
                    },
                },
            }
        },
    }
    solver["contract_sha256"] = hashlib.sha256(
        aggregate_sota._canonical_json_bytes(solver)
    ).hexdigest()
    solver_sha256 = hashlib.sha256(aggregate_sota._canonical_json_bytes(solver)).hexdigest()
    generator = {"name": "frozen-generator", "version": 1}
    generator_sha256 = hashlib.sha256(aggregate_sota._canonical_json_bytes(generator)).hexdigest()
    semantic = ["1" * 64]
    rows = ["2" * 64]
    references = ["3" * 64]
    evidence = ["4" * 64]
    exports = ["5" * 64]
    files = [{"relative_path": "test.json", "sha256": "6" * 64, "size_bytes": 1, "rows": 1}]
    forbidden_sources = [
        {
            "task": "tsp",
            "label": label,
            "source_locator": f"forbidden/{label}.json",
            "files": files,
            "file_manifest_sha256": hashlib.sha256(
                aggregate_sota._canonical_json_bytes(files)
            ).hexdigest(),
            "rows": 1,
            "unique_instance_hashes": 1,
        }
        for label in ("sft-train", "rl-train", "historical-eval")
    ]
    manifest = {
        "schema_version": 3,
        "status": "frozen_before_inference",
        "minimum_instances_per_task": 1,
        "tasks": ["tsp"],
        "overlap_audit": {
            "passed": True,
            "collisions": 0,
            "tested_forbidden_sources": len(forbidden_sources),
        },
        "forbidden_sources": forbidden_sources,
        "contracts": {
            "generator": generator,
            "generator_contract_sha256": generator_sha256,
            "reference_solver": solver,
            "reference_solver_contract_sha256": solver_sha256,
        },
        "generation_receipt": {
            "status": "references_attached",
            "receipt_sha256": "b" * 64,
            "generator_code_sha256": "9" * 64,
            "reference_solver_contract_sha256": solver_sha256,
            "strict_verifier": verifier,
            "reference_evidence_schema_version": 1,
            "reference_export_summary_sha256": "7" * 64,
        },
        "strict_verifier": verifier,
        "test_sets": {
            "tsp": {
                "rows": 1,
                "source_locator": "source/tsp/test.json",
                "files": files,
                "file_manifest_sha256": hashlib.sha256(
                    aggregate_sota._canonical_json_bytes(files)
                ).hexdigest(),
                "ordered_instance_sha256": semantic,
                "ordered_semantic_instance_sha256": semantic,
                "instance_hashes_sha256": hashlib.sha256(
                    aggregate_sota._canonical_json_bytes(semantic)
                ).hexdigest(),
                "ordered_benchmark_row_sha256": rows,
                "row_hashes_sha256": hashlib.sha256(
                    aggregate_sota._canonical_json_bytes(rows)
                ).hexdigest(),
                "ordered_reference_output_sha256": references,
                "reference_hashes_sha256": hashlib.sha256(
                    aggregate_sota._canonical_json_bytes(references)
                ).hexdigest(),
                "reference_audit": {
                    "strictly_feasible": 1,
                    "zero_gap": 1,
                    "objective_min": 1.0,
                    "objective_max": 1.0,
                },
                "reference_evidence_schema_version": 1,
                "reference_evidence_summary_sha256": "8" * 64,
                "ordered_reference_evidence_record_sha256": evidence,
                "reference_evidence_records_sha256": hashlib.sha256(
                    aggregate_sota._canonical_json_bytes(evidence)
                ).hexdigest(),
                "ordered_reference_export_record_sha256": exports,
                "reference_export_records_sha256": hashlib.sha256(
                    aggregate_sota._canonical_json_bytes(exports)
                ).hexdigest(),
            }
        },
    }
    _attach_forbidden_source_registry(manifest)

    def write_manifest(payload: dict) -> Path:
        unsigned = deepcopy(payload)
        unsigned.pop("manifest_sha256", None)
        payload["manifest_sha256"] = hashlib.sha256(
            aggregate_sota._canonical_json_bytes(unsigned)
        ).hexdigest()
        path = tmp_path / "benchmark.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    path = write_manifest(manifest)
    aggregate_sota._load_frozen_benchmark(str(path))

    manifest["generation_receipt"]["reference_solver_contract_sha256"] = "0" * 64
    unsigned_receipt = deepcopy(manifest["generation_receipt"])
    unsigned_receipt.pop("receipt_sha256", None)
    manifest["generation_receipt"]["receipt_sha256"] = _digest(unsigned_receipt)
    path = write_manifest(manifest)
    with pytest.raises(ValueError, match="another solver contract"):
        aggregate_sota._load_frozen_benchmark(str(path))


def test_paired_comparison_rewards_feasibility_before_gap() -> None:
    key0 = (42, ("tsp", "eval", 0))
    key1 = (42, ("tsp", "eval", 1))
    baseline = {
        key0: _record(0, False, None),
        key1: _record(1, True, 0.02),
    }
    challenger = {
        key0: _record(0, True, 0.03),
        key1: _record(1, True, 0.01),
    }
    result = aggregate_sota.compare_paired(
        baseline,
        challenger,
        task="tsp",
        target_gap=0.02,
        repetitions=100,
        rng=aggregate_sota.random.Random(7),
    )
    assert result["challenger_minus_baseline_pass_at_k"] == 0.5
    assert result["pooled_for_descriptive_use_only"]["lexicographic"] == {
        "challenger_wins": 2,
        "ties": 0,
        "baseline_wins": 0,
    }
    assert "exact_two_sided_p" not in result["pooled_for_descriptive_use_only"]
    seed_test = result["per_training_seed"]["42"]["lexicographic_exact_test"]
    assert seed_test["challenger_wins"] == 2
    assert seed_test["exact_two_sided_p"] == pytest.approx(0.5)


def test_paired_comparison_reports_sample_std_across_training_seeds() -> None:
    baseline = {
        (42, ("tsp", "eval", 0)): _record(0, True, 0.03),
        (43, ("tsp", "eval", 0)): _record(0, True, 0.02),
    }
    challenger = {
        (42, ("tsp", "eval", 0)): _record(0, True, 0.01),
        (43, ("tsp", "eval", 0)): _record(0, True, 0.03),
    }
    result = aggregate_sota.compare_paired(
        baseline,
        challenger,
        task="tsp",
        target_gap=0.02,
        repetitions=25,
        rng=aggregate_sota.random.Random(11),
    )
    assert result["challenger_minus_baseline_pass_at_k_seed_sample_std"] == 0.0
    assert result[
        "challenger_minus_baseline_jointly_feasible_gap_seed_sample_std"
    ] == pytest.approx(0.021213203435596427)
    assert result[
        "challenger_minus_baseline_success_at_target_gap_seed_sample_std"
    ] == pytest.approx(2**0.5)


def test_confirmation_cli_requires_a_distinct_matched_comparison(tmp_path: Path) -> None:
    common = [
        "--run",
        "clasp_dense,42,tsp,/missing/details.jsonl",
        "--targets",
        str(ROOT / "experiments" / "sota_20260814" / "targets.json"),
        "--benchmark-manifest",
        "/missing/benchmark.json",
        "--campaign-manifest",
        "/missing/campaign.json",
        "--output",
        str(tmp_path / "aggregate.json"),
    ]
    with pytest.raises(SystemExit):
        aggregate_sota.main(common)
    with pytest.raises(SystemExit):
        aggregate_sota.main([*common, "--baseline", "clasp_dense", "--challenger", "clasp_dense"])


def test_development_aggregate_cannot_emit_external_strict_beat_claim(
    tmp_path: Path,
) -> None:
    details = tmp_path / "legacy.jsonl"
    details.write_text(
        "".join(
            json.dumps(record) + "\n" for record in (_record(0, True, 0.0), _record(1, True, 0.0))
        ),
        encoding="utf-8",
    )
    output = tmp_path / "aggregate.json"
    assert (
        aggregate_sota.main(
            [
                "--run",
                f"clasp_dense,42,{details}",
                "--targets",
                str(ROOT / "experiments" / "sota_20260814" / "targets.json"),
                "--lane",
                "raw_n8",
                "--development-only",
                "--allow-legacy-provenance",
                "--bootstrap-repetitions",
                "20",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["claim_status"] == "NO_SOTA_CLAIM"
    projection = payload["external_context_registry_projection"]
    assert projection["context_id"] == "llmcosolver_neurips2025_table1_n8"
    assert projection["comparison_scope"] == ("historical_first_100_author_protocol_only")
    assert projection["aggregate_point_comparison_eligible"] is False
    assert projection["strict_beat_claim_eligible"] is False
    external = payload["summaries"]["clasp_dense"]["tsp"]["external_context"]
    assert external["comparison_status"] == "CONTEXT_ONLY_NOT_COMPARABLE"
    assert external["external_point_comparison_eligible"] is False
    assert external["strict_beat_claim_eligible"] is False
    assert "descriptively_exceeds_frozen_reference_point" not in json.dumps(payload)
    assert (
        payload["summaries"]["clasp_dense"]["tsp"]["per_seed"]["42"]["success_at_target_gap"]
        is None
    )


def test_exact_binomial_and_holm_are_bounded() -> None:
    assert aggregate_sota.exact_two_sided_binomial(0, 0) == 1.0
    assert aggregate_sota.exact_two_sided_binomial(0, 6) == pytest.approx(0.03125)
    adjusted = aggregate_sota.holm_adjust({"a": 0.01, "b": 0.03, "c": 0.9})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.9}
