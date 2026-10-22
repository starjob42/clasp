from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.compare_development import (
    BOOTSTRAP_REPLICATES,
    DevelopmentComparisonError,
    _canonical_json_bytes,
    _load_artifact,
    _protocol_fingerprint_material,
    build_receipt,
    run_comparison,
)

from clasp.data.benchmark import (
    benchmark_row_sha256,
    reference_output_sha256,
    semantic_instance_sha256,
)

VERIFIER = {
    "name": "clasp_strict_shared_verifier",
    "implementation_manifest_sha256": "d" * 64,
}
DATA_FILES: list[dict] = []
DATA_MANIFEST = hashlib.sha256(_canonical_json_bytes(DATA_FILES)).hexdigest()
GLOBAL_SEED = 271828


def _seal(payload: dict, field: str) -> dict:
    sealed = copy.deepcopy(payload)
    sealed.pop(field, None)
    sealed[field] = hashlib.sha256(_canonical_json_bytes(sealed)).hexdigest()
    return sealed


def _per_example_seed(example_index: int) -> int:
    material = {
        "schema_version": 1,
        "global_seed": GLOBAL_SEED,
        "task": "tsp",
        "split": "eval",
        "example_index": example_index,
    }
    value = int.from_bytes(hashlib.sha256(_canonical_json_bytes(material)).digest()[:8], "big")
    return value % (2**63 - 1)


def _provenance(model_tag: str, *, k: int = 2, top_p: float = 0.9) -> dict:
    return {
        "schema_version": 1,
        "model": {"tag": model_tag},
        "data": {"manifest_sha256": DATA_MANIFEST, "files": DATA_FILES},
        "prompting": {"style": "synthetic-test"},
        "generation": {
            "seed": GLOBAL_SEED,
            "seed_policy": "per_example_sha256_v1",
            "num_return_sequences": k,
            "temperature": 0.7,
            "top_p": top_p,
        },
        "selection": {
            "evaluation_purpose": "development",
            "benchmark_manifest": None,
            "campaign_manifest": None,
            "tasks": ["tsp"],
            "split": "eval",
        },
        "verifier": VERIFIER,
        "code": {"manifest_sha256": "c" * 64, "files": []},
    }


def _fingerprints(provenance: dict) -> tuple[str, str]:
    experiment = hashlib.sha256(_canonical_json_bytes(provenance)).hexdigest()
    protocol = hashlib.sha256(
        _canonical_json_bytes(_protocol_fingerprint_material(provenance))
    ).hexdigest()
    return experiment, protocol


def _metric_fields(gaps: list[float | None]) -> dict:
    candidates = [
        {
            "candidate_index": index,
            "is_feasible": gap is not None,
            "gap_to_ground_truth": gap,
        }
        for index, gap in enumerate(gaps)
    ]
    feasible = [(index, float(gap)) for index, gap in enumerate(gaps) if gap is not None]
    best = min(feasible, key=lambda item: (item[1], item[0])) if feasible else None
    return {
        "num_return_sequences": len(candidates),
        "feasible_candidates": len(feasible),
        "candidate_feasibility_rate": len(feasible) / len(candidates),
        "best_candidate_index": best[0] if best else None,
        "best_of_n_feasible": best is not None,
        "best_feasible_gap": best[1] if best else None,
        "candidates": candidates,
    }


def _raw_record(
    example_index: int,
    gaps: list[float | None],
    provenance: dict,
) -> dict:
    experiment, protocol = _fingerprints(provenance)
    instance = [
        [0.0, 0.0],
        [1.0 + example_index / 10.0, 0.0],
        [0.0, 1.0 + example_index / 10.0],
    ]
    row = {
        "instruction": "Solve this synthetic TSP.",
        "input": f"Synthetic instance {example_index}.",
        "output": "Route: [0, 1, 2, 0]\nObjective: 3.414",
        "instance_json": instance,
    }
    record = {
        "task": "tsp",
        "split": "eval",
        "example_index": example_index,
        "instruction": row["instruction"],
        "input_problem": row["input"],
        "ground_truth_output": row["output"],
        "instance_json": instance,
        "instance_sha256": hashlib.sha256(_canonical_json_bytes(instance)).hexdigest(),
        "semantic_instance_sha256": semantic_instance_sha256("tsp", row, require_structured=True),
        "benchmark_row_sha256": benchmark_row_sha256(row),
        "reference_output_sha256": reference_output_sha256(row["output"]),
        "transformed_example_sha256": hashlib.sha256(_canonical_json_bytes(row)).hexdigest(),
        "source_row_locator": {
            "data_manifest_sha256": DATA_MANIFEST,
            "task": "tsp",
            "split": "eval",
            "merged_dataset_index": example_index,
        },
        "sampling_seed": _per_example_seed(example_index),
        "experiment_fingerprint": experiment,
        "protocol_fingerprint": protocol,
        "strict_verifier": VERIFIER,
        **_metric_fields(gaps),
    }
    return _seal(record, "record_sha256")


def _write_native(
    root: Path,
    name: str,
    gaps_by_row: list[list[float | None]],
    *,
    top_p: float = 0.9,
) -> tuple[Path, Path, list[dict], dict]:
    root.mkdir(parents=True, exist_ok=True)
    k = len(gaps_by_row[0])
    provenance = _provenance(name, k=k, top_p=top_p)
    experiment, protocol = _fingerprints(provenance)
    records = [_raw_record(index, gaps, provenance) for index, gaps in enumerate(gaps_by_row)]
    details = root / f"run_details__{name}.jsonl"
    details.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    raw = details.read_bytes()
    config = _seal(
        {
            "run_status": "complete",
            "evaluation_purpose": "development",
            "experiment_fingerprint": experiment,
            "protocol_fingerprint": protocol,
            "experiment_provenance": provenance,
            "verifier_identity": VERIFIER,
            "tasks": ["tsp"],
            "split": "eval",
            "generation": {**provenance["generation"], "best_of_n": k},
            "details_artifact": {
                "basename": details.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "record_count": len(records),
            },
        },
        "run_config_sha256",
    )
    companion = root / f"run_config__{name}.json"
    companion.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return details, companion, records, provenance


def _write_repair(
    root: Path,
    name: str,
    raw_records: list[dict],
    provenance: dict,
    repaired_gaps_by_row: list[list[float | None]],
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    source_experiment, protocol = _fingerprints(provenance)
    repair_fingerprint = "e" * 64
    repaired_records = []
    for raw_record, repaired_gaps in zip(raw_records, repaired_gaps_by_row, strict=True):
        repaired_view = {
            key: copy.deepcopy(value)
            for key, value in raw_record.items()
            if key
            not in {
                "record_sha256",
                "experiment_fingerprint",
                "protocol_fingerprint",
                "strict_verifier",
                "candidates",
            }
        }
        repaired_view.update(_metric_fields(repaired_gaps))
        wrapped = [
            {**candidate, "repaired_metrics": copy.deepcopy(candidate)}
            for candidate in repaired_view["candidates"]
        ]
        repaired = {
            **copy.deepcopy(repaired_view),
            "repair_schema_version": 1,
            "repair_algorithm_version": "clasp_deterministic_repair_v1",
            "repair_fingerprint": repair_fingerprint,
            "protocol_fingerprint": protocol,
            "strict_verifier": VERIFIER,
            "source_experiment_fingerprint": source_experiment,
            "raw_record": copy.deepcopy(raw_record),
            "raw_record_sha256": hashlib.sha256(_canonical_json_bytes(raw_record)).hexdigest(),
            "raw_strict_view": {
                key: copy.deepcopy(value)
                for key, value in raw_record.items()
                if key
                not in {
                    "record_sha256",
                    "experiment_fingerprint",
                    "protocol_fingerprint",
                    "strict_verifier",
                }
            },
            "repaired_strict_view": copy.deepcopy(repaired_view),
            "candidates": wrapped,
        }
        repaired_records.append(_seal(repaired, "repair_record_sha256"))
    details = root / f"{name}.jsonl"
    details.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in repaired_records),
        encoding="utf-8",
    )
    raw = details.read_bytes()
    summary = _seal(
        {
            "system_lane": "deterministic_repair_and_improve_hybrid",
            "repair_fingerprint": repair_fingerprint,
            "source_protocol_fingerprint": protocol,
            "source_experiment_provenance": provenance,
            "source_campaign_manifest": None,
            "source_campaign_run_binding": None,
            "source_confirmation_attempt_claim": None,
            "record_count": len(repaired_records),
            "candidate_count": sum(len(record["candidates"]) for record in repaired_records),
            "details_artifact": {
                "basename": details.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "record_count": len(repaired_records),
            },
        },
        "summary_sha256",
    )
    companion = root / f"{name}__summary.json"
    companion.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    return details, companion


def _args(
    tmp_path: Path,
    left: tuple[Path, Path],
    right: tuple[Path, Path],
) -> SimpleNamespace:
    return SimpleNamespace(
        left=str(left[0]),
        left_companion=str(left[1]),
        right=str(right[0]),
        right_companion=str(right[1]),
        left_label="baseline",
        right_label="challenger",
        output=str(tmp_path / "receipt.json"),
    )


def test_native_comparison_is_exact_deterministic_and_self_hashed(tmp_path: Path):
    left = _write_native(tmp_path / "left", "left", [[0.2, None], [None, None], [0.05, 0.1]])
    right = _write_native(tmp_path / "right", "right", [[0.1, 0.12], [0.3, None], [None, None]])
    args = _args(tmp_path, left[:2], right[:2])
    output = Path(run_comparison(args))
    receipt = json.loads(output.read_text(encoding="utf-8"))

    assert receipt["evidence_scope"] == "development_only"
    assert receipt["claim_eligibility"] == "not_confirmation_and_not_sota_evidence"
    assert receipt["pairing"]["key_fields"] == [
        "task",
        "semantic_instance_sha256",
        "per_example_seed",
    ]
    task = receipt["per_task"]["tsp"]
    assert task["pass_at_k"]["left"] == pytest.approx(2 / 3)
    assert task["pass_at_k"]["right"] == pytest.approx(2 / 3)
    assert task["pass_at_k"]["exact_mcnemar"]["exact_two_sided_p"] == 1.0
    assert task["candidate_feasibility"]["right_minus_left"] == 0.0
    gap = task["conditional_best_feasible_gap"]
    assert gap["jointly_feasible_instances"] == 1
    assert gap["right_minus_left_on_jointly_feasible_instances"] == pytest.approx(-0.1)
    assert gap["paired_bootstrap_95pct_ci"]["replicates"] == BOOTSTRAP_REPLICATES
    assert gap["paired_bootstrap_95pct_ci"]["lower"] == pytest.approx(-0.1)
    unsigned = dict(receipt)
    assert (
        unsigned.pop("receipt_sha256")
        == hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    )

    loaded_left = _load_artifact(str(left[0]), str(left[1]), label="left")
    loaded_right = _load_artifact(str(right[0]), str(right[1]), label="right")
    first = build_receipt(
        loaded_left,
        loaded_right,
        left_label="baseline",
        right_label="challenger",
        generated_at_utc="2026-08-14T00:00:00Z",
    )
    second = build_receipt(
        loaded_left,
        loaded_right,
        left_label="baseline",
        right_label="challenger",
        generated_at_utc="2026-08-14T00:00:00Z",
    )
    assert first == second


def test_native_to_repair_requires_and_records_exact_raw_linkage(tmp_path: Path):
    native = _write_native(tmp_path / "native", "native", [[0.2, None], [None, None]])
    repair = _write_repair(
        tmp_path / "repair",
        "repair",
        native[2],
        native[3],
        [[0.1, 0.2], [0.3, 0.4]],
    )
    args = _args(tmp_path, native[:2], repair)
    receipt = json.loads(Path(run_comparison(args)).read_text(encoding="utf-8"))
    assert receipt["artifacts"]["right"]["artifact_type"] == "repair"
    assert (
        receipt["pairing"]["native_repair_verbatim_linkage"]
        == "verified_exact_embedded_native_records"
    )
    assert receipt["per_task"]["tsp"]["pass_at_k"]["right_minus_left"] == 0.5


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("record_hash", "record_sha256 mismatch"),
        ("sampling_seed", "sampling_seed inconsistent"),
        ("source_row", "source_row_locator mismatch"),
        ("protocol", "different protocol fingerprints"),
        ("candidate_budget", "different candidate budgets"),
    ],
)
def test_comparison_fails_closed_on_integrity_or_pairing_mismatch(
    tmp_path: Path,
    mutation: str,
    message: str,
):
    left = _write_native(tmp_path / "left", "left", [[0.2, None], [None, None]])
    if mutation == "protocol":
        right = _write_native(tmp_path / "right", "right", [[0.1, None], [0.3, None]], top_p=0.8)
    elif mutation == "candidate_budget":
        right = _write_native(tmp_path / "right", "right", [[0.1, None, None], [0.3, None, None]])
    else:
        right = _write_native(tmp_path / "right", "right", [[0.1, None], [0.3, None]])
        records = copy.deepcopy(right[2])
        if mutation == "record_hash":
            records[0]["best_of_n_feasible"] = False
        elif mutation == "sampling_seed":
            records[0]["sampling_seed"] += 1
            records[0] = _seal(records[0], "record_sha256")
        elif mutation == "source_row":
            records[0]["source_row_locator"]["source_shard"] = "different.jsonl"
            records[0] = _seal(records[0], "record_sha256")
        details = right[0]
        details.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        config = json.loads(right[1].read_text(encoding="utf-8"))
        raw = details.read_bytes()
        config.pop("run_config_sha256")
        config["details_artifact"] = {
            "basename": details.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "record_count": len(records),
        }
        right[1].write_text(
            json.dumps(_seal(config, "run_config_sha256"), sort_keys=True),
            encoding="utf-8",
        )
    args = _args(tmp_path, left[:2], right[:2])
    with pytest.raises(DevelopmentComparisonError, match=message):
        run_comparison(args)
    assert not Path(args.output).exists()


def test_create_once_receipt_refuses_overwrite(tmp_path: Path):
    left = _write_native(tmp_path / "left", "left", [[0.2, None]])
    right = _write_native(tmp_path / "right", "right", [[0.1, None]])
    args = _args(tmp_path, left[:2], right[:2])
    run_comparison(args)
    original = Path(args.output).read_bytes()
    with pytest.raises(DevelopmentComparisonError, match="refusing to overwrite"):
        run_comparison(args)
    assert Path(args.output).read_bytes() == original
