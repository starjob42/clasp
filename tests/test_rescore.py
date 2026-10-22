from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clasp.entrypoints.evaluate import (
    _canonical_json_bytes,
    _experiment_fingerprint,
    _protocol_fingerprint_material,
)
from clasp.entrypoints.rescore import run_rescore


def _seal(payload: dict, field: str) -> dict:
    sealed = dict(payload)
    sealed[field] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return sealed


def test_shared_strict_rescore_writes_recomputed_metrics(tmp_path: Path):
    source_path = tmp_path / "external.jsonl"
    output_path = tmp_path / "strict.jsonl"
    source_record = {
        "task": "tsp",
        "split": "eval",
        "example_index": 7,
        "instruction": "Solve this TSP.",
        "input_problem": "three nodes",
        "ground_truth_output": "Route: [0, 1, 2, 0]\nObjective: 3.414",
        "instance_json": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        "semantic_instance_sha256": "a" * 64,
        "benchmark_row_sha256": "b" * 64,
        "reference_output_sha256": "c" * 64,
        "candidates": [
            {"completion": "Route: [0, 1, 2, 0]\nObjective: 3.414"},
            {"completion": "Route: [0, 1, 0]\nObjective: 2.0"},
        ],
        "experiment_fingerprint": "source-fingerprint",
    }
    source_path.write_text(json.dumps(source_record) + "\n", encoding="utf-8")

    details_path, summary_path = run_rescore(
        SimpleNamespace(
            details=[str(source_path)],
            output=str(output_path),
            system_name="external-system",
        )
    )

    rescored = json.loads(Path(details_path).read_text(encoding="utf-8").strip())
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    assert rescored["candidates"][0]["is_feasible"] is True
    assert rescored["candidates"][1]["is_feasible"] is False
    assert rescored["best_candidate_index"] == 0
    assert rescored["source_experiment_fingerprint"] == "source-fingerprint"
    assert rescored["semantic_instance_sha256"] == "a" * 64
    assert rescored["benchmark_row_sha256"] == "b" * 64
    assert rescored["reference_output_sha256"] == "c" * 64
    assert rescored["strict_verifier"]["name"] == "clasp_strict_shared_verifier"
    assert summary["overall"]["best_of_n_instance_feasibility_rate"] == 1.0
    assert summary["rescore_fingerprint"] == rescored["rescore_fingerprint"]
    assert summary["source_config_snapshot"] is None
    unsigned_record = dict(rescored)
    assert (
        unsigned_record.pop("rescore_record_sha256")
        == hashlib.sha256(_canonical_json_bytes(unsigned_record)).hexdigest()
    )
    details_bytes = Path(details_path).read_bytes()
    assert summary["details_artifact"] == {
        "basename": Path(details_path).name,
        "sha256": hashlib.sha256(details_bytes).hexdigest(),
        "size_bytes": len(details_bytes),
        "record_count": 1,
    }
    unsigned_summary = dict(summary)
    assert (
        unsigned_summary.pop("summary_sha256")
        == hashlib.sha256(_canonical_json_bytes(unsigned_summary)).hexdigest()
    )


def test_rescore_refuses_to_overwrite_source_details(tmp_path: Path):
    source_path = tmp_path / "external.jsonl"
    original = b'{"task":"tsp"}\n'
    source_path.write_bytes(original)
    with pytest.raises(ValueError, match="must not overwrite"):
        run_rescore(
            SimpleNamespace(
                details=[str(source_path)],
                output=str(source_path),
                source_config=None,
                system_name="external-system",
            )
        )
    assert source_path.read_bytes() == original


def test_rescore_binds_every_source_record_to_completed_config(tmp_path: Path):
    source_path = tmp_path / "run_details__native.jsonl"
    config_path = tmp_path / "run_config__native.json"
    output_path = tmp_path / "strict.jsonl"
    verifier = {"name": "test-verifier", "sha256": "d" * 64}
    provenance = {
        "model": {"checkpoint": "checkpoint-400"},
        "decoder": {"num_return_sequences": 1},
        "selection": {"tasks": ["tsp"], "split": "eval"},
        "verifier": verifier,
    }
    experiment_fingerprint = _experiment_fingerprint(provenance)
    protocol_fingerprint = _experiment_fingerprint(_protocol_fingerprint_material(provenance))
    training_sha256 = "e" * 64
    source_record = _seal(
        {
            "task": "tsp",
            "split": "eval",
            "example_index": 0,
            "instruction": "Solve this TSP.",
            "input_problem": "three nodes",
            "ground_truth_output": "Route: [0, 1, 2, 0]\nObjective: 3.414",
            "instance_json": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            "candidates": [{"completion": "Route: [0, 1, 2, 0]\nObjective: 3.414"}],
            "experiment_fingerprint": experiment_fingerprint,
            "experiment_fingerprint_schema_version": 1,
            "protocol_fingerprint": protocol_fingerprint,
            "strict_verifier": verifier,
            "training_run_contract_sha256": training_sha256,
        },
        "record_sha256",
    )
    source_path.write_text(json.dumps(source_record) + "\n", encoding="utf-8")
    source_bytes = source_path.read_bytes()
    source_config = _seal(
        {
            "run_status": "complete",
            "tasks": ["tsp"],
            "split": "eval",
            "experiment_provenance": provenance,
            "experiment_fingerprint": experiment_fingerprint,
            "protocol_fingerprint": protocol_fingerprint,
            "training_run_contract": {"schema_version": 1},
            "training_run_contract_sha256": training_sha256,
            "details_artifact": {
                "basename": source_path.name,
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "size_bytes": len(source_bytes),
                "record_count": 1,
            },
        },
        "run_config_sha256",
    )
    config_path.write_text(json.dumps(source_config), encoding="utf-8")

    _details_path, summary_path = run_rescore(
        SimpleNamespace(
            details=[str(source_path)],
            output=str(output_path),
            source_config=str(config_path),
            system_name="native-system",
        )
    )
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    snapshot = summary["source_config_snapshot"]
    assert snapshot["source_config_payload"] == source_config
    assert snapshot["source_config_identity"] == {
        "basename": config_path.name,
        "size_bytes": config_path.stat().st_size,
        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    assert snapshot["exact_peft_kl_runtime_receipt"] is None

    tampered = dict(source_record)
    tampered.pop("record_sha256")
    tampered["experiment_fingerprint"] = "0" * 64
    tampered = _seal(tampered, "record_sha256")
    source_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    tampered_bytes = source_path.read_bytes()
    source_config.pop("run_config_sha256")
    source_config["details_artifact"] = {
        "basename": source_path.name,
        "sha256": hashlib.sha256(tampered_bytes).hexdigest(),
        "size_bytes": len(tampered_bytes),
        "record_count": 1,
    }
    source_config = _seal(source_config, "run_config_sha256")
    config_path.write_text(json.dumps(source_config), encoding="utf-8")
    with pytest.raises(ValueError, match="experiment fingerprint mismatch"):
        run_rescore(
            SimpleNamespace(
                details=[str(source_path)],
                output=str(output_path),
                source_config=str(config_path),
                system_name="native-system",
            )
        )
