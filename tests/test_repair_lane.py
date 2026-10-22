from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clasp.data.benchmark import (
    benchmark_row_sha256,
    reference_output_sha256,
    semantic_instance_sha256,
)
from clasp.entrypoints.evaluate import (
    TASK_EVALUATORS,
    _canonical_json_bytes,
    _example_sampling_seed,
    _experiment_fingerprint,
    _protocol_fingerprint_material,
    _self_hash_payload,
    _strict_verifier_identity,
)
from clasp.entrypoints.repair import (
    REPAIR_ALGORITHM_VERSION,
    RepairContractError,
    _repair_record,
    repair_completion,
    run_repair,
)

TASK_CASES = {
    "tsp": {
        "instance": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "answer": "Route: [0, 1, 2, 3, 0]\nObjective: 4",
        "near": "Route: [0, 1, 2, 0]\nObjective: 3.4",
    },
    "cvrp": {
        "instance": [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [0, 1, 1, 1],
            2,
        ],
        "answer": "Routes: [[0, 1, 3, 0], [0, 2, 0]]\nObjective: 5.4",
        "near": "Routes: [[0, 1, 0]]\nObjective: 2",
    },
    "op": {
        "instance": [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]],
            [0, 2, 3, 4],
            2.5,
        ],
        "answer": "Route: [0, 1, 3]\nObjective: 6",
        "near": "Route: [0, 1, 1, 99]\nObjective: 2",
    },
    "mis": {
        "instance": {"num_nodes": 4, "edges": [[0, 1], [1, 2]]},
        "answer": "Set: [0, 2, 3]\nObjective: 3",
        "near": "Set: [0, 1, 3]\nObjective: 3",
    },
    "jssp": {
        "instance": [[0, 1, 1, 1], [1, 1, 0, 1]],
        "answer": "Schedule: [[0, 1], [0, 1]]\nMakespan: 4",
        "near": "Schedule: [[0], [1]]\nMakespan: 2",
    },
}


def _config_dict() -> dict:
    return {
        "schema_version": 1,
        "algorithm_version": REPAIR_ALGORITHM_VERSION,
        "tsp_max_passes": 4,
        "cvrp_max_passes": 4,
        "op_max_passes": 4,
        "jssp_max_passes": 2,
        "selection_policy": "strict_feasibility_then_task_objective",
        "ground_truth_access_policy": "verifier_only",
    }


def _source_artifacts() -> tuple[dict, list[dict]]:
    verifier = _strict_verifier_identity()
    data_files: list[dict] = []
    data_manifest = hashlib.sha256(_canonical_json_bytes(data_files)).hexdigest()
    code_files: list[dict] = []
    code_identity = {
        "manifest_sha256": hashlib.sha256(_canonical_json_bytes(code_files)).hexdigest(),
        "files": code_files,
    }
    provenance = {
        "schema_version": 1,
        "model": {"test_model": "immutable"},
        "data": {"manifest_sha256": data_manifest, "files": data_files},
        "prompting": {"no_reason": True},
        "generation": {
            "num_return_sequences": 2,
            "seed": 271828,
            "seed_policy": "per_example_sha256_v1",
        },
        "selection": {"tasks": sorted(TASK_CASES), "split": "eval"},
        "verifier": verifier,
        "code": code_identity,
        "runtime_versions": {"python": "test"},
    }
    experiment_fingerprint = _experiment_fingerprint(provenance)
    protocol_fingerprint = _experiment_fingerprint(_protocol_fingerprint_material(provenance))
    config = {
        "run_status": "complete",
        "experiment_fingerprint_schema_version": 1,
        "experiment_fingerprint": experiment_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
        "experiment_provenance": provenance,
        "code_identity": code_identity,
        "verifier_identity": verifier,
        "tasks": sorted(TASK_CASES),
        "split": "eval",
        "generation": {**provenance["generation"], "best_of_n": 2},
    }

    records = []
    for example_index, (task, case) in enumerate(TASK_CASES.items()):
        instruction = f"Solve the {task.upper()} instance."
        input_text = f"Structured {task} test instance."
        instance = case["instance"]
        answer = case["answer"]
        candidates = []
        for candidate_index, completion in enumerate(("", case["near"])):
            metrics = TASK_EVALUATORS[task](instruction, input_text, answer, instance, completion)
            candidates.append(
                {
                    **metrics,
                    "candidate_index": candidate_index,
                    "completion": completion,
                    "completion_tokens": 0 if not completion else 8,
                    "decoding_temperature": 0.7,
                    "decoding_top_p": 0.9,
                }
            )
        transformed = {
            "instruction": instruction,
            "input": input_text,
            "output": answer,
            "instance_json": instance,
        }
        records.append(
            {
                "task": task,
                "split": "eval",
                "example_index": example_index,
                "source_json": "test.json",
                "source_json_files": ["test.json"],
                "instruction": instruction,
                "input_problem": input_text,
                "ground_truth_output": answer,
                "instance_size": 4,
                "instance_json": instance,
                "instance_sha256": hashlib.sha256(_canonical_json_bytes(instance)).hexdigest(),
                "semantic_instance_sha256": semantic_instance_sha256(
                    task, transformed, require_structured=True
                ),
                "benchmark_row_sha256": benchmark_row_sha256(transformed),
                "reference_output_sha256": reference_output_sha256(answer),
                "transformed_example_sha256": hashlib.sha256(
                    _canonical_json_bytes(transformed)
                ).hexdigest(),
                "source_row_locator": {
                    "data_manifest_sha256": data_manifest,
                    "task": task,
                    "split": "eval",
                    "merged_dataset_index": example_index,
                },
                "sampling_seed": _example_sampling_seed(271828, task, "eval", example_index),
                "experiment_fingerprint": experiment_fingerprint,
                "experiment_fingerprint_schema_version": 1,
                "protocol_fingerprint": protocol_fingerprint,
                "strict_verifier": verifier,
                "prompt_tokens": 10,
                "prompt_was_truncated": False,
                "generation_time_sec": 0.1,
                "generated_tokens": sum(candidate["completion_tokens"] for candidate in candidates),
                "peak_gpu_memory_allocated_bytes": 1024,
                "peak_gpu_memory_reserved_bytes": 2048,
                "num_return_sequences": len(candidates),
                "candidates": candidates,
            }
        )
    return config, records


def _write_source(tmp_path: Path, config: dict, records: list[dict]) -> tuple[Path, Path]:
    config_path = tmp_path / "run_config.json"
    details_path = tmp_path / "run_details.jsonl"
    sealed_records = [_self_hash_payload(record, "record_sha256") for record in records]
    details_path.write_text(
        "".join(json.dumps(record) + "\n" for record in sealed_records),
        encoding="utf-8",
    )
    details_bytes = details_path.read_bytes()
    sealed_config = dict(config)
    sealed_config["details_artifact"] = {
        "basename": details_path.name,
        "sha256": hashlib.sha256(details_bytes).hexdigest(),
        "size_bytes": len(details_bytes),
        "record_count": len(sealed_records),
    }
    sealed_config = _self_hash_payload(sealed_config, "run_config_sha256")
    config_path.write_text(json.dumps(sealed_config), encoding="utf-8")
    return config_path, details_path


def _args(tmp_path: Path, config_path: Path, details_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        details=[str(details_path)],
        source_config=str(config_path),
        output=str(tmp_path / "repaired.jsonl"),
        system_name="test-repair",
        tsp_max_passes=4,
        cvrp_max_passes=4,
        op_max_passes=4,
        jssp_max_passes=2,
    )


def test_blank_and_nearly_valid_candidates_repair_for_all_tasks(tmp_path: Path):
    config, records = _source_artifacts()
    original_records = [
        _self_hash_payload(record, "record_sha256") for record in copy.deepcopy(records)
    ]
    config_path, details_path = _write_source(tmp_path, config, records)
    output_path, summary_path = run_repair(_args(tmp_path, config_path, details_path))

    output_records = [
        json.loads(line) for line in Path(output_path).read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    assert len(output_records) == len(TASK_CASES)
    assert summary["system_lane"] == "deterministic_repair_and_improve_hybrid"
    assert summary["repaired_strict"]["candidate_feasibility_rate"] == 1.0
    assert summary["raw_strict"]["candidate_feasibility_rate"] < 1.0
    assert "FALCON" in summary["limitations"][1]
    snapshot = summary["source_config_snapshot"]
    assert snapshot["source_config_payload"] == json.loads(config_path.read_text())
    assert snapshot["source_config_identity"] == {
        "basename": config_path.name,
        "size_bytes": config_path.stat().st_size,
        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    assert snapshot["exact_peft_kl_runtime_receipt"] is None
    details_bytes = Path(output_path).read_bytes()
    assert summary["details_artifact"] == {
        "basename": Path(output_path).name,
        "sha256": hashlib.sha256(details_bytes).hexdigest(),
        "size_bytes": len(details_bytes),
        "record_count": len(output_records),
    }
    unsigned_summary = dict(summary)
    assert (
        unsigned_summary.pop("summary_sha256")
        == hashlib.sha256(_canonical_json_bytes(unsigned_summary)).hexdigest()
    )
    for original, repaired in zip(original_records, output_records):
        assert repaired["raw_record"] == original
        assert (
            repaired["raw_record_sha256"]
            == hashlib.sha256(_canonical_json_bytes(original)).hexdigest()
        )
        unsigned_repair = dict(repaired)
        assert (
            unsigned_repair.pop("repair_record_sha256")
            == hashlib.sha256(_canonical_json_bytes(unsigned_repair)).hexdigest()
        )
        for candidate in repaired["candidates"]:
            assert candidate["is_feasible"] is True
            assert candidate["repaired_metrics"]["is_feasible"] is True
            assert (
                candidate["raw_completion"]
                == original["candidates"][candidate["candidate_index"]]["completion"]
            )
            assert candidate["repaired_completion"]


def test_repair_artifact_is_consumable_by_publication_aggregator(tmp_path: Path):
    from scripts.aggregate_sota import RunSpec, load_run, summarize_records

    config, records = _source_artifacts()
    config_path, details_path = _write_source(tmp_path, config, records)
    output_path, _summary_path = run_repair(_args(tmp_path, config_path, details_path))
    loaded = load_run(RunSpec(method="clasp-repair", seed=271828, path=output_path), expected_k=2)
    aggregate = summarize_records(list(loaded.values()))
    assert aggregate["pass_at_k"] == 1.0
    assert aggregate["candidate_feasibility"] == 1.0


@pytest.mark.parametrize("task", sorted(TASK_CASES))
def test_repair_completion_is_deterministic_and_has_no_ground_truth_argument(task: str):
    case = TASK_CASES[task]
    kwargs = {
        "task": task,
        "instance": case["instance"],
        "completion": case["near"],
        "config": _config_dict(),
    }
    first = repair_completion(**kwargs)
    second = repair_completion(**kwargs)
    assert first == second
    assert "answer" not in repair_completion.__annotations__
    assert "instruction" not in repair_completion.__annotations__
    assert "input_text" not in repair_completion.__annotations__


def test_ground_truth_changes_metrics_but_not_repaired_completion():
    config, records = _source_artifacts()
    del config
    record = records[0]
    altered = copy.deepcopy(record)
    altered["ground_truth_output"] = "Route: [0, 3, 2, 1, 0]\nObjective: 999"
    for candidate in altered["candidates"]:
        candidate.pop("verification_context", None)
        candidate.pop("verification_context_sha256", None)
    verifier = _strict_verifier_identity()
    first, _, _ = _repair_record(
        record,
        repair_fingerprint="f" * 64,
        verifier=verifier,
        config=_config_dict(),
    )
    second, _, _ = _repair_record(
        altered,
        repair_fingerprint="f" * 64,
        verifier=verifier,
        config=_config_dict(),
    )
    assert [item["repaired_completion"] for item in first["candidates"]] == [
        item["repaired_completion"] for item in second["candidates"]
    ]


@pytest.mark.parametrize(
    "corruption",
    [
        "experiment",
        "instance",
        "transformed",
        "schema",
        "locator",
        "sampling_seed",
        "candidate_index",
    ],
)
def test_corrupt_or_legacy_provenance_fails_before_writing(tmp_path: Path, corruption: str):
    config, records = _source_artifacts()
    record = records[0]
    if corruption == "experiment":
        record["experiment_fingerprint"] = "0" * 64
    elif corruption == "instance":
        record["instance_json"][0][0] = 123.0
    elif corruption == "transformed":
        record["transformed_example_sha256"] = "0" * 64
    elif corruption == "schema":
        record["experiment_fingerprint_schema_version"] = 0
    elif corruption == "locator":
        record["source_row_locator"]["merged_dataset_index"] = 999
    elif corruption == "sampling_seed":
        record["sampling_seed"] += 1
    else:
        record["candidates"][0]["candidate_index"] = 9
    config_path, details_path = _write_source(tmp_path, config, records)
    args = _args(tmp_path, config_path, details_path)
    with pytest.raises(RepairContractError):
        run_repair(args)
    assert not Path(args.output).exists()
    assert not Path(str(args.output).replace(".jsonl", "__summary.json")).exists()


def test_corrupt_candidate_specific_context_fails_closed(tmp_path: Path):
    config, records = _source_artifacts()
    record = records[0]
    for candidate in record["candidates"]:
        context = {
            "instruction": record["instruction"],
            "input_problem": record["input_problem"],
            "ground_truth_output": record["ground_truth_output"],
            "instance_json": record["instance_json"],
        }
        candidate["tsp_label_tta_variant"] = "reverse"
        candidate["verification_context"] = context
        candidate["verification_context_sha256"] = hashlib.sha256(
            _canonical_json_bytes(context)
        ).hexdigest()
    record["candidates"][1]["verification_context"]["input_problem"] = "tampered"
    config_path, details_path = _write_source(tmp_path, config, records)
    with pytest.raises(RepairContractError, match="corrupted"):
        run_repair(_args(tmp_path, config_path, details_path))


def test_duplicate_row_identity_fails_closed(tmp_path: Path):
    config, records = _source_artifacts()
    records.append(copy.deepcopy(records[0]))
    config_path, details_path = _write_source(tmp_path, config, records)
    with pytest.raises(RepairContractError, match="duplicate"):
        run_repair(_args(tmp_path, config_path, details_path))


def test_output_cannot_overwrite_source_artifact(tmp_path: Path):
    config, records = _source_artifacts()
    config_path, details_path = _write_source(tmp_path, config, records)
    args = _args(tmp_path, config_path, details_path)
    args.output = str(details_path)
    original = details_path.read_bytes()
    with pytest.raises(RepairContractError, match="must not overwrite"):
        run_repair(args)
    assert details_path.read_bytes() == original
