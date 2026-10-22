from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_benchmark", ROOT / "scripts" / "freeze_benchmark.py"
)
assert SPEC is not None and SPEC.loader is not None
freeze_benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = freeze_benchmark
SPEC.loader.exec_module(freeze_benchmark)


def _tsp_row(coords: list[list[float]] | None = None) -> dict:
    return {
        "instruction": "Solve the TSP and return a closed route.",
        "input": "Three nodes.",
        "output": "Route: [0, 1, 2, 0]\nObjective: 3.41421356237",
        "instance_json": coords
        or [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    }


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _fake_evaluator(_instruction, _input, _answer, _instance, _completion) -> dict:
    return {
        "is_feasible": True,
        "ground_truth_objective": 1.0,
        "gap_to_ground_truth": 0.0,
    }


def _build(test_path: Path, forbidden: list | None = None) -> dict:
    return freeze_benchmark.build_manifest(
        test_specs=[
            freeze_benchmark.DatasetSpec("tsp", "iid-confirmation-v1", str(test_path))
        ],
        forbidden_specs=forbidden or [],
        forbidden_source_registry=None,
        publication_mode=False,
        generator_contract={"name": "frozen-generator", "seed": 1001},
        solver_contract={"name": "exact-reference", "version": "1"},
        minimum_instances=1,
        evaluators={"tsp": _fake_evaluator},
        verifier_identity={"name": "test-strict-verifier", "sha256": "v1"},
    )


def test_manifest_hash_and_exact_file_identity_are_self_verifying(tmp_path: Path) -> None:
    test_path = _write_rows(tmp_path / "iid.json", [_tsp_row()])
    manifest = _build(test_path)
    freeze_benchmark.verify_manifest_integrity(manifest)
    assert manifest["status"] == "frozen_before_inference"
    assert manifest["test_sets"]["tsp"]["rows"] == 1
    assert manifest["overlap_audit"]["passed"] is True

    manifest["minimum_instances_per_task"] = 2
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        freeze_benchmark.verify_manifest_integrity(manifest)


def test_freeze_rejects_duplicate_or_previously_seen_instances(tmp_path: Path) -> None:
    duplicate_path = _write_rows(
        tmp_path / "duplicate.json", [_tsp_row(), _tsp_row()]
    )
    with pytest.raises(ValueError, match="duplicate independent instance"):
        _build(duplicate_path)

    test_path = _write_rows(tmp_path / "iid.json", [_tsp_row()])
    train_path = _write_rows(tmp_path / "train.json", [_tsp_row()])
    forbidden = [
        freeze_benchmark.DatasetSpec("tsp", "sft-train", str(train_path))
    ]
    with pytest.raises(ValueError, match="leakage detected"):
        _build(test_path, forbidden)


def test_reference_solution_must_pass_shared_strict_verifier(tmp_path: Path) -> None:
    valid_path = _write_rows(tmp_path / "valid.json", [_tsp_row()])
    manifest = freeze_benchmark.build_manifest(
        test_specs=[freeze_benchmark.DatasetSpec("tsp", "iid", str(valid_path))],
        forbidden_specs=[],
        forbidden_source_registry=None,
        publication_mode=False,
        generator_contract={"name": "generator"},
        solver_contract={"name": "solver"},
        minimum_instances=1,
    )
    assert manifest["test_sets"]["tsp"]["reference_audit"]["strictly_feasible"] == 1

    invalid = _tsp_row()
    invalid["output"] = "Route: [0, 1, 0]\nObjective: 2"
    invalid_path = _write_rows(tmp_path / "invalid.json", [invalid])
    with pytest.raises(ValueError, match="strict verifier rejects reference"):
        freeze_benchmark.build_manifest(
            test_specs=[
                freeze_benchmark.DatasetSpec("tsp", "iid", str(invalid_path))
            ],
            forbidden_specs=[],
            forbidden_source_registry=None,
            publication_mode=False,
            generator_contract={"name": "generator"},
            solver_contract={"name": "solver"},
            minimum_instances=1,
        )


def test_main_refuses_to_overwrite_frozen_manifest(tmp_path: Path) -> None:
    test_path = _write_rows(tmp_path / "iid.json", [_tsp_row()])
    generator_path = tmp_path / "generator.json"
    solver_path = tmp_path / "solver.json"
    output_path = tmp_path / "manifest.json"
    generator_path.write_text('{"name":"generator"}', encoding="utf-8")
    solver_path.write_text('{"name":"solver"}', encoding="utf-8")
    output_path.write_text("already frozen", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_benchmark.main(
            [
                "--test",
                f"tsp,iid,{test_path}",
                "--generator-contract",
                str(generator_path),
                "--solver-contract",
                str(solver_path),
                "--minimum-instances-per-task",
                "1",
                "--output",
                str(output_path),
            ]
        )


def test_production_freeze_requires_registry_and_unique_forbidden_specs(
    tmp_path: Path,
) -> None:
    test_path = _write_rows(tmp_path / "iid.json", [_tsp_row()])
    specs = [
        freeze_benchmark.DatasetSpec(task, "iid", str(test_path))
        for task in freeze_benchmark.SUPPORTED_TASKS
    ]

    def build(forbidden):
        return freeze_benchmark.build_manifest(
            test_specs=specs,
            forbidden_specs=forbidden,
            forbidden_source_registry=None,
            publication_mode=True,
            generator_contract={"name": "generator"},
            solver_contract={"name": "solver"},
            minimum_instances=freeze_benchmark.PUBLICATION_INSTANCES_PER_TASK,
            evaluators={"tsp": _fake_evaluator},
            verifier_identity={"name": "verifier"},
            generation_receipt={"status": "references_attached"},
        )

    with pytest.raises(ValueError, match="forbidden-source-registry"):
        build([])

    complete = [
        freeze_benchmark.DatasetSpec("tsp", label, str(test_path))
        for label in ("sft-train", "rl-train", "historical-eval")
    ]
    with pytest.raises(ValueError, match="duplicate forbidden source"):
        build([*complete, complete[0]])
