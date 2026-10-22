from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from clasp.data.benchmark import (
    benchmark_row_sha256,
    instance_sha256,
    reference_output_sha256,
    sha256_json,
)
from clasp.data.loaders import normalize_rows

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "generate_iid_benchmark", ROOT / "scripts" / "generate_iid_benchmark.py"
)
assert GENERATOR_SPEC is not None and GENERATOR_SPEC.loader is not None
generator = importlib.util.module_from_spec(GENERATOR_SPEC)
sys.modules[GENERATOR_SPEC.name] = generator
GENERATOR_SPEC.loader.exec_module(generator)

FREEZE_SPEC = importlib.util.spec_from_file_location(
    "freeze_benchmark_iid", ROOT / "scripts" / "freeze_benchmark.py"
)
assert FREEZE_SPEC is not None and FREEZE_SPEC.loader is not None
freeze = importlib.util.module_from_spec(FREEZE_SPEC)
sys.modules[FREEZE_SPEC.name] = freeze
FREEZE_SPEC.loader.exec_module(freeze)

CONTRACT = ROOT / "experiments" / "sota_20260814" / "iid_generator_contract.json"


def _generate(tmp_path: Path, name: str) -> tuple[Path, dict]:
    output = tmp_path / name
    receipt = generator.generate_instances(
        contract_path=str(CONTRACT),
        output_dir=str(output),
        count=1,
        smoke=True,
        official_source_root=None,
    )
    return output, receipt


def test_five_task_smoke_generation_is_deterministic_and_unlabeled(
    tmp_path: Path,
) -> None:
    first, first_receipt = _generate(tmp_path, "first")
    second, second_receipt = _generate(tmp_path, "second")

    assert first_receipt["status"] == "smoke_instances_generated"
    for task in generator.TASKS:
        first_task = first_receipt["tasks"][task]
        second_task = second_receipt["tasks"][task]
        assert first_task["ordered_instance_sha256"] == second_task[
            "ordered_instance_sha256"
        ]
        rows = json.loads((first / task / "eval" / "test.json").read_text())
        assert len(rows) == 1
        assert rows[0]["reference_status"] == "pending"
        assert "output" not in rows[0]
        assert rows[0]["semantic_instance_sha256"] == first_task[
            "ordered_semantic_instance_sha256"
        ][0]
        assert sha256_json(rows[0]["instance_json"]) != rows[0][
            "semantic_instance_sha256"
        ]
        assert first_task["ordered_benchmark_row_sha256"] == [
            benchmark_row_sha256(rows[0])
        ]
        if task == "op":
            displayed_budget = float(rows[0]["max_route_length"])
            assert rows[0]["instance_json"][2] == displayed_budget
            assert f"T = {displayed_budget:.1f}" in rows[0]["instruction"]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generator.generate_instances(
            contract_path=str(CONTRACT),
            output_dir=str(first),
            count=1,
            smoke=True,
            official_source_root=None,
        )


def test_smoke_references_assemble_and_bind_to_freeze_receipt(tmp_path: Path) -> None:
    draft, _ = _generate(tmp_path, "draft")
    references = tmp_path / "references"
    solver_contract_path = tmp_path / "solver.json"
    generator.make_smoke_references(
        draft_root=str(draft),
        references_dir=str(references),
        solver_contract_output=str(solver_contract_path),
    )
    assembled = tmp_path / "assembled"
    assembled_receipt = generator.assemble_references(
        draft_root=str(draft),
        references_root=str(references),
        generator_contract_path=str(CONTRACT),
        solver_contract_path=str(solver_contract_path),
        output_dir=str(assembled),
        smoke=True,
    )
    assert assembled_receipt["status"] == "smoke_references_attached"
    assert assembled_receipt["publishable"] is False

    def fake_evaluator(_instruction, _input, _answer, _instance, _completion):
        return {
            "is_feasible": True,
            "ground_truth_objective": 1.0,
            "gap_to_ground_truth": 0.0,
        }

    specs = [
        freeze.DatasetSpec(task, "smoke", str(assembled / task / "eval" / "test.json"))
        for task in generator.TASKS
    ]
    manifest = freeze.build_manifest(
        test_specs=specs,
        forbidden_specs=[],
        forbidden_source_registry=None,
        publication_mode=False,
        generator_contract=json.loads(CONTRACT.read_text()),
        solver_contract=json.loads(solver_contract_path.read_text()),
        minimum_instances=1,
        evaluators={task: fake_evaluator for task in generator.TASKS},
        verifier_identity=assembled_receipt["strict_verifier"],
        generation_receipt=assembled_receipt,
        allow_smoke_receipt=True,
    )
    assert manifest["generation_receipt"]["receipt_sha256"] == assembled_receipt[
        "receipt_sha256"
    ]
    for task in generator.TASKS:
        rows = json.loads((assembled / task / "eval" / "test.json").read_text())
        task_manifest = manifest["test_sets"][task]
        assert task_manifest["ordered_benchmark_row_sha256"] == [
            benchmark_row_sha256(rows[0])
        ]
        normalized = normalize_rows(rows, f"{task}.json")[0]
        assert benchmark_row_sha256(normalized) == benchmark_row_sha256(rows[0])
        assert instance_sha256(task, normalized) == instance_sha256(
            task, rows[0], require_structured=True
        )
        assert task_manifest["ordered_reference_output_sha256"] == [
            reference_output_sha256(rows[0]["output"])
        ]

    wrong_solver = json.loads(solver_contract_path.read_text())
    wrong_solver["identity_tamper"] = True
    with pytest.raises(ValueError, match="different reference solver contract"):
        freeze.build_manifest(
            test_specs=specs,
            forbidden_specs=[],
            forbidden_source_registry=None,
            publication_mode=False,
            generator_contract=json.loads(CONTRACT.read_text()),
            solver_contract=wrong_solver,
            minimum_instances=1,
            evaluators={task: fake_evaluator for task in generator.TASKS},
            verifier_identity=assembled_receipt["strict_verifier"],
            generation_receipt=assembled_receipt,
            allow_smoke_receipt=True,
        )

    with pytest.raises(ValueError, match="different strict verifier"):
        freeze.build_manifest(
            test_specs=specs,
            forbidden_specs=[],
            forbidden_source_registry=None,
            publication_mode=False,
            generator_contract=json.loads(CONTRACT.read_text()),
            solver_contract=json.loads(solver_contract_path.read_text()),
            minimum_instances=1,
            evaluators={task: fake_evaluator for task in generator.TASKS},
            verifier_identity={"name": "tampered-verifier"},
            generation_receipt=assembled_receipt,
            allow_smoke_receipt=True,
        )

    tampered_receipt = deepcopy(assembled_receipt)
    tampered_receipt["tasks"]["tsp"]["ordered_reference_output_sha256"][0] = "0" * 64
    tampered_receipt = generator._self_hashed(tampered_receipt, "receipt_sha256")
    with pytest.raises(ValueError, match="ordered reference hashes"):
        freeze.build_manifest(
            test_specs=specs,
            forbidden_specs=[],
            forbidden_source_registry=None,
            publication_mode=False,
            generator_contract=json.loads(CONTRACT.read_text()),
            solver_contract=json.loads(solver_contract_path.read_text()),
            minimum_instances=1,
            evaluators={task: fake_evaluator for task in generator.TASKS},
            verifier_identity=assembled_receipt["strict_verifier"],
            generation_receipt=tampered_receipt,
            allow_smoke_receipt=True,
        )

    with pytest.raises(ValueError, match="not freeze-eligible"):
        freeze.build_manifest(
            test_specs=specs,
            forbidden_specs=[],
            forbidden_source_registry=None,
            publication_mode=False,
            generator_contract=json.loads(CONTRACT.read_text()),
            solver_contract=json.loads(solver_contract_path.read_text()),
            minimum_instances=1,
            evaluators={task: fake_evaluator for task in generator.TASKS},
            verifier_identity=assembled_receipt["strict_verifier"],
            generation_receipt=assembled_receipt,
        )

    tsp_path = assembled / "tsp" / "eval" / "test.json"
    tampered_rows = json.loads(tsp_path.read_text())
    tampered_rows[0]["output"] += "\n"
    tsp_path.write_text(json.dumps(tampered_rows), encoding="utf-8")
    with pytest.raises(ValueError, match="ordered benchmark row hashes"):
        freeze.build_manifest(
            test_specs=specs,
            forbidden_specs=[],
            forbidden_source_registry=None,
            publication_mode=False,
            generator_contract=json.loads(CONTRACT.read_text()),
            solver_contract=json.loads(solver_contract_path.read_text()),
            minimum_instances=1,
            evaluators={task: fake_evaluator for task in generator.TASKS},
            verifier_identity=assembled_receipt["strict_verifier"],
            generation_receipt=assembled_receipt,
            allow_smoke_receipt=True,
        )


def test_prompt_only_forbidden_copy_has_same_semantic_instance_hash(
    tmp_path: Path,
) -> None:
    draft, _ = _generate(tmp_path, "draft")
    for task in generator.TASKS:
        row = json.loads((draft / task / "eval" / "test.json").read_text())[0]
        structured_hash = instance_sha256(task, row, require_structured=True)
        prompt_only = dict(row)
        prompt_only.pop("instance_json")
        assert instance_sha256(task, prompt_only) == structured_hash


def test_confirmation_generation_fails_without_verified_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="official-source-root"):
        generator.generate_instances(
            contract_path=str(CONTRACT),
            output_dir=str(tmp_path / "confirmation"),
            count=1_000,
            smoke=False,
            official_source_root=None,
        )
