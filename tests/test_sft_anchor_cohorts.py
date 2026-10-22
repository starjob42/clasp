"""Fresh development generation, numerical witnesses and prompt projection."""
import ast
import importlib.util
import json
from pathlib import Path
import random
import re

import pytest

from clasp.baselines.sft_anchor_pilot import (
    graph_semantic_id, score_completion, validate_contract,
)
from clasp.data.prompts import build_jiang_f638_inference_prompt

ROOT = Path(__file__).resolve().parents[1]
PREPARER = ROOT / "experiments/rloo_sft_anchor_20261013/prepare_study.py"
SPEC = importlib.util.spec_from_file_location("fresh_sft_anchor_prepare", PREPARER)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


@pytest.fixture(scope="module", params=("op", "cvrp"))
def cohorts(request):
    return request.param, prepare.build_task_cohorts(request.param, seed=2026101309)


def test_exact_size_counts_bands_and_deterministic_graph_disjointness(cohorts):
    task, value = cohorts
    assert len(value["training_rows"]) == 128 and len(value["validation_rows"]) == 96
    for name, counts in (("training_rows", (43, 43, 42)), ("validation_rows", (32, 32, 32))):
        rows = value[name]
        assert [sum(row["development_band"] == list(band) for row in rows)
                for band in prepare.BANDS] == list(counts)
        for row in rows:
            size = len(row["instance_json"][0]) - int(task == "cvrp")
            assert row["development_band"][0] <= size <= row["development_band"][1]
    train = {row["graph_semantic_id"] for row in value["training_rows"]}
    validation = {row["graph_semantic_id"] for row in value["validation_rows"]}
    assert len(train) == 128 and len(validation) == 96 and not train & validation
    assert value == prepare.build_task_cohorts(task, seed=2026101309)


def test_raw_integer_instances_and_all_reference_witnesses_verified(cohorts):
    task, value = cohorts
    for row in value["training_rows"] + value["validation_rows"]:
        coords, weights, limit = row["instance_json"]
        assert all(type(number) is int and 0 <= number <= 1000 for point in coords for number in point)
        assert len({tuple(point) for point in coords}) == len(coords)
        assert weights[0] == 0 and all(type(weight) is int and 1 <= weight <= 10 for weight in weights[1:])
        score = score_completion(task, row["output"], row["instance_json"], row["output"])
        assert score["feasible"] and score["reference_objective"] > 0
        assert score["signed_gap"] == 0 and score["u10"] == 1
        assert row["reference_evidence"]["objective"] == score["objective"]
        assert not row["reference_evidence"]["optimality_proven"]
        assert row["reference_evidence"]["feasibility_independently_verified"]
        if task == "cvrp":
            assert type(limit) is int and 30 <= limit <= 90
        else:
            assert limit > 0 and round(limit, 1) == limit


def test_exact_two_other_node_neighbors_lossless_numeric_fields_and_context(cohorts):
    task, value = cohorts
    label = "demand" if task == "cvrp" else "prize"
    expression = re.compile(r"Node (\d+), coordinates: (\[[^\]]+\]), " + label +
                            r": (\d+), neighbors: \[([^\]]+)\]([;.])")
    for row in value["training_rows"] + value["validation_rows"]:
        coords, weights, _ = row["instance_json"]
        matches = list(expression.finditer(row["input"]))
        assert len(matches) == len(coords) and "".join(match[0] for match in matches) == row["input"]
        for node, match in enumerate(matches):
            assert int(match[1]) == node and ast.literal_eval(match[2]) == coords[node]
            assert int(match[3]) == weights[node]
            parsed = [entry.split(": ") for entry in match[4].split(", ")]
            expected = prepare.neighbors(coords, node)
            assert len(parsed) == 2
            for (identifier, printed), (neighbor, cost) in zip(parsed, expected):
                assert int(identifier) == neighbor and neighbor != node
                assert abs(float(printed) - cost) <= .05000000001
            assert match[5] == ("." if node == len(coords) - 1 else ";")
        assert row["prompt"] == build_jiang_f638_inference_prompt(row["instruction"], row["input"])
        assert row["prompt"].endswith("    ### Response:")
        assert row["output"] not in row["prompt"]


def test_duplicate_graph_exclusion_replays_rejection_without_outcome_selection():
    excluded = prepare.generate_row("op", random.Random(7), split="old", position=0,
                                    band=prepare.BANDS[0], draw=1)
    value = prepare.build_task_cohorts("op", seed=7, train_count=3, validation_count=3,
                                      excluded_rows=[excluded])
    assert value["generation"]["rejected_reused_graphs"] == 1
    assert all(row["graph_semantic_id"] != excluded["graph_semantic_id"]
               for row in value["training_rows"] + value["validation_rows"])
    assert not value["generation"]["model_outcome_informed_selection"]


def test_known_witness_algorithms_enforce_open_budget_and_capacity():
    op = [[[0, 0], [1, 0], [4, 0]], [0, 4, 9], 1.0]
    assert prepare.op_witness(op) == [0, 1]
    cvrp = [[[0, 0], [1, 0], [0, 1]], [0, 2, 2], 2]
    assert prepare.cvrp_witness(cvrp) == [[0, 1, 0], [0, 2, 0]]
    op_text = "Route: " + repr(prepare.op_witness(op))
    cvrp_text = "Routes: " + repr(prepare.cvrp_witness(cvrp))
    assert score_completion("op", op_text, op, op_text)["feasible"]
    assert score_completion("cvrp", cvrp_text, cvrp, cvrp_text)["feasible"]


def test_exclusion_reader_projects_inputs_and_forbids_protected_performance_paths(tmp_path):
    row = prepare.generate_row("op", random.Random(1), split="train", position=0,
                               band=prepare.BANDS[0], draw=1)
    source = tmp_path / "public_op_input.json"
    source.write_text(json.dumps([row]))
    projected, binding = prepare.read_exclusion_input("op", source)
    assert list(projected[0]) == ["instance_json"] and binding["input_rows"] == 1
    assert graph_semantic_id("op", projected[0]) == row["graph_semantic_id"]
    protected = tmp_path / "protected_performance.json"
    protected.write_text("This file must not be opened")
    with pytest.raises(ValueError, match="reads forbidden"):
        prepare.read_exclusion_input("op", protected)


def test_sealed_production_study_has_bound_initializer_and_explicit_admission_limits(tmp_path):
    native = tmp_path / "native_inputs"
    for task in prepare.TASKS:
        folder = native / task / "sft"
        folder.mkdir(parents=True)
        (folder / "adapter_model.safetensors").write_bytes(b"test-only-identity-fixture-" + task.encode())
        (folder / "adapter_config.json").write_text("{}")
        (folder / "checkpoint_fidelity.json").write_text("{}")
        training = native / task / "train"
        training.mkdir()
        row = prepare.generate_row(task, random.Random(6), split="old_train", position=0,
                                   band=prepare.BANDS[0], draw=1)
        (training / "train_rl.json").write_text(json.dumps([row]))
    output = tmp_path / "new_study"
    study = prepare.prepare_study(output, native, train_seeds=(43, 44))
    assert len(study["contracts"]) == 4 and study["actual_training_or_evaluation_responses"] == 0
    assert not study["superiority_established"]
    assert not study["provenance"]["SFT500K_full_exclusion_certified"]
    assert not study["provenance"]["protected500_inputs_or_performance_read"]
    assert (output / "study.json").is_file()
    assert study["recipe"]["updates"] == 100 and study["recipe"]["completion_cap"] == 3000
    for item in study["contracts"]:
        c = validate_contract(json.loads(Path(item["file"]["path"]).read_text()))
        assert len(c["training_rows"]) == 128 and len(c["validation_rows"]) == 96
        assert c["scope"] == "production" and c["training"]["fixed_loss_denominator"] == 3000
        assert c["initializer"]["sha256"] == study["initializers"][c["task"]]["sha256"]
        assert c["cohort_provenance"]["study_scope"] == "development_pilot"
        assert not c["cohort_provenance"]["original_FOARL_exact_replica"]
    with pytest.raises(ValueError, match="Preserve existing study"):
        prepare.prepare_study(output, native)
