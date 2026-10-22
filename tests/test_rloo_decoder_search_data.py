"""Mathematical and split-isolation checks for the prospective decoder search."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "decoder_search_prepare_data_test", ROOT / "experiments/rloo_decoder_search_20261014/prepare_data.py")
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


@pytest.fixture(scope="module")
def deps():
    return DATA.dependencies()


@pytest.mark.parametrize("task,instance", [
    ("tsp", [[0, 0], [3, 0], [3, 4], [0, 4]]),
    ("cvrp", [[[0, 0], [3, 0], [3, 4], [0, 4]], [0, 3, 4, 3], 6]),
    ("op", [[[0, 0], [3, 0], [3, 4], [0, 4]], [0, 5, 7, 3], 7]),
    ("mis", {"num_nodes": 5, "edges": [[0, 1], [1, 2], [2, 3], [3, 4]]}),
    ("jssp", [[0, 3, 1, 2], [1, 4, 0, 1]]),
])
def test_heuristic_witness_satisfies_independent_math(task, instance, deps):
    row = DATA.attach_reference(task, {"instance_json": instance}, deps[1])
    action, error, _ = deps[1].raw_literal(row["output"], deps[1].LABELS[task])
    feasible, objective, reasons = deps[1].independent_check(task, instance, action, error)
    assert feasible, reasons
    assert objective == row["reference_objective"] > 0
    assert row["reference_kind"] == "verified_feasible_heuristic"
    assert row["reference_evidence"]["optimality_proven"] is False


def test_development_subsets_balanced_test_disjoint_and_seed_reproducible(deps):
    options = dict(seed=20261014, dev_count=24, test_count=3, forbidden=set(), deps=deps)
    cohorts, _ = DATA.build_task("tsp", **options)
    repeated, _ = DATA.build_task("tsp", **options)
    assert cohorts == repeated
    dev_ids = {row["semantic_id"] for row in cohorts["dev"]}
    test_ids = {row["semantic_id"] for row in cohorts["test"]}
    assert len(dev_ids) == 24 and len(test_ids) == 3 and not dev_ids & test_ids
    assert all(sum(row["band"] == band for row in cohorts["dev"]) == 8 for band in DATA.BANDS)
    assert all(sum(row["band"] == band for row in cohorts["test"]) == 1 for band in DATA.BANDS)
    for subset in ("screen", "coarse"):
        assert {row["semantic_id"] for row in cohorts[subset]} <= dev_ids
    for row in cohorts["dev"] + cohorts["test"]:
        assert row["semantic_id"] == row["semantic_instance_sha256"]
        assert "neighbors: ['" not in row["input"]
        assert row["prompt"].endswith("    ### Response:")


def test_forbidden_instance_is_rejected_without_changing_quotas(deps):
    options = dict(seed=1, dev_count=24, test_count=3, deps=deps)
    baseline, _ = DATA.build_task("mis", forbidden=set(), **options)
    forbidden = {baseline["dev"][0]["semantic_id"]}
    replacement, details = DATA.build_task("mis", forbidden=forbidden, **options)
    assert not forbidden & {row["semantic_id"] for split in ("dev", "test") for row in replacement[split]}
    assert details["dev"]["rejected_draws"]["previous_or_forbidden_instance"] >= 1
    assert all(sum(row["band"] == band for row in replacement["dev"]) == 8 for band in DATA.BANDS)


def test_generation_refuses_unbalanced_cohort_counts(deps):
    with pytest.raises(ValueError, match="multiples of three"):
        DATA.build_task("tsp", seed=1, dev_count=25, test_count=3, forbidden=set(), deps=deps)
