from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/analyze_small_tsp_canonical_pilot_v1.py"


@pytest.fixture(scope="module")
def analysis_module():
    spec = importlib.util.spec_from_file_location("small_tsp_pilot_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> dict:
    output = tmp_path_factory.mktemp("small_tsp_pilot")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(output),
            "--bootstrap-replicates",
            "100",
            "--bootstrap-seed",
            "20260815",
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (output / "exact_k8_subset_rows.csv").is_file()
    assert (output / "per_seed_summary.csv").is_file()
    assert (output / "training_mechanism_trajectory.csv").is_file()
    return json.loads((output / "analysis.json").read_text(encoding="utf-8"))


def test_exact_leave_one_out_combinatorics(analysis_module) -> None:
    no_feasible = analysis_module.enumerate_leave_one_out([False] * 9, [None] * 9)
    assert sum(not row["pass"] for row in no_feasible) == 9

    one_feasible = analysis_module.enumerate_leave_one_out(
        [True] + [False] * 8, [0.25] + [None] * 8
    )
    assert sum(not row["pass"] for row in one_feasible) == 1
    assert one_feasible[0]["pass"] is False
    assert all(row["best_feasible_gap"] == 0.25 for row in one_feasible[1:])

    two_feasible = analysis_module.enumerate_leave_one_out(
        [True, True] + [False] * 7, [0.10, 0.20] + [None] * 7
    )
    assert all(row["pass"] for row in two_feasible)
    assert two_feasible[0]["best_feasible_gap"] == 0.20
    assert two_feasible[1]["best_feasible_gap"] == 0.10
    assert all(len(row["selected_candidate_indices"]) == 8 for row in two_feasible)


def test_exact_k8_metrics_and_denominators_are_sealed(bundle: dict) -> None:
    assert bundle["evidence_scope"] == "development_only_not_publication_or_sota_evidence"
    assert bundle["experiment"]["held_out_instances"] == 31
    assert bundle["experiment"]["training_seeds"] == [101, 202, 303]
    assert bundle["experiment"]["deployment_set_K"] == 8
    assert bundle["experiment"]["exact_subsets_per_instance"] == 9

    sft = bundle["runs"]["sft"]
    assert sft["primary_k8"]["pass_at_8"] == {
        "count": 259,
        "denominator": 279,
        "rate": 259 / 279,
    }
    expected_passes = {"101": 276, "202": 267, "303": 268}
    expected_candidates = {"101": 152, "202": 148, "303": 152}
    for seed in ("101", "202", "303"):
        run = bundle["runs"]["canonical"][seed]
        assert run["primary_k8"]["pass_at_8"]["count"] == expected_passes[seed]
        assert run["primary_k8"]["pass_at_8"]["denominator"] == 279
        assert (
            run["secondary_candidate_level"]["feasibility"]["count"]
            == expected_candidates[seed]
        )


def test_bootstrap_hierarchy_common_coverage_and_mechanism(bundle: dict) -> None:
    bootstrap = bundle["bootstrap"]
    assert bootstrap["design"]["unit_hierarchy"] == [
        "training_seed",
        "paired_test_instance",
    ]
    assert bootstrap["design"]["replicates"] == 100
    assert bootstrap["design"]["base_seed"] == 20260815
    common = bootstrap["common_feasible_quality"]
    assert [row["count"] for row in common["per_training_seed"]] == [259, 249, 258]
    assert all(
        row["canonical_minus_sft_gap"] < 0 for row in common["per_training_seed"]
    )

    mechanism = bundle["training_mechanism"]
    assert mechanism["trajectory_rows"] == 120
    for seed in ("101", "202", "303"):
        summary = mechanism["per_seed"][seed]
        assert summary["optimizer_steps"] == 40
        assert summary["exact_k8_subsets_evaluated"] == 360
        assert summary["recovery_active_steps"] > 0
        assert summary["controller_positive_updates"] > 0
        assert summary["controller_negative_updates"] > 0
        assert summary["max_centering_residual"] < 1e-5
        assert summary["max_decomposition_residual"] < 1e-5
