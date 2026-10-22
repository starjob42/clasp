from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_sealed_additional_evidence_recomputes_expected_diagnostics(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "additional_evidence"
    subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/analyze_additional_evidence.py"),
            "--output-dir",
            str(output),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(
        (output / "paired_tsp_dense40_diagnostics.json").read_text(encoding="utf-8")
    )
    assert payload["evidence_scope"] == "development_only_not_publication_eligible"
    assert payload["pairing"]["instances"] == 200
    assert payload["paired_prefix_metrics"]["8"]["sft"]["pass_at_k"] == 1.0
    assert payload["paired_prefix_metrics"]["8"]["dense40"]["pass_at_k"] == 1.0
    assert (
        payload["paired_prefix_metrics"]["8"]["common_feasible_gap"]["right_minus_left"]
        > 0
    )
    training = payload["training_diagnostics"]
    assert training["observed_groups"] == 160
    assert training["group_exposure"]["failed_groups"] == 0
    assert training["set_failure_controller"]["lambda_max"] == 0.0
    assert training["credit_diagnostics"]["candidate_credits_nonzero"] == 1250

    exposure = payload["historical_sft_subsampled_failure_exposure"]["systems"]
    assert (
        exposure["sft100k_unified_attn_137500::cvrp"]
        ["expected_failure_of_uniform_random_k_subset"]["8"]
        == 0.33
    )
    assert (
        exposure["sft500k_specialist_mlp_tsp_31250::tsp"]
        ["expected_failure_of_uniform_random_k_subset"]["8"]
        == 0.0
    )

