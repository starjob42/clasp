#!/usr/bin/env python3
"""Generate claim-scoped ICLR mechanism tables from immutable local records.

This generator intentionally excludes the confirmatory five-task table.  It
only consumes the exact categorical-bandit bundle and sealed development-only
pilot records.  The small-TSP comparison is emitted into a guarded appendix
hook and cannot populate any claim-bearing result table.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper" / "iclr2027" / "generated"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _ratio_row(
    rows: list[dict[str, str]], method: str, context: str
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["method"] == method and row["context"] == context
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one variance row for method={method}, context={context}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _robustness_value(
    rows: list[dict[str, str]],
    *,
    stress_test: str,
    context: str,
    level: float,
    method: str,
    metric: str,
) -> float:
    matches = [
        row
        for row in rows
        if row["stress_test"] == stress_test
        and row["context"] == context
        and math.isclose(float(row["level"]), level, rel_tol=0.0, abs_tol=1e-12)
        and row["method"] == method
        and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one robustness row for "
            f"{stress_test}/{context}/{level}/{method}/{metric}; found {len(matches)}"
        )
    return float(matches[0]["mean"])


def _scientific_tex(value: float, digits: int = 2) -> str:
    if value == 0.0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10.0**exponent)
    return rf"${mantissa:.{digits}f}\!\times\!10^{{{exponent}}}$"


def generate_bandit_table() -> None:
    result_root = ROOT / "results" / "canonical_bandit"
    metadata_path = result_root / "metadata.json"
    exact_path = result_root / "exact_metrics.csv"
    variance_path = result_root / "paired_variance_summary.csv"
    robustness_path = result_root / "robustness_summary.csv"

    metadata = _read_json(metadata_path)
    if metadata.get("scope") != "synthetic_mechanistic_only":
        raise RuntimeError("Refusing to relabel a non-mechanistic artifact as synthetic")
    if len(metadata.get("policy_seeds", [])) != 12:
        raise RuntimeError("Expected 12 frozen policy perturbations")
    if len(metadata.get("settings", [])) != 6:
        raise RuntimeError("Expected six K/B settings")

    exact = _read_csv(exact_path)
    variance = _read_csv(variance_path)
    robustness = _read_csv(robustness_path)
    nonzero_target = [
        row
        for row in exact
        if float(row["true_gradient_l2"]) > 1e-12
    ]
    max_relative_error = {
        method: max(
            float(row["relative_bias_l2"])
            for row in nonzero_target
            if row["method"] == method
        )
        for method in ("canonical", "fixed_null", "raw", "oracle_constant")
    }
    max_centering = max(
        float(row["centering_residual_rms"])
        for row in exact
        if row["method"] == "canonical"
    )
    naive_biases = [
        float(row["relative_bias_l2"])
        for row in nonzero_target
        if row["method"] == "naive_centered"
    ]

    tied_fixed_variances = [
        float(row["total_variance"])
        for row in exact
        if row["context"] == "all_infeasible_tied"
        and row["method"] == "fixed_null"
    ]
    tied_canonical_variances = [
        float(row["total_variance"])
        for row in exact
        if row["context"] == "all_infeasible_tied"
        and row["method"] == "canonical"
    ]
    if not tied_fixed_variances or max(map(abs, tied_canonical_variances)) > 1e-13:
        raise RuntimeError("Tied action-independent zero-variance identity did not hold")
    tied_fixed_median = statistics.median(tied_fixed_variances)

    ratio_specs = [
        ("Fixed-null / canonical", "ALL_NONDEGENERATE", "all nondegenerate"),
        ("Fixed-null / canonical", "all_infeasible_ordered", "ordered all-infeasible"),
        ("Fixed-null / canonical", "rare_feasible", "rare feasible"),
        ("Fixed-null / canonical", "tied_feasible", "tied feasible quality"),
        ("Fixed-null / canonical", "mixed_quality", "mixed feasible quality"),
        ("Raw REINFORCE / canonical", "ALL_NONDEGENERATE", "all nondegenerate"),
    ]
    ratio_lines: list[str] = []
    for display_method, context, display_context in ratio_specs:
        method = "raw" if display_method.startswith("Raw") else "fixed_null"
        row = _ratio_row(variance, method, context)
        ratio_lines.append(
            f"    {display_method} & {display_context} & "
            f"{float(row['geometric_mean_ratio']):.3f} "
            f"[{float(row['cluster_bootstrap_95_low']):.3f}, "
            f"{float(row['cluster_bootstrap_95_high']):.3f}] \\\\"
        )

    noise_mixed = _robustness_value(
        robustness,
        stress_test="candidate_observation_noise",
        context="mixed_quality",
        level=0.2,
        method="canonical",
        metric="relative_bias_l2",
    )
    noise_ordered = _robustness_value(
        robustness,
        stress_test="candidate_observation_noise",
        context="all_infeasible_ordered",
        level=0.2,
        method="canonical",
        metric="relative_bias_l2",
    )
    dropout_levels = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75)
    dropout_errors = [
        _robustness_value(
            robustness,
            stress_test="dense_component_mask_dropout",
            context="all_infeasible_ordered",
            level=level,
            method="canonical",
            metric="relative_bias_l2",
        )
        for level in dropout_levels
    ]
    dropout_cosine_75 = _robustness_value(
        robustness,
        stress_test="dense_component_mask_dropout",
        context="all_infeasible_ordered",
        level=0.75,
        method="canonical",
        metric="gradient_cosine",
    )

    hashes = ", ".join(
        f"{path.name}={_sha256(path)[:12]}"
        for path in (metadata_path, exact_path, variance_path, robustness_path)
    )
    macro_lines = [
        "% AUTO-GENERATED by scripts/generate_iclr_mechanism_tables.py.",
        f"% Input hashes (12 hex): {hashes}",
        rf"\providecommand{{\BanditCanonicalMaxRelativeError}}{{{_scientific_tex(max_relative_error['canonical'])}}}",
        rf"\providecommand{{\BanditFixedNullMaxRelativeError}}{{{_scientific_tex(max_relative_error['fixed_null'])}}}",
        rf"\providecommand{{\BanditRawMaxRelativeError}}{{{_scientific_tex(max_relative_error['raw'])}}}",
        rf"\providecommand{{\BanditOracleMaxRelativeError}}{{{_scientific_tex(max_relative_error['oracle_constant'])}}}",
        rf"\providecommand{{\BanditCanonicalMaxCenteringRMS}}{{{_scientific_tex(max_centering)}}}",
        rf"\providecommand{{\BanditNaiveBiasRange}}{{{min(naive_biases):.2f}--{max(naive_biases):.2f}}}",
        rf"\providecommand{{\BanditTiedFixedVariance}}{{{_scientific_tex(tied_fixed_median)}}}",
        rf"\providecommand{{\BanditNoiseMixedError}}{{{noise_mixed:.3f}}}",
        rf"\providecommand{{\BanditNoiseOrderedError}}{{{noise_ordered:.3f}}}",
        rf"\providecommand{{\BanditDropoutErrors}}{{{', '.join(f'{value:.3f}' for value in dropout_errors)}}}",
        rf"\providecommand{{\BanditDropoutCosineSeventyFive}}{{{dropout_cosine_75:.3f}}}",
        "",
    ]
    lines = [
        "% AUTO-GENERATED by scripts/generate_iclr_mechanism_tables.py.",
        f"% Input hashes (12 hex): {hashes}",
        r"\input{generated/canonical_bandit_macros}",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{\\textbf{Synthetic mechanism experiment; no LLM is trained or evaluated.} "
        "Exact paired total-gradient variance ratios over 12 frozen policy perturbations "
        "and six $K/B$ settings. Ratios below one favor the named numerator; intervals "
        "are policy-seed cluster bootstraps and are not benchmark confidence intervals.}",
        "  \\label{tab:synthetic-mechanism}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{lll}",
        "    \\toprule",
        "    Estimator ratio & Context & Geometric mean [95\\% interval] \\\\",
        "    \\midrule",
        *ratio_lines,
        "    \\bottomrule",
        "  \\end{tabular}}",
        "\\end{table}",
        "",
    ]
    _atomic_write(OUTPUT_DIR / "canonical_bandit_macros.tex", "\n".join(macro_lines))
    _atomic_write(OUTPUT_DIR / "canonical_bandit_mechanism.tex", "\n".join(lines))


def _canonical_training_rows() -> tuple[list[dict[str, Any]], list[Path]]:
    experiment_root = (
        ROOT / "experiments" / "additional_evidence" / "small_tsp_canonical_v1"
    )
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for seed in (101, 202, 303):
        run_root = experiment_root / f"canonical_b9k8_seed{seed}_steps40_v1"
        status = _read_json(run_root / "stage_status.json")
        if not status.get("completed") or int(status.get("global_step", -1)) != 40:
            raise RuntimeError(f"Canonical development run {seed} is not a complete 40-step run")
        log_path = run_root / "grpo_log_history.json"
        records = [
            row
            for row in _read_json(log_path)
            if "constrained_grpo/clasp/canonical_set" in row
        ]
        if len(records) != 40:
            raise RuntimeError(f"Expected 40 canonical telemetry rows for seed {seed}")

        key = "constrained_grpo/clasp/"
        row = {
            "seed": seed,
            "steps": len(records),
            "all_failed": sum(
                item[f"{key}canonical_failure_at_k"] == 1 for item in records
            ),
            "k_not_b": sum(
                item[f"{key}canonical_failure_k_minus_failure_b"] > 0
                for item in records
            ),
            "recovery": sum(
                item[f"{key}canonical_recovery_coefficient_abs_mean"] > 0
                for item in records
            ),
            "failure_credit": sum(
                item[f"{key}canonical_failure_coefficient_abs_mean"] > 0
                for item in records
            ),
            "zero_credit": sum(
                item[f"{key}canonical_coefficient_abs_mean"] == 0
                for item in records
            ),
            "increase": sum(
                item[f"{key}bon_committed_actual_delta/tsp"] > 0
                for item in records
            ),
            "decrease": sum(
                item[f"{key}bon_committed_actual_delta/tsp"] < 0
                for item in records
            ),
            "max_lambda": max(
                item[f"{key}bon_committed_lambda_next/tsp"] for item in records
            ),
            "max_centering": max(
                item[f"{key}canonical_advantage_centering_residual_abs_max"]
                for item in records
            ),
            "max_decomposition": max(
                item[f"{key}canonical_component_decomposition_residual_abs_max"]
                for item in records
            ),
            "max_subset_quality_mean": max(
                item[f"{key}canonical_subset_quality_mean"] for item in records
            ),
        }
        rows.append(row)
        paths.append(log_path)
    return rows, paths


def generate_canonical_training_table() -> None:
    rows, paths = _canonical_training_rows()
    totals: dict[str, Any] = {
        "seed": "All",
        "steps": sum(row["steps"] for row in rows),
        "all_failed": sum(row["all_failed"] for row in rows),
        "k_not_b": sum(row["k_not_b"] for row in rows),
        "recovery": sum(row["recovery"] for row in rows),
        "failure_credit": sum(row["failure_credit"] for row in rows),
        "zero_credit": sum(row["zero_credit"] for row in rows),
        "increase": sum(row["increase"] for row in rows),
        "decrease": sum(row["decrease"] for row in rows),
        "max_lambda": max(row["max_lambda"] for row in rows),
        "max_centering": max(row["max_centering"] for row in rows),
        "max_decomposition": max(row["max_decomposition"] for row in rows),
        "max_subset_quality_mean": max(
            row["max_subset_quality_mean"] for row in rows
        ),
    }
    hashes = ", ".join(f"seed{row['seed']}={_sha256(path)[:12]}" for row, path in zip(rows, paths))
    body: list[str] = []
    for row in [*rows, totals]:
        body.append(
            f"    {row['seed']} & {row['steps']} & {row['all_failed']} & "
            f"{row['k_not_b']} & {row['recovery']} & {row['increase']} & "
            f"{row['decrease']} & {row['max_lambda']:.3f} \\\\"
        )
    lines = [
        "% AUTO-GENERATED by scripts/generate_iclr_mechanism_tables.py.",
        f"% Training-log hashes (12 hex): {hashes}",
        rf"\providecommand{{\CanonicalPilotMaxCenteringResidual}}{{{_scientific_tex(totals['max_centering'])}}}",
        rf"\providecommand{{\CanonicalPilotMaxDecompositionResidual}}{{{_scientific_tex(totals['max_decomposition'])}}}",
        rf"\providecommand{{\CanonicalPilotMaxSubsetQualityMean}}{{{totals['max_subset_quality_mean']:.3f}}}",
        rf"\providecommand{{\CanonicalPilotFailureCreditSteps}}{{{totals['failure_credit']}}}",
        rf"\providecommand{{\CanonicalPilotZeroCreditSteps}}{{{totals['zero_credit']}}}",
        "\\begin{table}[h]",
        "  \\centering",
        "  \\caption{\\textbf{Development-only canonical-training mechanism smoke.} "
        "Each Qwen2.5-0.5B run uses 40 optimizer steps with $B=9$, $K=8$. "
        "``All failed'' means that all nine rollout candidates are infeasible. "
        "``$K\\ne B$'' counts one-feasible groups for which failure@8 is $1/9$ "
        "while full-batch failure is zero. This table is not an efficacy comparison.}",
        "  \\label{tab:canonical-training-smoke}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{lrrrrrrr}",
        "    \\toprule",
        "    Seed & Steps & All failed & $K\\ne B$ & Recovery active & $\\lambda\\uparrow$ & $\\lambda\\downarrow$ & Max. $\\lambda$ \\\\",
        "    \\midrule",
        *body[:3],
        "    \\midrule",
        body[3],
        "    \\bottomrule",
        "  \\end{tabular}}",
        "\\end{table}",
        "",
    ]
    _atomic_write(OUTPUT_DIR / "canonical_training_diagnostics.tex", "\n".join(lines))


def generate_legacy_negative_pilot_table() -> None:
    source = ROOT / "results" / "additional_evidence" / "paired_tsp_dense40_diagnostics.json"
    payload = _read_json(source)
    if payload.get("evidence_scope") != "development_only_not_publication_eligible":
        raise RuntimeError("Legacy pilot is missing its development-only evidence boundary")
    if payload.get("comparison_direction") != "dense40_minus_sft":
        raise RuntimeError("Unexpected legacy-pilot comparison direction")
    rows: list[str] = []
    for budget in (1, 2, 4, 8):
        record = payload["paired_prefix_metrics"][str(budget)]
        gap = record["common_feasible_gap"]
        interval = gap["paired_bootstrap_95pct_ci"]
        rows.append(
            f"    {budget} & {100.0 * record['sft']['pass_at_k']:.1f} & "
            f"{100.0 * record['dense40']['pass_at_k']:.1f} & "
            f"{100.0 * gap['right_minus_left']:+.4f} "
            f"[{100.0 * interval['lower']:+.4f}, {100.0 * interval['upper']:+.4f}] \\\\"
        )
    k8 = payload["paired_prefix_metrics"]["8"]
    training = payload["training_diagnostics"]
    lines = [
        "% AUTO-GENERATED by scripts/generate_iclr_mechanism_tables.py.",
        f"% Input sha256: {_sha256(source)}",
        rf"\providecommand{{\LegacyPilotSuccessTwoSFT}}{{{100.0 * k8['sft']['success_at_gap_threshold']['0.0200']:.1f}\%}}",
        rf"\providecommand{{\LegacyPilotSuccessTwoDense}}{{{100.0 * k8['dense40']['success_at_gap_threshold']['0.0200']:.1f}\%}}",
        rf"\providecommand{{\LegacyPilotSuccessTwoP}}{{{k8['success_at_gap_threshold']['0.0200']['exact_mcnemar']['exact_two_sided_p']:.4f}}}",
        rf"\providecommand{{\LegacyPilotObservedGroups}}{{{training['observed_groups']}}}",
        rf"\providecommand{{\LegacyPilotFailedGroups}}{{{training['group_exposure']['failed_groups']}}}",
        "\\begin{table}[h]",
        "  \\centering",
        "  \\caption{\\textbf{Negative historical TSP pilot (development only; retired "
        "dense-rank method, not canonical \\method).} Gap is dense40 minus SFT on "
        "common-feasible instances, in percentage points; positive is worse. Smaller "
        "$K$ values reuse prefixes of the saved eight draws. Intervals resample 200 "
        "instances and do not include training-seed uncertainty.}",
        "  \\label{tab:legacy-tsp-negative-pilot}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{rrrr}",
        "    \\toprule",
        "    $K$ & SFT pass@$K$ (\\%) & Dense40 pass@$K$ (\\%) & Gap delta [95\\% interval] (pp) \\\\",
        "    \\midrule",
        *rows,
        "    \\bottomrule",
        "  \\end{tabular}}",
        "\\end{table}",
        "",
    ]
    _atomic_write(OUTPUT_DIR / "legacy_tsp_negative_pilot.tex", "\n".join(lines))


def _signed(value: float, digits: int) -> str:
    if abs(value) < 0.5 * (10.0 ** (-digits)):
        value = 0.0
    return f"{value:+.{digits}f}" if value != 0.0 else f"{value:.{digits}f}"


def generate_development_small_tsp_table() -> None:
    source = (
        ROOT
        / "results"
        / "additional_evidence"
        / "small_tsp_canonical_pilot_v1"
        / "analysis.json"
    )
    payload = _read_json(source)
    if payload.get("schema_version") != 1:
        raise RuntimeError("Unexpected small-TSP pilot schema version")
    if payload.get("evidence_scope") != "development_only_not_publication_or_sota_evidence":
        raise RuntimeError("Small-TSP pilot is missing its development-only scope")

    experiment = payload.get("experiment", {})
    expected_experiment = {
        "deployment_set_K": 8,
        "rollout_batch_B": 9,
        "exact_subsets_per_instance": 9,
        "held_out_instances": 31,
        "training_seeds": [101, 202, 303],
        "training_updates": 40,
    }
    for key, expected in expected_experiment.items():
        if experiment.get(key) != expected:
            raise RuntimeError(
                f"Unexpected small-TSP pilot field {key}: "
                f"{experiment.get(key)!r} != {expected!r}"
            )
    validation = payload.get("validation", {})
    required_checks = (
        "actual_prompt_text_and_transformed_hash_matched",
        "candidate_positions_paired_by_common_per_example_sampling_seed",
        "instances_exactly_paired",
        "strict_verifier_manifest_matched",
    )
    if not all(validation.get(key) is True for key in required_checks):
        raise RuntimeError("Small-TSP pairing/verifier validation is incomplete")

    bootstrap = payload["bootstrap"]
    design = bootstrap["design"]
    if (
        design.get("replicates") != 5000
        or design.get("leave_one_out_subsets")
        != "exactly enumerated; never resampled"
    ):
        raise RuntimeError("Unexpected small-TSP bootstrap design")

    sft = bootstrap["sft"]
    canonical = bootstrap["canonical"]
    delta = bootstrap["canonical_minus_sft"]
    common = bootstrap["common_feasible_quality"]

    rate_specs = (
        ("Pass@8", "pass_at_8"),
        ("Candidate feasibility", "candidate_feasibility"),
        ("Success@2\\% gap", "success_gap_le_0.02"),
        ("Success@5\\% gap", "success_gap_le_0.05"),
        ("Success@10\\% gap", "success_gap_le_0.10"),
    )
    rows: list[str] = []
    for label, key in rate_specs:
        sft_point = 100.0 * float(sft[key]["point"])
        can_point = 100.0 * float(canonical[key]["point_mean_across_training_seeds"])
        can_sd = 100.0 * float(canonical[key]["seed_sample_sd"])
        effect = 100.0 * float(delta[key]["point"])
        low = 100.0 * float(delta[key]["interval"]["lower"])
        high = 100.0 * float(delta[key]["interval"]["upper"])
        rows.append(
            f"    {label} (\\%) & {sft_point:.2f} & {can_point:.2f} ({can_sd:.2f}) & "
            f"{_signed(effect, 2)} [{_signed(low, 2)}, {_signed(high, 2)}] pp \\\\"
        )

    sft_gap = common["sft_gap"]
    canonical_gap = common["canonical_gap"]
    gap_delta = common["canonical_minus_sft_gap"]
    coverage = float(common["coverage"]["point_mean_across_training_seeds"])
    rows.append(
        "    Common-feasible gap & "
        f"{float(sft_gap['point_mean_across_training_seeds']):.4f} & "
        f"{float(canonical_gap['point_mean_across_training_seeds']):.4f} "
        f"({float(canonical_gap['seed_sample_sd']):.4f}) & "
        f"{_signed(float(gap_delta['point_mean_across_training_seeds']), 4)} "
        f"[{_signed(float(gap_delta['interval']['lower']), 4)}, "
        f"{_signed(float(gap_delta['interval']['upper']), 4)}] \\\\"
    )

    pass_delta = delta["pass_at_8"]
    candidate_delta = delta["candidate_feasibility"]
    lines = [
        "% AUTO-GENERATED by scripts/generate_iclr_mechanism_tables.py.",
        f"% Input sha256: {_sha256(source)}",
        rf"\providecommand{{\DevPilotPassSFT}}{{{100.0 * float(sft['pass_at_8']['point']):.2f}\%}}",
        rf"\providecommand{{\DevPilotPassCanonical}}{{{100.0 * float(canonical['pass_at_8']['point_mean_across_training_seeds']):.2f}\%}}",
        rf"\providecommand{{\DevPilotPassDelta}}{{{_signed(100.0 * float(pass_delta['point']), 2)} percentage points}}",
        rf"\providecommand{{\DevPilotPassDeltaCI}}{{[{_signed(100.0 * float(pass_delta['interval']['lower']), 2)}, {_signed(100.0 * float(pass_delta['interval']['upper']), 2)}] percentage points}}",
        rf"\providecommand{{\DevPilotCandidateDelta}}{{{_signed(100.0 * float(candidate_delta['point']), 2)} percentage points}}",
        rf"\providecommand{{\DevPilotCandidateDeltaCI}}{{[{_signed(100.0 * float(candidate_delta['interval']['lower']), 2)}, {_signed(100.0 * float(candidate_delta['interval']['upper']), 2)}] percentage points}}",
        rf"\providecommand{{\DevPilotCommonCoverage}}{{{100.0 * coverage:.2f}\%}}",
        rf"\providecommand{{\DevPilotGapDelta}}{{{_signed(float(gap_delta['point_mean_across_training_seeds']), 4)}}}",
        rf"\providecommand{{\DevPilotGapDeltaCI}}{{[{_signed(float(gap_delta['interval']['lower']), 4)}, {_signed(float(gap_delta['interval']['upper']), 4)}]}}",
        "\\begin{table}[h]",
        "  \\centering",
        "  \\caption{\\textbf{Held-out small-TSP development pilot; not publication evidence.} "
        "Qwen2.5-0.5B, 31 development instances, 40 updates, and three canonical "
        "training seeds. The primary $K=8$ analysis exactly enumerates the nine "
        "leave-one-out subsets of each $B=9$ draw (279 subset outcomes per system/seed). "
        "Canonical entries are mean (training-seed sample SD). Rate deltas are percentage "
        "points; gap is a normalized reference-gap fraction. Intervals use 5,000 "
        "two-way crossed training-seed by paired-instance bootstrap replicates; exact "
        "subsets are not resampled. Common-feasible coverage is "
        f"{100.0 * coverage:.2f}\\%.}}",
        "  \\label{tab:development-small-tsp-pilot}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{lrrr}",
        "    \\toprule",
        "    Metric & SFT & Canonical mean (SD) & Canonical $-$ SFT [95\\% interval] \\\\ ",
        "    \\midrule",
        *rows,
        "    \\bottomrule",
        "  \\end{tabular}}",
        "\\end{table}",
        "",
    ]
    _atomic_write(OUTPUT_DIR / "development_small_tsp_pilot.tex", "\n".join(lines))


def main() -> None:
    generate_bandit_table()
    generate_canonical_training_table()
    generate_legacy_negative_pilot_table()
    generate_development_small_tsp_table()


if __name__ == "__main__":
    main()
