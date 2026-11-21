#!/usr/bin/env python3
"""Recompute the complete, predeclared fresh CVRP/MIS development comparison.

Consumes eight immutable result indices (SFT, GRPO, CLASP v3/v4 per task).
The existing strict comparator validates raw artifacts and exact pairing.
Table rates and gaps are percentages; strict comparison JSON retains fractions.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("fresh_strict_comparator", ROOT / "scripts/compare_development.py")
strict = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = strict
SPEC.loader.exec_module(strict)
TASKS = ("cvrp", "mis")
METHODS = ("sft", "grpo_z", "v3", "v4")
LABELS = {"sft": "SFT initialization", "grpo_z": "Classic GRPO", "v3": "CLASP v3", "v4": "CLASP v4"}


def read_signed(path, field):
    value = json.loads(Path(path).read_text())
    strict._validate_self_hash(value, field, label=str(path))
    return value


def check_identity(value):
    snapshot = strict._read_binary_snapshot(value["path"], label=value["path"])
    if snapshot.sha256 != value["sha256"] or snapshot.size_bytes != value["size_bytes"]:
        raise ValueError("artifact identity mismatch: " + value["path"])
    return snapshot


def quality(record):
    return math.exp(-3 * max(float(record.best_feasible_gap), 0.0)) if record.pass_at_k else 0.0


def signed_gap(record, task):
    if not record.pass_at_k:
        return None
    objective, reference = (float(record.raw[key]) for key in ("best_objective_value", "ground_truth_objective"))
    if not math.isfinite(objective) or not math.isfinite(reference) or reference <= 0:
        raise ValueError("invalid objective/reference for supplementary signed gap")
    gap = (objective - reference) / reference if task == "cvrp" else (reference - objective) / reference
    if not math.isclose(max(0.0, gap), record.best_feasible_gap, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("signed objective gap disagrees with the evaluator's clipped gap")
    return gap


def summarize(records, task, method, band="all"):
    raw = [record.raw for record in records]
    candidates = [candidate for row in raw for candidate in row["candidates"]]
    gaps = [record.best_feasible_gap for record in records if record.pass_at_k]
    signed_gaps = [signed_gap(record, task) for record in records if record.pass_at_k]
    return {
        "task": task, "method": method, "size_band": band, "instances": len(records),
        "candidate_count": len(candidates), "feasible_at_8_count": len(gaps),
        "feasible_at_8_percent": 100 * len(gaps) / len(records),
        "candidate_feasibility_percent": 100 * statistics.fmean(r.candidate_feasibility for r in records),
        "mean_feasible_gap_percent": 100 * statistics.fmean(gaps) if gaps else None,
        "mean_signed_reference_gap_percent": 100 * statistics.fmean(signed_gaps) if signed_gaps else None,
        "better_than_reference_instances": sum(gap < -1e-12 for gap in signed_gaps),
        "unconditional_solved_quality": statistics.fmean(quality(r) for r in records),
        "completion_tokens": sum(c["completion_tokens"] for c in candidates),
        "parse_failure_candidates": sum(bool(c["parse_failure"]) for c in candidates),
        "truncated_prompts": sum(bool(row["prompt_was_truncated"]) for row in raw),
        "mean_distinct_completions": statistics.fmean(len({c["completion"] for c in row["candidates"]}) for row in raw),
        "gap_reference": "verified heuristic" if task == "cvrp" else "certified optimum",
    }


def paired_quality(left, right):
    strict._validate_pairing(left, right)
    deltas = np.array([quality(right.records[k]) - quality(left.records[k]) for k in sorted(left.records)])
    rng = np.random.Generator(np.random.PCG64(20260905))
    draws = deltas[rng.integers(0, len(deltas), size=(20_000, len(deltas)))].mean(axis=1)
    return {"right_minus_left": float(deltas.mean()),
            "paired_bootstrap_95pct_ci": [float(v) for v in np.quantile(draws, [0.025, 0.975])],
            "bootstrap_replicates": 20_000, "bootstrap_seed": 20260905,
            "bootstrap_rng": "numpy.random.PCG64", "resampling_unit": "paired_instance"}


def collect(indices):
    loaded, sources, launches = {}, {}, {}
    for path in indices:
        index = read_signed(path, "index_sha256")
        key = (index["task"], index["method"])
        if key in loaded or key not in {(task, method) for task in TASKS for method in METHODS}:
            raise ValueError("unexpected or duplicate task/method: " + str(key))
        if index["status"] != "complete_fresh_development_only" or index["repair"]:
            raise ValueError("only completed raw fresh development artifacts are accepted")
        if index["candidate_budget"] != 8 or index["generation_seed"] != 271828:
            raise ValueError("fresh generation budget/seed mismatch")
        for name in ("launch_receipt", "details", "run_config"):
            check_identity(index[name])
        launch = read_signed(index["launch_receipt"]["path"], "launch_sha256")
        if (launch["task"], launch["method"]) != key:
            raise ValueError("launch/result task or method mismatch")
        artifact = strict._load_artifact(index["details"]["path"], index["run_config"]["path"], label="/".join(key))
        if len(artifact.records) != 100 or {record.task for record in artifact.records.values()} != {key[0]}:
            raise ValueError("expected exactly 100 instances of the declared task")
        loaded[key], launches[key] = artifact, launch
        sources["/".join(key)] = strict._read_binary_snapshot(str(path), label=str(path)).identity()
    expected = {(task, method) for task in TASKS for method in METHODS}
    if set(loaded) != expected:
        raise ValueError("missing predeclared arms: " + str(sorted(expected - set(loaded))))
    source_hashes = {launch["source_manifest"]["sha256"] for launch in launches.values()}
    benchmark_hashes = {launch["benchmark"]["manifest_sha256"] for launch in launches.values()}
    audit_hashes = {launch["benchmark"]["audit_sha256"] for launch in launches.values()}
    if any(len(values) != 1 for values in (source_hashes, benchmark_hashes, audit_hashes)):
        raise ValueError("methods differ in source, benchmark, or overlap audit")
    return loaded, sources, launches


def build_report(indices):
    loaded, sources, launches = collect(indices)
    rows, sizes, comparisons, sft_pairing = [], [], {}, {}
    for task in TASKS:
        for method in METHODS:
            records = list(loaded[task, method].records.values())
            rows.append(summarize(records, task, method))
            for low, high, band in ((0, 25, "<=25"), (26, 50, "26-50"), (51, 75, "51-75"), (76, 10000, ">75")):
                subset = [r for r in records if low <= r.raw["instance_size"] <= high]
                if subset:
                    sizes.append(summarize(subset, task, method, band))
        pairs = [("grpo_z", "v3"), ("grpo_z", "v4"), ("v3", "v4")]
        # SFT is always descriptive context. Do not relax any identity check to
        # obtain a paired statistic if its inference protocol differs.
        for method in METHODS[1:]:
            try:
                strict._validate_pairing(loaded[task, "sft"], loaded[task, method])
            except strict.DevelopmentComparisonError as exc:
                sft_pairing[task + "/" + method] = str(exc)
            else:
                pairs.append(("sft", method))
                sft_pairing[task + "/" + method] = "exact"
        for left, right in pairs:
            a, b = loaded[task, left], loaded[task, right]
            receipt = strict.build_receipt(a, b, left_label=left, right_label=right)
            comparisons[f"{task}/{left}_vs_{right}"] = {
                "strict_comparison": receipt, "unconditional_solved_quality": paired_quality(a, b)}
    return {"scope": "fresh_development_single_training_seed_43", "sources": sources,
            "summary": rows, "size_summary": sizes, "comparisons": comparisons,
            "sft_pairing": sft_pairing,
            "benchmark_manifest_sha256": next(iter(launches.values()))["benchmark"]["manifest_sha256"],
            "definitions": {"gap": "nonnegative reference excess gap=max(0,directional relative gap); each method's own feasible set in headline table",
                            "paired_gap": "right-minus-left on jointly feasible instances",
                            "signed_gap_secondary": "unclipped directional relative gap recomputed from the selected feasible objective and reference; descriptive only",
                            "better_than_reference": "signed relative gap < -1e-12; avoids roundoff-only improvements",
                            "size_bands": "evaluator instance_size; CVRP includes the depot",
                            "units": "CSV/Markdown rates and gaps are percentages; strict comparison JSON rates, gaps, and deltas are fractions",
                            "solved_quality": "mean(feasible * exp(-3 * max(gap, 0))); failures contribute zero",
                            "uncertainty": "paired instance bootstrap; 20000 draws; no training-seed uncertainty or multiplicity adjustment"},
            "limitations": ["One training seed and 100 development instances per task; not confirmation evidence.",
                            "CVRP references are feasible heuristics, not certified optima.",
                            "Conditional gap must be interpreted alongside feasibility; failed instances are never assigned zero gap."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    report = build_report(args.index)
    args.output.mkdir(parents=True, exist_ok=False)
    strict._write_create_once(str(args.output / "comparison.json"), report)
    for name, rows in (("table.csv", report["summary"]), ("size_table.csv", report["size_summary"])):
        with (args.output / name).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["| Task | Method | Feasible candidates ↑ | Feasible@8 ↑ | Mean feasible gap ↓ | Solved quality ↑ |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in report["summary"]:
        gap = "—" if row["mean_feasible_gap_percent"] is None else f"{row['mean_feasible_gap_percent']:.3f}%"
        lines.append(f"| {row['task'].upper()} | {LABELS[row['method']]} | {row['candidate_feasibility_percent']:.3f}% | "
                     f"{row['feasible_at_8_count']}/100 | {gap} | {row['unconditional_solved_quality']:.6f} |")
    lines += ["", "Gap is nonnegative reference excess, conditional on each method succeeding. CVRP uses heuristic references; MIS uses certified optima.",
              "Supplementary signed reference gaps and counts of better-than-reference solutions are retained in table.csv.",
              "One training seed, 100 instances/task, eight raw candidates/instance. Paired confidence intervals are in comparison.json.", ""]
    (args.output / "table.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
