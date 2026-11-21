#!/usr/bin/env python3
"""Paired comparison of two exact CLASP evaluator JSONL artifacts."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("semantic_instance_sha256") or row.get("instance_sha256") or "")
            if not key:
                raise ValueError(f"{path}:{line_number}: missing instance identity")
            if key in records:
                raise ValueError(f"{path}:{line_number}: duplicate instance identity {key}")
            records[key] = row
    if not records:
        raise ValueError(f"empty details artifact: {path}")
    return records


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [math.nan, math.nan]
    p = successes / total
    scale = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / scale
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / scale
    return [center - radius, center + radius]


def exact_mcnemar_p(fail_to_pass: int, pass_to_fail: int) -> float:
    discordant = fail_to_pass + pass_to_fail
    if discordant == 0:
        return 1.0
    tail = min(fail_to_pass, pass_to_fail)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(
    values: list[float], *, seed: int, draws: int
) -> list[float | None]:
    if not values:
        return [None, None]
    generator = random.Random(seed)
    count = len(values)
    means = [
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(draws)
    ]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def feasible(row: dict) -> bool:
    return bool(row.get("best_of_n_feasible"))


def gap(row: dict) -> float | None:
    value = row.get("best_feasible_gap")
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite best_feasible_gap")
    return result


def aligned_quality(row: dict, slope: float) -> float:
    value = gap(row)
    return math.exp(-slope * value) if feasible(row) and value is not None else 0.0


def aggregate(rows: list[dict], slope: float) -> dict:
    total = len(rows)
    successes = sum(feasible(row) for row in rows)
    gaps = [value for row in rows if (value := gap(row)) is not None and feasible(row)]
    candidate_count = sum(len(row.get("candidates") or []) for row in rows)
    feasible_candidates = sum(int(row.get("feasible_candidates") or 0) for row in rows)
    qualities = [aligned_quality(row, slope) for row in rows]
    return {
        "instances": total,
        "best_of_n_feasible": successes,
        "best_of_n_feasibility_rate": successes / total,
        "best_of_n_feasibility_wilson95": wilson(successes, total),
        "candidates": candidate_count,
        "feasible_candidates": feasible_candidates,
        "candidate_feasibility_rate": feasible_candidates / candidate_count,
        "conditional_gap_count": len(gaps),
        "conditional_gap_mean": statistics.fmean(gaps) if gaps else None,
        "conditional_gap_median": statistics.median(gaps) if gaps else None,
        "unconditional_aligned_quality_mean": statistics.fmean(qualities),
    }


def size_band(task: str, instance_size: int) -> str:
    if task == "jssp":
        limits = (100, 200, 300)
    else:
        limits = (25, 50, 75)
    if instance_size <= limits[0]:
        return f"<={limits[0]}"
    if instance_size <= limits[1]:
        return f"{limits[0] + 1}-{limits[1]}"
    if instance_size <= limits[2]:
        return f"{limits[1] + 1}-{limits[2]}"
    return f">{limits[2]}"


def paired_summary(before_rows: list[dict], after_rows: list[dict], slope: float) -> dict:
    transitions = {"fail_fail": 0, "fail_pass": 0, "pass_fail": 0, "pass_pass": 0}
    candidate_transitions = {"fail_fail": 0, "fail_pass": 0, "pass_fail": 0, "pass_pass": 0}
    exact_completion_matches = 0
    exact_completion_groups = 0
    candidate_total = 0
    quality_deltas: list[float] = []
    joint_gap_deltas: list[float] = []
    for first, second in zip(before_rows, after_rows, strict=True):
        first_ok, second_ok = feasible(first), feasible(second)
        transitions[("pass" if first_ok else "fail") + "_" + ("pass" if second_ok else "fail")] += 1
        quality_deltas.append(aligned_quality(second, slope) - aligned_quality(first, slope))
        first_gap, second_gap = gap(first), gap(second)
        if first_ok and second_ok and first_gap is not None and second_gap is not None:
            joint_gap_deltas.append(second_gap - first_gap)
        first_candidates = first.get("candidates") or []
        second_candidates = second.get("candidates") or []
        if len(first_candidates) != len(second_candidates):
            raise ValueError("paired rows have different candidate counts")
        group_exact = True
        for first_candidate, second_candidate in zip(
            first_candidates, second_candidates, strict=True
        ):
            if first_candidate.get("candidate_index") != second_candidate.get("candidate_index"):
                raise ValueError("paired candidate indices differ")
            first_candidate_ok = bool(first_candidate.get("is_feasible"))
            second_candidate_ok = bool(second_candidate.get("is_feasible"))
            candidate_transitions[
                ("pass" if first_candidate_ok else "fail")
                + "_"
                + ("pass" if second_candidate_ok else "fail")
            ] += 1
            completion_exact = first_candidate.get("completion") == second_candidate.get("completion")
            exact_completion_matches += int(completion_exact)
            group_exact = group_exact and completion_exact
            candidate_total += 1
        exact_completion_groups += int(group_exact)
    return {
        "feasibility_transitions": transitions,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(
            transitions["fail_pass"], transitions["pass_fail"]
        ),
        "candidate_feasibility_transitions": candidate_transitions,
        "candidate_count": candidate_total,
        "exact_completion_matches": exact_completion_matches,
        "exact_completion_match_rate": exact_completion_matches / candidate_total,
        "all_completions_exact_groups": exact_completion_groups,
        "quality_deltas": quality_deltas,
        "joint_gap_deltas": joint_gap_deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--gap-slope", type=float, default=3.0)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = parser.parse_args()
    if not math.isfinite(args.gap_slope) or args.gap_slope <= 0:
        parser.error("--gap-slope must be finite and positive")
    if args.bootstrap_draws <= 0:
        parser.error("--bootstrap-draws must be positive")

    before = load(args.before)
    after = load(args.after)
    if set(before) != set(after):
        missing_after = sorted(set(before) - set(after))
        missing_before = sorted(set(after) - set(before))
        raise ValueError(
            "instance sets differ: "
            f"missing_after={missing_after[:5]}, missing_before={missing_before[:5]}"
        )

    keys = sorted(before)
    before_rows = [before[key] for key in keys]
    after_rows = [after[key] for key in keys]
    for key, first, second in zip(keys, before_rows, after_rows, strict=True):
        identity_fields = (
            "task",
            "split",
            "example_index",
            "source_row_locator",
            "benchmark_row_sha256",
            "instance_sha256",
            "semantic_instance_sha256",
            "transformed_example_sha256",
            "reference_output_sha256",
            "instance_size",
            "input_problem",
            "instruction",
            "ground_truth_objective",
            "num_return_sequences",
            "sampling_seed",
            "prompt_tokens",
            "prompt_was_truncated",
        )
        drift = {field: [first.get(field), second.get(field)] for field in identity_fields if first.get(field) != second.get(field)}
        if drift:
            raise ValueError(f"paired protocol/row drift for {key}: {drift}")

    before_aggregate = aggregate(before_rows, args.gap_slope)
    after_aggregate = aggregate(after_rows, args.gap_slope)
    paired = paired_summary(before_rows, after_rows, args.gap_slope)
    quality_deltas = paired.pop("quality_deltas")
    joint_gap_deltas = paired.pop("joint_gap_deltas")

    task = str(before_rows[0].get("task"))
    bands: dict[str, dict] = {}
    labels = [size_band(task, int(row["instance_size"])) for row in before_rows]
    for label in dict.fromkeys(labels):
        indices = [index for index, value in enumerate(labels) if value == label]
        first_band = [before_rows[index] for index in indices]
        second_band = [after_rows[index] for index in indices]
        first_aggregate = aggregate(first_band, args.gap_slope)
        second_aggregate = aggregate(second_band, args.gap_slope)
        band_paired = paired_summary(first_band, second_band, args.gap_slope)
        band_paired.pop("quality_deltas")
        band_paired.pop("joint_gap_deltas")
        bands[label] = {
            "before": first_aggregate,
            "after": second_aggregate,
            "candidate_feasibility_rate_delta": (
                second_aggregate["candidate_feasibility_rate"]
                - first_aggregate["candidate_feasibility_rate"]
            ),
            "best_of_n_feasibility_rate_delta": (
                second_aggregate["best_of_n_feasibility_rate"]
                - first_aggregate["best_of_n_feasibility_rate"]
            ),
            "unconditional_aligned_quality_mean_delta": (
                second_aggregate["unconditional_aligned_quality_mean"]
                - first_aggregate["unconditional_aligned_quality_mean"]
            ),
            "paired_feasibility": band_paired,
        }

    result = {
        "protocol": "clasp-paired-checkpoint-evaluation-comparison-v1",
        "before_path": str(args.before.resolve()),
        "after_path": str(args.after.resolve()),
        "task": task,
        "num_paired_instances": len(keys),
        "gap_slope": args.gap_slope,
        "before": before_aggregate,
        "after": after_aggregate,
        "paired": {
            **paired,
            "candidate_feasibility_rate_delta": (
                after_aggregate["candidate_feasibility_rate"]
                - before_aggregate["candidate_feasibility_rate"]
            ),
            "best_of_n_feasibility_rate_delta": (
                after_aggregate["best_of_n_feasibility_rate"]
                - before_aggregate["best_of_n_feasibility_rate"]
            ),
            "unconditional_aligned_quality_mean_delta": statistics.fmean(quality_deltas),
            "unconditional_aligned_quality_mean_delta_bootstrap95": bootstrap_mean_ci(
                quality_deltas, seed=args.bootstrap_seed, draws=args.bootstrap_draws
            ),
            "jointly_feasible_instances": len(joint_gap_deltas),
            "jointly_feasible_gap_delta_mean": (
                statistics.fmean(joint_gap_deltas) if joint_gap_deltas else None
            ),
            "jointly_feasible_gap_delta_median": (
                statistics.median(joint_gap_deltas) if joint_gap_deltas else None
            ),
            "jointly_feasible_gap_delta_mean_bootstrap95": bootstrap_mean_ci(
                joint_gap_deltas,
                seed=args.bootstrap_seed + 1,
                draws=args.bootstrap_draws,
            ),
        },
        "size_bands": bands,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
