#!/usr/bin/env python3
"""Seal the development-only small-TSP SFT versus canonical CLASP pilot.

The deployment budget is K=8, while evaluation generated B=9 candidates.  This
script therefore enumerates all C(9, 8)=9 leave-one-out subsets for every test
instance.  It never substitutes best-of-9 for the deployment metric.

Uncertainty is descriptive: a deterministic crossed bootstrap resamples the
three training seeds and the 31 paired test instances.  The experiment is a
0.5B, 40-update, development-only mechanism pilot, not publication/benchmark
evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

B = 9
K = 8
N_INSTANCES = 31
TRAINING_SEEDS = (101, 202, 303)
GAP_THRESHOLDS = (0.02, 0.05, 0.10)
DEFAULT_BOOTSTRAP_REPLICATES = 5_000
DEFAULT_BOOTSTRAP_SEED = 20_260_815
SCHEMA_VERSION = 1

RAW_ROOT = Path("experiments/additional_evidence/small_tsp_canonical_v1")
DEFAULT_SFT_GLOB = (
    RAW_ROOT
    / "eval_sft_b9_seed314159/sft/sft_qwen05b_seed20260815_v5/run_details*.jsonl"
)
DEFAULT_CANONICAL_GLOBS = {
    seed: RAW_ROOT
    / f"eval_canonical_b9k8_seed{seed}_steps40_b9_seed314159/grpo/"
    / f"canonical_b9k8_seed{seed}_steps40_v1/run_details*.jsonl"
    for seed in TRAINING_SEEDS
}
DEFAULT_TRAINING_LOGS = {
    seed: RAW_ROOT / f"canonical_b9k8_seed{seed}_steps40_v1/grpo_log_history.json"
    for seed in TRAINING_SEEDS
}
DEFAULT_OUTPUT = Path("results/additional_evidence/small_tsp_canonical_pilot_v1")


class AnalysisError(ValueError):
    """Raised when a raw artifact violates the sealed pilot contract."""


@dataclass(frozen=True)
class Candidate:
    feasible: bool
    gap: float | None
    completion_tokens: int


@dataclass(frozen=True)
class Record:
    example_index: int
    semantic_hash: str
    transformed_hash: str
    benchmark_hash: str
    sampling_seed: int
    instance_size: int
    ground_truth_objective: float
    prompt_tokens: int
    prompt_was_truncated: bool
    instruction: str
    input_problem: str
    verifier_manifest: str
    generation_time_sec: float
    generated_tokens: int
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class RunArrays:
    label: str
    seed: str
    details_path: Path
    records: tuple[Record, ...]
    subset_pass: np.ndarray  # [instance, omitted candidate]
    subset_best_gap: np.ndarray  # NaN exactly when the subset fails
    candidate_feasible: np.ndarray  # [instance, candidate]
    candidate_tokens: np.ndarray  # [instance, candidate]
    generated_tokens: np.ndarray  # [instance]
    generation_time_sec: np.ndarray  # [instance]


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisError(f"{label} must be finite")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _one_match(pattern: Path) -> Path:
    matches = sorted(pattern.parent.glob(pattern.name))
    if len(matches) != 1:
        raise AnalysisError(f"expected one match for {pattern}, found {len(matches)}")
    return matches[0]


def enumerate_leave_one_out(
    feasible: Sequence[bool], gaps: Sequence[float | None]
) -> list[dict[str, Any]]:
    """Enumerate and validate every B=9, K=8 subset.

    The returned order is the omitted candidate index, 0 through 8.  A best gap
    is defined iff the subset contains at least one feasible candidate.
    """

    if len(feasible) != B or len(gaps) != B:
        raise AnalysisError("leave-one-out enumeration requires exactly 9 candidates")
    rows: list[dict[str, Any]] = []
    appearances = [0] * B
    for omitted in range(B):
        selected = tuple(index for index in range(B) if index != omitted)
        if len(selected) != K:
            raise AssertionError("invalid K-subset construction")
        for index in selected:
            appearances[index] += 1
        feasible_gaps = [
            float(gaps[index])
            for index in selected
            if feasible[index] and gaps[index] is not None
        ]
        passed = bool(feasible_gaps)
        rows.append(
            {
                "omitted_candidate_index": omitted,
                "selected_candidate_indices": selected,
                "feasible_candidates_in_subset": len(feasible_gaps),
                "pass": passed,
                "best_feasible_gap": min(feasible_gaps) if passed else None,
            }
        )
    if appearances != [K] * B:
        raise AssertionError("leave-one-out subsets do not give equal candidate exposure")
    feasible_count = sum(bool(value) for value in feasible)
    expected_failures = math.comb(B - feasible_count, K) if B - feasible_count >= K else 0
    observed_failures = sum(not row["pass"] for row in rows)
    if observed_failures != expected_failures:
        raise AssertionError("failure enumeration disagrees with exact combinatorics")
    return rows


def _load_records(path: Path, *, label: str) -> tuple[Record, ...]:
    records: list[Record] = []
    seen_indices: set[int] = set()
    seen_hashes: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"{label}:{line_number} is malformed JSON") from exc
            if raw.get("task") != "tsp" or raw.get("split") != "eval":
                raise AnalysisError(f"{label}:{line_number} is not a TSP eval record")
            example_index = raw.get("example_index")
            semantic_hash = raw.get("semantic_instance_sha256")
            if isinstance(example_index, bool) or not isinstance(example_index, int):
                raise AnalysisError(f"{label}:{line_number} has invalid example_index")
            if not isinstance(semantic_hash, str) or len(semantic_hash) != 64:
                raise AnalysisError(f"{label}:{line_number} has invalid semantic hash")
            if example_index in seen_indices or semantic_hash in seen_hashes:
                raise AnalysisError(f"{label} contains a duplicate instance")
            seen_indices.add(example_index)
            seen_hashes.add(semantic_hash)

            raw_candidates = raw.get("candidates")
            if not isinstance(raw_candidates, list) or len(raw_candidates) != B:
                raise AnalysisError(f"{label}:{line_number} must have exactly B=9 candidates")
            candidates: list[Candidate] = []
            for candidate_index, candidate in enumerate(raw_candidates):
                if candidate.get("candidate_index") != candidate_index:
                    raise AnalysisError("candidate indices must be ordered 0 through 8")
                feasible = candidate.get("is_feasible")
                if not isinstance(feasible, bool):
                    raise AnalysisError("candidate feasibility must be boolean")
                raw_gap = candidate.get("gap_to_ground_truth")
                if feasible:
                    gap = _finite_float(raw_gap, "feasible candidate gap")
                    if gap < 0:
                        raise AnalysisError("feasible candidate gap must be nonnegative")
                else:
                    if raw_gap is not None:
                        raise AnalysisError("infeasible candidate cannot carry a gap")
                    gap = None
                completion_tokens = candidate.get("completion_tokens")
                if (
                    isinstance(completion_tokens, bool)
                    or not isinstance(completion_tokens, int)
                    or completion_tokens < 0
                ):
                    raise AnalysisError("completion_tokens must be a nonnegative integer")
                candidates.append(Candidate(feasible, gap, completion_tokens))

            feasible_count = sum(candidate.feasible for candidate in candidates)
            if raw.get("num_return_sequences") != B:
                raise AnalysisError("num_return_sequences does not equal B=9")
            if raw.get("feasible_candidates") != feasible_count:
                raise AnalysisError("record feasible_candidates is inconsistent")
            if not math.isclose(
                _finite_float(raw.get("candidate_feasibility_rate"), "candidate rate"),
                feasible_count / B,
                abs_tol=1e-12,
            ):
                raise AnalysisError("record candidate feasibility rate is inconsistent")
            expected_best = min(
                (candidate.gap for candidate in candidates if candidate.gap is not None),
                default=None,
            )
            if bool(raw.get("best_of_n_feasible")) != (expected_best is not None):
                raise AnalysisError("record best_of_n_feasible is inconsistent")
            raw_best = raw.get("best_feasible_gap")
            if expected_best is None:
                if raw_best is not None:
                    raise AnalysisError("failed B=9 record carries a best gap")
            elif not math.isclose(
                _finite_float(raw_best, "best_feasible_gap"), expected_best, abs_tol=1e-12
            ):
                raise AnalysisError("record best gap is inconsistent")

            generated_tokens = raw.get("generated_tokens")
            if isinstance(generated_tokens, bool) or not isinstance(generated_tokens, int):
                raise AnalysisError("generated_tokens must be an integer")
            if generated_tokens != sum(candidate.completion_tokens for candidate in candidates):
                raise AnalysisError("record generated token total is inconsistent")
            verifier = raw.get("strict_verifier")
            if not isinstance(verifier, dict):
                raise AnalysisError("strict verifier metadata is absent")
            verifier_manifest = verifier.get("implementation_manifest_sha256")
            if not isinstance(verifier_manifest, str) or len(verifier_manifest) != 64:
                raise AnalysisError("strict verifier manifest is invalid")

            transformed_hash = raw.get("transformed_example_sha256")
            benchmark_hash = raw.get("benchmark_row_sha256")
            sampling_seed = raw.get("sampling_seed")
            if not isinstance(transformed_hash, str) or len(transformed_hash) != 64:
                raise AnalysisError("transformed example hash is invalid")
            if not isinstance(benchmark_hash, str) or len(benchmark_hash) != 64:
                raise AnalysisError("benchmark row hash is invalid")
            if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
                raise AnalysisError("sampling_seed must be an integer")
            instruction = raw.get("instruction")
            input_problem = raw.get("input_problem")
            if not isinstance(instruction, str) or not isinstance(input_problem, str):
                raise AnalysisError("prompt fields must be strings")

            records.append(
                Record(
                    example_index=example_index,
                    semantic_hash=semantic_hash,
                    transformed_hash=transformed_hash,
                    benchmark_hash=benchmark_hash,
                    sampling_seed=sampling_seed,
                    instance_size=int(raw["instance_size"]),
                    ground_truth_objective=_finite_float(
                        raw["ground_truth_objective"], "ground_truth_objective"
                    ),
                    prompt_tokens=int(raw["prompt_tokens"]),
                    prompt_was_truncated=bool(raw["prompt_was_truncated"]),
                    instruction=instruction,
                    input_problem=input_problem,
                    verifier_manifest=verifier_manifest,
                    generation_time_sec=_finite_float(
                        raw["generation_time_sec"], "generation_time_sec"
                    ),
                    generated_tokens=generated_tokens,
                    candidates=tuple(candidates),
                )
            )
    records.sort(key=lambda record: record.example_index)
    if len(records) != N_INSTANCES:
        raise AnalysisError(f"{label} must contain exactly {N_INSTANCES} instances")
    if [record.example_index for record in records] != list(range(N_INSTANCES)):
        raise AnalysisError(f"{label} example indices are not exactly 0..30")
    return tuple(records)


def _validate_run_config(details_path: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(details_path.parent.glob("run_config*.json"))
    if len(candidates) != 1:
        raise AnalysisError(f"expected one run_config next to {details_path}")
    path = candidates[0]
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("evaluation_purpose") != "development" or config.get("run_status") != "complete":
        raise AnalysisError("evaluation must be complete and development-only")
    generation = config.get("generation")
    expected = {
        "seed": 314159,
        "seed_policy": "per_example_sha256_v1",
        "max_new_tokens": 160,
        "max_prompt_length": 1536,
        "num_return_sequences": B,
        "generation_batch_size": B,
        "best_of_n": B,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
    }
    if not isinstance(generation, dict):
        raise AnalysisError("run config lacks generation protocol")
    for key, value in expected.items():
        if generation.get(key) != value:
            raise AnalysisError(f"generation protocol mismatch for {key}")
    return path, expected


def _validate_pairing(reference: Sequence[Record], other: Sequence[Record]) -> None:
    for left, right in zip(reference, other, strict=True):
        fields = (
            "example_index",
            "semantic_hash",
            "transformed_hash",
            "benchmark_hash",
            "sampling_seed",
            "instance_size",
            "prompt_tokens",
            "prompt_was_truncated",
            "instruction",
            "input_problem",
            "verifier_manifest",
        )
        for field in fields:
            if getattr(left, field) != getattr(right, field):
                raise AnalysisError(f"paired records disagree on {field}")
        if not math.isclose(
            left.ground_truth_objective, right.ground_truth_objective, abs_tol=1e-12
        ):
            raise AnalysisError("paired records disagree on ground truth objective")


def _make_run(label: str, seed: str, details_path: Path) -> RunArrays:
    records = _load_records(details_path, label=f"{label}:{seed}")
    subset_pass = np.zeros((N_INSTANCES, B), dtype=bool)
    subset_best_gap = np.full((N_INSTANCES, B), np.nan, dtype=np.float64)
    candidate_feasible = np.zeros((N_INSTANCES, B), dtype=bool)
    candidate_tokens = np.zeros((N_INSTANCES, B), dtype=np.int64)
    for row_index, record in enumerate(records):
        feasible = [candidate.feasible for candidate in record.candidates]
        gaps = [candidate.gap for candidate in record.candidates]
        subsets = enumerate_leave_one_out(feasible, gaps)
        for subset in subsets:
            omitted = subset["omitted_candidate_index"]
            subset_pass[row_index, omitted] = subset["pass"]
            if subset["best_feasible_gap"] is not None:
                subset_best_gap[row_index, omitted] = subset["best_feasible_gap"]
        candidate_feasible[row_index] = feasible
        candidate_tokens[row_index] = [
            candidate.completion_tokens for candidate in record.candidates
        ]
    if not np.array_equal(subset_pass, np.isfinite(subset_best_gap)):
        raise AssertionError("pass and conditional-gap masks must coincide")
    return RunArrays(
        label=label,
        seed=seed,
        details_path=details_path,
        records=records,
        subset_pass=subset_pass,
        subset_best_gap=subset_best_gap,
        candidate_feasible=candidate_feasible,
        candidate_tokens=candidate_tokens,
        generated_tokens=np.asarray([record.generated_tokens for record in records]),
        generation_time_sec=np.asarray(
            [record.generation_time_sec for record in records], dtype=np.float64
        ),
    )


def _rate_entry(count: int, denominator: int) -> dict[str, Any]:
    return {"count": int(count), "denominator": int(denominator), "rate": count / denominator}


def _seed_summary(run: RunArrays) -> dict[str, Any]:
    subset_denominator = N_INSTANCES * B
    candidate_denominator = N_INSTANCES * B
    passing = int(run.subset_pass.sum())
    gaps = run.subset_best_gap[np.isfinite(run.subset_best_gap)]
    feasible_candidates_per_instance = run.candidate_feasible.sum(axis=1)
    return {
        "denominators": {
            "instances": N_INSTANCES,
            "exact_leave_one_out_k8_subsets": subset_denominator,
            "candidates": candidate_denominator,
            "conditional_quality_subsets": int(gaps.size),
        },
        "primary_k8": {
            "pass_at_8": _rate_entry(passing, subset_denominator),
            "failure_at_8": _rate_entry(subset_denominator - passing, subset_denominator),
            "success_thresholds": {
                f"gap_le_{threshold:.2f}": _rate_entry(
                    int(np.sum(run.subset_best_gap <= threshold)), subset_denominator
                )
                for threshold in GAP_THRESHOLDS
            },
            "best_feasible_gap_conditional": {
                "mean": float(np.mean(gaps)),
                "median": float(np.median(gaps)),
                "denominator_passing_subsets": int(gaps.size),
            },
        },
        "secondary_candidate_level": {
            "feasibility": _rate_entry(
                int(run.candidate_feasible.sum()), candidate_denominator
            ),
            "instances_by_feasible_candidate_count": {
                str(count): int(np.sum(feasible_candidates_per_instance == count))
                for count in range(B + 1)
                if np.any(feasible_candidates_per_instance == count)
            },
        },
        "secondary_b9_full_set": {
            "pass": _rate_entry(
                int(np.sum(feasible_candidates_per_instance > 0)), N_INSTANCES
            )
        },
        "resources": {
            "generated_tokens_total": int(run.generated_tokens.sum()),
            "generated_tokens_per_instance_mean": float(run.generated_tokens.mean()),
            "generated_tokens_per_candidate_mean": float(run.candidate_tokens.mean()),
            "generation_time_sec_total": float(run.generation_time_sec.sum()),
            "generation_time_sec_per_instance_mean": float(
                run.generation_time_sec.mean()
            ),
            "runtime_note": "Descriptive only; evaluations ran concurrently on different GPUs.",
        },
    }


Metric = Callable[[RunArrays, np.ndarray], float]


def _metric_functions() -> dict[str, Metric]:
    metrics: dict[str, Metric] = {
        "pass_at_8": lambda run, idx: float(run.subset_pass[idx].mean()),
        "failure_at_8": lambda run, idx: float((~run.subset_pass[idx]).mean()),
        "best_feasible_gap_conditional": lambda run, idx: float(
            np.nanmean(run.subset_best_gap[idx])
        ),
        "candidate_feasibility": lambda run, idx: float(
            run.candidate_feasible[idx].mean()
        ),
        "generated_tokens_per_instance": lambda run, idx: float(
            run.generated_tokens[idx].mean()
        ),
        "generation_time_sec_per_instance": lambda run, idx: float(
            run.generation_time_sec[idx].mean()
        ),
    }
    for threshold in GAP_THRESHOLDS:
        metrics[f"success_gap_le_{threshold:.2f}"] = (
            lambda run, idx, threshold=threshold: float(
                (run.subset_best_gap[idx] <= threshold).mean()
            )
        )
    return metrics


def _sample_sd(values: Sequence[float]) -> float | None:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1)) if len(values) > 1 else None


def _interval(values: np.ndarray, *, replicates: int, seed: int) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise AnalysisError("bootstrap produced no finite replicates")
    lower, upper = np.quantile(finite, (0.025, 0.975))
    return {
        "lower": float(lower),
        "upper": float(upper),
        "confidence": 0.95,
        "method": "percentile_two_way_crossed_seed_by_paired_instance_bootstrap",
        "replicates": replicates,
        "valid_replicates": int(finite.size),
        "rng": "numpy.random.PCG64",
        "base_seed": seed,
    }


def _bootstrap(
    sft: RunArrays,
    canonical: Sequence[RunArrays],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Crossed bootstrap: training seeds and paired test instances are clusters."""

    metrics = _metric_functions()
    rng = np.random.default_rng(seed)
    sft_draws = {name: np.empty(replicates) for name in metrics}
    canonical_draws = {name: np.empty(replicates) for name in metrics}
    delta_draws = {name: np.empty(replicates) for name in metrics if "best_feasible" not in name}
    common_names = ("coverage", "sft_gap", "canonical_gap", "canonical_minus_sft_gap")
    common_draws = {name: np.empty(replicates) for name in common_names}

    for replicate in range(replicates):
        instance_indices = rng.integers(0, N_INSTANCES, size=N_INSTANCES)
        seed_indices = rng.integers(0, len(canonical), size=len(canonical))
        for name, metric in metrics.items():
            left = metric(sft, instance_indices)
            right = float(
                np.mean([metric(canonical[index], instance_indices) for index in seed_indices])
            )
            sft_draws[name][replicate] = left
            canonical_draws[name][replicate] = right
            if name in delta_draws:
                delta_draws[name][replicate] = right - left

        coverage_values: list[float] = []
        left_gap_values: list[float] = []
        right_gap_values: list[float] = []
        delta_gap_values: list[float] = []
        for index in seed_indices:
            right = canonical[index]
            left_gaps = sft.subset_best_gap[instance_indices]
            right_gaps = right.subset_best_gap[instance_indices]
            common = np.isfinite(left_gaps) & np.isfinite(right_gaps)
            coverage_values.append(float(common.mean()))
            left_gap_values.append(float(np.mean(left_gaps[common])))
            right_gap_values.append(float(np.mean(right_gaps[common])))
            delta_gap_values.append(float(np.mean(right_gaps[common] - left_gaps[common])))
        common_draws["coverage"][replicate] = np.mean(coverage_values)
        common_draws["sft_gap"][replicate] = np.mean(left_gap_values)
        common_draws["canonical_gap"][replicate] = np.mean(right_gap_values)
        common_draws["canonical_minus_sft_gap"][replicate] = np.mean(delta_gap_values)

    all_indices = np.arange(N_INSTANCES)
    sft_point = {name: metric(sft, all_indices) for name, metric in metrics.items()}
    canonical_seed_points = {
        name: [metric(run, all_indices) for run in canonical]
        for name, metric in metrics.items()
    }
    canonical_point = {
        name: float(np.mean(values)) for name, values in canonical_seed_points.items()
    }
    common_seed_points = []
    for run in canonical:
        common = np.isfinite(sft.subset_best_gap) & np.isfinite(run.subset_best_gap)
        common_seed_points.append(
            {
                "seed": int(run.seed),
                "count": int(common.sum()),
                "denominator": int(common.size),
                "coverage": float(common.mean()),
                "sft_gap": float(np.mean(sft.subset_best_gap[common])),
                "canonical_gap": float(np.mean(run.subset_best_gap[common])),
                "canonical_minus_sft_gap": float(
                    np.mean(run.subset_best_gap[common] - sft.subset_best_gap[common])
                ),
            }
        )

    return {
        "design": {
            "unit_hierarchy": ["training_seed", "paired_test_instance"],
            "training_seeds_resampled_with_replacement": len(canonical),
            "paired_instances_resampled_with_replacement": N_INSTANCES,
            "leave_one_out_subsets": "exactly enumerated; never resampled",
            "evaluation_sampling_seed": 314159,
            "replicates": replicates,
            "base_seed": seed,
            "rng": "numpy.random.PCG64",
            "interval": "two-sided percentile 95%",
        },
        "sft": {
            name: {
                "point": sft_point[name],
                "interval": _interval(draws, replicates=replicates, seed=seed),
            }
            for name, draws in sft_draws.items()
        },
        "canonical": {
            name: {
                "point_mean_across_training_seeds": canonical_point[name],
                "seed_points": {
                    str(run.seed): canonical_seed_points[name][index]
                    for index, run in enumerate(canonical)
                },
                "seed_sample_sd": _sample_sd(canonical_seed_points[name]),
                "interval": _interval(draws, replicates=replicates, seed=seed),
            }
            for name, draws in canonical_draws.items()
        },
        "canonical_minus_sft": {
            name: {
                "point": canonical_point[name] - sft_point[name],
                "interval": _interval(draws, replicates=replicates, seed=seed),
            }
            for name, draws in delta_draws.items()
        },
        "common_feasible_quality": {
            "per_training_seed": common_seed_points,
            **{
                name: {
                    "point_mean_across_training_seeds": float(
                        np.mean([row[name] for row in common_seed_points])
                    ),
                    "seed_sample_sd": _sample_sd(
                        [row[name] for row in common_seed_points]
                    ),
                    "interval": _interval(draws, replicates=replicates, seed=seed),
                }
                for name, draws in common_draws.items()
            },
            "interpretation": (
                "Gap is paired by instance and omitted-candidate index and is conditional "
                "on both systems passing; coverage must be reported with the gap delta."
            ),
        },
    }


MECHANISM_FIELDS = {
    "candidate_feasibility_rate": "constrained_grpo/candidate_feasibility_rate",
    "failure_at_k": "constrained_grpo/clasp/canonical_failure_at_k",
    "full_b9_failure": "constrained_grpo/clasp/canonical_full_rollout_failure",
    "failure_k_minus_failure_b": "constrained_grpo/clasp/canonical_failure_k_minus_failure_b",
    "subset_quality_mean": "constrained_grpo/clasp/canonical_subset_quality_mean",
    "failed_recovery_mean": "constrained_grpo/clasp/canonical_failed_recovery_mean",
    "recovery_coefficient_abs_mean": "constrained_grpo/clasp/canonical_recovery_coefficient_abs_mean",
    "failure_coefficient_abs_mean": "constrained_grpo/clasp/canonical_failure_coefficient_abs_mean",
    "coefficient_abs_mean": "constrained_grpo/clasp/canonical_coefficient_abs_mean",
    "centering_residual_abs_max": "constrained_grpo/clasp/canonical_advantage_centering_residual_abs_max",
    "decomposition_residual_abs_max": "constrained_grpo/clasp/canonical_component_decomposition_residual_abs_max",
    "lambda_used": "constrained_grpo/clasp/bon_committed_lambda_used/tsp",
    "lambda_next": "constrained_grpo/clasp/bon_committed_lambda_next/tsp",
    "failure_ema": "constrained_grpo/clasp/bon_committed_failure_ema/tsp",
    "signed_residual": "constrained_grpo/clasp/bon_committed_signed_residual/tsp",
    "requested_delta": "constrained_grpo/clasp/bon_committed_requested_delta/tsp",
    "actual_delta": "constrained_grpo/clasp/bon_committed_actual_delta/tsp",
    "grad_norm": "grad_norm",
    "completion_mean_length": "completions/mean_length",
    "cumulative_processed_tokens": "num_tokens",
}


def _training_mechanism(log_path: Path, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = json.loads(log_path.read_text(encoding="utf-8"))
    steps = [entry for entry in raw if "loss" in entry and "step" in entry]
    terminal = [entry for entry in raw if "train_runtime" in entry]
    if len(steps) != 40 or [entry["step"] for entry in steps] != list(range(1, 41)):
        raise AnalysisError(f"seed {seed} training log does not contain steps 1..40")
    if len(terminal) != 1:
        raise AnalysisError(f"seed {seed} training log lacks one terminal record")
    rows: list[dict[str, Any]] = []
    for entry in steps:
        for key, expected in (
            ("constrained_grpo/clasp/canonical_set", 1.0),
            ("constrained_grpo/clasp/canonical_rollout_batch_size", 9.0),
            ("constrained_grpo/clasp/canonical_deployment_set_size", 8.0),
            ("constrained_grpo/clasp/canonical_subset_count", 9.0),
            ("constrained_grpo/clasp/bon_controller_commit", 1.0),
            ("constrained_grpo/clasp/bon_committed_group_count/tsp", 1.0),
        ):
            if not math.isclose(float(entry.get(key, math.nan)), expected, abs_tol=1e-8):
                raise AnalysisError(f"seed {seed} step {entry['step']} violates {key}")
        row: dict[str, Any] = {"training_seed": seed, "step": int(entry["step"])}
        for output_name, input_name in MECHANISM_FIELDS.items():
            row[output_name] = _finite_float(entry[input_name], input_name)
        if not math.isclose(
            row["lambda_next"] - row["lambda_used"], row["actual_delta"], abs_tol=2e-8
        ):
            raise AnalysisError("logged controller update identity does not hold")
        rows.append(row)
    failure_counts = np.asarray([row["failure_at_k"] * B for row in rows])
    if not np.allclose(failure_counts, np.rint(failure_counts), atol=2e-6):
        raise AnalysisError("failure@K is not an exact nine-subset U-statistic")
    summary = {
        "training_seed": seed,
        "optimizer_steps": 40,
        "rollout_groups": 40,
        "rollout_candidates": 40 * B,
        "exact_k8_subsets_evaluated": 40 * B,
        "failed_k8_subsets": int(np.rint(failure_counts).sum()),
        "all_infeasible_b9_groups": sum(row["full_b9_failure"] > 0.5 for row in rows),
        "one_feasible_b9_groups": sum(
            math.isclose(row["failure_at_k"], 1 / B, abs_tol=2e-6)
            and row["full_b9_failure"] < 0.5
            for row in rows
        ),
        "recovery_active_steps": sum(
            row["recovery_coefficient_abs_mean"] > 0 for row in rows
        ),
        "failure_credit_active_steps": sum(
            row["failure_coefficient_abs_mean"] > 0 for row in rows
        ),
        "controller_positive_updates": sum(row["actual_delta"] > 1e-12 for row in rows),
        "controller_negative_updates": sum(row["actual_delta"] < -1e-12 for row in rows),
        "controller_projected_zero_updates": sum(
            row["requested_delta"] < 0 and abs(row["actual_delta"]) <= 1e-12
            for row in rows
        ),
        "lambda_max": max(row["lambda_next"] for row in rows),
        "lambda_final": rows[-1]["lambda_next"],
        "max_centering_residual": max(row["centering_residual_abs_max"] for row in rows),
        "max_decomposition_residual": max(
            row["decomposition_residual_abs_max"] for row in rows
        ),
        "nonzero_gradient_steps": sum(row["grad_norm"] > 1e-12 for row in rows),
        "logged_completion_tokens": int(
            round(sum(row["completion_mean_length"] * B for row in rows))
        ),
        "final_cumulative_processed_tokens": int(rows[-1]["cumulative_processed_tokens"]),
        "training_runtime_sec": float(terminal[0]["train_runtime"]),
    }
    return rows, summary


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(path: Path, payload: dict[str, Any]) -> None:
    boot = payload["bootstrap"]
    sft = payload["runs"]["sft"]["primary_k8"]
    can = boot["canonical"]
    delta = boot["canonical_minus_sft"]
    common = boot["common_feasible_quality"]
    pass_ci = delta["pass_at_8"]["interval"]
    gap_ci = common["canonical_minus_sft_gap"]["interval"]
    text = f"""# Small-TSP canonical pilot v1

**Scope:** development-only; Qwen2.5-0.5B-Instruct; 31 held-out TSP instances;
40 GRPO updates; three training seeds (101, 202, 303). This is a mechanism and
system sanity check, not publication, benchmark, generalization, or SOTA evidence.

Primary deployment analysis exactly enumerates all nine K=8 leave-one-out subsets
from each B=9 evaluation rollout. Best-of-9 is secondary only.

- SFT pass@8: {sft['pass_at_8']['count']}/{sft['pass_at_8']['denominator']} = {sft['pass_at_8']['rate']:.4f}.
- Canonical pass@8: {can['pass_at_8']['point_mean_across_training_seeds']:.4f} (seed SD {can['pass_at_8']['seed_sample_sd']:.4f}).
- Canonical minus SFT pass@8: {delta['pass_at_8']['point']:+.4f}; crossed-bootstrap 95% CI [{pass_ci['lower']:+.4f}, {pass_ci['upper']:+.4f}].
- Common-feasible canonical minus SFT best gap: {common['canonical_minus_sft_gap']['point_mean_across_training_seeds']:+.4f}; 95% CI [{gap_ci['lower']:+.4f}, {gap_ci['upper']:+.4f}]. Report this only with common-feasible coverage ({common['coverage']['point_mean_across_training_seeds']:.4f}).
- Within-2%, 5%, and 10% success is effectively unchanged; the descriptive signal is feasibility, not near-optimality.

The 5,000-replicate deterministic bootstrap resamples training seeds and paired
instances as crossed clusters. Exact leave-one-out subsets are not resampled.
Runtime is descriptive because evaluations ran concurrently on different GPUs.
"""
    path.write_text(text, encoding="utf-8")


def analyze(output_dir: Path, *, bootstrap_replicates: int, bootstrap_seed: int) -> dict[str, Any]:
    sft_path = _one_match(DEFAULT_SFT_GLOB)
    canonical_paths = {
        seed: _one_match(DEFAULT_CANONICAL_GLOBS[seed]) for seed in TRAINING_SEEDS
    }
    sft_config_path, protocol = _validate_run_config(sft_path)
    for path in canonical_paths.values():
        _, candidate_protocol = _validate_run_config(path)
        if candidate_protocol != protocol:
            raise AnalysisError("evaluation generation protocols are not matched")

    sft = _make_run("sft", "sft", sft_path)
    canonical = [
        _make_run("canonical", str(seed), canonical_paths[seed])
        for seed in TRAINING_SEEDS
    ]
    for run in canonical:
        _validate_pairing(sft.records, run.records)
    if any(record.prompt_was_truncated for record in sft.records):
        raise AnalysisError("the pilot unexpectedly contains truncated prompts")

    bootstrap = _bootstrap(
        sft, canonical, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    mechanism_rows: list[dict[str, Any]] = []
    mechanism_summaries: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        rows, summary = _training_mechanism(DEFAULT_TRAINING_LOGS[seed], seed)
        mechanism_rows.extend(rows)
        mechanism_summaries[str(seed)] = summary

    input_files = [sft_path, sft_config_path]
    for seed in TRAINING_SEEDS:
        config_path, _ = _validate_run_config(canonical_paths[seed])
        input_files.extend([canonical_paths[seed], config_path, DEFAULT_TRAINING_LOGS[seed]])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_scope": "development_only_not_publication_or_sota_evidence",
        "experiment": {
            "task": "small TSP (sizes 10-20)",
            "model": "Qwen2.5-0.5B-Instruct with LoRA",
            "held_out_instances": N_INSTANCES,
            "training_updates": 40,
            "training_seeds": list(TRAINING_SEEDS),
            "evaluation_sampling_seed": 314159,
            "rollout_batch_B": B,
            "deployment_set_K": K,
            "exact_subsets_per_instance": math.comb(B, K),
            "gap_thresholds": list(GAP_THRESHOLDS),
        },
        "claim_boundary": (
            "Mechanism/system sanity check only. The 31-instance, 0.5B, 40-update "
            "development pilot cannot establish benchmark gains, broad generalization, "
            "or state of the art. Report threshold results and common-feasible coverage "
            "alongside any feasibility or quality statement."
        ),
        "validation": {
            "instances_exactly_paired": True,
            "candidate_positions_paired_by_common_per_example_sampling_seed": True,
            "strict_verifier_manifest_matched": True,
            "actual_prompt_text_and_transformed_hash_matched": True,
            "generation_protocol": protocol,
            "prompt_truncations": 0,
        },
        "inputs": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in input_files
        ],
        "runs": {
            "sft": _seed_summary(sft),
            "canonical": {
                str(run.seed): _seed_summary(run) for run in canonical
            },
        },
        "bootstrap": bootstrap,
        "training_mechanism": {
            "per_seed": mechanism_summaries,
            "trajectory_rows": len(mechanism_rows),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    subset_rows: list[dict[str, Any]] = []
    for run in [sft, *canonical]:
        for instance_index, record in enumerate(run.records):
            for omitted in range(B):
                gap = run.subset_best_gap[instance_index, omitted]
                subset_rows.append(
                    {
                        "system": run.label,
                        "training_seed": run.seed,
                        "example_index": record.example_index,
                        "semantic_instance_sha256": record.semantic_hash,
                        "sampling_seed": record.sampling_seed,
                        "omitted_candidate_index": omitted,
                        "subset_size_k": K,
                        "feasible_candidates_in_subset": int(
                            run.candidate_feasible[instance_index].sum()
                            - run.candidate_feasible[instance_index, omitted]
                        ),
                        "pass_at_8": int(run.subset_pass[instance_index, omitted]),
                        "best_feasible_gap": "" if not np.isfinite(gap) else float(gap),
                        "success_gap_le_0.02": int(np.isfinite(gap) and gap <= 0.02),
                        "success_gap_le_0.05": int(np.isfinite(gap) and gap <= 0.05),
                        "success_gap_le_0.10": int(np.isfinite(gap) and gap <= 0.10),
                    }
                )
    seed_rows: list[dict[str, Any]] = []
    for run in [sft, *canonical]:
        summary = _seed_summary(run)
        seed_rows.append(
            {
                "system": run.label,
                "training_seed": run.seed,
                "instances": N_INSTANCES,
                "k8_subsets": N_INSTANCES * B,
                "pass_at_8": summary["primary_k8"]["pass_at_8"]["rate"],
                "failure_at_8": summary["primary_k8"]["failure_at_8"]["rate"],
                "success_gap_le_0.02": summary["primary_k8"]["success_thresholds"]["gap_le_0.02"]["rate"],
                "success_gap_le_0.05": summary["primary_k8"]["success_thresholds"]["gap_le_0.05"]["rate"],
                "success_gap_le_0.10": summary["primary_k8"]["success_thresholds"]["gap_le_0.10"]["rate"],
                "conditional_gap_mean": summary["primary_k8"]["best_feasible_gap_conditional"]["mean"],
                "conditional_gap_denominator": summary["primary_k8"]["best_feasible_gap_conditional"]["denominator_passing_subsets"],
                "candidate_feasibility": summary["secondary_candidate_level"]["feasibility"]["rate"],
                "candidate_denominator": N_INSTANCES * B,
                "generated_tokens_total": summary["resources"]["generated_tokens_total"],
                "runtime_sec_total": summary["resources"]["generation_time_sec_total"],
            }
        )
    _write_csv(output_dir / "exact_k8_subset_rows.csv", subset_rows)
    _write_csv(output_dir / "per_seed_summary.csv", seed_rows)
    _write_csv(output_dir / "training_mechanism_trajectory.csv", mechanism_rows)
    (output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_readme(output_dir / "README.md", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        raise SystemExit("--bootstrap-replicates must be positive")
    payload = analyze(
        args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    delta = payload["bootstrap"]["canonical_minus_sft"]["pass_at_8"]
    print(
        f"wrote {args.output_dir}; pass@8 delta={delta['point']:+.6f}, "
        f"95% CI=[{delta['interval']['lower']:+.6f}, "
        f"{delta['interval']['upper']:+.6f}]"
    )


if __name__ == "__main__":
    main()
