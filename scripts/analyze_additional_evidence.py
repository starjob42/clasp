#!/usr/bin/env python3
"""Recompute development-only evidence from the sealed local TSP artifacts.

This script deliberately does not evaluate a model or touch a GPU.  It consumes the
exact instance-level SFT/CLASP records and the dense40 training log already present in
the repository, checks their pairing, and writes a compact diagnostic bundle.  The
bundle is useful for deciding what to run next; it is not confirmation, publication,
or SOTA evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

SCHEMA_VERSION = 1
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_BASE_SEED = 20_260_814
PREFIX_BUDGETS = (1, 2, 4, 8)
GAP_THRESHOLDS = (0.0, 0.005, 0.01, 0.0107, 0.02, 0.05, 0.10)

DEFAULT_SFT = Path(
    "experiments/sota_20260814/local_development/"
    "tsp_dense40_posttrain_k8_seed271828_v1/raw/sft/sft/"
    "local_tsp_sft500k_seed42_instruct/"
    "run_details__sft_only_local_tsp_sft500k_seed42_instruct_"
    "eval_k8_contracted_eval200_k8_seed271828_v1_fp81c9e70b005a.jsonl"
)
DEFAULT_DENSE40 = Path(
    "experiments/sota_20260814/local_development/"
    "tsp_dense40_posttrain_k8_seed271828_v1/raw/dense40/grpo/"
    "contracted_clasp_dense40_seed42_v1-checkpoint-40/"
    "run_details__grpo_contracted_clasp_dense40_seed42_v1-checkpoint-40_"
    "eval_k8_contracted_eval200_k8_seed271828_v1_fp6e1de90fb5fd.jsonl"
)
DEFAULT_TRAINING_LOG = Path(
    "outputs/tsp_k8_development_20260814/"
    "contracted_clasp_dense40_seed42_v1/grpo_log_history.json"
)
DEFAULT_CONTROLLER_STATE = Path(
    "outputs/tsp_k8_development_20260814/"
    "contracted_clasp_dense40_seed42_v1/multitask_primal_dual_state.json"
)
DEFAULT_SFT_DIAGNOSTICS = (
    Path(
        "results/sft_validation_20260812/diagnostics/"
        "sft100k_unified_attn_137500/failure_diagnostics.json"
    ),
    Path(
        "results/sft_validation_20260812/diagnostics/"
        "sft100k_specialist_mlp_cvrp_6250/failure_diagnostics.json"
    ),
    Path(
        "results/sft_validation_20260812/diagnostics/"
        "sft500k_specialist_mlp_cvrp_31250/failure_diagnostics.json"
    ),
    Path(
        "results/sft_validation_20260812/diagnostics/"
        "sft500k_specialist_mlp_tsp_31250/failure_diagnostics.json"
    ),
)
DEFAULT_OUTPUT = Path("results/additional_evidence")


class EvidenceError(ValueError):
    """Raised when the local artifacts cannot support the requested analysis."""


@dataclass(frozen=True)
class Candidate:
    completion: str
    completion_tokens: int
    feasible: bool
    gap: float | None


@dataclass(frozen=True)
class Record:
    key: tuple[str, str, int]
    example_index: int
    instance_size: int
    protocol_fingerprint: str
    verifier_manifest: str
    generation_time: float
    generated_tokens: int
    candidates: tuple[Candidate, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} must be finite")
    return result


def _load_records(path: Path, *, label: str) -> dict[tuple[str, str, int], Record]:
    records: dict[tuple[str, str, int], Record] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceError(f"{label}:{line_number} is malformed JSON") from exc
            if raw.get("task") != "tsp" or raw.get("split") != "eval":
                raise EvidenceError(f"{label}:{line_number} is not a TSP eval record")
            semantic_hash = raw.get("semantic_instance_sha256")
            sampling_seed = raw.get("sampling_seed")
            if not isinstance(semantic_hash, str) or len(semantic_hash) != 64:
                raise EvidenceError(f"{label}:{line_number} has invalid semantic hash")
            if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
                raise EvidenceError(f"{label}:{line_number} has invalid sampling seed")
            key = ("tsp", semantic_hash, sampling_seed)
            if key in records:
                raise EvidenceError(f"{label} contains duplicate key {key}")

            raw_candidates = raw.get("candidates")
            if not isinstance(raw_candidates, list) or len(raw_candidates) != 8:
                raise EvidenceError(f"{label}:{line_number} must contain exactly 8 candidates")
            candidates: list[Candidate] = []
            for index, candidate in enumerate(raw_candidates):
                if candidate.get("candidate_index") != index:
                    raise EvidenceError(
                        f"{label}:{line_number} candidate indices must be ordered 0..7"
                    )
                feasible = candidate.get("is_feasible")
                if not isinstance(feasible, bool):
                    raise EvidenceError(
                        f"{label}:{line_number}:candidate-{index} has invalid feasibility"
                    )
                raw_gap = candidate.get("gap_to_ground_truth")
                if feasible:
                    gap = _as_finite_float(
                        raw_gap, label=f"{label}:{line_number}:candidate-{index}.gap"
                    )
                    if gap < 0:
                        raise EvidenceError("feasible gaps must be nonnegative")
                else:
                    if raw_gap is not None:
                        raise EvidenceError("infeasible candidates must not carry a gap")
                    gap = None
                completion = candidate.get("completion")
                completion_tokens = candidate.get("completion_tokens")
                if not isinstance(completion, str):
                    raise EvidenceError("candidate completion must be a string")
                if isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int):
                    raise EvidenceError("candidate completion_tokens must be an integer")
                candidates.append(
                    Candidate(
                        completion=completion,
                        completion_tokens=completion_tokens,
                        feasible=feasible,
                        gap=gap,
                    )
                )

            verifier = raw.get("strict_verifier")
            if not isinstance(verifier, dict):
                raise EvidenceError(f"{label}:{line_number} lacks strict verifier metadata")
            verifier_manifest = verifier.get("implementation_manifest_sha256")
            if not isinstance(verifier_manifest, str) or len(verifier_manifest) != 64:
                raise EvidenceError(f"{label}:{line_number} has invalid verifier manifest")
            protocol = raw.get("protocol_fingerprint")
            if not isinstance(protocol, str) or len(protocol) != 64:
                raise EvidenceError(f"{label}:{line_number} has invalid protocol fingerprint")

            record = Record(
                key=key,
                example_index=int(raw["example_index"]),
                instance_size=int(raw["instance_size"]),
                protocol_fingerprint=protocol,
                verifier_manifest=verifier_manifest,
                generation_time=_as_finite_float(
                    raw["generation_time_sec"], label="generation_time_sec"
                ),
                generated_tokens=int(raw["generated_tokens"]),
                candidates=tuple(candidates),
            )
            if record.generated_tokens != sum(c.completion_tokens for c in record.candidates):
                raise EvidenceError(f"{label}:{line_number} has inconsistent token totals")
            records[key] = record
    if len(records) != 200:
        raise EvidenceError(f"{label} must contain the complete 200-instance lane")
    return records


def _validate_pairing(
    left: dict[tuple[str, str, int], Record],
    right: dict[tuple[str, str, int], Record],
) -> list[tuple[Record, Record]]:
    if set(left) != set(right):
        raise EvidenceError("SFT and dense40 key sets differ")
    paired: list[tuple[Record, Record]] = []
    # Match the established development comparator's lexical PairKey ordering so
    # K=8 bootstrap intervals reproduce its sealed receipt exactly.
    for key in sorted(left):
        a, b = left[key], right[key]
        if a.example_index != b.example_index or a.instance_size != b.instance_size:
            raise EvidenceError(f"paired source identity mismatch for {key}")
        if a.protocol_fingerprint != b.protocol_fingerprint:
            raise EvidenceError(f"paired protocol mismatch for {key}")
        if a.verifier_manifest != b.verifier_manifest:
            raise EvidenceError(f"paired verifier mismatch for {key}")
        paired.append((a, b))
    return paired


def _best_gap(record: Record, budget: int) -> float | None:
    gaps = [c.gap for c in record.candidates[:budget] if c.feasible]
    return min(gap for gap in gaps if gap is not None) if gaps else None


def _candidate_feasibility(record: Record, budget: int) -> float:
    return sum(c.feasible for c in record.candidates[:budget]) / budget


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _derived_seed(label: str) -> int:
    material = f"{BOOTSTRAP_BASE_SEED}:{label}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _bootstrap_mean_ci(values: Sequence[float], *, label: str) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(_derived_seed(label))
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    # Chunking bounds peak memory while preserving a deterministic PCG64 stream.
    chunk = 1_000
    for start in range(0, BOOTSTRAP_REPLICATES, chunk):
        count = min(chunk, BOOTSTRAP_REPLICATES - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        means[start : start + count] = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {
        "lower": float(lower),
        "upper": float(upper),
        "replicates": BOOTSTRAP_REPLICATES,
        "base_seed": BOOTSTRAP_BASE_SEED,
        "derived_seed": _derived_seed(label),
        "rng": "numpy.random.PCG64",
        "interval": "paired_percentile_95pct",
    }


def _exact_sign_test(differences: Iterable[float], *, tolerance: float = 1e-15) -> dict[str, Any]:
    values = list(differences)
    positive = sum(value > tolerance for value in values)
    negative = sum(value < -tolerance for value in values)
    non_tied = positive + negative
    if non_tied == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(non_tied, j) for j in range(min(positive, negative) + 1))
        p_value = min(1.0, 2.0 * tail / (2**non_tied))
    return {
        "positive_right_minus_left": positive,
        "negative_right_minus_left": negative,
        "ties": len(values) - non_tied,
        "non_tied": non_tied,
        "exact_two_sided_p": p_value,
    }


def _policy_summary(records: Sequence[Record], budget: int) -> dict[str, Any]:
    best_gaps = [_best_gap(record, budget) for record in records]
    successful_gaps = [gap for gap in best_gaps if gap is not None]
    candidate_rates = [_candidate_feasibility(record, budget) for record in records]
    return {
        "instances": len(records),
        "candidate_budget": budget,
        "pass_at_k": len(successful_gaps) / len(records),
        "failure_at_k": 1.0 - len(successful_gaps) / len(records),
        "failed_groups": len(records) - len(successful_gaps),
        "candidate_feasibility": float(np.mean(candidate_rates)),
        "conditional_best_feasible_gap": {
            "coverage": len(successful_gaps) / len(records),
            "mean": float(np.mean(successful_gaps)) if successful_gaps else None,
            "median": _percentile(successful_gaps, 0.5),
            "p90": _percentile(successful_gaps, 0.9),
        },
        "success_at_gap_threshold": {
            f"{threshold:.4f}": sum(
                gap is not None and gap <= threshold for gap in best_gaps
            )
            / len(records)
            for threshold in GAP_THRESHOLDS
        },
    }


def _paired_summary(pairs: Sequence[tuple[Record, Record]], budget: int) -> dict[str, Any]:
    left_records = [pair[0] for pair in pairs]
    right_records = [pair[1] for pair in pairs]
    left_gaps = [_best_gap(record, budget) for record in left_records]
    right_gaps = [_best_gap(record, budget) for record in right_records]
    left_pass = [gap is not None for gap in left_gaps]
    right_pass = [gap is not None for gap in right_gaps]
    pass_differences = [float(b) - float(a) for a, b in zip(left_pass, right_pass)]
    candidate_differences = [
        _candidate_feasibility(b, budget) - _candidate_feasibility(a, budget)
        for a, b in pairs
    ]
    common_indices = [
        index
        for index, (left_gap, right_gap) in enumerate(zip(left_gaps, right_gaps))
        if left_gap is not None and right_gap is not None
    ]
    gap_differences = [
        float(right_gaps[index]) - float(left_gaps[index]) for index in common_indices
    ]

    scope = "task:tsp" if budget == 8 else f"task:tsp:k{budget}"
    threshold_results: dict[str, Any] = {}
    for threshold in GAP_THRESHOLDS:
        left_success = [gap is not None and gap <= threshold for gap in left_gaps]
        right_success = [gap is not None and gap <= threshold for gap in right_gaps]
        differences = [float(b) - float(a) for a, b in zip(left_success, right_success)]
        threshold_results[f"{threshold:.4f}"] = {
            "left": sum(left_success) / len(pairs),
            "right": sum(right_success) / len(pairs),
            "right_minus_left": float(np.mean(differences)),
            "right_only_success": sum(not a and b for a, b in zip(left_success, right_success)),
            "left_only_success": sum(a and not b for a, b in zip(left_success, right_success)),
            "exact_mcnemar": _exact_sign_test(differences),
            "paired_bootstrap_95pct_ci": _bootstrap_mean_ci(
                differences, label=f"{scope}:success_at_gap_{threshold}"
            ),
        }

    return {
        "direction": "dense40_minus_sft",
        "candidate_budget": budget,
        "sft": _policy_summary(left_records, budget),
        "dense40": _policy_summary(right_records, budget),
        "pass_at_k": {
            "right_minus_left": float(np.mean(pass_differences)),
            "right_only_success": sum(not a and b for a, b in zip(left_pass, right_pass)),
            "left_only_success": sum(a and not b for a, b in zip(left_pass, right_pass)),
            "exact_mcnemar": _exact_sign_test(pass_differences),
            "paired_bootstrap_95pct_ci": _bootstrap_mean_ci(
                pass_differences, label=f"{scope}:pass_at_k"
            ),
        },
        "candidate_feasibility": {
            "right_minus_left": float(np.mean(candidate_differences)),
            "instance_level_exact_sign_test": _exact_sign_test(candidate_differences),
            "paired_bootstrap_95pct_ci": _bootstrap_mean_ci(
                candidate_differences, label=f"{scope}:candidate_feasibility"
            ),
        },
        "common_feasible_gap": {
            "jointly_feasible_instances": len(common_indices),
            "coverage": len(common_indices) / len(pairs),
            "sft_mean": float(np.mean([left_gaps[index] for index in common_indices]))
            if common_indices
            else None,
            "dense40_mean": float(np.mean([right_gaps[index] for index in common_indices]))
            if common_indices
            else None,
            "right_minus_left": float(np.mean(gap_differences)) if gap_differences else None,
            "instance_level_exact_sign_test": _exact_sign_test(gap_differences),
            "paired_bootstrap_95pct_ci": _bootstrap_mean_ci(
                gap_differences, label=f"{scope}:conditional_best_feasible_gap"
            ),
        },
        "success_at_gap_threshold": threshold_results,
    }


def _identity_summary(pairs: Sequence[tuple[Record, Record]]) -> dict[str, Any]:
    exact_completion = 0
    same_feasibility = 0
    jointly_feasible = 0
    identical_joint_gap = 0
    sft_only_feasible = 0
    dense_only_feasible = 0
    all_candidates_identical = 0
    for left, right in pairs:
        instance_identical = True
        for a, b in zip(left.candidates, right.candidates):
            if a.completion == b.completion:
                exact_completion += 1
            else:
                instance_identical = False
            same_feasibility += int(a.feasible == b.feasible)
            jointly_feasible += int(a.feasible and b.feasible)
            identical_joint_gap += int(
                a.feasible and b.feasible and a.gap is not None and a.gap == b.gap
            )
            sft_only_feasible += int(a.feasible and not b.feasible)
            dense_only_feasible += int(b.feasible and not a.feasible)
        all_candidates_identical += int(instance_identical)
    candidate_count = len(pairs) * 8
    return {
        "candidate_pairs": candidate_count,
        "exact_completion_pairs": exact_completion,
        "exact_completion_rate": exact_completion / candidate_count,
        "instances_with_all_eight_completions_identical": all_candidates_identical,
        "same_feasibility_label_pairs": same_feasibility,
        "same_feasibility_label_rate": same_feasibility / candidate_count,
        "jointly_feasible_candidate_pairs": jointly_feasible,
        "identical_gap_among_jointly_feasible_pairs": identical_joint_gap,
        "identical_gap_rate_among_jointly_feasible_pairs": identical_joint_gap
        / jointly_feasible,
        "sft_only_feasible_candidate_pairs": sft_only_feasible,
        "dense40_only_feasible_candidate_pairs": dense_only_feasible,
        "interpretation": (
            "The two policies are highly correlated under matched seeds. Candidate pairs "
            "remain nested within instances and are not independent replicates."
        ),
    }


def _systems_summary(pairs: Sequence[tuple[Record, Record]]) -> dict[str, Any]:
    left_time = [a.generation_time for a, _ in pairs]
    right_time = [b.generation_time for _, b in pairs]
    time_delta = [b - a for a, b in zip(left_time, right_time)]
    left_tokens = [a.generated_tokens for a, _ in pairs]
    right_tokens = [b.generated_tokens for _, b in pairs]
    token_delta = [float(b - a) for a, b in zip(left_tokens, right_tokens)]
    return {
        "sft": {
            "mean_generation_seconds_per_instance_k8": float(np.mean(left_time)),
            "median_generation_seconds_per_instance_k8": _percentile(left_time, 0.5),
            "mean_generated_tokens_per_instance_k8": float(np.mean(left_tokens)),
            "seconds_per_candidate_from_instance_total": float(np.mean(left_time)) / 8,
        },
        "dense40": {
            "mean_generation_seconds_per_instance_k8": float(np.mean(right_time)),
            "median_generation_seconds_per_instance_k8": _percentile(right_time, 0.5),
            "mean_generated_tokens_per_instance_k8": float(np.mean(right_tokens)),
            "seconds_per_candidate_from_instance_total": float(np.mean(right_time)) / 8,
        },
        "dense40_minus_sft": {
            "generation_seconds_per_instance": float(np.mean(time_delta)),
            "generation_seconds_paired_bootstrap_95pct_ci": _bootstrap_mean_ci(
                time_delta, label="generation-time"
            ),
            "generated_tokens_per_instance": float(np.mean(token_delta)),
            "generated_tokens_paired_bootstrap_95pct_ci": _bootstrap_mean_ci(
                token_delta, label="generated-tokens"
            ),
        },
        "limitation": (
            "Wall-clock records came from sequential evaluations on one local GPU. They "
            "are matched descriptive telemetry, not a controlled throughput benchmark."
        ),
    }


def _size_stratum(size: int) -> str:
    if size <= 40:
        return "11-40"
    if size <= 60:
        return "41-60"
    if size <= 80:
        return "61-80"
    return "81-100"


def _size_summary(pairs: Sequence[tuple[Record, Record]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum in ("11-40", "41-60", "61-80", "81-100"):
        subset = [pair for pair in pairs if _size_stratum(pair[0].instance_size) == stratum]
        left_gaps = [_best_gap(a, 8) for a, _ in subset]
        right_gaps = [_best_gap(b, 8) for _, b in subset]
        common = [
            (a, b) for a, b in zip(left_gaps, right_gaps) if a is not None and b is not None
        ]
        result[stratum] = {
            "instances": len(subset),
            "sft_pass_at_8": sum(gap is not None for gap in left_gaps) / len(subset),
            "dense40_pass_at_8": sum(gap is not None for gap in right_gaps) / len(subset),
            "common_feasible_instances": len(common),
            "common_feasible_gap_dense40_minus_sft": float(
                np.mean([float(b) - float(a) for a, b in common])
            )
            if common
            else None,
        }
    return result


def _load_training_diagnostics(
    log_path: Path, controller_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_log = json.loads(log_path.read_text(encoding="utf-8"))
    if not isinstance(raw_log, list):
        raise EvidenceError("training log must be a JSON list")
    required = {
        "loss",
        "step",
        "constrained_grpo/clasp/bon_committed_group_count/tsp",
        "constrained_grpo/clasp/bon_committed_failure_rate/tsp",
        "constrained_grpo/clasp/bon_lambda_used/tsp",
        "constrained_grpo/clasp/bon_lambda_next/tsp",
        "constrained_grpo/clasp/bon_mean_candidate_feasibility",
    }
    steps = [row for row in raw_log if isinstance(row, dict) and required <= set(row)]
    if [row["step"] for row in steps] != list(range(1, 41)):
        raise EvidenceError("expected one complete dense40 telemetry row for steps 1..40")

    group_counts = [row["constrained_grpo/clasp/bon_committed_group_count/tsp"] for row in steps]
    failure_rates = [
        row["constrained_grpo/clasp/bon_committed_failure_rate/tsp"] for row in steps
    ]
    total_groups = int(round(sum(group_counts)))
    failed_groups = int(round(sum(a * b for a, b in zip(group_counts, failure_rates))))
    fully_feasible_groups = int(
        round(
            sum(
                count * row["constrained_grpo/clasp/bon_fully_feasible_group_rate"]
                for count, row in zip(group_counts, steps)
            )
        )
    )
    singleton_groups = int(
        round(
            sum(
                count * row["constrained_grpo/clasp/bon_singleton_feasible_group_rate"]
                for count, row in zip(group_counts, steps)
            )
        )
    )
    successful_groups = total_groups - failed_groups
    zero_credit_groups = int(
        round(
            sum(
                count * row["constrained_grpo/clasp/bon_zero_success_credit_group_rate"]
                for count, row in zip(group_counts, steps)
            )
        )
    )
    quality_tie_groups = int(
        round(
            sum(
                count * row["constrained_grpo/clasp/bon_best_quality_tie_rate"]
                for count, row in zip(group_counts, steps)
            )
        )
    )
    candidate_count = total_groups * 8
    candidate_feasible = int(
        round(
            sum(
                count * 8 * row["constrained_grpo/clasp/bon_mean_candidate_feasibility"]
                for count, row in zip(group_counts, steps)
            )
        )
    )
    quality_nonzero = int(
        round(
            sum(
                count * 8 * row["constrained_grpo/clasp/bon_dense_quality_nonzero_rate"]
                for count, row in zip(group_counts, steps)
            )
        )
    )
    feasibility_nonzero = int(
        round(
            sum(
                count * 8 * row["constrained_grpo/clasp/bon_dense_feasibility_nonzero_rate"]
                for count, row in zip(group_counts, steps)
            )
        )
    )
    success_credit_nonzero = int(
        round(
            sum(
                count * 8 * row["constrained_grpo/clasp/bon_success_credit_nonzero_rate"]
                for count, row in zip(group_counts, steps)
            )
        )
    )

    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    bon_state = controller.get("__clasp_bon__")
    if not isinstance(bon_state, dict):
        raise EvidenceError("terminal state lacks __clasp_bon__")
    if bon_state.get("observed_groups", {}).get("tsp") != total_groups:
        raise EvidenceError("training log and controller observed-group counts disagree")

    lambda_used = [row["constrained_grpo/clasp/bon_lambda_used/tsp"] for row in steps]
    lambda_next = [row["constrained_grpo/clasp/bon_lambda_next/tsp"] for row in steps]
    aggregate_advantage_means = [
        row["constrained_grpo/mean_combined_advantage"] for row in steps
    ]
    component_lambdas: dict[str, Any] = {}
    for field in sorted(key for key in steps[0] if key.startswith("PD/lambda/tsp/")):
        name = field.removeprefix("PD/lambda/tsp/")
        trajectory = [row[field] for row in steps]
        component_lambdas[name] = {
            "first_logged": trajectory[0],
            "last_logged": trajectory[-1],
            "maximum_logged": max(trajectory),
        }

    def mean_field(field: str, rows: Sequence[dict[str, Any]]) -> float:
        return float(np.mean([row[field] for row in rows]))

    summary = {
        "steps": len(steps),
        "rollout_budget_k": 8,
        "observed_groups": total_groups,
        "candidate_rollouts": candidate_count,
        "group_exposure": {
            "failed_groups": failed_groups,
            "successful_groups": successful_groups,
            "singleton_feasible_groups": singleton_groups,
            "fully_feasible_groups": fully_feasible_groups,
            "mixed_feasible_groups": successful_groups
            - singleton_groups
            - fully_feasible_groups,
            "failed_group_recovery_exposures": failed_groups,
        },
        "set_failure_controller": {
            "target_failure_rate": steps[0][
                "constrained_grpo/clasp/bon_target_failure_rate_mean"
            ],
            "lambda_min": min(lambda_used + lambda_next),
            "lambda_max": max(lambda_used + lambda_next),
            "lambda_terminal": bon_state["lambdas"]["tsp"],
            "failure_ema_terminal": bon_state["failure_ema"]["tsp"],
            "steps_with_nonzero_lambda": sum(abs(value) > 1e-15 for value in lambda_used),
            "conclusion": (
                "The set-failure controller never activated because no K=8 group failed. "
                "This run provides no empirical test of the controller or failed-group recovery."
            ),
        },
        "credit_diagnostics": {
            "candidate_rollouts_feasible": candidate_feasible,
            "candidate_feasibility": candidate_feasible / candidate_count,
            "candidate_credits_nonzero": success_credit_nonzero,
            "candidate_credit_nonzero_rate": success_credit_nonzero / candidate_count,
            "dense_quality_credits_nonzero": quality_nonzero,
            "dense_quality_credit_nonzero_rate": quality_nonzero / candidate_count,
            "dense_feasibility_credits_nonzero": feasibility_nonzero,
            "dense_feasibility_credit_nonzero_rate": feasibility_nonzero / candidate_count,
            "successful_groups_with_zero_total_credit": zero_credit_groups,
            "multiple_feasible_groups_with_best_quality_tie": quality_tie_groups,
            "mean_advantage_abs": mean_field(
                "constrained_grpo/clasp/advantage_abs_mean", steps
            ),
            "mean_advantage_l2": mean_field("constrained_grpo/clasp/advantage_l2", steps),
            "maximum_absolute_logged_aggregate_advantage_mean": max(
                abs(value) for value in aggregate_advantage_means
            ),
            "missing_for_independent_credit_audit": [
                "candidate-level advantages",
                "per-group advantage sums",
                "candidate-level quality and violation arrays",
                "rollout texts and strict verifier outputs for training groups",
            ],
        },
        "first_10_vs_last_10": {
            "candidate_feasibility": {
                "first_10": mean_field(
                    "constrained_grpo/clasp/bon_mean_candidate_feasibility", steps[:10]
                ),
                "last_10": mean_field(
                    "constrained_grpo/clasp/bon_mean_candidate_feasibility", steps[-10:]
                ),
            },
            "success_credit_nonzero_rate": {
                "first_10": mean_field(
                    "constrained_grpo/clasp/bon_success_credit_nonzero_rate", steps[:10]
                ),
                "last_10": mean_field(
                    "constrained_grpo/clasp/bon_success_credit_nonzero_rate", steps[-10:]
                ),
            },
            "raw_positive_gap_telemetry": {
                "first_10": mean_field(
                    "constrained_grpo/task/tsp/mean_raw_positive_gap", steps[:10]
                ),
                "last_10": mean_field(
                    "constrained_grpo/task/tsp/mean_raw_positive_gap", steps[-10:]
                ),
            },
        },
        "component_lambda_trajectories": component_lambdas,
        "limitations": [
            "One TSP training seed and only 40 optimizer steps.",
            "Telemetry summarizes on-policy training groups, not a fixed evaluation set.",
            "The saved adapter omitted trained Q/K/V bias tensors and is not the terminal in-memory policy.",
            "This run used dense-rank credit, not the canonical leave-out estimator in the ICLR method specification.",
        ],
    }
    return summary, steps


def _subsampled_failure_exposure(paths: Sequence[Path]) -> dict[str, Any]:
    """Compute exact failure probability for a random K-subset of saved K=8 draws."""

    result: dict[str, Any] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        tasks = payload.get("tasks")
        if not isinstance(tasks, dict) or not tasks:
            raise EvidenceError(f"invalid SFT failure diagnostics: {path}")
        checkpoint = path.parent.name
        for task, task_payload in tasks.items():
            histogram = task_payload.get("feasible_candidates_histogram")
            if not isinstance(histogram, dict):
                raise EvidenceError(f"missing feasible-candidate histogram: {path}:{task}")
            counts: dict[int, int] = {}
            for raw_feasible, raw_count in histogram.items():
                feasible = int(raw_feasible)
                if feasible < 0 or feasible > 8:
                    raise EvidenceError(f"invalid feasible count in {path}:{task}")
                if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
                    raise EvidenceError(f"invalid histogram count in {path}:{task}")
                counts[feasible] = raw_count
            instances = sum(counts.values())
            if instances != task_payload.get("instances") or instances != 100:
                raise EvidenceError(f"incomplete SFT failure diagnostics: {path}:{task}")
            failure_by_budget: dict[str, float] = {}
            for budget in PREFIX_BUDGETS:
                failure = 0.0
                for feasible, count in counts.items():
                    infeasible = 8 - feasible
                    if infeasible >= budget:
                        failure += count * math.comb(infeasible, budget) / math.comb(8, budget)
                failure_by_budget[str(budget)] = failure / instances
            result[f"{checkpoint}::{task}"] = {
                "instances": instances,
                "observed_feasible_candidate_histogram_k8": {
                    str(index): counts.get(index, 0) for index in range(9)
                },
                "expected_failure_of_uniform_random_k_subset": failure_by_budget,
                "observed_failure_at_8": counts.get(0, 0) / instances,
            }
    return {
        "systems": result,
        "estimand": (
            "For each saved eight-candidate instance, choose a size-K subset uniformly "
            "without replacement; report the exact expected all-infeasible rate, then "
            "average over the 100 development instances."
        ),
        "interpretation": (
            "Historical CVRP and MIS have substantial failed-set exposure, whereas TSP "
            "at K=8 is saturated. These diagnostics motivate task selection only."
        ),
        "limitations": [
            "The subsets reuse one saved K=8 decode rather than performing fresh K-specific decoding.",
            "These are historical SFT development artifacts, not canonical CLASP training runs.",
            "All checkpoints use Qwen2.5-7B-Instruct and affected lora_only families may omit trained biases on save.",
        ],
    }


def _write_instance_csv(
    path: Path, pairs: Sequence[tuple[Record, Record]]
) -> None:
    fields = [
        "example_index",
        "semantic_instance_sha256",
        "sampling_seed",
        "instance_size",
        "size_stratum",
    ]
    for budget in PREFIX_BUDGETS:
        fields.extend(
            [
                f"sft_pass_at_{budget}",
                f"dense40_pass_at_{budget}",
                f"sft_best_gap_at_{budget}",
                f"dense40_best_gap_at_{budget}",
                f"sft_candidate_feasibility_at_{budget}",
                f"dense40_candidate_feasibility_at_{budget}",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for left, right in sorted(pairs, key=lambda pair: pair[0].example_index):
            row: dict[str, Any] = {
                "example_index": left.example_index,
                "semantic_instance_sha256": left.key[1],
                "sampling_seed": left.key[2],
                "instance_size": left.instance_size,
                "size_stratum": _size_stratum(left.instance_size),
            }
            for budget in PREFIX_BUDGETS:
                left_gap = _best_gap(left, budget)
                right_gap = _best_gap(right, budget)
                row.update(
                    {
                        f"sft_pass_at_{budget}": int(left_gap is not None),
                        f"dense40_pass_at_{budget}": int(right_gap is not None),
                        f"sft_best_gap_at_{budget}": "" if left_gap is None else left_gap,
                        f"dense40_best_gap_at_{budget}": "" if right_gap is None else right_gap,
                        f"sft_candidate_feasibility_at_{budget}": _candidate_feasibility(
                            left, budget
                        ),
                        f"dense40_candidate_feasibility_at_{budget}": _candidate_feasibility(
                            right, budget
                        ),
                    }
                )
            writer.writerow(row)


def _write_trajectory_csv(path: Path, steps: Sequence[dict[str, Any]]) -> None:
    fields = {
        "step": "step",
        "candidate_feasibility": "constrained_grpo/clasp/bon_mean_candidate_feasibility",
        "group_failure_rate": "constrained_grpo/clasp/bon_group_failure_rate",
        "fully_feasible_group_rate": "constrained_grpo/clasp/bon_fully_feasible_group_rate",
        "lambda_used": "constrained_grpo/clasp/bon_lambda_used/tsp",
        "lambda_next": "constrained_grpo/clasp/bon_lambda_next/tsp",
        "failure_ema": "constrained_grpo/clasp/bon_failure_ema/tsp",
        "success_credit_nonzero_rate": "constrained_grpo/clasp/bon_success_credit_nonzero_rate",
        "quality_credit_nonzero_rate": "constrained_grpo/clasp/bon_dense_quality_nonzero_rate",
        "feasibility_credit_nonzero_rate": "constrained_grpo/clasp/bon_dense_feasibility_nonzero_rate",
        "zero_credit_success_group_rate": "constrained_grpo/clasp/bon_zero_success_credit_group_rate",
        "best_quality_tie_rate": "constrained_grpo/clasp/bon_best_quality_tie_rate",
        "advantage_abs_mean": "constrained_grpo/clasp/advantage_abs_mean",
        "advantage_l2": "constrained_grpo/clasp/advantage_l2",
        "aggregate_combined_advantage_mean": "constrained_grpo/mean_combined_advantage",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for raw in steps:
            writer.writerow({output: raw[source] for output, source in fields.items()})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-details", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--dense40-details", type=Path, default=DEFAULT_DENSE40)
    parser.add_argument("--training-log", type=Path, default=DEFAULT_TRAINING_LOG)
    parser.add_argument("--controller-state", type=Path, default=DEFAULT_CONTROLLER_STATE)
    parser.add_argument(
        "--sft-diagnostic",
        action="append",
        type=Path,
        default=None,
        help="Repeat to replace the default historical SFT failure-diagnostic inputs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    diagnostic_paths = tuple(args.sft_diagnostic or DEFAULT_SFT_DIAGNOSTICS)
    input_paths = {
        "sft_details": args.sft_details,
        "dense40_details": args.dense40_details,
        "training_log": args.training_log,
        "controller_state": args.controller_state,
        **{
            f"sft_failure_diagnostic_{index}": path
            for index, path in enumerate(diagnostic_paths)
        },
    }
    for label, path in input_paths.items():
        if not path.is_file():
            raise EvidenceError(f"missing {label}: {path}")

    sft = _load_records(args.sft_details, label="sft")
    dense40 = _load_records(args.dense40_details, label="dense40")
    pairs = _validate_pairing(sft, dense40)
    training_summary, training_steps = _load_training_diagnostics(
        args.training_log, args.controller_state
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "evidence_scope": "development_only_not_publication_eligible",
        "comparison_direction": "dense40_minus_sft",
        "analysis_implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
            "size_bytes": Path(__file__).resolve().stat().st_size,
        },
        "inputs": {
            label: {
                "path": str(path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for label, path in input_paths.items()
        },
        "pairing": {
            "instances": len(pairs),
            "key": ["task", "semantic_instance_sha256", "sampling_seed"],
            "candidate_budget": 8,
            "protocol_fingerprint": pairs[0][0].protocol_fingerprint,
            "strict_verifier_manifest_sha256": pairs[0][0].verifier_manifest,
            "status": "exact",
        },
        "prefix_budget_note": (
            "K=1,2,4 are deterministic prefixes of each completed K=8 draw, not fresh "
            "K-specific generations. They estimate sampling-budget scaling without new inference."
        ),
        "paired_candidate_identity": _identity_summary(pairs),
        "paired_prefix_metrics": {
            str(budget): _paired_summary(pairs, budget) for budget in PREFIX_BUDGETS
        },
        "instance_size_descriptive_k8": _size_summary(pairs),
        "systems_telemetry": _systems_summary(pairs),
        "training_diagnostics": training_summary,
        "historical_sft_subsampled_failure_exposure": _subsampled_failure_exposure(
            diagnostic_paths
        ),
        "claim_boundary": [
            "One TSP seed, one historical development file, and a saved-policy fidelity defect.",
            "No comparison with GRPO, constrained GRPO, or canonical CLASP is present.",
            "Bootstrap intervals quantify paired instance resampling only, not training-seed uncertainty.",
            "Threshold and prefix analyses are exploratory and receive no multiplicity correction.",
            "Values may guide experiments or appear in a clearly labeled diagnostic appendix; they cannot support the paper's central efficacy claim.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "paired_tsp_dense40_diagnostics.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_instance_csv(args.output_dir / "paired_tsp_dense40_by_instance.csv", pairs)
    _write_trajectory_csv(args.output_dir / "dense40_training_trajectory.csv", training_steps)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
