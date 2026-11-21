#!/usr/bin/env python3
"""Analyze raw B=9 small-TSP evaluations at the contracted deployment K=8.

Every primary metric is the exact uniform average over all nine leave-one-out
size-eight subsets.  Best-of-nine is emitted only as a secondary diagnostic.
The script also accepts optional 40-step run directories for trajectory checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "experiments/additional_evidence/small_tsp_canonical_v1/matched40_v1.contract.json"
)
ERROR_PREFIX = "[SmallTSPK8AnalysisError]"


def fail(message: str) -> NoReturn:
    raise ValueError(f"{ERROR_PREFIX} {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class Candidate:
    feasible: bool
    gap: float | None


@dataclass(frozen=True)
class EvaluationRecord:
    example_index: int
    semantic_hash: str
    candidates: tuple[Candidate, ...]
    generated_tokens: int
    generation_time_sec: float


def load_contract() -> dict[str, Any]:
    try:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot load matched40 contract: {exc}")
    analysis = contract.get("post_training_analysis")
    if not (
        contract.get("development_only") is True
        and contract.get("publication_eligible") is False
        and isinstance(analysis, dict)
        and analysis.get("primary_candidate_budget") == 8
        and analysis.get("raw_candidates_per_instance") == 9
    ):
        fail("matched40 contract does not declare the exact B=9/K=8 development analysis")
    return contract


def load_records(path: Path, *, expected_records: int = 31) -> dict[str, EvaluationRecord]:
    records: dict[str, EvaluationRecord] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"{path}:{line_number} is malformed JSON: {exc}")
            if raw.get("task") != "tsp" or raw.get("split") != "eval":
                fail(f"{path}:{line_number} is not a TSP eval record")
            semantic_hash = raw.get("semantic_instance_sha256")
            if not isinstance(semantic_hash, str) or len(semantic_hash) != 64:
                fail(f"{path}:{line_number} has no valid semantic instance hash")
            if semantic_hash in records:
                fail(f"{path} duplicates semantic instance {semantic_hash}")
            raw_candidates = raw.get("candidates")
            if not isinstance(raw_candidates, list) or len(raw_candidates) != 9:
                fail(f"{path}:{line_number} must contain exactly B=9 candidates")
            candidates: list[Candidate] = []
            for candidate_index, raw_candidate in enumerate(raw_candidates):
                if not isinstance(raw_candidate, dict):
                    fail(f"{path}:{line_number} candidate {candidate_index} is not an object")
                if raw_candidate.get("candidate_index") != candidate_index:
                    fail(f"{path}:{line_number} candidate indices must be ordered 0..8")
                feasible = raw_candidate.get("is_feasible")
                if not isinstance(feasible, bool):
                    fail(f"{path}:{line_number} candidate feasibility is not boolean")
                raw_gap = raw_candidate.get("gap_to_ground_truth")
                if feasible:
                    gap = finite_float(
                        raw_gap,
                        label=f"{path}:{line_number}:candidate-{candidate_index}.gap",
                    )
                    if gap < 0.0:
                        fail("feasible gaps must be nonnegative")
                else:
                    if raw_gap is not None:
                        fail("infeasible candidates must not carry a ground-truth gap")
                    gap = None
                candidates.append(Candidate(feasible=feasible, gap=gap))
            records[semantic_hash] = EvaluationRecord(
                example_index=int(raw["example_index"]),
                semantic_hash=semantic_hash,
                candidates=tuple(candidates),
                generated_tokens=int(raw["generated_tokens"]),
                generation_time_sec=finite_float(
                    raw["generation_time_sec"], label="generation_time_sec"
                ),
            )
    if len(records) != expected_records:
        fail(f"{path} must contain {expected_records} records; observed={len(records)}")
    return records


def quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def leave_one_out_instance(
    record: EvaluationRecord,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    subset_best_gaps: list[float | None] = []
    threshold_successes = {float(threshold): 0 for threshold in thresholds}
    for omitted in range(9):
        gaps = [
            candidate.gap
            for index, candidate in enumerate(record.candidates)
            if index != omitted and candidate.feasible and candidate.gap is not None
        ]
        best_gap = min(gaps) if gaps else None
        subset_best_gaps.append(best_gap)
        for threshold in thresholds:
            threshold_successes[float(threshold)] += int(
                best_gap is not None and best_gap <= threshold
            )

    feasible_count = sum(candidate.feasible for candidate in record.candidates)
    infeasible_count = 9 - feasible_count
    successful_best_gaps = [gap for gap in subset_best_gaps if gap is not None]
    exact_failure_formula = (
        math.comb(infeasible_count, 8) / math.comb(9, 8)
        if infeasible_count >= 8
        else 0.0
    )
    enumerated_failure = 1.0 - len(successful_best_gaps) / 9.0
    if abs(exact_failure_formula - enumerated_failure) > 1e-15:
        fail("leave-one-out enumeration disagrees with the failure@8 U-statistic")

    all_nine_gaps = [
        candidate.gap
        for candidate in record.candidates
        if candidate.feasible and candidate.gap is not None
    ]
    bo9_best_gap = min(all_nine_gaps) if all_nine_gaps else None
    return {
        "example_index": record.example_index,
        "semantic_instance_sha256": record.semantic_hash,
        "feasible_candidates": feasible_count,
        "candidate_feasibility": feasible_count / 9.0,
        "pass_at_8": len(successful_best_gaps) / 9.0,
        "failure_at_8": enumerated_failure,
        "success_at_gap_threshold": {
            f"{threshold:g}": threshold_successes[float(threshold)] / 9.0
            for threshold in thresholds
        },
        "successful_size8_subsets": len(successful_best_gaps),
        "conditional_best_feasible_gap": (
            statistics.fmean(successful_best_gaps) if successful_best_gaps else None
        ),
        "size8_best_gaps": subset_best_gaps,
        "bo9_diagnostic": {
            "pass": float(bo9_best_gap is not None),
            "best_feasible_gap": bo9_best_gap,
            "success_at_gap_threshold": {
                f"{threshold:g}": float(
                    bo9_best_gap is not None and bo9_best_gap <= threshold
                )
                for threshold in thresholds
            },
        },
        "generated_tokens": record.generated_tokens,
        "generation_time_sec": record.generation_time_sec,
    }


def summarize_records(
    records: dict[str, EvaluationRecord], thresholds: Sequence[float]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_instance = [
        leave_one_out_instance(records[key], thresholds) for key in sorted(records)
    ]
    total_subsets = len(by_instance) * 9
    all_successful_gaps = [
        gap
        for row in by_instance
        for gap in row["size8_best_gaps"]
        if gap is not None
    ]
    successful_subsets = len(all_successful_gaps)
    pass_at_8 = sum(row["pass_at_8"] for row in by_instance) / len(by_instance)
    failure_at_8 = sum(row["failure_at_8"] for row in by_instance) / len(by_instance)
    if abs(pass_at_8 + failure_at_8 - 1.0) > 1e-15:
        fail("aggregate pass@8 and failure@8 do not sum to one")
    bo9_gaps = [
        row["bo9_diagnostic"]["best_feasible_gap"]
        for row in by_instance
        if row["bo9_diagnostic"]["best_feasible_gap"] is not None
    ]
    summary = {
        "estimand": "uniform average over all nine leave-one-out size-8 subsets per instance",
        "instances": len(by_instance),
        "rollout_candidates_B": 9,
        "deployment_candidates_K": 8,
        "enumerated_size8_subsets": total_subsets,
        "pass_at_8": pass_at_8,
        "failure_at_8": failure_at_8,
        "candidate_feasibility": statistics.fmean(
            row["candidate_feasibility"] for row in by_instance
        ),
        "success_at_gap_threshold": {
            f"{threshold:g}": statistics.fmean(
                row["success_at_gap_threshold"][f"{threshold:g}"] for row in by_instance
            )
            for threshold in thresholds
        },
        "conditional_best_feasible_gap": {
            "coverage": successful_subsets / total_subsets,
            "successful_size8_subsets": successful_subsets,
            "mean": statistics.fmean(all_successful_gaps) if all_successful_gaps else None,
            "median": quantile(all_successful_gaps, 0.5),
            "p90": quantile(all_successful_gaps, 0.9),
        },
        "compute": {
            "generated_tokens": sum(row["generated_tokens"] for row in by_instance),
            "generation_time_sec": sum(row["generation_time_sec"] for row in by_instance),
        },
        "bo9_secondary_diagnostic": {
            "role": "secondary_only_not_primary",
            "pass_at_9": len(bo9_gaps) / len(by_instance),
            "conditional_best_feasible_gap_mean": (
                statistics.fmean(bo9_gaps) if bo9_gaps else None
            ),
            "success_at_gap_threshold": {
                f"{threshold:g}": statistics.fmean(
                    row["bo9_diagnostic"]["success_at_gap_threshold"][f"{threshold:g}"]
                    for row in by_instance
                )
                for threshold in thresholds
            },
        },
    }
    return summary, by_instance


def parse_assignment(raw: str, *, kind: str) -> tuple[str, Path]:
    if "=" not in raw:
        fail(f"--{kind} must use LABEL=PATH")
    label, raw_path = raw.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().absolute()
    path_is_valid = path.is_file() if kind == "details" else path.is_dir()
    if not label or not path_is_valid:
        fail(f"invalid --{kind} assignment: {raw}")
    return label, path


def summarize_training_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    status_path = path / "stage_status.json"
    history_path = path / "grpo_log_history.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot load training run {path}: {exc}")
    if not (
        status.get("completed") is True
        and status.get("global_step") == 40
        and status.get("max_steps") == 40
        and status.get("has_saved_model") is True
        and isinstance(history, list)
    ):
        fail(f"training run is not a complete saved 40-step trajectory: {path}")
    steps = [row for row in history if isinstance(row, dict) and isinstance(row.get("step"), int)]
    if not steps or max(row["step"] for row in steps) != 40:
        fail(f"training log lacks a complete step-40 trajectory: {path}")
    summary = {
        "path": str(path),
        "stage_status_sha256": sha256_file(status_path),
        "history_sha256": sha256_file(history_path),
        "logged_optimizer_steps": len(steps),
        "final_step": max(row["step"] for row in steps),
        "final_train_metrics": steps[-1],
    }
    return summary, steps


def paired_point_differences(
    by_label: dict[str, list[dict[str, Any]]],
    *,
    reference: str,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    if reference not in by_label:
        fail(f"paired reference label is absent: {reference}")
    keyed = {
        label: {row["semantic_instance_sha256"]: row for row in rows}
        for label, rows in by_label.items()
    }
    reference_keys = set(keyed[reference])
    for label, rows in keyed.items():
        if set(rows) != reference_keys:
            fail(f"{label} and {reference} do not contain identical semantic instances")
    output: dict[str, Any] = {}
    for label, rows in keyed.items():
        if label == reference:
            continue
        pass_differences = [
            rows[key]["pass_at_8"] - keyed[reference][key]["pass_at_8"]
            for key in sorted(reference_keys)
        ]
        output[label] = {
            "right_minus_reference": label,
            "reference": reference,
            "instances": len(reference_keys),
            "pass_at_8_mean_difference": statistics.fmean(pass_differences),
            "success_at_gap_threshold_mean_difference": {
                f"{threshold:g}": statistics.fmean(
                    rows[key]["success_at_gap_threshold"][f"{threshold:g}"]
                    - keyed[reference][key]["success_at_gap_threshold"][f"{threshold:g}"]
                    for key in sorted(reference_keys)
                )
                for threshold in thresholds
            },
            "inference": "point_estimate_only_no_interval_or_significance_claim",
        }
    return output


def write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    by_label: dict[str, list[dict[str, Any]]],
    training_steps: dict[str, list[dict[str, Any]]],
) -> None:
    try:
        output_dir.mkdir(parents=True, mode=0o750)
    except FileExistsError:
        fail(f"analysis output is create-once and already exists: {output_dir}")
    (output_dir / "k8_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    threshold_names = [f"success_at_{value:g}" for value in payload["gap_thresholds"]]
    with (output_dir / "leave_one_out_k8_by_instance.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "label",
            "example_index",
            "semantic_instance_sha256",
            "feasible_candidates",
            "candidate_feasibility",
            "pass_at_8",
            "failure_at_8",
            "successful_size8_subsets",
            "conditional_best_feasible_gap",
            "bo9_pass_secondary",
            "bo9_best_feasible_gap_secondary",
            *threshold_names,
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, rows in sorted(by_label.items()):
            for row in rows:
                output = {
                    key: row.get(key)
                    for key in fieldnames
                    if key not in threshold_names
                    and key not in {"label", "bo9_pass_secondary", "bo9_best_feasible_gap_secondary"}
                }
                output["label"] = label
                output["bo9_pass_secondary"] = row["bo9_diagnostic"]["pass"]
                output["bo9_best_feasible_gap_secondary"] = row["bo9_diagnostic"][
                    "best_feasible_gap"
                ]
                for threshold, column in zip(payload["gap_thresholds"], threshold_names):
                    output[column] = row["success_at_gap_threshold"][f"{threshold:g}"]
                writer.writerow(output)
    if training_steps:
        with (output_dir / "training_curves.csv").open("x", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "label",
                "step",
                "loss",
                "grad_norm",
                "reward",
                "completion_mean_length",
                "candidate_feasibility",
                "canonical_failure_at_8",
                "canonical_lambda",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for label, rows in sorted(training_steps.items()):
                for row in rows:
                    writer.writerow(
                        {
                            "label": label,
                            "step": row.get("step"),
                            "loss": row.get("loss"),
                            "grad_norm": row.get("grad_norm"),
                            "reward": row.get("reward"),
                            "completion_mean_length": row.get("completions/mean_length"),
                            "candidate_feasibility": row.get(
                                "constrained_grpo/candidate_feasibility_rate"
                            ),
                            "canonical_failure_at_8": row.get(
                                "constrained_grpo/clasp/canonical_failure_at_k"
                            ),
                            "canonical_lambda": row.get(
                                "constrained_grpo/clasp/bon_lambda/tsp"
                            ),
                        }
                    )
    readme = """# Small-TSP matched40 K=8 analysis

Development-only screening evidence. The primary deployment estimator is the
exact uniform average over all nine leave-one-out size-eight subsets for every
instance. `pass_at_8`, `failure_at_8`, every success threshold, and conditional
best-feasible gap/coverage all use that estimator. Best-of-nine is retained only
under `bo9_secondary_diagnostic` and must not be reported as the primary result.

This 31-instance historical-data-derived holdout, 0.5B instruction model, and
40-step horizon are not independent, confirmatory, publication, or SOTA evidence.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--details",
        action="append",
        default=[],
        metavar="LABEL=JSONL",
        help="Raw B=9 evaluation JSONL; repeat for each policy/seed.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Optional complete 40-step training run directory.",
    )
    parser.add_argument("--reference", default=None, help="Label for paired point differences.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.details:
        fail("at least one --details LABEL=JSONL input is required")
    contract = load_contract()
    thresholds = [float(value) for value in contract["post_training_analysis"]["gap_thresholds"]]
    detail_assignments = [parse_assignment(raw, kind="details") for raw in args.details]
    if len({label for label, _ in detail_assignments}) != len(detail_assignments):
        fail("--details labels must be unique")
    summaries: dict[str, Any] = {}
    by_label: dict[str, list[dict[str, Any]]] = {}
    input_receipts: dict[str, Any] = {}
    for label, path in detail_assignments:
        summary, rows = summarize_records(load_records(path), thresholds)
        summaries[label] = summary
        by_label[label] = rows
        input_receipts[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    run_assignments = [parse_assignment(raw, kind="run") for raw in args.run]
    if len({label for label, _ in run_assignments}) != len(run_assignments):
        fail("--run labels must be unique")
    training_summaries: dict[str, Any] = {}
    training_steps: dict[str, list[dict[str, Any]]] = {}
    for label, path in run_assignments:
        training_summaries[label], training_steps[label] = summarize_training_run(path)

    reference = args.reference or detail_assignments[0][0]
    payload = {
        "schema_version": 1,
        "protocol": "clasp-small-tsp-exact-leave-one-out-k8-analysis-v1",
        "evidence_scope": "development_only_not_publication_eligible",
        "primary_candidate_budget": 8,
        "raw_candidates_per_instance": 9,
        "subset_rule": "enumerate every one of nine leave-one-out size-8 subsets",
        "gap_thresholds": thresholds,
        "contract": {"path": str(CONTRACT_PATH), "sha256": sha256_file(CONTRACT_PATH)},
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "inputs": input_receipts,
        "policy_summaries": summaries,
        "paired_point_differences": paired_point_differences(
            by_label, reference=reference, thresholds=thresholds
        ),
        "training_summaries": training_summaries,
        "claim_gate": {
            "publication_eligible": False,
            "confirmatory": False,
            "sota_eligible": False,
            "reason": contract["limitations"],
        },
    }
    output_dir = (
        args.output_dir.absolute()
        if args.output_dir is not None
        else (PROJECT_ROOT / contract["paths"]["analysis_root"]).absolute()
    )
    write_outputs(output_dir, payload, by_label, training_steps)
    print(json.dumps({"output_dir": str(output_dir), "labels": sorted(by_label)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
