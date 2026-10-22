"""Prospective, verified quality-credit ablations on a calibrated RLOO anchor.

This module does not change any existing reward head. The caller must bind the
training dataset and source in its run contract; references are recomputed from
each aligned training answer, never read from a declared objective field.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real

PROTOCOL = "feasible-quality-credit-20260919-v1"
METHODS = ("control", "cardinal", "scalar_reweight", "midrank")


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def make_advantage_contract(method, *, gap_scale=0.05, weight=0.25):
    """Return the fixed initial experiment contract; control adds exactly zero."""
    if method not in METHODS:
        raise ValueError("Unknown feasible-quality method")
    if _finite(gap_scale, "gap_scale") != 0.05:
        raise ValueError("This protocol fixes gap_scale=0.05")
    if _finite(weight, "weight") != 0.25:
        raise ValueError("This protocol fixes noncontrol weight=0.25")
    return {
        "protocol": PROTOCOL, "method": method, "num_generations": 9,
        "gap_scale": 0.05, "weight": 0.0 if method == "control" else 0.25,
        "denominator": "all_siblings",
        "anchor": "reference_sigmoid_v2_rloo",
        "quality": "negative_signed_training_reference_gap",
        "scalar_invalid_quality": 0.0,
        "midrank": "feasible_pairwise_sign",
    }


def validate_advantage_contract(value):
    if not isinstance(value, dict) or value.get("method") not in METHODS:
        raise ValueError("Invalid feasible-quality advantage contract")
    expected = make_advantage_contract(value["method"])
    # JSON comparison rejects bools masquerading as numeric fields and extras.
    if json.dumps(value, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("Changed feasible-quality advantage contract")
    return expected


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def verified_reward_metadata(task, prompts, completions, inputs):
    """Reverify local rows in their reward-call order, including the anchor.

    An invalid sampled output never receives an objective or gap. Its training
    reference is still validated. Returning its original negative anchor reward
    makes exact comparison against TRL's globally gathered reward possible.
    """
    from clasp.rewards.quality_signal import (
        _context, _objective, _parse, reward_batch, validate_training_reference,
    )

    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Expected nonempty completion-aligned training rows")
    if len(prompts) != len(inputs) or len(completions) != len(inputs):
        raise ValueError("Reward metadata is not completion-aligned")
    keys = ("task_type", "answer", "instance_json", "input")
    if any(not isinstance(row, dict) or any(key not in row for key in keys)
           or row["task_type"] != task for row in inputs):
        raise ValueError("Reward metadata has missing or mismatched training fields")
    kwargs = {key: [row[key] for row in inputs] for key in keys}
    anchors = reward_batch(task, prompts, completions, mode="reference_sigmoid_v2", **kwargs)
    records = []
    for row, completion, anchor in zip(inputs, completions, anchors):
        reference = validate_training_reference(task, row["answer"], row["instance_json"], row["input"])
        context = _context(task, row["instance_json"], row["input"])
        objective = _objective(task, _parse(task, completion), context)
        feasible = objective is not None
        gap = None
        if feasible:
            ratio = objective / reference
            gap = _finite(ratio - 1.0 if task in ("tsp", "cvrp", "jssp") else 1.0 - ratio,
                          "verified signed gap")
        if (anchor >= 0) != feasible:
            raise ValueError("Canonical feasibility and anchor reward disagree")
        records.append({
            "feasible": feasible, "signed_gap": gap, "objective": objective,
            "reference": reference, "anchor_reward": _finite(anchor, "anchor reward"),
            "prompt_key": _digest([task, row.get("instance_json"), row.get("input"), row.get("prompt")]),
            "completion_sha256": _digest(completion),
        })
    return records


def decompose_advantages(base_advantages, metadata, contract):
    """Compute globally ordered coefficients; no current-batch normalization.

    Cardinal is f_i sum_j f_j(q_i-q_j)/(B-1). Scalar-reweight instead
    applies ordinary RLOO to f*q, assigning invalid outputs scalar quality zero.
    Midrank replaces each feasible pair difference with its sign. Control
    returns the original base values with no additional arithmetic.
    """
    contract = validate_advantage_contract(contract)
    b = contract["num_generations"]
    if (not isinstance(metadata, (list, tuple)) or not metadata or len(metadata) % b
            or not isinstance(base_advantages, (list, tuple))
            or len(base_advantages) != len(metadata)):
        raise ValueError("Quality coefficients require aligned complete B9 groups")
    base = [_finite(value, "base advantage") for value in base_advantages]
    feasible, quality = [], []
    for row in metadata:
        if not isinstance(row, dict) or type(row.get("feasible")) is not bool or "signed_gap" not in row:
            raise ValueError("Quality metadata requires strict feasibility and signed_gap")
        f, gap = row["feasible"], row["signed_gap"]
        if not f and gap is not None:
            raise ValueError("Invalid candidates must have signed_gap=None")
        feasible.append(f)
        quality.append(_finite(-_finite(gap, "signed_gap") / contract["gap_scale"], "quality") if f else 0.0)
    auxiliary, feasible_counts = [], []
    method = contract["method"]
    try:
        for start in range(0, len(base), b):
            flags, q = feasible[start:start + b], quality[start:start + b]
            feasible_counts.append(sum(flags))
            for i in range(b):
                if method == "control":
                    coefficient = 0.0
                elif method == "scalar_reweight":
                    coefficient = q[i] - math.fsum(q[j] for j in range(b) if j != i) / (b - 1)
                elif not flags[i]:
                    coefficient = 0.0
                elif method == "cardinal":
                    coefficient = math.fsum(q[i] - q[j] for j in range(b)
                                            if j != i and flags[j]) / (b - 1)
                else:
                    coefficient = math.fsum((q[i] > q[j]) - (q[i] < q[j])
                                            for j in range(b) if j != i and flags[j]) / (b - 1)
                auxiliary.append(_finite(coefficient, "quality coefficient"))
        combined = (list(base) if method == "control" else
                    [_finite(a + contract["weight"] * q, "combined advantage")
                     for a, q in zip(base, auxiliary)])
    except (OverflowError, ValueError) as exc:
        raise ValueError("Quality coefficient computation failed") from exc
    n, groups = len(base), len(feasible_counts)
    diagnostics = {
        "quality/feasible_candidate_fraction": sum(feasible) / n,
        "quality/all_invalid_group_fraction": sum(m == 0 for m in feasible_counts) / groups,
        "quality/one_feasible_group_fraction": sum(m == 1 for m in feasible_counts) / groups,
        "quality/active_group_fraction": sum(any(v != 0 for v in auxiliary[s:s + b])
                                             for s in range(0, n, b)) / groups,
        "quality/auxiliary_abs_mean": math.fsum(abs(v) for v in auxiliary) / n,
        "quality/weighted_auxiliary_abs_mean": contract["weight"] * math.fsum(abs(v) for v in auxiliary) / n,
        "quality/base_abs_mean": math.fsum(abs(v) for v in base) / n,
        "quality/combined_abs_mean": math.fsum(abs(v) for v in combined) / n,
        "quality/auxiliary_negative_feasible_fraction": sum(f and v < 0 for f, v in zip(feasible, auxiliary)) / n,
        "quality/combined_negative_feasible_fraction": sum(f and v < 0 for f, v in zip(feasible, combined)) / n,
        "quality/invalid_nonzero_auxiliary_fraction": sum(not f and v != 0 for f, v in zip(feasible, auxiliary)) / n,
    }
    if any(not math.isfinite(v) for v in diagnostics.values()):
        raise ValueError("Nonfinite quality diagnostics")
    return {"base": base, "quality": auxiliary, "combined": combined, "diagnostics": diagnostics}
