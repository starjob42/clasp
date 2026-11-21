"""Strict, per-response linear physical utility for a new one-shot experiment.

No reference objective enters the quality scale. OP divides collected prize by
all non-depot prize. Euclidean unlimited-fleet CVRP divides cost by the cost of
serving every customer on its own depot round trip, a triangle-inequality upper
bound. These instance-dependent scales change cross-instance weighting; they
are not claimed to equal mean relative reference gap. Invalid responses have
no objective or quality utility. No search, repair or best-of-k is performed.
"""
from __future__ import annotations

import math

from clasp.baselines.sft_anchor_pilot import digest, numeric_instance
from clasp.rewards.strict_raw_training import strict_reward_metadata

PROTOCOL = "strict-linear-physical-one-shot-quality-v1"
ROUND_OFF_TOLERANCE = 1e-12


def physical_scale(task, instance):
    coords, values, limit = numeric_instance(task, instance)
    try:
        if task == "op":
            bound = math.fsum(values[1:])
            kind = "all_non_depot_prize"
        else:
            # Complete original coordinates, not a rounded/nearest-neighbor graph.
            bound = 2.0 * math.fsum(math.hypot(point[0] - coords[0][0], point[1] - coords[0][1])
                                    for point in coords[1:])
            kind = "sum_individual_depot_round_trip_cost"
    except OverflowError as exc:
        raise ValueError("One-shot physical normalization overflowed") from exc
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("One-shot physical normalization requires a positive finite instance bound")
    return {"value": bound, "kind": kind, "task": task,
            "instance_sha256": digest({"task": task, "numeric_instance": [coords, values, limit]})}


def physical_utility(task, objective, scale):
    if (task not in ("op", "cvrp") or isinstance(objective, bool) or
            type(objective) not in (int, float) or not math.isfinite(objective) or objective < 0 or
            type(scale) is not dict or scale.get("task") != task or
            type(scale.get("value")) not in (int, float) or not math.isfinite(scale["value"]) or scale["value"] <= 0):
        raise ValueError("Finite strict physical objective and matching positive scale required")
    ratio = objective / scale["value"]
    if not math.isfinite(ratio) or ratio < 0 or ratio > 1 + ROUND_OFF_TOLERANCE:
        raise ValueError("Verified objective violates its declared physical bound; objective clipping is forbidden")
    # The only projection is a declared <=1e-12 floating-point bound overshoot.
    ratio = min(1.0, ratio)
    return ratio if task == "op" else 1.0 - ratio


def reward_batch(task, prompts, completions, inputs):
    """Return one shared scalar and an explicitly changed two-head decomposition.

    Valid reward is 1+utility in [1,2]. Invalid reward is the fraction of strict
    diagnostic checks passed minus1, in [-1,0). This is an expected bounded
    utility objective, not lexicographic feasibility or a greedy guarantee.
    """
    strict = strict_reward_metadata(task, prompts, completions, inputs)
    rows, rewards, quality_head, validity_head = [], [], [], []
    for source, raw in zip(strict, inputs):
        scale = physical_scale(task, raw["instance_json"])
        utility = physical_utility(task, source["objective"], scale) if source["feasible"] else 0.0
        validity = 1.0 if source["feasible"] else source["strict_constraint_fraction"] - 1.0
        reward = validity + utility
        if source["feasible"]:
            if not 1 <= reward <= 2:
                raise ValueError("Valid one-shot reward escaped [1,2]")
        elif not -1 <= reward < 0:
            raise ValueError("Invalid one-shot reward escaped [-1,0)")
        rows.append({"protocol": PROTOCOL, "task": task, "prompt_key": source["prompt_key"],
            "completion_sha256": source["completion_sha256"], "feasible": source["feasible"],
            "objective": source["objective"], "physical_scale": scale, "utility": utility,
            "strict_constraint_fraction": source["strict_constraint_fraction"],
            "strict_constraint_checks": source["strict_constraint_checks"], "violations": source["violations"],
            "reference": source["reference"], "reference_signed_gap_diagnostic": source["signed_gap"],
            "quality_head": utility, "validity_head": validity, "anchor_reward": reward,
            "constraint_view": "strict_typed_no_repair"})
        rewards.append(reward); quality_head.append(utility); validity_head.append(validity)
    return {"protocol": PROTOCOL, "rewards": rewards, "quality_head": quality_head,
            "validity_head": validity_head, "metadata": rows,
            "objective": "expected_per_response_strict_linear_physical_utility",
            "greedy_improvement_guaranteed": False, "reference_used_for_quality_scale": False}


def contract():
    return {"protocol": PROTOCOL, "tasks": ["op", "cvrp"], "feasible_reward": "1+physical_utility",
        "invalid_reward": "strict_check_fraction-1", "invalid_quality": 0.0,
        "validity_range": [-1.0, 1.0], "scalar_range": [-1.0, 2.0],
        "physical_utility_range": [0.0, 1.0], "round_off_tolerance": ROUND_OFF_TOLERANCE,
        "op_scale": "sum_non_depot_prize", "cvrp_scale": "2*sum_depot_customer_euclidean_distance",
        "reference_quality_scale": False, "best8_auxiliary": 0.0, "cardinal_auxiliary": 0.0,
        "repair": False, "search": False, "greedy_improvement_guaranteed": False,
        "limitations": ["Instance normalization weights tasks/instances differently from reference-relative mean gap.",
                        "Feasibility and quality share a bounded scalar; no lexicographic policy guarantee.",
                        "Expected sampled-response utility does not guarantee deterministic greedy improvement.",
                        "TSP and JSSP require separately verified normalization and training adapters."]}
