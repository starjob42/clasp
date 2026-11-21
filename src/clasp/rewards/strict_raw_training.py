"""Opt-in successor reward gates for OP/CVRP; no existing recipe imports this.

Both successor methods use the independent, typed physical scorer. Generated
IDs are never cast or repaired. Strictly invalid actions have no objective,
gap, or quality credit, while a separate nonpositive/fractional signal can
still distinguish partial constraint progress. These are changed reward heads,
not the released FOARL control or the frozen SFT-anchor pilot rewards.
"""
from __future__ import annotations

import ast
from collections import Counter
import math
from numbers import Real
import re

from clasp.baselines.sft_anchor_pilot import digest, numeric_instance, score_completion

PROTOCOL = "strict-raw-training-op-cvrp-successor-v2"
CHECKS = {
    "op": ("list_format", "integer_ids", "depot_and_minimum_visits",
           "unique_visits", "node_range", "raw_budget"),
    "cvrp": ("list_format", "integer_ids", "route_boundary", "node_range",
             "no_interior_depot", "unique_customers", "customer_coverage", "capacity"),
}


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _literal_list(task, text):
    """Diagnostic structure only: a noninteger leaf never enters geometry."""
    label = "Routes" if task == "cvrp" else "Route"
    match = re.search(r"\b" + label + r"\s*:\s*", text, re.IGNORECASE)
    start = text.find("[", match.end() if match else 0)
    if start < 0:
        return None
    depth = 0
    for end in range(start, len(text)):
        depth += (text[end] == "[") - (text[end] == "]")
        if depth == 0:
            try:
                value = ast.literal_eval(text[start:end + 1])
            except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError):
                return None
            return value if type(value) is list else None
    return None


def _constraint_fraction(task, text, instance, strict_score):
    coords, values, limit = numeric_instance(task, instance)
    value = _literal_list(task, text)
    shape = bool(value) and (task == "op" or all(
        type(route) is list and bool(route) for route in value))
    flags = dict.fromkeys(CHECKS[task], False)
    flags["list_format"] = bool(shape)
    nodes = (value if task == "op" else
             [node for route in value for node in route]) if shape else []
    typed = bool(shape) and all(type(node) is int for node in nodes)
    flags["integer_ids"] = typed
    if typed:
        flags["node_range"] = all(0 <= node < len(coords) for node in nodes)
        if task == "op":
            flags["depot_and_minimum_visits"] = len(value) >= 2 and value[0] == 0
            flags["unique_visits"] = len(value) == len(set(value))
            if flags["node_range"] and len(value) >= 2:
                distance = math.fsum(math.hypot(
                    coords[a][0] - coords[b][0], coords[a][1] - coords[b][1])
                    for a, b in zip(value, value[1:]))
                flags["raw_budget"] = math.isfinite(distance) and distance <= limit + 1e-9
        else:
            customers = [node for node in nodes if node != 0]
            flags["route_boundary"] = all(
                len(route) >= 2 and route[0] == 0 and route[-1] == 0 for route in value)
            flags["no_interior_depot"] = all(0 not in route[1:-1] for route in value)
            flags["unique_customers"] = all(count == 1 for count in Counter(customers).values())
            flags["customer_coverage"] = set(customers) == set(range(1, len(coords)))
            if flags["node_range"]:
                flags["capacity"] = all(math.fsum(values[node] for node in route if node != 0)
                                        <= limit + 1e-9 for route in value)
    fraction = sum(flags.values()) / len(flags)
    if (fraction == 1.0) != strict_score["feasible"]:
        raise ValueError("Strict diagnostic checks disagree with the physical scorer")
    return fraction, flags


def strict_reward_metadata(task, prompts, completions, inputs):
    """Validate references before sampling credit; return completion-aligned rows."""
    if task not in CHECKS:
        raise ValueError("Strict raw training supports only OP and CVRP")
    if (not isinstance(inputs, (list, tuple)) or not inputs or
            not isinstance(prompts, (list, tuple)) or
            not isinstance(completions, (list, tuple)) or
            len(prompts) != len(inputs) or len(completions) != len(inputs)):
        raise ValueError("Strict raw training requires nonempty aligned rows")
    required = ("task_type", "answer", "instance_json", "input")
    for row, completion in zip(inputs, completions):
        if (not isinstance(row, dict) or any(key not in row for key in required) or
                row["task_type"] != task or type(row["answer"]) is not str or
                type(completion) is not str):
            raise ValueError("Strict raw training has missing, nontext, or mismatched fields")
    # Complete strict reference validation for all rows before reward generation.
    from clasp.baselines.sft_anchor_pilot import reference_value
    for row in inputs:
        reference_value(task, row["instance_json"], row["answer"])
    metadata = []
    for row, completion in zip(inputs, completions):
        strict = score_completion(task, completion, row["instance_json"], row["answer"])
        fraction, checks = _constraint_fraction(task, completion, row["instance_json"], strict)
        gap = strict["signed_gap"]
        quality = 0.0
        if strict["feasible"]:
            z = -8.0 * gap
            quality = (1.0 / (1.0 + math.exp(-z)) if z >= 0 else
                       math.exp(z) / (1.0 + math.exp(z)))
        metadata.append({
            "protocol": PROTOCOL, "task": task, "feasible": strict["feasible"],
            "objective": strict["objective"], "signed_gap": gap,
            "reference": strict["reference_objective"], "quality_utility": quality,
            "strict_constraint_fraction": fraction, "strict_constraint_checks": checks,
            "violations": list(strict["violations"]),
            "constraint_view": "strict_typed_no_repair",
            "prompt_key": digest([task, row["instance_json"], row["input"], row.get("prompt")]),
            "completion_sha256": digest(completion),
        })
    return metadata


def _validate_metadata(metadata):
    if not isinstance(metadata, (list, tuple)) or not metadata:
        raise ValueError("Expected nonempty strict metadata")
    for row in metadata:
        if (not isinstance(row, dict) or row.get("protocol") != PROTOCOL or
                row.get("task") not in CHECKS or type(row.get("feasible")) is not bool):
            raise ValueError("Changed strict raw training metadata contract")
        fraction = _finite(row.get("strict_constraint_fraction"), "constraint fraction")
        quality = _finite(row.get("quality_utility"), "quality utility")
        if not 0 <= fraction <= 1 or not 0 <= quality <= 1:
            raise ValueError("Strict metadata fractions must be within [0,1]")
        checks = row.get("strict_constraint_checks")
        if (not isinstance(checks, dict) or set(checks) != set(CHECKS[row["task"]]) or
                any(type(flag) is not bool for flag in checks.values()) or
                sum(checks.values()) / len(checks) != fraction):
            raise ValueError("Changed strict diagnostic checks")
        if _finite(row.get("reference"), "reference") <= 0:
            raise ValueError("Strict reference must be positive")
        if row["feasible"]:
            if fraction != 1 or row.get("objective") is None or row.get("signed_gap") is None:
                raise ValueError("Feasible metadata requires a verified objective and gap")
            if _finite(row["objective"], "objective") < 0:
                raise ValueError("Strict objective must be nonnegative")
            _finite(row["signed_gap"], "signed gap")
        elif fraction == 1 or row.get("objective") is not None or row.get("signed_gap") is not None or quality != 0:
            raise ValueError("Invalid outputs cannot receive feasible objective or quality credit")


def gate_rloo_rewards(metadata, canonical_rewards):
    """Keep old negative grading; mask false positives with strict fraction - 1.

    Valid quality is recomputed with the same sigmoid slope eight. The new
    fallback is explicit: six equally weighted OP or eight CVRP strict checks,
    including parse/type checks, supply graded credit in [-1,0).
    """
    _validate_metadata(metadata)
    if not isinstance(canonical_rewards, (list, tuple)) or len(canonical_rewards) != len(metadata):
        raise ValueError("RLOO rewards must align with strict metadata")
    result = []
    for row, old in zip(metadata, canonical_rewards):
        old = _finite(old, "canonical reward")
        if not -1 <= old <= 2:
            raise ValueError("Canonical RLOO reward must be bounded by [-1,2]")
        if row["feasible"]:
            result.append(1.0 + row["quality_utility"])
        else:
            result.append(old if old < 0 else row["strict_constraint_fraction"] - 1.0)
    return result


def gate_foarl_reward_heads(metadata, optimality, feasibility_fraction):
    """Changed heads: invalid quality zero, graded feasibility capped below one."""
    _validate_metadata(metadata)
    if any(not isinstance(head, (list, tuple)) or len(head) != len(metadata)
           for head in (optimality, feasibility_fraction)):
        raise ValueError("FOARL heads must align with strict metadata")
    quality, graded = [], []
    for row, opt, fraction in zip(metadata, optimality, feasibility_fraction):
        opt = _finite(opt, "FOARL optimality")
        fraction = _finite(fraction, "FOARL fractional feasibility")
        if opt < 0 or not 0 <= fraction <= 1:
            raise ValueError("FOARL heads require nonnegative quality and [0,1] fractional feasibility")
        quality.append(opt if row["feasible"] else 0.0)
        graded.append(fraction if row["feasible"] else min(
            fraction, row["strict_constraint_fraction"]))
    return {"optimality": quality, "feasibility_fraction": graded,
            "strict_feasible": [row["feasible"] for row in metadata],
            "protocol": PROTOCOL, "method_label": "FOARL strict-aligned successor heads"}


def strict_rloo_reward_batch(task, prompts, completions, inputs):
    """Opt-in adapter; the caller still seals the existing graded environment."""
    metadata = strict_reward_metadata(task, prompts, completions, inputs)
    from clasp.rewards.quality_signal import reward_batch
    canonical = reward_batch(task, prompts, completions, mode="reference_sigmoid_v2", **{
        key: [row[key] for row in inputs]
        for key in ("task_type", "answer", "instance_json", "input")})
    rewards = gate_rloo_rewards(metadata, canonical)
    return {"rewards": rewards,
            "metadata": [{**row, "anchor_reward": reward} for row, reward in zip(metadata, rewards)],
            "protocol": PROTOCOL, "method_label": "RLOO strict-aligned successor reward"}


def strict_aligned_foarl_reward_batch(task, prompts, completions, inputs):
    """Apply released quality formulas to the shared strictly verified objective.

    Strictly valid compact/spaced grammars both receive full feasibility credit.
    OP uses 2*objective/max(.1,reference); CVRP uses reference/objective. Neither
    raw IDs nor declared objectives are sent through the released quality parser.
    Invalid fractional credit keeps the released partial head subject to strict
    diagnostics. This is a changed successor, not the released FOARL control.
    """
    metadata = strict_reward_metadata(task, prompts, completions, inputs)
    from clasp.baselines.llmcosolver_f638_training import FOARL_F638_REWARD_FUNCS
    kwargs = {key: [row[key] for row in inputs]
              for key in ("task_type", "answer", "instance_json")}
    fractions = FOARL_F638_REWARD_FUNCS[1](prompts, completions, **kwargs)
    optimality = [0.0] * len(metadata)
    for index, row in enumerate(metadata):
        if row["feasible"]:
            if task == "cvrp" and row["objective"] <= 0:
                raise ValueError("Strict valid CVRP objective must be positive under a positive reference")
            optimality[index] = (2.0 * row["objective"] / max(.1, row["reference"])
                                if task == "op" else row["reference"] / row["objective"])
            fractions[index] = 1.0
    return {**gate_foarl_reward_heads(metadata, optimality, fractions), "metadata": metadata}
