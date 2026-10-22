"""Prospective pure credit for a best-of-eight target from nine IID responses.

The repository already has best-of-K/exact-LOO credit in ``clasp_math``. This
helper makes a different target explicit: average the maxima of all nine
eight-response subsets, then subtract a focal-response-independent baseline.
It does not alter any trainer or frozen experiment.

The score identity requires scores from the actual sampling distribution.
The pinned native GRPO helper divides logits by the sampling temperature before
log-softmax, so the current temperature .7 is already aligned. Its scores omit
the top-p/top-k support restriction and renormalization used during generation.
The identity below therefore does not establish an unbiased gradient for that
filtered recipe. Clipping, KL, and further updates require separate analysis.
"""
from __future__ import annotations

import math
from numbers import Real
import re

PROTOCOL = "prospective-best8-from9-symmetric-credit-v1"
NUM_RESPONSES = 9
TARGET_K = 8
UTILITY_ABS_TOL = 1e-12
UTILITY_REL_TOL = 0.0


def best8_from9_coefficients(utilities):
    """Return detached scalar coefficients for complete groups of nine.

    Let H9 = sum_j max_{i != j} u_i / 9 and b_i = max_{k != i} u_k.
    A native mean-of-nine score loss receives coefficients 9*(H9-b_i):
    the unique best response gets 8*Delta, others get -Delta, and a top tie
    produces zero. Here Delta is the gap between the top two utilities.

    Since E[H9]=E[max(u1,...,u8)] and b_i is independent of focal action i,
    E[mean_i coefficient_i * score_i] equals the gradient of expected best8
    utility under IID draws and a matched on-policy score. Invalid utility
    must be zero; a negative coefficient on an invalid response is baseline
    pressure, not feasible quality credit. The graded RLOO anchor remains a
    separate objective and provides signal when this sparse term is zero.
    """
    if (not isinstance(utilities, (list, tuple)) or not utilities or
            len(utilities) % NUM_RESPONSES):
        raise ValueError("Best8-from9 credit requires complete nonempty groups of nine")
    values = []
    for value in utilities:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Utilities must be finite numbers in [0,1]")
        try:
            value = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError("Utilities must be finite numbers in [0,1]") from exc
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Utilities must be finite numbers in [0,1]")
        values.append(value)
    coefficients, baselines, groups = [], [], []
    for start in range(0, len(values), NUM_RESPONSES):
        row = values[start:start + NUM_RESPONSES]
        top, second = sorted(row, reverse=True)[:2]
        margin = top - second
        h9 = (TARGET_K * top + second) / NUM_RESPONSES
        unique = margin > 0
        coefficients.extend((TARGET_K * margin if value == top else -margin)
                            if unique else 0.0 for value in row)
        baselines.extend(second if unique and value == top else top for value in row)
        groups.append({"h9": h9, "top_utility": top, "second_utility": second,
                       "margin": margin, "active": unique})
    return {"protocol": PROTOCOL, "num_responses": NUM_RESPONSES, "target_k": TARGET_K,
            "native_mean_loss_scale": NUM_RESPONSES, "coefficients": coefficients,
            "focal_independent_baselines": baselines, "groups": groups,
            "sampling_score_identity_applies_to_current_pilot": False}


def strict_best8_from9_coefficients(metadata):
    """Admit one prompt/task per B9 group and independently recompute utility.

    Prompt keys are canonical nonempty lowercase 64-hex digests. Feasible
    stored utilities must match sigmoid(-8*signed_gap) with absolute tolerance
    1e-12 and relative tolerance zero; recomputed values supply the coefficients.
    The shared strict gate requires invalid stored utility to be exactly zero.
    Duplicate valid actions are allowed and receive the ordinary top-tie rule.
    """
    from clasp.rewards.strict_raw_training import _validate_metadata
    if (not isinstance(metadata, (list, tuple)) or not metadata or
            len(metadata) % NUM_RESPONSES):
        raise ValueError("Strict best8 metadata requires complete nonempty groups of nine")
    _validate_metadata(metadata)
    for start in range(0, len(metadata), NUM_RESPONSES):
        group = metadata[start:start + NUM_RESPONSES]
        keys = [row.get("prompt_key") for row in group]
        if any(type(key) is not str or re.fullmatch(r"[0-9a-f]{64}", key) is None for key in keys):
            raise ValueError("Strict best8 metadata requires valid nonempty lowercase 64-hex prompt keys")
        if len(set(keys)) != 1:
            raise ValueError("Strict best8 B9 groups must belong to one prompt")
        if len({row["task"] for row in group}) != 1:
            raise ValueError("Strict best8 B9 groups must belong to one task")
    utilities = []
    for row in metadata:
        utility = 0.0
        if row["feasible"]:
            z = -8.0 * float(row["signed_gap"])
            if z >= 0:
                utility = 1.0 / (1.0 + math.exp(-z))
            else:
                exp_z = math.exp(z)
                utility = exp_z / (1.0 + exp_z)
        if not math.isclose(float(row["quality_utility"]), utility,
                            abs_tol=UTILITY_ABS_TOL, rel_tol=UTILITY_REL_TOL):
            raise ValueError("Stored strict quality utility differs from independently recomputed feasibility/gap utility")
        utilities.append(utility)
    return best8_from9_coefficients(utilities)
