"""Versioned B9 reward ablation; the existing external-baseline recipe is fixed.

This profile distinguishes legacy rewards, invalid-diagnostic corrections, and
reference-normalized feasible quality. It changes no evaluation metric and
never claims an improvement before a matched policy comparison.
"""
from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass

from clasp.baselines.external_baseline_dev import (
    SUPPORTED_CO_TASKS,
    ExternalBaselineRuntime,
)
from clasp.rewards.quality_signal import (
    QUALITY_SLOPE,
    REWARD_MODES,
    reward_batch,
    validate_training_reference,
)

PROFILE_ID = "quality-signal-b9-development-20260916-v1"


def _head(task, mode):
    def reward(prompts, completions, **kwargs):
        return reward_batch(task, prompts, completions, mode=mode, **kwargs)

    reward.__name__ = f"quality_signal_{mode}_{task}_reward"
    reward.__qualname__ = reward.__name__
    return reward


REWARD_HEADS = {(task, mode): _head(task, mode)
                for task in SUPPORTED_CO_TASKS for mode in REWARD_MODES}
globals().update({head.__name__: head for head in REWARD_HEADS.values()})


def validate_reward_contract(value):
    expected = {"profile_id", "task", "mode", "quality_slope", "reward_head",
                "source_manifest_sha256"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Invalid quality-signal reward contract fields")
    if (value["profile_id"] != PROFILE_ID or value["task"] not in SUPPORTED_CO_TASKS
            or value["mode"] not in REWARD_MODES):
        raise ValueError("Unknown quality-signal reward profile/task/mode")
    if (type(value["quality_slope"]) not in (int, float)
            or value["quality_slope"] != QUALITY_SLOPE
            or value["reward_head"] != REWARD_HEADS[value["task"], value["mode"]].__name__
            or not isinstance(value["source_manifest_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["source_manifest_sha256"]) is None):
        raise ValueError("Changed quality-signal reward contract")
    return copy.deepcopy(value)


def check_resume_reward_contract(expected, payload):
    """Prevent both legacy→new and new→legacy cross-recipe continuation."""
    observed = payload.get("reward_contract")
    if expected is not None:
        validate_reward_contract(expected)
    if observed is not None:
        validate_reward_contract(observed)
    if observed != expected:
        raise ValueError("Resume reward contract differs; start a fresh matched SFT run")


@dataclass(frozen=True)
class QualitySignalRuntime(ExternalBaselineRuntime):
    quality_reward: str = "reference_sigmoid_v2"

    def validate_args(self, args):
        super().validate_args(args)
        if self.num_generations != 9:
            raise ValueError("Quality-signal attribution fixes B9; no B1 prompt-coverage change")
        if (self.quality_reward not in REWARD_MODES
                or getattr(args, "quality_reward", None) != self.quality_reward):
            raise ValueError("Quality-signal reward mode differs from the declaration")
        self.reward_contract()

    def reward_functions(self):
        return (REWARD_HEADS[self.task, self.quality_reward],)

    def reward_contract(self):
        return validate_reward_contract({
            "profile_id": PROFILE_ID, "task": self.task, "mode": self.quality_reward,
            "quality_slope": QUALITY_SLOPE,
            "reward_head": self.reward_functions()[0].__name__,
            "source_manifest_sha256": self.source_binding.get("manifest", {}).get("sha256"),
        })

    def validate_dataset(self, dataset, *, label):
        result = super().validate_dataset(dataset, label=label)
        if label == "train" and self.quality_reward == "reference_sigmoid_v2":
            # Verify every retained training target before any optimization.
            # No evaluation target is consulted by this training preflight.
            for index, row in enumerate(dataset):
                try:
                    validate_training_reference(self.task, row.get("answer"),
                                                row.get("instance_json"), row.get("input", ""))
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"Invalid {self.task} training reference at row {index}") from exc
            result = {**result, "quality_signal_training_references": {
                "status": "all_retained_training_references_canonically_verified",
                "count": len(dataset), "declared_objectives_used": False,
                "reference_optimality_claimed": False,
            }}
        return result

    def metadata(self):
        result = super().metadata()
        corrected = self.quality_reward != "legacy"
        calibrated = self.quality_reward == "reference_sigmoid_v2"
        result.update(
            profile_id=PROFILE_ID,
            method_id=f"{self.advantage_estimator}_b9_{self.quality_reward}",
            reward_contract=self.reward_contract(),
            reward_semantics=self.quality_reward,
            invalid_reward=("strict_negative_violation_with_conservative_unavailable_diagnostics"
                            if corrected else result["invalid_reward"]),
            feasible_quality=("sigmoid(-8*signed_reference_gap)"
                              if calibrated else result["feasible_quality"]),
            quality_calibration={
                "enabled": calibrated, "slope": QUALITY_SLOPE,
                "reference_source": "aligned_canonically_verified_train_answer" if calibrated else "legacy",
                "reference_dependency_added_for_task": calibrated and self.task in ("mis", "op"),
                "invalid_diagnostic_fix_applies": corrected and self.task in ("op", "jssp"),
                "all_invalid_zero_signal_guaranteed_solved": False,
                "method_superiority_established": False,
            },
            claim_limit="versioned_development_candidate_requires_matched_policy_evaluation",
        )
        return result


def build_runtime(source_binding, args):
    return QualitySignalRuntime(
        task=str(args.tasks[0]), advantage_estimator=args.advantage_estimator,
        num_generations=args.num_generations, baseline_initial=args.external_baseline_initial,
        baseline_scale=args.external_baseline_scale, diagnostic=args.external_baseline_diagnostic,
        world_size=int(os.environ.get("WORLD_SIZE", "1")), seed=args.seed,
        source_binding=copy.deepcopy(dict(source_binding)), quality_reward=args.quality_reward,
    )
