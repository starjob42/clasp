"""Strict-objective, beta-zero GRPO control for the fixed 400-update study.

Only the reward/profile interface is new. The existing weight-zero teacher
trainer delegates policy loss to stock TRL and restores checkpointing kwargs.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from clasp.baselines.foarl_f638_dev_20260907 import COMMON_PROMPT_SHA256
from clasp.baselines.foarl_teacher_dev import COMMON_DEFAULTS
from clasp.baselines.llmcosolver_f638_training import (
    validate_foarl_f638_dataset,
    validate_trl_foarl_implementation,
)
from clasp.rewards.co_tasks import cvrp_objective_reward, mis_objective_reward

METHOD_ID = "classic_grpo"
PROFILE_ID = "classic-grpo-strict-objective-400-development-20260909-v1"
REWARD_ENVIRONMENT = {
    "GRPO_GRADED_REWARD": "0",
    "GRPO_VERIFIED_IMPROVEMENT": "0",
    "CLASP_OBJECTIVE_QUALITY_MODE": "exp",
    "CLASP_OBJECTIVE_GAP_SLOPE": "3.0",
    "CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT": "0.0",
}
KL_ENVIRONMENT = ("CLASP_KL_REFERENCE_MODEL_PATH", "CLASP_KL_REFERENCE_RECEIPT")
CONFIG = {
    "beta": 0.0,
    "epsilon": 0.2,
    "epsilon_high": 0.2,
    "loss_type": "grpo",
    "scale_rewards": "group",
    "num_iterations": 1,
    "sync_ref_model": False,
    "reward_weights": [1.0],
    "importance_sampling_level": "token",
    "mask_truncated_completions": False,
    "use_liger_loss": False,
}


def validate_reward_environment() -> None:
    drift = {name: os.environ.get(name) for name, expected in REWARD_ENVIRONMENT.items()
             if os.environ.get(name) != expected}
    if drift:
        raise ValueError(f"Classic strict-objective environment differs: {drift}")
    if any(os.environ.get(name, "").strip() for name in KL_ENVIRONMENT):
        raise ValueError("Classic beta-zero control forbids an explicit KL reference")


def seal_reward_environment() -> dict[str, str]:
    """Reject inherited drift before establishing the explicit old reward recipe."""
    drift = {name: os.environ[name] for name, expected in REWARD_ENVIRONMENT.items()
             if name in os.environ and os.environ[name] != expected}
    if drift:
        raise ValueError(f"Inherited classic reward environment differs: {drift}")
    if any(os.environ.get(name, "").strip() for name in KL_ENVIRONMENT):
        raise ValueError("Classic beta-zero control forbids an explicit KL reference")
    os.environ.update(REWARD_ENVIRONMENT)
    return dict(REWARD_ENVIRONMENT)


def _strict_task_reward(task, prompts, completions, kwargs):
    validate_reward_environment()
    if not isinstance(prompts, (list, tuple)) or not isinstance(completions, (list, tuple)):
        raise ValueError("Classic reward requires aligned prompt/completion sequences")
    count = len(completions)
    if len(prompts) != count:
        raise ValueError("Classic prompt/completion counts differ")
    for name in ("task_type", "answer", "instance_json", "input"):
        values = kwargs.get(name)
        if not isinstance(values, (list, tuple)) or len(values) != count:
            raise ValueError(f"Classic reward requires one {name} per completion")
    if any(value != task for value in kwargs["task_type"]):
        raise ValueError("Classic reward received another specialist task")
    # These are the same wrapper settings as the old zero-dual grpo_z arm.
    # Never add the structure/constraint heads or let caller kwargs enable
    # infeasible partial credit. Preserve the objective head's factor two.
    local = dict(kwargs)
    local.update(apply_constraint_gate=True, allow_infeasible_partial_credit=False)
    function = cvrp_objective_reward if task == "cvrp" else mis_objective_reward
    rewards = function(list(prompts), list(completions), **local)
    if len(rewards) != count or any(type(v) not in (int, float) or not math.isfinite(v)
                                    for v in rewards):
        raise ValueError("Classic strict objective produced invalid rewards")
    return rewards


def classic_cvrp_strict_objective_reward(prompts, completions, **kwargs):
    return _strict_task_reward("cvrp", prompts, completions, kwargs)


def classic_mis_strict_objective_reward(prompts, completions, **kwargs):
    return _strict_task_reward("mis", prompts, completions, kwargs)


@dataclass(frozen=True)
class ClassicGRPO400Runtime:
    task: str
    source_binding: Mapping[str, Any] = field(default_factory=dict)
    diagnostic: bool = False
    requires_explicit_fixed_reference: bool = False
    # This existing hook selects stock TRL plus the shared checkpointing fix.
    # Weight zero executes no teacher construction, tokenization or forward.
    verified_teacher_extension: bool = True
    teacher_weight: float = 0.0

    def validate_args(self, args):
        if self.task not in ("cvrp", "mis") or tuple(args.tasks) != (self.task,):
            raise ValueError("Classic control requires exactly one declared CVRP/MIS task")
        if self.teacher_weight != 0.0 or self.requires_explicit_fixed_reference:
            raise ValueError("Classic control must remain teacher-free and beta zero")
        from clasp.data.prompts import JIANG_F638_INFERENCE_TEMPLATE
        if hashlib.sha256(JIANG_F638_INFERENCE_TEMPLATE.encode()).hexdigest() != COMMON_PROMPT_SHA256:
            raise ValueError("Classic control common prompt bytes changed")
        expected = {**COMMON_DEFAULTS, "classic_grpo_diagnostic": self.diagnostic,
                    "dtype": "bfloat16", "prompt_style": "jiang_f638", "num_train_epochs": 1,
                    "use_primal_dual": False, "use_clasp": False, "full_finetune": False,
                    "verify_gt_rewards": True, "grpo_beta": 0.0, "grpo_epsilon": 0.2,
                    "grpo_epsilon_high": 0.2, "grpo_loss_type": "grpo"}
        if self.diagnostic:
            expected["save_steps"] = 5
        drift = {key: (value, getattr(args, key, None)) for key, value in expected.items()
                 if getattr(args, key, None) != value}
        if drift:
            raise ValueError(f"Classic common recipe mismatch: {drift}")
        validate_reward_environment()
        if os.environ.get("WORLD_SIZE", "1") != "4" or os.environ.get("GRPO_DATA_SEED", "43") != "43":
            raise ValueError("Classic control requires four ranks and policy/data seed43")
        if os.environ.get("GRPO_STRICT_OPTIMIZER_RESUME") != "1":
            raise ValueError("Classic control requires strict optimizer resume")
        stop = os.environ.get("GRPO_STOP_AT_GLOBAL_STEP", "")
        if not stop.isdigit() or int(stop) not in ((5, 10) if self.diagnostic else (100, 200, 300, 400)):
            raise ValueError("Classic control requires a declared chunk boundary")
        if getattr(args, "resolved_generation_batch_size", 36) != 36:
            raise ValueError("Classic rollout batch must remain36")
        if getattr(args, "resolved_steps_per_generation", None) not in (None, 3):
            raise ValueError("Classic rollout buffer must span three microbatches")

    def grpo_config_overrides(self, grpo_signature):
        if missing := set(CONFIG) - set(grpo_signature):
            raise ValueError(f"Stock TRL cannot express classic recipe: {sorted(missing)}")
        return copy.deepcopy(CONFIG)

    def validate_grpo_config(self, config):
        expected = {**CONFIG, "num_generations": 9, "max_steps": 400,
                    "generation_batch_size": 36, "gradient_accumulation_steps": 3,
                    "steps_per_generation": 3, "ignore_data_skip": False}
        for name, value in expected.items():
            if getattr(config, name, None) != value:
                raise ValueError(f"Resolved classic GRPOConfig.{name} differs from {value!r}")

    def reward_functions(self):
        if self.task not in ("cvrp", "mis"):
            raise ValueError("Unsupported classic task")
        return (classic_cvrp_strict_objective_reward if self.task == "cvrp"
                else classic_mis_strict_objective_reward,)

    def validate_dataset(self, dataset, *, label):
        # Reuse only the common row-schema check, not FOARL's reward heads.
        if "input" not in getattr(dataset, "column_names", ()):
            raise ValueError("Classic reward requires the prepared input column")
        receipt = validate_foarl_f638_dataset(dataset, label=label)
        if set(receipt["task_counts"]) - {self.task}:
            raise ValueError("Classic dataset contains another specialist task")
        if any(not isinstance(dataset[i]["input"], str) for i in range(len(dataset))):
            raise ValueError("Classic prepared input values must be text")
        if label == "train" and len(dataset) < 1600:
            raise ValueError("Classic400 requires at least1600 retained training prompts")
        return receipt

    def validate_trainer_implementation(self, trainer_class):
        # This checker audits stock sample-std+epsilon and GRPO loss reduction.
        return validate_trl_foarl_implementation(trainer_class)

    def metadata(self):
        if self.source_binding.get("status") != "passed_frozen_manifest_successor_source":
            raise ValueError("Classic control requires a verified frozen source manifest")
        return {
            "method_id": METHOD_ID, "profile_id": PROFILE_ID,
            "study_scope": "distributed_diagnostic" if self.diagnostic else "development_only",
            "reproduction_claim": "matched_classic_control_not_official_reproduction",
            "task": self.task, "beta": 0.0, "teacher_weight": 0.0,
            "source_binding": copy.deepcopy(dict(self.source_binding)),
            "reward_heads": [self.reward_functions()[0].__name__], "reward_weights": [1.0],
            "reward_semantics": "legacy_grpo_z_strict_objective_only_exp3_declared0_no_infeasible_credit",
            "reward_environment": dict(REWARD_ENVIRONMENT), "training_rewards_use_reference_labels": True,
            "algorithm_config": copy.deepcopy(CONFIG),
            "common_budget": {**COMMON_DEFAULTS, "world_size": 4,
                              "save_steps": 5 if self.diagnostic else 100},
            "prompt_sha256": COMMON_PROMPT_SHA256, "no_reason": True,
            "trainer_path": "stock_TRL_via_verified_teacher_weight_zero_with_checkpointing_restore",
            "continuation": {"fixed_scheduler_horizon": 400, "strict_optimizer_resume": True,
                             "kl_reference": "none_beta_zero", "native_full_state_parity_required": True},
        }


def build_runtime(source_binding, task, diagnostic=False):
    return ClassicGRPO400Runtime(task=task, source_binding=copy.deepcopy(dict(source_binding)),
                                diagnostic=diagnostic)
