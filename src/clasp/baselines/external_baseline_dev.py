"""Opt-in raw-policy RLOO/external-baseline development experiment.

This profile preserves the existing graded CVRP/MIS reward and extends the
same strict-feasibility, bounded-reward contract to TSP, OP and JSSP.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from clasp.baselines.foarl_f638_dev_20260907 import COMMON_PROMPT_SHA256
from clasp.baselines.llmcosolver_f638_training import validate_foarl_f638_dataset
from clasp.rewards.co_tasks import cvrp_objective_reward, mis_objective_reward
from clasp.rewards.external_baseline import (
    SUPPORTED_CO_TASKS, TASK_SIZE_SEMANTICS, baseline_instance_size,
)
from clasp.rewards.external_baseline_tasks import additional_task_graded_reward, validate_graded_instance

PROFILE_ID = "external-baseline-graded-raw-development-20260910-v1"
COMMON_DEFAULTS = {
    "no_reason": True, "prompt_style": "jiang_f638", "dtype": "bfloat16",
    "num_generations": 9, "generation_batch_size": 36,
    "per_device_train_batch_size": 3, "per_device_eval_batch_size": 9,
    "gradient_accumulation_steps": 3, "learning_rate": 1e-6,
    "optim": "adamw_8bit", "weight_decay": 0.01, "warmup_ratio": 0.03,
    "lr_scheduler_type": "linear", "seed": 43, "max_steps": 100,
    "save_steps": 50, "logging_steps": 1, "max_prompt_length": 29_768,
    "max_completion_length": 3_000, "generation_temperature": 1.0,
    "generation_top_p": 1.0, "generation_top_k": 0,
    "generation_repetition_penalty": 1.0, "lora_r": 64, "lora_alpha": 64,
    "lora_dropout": 0.0, "num_train_epochs": 1, "use_primal_dual": False,
    "use_clasp": False, "full_finetune": False, "verify_gt_rewards": True,
    "grpo_beta": 0.0, "grpo_epsilon": 0.2, "grpo_epsilon_high": 0.2,
    "grpo_loss_type": "dr_grpo", "eval_ratio": 0.0,
}
REWARD_ENVIRONMENT = {
    "GRPO_GRADED_REWARD": "1", "GRPO_VERIFIED_IMPROVEMENT": "0",
    "GRPO_VERIFIED_IMPROVEMENT_WEIGHT": "0.0",
    "CLASP_OBJECTIVE_QUALITY_MODE": "exp", "CLASP_OBJECTIVE_GAP_SLOPE": "3.0",
    "CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT": "0.0",
}
CONFIG = {
    "beta": 0.0, "epsilon": 0.2, "epsilon_high": 0.2,
    "loss_type": "dr_grpo", "scale_rewards": "none", "num_iterations": 1,
    "sync_ref_model": False, "reward_weights": [1.0],
    "importance_sampling_level": "token", "mask_truncated_completions": False,
    "use_liger_loss": False, "use_vllm": False,
}
SOURCE_REQUIRED_FILES = (
    "src/clasp/baselines/external_baseline_dev.py",
    "src/clasp/entrypoints/train_external_baseline_grpo.py",
    "src/clasp/entrypoints/train_grpo.py",
    "src/clasp/rewards/co_tasks.py",
    "src/clasp/models/training_fidelity.py",
)


def verify_source_manifest(manifest_path, expected_sha256, *, source_root=None):
    """Bind this opt-in profile to a caller-declared frozen source inventory."""
    root = (Path(source_root) if source_root else Path(__file__).resolve().parents[3]).resolve()
    manifest = Path(manifest_path).absolute()
    if manifest.is_symlink() or manifest.parent.resolve() != root or not manifest.is_file():
        raise ValueError("External-baseline manifest must be a regular file in its source root")
    raw = manifest.read_bytes()
    if not re.fullmatch(r"[a-f0-9]{64}", str(expected_sha256)) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("External-baseline source-manifest digest mismatch")
    identities = {}
    for line in raw.decode().splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        if match is None:
            raise ValueError("External-baseline source manifest requires sha256sum lines")
        digest, name = match.groups()
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != name or name in identities:
            raise ValueError(f"Invalid or repeated source path: {name}")
        path = root / name
        if path.is_symlink() or path.resolve() != path or not path.is_file():
            raise ValueError(f"Source must be an unlinked regular file: {name}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"External-baseline source mismatch: {name}")
        identities[name] = {"sha256": digest, "size_bytes": len(payload)}
    required = set(SOURCE_REQUIRED_FILES)
    required.update(p.relative_to(root).as_posix() for p in (root / "src/clasp").rglob("*.py"))
    if missing := required - identities.keys():
        raise ValueError(f"Source manifest omits executing Python files: {sorted(missing)}")
    return {
        "status": "passed_frozen_manifest_successor_source", "source_root": str(root),
        "manifest": {"path": str(manifest), "sha256": expected_sha256},
        "verified_file_count": len(identities),
        "required_files": {name: identities[name] for name in SOURCE_REQUIRED_FILES},
    }


def seal_reward_environment():
    drift = {name: os.environ[name] for name, expected in REWARD_ENVIRONMENT.items()
             if name in os.environ and os.environ[name] != expected}
    if drift:
        raise ValueError(f"Inherited external-baseline reward configuration differs: {drift}")
    for name in ("CLASP_KL_REFERENCE_MODEL_PATH", "CLASP_KL_REFERENCE_RECEIPT"):
        if os.environ.get(name, "").strip():
            raise ValueError("Beta-zero external-baseline experiment forbids an explicit KL reference")
    os.environ.update(REWARD_ENVIRONMENT)
    return dict(REWARD_ENVIRONMENT)


def _graded_reward(task, prompts, completions, kwargs):
    if any(os.environ.get(k) != v for k, v in REWARD_ENVIRONMENT.items()):
        raise ValueError("External-baseline graded reward environment changed")
    if not isinstance(prompts, (list, tuple)) or not isinstance(completions, (list, tuple)):
        raise ValueError("Graded reward needs aligned prompt and completion lists")
    count = len(completions)
    if len(prompts) != count:
        raise ValueError("Graded reward prompt/completion counts differ")
    for key in ("task_type", "answer", "instance_json", "input"):
        if not isinstance(kwargs.get(key), (list, tuple)) or len(kwargs[key]) != count:
            raise ValueError(f"Graded reward requires one {key} per completion")
    if any(t != task for t in kwargs["task_type"]):
        raise ValueError("Graded specialist reward received a different task")
    local = dict(kwargs)
    local.update(apply_constraint_gate=True, allow_infeasible_partial_credit=False)
    if task in ("cvrp", "mis"):
        function = cvrp_objective_reward if task == "cvrp" else mis_objective_reward
        values = function(list(prompts), list(completions), **local)
    else:
        values = additional_task_graded_reward(task, list(prompts), list(completions), **local)
    if len(values) != count or any(type(v) not in (int, float) or not math.isfinite(v) or not -1 <= v <= 2 for v in values):
        raise ValueError("Existing graded objective returned invalid/out-of-range rewards")
    return values


def external_cvrp_graded_reward(prompts, completions, **kwargs):
    return _graded_reward("cvrp", prompts, completions, kwargs)


def external_mis_graded_reward(prompts, completions, **kwargs):
    return _graded_reward("mis", prompts, completions, kwargs)


def external_tsp_graded_reward(prompts, completions, **kwargs):
    return _graded_reward("tsp", prompts, completions, kwargs)


def external_op_graded_reward(prompts, completions, **kwargs):
    return _graded_reward("op", prompts, completions, kwargs)


def external_jssp_graded_reward(prompts, completions, **kwargs):
    return _graded_reward("jssp", prompts, completions, kwargs)


TASK_GRADED_REWARDS = {
    "tsp": external_tsp_graded_reward, "cvrp": external_cvrp_graded_reward,
    "op": external_op_graded_reward, "mis": external_mis_graded_reward,
    "jssp": external_jssp_graded_reward,
}


@dataclass(frozen=True)
class ExternalBaselineRuntime:
    task: str
    advantage_estimator: str
    num_generations: int
    baseline_initial: float = 0.0
    baseline_scale: float = 1.0
    diagnostic: bool = False
    world_size: int = 4
    seed: int = 43
    source_binding: Mapping[str, Any] = field(default_factory=dict)
    external_baseline_extension: bool = True
    requires_explicit_fixed_reference: bool = False

    def validate_args(self, args):
        if self.task not in SUPPORTED_CO_TASKS or tuple(args.tasks) != (self.task,):
            raise ValueError(f"External-baseline experiment requires one specialist from {SUPPORTED_CO_TASKS}")
        if self.advantage_estimator not in ("rloo", "external"):
            raise ValueError("Unknown external-baseline advantage estimator")
        if self.num_generations not in (1, 9) or (self.advantage_estimator == "rloo" and self.num_generations != 9):
            raise ValueError("This comparison supports RLOO B9, external B9, and external B1")
        if self.baseline_initial != 0.0 or self.baseline_scale != 1.0:
            raise ValueError("First external-baseline screen fixes baseline initial=0 and scale=1")
        if self.world_size not in ((1, 4) if self.diagnostic else (4,)):
            raise ValueError("Pilot requires four ranks; diagnostic supports one or four")
        if str(self.world_size) != os.environ.get("WORLD_SIZE", "1"):
            raise ValueError("External-baseline world size differs from the declared profile")
        if self.seed < 0 or str(self.seed) != os.environ.get("GRPO_DATA_SEED", str(self.seed)):
            raise ValueError("External-baseline policy and dataset seeds must match")
        expected = {
            **COMMON_DEFAULTS, "num_generations": self.num_generations, "seed": self.seed,
            "generation_batch_size": self.world_size * 9,
            "max_steps": 100,
            "save_steps": 5 if self.diagnostic else 50,
            "advantage_estimator": self.advantage_estimator,
            "external_baseline_initial": self.baseline_initial,
            "external_baseline_scale": self.baseline_scale,
            "external_baseline_diagnostic": self.diagnostic,
        }
        drift = {k: {"expected": v, "observed": getattr(args, k, None)}
                 for k, v in expected.items() if getattr(args, k, None) != v}
        if drift:
            raise ValueError(f"External-baseline common recipe mismatch: {drift}")
        if any(os.environ.get(k) != v for k, v in REWARD_ENVIRONMENT.items()):
            raise ValueError("External-baseline reward recipe changed")
        stop = os.environ.get("GRPO_STOP_AT_GLOBAL_STEP", "")
        if not stop.isdigit() or int(stop) not in ((5, 10) if self.diagnostic else (50, 100)):
            raise ValueError("External-baseline runs require a declared diagnostic/pilot chunk boundary")
        if os.environ.get("GRPO_STRICT_OPTIMIZER_RESUME") != "1":
            raise ValueError("External-baseline chunks require strict optimizer resume")
        if any(value for name, value in os.environ.items()
               if name == "GRPO_REBASE_LOADED_SCHEDULER_AT_STEP" or name.startswith("GRPO_EXTENSION_")):
            raise ValueError("External-baseline fixed horizon forbids scheduler extension/rebase settings")
        if getattr(args, "resolved_generation_batch_size", self.world_size * 9) != self.world_size * 9:
            raise ValueError("External-baseline resolved rollout batch differs")
        if getattr(args, "resolved_steps_per_generation", None) not in (None, 3):
            raise ValueError("External-baseline rollout buffer must span three microbatches")
        from clasp.data.prompts import JIANG_F638_INFERENCE_TEMPLATE
        if hashlib.sha256(JIANG_F638_INFERENCE_TEMPLATE.encode()).hexdigest() != COMMON_PROMPT_SHA256:
            raise ValueError("Common no-reasoning prompt changed")

    def grpo_config_overrides(self, grpo_signature):
        if missing := set(CONFIG) - set(grpo_signature):
            raise ValueError(f"TRL cannot express the external-baseline recipe: {sorted(missing)}")
        return copy.deepcopy(CONFIG)

    def validate_grpo_config(self, config):
        expected = {**CONFIG, "num_generations": self.num_generations,
                    "max_steps": 100,
                    "generation_batch_size": self.world_size * 9,
                    "gradient_accumulation_steps": 3, "steps_per_generation": 3,
                    "ignore_data_skip": False, "use_vllm": False}
        for name, value in expected.items():
            if getattr(config, name, None) != value:
                raise ValueError(f"Resolved external-baseline GRPOConfig.{name} differs from {value!r}")

    def reward_functions(self):
        if self.task not in TASK_GRADED_REWARDS:
            raise ValueError("Unsupported external-baseline task")
        return (TASK_GRADED_REWARDS[self.task],)

    def validate_dataset(self, dataset, *, label):
        if "input" not in getattr(dataset, "column_names", ()):
            raise ValueError("External-baseline reward requires prepared input text")
        receipt = validate_foarl_f638_dataset(dataset, label=label)
        if set(receipt["task_counts"]) - {self.task}:
            raise ValueError("External-baseline dataset mixes specialist tasks")
        if any(not isinstance(dataset[i]["input"], str) for i in range(len(dataset))):
            raise ValueError("External-baseline input values must be text")
        for row in dataset:
            baseline_instance_size(self.task, row)
            if self.task in ("tsp", "op", "jssp"):
                validate_graded_instance(self.task, row["instance_json"])
        if label == "train" and len(dataset) < (36 if self.diagnostic else 1900):
            raise ValueError("Too few retained prompts for the declared external-baseline screen")
        return receipt

    def validate_trainer_implementation(self, trainer_class):
        version = importlib.metadata.version("trl")
        if not version.startswith("0.24."):
            raise RuntimeError(f"External-baseline integration is audited for TRL0.24.x, got {version}")
        source = inspect.getsource(trainer_class)
        compact = re.sub(r"\s+", "", source)
        fixed_reduction = (
            "loss=(per_token_loss*completion_mask).sum()/"
            "(per_token_loss.size(0)*self.max_completion_length)"
        )
        if ('self.loss_type=="dr_grpo"' not in compact or fixed_reduction not in compact
                or "loss=loss/self.current_gradient_accumulation_steps" not in compact):
            raise RuntimeError("TRL does not expose the required fixed-horizon dr_grpo reduction")
        if not callable(getattr(trainer_class, "_generate_and_score_completions", None)):
            raise RuntimeError("TRL lacks the completion-scoring hook")
        return {"status": "PASS", "trl_version": version,
                "base_trainer_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "loss_reduction": "dr_grpo_fixed_max_completion_horizon",
                "advantage_path": "separate_external_baseline_extension_required"}

    def metadata(self):
        if self.source_binding.get("status") != "passed_frozen_manifest_successor_source":
            raise ValueError("External-baseline runtime needs a verified frozen source manifest")
        return {
            "profile_id": PROFILE_ID, "method_id": f"{self.advantage_estimator}_b{self.num_generations}",
            "study_scope": "engineering_smoke_no_performance_claim" if self.diagnostic else "development_only",
            "task": self.task, "seed": self.seed, "world_size": self.world_size,
            "supported_tasks": list(SUPPORTED_CO_TASKS),
            "baseline_size_semantics": TASK_SIZE_SEMANTICS[self.task],
            "advantage_estimator": self.advantage_estimator,
            "baseline": {"kind": ("past_only_pooled_task_size" if self.advantage_estimator == "external"
                                  else "same_prompt_leave_one_out"), "initial": self.baseline_initial,
                         "scale": self.baseline_scale, "current_batch_centering": False,
                         "diagnostic_tracker_updated_after_global_rollout_batch": True,
                         "pooled_values_used_for_actor": self.advantage_estimator == "external"},
            "num_generations": self.num_generations,
            "max_steps": 100,
            "generation_batch_size": self.world_size * 9,
            "reward_heads": [self.reward_functions()[0].__name__],
            "reward_environment": dict(REWARD_ENVIRONMENT),
            "reward_semantics": ("existing_graded_verifier_reward_no_teachers"
                                 if self.task in ("cvrp", "mis")
                                 else "strict_graded_all_co_extension_v1_no_teachers"),
            "reward_range": [-1.0, 2.0],
            "invalid_reward": "empty_parse_minus_one_else_minus_V_over_one_plus_V",
            "feasible_quality": ({"tsp": "reference_over_reference_plus_tour_distance",
                                  "cvrp": "reference_over_reference_plus_route_distance",
                                  "op": "collected_prize_over_total_instance_prize_zero_if_all_zero",
                                  "mis": "independent_set_cardinality_over_num_vertices",
                                  "jssp": "reference_over_reference_plus_makespan"}[self.task]),
            "algorithm_config": copy.deepcopy(CONFIG),
            "source_binding": copy.deepcopy(dict(self.source_binding)),
            "continuation": {"fixed_scheduler_horizon": 100, "strict_optimizer_resume": True,
                             "state_contract": "new_external_baseline_rank_local_optimizer_scheduler_and_baseline_state"},
            "sampling_difference_from_historical_anchors": {
                "previous": {"temperature": 0.7, "top_p": 0.9, "top_k": 50},
                "attribution_arms": {"temperature": 1.0, "top_p": 1.0, "top_k": 0}},
            "claim_limit": "matched_rollout_budget_not_equal_tokens_or_guaranteed_superiority",
        }


def build_runtime(source_binding, args):
    return ExternalBaselineRuntime(
        task=str(args.tasks[0]), advantage_estimator=args.advantage_estimator,
        num_generations=args.num_generations, baseline_initial=args.external_baseline_initial,
        baseline_scale=args.external_baseline_scale, diagnostic=args.external_baseline_diagnostic,
        world_size=int(os.environ.get("WORLD_SIZE", "1")), seed=args.seed,
        source_binding=copy.deepcopy(dict(source_binding)),
    )
