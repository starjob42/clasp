"""Executable semantics for the corrected LLMCoSolver/FOARL f638 baseline.

The reward formulas remain byte-behavior-compatible with the vendored f638
port.  This module only supplies the missing common-contract row adapter and a
fail-closed runtime profile for TRL 0.24.  Publication evaluation must still use
CLASP's strict evaluator, never these training rewards.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from numbers import Integral, Real
from typing import Any, Callable, Mapping, Sequence

from clasp.training_source_identity import (
    FOARL_F638_CORRECTED_TRAINER_SHA256,
    FOARL_F638_CORRECTED_TRAINER_VERSION,
)

from .llmcosolver_f638_rewards import (
    OFFICIAL_REWARD_HEADS,
    UPSTREAM_COMMIT,
    UPSTREAM_REWARDS_SHA256,
)


FOARL_F638_METHOD_ID = "foarl_f638_corrected"
FOARL_F638_PROFILE_ID = "foarl-f638-corrected-common-contract-v1"
FOARL_F638_PUBLICATION_LABEL = "FOARL-f638-corrected (common-contract reimplementation)"
FOARL_F638_TASKS = ("tsp", "cvrp", "op", "mis", "jssp")
FOARL_F638_REWARD_HEAD_ORDER = ("optimality", "feasibility")
FOARL_F638_REWARD_WEIGHTS = (1.0, 1.0)
FOARL_F638_NUM_GENERATIONS = 8
FOARL_F638_BETA = 0.05
FOARL_F638_ADVANTAGE_EPSILON = 1e-4
FOARL_F638_SCALE_REWARDS = "group"
FOARL_F638_STD_ESTIMATOR = "sample_unbiased"
FOARL_F638_NUM_ITERATIONS = 1
FOARL_F638_LOSS_TYPE = "grpo"
FOARL_F638_TOKEN_AGGREGATION = "completion_token_mean_then_sequence_mean"
FOARL_F638_SYNC_REF_MODEL = False
FOARL_F638_IMPORTANCE_SAMPLING_LEVEL = "token"
FOARL_F638_CLIP_EPSILON = 0.2
FOARL_F638_PROMPT_STYLE = "jiang_f638"
FOARL_F638_RL_PROMPT_SHA256 = "8e90f30020283d017f2b7a9c3109d10fde262488a8f36fd16f0da17ba984a606"
FOARL_F638_TRL_MINOR = "0.24"


def validate_foarl_f638_prompt_template() -> dict[str, str]:
    """Bind the live prompt constant, not only a copied metadata digest."""

    from clasp.data.prompts import JIANG_F638_RL_TEMPLATE

    observed = hashlib.sha256(JIANG_F638_RL_TEMPLATE.encode("utf-8")).hexdigest()
    if observed != FOARL_F638_RL_PROMPT_SHA256:
        raise RuntimeError(
            "live Jiang f638 RL prompt template drift: "
            f"expected={FOARL_F638_RL_PROMPT_SHA256}, observed={observed}"
        )
    return {"status": "PASS", "sha256": observed}


@dataclass(frozen=True)
class FoarlF638Semantics:
    """Immutable scientific contract for the corrected executable baseline."""

    method_id: str = FOARL_F638_METHOD_ID
    profile_id: str = FOARL_F638_PROFILE_ID
    publication_label: str = FOARL_F638_PUBLICATION_LABEL
    upstream_commit: str = UPSTREAM_COMMIT
    upstream_rewards_sha256: str = UPSTREAM_REWARDS_SHA256
    supported_tasks: tuple[str, ...] = FOARL_F638_TASKS
    reward_head_order: tuple[str, ...] = FOARL_F638_REWARD_HEAD_ORDER
    reward_weights: tuple[float, ...] = FOARL_F638_REWARD_WEIGHTS
    num_generations: int = FOARL_F638_NUM_GENERATIONS
    beta: float = FOARL_F638_BETA
    advantage_epsilon: float = FOARL_F638_ADVANTAGE_EPSILON
    scale_rewards: str = FOARL_F638_SCALE_REWARDS
    std_estimator: str = FOARL_F638_STD_ESTIMATOR
    num_iterations: int = FOARL_F638_NUM_ITERATIONS
    loss_type: str = FOARL_F638_LOSS_TYPE
    token_aggregation: str = FOARL_F638_TOKEN_AGGREGATION
    sync_ref_model: bool = FOARL_F638_SYNC_REF_MODEL
    importance_sampling_level: str = FOARL_F638_IMPORTANCE_SAMPLING_LEVEL
    clip_epsilon: float = FOARL_F638_CLIP_EPSILON
    prompt_style: str = FOARL_F638_PROMPT_STYLE
    prompt_sha256: str = FOARL_F638_RL_PROMPT_SHA256
    trl_minor: str = FOARL_F638_TRL_MINOR
    trainer_source_version: str = FOARL_F638_CORRECTED_TRAINER_VERSION
    trainer_source_sha256: str = FOARL_F638_CORRECTED_TRAINER_SHA256

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-safe receipt without exposing mutable contract state."""

        return {
            "method_id": self.method_id,
            "profile_id": self.profile_id,
            "publication_label": self.publication_label,
            "classification": "common-contract reimplementation; not an official reproduction",
            "upstream": {
                "commit": self.upstream_commit,
                "rewards_sha256": self.upstream_rewards_sha256,
            },
            "supported_tasks": list(self.supported_tasks),
            "reward_heads": list(self.reward_head_order),
            "reward_weights": list(self.reward_weights),
            "reward_aggregation": "sum",
            "num_generations": self.num_generations,
            "beta": self.beta,
            "advantage_normalization": {
                "scope": self.scale_rewards,
                "std_estimator": self.std_estimator,
                "epsilon": self.advantage_epsilon,
            },
            "num_iterations": self.num_iterations,
            "loss_type": self.loss_type,
            "token_aggregation": self.token_aggregation,
            "sync_ref_model": self.sync_ref_model,
            "reference": "explicit_frozen_corrected_sft_step_zero",
            "importance_sampling_level": self.importance_sampling_level,
            # This is PPO clipping epsilon.  It is intentionally distinct from
            # the 1e-4 advantage-normalization epsilon above.
            "clip_epsilon": self.clip_epsilon,
            "prompt_style": self.prompt_style,
            "prompt_sha256": self.prompt_sha256,
            "trl_minor": self.trl_minor,
            "shared_trainer_source": {
                "version": self.trainer_source_version,
                "sha256": self.trainer_source_sha256,
            },
            "publication_evaluator": "clasp_strict",
        }


FOARL_F638_SEMANTICS = FoarlF638Semantics()


@dataclass(frozen=True)
class FoarlF638RewardRow:
    """One validated standardized GRPO row."""

    task_type: str
    ground_truth: str
    reward_kwargs: Mapping[str, Any]


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _require_finite_number(value: Any, *, label: str, integral: bool = False) -> Real:
    expected = Integral if integral else Real
    if isinstance(value, bool) or not isinstance(value, expected):
        kind = "integer" if integral else "number"
        raise ValueError(f"{label} must be a JSON {kind}, got {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return value


def _validate_coordinates(value: Any, *, label: str) -> list[Any]:
    if not _is_sequence(value) or len(value) < 2:
        raise ValueError(f"{label} must contain at least two coordinate pairs")
    coordinates = list(value)
    for node_idx, coordinate in enumerate(coordinates):
        if not _is_sequence(coordinate) or len(coordinate) != 2:
            raise ValueError(f"{label}[{node_idx}] must be a length-2 coordinate")
        _require_finite_number(coordinate[0], label=f"{label}[{node_idx}][0]")
        _require_finite_number(coordinate[1], label=f"{label}[{node_idx}][1]")
    return coordinates


def _validate_numeric_vector(
    value: Any,
    *,
    label: str,
    expected_length: int,
    nonnegative: bool,
) -> list[Any]:
    if not _is_sequence(value) or len(value) != expected_length:
        raise ValueError(f"{label} must contain exactly {expected_length} values")
    result = list(value)
    for idx, item in enumerate(result):
        _require_finite_number(item, label=f"{label}[{idx}]")
        if nonnegative and float(item) < 0.0:
            raise ValueError(f"{label}[{idx}] must be nonnegative")
    return result


def _decode_instance(value: Any) -> Any:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("instance_json must be a non-empty JSON value")
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"instance_json is not valid JSON: {exc.msg}") from exc
    if value is None:
        raise ValueError("instance_json must not be null")
    return value


def _adapt_instance(task_type: str, value: Any) -> dict[str, Any]:
    instance = _decode_instance(value)
    if task_type == "tsp":
        return {"instance": _validate_coordinates(instance, label="tsp.instance")}

    if task_type == "cvrp":
        if not _is_sequence(instance) or len(instance) != 3:
            raise ValueError("cvrp.instance_json must be [coordinates, demands, capacity]")
        coordinates = _validate_coordinates(instance[0], label="cvrp.coordinates")
        demands = _validate_numeric_vector(
            instance[1],
            label="cvrp.demands",
            expected_length=len(coordinates),
            nonnegative=True,
        )
        capacity = _require_finite_number(instance[2], label="cvrp.capacity")
        if float(capacity) <= 0.0:
            raise ValueError("cvrp.capacity must be positive")
        return {
            "instance_coords": coordinates,
            "instance_demands": demands,
            "instance_capacity": capacity,
        }

    if task_type == "op":
        if not _is_sequence(instance) or len(instance) != 3:
            raise ValueError("op.instance_json must be [coordinates, prizes, max_distance]")
        coordinates = _validate_coordinates(instance[0], label="op.coordinates")
        prizes = _validate_numeric_vector(
            instance[1],
            label="op.prizes",
            expected_length=len(coordinates),
            nonnegative=True,
        )
        max_distance = _require_finite_number(instance[2], label="op.max_distance")
        if float(max_distance) <= 0.0:
            raise ValueError("op.max_distance must be positive")
        return {
            "instance_coords": coordinates,
            "instance_prizes": prizes,
            "instance_max_dist": max_distance,
        }

    if task_type == "mis":
        if not isinstance(instance, Mapping):
            raise ValueError("mis.instance_json must be an object with an edges field")
        edges = instance.get("edges")
        if not _is_sequence(edges):
            raise ValueError("mis.instance_json.edges must be a list")
        for edge_idx, edge in enumerate(edges):
            if not _is_sequence(edge) or len(edge) != 2:
                raise ValueError(f"mis.edges[{edge_idx}] must be a length-2 edge")
            _require_finite_number(edge[0], label=f"mis.edges[{edge_idx}][0]", integral=True)
            _require_finite_number(edge[1], label=f"mis.edges[{edge_idx}][1]", integral=True)
        return {"instance": dict(instance)}

    if task_type == "jssp":
        if not _is_sequence(instance) or not instance:
            raise ValueError("jssp.instance_json must be a non-empty operation matrix")
        rows = list(instance)
        if not _is_sequence(rows[0]) or not rows[0] or len(rows[0]) % 2:
            raise ValueError("jssp rows must contain machine,duration pairs")
        width = len(rows[0])
        for job_idx, row in enumerate(rows):
            if not _is_sequence(row) or len(row) != width:
                raise ValueError("jssp operation matrix must be rectangular")
            for column_idx, item in enumerate(row):
                integral = column_idx % 2 == 0
                _require_finite_number(
                    item,
                    label=f"jssp.instance[{job_idx}][{column_idx}]",
                    integral=integral,
                )
                if float(item) < 0.0:
                    raise ValueError("jssp machine identifiers and durations must be nonnegative")
        return {"instance": rows}

    raise ValueError(f"Unsupported FOARL-f638 task: {task_type!r}")


def adapt_foarl_f638_row(row: Mapping[str, Any]) -> FoarlF638RewardRow:
    """Validate one row emitted by :func:`prepare_grpo_dataset_for_task`."""

    task_type = str(row.get("task_type", "")).strip().lower()
    if task_type not in FOARL_F638_TASKS:
        raise ValueError(
            f"task_type must be one of {FOARL_F638_TASKS}, got {task_type or '<missing>'!r}"
        )
    ground_truth = row.get("answer")
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        raise ValueError(f"{task_type} row requires a non-empty string answer")
    if "instance_json" not in row:
        raise ValueError(f"{task_type} row is missing instance_json")
    reward_kwargs = _adapt_instance(task_type, row["instance_json"])
    return FoarlF638RewardRow(
        task_type=task_type,
        ground_truth=ground_truth,
        reward_kwargs=reward_kwargs,
    )


def _aligned_column(kwargs: Mapping[str, Any], name: str, total: int) -> list[Any]:
    value = kwargs.get(name)
    if not _is_sequence(value) or len(value) != total:
        raise ValueError(f"FOARL reward requires one {name!r} value per completion")
    return list(value)


def _mixed_foarl_reward(
    head_index: int,
    prompts: Sequence[Any] | None,
    completions: Sequence[Any] | None,
    kwargs: Mapping[str, Any],
) -> list[float]:
    if not _is_sequence(completions):
        raise ValueError("FOARL reward requires a completion sequence")
    completion_values = list(completions)
    total = len(completion_values)
    if total == 0:
        return []
    if prompts is not None and (not _is_sequence(prompts) or len(prompts) != total):
        raise ValueError("FOARL reward requires one prompt per completion")
    for idx, completion in enumerate(completion_values):
        if not isinstance(completion, str):
            raise ValueError(f"completion[{idx}] must be plain text, got {type(completion).__name__}")

    task_types = _aligned_column(kwargs, "task_type", total)
    answers = _aligned_column(kwargs, "answer", total)
    instances = _aligned_column(kwargs, "instance_json", total)
    adapted = [
        adapt_foarl_f638_row(
            {"task_type": task_types[idx], "answer": answers[idx], "instance_json": instances[idx]}
        )
        for idx in range(total)
    ]

    indices_by_task: dict[str, list[int]] = {}
    for idx, row in enumerate(adapted):
        indices_by_task.setdefault(row.task_type, []).append(idx)

    rewards: list[float | None] = [None] * total
    for task_type, indices in indices_by_task.items():
        reward_fn = OFFICIAL_REWARD_HEADS[task_type][head_index]
        batched_kwargs: dict[str, list[Any]] = {
            "ground_truth": [adapted[idx].ground_truth for idx in indices]
        }
        for key in adapted[indices[0]].reward_kwargs:
            batched_kwargs[key] = [adapted[idx].reward_kwargs[key] for idx in indices]
        try:
            task_rewards = reward_fn(
                completions=[completion_values[idx] for idx in indices],
                **batched_kwargs,
            )
        except Exception as exc:
            head_name = FOARL_F638_REWARD_HEAD_ORDER[head_index]
            raise RuntimeError(f"f638 {task_type} {head_name} reward failed") from exc
        if not _is_sequence(task_rewards) or len(task_rewards) != len(indices):
            raise RuntimeError(
                f"f638 {task_type} reward returned {len(task_rewards) if _is_sequence(task_rewards) else 'non-sequence'} "
                f"values for {len(indices)} completions"
            )
        for original_idx, value in zip(indices, task_rewards):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise RuntimeError(f"f638 reward[{original_idx}] is not numeric: {value!r}")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise RuntimeError(f"f638 reward[{original_idx}] is not finite: {value!r}")
            rewards[original_idx] = numeric

    if any(value is None for value in rewards):
        raise RuntimeError("f638 mixed reward failed to populate every completion")
    return [float(value) for value in rewards if value is not None]


def foarl_f638_optimality_reward(
    prompts: Sequence[Any] | None = None,
    completions: Sequence[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Ordered reward head 0: upstream f638 optimality."""

    return _mixed_foarl_reward(0, prompts, completions, kwargs)


def foarl_f638_feasibility_reward(
    prompts: Sequence[Any] | None = None,
    completions: Sequence[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Ordered reward head 1: upstream f638 feasibility."""

    return _mixed_foarl_reward(1, prompts, completions, kwargs)


FOARL_F638_REWARD_FUNCS: tuple[Callable[..., list[float]], ...] = (
    foarl_f638_optimality_reward,
    foarl_f638_feasibility_reward,
)


def validate_foarl_f638_dataset(dataset: Any, *, label: str) -> dict[str, Any]:
    """Validate every prepared row before the first rollout."""

    columns = set(getattr(dataset, "column_names", ()))
    required = {"prompt", "answer", "instance_json", "task_type"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{label} dataset is missing FOARL columns: {missing}")
    counts = {task: 0 for task in FOARL_F638_TASKS}
    for row_idx in range(len(dataset)):
        row = dict(dataset[row_idx])
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{label} row {row_idx} has an empty/non-text prompt")
        try:
            adapted = adapt_foarl_f638_row(row)
        except ValueError as exc:
            raise ValueError(f"{label} row {row_idx}: {exc}") from exc
        counts[adapted.task_type] += 1
    return {
        "label": str(label),
        "rows": len(dataset),
        "task_counts": {task: count for task, count in counts.items() if count},
        "status": "PASS",
    }


def _normalized_source(callable_obj: Callable[..., Any], *, label: str) -> str:
    try:
        source = inspect.getsource(inspect.unwrap(callable_obj))
    except (OSError, TypeError) as exc:
        raise RuntimeError(f"Cannot inspect {label}; refusing unverified FOARL semantics") from exc
    return re.sub(r"\s+", "", source)


def validate_trl_foarl_implementation(
    trainer_class: type,
    *,
    trl_version: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless installed TRL implements the pinned reductions.

    TRL 0.24's ``grpo`` branch performs a token mean within each completion
    followed by a sequence mean.  Its ``group`` reward scaling uses
    ``torch.std``'s default sample correction and adds exactly ``1e-4``.  These
    checks deliberately reject a source change instead of silently weakening
    the comparison contract.
    """

    if trl_version is None:
        try:
            trl_version = importlib_metadata.version("trl")
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError("TRL is not installed; FOARL training cannot be verified") from exc
    if not str(trl_version).startswith(f"{FOARL_F638_TRL_MINOR}."):
        raise RuntimeError(
            f"FOARL requires audited TRL {FOARL_F638_TRL_MINOR}.x, got {trl_version!r}"
        )

    loss_source = _normalized_source(trainer_class.compute_loss, label="GRPOTrainer.compute_loss")
    sequence_mean_patterns = (
        "loss=((per_token_loss*completion_mask).sum(-1)/completion_mask.sum(-1).clamp(min=1.0)).mean()",
        "loss=((per_token_loss*mask).sum(-1)/mask.sum(-1).clamp(min=1.0)).mean()",
    )
    loss_matches = 'ifself.loss_type=="grpo":' in loss_source and any(
        pattern in loss_source for pattern in sequence_mean_patterns
    )
    if not loss_matches:
        # TRL decorates trainer methods for profiling.  Well-behaved decorators
        # preserve ``__wrapped__``, but inspect the owning class as a strict
        # fallback for builds whose profiler does not.
        class_source = _normalized_source(trainer_class, label="GRPOTrainer")
        loss_matches = 'ifself.loss_type=="grpo":' in class_source and any(
            pattern in class_source for pattern in sequence_mean_patterns
        )
    if not loss_matches:
        raise RuntimeError(
            "Installed GRPOTrainer does not expose the audited per-sequence-token-mean "
            "loss_type='grpo' branch"
        )

    generation_method = getattr(trainer_class, "_generate_and_score_completions", None)
    if generation_method is None:
        raise RuntimeError("Installed GRPOTrainer lacks _generate_and_score_completions")
    reward_source = _normalized_source(
        generation_method,
        label="GRPOTrainer._generate_and_score_completions",
    )
    required_reward_fragments = (
        "std_rewards=rewards.view(-1,self.num_generations).std(dim=1)",
        "advantages=advantages/(std_rewards+1e-4)",
    )
    reward_matches = all(fragment in reward_source for fragment in required_reward_fragments)
    if not reward_matches:
        class_source = _normalized_source(trainer_class, label="GRPOTrainer")
        reward_source = class_source
        reward_matches = all(fragment in reward_source for fragment in required_reward_fragments)
    if not reward_matches:
        raise RuntimeError(
            "Installed GRPOTrainer does not expose the audited group sample-std + 1e-4 "
            "advantage normalization"
        )
    if "unbiased=False" in reward_source or "correction=0" in reward_source:
        raise RuntimeError("Installed GRPOTrainer uses population rather than sample std")
    return {
        "status": "PASS",
        "trl_version": str(trl_version),
        "loss_reduction": FOARL_F638_TOKEN_AGGREGATION,
        "advantage_normalization": {
            "scope": FOARL_F638_SCALE_REWARDS,
            "std_estimator": FOARL_F638_STD_ESTIMATOR,
            "epsilon": FOARL_F638_ADVANTAGE_EPSILON,
        },
    }


def _require_exact_float(value: Any, expected: float, *, label: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must equal {expected:g}, got {value!r}") from exc
    if not math.isfinite(numeric) or not math.isclose(numeric, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} must equal {expected:g}, got {value!r}")


@dataclass(frozen=True)
class FoarlF638CorrectedRuntimeProfile:
    """Hooks consumed by the shared GRPO model/checkpoint runtime."""

    semantics: FoarlF638Semantics = FOARL_F638_SEMANTICS
    requires_explicit_fixed_reference: bool = True

    def validate_args(self, args: Any) -> None:
        tasks = tuple(str(task).lower() for task in getattr(args, "tasks", ()))
        if len(tasks) != 1 or any(task not in self.semantics.supported_tasks for task in tasks):
            raise ValueError(
                "FOARL tasks require exactly one task-specialist checkpoint and "
                f"one task from {self.semantics.supported_tasks}, got {tasks}"
            )
        validate_foarl_f638_prompt_template()
        if bool(getattr(args, "use_primal_dual", False)):
            raise ValueError("FOARL-f638-corrected forbids primal-dual/CLASP training")
        if bool(getattr(args, "use_clasp", False)):
            raise ValueError("FOARL-f638-corrected forbids --use_clasp")
        if bool(getattr(args, "full_finetune", False)):
            raise ValueError("FOARL-f638-corrected must continue the corrected SFT adapter")
        if bool(getattr(args, "verify_gt_rewards", False)):
            raise ValueError("FOARL must not substitute CLASP strict heads for f638 training rewards")
        if int(getattr(args, "num_generations", 0)) != self.semantics.num_generations:
            raise ValueError(
                "FOARL-f638-corrected requires "
                f"num_generations={self.semantics.num_generations}"
            )
        if str(getattr(args, "prompt_style", "")).strip().lower() != self.semantics.prompt_style:
            raise ValueError("FOARL-f638-corrected requires prompt_style=jiang_f638")
        if bool(getattr(args, "no_reason", True)):
            raise ValueError("FOARL-f638-corrected requires the reasoning-enabled Jiang RL prompt")
        _require_exact_float(getattr(args, "grpo_beta", None), self.semantics.beta, label="grpo_beta")
        _require_exact_float(
            getattr(args, "grpo_epsilon", None),
            self.semantics.clip_epsilon,
            label="grpo_epsilon (PPO clipping)",
        )
        _require_exact_float(
            getattr(args, "grpo_epsilon_high", None),
            self.semantics.clip_epsilon,
            label="grpo_epsilon_high (PPO clipping)",
        )
        if str(getattr(args, "grpo_loss_type", "")).strip().lower() != self.semantics.loss_type:
            raise ValueError("FOARL-f638-corrected requires grpo_loss_type=grpo")
        _require_exact_float(
            getattr(args, "foarl_advantage_epsilon", None),
            self.semantics.advantage_epsilon,
            label="foarl_advantage_epsilon",
        )
        if str(getattr(args, "foarl_scale_rewards", "")).strip().lower() != self.semantics.scale_rewards:
            raise ValueError("FOARL-f638-corrected requires sample-std scale_rewards=group")
        if int(getattr(args, "foarl_num_iterations", 0)) != self.semantics.num_iterations:
            raise ValueError("FOARL-f638-corrected requires num_iterations=1")
        if str(getattr(args, "foarl_token_aggregation", "")).strip().lower() != self.semantics.token_aggregation:
            raise ValueError(
                "FOARL-f638-corrected requires completion-token mean followed by sequence mean"
            )
        if bool(getattr(args, "foarl_sync_ref_model", True)):
            raise ValueError("FOARL-f638-corrected requires sync_ref_model=false")

    def grpo_config_overrides(self, grpo_signature: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "beta",
            "epsilon",
            "epsilon_high",
            "loss_type",
            "scale_rewards",
            "num_iterations",
            "sync_ref_model",
            "reward_weights",
            "importance_sampling_level",
            "mask_truncated_completions",
            "use_liger_loss",
        }
        missing = sorted(required - set(grpo_signature))
        if missing:
            raise RuntimeError(
                "Installed GRPOConfig cannot express the exact FOARL contract; missing "
                + ", ".join(missing)
            )
        return {
            "beta": self.semantics.beta,
            "epsilon": self.semantics.clip_epsilon,
            "epsilon_high": self.semantics.clip_epsilon,
            "loss_type": self.semantics.loss_type,
            "scale_rewards": self.semantics.scale_rewards,
            "num_iterations": self.semantics.num_iterations,
            "sync_ref_model": self.semantics.sync_ref_model,
            "reward_weights": list(self.semantics.reward_weights),
            "importance_sampling_level": self.semantics.importance_sampling_level,
            "mask_truncated_completions": False,
            "use_liger_loss": False,
        }

    def validate_grpo_config(self, config: Any) -> None:
        checks = {
            "num_generations": self.semantics.num_generations,
            "scale_rewards": self.semantics.scale_rewards,
            "num_iterations": self.semantics.num_iterations,
            "loss_type": self.semantics.loss_type,
            "sync_ref_model": self.semantics.sync_ref_model,
            "importance_sampling_level": self.semantics.importance_sampling_level,
            "mask_truncated_completions": False,
            "use_liger_loss": False,
        }
        for name, expected in checks.items():
            actual = getattr(config, name, None)
            if actual != expected:
                raise RuntimeError(f"Resolved GRPOConfig.{name}={actual!r}; expected {expected!r}")
        _require_exact_float(getattr(config, "beta", None), self.semantics.beta, label="GRPOConfig.beta")
        _require_exact_float(
            getattr(config, "epsilon", None),
            self.semantics.clip_epsilon,
            label="GRPOConfig.epsilon",
        )
        _require_exact_float(
            getattr(config, "epsilon_high", None),
            self.semantics.clip_epsilon,
            label="GRPOConfig.epsilon_high",
        )
        reward_weights = tuple(float(value) for value in (getattr(config, "reward_weights", None) or ()))
        if reward_weights != self.semantics.reward_weights:
            raise RuntimeError(
                f"Resolved GRPOConfig.reward_weights={reward_weights!r}; expected {self.semantics.reward_weights!r}"
            )

    def reward_functions(self) -> tuple[Callable[..., list[float]], ...]:
        return FOARL_F638_REWARD_FUNCS

    def validate_dataset(self, dataset: Any, *, label: str) -> dict[str, Any]:
        return validate_foarl_f638_dataset(dataset, label=label)

    def metadata(self) -> dict[str, Any]:
        return self.semantics.metadata()

    def validate_trainer_implementation(self, trainer_class: type) -> dict[str, Any]:
        return validate_trl_foarl_implementation(trainer_class)


FOARL_F638_CORRECTED_RUNTIME = FoarlF638CorrectedRuntimeProfile()


__all__ = [
    "FOARL_F638_ADVANTAGE_EPSILON",
    "FOARL_F638_BETA",
    "FOARL_F638_CORRECTED_RUNTIME",
    "FOARL_F638_METHOD_ID",
    "FOARL_F638_NUM_GENERATIONS",
    "FOARL_F638_NUM_ITERATIONS",
    "FOARL_F638_PROFILE_ID",
    "FOARL_F638_REWARD_FUNCS",
    "FOARL_F638_REWARD_HEAD_ORDER",
    "FOARL_F638_SCALE_REWARDS",
    "FOARL_F638_SEMANTICS",
    "FOARL_F638_SYNC_REF_MODEL",
    "FOARL_F638_TASKS",
    "FOARL_F638_TOKEN_AGGREGATION",
    "FoarlF638CorrectedRuntimeProfile",
    "FoarlF638RewardRow",
    "FoarlF638Semantics",
    "adapt_foarl_f638_row",
    "foarl_f638_feasibility_reward",
    "foarl_f638_optimality_reward",
    "validate_foarl_f638_dataset",
    "validate_foarl_f638_prompt_template",
    "validate_trl_foarl_implementation",
]
