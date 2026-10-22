"""Reward-head construction and the multi-task primal-dual manager.

:func:`build_mixed_reward_head` dispatches each element of a mixed batch to the correct
per-task reward head using ``kwargs["task_type"]``.
:class:`MultiTaskPrimalDualManager` owns the per-task
:class:`~clasp.rewards.primal_dual.PrimalDualConstrainedReward` instances and their dual
(lambda) state, and :func:`build_component_profile` resolves a ``pd_component_profile`` name
into concrete cost tolerances and weights.
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import os
from typing import Callable

from transformers import TrainerCallback  # type: ignore[import-not-found]

from clasp.data.loaders import (
    coerce_json_string,
    list_split_json_files,
    load_and_merge_task_datasets,
    transform_example_for_task,
)
from clasp.data.prompts import build_alpaca_prompt
from clasp.rewards.co_tasks import (
    _check_mis_constraints,
    _grpo_graded_reward_enabled,
    _mis_meta,
    _parse_mis_solution,
)
from clasp.rewards.primal_dual import PrimalDualConstrainedReward
from clasp.rewards.registry import TASK_COST_COMPONENTS, TASK_REWARD_TRIPLETS
from clasp.runtime import find_latest_checkpoint_dir, print_primary

# Intentional re-export: this is the active trainer copy (not the stale
# package-root one); downstream code may import it via clasp.rewards.heads.
from clasp.trainers.primal_dual_grpo import PrimalDualGRPOTrainer  # noqa: F401


def slice_kwargs(kwargs: dict, indices: list[int], total_len: int) -> dict:
    sliced = {}
    for key, value in kwargs.items():
        if isinstance(value, list) and len(value) == total_len:
            sliced[key] = [value[i] for i in indices]
        else:
            sliced[key] = value
    return sliced


def build_mixed_reward_head(
    task_triplets: dict[str, tuple[Callable, Callable, Callable]],
    head_idx: int,
):
    def mixed_reward(prompts, completions, **kwargs):
        total = len(completions)
        if total == 0:
            return []

        task_types = kwargs.get("task_type")
        if not isinstance(task_types, list) or len(task_types) != total:
            raise ValueError("Mixed reward expects kwargs['task_type'] with one label per completion.")

        indices_by_task: dict[str, list[int]] = {}
        for idx, task_name in enumerate(task_types):
            task_name = str(task_name)
            if task_name not in task_triplets:
                raise ValueError(f"No reward triplet registered for task '{task_name}'")
            indices_by_task.setdefault(task_name, []).append(idx)

        rewards = [0.0] * total
        for task_name, indices in indices_by_task.items():
            reward_fn = task_triplets[task_name][head_idx]
            sub_rewards = reward_fn(
                [prompts[i] for i in indices],
                [completions[i] for i in indices],
                **slice_kwargs(kwargs, indices, total),
            )
            for idx, value in zip(indices, sub_rewards):
                rewards[idx] = float(value)
        return rewards

    mixed_reward.__name__ = f"mixed_reward_head_{head_idx}"
    return mixed_reward


class MultiTaskPrimalDualManager:
    def __init__(
        self,
        wrappers: dict[str, PrimalDualConstrainedReward],
        *,
        fixed_null_score_receipt=None,
    ):
        self.wrappers = wrappers
        self.fixed_null_score_receipt = fixed_null_score_receipt
        self.fixed_null_receipt_identity = (
            None
            if fixed_null_score_receipt is None
            else dict(fixed_null_score_receipt.identity)
        )
        # Set-level CLASP multipliers live beside the per-constraint wrappers so
        # the existing checkpoint callback persists them across resumes.
        self._clasp_bon_lambdas: dict[str, float] = {}
        self._clasp_bon_failure_ema: dict[str, float] = {}
        self._clasp_bon_observed_groups: dict[str, int] = {}
        self._clasp_trainer_state: dict | None = None
        self._clasp_resume_checkpoint_requested = False
        self._clasp_resume_checkpoint_has_trainer_state = False
        self._clasp_resume_load_serial = 0
        self._last_components = {
            "primal_scores": [],
            "objective_scores": [],
            "structure_scores": [],
            "weighted_gap_scores": [],
            "positive_weighted_gap_scores": [],
            "raw_positive_gap_scores": [],
            "dual_aware_recovery_gap_scores": [],
            "constraint_violation_scores": [],
            "max_constraint_violation_scores": [],
            "feasibility_flags": [],
            "fixed_null_objective_scores": [],
            "fixed_null_max_constraint_violation_scores": [],
            "fixed_null_feasibility_flags": [],
            "constraint_scores": [],
            "combined_scores": [],
            "task_types": [],
        }
        self.__name__ = self.__class__.__name__

    def __call__(self, prompts, completions, **kwargs):
        total = len(completions)
        if total == 0:
            return []

        task_types = kwargs.get("task_type")
        if not isinstance(task_types, list) or len(task_types) != total:
            raise ValueError(
                "Multi-task primal-dual reward expects kwargs['task_type'] with one label per completion."
            )

        indices_by_task: dict[str, list[int]] = {}
        for idx, task_name in enumerate(task_types):
            task_name = str(task_name)
            if task_name not in self.wrappers:
                raise ValueError(f"No primal-dual wrapper registered for task '{task_name}'")
            indices_by_task.setdefault(task_name, []).append(idx)

        fixed_null_scores = None
        if self.fixed_null_score_receipt is not None:
            fixed_null_scores = [
                self.fixed_null_score_receipt.scores_for(task_name, prompt)
                for task_name, prompt in zip(task_types, prompts)
            ]

        rewards = [0.0] * total
        primal_scores = [0.0] * total
        objective_scores = [0.0] * total
        structure_scores = [0.0] * total
        weighted_gap_scores = [0.0] * total
        positive_weighted_gap_scores = [0.0] * total
        raw_positive_gap_scores = [0.0] * total
        dual_aware_recovery_gap_scores = [0.0] * total
        constraint_violation_scores = [0.0] * total
        max_constraint_violation_scores = [0.0] * total
        feasibility_flags = [0.0] * total
        constraint_scores = [0.0] * total
        combined_scores = [0.0] * total

        for task_name, indices in indices_by_task.items():
            wrapper = self.wrappers[task_name]
            sub_rewards = wrapper(
                [prompts[i] for i in indices],
                [completions[i] for i in indices],
                **slice_kwargs(kwargs, indices, total),
            )
            sub_components = wrapper.get_last_components()
            raw_weighted_gap_scores = sub_components.get(
                "weighted_gap_scores",
                sub_components.get("weighted_cost_scores", []),
            )
            raw_positive_weighted_gap_scores = sub_components.get(
                "positive_weighted_gap_scores",
                [max(0.0, float(value)) for value in raw_weighted_gap_scores],
            )
            raw_positive_gap_values = sub_components.get(
                "raw_positive_gap_scores",
                raw_positive_weighted_gap_scores,
            )
            dual_aware_recovery_gap_values = sub_components.get(
                "dual_aware_recovery_gap_scores",
                raw_positive_gap_values,
            )
            raw_objective_scores = sub_components.get(
                "objective_scores",
                sub_components.get("primal_scores", []),
            )
            raw_structure_scores = sub_components.get(
                "structure_scores",
                [0.0] * len(indices),
            )
            for component_name, component_values in (
                ("objective_scores", raw_objective_scores),
                ("structure_scores", raw_structure_scores),
            ):
                if not isinstance(component_values, list) or len(component_values) != len(indices):
                    raise RuntimeError(
                        f"Task {task_name!r} returned {len(component_values) if isinstance(component_values, list) else 'non-list'} "
                        f"{component_name} for {len(indices)} completions."
                    )

            for idx, value in zip(indices, sub_rewards):
                rewards[idx] = float(value)
            for idx, value in zip(indices, sub_components["primal_scores"]):
                primal_scores[idx] = float(value)
            for idx, value in zip(indices, raw_objective_scores):
                objective_scores[idx] = float(value)
            for idx, value in zip(indices, raw_structure_scores):
                structure_scores[idx] = float(value)
            for idx, value in zip(indices, raw_weighted_gap_scores):
                weighted_gap_scores[idx] = float(value)
            for idx, value in zip(indices, raw_positive_weighted_gap_scores):
                positive_weighted_gap_scores[idx] = float(value)
            for idx, value in zip(indices, raw_positive_gap_values):
                raw_positive_gap_scores[idx] = float(value)
            for idx, value in zip(indices, dual_aware_recovery_gap_values):
                dual_aware_recovery_gap_scores[idx] = float(value)
            for idx, value in zip(indices, sub_components.get("constraint_violation_scores", [])):
                constraint_violation_scores[idx] = float(value)
            for idx, value in zip(
                indices,
                sub_components.get(
                    "max_constraint_violation_scores",
                    sub_components.get("constraint_violation_scores", []),
                ),
            ):
                max_constraint_violation_scores[idx] = float(value)
            for idx, value in zip(indices, sub_components.get("feasibility_flags", [])):
                feasibility_flags[idx] = float(value)
            for idx, value in zip(indices, sub_components["constraint_scores"]):
                constraint_scores[idx] = float(value)
            for idx, value in zip(indices, sub_components["combined_scores"]):
                combined_scores[idx] = float(value)

        self._last_components = {
            "primal_scores": primal_scores,
            "objective_scores": objective_scores,
            "structure_scores": structure_scores,
            "weighted_gap_scores": weighted_gap_scores,
            "weighted_cost_scores": list(weighted_gap_scores),
            "positive_weighted_gap_scores": positive_weighted_gap_scores,
            "raw_positive_gap_scores": raw_positive_gap_scores,
            "dual_aware_recovery_gap_scores": dual_aware_recovery_gap_scores,
            "constraint_violation_scores": constraint_violation_scores,
            "max_constraint_violation_scores": max_constraint_violation_scores,
            "feasibility_flags": feasibility_flags,
            "constraint_scores": constraint_scores,
            "combined_scores": combined_scores,
            "task_types": [str(task_name) for task_name in task_types],
        }
        if fixed_null_scores is not None:
            self._last_components.update(
                {
                    "fixed_null_objective_scores": [
                        float(scores.objective_score) for scores in fixed_null_scores
                    ],
                    "fixed_null_max_constraint_violation_scores": [
                        float(scores.max_constraint_violation_score)
                        for scores in fixed_null_scores
                    ],
                    "fixed_null_feasibility_flags": [
                        float(scores.feasibility_flag) for scores in fixed_null_scores
                    ],
                }
            )
        return rewards

    def reset_pending(self):
        for wrapper in self.wrappers.values():
            wrapper.reset_pending()

    def set_collect_stats(self, enabled: bool):
        for wrapper in self.wrappers.values():
            wrapper.set_collect_stats(enabled)

    def get_last_components(self):
        out = {}
        for key, values in self._last_components.items():
            if key == "task_types":
                out[key] = [str(value) for value in values]
            else:
                out[key] = [float(value) for value in values]
        return out

    def get_clasp_bon_lambdas(self) -> dict[str, float]:
        return {
            str(task).lower(): max(0.0, float(value))
            for task, value in self._clasp_bon_lambdas.items()
            if math.isfinite(float(value))
        }

    def set_clasp_bon_lambdas(self, lambdas: dict[str, float]) -> None:
        clean: dict[str, float] = {}
        for task, value in (lambdas or {}).items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(
                    f"Invalid CLASP BoN lambda for task {task!r}: {value!r}"
                )
            clean[str(task).lower()] = numeric
        self._clasp_bon_lambdas = clean

    def get_clasp_bon_controller_state(self) -> dict[str, dict[str, float | int]]:
        return {
            "lambdas": self.get_clasp_bon_lambdas(),
            "failure_ema": dict(self._clasp_bon_failure_ema),
            "observed_groups": dict(self._clasp_bon_observed_groups),
        }

    def set_clasp_bon_controller_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        self.set_clasp_bon_lambdas(state.get("lambdas", {}))
        failure_ema: dict[str, float] = {}
        for task, value in (state.get("failure_ema", {}) or {}).items():
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError(
                    f"Invalid CLASP BoN failure EMA for task {task!r}: {value!r}"
                )
            failure_ema[str(task).lower()] = numeric
        observed_groups: dict[str, int] = {}
        for task, value in (state.get("observed_groups", {}) or {}).items():
            numeric = int(value)
            if numeric < 0:
                raise ValueError(
                    f"Invalid CLASP BoN observed-group count for task {task!r}: {value!r}"
                )
            observed_groups[str(task).lower()] = numeric
        self._clasp_bon_failure_ema = failure_ema
        self._clasp_bon_observed_groups = observed_groups

    def get_clasp_trainer_state(self) -> dict | None:
        if self._clasp_trainer_state is None:
            return None
        return copy.deepcopy(self._clasp_trainer_state)

    def set_clasp_trainer_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise TypeError("CLASP trainer runtime state must be a dictionary")
        self._clasp_trainer_state = copy.deepcopy(state)

    def mark_resume_checkpoint_requested(self, *, has_trainer_state: bool = False) -> None:
        self._clasp_resume_checkpoint_requested = True
        self._clasp_resume_checkpoint_has_trainer_state = bool(has_trainer_state)

    def get_clasp_resume_state_status(self) -> tuple[bool, bool, int]:
        return (
            bool(self._clasp_resume_checkpoint_requested),
            bool(self._clasp_resume_checkpoint_has_trainer_state),
            int(self._clasp_resume_load_serial),
        )

    def update_lambda(self):
        for task_name in sorted(self.wrappers):
            self.wrappers[task_name].update_lambda()

    def uses_clasp_proper_dual(self) -> bool:
        return any(
            bool(getattr(wrapper, "clasp_proper_dual", False))
            for wrapper in self.wrappers.values()
        )

    def set_cost_limit(self, cost_limit: float):
        for wrapper in self.wrappers.values():
            wrapper.set_cost_limit(float(cost_limit))

    def set_dual_learning_rates(self, dual_lr_up: float, dual_lr_down: float | None = None):
        for wrapper in self.wrappers.values():
            wrapper.set_dual_learning_rates(float(dual_lr_up), dual_lr_down)

    def get_last_mean_constraint(self) -> float | None:
        total = 0.0
        count = 0
        for wrapper in self.wrappers.values():
            mean_value, local_count = wrapper.get_last_constraint_stats()
            if mean_value is None or local_count <= 0:
                continue
            total += float(mean_value) * int(local_count)
            count += int(local_count)
        if count <= 0:
            return None
        return float(total / count)

    def state_dict(self):
        state = {task: wrapper.state_dict() for task, wrapper in self.wrappers.items()}
        state["__clasp_bon__"] = self.get_clasp_bon_controller_state()
        if self._clasp_trainer_state is not None:
            state["__clasp_trainer__"] = self.get_clasp_trainer_state()
        return state

    def load_state_dict(self, state, *, from_checkpoint: bool = False):
        if not isinstance(state, dict):
            return
        self._clasp_resume_load_serial += 1
        if from_checkpoint:
            self._clasp_resume_checkpoint_requested = True
            self._clasp_resume_checkpoint_has_trainer_state = isinstance(
                state.get("__clasp_trainer__"),
                dict,
            )
            if not self._clasp_resume_checkpoint_has_trainer_state:
                self._clasp_trainer_state = None
        else:
            self._clasp_resume_checkpoint_requested = False
            self._clasp_resume_checkpoint_has_trainer_state = False
        trainer_state = state.get("__clasp_trainer__")
        if isinstance(trainer_state, dict):
            self.set_clasp_trainer_state(trainer_state)
        elif not from_checkpoint:
            self._clasp_trainer_state = None
        bon_state = state.get("__clasp_bon__")
        if isinstance(bon_state, dict):
            self.set_clasp_bon_controller_state(bon_state)
        for task, wrapper_state in state.items():
            if task in self.wrappers:
                self.wrappers[task].load_state_dict(wrapper_state)


class MultiTaskPrimalDualStateCallback(TrainerCallback):
    def __init__(self, manager: MultiTaskPrimalDualManager):
        self.manager = manager

    def _save_state(self, output_dir: str) -> None:
        if not output_dir:
            return
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "multitask_primal_dual_state.json")
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(self.manager.state_dict(), handle, indent=2, ensure_ascii=True)

    def on_save(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return control
        self._save_state(os.path.join(args.output_dir, f"checkpoint-{state.global_step}"))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return control
        self._save_state(args.output_dir)
        return control


def save_multitask_primal_dual_state(output_dir: str, manager: MultiTaskPrimalDualManager) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "multitask_primal_dual_state.json")
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(manager.state_dict(), handle, indent=2, ensure_ascii=True)


class MultiTaskPrimalDualUpdateCallback(TrainerCallback):
    def __init__(self, manager: MultiTaskPrimalDualManager):
        self.manager = manager

    def on_train_begin(self, args, state, control, **kwargs):
        self.manager.reset_pending()
        self.manager.set_collect_stats(False)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        if not self.manager.uses_clasp_proper_dual():
            self.manager.reset_pending()
        self.manager.set_collect_stats(True)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self.manager.update_lambda()
        self.manager.set_collect_stats(False)
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        self.manager.set_collect_stats(False)
        self.manager.reset_pending()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.manager.set_collect_stats(False)
        self.manager.reset_pending()
        return control


class ClaspValidationGateCheckpointCallback(TrainerCallback):
    def __init__(self, targets: dict[str, float]):
        self.targets = {
            str(task).lower(): float(target)
            for task, target in (targets or {}).items()
            if 0.0 <= float(target) <= 1.0
        }

    def _find_latest_metric(self, state, task_name: str) -> float | None:
        history = getattr(state, "log_history", None)
        if not isinstance(history, list):
            return None
        task_suffix = f"task/{task_name}/candidate_feasibility_rate"
        for record in reversed(history):
            if not isinstance(record, dict):
                continue
            task_matches = [
                value
                for key, value in record.items()
                if str(key).startswith("eval") and str(key).endswith(task_suffix)
            ]
            if task_matches:
                try:
                    return float(task_matches[-1])
                except (TypeError, ValueError):
                    return None
        return None

    def _gate_save(self, state, control):
        if not self.targets or not bool(getattr(control, "should_save", False)):
            return control
        failures = []
        for task_name, target in sorted(self.targets.items()):
            observed = self._find_latest_metric(state, task_name)
            if observed is None or observed < target:
                failures.append((task_name, target, observed))
        if failures:
            control.should_save = False
            details = ", ".join(
                f"{task} observed={observed if observed is not None else 'missing'} target={target:.4f}"
                for task, target, observed in failures
            )
            print_primary(f"[CLASP][ValidationGate] Suppressing checkpoint save: {details}")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        return self._gate_save(state, control)

    def on_save(self, args, state, control, **kwargs):
        return self._gate_save(state, control)


class MultiTaskPrimalDualScheduleCallback(TrainerCallback):
    def __init__(
        self,
        manager: MultiTaskPrimalDualManager,
        *,
        stage_a_fraction: float,
        stage_b_fraction: float,
        stage_a_cost_limit: float,
        stage_b_cost_limit: float,
        stage_c_cost_limit: float,
        stage_a_dual_lr_scale: float,
        stage_b_dual_lr_scale: float,
        stage_c_dual_lr_scale: float,
        stage_c_constraint_threshold: float,
        base_dual_lr_up: float,
        base_dual_lr_down: float,
    ):
        self.manager = manager
        self.stage_a_fraction = float(stage_a_fraction)
        self.stage_b_fraction = float(stage_b_fraction)
        self.stage_a_cost_limit = float(stage_a_cost_limit)
        self.stage_b_cost_limit = float(stage_b_cost_limit)
        self.stage_c_cost_limit = float(stage_c_cost_limit)
        self.stage_a_dual_lr_scale = float(stage_a_dual_lr_scale)
        self.stage_b_dual_lr_scale = float(stage_b_dual_lr_scale)
        self.stage_c_dual_lr_scale = float(stage_c_dual_lr_scale)
        self.stage_c_constraint_threshold = float(stage_c_constraint_threshold)
        self.base_dual_lr_up = float(base_dual_lr_up)
        self.base_dual_lr_down = float(base_dual_lr_down)
        self._last_stage_name: str | None = None

    def _resolve_stage(self, state) -> tuple[str, float, float]:
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        if max_steps <= 0:
            progress = 0.0
        else:
            progress = max(0.0, min(1.0, float(getattr(state, "global_step", 0)) / float(max_steps)))

        if progress < self.stage_a_fraction:
            return (
                "stage_a",
                self.stage_a_cost_limit,
                self.stage_a_dual_lr_scale,
            )

        if progress < self.stage_b_fraction:
            return (
                "stage_b",
                self.stage_b_cost_limit,
                self.stage_b_dual_lr_scale,
            )

        mean_constraint = self.manager.get_last_mean_constraint()
        if mean_constraint is not None and mean_constraint >= self.stage_c_constraint_threshold:
            return (
                "stage_c",
                self.stage_c_cost_limit,
                self.stage_c_dual_lr_scale,
            )

        return (
            "stage_b_hold",
            self.stage_b_cost_limit,
            self.stage_b_dual_lr_scale,
        )

    def _apply_stage(self, stage_name: str, cost_limit: float, lr_scale: float):
        dual_lr_up = self.base_dual_lr_up * float(lr_scale)
        dual_lr_down = self.base_dual_lr_down * float(lr_scale)
        self.manager.set_cost_limit(cost_limit)
        self.manager.set_dual_learning_rates(dual_lr_up=dual_lr_up, dual_lr_down=dual_lr_down)
        if stage_name != self._last_stage_name:
            mean_constraint = self.manager.get_last_mean_constraint()
            mean_text = "n/a" if mean_constraint is None else f"{mean_constraint:.4f}"
            print_primary(
                "[ConstrainedGRPO][Schedule] "
                f"stage={stage_name} cost_limit={cost_limit:.4f} "
                f"dual_lr_up={dual_lr_up:.6f} dual_lr_down={dual_lr_down:.6f} "
                f"mean_constraint={mean_text}"
            )
            self._last_stage_name = stage_name

    def on_train_begin(self, args, state, control, **kwargs):
        stage_name, cost_limit, lr_scale = self._resolve_stage(state)
        self._apply_stage(stage_name, cost_limit, lr_scale)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        stage_name, cost_limit, lr_scale = self._resolve_stage(state)
        self._apply_stage(stage_name, cost_limit, lr_scale)
        return control


def build_task_triplets(tasks: list[str]) -> dict[str, tuple[Callable, Callable, Callable]]:
    enabled = os.environ.get("GRPO_VERIFIED_TRAIN_REFERENCE_TEACHER", "0")
    if enabled not in ("0", "1"):
        raise ValueError("GRPO_VERIFIED_TRAIN_REFERENCE_TEACHER must be0 or1")
    if enabled == "1" and not _grpo_graded_reward_enabled():
        raise ValueError("Training-reference teacher requires the graded reward recipe")
    result = {task: TASK_REWARD_TRIPLETS[task] for task in tasks}
    if enabled == "1":
        from clasp.rewards.verified_train_reference import training_reference_graded_reward
        def head(task):
            def objective(prompts, completions, **kwargs):
                return training_reference_graded_reward(task, prompts, completions, **kwargs)
            objective.__name__ = f"verified_reference_{task}_graded_objective"
            return objective
        for task in ("tsp", "op", "jssp"):
            if task in result:
                result[task] = (*result[task][:2], head(task))
    return result


def _infer_static_cost_names(cost_components_fn: Callable, task_name: str) -> list[str]:
    """Probe a task cost head on an empty batch to bootstrap a stable cost vector schema."""
    try:
        probe_output = cost_components_fn(
            [],
            [],
            task_type=[],
            input=[],
            answer=[],
            instance_json=[],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to probe cost component schema for task '{task_name}'. "
            "The cost head must return a stable non-empty cost_names list even on empty batches."
        ) from exc

    if isinstance(probe_output, dict):
        raw_cost_names = probe_output.get("cost_names")
    elif isinstance(probe_output, tuple) and len(probe_output) == 2:
        raw_cost_names = probe_output[0]
    else:
        raise RuntimeError(
            f"Cost head for task '{task_name}' returned unsupported probe payload type "
            f"{type(probe_output).__name__}. Expected dict or (cost_names, cost_vectors) tuple."
        )

    if not isinstance(raw_cost_names, list) or not raw_cost_names:
        raise RuntimeError(
            f"Cost head for task '{task_name}' must expose a non-empty cost_names list "
            "during empty-batch probing."
        )

    cost_names = [str(name) for name in raw_cost_names]
    if len(set(cost_names)) != len(cost_names):
        raise RuntimeError(
            f"Cost head for task '{task_name}' returned duplicate cost names during probing: {cost_names!r}."
        )
    return cost_names


def _co_default_component_profile(task_name: str) -> tuple[dict[str, float], dict[str, float]]:
    """Return (cost_limits, penalty_weights) for named CO constraint components."""
    strict_limits = {
        "parse_failure": 0.0,
        "depot_boundary": 0.0,
        "start_at_depot": 0.0,
        "end_at_depot": 0.0,
        "route_shape_invalid": 0.0,
        "invalid_node": 0.0,
        "duplicate_customer": 0.0,
        "missing_customer": 0.0,
        "customer_count_mismatch": 0.0,
        "duplicate_visit": 0.0,
        "duplicate_node": 0.0,
        "edge_conflict": 0.0,
        "conflict_node_fraction": 0.0,
        "machine_row_mismatch": 0.0,
        "row_length_mismatch": 0.0,
        "invalid_job": 0.0,
        "duplicate_job": 0.0,
        "missing_job": 0.0,
        "row_set_mismatch": 0.0,
        "schedule_invalid": 0.0,
        "cycle_node_fraction": 0.0,
        "processed_node_deficit": 0.0,
        "parse_bad_token": 0.0,
        "makespan_gap_over_ref": 0.0,
        "interior_depot": 0.0,
        "route_length_mismatch": 0.0,
        "prize_surplus_overbudget": 0.0,
        "prize_cap_over_ref": 0.0,
        "size_deficit_over_ref": 0.0,
        "size_excess_over_ref": 0.0,
        "repair_size_deficit_over_ref": 0.0,
        "addable_vertex_fraction": 0.0,
        "tsp_objective_gap": 0.0,
        "cvrp_target_gap_excess": 0.0,
        "cvrp_objective_gap": 0.0,
    }
    soft_limits = {
        "route_count_mismatch": 0.02,
        "capacity_overflow": 0.02,
        "over_capacity_route_fraction": 0.0,
        "max_route_capacity_overflow": 0.0,
        "length_overflow": 0.015,
        # length_margin_overflow is opt-in (CLASP_OP_LENGTH_MARGIN_RATIO): only the
        # co_op_strict_length_margin profile carries its limit; the lock profile
        # neutralizes it at 1.0.
    }
    weights = {
        "parse_failure": 5.0,
        "invalid_node": 4.0,
        "depot_boundary": 3.5,
        "start_at_depot": 3.5,
        "end_at_depot": 3.5,
        "route_shape_invalid": 4.0,
        "interior_depot": 2.5,
        "duplicate_customer": 4.0,
        "missing_customer": 5.0,
        "customer_count_mismatch": 4.0,
        "route_count_mismatch": 1.5,
        "capacity_overflow": 3.0,
        "over_capacity_route_fraction": 4.0,
        "max_route_capacity_overflow": 4.0,
        "duplicate_visit": 2.0,
        "length_overflow": 3.0,
        "length_margin_overflow": 2.0,
        "prize_surplus_overbudget": 2.0,
        "prize_cap_over_ref": 2.0,
        "duplicate_node": 1.5,
        "edge_conflict": 5.0,
        "conflict_node_fraction": 5.0,
        "machine_row_mismatch": 3.5,
        "row_length_mismatch": 3.5,
        "invalid_job": 4.0,
        "duplicate_job": 3.5,
        "missing_job": 4.5,
        "row_set_mismatch": 4.0,
        "schedule_invalid": 5.0,
        "cycle_node_fraction": 6.0,
        "processed_node_deficit": 6.0,
        "parse_bad_token": 4.0,
        "makespan_gap_over_ref": 4.0,
        "route_length_mismatch": 4.0,
        "size_deficit_over_ref": 4.0,
        "size_excess_over_ref": 4.0,
        "repair_size_deficit_over_ref": 4.0,
        "addable_vertex_fraction": 4.0,
    }

    if task_name == "cvrp":
        weights.update(
            {
                "missing_customer": 6.0,
                "duplicate_customer": 5.0,
                "capacity_overflow": 4.0,
                "over_capacity_route_fraction": 8.0,
                "max_route_capacity_overflow": 8.0,
                "depot_boundary": 4.0,
                "start_at_depot": 5.0,
                "end_at_depot": 5.0,
                "route_shape_invalid": 5.0,
                "customer_count_mismatch": 5.0,
                "cvrp_target_gap_excess": 5.0,
                "cvrp_objective_gap": 3.0,
            }
        )
    elif task_name == "mis":
        weights.update(
            {
                "edge_conflict": 6.0,
                "conflict_node_fraction": 8.0,
                "duplicate_node": 1.0,
                "size_deficit_over_ref": 8.0,
                "size_excess_over_ref": 8.0,
                "addable_vertex_fraction": 8.0,
            }
        )
    elif task_name == "op":
        weights.update({"length_overflow": 4.0, "start_at_depot": 4.0})
    elif task_name == "jssp":
        weights.update(
            {
                "schedule_invalid": 6.0,
                "invalid_job": 5.0,
                "missing_job": 5.0,
                "row_set_mismatch": 5.0,
                "makespan_gap_over_ref": 8.0,
                "cycle_node_fraction": 8.0,
                "processed_node_deficit": 8.0,
                "parse_bad_token": 5.0,
            }
        )
    elif task_name == "tsp":
        weights.update({"tsp_objective_gap": 1.0})

    limits = dict(strict_limits)
    limits.update(soft_limits)
    return limits, weights


def build_component_profile(task_name: str, profile_name: str) -> tuple[dict[str, float], dict[str, float]]:
    profile = str(profile_name or "none").lower()
    if profile == "none":
        return {}, {}
    if profile == "co_default":
        return _co_default_component_profile(str(task_name).lower())
    if profile == "co_op_strict_length":
        limits, weights = _co_default_component_profile(str(task_name).lower())
        if str(task_name).lower() == "op":
            # OP evaluation treats any positive route-length overflow as infeasible.
            # This opt-in profile aligns training pressure with that fair gate.
            limits["length_overflow"] = 0.0
            weights["length_overflow"] = max(float(weights.get("length_overflow", 0.0)), 8.0)
        return limits, weights
    if profile == "co_op_strict_length_prizecap":
        limits, weights = _co_default_component_profile(str(task_name).lower())
        if str(task_name).lower() == "op":
            # Hard budget pressure plus reference-prize caps, without the
            # extra margin that can over-shorten near-budget feasible OP rows.
            limits["length_overflow"] = 0.0
            weights["length_overflow"] = max(float(weights.get("length_overflow", 0.0)), 8.0)
            limits["prize_surplus_overbudget"] = 0.0
            weights["prize_surplus_overbudget"] = max(
                float(weights.get("prize_surplus_overbudget", 0.0)),
                5.0,
            )
            limits["prize_cap_over_ref"] = 0.0
            weights["prize_cap_over_ref"] = max(
                float(weights.get("prize_cap_over_ref", 0.0)),
                5.0,
            )
        return limits, weights
    if profile == "co_op_strict_length_margin":
        limits, weights = _co_default_component_profile(str(task_name).lower())
        if str(task_name).lower() == "op":
            # Training-only slack before the hard OP budget. The component is
            # present only when CLASP_OP_LENGTH_MARGIN_RATIO is set.
            limits["length_overflow"] = 0.0
            weights["length_overflow"] = max(float(weights.get("length_overflow", 0.0)), 8.0)
            limits["length_margin_overflow"] = 0.0
            weights["length_margin_overflow"] = max(
                float(weights.get("length_margin_overflow", 0.0)),
                4.0,
            )
            limits["prize_surplus_overbudget"] = 0.0
            weights["prize_surplus_overbudget"] = max(
                float(weights.get("prize_surplus_overbudget", 0.0)),
                5.0,
            )
            limits["prize_cap_over_ref"] = 0.0
            weights["prize_cap_over_ref"] = max(
                float(weights.get("prize_cap_over_ref", 0.0)),
                5.0,
            )
        return limits, weights
    if profile == "co_jssp_valid_target_gap":
        limits, weights = _co_default_component_profile(str(task_name).lower())
        if str(task_name).lower() == "jssp":
            # JSSP feasibility is already strong in the checkpoint-2020 gate;
            # this opt-in profile preserves validity while adding direct
            # pressure against feasible schedules above the target makespan gap.
            for name, weight in {
                "parse_failure": 6.0,
                "machine_row_mismatch": 6.0,
                "row_length_mismatch": 6.0,
                "invalid_job": 6.0,
                "duplicate_job": 6.0,
                "missing_job": 8.0,
                "row_set_mismatch": 8.0,
                "schedule_invalid": 10.0,
                "cycle_node_fraction": 10.0,
                "processed_node_deficit": 10.0,
                "parse_bad_token": 6.0,
                "makespan_gap_over_ref": 12.0,
            }.items():
                limits[name] = 0.0
                weights[name] = max(float(weights.get(name, 0.0)), weight)
        return limits, weights
    if profile in {"co_feasibility_lock_all", "co_jssp_tail_feas_lock_all"}:
        task = str(task_name).lower()
        limits, weights = _co_default_component_profile(task)

        def lock(names: dict[str, float]) -> None:
            for name, min_weight in names.items():
                limits[name] = 0.0
                weights[name] = max(float(weights.get(name, 0.0)), float(min_weight))

        if task == "tsp":
            lock(
                {
                    "parse_failure": 10.0,
                    "depot_boundary": 10.0,
                    "route_length_mismatch": 10.0,
                    "invalid_node": 10.0,
                    "duplicate_customer": 10.0,
                    "missing_customer": 10.0,
                    "interior_depot": 8.0,
                }
            )
        elif task == "op":
            lock(
                {
                    "parse_failure": 10.0,
                    "start_at_depot": 10.0,
                    "invalid_node": 10.0,
                    "duplicate_visit": 8.0,
                    "length_overflow": 12.0,
                }
            )
            for auxiliary_name in (
                "length_margin_overflow",
                "prize_surplus_overbudget",
                "prize_cap_over_ref",
            ):
                limits[auxiliary_name] = 1.0
        elif task == "cvrp":
            lock(
                {
                    "parse_failure": 10.0,
                    "start_at_depot": 10.0,
                    "end_at_depot": 10.0,
                    "depot_boundary": 10.0,
                    "route_shape_invalid": 10.0,
                    "invalid_node": 10.0,
                    "duplicate_customer": 10.0,
                    "missing_customer": 12.0,
                    "customer_count_mismatch": 10.0,
                    "capacity_overflow": 12.0,
                    "over_capacity_route_fraction": 14.0,
                    "max_route_capacity_overflow": 14.0,
                }
            )
            # The benchmark CVRP checker allows depot separators inside a
            # route and does not enforce an exact vehicle count as long as
            # customer coverage, boundary, and capacity are valid. Keep the
            # diagnostic heads, but do not let them remove rare eval-feasible
            # anchors from CLASP groups.
            limits["route_count_mismatch"] = 1.0
            limits["interior_depot"] = 1.0
            weights["route_count_mismatch"] = max(float(weights.get("route_count_mismatch", 0.0)), 1.5)
            weights["interior_depot"] = max(float(weights.get("interior_depot", 0.0)), 2.5)
        elif task == "mis":
            lock(
                {
                    "parse_failure": 10.0,
                    "invalid_node": 10.0,
                    "duplicate_node": 8.0,
                    "edge_conflict": 24.0,
                    "conflict_node_fraction": 24.0,
                }
            )
            for auxiliary_name in (
                "size_deficit_over_ref",
                "size_excess_over_ref",
                "repair_size_deficit_over_ref",
                "addable_vertex_fraction",
            ):
                limits[auxiliary_name] = 1.0
        elif task == "jssp":
            lock(
                {
                    "parse_failure": 10.0,
                    "machine_row_mismatch": 8.0,
                    "row_length_mismatch": 8.0,
                    "invalid_job": 10.0,
                    "duplicate_job": 8.0,
                    "missing_job": 10.0,
                    "row_set_mismatch": 10.0,
                    "schedule_invalid": 12.0,
                    "cycle_node_fraction": 12.0,
                    "processed_node_deficit": 12.0,
                    "parse_bad_token": 8.0,
                }
            )
            if profile == "co_jssp_tail_feas_lock_all":
                lock({"makespan_gap_over_ref": 24.0})
            else:
                limits["makespan_gap_over_ref"] = 1.0
        return limits, weights
    if profile == "co_mis_strict_edges":
        limits, weights = _co_default_component_profile(str(task_name).lower())
        if str(task_name).lower() == "mis":
            # E692/E693 hard MIS rows were validly parsed near-independent sets
            # with a few edge conflicts. Push those conflicts harder during RL
            # while leaving inference unchanged.
            limits["edge_conflict"] = 0.0
            weights["edge_conflict"] = max(float(weights.get("edge_conflict", 0.0)), 20.0)
            weights["duplicate_node"] = max(float(weights.get("duplicate_node", 0.0)), 2.0)
        return limits, weights
    if profile == "co_mis_conflict_nodes":
        limits, weights = _co_default_component_profile(str(task_name).lower())
        if str(task_name).lower() == "mis":
            # Edge count alone can under-focus near-independent sets where a
            # few selected vertices cause the conflicts. This profile adds a
            # stricter selected-conflict-node component when that cost head is
            # explicitly enabled.
            limits["edge_conflict"] = 0.0
            weights["edge_conflict"] = max(float(weights.get("edge_conflict", 0.0)), 20.0)
            limits["conflict_node_fraction"] = 0.0
            weights["conflict_node_fraction"] = max(
                float(weights.get("conflict_node_fraction", 0.0)),
                24.0,
            )
            weights["duplicate_node"] = max(float(weights.get("duplicate_node", 0.0)), 2.0)
        return limits, weights
    if profile == "co_mis_conflict_target_gap":
        limits, weights = _co_default_component_profile(str(task_name).lower())
        if str(task_name).lower() == "mis":
            # E759 improved feasibility by removing reference imitation but
            # worsened underselection. Combine conflict-node pressure with a
            # target-size cost; planned runs ignore the size cost for binary
            # feasibility while keeping it in raw/dual CLASP gaps.
            limits["edge_conflict"] = 0.0
            weights["edge_conflict"] = max(float(weights.get("edge_conflict", 0.0)), 20.0)
            limits["conflict_node_fraction"] = 0.0
            weights["conflict_node_fraction"] = max(
                float(weights.get("conflict_node_fraction", 0.0)),
                24.0,
            )
            limits["size_deficit_over_ref"] = 0.0
            weights["size_deficit_over_ref"] = max(
                float(weights.get("size_deficit_over_ref", 0.0)),
                12.0,
            )
            weights["duplicate_node"] = max(float(weights.get("duplicate_node", 0.0)), 2.0)
        return limits, weights
    if profile == "co_mis_conflict_target_gap_sizecap":
        limits, weights = _co_default_component_profile(str(task_name).lower())
        if str(task_name).lower() == "mis":
            # Dense low-optimum MIS rows can fail by selecting too many
            # vertices before conflict pressure has a useful frontier. This
            # opt-in profile keeps the target-gap pressure, but adds a
            # training-only reference-size cap cost when the corresponding
            # environment flag enables the cost head.
            limits["edge_conflict"] = 0.0
            weights["edge_conflict"] = max(float(weights.get("edge_conflict", 0.0)), 20.0)
            limits["conflict_node_fraction"] = 0.0
            weights["conflict_node_fraction"] = max(
                float(weights.get("conflict_node_fraction", 0.0)),
                24.0,
            )
            limits["size_deficit_over_ref"] = 0.0
            weights["size_deficit_over_ref"] = max(
                float(weights.get("size_deficit_over_ref", 0.0)),
                12.0,
            )
            limits["size_excess_over_ref"] = 0.0
            weights["size_excess_over_ref"] = max(
                float(weights.get("size_excess_over_ref", 0.0)),
                10.0,
            )
            weights["duplicate_node"] = max(float(weights.get("duplicate_node", 0.0)), 2.0)
        return limits, weights
    if profile == "co_mis_conflict_target_gap_maximality":
        limits, weights = _co_default_component_profile(str(task_name).lower())
        if str(task_name).lower() == "mis":
            # Hard MIS diagnostics now fail mainly as undersized near-independent
            # sets. This opt-in profile adds a maximality cost: candidates that
            # leave graph vertices safely addable receive extra CLASP pressure.
            limits["edge_conflict"] = 0.0
            weights["edge_conflict"] = max(float(weights.get("edge_conflict", 0.0)), 20.0)
            limits["conflict_node_fraction"] = 0.0
            weights["conflict_node_fraction"] = max(
                float(weights.get("conflict_node_fraction", 0.0)),
                24.0,
            )
            limits["size_deficit_over_ref"] = 0.0
            weights["size_deficit_over_ref"] = max(
                float(weights.get("size_deficit_over_ref", 0.0)),
                12.0,
            )
            limits["addable_vertex_fraction"] = 0.0
            weights["addable_vertex_fraction"] = max(
                float(weights.get("addable_vertex_fraction", 0.0)),
                14.0,
            )
            weights["duplicate_node"] = max(float(weights.get("duplicate_node", 0.0)), 2.0)
        return limits, weights
    if profile == "co_mis_conflict_repair_target_gap":
        limits, weights = _co_default_component_profile(str(task_name).lower())
        if str(task_name).lower() == "mis":
            # Combine the best E760 conflict/target-size pressure with the
            # existing repaired-subset deficit signal. The repair signal only
            # shapes training-time CLASP gaps; inference candidates are never
            # edited or filtered.
            limits["edge_conflict"] = 0.0
            weights["edge_conflict"] = max(float(weights.get("edge_conflict", 0.0)), 20.0)
            limits["conflict_node_fraction"] = 0.0
            weights["conflict_node_fraction"] = max(
                float(weights.get("conflict_node_fraction", 0.0)),
                24.0,
            )
            limits["size_deficit_over_ref"] = 0.0
            weights["size_deficit_over_ref"] = max(
                float(weights.get("size_deficit_over_ref", 0.0)),
                12.0,
            )
            limits["repair_size_deficit_over_ref"] = 0.0
            weights["repair_size_deficit_over_ref"] = max(
                float(weights.get("repair_size_deficit_over_ref", 0.0)),
                14.0,
            )
            weights["duplicate_node"] = max(float(weights.get("duplicate_node", 0.0)), 2.0)
        return limits, weights
    if profile == "co_mis_target_gap":
        limits, weights = _co_default_component_profile(str(task_name).lower())
        if str(task_name).lower() == "mis":
            # Default-off training-only objective pressure for feasible MIS sets
            # that fall short of the reference size. In planned runs this cost
            # is ignored for binary feasibility but kept in raw/dual CLASP gaps.
            limits["edge_conflict"] = 0.0
            weights["edge_conflict"] = max(float(weights.get("edge_conflict", 0.0)), 16.0)
            weights["duplicate_node"] = max(float(weights.get("duplicate_node", 0.0)), 2.0)
            limits["size_deficit_over_ref"] = 0.0
            weights["size_deficit_over_ref"] = max(
                float(weights.get("size_deficit_over_ref", 0.0)),
                12.0,
            )
        return limits, weights
    if profile == "co_mis_repair_target":
        limits, weights = _co_default_component_profile(str(task_name).lower())
        if str(task_name).lower() == "mis":
            # Default-off training-only pressure for the size of the independent
            # subset implied by greedily dropping conflict-causing vertices.
            # This does not repair inference outputs; it only shapes CLASP gaps.
            limits["edge_conflict"] = 0.0
            weights["edge_conflict"] = max(float(weights.get("edge_conflict", 0.0)), 20.0)
            weights["duplicate_node"] = max(float(weights.get("duplicate_node", 0.0)), 2.0)
            limits["repair_size_deficit_over_ref"] = 0.0
            weights["repair_size_deficit_over_ref"] = max(
                float(weights.get("repair_size_deficit_over_ref", 0.0)),
                14.0,
            )
        return limits, weights
    raise ValueError(f"Unknown primal-dual component profile: {profile_name!r}")


def build_primal_dual_manager(args, tasks: list[str]) -> MultiTaskPrimalDualManager:
    task_triplets = build_task_triplets(tasks)
    wrappers = {}

    for task_name, (structure_fn, constraint_fn, objective_fn) in task_triplets.items():
        cost_components_fn = TASK_COST_COMPONENTS[task_name]
        component_cost_limits, component_weights = build_component_profile(
            task_name,
            getattr(args, "pd_component_profile", "none"),
        )
        objective_supports_apply_gate = False
        try:
            objective_sig = inspect.signature(objective_fn)
            objective_supports_apply_gate = (
                "apply_constraint_gate" in objective_sig.parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in objective_sig.parameters.values()
                )
            )
        except (TypeError, ValueError):
            objective_supports_apply_gate = False

        def wrap_objective(fn, supports_gate):
            def objective_for_primal_dual(prompts, completions, **kwargs):
                local_kwargs = dict(kwargs)
                local_kwargs.pop("apply_constraint_gate", None)
                local_kwargs["allow_infeasible_partial_credit"] = bool(
                    getattr(args, "pd_soft_objective_for_infeasible", True)
                )
                if supports_gate:
                    # Use soft gating inside primal-dual objective heads so near-feasible
                    # solutions still carry a learning signal before full feasibility.
                    return fn(prompts, completions, apply_constraint_gate=True, **local_kwargs)
                return fn(prompts, completions, **local_kwargs)

            return objective_for_primal_dual

        wrapper = PrimalDualConstrainedReward(
            structure_fn=structure_fn,
            constraint_fn=constraint_fn,
            objective_fn=wrap_objective(objective_fn, objective_supports_apply_gate),
            cost_components_fn=cost_components_fn,
            constraint_target=args.pd_constraint_target,
            cost_limit=args.pd_cost_limit,
            lambda_init=args.pd_lambda_init,
            dual_lr=args.pd_dual_lr,
            dual_lr_up=args.pd_dual_lr_up,
            dual_lr_down=args.pd_dual_lr_down,
            dual_update_interval=args.pd_dual_update_interval,
            violation_ema_beta=args.pd_violation_ema_beta,
            lambda_max=args.pd_lambda_max,
            violation_tolerance=args.pd_violation_tolerance,
            objective_weight=args.pd_objective_weight,
            structure_weight=args.pd_structure_weight,
            sync_across_ranks=args.pd_sync_across_ranks,
            strict_constraint_range=args.pd_strict_constraint_range,
            log_every=args.pd_log_every,
            component_weight_map=component_weights,
            component_cost_limit_map=component_cost_limits,
        )
        # Keep distributed dual-update collectives rank-aligned from step 0, even
        # when local batches do not contain every task yet.
        wrapper._initialize_cost_state(
            _infer_static_cost_names(cost_components_fn, task_name)
        )
        wrappers[task_name] = wrapper

    fixed_null_receipt = None
    if bool(getattr(args, "clasp_fixed_null_set", False)):
        from clasp.set_grpo import load_fixed_null_score_receipt

        receipt_path = str(
            getattr(args, "clasp_fixed_null_receipt_path", "") or ""
        ).strip()
        if not receipt_path:
            raise ValueError(
                "Fixed-null SET-GRPO requires --clasp_fixed_null_receipt_path."
            )
        fixed_null_receipt = load_fixed_null_score_receipt(receipt_path)
        missing_tasks = sorted(
            set(str(task).strip().lower() for task in tasks)
            - set(fixed_null_receipt.identity["tasks"])
        )
        if missing_tasks:
            raise ValueError(
                "Fixed-null construction receipt has no records for active tasks: "
                + ", ".join(missing_tasks)
            )

    return MultiTaskPrimalDualManager(
        wrappers,
        fixed_null_score_receipt=fixed_null_receipt,
    )


def maybe_load_primal_dual_state(
    output_dir: str,
    manager: MultiTaskPrimalDualManager,
    *,
    required: bool,
    checkpoint_dir: str | None = None,
    default_state: dict | None = None,
) -> None:
    if default_state is not None:
        manager.load_state_dict(default_state, from_checkpoint=False)

    candidate_checkpoint_dir = checkpoint_dir or find_latest_checkpoint_dir(output_dir)
    if not candidate_checkpoint_dir:
        return

    state_path = os.path.join(candidate_checkpoint_dir, "multitask_primal_dual_state.json")
    manager.mark_resume_checkpoint_requested(has_trainer_state=False)
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as handle:
            manager.load_state_dict(json.load(handle), from_checkpoint=True)
        print_primary(f"[GRPO] Loaded multi-task primal-dual state from: {state_path}")
        return

    if required:
        raise RuntimeError(
            f"Expected {state_path} for resume, but it does not exist. "
            "Use --no-pd_require_state_on_resume to continue anyway."
        )


def verify_ground_truth_rewards(
    *,
    tasks: list[str],
    data_root: str,
    max_samples_per_task: int,
    prefer_subset_files: bool,
    no_reason: bool = False,
) -> None:
    failures = []
    checked = 0
    graded_reward = _grpo_graded_reward_enabled()

    def expected_objective_for_task(
        task: str, *, output_text: str, input_text: str, instance_json,
    ) -> float:
        if graded_reward and task == "cvrp":
            # Candidate and reference are the same verified routes: d / d_ref = 1.
            # Do not derive the expected value from the objective head under test.
            return 1.5
        if graded_reward and task == "mis":
            instance = json.loads(coerce_json_string(instance_json))
            num_nodes, edges = _mis_meta(instance, input_text)
            nodes = _parse_mis_solution(output_text)["node_set"]
            if (
                num_nodes is None or num_nodes <= 0
                or not _check_mis_constraints(nodes, num_nodes, edges)[2]
            ):
                raise RuntimeError("Ground-truth reward verification failed: invalid MIS reference")
            # Compute the normalized verified cardinality independently of both
            # the objective head and the possibly incorrect declared Objective.
            return 1.0 + len(nodes) / num_nodes
        if graded_reward and os.environ.get("GRPO_VERIFIED_TRAIN_REFERENCE_TEACHER", "0") == "1" and task in ("tsp", "op", "jssp"):
            from clasp.rewards.verified_train_reference import canonical_verified_solution
            from clasp.rewards.external_baseline_tasks import validate_graded_instance
            instance = json.loads(coerce_json_string(instance_json))
            verified = canonical_verified_solution(task, output_text, instance, input_text)
            if verified is None:
                raise RuntimeError("Ground-truth reward verification failed: invalid training reference")
            if task != "op":
                return 1.5
            total = sum(validate_graded_instance(task, instance)[1])
            return 1.0 + (verified[1] / total if total > 0 else 0.0)
        if task != "tsp":
            return 2.0
        raw_gap = os.environ.get("CLASP_TSP_TARGET_GAP", "")
        raw_bonus = os.environ.get("CLASP_TSP_TARGET_GAP_BONUS", "")
        if not raw_gap or not raw_bonus:
            return 2.0
        try:
            target_gap = float(raw_gap)
            bonus = float(raw_bonus)
        except ValueError:
            return 2.0
        if not (math.isfinite(target_gap) and math.isfinite(bonus)):
            return 2.0
        if target_gap <= 0.0 or bonus <= 0.0:
            return 2.0
        return 2.0 + (2.0 * bonus)

    for task in tasks:
        triplet = build_task_triplets([task])[task]
        json_files = list_split_json_files(
            data_root,
            task,
            "train",
            prefer_subset_files=prefer_subset_files,
            required=True,
        )
        dataset = load_and_merge_task_datasets(json_files)
        take = min(len(dataset), max(1, int(max_samples_per_task)))

        for idx in range(take):
            row = dataset[idx]
            instruction = str(row.get("instruction", ""))
            transformed = transform_example_for_task(
                task,
                instruction,
                str(row.get("input", "")),
                str(row.get("output", "")),
                row.get("instance_json"),
            )
            input_text = str(transformed["input"])
            output_text = str(transformed["output"])
            instance_json = transformed["instance_json"]
            instruction = str(transformed["instruction"])

            prompt = build_alpaca_prompt(instruction, input_text, no_reason=no_reason)
            kwargs = {
                "answer": [output_text],
                "input": [input_text],
                "task_type": [task],
                "instance_json": [coerce_json_string(instance_json)],
            }

            structure_score = float(triplet[0]([prompt], [output_text], **kwargs)[0])
            constraint_score = float(triplet[1]([prompt], [output_text], **kwargs)[0])
            objective_score = float(triplet[2]([prompt], [output_text], **kwargs)[0])
            expected_objective_score = expected_objective_for_task(
                task, output_text=output_text, input_text=input_text,
                instance_json=instance_json,
            )
            checked += 1

            if not (
                abs(structure_score - 1.0) < 1e-9
                and abs(constraint_score - 1.0) < 1e-9
                and abs(objective_score - expected_objective_score) < 1e-9
            ):
                failures.append(
                    {
                        "task": task,
                        "sample_index": idx,
                        "structure": structure_score,
                        "constraint": constraint_score,
                        "objective": objective_score,
                        "expected_objective": expected_objective_score,
                    }
                )

    if failures:
        preview = "; ".join(
            f"{item['task']}#{item['sample_index']} "
            f"(s={item['structure']:.4f}, c={item['constraint']:.4f}, "
            f"o={item['objective']:.4f}, expected_o={item['expected_objective']:.4f})"
            for item in failures[:10]
        )
        raise RuntimeError(
            "Ground-truth reward verification failed. "
            f"Checked={checked}, failures={len(failures)}. Examples: {preview}"
        )

    print_primary(
        "[RewardCheck] Ground-truth verification passed for all checked samples. "
        f"Checked={checked}, expected=(structure=1.0,constraint=1.0,"
        f"objective=task-specific reference reward), GRPO_GRADED_REWARD={int(graded_reward)}."
    )
