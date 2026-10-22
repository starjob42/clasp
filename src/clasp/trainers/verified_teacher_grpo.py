"""Stock TRL GRPO with a separately verified teacher likelihood objective.

The rollout rewards, advantages, clipping, KL term and on-policy reduction are
computed by GRPOTrainer itself. Only the teacher utilities are shared with the
older constrained trainer; none of its policy/advantage overrides are inherited.
"""
from __future__ import annotations

import math

from trl import GRPOTrainer

from .primal_dual_grpo import PrimalDualGRPOTrainer as _TeacherUtilities


class VerifiedTeacherGRPOTrainer(GRPOTrainer):
    # Reuse the tested verifier/selector and distributed window normalization.
    # Explicit aliases make the shared surface reviewable and leave historical
    # trainer implementations unchanged.
    _append_metric = _TeacherUtilities._append_metric
    _extract_task_types = _TeacherUtilities._extract_task_types
    _extract_batch_raw_values = staticmethod(_TeacherUtilities._extract_batch_raw_values)
    _maybe_json_load = staticmethod(_TeacherUtilities._maybe_json_load)
    _model_token_vocab_size = _TeacherUtilities._model_token_vocab_size
    _sanitize_token_ids = _TeacherUtilities._sanitize_token_ids
    _safe_pad_token_id = _TeacherUtilities._safe_pad_token_id
    _verified_improvement_target = _TeacherUtilities._verified_improvement_target
    _tokenize_verified_target = _TeacherUtilities._tokenize_verified_target
    _verified_completion_objective = staticmethod(_TeacherUtilities._verified_completion_objective)
    _verified_teacher_prompt_key = staticmethod(_TeacherUtilities._verified_teacher_prompt_key)
    _gather_verified_teacher_rows = _TeacherUtilities._gather_verified_teacher_rows
    _validate_verified_teacher_window = _TeacherUtilities._validate_verified_teacher_window
    _verified_raw_target = _TeacherUtilities._verified_raw_target
    _verified_constructive_target = _TeacherUtilities._verified_constructive_target
    _attach_verified_improvement_tensors = _TeacherUtilities._attach_verified_improvement_tensors
    _compute_improvement_distillation_losses = _TeacherUtilities._compute_improvement_distillation_losses
    _restore_configured_gradient_checkpointing = _TeacherUtilities._restore_configured_gradient_checkpointing
    _validate_fixed_kl_reference = _TeacherUtilities._validate_fixed_kl_reference

    def __init__(self, *args, verified_teacher_weight=0.0, **kwargs):
        weight = float(verified_teacher_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("verified_teacher_weight must be finite and nonnegative")
        self.grpo_verified_improvement = weight > 0
        self.grpo_verified_improvement_weight = weight
        self.grpo_verified_improvement_max_tokens = 3000
        self.grpo_verified_improvement_max_passes = 2
        self.grpo_verified_teacher_selection = "feasibility_incumbent"
        self.grpo_verified_teacher_normalization = "optimizer_window"
        super().__init__(*args, **kwargs)
        if self.grpo_verified_improvement:
            self._validate_verified_teacher_window()

    def _improvement_distillation_specs(self):
        if not self.grpo_verified_improvement:
            return []
        return [
            {"prefix": f"grpo_verified_{task}", "task": task,
             "weight": self.grpo_verified_improvement_weight,
             "max_tokens": self.grpo_verified_improvement_max_tokens,
             "min_gain": 0.0, "verified_grpo": True}
            for task in ("cvrp", "mis")
        ]

    def _generate_and_score_completions(self, inputs):
        self._validate_fixed_kl_reference()
        try:
            output = super()._generate_and_score_completions(inputs)
        finally:
            # TRL 0.24 loses use_reentrant=False when leaving generation.
            # Apply the same restoration in teacher and weight-zero control.
            self._restore_configured_gradient_checkpointing()
        if self.model.training and self.grpo_verified_improvement:
            self._attach_verified_improvement_tensors(
                inputs, output, self._extract_task_types(inputs))
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(
            model, inputs, return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch)
        if not self.model.training or not self.grpo_verified_improvement:
            return loss
        teacher_loss, stats = self._compute_improvement_distillation_losses(
            model, inputs, inputs["prompt_ids"], inputs["prompt_mask"],
            inputs["completion_ids"], loss.reshape(1))
        for name, value in stats.items():
            self._append_metric("train", name, value)
        if teacher_loss is not None:
            # Stock TRL already divides its loss by accumulation. The teacher
            # utility scales by A*W/M, so divide this additional term once too.
            loss = loss + teacher_loss / self.current_gradient_accumulation_steps
        return loss
