"""Past-only value baselines and RLOO on the pinned TRL rollout/loss machinery.

All arms use the native repeated sampler, one optimizer step per rollout buffer,
no teacher, and the same fixed-horizon actor reduction. The external baseline is
snapshotted before generation and only updated from the globally gathered batch
for use by subsequent batches. This is an experimental scalar-reward estimator,
not canonical CLASP or a complete SAO implementation.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
import warnings
from collections import deque
from pathlib import Path

import torch
from trl import GRPOConfig

from clasp.rewards.external_baseline import baseline_instance_size
from clasp.rewards.external_baseline import ExternalBaselineTracker
from clasp.trainers.verified_teacher_grpo import VerifiedTeacherGRPOTrainer

STATE_FILENAME = "external_baseline_state.json"


class ExternalBaselineGRPOConfig(GRPOConfig):
    """Allow G=1 only for this opt-in external-advantage trainer.

    TRL 0.24's minimum-group guard is specific to sibling advantages. Let the
    parent perform all preceding configuration/distribution checks, catch only
    that precise guard, and preserve the one remaining Liger compatibility
    check. Fail on upstream control-flow drift instead of suppressing arbitrary
    configuration errors. No generation count is temporarily falsified.
    """

    def __post_init__(self):
        if self.num_generations != 1:
            return super().__post_init__()
        source = inspect.getsource(GRPOConfig.__post_init__)
        tail = source.split("        if self.num_generations < 2:", 1)
        expected_tail = '''
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )

        if self.delta is not None and self.use_liger_loss:
            raise ValueError("Liger loss does not support two-sided GRPO loss yet.")'''
        if len(tail) != 2 or tail[1].strip() != expected_tail.strip():
            raise RuntimeError("Single-rollout config requires the audited TRL 0.24 validation tail")
        expected_error = (
            "GRPO requires at least 2 generations per prompt to calculate the advantages. "
            "You provided 1, which is less than the minimum required."
        )
        try:
            super().__post_init__()
        except ValueError as exc:
            if str(exc) != expected_error:
                raise
        else:
            raise RuntimeError("TRL minimum-generation guard changed; review single-rollout support")
        if self.delta is not None and self.use_liger_loss:
            raise ValueError("Liger loss does not support two-sided GRPO loss yet.")


class ExternalBaselineGRPOTrainer(VerifiedTeacherGRPOTrainer):
    """Replace advantages while retaining TRL generation, clipping and loss."""

    def __init__(self, *args, advantage_estimator="external", baseline_initial=0.0,
                 baseline_scale=1.0, reward_contract=None, **kwargs):
        if advantage_estimator not in ("rloo", "external"):
            raise ValueError("advantage_estimator must be rloo or external")
        if kwargs.pop("verified_teacher_weight", 0.0) != 0.0:
            raise ValueError("External-baseline attribution forbids teacher loss")
        self.advantage_estimator = advantage_estimator
        self.reward_contract = None
        if reward_contract is not None:
            from clasp.baselines.quality_signal_dev import validate_reward_contract
            self.reward_contract = validate_reward_contract(reward_contract)
        self.external_baseline = ExternalBaselineTracker(
            initial_value=baseline_initial, scale=baseline_scale)
        self._baseline_reward_buffer = None
        self._baseline_generation_active = False
        super().__init__(*args, verified_teacher_weight=0.0, **kwargs)
        expected = {
            "scale_rewards": "none", "loss_type": "dr_grpo", "beta": 0.0,
            "num_iterations": 1, "importance_sampling_level": "token",
            "use_vllm": False, "use_liger_loss": False,
            "mask_truncated_completions": False,
        }
        for name, value in expected.items():
            if getattr(self, name, getattr(self.args, name, None)) != value:
                raise ValueError(f"External-baseline attribution requires {name}={value!r}")
        if int(self.num_generations) < (2 if advantage_estimator == "rloo" else 1):
            raise ValueError("RLOO requires at least two siblings")
        if self.args.steps_per_generation != self.args.gradient_accumulation_steps:
            raise ValueError("One full rollout buffer per optimizer step is required")
        if self.reward_weights.numel() != 1 or self.reward_weights.item() != 1.0:
            raise ValueError("Attribution requires one unscaled graded verifier reward")
        if self.args.eval_strategy != "no":
            raise ValueError("Use separate raw-policy evaluation; online value updates are train-only")

    def _gather_objects(self, value):
        if int(self.accelerator.num_processes) == 1:
            return [value]
        if not torch.distributed.is_initialized():
            raise RuntimeError("Distributed baseline requires an initialized process group")
        gathered = [None] * int(self.accelerator.num_processes)
        torch.distributed.all_gather_object(gathered, value)
        return gathered

    def _consensus(self, value, label):
        values = self._gather_objects(value)
        if any(item != values[0] for item in values):
            raise RuntimeError(f"Distributed external-baseline {label} differs across ranks")

    def _input_metadata(self, inputs):
        local = []
        local_error = None
        try:
            if not isinstance(inputs, list) or not inputs:
                raise ValueError("Baseline inputs must be nonempty completion-aligned rows")
            for row in inputs:
                if not isinstance(row, dict):
                    raise ValueError("Baseline input rows must be dictionaries")
                task = str(row.get("task_type", "")).lower()
                size = baseline_instance_size(task, row)
                # No reference solution, sampled action, or reward enters baseline features.
                prompt_key = hashlib.sha256(json.dumps(
                    [task, row.get("instance_json"), row.get("input"), row.get("prompt")],
                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                local.append((task, int(size), prompt_key))
        except Exception as exc:
            # Every rank must enter the same collective even if only one rank
            # encounters bad metadata. Never admit a partially built local list.
            local_error = f"{type(exc).__name__}: {exc}"
            local = []
        gathered = self._gather_objects({"rows": local, "error": local_error})
        if (not isinstance(gathered, list) or len(gathered) != int(self.accelerator.num_processes)
                or any(not isinstance(item, dict) or set(item) != {"rows", "error"}
                       or not isinstance(item["rows"], list)
                       or (item["error"] is not None and not isinstance(item["error"], str))
                       for item in gathered)):
            raise RuntimeError("Invalid gathered baseline metadata envelope")
        errors = [(rank, item["error"]) for rank, item in enumerate(gathered)
                  if item["error"] is not None]
        if errors:
            raise RuntimeError(f"Baseline metadata validation failed across ranks: {errors}")
        parts = [item["rows"] for item in gathered]
        if any(len(rows) != len(local) for rows in parts):
            raise RuntimeError("Rank rollout sizes differ")
        rows = [row for part in parts for row in part]
        g = int(self.num_generations)
        if len(rows) % g:
            raise RuntimeError("Global rollout batch contains incomplete groups")
        for start in range(0, len(rows), g):
            if any(row != rows[start] for row in rows[start:start + g]):
                raise RuntimeError("Native sampler did not preserve sibling prompt identity")
        return rows

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        if not self._baseline_generation_active or self._baseline_reward_buffer is not None:
            raise RuntimeError("Expected one reward calculation per snapshotted rollout buffer")
        if not torch.isfinite(rewards).all():
            raise RuntimeError("Graded verifier produced nonfinite/missing rewards")
        self._baseline_reward_buffer = rewards.detach().clone()
        return rewards

    def _generate_and_score_completions(self, inputs):
        if not self.model.training:
            raise RuntimeError("External-baseline trainer uses separate policy evaluation")
        if self._baseline_generation_active:
            raise RuntimeError("Nested baseline generation is unsupported")
        rows = self._input_metadata(inputs)
        snapshot = self.external_baseline.snapshot([(task, size) for task, size, _ in rows])
        self._consensus(snapshot.state_sha256, "pre-generation state")
        self._baseline_reward_buffer = None
        self._baseline_generation_active = True
        started = time.monotonic()
        try:
            with warnings.catch_warnings():
                # Stock G=1 sample-STD is undefined and used only for logging
                # when scale_rewards='none'. Replace its metrics below.
                if self.num_generations == 1:
                    warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0.*")
                output = super()._generate_and_score_completions(inputs)
        finally:
            self._baseline_generation_active = False
        raw = self._baseline_reward_buffer
        if raw is None or tuple(raw.shape) != (len(rows), 1):
            raise RuntimeError("Missing globally gathered scalar rewards")
        rewards = raw[:, 0]
        values = rewards.cpu().tolist()
        if self.advantage_estimator == "rloo":
            grouped = rewards.view(-1, self.num_generations)
            advantages = (grouped - (grouped.sum(1, keepdim=True) - grouped)
                          / (self.num_generations - 1)).reshape(-1) / snapshot.scale
        else:
            advantages = torch.tensor(self.external_baseline.advantages(snapshot, values),
                                      dtype=torch.float32, device=rewards.device)
        if not torch.isfinite(advantages).all():
            raise RuntimeError("Nonfinite external/RLOO advantages")
        self._consensus(advantages.cpu().tolist(), "global advantages")
        local_count = len(inputs)
        start = int(self.accelerator.process_index) * local_count
        output["advantages"] = advantages[start:start + local_count].detach()
        grouped = rewards.view(-1, self.num_generations)
        group_std = grouped.std(1, unbiased=False)
        metrics = self._metrics["train"]
        metrics["reward_std"][-1] = group_std.mean().item()
        metrics["frac_reward_zero_std"][-1] = (group_std == 0).float().mean().item()
        recorded = self._logs["advantages"]
        replacement = advantages.cpu().tolist()
        if isinstance(recorded, deque):
            for _ in range(min(len(recorded), len(replacement))):
                recorded.pop()
            recorded.extend(replacement)
        else:
            del recorded[-len(replacement):]
            recorded.extend(replacement)
        baseline_values = torch.tensor(snapshot.values, device=rewards.device)
        invalid = rewards < 0
        metrics_to_add = {
            "external/baseline_revision": snapshot.revision,
            "external/baseline_mean": baseline_values.mean().item(),
            "external/baseline_mse": ((rewards - baseline_values) ** 2).mean().item(),
            "external/advantage_mean": advantages.mean().item(),
            "external/advantage_abs_mean": advantages.abs().mean().item(),
            "external/zero_advantage_fraction": (advantages == 0).float().mean().item(),
            "external/all_invalid_group_fraction": (grouped < 0).all(1).float().mean().item(),
            "external/invalid_nonzero_advantage_fraction": (
                (advantages[invalid] != 0).float().mean().item() if invalid.any() else 0.0),
            "external/rollout_responses": len(rows),
            "external/unique_prompts": len({row[2] for row in rows}),
            "external/prompt_groups": len(rows) // self.num_generations,
            "external/generation_and_scoring_seconds": time.monotonic() - started,
        }
        for name, value in metrics_to_add.items():
            if not math.isfinite(value):
                raise RuntimeError(f"Nonfinite baseline diagnostic: {name}")
            metrics[name].append(float(value))
        self.external_baseline.update(snapshot, values)
        next_state = self.external_baseline.state_dict()
        self._consensus(next_state, "post-update state")
        if self.accelerator.is_main_process:
            record = {"global_step_before_update": int(self.state.global_step),
                      "recorded_at_unix": time.time(),
                      "estimator": self.advantage_estimator,
                      "pre_batch_state_sha256": snapshot.state_sha256,
                      "post_batch_state_sha256": next_state["sha256"],
                      **metrics_to_add}
            with (Path(self.args.output_dir) / "external_baseline_batches.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        self._baseline_reward_buffer = None
        return output

    def _baseline_checkpoint_payload(self):
        payload = {"schema_version": 1, "global_step": int(self.state.global_step),
                "advantage_estimator": self.advantage_estimator,
                "num_generations": int(self.num_generations),
                "max_completion_length": int(self.max_completion_length),
                "tracker": self.external_baseline.state_dict()}
        if self.reward_contract is not None:
            # Retain the exact historical payload for all old profiles.
            payload["reward_contract"] = json.loads(json.dumps(self.reward_contract, allow_nan=False))
        return payload

    def _save_checkpoint(self, model, trial):
        if self._baseline_generation_active:
            raise RuntimeError("Cannot checkpoint an unfinished baseline rollout")
        super()._save_checkpoint(model, trial)
        self.accelerator.wait_for_everyone()
        directory = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
        from clasp.models.external_baseline_state import save_rank_state

        failure = None
        try:
            save_rank_state(directory, self.optimizer, self.lr_scheduler,
                            rank=int(self.accelerator.process_index),
                            world_size=int(self.accelerator.num_processes),
                            step=int(self.state.global_step))
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        failures = self._gather_objects(failure)
        if any(item is not None for item in failures):
            raise RuntimeError(f"Rank-owned checkpoint verification failed: {failures}")
        payload = self._baseline_checkpoint_payload()
        if payload["tracker"]["revision"] != payload["global_step"]:
            raise RuntimeError("Baseline revision differs from optimizer step at checkpoint")
        self._consensus(payload, "checkpoint state")
        if self.accelerator.is_main_process:
            target = directory / STATE_FILENAME
            temporary = target.with_suffix(".json.tmp")
            with temporary.open("w") as handle:
                json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        self.accelerator.wait_for_everyone()

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint:
            path = Path(checkpoint) / STATE_FILENAME
            payload = json.loads(path.read_text())
            if self.reward_contract is not None or "reward_contract" in payload:
                from clasp.baselines.quality_signal_dev import check_resume_reward_contract
                check_resume_reward_contract(self.reward_contract, payload)
            expected = self._baseline_checkpoint_payload()
            for name in ("schema_version", "advantage_estimator", "num_generations", "max_completion_length"):
                if payload.get(name) != expected[name]:
                    raise ValueError(f"Resume baseline configuration differs: {name}")
            state = json.loads((Path(checkpoint) / "trainer_state.json").read_text())
            if payload["global_step"] != state["global_step"]:
                raise ValueError("Baseline checkpoint step differs from trainer state")
            if payload["tracker"]["revision"] != payload["global_step"]:
                raise ValueError("Baseline revision differs from saved optimizer step")
            restored = ExternalBaselineTracker.from_state_dict(payload["tracker"])
            # Do not accept a valid checkpoint from a different baseline recipe.
            reference = self.external_baseline.state_dict()
            if restored.state_dict()["config"] != reference["config"]:
                raise ValueError("Resume baseline tracker configuration differs")
            self._consensus(payload, "restored checkpoint")
            self.external_baseline = restored
        result = super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint:
            from clasp.models.external_baseline_state import (
                _write_json_atomic,
                restore_rank_state,
            )

            failure = None
            try:
                rank = int(self.accelerator.process_index)
                receipt = restore_rank_state(
                    checkpoint, self.optimizer, self.lr_scheduler,
                    rank=rank, world_size=int(self.accelerator.num_processes),
                    step=int(payload["global_step"]))
                _write_json_atomic(Path(self.args.output_dir)
                                   / f"external_resume_{payload['global_step']}_rank{rank}.json",
                                   receipt)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
            failures = self._gather_objects(failure)
            if any(item is not None for item in failures):
                raise RuntimeError(f"Rank-owned restoration failed: {failures}")
        return result
