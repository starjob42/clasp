"""Opt-in checkpoint fidelity for the declared four-arm budget comparison.

This wrapper changes no sampler, generation, reward, advantage, or loss method.
Controls receive the same rank-owned optimizer/scheduler persistence already
used by ExternalBaselineGRPOTrainer. All arms bind their immutable run contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from clasp.models.external_baseline_state import (
    _write_json_atomic,
    restore_rank_state,
    save_rank_state,
)

CONTRACT_FILENAME = "budget_resume_contract.json"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class BudgetMatchedCheckpointMixin:
    """Seal the run identity and restore rank-owned state without a fallback."""

    _budget_base_owns_rank_state = False

    def __init__(self, *args, budget_resume_contract, **kwargs):
        contract = json.loads(_canonical(budget_resume_contract))
        if not isinstance(contract, dict):
            raise ValueError("Budget resume contract must be an object")
        for name in ("protocol", "method", "task", "seed", "horizon"):
            if name not in contract:
                raise ValueError(f"Budget resume contract is missing {name}")
        if type(contract["horizon"]) is not int or contract["horizon"] <= 0:
            raise ValueError("Budget resume horizon must be a positive integer")
        self._budget_contract_json = _canonical(contract)
        self._budget_contract_sha256 = hashlib.sha256(
            self._budget_contract_json.encode()).hexdigest()
        super().__init__(*args, **kwargs)
        if self.args.max_steps != contract["horizon"]:
            raise ValueError("Trainer horizon differs from budget resume contract")
        if self.args.ignore_data_skip:
            raise ValueError("Budget continuation requires ignore_data_skip=False")

    def _budget_gather(self, value):
        size = int(self.accelerator.num_processes)
        if size == 1:
            return [value]
        if not torch.distributed.is_initialized():
            raise RuntimeError("Budget checkpoint requires an initialized process group")
        values = [None] * size
        torch.distributed.all_gather_object(values, value)
        return values

    def _budget_collective(self, action, label):
        result, failure = None, None
        try:
            result = action()
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        failures = self._budget_gather(failure)
        if any(item is not None for item in failures):
            raise RuntimeError(f"Budget {label} failed: {failures}")
        return result

    def _budget_payload(self, step):
        return {
            "schema_version": 1,
            "status": "passed",
            "global_step": int(step),
            "world_size": int(self.accelerator.num_processes),
            "contract_sha256": self._budget_contract_sha256,
            "contract": json.loads(self._budget_contract_json),
        }

    def _budget_validate_checkpoint(self, checkpoint):
        directory = Path(checkpoint)
        state = json.loads((directory / "trainer_state.json").read_text())
        step = state["global_step"]
        contract = json.loads(self._budget_contract_json)
        if type(step) is not int or not 0 < step <= contract["horizon"]:
            raise ValueError("Checkpoint has an invalid optimizer step")
        if state.get("max_steps") != contract["horizon"]:
            raise ValueError("Checkpoint trainer horizon differs from run contract")
        payload = json.loads((directory / CONTRACT_FILENAME).read_text())
        if _canonical(payload) != _canonical(self._budget_payload(step)):
            raise ValueError("Checkpoint budget contract differs from the declared run")
        return step

    def _save_checkpoint(self, model, trial):
        # Existing RLOO performs its own rank writes and baseline-state sealing.
        super()._save_checkpoint(model, trial)
        self.accelerator.wait_for_everyone()
        step = int(self.state.global_step)
        directory = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{step}"
        if not self._budget_base_owns_rank_state:
            self._budget_collective(
                lambda: save_rank_state(
                    directory, self.optimizer, self.lr_scheduler,
                    rank=int(self.accelerator.process_index),
                    world_size=int(self.accelerator.num_processes), step=step),
                "rank-owned checkpoint verification")
        payload = self._budget_payload(step)
        all_payloads = self._budget_gather(payload)
        if any(item != payload for item in all_payloads):
            raise RuntimeError("Budget checkpoint contracts differ across ranks")

        def seal():
            if self.accelerator.is_main_process:
                state = json.loads((directory / "trainer_state.json").read_text())
                if state.get("global_step") != step or state.get("max_steps") != self.args.max_steps:
                    raise ValueError("Checkpoint trainer state differs from active run")
                _write_json_atomic(directory / CONTRACT_FILENAME, payload)

        self._budget_collective(seal, "checkpoint contract sealing")
        self.accelerator.wait_for_everyone()

    def _load_optimizer_and_scheduler(self, checkpoint):
        step = None
        if checkpoint:
            # Refuse cross-method/source/horizon/diagnostic resumes before loading.
            step = self._budget_collective(
                lambda: self._budget_validate_checkpoint(checkpoint),
                "resume contract validation")
        result = super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint:
            rank = int(self.accelerator.process_index)

            def restore_and_receipt():
                if not self._budget_base_owns_rank_state:
                    rank_receipt = restore_rank_state(
                        checkpoint, self.optimizer, self.lr_scheduler,
                        rank=rank, world_size=int(self.accelerator.num_processes),
                        step=step)
                    _write_json_atomic(
                        Path(self.args.output_dir) / f"external_resume_{step}_rank{rank}.json",
                        rank_receipt)
                receipt = self._budget_payload(step)
                receipt.update(rank=rank, checkpoint=str(Path(checkpoint)))
                _write_json_atomic(
                    Path(self.args.output_dir) / f"budget_resume_{step}_rank{rank}.json",
                    receipt)

            self._budget_collective(restore_and_receipt, "rank-owned restoration")
        return result


def budget_matched_trainer_class(base_class, *, external_baseline):
    """Wrap the selected trainer; the external trainer keeps its rank hooks."""
    if issubclass(base_class, BudgetMatchedCheckpointMixin):
        raise ValueError("Budget checkpoint wrapper must be installed exactly once")

    class BudgetMatchedTrainer(BudgetMatchedCheckpointMixin, base_class):
        _budget_base_owns_rank_state = bool(external_baseline)

    return BudgetMatchedTrainer
