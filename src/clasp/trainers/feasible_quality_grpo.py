"""Opt-in verified quality credit layered over the calibrated B9 RLOO trainer.

The entrypoint applies the existing budget checkpoint wrapper outside this
class. Generation, token loss, rank-owned optimizer state and baseline tracking
remain inherited. No historical trainer selects this extension implicitly.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import time

import torch

from clasp.models.external_baseline_state import _write_json_atomic
from clasp.rewards.feasible_quality import (
    decompose_advantages, validate_advantage_contract, verified_reward_metadata,
)
from clasp.trainers.external_baseline_grpo import ExternalBaselineGRPOTrainer

CONTRACT_FILENAME = "feasible_quality_contract.json"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class FeasibleQualityGRPOTrainer(ExternalBaselineGRPOTrainer):
    """Add independently audited quality coefficients after native collection."""

    def __init__(self, *args, advantage_contract, **kwargs):
        self.advantage_contract = validate_advantage_contract(advantage_contract)
        self._quality_contract_json = _canonical(self.advantage_contract)
        self._quality_contract_sha256 = hashlib.sha256(self._quality_contract_json.encode()).hexdigest()
        if kwargs.get("advantage_estimator", "external") != "rloo":
            raise ValueError("Feasible-quality credit requires the calibrated RLOO anchor")
        if kwargs.get("baseline_scale", 1.0) != 1.0:
            raise ValueError("Feasible-quality credit requires baseline_scale=1")
        self._quality_generation_active = False
        self._quality_metadata_buffer = None
        self._quality_reward_buffer = None
        self._quality_prompt_rows = None
        super().__init__(*args, **kwargs)
        if self.num_generations != 9:
            raise ValueError("Feasible-quality attribution requires B9")
        if self.reward_contract is None or self.reward_contract["mode"] != "reference_sigmoid_v2":
            raise ValueError("Feasible-quality attribution requires a calibrated reward contract")
        if not hasattr(self, "_budget_contract_json"):
            raise ValueError("Feasible-quality trainer requires the budget checkpoint wrapper")
        budget = json.loads(self._budget_contract_json)
        if (budget.get("advantage_contract") != self.advantage_contract
                or budget.get("task") != self.reward_contract["task"]):
            raise ValueError("Budget and feasible-quality contracts differ")
        self._quality_reward_contract_json = _canonical(self.reward_contract)
        self._consensus(self._quality_contract_json, "feasible-quality contract")

    def _quality_assert_contract(self):
        if (_canonical(self.advantage_contract) != self._quality_contract_json
                or _canonical(self.reward_contract) != self._quality_reward_contract_json):
            raise ValueError("Active feasible-quality contract changed after initialization")

    def _input_metadata(self, inputs):
        rows = super()._input_metadata(inputs)
        self._quality_prompt_rows = rows
        return rows

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        if not self._quality_generation_active or self._quality_metadata_buffer is not None:
            raise RuntimeError("Expected one verified quality collection per rollout")
        local, error = [], None
        try:
            local = verified_reward_metadata(
                self.reward_contract["task"], prompts, completions, inputs)
            if len(completion_ids_list) != len(local):
                raise ValueError("Completion token rows differ from verified metadata")
            for row, tokens in zip(local, completion_ids_list):
                if (not isinstance(tokens, (list, tuple))
                        or any(type(token) is not int or token < 0 for token in tokens)):
                    raise ValueError("Completion token IDs must be aligned nonnegative integers")
                row["completion_token_count"] = len(tokens)
                row["completion_ids_sha256"] = hashlib.sha256(_canonical(list(tokens)).encode()).hexdigest()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        envelopes = self._gather_objects({
            "rank": int(self.accelerator.process_index), "rows": local, "error": error})
        world = int(self.accelerator.num_processes)
        if (len(envelopes) != world or any(
                not isinstance(item, dict) or set(item) != {"rank", "rows", "error"}
                or type(item["rank"]) is not int or item["rank"] != rank
                or not isinstance(item["rows"], list)
                or (item["error"] is not None and not isinstance(item["error"], str))
                for rank, item in enumerate(envelopes))):
            raise RuntimeError("Malformed gathered quality metadata envelope")
        failures = [(rank, item["error"]) for rank, item in enumerate(envelopes) if item["error"] is not None]
        if failures:
            raise RuntimeError(f"Quality verification failed across ranks: {failures}")
        if any(len(item["rows"]) != len(inputs) for item in envelopes):
            raise RuntimeError("Rank quality metadata sizes differ")
        rows = [row for item in envelopes for row in item["rows"]]
        expected = self._quality_prompt_rows
        if (expected is None or len(rows) != len(expected)
                or any(row.get("prompt_key") != prompt[2] for row, prompt in zip(rows, expected))):
            raise RuntimeError("Quality metadata differs from native global prompt order")
        for start in range(0, len(rows), self.num_generations):
            siblings = rows[start:start + self.num_generations]
            if (len(siblings) != self.num_generations
                    or any(row["reference"] != siblings[0]["reference"] for row in siblings)):
                raise RuntimeError("Quality siblings have different verified training references")
        rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        expected_rewards = torch.tensor([row["anchor_reward"] for row in rows],
                                        dtype=rewards.dtype, device=rewards.device).reshape(-1, 1)
        if not torch.equal(rewards, expected_rewards):
            raise RuntimeError("Verified metadata differs from globally gathered anchor rewards")
        self._quality_metadata_buffer = rows
        self._quality_reward_buffer = rewards.detach().clone()
        return rewards

    def _generate_and_score_completions(self, inputs):
        if self._quality_generation_active:
            raise RuntimeError("Nested feasible-quality generation is unsupported")
        self._budget_collective(self._quality_assert_contract, "active feasible-quality contract")
        self._quality_generation_active = True
        self._quality_metadata_buffer = self._quality_reward_buffer = None
        self._quality_prompt_rows = None
        try:
            output = super()._generate_and_score_completions(inputs)
            rows, rewards = self._quality_metadata_buffer, self._quality_reward_buffer
            if rows is None or rewards is None or tuple(rewards.shape) != (len(rows), 1):
                raise RuntimeError("Missing verified global quality rollout")
            # This is the inherited B9 RLOO expression, including its fixed scale.
            grouped = rewards[:, 0].view(-1, self.num_generations)
            base = (grouped - (grouped.sum(1, keepdim=True) - grouped)
                    / (self.num_generations - 1)).reshape(-1)
            count = len(inputs)
            start = int(self.accelerator.process_index) * count
            if not torch.equal(output["advantages"], base[start:start + count]):
                raise RuntimeError("Inherited RLOO advantages differ from the declared anchor")
            parts = decompose_advantages(base.cpu().tolist(), rows, self.advantage_contract)
            auxiliary = torch.tensor(parts["quality"], dtype=base.dtype, device=base.device)
            weight = self.advantage_contract["weight"]
            combined = base if weight == 0.0 else base + weight * auxiliary
            if not torch.isfinite(combined).all():
                raise RuntimeError("Nonfinite combined feasible-quality advantages")
            self._consensus(combined.cpu().tolist(), "combined quality advantages")
            if weight != 0.0:
                output["advantages"] = combined[start:start + count].detach()
                recorded, replacement = self._logs["advantages"], combined.cpu().tolist()
                if isinstance(recorded, deque):
                    for _ in range(min(len(recorded), len(replacement))):
                        recorded.pop()
                    recorded.extend(replacement)
                else:
                    del recorded[-len(replacement):]
                    recorded.extend(replacement)
            for name, value in parts["diagnostics"].items():
                self._metrics["train"][name].append(float(value))
            def record_batch():
                if self.accelerator.is_main_process:
                    record = {
                        "global_step_before_update": int(self.state.global_step),
                        "recorded_at_unix": time.time(),
                        "advantage_contract_sha256": self._quality_contract_sha256,
                        "metadata_sha256": hashlib.sha256(_canonical(rows).encode()).hexdigest(),
                        "metadata": rows, "base": base.cpu().tolist(),
                        "quality": auxiliary.cpu().tolist(), "combined": combined.cpu().tolist(),
                        **parts["diagnostics"],
                    }
                    with (Path(self.args.output_dir) / "feasible_quality_batches.jsonl").open("a") as handle:
                        handle.write(_canonical(record) + "\n")

            self._budget_collective(record_batch, "feasible-quality batch audit")
            return output
        finally:
            self._quality_generation_active = False
            self._quality_metadata_buffer = self._quality_reward_buffer = None
            self._quality_prompt_rows = None

    def _quality_checkpoint_payload(self, step):
        self._quality_assert_contract()
        return {
            "schema_version": 1, "status": "passed", "global_step": int(step),
            "world_size": int(self.accelerator.num_processes),
            "contract_sha256": self._quality_contract_sha256,
            "contract": json.loads(self._quality_contract_json),
            "budget_contract_sha256": self._budget_contract_sha256,
            "reward_contract": self.reward_contract,
        }

    def _quality_validate_checkpoint(self, checkpoint):
        directory = Path(checkpoint)
        state = json.loads((directory / "trainer_state.json").read_text())
        step = state["global_step"]
        if type(step) is not int or not 0 < step <= self.args.max_steps:
            raise ValueError("Invalid feasible-quality checkpoint step")
        observed = json.loads((directory / CONTRACT_FILENAME).read_text())
        if _canonical(observed) != _canonical(self._quality_checkpoint_payload(step)):
            raise ValueError("Resume feasible-quality advantage contract differs")
        return step

    def _save_checkpoint(self, model, trial):
        if self._quality_generation_active:
            raise RuntimeError("Cannot checkpoint an unfinished quality rollout")
        self._budget_collective(self._quality_assert_contract, "active feasible-quality contract")
        super()._save_checkpoint(model, trial)
        step = int(self.state.global_step)
        payload = self._quality_checkpoint_payload(step)
        self._consensus(payload, "feasible-quality checkpoint")

        def seal():
            if self.accelerator.is_main_process:
                directory = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{step}"
                _write_json_atomic(directory / CONTRACT_FILENAME, payload)

        self._budget_collective(seal, "feasible-quality checkpoint sealing")
        self.accelerator.wait_for_everyone()

    def _load_optimizer_and_scheduler(self, checkpoint):
        self._budget_collective(self._quality_assert_contract, "active feasible-quality contract")
        if checkpoint:
            self._budget_collective(lambda: self._quality_validate_checkpoint(checkpoint),
                                    "feasible-quality resume validation")
        return super()._load_optimizer_and_scheduler(checkpoint)
