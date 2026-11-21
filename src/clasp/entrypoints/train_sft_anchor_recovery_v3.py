"""Prospective periodic-checkpoint/rescue driver; frozen v2 remains unchanged.

Production retains the original reward/loss/reference, sealed prompt draws and
100-update terminal selection. Checkpoints25/50/75 exist only for continuation.
No function submits jobs or changes an allocation. Exact future trajectory is
admitted only by the pinned actual-PEFT regression, never inferred from a saved
adapter alone. Evaluation-only rescue requires durable original training proof.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainerCallback, set_seed

from clasp.baselines.budget_matched_dev import reward_environment
from clasp.baselines.llmcosolver_f638_training import FOARL_F638_REWARD_FUNCS
from clasp.baselines.quality_signal_dev import PROFILE_ID, REWARD_HEADS
from clasp.baselines.sft_anchor_pilot import ARMS, digest, evaluate_arm, make_contract, validate_contract
from clasp.entrypoints import train_sft_anchor_pilot_v2 as v2
from clasp.models.external_baseline_state import restore_rank_state, semantic_state_sha256
from clasp.rewards.feasible_quality import make_advantage_contract

VERSION = 3
CHECKPOINT_SEAL = "recovery_checkpoint_complete.json"
CHECKPOINT_PREFIX = "recovery_rollout_prefix.json"
PROTOCOL = "sft-anchor-periodic-recovery-v3"
write_json_atomic = v2.write_json_atomic
file_identity = v2.file_identity


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _rank():
    return int(os.environ.get("RANK", "0"))


def _world():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def _gather(value):
    if not torch.distributed.is_initialized():
        return [value]
    values = [None] * _world()
    torch.distributed.all_gather_object(values, value)
    return values


def _normal_rng(value):
    if isinstance(value, np.ndarray):
        return {"numpy_dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _normal_rng(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_normal_rng(item) for item in value)
    return value


def rng_state():
    result = {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.random.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = (torch.cuda.random.get_rng_state_all() if _world() > 1 else
                          torch.cuda.random.get_rng_state())
    return result


def rng_sha256(value=None):
    return semantic_state_sha256(_normal_rng(rng_state() if value is None else value))


def _rng_path(directory, rank, world):
    return Path(directory) / (f"rng_state_{rank}.pth" if world > 1 else "rng_state.pth")


def _checkpoint_tree(directory):
    tree = v2.artifact_tree(directory)
    records = [row for row in tree["files"] if row["path"] != CHECKPOINT_SEAL]
    return {"files": records, "tree_sha256": digest(records)}


def _validate_prefix(rows, contract, step):
    need(type(rows) is list and len(rows) == step, "Committed rollout-prefix length differs")
    for index, row in enumerate(rows):
        expected = [identifier for identifier in contract["training"]["prompt_draws"][index] for _ in range(9)]
        need(row.get("step") == index and row.get("instance_ids") == expected and
             type(row.get("generated_tokens")) is int and row["generated_tokens"] >= 0 and
             type(row.get("completion_fingerprints")) is list and len(row["completion_fingerprints"]) == 36 and
             all(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) for value in row["completion_fingerprints"]) and
             type(row.get("rank_rng_before_generation")) is list and len(row["rank_rng_before_generation"]) == _world() and
             all(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) for value in row["rank_rng_before_generation"]),
             "Committed rollout order/token/fingerprint differs")
    return sum(row["generated_tokens"] for row in rows)


def read_sealed_checkpoint(directory, contract, arm):
    """Read-only admission; incomplete directories and changed rank files fail."""
    directory = Path(directory)
    need(not directory.is_symlink(), "Linked recovery checkpoint forbidden")
    seal = json.loads((directory / CHECKPOINT_SEAL).read_text())
    need(seal.get("protocol") == PROTOCOL and seal.get("status") == "passed" and
         seal.get("operational_version") == VERSION and seal.get("arm") == arm and
         seal.get("pilot_contract_sha256") == contract["contract_sha256"] and
         seal.get("world_size") == _world() and seal.get("source") == file_identity(Path(__file__)),
         "Recovery checkpoint identity/world/source differs")
    step = seal.get("global_step")
    need(type(step) is int and 0 < step <= contract["training"]["horizon"], "Invalid recovery checkpoint step")
    for candidate in directory.parent.glob("checkpoint-*/" + CHECKPOINT_SEAL):
        other = json.loads(candidate.read_text())
        need(not (other.get("status") == "passed" and other.get("pilot_contract_sha256") == contract["contract_sha256"] and
                  type(other.get("global_step")) is int and other["global_step"] > step),
             "Use the newest sealed checkpoint for continuation; intermediate selection forbidden")
    need(_checkpoint_tree(directory) == seal.get("artifact_tree"), "Recovery checkpoint artifact bytes changed")
    prefix = json.loads((directory / CHECKPOINT_PREFIX).read_text())
    tokens = _validate_prefix(prefix.get("rows"), contract, step)
    need(prefix.get("pilot_contract_sha256") == contract["contract_sha256"] and
         prefix.get("global_step") == step and prefix.get("generated_tokens") == tokens and
         seal.get("rollout_prefix_sha256") == file_identity(directory / CHECKPOINT_PREFIX)["sha256"],
         "Recovery committed-prefix binding differs")
    ranks = seal.get("rank_closure", [])
    need(len(ranks) == _world() and all(row.get("rank") == index for index, row in enumerate(ranks)) and
         all(row.get("actor_state_sha256") == ranks[0].get("actor_state_sha256") and
             row.get("global_step") == step for row in ranks), "Recovery rank closure differs")
    return seal, prefix["rows"]


def _complete_lines(path):
    """Complete lines only; preserve and report a truncated final write."""
    path = Path(path)
    if not path.exists():
        return [], None
    need(not path.is_symlink(), "Linked progress file forbidden")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read()
        after = os.fstat(stream.fileno())
    need(all(getattr(before, key) == getattr(after, key) for key in ("st_ino", "st_size", "st_mtime_ns")),
         "Progress file changed during read; preserve a stable completed snapshot")
    chunks = raw.splitlines(keepends=True)
    tail = chunks.pop() if chunks and not chunks[-1].endswith(b"\n") else None
    rows = [json.loads(chunk) for chunk in chunks]
    return rows, ({"size_bytes": len(tail), "sha256": hashlib.sha256(tail).hexdigest()} if tail else None)


class RecoveryClock(TrainerCallback):
    def __init__(self, trainer, stop_after):
        self.trainer, self.stop_after = trainer, stop_after

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        need(len(trainer.recovery_rows) == state.global_step, "Optimizer step has no matching committed rollout")
        if trainer.accelerator.is_main_process:
            v2._append_durable(Path(args.output_dir) / "recovery_optimizer_steps.jsonl", {
                "global_step": state.global_step, "segment_start_step": trainer.recovery_start_step,
                "segment_elapsed_seconds": time.monotonic() - trainer.recovery_started})
        if self.stop_after is not None and state.global_step == self.stop_after:
            control.should_training_stop = True
            control.should_save = True
        return control


class RecoveryStateMixin:
    """Seal periodic native checkpoints and restore observation/RNG continuity."""
    def __init__(self, *args, recovery_checkpoint=None, recovery_stop_after=None, **kwargs):
        self.recovery_checkpoint = Path(recovery_checkpoint) if recovery_checkpoint else None
        self.recovery_rows = []
        self.recovery_start_step = 0
        self.recovery_parent_training_seconds = 0.0
        self.recovery_parent_clock_complete = True
        self.recovery_lineage = []
        self.recovery_discarded = {"responses": 0, "generated_tokens": 0, "optimizer_updates": 0,
                                   "training_seconds_lower_bound": 0.0, "truncated_progress_tail": None}
        self.recovery_started = time.monotonic()
        self.resume_admission = None
        self.resume_rng_audit = None
        super().__init__(*args, **kwargs)
        if self.recovery_checkpoint:
            seal, prefix = self._collective_check(lambda: read_sealed_checkpoint(
                self.recovery_checkpoint, self.pilot_contract, self.recipe_contract["arm"]), "recovery seal admission")
            self._collective_check(lambda: self._validate_checkpoint(self.recovery_checkpoint), "native resume contract")
            self.recovery_rows = prefix
            self.recovery_start_step = seal["global_step"]
            self.actual_windows = [row["instance_ids"] for row in prefix]
            self.actual_generated_tokens = sum(row["generated_tokens"] for row in prefix)
            self.recovery_parent_training_seconds = seal["accounting"]["accepted_training_seconds"]
            self.recovery_parent_clock_complete = False
            self.recovery_lineage = seal["lineage"] + [{"checkpoint": str(self.recovery_checkpoint.resolve()),
                                                       "seal_sha256": file_identity(self.recovery_checkpoint / CHECKPOINT_SEAL)["sha256"],
                                                       "global_step": self.recovery_start_step}]
            closed_segment = self.recovery_checkpoint.parent.parent / "segment_complete.json"
            if closed_segment.exists():
                closure = json.loads(closed_segment.read_text())
                if (closure.get("global_step") == self.recovery_start_step and
                        closure.get("checkpoint_seal_sha256") == file_identity(self.recovery_checkpoint / CHECKPOINT_SEAL)["sha256"]):
                    clock = closure["accounting"]
                    need(clock["accepted_training_seconds"] >= self.recovery_parent_training_seconds,
                         "Closed parent segment clock is shorter than its checkpoint prefix")
                    self.recovery_parent_training_seconds = clock["accepted_training_seconds"]
                    self.recovery_parent_clock_complete = clock.get("accepted_training_clock_complete") is True
                    self.recovery_lineage[-1]["closed_segment_sha256"] = file_identity(closed_segment)["sha256"]
            old, tail = _complete_lines(self.recovery_checkpoint.parent / "recovery_rollout_observations.jsonl")
            need(old[:len(prefix)] == prefix, "Parent observed-prefix differs from sealed checkpoint")
            trailing = old[len(prefix):]
            old_steps, _ = _complete_lines(self.recovery_checkpoint.parent / "recovery_optimizer_steps.jsonl")
            self.recovery_discarded = {
                "responses": seal["discarded_work"]["responses"] + 36 * len(trailing),
                "generated_tokens": seal["discarded_work"]["generated_tokens"] + sum(row["generated_tokens"] for row in trailing),
                "optimizer_updates": seal["discarded_work"]["optimizer_updates"] + sum(row["global_step"] > self.recovery_start_step for row in old_steps),
                # Native allocation accounting captures all charged time. A
                # pre-seal timestamp cannot isolate discarded compute from the
                # checkpoint's own later I/O, so do not invent that split.
                "training_seconds_lower_bound": seal["discarded_work"]["training_seconds_lower_bound"],
                "discarded_training_seconds_measured": None,
                "post_checkpoint_elapsed_seconds_observed": max(
                    0.0, max((row["segment_elapsed_seconds"] for row in old_steps), default=0) - seal["accounting"]["segment_training_seconds"]),
                "truncated_progress_tail": tail}
            self.resume_admission = seal
        self.add_callback(RecoveryClock(self, recovery_stop_after))
        if self.accelerator.is_main_process:
            directory = Path(self.args.output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            need(not (directory / "actual_prompt_windows.jsonl").exists(), "Preserve previous observation output")
            with (directory / "actual_prompt_windows.jsonl").open("x") as stream:
                for row in self.recovery_rows:
                    stream.write(json.dumps({key: row[key] for key in ("step", "instance_ids", "generated_tokens")}) + "\n")
                stream.flush(); os.fsync(stream.fileno())
            for row in self.recovery_rows:
                v2._append_durable(directory / "recovery_rollout_observations.jsonl", row)

    def _load_from_checkpoint(self, checkpoint, model=None):
        result = self._collective_check(lambda: super(RecoveryStateMixin, self)._load_from_checkpoint(
            checkpoint, model=model), "native actor checkpoint restoration")
        self._checkpointing_admit("after_checkpoint_load")
        expected = self.resume_admission["rank_closure"][_rank()]["actor_state_sha256"]
        self._collective_check(lambda: need(semantic_state_sha256(
            self.accelerator.unwrap_model(self.model).state_dict()) == expected,
            "Resumed whole actor differs from original checkpoint actor"), "restored whole actor")
        return result

    def _load_rng_state(self, checkpoint):
        result = self._collective_check(lambda: super(RecoveryStateMixin, self)._load_rng_state(checkpoint),
                                        "native RNG restoration")
        expected = self.resume_admission["rank_closure"][_rank()]["rng_state_sha256"]
        observed = rng_sha256()
        self._collective_check(lambda: need(observed == expected, "Native restored RNG differs"), "restored RNG bytes")
        self.resume_rng_audit = {"expected": expected, "observed": observed, "bitwise_equal": True}
        return result

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Native guarded load, followed by collective rank-owned restoration.

        Pinned Accelerate labels a MULTI_CPU device ``cpu:0``. Transformers
        passes it to torch.load in DDP, but torch serialization has no indexed
        CPU storage backend. Normalize only that equivalent CPU destination;
        CUDA map locations and the native Torch>=2.6 safety guard stay intact.
        Calling the native loader directly lets a local deserialization failure
        join peers before the separate rank-restoration collective begins.
        """
        self._collective_check(self._assert_contract, "resume active contract")
        step = (self._collective_check(lambda: self._validate_checkpoint(checkpoint), "resume native recipe")
                if checkpoint else None)
        if self.args.device.type == "cpu" and self.args.device.index is not None:
            self.args.__dict__["_setup_devices"] = torch.device("cpu")
        result = self._collective_check(lambda: Trainer._load_optimizer_and_scheduler(self, checkpoint),
                                        "native guarded optimizer/scheduler loading")
        if checkpoint:
            self._collective_check(lambda: restore_rank_state(checkpoint, self.optimizer, self.lr_scheduler,
                rank=_rank(), world_size=_world(), step=step), "rank-owned optimizer/scheduler restoration")
            self._resuming = self._reference_prepared_checked = True
        return result

    def _generate_and_score_completions(self, inputs):
        before_rng = rng_sha256()
        output = super()._generate_and_score_completions(inputs)
        ids, mask = output["completion_ids"].detach().cpu().tolist(), output["completion_mask"].detach().cpu().tolist()
        local = [digest([tokens, flags]) for tokens, flags in zip(ids, mask)]
        fingerprints = [value for rank_values in _gather(local) for value in rank_values]
        row = {"step": int(self.state.global_step), "instance_ids": list(self.actual_windows[-1]),
               "generated_tokens": self.actual_generated_tokens - sum(item["generated_tokens"] for item in self.recovery_rows),
               "completion_fingerprints": fingerprints, "rank_rng_before_generation": _gather(before_rng)}
        need(len(self.recovery_rows) == row["step"], "Recovery rollout prefix is not sequential")
        self.recovery_rows.append(row)
        if self.accelerator.is_main_process:
            v2._append_durable(Path(self.args.output_dir) / "recovery_rollout_observations.jsonl", row)
        return output

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        step = int(self.state.global_step)
        directory = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{step}"
        self._collective_check(lambda: _validate_prefix(self.recovery_rows, self.pilot_contract, step), "checkpoint rollout-prefix closure")

        def rank_proof():
            self._assert_reference_integrity(full=True)
            path = _rng_path(directory, _rank(), _world())
            saved = torch.load(path, map_location="cpu", weights_only=False)
            need(rng_sha256(saved) == rng_sha256(), "Saved native RNG differs from live checkpoint RNG")
            return {"rank": _rank(), "status": "passed", "global_step": step, "optimizer_steps": step,
                    "actor_state_sha256": semantic_state_sha256(self.accelerator.unwrap_model(self.model).state_dict()),
                    "reference_state_sha256": self.reference_state_sha256(),
                    "optimizer_state_sha256": semantic_state_sha256(self.optimizer.state_dict()),
                    "scheduler_state_sha256": semantic_state_sha256(self.lr_scheduler.state_dict()),
                    "rng_state_sha256": rng_sha256(saved), "rng_file_sha256": file_identity(path)["sha256"],
                    "reference_integrity_verified": bool(self.beta),
                    "reference_full_integrity_verified": True if self.beta else None,
                    "effective_checkpointing": self.last_checkpointing_audit,
                    "checkpointing_stage_counts": dict(self.checkpointing_stage_counts),
                    "checkpointing_total_rollouts_observed": self.recovery_start_step + self.checkpointing_stage_counts.get("after_generation", 0),
                    "prepared_reference_audit_original_segment": (self.prepared_reference_audit or
                        (self.resume_admission["reference"]["prepared_equivalence"] if self.resume_admission else None))}
        proofs = _gather(self._collective_check(rank_proof, "periodic rank proof"))
        self._consensus([row["actor_state_sha256"] for row in proofs], "periodic actor closure")
        need(all(row["actor_state_sha256"] == proofs[0]["actor_state_sha256"] for row in proofs), "Periodic actors differ")

        def seal():
            if not self.accelerator.is_main_process:
                return None
            elapsed = time.monotonic() - self.recovery_started
            prefix = {"schema_version": VERSION, "pilot_contract_sha256": self.pilot_contract["contract_sha256"],
                      "global_step": step, "generated_tokens": self.actual_generated_tokens, "rows": self.recovery_rows}
            write_json_atomic(directory / CHECKPOINT_PREFIX, prefix)
            payload = {"schema_version": VERSION, "protocol": PROTOCOL, "operational_version": VERSION,
                       "status": "passed", "source": file_identity(Path(__file__)),
                       "pilot_contract_sha256": self.pilot_contract["contract_sha256"],
                       "arm": self.recipe_contract["arm"], "scope": self.pilot_contract["scope"],
                       "global_step": step, "world_size": _world(), "rank_closure": proofs,
                       "reference": {"beta": self.beta, "state_sha256": self.reference_state_sha256(),
                                     "initial_equivalence": self.initial_reference_audit,
                                     "prepared_equivalence": proofs[0]["prepared_reference_audit_original_segment"]},
                       "lineage": self.recovery_lineage,
                       "rollout_prefix_sha256": file_identity(directory / CHECKPOINT_PREFIX)["sha256"],
                       "accounting": {"accepted_optimizer_steps": step, "accepted_responses": step * 36,
                                      "accepted_generated_tokens": self.actual_generated_tokens,
                                      "segment_start_step": self.recovery_start_step,
                                      "segment_training_seconds": elapsed,
                                      "accepted_training_seconds": self.recovery_parent_training_seconds + elapsed,
                                      "accepted_training_clock_complete": False,
                                      "clock_scope": "elapsed through pre-seal; current seal I/O and trainer epilogue not yet observed"},
                       "discarded_work": self.recovery_discarded, "artifact_tree": _checkpoint_tree(directory)}
            write_json_atomic(directory / CHECKPOINT_SEAL, payload)
            return payload
        self._collective_check(seal, "atomic recovery checkpoint seal")


class RecoveryRLOOTrainer(RecoveryStateMixin, v2.ObservedRLOOV2):
    pass


class RecoveryFOARLTrainer(RecoveryStateMixin, v2.ObservedFOARLV2):
    pass


def make_recovery_trainer(contract, arm, output, tokenizer, base, sft, device, *, checkpoint_interval=25,
                          resume_from=None, stop_after=None, production_fixture=False):
    """Rebuild original SFT actor/reference before native resumed actor loading."""
    contract = validate_contract(contract)
    need(arm in ARMS and arm != "sft", "Periodic recovery requires an RL arm")
    need(type(checkpoint_interval) is int and checkpoint_interval > 0, "Positive checkpoint interval required")
    if contract["scope"] == "production":
        need(checkpoint_interval == 25 and stop_after is None, "Production fixes periodic25 and terminal100 selection")
    if stop_after is not None:
        need(type(stop_after) is int and 0 < stop_after < contract["training"]["horizon"] and
             stop_after % checkpoint_interval == 0, "Fixture interruption must use a checkpoint boundary")
    output = Path(output)
    v2._collective_phase(lambda: output.mkdir(parents=True, exist_ok=False) if _rank() == 0 else None, "new recovery segment output")
    set_seed(contract["train_seed"])
    os.environ.update(reward_environment("foarl" if arm == "foarl" else "rloo_quality"))
    dtype = torch.float32 if production_fixture else torch.bfloat16
    actor = v2.loaded_model(base, sft, device, trainable=True, dtype=dtype)
    reference = v2.loaded_model(base, sft, device, trainable=False, dtype=dtype) if ARMS[arm]["beta"] else None
    identity = ({"source_kind": "original_sft", "source_sha256": contract["initializer"]["sha256"],
                 "immutable": True, "scope": contract["scope"]} if reference is not None else None)
    need(file_identity(Path(sft) / "adapter_model.safetensors")["sha256"] == contract["initializer"]["sha256"],
         "Original SFT adapter byte identity differs")
    args = v2._training_args(contract, arm, output, device, production_peft=True)
    args.save_steps = checkpoint_interval
    by_id = {row["instance_id"]: row for row in contract["training_rows"]}
    rows = [{**by_id[identifier], "answer": by_id[identifier]["output"],
             "instance_json": json.dumps(by_id[identifier]["instance_json"])}
            for draw in contract["training"]["prompt_draws"] for identifier in draw]
    common = dict(model=actor, args=args, processing_class=tokenizer, train_dataset=Dataset.from_list(rows),
                  reference_model=reference, reference_identity=identity, pilot_contract=contract,
                  recovery_checkpoint=resume_from, recovery_stop_after=stop_after,
                  recipe_contract={"pilot_sha256": contract["contract_sha256"], "arm": arm,
                                   "operational_version": VERSION, "entrypoint_sha256": file_identity(Path(__file__))["sha256"],
                                   "inherited_v2_sha256": file_identity(Path(v2.__file__))["sha256"],
                                   "checkpoint_interval": checkpoint_interval, "terminal_selection": contract["training"]["horizon"]})
    if arm == "foarl":
        return RecoveryFOARLTrainer(reward_funcs=list(FOARL_F638_REWARD_FUNCS), **common)
    reward = REWARD_HEADS[contract["task"], "reference_sigmoid_v2"]
    return RecoveryRLOOTrainer(reward_funcs=[reward],
        advantage_contract=make_advantage_contract("cardinal" if ARMS[arm]["quality_credit_lambda"] else "control"),
        reward_contract={"profile_id": PROFILE_ID, "task": contract["task"], "mode": "reference_sigmoid_v2",
                         "quality_slope": 8.0, "reward_head": reward.__name__,
                         "source_manifest_sha256": file_identity(Path(__file__))["sha256"]}, **common)


def run_training_segment(contract, arm, output, tokenizer, base, sft, device, **kwargs):
    """Return a live trainer for explicit caller cleanup and state validation."""
    trainer = make_recovery_trainer(contract, arm, output, tokenizer, base, sft, device, **kwargs)
    trainer.recovery_started = time.monotonic()
    trainer.train(resume_from_checkpoint=str(trainer.recovery_checkpoint) if trainer.recovery_checkpoint else None)
    train_call_seconds = time.monotonic() - trainer.recovery_started
    step = trainer.state.global_step
    need(len(trainer.actual_windows) == step and len(trainer.recovery_rows) == step, "Segment optimizer/rollout closure differs")
    directory = Path(trainer.args.output_dir) / f"checkpoint-{step}"
    seal, _ = v2._collective_phase(lambda: read_sealed_checkpoint(directory, contract, arm), "completed segment checkpoint")
    terminal = step == contract["training"]["horizon"]
    accounting = {**seal["accounting"], "segment_training_seconds": train_call_seconds,
                  "accepted_training_seconds": trainer.recovery_parent_training_seconds + train_call_seconds,
                  "accepted_training_clock_complete": trainer.recovery_parent_clock_complete,
                  "clock_scope": "sum of measured completed native train calls; unavailable prior epilogues remain an explicit lower bound"}
    proof = None
    if terminal:
        def final_integrity():
            trainer._assert_reference_integrity(full=True)
            expected = seal["rank_closure"][_rank()]
            need(semantic_state_sha256(trainer.accelerator.unwrap_model(trainer.model).state_dict()) == expected["actor_state_sha256"] and
                 semantic_state_sha256(trainer.optimizer.state_dict()) == expected["optimizer_state_sha256"] and
                 semantic_state_sha256(trainer.lr_scheduler.state_dict()) == expected["scheduler_state_sha256"] and
                 rng_sha256() == expected["rng_state_sha256"], "Terminal live states changed after final checkpoint seal")
        v2._collective_phase(final_integrity, "terminal live state integrity")
        def persist():
            if _rank() != 0:
                return None
            model = trainer.accelerator.unwrap_model(trainer.model)
            terminal_dir = Path(output) / "terminal"
            model.save_pretrained(terminal_dir); tokenizer.save_pretrained(terminal_dir)
            payload = {"schema_version": VERSION, "protocol": "sft-anchor-training-complete-v3", "status": "passed",
                       "operational_version": VERSION, "scope": contract["scope"], "task": contract["task"], "arm": arm,
                       "pilot_contract_sha256": contract["contract_sha256"], "source": file_identity(Path(__file__)),
                       "checkpoint": {"initializer_sha256": contract["initializer_sha256"], "train_seed": contract["train_seed"],
                                      "step": step, "state_sha256": seal["rank_closure"][0]["actor_state_sha256"],
                                      "reference_initializer_sha256": contract["initializer_sha256"] if trainer.beta else None},
                       "training": {"optimizer_steps": step, "responses": step * 36,
                                    "generated_tokens": trainer.actual_generated_tokens,
                                    "wall_seconds": accounting["accepted_training_seconds"],
                                    "wall_seconds_complete": accounting["accepted_training_clock_complete"],
                                    "prompt_draw_sha256": contract["training"]["prompt_draw_sha256"],
                                    "rollout_windows_count": step,
                                    "actual_prompt_windows_sha256": file_identity(Path(trainer.args.output_dir) / "actual_prompt_windows.jsonl")["sha256"]},
                       "reference": {**seal["reference"], "full_integrity_verified": True if trainer.beta else None},
                       "distributed": {"world_size": _world(), "rank_closure": seal["rank_closure"]},
                       "terminal_artifact_tree": v2.artifact_tree(terminal_dir), "lineage": seal["lineage"],
                       "accounting": accounting, "discarded_work": seal["discarded_work"],
                       "checkpoint_seal_sha256": file_identity(directory / CHECKPOINT_SEAL)["sha256"]}
            write_json_atomic(Path(output) / "training_complete.json", payload)
            return payload
        proof = v2._collective_phase(persist, "durable terminal100 proof")
    report = {"status": "terminal_training_complete" if terminal else "fixture_interrupted_at_sealed_checkpoint",
              "global_step": step, "checkpoint": str(directory), "checkpoint_seal_sha256": file_identity(directory / CHECKPOINT_SEAL)["sha256"],
              "whole_actor_state_sha256": seal["rank_closure"][0]["actor_state_sha256"],
              "fixed_terminal_selection": contract["training"]["horizon"], "lineage": seal["lineage"],
              "accounting": accounting, "discarded_work": seal["discarded_work"], "training_complete": proof}
    if _rank() == 0:
        write_json_atomic(Path(output) / "segment_complete.json", report)
    return trainer, report


def close_training_group():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _mode_seed(contract, row, mode):
    return int.from_bytes(hashlib.sha256(f'{contract["eval_seed"]}/{row["semantic_id"]}/{mode}'.encode()).digest()[:4], "big")


def _validate_group(group, contract, tokenizer):
    rows = {row["instance_id"]: row for row in contract["validation_rows"]}
    identifier, mode = group.get("instance_id"), group.get("decode")
    need(identifier in rows and mode in ("greedy", "k8"), "Recovery group cohort/decoder differs")
    row = rows[identifier]; seed = _mode_seed(contract, row, mode); count = 1 if mode == "greedy" else 8
    samples = group.get("raw_completions")
    need(group.get("schema_version") == 2 and group.get("semantic_id") == row["semantic_id"] and
         group.get("seed") == seed and type(samples) is list and len(samples) == count and
         type(group.get("wall_seconds")) in (int, float) and math.isfinite(group["wall_seconds"]) and group["wall_seconds"] >= 0,
         "Recovery group seed/count/time differs")
    for index, item in enumerate(samples):
        ids = item.get("completion_token_ids")
        need(type(ids) is list and all(type(value) is int and 0 <= value < len(tokenizer) for value in ids) and
             len(ids) <= contract["evaluation"]["max_completion_tokens"] and
             item.get("instance_id") == identifier and item.get("decode") == mode and
             item.get("sample_index") == index and item.get("generation_seed") == seed and
             item.get("completion_id") == f"{identifier}/{mode}/{index}" and item.get("runtime_complete") is True and
             item.get("generated_tokens") == len(ids) and item.get("text") == tokenizer.decode(ids, skip_special_tokens=True),
             "Recovery raw generation text/tokens/identity differs")
        need(tokenizer.eos_token_id not in ids[:-1] and item.get("finish_reason") ==
             ("eos" if ids and ids[-1] == tokenizer.eos_token_id else "length"), "Recovery stop semantics differ")
    need(group.get("generated_tokens") == sum(item["generated_tokens"] for item in samples), "Recovery group token total differs")
    return identifier, mode


def _generate_group(model, tokenizer, contract, row, mode, device):
    """Exact v2 decoder/seed for one missing whole modegroup."""
    encoded = tokenizer(row["prompt"], return_tensors="pt", truncation=False).to(device)
    need(encoded.input_ids.shape[1] + contract["evaluation"]["max_completion_tokens"] <= model.config.max_position_embeddings,
         "Recovery input exceeds context; truncation forbidden")
    seed = _mode_seed(contract, row, mode); set_seed(seed)
    kwargs = {"max_new_tokens": contract["evaluation"]["max_completion_tokens"], "do_sample": mode == "k8",
              "num_return_sequences": 1 if mode == "greedy" else 8, "pad_token_id": tokenizer.pad_token_id,
              "eos_token_id": tokenizer.eos_token_id}
    if mode == "k8":
        kwargs.update(temperature=.7, top_p=.9, top_k=50)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and contract["scope"] != "system_smoke" else nullcontext()
    started = time.monotonic()
    with torch.inference_mode(), autocast:
        generated = model.generate(**encoded, **kwargs)
    samples = []
    for index, values in enumerate(generated[:, encoded.input_ids.shape[1]:].cpu().tolist()):
        if tokenizer.eos_token_id in values:
            values = values[:values.index(tokenizer.eos_token_id) + 1]
        samples.append({"instance_id": row["instance_id"], "decode": mode, "sample_index": index,
                        "completion_id": f'{row["instance_id"]}/{mode}/{index}', "completion_token_ids": values,
                        "text": tokenizer.decode(values, skip_special_tokens=True), "generated_tokens": len(values),
                        "generation_seed": seed, "runtime_complete": True,
                        "finish_reason": "eos" if values and values[-1] == tokenizer.eos_token_id else "length"})
    return {"schema_version": 2, "instance_id": row["instance_id"], "semantic_id": row["semantic_id"],
            "decode": mode, "seed": seed, "raw_completions": samples,
            "generated_tokens": sum(row["generated_tokens"] for row in samples), "wall_seconds": time.monotonic() - started}


def allocation_accounting(records, *, authorized_requested_gpu_hours=92):
    """Account submitted/failed/rescue allocations; this grants no job authority."""
    need(type(records) is list, "Allocation accounting must be a list")
    identities = set(); requested = allocated = 0.0
    for row in records:
        need(type(row) is dict and type(row.get("id")) is str and row["id"] not in identities and
             type(row.get("gpus")) is int and row["gpus"] >= 0 and
             all(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] >= 0
                 for key in ("requested_wall_seconds", "elapsed_seconds")), "Invalid allocation resource record")
        identities.add(row["id"]); requested += row["gpus"] * row["requested_wall_seconds"]
        allocated += row["gpus"] * row["elapsed_seconds"]
    need(requested / 3600 <= authorized_requested_gpu_hours, "Recovery accounting exceeds declared requested GPU-hour authority")
    return {"records": records, "total_requested_gpu_hours": requested / 3600,
            "total_allocated_gpu_seconds": allocated, "allocation_accounting_complete": False,
            "accounting_certificate_verified": False, "scope": "caller supplied arithmetic; not scheduler verified",
            "authorized_requested_gpu_hours": authorized_requested_gpu_hours, "new_job_authorized": False}


def admit_original_training_proof(contract, arm, training_output, *, expected_sha256=None):
    """Admit complete producer proof, not only an actor filename or step label."""
    training_output = Path(training_output)
    path = training_output / "training_complete.json"
    need(not path.is_symlink(), "Linked original training proof forbidden")
    if expected_sha256 is not None:
        need(file_identity(path)["sha256"] == expected_sha256, "Original durable training-proof byte binding differs")
    proof = json.loads(path.read_text())
    version = proof.get("schema_version")
    steps = 0 if arm == "sft" else contract["training"]["horizon"]
    checkpoint, counts = proof.get("checkpoint", {}), proof.get("training", {})
    source_path = Path(v2.__file__) if version == 2 else Path(__file__)
    need(version in (2, 3) and proof.get("status") == "passed" and
         proof.get("protocol") == f"sft-anchor-training-complete-v{version}" and
         proof.get("operational_version") == version and proof.get("source", {}).get("sha256") == file_identity(source_path)["sha256"] and
         proof.get("source", {}).get("size_bytes") == file_identity(source_path)["size_bytes"] and
         proof.get("pilot_contract_sha256") == contract["contract_sha256"] and proof.get("task") == contract["task"] and
         proof.get("arm") == arm and proof.get("scope") == contract["scope"] and checkpoint.get("step") == steps and
         checkpoint.get("initializer_sha256") == contract["initializer_sha256"] and checkpoint.get("train_seed") == contract["train_seed"] and
         counts.get("optimizer_steps") == steps and counts.get("responses") == steps * 36 and
         counts.get("rollout_windows_count") == steps and counts.get("prompt_draw_sha256") == contract["training"]["prompt_draw_sha256"] and
         type(counts.get("generated_tokens")) is int and counts["generated_tokens"] >= 0 and
         type(counts.get("wall_seconds")) in (int, float) and math.isfinite(counts["wall_seconds"]) and counts["wall_seconds"] >= 0,
         "Original durable terminal training proof required")
    world = proof.get("distributed", {}).get("world_size")
    ranks = proof.get("distributed", {}).get("rank_closure", [])
    need(type(world) is int and world in (1, 2, 4) and len(ranks) == world and
         all(row.get("rank") == index and row.get("status") == "passed" and row.get("optimizer_steps") == steps and
             row.get("actor_state_sha256") == checkpoint.get("state_sha256") for index, row in enumerate(ranks)),
         "Original distributed actor/update closure differs")
    if contract["scope"] == "production":
        need(world == (1 if arm == "sft" else 4) and contract["training"]["horizon"] == 100 and
             len(contract["training_rows"]) == 128 and len(contract["validation_rows"]) == 96 and
             contract["training"]["max_completion_tokens"] == 3000 and expected_sha256 is not None,
             "Production rescue needs fixed recipe/world and externally bound original proof")
    if steps:
        windows_path = training_output / "training/actual_prompt_windows.jsonl"
        need(file_identity(windows_path)["sha256"] == counts.get("actual_prompt_windows_sha256"),
             "Original prompt-window bytes changed")
        windows, tail = _complete_lines(windows_path)
        need(tail is None and len(windows) == steps and all(
            row.get("step") == index and row.get("instance_ids") == [identifier for identifier in
                contract["training"]["prompt_draws"][index] for _ in range(9)] and
            type(row.get("generated_tokens")) is int and row["generated_tokens"] >= 0
            for index, row in enumerate(windows)) and sum(row["generated_tokens"] for row in windows) == counts["generated_tokens"],
            "Original complete prompt/token schedule differs")
        for row in ranks:
            gc_audit = row.get("effective_checkpointing", {})
            counters = row.get("checkpointing_stage_counts", {})
            total_rollouts = row.get("checkpointing_total_rollouts_observed", counters.get("after_generation", 0))
            need(gc_audit.get("effective_use_reentrant") is False and gc_audit.get("frozen_embedding_weights") is True and
                 counters.get("construction") == 1 and total_rollouts == steps and
                 counters.get("before_actor_loss", 0) >= counters.get("after_generation", 0) * (36 // world),
                 "Original effective non-reentrant checkpointing admission differs")
    beta = ARMS[arm]["beta"]; reference = proof.get("reference", {})
    need(reference.get("beta") == beta, "Original frozen-reference beta differs")
    if beta:
        need(checkpoint.get("reference_initializer_sha256") == contract["initializer_sha256"] and
             reference.get("full_integrity_verified") is True and
             reference.get("initial_equivalence", {}).get("bitwise_equal") is True and
             reference.get("prepared_equivalence", {}).get("bitwise_equal") is True and
             all(row.get("reference_full_integrity_verified") is True and
                 row.get("reference_state_sha256") == reference.get("state_sha256") for row in ranks),
             "Original independent frozen SFT reference admission differs")
    else:
        need(reference.get("state_sha256") is None and checkpoint.get("reference_initializer_sha256") is None,
             "Beta-zero control acquired a reference")
    terminal = training_output / "terminal"
    need(v2.artifact_tree(terminal) == proof.get("terminal_artifact_tree"), "Original terminal model/config bytes changed")
    if version == 3:
        sealed = training_output / "training" / f"checkpoint-{steps}" / CHECKPOINT_SEAL
        need(file_identity(sealed)["sha256"] == proof.get("checkpoint_seal_sha256"), "Original terminal checkpoint seal changed")
    return proof


def rescue_evaluation(contract, arm, training_output, output, tokenizer, base, device, *,
                      production_fixture=False, progress_paths=(), allocation_records=(), authorized_requested_gpu_hours=92,
                      expected_training_complete_sha256=None, accounting_certificate=None,
                      accounting_certificate_sha256=None):
    """Reuse only closed seeded groups after whole-actor/tree verification."""
    need(not torch.distributed.is_initialized(), "Close all training ranks before evaluation-only rescue")
    contract = validate_contract(contract); training_output, output = Path(training_output), Path(output)
    proof_path = training_output / "training_complete.json"
    proof = admit_original_training_proof(contract, arm, training_output, expected_sha256=expected_training_complete_sha256)
    expected_step = 0 if arm == "sft" else contract["training"]["horizon"]
    terminal = training_output / "terminal"
    if contract["scope"] == "production":
        need(not production_fixture and accounting_certificate is not None and accounting_certificate_sha256 is not None,
             "Production rescue requires an externally bound scheduler accounting certificate")
    accounting = allocation_accounting(list(allocation_records), authorized_requested_gpu_hours=authorized_requested_gpu_hours)
    if accounting_certificate is not None:
        from clasp.baselines.sft_anchor_recovery_accounting import validate_accounting_certificate
        accounting = validate_accounting_certificate(accounting_certificate, expected_sha256=accounting_certificate_sha256,
            training_complete_sha256=file_identity(proof_path)["sha256"], pilot_contract_sha256=contract["contract_sha256"],
            source_sha256=proof["source"]["sha256"], task=contract["task"], arm=arm,
            current_job_id=os.environ.get("SLURM_JOB_ID"), authorized_requested_gpu_hours=authorized_requested_gpu_hours)
    model = v2.loaded_model(base, terminal, device, trainable=False,
                            dtype=torch.float32 if production_fixture else torch.bfloat16)
    expected = proof["checkpoint"]["state_sha256"]; observed = semantic_state_sha256(model.state_dict())
    need(observed == expected, "Rescue reloaded actor differs from original in-memory final actor")
    groups, bindings, tails = {}, [], []
    paths = list(progress_paths) or [training_output / "evaluation_groups.jsonl"]
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        records, tail = _complete_lines(path)
        bindings.append(file_identity(path))
        if tail:
            tails.append({"path": str(path.resolve()), **tail})
        for group in records:
            key = _validate_group(group, contract, tokenizer)
            need(key not in groups, "Duplicate/conflicting completed evaluation group")
            groups[key] = group
    output.mkdir(parents=True, exist_ok=False)
    reload_proof = {"schema_version": VERSION, "protocol": "sft-anchor-checkpoint-reload-rescue-v3", "status": "passed",
                    "source": file_identity(Path(__file__)), "training_complete_sha256": file_identity(proof_path)["sha256"],
                    "expected_actor_state_sha256": expected, "reloaded_actor_state_sha256": observed, "bitwise_state_equal": True,
                    "checkpoint_path": str(terminal.resolve()), "distributed_group_closed_before_evaluation": True}
    write_json_atomic(output / "checkpoint_reload_exact.json", reload_proof)
    expected_groups = [(row, mode) for row in contract["validation_rows"] for mode in ("greedy", "k8")]
    reused = list(groups); regenerated = []; raw = []; old_seconds = sum(group["wall_seconds"] for group in groups.values())
    started = time.monotonic(); new_seconds = 0.0
    model.eval()
    for row, mode in expected_groups:
        key = row["instance_id"], mode
        if key not in groups:
            groups[key] = _generate_group(model, tokenizer, contract, row, mode, device)
            _validate_group(groups[key], contract, tokenizer)
            regenerated.append(key); new_seconds += groups[key]["wall_seconds"]
        v2._append_durable(output / "evaluation_groups.jsonl", groups[key])
        raw.extend(groups[key]["raw_completions"])
    recovery = {"schema_version": VERSION, "protocol": "sft-anchor-evaluation-rescue-v3", "source": file_identity(Path(__file__)),
                "training_complete": file_identity(proof_path), "terminal_checkpoint_selection": expected_step,
                "prior_progress_bindings": bindings, "discarded_partial_tails": tails,
                "reused_modegroups": reused, "regenerated_modegroups": regenerated,
                "accepted_evaluation_responses": len(raw), "reused_evaluation_responses": sum(len(groups[key]["raw_completions"]) for key in reused),
                "new_evaluation_responses": sum(len(groups[key]["raw_completions"]) for key in regenerated),
                "accepted_prior_group_seconds": old_seconds, "new_group_seconds": new_seconds,
                "rescue_evaluation_wall_seconds": time.monotonic() - started,
                "wasted_inflight_generation_seconds": None, "unmeasured_tail_time_not_fabricated": True,
                "training_discarded_work": proof.get("discarded_work", {}), "allocation_accounting": accounting,
                "prior_scheduler_failure_preserved": True, "no_model_quality_claim": True}
    write_json_atomic(output / "evaluation_recovery.json", recovery)
    resources = {"training_optimizer_steps": proof["training"]["optimizer_steps"], "training_responses": proof["training"]["responses"],
                 "training_generated_tokens": proof["training"]["generated_tokens"], "training_wall_seconds": proof["training"]["wall_seconds"],
                 "evaluation_wall_seconds": old_seconds + new_seconds, "prompt_draw_sha256": contract["training"]["prompt_draw_sha256"],
                 "checkpoint_reload_exact": True, "operational_version": VERSION, "production_fixture": production_fixture,
                 "training_complete_sha256": file_identity(proof_path)["sha256"],
                 "checkpoint_reload_exact_sha256": file_identity(output / "checkpoint_reload_exact.json")["sha256"],
                 "evaluation_groups_sha256": file_identity(output / "evaluation_groups.jsonl")["sha256"],
                 "evaluation_recovery_sha256": file_identity(output / "evaluation_recovery.json")["sha256"],
                 "evaluation_groups_closed": True, "distributed_group_closed_before_evaluation": True,
                 "original_training_output": str(training_output.resolve())}
    result = evaluate_arm(contract, arm, raw, proof["checkpoint"], resources)
    write_json_atomic(output / "result.json", result)
    del model; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-fixture", action="store_true")
    parser.add_argument("--fixture-horizon", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--evaluate-only", type=Path)
    parser.add_argument("--task", choices=("op", "cvrp"), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--base", type=Path); parser.add_argument("--sft", type=Path)
    parser.add_argument("--contract", type=Path); parser.add_argument("--contract-sha256")
    parser.add_argument("--progress", action="append", type=Path, default=[])
    parser.add_argument("--allocation-records", type=Path)
    parser.add_argument("--training-complete-sha256")
    parser.add_argument("--accounting-certificate", type=Path)
    parser.add_argument("--accounting-certificate-sha256")
    args = parser.parse_args(); torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        need(not args.evaluate_only, "Evaluation-only is a single-process rescue after group closure")
        torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    if args.production_fixture and not args.contract:
        need(not args.resume_from and not args.evaluate_only, "Fixture resume/rescue requires its original contract/base/SFT")
        if _rank() == 0:
            args.output.mkdir(parents=True, exist_ok=False)
            args.base, args.sft = args.output / "fixture_base", args.output / "fixture_sft"
            model, tokenizer, train, validation = v2.tiny_peft_initializer(args.task, args.base, args.sft, device)
            del model; gc.collect()
            contract = make_contract(args.task, {"id": "v3-tiny-PEFT-recovery-fixture",
                "sha256": file_identity(args.sft / "adapter_model.safetensors")["sha256"]}, train, validation,
                train_seed=61, horizon=args.fixture_horizon, scope="system_smoke", max_completion_tokens=1)
            write_json_atomic(args.output / "contract.json", contract)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        args.base, args.sft, args.contract = args.output / "fixture_base", args.output / "fixture_sft", args.output / "contract.json"
    else:
        need(args.contract and args.base and (args.sft or args.evaluate_only), "Original contract/base/SFT required")
        if not args.production_fixture:
            need(device.type == "cuda" and args.contract_sha256 == file_identity(args.contract)["sha256"], "Production requires CUDA and exact contract bytes")
    contract = validate_contract(json.loads(args.contract.read_text()))
    need(contract["task"] == args.task, "Contract task differs")
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"; tokenizer.model_input_names = ["input_ids", "attention_mask"]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.evaluate_only:
        records = json.loads(args.allocation_records.read_text()) if args.allocation_records else []
        result, recovery = rescue_evaluation(contract, args.arm, args.evaluate_only, args.output, tokenizer, args.base, device,
            production_fixture=args.production_fixture, progress_paths=args.progress, allocation_records=records,
            expected_training_complete_sha256=args.training_complete_sha256,
            accounting_certificate=args.accounting_certificate,
            accounting_certificate_sha256=args.accounting_certificate_sha256)
        print(json.dumps({"status": "evaluation_rescue_complete", "responses": result["resources"]["evaluation_responses"],
                          "reused_groups": len(recovery["reused_modegroups"]), "new_groups": len(recovery["regenerated_modegroups"])}))
        return
    segment = args.output / "segment" if args.production_fixture and args.contract == args.output / "contract.json" else args.output
    trainer, report = run_training_segment(contract, args.arm, segment, tokenizer, args.base, args.sft, device,
        checkpoint_interval=args.checkpoint_interval, resume_from=args.resume_from, stop_after=args.stop_after,
        production_fixture=args.production_fixture)
    del trainer; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rank = _rank(); close_training_group()
    if rank == 0:
        print(json.dumps({key: report[key] for key in ("status", "global_step", "checkpoint", "fixed_terminal_selection")}))


if __name__ == "__main__":
    main()
