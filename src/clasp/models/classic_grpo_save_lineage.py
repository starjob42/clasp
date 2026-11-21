"""Capture current-step classic lineage immediately after a complete Trainer save.

These sidecars are provisional checkpoint evidence. A later consumer must also
require the owning training subprocess and scheduler job to complete successfully.
No loss, optimizer, RNG, saved tensor or historical validator is changed here.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch.distributed as dist
from transformers import TrainerCallback

from clasp.baselines.co_deterministic_dev import validate_strict_determinism
from clasp.models.classic_grpo_resume import seal_classic_grpo_checkpoint
from clasp.models.training_fidelity import (
    PublicationFidelityCheckpointCallback,
    install_publication_fidelity_callback,
)

WORLD_SIZE = 4


class ClassicSaveLineageCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.saved_steps = []

    def _local_admission(self, args, state, rank):
        validate_strict_determinism()
        step = state.global_step
        if type(step) is not int or step <= 0 or step > 400 or step in self.saved_steps:
            raise ValueError("Invalid or repeated classic save boundary")
        if bool(state.is_world_process_zero) != (rank == 0):
            raise ValueError("Trainer world-zero state differs from distributed rank")
        contract_path = Path(os.environ["CLASP_TRAINING_RUN_CONTRACT_PATH"])
        if (not contract_path.is_absolute() or contract_path.resolve() != contract_path
                or contract_path.is_symlink() or not contract_path.is_file()):
            raise ValueError("Original classic contract must be a regular absolute file")
        contract = json.loads(contract_path.read_text())
        expected_output = contract_path.parent / "checkpoints"
        if (Path(args.output_dir) != expected_output or expected_output.resolve() != expected_output
                or int(state.max_steps) != 400 or int(args.max_steps) != 400):
            raise ValueError("Classic callback checkpoint root or fixed horizon changed")
        save_steps = 5 if contract.get("diagnostic") is True else 100
        if (contract.get("method") != "classic_grpo" or contract.get("beta") != 0
                or contract.get("teacher_weight") != 0
                or contract.get("schedule", {}).get("save_steps") != save_steps
                or step % save_steps):
            raise ValueError("Classic callback method or save schedule changed")
        return {"rank": rank, "step": step, "contract": str(contract_path),
                "output": str(expected_output), "error": None}, contract

    def on_save(self, args, state, control, **kwargs):
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE:
            raise RuntimeError("Classic save lineage requires the declared four-rank process group")
        rank = dist.get_rank()
        # Every rank has already returned from Trainer's checkpoint/RNG writes
        # and the original fidelity callback before this synchronization.
        dist.barrier()
        cause = None
        try:
            local, contract = self._local_admission(args, state, rank)
        except Exception as exc:
            cause = exc
            local, contract = {"rank": rank, "error": type(exc).__name__}, None
        gathered = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, local)
        outcome = [None]
        if rank == 0:
            try:
                if any(item.get("error") is not None for item in gathered):
                    raise ValueError("At least one rank rejected the provisional save boundary")
                expected = [{**local, "rank": index} for index in range(WORLD_SIZE)]
                if gathered != expected:
                    raise ValueError("Ranks disagree on the saved boundary or contract location")
                checkpoint = Path(local["output"]) / f"checkpoint-{local['step']}"
                # At step5 the ORIGINAL policy report still ends at5. The
                # unmodified sealer binds its exact bytes to saved5 fidelity;
                # this is not a truncated/derived report produced after step10.
                seal_classic_grpo_checkpoint(
                    checkpoint, expected_step=local["step"], expected_contract=contract,
                    run_contract_path=Path(local["contract"]))
                outcome[0] = {"status": "sealed_provisional_save", "step": local["step"]}
            except Exception as exc:
                cause = exc
                outcome[0] = {"status": "failed_provisional_save", "exception_class": type(exc).__name__}
        dist.broadcast_object_list(outcome, src=0)
        # Rank0 file failure is communicated before any rank can start another
        # update. The same final barrier also completes successful publication.
        dist.barrier()
        if outcome[0].get("status") != "sealed_provisional_save":
            raise RuntimeError("Classic provisional save failed on the distributed group: "
                               + str(outcome[0].get("exception_class"))) from cause
        self.saved_steps.append(outcome[0]["step"])
        return control


@contextmanager
def install_classic_save_lineage(training_module, method):
    """Scope one installer addition to the new deterministic classic entry."""
    if method != "classic_grpo":
        yield
        return
    original = training_module.install_publication_fidelity_callback
    if original is not install_publication_fidelity_callback:
        raise ValueError("Fidelity callback installer is already replaced or from another source")
    installed = []

    def install(trainer):
        if installed:
            raise ValueError("Classic save lineage must be installed exactly once")
        result = original(trainer)
        callbacks = trainer.callback_handler.callbacks
        if (result is not True or not callbacks
                or type(callbacks[-1]) is not PublicationFidelityCheckpointCallback
                or callbacks[-1]._trainer is not trainer):
            raise ValueError("Classic save lineage must follow the original enabled fidelity callback")
        callback = ClassicSaveLineageCallback(trainer)
        trainer.add_callback(callback)
        installed.append(callback)
        return result

    training_module.install_publication_fidelity_callback = install
    try:
        yield
        if len(installed) != 1:
            raise ValueError("Classic training never installed its save-lineage callback")
    finally:
        changed = training_module.install_publication_fidelity_callback is not install
        training_module.install_publication_fidelity_callback = original
        if changed:
            raise ValueError("Scoped classic fidelity installer was unexpectedly replaced")
