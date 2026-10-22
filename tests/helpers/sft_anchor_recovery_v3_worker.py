"""Actual two-rank PEFT trajectory/resume regression, Gloo or permitted CUDA."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from clasp.baselines.sft_anchor_pilot import make_contract, validate_contract
from clasp.entrypoints import train_sft_anchor_recovery_v3 as v3
from clasp.entrypoints import train_sft_anchor_pilot_v2 as v2
from clasp.models.external_baseline_state import semantic_state_sha256


def snapshot(trainer):
    trainer._assert_reference_integrity(full=True)
    return {"actor": semantic_state_sha256(trainer.accelerator.unwrap_model(trainer.model).state_dict()),
            "optimizer": semantic_state_sha256(trainer.optimizer.state_dict()),
            "scheduler": semantic_state_sha256(trainer.lr_scheduler.state_dict()),
            "rng": v3.rng_sha256(), "reference": trainer.reference_state_sha256(),
            "windows": copy.deepcopy(trainer.recovery_rows), "tokens": trainer.actual_generated_tokens,
            "step": trainer.state.global_step, "resume_rng_audit": trainer.resume_rng_audit,
            "checkpointing": trainer.last_checkpointing_audit}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="rloo_cardinal_beta005")
    parser.add_argument("--task", default="op")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=2)
    parser.add_argument("--interrupt-step", type=int, default=2)
    args = parser.parse_args(); torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    rank = torch.distributed.get_rank()
    base, sft = args.output / "base", args.output / "sft"
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        policy, tokenizer, train, validation = v2.tiny_peft_initializer(args.task, base, sft, device)
        del policy; gc.collect()
        contract = make_contract(args.task, {"id": "tiny-PEFT-DDP-resume-regression",
            "sha256": v2.file_identity(sft / "adapter_model.safetensors")["sha256"]}, train, validation,
            scope="system_smoke", horizon=args.horizon, max_completion_tokens=1, train_seed=61)
        v3.write_json_atomic(args.output / "contract.json", contract)
    torch.distributed.barrier()
    contract = validate_contract(json.loads((args.output / "contract.json").read_text()))
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokenizer.padding_side = "left"; tokenizer.model_input_names = ["input_ids", "attention_mask"]
    common = dict(contract=contract, arm=args.arm, tokenizer=tokenizer, base=base, sft=sft,
                  device=device, checkpoint_interval=args.checkpoint_interval, production_fixture=True)
    full, _ = v3.run_training_segment(output=args.output / "full", **common)
    expected = snapshot(full); del full; gc.collect()
    partial, partial_report = v3.run_training_segment(output=args.output / "partial", stop_after=args.interrupt_step, **common)
    first = snapshot(partial); del partial; gc.collect()
    resumed, report = v3.run_training_segment(output=args.output / "resumed", resume_from=partial_report["checkpoint"], **common)
    actual = snapshot(resumed)
    for key in ("actor", "optimizer", "scheduler", "rng", "reference", "windows", "tokens", "step"):
        assert actual[key] == expected[key], (rank, key, actual[key], expected[key])
    assert actual["step"] == args.horizon and actual["tokens"] == 36 * args.horizon
    assert actual["resume_rng_audit"]["bitwise_equal"]
    assert first["reference"] == actual["reference"]
    assert actual["checkpointing"]["effective_use_reentrant"] is False
    assert actual["checkpointing"]["frozen_embedding_weights"]
    del resumed; gc.collect()
    # A rank-local read failure must reach both peers before native training.
    original_reader = v3.read_sealed_checkpoint
    if rank == 1:
        def corrupt(*args, **kwargs):
            raise ValueError("simulated peer checkpoint corruption")
        v3.read_sealed_checkpoint = corrupt
    peer_fault = None
    try:
        v3.make_recovery_trainer(output=args.output / "peer_fault", resume_from=partial_report["checkpoint"], **common)
    except RuntimeError as exc:
        peer_fault = str(exc)
    finally:
        v3.read_sealed_checkpoint = original_reader
    assert peer_fault and "simulated peer checkpoint corruption" in peer_fault
    faults = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(faults, peer_fault)
    assert faults[0] == faults[1]
    receipt = {"rank": rank, "world_size": torch.distributed.get_world_size(), "status": "passed",
               "scope": "actual_tiny_PEFT_resume_not7B_performance", "full": expected, "resumed": actual,
               "interrupted_step": args.interrupt_step, "checkpoint_interval": args.checkpoint_interval,
               "peer_fault_rejected_collectively": True,
               "whole_actor_optimizer_scheduler_rng_prefix_equal": True,
               "source": v3.file_identity(Path(v3.__file__))}
    v3.write_json_atomic(args.output / f"rank{rank}_trajectory_receipt.json", receipt)
    v3.close_training_group()
    if rank == 0:
        print(json.dumps({"status": "passed_actual_two_rank_PEFT_resume", "device": args.device,
                          "task": args.task, "arm": args.arm, "source_sha256": receipt["source"]["sha256"]}))


if __name__ == "__main__":
    main()
