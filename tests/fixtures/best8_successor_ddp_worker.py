"""Local CPU/Gloo trainer-only regression; never a study launcher."""
from datetime import timedelta
import argparse
import gc
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from clasp.entrypoints import train_sft_anchor_best8_v4 as v4
from clasp.entrypoints import train_sft_anchor_recovery_v3 as v3
from clasp.models.external_baseline_state import semantic_state_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", choices=("rloo_strict_best8", "foarl_released"), required=True)
    parser.add_argument("--fault", action="store_true")
    args = parser.parse_args(); torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", timeout=timedelta(seconds=120))
    rank = torch.distributed.get_rank()
    contract = json.loads((args.fixture / "contract.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.fixture / "sft", local_files_only=True)
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.padding_side = "left"
    trainer = v4.make_successor_trainer(contract, args.recipe, args.output, tokenizer,
        args.fixture / "base", args.fixture / "sft")
    if args.fault:
        by_id = {row["instance_id"]: row for row in contract["training_rows"]}
        rows = [{**by_id[value], "answer": by_id[value]["output"], "instance_json": json.dumps(by_id[value]["instance_json"])}
                for value in contract["training"]["prompt_draws"][0] for _ in range(9)]
        own = rows[rank * 18:(rank + 1) * 18]
        if rank == 1:
            own[0]["task_type"] = "cvrp"
        native_called = []
        trainer._generate = lambda *args: native_called.append(True)
        try:
            trainer._generate_and_score_completions(own)
        except RuntimeError as error:
            assert "strict prompt identities" in str(error)
            assert not native_called
            report = {"rank": rank, "peer_fault_collectively_rejected_before_generation": True, "error": str(error)}
        else:
            raise AssertionError("Cross-rank task fault reached native generation")
    else:
        trainer.recovery_started = time.monotonic(); trainer.train()
        trainer._assert_reference_integrity(full=True)
        report = {"rank": rank, "steps": trainer.state.global_step, "RL_responses": 36 * trainer.state.global_step,
            "tokens": trainer.actual_generated_tokens, "actor": semantic_state_sha256(trainer.accelerator.unwrap_model(trainer.model).state_dict()),
            "reference": trainer.reference_state_sha256(), "optimizer": semantic_state_sha256(trainer.optimizer.state_dict()),
            "scheduler": semantic_state_sha256(trainer.lr_scheduler.state_dict()), "RNG": v3.rng_sha256(),
            "windows": trainer.actual_windows, "strict_prompt_keys": [row["prompt_key"] for row in trainer.last_coefficient_record["strict_metadata"]],
            "combined": trainer.last_coefficient_record["combined"], "effective_checkpointing": trainer.last_checkpointing_audit,
            "reference_frozen": all(not p.requires_grad and p.grad is None for p in trainer.ref_model.parameters()),
            "scope": "system_smoke_CPU_Gloo_trainer_only", "superiority_established": False}
    v4.write_json_atomic(args.output / f"rank{rank}_trainer_only.json", report)
    del trainer; gc.collect()
    v3.close_training_group()


if __name__ == "__main__":
    main()
