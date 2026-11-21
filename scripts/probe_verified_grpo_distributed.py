"""Small real Qwen/PEFT/DDP probe for the verified-GRPO training failure.

Run directly with the training Python; pytest and model downloads are unnecessary.
The probe compares ordinary GRPO, the actual separate-forward teacher helper, and
an equivalent teacher objective evaluated in one padded forward. It reports
gradient, optimizer-state, and parameter differences without repairing them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class _Accelerator:
    def __init__(self, world_size):
        self.num_processes = world_size
        self.state = SimpleNamespace(deepspeed_plugin=None)

    @staticmethod
    def unwrap_model(model):
        import torch

        return model.module if isinstance(
            model, torch.nn.parallel.DistributedDataParallel
        ) else model

    def gather(self, value):
        import torch
        import torch.distributed as dist

        result = [torch.empty_like(value) for _ in range(self.num_processes)]
        dist.all_gather(result, value)
        return torch.cat(result)


def _tensor_difference(local, rank_zero):
    """Compare raw storage values, including signed zero, as fidelity does."""
    import torch

    local_bytes = local.contiguous().reshape(-1).view(torch.uint8)
    reference_bytes = rank_zero.contiguous().reshape(-1).view(torch.uint8)
    different_bytes = local_bytes != reference_bytes
    if not bool(different_bytes.any().item()):
        return None
    delta = (local.double() - rank_zero.double()).abs()
    return {"dtype": str(local.dtype),
            "max_abs_difference": float(delta.max().item()),
            "different_elements": int((local != rank_zero).sum().item()),
            "different_bytes": int(different_bytes.sum().item()),
            "numerically_equal": bool(torch.equal(local, rank_zero))}


def _rank_difference(named_tensors, device):
    import torch
    import torch.distributed as dist

    different = []
    maximum = 0.0
    for name, value in named_tensors:
        if not torch.is_tensor(value):
            continue
        local = value.detach().to(device=device).contiguous()
        rank_zero = local.clone()
        dist.broadcast(rank_zero, src=0)
        difference = _tensor_difference(local, rank_zero)
        if difference is not None:
            maximum = max(maximum, difference["max_abs_difference"])
            different.append({"name": name, **difference})
    return {"different_tensor_count": len(different), "max_abs_difference": maximum,
            "first_different_tensors": different[:12]}


def _optimizer_tensors(optimizer, model):
    for name, parameter in model.named_parameters():
        for key, value in sorted(optimizer.state.get(parameter, {}).items()):
            yield name + "." + key, value


def _worker(rank, config, rendezvous, output_dir):
    import torch
    import torch.distributed as dist
    import transformers
    import trl
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from clasp.trainers.primal_dual_grpo import PrimalDualGRPOTrainer

    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if config["device"] == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                            init_method="file://" + rendezvous,
                            rank=rank, world_size=config["world_size"])
    report = {"rank": rank, "versions": {"torch": torch.__version__,
              "transformers": transformers.__version__, "trl": trl.__version__}, "modes": {}}
    try:
        for mode in config["modes"]:
            torch.manual_seed(3)
            base = Qwen2ForCausalLM(Qwen2Config(
                vocab_size=43, hidden_size=128, intermediate_size=256,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                attn_implementation="eager",
            )).to(device=device, dtype=getattr(torch, config["dtype"]))
            policy = get_peft_model(base, LoraConfig(
                r=32, lora_alpha=32, lora_dropout=0, bias="all", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                "up_proj", "down_proj"],
            ))
            policy.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            policy.enable_input_require_grads()
            policy.train()
            model = torch.nn.parallel.DistributedDataParallel(
                policy, device_ids=[rank] if device.type == "cuda" else None,
                find_unused_parameters=False, bucket_cap_mb=config["bucket_cap_mb"],
            )
            trainer = object.__new__(PrimalDualGRPOTrainer)
            trainer.model = policy
            trainer.args = SimpleNamespace(report_to=[], gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
            trainer.accelerator = _Accelerator(config["world_size"])
            trainer.model_kwarg_keys = ["logits_to_keep"]
            trainer.temperature = 0.7
            trainer.grpo_verified_improvement = True
            trainer.grpo_verified_improvement_weight = 0.05
            trainer.grpo_verified_improvement_max_tokens = 3000
            trainer.grpo_verified_improvement_max_passes = 2
            trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
            trainer.processing_class = SimpleNamespace(pad_token_id=0, eos_token_id=2)
            trainable = [p for p in model.parameters() if p.requires_grad]
            if config["optimizer"] == "adamw8bit":
                import bitsandbytes as bnb

                optimizer = bnb.optim.AdamW8bit(trainable, lr=1e-4, weight_decay=0.01)
                report["versions"]["bitsandbytes"] = bnb.__version__
            else:
                optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
            observations = []
            for step in range(config["steps"]):
                checkpointing = None
                if config["generation_lifecycle"] != "none":
                    from trl.models.utils import unwrap_model_for_generation

                    layer = policy.base_model.model.model.layers[0]
                    before = dict(layer._gradient_checkpointing_func.keywords)
                    # Invoke the actual installed TRL context; its restoration
                    # behavior is the regression trigger, with no model download
                    # or stochastic generation needed to exercise it.
                    with unwrap_model_for_generation(model, trainer.accelerator):
                        assert not policy.is_gradient_checkpointing
                    after_trl = dict(layer._gradient_checkpointing_func.keywords)
                    if config["generation_lifecycle"] == "restored":
                        trainer._restore_configured_gradient_checkpointing()
                    checkpointing = {"before": before, "after_trl": after_trl,
                                     "before_loss": dict(layer._gradient_checkpointing_func.keywords)}
                optimizer.zero_grad()
                counts = []
                for micro in range(3):
                    with model.no_sync() if micro < 2 else nullcontext():
                        prompt = (torch.arange(8, device=device).reshape(2, 4) + rank + micro) % 43
                        prompt_mask = torch.ones_like(prompt)
                        completion = (prompt[:, :3] + step + 3) % 43
                        completion_mask = torch.ones_like(completion)
                        target = (completion + 5) % 43
                        target_mask = completion_mask.clone()
                        # Includes unequal counts, a rank with no teachers, and
                        # a microbatch with no teachers anywhere in the group.
                        if (rank + micro + step) % 3 == 0:
                            target_mask.zero_()
                        elif (rank + micro + step) % 3 == 1:
                            target_mask[0].zero_()
                        if step % 4 == 2 and micro == 1:
                            target_mask.zero_()
                        valid = target_mask.sum(dim=1) > 0
                        global_count = trainer.accelerator.gather(valid.sum().reshape(1)).sum()
                        counts.append({"local": int(valid.sum().item()),
                                       "global": int(global_count.item())})
                        input_ids = torch.cat([prompt, completion], dim=1)
                        attention = torch.cat([prompt_mask, completion_mask], dim=1)
                        if mode == "fused_single_forward" and int(global_count.item()) > 0:
                            # A dummy teacher shares the forward but contributes
                            # zero when this rank has no eligible teacher.
                            chosen = valid if bool(valid.any().item()) else torch.tensor(
                                [True, False], device=device)
                            teacher_ids = torch.cat([prompt[chosen], target[chosen]], dim=1)
                            teacher_attention = torch.cat(
                                [prompt_mask[chosen], target_mask[chosen]], dim=1)
                            joint_logps, _ = trainer._get_per_token_logps_and_entropies(
                                model, torch.cat([input_ids, teacher_ids]),
                                torch.cat([attention, teacher_attention]), completion.size(1))
                            logps = joint_logps[:2]
                            teacher_logps = joint_logps[2:]
                            teacher_mask = target_mask[chosen]
                            nll = -(teacher_logps * teacher_mask).sum(dim=1) / (
                                teacher_mask.sum(dim=1).clamp(min=1))
                            teacher_loss = 0.05 * nll.sum() * config["world_size"] / global_count
                        else:
                            logps, _ = trainer._get_per_token_logps_and_entropies(
                                model, input_ids, attention, completion.size(1))
                            teacher_loss = logps.sum() * 0
                            if mode == "existing_double_forward":
                                teacher_loss, _ = trainer._compute_improvement_distillation_losses(
                                    model, {"grpo_verified_mis_completion_ids": target,
                                            "grpo_verified_mis_completion_mask": target_mask},
                                    prompt, prompt_mask, completion, logps)
                        advantages = torch.tensor([1.0, -0.8], device=device)
                        loss = -(logps.mean(dim=1) * advantages).mean() + teacher_loss
                        (loss / 3).backward()
                inspect_now = bool(config["inspect_every_step"])
                inspect_final = step == config["steps"] - 1
                gradients = None
                if inspect_now:
                    gradients = _rank_difference(
                        ((name, p.grad) for name, p in model.named_parameters() if p.requires_grad),
                        device,
                    )
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                parameters = states = None
                if inspect_now or inspect_final:
                    if not inspect_now:
                        # Final-only mode deliberately adds no pre-optimizer
                        # collective or host synchronization. These surviving
                        # gradients have already been clipped by the optimizer
                        # loop; their comparison cannot localize an earlier
                        # step's gradient divergence.
                        gradients = _rank_difference(
                            ((name, p.grad) for name, p in model.named_parameters()
                             if p.requires_grad), device,
                        )
                    parameters = _rank_difference(model.named_parameters(), device)
                    states = _rank_difference(_optimizer_tensors(optimizer, model), device)
                observation = {"step": step + 1, "gradients": gradients, "parameters": parameters,
                               "optimizer_states": states, "gradient_norm": norm.detach(),
                               "gradient_inspection_phase": (
                                   "before_clipping" if inspect_now else
                                   "after_final_optimizer_clipped" if inspect_final else "not_inspected"),
                               "teacher_counts": counts, "checkpointing": checkpointing}
                observations.append(observation)
                if rank == 0:
                    print(json.dumps({"mode": mode, "step": step + 1,
                          "local_rank": rank, "inspected": inspect_now or inspect_final}), flush=True)
            # Delaying scalar extraction matters in final-only mode: .item()
            # would otherwise synchronize CUDA after every optimizer update.
            for observation in observations:
                observation["gradient_norm"] = float(observation["gradient_norm"].item())
            report["modes"][mode] = observations
            report.setdefault("ddp_logging", {})[mode] = model._get_ddp_logging_data()
            del trainer, optimizer, model, policy, base
            if device.type == "cuda":
                torch.cuda.empty_cache()
            dist.barrier()
        Path(output_dir, f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


def run_probe(config, output):
    import torch.multiprocessing as mp

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verified-grpo-ddp-") as directory:
        mp.spawn(_worker, args=(config, str(Path(directory, "rendezvous")), directory),
                 nprocs=config["world_size"], join=True)
        ranks = [json.loads(Path(directory, f"rank{rank}.json").read_text())
                 for rank in range(config["world_size"])]
    consensus = all(
        entry[field]["different_tensor_count"] == 0
        for rank in ranks for mode in rank["modes"].values() for entry in mode
        for field in ("gradients", "parameters", "optimizer_states")
        if entry[field] is not None
    )
    report = {"schema_version": 2, "comparison": "raw_tensor_bytes_including_signed_zero",
              "config": config, "exact_rank_consensus": consensus,
              "ranks": ranks,
              "inspection_semantics": (
                  "gradients before clipping and parameters/state after every optimizer step"
                  if config["inspect_every_step"] else
                  "no probe synchronization around intermediate optimizer steps; final clipped "
                  "gradients and final parameters/state only; normal trainer collectives remain"
              ),
              "scope": "Tiny randomly initialized model diagnostic; no task-performance claim."}
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--optimizer", choices=("adamw", "adamw8bit"), default="adamw")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--inspect-every-step", type=int, choices=(0, 1), default=1)
    parser.add_argument("--bucket-cap-mb", type=float, default=25.0)
    parser.add_argument("--generation-lifecycle", choices=("none", "trl", "restored"), default="none")
    parser.add_argument("--modes", nargs="+", choices=("grpo", "existing_double_forward",
                         "fused_single_forward"),
                        default=["grpo", "existing_double_forward", "fused_single_forward"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.world_size < 2 or args.steps < 1:
        parser.error("world-size must be at least two and steps must be positive")
    if not 0 < args.bucket_cap_mb <= 1024:
        parser.error("bucket-cap-mb must be positive and at most 1024")
    if args.optimizer == "adamw8bit" and args.device != "cuda":
        parser.error("adamw8bit diagnostic requires CUDA")
    config = vars(args).copy()
    output = config.pop("output")
    report = run_probe(config, output)
    print(json.dumps({"output": str(output), "exact_rank_consensus": report["exact_rank_consensus"]}),
          flush=True)
    return 0 if report["exact_rank_consensus"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
