"""Run the prospective SFT-anchor factorial, with a real tiny-model E2E mode.

System smokes exercise generation, both reward packages, optimization, frozen
reference scoring, checkpoint reload, typed evaluation and paired reporting.
They cannot establish a Qwen-7B performance result. Production consumes a
sealed, fresh train/validation contract and the original local SFT checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Qwen2Config,
    Qwen2ForCausalLM, TrainerCallback, set_seed,
)
from trl import GRPOConfig
from trl.trainer.grpo_trainer import RepeatSampler

from clasp.baselines.sft_anchor_pilot import (
    ARMS, compare_results, digest, evaluate_arm, make_contract, validate_contract,
)
from clasp.baselines.quality_signal_dev import PROFILE_ID, REWARD_HEADS
from clasp.baselines.budget_matched_dev import reward_environment
from clasp.baselines.llmcosolver_f638_training import FOARL_F638_REWARD_FUNCS
from clasp.models.external_baseline_state import semantic_state_sha256
from clasp.rewards.feasible_quality import make_advantage_contract
from clasp.trainers.sft_anchored_rloo import SFTAnchoredRLOOTrainer, SFTReferenceGRPOTrainer


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Preserve previous attempt: {path}")
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def file_identity(path):
    path = Path(path)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return {"path": str(path.resolve()), "sha256": sha.hexdigest(), "size_bytes": path.stat().st_size}


class FiniteUpdates(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        gradients = [p.grad for p in self.trainer.model.parameters() if p.grad is not None]
        finite = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
        if torch.distributed.is_initialized():
            flag = torch.tensor(int(finite), device=self.trainer.accelerator.device)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
            finite = bool(flag.item())
        if not finite:
            raise FloatingPointError("Nonfinite/missing gradients before optimizer; no update admitted")

    def on_step_end(self, args, state, control, **kwargs):
        finite = all(bool(torch.isfinite(p).all()) for p in self.trainer.model.parameters())
        if torch.distributed.is_initialized():
            flag = torch.tensor(int(finite), device=self.trainer.accelerator.device)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
            finite = bool(flag.item())
        if not finite:
            raise FloatingPointError("Nonfinite parameters after optimizer")


class OrderedObservationMixin:
    """One predeclared draw schedule, joined across ranks before scoring."""
    def __init__(self, *args, pilot_contract, **kwargs):
        self.pilot_contract = pilot_contract
        self.actual_windows = []
        self.actual_generated_tokens = 0
        super().__init__(*args, **kwargs)
        self.add_callback(FiniteUpdates(self))

    def _get_train_sampler(self, dataset=None):
        return RepeatSampler(
            data_source=self.train_dataset if dataset is None else dataset,
            mini_repeat_count=9, batch_size=4,
            repeat_count=self.args.steps_per_generation, shuffle=False, seed=self.args.seed,
        )

    def _generate_and_score_completions(self, inputs):
        local = [row["instance_id"] for row in inputs]
        if torch.distributed.is_initialized():
            parts = [None] * self.accelerator.num_processes
            torch.distributed.all_gather_object(parts, local)
            actual = [item for part in parts for item in part]
        else:
            actual = local
        step = int(self.state.global_step)
        expected = [item for item in self.pilot_contract["training"]["prompt_draws"][step]
                    for _ in range(9)]
        if actual != expected:
            raise RuntimeError("Actual rollout order differs from sealed common prompt schedule")
        output = super()._generate_and_score_completions(inputs)
        tokens = output["completion_mask"].sum().detach()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(tokens)
        self.actual_generated_tokens += int(tokens.item())
        self.actual_windows.append(actual)
        if self.accelerator.is_main_process:
            with (Path(self.args.output_dir) / "actual_prompt_windows.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, "instance_ids": actual,
                                         "generated_tokens": int(tokens.item())}) + "\n")
        return output

    def compute_loss(self, *args, **kwargs):
        loss = super().compute_loss(*args, **kwargs)
        finite = torch.isfinite(loss).all().detach().to(dtype=torch.int32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        if not bool(finite.item()):
            raise FloatingPointError("Nonfinite actor loss before backward")
        return loss


class ObservedRLOO(OrderedObservationMixin, SFTAnchoredRLOOTrainer):
    pass


class ObservedFOARL(OrderedObservationMixin, SFTReferenceGRPOTrainer):
    pass


def smoke_rows(task):
    rows = []
    for index in range(8):
        coords = [[0, 0], [index + 1, 0], [0, index + 2]]
        values = [0, index + 1, index + 3] if task == "op" else [0, 1, 1]
        instance = [coords, values, 3 * (index + 3) if task == "op" else 2]
        objective = sum(values) if task == "op" else (
            index + 1 + math.hypot(index + 1, index + 2) + index + 2)
        answer = ("Route: [0, 1, 2]" if task == "op" else "Routes: [[0, 1, 2, 0]]")
        answer += f", Objective: {objective:.8f}"
        rows.append({"instance_id": f"{task}_{index}", "task_type": task,
                     "instance_json": instance, "instruction": f"x{index}",
                     "input": "", "prompt": f"x{index}", "output": answer})
    return rows[:4], rows[4:]


def tiny_initializer(task, output, device):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    train, validation = smoke_rows(task)
    tokens = ["[PAD]", "[UNK]", "[EOS]", *[f"x{i}" for i in range(8)],
              "Route: [0, 1]", "Route: [0, 2]", "Route: [0, 1, 2]",
              "Route: [0, 1, 1]", "Routes: [[0, 1, 2, 0]]",
              "Routes: [[0, 1, 0], [0, 2, 0]]", "Routes: [[0, 0]]",
              *[row["output"] for row in train + validation]]
    raw = Tokenizer(WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]",
                                        unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.padding_side = "left"
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    set_seed(61)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(tokens), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, pad_token_id=0, eos_token_id=2,
        bos_token_id=3, tie_word_embeddings=False, attention_dropout=0.0,
    )).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.train()
    for _ in range(8):
        batch = [[tokens.index(row["prompt"]), tokens.index(row["output"])] for row in train]
        ids = torch.tensor(batch, device=device)
        labels = ids.clone()
        labels[:, 0] = -100
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    return model, tokenizer, train, validation


def loaded_model(base, sft, device, *, trainable):
    from peft import PeftConfig, PeftModel
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(model, sft, config=PeftConfig.from_pretrained(sft),
                                     is_trainable=trainable, autocast_adapter_dtype=True,
                                     local_files_only=True, low_cpu_mem_usage=False)
    model.to(device)
    if not trainable:
        model.requires_grad_(False)
        model.eval()
    return model


def collect_outputs(model, tokenizer, contract, device):
    records = []
    model.eval()
    started = time.monotonic()
    for row in contract["validation_rows"]:
        encoded = tokenizer(row["prompt"], return_tensors="pt", truncation=False).to(device)
        if encoded.input_ids.shape[1] + contract["evaluation"]["max_completion_tokens"] > model.config.max_position_embeddings:
            raise ValueError("Validation input exceeds full context; truncation is forbidden")
        for mode, count in (("greedy", 1), ("k8", 8)):
            seed = int.from_bytes(hashlib.sha256(
                f'{contract["eval_seed"]}/{row["semantic_id"]}/{mode}'.encode()).digest()[:4], "big")
            set_seed(seed)
            kwargs = {"max_new_tokens": contract["evaluation"]["max_completion_tokens"],
                      "do_sample": mode == "k8", "num_return_sequences": count,
                      "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
            if mode == "k8":
                kwargs.update(temperature=.7, top_p=.9, top_k=50)
            autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" and contract["scope"] != "system_smoke"
                        else nullcontext())
            with torch.inference_mode(), autocast:
                generated = model.generate(**encoded, **kwargs)
            ids = generated[:, encoded.input_ids.shape[1]:].cpu().tolist()
            for index, values in enumerate(ids):
                if tokenizer.eos_token_id in values:
                    values = values[:values.index(tokenizer.eos_token_id) + 1]
                records.append({"instance_id": row["instance_id"], "decode": mode,
                                "sample_index": index, "completion_id": f'{row["instance_id"]}/{mode}/{index}',
                                "completion_token_ids": values,
                                "text": tokenizer.decode(values, skip_special_tokens=True),
                                "generated_tokens": len(values), "generation_seed": seed,
                                "runtime_complete": True,
                                "finish_reason": "eos" if values and values[-1] == tokenizer.eos_token_id else "length"})
    return records, time.monotonic() - started


def run_arm(contract, arm, output, tokenizer, initializer, device, *, base=None, sft=None):
    os.environ.update(reward_environment("foarl" if arm == "foarl" else "rloo_quality"))
    main = int(os.environ.get("RANK", "0")) == 0
    if main:
        output.mkdir(parents=True, exist_ok=False)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    set_seed(contract["train_seed"])
    model = copy.deepcopy(initializer).to(device) if initializer is not None else loaded_model(base, sft, device, trainable=True)
    beta = ARMS[arm]["beta"]
    reference = None
    identity = None
    if beta:
        reference = copy.deepcopy(initializer).to(device) if initializer is not None else loaded_model(base, sft, device, trainable=False)
        reference.requires_grad_(False)
        reference.eval()
        identity = {"source_kind": "original_sft", "source_sha256": contract["initializer"]["sha256"],
                    "immutable": True, "scope": contract["scope"]}
    train_seconds = 0.0
    train_tokens = 0
    trainer = None
    steps = 0 if arm == "sft" else contract["training"]["horizon"]
    if steps:
        world = int(os.environ.get("WORLD_SIZE", "1"))
        accumulation = 36 // world
        if world not in (1, 2, 4):
            raise ValueError("Pilot supports world sizes1,2,4 only")
        args = GRPOConfig(
            output_dir=str(output / "training"), use_cpu=device.type == "cpu",
            bf16=initializer is None, fp16=False, max_steps=steps,
            per_device_train_batch_size=1, gradient_accumulation_steps=accumulation,
            generation_batch_size=36, num_generations=9, max_prompt_length=16384,
            max_completion_length=contract["training"]["max_completion_tokens"],
            temperature=.7, top_p=.9, top_k=50, beta=beta,
            scale_rewards="group" if arm == "foarl" else "none",
            loss_type="grpo" if arm == "foarl" else "dr_grpo",
            num_iterations=1, importance_sampling_level="token", mask_truncated_completions=False,
            eval_strategy="no", save_strategy="steps", save_steps=steps, report_to="none",
            seed=contract["train_seed"], learning_rate=contract["training"]["learning_rate"],
            lr_scheduler_type="constant", warmup_ratio=0.0, optim="adamw_torch",
            gradient_checkpointing=initializer is None,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_pin_memory=False, disable_tqdm=True, logging_steps=1,
            remove_unused_columns=False, ddp_find_unused_parameters=False,
        )
        by_id = {row["instance_id"]: row for row in contract["training_rows"]}
        draw_rows = [by_id[identifier] for draw in contract["training"]["prompt_draws"] for identifier in draw]
        data = Dataset.from_list([{**row, "answer": row["output"],
                                   "instance_json": json.dumps(row["instance_json"])} for row in draw_rows])
        common = dict(model=model, args=args, processing_class=tokenizer, train_dataset=data,
                      reference_model=reference, reference_identity=identity,
                      recipe_contract={"pilot_sha256": contract["contract_sha256"], "arm": arm},
                      pilot_contract=contract)
        if arm == "foarl":
            trainer = ObservedFOARL(reward_funcs=list(FOARL_F638_REWARD_FUNCS), **common)
        else:
            reward = REWARD_HEADS[contract["task"], "reference_sigmoid_v2"]
            reward_contract = {"profile_id": PROFILE_ID, "task": contract["task"],
                               "mode": "reference_sigmoid_v2", "quality_slope": 8.0,
                               "reward_head": reward.__name__,
                               "source_manifest_sha256": file_identity(Path(__file__))["sha256"]}
            trainer = ObservedRLOO(reward_funcs=[reward], reward_contract=reward_contract,
                                   advantage_contract=make_advantage_contract(
                                       "cardinal" if ARMS[arm]["quality_credit_lambda"] else "control"), **common)
        before = semantic_state_sha256(model.state_dict())
        started = time.monotonic()
        trainer.train()
        train_seconds = time.monotonic() - started
        if trainer.state.global_step != steps or len(trainer.actual_windows) != steps:
            raise RuntimeError("Terminal optimizer/rollout counts differ")
        if before == semantic_state_sha256(trainer.accelerator.unwrap_model(trainer.model).state_dict()):
            raise RuntimeError("Actual training did not change actor tensors")
        train_tokens = trainer.actual_generated_tokens
        if beta:
            trainer._assert_reference_integrity(full=True)
        model = trainer.accelerator.unwrap_model(trainer.model)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    result = None
    if main:
        checkpoint_dir = output / "terminal"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        checkpoint = {"initializer_sha256": contract["initializer_sha256"],
                      "train_seed": contract["train_seed"], "step": steps,
                      "state_sha256": semantic_state_sha256(model.state_dict()),
                      "reference_initializer_sha256": contract["initializer_sha256"] if beta else None}
        initial_equivalence = trainer.initial_reference_audit if trainer is not None and beta else None
    else:
        initial_equivalence = None
    # Free optimizer, actor and reference before loading a third full model.
    # The constructor dictionary is another live owner of the two models.
    if steps:
        del common
    del trainer, reference, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if main:
        # Reload the actual saved artifact and require exact tensor-state fidelity.
        if initializer is not None:
            reloaded = Qwen2ForCausalLM.from_pretrained(checkpoint_dir, local_files_only=True).to(device)
        else:
            reloaded = loaded_model(base, checkpoint_dir, device, trainable=False)
        if semantic_state_sha256(reloaded.state_dict()) != checkpoint["state_sha256"]:
            raise RuntimeError("Reloaded terminal checkpoint differs from the trained tensors")
        raw, eval_seconds = collect_outputs(reloaded, tokenizer, contract, device)
        result = evaluate_arm(contract, arm, raw, checkpoint, {
            "training_optimizer_steps": steps, "training_responses": steps * 36,
            "training_generated_tokens": train_tokens, "training_wall_seconds": train_seconds,
            "evaluation_wall_seconds": eval_seconds,
            "prompt_draw_sha256": contract["training"]["prompt_draw_sha256"],
            "checkpoint_reload_exact": True, "reference_initial_equivalence":
                initial_equivalence,
        })
        write_json(output / "result.json", result)
        del reloaded
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--task", choices=("op", "cvrp", "both"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--base", type=Path)
    parser.add_argument("--sft", type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA runtime/device is available")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", rank) if args.device == "cuda" else torch.device("cpu")
    if args.device == "cuda":
        torch.cuda.set_device(device)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl" if args.device == "cuda" else "gloo")
    main_rank = int(os.environ.get("RANK", "0")) == 0
    if main_rank:
        args.output.mkdir(parents=True, exist_ok=False)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    tasks = ("op", "cvrp") if args.task == "both" else (args.task,)
    all_results = []
    for task in tasks:
        initializer = None
        if args.smoke:
            if main_rank:
                initializer, tokenizer, train, validation = tiny_initializer(task, args.output / task / "initializer", device)
                state_sha = semantic_state_sha256(initializer.state_dict())
                contract = make_contract(task, {"id": "tiny_qwen_locally_fitted_SFT", "sha256": state_sha,
                                                "scope": "system_smoke"}, train, validation,
                                         train_seed=61, horizon=4, scope="system_smoke", max_completion_tokens=1)
                write_json(args.output / task / "contract.json", contract)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            if not main_rank:
                initializer = Qwen2ForCausalLM.from_pretrained(args.output / task / "initializer",
                                                              local_files_only=True).to(device)
                tokenizer = AutoTokenizer.from_pretrained(args.output / task / "initializer", local_files_only=True)
                tokenizer.padding_side = "left"
                tokenizer.model_input_names = ["input_ids", "attention_mask"]
                contract = validate_contract(json.loads((args.output / task / "contract.json").read_text()))
        else:
            if args.device != "cuda" or args.task == "both" or not all(
                (args.contract, args.contract_sha256, args.base, args.sft, args.arm)
            ):
                raise ValueError("Real-model run requires CUDA, one task/arm, sealed contract+SHA, base and SFT")
            if file_identity(args.contract)["sha256"] != args.contract_sha256:
                raise ValueError("Contract byte identity differs from the launch declaration")
            contract = validate_contract(json.loads(args.contract.read_text()))
            if contract["task"] != task or contract["initializer"]["sha256"] != file_identity(args.sft / "adapter_model.safetensors")["sha256"]:
                raise ValueError("Task or original SFT source differs from the launch declaration")
            tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            for row in [*contract["training_rows"], *contract["validation_rows"]]:
                if len(tokenizer(row["prompt"], truncation=False).input_ids) > 16384:
                    raise ValueError("Input would be truncated by the fixed prompt cap")
        for arm in ((args.arm,) if args.arm else tuple(ARMS)):
            result = run_arm(contract, arm, args.output / task / arm, tokenizer, initializer, device,
                             base=str(args.base) if args.base else None, sft=str(args.sft) if args.sft else None)
            if result is not None:
                all_results.append(result)
        del initializer
    if main_rank:
        report = compare_results(all_results, bootstrap_reps=200 if args.smoke else 50000)
        report.update(status="completed_system_smoke" if args.smoke else "completed_single_arm_model_evaluation",
                      source=file_identity(Path(__file__)), device=str(device),
                      runtime={"torch": torch.__version__, "cuda": torch.version.cuda},
                      actual_RL_optimizer_steps=sum(r["resources"]["training_optimizer_steps"] for r in all_results),
                      actual_RL_responses=sum(r["resources"]["training_responses"] for r in all_results),
                      actual_evaluation_responses=sum(r["resources"]["evaluation_responses"] for r in all_results),
                      checkpoint_reload_exact=all(r["resources"]["checkpoint_reload_exact"] for r in all_results))
        write_json(args.output / "complete.json", report)
        print(json.dumps({key: report[key] for key in ("status", "device", "actual_RL_optimizer_steps",
                                                      "actual_RL_responses", "actual_evaluation_responses",
                                                      "superiority_established")}))
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
