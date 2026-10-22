"""Versioned operational repair of the frozen SFT-anchor pilot.

The statistical recipe is inherited from v1. Effective non-reentrant PEFT
checkpointing is restored after TRL generation, terminal training/model proof
is sealed before evaluation, and single-arm distributed workers close their
process group before rank zero performs serial evaluation. The production
fixture uses actual tiny PEFT loads, frozen embeddings and checkpointed RL;
it is integration evidence only.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM, set_seed
from trl import GRPOConfig

from clasp.baselines.budget_matched_dev import reward_environment
from clasp.baselines.llmcosolver_f638_training import FOARL_F638_REWARD_FUNCS
from clasp.baselines.quality_signal_dev import PROFILE_ID, REWARD_HEADS
from clasp.baselines.sft_anchor_pilot import ARMS, compare_results, digest, evaluate_arm, make_contract, validate_contract
from clasp.entrypoints import train_sft_anchor_pilot as v1
from clasp.models.external_baseline_state import semantic_state_sha256
from clasp.rewards.feasible_quality import make_advantage_contract

OPERATIONAL_VERSION = 2
file_identity = v1.file_identity


def write_json_atomic(path, value):
    """Create a durable receipt once, without exposing partially written JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Preserve previous attempt: {path}")
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # link is an atomic create-if-absent operation, preserving an existing
        # destination even if another writer creates it after the first check.
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _append_durable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _collective_phase(action, label):
    result, error = None, None
    try:
        result = action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [error]
    if torch.distributed.is_initialized():
        errors = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(errors, error)
    if any(item is not None for item in errors):
        raise RuntimeError(f"{label} failed across ranks: {errors}")
    return result


def restore_nonreentrant_checkpointing(model):
    """Restore and inspect the effective function on every checkpointed layer.

    TRL 0.24 enables PEFT checkpointing without kwargs during construction and
    re-enables it without kwargs after generation. Inspecting arguments alone
    therefore does not establish the checkpointing mode used by actor loss.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    enabled = []
    for name, module in base.named_modules():
        if not getattr(module, "gradient_checkpointing", False):
            continue
        function = getattr(module, "_gradient_checkpointing_func", None)
        keywords = getattr(function, "keywords", None)
        if not isinstance(keywords, dict) or keywords.get("use_reentrant") is not False:
            raise ValueError(f"Effective checkpoint function is not explicitly non-reentrant: {name}")
        enabled.append(name)
    if not enabled or not base.is_gradient_checkpointing:
        raise ValueError("No effective checkpointed layers were enabled")
    return {"status": "passed", "declared_use_reentrant": False,
            "effective_use_reentrant": False, "enabled_modules": enabled,
            "frozen_embedding_weights": not model.get_input_embeddings().weight.requires_grad}


class EffectiveCheckpointingMixin:
    """Keep v1 credit/loss while admitting the actual declared GC semantics."""

    def __init__(self, *args, **kwargs):
        self.checkpointing_stage_counts = {}
        self.last_checkpointing_audit = None
        super().__init__(*args, **kwargs)
        self._checkpointing_admit("construction")

    def _checkpointing_admit(self, stage):
        def restore():
            if not self.args.gradient_checkpointing:
                return {"status": "disabled", "effective_use_reentrant": None,
                        "enabled_modules": []}
            if self.args.gradient_checkpointing_kwargs != {"use_reentrant": False}:
                raise ValueError("The declared pilot checkpointing kwargs changed")
            return restore_nonreentrant_checkpointing(self.accelerator.unwrap_model(self.model))

        audit = self._collective_check(restore, f"effective checkpointing {stage}")
        self.last_checkpointing_audit = audit
        self.checkpointing_stage_counts[stage] = self.checkpointing_stage_counts.get(stage, 0) + 1
        return audit

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        self._checkpointing_admit("after_generation")
        return output

    def _compute_loss(self, model, inputs):
        self._checkpointing_admit("before_actor_loss")
        return super()._compute_loss(model, inputs)


class ObservedRLOOV2(EffectiveCheckpointingMixin, v1.ObservedRLOO):
    pass


class ObservedFOARLV2(EffectiveCheckpointingMixin, v1.ObservedFOARL):
    pass


def loaded_model(base, sft, device, *, trainable, dtype=torch.bfloat16):
    from peft import PeftConfig, PeftModel
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(
        model, sft, config=PeftConfig.from_pretrained(sft), is_trainable=trainable,
        autocast_adapter_dtype=True, local_files_only=True, low_cpu_mem_usage=False,
    ).to(device)
    if not trainable:
        model.requires_grad_(False)
        model.eval()
    return model


def tiny_peft_initializer(task, base_path, sft_path, device):
    """Save a local tiny supervised policy using the production PEFT loader."""
    from peft import LoraConfig, get_peft_model
    base, tokenizer, train, validation = v1.tiny_initializer(task, base_path, device)
    policy = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=2, lora_dropout=0.0, bias="all", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    policy.peft_config["default"].base_model_name_or_path = str(Path(base_path).resolve())
    policy.save_pretrained(sft_path)
    tokenizer.save_pretrained(sft_path)
    if policy.get_input_embeddings().weight.requires_grad:
        raise ValueError("Production fixture must have frozen embedding weights")
    return policy, tokenizer, train, validation


def artifact_tree(directory):
    directory = Path(directory)
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Linked model artifact forbidden")
        if path.is_file():
            records.append({"path": str(path.relative_to(directory)),
                            "sha256": file_identity(path)["sha256"], "size_bytes": path.stat().st_size})
    if not records:
        raise ValueError("Empty model artifact")
    return {"files": records, "tree_sha256": digest(records)}


def collect_outputs(model, tokenizer, contract, device, *, progress_path):
    """Preserve the v1 decoder and seeds, sealing each completed mode group."""
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
            call_started = time.monotonic()
            with torch.inference_mode(), autocast:
                generated = model.generate(**encoded, **kwargs)
            ids = generated[:, encoded.input_ids.shape[1]:].cpu().tolist()
            group = []
            for index, values in enumerate(ids):
                if tokenizer.eos_token_id in values:
                    values = values[:values.index(tokenizer.eos_token_id) + 1]
                group.append({"instance_id": row["instance_id"], "decode": mode,
                              "sample_index": index, "completion_id": f'{row["instance_id"]}/{mode}/{index}',
                              "completion_token_ids": values, "text": tokenizer.decode(values, skip_special_tokens=True),
                              "generated_tokens": len(values), "generation_seed": seed,
                              "runtime_complete": True,
                              "finish_reason": "eos" if values and values[-1] == tokenizer.eos_token_id else "length"})
            records.extend(group)
            _append_durable(progress_path, {"schema_version": 2, "instance_id": row["instance_id"],
                                            "semantic_id": row["semantic_id"], "decode": mode, "seed": seed,
                                            "raw_completions": group,
                                            "generated_tokens": sum(item["generated_tokens"] for item in group),
                                            "wall_seconds": time.monotonic() - call_started})
    return records, time.monotonic() - started


def _training_args(contract, arm, output, device, *, production_peft):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (1, 2, 4):
        raise ValueError("Pilot supports world sizes1,2,4 only")
    return GRPOConfig(
        output_dir=str(output / "training"), use_cpu=device.type == "cpu",
        bf16=production_peft and contract["scope"] != "system_smoke", fp16=False,
        max_steps=contract["training"]["horizon"],
        per_device_train_batch_size=1, gradient_accumulation_steps=36 // world,
        generation_batch_size=36, num_generations=9, max_prompt_length=16384,
        max_completion_length=contract["training"]["max_completion_tokens"],
        temperature=.7, top_p=.9, top_k=50, beta=ARMS[arm]["beta"],
        scale_rewards="group" if arm == "foarl" else "none",
        loss_type="grpo" if arm == "foarl" else "dr_grpo",
        num_iterations=1, importance_sampling_level="token", mask_truncated_completions=False,
        eval_strategy="no", save_strategy="steps", save_steps=contract["training"]["horizon"], report_to="none",
        seed=contract["train_seed"], learning_rate=contract["training"]["learning_rate"],
        lr_scheduler_type="constant", warmup_ratio=0.0, optim="adamw_torch",
        gradient_checkpointing=production_peft, gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False, disable_tqdm=True, logging_steps=1,
        remove_unused_columns=False, ddp_find_unused_parameters=False,
    )


def run_arm(contract, arm, output, tokenizer, initializer, device, *, base=None, sft=None,
            production_fixture=False, close_distributed_before_evaluation=False):
    os.environ.update(reward_environment("foarl" if arm == "foarl" else "rloo_quality"))
    main = int(os.environ.get("RANK", "0")) == 0
    if main:
        output.mkdir(parents=True, exist_ok=False)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    set_seed(contract["train_seed"])
    dtype = torch.float32 if production_fixture else torch.bfloat16
    model = (v1.copy.deepcopy(initializer).to(device) if initializer is not None
             else loaded_model(base, sft, device, trainable=True, dtype=dtype))
    beta = ARMS[arm]["beta"]
    reference, identity = None, None
    if beta:
        reference = (v1.copy.deepcopy(initializer).to(device) if initializer is not None
                     else loaded_model(base, sft, device, trainable=False, dtype=dtype))
        reference.requires_grad_(False)
        reference.eval()
        identity = {"source_kind": "original_sft", "source_sha256": contract["initializer"]["sha256"],
                    "immutable": True, "scope": contract["scope"]}
    trainer, train_seconds, train_tokens = None, 0.0, 0
    steps = 0 if arm == "sft" else contract["training"]["horizon"]
    initial_actor_sha = semantic_state_sha256(model.state_dict())
    if steps:
        args = _training_args(contract, arm, output, device, production_peft=initializer is None)
        by_id = {row["instance_id"]: row for row in contract["training_rows"]}
        draw_rows = [by_id[identifier] for draw in contract["training"]["prompt_draws"] for identifier in draw]
        data = Dataset.from_list([{**row, "answer": row["output"],
                                   "instance_json": json.dumps(row["instance_json"])} for row in draw_rows])
        common = dict(model=model, args=args, processing_class=tokenizer, train_dataset=data,
                      reference_model=reference, reference_identity=identity,
                      recipe_contract={"pilot_sha256": contract["contract_sha256"], "arm": arm,
                                       "operational_version": 2, "entrypoint_sha256": file_identity(Path(__file__))["sha256"]},
                      pilot_contract=contract)
        if arm == "foarl":
            trainer = ObservedFOARLV2(reward_funcs=list(FOARL_F638_REWARD_FUNCS), **common)
        else:
            reward = REWARD_HEADS[contract["task"], "reference_sigmoid_v2"]
            reward_contract = {"profile_id": PROFILE_ID, "task": contract["task"], "mode": "reference_sigmoid_v2",
                               "quality_slope": 8.0, "reward_head": reward.__name__,
                               "source_manifest_sha256": file_identity(Path(__file__))["sha256"]}
            trainer = ObservedRLOOV2(reward_funcs=[reward], reward_contract=reward_contract,
                                     advantage_contract=make_advantage_contract(
                                         "cardinal" if ARMS[arm]["quality_credit_lambda"] else "control"), **common)
        started = time.monotonic()
        trainer.train()
        train_seconds = time.monotonic() - started
        if trainer.state.global_step != steps or len(trainer.actual_windows) != steps:
            raise RuntimeError("Terminal optimizer/rollout counts differ")
        train_tokens = trainer.actual_generated_tokens
        model = trainer.accelerator.unwrap_model(trainer.model)

    def rank_proof():
        actor_sha = semantic_state_sha256(model.state_dict())
        if steps and initial_actor_sha == actor_sha:
            raise RuntimeError("Actual training did not change actor tensors")
        if beta:
            trainer._assert_reference_integrity(full=True)
        return {"rank": int(os.environ.get("RANK", "0")), "status": "passed",
                "optimizer_steps": steps, "actor_state_sha256": actor_sha,
                "training_wall_seconds": train_seconds,
                "generated_tokens": train_tokens,
                "reference_state_sha256": trainer.reference_state_sha256() if beta else None,
                "reference_full_integrity_verified": True if beta else None,
                "checkpointing_stage_counts": trainer.checkpointing_stage_counts if trainer else {},
                "effective_checkpointing": trainer.last_checkpointing_audit if trainer else None}

    local_proof = _collective_phase(rank_proof, "terminal rank integrity")
    proofs = [local_proof]
    if torch.distributed.is_initialized():
        proofs = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(proofs, local_proof)
    if any(item["actor_state_sha256"] != proofs[0]["actor_state_sha256"]
           or item["optimizer_steps"] != steps for item in proofs):
        raise RuntimeError("Terminal actors/update counts differ across ranks")
    checkpoint = {"initializer_sha256": contract["initializer_sha256"], "train_seed": contract["train_seed"],
                  "step": steps, "state_sha256": proofs[0]["actor_state_sha256"],
                  "reference_initializer_sha256": contract["initializer_sha256"] if beta else None}
    checkpoint_dir = output / "terminal"

    def seal_training():
        if not main:
            return None
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        actual_windows = output / "training/actual_prompt_windows.jsonl"
        payload = {"schema_version": 2, "protocol": "sft-anchor-training-complete-v2", "status": "passed",
                   "operational_version": 2, "task": contract["task"], "arm": arm, "scope": contract["scope"],
                   "pilot_contract_sha256": contract["contract_sha256"], "checkpoint": checkpoint,
                   "training": {"optimizer_steps": steps, "responses": steps * 36, "generated_tokens": train_tokens,
                                "wall_seconds": train_seconds, "prompt_draw_sha256": contract["training"]["prompt_draw_sha256"],
                                "rollout_windows_count": len(trainer.actual_windows) if trainer else 0,
                                "actual_prompt_windows_sha256": file_identity(actual_windows)["sha256"] if steps else None},
                   "reference": {"beta": beta, "state_sha256": proofs[0]["reference_state_sha256"],
                                 "full_integrity_verified": True if beta else None,
                                 "initial_equivalence": trainer.initial_reference_audit if beta else None,
                                 "prepared_equivalence": trainer.prepared_reference_audit if beta else None},
                   "distributed": {"world_size": len(proofs), "rank_closure": proofs},
                   "terminal_artifact_tree": artifact_tree(checkpoint_dir), "source": file_identity(Path(__file__))}
        write_json_atomic(output / "training_complete.json", payload)
        return payload

    training_proof = _collective_phase(seal_training, "durable terminal training proof")
    if steps:
        del common
    del trainer, reference, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    group_closed = False
    if close_distributed_before_evaluation and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        group_closed = True
    elif torch.distributed.is_initialized():
        torch.distributed.barrier()
    result = None
    if main:
        if initializer is not None:
            reloaded = Qwen2ForCausalLM.from_pretrained(checkpoint_dir, local_files_only=True).to(device)
        else:
            reloaded = loaded_model(base, checkpoint_dir, device, trainable=False, dtype=dtype)
        reloaded_sha = semantic_state_sha256(reloaded.state_dict())
        if reloaded_sha != checkpoint["state_sha256"]:
            raise RuntimeError("Reloaded terminal checkpoint differs from original trained tensors")
        reload_proof = {"schema_version": 2, "protocol": "sft-anchor-checkpoint-reload-v2", "status": "passed",
                        "training_complete_sha256": file_identity(output / "training_complete.json")["sha256"],
                        "expected_actor_state_sha256": checkpoint["state_sha256"],
                        "reloaded_actor_state_sha256": reloaded_sha, "bitwise_state_equal": True,
                        "checkpoint_path": str(checkpoint_dir), "source": file_identity(Path(__file__)),
                        "distributed_group_closed_before_evaluation": group_closed}
        write_json_atomic(output / "checkpoint_reload_exact.json", reload_proof)
        raw, eval_seconds = collect_outputs(reloaded, tokenizer, contract, device,
                                            progress_path=output / "evaluation_groups.jsonl")
        result = evaluate_arm(contract, arm, raw, checkpoint, {
            "training_optimizer_steps": steps, "training_responses": steps * 36,
            "training_generated_tokens": train_tokens, "training_wall_seconds": train_seconds,
            "evaluation_wall_seconds": eval_seconds, "prompt_draw_sha256": contract["training"]["prompt_draw_sha256"],
            "checkpoint_reload_exact": True, "reference_initial_equivalence": training_proof["reference"]["initial_equivalence"],
            "operational_version": 2, "production_fixture": production_fixture,
            "training_complete_sha256": file_identity(output / "training_complete.json")["sha256"],
            "checkpoint_reload_exact_sha256": file_identity(output / "checkpoint_reload_exact.json")["sha256"],
            "evaluation_groups_sha256": file_identity(output / "evaluation_groups.jsonl")["sha256"],
            "evaluation_groups_closed": True,
            "distributed_group_closed_before_evaluation": group_closed,
        })
        write_json_atomic(output / "result.json", result)
        del reloaded
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    fixtures = parser.add_mutually_exclusive_group()
    fixtures.add_argument("--smoke", action="store_true")
    fixtures.add_argument("--production-fixture", action="store_true")
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
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    main_rank = int(os.environ.get("RANK", "0")) == 0
    device = torch.device("cuda", rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1 and (not args.arm or args.task == "both"):
        raise ValueError("Distributed v2 must select one task and one arm before group teardown")
    if world > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    if main_rank:
        args.output.mkdir(parents=True, exist_ok=False)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    fixture = args.smoke or args.production_fixture
    tasks = ("op", "cvrp") if args.task == "both" else (args.task,)
    results = []
    for task in tasks:
        initializer = None
        base_path, sft_path = args.base, args.sft
        if fixture:
            if main_rank:
                if args.production_fixture:
                    base_path = args.output / task / "fixture_base"
                    sft_path = args.output / task / "fixture_sft"
                    model, tokenizer, train, validation = tiny_peft_initializer(task, base_path, sft_path, device)
                    identity = {"id": "tiny_peft_frozen_embeddings_checkpointed_SFT",
                                "sha256": file_identity(sft_path / "adapter_model.safetensors")["sha256"],
                                "scope": "system_smoke"}
                    del model
                    gc.collect()
                else:
                    initializer, tokenizer, train, validation = v1.tiny_initializer(task, args.output / task / "initializer", device)
                    identity = {"id": "tiny_qwen_locally_fitted_SFT", "sha256": semantic_state_sha256(initializer.state_dict()),
                                "scope": "system_smoke"}
                contract = make_contract(task, identity, train, validation, train_seed=61, horizon=4,
                                         scope="system_smoke", max_completion_tokens=1)
                write_json_atomic(args.output / task / "contract.json", contract)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            if not main_rank:
                base_path = args.output / task / ("fixture_base" if args.production_fixture else "initializer")
                sft_path = args.output / task / "fixture_sft" if args.production_fixture else None
                tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
                tokenizer.padding_side = "left"
                tokenizer.model_input_names = ["input_ids", "attention_mask"]
                contract = validate_contract(json.loads((args.output / task / "contract.json").read_text()))
                if not args.production_fixture:
                    initializer = Qwen2ForCausalLM.from_pretrained(base_path, local_files_only=True).to(device)
        else:
            if device.type != "cuda" or args.task == "both" or not all(
                    (args.contract, args.contract_sha256, args.base, args.sft, args.arm)):
                raise ValueError("Real-model run requires CUDA, one task/arm, sealed contract+SHA, base and SFT")
            if file_identity(args.contract)["sha256"] != args.contract_sha256:
                raise ValueError("Contract byte identity differs")
            contract = validate_contract(json.loads(args.contract.read_text()))
            if contract["task"] != task or contract["initializer"]["sha256"] != file_identity(args.sft / "adapter_model.safetensors")["sha256"]:
                raise ValueError("Task or original SFT source differs")
            tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            for row in [*contract["training_rows"], *contract["validation_rows"]]:
                if len(tokenizer(row["prompt"], truncation=False).input_ids) > 16384:
                    raise ValueError("Input would be truncated by the fixed prompt cap")
        for arm in ((args.arm,) if args.arm else tuple(ARMS)):
            result = run_arm(contract, arm, args.output / task / arm, tokenizer, initializer, device,
                             base=str(base_path) if base_path else None, sft=str(sft_path) if sft_path else None,
                             production_fixture=args.production_fixture,
                             close_distributed_before_evaluation=world > 1)
            if result is not None:
                results.append(result)
        del initializer
    if main_rank:
        report = compare_results(results, bootstrap_reps=200 if fixture else 50000)
        report.update(status="completed_system_smoke" if fixture else "completed_single_arm_model_evaluation",
                      source=file_identity(Path(__file__)), operational_version=2,
                      production_fixture=args.production_fixture, device=str(device),
                      runtime={"torch": torch.__version__, "cuda": torch.version.cuda},
                      actual_RL_optimizer_steps=sum(r["resources"]["training_optimizer_steps"] for r in results),
                      actual_RL_responses=sum(r["resources"]["training_responses"] for r in results),
                      actual_evaluation_responses=sum(r["resources"]["evaluation_responses"] for r in results),
                      checkpoint_reload_exact=all(r["resources"]["checkpoint_reload_exact"] for r in results))
        write_json_atomic(args.output / "complete.json", report)
        print(json.dumps({key: report[key] for key in ("status", "operational_version", "production_fixture",
                                                      "actual_RL_optimizer_steps", "actual_RL_responses",
                                                      "actual_evaluation_responses", "superiority_established")}))
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
