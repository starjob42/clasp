"""Bounded resource measurement of the frozen CVRP V2 training prefix.

The original H100 schedule/configuration remains intact. A callback stops after
three committed optimizer updates. There is no evaluation, checkpoint, model
selection or study admission; only timing/count/integrity evidence is retained.
Native execution belongs exclusively to the separately admitted operator.
"""
from __future__ import annotations
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import types

import numpy as np
import torch
from transformers import TrainerCallback, set_seed
from clasp.entrypoints import train_sft_anchor_pilot_v2 as v2

PROTOCOL = "sft-anchor-cvrp-prefix-timing-probe-v1-result"
SOURCE2_SHA = "79fa6e424c6bcd71b91fe82af9f74ff0bdf37bd153b65d9dce94ffe2b77a3dc2"
ARM = "rloo_cardinal_beta005"
PREFIX_UPDATES = 3
PROMOTION_PATH = "experiments/rloo_sft_anchor_20261013/verify_registration_promotion_v1.py"
PROMOTION_SHA = "9b1f112935adea22d266d3aefcf0b8529d578784deec144bdf407c45b6a76b44"

def need(ok, message):
    if not ok: raise ValueError(message)

def admit_native_operator(admission, *, contract, base, sft, output):
    """Recheck the independent signature; a producer boolean is insufficient.

    A key-bearing verifier must be independently frozen before the actual source
    packet. This source-only candidate intentionally pins the empty-key verifier.
    """
    need(type(admission) is dict and admission.get("protocol") == "sft-anchor-cvrp-timing-probe-native-admission-v1",
        "Separately signed native operator admission required")
    expected = admission.get("signed_expectations")
    executed_sha = globals().get("__executed_source_sha256__")
    need(type(expected) is dict and expected.get("consumer") == PROTOCOL and type(executed_sha) is str and
        expected.get("kernel_sha256") == executed_sha == hashlib.sha256(Path(__file__).read_bytes()).hexdigest() and
        expected.get("source2_sha256") == SOURCE2_SHA and expected.get("job_id") == os.environ.get("SLURM_JOB_ID") and
        expected.get("contract_semantic_sha256") == contract["contract_sha256"] ==
            "777f6a911e2bca723ccc992d5e07978f7930fd2eeec89e592c3bd963c6e7fcef" and
        expected.get("base") == str(base) and expected.get("sft") == str(sft) and expected.get("output") == str(output) and
        expected.get("native_source_root") == str(Path(__file__).resolve().parents[3]) and
        expected.get("committed_prefix_updates") == 3 and expected.get("original_horizon") == 100 and
        expected.get("study_admission") is False and expected.get("policy_outcomes_read") is False,
        "Signed operator kernel/job/scope differs")
    path = Path(v2.__file__).resolve().parents[3] / PROMOTION_PATH
    raw = path.read_bytes()
    need(not path.is_symlink() and hashlib.sha256(raw).hexdigest() == PROMOTION_SHA,
        "Exact source-bound promotion verifier required")
    verifier = types.ModuleType("probe_bound_promotion"); verifier.__file__ = str(path)
    exec(compile(raw, str(path), "exec", dont_inherit=True), verifier.__dict__)
    verifier.verify_promotion(admission["certificate"], expected)

def fingerprint(value):
    """Semantic diagnostic digest; no pickle or tensor checkpoint is written."""
    whole = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            whole.update(json.dumps(["tensor", str(tensor.dtype), list(tensor.shape)]).encode())
            whole.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            whole.update(json.dumps(["array", str(item.dtype), list(item.shape)]).encode()); whole.update(item.tobytes())
        elif isinstance(item, dict):
            whole.update(b"dict:")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))): visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            whole.update(type(item).__name__.encode() + str(len(item)).encode())
            for member in item: visit(member)
        else:
            whole.update(type(item).__name__.encode() + b":" + repr(item).encode())
    visit(value); return whole.hexdigest()

def rng_fingerprint(device):
    states = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.random.get_rng_state()}
    if device.type == "cuda": states["torch_local_cuda"] = torch.cuda.get_rng_state(device)
    return fingerprint(states)

def environment(device):
    libraries = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "peft", "accelerate")}
    result = {"python": sys.version, "executable": sys.executable, "libraries": libraries,
        "device": str(device), "torch_cuda": torch.version.cuda,
        "rank": int(os.environ.get("RANK", "0")), "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(device_name=props.name, device_uuid=str(getattr(props, "uuid", "unavailable")),
            device_total_memory=props.total_memory, bfloat16_supported=torch.cuda.is_bf16_supported())
    return result

class StopAtThree(TrainerCallback):
    """Operational prefix stop; never changes args.max_steps or sampler state."""
    def __init__(self, trainer, *, instrumented):
        self.trainer, self.instrumented = trainer, instrumented
        self.steps = []; self.started = None; self.pre_optimizer = None

    def on_step_begin(self, args, state, control, **kwargs):
        need(args.max_steps == 100 and state.global_step < PREFIX_UPDATES, "Only the first3 updates of originalH100 may execute")
        self.started = time.monotonic(); self.pre_optimizer = None

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.pre_optimizer = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        ended = time.monotonic()
        need(0 < state.global_step <= PREFIX_UPDATES and len(self.trainer.actual_windows) == state.global_step,
            "Prefix optimizer/rollout closure differs")
        self.steps.append({"optimizer_step": state.global_step, "wall_seconds": ended - self.started,
            "through_backward_seconds": self.pre_optimizer - self.started if self.pre_optimizer is not None else None,
            "optimizer_and_existing_finite_guard_seconds": ended - self.pre_optimizer if self.pre_optimizer is not None else None,
            "cumulative_generated_tokens": self.trainer.actual_generated_tokens,
            "actual_window_sha256": v2.digest(self.trainer.actual_windows[-1])})
        if state.global_step == PREFIX_UPDATES:
            control.should_training_stop = True
            # Original save_steps remains100. Explicitly exclude terminal
            # artifacts for this profiling actor even on a future HF revision.
            control.should_save = False; control.should_evaluate = False
        return control

class PassiveTiming(v2.ObservedRLOOV2):
    def __init__(self, *args, **kwargs):
        self.stage_measurements = []
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, inputs):
        started = time.monotonic()
        try: return super()._generate_and_score_completions(inputs)
        finally: self.stage_measurements.append({"stage": "generation_and_reference_reward_scoring",
            "step": int(self.state.global_step), "wall_seconds": time.monotonic() - started})

    def compute_loss(self, *args, **kwargs):
        started = time.monotonic()
        try: return super().compute_loss(*args, **kwargs)
        finally: self.stage_measurements.append({"stage": "actor_forward_and_loss_finite_guard",
            "step": int(self.state.global_step), "wall_seconds": time.monotonic() - started})

    def log(self, *args, **kwargs):
        # Suppress quality/loss diagnostics only. Logging does not feed the
        # original loss, optimizer, sampler or RNG; actual4-rank equality tests
        # compare this instrumentation with unmodified ObservedRLOOV2 logging.
        return None

def admit_contract(contract, *, fixture):
    v2.validate_contract(contract)
    training = contract["training"]
    need(contract["task"] == "cvrp" and training["horizon"] == 100 and
        training["num_generations"] == 9 and training["responses_per_step"] == 36 and
        training["prompt_groups_per_step"] == 4 and len(training["prompt_draws"]) == 100 and
        training["learning_rate"] == 1e-5 and training["temperature"] == .7 and training["top_p"] == .9 and training["top_k"] == 50,
        "Profiling must preserve originalCVRP H100 schedule and numeric recipe")
    need(contract["scope"] == ("system_smoke" if fixture else "production") and
        (fixture or len(contract["training_rows"]) == 128) and
        (fixture or training["max_completion_tokens"] == 3000), "Tiny diagnostic and native original128/3000 scopes differ")

def make_trainer(contract, output, tokenizer, device, *, base, sft, instrumented=True):
    started = time.monotonic(); set_seed(contract["train_seed"])
    dtype = torch.float32 if contract["scope"] == "system_smoke" else torch.bfloat16
    model = v2.loaded_model(base, sft, device, trainable=True, dtype=dtype)
    actor_loaded = time.monotonic()
    reference = v2.loaded_model(base, sft, device, trainable=False, dtype=dtype)
    reference.requires_grad_(False); reference.eval(); reference_loaded = time.monotonic()
    initial_actor_sha = v2.semantic_state_sha256(model.state_dict())
    args = v2._training_args(contract, ARM, output, device, production_peft=True)
    by_id = {row["instance_id"]: row for row in contract["training_rows"]}
    draw_rows = [by_id[identifier] for draw in contract["training"]["prompt_draws"] for identifier in draw]
    data = v2.Dataset.from_list([{**row, "answer": row["output"], "instance_json": json.dumps(row["instance_json"])} for row in draw_rows])
    source = v2.file_identity(Path(v2.__file__))
    need(source["sha256"] == SOURCE2_SHA, "Frozen producer source changed")
    reward = v2.REWARD_HEADS["cvrp", "reference_sigmoid_v2"]
    trainer = (PassiveTiming if instrumented else v2.ObservedRLOOV2)(
        model=model, args=args, processing_class=tokenizer, train_dataset=data,
        reference_model=reference, reference_identity={"source_kind": "original_sft",
            "source_sha256": contract["initializer"]["sha256"], "immutable": True, "scope": contract["scope"]},
        recipe_contract={"pilot_sha256": contract["contract_sha256"], "arm": ARM,
            "operational_version": 2, "entrypoint_sha256": SOURCE2_SHA}, pilot_contract=contract,
        reward_funcs=[reward], reward_contract={"profile_id": v2.PROFILE_ID, "task": "cvrp", "mode": "reference_sigmoid_v2",
            "quality_slope": 8.0, "reward_head": reward.__name__, "source_manifest_sha256": SOURCE2_SHA},
        advantage_contract=v2.make_advantage_contract("cardinal"))
    callback = StopAtThree(trainer, instrumented=instrumented); trainer.add_callback(callback)
    need(trainer.args.max_steps == trainer.args.save_steps == 100 and trainer.beta == .05 and
        trainer.advantage_contract["weight"] == .25, "Final originalH100/cardinal/KL coefficients changed")
    return trainer, callback, initial_actor_sha, {"actor_load_seconds": actor_loaded - started,
        "reference_load_seconds": reference_loaded - actor_loaded,
        "initial_hash_dataset_and_trainer_admission_seconds": time.monotonic() - reference_loaded}

def run_probe(contract, output, tokenizer, device, *, base, sft, source_admission, fixture=False, instrumented=True):
    """Train exactly3 updates; retain diagnostic hashes and discard actor/ref."""
    admit_contract(contract, fixture=fixture)
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0"))
    need((fixture and device.type == "cpu" and world in (1, 4)) or
        (not fixture and device.type == "cuda" and world == 4),
        "Native requires separately admitted four-A100 operator; fixture scope isCPU1/4 only")
    if not fixture:
        admit_native_operator(source_admission, contract=contract, base=base, sft=sft, output=output)
        need(torch.distributed.is_initialized() and torch.distributed.get_world_size() == 4 and
            str(os.environ.get("SLURM_JOB_ID", "")).isdecimal() and
            os.environ.get("SLURM_STEP_ID") in (None, "batch") and
            "A100" in torch.cuda.get_device_name(device) and torch.cuda.is_bf16_supported(),
            "Native timing requires the actual assigned four-A100 batch allocation and BF16 support")
    output = Path(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False); (output / "training").mkdir()
    if torch.distributed.is_initialized(): torch.distributed.barrier()
    os.environ.update(v2.reward_environment("rloo_quality"))
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    trainer, callback, initial, stages = make_trainer(contract, output, tokenizer, device,
        base=base, sft=sft, instrumented=instrumented)
    started = time.monotonic(); trainer.train(); stages["training_seconds"] = time.monotonic() - started
    integrity_started = time.monotonic()
    def close_rank():
        need(trainer.state.global_step == PREFIX_UPDATES and len(trainer.actual_windows) == PREFIX_UPDATES and
            trainer.actual_windows == [[identifier for identifier in draw for _ in range(9)]
                for draw in contract["training"]["prompt_draws"][:PREFIX_UPDATES]], "First3 original windows/update counts differ")
        trainer._assert_reference_integrity(full=True)
        actor = trainer.accelerator.unwrap_model(trainer.model)
        actor_sha = v2.semantic_state_sha256(actor.state_dict())
        need(actor_sha != initial, "Committed profile updates did not change actor tensors")
        need(trainer.last_checkpointing_audit["effective_use_reentrant"] is False and
            trainer.last_checkpointing_audit["frozen_embedding_weights"] is True, "Effective PEFT checkpoint closure differs")
        return {"rank": rank, "optimizer_steps": 3, "responses": 108, "actual_windows_count": 3,
            "actual_windows_sha256": v2.digest(trainer.actual_windows), "generated_tokens": trainer.actual_generated_tokens,
            "initial_actor_state_sha256": initial, "actor_state_sha256": actor_sha,
            "reference_state_sha256": trainer.reference_state_sha256(), "reference_full_integrity_verified": True,
            "optimizer_state_sha256": fingerprint(trainer.optimizer.state_dict()),
            "lr_scheduler_state_sha256": fingerprint(trainer.lr_scheduler.state_dict()),
            "rng_state_sha256": rng_fingerprint(device),
            "actor_parameter_dtype_counts": {str(dtype): sum(p.numel() for p in actor.parameters() if p.dtype == dtype)
                for dtype in {p.dtype for p in actor.parameters()}},
            "effective_checkpointing": trainer.last_checkpointing_audit,
            "checkpointing_stage_counts": trainer.checkpointing_stage_counts, "environment": environment(device),
            "step_timings": callback.steps, "stage_measurements": getattr(trainer, "stage_measurements", []),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
    local = v2._collective_phase(close_rank, "prefix3 actor/reference/GC/count integrity")
    proofs = [local]
    if torch.distributed.is_initialized():
        proofs = [None] * world; torch.distributed.all_gather_object(proofs, local)
    need(len(proofs) == world and all(row["actor_state_sha256"] == proofs[0]["actor_state_sha256"] and
        row["reference_state_sha256"] == proofs[0]["reference_state_sha256"] and
        row["optimizer_state_sha256"] == proofs[0]["optimizer_state_sha256"] and
        row["lr_scheduler_state_sha256"] == proofs[0]["lr_scheduler_state_sha256"] for row in proofs),
        "All committed distributed actor/ref/optimizer/scheduler closures must match")
    stages["integrity_seconds"] = time.monotonic() - integrity_started
    callback.trainer = None
    del callback, trainer; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    stages["cleanup_seconds"] = time.monotonic() - integrity_started - stages["integrity_seconds"]
    if torch.distributed.is_initialized(): torch.distributed.barrier()
    if rank == 0:
        need(not any(path.name.startswith("checkpoint-") or path.name in ("optimizer.pt", "adapter_model.safetensors", "model.safetensors", "trainer_state.json")
            for path in output.rglob("*")), "Profiling actor/optimizer checkpoint must never be persisted")
        result = {"protocol": PROTOCOL, "schema_version": 1, "status": "completed_three_update_timing_probe",
            "scope": "system_smoke" if fixture else "resource_timing_measurement", "synthetic_fixture": fixture,
            "source_admission": source_admission, "source2": v2.file_identity(Path(v2.__file__)),
            "profiler_source": v2.file_identity(Path(__file__)), "contract_sha256": contract["contract_sha256"],
            "task": "cvrp", "recipe_arm": ARM, "original_horizon": 100, "committed_prefix_updates": 3,
            "responses": 108, "num_generations": 9, "global_responses_per_step": 36,
            "quality_credit_lambda": .25, "beta": .05, "learning_rate": 1e-5,
            "temperature": .7, "top_p": .9, "top_k": 50, "max_completion_tokens": contract["training"]["max_completion_tokens"],
            "distributed": {"world_size": world, "rank_closure": proofs}, "stage_timings": stages,
            "actor_discarded": True, "reference_discarded": True, "model_checkpoint_written": False, "evaluation_responses": 0,
            "heldout_policy_calls": 0, "intermediate_checkpoint_selection": False,
            "study_admission": False, "superiority_established": False,
            "timing_representativeness_or_full_budget_adequacy_proven": False,
            "stage_timing_scope": "host_monotonic_wall_with_original_finite_guard_sync;no_added_cuda_synchronize;not_full100_budget_proof",
            "logging_quality_metrics_suppressed": instrumented}
        result["receipt_sha256"] = v2.digest(result); v2.write_json_atomic(output / "timing_probe.json", result)
    else: result = None
    if torch.distributed.is_initialized(): torch.distributed.barrier(); torch.distributed.destroy_process_group()
    return result
