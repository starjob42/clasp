"""Separate V4 CUDA/DDP execution path; default native registry is empty.

System-smoke fixtures exercise actual tiny PEFT training and three decoders.
They cannot authorize native7B work or establish a quality comparison. The
original V4 statistical helper and every frozen V2 byte remain unchanged.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import torch
from datasets import Dataset
from transformers import AutoTokenizer, GenerationConfig, set_seed

from clasp.baselines.sft_anchor_pilot import digest, make_contract as base_contract, score_completion
from clasp.entrypoints import train_sft_anchor_best8_v4 as v4
from clasp.entrypoints import train_sft_anchor_pilot_v2 as v2
from clasp.entrypoints import train_sft_anchor_recovery_v3 as v3
from clasp.models.external_baseline_state import semantic_state_sha256
from clasp.rewards.feasible_quality import make_advantage_contract

PROTOCOL = "sft-anchor-strict-best8-cuda-ddp-v4-v1"
STATISTICAL_SHA = "9c474eee64ce56dc7bfe78cf40e7fc41fcac14bf3e7da687000637bcc9522064"
# A caller-created registry and receipt cannot authorize native execution.
# An actual independently registered source version must close this allowlist.
# Contracts exclude driver/manifest hashes, so source closure can precede the
# independent registry without a source<->registry hash fixed-point cycle.
KNOWN_NATIVE_CONTRACTS = {}
PROMOTION_SHA = "9b1f112935adea22d266d3aefcf0b8529d578784deec144bdf407c45b6a76b44"
PROMOTION_REL = "experiments/rloo_sft_anchor_20261013/verify_registration_promotion_v1.py"
need, identity, write = v3.need, v2.file_identity, v2.write_json_atomic


def production_declaration():
    """No inputs/seeds/resources/registration are instantiated by this API."""
    return {**v4.prospective_template(), "operational_protocol": PROTOCOL,
        "world_size_trained": 4, "world_size_sft": 1, "scope": "prospective_declaration",
        "native_registration_created": False, "native_submission_authorized": False}


def validate_contract(contract):
    need(identity(Path(v4.__file__))["sha256"] == STATISTICAL_SHA, "Original V4 statistical source changed")
    if contract.get("scope") == "system_smoke":
        return v4.validate_smoke_contract(contract)
    need(contract.get("scope") == "production" and contract.get("protocol") == PROTOCOL + "-contract",
         "Explicit production contract or labelled system_smoke required")
    expected = base_contract(contract["task"], contract["initializer"], contract["training_rows"], contract["validation_rows"],
        scope="production", horizon=100, max_completion_tokens=3000, train_seed=contract["train_seed"],
        eval_seed=contract["eval_seed"], cohort_provenance=contract["cohort_provenance"])
    for key in ("forbidden_semantic_sha256", "forbidden_graph_semantic_sha256", "forbidden_instance_count"):
        expected[key] = contract[key]
    expected.update(protocol=PROTOCOL + "-contract", schema_version=4, recipes=copy.deepcopy(v4.RECIPES),
        evaluation_streams=copy.deepcopy(v4.DECODE_STREAMS), registration_status="independent_native_registration_required")
    expected.pop("arms"); expected["training"].update(top_p=1., top_k=0)
    expected["evaluation"] = {"max_completion_tokens": 3000, "repair": False,
        "primary": "k8_full_support", "secondary": "k8_filtered_secondary", "actual_greedy": True}
    expected["contract_sha256"] = digest({k: v for k, v in expected.items() if k != "contract_sha256"})
    need(contract == expected and len(contract["training_rows"]) == 128 and len(contract["validation_rows"]) == 96,
         "V4 fixed128/96/H100/G9x4/LR1e-5/cap3000 recipe or cohort differs")
    return copy.deepcopy(expected)


def admit_production(contract, recipe, registry, *, registry_sha256, base, sft, device, output=None,
                     promotion=None, promotion_sha256=None):
    """Independent byte-bound native admission, never a self-authorized switch."""
    import types
    anchor = KNOWN_NATIVE_CONTRACTS.get(contract["contract_sha256"])
    need(anchor is not None and anchor.get("task") == contract["task"] and recipe in anchor.get("recipes", []),
         "Known native V4 registry is empty; source-only preparation cannot authorize a policy call")
    need(contract["scope"] == "production" and device.type == "cuda" and torch.cuda.is_bf16_supported() and
         torch.cuda.get_device_capability(device)[0] >= 8,
         "Production requires actual CUDA BF16, not a tiny FP32 smoke")
    need(globals().get("__executed_source_sha256__") == identity(Path(__file__))["sha256"] and
         globals().get("__bound_project_manifest_sha256__"), "Production requires exact-buffer closed project bootstrap")
    raw = Path(registry).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == registry_sha256, "Independent registry byte identity differs")
    registration = json.loads(raw)
    need(registration.get("protocol") == PROTOCOL + "-registry" and type(registration.get("entries")) is list,
         "Closed versioned V4 native registry required")
    entries = [e for e in registration["entries"] if e.get("contract_sha256") == contract["contract_sha256"] and e.get("recipe") == recipe]
    need(len(entries) == 1, "Default V4 native registry is empty; no production policy call authorized")
    e = entries[0]
    required = ("independent_prepolicy_registration", "fresh_training_validation_inputs_certified", "original_SFT_artifacts_admitted",
        "full_source_environment_admitted", "bounded_cumulative_authority_admitted", "native_BF16_7B_DDP_path_validated")
    need(all(e.get(k) is True for k in required) and e.get("policy_outcomes_read") is False and
         e.get("statistical_source_sha256") == STATISTICAL_SHA and
         e.get("driver_sha256") == globals()["__executed_source_sha256__"] and
         e.get("project_manifest_sha256") == globals()["__bound_project_manifest_sha256__"] and
         e.get("base") == str(base) and e.get("sft") == str(sft) and
         e.get("output") == str(output) and e.get("first_attempt") is True and e.get("automatic_retry") is False and
         e.get("world_size") == (1 if recipe == "sft" else 4) and
         e.get("job_id") == os.environ.get("SLURM_JOB_ID") and
         type(e.get("job_id")) is str and e["job_id"].isdecimal() and
         set(e.get("expected_rank_environments", {})) == {str(i) for i in range(e["world_size"])},
         "Native source/model/authority/recipe/first-job custody is incomplete")
    proof_binding = e["independent_registration_binding"]
    proof_raw = Path(proof_binding["path"]).read_bytes()
    need(len(proof_raw) == proof_binding["size_bytes"] and hashlib.sha256(proof_raw).hexdigest() == proof_binding["sha256"],
         "Independent native registration bytes differ")
    proof = json.loads(proof_raw)
    need(proof.get("protocol") == PROTOCOL + "-independent-prepolicy-registration" and
         proof.get("contract_sha256") == contract["contract_sha256"] and
         proof.get("registry_entry_sha256") == digest({k: v for k, v in e.items() if k != "independent_registration_binding"}) and
         proof.get("independent_read_only_custody") is True and proof.get("policy_outcomes_read") is False,
         "Source-only declarations do not establish independent native registration")
    need(not any(v is True for k, v in proof.items() if k in ("synthetic_fixture", "system_smoke")),
         "Fixture cannot authorize native V4")
    buffers = globals().get("__bound_project_buffers__", {})
    verifier_raw = buffers.get(PROMOTION_REL)
    need(verifier_raw is not None and hashlib.sha256(verifier_raw).hexdigest() == PROMOTION_SHA,
         "Independent promotion verifier must be in the exact-buffer closed source inventory")
    verifier = types.ModuleType("_v4_native_promotion"); verifier.__file__ = str(Path(__file__).parents[3] / PROMOTION_REL)
    exec(compile(verifier_raw, verifier.__file__, "exec", dont_inherit=True), verifier.__dict__)
    promotion_raw = Path(promotion).read_bytes()
    need(hashlib.sha256(promotion_raw).hexdigest() == promotion_sha256, "External signed promotion bytes differ")
    verifier.verify_promotion(json.loads(promotion_raw), {"consumer": PROTOCOL + "-native-execution",
        "contract_sha256": contract["contract_sha256"], "recipe": recipe, "registry_sha256": registry_sha256,
        "driver_sha256": globals()["__executed_source_sha256__"],
        "effective_source_manifest_sha256": globals()["__bound_project_manifest_sha256__"],
        "statistical_source_sha256": STATISTICAL_SHA, "job_id": e["job_id"], "first_attempt": True,
        "output": str(output), "independent_prepolicy_registration_sha256": proof_binding["sha256"],
        "resource_authority_binding": e["resource_authority_binding"], "policy_outcomes_read": False}, synthetic_fixture=False)
    # The independent registration supplies an exhaustive pre-policy loader
    # inventory. Each rank replays it before model loading, including configs.
    loader = e["initializer_inventory"]
    verify_loader_inventory(base, sft, loader)
    return {**e, "_admission_replay": {"registry":str(registry), "registry_sha256":registry_sha256,
        "promotion":str(promotion), "promotion_sha256":promotion_sha256}}


def verify_loader_inventory(base, sft, inventory):
    need(set(inventory) == {"base", "sft"}, "Complete original loader inventory required")
    for kind, root in (("base", Path(base)), ("sft", Path(sft))):
        need(root.is_dir() and not root.is_symlink() and root.resolve() == root, "Canonical unlinked loader roots required")
        rows = inventory[kind]
        need(type(rows) is list and rows and len({r["path"] for r in rows}) == len(rows), "Unique loader files required")
        expected = {r["path"]: r for r in rows}
        def files():
            found = set()
            for path in root.rglob("*"):
                need(not path.is_symlink() and (path.is_file() or path.is_dir()), "Linked/nonregular loader artifact")
                if path.is_file(): found.add(path.relative_to(root).as_posix())
            return found
        need(files() == set(expected), "Loader artifact inventory differs")
        snapshots = {}
        for rel, row in expected.items():
            p = Path(rel)
            need(not p.is_absolute() and p.parts and p.as_posix() == rel and all(x not in (".", "..") for x in p.parts), "Unsafe loader path")
            path = root / rel; digest_ = hashlib.sha256(); count = 0
            with path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                while block := stream.read(1024 * 1024): digest_.update(block); count += len(block)
                after = os.fstat(stream.fileno())
            keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            snap = tuple(getattr(before, k) for k in keys)
            need(snap == tuple(getattr(after, k) for k in keys) == tuple(getattr(path.stat(), k) for k in keys) and
                 count == row["size_bytes"] and digest_.hexdigest() == row["sha256"], "Loader bytes/config changed")
            snapshots[rel] = snap
        need(files() == set(expected) and all(snapshots[r] == tuple(getattr((root / r).stat(), k) for k in
             ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")) for r in expected), "Loader changed across replay")


def environment(model=None):
    import peft, transformers, trl, accelerate
    value = {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__, "accelerate": accelerate.__version__,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "rank": v3._rank(), "world_size": v3._world(),
        "backend": torch.distributed.get_backend() if torch.distributed.is_initialized() else None}
    if torch.cuda.is_available():
        value.update(cuda_name=torch.cuda.get_device_name(), cuda_capability=list(torch.cuda.get_device_capability()),
            bf16_supported=torch.cuda.is_bf16_supported())
    if model is not None:
        counts = {}
        for p in model.parameters(): counts[str(p.dtype)] = counts.get(str(p.dtype), 0) + p.numel()
        value.update(parameter_dtype_counts=counts, parameter_count=sum(p.numel() for p in model.parameters()))
    return value


def make_trainer(contract, recipe, output, tokenizer, base, sft, device, *, dtype):
    settings = v4.RECIPES[recipe]
    v2._collective_phase(lambda: Path(output).mkdir(parents=True, exist_ok=False) if v3._rank() == 0 else None, "new V4 output")
    set_seed(contract["train_seed"])
    os.environ.update(v4.reward_environment("foarl" if settings["method"] == "grpo" else "rloo_quality"))
    actor = v2.loaded_model(base, sft, device, trainable=True, dtype=dtype)
    reference = v2.loaded_model(base, sft, device, trainable=False, dtype=dtype) if settings["beta"] else None
    # HF may merge saved model defaults into a supplied trainer config. Clear
    # those defaults explicitly so the unchanged V4 sampler cannot inherit a
    # repetition penalty or hidden token support constraint from an asset.
    clean_config = evaluation_config({"do_sample":True,"count":1,"temperature":.7,"top_p":1.,"top_k":0},
        tokenizer, contract["training"]["max_completion_tokens"])
    actor.generation_config = copy.deepcopy(clean_config)
    actor.get_base_model().generation_config = copy.deepcopy(clean_config)
    need(identity(Path(sft) / "adapter_model.safetensors")["sha256"] == contract["initializer"]["sha256"], "Original SFT adapter changed")
    ref_identity = {"source_kind": "original_sft", "source_sha256": contract["initializer"]["sha256"],
        "immutable": True, "scope": contract["scope"]} if reference is not None else None
    alias = "foarl" if settings["method"] == "grpo" else "rloo_scalar_beta005" if settings["beta"] else "rloo_scalar_beta0"
    args = v2._training_args(contract, alias, Path(output), device, production_peft=True)
    args.temperature = .7; args.top_p = 1.; args.top_k = 0
    args.save_steps = 25 if contract["scope"] == "production" else 2
    args.bf16 = dtype == torch.bfloat16; args.fp16 = False
    by_id = {r["instance_id"]: r for r in contract["training_rows"]}
    rows = [{**by_id[i], "answer": by_id[i]["output"], "instance_json": json.dumps(by_id[i]["instance_json"])}
        for draw in contract["training"]["prompt_draws"] for i in draw]
    common = dict(model=actor, args=args, processing_class=tokenizer, train_dataset=Dataset.from_list(rows),
        reference_model=reference, reference_identity=ref_identity, pilot_contract=contract, successor_recipe=recipe,
        recipe_contract={"protocol": PROTOCOL, "pilot_sha256": contract["contract_sha256"], "arm": recipe,
            "operational_version": "4.cuda1", "entrypoint_sha256": identity(Path(__file__))["sha256"],
            "statistical_source_sha256": STATISTICAL_SHA, "checkpoint_helper_sha256": identity(Path(v3.__file__))["sha256"],
            "checkpoint_interval": args.save_steps})
    if settings["method"] == "grpo":
        heads = list(v4.FOARL_F638_REWARD_FUNCS) if settings["reward"] == "released_foarl" else [v4.strict_foarl_optimality, v4.strict_foarl_fraction]
        trainer = v4.SuccessorFOARLTrainer(reward_funcs=heads, **common)
    else:
        trainer = v4.SuccessorRLOOTrainer(reward_funcs=[v4.strict_scalar_reward], advantage_contract=make_advantage_contract("control"), **common)
    trainer._assert_contract()
    need(actor.generation_config.to_dict() == clean_config.to_dict() and
         actor.get_base_model().generation_config.to_dict() == clean_config.to_dict(),
         "Effective actor/base training generation defaults changed")
    trainer.cuda_v4_installed_generation_config = clean_config.to_dict()
    return trainer


def evaluation_config(settings, tokenizer, cap):
    """Construct all effective defaults independently of model-saved filters."""
    return GenerationConfig(max_new_tokens=cap, do_sample=settings["do_sample"], num_return_sequences=settings["count"],
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        temperature=settings.get("temperature", 1.), top_p=settings.get("top_p", 1.), top_k=settings.get("top_k", 0),
        repetition_penalty=1., num_beams=1, num_beam_groups=1, typical_p=1., min_p=None,
        epsilon_cutoff=0., eta_cutoff=0., no_repeat_ngram_size=0, min_length=0, min_new_tokens=None,
        bad_words_ids=None, force_words_ids=None, suppress_tokens=None, begin_suppress_tokens=None,
        forced_bos_token_id=None, forced_eos_token_id=None, constraints=None, penalty_alpha=None)


def collect_outputs(model, tokenizer, contract, output, device):
    """Actual greedy plus separately named full-support and filtered K8 streams."""
    rows = []; model.eval()
    for instance in contract["validation_rows"]:
        encoded = tokenizer(instance["prompt"], return_tensors="pt", truncation=False).to(device)
        prompt_length = encoded.input_ids.shape[1]
        need(prompt_length + contract["evaluation"]["max_completion_tokens"] <= model.config.max_position_embeddings,
             "Full V4 input exceeds context; truncation forbidden")
        for stream, settings in v4.DECODE_STREAMS.items():
            seed = int.from_bytes(hashlib.sha256(f'{contract["eval_seed"]}/{instance["semantic_id"]}/{stream}'.encode()).digest()[:4], "big")
            set_seed(seed); started = time.monotonic()
            config = evaluation_config(settings, tokenizer, contract["evaluation"]["max_completion_tokens"])
            with torch.inference_mode(): generated = model.generate(**encoded, generation_config=config, use_model_defaults=False)
            group = []
            for index, ids in enumerate(generated[:, prompt_length:].cpu().tolist()):
                if tokenizer.eos_token_id in ids: ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
                text = tokenizer.decode(ids, skip_special_tokens=True)
                score = score_completion(contract["task"], text, instance["instance_json"], instance["reference_answer"])
                z = -8. * score["signed_gap"] if score["feasible"] else None
                utility = (1. / (1. + math.exp(-z)) if z >= 0 else math.exp(z) / (1. + math.exp(z))) if z is not None else 0.
                group.append({"instance_id": instance["instance_id"], "semantic_id": instance["semantic_id"], "stream": stream,
                    "sample_index": index, "completion_token_ids": ids, "text": text, "generated_tokens": len(ids), "generation_seed": seed,
                    "runtime_complete": True, "finish_reason": "eos" if ids and ids[-1] == tokenizer.eos_token_id else "length",
                    "strict_score": score, "quality_utility": utility})
            need(len(group) == settings["count"], "V4 raw candidate count differs")
            rows.extend(group)
            v2._append_durable(Path(output) / "successor_evaluation_groups.jsonl", {"protocol": PROTOCOL,
                "stream": stream, "endpoint": settings["endpoint"], "instance_id": instance["instance_id"], "seed": seed,
                "generation_configuration": config.to_dict(), "extra_generation_kwargs": {"use_model_defaults": False},
                "prompt_tokens": prompt_length, "input_truncated": False,
                "raw_completions": group, "generated_tokens": sum(r["generated_tokens"] for r in group), "wall_seconds": time.monotonic() - started})
    return rows


def replay_raw(contract, raw, groups, tokenizer):
    need(len(groups) == 3 * len(contract["validation_rows"]) and len(raw) == 17 * len(contract["validation_rows"]), "V4 raw closure/count differs")
    flattened = []; cursor = 0
    for instance in contract["validation_rows"]:
        for stream, settings in v4.DECODE_STREAMS.items():
            group = groups[cursor]; cursor += 1
            expected_seed = int.from_bytes(hashlib.sha256(f'{contract["eval_seed"]}/{instance["semantic_id"]}/{stream}'.encode()).digest()[:4], "big")
            need(group["protocol"] == PROTOCOL and group["instance_id"] == instance["instance_id"] and group["stream"] == stream and
                 type(group["seed"]) is int and group["seed"] == expected_seed and
                 group["input_truncated"] is False and group["endpoint"] == settings["endpoint"] and
                 len(group["raw_completions"]) == settings["count"], "V4 raw group order/seed/source differs")
            expected_config = evaluation_config(settings, tokenizer, contract["evaluation"]["max_completion_tokens"]).to_dict()
            prompt_tokens = len(tokenizer(instance["prompt"], truncation=False)["input_ids"])
            need(group["generation_configuration"] == expected_config and group["extra_generation_kwargs"] == {"use_model_defaults": False} and
                 type(group["prompt_tokens"]) is int and group["prompt_tokens"] == prompt_tokens and
                 type(group["generated_tokens"]) is int and group["generated_tokens"] > 0 and
                 group["generated_tokens"] == sum(r["generated_tokens"] for r in group["raw_completions"]),
                 "V4 resolved decoder/prompt/token configuration changed")
            for index, row in enumerate(group["raw_completions"]):
                ids = row["completion_token_ids"]
                need(type(ids) is list and all(type(t) is int and 0 <= t < len(tokenizer) for t in ids) and
                     tokenizer.decode(ids, skip_special_tokens=True) == row["text"] and
                     (tokenizer.eos_token_id not in ids or ids.index(tokenizer.eos_token_id) == len(ids) - 1) and
                     row["finish_reason"] == ("eos" if ids and ids[-1] == tokenizer.eos_token_id else "length"),
                     "V4 typed token/text/EOS closure differs")
                score = score_completion(contract["task"], row["text"], instance["instance_json"], instance["reference_answer"])
                z = -8. * score["signed_gap"] if score["feasible"] else None
                u = (1. / (1. + math.exp(-z)) if z >= 0 else math.exp(z) / (1. + math.exp(z))) if z is not None else 0.
                need(row["instance_id"] == instance["instance_id"] and row["semantic_id"] == instance["semantic_id"] and
                     row["stream"] == stream and row["runtime_complete"] is True and
                     type(row["sample_index"]) is int and row["sample_index"] == index and row["strict_score"] == score and row["quality_utility"] == u and
                     type(row["generation_seed"]) is int and row["generation_seed"] == expected_seed and
                     type(row["generated_tokens"]) is int and 0 < row["generated_tokens"] == len(row["completion_token_ids"]) <= contract["evaluation"]["max_completion_tokens"],
                     "V4 independent physical/raw replay differs")
            flattened.extend(group["raw_completions"])
    need(flattened == raw, "V4 durable group and result raw rows differ")
    return {"groups": len(groups), "responses": len(raw), "independent_strict_score_replay": True}


def admit_run_arm(contract, recipe, admission, base, sft, device, output):
    """Repeat authentication for a direct library call; no fixture model calls."""
    if contract.get("scope") == "production":
        anchor = KNOWN_NATIVE_CONTRACTS.get(contract.get("contract_sha256"))
        need(anchor is not None and anchor.get("task") == contract.get("task") and recipe in anchor.get("recipes", []),
             "Unknown native V4 contract cannot train through the library API")
        need(type(admission) is dict and type(admission.get("_admission_replay")) is dict,
             "Production library API requires replay of independently authenticated admission")
        args = admission["_admission_replay"]
        admission = admit_production(contract,recipe,args["registry"],registry_sha256=args["registry_sha256"],base=base,sft=sft,
            device=device,output=output,promotion=args["promotion"],promotion_sha256=args["promotion_sha256"])
    return admission


def run_arm(contract, recipe, output, tokenizer, base, sft, device, *, admission=None):
    admission = admit_run_arm(contract, recipe, admission, base, sft, device, output)
    contract = validate_contract(contract)
    need(recipe in v4.RECIPES or recipe == "sft", "Unknown fixed V4 recipe")
    need(contract["scope"] == "system_smoke" or admission is not None, "Unregistered production cannot train")
    dtype = torch.float32 if contract["scope"] == "system_smoke" else torch.bfloat16
    output = Path(output); trainer = reference = None; main_rank = v3._rank() == 0
    steps = 0 if recipe == "sft" else contract["training"]["horizon"]
    if steps:
        trainer = make_trainer(contract, recipe, output, tokenizer, base, sft, device, dtype=dtype)
        model = trainer.accelerator.unwrap_model(trainer.model)
    else:
        v2._collective_phase(lambda: output.mkdir(parents=True, exist_ok=False) if main_rank else None, "new V4 SFT output")
        model = v2.loaded_model(base, sft, device, trainable=False, dtype=dtype)
    initial_sha = semantic_state_sha256(model.state_dict()); env = environment(model)
    if admission is not None:
        need(env["parameter_count"] >= 6_000_000_000 and
             env == admission["expected_rank_environments"][str(v3._rank())],
             "Actual native per-rank environment/dtypes/model do not match registered witness")
    started = time.monotonic()
    if steps:
        trainer.recovery_started = started; trainer.train()
        need(trainer.state.global_step == steps == len(trainer.actual_windows), "V4 fixed terminal/window count differs")
        trainer._assert_reference_integrity(full=True)
        model = trainer.accelerator.unwrap_model(trainer.model)
    seconds = time.monotonic() - started
    def rank_proof():
        actor_sha = semantic_state_sha256(model.state_dict())
        need(not steps or actor_sha != initial_sha, "Actual V4 optimizer steps did not change actor")
        return {"rank": v3._rank(), "optimizer_steps": steps, "actor_state_sha256": actor_sha,
            "initial_actor_state_sha256": initial_sha, "actor_changed": actor_sha != initial_sha,
            "optimizer_state_sha256": semantic_state_sha256(trainer.optimizer.state_dict()) if trainer else None,
            "scheduler_state_sha256": semantic_state_sha256(trainer.lr_scheduler.state_dict()) if trainer else None,
            "reference_state_sha256": trainer.reference_state_sha256() if trainer else None,
            "reference_full_integrity_verified": bool(trainer and trainer.beta),
            "initial_reference_audit": trainer.initial_reference_audit if trainer else None,
            "prepared_reference_audit": trainer.prepared_reference_audit if trainer else None,
            "effective_checkpointing": trainer.last_checkpointing_audit if trainer else None,
            "checkpointing_stage_counts": trainer.checkpointing_stage_counts if trainer else {},
            "installed_actor_generation_config": trainer.cuda_v4_installed_generation_config if trainer else None,
            "effective_trainer_generation_config": trainer.generation_config.to_dict() if trainer else None,
            "environment": env, "generated_tokens": trainer.actual_generated_tokens if trainer else 0}
    local = v2._collective_phase(rank_proof, "V4 all-rank terminal integrity"); proofs = v3._gather(local)
    for key in ("optimizer_steps", "actor_state_sha256", "optimizer_state_sha256", "scheduler_state_sha256", "reference_state_sha256"):
        need(all(p[key] == proofs[0][key] for p in proofs), "V4 all-rank terminal " + key + " differs")
    terminal = output / "terminal"
    def seal():
        if not main_rank: return None
        model.save_pretrained(terminal); tokenizer.save_pretrained(terminal)
        proof = {"protocol": PROTOCOL + "-training-complete", "scope": contract["scope"], "recipe": recipe,
            "source": identity(Path(__file__)), "statistical_source_sha256": STATISTICAL_SHA,
            "contract_sha256": contract["contract_sha256"], "fixed_terminal_step": steps,
            "RL_responses": steps * 36, "training_wall_seconds": seconds,
            "distributed": {"world_size": len(proofs), "rank_closure": proofs}, "terminal_artifact_tree": v2.artifact_tree(terminal),
            "actual_prompt_windows": identity(output / "training/actual_prompt_windows.jsonl") if steps else None,
            "strict_coefficients": identity(output / "training/successor_coefficients.jsonl") if steps else None,
            "superiority_established": False}
        write(output / "training_complete.json", proof); return proof
    v2._collective_phase(seal, "V4 rank0 durable terminal save")
    del model, trainer, reference; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier(); torch.distributed.destroy_process_group()
    result = None
    if main_rank:
        need(not torch.distributed.is_initialized(), "Rank group must close before rank0 serial evaluation")
        loaded = v2.loaded_model(base, terminal, device, trainable=False, dtype=dtype)
        need(semantic_state_sha256(loaded.state_dict()) == proofs[0]["actor_state_sha256"], "V4 bitwise whole-actor terminal reload differs")
        write(output / "checkpoint_reload_exact.json", {"protocol": PROTOCOL + "-reload", "bitwise_equal": True,
            "actor_state_sha256": proofs[0]["actor_state_sha256"], "training_complete": identity(output / "training_complete.json"),
            "process_group_destroyed_before_evaluation": True, "environment": environment(loaded)})
        raw = collect_outputs(loaded, tokenizer, contract, output, device)
        groups = [json.loads(line) for line in (output / "successor_evaluation_groups.jsonl").read_text().splitlines()]
        closure = replay_raw(contract, raw, groups, tokenizer)
        result = {"protocol": PROTOCOL, "scope": contract["scope"], "recipe": recipe, "trained_updates": steps,
            "RL_responses": steps * 36, "evaluation_responses": len(raw), "raw_completions": raw,
            "raw_closure": closure, "primary_endpoint": "k8_full_support", "secondary_endpoint": "k8_filtered_secondary",
            "actual_greedy": True, "checkpoint_reload_exact": True, "superiority_established": False,
            "production_admission": admission is not None, "native_registration_created": False,
            "training_complete": identity(output / "training_complete.json"), "reload": identity(output / "checkpoint_reload_exact.json"),
            "group_file": identity(output / "successor_evaluation_groups.jsonl")}
        write(output / "result.json", result); del loaded; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true"); p.add_argument("--task", choices=("op", "cvrp", "both"), default="both")
    p.add_argument("--recipe", choices=(*v4.RECIPES, "sft", "all"), default="all")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu"); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contract", type=Path); p.add_argument("--contract-sha256"); p.add_argument("--base", type=Path); p.add_argument("--sft", type=Path)
    p.add_argument("--registry", type=Path); p.add_argument("--registry-sha256")
    p.add_argument("--promotion",type=Path);p.add_argument("--promotion-sha256")
    a = p.parse_args(); torch.set_num_threads(1)
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0"))
    need(world in (1, 2, 4), "V4 rank world must be1,2,4")
    need(world == 1 or (a.task != "both" and a.recipe != "all"), "Distributed V4 must select one task/recipe before teardown")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if a.device == "cuda" else torch.device("cpu")
    if device.type == "cuda": need(torch.cuda.is_available(), "CUDA unavailable"); torch.cuda.set_device(device)
    admission = None
    if not a.smoke:
        need(a.task != "both" and a.recipe != "all" and all((a.contract, a.contract_sha256, a.base, a.sft, a.registry, a.registry_sha256,a.promotion,a.promotion_sha256)),
             "Production requires one fixed task/recipe and externally registered contract/source/initializer/authority")
        raw = a.contract.read_bytes(); need(hashlib.sha256(raw).hexdigest() == a.contract_sha256, "Contract byte identity differs")
        contract = validate_contract(json.loads(raw))
        need(world == (1 if a.recipe == "sft" else 4), "Production trained recipes require four ranks; SFT one")
        admission = admit_production(contract, a.recipe, a.registry, registry_sha256=a.registry_sha256, base=a.base, sft=a.sft,
            device=device,output=a.output/a.task/a.recipe,promotion=a.promotion,promotion_sha256=a.promotion_sha256)
    if world > 1: torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    if rank == 0: a.output.mkdir(parents=True, exist_ok=False)
    if torch.distributed.is_initialized(): torch.distributed.barrier()
    results = []
    for task in (("op", "cvrp") if a.task == "both" else (a.task,)):
        base, sft = a.base, a.sft
        if a.smoke:
            base, sft = a.output / task / "fixture_base", a.output / task / "fixture_sft"
            if rank == 0:
                model, tokenizer, train, validation = v2.tiny_peft_initializer(task, base, sft, torch.device("cpu"))
                del model; gc.collect()
                contract = v4.make_smoke_contract(task, {"id": "tiny-V4-CUDA-DDP-system-smoke", "sha256": identity(sft / "adapter_model.safetensors")["sha256"]}, train, validation)
                write(a.output / task / "contract.json", contract)
            if torch.distributed.is_initialized(): torch.distributed.barrier()
            contract = validate_contract(json.loads((a.output / task / "contract.json").read_text()))
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True); tokenizer.padding_side = "left"
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
        if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
        for recipe in ((*v4.RECIPES, "sft") if a.recipe == "all" else (a.recipe,)):
            result = run_arm(contract, recipe, a.output / task / recipe, tokenizer, base, sft, device, admission=admission)
            if rank == 0: results.append(result)
    if rank == 0:
        summary = {"protocol": PROTOCOL, "status": "completed_system_smoke" if a.smoke else "completed_registered_model_execution",
            "scope": "system_smoke" if a.smoke else "production", "source": identity(Path(__file__)),
            "trained_updates": sum(r["trained_updates"] for r in results), "RL_responses": sum(r["RL_responses"] for r in results),
            "evaluation_responses": sum(r["evaluation_responses"] for r in results), "runs": len(results),
            "world_size_at_launch": world, "requested_device": a.device, "native_registration_created": False,
            "production_admission": admission is not None, "superiority_established": False}
        write(a.output / "complete.json", summary); print(json.dumps(summary))


if __name__ == "__main__": main()
