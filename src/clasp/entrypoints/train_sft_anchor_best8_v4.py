"""Isolated, unregistered successor integration; tiny CPU PEFT execution only.

The current pilot and confirmation sources stay unchanged. Full-support
tempered sampling, strict scalar rewards and best8 credit are a new statistical
recipe, not an operational recovery of v2. Production remains unavailable until
pilot eligibility, fresh cohorts, native validation and bounded authority close.
The unchanged v3 mixin supplies periodic checkpoint/RNG/prefix mechanics only;
the byte-bound native checkpoint contract separately identifies this v4 driver.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time

import torch
from datasets import Dataset
from transformers import AutoTokenizer, set_seed
from trl import GRPOTrainer

from clasp.baselines.budget_matched_dev import reward_environment
from clasp.baselines.llmcosolver_f638_training import FOARL_F638_REWARD_FUNCS
from clasp.baselines.sft_anchor_pilot import digest, make_contract, score_completion
from clasp.entrypoints import train_sft_anchor_pilot as v1
from clasp.entrypoints import train_sft_anchor_pilot_v2 as v2
from clasp.entrypoints import train_sft_anchor_recovery_v3 as v3
from clasp.models.external_baseline_state import semantic_state_sha256
from clasp.rewards import best8_from9, strict_raw_training
from clasp.rewards.feasible_quality import decompose_advantages, make_advantage_contract
from clasp.trainers.sft_anchored_rloo import SFTAnchoredRLOOTrainer, SFTReferenceGRPOTrainer

PROTOCOL = "unregistered-strict-best8-successor-v4"
RECIPES = {
    "foarl_released": {"method": "grpo", "reward": "released_foarl", "beta": .05, "best8_eta": 0., "cardinal_lambda": 0.},
    "foarl_strict_aligned": {"method": "grpo", "reward": "strict_foarl", "beta": .05, "best8_eta": 0., "cardinal_lambda": 0.},
    "rloo_strict_scalar": {"method": "rloo", "reward": "strict_scalar", "beta": .05, "best8_eta": 0., "cardinal_lambda": 0.},
    "rloo_strict_best8": {"method": "rloo", "reward": "strict_scalar", "beta": .05, "best8_eta": .25, "cardinal_lambda": 0.},
    "rloo_strict_cardinal": {"method": "rloo", "reward": "strict_scalar", "beta": .05, "best8_eta": 0., "cardinal_lambda": .25},
    "rloo_strict_best8_beta0": {"method": "rloo", "reward": "strict_scalar", "beta": 0., "best8_eta": .25, "cardinal_lambda": 0.},
}
DECODE_STREAMS = {
    "greedy": {"count": 1, "do_sample": False, "endpoint": "actual_greedy"},
    "k8_full_support": {"count": 8, "do_sample": True, "temperature": .7, "top_p": 1., "top_k": 0, "endpoint": "primary"},
    "k8_filtered_secondary": {"count": 8, "do_sample": True, "temperature": .7, "top_p": .9, "top_k": 50, "endpoint": "secondary_continuity"},
}
file_identity = v2.file_identity
write_json_atomic = v2.write_json_atomic


def prospective_template():
    """Declaration only: no cohorts, policy draws, launch or registration."""
    return {"protocol": PROTOCOL, "status": "unregistered_source_only", "recipes": copy.deepcopy(RECIPES),
            "tasks": ["op", "cvrp"], "training_rows_per_task": 128, "validation_rows_per_task": 96,
            "horizon": 100, "terminal_selection": 100, "global_responses_per_update": 36,
            "groups_per_update": 4, "siblings_per_group": 9, "learning_rate": 1e-5,
            "max_completion_tokens": 3000, "temperature": .7, "top_p": 1., "top_k": 0,
            "checkpoint_steps": [25, 50, 75, 100], "evaluation_streams": copy.deepcopy(DECODE_STREAMS),
            "frozen_sft_evaluation_only": True, "actual_fresh_cohorts_created": False,
            "production_registration_ready": False, "production_launch_enabled": False, "superiority_established": False,
            "preconditions": ["current pilot completed with no eligible shared arm", "fresh certified tuning cohorts",
                              "new frozen selection/registration", "native CUDA/7B validation", "new bounded resource authority"]}


def make_smoke_contract(task, initializer, train, validation, *, horizon=4, max_completion_tokens=1, train_seed=61):
    """Use only synthetic smoke inputs; no production dataset admission API."""
    base = make_contract(task, initializer, train, validation, scope="system_smoke", horizon=horizon,
                         max_completion_tokens=max_completion_tokens, train_seed=train_seed)
    base.update(protocol=PROTOCOL, schema_version=4, recipes=copy.deepcopy(RECIPES),
                evaluation_streams=copy.deepcopy(DECODE_STREAMS), registration_status="unregistered_system_smoke")
    base.pop("arms")
    base["training"].update(top_p=1., top_k=0)
    base["evaluation"] = {"max_completion_tokens": max_completion_tokens, "repair": False,
                          "primary": "k8_full_support", "secondary": "k8_filtered_secondary", "actual_greedy": True}
    base["contract_sha256"] = digest({key: value for key, value in base.items() if key != "contract_sha256"})
    return base


def validate_smoke_contract(contract):
    v3.need(type(contract) is dict and contract.get("protocol") == PROTOCOL and contract.get("scope") == "system_smoke",
            "Successor is unregistered; only explicit synthetic system_smoke is executable")
    expected = make_smoke_contract(contract["task"], contract["initializer"], contract["training_rows"], contract["validation_rows"],
        horizon=contract["training"]["horizon"], max_completion_tokens=contract["training"]["max_completion_tokens"], train_seed=contract["train_seed"])
    v3.need(contract == expected, "Successor contract changed")
    return copy.deepcopy(expected)


def _rows_from_kwargs(task, prompts, kwargs):
    count = len(prompts)
    required = ("task_type", "answer", "instance_json", "input")
    v3.need(all(isinstance(kwargs.get(key), (list, tuple)) and len(kwargs[key]) == count for key in required),
            "Strict reward fields are not completion aligned")
    return [{**{key: kwargs[key][index] for key in required}, "prompt": prompts[index]} for index in range(count)]


def strict_scalar_reward(prompts, completions, **kwargs):
    tasks = kwargs.get("task_type", [])
    v3.need(bool(tasks) and len(set(tasks)) == 1, "Strict scalar reward requires one task")
    return strict_raw_training.strict_rloo_reward_batch(tasks[0], prompts, completions,
        _rows_from_kwargs(tasks[0], prompts, kwargs))["rewards"]


def strict_foarl_optimality(prompts, completions, **kwargs):
    tasks = kwargs.get("task_type", [])
    v3.need(bool(tasks) and len(set(tasks)) == 1, "Strict FOARL reward requires one task")
    return strict_raw_training.strict_aligned_foarl_reward_batch(tasks[0], prompts, completions,
        _rows_from_kwargs(tasks[0], prompts, kwargs))["optimality"]


def strict_foarl_fraction(prompts, completions, **kwargs):
    tasks = kwargs.get("task_type", [])
    v3.need(bool(tasks) and len(set(tasks)) == 1, "Strict FOARL reward requires one task")
    return strict_raw_training.strict_aligned_foarl_reward_batch(tasks[0], prompts, completions,
        _rows_from_kwargs(tasks[0], prompts, kwargs))["feasibility_fraction"]


class SuccessorCreditMixin:
    """Shared ordered strict metadata; released FOARL rewards stay unmasked."""
    def __init__(self, *args, successor_recipe, **kwargs):
        v3.need(successor_recipe in RECIPES, "Unknown successor recipe")
        self.successor_recipe = successor_recipe
        self.successor_settings = copy.deepcopy(RECIPES[successor_recipe])
        self.last_coefficient_record = None
        self._strict_metadata_buffer = None
        super().__init__(*args, **kwargs)

    def _recipe(self):
        contract = super()._recipe()
        contract.update(protocol=PROTOCOL, successor_recipe=self.successor_recipe,
            successor_settings=copy.deepcopy(self.successor_settings), statistical_source=file_identity(Path(__file__)),
            strict_reward_source=file_identity(Path(strict_raw_training.__file__)), best8_source=file_identity(Path(best8_from9.__file__)),
            training_sampling={"temperature": .7, "top_p": 1., "top_k": 0},
            generation_configuration=copy.deepcopy(self.generation_config.to_dict()),
            score_source_sha256=hashlib.sha256(inspect.getsource(GRPOTrainer._get_per_token_logps_and_entropies).encode()).hexdigest())
        if "advantage_contract" in contract:
            contract["inherited_initialization_control_contract"] = contract.pop("advantage_contract")
            contract["quality_weight"] = self.successor_settings["best8_eta"] + self.successor_settings["cardinal_lambda"]
            contract["reward_contract"] = {"profile": "strict_physical_calibrated_scalar_v4", "invalid_quality": 0.}
        return contract

    def _assert_contract(self):
        super()._assert_contract()
        v3.need(self.successor_settings == RECIPES[self.successor_recipe], "Successor coefficient settings changed")
        v3.need(self.args.temperature == self.temperature == .7 and self.args.top_p == 1. and self.args.top_k == 0,
                "Successor requires matched full-support tempered training")
        v3.need(self.generation_config.temperature == .7 and self.generation_config.top_p == 1. and self.generation_config.top_k == 0,
                "Actual generation configuration differs from full-support sampler")
        v3.need(self.generation_config.repetition_penalty == 1. and self.generation_config.no_repeat_ngram_size == 0 and
                self.generation_config.typical_p == 1. and self.generation_config.min_p is None and
                self.generation_config.epsilon_cutoff == self.generation_config.eta_cutoff == 0. and
                self.generation_config.min_length == 0 and self.generation_config.min_new_tokens is None and
                self.generation_config.num_beams == self.generation_config.num_beam_groups == 1 and
                all(getattr(self.generation_config, name, None) is None for name in
                    ("bad_words_ids", "force_words_ids", "suppress_tokens", "begin_suppress_tokens", "forced_bos_token_id", "forced_eos_token_id", "penalty_alpha", "constraints", "top_h")),
                "Full-support sampling forbids additional token penalties or support constraints")
        v3.need(self.beta == self.successor_settings["beta"], "Successor frozen-reference beta changed")

    def _input_metadata(self, inputs):
        def local():
            v3.need(type(inputs) is list and bool(inputs) and all(type(row) is dict and row.get("task_type") == self.pilot_contract["task"] for row in inputs),
                    "Successor input task/row metadata differs")
            return [digest([row.get("task_type"), row.get("instance_json"), row.get("input"), row.get("prompt")]) for row in inputs]
        own = self._collective_check(local, "strict prompt identities")
        parts = self._gather_objects(own)
        v3.need(all(len(part) == len(own) for part in parts), "Successor rank rollout lengths differ")
        keys = [value for part in parts for value in part]
        v3.need(len(keys) == 36 and all(len(set(keys[start:start + 9])) == 1 for start in range(0, 36, 9)),
                "Successor requires exactly four complete same-prompt B9 groups")
        return keys

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        def verify():
            task = self.pilot_contract["task"]
            if self.successor_settings["reward"] == "strict_scalar":
                result = strict_raw_training.strict_rloo_reward_batch(task, prompts, completions, inputs)
                return result["metadata"], [[value] for value in result["rewards"]]
            metadata = strict_raw_training.strict_reward_metadata(task, prompts, completions, inputs)
            if self.successor_settings["reward"] == "strict_foarl":
                result = strict_raw_training.strict_aligned_foarl_reward_batch(task, prompts, completions, inputs)
                return metadata, [list(values) for values in zip(result["optimality"], result["feasibility_fraction"])]
            return metadata, None
        local, expected_local = self._collective_check(verify, "strict raw reward metadata")
        parts = self._gather_objects(local)
        metadata = [row for part in parts for row in part]
        self._collective_check(lambda: best8_from9.strict_best8_from9_coefficients(metadata), "global strict utility/prompt/task admission")
        v3.need([row["prompt_key"] for row in metadata] == self._prompt_rows, "Strict reward metadata lost native response order")
        v3.need(all(len({row["reference"] for row in metadata[start:start + 9]}) == 1 for start in range(0, 36, 9)),
                "Sibling strict references differ")
        rewards = GRPOTrainer._calculate_rewards(self, inputs, prompts, completions, completion_ids_list)
        expected_heads = 1 if self.successor_settings["method"] == "rloo" else 2
        v3.need(tuple(rewards.shape) == (36, expected_heads) and bool(torch.isfinite(rewards).all()),
                "Successor gathered rewards are missing or nonfinite")
        if expected_local is not None:
            expected = [row for part in self._gather_objects(expected_local) for row in part]
            v3.need(torch.equal(rewards, torch.tensor(expected, dtype=rewards.dtype, device=rewards.device)),
                    "Native reward head and independently verified strict reward disagree")
        self._reward_buffer = rewards.detach().clone()
        self._strict_metadata_buffer = metadata
        return rewards

    def _generate_and_score_completions(self, inputs):
        self._collective_check(self._assert_contract, "successor active recipe")
        if self.beta and not self._reference_prepared_checked:
            if not self._resuming:
                self.prepared_reference_audit = self._collective_check(self._audit_initial_pair, "successor prepared original SFT equality")
            self._reference_prepared_checked = True
        self._prompt_rows = self._input_metadata(inputs)
        self._reward_buffer = self._strict_metadata_buffer = None
        self._generation_active = True
        try:
            output = GRPOTrainer._generate_and_score_completions(self, inputs)
            rewards, metadata = self._reward_buffer, self._strict_metadata_buffer
            v3.need(rewards is not None and metadata is not None, "Missing successor reward/metadata buffer")
            best8 = best8_from9.strict_best8_from9_coefficients(metadata)
            auxiliary = torch.tensor(best8["coefficients"], dtype=rewards.dtype, device=rewards.device)
            if self.successor_settings["method"] == "rloo":
                grouped = rewards[:, 0].view(-1, 9)
                base = (grouped - (grouped.sum(1, keepdim=True) - grouped) / 8).reshape(-1).detach()
                cardinal = decompose_advantages(base.cpu().tolist(), metadata, make_advantage_contract("cardinal"))["quality"]
                cardinal = torch.tensor(cardinal, dtype=base.dtype, device=base.device)
                combined = base + self.successor_settings["best8_eta"] * auxiliary + self.successor_settings["cardinal_lambda"] * cardinal
            else:
                scalar = (rewards * self.reward_weights.to(rewards.device).unsqueeze(0)).sum(1)
                grouped = scalar.view(-1, 9)
                base = ((grouped - grouped.mean(1, keepdim=True)) / (grouped.std(1, keepdim=True) + 1e-4)).reshape(-1).detach()
                combined, cardinal = base, torch.zeros_like(base)
            v3.need(not combined.requires_grad and bool(torch.isfinite(combined).all()), "Successor coefficients must be finite and detached")
            self._consensus(combined.cpu().tolist(), "successor globally ordered coefficients")
            start = int(self.accelerator.process_index) * len(inputs)
            if self.successor_settings["method"] == "grpo":
                v3.need(torch.equal(output["advantages"], combined[start:start + len(inputs)]), "Stock GRPO group normalization changed")
            output["advantages"] = combined[start:start + len(inputs)].detach()
            recorded, replacement = self._logs["advantages"], combined.cpu().tolist()
            for _ in range(min(len(recorded), len(replacement))):
                recorded.pop()
            recorded.extend(replacement)
            record = {"protocol": PROTOCOL, "recipe": self.successor_recipe, "global_step_before_update": int(self.state.global_step),
                "beta": self.beta, "best8_eta": self.successor_settings["best8_eta"], "cardinal_lambda": self.successor_settings["cardinal_lambda"],
                "base": base.cpu().tolist(), "best8": auxiliary.cpu().tolist(), "cardinal": cardinal.cpu().tolist(),
                "combined": combined.cpu().tolist(), "strict_metadata": metadata, "gathered_reward_heads": rewards.cpu().tolist(),
                "best8_active_group_fraction": sum(group["active"] for group in best8["groups"]) / len(best8["groups"]),
                "sampling": {"temperature": .7, "top_p": 1., "top_k": 0}, "source": file_identity(Path(__file__))}
            self.last_coefficient_record = record
            if self.accelerator.is_main_process:
                v2._append_durable(Path(self.args.output_dir) / "successor_coefficients.jsonl", record)
            return output
        finally:
            self._generation_active = False
            self._reward_buffer = self._strict_metadata_buffer = self._prompt_rows = None


class SuccessorRLOOCore(SuccessorCreditMixin, SFTAnchoredRLOOTrainer):
    pass


class SuccessorFOARLCore(SuccessorCreditMixin, SFTReferenceGRPOTrainer):
    pass


class SuccessorRLOOTrainer(v3.RecoveryStateMixin, v2.EffectiveCheckpointingMixin, v1.OrderedObservationMixin, SuccessorRLOOCore):
    pass


class SuccessorFOARLTrainer(v3.RecoveryStateMixin, v2.EffectiveCheckpointingMixin, v1.OrderedObservationMixin, SuccessorFOARLCore):
    pass


def make_successor_trainer(contract, recipe, output, tokenizer, base, sft, *, checkpoint_interval=2, resume_from=None, stop_after=None):
    """Actual tiny CPU PEFT branch; real-model loading/launch deliberately absent."""
    contract = validate_smoke_contract(contract)
    v3.need(recipe in RECIPES, "Unknown successor recipe")
    v3.need(type(checkpoint_interval) is int and checkpoint_interval > 0 and
            contract["training"]["horizon"] % checkpoint_interval == 0,
            "Smoke checkpoint interval must divide the fixed terminal horizon")
    if stop_after is not None:
        v3.need(type(stop_after) is int and 0 < stop_after < contract["training"]["horizon"] and
                stop_after % checkpoint_interval == 0, "Smoke interruption must use an earlier checkpoint boundary")
    settings = RECIPES[recipe]; output = Path(output)
    v2._collective_phase(lambda: output.mkdir(parents=True, exist_ok=False) if v3._rank() == 0 else None, "new successor output")
    set_seed(contract["train_seed"])
    os.environ.update(reward_environment("foarl" if settings["method"] == "grpo" else "rloo_quality"))
    actor = v2.loaded_model(base, sft, torch.device("cpu"), trainable=True, dtype=torch.float32)
    reference = v2.loaded_model(base, sft, torch.device("cpu"), trainable=False, dtype=torch.float32) if settings["beta"] else None
    identity = ({"source_kind": "original_sft", "source_sha256": contract["initializer"]["sha256"], "immutable": True, "scope": contract["scope"]}
                if reference is not None else None)
    v3.need(file_identity(Path(sft) / "adapter_model.safetensors")["sha256"] == contract["initializer"]["sha256"], "Original SFT adapter changed")
    alias = "foarl" if settings["method"] == "grpo" else "rloo_scalar_beta005" if settings["beta"] else "rloo_scalar_beta0"
    args = v2._training_args(contract, alias, output, torch.device("cpu"), production_peft=True)
    args.temperature = .7; args.top_p = 1.; args.top_k = 0; args.save_steps = checkpoint_interval
    by_id = {row["instance_id"]: row for row in contract["training_rows"]}
    rows = [{**by_id[identifier], "answer": by_id[identifier]["output"], "instance_json": json.dumps(by_id[identifier]["instance_json"])}
            for draw in contract["training"]["prompt_draws"] for identifier in draw]
    common = dict(model=actor, args=args, processing_class=tokenizer, train_dataset=Dataset.from_list(rows),
        reference_model=reference, reference_identity=identity, pilot_contract=contract, successor_recipe=recipe,
        recovery_checkpoint=resume_from, recovery_stop_after=stop_after,
        recipe_contract={"protocol": PROTOCOL, "pilot_sha256": contract["contract_sha256"], "arm": recipe,
                         "operational_version": 4, "entrypoint_sha256": file_identity(Path(__file__))["sha256"],
                         "checkpoint_helper_sha256": file_identity(Path(v3.__file__))["sha256"], "checkpoint_interval": checkpoint_interval})
    if settings["method"] == "grpo":
        heads = list(FOARL_F638_REWARD_FUNCS) if settings["reward"] == "released_foarl" else [strict_foarl_optimality, strict_foarl_fraction]
        return SuccessorFOARLTrainer(reward_funcs=heads, **common)
    return SuccessorRLOOTrainer(reward_funcs=[strict_scalar_reward], advantage_contract=make_advantage_contract("control"), **common)


def collect_successor_outputs(model, tokenizer, contract, output):
    """Preserve three fixed named streams; no decoder selection after results."""
    rows = []; model.eval(); device = torch.device("cpu")
    for instance in contract["validation_rows"]:
        encoded = tokenizer(instance["prompt"], return_tensors="pt", truncation=False).to(device)
        v3.need(encoded.input_ids.shape[1] + contract["evaluation"]["max_completion_tokens"] <= model.config.max_position_embeddings,
                "Full successor input exceeds context; no truncation")
        for stream, settings in DECODE_STREAMS.items():
            seed = int.from_bytes(hashlib.sha256(f'{contract["eval_seed"]}/{instance["semantic_id"]}/{stream}'.encode()).digest()[:4], "big")
            set_seed(seed); started = time.monotonic()
            kwargs = {"max_new_tokens": contract["evaluation"]["max_completion_tokens"], "do_sample": settings["do_sample"],
                      "num_return_sequences": settings["count"], "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
            if settings["do_sample"]:
                kwargs.update({key: settings[key] for key in ("temperature", "top_p", "top_k")})
            with torch.inference_mode():
                generated = model.generate(**encoded, **kwargs)
            group = []
            for index, ids in enumerate(generated[:, encoded.input_ids.shape[1]:].cpu().tolist()):
                if tokenizer.eos_token_id in ids:
                    ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
                text = tokenizer.decode(ids, skip_special_tokens=True)
                score = score_completion(contract["task"], text, instance["instance_json"], instance["reference_answer"])
                gap = score["signed_gap"]; utility = 0.
                if score["feasible"]:
                    z = -8. * gap; utility = 1. / (1. + math.exp(-z)) if z >= 0 else math.exp(z) / (1. + math.exp(z))
                group.append({"instance_id": instance["instance_id"], "semantic_id": instance["semantic_id"], "stream": stream,
                    "sample_index": index, "completion_token_ids": ids, "text": text, "generated_tokens": len(ids), "generation_seed": seed,
                    "runtime_complete": True, "strict_score": score, "quality_utility": utility})
            rows.extend(group)
            v2._append_durable(Path(output) / "successor_evaluation_groups.jsonl", {"protocol": PROTOCOL, "stream": stream,
                "endpoint": settings["endpoint"], "instance_id": instance["instance_id"], "seed": seed, "raw_completions": group,
                "generated_tokens": sum(row["generated_tokens"] for row in group), "wall_seconds": time.monotonic() - started})
    return rows


def run_successor_arm(contract, recipe, output, tokenizer, base, sft, *, checkpoint_interval=2):
    """Train/evaluate the fixed tiny terminal, keeping original whole actor proof."""
    v3.need(v3._world() == 1 and int(os.environ.get("WORLD_SIZE", "1")) == 1,
            "Serial successor train/reload/evaluation supports one rank only; distributed trainer API needs separate rank closure")
    contract = validate_smoke_contract(contract)
    v3.need(recipe in RECIPES or recipe == "sft", "Unknown successor recipe")
    output = Path(output); trainer = None; steps = 0; tokens = 0
    if recipe == "sft":
        output.mkdir(parents=True, exist_ok=False)
        model = v2.loaded_model(base, sft, torch.device("cpu"), trainable=False, dtype=torch.float32)
        reference_sha = None; initial_actor_sha = semantic_state_sha256(model.state_dict())
    else:
        trainer = make_successor_trainer(contract, recipe, output, tokenizer, base, sft, checkpoint_interval=checkpoint_interval)
        initial_actor_sha = semantic_state_sha256(trainer.accelerator.unwrap_model(trainer.model).state_dict())
        trainer.recovery_started = time.monotonic(); trainer.train()
        steps, tokens = trainer.state.global_step, trainer.actual_generated_tokens
        v3.need(steps == contract["training"]["horizon"] == len(trainer.actual_windows), "Successor terminal update/window count differs")
        trainer._assert_reference_integrity(full=True)
        model = trainer.accelerator.unwrap_model(trainer.model); reference_sha = trainer.reference_state_sha256()
    actor_sha = semantic_state_sha256(model.state_dict()); terminal = output / "terminal"
    model.save_pretrained(terminal); tokenizer.save_pretrained(terminal)
    proof = {"protocol": PROTOCOL, "scope": "system_smoke", "recipe": recipe, "source": file_identity(Path(__file__)),
             "steps": steps, "responses": steps * 36, "generated_tokens": tokens, "whole_actor_sha256": actor_sha,
             "initial_whole_actor_sha256": initial_actor_sha, "actor_changed": actor_sha != initial_actor_sha,
             "reference_sha256": reference_sha, "reference_frozen": bool(trainer and trainer.beta),
             "reference_full_integrity_verified": bool(trainer and trainer.beta),
             "initial_reference_audit": trainer.initial_reference_audit if trainer else None,
             "prepared_reference_audit": trainer.prepared_reference_audit if trainer else None,
             "effective_checkpointing": trainer.last_checkpointing_audit if trainer else None,
             "terminal_artifact_tree": v2.artifact_tree(terminal), "statistical_contract_sha256": contract["contract_sha256"],
             "checkpoint_helper": file_identity(Path(v3.__file__)), "superiority_established": False}
    write_json_atomic(output / "successor_training_complete.json", proof)
    del model, trainer; gc.collect()
    reloaded = v2.loaded_model(base, terminal, torch.device("cpu"), trainable=False, dtype=torch.float32)
    v3.need(semantic_state_sha256(reloaded.state_dict()) == actor_sha, "Successor saved/reloaded whole actor differs")
    write_json_atomic(output / "successor_checkpoint_reload.json", {"bitwise_equal": True, "whole_actor_sha256": actor_sha,
        "training_proof_sha256": file_identity(output / "successor_training_complete.json")["sha256"]})
    raw = collect_successor_outputs(reloaded, tokenizer, contract, output)
    report = {"protocol": PROTOCOL, "scope": "system_smoke", "recipe": recipe, "trained_updates": steps,
              "RL_responses": steps * 36, "evaluation_responses": len(raw), "primary_endpoint": "k8_full_support",
              "secondary_endpoint": "k8_filtered_secondary", "actual_greedy": True, "checkpoint_reload_exact": True,
              "raw_completions": raw, "superiority_established": False}
    write_json_atomic(output / "result.json", report)
    del reloaded; gc.collect()
    return report


def optional_classic_comparator_api(task):
    """Supplementary CVRP API, not an extra recipe or executable400 runtime."""
    from clasp.baselines import classic_grpo_400_dev as classic
    v3.need(task == "cvrp", "Released classic400 reward API has no OP implementation")
    return {"label": "optional Classic GRPO CVRP matched successor extension", "reward_callable": classic.classic_cvrp_strict_objective_reward,
            "original_profile": classic.PROFILE_ID, "source": file_identity(Path(classic.__file__)),
            "original_config": copy.deepcopy(classic.CONFIG), "reward_environment": copy.deepcopy(classic.REWARD_ENVIRONMENT),
            "matched_template": prospective_template(), "appended_to_six_recipes": False, "resource_protocol_extension_required": True,
            "matched_beta": 0., "stock_group_normalization": "unbiased_sample_std_plus_1e-4", "stock_loss": "response_token_mean_grpo",
            "additional_adapter_required": "A separately labelled beta0 stock GRPO trainer; the current explicit-SFT FOARL class fixes beta.05 and cannot serve this control.",
            "original_runtime_compatible": False, "reason": "Original runtime pins400 updates/atleast1600 prompts/cadence3; matched successor horizon100 requires a separate adapter and registration."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--task", choices=("op", "cvrp"), required=True)
    parser.add_argument("--recipe", choices=(*RECIPES, "sft", "all"), default="all"); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v3.need(args.smoke and int(os.environ.get("WORLD_SIZE", "1")) == 1,
            "V4 is unregistered; CLI currently supports tiny CPU smoke only, not production or DDP launches")
    torch.set_num_threads(1); args.output.mkdir(parents=True, exist_ok=False)
    base, sft = args.output / "fixture_base", args.output / "fixture_sft"
    model, tokenizer, train, validation = v2.tiny_peft_initializer(args.task, base, sft, torch.device("cpu"))
    del model; gc.collect()
    contract = make_smoke_contract(args.task, {"id": "tiny-v4-PEFT-integration-fixture", "sha256": file_identity(sft / "adapter_model.safetensors")["sha256"]}, train, validation)
    write_json_atomic(args.output / "contract.json", contract)
    recipes = (*RECIPES, "sft") if args.recipe == "all" else (args.recipe,)
    reports = [run_successor_arm(contract, recipe, args.output / recipe, tokenizer, base, sft) for recipe in recipes]
    summary = {"status": "completed_unregistered_tiny_successor_smoke", "source": file_identity(Path(__file__)),
               "protocol": PROTOCOL, "scope": "system_smoke", "recipes": list(recipes),
               "trained_updates": sum(row["trained_updates"] for row in reports), "RL_responses": sum(row["RL_responses"] for row in reports),
               "evaluation_responses": sum(row["evaluation_responses"] for row in reports), "superiority_established": False,
               "unsupported": ["registered fresh cohorts", "CUDA/7B", "native distributed production", "formal publication admission", "new resource authority"]}
    write_json_atomic(args.output / "complete.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "trained_updates", "RL_responses", "evaluation_responses", "superiority_established")}))


if __name__ == "__main__":
    main()
