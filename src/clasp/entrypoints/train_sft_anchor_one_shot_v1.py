"""Separate one-shot RLOO core and actual tiny CPU PEFT integration.

The runnable entrypoint is synthetic CPU smoke only. Reusable trainer classes
admit a contract-declared number of complete B9 groups; a separately registered
real-model runner owns its data, loader, resources and native admission. No old
campaign, V4 objective or deployed decoder is changed. Greedy quality gains are
unproved. TSP/JSSP and training-only teachers are separate future extensions.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch
from datasets import Dataset
from transformers import GenerationConfig, set_seed
from trl import GRPOTrainer

from clasp.baselines.budget_matched_dev import reward_environment
from clasp.baselines.llmcosolver_f638_training import FOARL_F638_REWARD_FUNCS
from clasp.baselines.sft_anchor_pilot import digest, make_contract as base_contract, score_completion
from clasp.entrypoints import train_sft_anchor_pilot as v1
from clasp.entrypoints import train_sft_anchor_pilot_v2 as v2
from clasp.models.external_baseline_state import semantic_state_sha256
from clasp.rewards import one_shot_quality
from clasp.rewards.feasible_quality import make_advantage_contract
from clasp.trainers.sft_anchored_rloo import SFTAnchoredRLOOTrainer, SFTReferenceGRPOTrainer

PROTOCOL = "sft-anchor-strict-linear-one-shot-v1"
RECIPES = {
    "rloo_cost": {"method": "rloo", "reward": "strict_linear", "heads": 1, "loss_type": "dr_grpo", "scale_rewards": "none", "beta": .05},
    "grpo_cost": {"method": "grpo", "reward": "strict_linear", "heads": 1, "loss_type": "grpo", "scale_rewards": "group", "beta": .05},
    "grpo_cost_fixed_horizon": {"method": "grpo", "reward": "strict_linear", "heads": 1, "loss_type": "dr_grpo", "scale_rewards": "group", "beta": .05},
    "foarl_cost_changed": {"method": "grpo", "reward": "strict_linear_split", "heads": 2, "loss_type": "grpo", "scale_rewards": "group", "beta": .05},
    "foarl_released": {"method": "grpo", "reward": "released_foarl", "heads": 2, "loss_type": "grpo", "scale_rewards": "group", "beta": .05},
}
PRIMARY = {"name": "actual_greedy_one_shot", "do_sample": False, "num_return_sequences": 1,
           "repair": False, "search": False, "best_of_k": False, "retry": False}
identity, write = v2.file_identity, v2.write_json_atomic


def need(value, message):
    if not value:
        raise ValueError(message)


def prospective_template(groups_per_update=4):
    need(type(groups_per_update) is int and groups_per_update > 0, "Positive complete B9 group count required")
    return {"protocol": PROTOCOL, "status": "source_only_prospective", "recipes": copy.deepcopy(RECIPES),
        "tasks": ["op", "cvrp"], "phase2_tasks_not_implemented": ["tsp", "jssp"],
        "groups_per_update": groups_per_update, "siblings_per_group": 9, "global_responses_per_update": 9*groups_per_update,
        "horizon": 100, "terminal_selection": 100, "learning_rate": 1e-5, "max_completion_tokens": 3000,
        "training_sampler": {"temperature": .7, "top_p": 1., "top_k": 0}, "evaluation": copy.deepcopy(PRIMARY),
        "reward": one_shot_quality.contract(), "best8_auxiliary": 0., "cardinal_auxiliary": 0.,
        "training_teacher": False, "production_registration_ready": False, "production_launch_enabled": False,
        "superiority_established": False, "compute_scope": "Matched rollout/update/token caps; actual training and scoring costs must also be reported.",
        "control_labels": {"grpo_cost": "Same reward, stock response-mean GRPO; changed reward relative to classic released controls.",
            "grpo_cost_fixed_horizon": "Same reward and fixed-horizon loss; stock group-STD credit ablation.",
            "foarl_cost_changed": "Changed FOARL head decomposition; not the released reward.",
            "foarl_released": "Original released two reward heads; matched SFT reference and sampler."}}


def make_smoke_contract(task, initializer, train, validation, *, horizon=4, max_completion_tokens=1,
                        groups_per_update=4, train_seed=61):
    prospective_template(groups_per_update)
    base = base_contract(task, initializer, train, validation, scope="system_smoke", horizon=horizon,
                         max_completion_tokens=max_completion_tokens, train_seed=train_seed)
    # A new declared schedule, not a relabel of the original four-group study.
    draws = base["training"]["prompt_draws"]
    if groups_per_update > 4:
        draws = [[row[index % 4] for index in range(groups_per_update)] for row in draws]
    else:
        draws = [row[:groups_per_update] for row in draws]
    base.update(protocol=PROTOCOL, schema_version=1, recipes=copy.deepcopy(RECIPES),
                one_shot_reward=one_shot_quality.contract())
    base.pop("arms")
    base["training"].update(prompt_draws=draws, groups_per_update=groups_per_update,
        global_responses_per_update=9*groups_per_update, prompt_groups_per_step=groups_per_update,
        responses_per_step=9*groups_per_update, prompt_draw_sha256=digest(draws), top_p=1., top_k=0)
    base["evaluation"] = {**copy.deepcopy(PRIMARY), "max_completion_tokens": max_completion_tokens}
    base["contract_sha256"] = digest({k:v for k,v in base.items() if k != "contract_sha256"})
    return base


def validate_training_contract(contract):
    """Trainer invariants only; this does not admit a real-model policy call."""
    need(type(contract) is dict and contract.get("protocol") == PROTOCOL and contract.get("task") in ("op", "cvrp"),
         "Explicit separate OP/CVRP one-shot contract required")
    need(contract.get("contract_sha256") == digest({k:v for k,v in contract.items() if k != "contract_sha256"}),
         "One-shot semantic contract identity differs")
    training = contract["training"]; groups = training.get("groups_per_update")
    need(type(groups) is int and groups > 0 and training.get("global_responses_per_update") == 9*groups and
         type(training.get("global_responses_per_update")) is int and
         type(training.get("horizon")) is int and training["horizon"] > 0,
         "Declared global rollout must consist of positive complete B9 groups")
    need(type(training.get("prompt_groups_per_step")) is int and training["prompt_groups_per_step"] == groups and
         type(training.get("responses_per_step")) is int and training["responses_per_step"] == 9*groups and
         type(training.get("num_generations")) is int and training["num_generations"] == 9,
         "Training sibling/group/response aliases must state the actual declared rollout")
    need(training["temperature"] == .7 and training["top_p"] == 1. and training["top_k"] == 0 and
         contract.get("recipes") == RECIPES and contract.get("one_shot_reward") == one_shot_quality.contract(),
         "Shared full-support sampler or pure scalar reward changed")
    draws = training["prompt_draws"]; ids = {row["instance_id"] for row in contract["training_rows"]}
    need(len(draws) == training["horizon"] and all(type(row) is list and len(row) == groups and
         all(type(value) is str and value in ids for value in row) for row in draws), "Complete fixed prompt schedule required")
    need(training.get("prompt_draw_sha256") == digest(draws), "Actual prompt schedule hash differs")
    need(contract["evaluation"] == {**PRIMARY, "max_completion_tokens": training["max_completion_tokens"]},
         "Exactly one unassisted greedy endpoint required")
    for row in contract["training_rows"] + contract["validation_rows"]:
        one_shot_quality.physical_scale(contract["task"], row["instance_json"])
    return contract


def validate_smoke_contract(contract):
    validate_training_contract(contract)
    need(contract.get("scope") == "system_smoke", "CPU factory supports synthetic system_smoke only; real runner requires separate admission")
    expected = make_smoke_contract(contract["task"], contract["initializer"], contract["training_rows"], contract["validation_rows"],
        horizon=contract["training"]["horizon"], max_completion_tokens=contract["training"]["max_completion_tokens"],
        groups_per_update=contract["training"]["groups_per_update"], train_seed=contract["train_seed"])
    need(contract == expected, "Tiny one-shot contract differs")
    return copy.deepcopy(expected)


def _aligned(task, prompts, kwargs):
    required = ("task_type", "answer", "instance_json", "input")
    need(all(isinstance(kwargs.get(k), (list,tuple)) and len(kwargs[k]) == len(prompts) for k in required),
         "Completion-aligned strict reward inputs required")
    return [{**{key:kwargs[key][index] for key in required}, "prompt":prompt} for index,prompt in enumerate(prompts)]


def strict_cost_reward(prompts, completions, **kwargs):
    tasks = kwargs.get("task_type", [])
    need(tasks and len(set(tasks)) == 1, "One task per one-shot reward window required")
    return one_shot_quality.reward_batch(tasks[0], prompts, completions, _aligned(tasks[0], prompts, kwargs))["rewards"]


def strict_cost_quality_head(prompts, completions, **kwargs):
    tasks = kwargs.get("task_type", [])
    need(tasks and len(set(tasks)) == 1, "One task per one-shot reward window required")
    return one_shot_quality.reward_batch(tasks[0], prompts, completions, _aligned(tasks[0], prompts, kwargs))["quality_head"]


def strict_cost_validity_head(prompts, completions, **kwargs):
    tasks = kwargs.get("task_type", [])
    need(tasks and len(set(tasks)) == 1, "One task per one-shot reward window required")
    return one_shot_quality.reward_batch(tasks[0], prompts, completions, _aligned(tasks[0], prompts, kwargs))["validity_head"]


def generation_config(tokenizer, cap, *, sample):
    return GenerationConfig(max_new_tokens=cap, do_sample=sample, num_return_sequences=1,
        temperature=.7 if sample else 1., top_p=1., top_k=0,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
        repetition_penalty=1., num_beams=1, num_beam_groups=1, typical_p=1., min_p=None,
        epsilon_cutoff=0., eta_cutoff=0., no_repeat_ngram_size=0, min_length=0, min_new_tokens=None,
        bad_words_ids=None, force_words_ids=None, suppress_tokens=None, begin_suppress_tokens=None,
        forced_bos_token_id=None, forced_eos_token_id=None, constraints=None, penalty_alpha=None)


def generation_config_snapshot(config):
    """Full policy config; normalize only metadata and singleton EOS notation.

    Transformers Trainer may align an integer EOS to its equivalent [integer]
    representation when reconciling the tokenizer. Neither notation changes
    termination. All processor, cache, length and unknown fields remain pinned.
    """
    value = copy.deepcopy(config.to_dict())
    for key in ("transformers_version","_from_model_config","_commit_hash"):
        value.pop(key,None)
    eos = value.get("eos_token_id")
    if type(eos) is list and len(eos) == 1 and type(eos[0]) is int:
        value["eos_token_id"] = eos[0]
    return value


class OneShotOrderedObservationMixin(v1.OrderedObservationMixin):
    def __init__(self, *args, pilot_contract, **kwargs):
        validate_training_contract(pilot_contract)
        self.groups_per_update = pilot_contract["training"]["groups_per_update"]
        super().__init__(*args, pilot_contract=pilot_contract, **kwargs)

    def _get_train_sampler(self, dataset=None):
        return v1.RepeatSampler(data_source=self.train_dataset if dataset is None else dataset,
            mini_repeat_count=9, batch_size=self.groups_per_update,
            repeat_count=self.args.steps_per_generation, shuffle=False, seed=self.args.seed)

    def _generate_and_score_completions(self, inputs):
        # The inherited ordered observer indexes instance_id before its gather.
        # Admit local IDs collectively first so malformed peers abort together.
        def admit_ids():
            need(type(inputs) is list and inputs and all(type(row) is dict and
                 type(row.get("instance_id")) is str and row["instance_id"] for row in inputs),
                 "Complete local instance IDs required before ordered observation")
        self._collective_check(admit_ids,"one-shot ordered input admission")
        return super()._generate_and_score_completions(inputs)


class OneShotCreditAuditMixin:
    """Verify shared physical rewards; retain native RLOO or stock GRPO credit."""
    def __init__(self, *args, one_shot_recipe, **kwargs):
        need(one_shot_recipe in RECIPES, "Unknown one-shot recipe")
        self.one_shot_recipe = one_shot_recipe
        self.one_shot_settings = copy.deepcopy(RECIPES[one_shot_recipe])
        self.last_one_shot_record = None
        self._one_shot_metadata = self._one_shot_rewards = self._one_shot_keys = None
        super().__init__(*args, **kwargs)

    def _recipe(self):
        value = super()._recipe()
        value.update(protocol=PROTOCOL, one_shot_recipe=self.one_shot_recipe,
            one_shot_settings=copy.deepcopy(self.one_shot_settings), one_shot_reward=one_shot_quality.contract(),
            one_shot_source=identity(Path(__file__)), reward_source=identity(Path(one_shot_quality.__file__)),
            evaluation_endpoint=copy.deepcopy(PRIMARY), training_sampler={"temperature":.7,"top_p":1.,"top_k":0},
            groups_per_update=self.groups_per_update,
            score_source_sha256=hashlib.sha256(inspect.getsource(GRPOTrainer._get_per_token_logps_and_entropies).encode()).hexdigest())
        return value

    def _assert_contract(self):
        super()._assert_contract()
        validate_training_contract(self.pilot_contract)
        settings = RECIPES[self.one_shot_recipe]
        need(self.one_shot_settings == settings and self.beta == self.args.beta == settings["beta"] and
             self.loss_type == settings["loss_type"] and self.scale_rewards == settings["scale_rewards"], "One-shot method/anchor drift")
        count = self.groups_per_update*9
        need(self.num_generations == 9 and self.args.generation_batch_size == count and
             count % self.accelerator.num_processes == 0 and
             self.args.steps_per_generation == self.args.gradient_accumulation_steps and
             self.args.per_device_train_batch_size*self.args.gradient_accumulation_steps*self.accelerator.num_processes == count,
             "One complete declared rollout window per optimizer step required")
        need(self.args.max_steps == self.pilot_contract["training"]["horizon"] and
             self.max_completion_length == self.pilot_contract["training"]["max_completion_tokens"] and
             self.args.learning_rate == self.pilot_contract["training"]["learning_rate"] == 1e-5 and
             self.num_iterations == 1 and self.args.importance_sampling_level == "token" and
             not self.args.mask_truncated_completions and not self.args.use_vllm and
             not self.args.use_liger_loss and not self.args.sync_ref_model and self.args.top_entropy_quantile == 1.,
             "Declared terminal/score/update or reward-window semantics differ")
        cfg = self.generation_config
        # Check the complete config, including newly introduced Transformers
        # processors. Native TRL scores the temperature-only categorical policy.
        # Model defaults must also be clean: TRL's generate call may merge them.
        expected_cfg = generation_config_snapshot(generation_config(self.processing_class,self.max_completion_length,sample=True))
        need(generation_config_snapshot(cfg) == expected_cfg,
             "Full-support training requires the complete declared generation configuration")
        actor = self.accelerator.unwrap_model(self.model)
        model_configs = [actor.generation_config]
        if hasattr(actor,"get_base_model"):
            model_configs.append(actor.get_base_model().generation_config)
        need(all(generation_config_snapshot(model_cfg) == expected_cfg for model_cfg in model_configs),
             "Model generation defaults differ from the declared full-support policy")
        need(cfg.do_sample is True and cfg.num_return_sequences == 1 and
             self.args.temperature == self.temperature == cfg.temperature == .7 and
             self.args.top_p == cfg.top_p == 1. and self.args.top_k == cfg.top_k == 0 and
             cfg.repetition_penalty == 1. and cfg.typical_p == 1. and cfg.min_p is None and
             cfg.epsilon_cutoff == cfg.eta_cutoff == 0. and cfg.no_repeat_ngram_size == 0 and
             cfg.num_beams == cfg.num_beam_groups == 1 and cfg.min_length == 0 and cfg.min_new_tokens is None and
             all(getattr(cfg,k,None) is None for k in ("bad_words_ids","force_words_ids","suppress_tokens","begin_suppress_tokens",
                "forced_bos_token_id","forced_eos_token_id","constraints","penalty_alpha","top_h")),
             "Full-support temperature-aligned training forbids hidden sampling penalties")

    def _prompt_keys(self, inputs):
        task = self.pilot_contract["task"]
        def validate_local():
            need(type(inputs) is list and inputs and all(type(r) is dict and r.get("task_type") == task for r in inputs),
                 "One-shot input task/order metadata differs")
            return [digest([r.get("task_type"),r.get("instance_json"),r.get("input"),r.get("prompt")]) for r in inputs]
        local = self._collective_check(validate_local,"one-shot local prompt admission")
        parts = self._gather_objects(local)
        need(all(len(part) == len(local) for part in parts), "One-shot rank rollout lengths differ")
        keys = [key for part in parts for key in part]
        need(len(keys) == self.groups_per_update*9 and all(len(set(keys[i:i+9])) == 1 for i in range(0,len(keys),9)),
             "One-shot requires complete same-prompt B9 groups")
        return keys

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        verified = self._collective_check(lambda: one_shot_quality.reward_batch(self.pilot_contract["task"],prompts,completions,inputs),
                                          "one-shot strict physical reward")
        parts = self._gather_objects(verified["metadata"])
        metadata = [row for part in parts for row in part]
        need([row["prompt_key"] for row in metadata] == self._one_shot_keys, "One-shot reward metadata lost prompt order")
        for index in range(0,len(metadata),9):
            need(len({row["physical_scale"]["instance_sha256"] for row in metadata[index:index+9]}) == 1,
                 "Sibling physical normalization differs")
        rewards = super()._calculate_rewards(inputs,prompts,completions,completion_ids_list)
        need(tuple(rewards.shape) == (self.groups_per_update*9,self.one_shot_settings["heads"]) and
             bool(torch.isfinite(rewards).all()), "One-shot gathered reward shape/numerics differ")
        if self.one_shot_settings["reward"] != "released_foarl":
            rows = ([[row["anchor_reward"]] for row in metadata] if self.one_shot_settings["heads"] == 1 else
                    [[row["quality_head"],row["validity_head"]] for row in metadata])
            need(torch.equal(rewards,torch.tensor(rows,dtype=rewards.dtype,device=rewards.device)),
                 "Shared strict physical reward and native head calculation differ")
        self._one_shot_metadata, self._one_shot_rewards = metadata,rewards.detach().clone()
        return rewards

    def _generate_and_score_completions(self, inputs):
        self._collective_check(self._assert_contract,"one-shot active contract")
        self._one_shot_keys = self._prompt_keys(inputs)
        self._one_shot_metadata = self._one_shot_rewards = None
        try:
            output = super()._generate_and_score_completions(inputs)
            rewards = self._one_shot_rewards
            need(rewards is not None and self._one_shot_metadata is not None,"Missing one-shot reward ledger")
            scalar = rewards.sum(1).view(-1,9)
            expected = ((scalar-(scalar.sum(1,keepdim=True)-scalar)/8).reshape(-1) if self.one_shot_settings["method"] == "rloo" else
                        ((scalar-scalar.mean(1,keepdim=True))/(scalar.std(1,keepdim=True)+1e-4)).reshape(-1)).detach()
            start = self.accelerator.process_index*len(inputs)
            need(torch.equal(output["advantages"],expected[start:start+len(inputs)]) and not output["advantages"].requires_grad,
                 "Native per-response RLOO/GRPO coefficients changed")
            record = {"protocol":PROTOCOL,"recipe":self.one_shot_recipe,"step":int(self.state.global_step),
                "global_responses":len(expected),"gathered_reward_heads":rewards.cpu().tolist(),
                "advantages":expected.cpu().tolist(),"strict_metadata":self._one_shot_metadata,
                "best8_auxiliary":0.,"cardinal_auxiliary":0.,"beta":self.beta,"loss_type":self.loss_type,
                "scale_rewards":self.scale_rewards,"endpoint":PRIMARY["name"]}
            self.last_one_shot_record = record
            if self.accelerator.is_main_process:
                v2._append_durable(Path(self.args.output_dir)/"one_shot_coefficients.jsonl",record)
            return output
        finally:
            self._one_shot_keys = self._one_shot_metadata = self._one_shot_rewards = None


class OneShotRLOOTrainer(v2.EffectiveCheckpointingMixin, OneShotOrderedObservationMixin,
                          OneShotCreditAuditMixin, SFTAnchoredRLOOTrainer):
    pass


class OneShotGRPOTrainer(v2.EffectiveCheckpointingMixin, OneShotOrderedObservationMixin,
                          OneShotCreditAuditMixin, SFTReferenceGRPOTrainer):
    pass


def make_cpu_trainer(contract, recipe, output, tokenizer, base, sft):
    contract = validate_smoke_contract(contract)
    need(int(os.environ.get("WORLD_SIZE","1")) == 1,"CPU smoke factory supports one rank; native runner is separate")
    settings = RECIPES[recipe]; output = Path(output); output.mkdir(parents=True,exist_ok=False)
    set_seed(contract["train_seed"])
    os.environ.update(reward_environment("foarl" if settings["method"] == "grpo" else "rloo_quality"))
    actor = v2.loaded_model(base,sft,torch.device("cpu"),trainable=True,dtype=torch.float32)
    reference = v2.loaded_model(base,sft,torch.device("cpu"),trainable=False,dtype=torch.float32)
    clean = generation_config(tokenizer,contract["training"]["max_completion_tokens"],sample=True)
    actor.generation_config = copy.deepcopy(clean); actor.get_base_model().generation_config = copy.deepcopy(clean)
    need(identity(Path(sft)/"adapter_model.safetensors")["sha256"] == contract["initializer"]["sha256"],"Original SFT adapter changed")
    alias = "rloo_scalar_beta005" if settings["method"] == "rloo" else "foarl"
    args = v2._training_args(contract,alias,output,torch.device("cpu"),production_peft=True)
    count = contract["training"]["global_responses_per_update"]
    args.generation_batch_size = count; args.gradient_accumulation_steps = count; args.steps_per_generation = count
    args.top_p = 1.; args.top_k = 0; args.loss_type = settings["loss_type"]; args.scale_rewards = settings["scale_rewards"]
    by_id = {r["instance_id"]:r for r in contract["training_rows"]}
    rows = [{**by_id[i],"answer":by_id[i]["output"],"instance_json":json.dumps(by_id[i]["instance_json"])}
            for draw in contract["training"]["prompt_draws"] for i in draw]
    kwargs = dict(model=actor,args=args,processing_class=tokenizer,train_dataset=Dataset.from_list(rows),
        reference_model=reference,reference_identity={"source_kind":"original_sft","source_sha256":contract["initializer"]["sha256"],
            "immutable":True,"scope":contract["scope"]},pilot_contract=contract,one_shot_recipe=recipe,
        recipe_contract={"protocol":PROTOCOL,"contract_sha256":contract["contract_sha256"],"recipe":recipe,
            "entrypoint_sha256":identity(Path(__file__))["sha256"],"reward_sha256":identity(Path(one_shot_quality.__file__))["sha256"]})
    if settings["method"] == "rloo":
        trainer = OneShotRLOOTrainer(reward_funcs=[strict_cost_reward],advantage_contract=make_advantage_contract("control"),**kwargs)
    else:
        heads = (list(FOARL_F638_REWARD_FUNCS) if settings["reward"] == "released_foarl" else
                 [strict_cost_quality_head,strict_cost_validity_head] if settings["heads"] == 2 else [strict_cost_reward])
        trainer = OneShotGRPOTrainer(reward_funcs=heads,**kwargs)
    trainer._assert_contract()
    return trainer


def collect_greedy_outputs(model, tokenizer, contract, output):
    validate_training_contract(contract); model.eval(); device = next(model.parameters()).device
    cfg = generation_config(tokenizer,contract["evaluation"]["max_completion_tokens"],sample=False)
    rows = []; seen = set()
    for instance in contract["validation_rows"]:
        need(instance["instance_id"] not in seen,"One-shot validation IDs must be unique")
        seen.add(instance["instance_id"])
        encoded = tokenizer(instance["prompt"],return_tensors="pt",truncation=False)
        inputs = {k:v.to(device) for k,v in encoded.items() if k in ("input_ids","attention_mask")}
        need(inputs["input_ids"].shape[1]+cfg.max_new_tokens <= model.config.max_position_embeddings,
             "Full untruncated one-shot prompt exceeds context")
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs,generation_config=copy.deepcopy(cfg),use_model_defaults=False)
        need(generated.shape[0] == 1,"One-shot decoder returned more than one candidate")
        ids = generated[0,inputs["input_ids"].shape[1]:].cpu().tolist()
        if tokenizer.eos_token_id in ids: ids = ids[:ids.index(tokenizer.eos_token_id)+1]
        text = tokenizer.decode(ids,skip_special_tokens=True)
        record = {"protocol":PROTOCOL,"instance_id":instance["instance_id"],"semantic_id":instance["semantic_id"],
            "sample_index":0,"generation_calls":1,"completion_token_ids":ids,"text":text,"generated_tokens":len(ids),
            "endpoint":copy.deepcopy(PRIMARY),"generation_config":cfg.to_dict(),"use_model_defaults":False,
            "strict_score":score_completion(contract["task"],text,instance["instance_json"],instance["reference_answer"]),
            "wall_seconds":time.monotonic()-started,"runtime_complete":True}
        rows.append(record); v2._append_durable(Path(output)/"one_shot_evaluation_groups.jsonl",record)
    return rows


def run_cpu_arm(contract, recipe, output, tokenizer, base, sft):
    need(int(os.environ.get("WORLD_SIZE","1")) == 1,"Serial CPU smoke only")
    contract = validate_smoke_contract(contract); output = Path(output); trainer = None
    if recipe == "sft":
        output.mkdir(parents=True,exist_ok=False)
        model = v2.loaded_model(base,sft,torch.device("cpu"),trainable=False,dtype=torch.float32)
        initial = actor_sha = semantic_state_sha256(model.state_dict()); steps = tokens = 0; ref_sha = None
    else:
        trainer = make_cpu_trainer(contract,recipe,output,tokenizer,base,sft)
        initial = semantic_state_sha256(trainer.accelerator.unwrap_model(trainer.model).state_dict())
        started = time.monotonic(); trainer.train(); elapsed = time.monotonic()-started
        trainer._assert_reference_integrity(full=True)
        model = trainer.accelerator.unwrap_model(trainer.model); actor_sha = semantic_state_sha256(model.state_dict())
        steps,tokens = trainer.state.global_step,trainer.actual_generated_tokens; ref_sha = trainer.reference_state_sha256()
        need(steps == len(trainer.actual_windows) == contract["training"]["horizon"],"Fixed terminal/window count differs")
    terminal = output/"terminal"; model.save_pretrained(terminal); tokenizer.save_pretrained(terminal)
    proof = {"protocol":PROTOCOL,"scope":"system_smoke","recipe":recipe,"steps":steps,
        "RL_responses":steps*contract["training"]["global_responses_per_update"],"generated_tokens":tokens,
        "training_wall_seconds":elapsed if trainer else 0.,"whole_actor_sha256":actor_sha,
        "initial_whole_actor_sha256":initial,"actor_changed":initial != actor_sha,"reference_sha256":ref_sha,
        "reference_integrity":bool(trainer),"initial_reference_audit":trainer.initial_reference_audit if trainer else None,
        "prepared_reference_audit":trainer.prepared_reference_audit if trainer else None,
        "effective_checkpointing":trainer.last_checkpointing_audit if trainer else None,
        "actual_windows":trainer.actual_windows if trainer else [],"terminal_tree":v2.artifact_tree(terminal),
        "contract_sha256":contract["contract_sha256"],"source":identity(Path(__file__)),"superiority_established":False}
    write(output/"one_shot_training_complete.json",proof)
    del model,trainer; gc.collect()
    reloaded = v2.loaded_model(base,terminal,torch.device("cpu"),trainable=False,dtype=torch.float32)
    need(semantic_state_sha256(reloaded.state_dict()) == actor_sha,"Typed saved/reloaded actor differs")
    write(output/"one_shot_checkpoint_reload.json",{"whole_actor_sha256":actor_sha,"bitwise_equal":True,
        "training_proof_sha256":identity(output/"one_shot_training_complete.json")["sha256"]})
    raw = collect_greedy_outputs(reloaded,tokenizer,contract,output)
    report = {"protocol":PROTOCOL,"scope":"system_smoke","recipe":recipe,"trained_updates":steps,
        "RL_responses":proof["RL_responses"],"evaluation_responses":len(raw),"primary_endpoint":copy.deepcopy(PRIMARY),
        "checkpoint_reload_exact":True,"raw_completions":raw,"superiority_established":False}
    write(output/"result.json",report); del reloaded; gc.collect()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true"); parser.add_argument("--task",choices=("op","cvrp"),required=True)
    parser.add_argument("--recipe",choices=(*RECIPES,"sft","all"),default="rloo_cost")
    parser.add_argument("--groups-per-update",type=int,default=4); parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    need(args.smoke and int(os.environ.get("WORLD_SIZE","1")) == 1,"Entrypoint supports explicit tiny CPU smoke only")
    torch.set_num_threads(1); args.output.mkdir(parents=True,exist_ok=False)
    base,sft = args.output/"fixture_base",args.output/"fixture_sft"
    model,tokenizer,train,validation = v2.tiny_peft_initializer(args.task,base,sft,torch.device("cpu")); del model; gc.collect()
    contract = make_smoke_contract(args.task,{"id":"tiny-one-shot-PEFT","sha256":identity(sft/"adapter_model.safetensors")["sha256"]},
                                  train,validation,groups_per_update=args.groups_per_update)
    write(args.output/"contract.json",contract)
    recipes = (*RECIPES,"sft") if args.recipe == "all" else (args.recipe,)
    reports = [run_cpu_arm(contract,recipe,args.output/recipe,tokenizer,base,sft) for recipe in recipes]
    write(args.output/"complete.json",{"protocol":PROTOCOL,"status":"completed_actual_tiny_cpu_one_shot_smoke",
        "scope":"system_smoke","recipes":list(recipes),"RL_responses":sum(r["RL_responses"] for r in reports),
        "evaluation_responses":sum(r["evaluation_responses"] for r in reports),"superiority_established":False,
        "limitations":["No actual native/CUDA/BF16/7B validation","No empirical greedy/FOARL win","No TSP/JSSP or teacher implementation"]})


if __name__ == "__main__": main()
