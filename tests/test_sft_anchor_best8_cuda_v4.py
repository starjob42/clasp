"""NEW source-only CUDA/DDP driver guards; fixtures never publish outcomes."""
from __future__ import annotations
import copy
import gc
import hashlib
import json
from pathlib import Path
import pytest
import torch
from clasp.entrypoints import train_sft_anchor_best8_cuda_v4 as m
from clasp.entrypoints import train_sft_anchor_best8_v4 as v4
from clasp.entrypoints import train_sft_anchor_pilot_v2 as v2


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(1)
    base, sft = tmp_path / "base", tmp_path / "sft"
    model, tokenizer, train, validation = v2.tiny_peft_initializer("op", base, sft, torch.device("cpu"))
    del model; gc.collect()
    contract = v4.make_smoke_contract("op", {"id": "synthetic_cuda_path_unit_fixture",
        "sha256": v2.file_identity(sft / "adapter_model.safetensors")["sha256"]}, train, validation)
    return contract, tokenizer, base, sft


def synthetic_groups(contract, tokenizer):
    raw, groups = [], []
    for instance in contract["validation_rows"]:
        for stream, settings in v4.DECODE_STREAMS.items():
            seed = int.from_bytes(hashlib.sha256(f'{contract["eval_seed"]}/{instance["semantic_id"]}/{stream}'.encode()).digest()[:4], "big")
            text = instance["reference_answer"]; ids = tokenizer.encode(text, add_special_tokens=False)
            assert len(ids) == 1
            score = m.score_completion("op", text, instance["instance_json"], text)
            rows = [{"instance_id": instance["instance_id"], "semantic_id": instance["semantic_id"], "stream": stream,
                "sample_index": i, "completion_token_ids": ids, "text": text, "generated_tokens": len(ids), "generation_seed": seed,
                "runtime_complete": True, "finish_reason": "length", "strict_score": score, "quality_utility": .5}
                for i in range(settings["count"])]
            groups.append({"protocol":m.PROTOCOL,"instance_id": instance["instance_id"], "stream": stream, "seed": seed,
                "endpoint": settings["endpoint"], "input_truncated": False, "raw_completions": rows,
                "generation_configuration": m.evaluation_config(settings,tokenizer,1).to_dict(), "extra_generation_kwargs": {"use_model_defaults":False},
                "prompt_tokens":len(tokenizer(instance["prompt"])["input_ids"]),"generated_tokens":len(rows)})
            raw.extend(rows)
    return raw, groups


def test_prospective_template_keeps_six_recipes_and_all_fixed_budgets():
    value = m.production_declaration()
    assert len(value["recipes"]) == 6 and value["horizon"] == value["terminal_selection"] == 100
    assert (value["global_responses_per_update"], value["groups_per_update"], value["siblings_per_group"]) == (36, 4, 9)
    assert (value["learning_rate"], value["max_completion_tokens"], value["temperature"], value["top_p"], value["top_k"]) == (1e-5,3000,.7,1.,0)
    assert not value["native_registration_created"] and not value["native_submission_authorized"]


def test_original_v4_statistical_and_frozen_v2_bytes_are_preserved():
    assert v2.file_identity(Path(v4.__file__))["sha256"] == m.STATISTICAL_SHA
    assert v2.file_identity(Path(v2.__file__))["sha256"] == "79fa6e424c6bcd71b91fe82af9f74ff0bdf37bd153b65d9dce94ffe2b77a3dc2"


def test_source_only_smoke_never_becomes_production_by_scope_rename(fixture):
    contract, _, _, _ = fixture
    assert m.validate_contract(contract) == contract
    wrong = copy.deepcopy(contract); wrong["scope"] = "production"
    with pytest.raises(ValueError): m.validate_contract(wrong)


def test_raw_closure_has_separate_greedy_primary_fullK8_and_filtered_secondary(fixture):
    contract, tokenizer, _, _ = fixture
    raw, groups = synthetic_groups(contract, tokenizer)
    result = m.replay_raw(contract, raw, groups, tokenizer)
    assert result == {"groups":12,"responses":68,"independent_strict_score_replay":True}


@pytest.mark.parametrize("field", ["score", "utility", "seed", "tokens", "text", "type", "order", "recipe", "truncation", "count",
    "penalty","suppress","extra_kwargs","endpoint","prompt_tokens","token_total","EOS","bool_tokens","bool_seed","protocol"])
def test_corrupt_raw_groups_physical_scores_decoder_or_token_identity_are_rejected(fixture, field):
    contract, tokenizer, _, _ = fixture
    raw, groups = synthetic_groups(contract, tokenizer)
    row = groups[0]["raw_completions"][0]
    if field == "score": row["strict_score"]["objective"] += 1
    elif field == "utility": row["quality_utility"] = 1.
    elif field == "seed": groups[0]["seed"] += 1
    elif field == "tokens": row["completion_token_ids"] = [tokenizer.eos_token_id]
    elif field == "text": row["text"] = "Route: [0]"
    elif field == "type": row["sample_index"] = False
    elif field == "order": groups[0], groups[1] = groups[1], groups[0]
    elif field == "recipe": groups[1]["generation_configuration"]["top_p"] = .9
    elif field == "truncation": groups[0]["input_truncated"] = True
    elif field == "penalty": groups[1]["generation_configuration"]["repetition_penalty"] = 2.
    elif field == "suppress": groups[1]["generation_configuration"]["suppress_tokens"] = [tokenizer.eos_token_id]
    elif field == "extra_kwargs": groups[1]["extra_generation_kwargs"] = {"repetition_penalty":2.}
    elif field == "endpoint": groups[1]["endpoint"] = "secondary_continuity"
    elif field == "prompt_tokens": groups[0]["prompt_tokens"] += 1
    elif field == "token_total": groups[0]["generated_tokens"] += 1
    elif field == "EOS": groups[1]["generation_configuration"]["eos_token_id"] = tokenizer.pad_token_id
    elif field == "bool_tokens": row["generated_tokens"] = True
    elif field == "bool_seed": row["generation_seed"] = True
    elif field == "protocol": groups[0]["protocol"] = "other"
    else: raw.pop()
    with pytest.raises(ValueError): m.replay_raw(contract, raw, groups, tokenizer)


def test_complete_loader_inventory_includes_configs_and_rejects_same_weight_config_change(fixture):
    _, _, base, sft = fixture
    trees = {kind:[{**v2.file_identity(p),"path":p.relative_to(root).as_posix()} for p in root.rglob("*") if p.is_file()]
        for kind,root in (("base",base),("sft",sft))}
    m.verify_loader_inventory(base,sft,trees)
    p=base/"config.json";value=json.loads(p.read_text());value["rope_theta"]=123456;p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="bytes/config"):m.verify_loader_inventory(base,sft,trees)


def test_tiny_cpu_or_cuda_fp32_cannot_supply_native_BF16_admission(fixture):
    contract, _, base, sft = fixture
    with pytest.raises(ValueError,match="registry is empty"):
        m.admit_production(contract,"rloo_strict_best8",Path("missing"),registry_sha256="0"*64,base=base,sft=sft,device=torch.device("cpu"))


def test_actual_model_default_penalties_cannot_change_explicit_eval_generation(fixture,tmp_path):
    contract, tokenizer, base, sft=fixture
    model=v2.loaded_model(base,sft,torch.device("cpu"),trainable=False,dtype=torch.float32)
    clean=tmp_path/"eval_clean";clean.mkdir()
    before=m.collect_outputs(model,tokenizer,contract,clean,torch.device("cpu"))
    model.generation_config.repetition_penalty=2.;model.generation_config.suppress_tokens=[tokenizer.eos_token_id]
    model.get_base_model().generation_config=model.generation_config
    poisoned=tmp_path/"eval_poisoned_defaults";poisoned.mkdir()
    after=m.collect_outputs(model,tokenizer,contract,poisoned,torch.device("cpu"))
    assert before==after
    groups=[json.loads(line) for line in (poisoned/"successor_evaluation_groups.jsonl").read_text().splitlines()]
    m.replay_raw(contract,after,groups,tokenizer)


def test_actual_peft_actor_and_generation_base_defaults_are_cleaned_before_native_training(fixture,tmp_path,monkeypatch):
    contract,tokenizer,base,sft=fixture
    loader=v2.loaded_model
    def poisoned(*args,**kwargs):
        model=loader(*args,**kwargs)
        if kwargs.get('trainable'):
            model.generation_config.repetition_penalty=2.;model.generation_config.suppress_tokens=[tokenizer.eos_token_id]
            model.get_base_model().generation_config=model.generation_config
        return model
    monkeypatch.setattr(v2,'loaded_model',poisoned)
    trainer=m.make_trainer(contract,'rloo_strict_best8_beta0',tmp_path/'trainer',tokenizer,base,sft,torch.device('cpu'),dtype=torch.float32)
    actor=trainer.accelerator.unwrap_model(trainer.model)
    config,_=actor.get_base_model()._prepare_generation_config(trainer.generation_config,use_model_defaults=True)
    assert config.repetition_penalty==1. and config.suppress_tokens is None
    assert (config.temperature,config.top_p,config.top_k,config.num_return_sequences)==(.7,1.,0,1)
    trainer._assert_contract()


def test_direct_production_library_call_cannot_bypass_empty_known_anchor_with_fake_admission(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"make_trainer",lambda *a,**k: (_ for _ in ()).throw(AssertionError("must never construct model")))
    with pytest.raises(ValueError,match="cannot train through"):
        m.run_arm({"scope":"production","task":"op","contract_sha256":"a"*64},"rloo_strict_best8",tmp_path,
            None,None,None,torch.device('cpu'),admission={"expected_rank_environments":{"0":{}}})
