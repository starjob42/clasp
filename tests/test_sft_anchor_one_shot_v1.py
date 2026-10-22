"""Actual tiny CPU PEFT tests; no native/current campaign policy outcomes."""
import copy
import gc
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from clasp.entrypoints import train_sft_anchor_one_shot_v1 as one
from clasp.entrypoints import train_sft_anchor_pilot_v2 as v2
from clasp.entrypoints import train_sft_anchor_recovery_v3 as recovery
from clasp.models.external_baseline_state import semantic_state_sha256


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("WORLD_SIZE","1"); monkeypatch.setenv("RANK","0"); monkeypatch.delenv("LOCAL_RANK",raising=False)
    keys = set(one.reward_environment("foarl")) | set(one.reward_environment("rloo_quality"))
    previous = {k:os.environ.get(k) for k in keys}
    yield
    for k,v in previous.items():
        if v is None: os.environ.pop(k,None)
        else: os.environ[k] = v


def fixture(path, task="op", groups=1, horizon=2):
    base,sft = path/"base",path/"sft"
    model,tokenizer,train,validation = v2.tiny_peft_initializer(task,base,sft,torch.device("cpu"))
    del model; gc.collect()
    contract = one.make_smoke_contract(task,{"id":"actual-tiny-one-shot-PEFT", "sha256":one.identity(sft/"adapter_model.safetensors")["sha256"]},
        train,validation,horizon=horizon,groups_per_update=groups)
    return dict(contract=contract,tokenizer=tokenizer,base=base,sft=sft)


def first_window(contract):
    rows = {r["instance_id"]:r for r in contract["training_rows"]}
    return [{**rows[i],"answer":rows[i]["output"],"instance_json":json.dumps(rows[i]["instance_json"])}
        for i in contract["training"]["prompt_draws"][0] for _ in range(9)]


def fixed(trainer, monkeypatch, replies):
    cap = trainer.pilot_contract["training"]["global_responses_per_update"]
    assert len(replies) == 9
    tokens = [trainer.processing_class.encode(text,add_special_tokens=False) for text in replies]*(cap//9)
    assert all(len(row) == 1 for row in tokens)
    def generate(prompts, images):
        assert len(prompts) == cap
        return [trainer.processing_class.encode(p,add_special_tokens=False) for p in prompts],tokens,torch.tensor(cap),None,{}
    monkeypatch.setattr(trainer,"_generate",generate)
    trainer.current_gradient_accumulation_steps = 1; trainer.model.train()


def test_protocol_and_actual_greedy_endpoint_are_separate_from_best8_and_scope_is_explicit(tmp_path):
    template = one.prospective_template(1)
    assert template["global_responses_per_update"] == 9 and template["siblings_per_group"] == 9
    assert template["evaluation"] == one.PRIMARY and template["best8_auxiliary"] == template["cardinal_auxiliary"] == 0
    assert not template["production_launch_enabled"] and template["phase2_tasks_not_implemented"] == ["tsp","jssp"]
    common = fixture(tmp_path)
    assert common["contract"]["training"]["responses_per_step"] == common["contract"]["training"]["global_responses_per_update"] == 9
    assert common["contract"]["training"]["prompt_groups_per_step"] == common["contract"]["training"]["groups_per_update"] == 1
    assert common["contract"]["training"]["prompt_draw_sha256"] == one.digest(common["contract"]["training"]["prompt_draws"])
    contract = copy.deepcopy(common["contract"]); contract["scope"] = "local_qualification"
    contract["contract_sha256"] = one.digest({k:v for k,v in contract.items() if k != "contract_sha256"})
    assert one.validate_training_contract(contract) == contract
    with pytest.raises(ValueError,match="synthetic system_smoke"):
        one.make_cpu_trainer(**{**common,"contract":contract},recipe="rloo_cost",output=tmp_path/"denied")
    assert not (tmp_path/"denied").exists()


@pytest.mark.parametrize("task",["op","cvrp"])
@pytest.mark.parametrize("recipe",["rloo_cost","grpo_cost_fixed_horizon","grpo_cost","foarl_cost_changed","foarl_released"])
def test_exact_native_credit_and_actor_gradient_for_same_reward_method_controls(tmp_path,monkeypatch,task,recipe):
    common = fixture(tmp_path,task)
    trainer = one.make_cpu_trainer(**common,recipe=recipe,output=tmp_path/"trainer")
    good,low = (("Route: [0, 1, 2]","Route: [0, 1]") if task == "op" else
                ("Routes: [[0, 1, 2, 0]]","Routes: [[0, 1, 0], [0, 2, 0]]"))
    fixed(trainer,monkeypatch,[good]*2+[low]*5+["invalid"]*2)
    batch = trainer._generate_and_score_completions(first_window(common["contract"]))
    record = trainer.last_one_shot_record
    heads = torch.tensor(record["gathered_reward_heads"]); grouped = heads.sum(1).view(-1,9)
    expected = ((grouped-(grouped.sum(1,keepdim=True)-grouped)/8).reshape(-1) if recipe == "rloo_cost" else
                ((grouped-grouped.mean(1,keepdim=True))/(grouped.std(1,keepdim=True)+1e-4)).reshape(-1))
    assert torch.equal(batch["advantages"],expected) and not batch["advantages"].requires_grad
    assert (trainer.loss_type,trainer.scale_rewards) == (one.RECIPES[recipe]["loss_type"],one.RECIPES[recipe]["scale_rewards"])
    assert record["best8_auxiliary"] == record["cardinal_auxiliary"] == 0
    assert all(row["utility"] == 0 and row["objective"] is None for row in record["strict_metadata"][-2:])
    params = tuple(p for p in trainer.model.parameters() if p.requires_grad)
    actual_loss = trainer.compute_loss(trainer.model,batch)
    zero_loss = trainer.compute_loss(trainer.model,{**batch,"advantages":torch.zeros_like(expected)})
    actual = torch.autograd.grad(actual_loss-zero_loss,params,allow_unused=True)
    logps,_ = trainer._get_per_token_logps_and_entropies(trainer.model,
        torch.cat([batch["prompt_ids"],batch["completion_ids"]],1),
        torch.cat([batch["prompt_mask"],batch["completion_mask"]],1),1)
    exact = torch.autograd.grad(-(logps*expected.unsqueeze(1)*batch["completion_mask"]).sum()/9,params,allow_unused=True)
    assert sum(value.abs().sum().item() for value in actual if value is not None) > 0
    for a,b in zip(actual,exact):
        if a is None or b is None: assert a is b
        else: torch.testing.assert_close(a,b,atol=3e-7,rtol=3e-6)
    trainer._assert_reference_integrity(full=True)
    assert trainer.initial_reference_audit["bitwise_equal"] and trainer.prepared_reference_audit["bitwise_equal"]
    assert trainer.last_checkpointing_audit["effective_use_reentrant"] is False
    assert all(p.grad is None and not p.requires_grad for p in trainer.ref_model.parameters())


@pytest.mark.parametrize("groups",[1,4,5])
def test_actual_sampler_obeys_contract_positive_complete_b9_group_count(tmp_path,groups):
    common = fixture(tmp_path,groups=groups,horizon=1)
    trainer = one.make_cpu_trainer(**common,recipe="rloo_cost",output=tmp_path/"trainer")
    trainer.train()
    assert trainer.state.global_step == len(trainer.actual_windows) == 1
    expected = [i for i in common["contract"]["training"]["prompt_draws"][0] for _ in range(9)]
    assert trainer.actual_windows == [expected] and len(expected) == 9*groups
    assert trainer.actual_generated_tokens == 9*groups
    trainer._assert_reference_integrity(full=True)


@pytest.mark.parametrize("fault",["top_p","top_k","temperature","loss","beta","repetition","group_count","wrong_prompt","reference","horizon","learning_rate",
    "sequence_bias","stop_strings","guidance_scale","token_healing","forced_decoder_ids","watermarking_config","model_default"])
def test_recipe_group_or_reference_drift_fails_before_generation(tmp_path,monkeypatch,fault):
    common = fixture(tmp_path); trainer = one.make_cpu_trainer(**common,recipe="rloo_cost",output=tmp_path/"trainer")
    inputs = first_window(common["contract"])
    if fault == "top_p": trainer.generation_config.top_p = .9
    elif fault == "top_k": trainer.generation_config.top_k = 50
    elif fault == "temperature": trainer.generation_config.temperature = 1.
    elif fault == "loss": trainer.loss_type = "grpo"
    elif fault == "beta": trainer.args.beta = 0.
    elif fault == "repetition": trainer.generation_config.repetition_penalty = 1.2
    elif fault == "group_count": trainer.groups_per_update = 2
    elif fault == "wrong_prompt": inputs[0]["input"] += "changed"
    elif fault == "horizon": trainer.args.max_steps += 1
    elif fault == "learning_rate": trainer.args.learning_rate *= 2
    elif fault == "sequence_bias": trainer.generation_config.sequence_bias = {(trainer.eos_token_id,):-2.}
    elif fault == "stop_strings": trainer.generation_config.stop_strings = ["Route"]
    elif fault == "guidance_scale": trainer.generation_config.guidance_scale = 2.
    elif fault == "token_healing": trainer.generation_config.token_healing = True
    elif fault == "forced_decoder_ids": trainer.generation_config.forced_decoder_ids = [[1,trainer.eos_token_id]]
    elif fault == "watermarking_config": trainer.generation_config.watermarking_config = {"bias":2.}
    elif fault == "model_default": trainer.model.get_base_model().generation_config.sequence_bias = {(trainer.eos_token_id,):-2.}
    else:
        with torch.no_grad(): next(trainer.ref_model.parameters()).add_(.01)
    generated = []; monkeypatch.setattr(trainer,"_generate",lambda *args: generated.append(True))
    with pytest.raises((RuntimeError,ValueError)):
        trainer._generate_and_score_completions(inputs)
    assert generated == []


@pytest.mark.parametrize("key,value",[("responses_per_step",36),("prompt_groups_per_step",4),("num_generations",1),("prompt_draw_sha256","0"*64)])
def test_declared_schedule_cannot_retain_old_four_group_or_one_sibling_aliases(tmp_path,key,value):
    common = fixture(tmp_path)
    contract = copy.deepcopy(common["contract"])
    contract["training"][key] = value
    contract["contract_sha256"] = one.digest({k:v for k,v in contract.items() if k != "contract_sha256"})
    with pytest.raises(ValueError): one.validate_training_contract(contract)


@pytest.mark.parametrize("peer_fault",[False,True])
def test_ordered_observer_collectively_rejects_missing_local_or_peer_id_before_parent_gather(monkeypatch,peer_fault):
    trainer = object.__new__(one.OneShotOrderedObservationMixin)
    entered = []
    monkeypatch.setattr(one.v1.OrderedObservationMixin,"_generate_and_score_completions",lambda *args: entered.append(True))
    def collective(fn,label):
        fn()
        if peer_fault: raise RuntimeError("simulated peer lacks instance_id")
    trainer._collective_check = collective
    with pytest.raises((ValueError,RuntimeError)):
        trainer._generate_and_score_completions([{"instance_id":"row"}] if peer_fault else [{}])
    assert not entered


def test_complete_generation_snapshot_only_normalizes_metadata_and_equivalent_single_eos():
    from transformers import GenerationConfig
    first = GenerationConfig(eos_token_id=2)
    second = copy.deepcopy(first); second.eos_token_id = [2]; second.transformers_version = "metadata-only"
    assert one.generation_config_snapshot(first) == one.generation_config_snapshot(second)
    second.eos_token_id = [2,3]
    assert one.generation_config_snapshot(first) != one.generation_config_snapshot(second)
    second = copy.deepcopy(first); second.future_processor = {"token":4}
    assert one.generation_config_snapshot(first) != one.generation_config_snapshot(second)


@pytest.mark.parametrize("text",["invalid","Route: [0, 1, 2]"])
def test_equal_scalar_group_has_zero_credit_without_best8_or_fake_winner(tmp_path,monkeypatch,text):
    common = fixture(tmp_path); trainer = one.make_cpu_trainer(**common,recipe="rloo_cost",output=tmp_path/"trainer")
    fixed(trainer,monkeypatch,[text]*9)
    batch = trainer._generate_and_score_completions(first_window(common["contract"]))
    assert torch.count_nonzero(batch["advantages"]) == 0
    loss = trainer.compute_loss(trainer.model,batch); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None)


def test_same_seed_cpu_trajectory_actor_optimizer_scheduler_rng_reference_and_tokens_are_exact(tmp_path):
    common = fixture(tmp_path)
    def run(name):
        trainer = one.make_cpu_trainer(**common,recipe="rloo_cost",output=tmp_path/name); trainer.train()
        trainer._assert_reference_integrity(full=True)
        result = {"actor":semantic_state_sha256(trainer.accelerator.unwrap_model(trainer.model).state_dict()),
            "optimizer":semantic_state_sha256(trainer.optimizer.state_dict()),"scheduler":semantic_state_sha256(trainer.lr_scheduler.state_dict()),
            "rng":recovery.rng_sha256(),"reference":trainer.reference_state_sha256(),"tokens":trainer.actual_generated_tokens,
            "windows":copy.deepcopy(trainer.actual_windows)}
        del trainer; gc.collect(); return result
    assert run("first") == run("second")


def test_one_greedy_call_per_instance_overrides_saved_sampling_defaults(tmp_path,monkeypatch):
    common = fixture(tmp_path)
    model = v2.loaded_model(common["base"],common["sft"],torch.device("cpu"),trainable=False,dtype=torch.float32)
    model.generation_config.repetition_penalty = 9.; model.generation_config.top_p = .1
    generate = model.generate; calls = []
    def observed(*args,**kwargs):
        cfg = kwargs["generation_config"]
        assert not cfg.do_sample and cfg.num_return_sequences == cfg.num_beams == 1
        assert cfg.repetition_penalty == cfg.top_p == 1. and cfg.top_k == 0
        assert kwargs["use_model_defaults"] is False
        calls.append(True); return generate(*args,**kwargs)
    monkeypatch.setattr(model,"generate",observed)
    output = tmp_path/"eval"; output.mkdir()
    before = semantic_state_sha256(model.state_dict())
    rows = one.collect_greedy_outputs(model,common["tokenizer"],common["contract"],output)
    assert len(calls) == len(rows) == len(common["contract"]["validation_rows"])
    assert all(row["generation_calls"] == 1 and row["sample_index"] == 0 and row["endpoint"] == one.PRIMARY for row in rows)
    assert semantic_state_sha256(model.state_dict()) == before


@pytest.mark.parametrize("task",["op","cvrp"])
@pytest.mark.parametrize("recipe",["rloo_cost","grpo_cost","foarl_cost_changed","foarl_released","sft"])
def test_actual_tiny_end_to_end_terminal_reload_and_unassisted_greedy(tmp_path,task,recipe):
    common = fixture(tmp_path,task)
    report = one.run_cpu_arm(**common,recipe=recipe,output=tmp_path/"out")
    proof = json.loads((tmp_path/"out/one_shot_training_complete.json").read_bytes())
    reload = json.loads((tmp_path/"out/one_shot_checkpoint_reload.json").read_bytes())
    assert report["evaluation_responses"] == len(common["contract"]["validation_rows"])
    assert report["primary_endpoint"] == one.PRIMARY and report["checkpoint_reload_exact"]
    assert reload["bitwise_equal"] and reload["whole_actor_sha256"] == proof["whole_actor_sha256"]
    assert report["RL_responses"] == (0 if recipe == "sft" else 18)
    assert not report["superiority_established"]
    if recipe != "sft":
        assert proof["reference_integrity"] and proof["initial_reference_audit"]["bitwise_equal"]
        assert proof["prepared_reference_audit"]["bitwise_equal"]
        assert proof["effective_checkpointing"]["effective_use_reentrant"] is False
        assert proof["actor_changed"] and len(proof["actual_windows"]) == 2
