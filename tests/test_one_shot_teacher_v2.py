"""Prospective stronger-teacher/calibration mechanics; no learned gain evidence."""
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from transformers import AutoModelForCausalLM,PreTrainedTokenizerFast,Qwen2Config,Qwen2ForCausalLM
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.decoders import Fuse
from clasp.entrypoints import train_sft_anchor_one_shot_v1 as core
from clasp.entrypoints import train_sft_anchor_pilot_v2 as v2
from clasp.models.external_baseline_state import semantic_state_sha256

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/"experiments/rloo_one_shot_20261016"

def module(name):
    path = HERE/(name+".py");spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

D = module("teacher_candidate_v2");C = module("calibrate_teacher_v2");T = module("train_teacher_candidate_v2")


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    torch.set_num_threads(1);monkeypatch.setenv("WORLD_SIZE","1");monkeypatch.setenv("RANK","0");monkeypatch.delenv("LOCAL_RANK",raising=False)
    keys = set(core.reward_environment("foarl"))|set(core.reward_environment("rloo_quality"));saved = {k:os.environ.get(k) for k in keys}
    yield
    for key,value in saved.items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key] = value


def fixture(path,task):
    base,sft = path/"base",path/"sft"
    model,tok,train,validation = v2.tiny_peft_initializer(task,base,sft,torch.device("cpu"));del model;gc.collect()
    # A NEW weak-head engineering initializer, declared before any collection.
    # It gives nondegenerate full-support samples; no frozen initializer changes.
    model = AutoModelForCausalLM.from_pretrained(base)
    with torch.no_grad():model.lm_head.weight.mul_(.05)
    model.save_pretrained(base);del model;gc.collect()
    contract = core.make_smoke_contract(task,{"id":"teacher-v2-engineering-headscale005-before-collection",
        "sha256":core.identity(sft/"adapter_model.safetensors")["sha256"]},train,validation,horizon=3,groups_per_update=1,max_completion_tokens=4)
    engine = D.adapter_v1().engine_module();excluded = {key:[] for key in engine.SPLITS};graphs = copy.deepcopy(excluded)
    excluded["validation"] = sorted(row["semantic_id"] for row in contract["validation_rows"])
    graphs["validation"] = sorted(row["graph_semantic_id"] for row in contract["validation_rows"])
    admission = engine.make_training_admission(task,contract["training_rows"],
        allowed_training_semantic_ids=sorted(row["semantic_id"] for row in contract["training_rows"]),
        excluded_semantic_ids_by_split=excluded,excluded_graph_ids_by_split=graphs,training_dataset_sha256=core.digest(contract["training_rows"]))
    context = {"split":"training","operation":"supervised_training_target","inference_search":False,"training_dataset_sha256":admission["training_dataset_sha256"]}
    spec = D.make_spec(contract,[row["instance_id"] for row in contract["training_rows"][:2]])
    model = v2.loaded_model(base,sft,torch.device("cpu"),trainable=False,dtype=torch.float32)
    data = D.collect_tiny_cpu_data(contract,model,tok,admission=admission,expected_admission_sha256=admission["admission_sha256"],
        context=context,spec=spec,expected_spec_sha256=spec["spec_sha256"]);del model;gc.collect()
    targets = D.prepare_stronger_targets(contract,data,data["shared_data_sha256"],tok)
    calibrations = {recipe:C.calibrate_cpu(contract,recipe,data,data["shared_data_sha256"],targets,targets["teacher_targets_sha256"],
                    tok,base,sft,path/("calibration_"+recipe)) for recipe in C.RECIPES}
    return dict(contract=contract,tokenizer=tok,base=base,sft=sft,data=data,targets=targets,calibrations=calibrations)


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    torch.set_num_threads(1);os.environ["WORLD_SIZE"] = "1";os.environ["RANK"] = "0";os.environ.pop("LOCAL_RANK",None)
    path = tmp_path_factory.mktemp("teacher_v2")
    return {task:fixture(path/task,task) for task in ("op","cvrp")}


def kwargs(f,recipe):
    return dict(shared_data=f["data"],expected_data_sha256=f["data"]["shared_data_sha256"],teacher_targets=f["targets"],
        expected_teacher_targets_sha256=f["targets"]["teacher_targets_sha256"],calibration=f["calibrations"][recipe],
        expected_calibration_sha256=f["calibrations"][recipe]["calibration_sha256"])


@pytest.mark.parametrize("task",["op","cvrp"])
@pytest.mark.parametrize("recipe",list(C.RECIPES))
def test_actual_preclip_native9_matches_independent_variable_length_full_window_and_preserves_every_state(fixtures,task,recipe):
    f = fixtures[task];cal = f["calibrations"][recipe]
    C.validate_calibration(f["contract"],recipe,f["data"],f["targets"],cal,cal["calibration_sha256"],f["tokenizer"])
    assert cal["alpha"] > 0 and cal["rho"] == 1. and len(cal["records"]) == 2
    assert cal["scope"] == cal["model_kind"] == "actual_tiny_cpu_only"
    assert all(row["actor_reference_optimizer_scheduler_unchanged"] and row["mode_and_RNG_restored"] and
               row["all_grads_cleared"] and row["optimizer_updates"] == 0 and row["native_RL_microbatches"] == 9 and
               row["initial_KL_gradient_norm"] <= 1e-6 and row["restored_active_CUDA_RNG_devices"] == [] for row in cal["records"])
    assert any(len(set(row["completion_lengths"])) > 1 for row in cal["records"])
    assert all(row["independent_policy_gradient_max_abs_error"] < 1e-6 for row in cal["records"])
    assert all(row["teacher_target_tokens_including_Objective_and_EOS"] >= 2 for row in cal["records"])
    names = [r["name"] for r in cal["parameter_inventory"]]
    assert any(name.endswith(".bias") for name in names) and any(".lora_A." in name for name in names)


@pytest.mark.parametrize("task",["op","cvrp"])
@pytest.mark.parametrize("recipe",list(C.RECIPES))
def test_actual_three_update_streamed_rl_plus_ce_ref_gc_full_target_and_typed_reload(fixtures,tmp_path,task,recipe):
    f = fixtures[task]
    result = T.run_cpu_training(f["contract"],recipe,tmp_path/"train",f["tokenizer"],f["base"],f["sft"],**kwargs(f,recipe))
    assert result["updates"] == result["teacher_CE_forwards"] == 3 and result["RL_responses"] == 27
    assert result["reference_integrity"] and result["GC"]["effective_use_reentrant"] is False
    assert result["checkpoint_reload_exact"] and result["actor_changed"] and result["evaluation_model_calls"] == 0
    assert all(row["after_native_RL_backward_microbatch"] == 9 and row["CE_RNG_restored"] and
               not row["CE_target_truncated"] and row["target_tokens_including_Objective_and_EOS"] >= 2 for row in result["teacher_records"])
    assert [row["instance_id"] for row in result["teacher_records"]] == [row[0] for row in f["contract"]["training"]["prompt_draws"]]


@pytest.mark.parametrize("task",["op","cvrp"])
def test_teacher_only_control_has_no_rl_path_and_same_objective_eos_targets(fixtures,tmp_path,monkeypatch,task):
    f = fixtures[task]
    def forbidden(*args,**kwargs):raise AssertionError("RL path must not be called by teacher-only control")
    monkeypatch.setattr(core.OneShotRLOOTrainer,"_generate_and_score_completions",forbidden)
    monkeypatch.setattr(core,"strict_cost_reward",forbidden)
    result = T.run_teacher_only_cpu(f["contract"],f["data"],f["targets"],f["tokenizer"],f["base"],f["sft"],tmp_path/"only")
    assert result["updates"] == result["CE_forwards"] == 3 and result["RL_model_calls"] == result["RL_reward_calls"] == 0
    assert not result["reference_KL"] and result["CE_multiplier"] == 1. and result["checkpoint_reload_exact"]
    assert all(row["target_tokens"] >= 2 for row in result["records"])


@pytest.mark.parametrize("recipe",list(C.RECIPES))
def test_streamed_nine_backward_gradient_is_exact_sum_rl_plus_calibrated_ce_without_concurrent_graph(fixtures,tmp_path,recipe):
    f = fixtures["op"];trainer = T.make_cpu_trainer(f["contract"],recipe,tmp_path/recipe,f["tokenizer"],f["base"],f["sft"],**kwargs(f,recipe))
    group = copy.deepcopy(f["data"]["calibration_groups"][0]);identifier = trainer.pilot_contract["training"]["prompt_draws"][0][0]
    # A real nonzero, variable-length diagnostic group, tied to actual first prompt.
    row = next(r for r in f["contract"]["training_rows"] if r["instance_id"] == identifier)
    replies = [row["output"]]*3+["Route: [0, 1]"]*3+["Route: [0, 1, 1]"]*3
    ids = [f["tokenizer"].encode(reply,add_special_tokens=False)+([f["tokenizer"].eos_token_id] if i%2 else []) for i,reply in enumerate(replies)]
    trainer._generate = lambda prompts,images:([f["tokenizer"].encode(p,add_special_tokens=False) for p in prompts],ids,torch.tensor(sum(map(len,ids))),None,{})
    inputs = [{**row,"answer":row["output"],"instance_json":json.dumps(row["instance_json"])} for _ in range(9)]
    trainer.model.train();trainer.current_gradient_accumulation_steps = 9
    full = trainer._generate_and_score_completions(inputs);trainer.calibration_batch = full
    inventory = C.parameter_inventory(trainer.model);pairs = C.parameter_pairs(trainer.model,inventory)
    policy,kl = C.policy_and_KL_gradients(trainer,inventory);assert C.finite_norm(policy) > 0
    util = D.adapter_v1();ce = util.teacher_cross_entropy(trainer.model,trainer._teacher_tokens)
    teacher = C.gradient_vector(pairs,torch.autograd.grad(ce,[p for _,p in pairs],allow_unused=True))
    expected = policy+kl+trainer.teacher_alpha*teacher
    for index in range(9):
        micro = {k:(v[index:index+1] if isinstance(v,torch.Tensor) and v.ndim and v.shape[0] == 9 else v) for k,v in full.items()}
        loss = trainer.compute_loss(trainer.model,micro);trainer.accelerator.backward(loss)
        assert trainer.teacher_forward_calls == 0
        if index < 8:assert trainer._teacher_backward_if_due(trainer.model) is None
    assert trainer.teacher_forward_calls == 0
    before = D.rng_sha(D.rng_state());trainer._teacher_backward_if_due(trainer.model)
    assert before == D.rng_sha(D.rng_state()) and trainer.teacher_forward_calls == 1
    torch.testing.assert_close(C.gradient_vector(pairs),expected,rtol=1e-4,atol=1e-7)
    with pytest.raises(ValueError,match="once"):trainer._teacher_backward_if_due(trainer.model)


def test_zeros_stay_in_fixed_median_no_drop_no_extra_factor9():
    result = C.choose_alpha([0.,0.,2.,4.,6.,8.,10.,12.],[0.,1.,2.,3.,4.,5.,6.,7.],rho=1.)
    assert result["alpha"] == 5/3.5 and result["zero_policy_prompt_count"] == 2
    assert result["zero_teacher_prompt_count"] == 1
    assert C.finite_norm(torch.zeros(3),positive=False) == 0.


@pytest.mark.parametrize("rl,ce,rho",[([],[],1.),([0.,0.,1.],[1.,1.,1.],1.),([1.,1.,1.],[0.,0.,1.],1.),
    ([float('nan')],[1.],1.),([1.],[float('inf')],1.),([True],[1.],1.),([1.],[1.],True),([1.],[1.],1),([1.],[1.],.1),([-1.],[1.],1.)])
def test_degenerate_nonfinite_boolean_or_changed_rho_is_rejected(rl,ce,rho):
    with pytest.raises(ValueError):C.choose_alpha(rl,ce,rho=rho)


@pytest.mark.parametrize("value",[torch.tensor([float('nan')]),torch.tensor([float('inf')]),torch.tensor([],dtype=torch.float64),torch.tensor([1e308],dtype=torch.float64)])
def test_gradient_finiteness_and_fp64_overflow_reject(value):
    with pytest.raises(ValueError):C.finite_norm(value)


@pytest.mark.parametrize("fault",["fourth_token","mask","sample_index","seed_bool","subset_order","group_drop","heldout","text"])
def test_fixed_shared_data_corruption_rejected_before_any_model_call(fixtures,fault):
    f = fixtures["op"];data = copy.deepcopy(f["data"])
    if fault == "fourth_token":data["calibration_groups"][0]["records"][3]["completion_token_ids"][0] = True
    elif fault == "mask":data["calibration_groups"][0]["records"][3]["completion_mask"][0] = 0
    elif fault == "sample_index":data["calibration_groups"][0]["records"][3]["sample_index"] = 4
    elif fault == "seed_bool":data["spec"]["seed"] = True;data["spec"]["spec_sha256"] = D.seal(data["spec"],"spec_sha256");data["spec_sha256"] = data["spec"]["spec_sha256"]
    elif fault == "subset_order":data["calibration_groups"].reverse()
    elif fault == "group_drop":data["calibration_groups"].pop()
    elif fault == "heldout":data["context"]["split"] = "validation"
    else:data["calibration_groups"][0]["records"][3]["text"] += " forged"
    data["shared_data_sha256"] = D.seal(data,"shared_data_sha256")
    with pytest.raises(ValueError):D.validate_shared_data(f["contract"],data,data["shared_data_sha256"],f["tokenizer"])


@pytest.mark.parametrize("fault",["alpha","norm","actor","inventory","source","recipe","scope"])
def test_calibration_arithmetic_actor_inventory_source_and_model_kind_cannot_be_rebound(fixtures,fault):
    f = fixtures["op"];cal = copy.deepcopy(f["calibrations"]["rloo_cost"])
    if fault == "alpha":cal["alpha"] *= 2
    elif fault == "norm":cal["records"][0]["native_policy_gradient_norm"] *= 3
    elif fault == "actor":cal["calibration_actor_identity"]["whole_actor_sha256"] = "0"*64
    elif fault == "inventory":cal["parameter_inventory"] = cal["parameter_inventory"][:-1]
    elif fault == "source":cal["source_sha256"] = "0"*64
    elif fault == "recipe":cal["recipe"] = "foarl_released"
    else:cal["scope"] = "production_training_collection_7b"
    cal["calibration_sha256"] = D.seal(cal,"calibration_sha256")
    with pytest.raises(ValueError):C.validate_calibration(f["contract"],"rloo_cost",f["data"],f["targets"],cal,cal["calibration_sha256"],f["tokenizer"])


def test_multitoken_objective_eos_ce_equals_independent_full_masked_reference():
    text = "Route: [0, 1], Objective: 3.00000000";prompt = "pfx"
    chars = sorted(set(text+prompt));vocab = {"[PAD]":0,"[UNK]":1,"[EOS]":2,**{c:i+3 for i,c in enumerate(chars)}}
    raw = Tokenizer(WordLevel(vocab,unk_token="[UNK]"));raw.pre_tokenizer = Split("",behavior="isolated");raw.decoder = Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw,pad_token="[PAD]",unk_token="[UNK]",eos_token="[EOS]")
    util = D.adapter_v1();tokens = util.prepare_teacher_tokens(tok,prompt,text,target_token_cap=100,context_limit=128,device=torch.device("cpu"))
    assert tokens["target_tokens"] == len(text)+1 and tokens["target_ids"][-1] == tok.eos_token_id
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok),hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,attention_dropout=0.))
    logits = model(input_ids=tokens["input_ids"],attention_mask=tokens["attention_mask"],use_cache=False).logits
    explicit = torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,len(tok)),tokens["labels"][:,1:].reshape(-1),ignore_index=-100)
    actual = util.teacher_cross_entropy(model,tokens);torch.testing.assert_close(actual,explicit)
    assert bool((tokens["labels"][:,:len(prompt)] == -100).all())
    with pytest.raises(ValueError,match="truncation"):util.prepare_teacher_tokens(tok,prompt,text,target_token_cap=3,context_limit=128,device=torch.device("cpu"))


def test_all_actual_trainables_in_norm_inventory_unexpected_base_or_modules_to_save_denied(fixtures):
    f = fixtures["op"];model = v2.loaded_model(f["base"],f["sft"],torch.device("cpu"),trainable=True,dtype=torch.float32)
    inventory = C.parameter_inventory(model);assert len(inventory) == sum(p.requires_grad for p in model.parameters())
    model.get_input_embeddings().weight.requires_grad_(True)
    with pytest.raises(ValueError,match="Unexpected"):C.parameter_inventory(model)
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError):C.gradient_vector([("test",p)],[torch.ones(3)])
    with pytest.raises(ValueError):C.gradient_vector([("test",p)],[torch.tensor([1.,float('nan')])])


def test_strict_better_sft_incumbent_and_witness_fallback_physical_retention(fixtures):
    f = fixtures["op"];row = f["contract"]["training_rows"][0]
    _,score,origin = D.select_incumbent("op",row["instance_json"],row["output"],"malformed")
    assert origin == "training_witness" and score["feasible"]
    for row,target in zip(f["contract"]["training_rows"],f["targets"]["targets"]):
        assert ", Objective: " in target["target_text"]
        final = core._raw_score("op",target["target_text"],row["instance_json"]) if hasattr(core,"_raw_score") else D._raw_score("op",target["target_text"],row["instance_json"])
        assert final["feasible"] and final["objective"] >= target["incumbent_objective"]


def test_zero_reward_group_is_captured_as_zero_and_kept_not_resampled(fixtures,tmp_path):
    f = fixtures["op"];group = copy.deepcopy(f["data"]["calibration_groups"][0])
    row = next(row for row in f["contract"]["training_rows"] if row["instance_id"] == group["instance_id"])
    cfg = core.generation_config(f["tokenizer"],4,sample=True)
    group["records"] = [D._record(row,[f["tokenizer"].eos_token_id],f["tokenizer"],cfg,
        role="gradient_calibration",index=index,count=9,group_index=0) for index in range(9)]
    target = next(target for target in f["targets"]["targets"] if target["instance_id"] == group["instance_id"])
    trainer = C.make_calibration_trainer(C._prompt_contract(f["contract"],group["instance_id"]),"rloo_cost",group,
        f["tokenizer"],f["base"],f["sft"],tmp_path/"zero")
    record,native,teacher = C.capture_prompt_gradient(trainer,group,target,f["tokenizer"],f["data"]["SFT_identity"],contract=f["contract"])
    assert record["native_policy_gradient_norm"] == C.finite_norm(native,positive=False) == 0.
    assert C.finite_norm(teacher) > 0 and trainer.replay_calls == 1 and trainer.state.global_step == 0
    with pytest.raises(ValueError,match="median"):C.choose_alpha([0.,0.],[1.,2.],rho=1.)


def test_replaying_same_actual_fixed_prompt_is_deterministic_without_rng_or_actor_updates(fixtures,tmp_path):
    f = fixtures["op"];group = f["data"]["calibration_groups"][0]
    target = next(target for target in f["targets"]["targets"] if target["instance_id"] == group["instance_id"])
    with D.isolated_cpu_seed(f["data"]["spec"]["calibration_seed"]):
        record,rl,ce = C.calibrate_prompt(f["contract"],"rloo_cost",group,target,f["tokenizer"],f["base"],f["sft"],tmp_path/"again",f["data"]["SFT_identity"])
    assert record == f["calibrations"]["rloo_cost"]["records"][0]


def test_cpu_receipt_cannot_be_relabelled_as_real7b_or_use_unadmitted_collection(fixtures):
    f = fixtures["op"]
    envelope = {"protocol":"one-shot-training-collection-admission-v1","kind":"synthetic_training_collection_fixture",
                "synthetic_fixture":True,"actual_7b_production_collection_admitted":True}
    envelope["admission_sha256"] = core.digest(envelope)
    with pytest.raises(ValueError,match="production7B"):
        D.import_production_shared_data(f["contract"],envelope,envelope["admission_sha256"],f["tokenizer"])
    model = v2.loaded_model(f["base"],f["sft"],torch.device("cpu"),trainable=True,dtype=torch.float32)
    with torch.no_grad():next(p for p in model.parameters() if p.requires_grad).add_(.0001)
    cal = f["calibrations"]["rloo_cost"]
    with pytest.raises(ValueError,match="initial model"):
        C.validate_calibration(f["contract"],"rloo_cost",f["data"],f["targets"],cal,cal["calibration_sha256"],f["tokenizer"],model=model)


def test_7b_capture_source_has_no_full_b9_gradient_graph_branch():
    import inspect,ast
    source = inspect.getsource(C.capture_prompt_gradient);tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id == "policy_and_KL_gradients"]
    assert len(calls) == 1
    conditions = [node for node in ast.walk(tree) if isinstance(node,ast.If) and ast.unparse(node.test) == "not cuda_devices"]
    assert any(calls[0] in list(ast.walk(node)) for node in conditions)


@pytest.mark.parametrize("fault",["incumbent","work_cap","model_calls","total_moves","engine_source"])
def test_teacher_physics_proof_source_incumbent_and_cpu_work_cannot_be_rebound(fixtures,fault):
    f = fixtures["op"];targets = copy.deepcopy(f["targets"]);result = targets["targets"][0]["teacher_result"]
    if fault == "incumbent":result["incumbent_text_sha256"] = "0"*64
    elif fault == "work_cap":result["move_evaluations"] = result["limits"]["max_move_evaluations"]+1
    elif fault == "model_calls":result["model_calls"] = True
    elif fault == "total_moves":targets["search_move_evaluations"] += 1
    else:targets["teacher_engine_sha256"] = "0"*64
    result["result_sha256"] = D.seal(result,"result_sha256");targets["teacher_targets_sha256"] = D.seal(targets,"teacher_targets_sha256")
    with pytest.raises(ValueError):D.validate_targets(f["contract"],f["data"],targets,targets["teacher_targets_sha256"],f["tokenizer"])


def test_best_strict_sft_training_candidate_beats_witness_before_search(fixtures):
    f = fixtures["op"];row = f["contract"]["training_rows"][0]
    _,score,origin = D.select_incumbent("op",row["instance_json"],"Route: [0, 1]",row["output"])
    witness = D._raw_score("op","Route: [0, 1]",row["instance_json"])
    assert origin == "original_SFT_greedy" and score["feasible"] and score["objective"] > witness["objective"]


def test_modules_to_save_bias_and_boolean_gradient_vector_are_not_calibrated_inventory():
    class Fake:
        def named_parameters(self):
            return [("x.lora_A.default.weight",torch.nn.Parameter(torch.ones(2,2))),
                    ("x.lora_B.default.weight",torch.nn.Parameter(torch.ones(2,2))),
                    ("x.modules_to_save.default.bias",torch.nn.Parameter(torch.ones(2)))]
    with pytest.raises(ValueError,match="modules_to_save"):C.parameter_inventory(Fake())
    with pytest.raises(ValueError):C.finite_norm(torch.tensor([True,False]))
    with pytest.raises(ValueError):C.gradient_vector([("x",torch.nn.Parameter(torch.ones(2)))],[1.])
