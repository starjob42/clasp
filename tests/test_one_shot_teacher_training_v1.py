"""Actual tiny CPU teacher CE integration, not model-quality evidence."""
import copy
import gc
import importlib.util
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")
from clasp.entrypoints import train_sft_anchor_one_shot_v1 as core
from clasp.entrypoints import train_sft_anchor_pilot_v2 as v2
from clasp.models.external_baseline_state import semantic_state_sha256

PATH = Path(__file__).resolve().parents[1]/"experiments/rloo_one_shot_20261016/train_teacher_candidate_v1.py"
spec = importlib.util.spec_from_file_location("one_shot_teacher_training_candidate_v1",PATH)
t = importlib.util.module_from_spec(spec);spec.loader.exec_module(t)


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    torch.set_num_threads(1);monkeypatch.setenv("WORLD_SIZE","1");monkeypatch.setenv("RANK","0")
    monkeypatch.delenv("LOCAL_RANK",raising=False)
    keys = set(core.reward_environment("foarl"))|set(core.reward_environment("rloo_quality"))
    before = {k:os.environ.get(k) for k in keys}
    yield
    for k,v in before.items():
        if v is None:os.environ.pop(k,None)
        else:os.environ[k] = v


def fixture(path,task="op",horizon=3):
    base,sft = path/"base",path/"sft"
    model,tokenizer,train,validation = v2.tiny_peft_initializer(task,base,sft,torch.device("cpu"));del model;gc.collect()
    contract = core.make_smoke_contract(task,{"id":"actual-tiny-PEFT-teacher-variant",
        "sha256":core.identity(sft/"adapter_model.safetensors")["sha256"]},train,validation,
        horizon=horizon,groups_per_update=1)
    engine = t.engine_module();excluded = {key:[] for key in engine.SPLITS};ex_graph = copy.deepcopy(excluded)
    excluded["validation"] = sorted(row["semantic_id"] for row in contract["validation_rows"])
    ex_graph["validation"] = sorted(row["graph_semantic_id"] for row in contract["validation_rows"])
    admission = engine.make_training_admission(task,contract["training_rows"],
        allowed_training_semantic_ids=sorted(row["semantic_id"] for row in contract["training_rows"]),
        excluded_semantic_ids_by_split=excluded,excluded_graph_ids_by_split=ex_graph,
        training_dataset_sha256=core.digest(contract["training_rows"]))
    context = {"split":"training","operation":"supervised_training_target","inference_search":False,
               "training_dataset_sha256":admission["training_dataset_sha256"]}
    bundle = t.prepare_teacher_bundle(contract,admission=admission,expected_admission_sha256=admission["admission_sha256"],context=context)
    return dict(contract=contract,tokenizer=tokenizer,base=base,sft=sft,teacher_bundle=bundle,
                expected_teacher_bundle_sha256=bundle["bundle_sha256"])


def first_window(contract):
    identifier = contract["training"]["prompt_draws"][0][0]
    row = next(r for r in contract["training_rows"] if r["instance_id"] == identifier)
    return [{**row,"answer":row["output"],"instance_json":json.dumps(row["instance_json"])} for _ in range(9)]


@pytest.mark.parametrize("task",["op","cvrp"])
@pytest.mark.parametrize("recipe",list(t.RECIPES))
def test_actual_three_update_teacher_training_ref_gc_terminal_reload_and_cost_closure(tmp_path,task,recipe):
    common = fixture(tmp_path,task)
    before = copy.deepcopy(common["teacher_bundle"])
    proof = t.run_cpu_training(**common,recipe=recipe,output=tmp_path/"out")
    assert proof["updates"] == proof["extra_CE_forward_calls"] == 3
    assert proof["RL_responses"] == 27 and proof["extra_CE_target_tokens_including_EOS"] == 6
    assert proof["extra_CE_forward_tokens"] == 9 and proof["alpha"] == .1
    assert proof["reference_integrity"] and proof["GC"]["effective_use_reentrant"] is False
    assert proof["checkpoint_reload_exact"] and proof["actor_changed"]
    assert proof["shared_teacher_CPU_search_calls"] == 4 and proof["evaluation_model_calls"] == 0
    assert [row["step"] for row in proof["teacher_records"]] == [0,1,2]
    assert [row["instance_id"] for row in proof["teacher_records"]] == [row[0] for row in common["contract"]["training"]["prompt_draws"]]
    assert all(row["teacher_forward_calls"] == 1 and row["masked_prompt_tokens"] == 1 and
               not row["target_truncated"] and row["native_RL_loss_already_divided_by"] == 9 for row in proof["teacher_records"])
    assert common["teacher_bundle"] == before
    assert proof["variant"] == recipe+"_plus_teacher_ce" and not proof["learned_gain_established"]


@pytest.mark.parametrize("recipe",list(t.RECIPES))
def test_actual_nine_microbatch_gradient_adds_exactly_alpha_ce_once_not_alpha_over9_or9alpha(tmp_path,monkeypatch,recipe):
    common = fixture(tmp_path,horizon=1)
    trainer = t.make_cpu_trainer(**common,recipe=recipe,output=tmp_path/"trainer")
    reply = "Route: [0, 1, 2]"
    completion = trainer.processing_class.encode(reply,add_special_tokens=False)
    assert len(completion) == 1
    def generate(prompts,images):
        return [trainer.processing_class.encode(p,add_special_tokens=False) for p in prompts],[completion]*9,torch.tensor(9),None,{}
    monkeypatch.setattr(trainer,"_generate",generate)
    trainer.model.train();trainer.current_gradient_accumulation_steps = 9
    full = trainer._generate_and_score_completions(first_window(common["contract"]))
    assert torch.count_nonzero(full["advantages"]) == 0
    tokens = trainer._teacher_tokens;params = tuple(p for p in trainer.model.parameters() if p.requires_grad)
    logits = trainer.model(input_ids=tokens["input_ids"],attention_mask=tokens["attention_mask"],use_cache=False).logits
    independently_masked_ce = torch.nn.functional.cross_entropy(logits[:,:-1,:].float().reshape(-1,logits.shape[-1]),
        tokens["labels"][:,1:].reshape(-1),ignore_index=-100)
    expected = torch.autograd.grad(t.ALPHA*independently_masked_ce,params,allow_unused=True)
    for index in range(9):
        micro = {k:(v[index:index+1] if isinstance(v,torch.Tensor) and v.ndim and v.shape[0] == 9 else v) for k,v in full.items()}
        loss = trainer.compute_loss(trainer.model,micro)
        trainer.accelerator.backward(loss)
    assert trainer.teacher_forward_calls == len(trainer.teacher_records) == 1 and trainer._teacher_micro == 9
    assert sum(p.grad.abs().sum().item() for p in params if p.grad is not None) > 0
    for actual,reference in zip(params,expected):
        if reference is None:assert actual.grad is None
        else:torch.testing.assert_close(actual.grad,reference,atol=2e-7,rtol=3e-6)
    assert trainer.accelerator.gradient_accumulation_steps == 1 and trainer.compute_loss_func is not None
    trainer._assert_reference_integrity(full=True)


@pytest.mark.parametrize("recipe",list(t.RECIPES))
def test_nonzero_native_rl_gradient_plus_single_teacher_gradient_is_additive(tmp_path,monkeypatch,recipe):
    common = fixture(tmp_path,horizon=1)
    trainer = t.make_cpu_trainer(**common,recipe=recipe,output=tmp_path/"trainer")
    replies = ["Route: [0, 1, 2]"]*3+["Route: [0, 1]"]*4+["Route: [0, 1, 1]"]*2
    ids = [trainer.processing_class.encode(text,add_special_tokens=False) for text in replies]
    assert all(len(row) == 1 for row in ids)
    def generate(prompts,images):
        return [trainer.processing_class.encode(p,add_special_tokens=False) for p in prompts],ids,torch.tensor(9),None,{}
    monkeypatch.setattr(trainer,"_generate",generate)
    trainer.model.train();trainer.current_gradient_accumulation_steps = 9
    full = trainer._generate_and_score_completions(first_window(common["contract"]))
    assert torch.count_nonzero(full["advantages"]) > 0
    logps,_ = trainer._get_per_token_logps_and_entropies(trainer.model,
        torch.cat([full["prompt_ids"],full["completion_ids"]],1),
        torch.cat([full["prompt_mask"],full["completion_mask"]],1),1)
    native_rl = -(logps.squeeze(1)*full["advantages"]).mean()
    ce = t.teacher_cross_entropy(trainer.model,trainer._teacher_tokens)
    params = tuple(p for p in trainer.model.parameters() if p.requires_grad)
    expected = torch.autograd.grad(native_rl+t.ALPHA*ce,params,allow_unused=True)
    for index in range(9):
        micro = {k:(v[index:index+1] if isinstance(v,torch.Tensor) and v.ndim and v.shape[0] == 9 else v) for k,v in full.items()}
        trainer.accelerator.backward(trainer.compute_loss(trainer.model,micro))
    for actual,reference in zip(params,expected):
        if reference is None:assert actual.grad is None
        else:torch.testing.assert_close(actual.grad,reference,atol=3e-7,rtol=5e-6)
    assert trainer.teacher_forward_calls == 1


def test_both_variants_consume_the_identical_shared_teacher_bundle(tmp_path):
    common = fixture(tmp_path,horizon=1)
    first = t.make_cpu_trainer(**common,recipe="rloo_cost",output=tmp_path/"rloo")
    second = t.make_cpu_trainer(**common,recipe="foarl_released",output=tmp_path/"foarl")
    assert first.teacher_bundle_sha256 == second.teacher_bundle_sha256 == common["teacher_bundle"]["bundle_sha256"]
    assert first.teacher_bundle == second.teacher_bundle
    assert first.beta == second.beta == .05 and first.loss_type == "dr_grpo" and second.loss_type == "grpo"
    assert first.scale_rewards == "none" and second.scale_rewards == "group"
    assert second.reward_funcs == list(core.FOARL_F638_REWARD_FUNCS)


@pytest.mark.parametrize("fault",["target","prompt","instance_id","admission","context","source","alpha"])
def test_teacher_bundle_corruption_and_nontraining_context_rejected_before_model_loading(tmp_path,monkeypatch,fault):
    common = fixture(tmp_path,horizon=1);bundle = copy.deepcopy(common["teacher_bundle"])
    if fault == "target":bundle["targets"][0]["target_text"] = "Route: [0,99]"
    elif fault == "prompt":bundle["targets"][0]["prompt_sha256"] = "0"*64
    elif fault == "instance_id":bundle["targets"][0]["instance_id"] = "validation_id"
    elif fault == "admission":bundle["admission"]["split"] = "test"
    elif fault == "context":bundle["request_context"]["split"] = "engineering_eval"
    elif fault == "source":bundle["engine_sha256"] = "0"*64
    else:bundle["alpha"] = .2
    bundle["bundle_sha256"] = core.digest({k:v for k,v in bundle.items() if k != "bundle_sha256"})
    loaded = [];monkeypatch.setattr(t.v2,"loaded_model",lambda *a,**kw:loaded.append(True))
    with pytest.raises((ValueError,KeyError)):
        t.make_cpu_trainer(**{**common,"teacher_bundle":bundle,"expected_teacher_bundle_sha256":bundle["bundle_sha256"]},
                           recipe="rloo_cost",output=tmp_path/"denied")
    assert not loaded and not (tmp_path/"denied").exists()


@pytest.mark.parametrize("fault",["negative_bundle_time","negative_search_time","negative_result_time","boolean_move_count","excessive_move_count","fake_model_call"])
def test_resealed_invalid_teacher_work_accounting_is_rejected(tmp_path,fault):
    common = fixture(tmp_path,horizon=1);bundle = copy.deepcopy(common["teacher_bundle"])
    result = bundle["targets"][0]["teacher_result"]
    if fault == "negative_bundle_time":bundle["bundle_generation_wall_seconds"] = -1.
    elif fault == "negative_search_time":bundle["search_wall_seconds"] = -1.
    elif fault == "negative_result_time":result["wall_seconds"] = -1.
    elif fault == "boolean_move_count":result["move_evaluations"] = True
    elif fault == "excessive_move_count":result["move_evaluations"] = 10**6
    else:result["model_calls"] = 1
    result["result_sha256"] = core.digest({k:v for k,v in result.items() if k != "result_sha256"})
    bundle["bundle_sha256"] = core.digest({k:v for k,v in bundle.items() if k != "bundle_sha256"})
    with pytest.raises(ValueError):t.validate_teacher_bundle(common["contract"],bundle,bundle["bundle_sha256"])


def test_token_mask_prompt_eos_roundtrip_and_no_truncation(tmp_path):
    common = fixture(tmp_path,horizon=1);tokenizer = common["tokenizer"]
    tokens = t.prepare_teacher_tokens(tokenizer,"x0","Route: [0, 1]",target_token_cap=2,context_limit=3,device=torch.device("cpu"))
    assert tokens["labels"].tolist() == [[-100,tokenizer.convert_tokens_to_ids("Route: [0, 1]"),tokenizer.eos_token_id]]
    assert tokens["input_ids"][0,-1].item() == tokenizer.eos_token_id
    with pytest.raises(ValueError,match="truncation is forbidden"):
        t.prepare_teacher_tokens(tokenizer,"x0","Route: [0, 1]",target_token_cap=1,context_limit=3,device="cpu")
    with pytest.raises(ValueError,match="truncation is forbidden"):
        t.prepare_teacher_tokens(tokenizer,"x0","Route: [0, 1]",target_token_cap=2,context_limit=2,device="cpu")
    with pytest.raises(ValueError,match="roundtrip"):
        t.prepare_teacher_tokens(tokenizer,"x0","unknown text",target_token_cap=3,context_limit=8,device="cpu")


def test_true_multi_token_prompt_is_masked_and_ce_matches_standard_causal_labels():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    raw = Tokenizer(WordLevel({"[UNK]":0,"[EOS]":1,"p":2,"q":3,"target":4},unk_token="[UNK]"));raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw,unk_token="[UNK]",eos_token="[EOS]")
    tokens = t.prepare_teacher_tokens(tokenizer,"p q","target",target_token_cap=2,context_limit=4,device="cpu")
    assert tokens["labels"].tolist() == [[-100,-100,4,1]]
    class LogitModel:
        def __call__(self,**kwargs):
            assert kwargs["use_cache"] is False and kwargs["logits_to_keep"] == 3
            class Output:pass
            output = Output();output.logits = logits[:,-3:];return output
    logits = torch.arange(20,dtype=torch.float32).reshape(1,4,5).requires_grad_()
    actual = t.teacher_cross_entropy(LogitModel(),tokens)
    expected = torch.nn.functional.cross_entropy(logits[:,:-1,:].reshape(-1,5),tokens["labels"][:,1:].reshape(-1),ignore_index=-100)
    torch.testing.assert_close(actual,expected)
    grad = torch.autograd.grad(actual,logits)[0]
    assert torch.count_nonzero(grad[:,0,:]) == 0 and torch.count_nonzero(grad[:,1:3,:]) > 0


def test_teacher_schedule_or_microbatch_prompt_cannot_be_rebound(tmp_path,monkeypatch):
    common = fixture(tmp_path,horizon=1);trainer = t.make_cpu_trainer(**common,recipe="rloo_cost",output=tmp_path/"trainer")
    inputs = first_window(common["contract"]);inputs[0]["prompt"] = "x7"
    generated = [];monkeypatch.setattr(trainer,"_generate",lambda *args:generated.append(True))
    with pytest.raises(ValueError,match="identity differs"):trainer._generate_and_score_completions(inputs)
    assert not generated
    trainer._teacher_step = 0;trainer._teacher_tokens = {"prompt_ids":[3]};trainer.current_gradient_accumulation_steps = 9
    trainer.model.train()
    with pytest.raises(ValueError,match="microbatch tokens"):
        trainer._compute_loss(trainer.model,{"prompt_ids":torch.tensor([[4]]),"prompt_mask":torch.tensor([[1]])})


@pytest.mark.parametrize("fault",["labels","EOS","repeat_CE"])
def test_teacher_labels_eos_and_one_forward_window_state_are_constant(tmp_path,monkeypatch,fault):
    common = fixture(tmp_path,horizon=1);trainer = t.make_cpu_trainer(**common,recipe="rloo_cost",output=tmp_path/"trainer")
    inputs = first_window(common["contract"])
    completion = trainer.processing_class.encode("Route: [0, 1, 2]",add_special_tokens=False)
    monkeypatch.setattr(trainer,"_generate",lambda prompts,images:([trainer.processing_class.encode(p,add_special_tokens=False) for p in prompts],
        [completion]*9,torch.tensor(9),None,{}))
    trainer.model.train();trainer.current_gradient_accumulation_steps = 9
    full = trainer._generate_and_score_completions(inputs)
    if fault == "labels":trainer._teacher_tokens["labels"][0,-2] = trainer.processing_class.unk_token_id
    elif fault == "EOS":trainer._teacher_tokens["input_ids"][0,-1] = trainer.processing_class.unk_token_id
    else:trainer.teacher_forward_calls = 1
    micro = {k:(v[:1] if isinstance(v,torch.Tensor) and v.ndim and v.shape[0] == 9 else v) for k,v in full.items()}
    with pytest.raises(ValueError,match="changed within its sibling window or CE was repeated"):
        trainer.compute_loss(trainer.model,micro)


@pytest.mark.parametrize("fault",["alpha","accelerator_GA","unsupported_recipe","resume","regenerate"])
def test_unsupported_scaling_recipe_and_resume_fail_closed(tmp_path,fault):
    common = fixture(tmp_path,horizon=1)
    if fault == "unsupported_recipe":
        with pytest.raises(ValueError):t.make_cpu_trainer(**common,recipe="grpo_cost",output=tmp_path/"denied")
        return
    trainer = t.make_cpu_trainer(**common,recipe="rloo_cost",output=tmp_path/"trainer")
    if fault == "alpha":
        trainer.teacher_alpha = .2
        with pytest.raises(ValueError):trainer._assert_contract()
    elif fault == "accelerator_GA":
        trainer.accelerator.gradient_accumulation_steps = 9
        with pytest.raises(ValueError):trainer._assert_contract()
    elif fault == "resume":
        with pytest.raises(ValueError,match="resume is unsupported"):trainer._load_optimizer_and_scheduler("fake_checkpoint")
    else:
        trainer._teacher_step = 0;trainer._teacher_micro = 1
        with pytest.raises(ValueError,match="incomplete or was regenerated"):
            trainer._generate_and_score_completions(first_window(common["contract"]))
