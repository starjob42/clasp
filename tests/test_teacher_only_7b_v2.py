"""Teacher-only guards and actual tiny CPU CE; no policy generation/GPU."""
import copy
import hashlib
import json
import os
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/rloo_one_shot_20261016"


def module(path, name):
    raw = path.read_bytes();value = types.ModuleType(name)
    value.__file__ = str(path);value.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    exec(compile(raw, str(path), "exec", dont_inherit=True), value.__dict__)
    return value


M = module(HERE / "teacher_only_7b_v2.py", "teacher_only_helper_test")


def test_import_is_stdlib_only_in_a_fresh_interpreter():
    import subprocess, sys
    raw = (HERE / "teacher_only_7b_v2.py").read_bytes()
    script = "import sys;exec(compile(" + repr(raw) + ",'<source>','exec'),{'__name__':'safe_import'});assert not any(n in sys.modules for n in ('torch','transformers','clasp','trl'))"
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_old_executed_source_cannot_rebind_changed_disk(tmp_path):
    path = tmp_path / "helper.py";path.write_bytes((HERE / "teacher_only_7b_v2.py").read_bytes())
    loaded = module(path, "earlier_teacher_source")
    path.write_bytes(path.read_bytes() + b"\n# later disk bytes\n")
    with pytest.raises(ValueError, match="externally pinned"):
        loaded._self_source()


@pytest.fixture
def production_context(tmp_path, monkeypatch):
    roster = json.loads((HERE / "execution_v1/execution_roster.json").read_bytes())
    runner = module(Path(roster["runner"]["path"]), "teacher_only_guard_runtime")
    manifest = json.loads(Path(roster["source_manifest"]["path"]).read_bytes());source = Path(manifest["root"])
    def source_stub(path):
        return types.SimpleNamespace(__file__=str(path), __executed_source_sha256__=hashlib.sha256(path.read_bytes()).hexdigest())
    core = source_stub(source / "src/clasp/entrypoints/train_sft_anchor_one_shot_v1.py")
    core.one_shot_quality = source_stub(source / "src/clasp/rewards/one_shot_quality.py")
    core.validate_training_contract = lambda value: value
    teacher = source_stub(HERE / "train_teacher_candidate_v2.py")
    teacher.PROTOCOL = "one-shot-calibrated-streamed-teacher-ce-v2"
    teacher.calibration_module = lambda: pytest.fail("Imported calibration before rejecting invalid outer context")
    contract = {"scope":"prospective_teacher_development","task":"op","train_seed":73,
                "training":{"horizon":100,"groups_per_update":1,"max_completion_tokens":3000,"learning_rate":1e-5},
                "training_rows":[{} for _ in range(128)]}
    kwargs = dict(base_runtime=runner,core=core,teacher_module=teacher,logical_contract=contract,
        actor=object(),tokenizer=object(),gpu_owner={"pid":os.getpid(),"uid":os.getuid(),"gpu_uuid":runner.GPU_UUID,
        "all_locks_held_until_GPU_lifetime_closed":True},expected_actor_sha256="1"*64,output=tmp_path,
        shared_data={"scope":"production_training_collection_7b","SFT_identity":{"whole_actor_sha256":"1"*64}},
        expected_data_sha256="2"*64,teacher_targets={},expected_teacher_targets_sha256="3"*64,
        local_execution_contract_sha256="4"*64,source_manifest_sha256="5"*64)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES",runner.GPU_UUID)
    monkeypatch.setenv("WORLD_SIZE","1");monkeypatch.setenv("RANK","0");monkeypatch.setenv("LOCAL_RANK","0")
    return kwargs


@pytest.mark.parametrize("field,value", [("task","cvrp"),("scope","system_smoke"),("train_seed",True)])
def test_wrong_production_contract_denied_before_actor_or_calibration(production_context, field, value):
    production_context["logical_contract"][field] = value
    with pytest.raises(ValueError):M.run_teacher_only(**production_context)


@pytest.mark.parametrize("field,value", [("horizon",99),("groups_per_update",4),("max_completion_tokens",2999),("learning_rate",2e-5)])
def test_changed_training_budget_denied_before_actor(production_context, field, value):
    production_context["logical_contract"]["training"][field] = value
    with pytest.raises(ValueError):M.run_teacher_only(**production_context)


@pytest.mark.parametrize("field,value", [("pid",True),("pid",999999),("uid",True),("uid",999999),("gpu_uuid","another"),("all_locks_held_until_GPU_lifetime_closed",False)])
def test_wrong_current_owner_denied_before_actor(production_context, field, value):
    production_context["gpu_owner"][field] = value
    with pytest.raises(ValueError):M.run_teacher_only(**production_context)


def test_tiny_shared_data_cannot_promote(production_context):
    production_context["shared_data"]["scope"] = "actual_tiny_cpu_only"
    with pytest.raises(ValueError,match="production training data"):M.run_teacher_only(**production_context)


@pytest.fixture
def tiny():
    import torch
    from transformers import PreTrainedTokenizerFast,Qwen2Config,Qwen2ForCausalLM
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from tokenizers.decoders import Fuse
    from peft import LoraConfig,get_peft_model
    from clasp.entrypoints import train_sft_anchor_pilot_v2 as anchor
    from clasp.models.external_baseline_state import semantic_state_sha256
    torch.set_num_threads(1);torch.manual_seed(7)
    text="Route: [0, 1], Objective: 3.00000000";prompt="pfx"
    chars=sorted(set(text+prompt));vocab={"[PAD]":0,"[UNK]":1,"[EOS]":2,**{c:i+3 for i,c in enumerate(chars)}}
    backend=Tokenizer(WordLevel(vocab,unk_token="[UNK]"));backend.pre_tokenizer=Split("",behavior="isolated");backend.decoder=Fuse()
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token="[PAD]",unk_token="[UNK]",eos_token="[EOS]")
    actor=Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok),hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,attention_dropout=0.))
    actor=get_peft_model(actor,LoraConfig(r=2,lora_alpha=4,target_modules=["q_proj","v_proj"],bias="all",lora_dropout=0.,task_type="CAUSAL_LM"))
    cal=module(HERE/"calibrate_teacher_v2.py","actual_tiny_teacher_calibration_parameters")
    util=module(HERE/"train_teacher_candidate_v1.py","frozen_complete_teacher_CE_utilities")
    rows=[{"instance_id":s,"semantic_id":hashlib.sha256(s.encode()).hexdigest(),"prompt":prompt} for s in ("a","b")]
    targets=[{"instance_id":s,"target_text":text,"target_text_sha256":hashlib.sha256(text.encode()).hexdigest()} for s in ("a","b")]
    args=anchor._training_args({"scope":"system_smoke","train_seed":73,"training":{"horizon":3,"learning_rate":1e-5,"max_completion_tokens":100}},
                              "rloo_scalar_beta005",Path("/tmp/teacher_only_args_no_output"),torch.device("cpu"),production_peft=True)
    return dict(actor=actor,tokenizer=tok,rows=rows,targets=targets,draws=[["a"],["b"],["a"]],args=args,
                cal=cal,util=util,anchor=anchor,semantic_state_sha256=semantic_state_sha256,
                expected_initial_sha256=semantic_state_sha256(actor.state_dict()))


def test_actual_tiny_multitoken_peft_matches_full_masked_reference_with_no_rl(tiny,tmp_path,monkeypatch):
    import torch
    from transformers import get_scheduler
    f=tiny;reference=copy.deepcopy(f["actor"])
    def forbidden(*args,**kwargs):pytest.fail("Teacher-only control attempted policy generation")
    monkeypatch.setattr(f["actor"],"generate",forbidden);monkeypatch.setattr(f["actor"].get_base_model(),"generate",forbidden)
    proof=M.run_cpu_diagnostic(**f,output=tmp_path)
    reference.train();f["anchor"].restore_nonreentrant_checkpointing(reference)
    inventory=f["cal"].parameter_inventory(reference);pairs=f["cal"].parameter_pairs(reference,inventory)
    args=f["args"];optimizer=torch.optim.AdamW([p for _,p in pairs],lr=args.learning_rate,
        betas=(args.adam_beta1,args.adam_beta2),eps=args.adam_epsilon,weight_decay=args.weight_decay)
    scheduler=get_scheduler("constant",optimizer,0,3)
    for draw in f["draws"]:
        tokens=f["util"].prepare_teacher_tokens(f["tokenizer"],"pfx",f["targets"][0]["target_text"],
            target_token_cap=3000,context_limit=128,device=torch.device("cpu"))
        optimizer.zero_grad(set_to_none=True)
        logits=reference(input_ids=tokens["input_ids"],attention_mask=tokens["attention_mask"],use_cache=False).logits
        loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,len(f["tokenizer"])),tokens["labels"][:,1:].reshape(-1),ignore_index=-100)
        loss.backward();torch.nn.utils.clip_grad_norm_([p for _,p in pairs],args.max_grad_norm,error_if_nonfinite=True)
        optimizer.step();scheduler.step()
    errors=[]
    for actual,expected in zip(f["actor"].parameters(),reference.parameters()):
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
        errors.append(float((actual.detach()-expected.detach()).abs().max()))
    assert proof["scope"]=="actual_tiny_cpu_only" and proof["actual_7b_production_execution"] is False
    assert proof["updates"]==proof["teacher_CE_forwards"]==proof["teacher_CE_backwards"]==3
    assert proof["RL_responses"]==proof["RL_reward_calls"]==proof["evaluation_model_calls"]==0
    assert not proof["reference_KL"] and not proof["BF16_autocast"] and proof["all_grads_cleared"]
    assert proof["GC"]["effective_use_reentrant"] is False and proof["GC"]["frozen_embedding_weights"]
    assert [r["instance_id"] for r in proof["teacher_records"]]==["a","b","a"]
    assert all(r["target_tokens_including_Objective_and_EOS"]==len(f["targets"][0]["target_text"])+1 and r["masked_prompt_tokens"]==3 for r in proof["teacher_records"])
    assert any(row["name"].endswith(".bias") for row in proof["parameter_inventory"])
    proof.update(diagnostic_actor_parameters=sum(p.numel() for p in f["actor"].parameters()),
                 diagnostic_parameter_dtype="torch.float32",diagnostic_device="cpu",policy_generation_calls=0,
                 independent_full_masked_CE_optimizer_reference_max_abs_error=max(errors))
    (tmp_path/"diagnostic_complete.json").write_text(json.dumps(proof,sort_keys=True,indent=2)+"\n")


def test_nonfinite_ce_fails_before_any_optimizer_step(tiny,tmp_path,monkeypatch):
    import torch
    initial=tiny["expected_initial_sha256"]
    monkeypatch.setattr(tiny["util"],"teacher_cross_entropy",lambda model,tokens: next(model.parameters()).sum()*float("nan"))
    with pytest.raises(ValueError,match="Nonfinite"):M.run_cpu_diagnostic(**tiny,output=tmp_path)
    assert tiny["semantic_state_sha256"](tiny["actor"].state_dict())==initial
    assert not (tmp_path/"teacher_only_CE_windows.jsonl").exists()
    assert all(p.grad is None for p in tiny["actor"].parameters())


def test_initial_gradients_rejected_before_update(tiny,tmp_path):
    import torch
    next(p for p in tiny["actor"].parameters() if p.requires_grad).grad=torch.zeros_like(next(p for p in tiny["actor"].parameters() if p.requires_grad))
    with pytest.raises(ValueError,match="clean original"):M.run_cpu_diagnostic(**tiny,output=tmp_path)


def test_nonfinite_gradient_fails_before_any_update(tiny,tmp_path):
    import torch
    param=next(p for p in tiny["actor"].parameters() if p.requires_grad)
    hook=param.register_hook(lambda gradient:torch.full_like(gradient,float("nan")))
    try:
        with pytest.raises(ValueError,match="Nonfinite or absent teacher gradient"):
            M.run_cpu_diagnostic(**tiny,output=tmp_path)
    finally:hook.remove()
    assert tiny["semantic_state_sha256"](tiny["actor"].state_dict())==tiny["expected_initial_sha256"]
    assert not (tmp_path/"teacher_only_CE_windows.jsonl").exists()


def test_label_without_objective_rejected_before_update(tiny,tmp_path):
    tiny["targets"][0]["target_text"]="Route: [0, 1]"
    tiny["targets"][0]["target_text_sha256"]=hashlib.sha256(tiny["targets"][0]["target_text"].encode()).hexdigest()
    with pytest.raises(ValueError,match="Objective-bearing"):M.run_cpu_diagnostic(**tiny,output=tmp_path)


def test_settings_equal_actual_frozen_args(tiny):
    settings=M._settings(tiny["args"])
    assert settings=={"learning_rate":1e-5,"adam_beta1":.9,"adam_beta2":.999,"adam_epsilon":1e-8,
                      "weight_decay":0.,"max_grad_norm":1.,"lr_scheduler_type":"constant","warmup_ratio":0.,"optim":"adamw_torch"}
