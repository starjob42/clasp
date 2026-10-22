"""Pure synthetic endpoint custody; no models, CUDA or study outcomes."""
import copy
import builtins
import importlib.util
import json
from pathlib import Path
import random
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/rloo_one_shot_20261016"


def module(name, path):
    raw = path.read_bytes(); value = types.ModuleType(name); value.__file__ = str(path)
    exec(compile(raw, str(path), "exec", dont_inherit=True), value.__dict__)
    return value


A = module("analysis_admission_under_test", STUDY / "analysis_admission_v1.py")
P = module("analysis_admission_fixture_preparer", STUDY / "prepare_execution.py")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(A.encoded(value) + b"\n")
    return A.binding(path)


def change(path, edit):
    value = json.loads(path.read_bytes()); edit(value); return write(path, value)


@pytest.fixture
def complete(tmp_path, monkeypatch):
    # Snapshot actual reviewed sources and original *input metadata* only.
    # All endpoint/model/dtype/runtime/resource rows below are synthetic.
    monkeypatch.setattr(__import__("sys"), "dont_write_bytecode", True)
    importer = builtins.__import__
    def no_model_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "clasp", "peft", "datasets", "trl"}, "Metadata admission imported model/project dependencies"
        return importer(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_model_import)
    bound, roster = P.prepare(tmp_path / "packet")
    runner, source, buffers = A.load_runner(roster)
    pure = A.pure_contract_builder(source, buffers)
    for slot in roster["slots"]:
        if slot["mode"] != "study" or slot["task"] != "op":
            continue
        out = Path(slot["output"]); out.mkdir()
        c = json.loads(A.read_bound(slot["contract"])); trained = slot["arm"] != "one_shot_sft"
        steps = 100 if trained else 0
        terminal = out / "terminal"; terminal.mkdir()
        records = []
        for name in sorted(("adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "tokenizer_config.json")):
            p = terminal / name; p.write_bytes(("SYNTHETIC-NOT-A-MODEL:" + name).encode())
            records.append({k:v for k,v in A.binding(p).items() if k != "path"} | {"path":name})
        tree = {"files":records, "tree_sha256":A.digest(records)}
        rows = json.loads(A.read_bound(c["training"])); rng = random.Random(73)
        draws = [[rng.choice([r["instance_id"] for r in rows]) for _ in range(4)][:1] for _ in range(100)]
        logical = A.expected_logical(c, runner, pure)
        write(out/"training_contract.json", logical)
        proof = {"protocol":"sft-anchor-strict-linear-one-shot-v1","mode":"study","arm":slot["arm"],
            "recipe":{"one_shot_sft":"sft","foarl":"foarl_released","one_shot_grpo":"grpo_cost",
                "one_shot_grpo_fixed_horizon":"grpo_cost_fixed_horizon","one_shot_rloo":"rloo_cost"}[slot["arm"]],
            "local_contract_sha256":c["contract_sha256"],"source_manifest":c["source_manifest"],
            "steps":steps,"RL_responses":steps*9,"generated_tokens":steps*9,"training_wall_seconds":600,
            "automatic_retry":False,"superiority_established":False,"intermediate_optimizer_resume_supported":False,
            "checkpoint_policy":"terminal_only_no_optimizer_checkpoint","whole_actor_sha256":"b"*64 if trained else "a"*64,
            "initial_whole_actor_sha256":"a"*64,"actor_changed":trained,"terminal_tree":tree,
            "reference_sha256":"a"*64 if trained else None,"reference_integrity":trained,
            "initial_reference_audit":{"bitwise_equal":True},"prepared_reference_audit":{"bitwise_equal":True},
            "effective_checkpointing":{"effective_use_reentrant":False},
            "actual_windows":[[v for v in draw for _ in range(9)] for draw in draws] if trained else []}
        proof_binding = write(out/"training_complete.json", proof)
        reload = {"status":"passed","bitwise_state_equal":True,"whole_actor_sha256":proof["whole_actor_sha256"],
                  "training_complete_binding":proof_binding,"terminal_tree":tree}
        reload_binding = write(out/"checkpoint_reload_exact.json", reload)
        # Invalid placeholder bytes deliberately prove the metadata gate never
        # opens raw quality or tries to claim these fixtures are measurements.
        (out/"raw_completions.jsonl").write_bytes(b"SYNTHETIC RAW MUST NEVER BE READ\n")
        result = {"status":"completed_development_arm","mode":"study","task":"op","arm":slot["arm"],
            "contract":c,"source_manifest":c["source_manifest"],"runtime":A.PACKAGES,"superiority_established":False,
            "quality_summary":None,"trained_updates":steps,"RL_responses":steps*9,"evaluation_responses":96,
            "raw_completions_binding":A.binding(out/"raw_completions.jsonl"),"training_complete_binding":proof_binding,
            "checkpoint_reload_binding":reload_binding,"resources":{"gpu_uuid":c["gpu_uuid"],"physical_gpus":1,
                "wall_seconds_inside_worker":800,"training_wall_seconds":600,"evaluation_wall_seconds":100,
                "peak_allocated_bytes":1000,"peak_reserved_bytes":2000}}
        write(out/"result.json", result)
        write(out/"attempt.json", {"contract_file_binding":slot["contract"],"contract_sha256":c["contract_sha256"],
            "automatic_retry":False,"gpu_uuid":c["gpu_uuid"],"maximum_wall_seconds":slot["wall_seconds"],
            "requested_gpu_hours":slot["wall_seconds"]/3600})
        write(out/"completion.json", {"status":"completed","error":None,"automatic_retry":False,
            "contract_file_sha256":slot["contract"]["sha256"],"source_manifest":c["source_manifest"],
            "requested_gpu_hours":slot["wall_seconds"]/3600,"parent_elapsed_wall_seconds":1000,
            "actual_GPU_hour_upper_bound":1000/3600})
        write(out/"runtime_admission.json", {"packages":A.PACKAGES,"gpu_uuid":c["gpu_uuid"],"world_size":1,
            "compute_capability":[8,6],"native_bfloat16":True,"training_dtype":"torch.bfloat16"})
        dtype = {"parameter_counts_by_dtype":{"torch.bfloat16":6_000_000_000,"torch.float32":100},
            "trainable_parameters":100 if trained else 0,"devices":["cuda:0"],"actual_BF16_base_weights_verified":True,
            "FP32_adapter_parameter_names":["lora_A.weight"],"local_contract_sha256":c["contract_sha256"],
            "gpu_uuid":c["gpu_uuid"]}
        write(out/"model_load_admission.json", {**dtype,"protocol":"one-shot-actual-BF16-model-admission-v1"})
        write(out/"reference_load_admission.json", {**dtype,"protocol":"one-shot-actual-frozen-reference-admission-v1",
            "trainable_parameters":0,"independent_original_SFT_load":True})
    original = A.stable_read
    def no_quality(path, **kwargs):
        assert Path(path).name != "raw_completions.jsonl", "Metadata gate opened raw quality"
        return original(path, **kwargs)
    monkeypatch.setattr(A, "stable_read", no_quality)
    return bound, roster


def slot_output(roster, index=4):
    return Path([s for s in roster["slots"] if s["mode"]=="study" and s["task"]=="op"][index]["output"])


def test_complete_five_synthetic_endpoints_admit_metadata_without_raw_or_models(complete):
    bound, _ = complete
    result = A.admit_complete_task(bound, "op")
    assert result["status"] == "complete_metadata_admission" and len(result["results"]) == 5
    assert result["raw_quality_files_read"] == result["model_calls"] == 0
    assert result["reserved_campaign_GPU_hours"] == 42 and not result["superiority_established"]


def test_missing_arm_returns_without_reading_any_existing_result(complete, monkeypatch):
    bound, roster = complete; (slot_output(roster)/"result.json").unlink()
    original = A.stable_read
    def guard(path, **kwargs):
        assert Path(path).name not in ("result.json","training_complete.json")
        return original(path, **kwargs)
    monkeypatch.setattr(A,"stable_read",guard)
    assert A.admit_complete_task(bound,"op")["quality_analyzed"] is False


def test_failed_fourth_arm_rejects_before_any_result_or_raw_read(complete, monkeypatch):
    bound, roster = complete; change(slot_output(roster,3)/"completion.json",lambda x:x.update(status="failed"))
    original = A.stable_read
    def guard(path, **kwargs):
        assert Path(path).name != "result.json"
        return original(path, **kwargs)
    monkeypatch.setattr(A,"stable_read",guard)
    with pytest.raises(ValueError,match="Failed"):A.admit_complete_task(bound,"op")


@pytest.mark.parametrize("field,value",[("parent_elapsed_wall_seconds",True),("parent_elapsed_wall_seconds",float("inf")),
    ("parent_elapsed_wall_seconds",15000),("actual_GPU_hour_upper_bound",2),("requested_gpu_hours",5),
    ("automatic_retry",True),("contract_file_sha256","0"*64)])
def test_completion_cost_identity_and_type_faults_reject_before_raw(complete,field,value):
    bound, roster=complete; p=slot_output(roster)/"completion.json"
    # Nonfinite JSON is deliberately admitted only by the test writer.
    if isinstance(value,float) and value==float("inf"):
        x=json.loads(p.read_bytes());x[field]=value;p.write_text(json.dumps(x))
    else:change(p,lambda x:x.update({field:value}))
    with pytest.raises(ValueError):A.admit_complete_task(bound,"op")


@pytest.mark.parametrize("fault",["terminal_bytes","extra_terminal","terminal_link","raw_escape","reload_actor",
    "reload_training_binding","reference_dtype","reference_hash","wrong_runtime","resource_negative","wrong_gpu","peak_bool","window","logical_sampler","source_bytes"])
def test_actual_artifact_source_reference_and_resource_faults_reject_before_raw(complete,fault):
    bound, roster=complete; out=slot_output(roster)
    if fault=="terminal_bytes":(out/"terminal/adapter_model.safetensors").write_bytes(b"changed")
    elif fault=="extra_terminal":(out/"terminal/extra.bin").write_bytes(b"extra")
    elif fault=="terminal_link":(out/"terminal/extra.bin").symlink_to(out/"attempt.json")
    elif fault=="raw_escape":change(out/"result.json",lambda x:x["raw_completions_binding"].update(path=str(slot_output(roster,1)/"raw_completions.jsonl")))
    elif fault.startswith("reload_"):
        rb=change(out/"checkpoint_reload_exact.json",lambda x:x.update(whole_actor_sha256="c"*64) if fault=="reload_actor" else x["training_complete_binding"].update(sha256="0"*64))
        change(out/"result.json",lambda x:x.update(checkpoint_reload_binding=rb))
    elif fault=="reference_dtype":change(out/"reference_load_admission.json",lambda x:x.update(trainable_parameters=1))
    elif fault in ("reference_hash","window"):
        pb=change(out/"training_complete.json",lambda x:x.update(reference_sha256="c"*64) if fault=="reference_hash" else x["actual_windows"][0].__setitem__(0,"wrong"))
        rb=change(out/"checkpoint_reload_exact.json",lambda x:x.update(training_complete_binding=pb))
        change(out/"result.json",lambda x:x.update(training_complete_binding=pb,checkpoint_reload_binding=rb))
    elif fault=="wrong_runtime":change(out/"runtime_admission.json",lambda x:x.update(world_size=4))
    elif fault=="logical_sampler":
        def edit(x):
            x["training"]["top_p"]=.9
            x["contract_sha256"]=A.digest({k:v for k,v in x.items() if k!="contract_sha256"})
        change(out/"training_contract.json",edit)
    elif fault in ("resource_negative","wrong_gpu","peak_bool"):
        field,value={"resource_negative":("evaluation_wall_seconds",-1),"wrong_gpu":("gpu_uuid","other"),"peak_bool":("peak_allocated_bytes",True)}[fault]
        change(out/"result.json",lambda x:x["resources"].update({field:value}))
    else:
        source=json.loads(A.read_bound(roster["source_manifest"]))
        p=Path(source["root"])/"src/clasp/rewards/one_shot_quality.py";p.chmod(0o600);p.write_bytes(p.read_bytes()+b"\n# mutation\n")
    with pytest.raises(ValueError):A.admit_complete_task(bound,"op")


def test_bound_artifact_symlink_is_not_followed(complete):
    bound,roster=complete;out=slot_output(roster);p=out/"completion.json";copy=out/"completion-copy.json"
    copy.write_bytes(p.read_bytes());p.unlink();p.symlink_to(copy)
    with pytest.raises(ValueError,match="Canonical"):A.admit_complete_task(bound,"op")


@pytest.mark.parametrize("fault",["reduced_roster","extra_attempt","wrong_seed","plan_sampler","reservation","source_rebind","resealed_scorer"])
def test_roster_contract_plan_and_source_registration_faults_reject(complete,fault):
    bound,roster=complete
    if fault=="reduced_roster":roster["slots"].pop()
    elif fault=="extra_attempt":roster["slots"][0]["maximum_attempts"]=2
    elif fault=="reservation":roster["maximum_requested_physical_gpu_hours"]=43
    elif fault=="plan_sampler":
        plan=json.loads(A.read_bound(roster["campaign_plan"]));plan["training"]["training_top_p"]=.9
        roster["campaign_plan"]=write(Path(bound["path"]).parent/"synthetic_modified_plan.json",plan)
    elif fault=="source_rebind":
        p=Path(bound["path"]).parent/"source/run_local_7b_v1.py";p.chmod(0o600);p.write_bytes(p.read_bytes()+b"\n# changed runner\n")
        roster["runner"]=A.binding(p)
    elif fault=="resealed_scorer":
        manifest=json.loads(A.read_bound(roster["source_manifest"]))
        relative="src/clasp/baselines/sft_anchor_pilot.py";p=Path(manifest["root"])/relative
        p.chmod(0o600);p.write_bytes(p.read_bytes()+b"\n# separately re-sealed scorer\n")
        for row in manifest["files"]:
            if row["path"]==relative:row.update({k:v for k,v in A.binding(p).items() if k!="path"})
        roster["source_manifest"]=write(Path(roster["source_manifest"]["path"]),manifest)
    else:
        slot=[s for s in roster["slots"] if s["slot"].startswith("study.op.one_shot_rloo")][0]
        c=json.loads(A.read_bound(slot["contract"]));c["seed"]=74
        c["contract_sha256"]=A.digest({k:v for k,v in c.items() if k!="contract_sha256"})
        slot["contract"]=write(Path(slot["contract"]["path"]),c)
    bound=write(Path(bound["path"]),roster)
    with pytest.raises(ValueError,match="Unreviewed numerical core" if fault=="resealed_scorer" else None):A.admit_complete_task(bound,"op")


@pytest.mark.parametrize("name",["completion.json","runtime_admission.json"])
def test_late_metadata_rewrite_cannot_rebind_earlier_admitted_values(complete,monkeypatch,name):
    bound,roster=complete;target=slot_output(roster)/name;original=A.stable_read;changed=[]
    def swap(path,**kwargs):
        raw=original(path,**kwargs)
        if Path(path)==target and not changed:
            changed.append(True);target.write_bytes(raw+b"\n")
        return raw
    monkeypatch.setattr(A,"stable_read",swap)
    with pytest.raises(ValueError,match="Metadata changed after admission"):
        A.admit_complete_task(bound,"op")
    assert changed
