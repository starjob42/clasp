"""Prospective operational plans/CPU observer fixtures; no native jobs or seeds."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"
spec = importlib.util.spec_from_file_location("formal_launch", EXP / "formal_srun_launch_v1.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fixture(tmp_path, *, sft=False):
    value = {"protocol": m.PROTOCOL + "-slot-spec", "schema_version": 1, "synthetic_fixture": True,
        "slot_id": "fixture/op.sft" if sft else "fixture/op.rloo_scalar_beta0", "draw_id": "fixture",
        "task": "op", "arm": "sft" if sft else "rloo_scalar_beta0", "train_seed": 123,
        "round_contract_sha256": "a" * 64, "first_attempt": True, "automatic_retry": False,
        "gpus": 1 if sft else 4, "training_ranks": 1 if sft else 4, "slurm_launcher_tasks": 1,
        "cpus_per_task": 40, "requested_wall_seconds": 3600,
        "python": "/synthetic/python", "source_root": "/synthetic/source", "output": "/synthetic/output",
        "base": "/synthetic/base", "sft": "/synthetic/sft", "observer_directory": "/synthetic/events",
        "source_manifest_binding":{"path":"/synthetic/source/source_manifest.json","sha256":"e"*64,"size_bytes":1},
        "original_source_manifest_binding":{"path":"/synthetic/source/original_source2_manifest.json","sha256":m.SOURCE_MANIFEST_SHA,"size_bytes":1},
        "worker_binding": {"path": "/synthetic/worker.py", "sha256": "c" * 64, "size_bytes": 1}}
    value["spec_sha256"] = m.sha(m.encoded(value))
    registry = {"protocol": m.PROTOCOL + "-registry", "schema_version": 1, "entries": [{
        "slot_id": value["slot_id"], "slot_spec_sha256": value["spec_sha256"], "round_contract_sha256": "a" * 64,
        "source_manifest_sha256": "e"*64,"original_source_manifest_sha256":m.SOURCE_MANIFEST_SHA,
        "original_source_manifest_binding":value["original_source_manifest_binding"], "source2_sha256": m.SOURCE2_SHA,
        "synthetic_fixture": True, "production_execution_authorized": False}]}
    p, r = tmp_path / "spec.json", tmp_path / "registry.json"
    p.write_bytes(m.encoded(value)); r.write_bytes(m.encoded(registry))
    return value, registry, p, r


def test_exact_srun_dotzero_strategy_has_one_launcher_and_four_torchrun_ranks(tmp_path):
    value, registry, p, r = fixture(tmp_path)
    argv = m.build_srun_argv(value, registry, spec_file=p, registry_file=r, synthetic_fixture=True)
    assert argv[:4] == ["srun", "--ntasks=1", "--cpus-per-task=40", "--gpus-per-task=4"]
    assert "--nproc-per-node=4" in argv and "--no-python" in argv
    assert argv.count("-I") == argv.count("-B") == 2
    assert "-c" in argv and m.WORKER_BOOTSTRAP in argv


def test_once_per_task_sft_command_has_one_gpu_and_one_rank(tmp_path):
    value, registry, p, r = fixture(tmp_path, sft=True)
    argv = m.build_srun_argv(value, registry, spec_file=p, registry_file=r, synthetic_fixture=True)
    assert "--gpus-per-task=1" in argv and "--nproc-per-node=1" in argv


def test_default_empty_formal_registry_denies_all_execution(tmp_path):
    value, _, _, _ = fixture(tmp_path)
    registry = json.loads((EXP / "planning/formal_execution_source_registry_v1.json").read_bytes())
    with pytest.raises(ValueError, match="empty"): m.validate_slot_spec(value, registry, synthetic_fixture=True)


@pytest.mark.parametrize("key,value", [("automatic_retry", True), ("first_attempt", False), ("gpus", 1),
    ("training_ranks", 2), ("slurm_launcher_tasks", 4), ("requested_wall_seconds", 72001),
    ("source_root", "/synthetic/../other")])
def test_sampler_world_attempt_budget_or_namespace_cannot_change(tmp_path, key, value):
    item, registry, _, _ = fixture(tmp_path); item[key] = value
    item["spec_sha256"] = m.sha(m.encoded({k: v for k, v in item.items() if k != "spec_sha256"}))
    registry["entries"][0]["slot_spec_sha256"] = item["spec_sha256"]
    with pytest.raises(ValueError): m.validate_slot_spec(item, registry, synthetic_fixture=True)


def test_reviewed_command_cannot_execute_another_spec_file(tmp_path):
    item, registry, p, r = fixture(tmp_path)
    p.write_bytes(m.encoded({"different": "spec"}))
    with pytest.raises(ValueError, match="exactly the reviewed"): m.build_srun_argv(item, registry, spec_file=p, registry_file=r, synthetic_fixture=True)


def test_fixture_or_self_flags_never_authorize_production(tmp_path):
    item, registry, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="Fixture"): m.validate_slot_spec(item, registry)
    item.pop("synthetic_fixture"); registry["entries"][0].pop("synthetic_fixture")
    item["spec_sha256"] = m.sha(m.encoded({k: v for k, v in item.items() if k != "spec_sha256"}))
    registry["entries"][0]["slot_spec_sha256"] = item["spec_sha256"]
    with pytest.raises(ValueError, match="Actual complete"): m.validate_slot_spec(item, registry)


def test_passive_first_forward_observer_preserves_actual_cpu_outputs_rng_and_parameters(tmp_path):
    import torch
    torch.manual_seed(123)
    model = torch.nn.Linear(3, 2)
    inputs = torch.tensor([[1., 2., 3.]])
    expected = model(inputs).detach().clone()
    before_rng = torch.get_rng_state().clone(); before_params = {k: v.detach().clone() for k, v in model.state_dict().items()}
    spec, _, _, _ = fixture(tmp_path)
    observed = m.FirstModelForward(tmp_path / "events", spec, rank=0, job_id="999", step_id="0",
        source_binding={"sha256": m.SOURCE2_SHA})
    observed.attach(model, "actor")
    assert torch.equal(model(inputs), expected) and torch.equal(torch.get_rng_state(), before_rng)
    assert all(torch.equal(v, before_params[k]) for k, v in model.state_dict().items())
    first = (tmp_path / "events/first_model_forward_rank0.json").read_bytes()
    model(inputs)
    assert (tmp_path / "events/first_model_forward_rank0.json").read_bytes() == first
    event = json.loads(first)
    assert event["outcomes_recorded"] is False and event["independent_custody_receipt"] is False
    assert event["requires_independent_read_only_collection"] and not event["formal_claim_eligible"]


def test_existing_first_call_event_blocks_retry_before_forward(tmp_path):
    import torch
    item, _, _, _ = fixture(tmp_path)
    directory = tmp_path / "events"; directory.mkdir(); (directory / "first_model_forward_rank0.json").write_bytes(b"prior")
    model = torch.nn.Linear(1, 1); observer = m.FirstModelForward(directory, item, rank=0, job_id="999", step_id="0", source_binding={})
    observer.attach(model, "actor")
    with pytest.raises(FileExistsError): model(torch.ones(1, 1))


def test_worker_cannot_execute_an_arbitrary_helper_before_registration(tmp_path):
    spec = importlib.util.spec_from_file_location("formal_worker", EXP / "formal_source2_worker_v1.py")
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    path = tmp_path / "untrusted.py"; path.write_text('raise RuntimeError("this must never execute")\n')
    item = m.binding(path)
    with pytest.raises(ValueError, match="bytes differ"): worker.module_from_binding(item, "_untrusted", worker.LAUNCHER_SHA)
    actual = worker.module_from_binding(m.binding(EXP / "formal_srun_launch_v1.py"), "_actual_gate", worker.LAUNCHER_SHA)
    assert actual.SOURCE2_SHA == m.SOURCE2_SHA
    assert actual.__executed_source_sha256__ == worker.LAUNCHER_SHA


def test_worker_bootstrap_executes_same_pinned_buffer_with_metadata_and_exact_arguments(tmp_path):
    import subprocess
    import sys
    path = tmp_path / "fixture_worker.py"
    path.write_text('import json,sys\nprint(json.dumps({"sha": __executed_source_sha256__, "args": sys.argv[1:]}))\n')
    item = m.binding(path)
    result = subprocess.run([sys.executable, "-I", "-B", "-c", m.WORKER_BOOTSTRAP,
        str(path), item["sha256"], str(item["size_bytes"]), "--fixture-only"], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"sha": item["sha256"], "args": ["--fixture-only"]}


def test_worker_bootstrap_rejects_changed_source_before_any_execution(tmp_path):
    import subprocess
    import sys
    marker = tmp_path / "must_not_exist"
    path = tmp_path / "fixture_worker.py"
    path.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("executed")\n')
    item = m.binding(path)
    result = subprocess.run([sys.executable, "-I", "-B", "-c", m.WORKER_BOOTSTRAP,
        str(path), "0" * 64, str(item["size_bytes"])], capture_output=True, text=True)
    assert result.returncode != 0 and not marker.exists()


def round_membership_schema_fixture(tmp_path, *, sft=False):
    """Local schema stub only: no inputs, model, real selection or registration."""
    roster = [{"draw_id": f"draw{i:02d}", "train_seed": 123} for i in range(20)]
    contract = {"protocol": "sft-anchor-exact-global-seed-confirmation-20261014-v1-contract",
        "n_seed_draws": 20, "seed_roster": roster, "candidate_arm": "rloo_scalar_beta0",
        "native_source_root": "/synthetic/source",
        "source_manifest_binding":{"sha256":"e"*64},
        "native_control_roots": {d["draw_id"]: "/synthetic/" + d["draw_id"] for d in roster},
        "task_contract_bindings": {d["draw_id"]: {"op": {"sha256": "b" * 64}} for d in roster}}
    contract["contract_sha256"] = m.sha(m.encoded(contract))
    path = tmp_path / "schema_only_round.json"; path.write_bytes(m.encoded(contract))
    item = {"arm": "sft" if sft else "rloo_scalar_beta0", "task": "op", "train_seed": 123,
        "draw_id": "sft_once" if sft else "draw03", "source_root": "/synthetic/source",
        "round_contract_binding": m.binding(path), "round_contract_sha256": contract["contract_sha256"],
        "task_contract_binding": {"sha256": "b" * 64}}
    item["source_manifest_binding"]=contract["source_manifest_binding"]
    item["original_source_manifest_binding"]={"sha256":m.SOURCE_MANIFEST_SHA}
    draw = "draw00" if sft else "draw03"
    item["slot_id"] = item["draw_id"] + "/op." + item["arm"]
    item["output"] = "/synthetic/" + draw + "/outputs/op." + item["arm"]
    item["observer_directory"] = "/synthetic/" + draw + "/observers/op." + item["arm"]
    return item, {"once_per_task_sft_sidecar": sft,"original_source_manifest_binding":item["original_source_manifest_binding"]}


def test_duplicate_seed_values_keep_registered_draw_output_and_observer_identity(tmp_path):
    item, entry = round_membership_schema_fixture(tmp_path)
    m.admit_round_membership(item, entry)
    item["output"] = item["output"].replace("draw03", "draw02")
    with pytest.raises(ValueError, match="output or observer"): m.admit_round_membership(item, entry)


def test_single_sft_sidecar_has_fixed_first_draw_root_and_distinct_observer(tmp_path):
    item, entry = round_membership_schema_fixture(tmp_path, sft=True)
    m.admit_round_membership(item, entry)
    item["observer_directory"] = item["observer_directory"].replace("op.sft", "op.foarl")
    with pytest.raises(ValueError, match="output or observer"): m.admit_round_membership(item, entry)


def test_actual_peft_generate_is_timestamped_before_base_forward_without_any_loss_call(tmp_path):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(789)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)).eval()
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).eval()
    inputs = {"input_ids": torch.tensor([[1, 4, 5]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    expected = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    before_rng = torch.get_rng_state().clone()
    before_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    item, _, _, _ = fixture(tmp_path)
    directory = tmp_path / "peft_events"
    observer = m.FirstModelForward(directory, item, rank=0, job_id="999", step_id="0",
        source_binding={"sha256": m.SOURCE2_SHA})
    observer.attach(model, "actor")
    base_checks = []
    def before_generation_forward(module, args):
        base_checks.append((directory / "first_model_forward_rank0.json").is_file())
    base.register_forward_pre_hook(before_generation_forward)
    actual = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    assert base_checks and all(base_checks)
    assert torch.equal(actual, expected) and torch.equal(torch.get_rng_state(), before_rng)
    assert all(torch.equal(v, before_state[k]) for k, v in model.state_dict().items())
    event = json.loads((directory / "first_model_forward_rank0.json").read_bytes())
    assert event["loaded_model_role"] == "actor" and not event["outcomes_recorded"]


def initializer_schema_fixture(tmp_path):
    """Diagnostic artifact tree only; never a production inventory/registration."""
    trees = {}
    for kind in ("base", "sft"):
        root = tmp_path / kind; root.mkdir()
        (root / "config.json").write_text('{"rope_theta":10000}\n')
        (root / "model.safetensors").write_bytes(b"synthetic unchanged tensor bytes")
        rows = []
        for path in sorted(root.iterdir()):
            row = m.binding(path); row["path"] = path.name; rows.append(row)
        trees[kind] = {"root": str(root), "files": rows}
    contract = tmp_path / "fixture_task_contract.json"
    contract.write_text(json.dumps({"initializer_sha256": "d" * 64}))
    inventory = {"protocol": m.PROTOCOL + "-initializer-artifacts", "task": "op",
        "initializer_sha256": "d" * 64, "registered_before_policy_calls": True, "artifacts": trees}
    inventory["inventory_sha256"] = m.sha(m.encoded(inventory))
    path = tmp_path / "fixture_inventory.json"; path.write_bytes(m.encoded(inventory))
    return {"task": "op", "base": str(tmp_path / "base"), "sft": str(tmp_path / "sft"),
        "task_contract_binding": m.binding(contract)}, {"initializer_artifacts_binding": m.binding(path)}


def test_complete_initializer_artifacts_are_replayed_without_model_loading(tmp_path):
    item, entry = initializer_schema_fixture(tmp_path)
    result = m.validate_initializer_inventory(item, entry)
    assert result["initializer_sha256"] == "d" * 64


@pytest.mark.parametrize("mutation", ["config", "missing", "extra", "link", "identity"])
def test_initializer_config_inventory_links_and_task_identity_cannot_change(tmp_path, mutation):
    item, entry = initializer_schema_fixture(tmp_path)
    if mutation == "config": (tmp_path / "base/config.json").write_text('{"rope_theta":123456}\n')
    elif mutation == "missing": (tmp_path / "sft/model.safetensors").unlink()
    elif mutation == "extra": (tmp_path / "sft/unregistered.json").write_text('{}\n')
    elif mutation == "link":
        target = tmp_path / "base/config.json"; raw = target.read_bytes(); target.unlink()
        outside = tmp_path / "outside.json"; outside.write_bytes(raw); target.symlink_to(outside)
    else: item["task"] = "cvrp"
    with pytest.raises(ValueError): m.validate_initializer_inventory(item, entry)
