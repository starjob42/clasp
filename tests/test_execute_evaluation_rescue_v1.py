"""Actual tiny PEFT inference and independently pinned synthetic cost guards.

No scheduler action, native allocation proof or production performance admission.
"""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("peft")

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

f = module("rescue_test_fixture", ROOT / "tests/test_prepare_evaluation_rescue_v1.py")
m = module("rescue_executor", EXP / "execute_evaluation_rescue_v1.py")
a = m.accounting

def dump(path, value):
    path.write_bytes(f.encoded(value)); return m.preparer().identity(path)["sha256"]

def sacct(job, state, seconds, gpus):
    rows = []
    for suffix in ("", ".batch", ".extern"):
        status = "COMPLETED" if suffix == ".extern" and state != "RUNNING" else state
        exitcode = "0:0" if status in ("RUNNING", "COMPLETED") else "1:0"
        rows.append(f"{job}{suffix}|{status}|{seconds}|{exitcode}|cpu=80,mem=300000M,node=1,billing=5,gres/gpu={gpus}\n")
    return {"argv": ["sacct", "--noheader", "--parsable2", "--jobs", job,
        "--format=JobIDRaw,State%40,ElapsedRaw,ExitCode,AllocTRES"], "returncode": 0, "stderr": "",
        "stdout_binding": f.binding("".join(rows).encode())}

def setup(tmp_path, task="op", arm="rloo_cardinal_beta005", prefix=2, tail=True):
    args, base, originals = f.fixture(tmp_path, task, arm, prefix=prefix, tail=tail)
    p = m.preparer(); plan = p.prepare(**args)
    plan_path = tmp_path / "plan.json"; plan_sha = dump(plan_path, plan)
    contract_path = tmp_path / "contract.json"; contract_sha = dump(contract_path, args["contract"])
    output = tmp_path / "operator"
    entry = {"protocol": m.PROTOCOL, "slot": plan["slot"], "scope": plan["scope"], "sources": m.sources(),
        "amendment_sha256": m.AMENDMENT_SHA, "preparation_semantic_sha256": plan["semantic_sha256"],
        "whole_actor_sha256": plan["whole_actor_sha256"], "contract_sha256": plan["contract_sha256"],
        "training_or_resume": False, "checkpoint_selection": False, "synthetic_offline_fixture": True,
        "output_directory": str(output.resolve()), "base_model_binding": {"path": str(base.resolve()),
         "config": p.identity(base / "config.json"), "generation_config": p.identity(base / "generation_config.json")}}
    cert_path = tmp_path / "certificate.json"
    if plan["mode"] == "one_gpu_evaluation_only":
        script_path = a.old.NAMESPACE + "/control_v2/job_scripts/" + plan["slot"] + ".slurm"
        script_raw = (EXP / "control_v2/job_scripts" / (plan["slot"] + ".slurm")).read_bytes()
        attempt = json.loads(a.old.bound_bytes(args["parent_attempt"]))
        attempt.update(schema_version=2, script_sha256=a.old.sha(script_raw), argv=["sbatch", "--parsable", script_path],
            automatic_retry=False, automatic_cancellation=False, returncode=0, stderr="", stdout="999\n")
        wrapped = {"reservation_id": "phase2/" + plan["slot"], "attempt_index": 1, "kind": "original_source2",
            "attempt_binding": f.binding(f.encoded(attempt), a.old.NAMESPACE + "/control_v2/attempts/" + plan["slot"] + ".json"),
            "script_binding": f.binding(script_raw, script_path)}
        inventory = {"protocol": "closed-original-phase2-attempt-inventory-v1", "source_manifest_sha256": a.old.SOURCE_SHA,
            "entries": [wrapped], "directory_listing_binding": f.binding(f.encoded({
                "directory": a.old.NAMESPACE + "/control_v2/attempts", "regular_json_files": [plan["slot"] + ".json"], "unresolved_intents": []}))}
        extra = {"protocol": "sft-anchor-terminal-evaluation-attempt-v1", "slot": plan["slot"], "job_id": "1000", "state": "submitted",
            "kind": "one_gpu_evaluation_only", "preparation_semantic_sha256": plan["semantic_sha256"],
            "authority_sha256": args["revised_authority_sha256"], "gpus": 1, "wall_seconds": 3600, "training_or_resume": False}
        extra_binding = f.binding(f.encoded(extra))
        extra_inventory = {"protocol": "closed-terminal-evaluation-attempt-inventory-v1", "entries": [{
            "reservation_id": "evalrescue-v1/" + plan["slot"], "attempt_index": 1, "kind": "one_gpu_evaluation_only", "attempt_binding": extra_binding}],
            "directory_listing_binding": f.binding(f.encoded({"directory": a.old.NAMESPACE + "/evaluation_rescue_operator_v1_control/attempts",
                "regular_json_files": [plan["slot"] + ".json"], "unresolved_intents": []}))}
        authority = json.loads(args["revised_authority"].read_bytes())
        certificate = {"protocol": a.PROTOCOL, "schema_version": 1, "synthetic_offline_fixture": True,
            "preparation_semantic_sha256": plan["semantic_sha256"], "current_job_id": "1000",
            "reservations": a.old.reservations() + [{key: row[key] for key in ("reservation_id", "slot", "gpus", "requested_wall_seconds", "attempt_limit", "purpose")}
                for row in authority["rescue_reservations"]], "original_attempts": [wrapped],
            "original_attempt_inventory_binding": f.binding(f.encoded(inventory)),
            "original_attempt_inventory_sha256": a.old.sha(f.encoded(inventory)),
            "rescue_attempt_inventory_binding": f.binding(f.encoded(extra_inventory)),
            "jobs": [{"job_id": "1030929", "reservation_id": "prior-v1/1030929", "accounting": sacct("1030929", "FAILED", 459, 4)},
                {"job_id": "999", "reservation_id": "phase2/" + plan["slot"], "accounting": sacct("999", "TIMEOUT", 7200, 4)},
                {"job_id": "1000", "reservation_id": "evalrescue-v1/" + plan["slot"], "accounting": sacct("1000", "RUNNING", 60, 1), "attempt_binding": extra_binding}]}
        cert_sha = dump(cert_path, certificate)
        entry.update(authority_sha256=args["revised_authority_sha256"], cumulative_maximum_requested_gpu_hours=112,
            original_attempt_inventory_sha256=a.old.sha(f.encoded(inventory)), rescue_attempt_inventory_sha256=a.old.sha(f.encoded(extra_inventory)))
    else:
        certificate = None; cert_sha = None
    registry_path = tmp_path / "registry.json"
    registry = {"protocol": m.PROTOCOL + "-source-registry", "schema_version": 1, "entries": [entry]}
    registry_sha = dump(registry_path, registry)
    kwargs = dict(plan_path=plan_path, plan_sha256=plan_sha, registry_path=registry_path, registry_sha256=registry_sha,
        amendment_path=EXP / "planning/evaluation_rescue_amendment_v1/amendment.json", contract_path=contract_path,
        contract_sha256=contract_sha, training_output=args["training_output"], base=base, output=output, device="cpu",
        certificate_path=cert_path if certificate is not None else None, certificate_sha256=cert_sha, current_job_id="1000")
    return kwargs, args, plan, registry, certificate, originals

@pytest.mark.parametrize("task,arm", [("op", "rloo_cardinal_beta005"), ("cvrp", "foarl")])
def test_actual_tiny_typed_reload_missing_only_generation_and_full_strict_replay(tmp_path, monkeypatch, task, arm):
    kwargs, args, plan, registry, certificate, originals = setup(tmp_path, task, arm)
    before = args["training_output"].joinpath("evaluation_groups.jsonl").read_bytes()
    monkeypatch.setattr(m.helper3, "make_recovery_trainer", lambda *args, **kwargs: pytest.fail("Created optimizer trainer"))
    result, complete = m.execute(**kwargs)
    actual = [json.loads(line) for line in (kwargs["output"] / "operator_evaluation_groups.jsonl").read_text().splitlines()]
    assert [group["raw_completions"] for group in actual] == [group["raw_completions"] for group in originals]
    assert args["training_output"].joinpath("evaluation_groups.jsonl").read_bytes() == before
    assert complete["new_responses"] == 27 and complete["reused_responses"] == 9
    assert complete["actual_new_model_calls"] == 6 and not complete["training_or_resume"]
    assert complete["accounting"]["total_reserved_gpu_hours"] == 112
    assert not complete["accounting"]["native_production_accounting_verified"]
    assert complete["requires_independent_post_job_completed_accounting"] and not complete["publication_accounting_complete"]
    assert result["resources"]["training_responses"] == 144
    assert not (kwargs["output"] / "complete.json").exists()
    with pytest.raises(ValueError, match="sole registered"):
        m.execute(**kwargs)

def test_actual_complete_prefix_offline_zero_model_calls(tmp_path, monkeypatch):
    kwargs, args, plan, registry, certificate, originals = setup(tmp_path, prefix=None, tail=False)
    monkeypatch.setattr(m.source2, "loaded_model", lambda *args, **kwargs: pytest.fail("Offline loader"))
    monkeypatch.setattr(m.helper3, "_generate_group", lambda *args, **kwargs: pytest.fail("Offline generation"))
    result, complete = m.execute(**kwargs)
    assert complete["actual_new_model_calls"] == complete["new_responses"] == 0
    assert complete["reused_responses"] == 36 and complete["new_reload_proof"] is None
    assert result["resources"]["evaluation_wall_seconds_is_lower_bound"]

def test_actual_saved_four_rank_terminal_proof_missing_only_executor(tmp_path, monkeypatch):
    fake = tmp_path / "fixture_root"
    destination = fake / "validation/v2_peft_cpu_all_arms_v1/op"; destination.parent.mkdir(parents=True)
    destination.symlink_to(EXP / "validation/v2_peft_cpu_ddp4_v1/op", target_is_directory=True)
    authority = fake / "collected_status_v2_20261014T0911/status"; authority.mkdir(parents=True)
    authority.joinpath("campaign_authority.json").write_bytes((EXP / "collected_status_v2_20261014T0911/status/campaign_authority.json").read_bytes())
    monkeypatch.setattr(f, "EXP", fake)
    kwargs, args, plan, registry, certificate, originals = setup(tmp_path)
    proof = json.loads((args["training_output"] / "training_complete.json").read_bytes())
    assert len(proof["distributed"]["rank_closure"]) == 4
    result, complete = m.execute(**kwargs)
    assert complete["new_responses"] == 27 and complete["reused_responses"] == 9
    assert result["resources"]["training_responses"] == 144
    assert complete["original_training_proof"]["sha256"] == plan["training_proof"]["sha256"]

@pytest.mark.parametrize("fault", ["empty_registry", "source", "actor", "prefix", "base_path", "base_config", "plan", "contract", "native_validation_cpu"])
def test_pre_generation_provenance_guards(tmp_path, monkeypatch, fault):
    kwargs, args, plan, registry, certificate, originals = setup(tmp_path)
    if fault == "empty_registry": registry["entries"] = []
    elif fault == "source": registry["entries"][0]["sources"]["executor"]["sha256"] = "0" * 64
    elif fault == "actor": registry["entries"][0]["whole_actor_sha256"] = "0" * 64
    elif fault == "prefix": args["training_output"].joinpath("evaluation_groups.jsonl").write_bytes(f.encoded(originals[0]))
    elif fault == "base_path": kwargs["base"] = tmp_path / "different_base"
    elif fault == "base_config":
        path = kwargs["base"] / "config.json"
        # Fixtures share the base; copy before tampering so originals stay exact.
        import shutil
        cloned = tmp_path / "base_copy"; shutil.copytree(kwargs["base"], cloned); kwargs["base"] = cloned
        registry["entries"][0]["base_model_binding"]["path"] = str(cloned.resolve())
        value = json.loads((cloned / "config.json").read_bytes()); value["rope_theta"] = 123456
        dump(cloned / "config.json", value)
    elif fault == "plan": kwargs["plan_sha256"] = "0" * 64
    elif fault == "contract": kwargs["contract_sha256"] = "0" * 64
    else: kwargs["native_validation_only"] = True
    kwargs["registry_sha256"] = dump(kwargs["registry_path"], registry)
    monkeypatch.setattr(m.source2, "loaded_model", lambda *args, **kwargs: pytest.fail("Unadmitted actor loaded"))
    with pytest.raises(ValueError): m.execute(**kwargs)

@pytest.mark.parametrize("fault", ["drop_prior", "drop_original", "drop_reservation", "zero_gpu", "extra_gpu", "elapsed", "duplicate_job", "inventory", "extra_listing", "original_script", "wrong_authority", "wrong_job", "synthetic_production"])
def test_new_cumulative_accounting_closed_inventory_and_cost_guards(tmp_path, fault):
    kwargs, args, plan, registry, value, _ = setup(tmp_path)
    entry = registry["entries"][0]
    if fault == "drop_prior": value["jobs"].pop(0)
    elif fault == "drop_original": value["jobs"].pop(1)
    elif fault == "drop_reservation": value["reservations"].pop()
    elif fault == "zero_gpu": value["jobs"][-1]["accounting"] = sacct("1000", "RUNNING", 60, 0)
    elif fault == "extra_gpu": value["jobs"][-1]["accounting"] = sacct("1000", "RUNNING", 60, 2)
    elif fault == "elapsed": value["jobs"][-1]["accounting"] = sacct("1000", "RUNNING", 7201, 1)
    elif fault == "duplicate_job": value["jobs"].append(copy.deepcopy(value["jobs"][-1]))
    elif fault == "inventory": entry["original_attempt_inventory_sha256"] = "0" * 64
    elif fault == "extra_listing":
        inventory = json.loads(a.old.bound_bytes(value["rescue_attempt_inventory_binding"]))
        listing = json.loads(a.old.bound_bytes(inventory["directory_listing_binding"])); listing["unresolved_intents"] = ["ambiguous"]
        inventory["directory_listing_binding"] = f.binding(f.encoded(listing)); value["rescue_attempt_inventory_binding"] = f.binding(f.encoded(inventory))
        entry["rescue_attempt_inventory_sha256"] = a.old.sha(f.encoded(inventory))
    elif fault == "original_script": value["original_attempts"][0]["script_binding"] = f.binding(b"forged", value["original_attempts"][0]["script_binding"]["native_path"])
    elif fault == "wrong_authority": args["revised_authority_sha256"] = "0" * 64
    elif fault == "wrong_job": kwargs["current_job_id"] = "1001"
    else: plan["scope"] = "production"
    sha = dump(kwargs["certificate_path"], value)
    with pytest.raises((ValueError, KeyError)):
        a.validate(kwargs["certificate_path"], expected_sha256=sha, preparation=plan, current_job_id=kwargs["current_job_id"],
            revised_authority_path=args["revised_authority"], revised_authority_sha256=args["revised_authority_sha256"],
            expected_cumulative_gpu_hours=112, original_attempt_inventory_sha256=entry["original_attempt_inventory_sha256"],
            rescue_attempt_inventory_sha256=entry["rescue_attempt_inventory_sha256"])

def test_executed_preparer_identity_uses_compiled_buffer_not_later_read(tmp_path, monkeypatch):
    p = m.preparer(); admitted = p.__executed_source_binding
    original_identity = p.identity
    monkeypatch.setattr(p, "identity", lambda path: {"path": str(path), "sha256": "0" * 64, "size_bytes": 1}
        if Path(path).name == "prepare_evaluation_rescue_v1.py" else original_identity(path))
    assert p.source_registration()["operator_source"] == admitted

def test_historical_measured_authority_keeps_prior_extra_attempt_and_all112_reservations(tmp_path):
    kwargs, args, plan, registry, value, _ = setup(tmp_path)
    other = "op.rloo_scalar_beta0"
    history = json.loads(args["revised_authority"].read_bytes())
    current_timing = next(row["measured_runtime"] for row in history["rescue_reservations"] if row["slot"] == plan["slot"])
    next(row for row in history["rescue_reservations"] if row["slot"] == other)["measured_runtime"] = copy.deepcopy(current_timing)
    history["semantic_sha256"] = f.m.digest({key: val for key, val in history.items() if key != "semantic_sha256"})
    historical_binding = f.binding(f.encoded(history))
    value["authority_history_bindings"] = [{"slot": other, "authority_binding": historical_binding}]
    old_attempt = json.loads(a.old.bound_bytes(value["jobs"][-1]["attempt_binding"]))
    old_attempt.update(slot=other, job_id="1002", authority_sha256=historical_binding["sha256"], preparation_semantic_sha256="1" * 64)
    old_binding = f.binding(f.encoded(old_attempt))
    value["jobs"].append({"job_id": "1002", "reservation_id": "evalrescue-v1/" + other,
        "accounting": sacct("1002", "TIMEOUT", 3600, 1), "attempt_binding": old_binding})
    inventory = json.loads(a.old.bound_bytes(value["rescue_attempt_inventory_binding"]))
    inventory["entries"].append({"reservation_id": "evalrescue-v1/" + other, "attempt_index": 1, "kind": "one_gpu_evaluation_only", "attempt_binding": old_binding})
    listing = json.loads(a.old.bound_bytes(inventory["directory_listing_binding"]))
    listing["regular_json_files"] = sorted([plan["slot"] + ".json", other + ".json"])
    inventory["directory_listing_binding"] = f.binding(f.encoded(listing))
    value["rescue_attempt_inventory_binding"] = f.binding(f.encoded(inventory))
    entry = registry["entries"][0]; entry["rescue_attempt_inventory_sha256"] = a.old.sha(f.encoded(inventory))
    def admit():
        sha = dump(kwargs["certificate_path"], value)
        return a.validate(kwargs["certificate_path"], expected_sha256=sha, preparation=plan, current_job_id="1000",
            revised_authority_path=args["revised_authority"], revised_authority_sha256=args["revised_authority_sha256"],
            expected_cumulative_gpu_hours=112, original_attempt_inventory_sha256=entry["original_attempt_inventory_sha256"],
            rescue_attempt_inventory_sha256=entry["rescue_attempt_inventory_sha256"])
    proof = admit()
    assert proof["total_reserved_gpu_hours"] == 112 and len(proof["extra_attempts"]) == 2
    value["authority_history_bindings"] = []
    with pytest.raises(ValueError, match="authority identity"):
        admit()

def test_default_registry_is_empty_and_cannot_admit_production():
    value = json.loads((EXP / "planning/evaluation_rescue_source_registry_v1.json").read_bytes())
    assert value["entries"] == []

def test_offline_registration_requires_zero_call_source_admission_without_fake_cuda_witness(tmp_path):
    kwargs, args, plan, registry, _, _ = setup(tmp_path, prefix=None, tail=False)
    # Admission-only schema exercise. Original smoke proof is never promoted.
    plan["scope"] = "production"; registry["entries"][0]["scope"] = "production"
    entry = registry["entries"][0]; entry.pop("synthetic_offline_fixture")
    entry.update(admission_kind="offline_zero_gpu_closure", production_source_admitted=True, zero_model_calls=True, new_allocation=False)
    sha = dump(kwargs["registry_path"], registry)
    admitted, _ = m.admit_registration(kwargs["registry_path"], sha, plan, amendment_path=kwargs["amendment_path"])
    assert admitted["new_allocation"] is False and "native_validation_receipt" not in admitted
    entry["zero_model_calls"] = False; sha = dump(kwargs["registry_path"], registry)
    with pytest.raises(ValueError, match="zero-model-call"):
        m.admit_registration(kwargs["registry_path"], sha, plan, amendment_path=kwargs["amendment_path"])

def test_actual_cuda_validation_first_group_is_reused_without_resampling(tmp_path, monkeypatch):
    """Run explicitly on an independently checked idle local device, FP32 tiny."""
    if not torch.cuda.is_available(): pytest.skip("Explicit local CUDA mechanics invocation only")
    kwargs, args, plan, registry, certificate, originals = setup(tmp_path, prefix=6, tail=False)
    kwargs["device"] = "cuda"; kwargs["native_validation_only"] = True
    witness_output = kwargs["output"]
    result, witness = m.execute(**kwargs)
    assert result is None and witness["scope"] == "local_cuda_fp32_tiny_fixture"
    assert witness["dtype"] == "torch.float32" and witness["model_parameter_count"] < 6_000_000_000
    assert witness["generated_responses"] == witness["model_calls"] == 1
    assert witness["typed_load_and_exact_actor"] and not witness["performance_admission"]
    assert witness["parameter_dtype_counts"] == {"torch.float32": witness["model_parameter_count"]}
    assert witness["invocation_wall_seconds"] > 0
    assert not (witness_output / "operator_result.json").exists()
    # A caller's native_production_admitted flag cannot promote the actual local
    # FP32/2080/Torch2.5/tiny witness to the BF16/7B production target.
    production_plan = copy.deepcopy(plan); production_plan["scope"] = "production"
    production_registry = copy.deepcopy(registry); production_entry = production_registry["entries"][0]
    production_entry.pop("synthetic_offline_fixture"); production_entry.update(scope="production", native_production_admitted=True,
        expected_native_environment=witness["environment"], expected_parameter_dtype_counts=witness["parameter_dtype_counts"],
        native_validation_receipt=f.binding((witness_output / "operator_native_validation.json").read_bytes()))
    production_path = tmp_path / "production_rejected_registry.json"; production_sha = dump(production_path, production_registry)
    with pytest.raises(ValueError, match="Actual source-bound native"):
        m.admit_registration(production_path, production_sha, production_plan, amendment_path=kwargs["amendment_path"])
    entry = registry["entries"][0]
    entry["native_validation_receipt"] = f.binding((witness_output / "operator_native_validation.json").read_bytes())
    entry["native_validation_reload_receipt"] = f.binding((witness_output / "operator_checkpoint_reload.json").read_bytes())
    output = tmp_path / "after_validation"
    entry["output_directory"] = str(output.resolve()); kwargs["output"] = output; kwargs["native_validation_only"] = False
    kwargs["registry_sha256"] = dump(kwargs["registry_path"], registry)
    generate = m.helper3._generate_group; calls = []
    def counted(*args, **kwargs):
        calls.append((args[3]["instance_id"], args[4])); return generate(*args, **kwargs)
    monkeypatch.setattr(m.helper3, "_generate_group", counted)
    result, complete = m.execute(**kwargs)
    assert calls == [tuple(plan["missing_group_keys"][1])]
    assert (output / "operator_evaluation_groups.jsonl").read_bytes().startswith(
        (witness_output / "operator_evaluation_groups.jsonl").read_bytes())
    assert complete["new_responses"] == 9 and complete["reused_responses"] == 27
    assert complete["actual_new_model_calls"] == 2
    assert complete["model_calls_in_current_invocation"] == complete["reused_validation_groups"] == 1
    assert complete["validation_invocation_wall_seconds"] == witness["invocation_wall_seconds"]
    assert not complete["publication_accounting_complete"]

def test_injected_valid_prefix_buffer_cannot_rebind_unchanged_original_binding(tmp_path, monkeypatch):
    kwargs, args, plan, registry, value, originals = setup(tmp_path)
    original_preparer = m.preparer
    def injected_preparer():
        p = original_preparer(); stable = p.closed_prefix_snapshot
        def forged(*args, **kwargs):
            rows, binding, tail, raw = stable(*args, **kwargs)
            return rows, binding, tail, f.encoded(originals[0])
        p.closed_prefix_snapshot = forged
        return p
    monkeypatch.setattr(m, "preparer", injected_preparer)
    monkeypatch.setattr(m.source2, "loaded_model", lambda *args, **kwargs: pytest.fail("Forged prefix loaded model"))
    with pytest.raises(ValueError, match="stable prefix"):
        m.execute(**kwargs)

