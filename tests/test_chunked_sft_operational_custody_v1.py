"""Synthetic timing/spec custody over actual saved tiny initializer inventories.

No original7B inventory is fabricated and no native policy is called. The
positive diagnostic remains explicitly synthetic; production signatures deny.
"""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from test_chunked_confirmation_analysis_v1 import m, EXP, ROOT
from test_sft_anchor_confirmatory_analysis import MemoryEvidence, seal
from test_sft_anchor_confirmation_custody import ByteProvider

def fixture(tmp_path, task="op"):
    packet = MemoryEvidence()
    actual = EXP / "validation/v2_peft_cpu_all_arms_v1" / task
    tiny = json.loads((actual / "contract.json").read_bytes())
    task_binding = packet.put("fixture/actual-tiny-contract.json", (actual / "contract.json").read_bytes())
    source_binding = packet.put("fixture/source-metadata.json", {"synthetic_fixture": True})
    contract = {"synthetic_fixture": True, "contract_sha256": tiny["contract_sha256"],
        "original_sft_initializers": {task: tiny["initializer_sha256"]}, "seed_roster": [{"draw_id": "draw00", "train_seed": 100}],
        "native_source_root": "/synthetic/source", "native_control_roots": {"draw00": "/synthetic/control/draw00"},
        "source_manifest_binding": source_binding, "task_contract_bindings": {"draw00": {task: task_binding}}}
    commitments = {name: packet.put("fixture/" + name + ".json", {"synthetic_fixture": True,
        "observed_utc": f"2030-01-01T00:00:0{second}+00:00"}) for name, second in
        (("selection", 1), ("seed_roster", 2), ("source", 3), ("round_registration", 4), ("first_policy_call", 5))}
    chronology = packet.put("fixture/chronology.json", {"synthetic_fixture": True, "commitments": commitments})
    A, C, _ = m.dependencies(lambda *args, **kwargs: False)
    root = contract["native_control_roots"]["draw00"]; op = "sft-anchor-formal-source2-operational-v1"
    source2 = ROOT / C.ENTRYPOINT; raw = source2.read_bytes()
    entries = {C.ENTRYPOINT: {"path": C.ENTRYPOINT, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}}
    inventory = {"protocol": op + "-initializer-artifacts", "synthetic_fixture": True, "scope": "system_smoke",
        "task": task, "initializer_sha256": tiny["initializer_sha256"], "registered_before_policy_calls": True, "artifacts": {}}
    for kind, folder in (("base", actual / "fixture_base"), ("sft", actual / "fixture_sft")):
        rows = []
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                data = path.read_bytes(); rows.append({"path": path.relative_to(folder).as_posix(),
                    "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})
        inventory["artifacts"][kind] = {"root": str(folder.resolve()), "files": rows}
    seal(inventory, "inventory_sha256"); inventory_binding = packet.put("sft/inventory.json", inventory)
    original_binding = packet.put("sft/original_source2_manifest.json", (EXP / "source_v2/source_manifest.json").read_bytes())
    spec = {"protocol": op + "-slot-spec", "schema_version": 1, "synthetic_fixture": True,
        "slot_id": "sft_once/" + task + ".sft", "draw_id": "sft_once", "task": task, "arm": "sft",
        "train_seed": 100, "gpus": 1, "training_ranks": 1, "slurm_launcher_tasks": 1,
        "first_attempt": True, "automatic_retry": False, "round_contract_sha256": contract["contract_sha256"],
        "source_root": contract["native_source_root"], "source_manifest_binding": contract["source_manifest_binding"],
        "original_source_manifest_binding": original_binding, "task_contract_binding": contract["task_contract_bindings"]["draw00"][task],
        "output": root + "/outputs/" + task + ".sft", "observer_directory": root + "/observers/" + task + ".sft",
        "base": inventory["artifacts"]["base"]["root"], "sft": inventory["artifacts"]["sft"]["root"]}
    for key, filename in (("launcher_binding", "formal_srun_launch_v1.py"), ("worker_binding", "formal_source2_worker_v1.py"),
        ("bootstrap_binding", "evaluation_rescue_bootstrap_v1.py"), ("source_layer_binding", "formal_source_layer_v2.py")):
        path = EXP / filename; data = path.read_bytes(); relative = path.relative_to(ROOT).as_posix()
        row = {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
        entries[relative] = row; spec[key] = {**row, "path": contract["native_source_root"] + "/" + relative}
    seal(spec, "spec_sha256"); spec_binding = packet.put("sft/spec.json", spec)
    env = packet.put("sft/environment.json", {"synthetic_fixture": True, "scope": "actual_saved_tiny_CPU_metadata",
        "device": "cpu", "dtype": "float32", "native_production_admitted": False})
    registration = {"protocol": op + "-independent-prepolicy-registration", "synthetic_fixture": True,
        "registered_round_contract_sha256": contract["contract_sha256"], "registered_slot_spec_sha256": spec["spec_sha256"],
        "source_manifest_sha256": contract["source_manifest_binding"]["sha256"], "original_source_manifest_sha256": m.ORIGINAL_SOURCE_MANIFEST_SHA,
        "initializer_artifacts_binding": inventory_binding, "native_environment_binding": env,
        "policy_outcomes_read": False, "independent_read_only_custody": True, "registered_utc": "2030-01-01T00:00:04+00:00"}
    reg_binding = packet.put("sft/registration.json", registration)
    native_entry = {"slot_id": spec["slot_id"], "slot_spec_sha256": spec["spec_sha256"], "job_id": "990000",
        "first_attempt": True, "slurm_step_id": "0", "source2_sha256": C.ENTRYPOINT_SHA256,
        "independent_registration_binding": reg_binding, "once_per_task_sft_sidecar": True,
        "initializer_artifacts_binding": inventory_binding, "native_environment_binding": env}
    operational = packet.put("sft/operational_registry.json", {"protocol": op + "-registry", "schema_version": 1,
        "synthetic_fixture": True, "entries": [native_entry]})
    entry = {"sidecar_slot_spec_bindings": {task: spec_binding}, "sidecar_independent_prepolicy_registration_bindings": {task: reg_binding},
        "sidecar_operational_registry_binding": operational, "original_source_manifest_binding": original_binding}
    context = {"contract": contract, "packet_reader": packet, "registration_entry": entry, "entries": entries,
        "analysis": A, "collector": C, "synthetic_fixture": True}
    store = C.ChunkStore(tmp_path / "chunks"); reader = C.ChunkReader(store.root)
    observer = {"protocol": op + "-first-loaded-model-forward", "synthetic_fixture": True, "schema_version": 1,
        "slot_id": spec["slot_id"], "draw_id": "sft_once", "task": task, "arm": "sft", "train_seed": 100,
        "round_contract_sha256": contract["contract_sha256"], "slot_spec_sha256": spec["spec_sha256"],
        "job_id": "990000", "slurm_step_id": "0", "rank": 0, "loaded_model_role": "reference_or_frozen_evaluation",
        "source2": {"path": contract["native_source_root"] + "/" + C.ENTRYPOINT, "sha256": C.ENTRYPOINT_SHA256, "size_bytes": len(raw)},
        "outcomes_recorded": False, "independent_custody_receipt": False, "requires_independent_read_only_collection": True,
        "model_arguments_outputs_or_rng_modified": False, "observed_utc": "2030-01-01T00:00:06+00:00"}
    seal(observer, "event_sha256"); path = root + f"/observers/{task}.sft/first_model_forward_rank0.json"
    observer_binding = store.ingest(ByteProvider({path: m.encoded(observer)}), path)
    custody = {"synthetic_fixture": True, "performance_admission": False,
        "consumer_protocol": m.PROTOCOL + "-once-sft-observer-custody", "consumer_core_source_sha256": m.__executed_source_sha256__,
        "verifier_source_sha256": m.PROMOTION_VERIFIER_SHA, "registered_contract_sha256": contract["contract_sha256"],
        "slot_spec_binding": spec_binding, "independent_prepolicy_registration_binding": reg_binding,
        "initializer_artifacts_binding": inventory_binding, "native_environment_binding": env, "observer_binding": observer_binding,
        "actual_job_id": "990000", "slurm_step_id": "0", "rank": 0, "independent_read_only_custody": True,
        "initializer_inventory_replayed_before_model_load": True, "producer_observer_verified_from_actual_native_readback": True}
    slot = {"task": task, "observer_binding": observer_binding,
        "independent_observer_custody_binding": packet.put("sft/observer_custody.json", custody)}
    cost = {"terminal": {("sft_baseline", task, "sft"): {"state": "COMPLETED", "job_id": "990000"}}}
    return context, slot, reader, cost, chronology, store

@pytest.mark.parametrize("task", ["op", "cvrp"])
def test_once_sft_prepolicy_source_spec_job_and_actual_tiny_initializer_inventory_diagnostic(tmp_path, task):
    context, side, reader, cost, chronology, _ = fixture(tmp_path, task)
    proof = m.verify_sft_operational_custody(context, side, reader, cost, chronology)
    assert not proof["independent_once_sft_prepolicy_custody"]
    assert proof["initializer_artifacts_binding"]["sha256"]

@pytest.mark.parametrize("fault", ["late_registration", "prior_global_call", "job", "rank", "step", "role", "source", "independent_producer", "inventory", "environment", "spec", "worker", "custody"])
def test_once_sft_timing_initializer_first_forward_and_independent_readback_guards(tmp_path, fault):
    context, side, reader, cost, chronology, store = fixture(tmp_path)
    packet = context["packet_reader"]; entry = context["registration_entry"]
    if fault in ("late_registration", "environment"):
        binding = entry["sidecar_independent_prepolicy_registration_bindings"]["op"]
        data = json.loads(packet(binding))
        if fault == "late_registration": data["registered_utc"] = "2030-01-01T00:00:07+00:00"
        else: data["native_environment_binding"] = {}
        packet.files[binding["path"]] = m.encoded(data); changed = packet.put("sft/changed-registration.json", data)
        entry["sidecar_independent_prepolicy_registration_bindings"]["op"] = changed
    elif fault in ("inventory", "spec", "worker"):
        if fault == "inventory":
            opreg = json.loads(packet(entry["sidecar_operational_registry_binding"])); binding = opreg["entries"][0]["initializer_artifacts_binding"]
            data = json.loads(packet(binding)); data["artifacts"]["base"]["files"][0]["path"] = "../config.json"
            seal(data, "inventory_sha256"); packet.files[binding["path"]] = m.encoded(data)
        elif fault == "spec": entry["sidecar_slot_spec_bindings"] = {}
        else: context["entries"].pop("experiments/rloo_sft_anchor_20261013/formal_source2_worker_v1.py")
    elif fault == "custody":
        data = json.loads(packet(side["independent_observer_custody_binding"])); data["performance_admission"] = True
        side["independent_observer_custody_binding"] = packet.put("sft/changed-custody.json", data)
    else:
        data = json.loads(reader.small(side["observer_binding"]))
        if fault == "prior_global_call": data["observed_utc"] = "2030-01-01T00:00:04+00:00"
        elif fault == "job": data["job_id"] = "990001"
        elif fault == "rank": data["rank"] = 1
        elif fault == "step": data["slurm_step_id"] = "batch"
        elif fault == "role": data["loaded_model_role"] = "actor"
        elif fault == "source": data["source2"]["sha256"] = "0" * 64
        else: data["independent_custody_receipt"] = True
        seal(data, "event_sha256"); path = reader.native_identity(side["observer_binding"])["native_path"]
        side["observer_binding"] = store.ingest(ByteProvider({path: m.encoded(data)}), path)
    with pytest.raises((ValueError, KeyError, AssertionError)):
        m.verify_sft_operational_custody(context, side, reader, cost, chronology)
