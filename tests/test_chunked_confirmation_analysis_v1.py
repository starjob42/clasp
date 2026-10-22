"""Synthetic closed20/82-slot custody and actual saved SFT fixture replay.

No real cohort, roster, candidate, authority, registration or native action.
"""
import copy
import hashlib
import json
from pathlib import Path
import types

import pytest
from test_sft_anchor_confirmation_custody import capture_slots, ByteProvider, native_path, repair_terminal
from test_sft_anchor_confirmatory_analysis import synthetic_round, encoded, seal
from test_sft_anchor_confirmatory_analysis import MemoryEvidence
from test_sft_anchor_admission_v3_collection import _durable_slot

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"
m = types.ModuleType("new_chunked_analysis"); m.__file__ = str(EXP / "analyze_chunked_confirmation_v1.py")
body = Path(m.__file__).read_bytes(); m.__executed_source_sha256__ = hashlib.sha256(body).hexdigest()
exec(compile(body, m.__file__, "exec"), m.__dict__)

def fixture(tmp_path, *, duplicate_seed=False):
    packet = synthetic_round(duplicate_seed=duplicate_seed); contract, bundle, packet_reader = packet
    A, C, proof = m.dependencies(lambda *args, **kwargs: False)
    plans = [{"draw_id": draw["draw_id"], "task": task, "arm": arm, "gpus": 4, "requested_wall_seconds": 60,
        "maximum_attempts": 1, "automatic_retry": False} for draw in contract["seed_roster"] for task in ("op", "cvrp")
        for arm in (contract["candidate_arm"], "foarl")]
    plans.extend({"draw_id": "sft_baseline", "task": task, "arm": "sft", "gpus": 1, "requested_wall_seconds": 60,
        "maximum_attempts": 1, "automatic_retry": False} for task in ("op", "cvrp"))
    old_binding = packet_reader.put("history/original_authority.json", (EXP / "collected_status_v2_20261014T0911/status/campaign_authority.json").read_bytes())
    failed_binding = packet_reader.put("history/prior_failed.json", (EXP / "validation/native_monitor/prior_failed_1030929_reverification_v2_bound.json").read_bytes())
    old = json.loads(packet_reader(old_binding))
    baseline = [{"reservation_id": "prior-v1/1030929", "gpus": 4, "requested_wall_seconds": 7200}]
    baseline += [{"reservation_id": "phase2/" + slot, "gpus": 1 if slot.endswith(".sft") else 4,
        "requested_wall_seconds": 7200} for slot in old["plan"]["order"]]
    history = seal({"protocol": m.PROTOCOL + "-historical-reservation-ledger", "synthetic_fixture": True,
        "independent_read_only_collection": True, "all_scoped_attempt_directories_reconciled": True, "unresolved_intents": [],
        "pilot_authority_binding": old_binding, "prior_failed_phase_binding": failed_binding,
        "original_pilot_reservations": baseline, "additional_pilot_authorities": [], "prior_formal_rounds": []}, "receipt_sha256")
    history_binding = packet_reader.put("history/reservation_ledger.json", history)
    new = sum(row["gpus"] * row["requested_wall_seconds"] for row in plans) / 3600
    authority = seal({"synthetic_fixture": True, "planned_slots": plans, "historical_pilot_requested_gpu_hours": 92,
        "historical_formal_requested_gpu_hours": 0, "new_round_requested_gpu_hours": new,
        "combined_reserved_requested_gpu_hours": 92 + new, "declared_requested_gpu_hour_ceiling": 100,
        "historical_resource_ledger_binding": history_binding}, "authority_sha256")
    authority_binding = packet_reader.put("new_registration/authority.json", authority)
    first = contract["seed_roster"][0]
    entry = {"protocol": m.PROTOCOL, "contract_sha256": contract["contract_sha256"], "synthetic_fixture": True,
        "performance_admission": False, "adapter_source_sha256": m.__executed_source_sha256__,
        "analysis_source_sha256": m.A_SHA, "collector_source_sha256": m.C_SHA, "primary_decoder": "k8", "global_test_count": 1,
        "fixed_draw_count": 20, "resource_authority_binding": authority_binding,
        "historical_resource_ledger_binding": history_binding,
        "sidecar_control_roots": {task: contract["native_control_roots"][first["draw_id"]] for task in ("op", "cvrp")},
        "sidecar_task_contract_bindings": contract["task_contract_bindings"][first["draw_id"]]}
    registry = {"protocol": m.PROTOCOL + "-registry", "schema_version": 1, "entries": [entry]}
    registry_binding = packet_reader.put("new_registration/registry.json", registry)
    context = m.prepare_context(contract, packet_reader, registry_binding,
        expected_registry_sha256=registry_binding["sha256"], synthetic_fixture=True)
    C = context["collector"]; store = C.ChunkStore(tmp_path / "chunks", chunk_bytes=16384)
    slots, sources = capture_slots(packet, store, count=80)
    reader = C.ChunkReader(store.root); C.verify_source_snapshot(context, sources, reader)
    sidecars = []; files = {}; native = entry["sidecar_control_roots"]["op"]
    for offset, task in enumerate(("op", "cvrp")):
        task_contract = context["task_contracts"][(first["draw_id"], task)]
        payloads, _ = _durable_slot(task + ".sft", task_contract,
            {"native_source_root": contract["native_source_root"], "native_control_root": native}, context["entries"][C.ENTRYPOINT])
        tree = repair_terminal(payloads, b"SYNTHETIC ZERO-TRAINING SFT BYTES")
        complete = json.loads(payloads["complete.json"]); complete["actual_evaluation_responses"] = 18; payloads["complete.json"] = encoded(complete)
        attempt = {"protocol": A.PROTOCOL + "-attempt", "synthetic_fixture": True, "state": "submitted", "job_id": str(990000 + offset),
            "attempt_number": 1, "draw_id": "sft_once", "train_seed": first["train_seed"], "task": task, "arm": "sft",
            "round_contract_sha256": contract["contract_sha256"], "native_control_root": native, "requested_gpus": 1, "replacement_or_retry": False}
        payloads["attempt.json"] = encoded(attempt)
        for name, raw in payloads.items():
            path = native + (f"/attempts/{task}.sft.json" if name == "attempt.json" else
                f"/outputs/{task}.sft/complete.json" if name == "complete.json" else f"/outputs/{task}.sft/{task}/sft/" + name)
            files[path] = raw
        terminal = native + f"/outputs/{task}.sft/{task}/sft/terminal"
        files[terminal + "/fixture_adapter.safetensors"] = b"SYNTHETIC ZERO-TRAINING SFT BYTES"
        sidecars.append({"synthetic_fixture": True, "draw_id": "sft_once", "task": task, "arm": "sft", "state": "complete",
            "native_terminal_artifact_proof": tree})
    provider = ByteProvider(files)
    for side in sidecars:
        task = side["task"]
        names = ["complete.json", "result.json", "training_complete.json", "checkpoint_reload_exact.json", "evaluation_groups.jsonl", "attempt.json"]
        side["artifacts"] = {}
        for name in names:
            path = native + (f"/attempts/{task}.sft.json" if name == "attempt.json" else
                f"/outputs/{task}.sft/complete.json" if name == "complete.json" else f"/outputs/{task}.sft/{task}/sft/" + name)
            side["artifacts"][name] = store.ingest(provider, path)
        _, side["terminal_file_chunks"] = C.capture_terminal_tree(store, provider, native + f"/outputs/{task}.sft/{task}/sft/terminal")
    costs = []; allocations = {}
    for slot in slots + sidecars:
        attempt_binding = slot["artifacts"]["attempt.json"]; attempt = json.loads(reader.small(attempt_binding)); job = attempt["job_id"]
        gpus = 1 if slot["arm"] == "sft" else 4
        query = {"synthetic_fixture": True, "independent_read_only_collection": True, "returncode": 0,
            "argv": ["sacct", "--noheader", "--parsable2", "--jobs", job, "--format=JobIDRaw,State%40,ElapsedRaw,ExitCode,AllocTRES"],
            "stdout": "\n".join(f"{job}{suffix}|COMPLETED|10|0:0|gres/gpu={gpus}" for suffix in ("", ".batch", ".extern", ".0"))}
        path = "/synthetic/independent-allocation/" + job + ".json"; allocations[path] = encoded(query)
        costs.append({"draw_id": "sft_baseline" if slot["arm"] == "sft" else slot["draw_id"], "task": slot["task"], "arm": slot["arm"],
            "state": "closed_attempt", "attempt_count": 1, "attempt_binding": attempt_binding, "allocation_native_path": path})
    alloc_provider = ByteProvider(allocations)
    for cost in costs: cost["allocation_binding"] = store.ingest(alloc_provider, cost.pop("allocation_native_path"))
    directories = []
    for draw in contract["seed_roster"]:
        names = sorted(task + "." + arm + ".json" for task in ("op", "cvrp") for arm in (contract["candidate_arm"], "foarl"))
        if draw["draw_id"] == first["draw_id"]: names = sorted(names + ["op.sft.json", "cvrp.sft.json"])
        directories.append({"native_control_root": contract["native_control_roots"][draw["draw_id"]], "regular_attempt_json_files": names, "unresolved_intents": []})
    listing = seal({"protocol": m.PROTOCOL + "-independent-attempt-directory-inventory", "synthetic_fixture": True,
        "contract_sha256": contract["contract_sha256"], "independent_read_only_collection": True, "unresolved_intents": [], "directories": directories}, "receipt_sha256")
    cost = seal({"protocol": m.PROTOCOL + "-closed-attempt-cost-custody", "synthetic_fixture": True,
        "contract_sha256": contract["contract_sha256"], "authority_byte_sha256": authority_binding["sha256"],
        "independent_read_only_collection": True, "all_scoped_attempt_directories_reconciled": True, "unresolved_intents": [],
        "attempt_directory_inventory_binding": store.write_receipt(listing), "slots": costs}, "receipt_sha256")
    return context, slots, sidecars, reader, store, bundle, cost, registry_binding

def finalize(f):
    context, slots, sidecars, reader, store, bundle, cost, _ = f; C = context["collector"]
    receipts = {}
    for draw in context["contract"]["seed_roster"]:
        chosen = [slot for slot in slots if slot["draw_id"] == draw["draw_id"]]
        receipts[draw["draw_id"]], _ = C.collect_draw(context, draw["draw_id"], chosen, reader, store)
    binding, _ = C.finalize_round(context, receipts, bundle["round_close"], reader, store, chronology_binding=bundle["chronology_binding"])
    return binding

@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    f = fixture(tmp_path_factory.mktemp("closed20"))
    return f, finalize(f)

def test_full20_chunked_receipts_two_once_sft_exact_math_and_duplicate_seed_retention(tmp_path):
    f = fixture(tmp_path, duplicate_seed=True); context, slots, sides, reader, store, bundle, cost, _ = f
    binding = finalize(f)
    _, report = m.analyze(context, binding, sides, reader, store, attempt_cost_binding=store.write_receipt(cost))
    assert len(report["seed_draws"]) == 20 and report["seed_draws"][0]["train_seed"] == report["seed_draws"][-1]["train_seed"]
    assert report["global_test"] == context["analysis"].exact_global_test(sum(row["D"] for row in report["seed_draws"]))
    assert report["global_test_count"] == 1 and report["resources"]["all82slots_retained"]
    assert set(report["once_sft_sidecars"]) == {"op", "cvrp"}
    assert all(row["raw_responses_verified"] == 18 for row in report["once_sft_sidecars"].values())
    assert not report["formal_claim_eligible"] and not report["legacy_analysis_enabled"] and report["legacy_collector_registry_entries"] == 0

@pytest.mark.parametrize("task", ["op", "cvrp"])
def test_actual_saved_sft_zero_training_typed_reload_tree_and_strict_stream_replay(tmp_path, task):
    _, C, proof = m.dependencies(lambda *args, **kwargs: False)
    parent = EXP / "validation/v2_peft_cpu_all_arms_v1" / task
    report = m.replay_saved_sft_fixture(json.loads((parent / "contract.json").read_bytes()), parent / "sft", C.ChunkStore(tmp_path / "actual"))
    assert report["raw_responses_verified"] == 36 and report["typed_reload_and_zero_training_verified"]
    assert not report["formal_claim_eligible"] and report["dependency_bridge"]["sft_horizon_replacements"] == 3

@pytest.mark.parametrize("fault", ["drop_cost", "retry", "gpus", "drop_directory", "extra_intent", "drop_sft", "endpoint", "source", "registry_empty", "registry_pin", "open_round", "drop_draw"])
def test_closed_registry_endpoint_roster_resource_attempt_and_sidecar_guards(shared, fault):
    f, original_binding = shared
    original_context, slots, original_sides, reader, store, bundle, original_cost, registry_binding = f
    context = {**original_context, "registration_entry": copy.deepcopy(original_context["registration_entry"])}
    sides = copy.deepcopy(original_sides); cost = copy.deepcopy(original_cost)
    if fault in ("endpoint", "source", "registry_empty", "registry_pin"):
        registry = json.loads(context["packet_reader"](registry_binding))
        if fault == "endpoint": registry["entries"][0]["primary_decoder"] = "greedy"
        elif fault == "source": registry["entries"][0]["adapter_source_sha256"] = "0" * 64
        elif fault == "registry_empty": registry["entries"] = []
        changed = context["packet_reader"].put("new_registration/changed.json", registry)
        with pytest.raises(ValueError): m.prepare_context(context["contract"], context["packet_reader"], changed,
            expected_registry_sha256="0" * 64 if fault == "registry_pin" else changed["sha256"], synthetic_fixture=True)
        return
    if fault in ("open_round", "drop_draw"):
        binding = original_binding; value = json.loads(reader(binding))
        if fault == "open_round": value["status"] = "open_custody_diagnostic"
        else: value["draw_receipts"].pop("draw19")
        seal(value, "receipt_sha256"); binding = store.write_receipt(value)
    else: binding = original_binding
    if fault == "drop_cost": cost["slots"].pop()
    elif fault == "retry": cost["slots"][0]["attempt_count"] = 2
    elif fault == "gpus":
        authority = json.loads(context["packet_reader"](context["registration_entry"]["resource_authority_binding"]))
        authority["planned_slots"][0]["gpus"] = 1; seal(authority, "authority_sha256")
        context["registration_entry"]["resource_authority_binding"] = context["packet_reader"].put("wrong_authority.json", authority)
    elif fault in ("drop_directory", "extra_intent"):
        listing = json.loads(reader(cost["attempt_directory_inventory_binding"]))
        if fault == "drop_directory": listing["directories"].pop()
        else: listing["directories"][0]["unresolved_intents"] = ["ambiguous"]
        seal(listing, "receipt_sha256"); cost["attempt_directory_inventory_binding"] = store.write_receipt(listing)
    elif fault == "drop_sft": sides.pop()
    seal(cost, "receipt_sha256")
    with pytest.raises(ValueError): m.analyze(context, binding, sides, reader, store, attempt_cost_binding=store.write_receipt(cost))

def test_closed_permanent_failure_stays_zero_and_keeps20_denominator(tmp_path):
    f = fixture(tmp_path); context, slots, sides, reader, store, bundle, cost, _ = f
    bundle["round_close"]["draw_states"][0].update(state="declared_permanent_failure", reason="synthetic closed technical failure")
    binding = finalize(f)
    _, report = m.analyze(context, binding, sides, reader, store, attempt_cost_binding=store.write_receipt(cost))
    assert report["seed_draws"][0]["D"] == 0 and report["seed_draws"][0]["permanent_failure"]
    assert report["global_test"]["n_registered_draws"] == 20

def test_default_registry_and_legacy_flags_remain_closed():
    default = json.loads((EXP / "planning/chunked_confirmation_analysis_registry_v1.json").read_bytes())
    A, C, proof = m.dependencies(lambda *args, **kwargs: False)
    assert default["entries"] == [] and A.PRODUCTION_ADMISSION_READY is False and C.KNOWN_REGISTRATIONS == {}
    assert proof["replaced_functions"] == ["analysis_module", "registration_anchor"]

def test_existing_fixture_flags_cannot_authorize_registered_production(shared):
    f, _ = shared; context, _, _, _, _, _, _, registry = f
    with pytest.raises(ValueError, match="signed promotion"):
        m.prepare_context(context["contract"], context["packet_reader"], registry,
            expected_registry_sha256=registry["sha256"], synthetic_fixture=False)

def test_mutated_prepared_registration_context_is_rejected(shared):
    f, binding = shared; context, slots, sides, reader, store, bundle, cost, _ = f
    changed = {**context, "registration_entry": copy.deepcopy(context["registration_entry"])}
    changed["registration_entry"]["performance_admission"] = True
    with pytest.raises(ValueError, match="context entry changed"):
        m.analyze(changed, binding, sides, reader, store, attempt_cost_binding=store.write_receipt(cost))

@pytest.mark.parametrize("fault", ["none", "drop_original", "alter_original", "drop_embedded", "old_as_full", "adapter", "promotion", "duplicate"])
def test_expanded_effective_manifest_preserves_actual_original121_independently(fault):
    reader = MemoryEvidence(); raw = (EXP / "source_v2/source_manifest.json").read_bytes()
    old = reader.put("original_source2_manifest.json", raw)
    rows = copy.deepcopy(json.loads(raw)["files"])
    rows += [{"path": "original_source2_manifest.json", "sha256": old["sha256"], "size_bytes": old["size_bytes"]}]
    for path in ("analyze_chunked_confirmation_v1.py", "verify_registration_promotion_v1.py", "confirmatory_analysis.py", "collect_confirmation_round_v1.py"):
        source = (EXP / path).read_bytes(); relative = "experiments/rloo_sft_anchor_20261013/" + path
        rows.append({"path": relative, "sha256": hashlib.sha256(source).hexdigest(), "size_bytes": len(source)})
    if fault == "drop_original": rows.pop(0)
    elif fault == "alter_original": rows[0]["sha256"] = "0" * 64
    elif fault == "drop_embedded": rows = [row for row in rows if row["path"] != "original_source2_manifest.json"]
    elif fault == "old_as_full": rows = json.loads(raw)["files"]
    elif fault in ("adapter", "promotion"):
        suffix = "analyze_chunked_confirmation_v1.py" if fault == "adapter" else "verify_registration_promotion_v1.py"
        next(row for row in rows if row["path"].endswith(suffix))["sha256"] = "0" * 64
    elif fault == "duplicate": rows.append(copy.deepcopy(rows[0]))
    effective = reader.put("effective.json", {"files": rows, "synthetic_fixture": True})
    args = ({"source_manifest_binding": effective}, {"original_source_manifest_binding": old,
        "adapter_source_sha256": m.__executed_source_sha256__}, reader)
    if fault == "none": assert len(m.validate_effective_source_layer(*args)) == 126
    else:
        with pytest.raises(ValueError): m.validate_effective_source_layer(*args)

@pytest.mark.parametrize("fault", ["drop_reservation", "rewrite_failed", "scalar_reset", "unresolved", "skip_round"])
def test_immutable_historical_reservations_failures_and_round_chain(shared, fault):
    f, _ = shared; context, _, _, _, _, _, _, _ = f
    context = {**context, "registration_entry": copy.deepcopy(context["registration_entry"])}
    entry = context["registration_entry"]; packet = context["packet_reader"]
    authority = json.loads(packet(entry["resource_authority_binding"]))
    history = json.loads(packet(entry["historical_resource_ledger_binding"]))
    if fault == "drop_reservation": history["original_pilot_reservations"].pop()
    elif fault == "rewrite_failed": history["prior_failed_phase_binding"]["sha256"] = "0" * 64
    elif fault == "scalar_reset": authority["historical_formal_requested_gpu_hours"] = 1
    elif fault == "unresolved": history["unresolved_intents"] = ["ambiguous first attempt"]
    else: context["contract"] = {**context["contract"], "round_number": 2}
    seal(history, "receipt_sha256"); history_binding = packet.put("history/tampered-" + fault + ".json", history)
    entry["historical_resource_ledger_binding"] = authority["historical_resource_ledger_binding"] = history_binding
    with pytest.raises((ValueError, AssertionError)): m.historical_resource_custody(context, authority)

@pytest.mark.parametrize("fault", ["trained_sft", "world", "tree", "raw", "optimizer"])
def test_once_sft_full_physical_identity_zero_training_and_tree_guards(shared, fault):
    f, _ = shared; context, _, sides, reader, store, _, cost, _ = f
    side = copy.deepcopy(sides[0]); artifacts = side["artifacts"]
    if fault == "optimizer": side["native_terminal_artifact_proof"]["files"][0]["path"] = "optimizer.pt"
    elif fault in ("trained_sft", "world"):
        name = "training_complete.json"; data = json.loads(reader.small(artifacts[name]))
        if fault == "trained_sft": data["training"]["optimizer_steps"] = 100
        else: data["distributed"]["world_size"] = 4
        path = reader.native_identity(artifacts[name])["native_path"]
        artifacts[name] = store.ingest(ByteProvider({path: encoded(data)}), path)
    elif fault == "tree": side["native_terminal_artifact_proof"]["tree_sha256"] = "0" * 64
    else:
        name = "evaluation_groups.jsonl"; raw = reader.small(artifacts[name]); data = [json.loads(line) for line in raw.splitlines()]
        data[0]["raw_completions"][0]["text"] += " forged"
        path = reader.native_identity(artifacts[name])["native_path"]
        artifacts[name] = store.ingest(ByteProvider({path: b"".join(encoded(row) + b"\n" for row in data)}), path)
    closed = m.resource_and_attempt_custody(context, store.write_receipt(cost), reader)
    with pytest.raises(ValueError): m.verify_sft_sidecar(context, side, reader, closed)

def test_closed_timeout_accounts_actual_grace_without_erasing_failed_draw(shared):
    f, _ = shared; context, _, _, reader, store, _, original, _ = f
    cost = copy.deepcopy(original); row = cost["slots"][0]
    allocation = json.loads(reader.small(row["allocation_binding"])); job = allocation["argv"][4]
    allocation["stdout"] = "\n".join(f"{job}{suffix}|TIMEOUT|75|0:15|gres/gpu=4" for suffix in ("", ".batch", ".extern"))
    path = reader.native_identity(row["allocation_binding"])["native_path"]
    row["allocation_binding"] = store.ingest(ByteProvider({path: encoded(allocation)}), path)
    grace = {"protocol": m.PROTOCOL + "-independent-scheduler-termination-grace", "synthetic_fixture": True,
        "job_id": job, "allocation_binding": row["allocation_binding"], "independent_read_only_collection": True, "maximum_overrun_seconds": 20}
    path = "/synthetic/independent-grace/" + job + ".json"
    row["scheduler_termination_grace_binding"] = store.ingest(ByteProvider({path: encoded(grace)}), path)
    seal(cost, "receipt_sha256"); result = m.resource_and_attempt_custody(context, store.write_receipt(cost), reader)
    key = row["draw_id"], row["task"], row["arm"]
    assert result["terminal"][key]["state"] == "TIMEOUT"
    assert result["allocated_gpu_seconds"] == 4 * (75 + 79 * 10) + 2 * 10
    assert result["reserved_gpu_hours"] == 92 + (80 * 4 + 2) / 60
    del row["scheduler_termination_grace_binding"]; seal(cost, "receipt_sha256")
    with pytest.raises(KeyError): m.resource_and_attempt_custody(context, store.write_receipt(cost), reader)

