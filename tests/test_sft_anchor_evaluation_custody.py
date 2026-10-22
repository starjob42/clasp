"""Synthetic native custody guards and actual saved tiny-model integration.

All scheduler/authority/extra-attempt rows here are fabricated. The saved
four-CPU-rank PEFT artifact is real integration evidence, never model-study
performance or native recovery admission.
"""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"
spec = importlib.util.spec_from_file_location("eval_custody", EXP / "verify_evaluation_rescue_v1.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def binding(raw, path=None):
    result = {"sha256": m.sha(raw), "size_bytes": len(raw), "content_base64": base64.b64encode(raw).decode()}
    if path is not None: result["native_path"] = path
    return result


def sealed(value):
    value["semantic_sha256"] = m.digest({k: v for k, v in value.items() if k != "semantic_sha256"})
    return value


def accounting(job, state="COMPLETED", elapsed=60, gpus=4):
    rows = []
    for suffix in ("", ".batch", ".extern"):
        status = "COMPLETED" if state in ("FAILED", "TIMEOUT") and suffix == ".extern" else state
        code = "1:0" if status in ("FAILED", "TIMEOUT") else "0:0"
        rows.append(f"{job}{suffix}|{status}|{elapsed}|{code}|cpu=80,mem=300000M,gres/gpu={gpus}\n")
    return {"argv": ["sacct", "--noheader", "--parsable2", "--jobs", job,
                      "--format=JobIDRaw,State%40,ElapsedRaw,ExitCode,AllocTRES"],
            "returncode": 0, "stderr": "", "stdout_binding": binding("".join(rows).encode())}


def change_stdout(row, fn):
    raw = base64.b64decode(row["accounting"]["stdout_binding"]["content_base64"]).decode()
    row["accounting"]["stdout_binding"] = binding(fn(raw).encode())


def cost_fixture():
    _, _, a = m.kernels()
    raw = (EXP / "collected_status_v2_20261014T0911/status/campaign_authority.json").read_bytes()
    original = binding(raw, a.NAMESPACE + "/control_v2/campaign_authority.json")
    slot, parent, rescue = "op.rloo_cardinal_beta005", "900", "901"
    progress = b'{"wall_seconds":3.5}\n{"wall_seconds":7.5}\n'
    authority = sealed({"schema_version": 1, "protocol": m.OPERATOR_PROTOCOL + "-cumulative-authority",
        "original_authority": original, "original_reservations": a.reservations(),
        "rescue_reservations": [{"reservation_id": "evalrescue-v1/" + slot, "slot": slot, "gpus": 1,
          "requested_wall_seconds": 3600, "attempt_limit": 1, "purpose": "terminal_evaluation_only",
          "measured_runtime": {"progress_binding": binding(progress), "complete_groups": 2,
            "measured_group_seconds": 11, "planned_wall_seconds": 3600}}],
        "cumulative_maximum_requested_gpu_hours": 93, "no_training": True, "no_checkpoint_selection": True})
    revised = binding(m.encoded(authority))
    original_attempt = {"schema_version": 2, "study": "rloo-sft-anchor-20261013-phase2", "slot": slot, "job_id": parent,
                        "state": "submitted", "manifest_sha256": a.LAUNCH_SHA, "requested_gpu_hours": 8}
    script_raw = (EXP / "control_v2/job_scripts" / (slot + ".slurm")).read_bytes()
    script_path = a.NAMESPACE + "/control_v2/job_scripts/" + slot + ".slurm"
    original_attempt.update(script_sha256=m.sha(script_raw), argv=["sbatch", "--parsable", script_path],
                            automatic_retry=False, automatic_cancellation=False, returncode=0, stderr="", stdout=parent + "\n")
    script_binding = binding(script_raw, script_path)
    rescue_attempt = {"protocol": m.PROTOCOL + "-attempt", "reservation_id": "evalrescue-v1/" + slot,
        "slot": slot, "job_id": rescue, "original_parent_job_id": parent, "attempt_index": 1, "state": "submitted",
        "purpose": "terminal_evaluation_only", "gpus": 1, "requested_wall_seconds": 3600, "terminal_step": 100,
        "training_or_resume": False, "source2_sha256": m.SOURCE2_SHA, "helper3_sha256": m.HELPER3_SHA,
        "operator_sha256": "c" * 64, "authority_sha256": revised["sha256"],
        "training_complete_sha256": "a" * 64, "contract_sha256": "b" * 64}
    original_root, rescue_root = a.NAMESPACE + "/control_v2/attempts", a.NAMESPACE + "/evaluation_rescue_v1/control/attempts"
    original_binding = binding(m.encoded(original_attempt), original_root + "/" + slot + ".json")
    rescue_binding = binding(m.encoded(rescue_attempt), rescue_root + "/" + slot + ".json")
    listing = {"protocol": m.PROTOCOL + "-scoped-attempt-listing", "stable_snapshot": True, "read_only": True,
        "original_root": original_root, "rescue_root": rescue_root,
        "original_entries": [{"path": slot + ".json", "binding": original_binding, "script_binding": script_binding}],
        "rescue_entries": [{"path": slot + ".json", "binding": rescue_binding}]}
    inventory = {"protocol": m.INVENTORY_PROTOCOL, "closed": True,
                 "attempts": [{"kind": "original_source2", "reservation_id": "phase2/" + slot, "attempt_index": 1,
                               "attempt_binding": original_binding, "script_binding": script_binding},
                              {"kind": "evaluation_rescue", "reservation_id": "evalrescue-v1/" + slot, "attempt_index": 1,
                               "attempt_binding": rescue_binding}]}
    cert = sealed({"schema_version": 1, "protocol": m.ACCOUNTING_PROTOCOL, "synthetic_fixture": True,
        "original_authority": original, "revised_authority": revised, "original_reservations": a.reservations(),
        "cumulative_maximum_requested_gpu_hours": 93, "attempt_inventory": binding(m.encoded(inventory)),
        "scoped_native_attempt_listing": binding(m.encoded(listing)),
        "jobs": [{"job_id": "1030929", "reservation_id": "prior-v1/1030929", "accounting": accounting("1030929", "FAILED", 459)},
                 {"job_id": parent, "reservation_id": "phase2/" + slot, "accounting": accounting(parent, "TIMEOUT", 7200)},
                 {"job_id": rescue, "reservation_id": "evalrescue-v1/" + slot, "accounting": accounting(rescue, "COMPLETED", 600, 1)}]})
    args = {"expected_authority_sha256": revised["sha256"], "expected_cumulative_gpu_hours": 93,
            "slot": slot, "training_sha256": "a" * 64, "contract_sha256": "b" * 64,
            "operator_sha256": "c" * 64, "rescue_job_id": rescue, "synthetic_fixture": True,
            "expected_listing_sha256": m.sha(m.encoded(listing))}
    return cert, args


def validate_cost(cert, args):
    sealed(cert)
    return m.validate_cost_certificate(cert, expected_sha256=m.digest(cert), **args)


def update_inventory(cert, fn, args):
    inventory = json.loads(base64.b64decode(cert["attempt_inventory"]["content_base64"]))
    attempts = [json.loads(base64.b64decode(row["attempt_binding"]["content_base64"])) for row in inventory["attempts"]]
    fn(attempts)
    listing = json.loads(base64.b64decode(cert["scoped_native_attempt_listing"]["content_base64"]))
    scripts = {row["path"]: row["script_binding"] for row in listing["original_entries"]}
    listing["original_entries"] = []; listing["rescue_entries"] = []; inventory["attempts"] = []
    for attempt in attempts:
        original = attempt.get("schema_version") == 2
        kind, key, root = (("original_source2", "original_entries", listing["original_root"]) if original else
                           ("evaluation_rescue", "rescue_entries", listing["rescue_root"]))
        raw_binding = binding(m.encoded(attempt), root + "/" + attempt["slot"] + ".json")
        entry = {"path": attempt["slot"] + ".json", "binding": raw_binding}
        wrapper = {"kind": kind, "reservation_id": ("phase2/" if original else "evalrescue-v1/") + attempt["slot"],
                   "attempt_index": 1, "attempt_binding": raw_binding}
        if original: entry["script_binding"] = wrapper["script_binding"] = scripts[entry["path"]]
        listing[key].append(entry); inventory["attempts"].append(wrapper)
    cert["attempt_inventory"] = binding(m.encoded(inventory))
    cert["scoped_native_attempt_listing"] = binding(m.encoded(listing))
    args["expected_listing_sha256"] = cert["scoped_native_attempt_listing"]["sha256"]


def test_cumulative_cost_retains_all_original_reservations_and_actual_failed_cost():
    cert, args = cost_fixture(); result = validate_cost(cert, args)
    assert result["total_requested_gpu_hours"] == 93 and result["retained_original_requested_gpu_hours"] == 92
    assert result["total_allocated_gpu_seconds"] == 4 * (459 + 7200) + 600
    assert result["allocation_accounting_complete"] and result["original_parent_failure_preserved"]
    assert not result["production_admitted"] and not result["new_job_authorized"]


@pytest.mark.parametrize("mutate", [
    lambda c: c["original_reservations"].pop(),
    lambda c: c.update(cumulative_maximum_requested_gpu_hours=92),
    lambda c: c["jobs"].pop(0),
    lambda c: c["jobs"].pop(1),
    lambda c: c["jobs"].append(copy.deepcopy(c["jobs"][-1])),
    lambda c: c["jobs"][-1].update(reservation_id="phase2/op.sft"),
    lambda c: c["jobs"][-1]["accounting"].update(stderr="unresolved"),
    lambda c: c["jobs"][-1]["accounting"].update(returncode=1),
    lambda c: change_stdout(c["jobs"][-1], lambda s: s.replace("gres/gpu=1", "gres/gpu=4")),
    lambda c: change_stdout(c["jobs"][-1], lambda s: s.replace("|600|", "|3601|")),
    lambda c: change_stdout(c["jobs"][-1], lambda s: s.replace("COMPLETED", "RUNNING")),
    lambda c: change_stdout(c["jobs"][-1], lambda s: s.replace("COMPLETED", "FAILED").replace("0:0", "1:0")),
    lambda c: change_stdout(c["jobs"][1], lambda s: s.replace("TIMEOUT", "COMPLETED").replace("1:0", "0:0")),
    lambda c: change_stdout(c["jobs"][1], lambda s: s.replace("900.batch|TIMEOUT|7200|1:0", "900.batch|RUNNING|7200|0:0")),
    lambda c: change_stdout(c["jobs"][0], lambda s: s.replace("|459|", "|458|")),
    lambda c: change_stdout(c["jobs"][-1], lambda s: s.splitlines(True)[0]),
    lambda c: change_stdout(c["jobs"][-1], lambda s: s.replace("901.batch", "foreign.batch")),
    lambda c: c["revised_authority"].update(sha256="0" * 64),
])
def test_raw_native_lineage_and_cumulative_cost_corruptions_rejected(mutate):
    cert, args = cost_fixture(); mutate(cert)
    with pytest.raises(ValueError): validate_cost(cert, args)


@pytest.mark.parametrize("key,value", [("attempt_index", 2), ("job_id", "900"), ("gpus", 4), ("terminal_step", 99),
    ("training_or_resume", True), ("purpose", "training"), ("requested_wall_seconds", 7200),
    ("source2_sha256", "0" * 64), ("helper3_sha256", "0" * 64), ("operator_sha256", "0" * 64),
    ("authority_sha256", "0" * 64), ("training_complete_sha256", "0" * 64),
    ("contract_sha256", "0" * 64), ("original_parent_job_id", "902")])
def test_extra_attempt_cannot_select_policy_change_recipe_or_resubmit(key, value):
    cert, args = cost_fixture(); update_inventory(cert, lambda rows: rows[-1].update({key: value}), args)
    with pytest.raises(ValueError): validate_cost(cert, args)


def test_failed_or_ambiguous_attempt_is_retained_and_prevents_second_attempt():
    cert, args = cost_fixture()
    def mutate(rows):
        old = copy.deepcopy(rows[-1]); old.update(state="ambiguous_submission", job_id=None)
        rows.insert(1, old)
    update_inventory(cert, mutate, args)
    with pytest.raises(ValueError): validate_cost(cert, args)


def test_no_production_admission_or_claim_from_fixture():
    cert, args = cost_fixture(); args["synthetic_fixture"] = False
    with pytest.raises(ValueError, match="unregistered"): validate_cost(cert, args)
    with pytest.raises(ValueError, match="production admission disabled"): m.admit_production(registration_sha256="a" * 64)


def actual_saved_fixture(tmp_path, *, partial=False):
    from transformers import AutoTokenizer
    import sys
    sys.path.insert(0, str(EXP))
    import prepare_evaluation_rescue_v1 as op
    original = EXP / "validation/v2_peft_cpu_ddp4_v1/op"
    training = tmp_path / "saved_original"
    shutil.copytree(original / "rloo_cardinal_beta005", training)
    contract = json.loads((original / "contract.json").read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(original / "fixture_base", local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"; tokenizer.model_input_names = ["input_ids", "attention_mask"]
    _, _, a = m.kernels()
    authority = binding((EXP / "collected_status_v2_20261014T0911/status/campaign_authority.json").read_bytes(),
                        a.NAMESPACE + "/control_v2/campaign_authority.json")
    slot, parent = "op.rloo_cardinal_beta005", "999"
    attempt = binding(m.encoded({"state": "submitted", "slot": slot, "job_id": parent,
                      "manifest_sha256": a.LAUNCH_SHA, "requested_gpu_hours": 8}),
                      a.NAMESPACE + "/control_v2/attempts/" + slot + ".json")
    raw = (training / "evaluation_groups.jsonl").read_bytes()
    options = {}
    if partial:
        raw = b"".join(raw.splitlines(keepends=True)[:-1])
        (training / "evaluation_groups.jsonl").write_bytes(raw)
        groups = [json.loads(line) for line in raw.splitlines()]
        revised = sealed({"schema_version": 1, "protocol": op.AUTHORITY_PROTOCOL,
            "original_authority": authority, "original_reservations": a.reservations(),
            "rescue_reservations": [{"reservation_id": "evalrescue-v1/" + logical, "slot": logical, "gpus": 1,
                "requested_wall_seconds": 7200, "attempt_limit": 1, "purpose": "terminal_evaluation_only",
                "measured_runtime": ({"progress_binding": binding(raw), "complete_groups": len(groups),
                    "measured_group_seconds": sum(g["wall_seconds"] for g in groups), "planned_wall_seconds": 3600}
                    if logical == slot else None)} for logical in a.ORDER if not logical.endswith('.sft')],
            "cumulative_maximum_requested_gpu_hours": 112, "no_training": True, "no_checkpoint_selection": True})
        revised_path = tmp_path / "authority.json"; revised_path.write_bytes(m.encoded(revised))
        ledger = tmp_path / "ledger.json"; ledger.write_bytes(m.encoded({"schema_version": 1, "protocol": op.LEDGER_PROTOCOL, "attempts": []}))
        options = {"revised_authority": revised_path, "revised_authority_sha256": m.identity(revised_path)["sha256"],
            "cumulative_gpu_hours": 112, "attempt_ledger": ledger, "attempt_ledger_sha256": m.identity(ledger)["sha256"]}
    account = accounting(parent, "TIMEOUT", 7200)
    plan = op.prepare(contract, "rloo_cardinal_beta005", training, tokenizer,
        training_sha256=m.identity(training / "training_complete.json")["sha256"], parent_accounting=account,
        parent_job_id=parent, parent_attempt=attempt, original_authority_binding=authority, **options)
    plan["synthetic_fixture"] = True; sealed(plan)  # Scheduler/custody is simulated.
    return op, plan, contract, training, tokenizer, account, original / "fixture_base"


def admit_saved(plan, contract, training, tokenizer, account):
    return m.validate_preparation(plan, contract, training, tokenizer, account,
        expected_plan_sha256=plan["semantic_sha256"], synthetic_fixture=True)


def test_actual_saved_four_rank_tiny_offline_closure_has_zero_new_model_calls(tmp_path):
    op, plan, contract, training, tokenizer, account, base = actual_saved_fixture(tmp_path)
    context = admit_saved(plan, contract, training, tokenizer, account)
    output = tmp_path / "offline"
    result, operator = op.execute_smoke(plan, contract, training, output, tokenizer, base)
    verified = m.validate_completed_diagnostic(context, plan, output, tokenizer)
    assert verified["terminal_step"] == 4 and verified["training_responses"] == 144
    assert verified["raw_responses_replayed"] == 36 and verified["new_responses"] == 0
    assert result["resources"]["model_calls_in_this_operator"] == 0 and not verified["production_admitted"]


def test_actual_tiny_missing_only_model_decode_preserves_original_group_bytes(tmp_path):
    import torch
    torch.set_num_threads(1)
    op, plan, contract, training, tokenizer, account, base = actual_saved_fixture(tmp_path, partial=True)
    context = admit_saved(plan, contract, training, tokenizer, account)
    output = tmp_path / "rescue"
    op.execute_smoke(plan, contract, training, output, tokenizer, base)
    verified = m.validate_completed_diagnostic(context, plan, output, tokenizer)
    assert verified["raw_responses_replayed"] == 36 and verified["new_responses"] == 8 and verified["reused_responses"] == 28
    assert (output / "evaluation_groups.jsonl").read_bytes().startswith(context["prefix_bytes"])


@pytest.mark.parametrize("change", ["rank_actor", "rank_gc", "reference", "windows", "terminal", "reload", "prefix", "tokenizer"])
def test_original_training_actor_prefix_and_reload_cannot_be_reconstructed(tmp_path, change):
    _, plan, contract, training, tokenizer, account, _ = actual_saved_fixture(tmp_path)
    if change in ("rank_actor", "rank_gc", "reference"):
        path = training / "training_complete.json"; proof = json.loads(path.read_bytes())
        if change == "rank_actor": proof["distributed"]["rank_closure"][2]["actor_state_sha256"] = "0" * 64
        if change == "rank_gc": proof["distributed"]["rank_closure"][2]["effective_checkpointing"]["effective_use_reentrant"] = True
        if change == "reference": proof["reference"]["initial_equivalence"]["bitwise_equal"] = False
        path.write_bytes(m.encoded(proof)); plan["training_proof"] = m.identity(path); sealed(plan)
    elif change == "windows":
        with (training / "training/actual_prompt_windows.jsonl").open("ab") as f: f.write(b"{}\n")
    elif change == "terminal":
        with (training / "terminal/adapter_model.safetensors").open("ab") as f: f.write(b"changed")
    elif change == "reload":
        plan["original_reload_proof"] = None; plan["new_exact_reload_required"] = True; sealed(plan)
    elif change == "prefix":
        path = training / "evaluation_groups.jsonl"; rows = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(rows[1:] + rows[:1])); plan["prior_progress"] = m.identity(path); sealed(plan)
    elif change == "tokenizer": tokenizer.padding_side = "right"
    with pytest.raises(ValueError): admit_saved(plan, contract, training, tokenizer, account)


def test_result_numerical_cell_cannot_bypass_independent_full_raw_replay(tmp_path):
    op, plan, contract, training, tokenizer, account, base = actual_saved_fixture(tmp_path)
    context = admit_saved(plan, contract, training, tokenizer, account)
    output = tmp_path / "offline"; op.execute_smoke(plan, contract, training, output, tokenizer, base)
    result_path = output / "result.json"; result = json.loads(result_path.read_bytes())
    result["raw_completions"][0]["score"]["u10"] = 17
    result_path.write_bytes(m.encoded(result))
    operator_path = output / "evaluation_operator_complete.json"; operator = json.loads(operator_path.read_bytes())
    operator["result"] = m.identity(result_path); operator_path.write_bytes(m.encoded(operator))
    with pytest.raises(ValueError, match="scorer replay"): m.validate_completed_diagnostic(context, plan, output, tokenizer)


@pytest.mark.parametrize("change", ["borrow", "drop_original", "target", "measurement", "second_attempt", "wall", "training"])
def test_preparation_budget_and_one_attempt_guards_are_replayed_from_bytes(tmp_path, change):
    _, plan, contract, training, tokenizer, account, _ = actual_saved_fixture(tmp_path, partial=True)
    if change == "second_attempt":
        path = Path(plan["first_attempt_ledger"]["path"])
        value = json.loads(path.read_bytes())
        value["attempts"] = [{"slot": plan["slot"], "kind": "one_gpu_evaluation_only", "job_id": "1000"}]
        path.write_bytes(m.encoded(value)); plan["first_attempt_ledger"] = m.identity(path)
    else:
        path = Path(plan["cumulative_authority"]["binding"]["path"]); value = json.loads(path.read_bytes())
        if change == "borrow": value["cumulative_maximum_requested_gpu_hours"] = 92
        if change == "drop_original": value["original_reservations"].pop()
        if change == "target": value["rescue_reservations"][0]["slot"] = "op.foarl"
        if change == "measurement": value["rescue_reservations"][0]["measured_runtime"]["measured_group_seconds"] += 1
        if change == "wall": value["rescue_reservations"][0]["requested_wall_seconds"] = 7201
        if change == "training": value["no_training"] = False
        sealed(value); path.write_bytes(m.encoded(value)); plan["cumulative_authority"]["binding"] = m.identity(path)
    sealed(plan)
    with pytest.raises(ValueError): admit_saved(plan, contract, training, tokenizer, account)


def test_complete_json_without_newline_is_a_discarded_partial_tail(tmp_path):
    _, plan, contract, training, tokenizer, account, _ = actual_saved_fixture(tmp_path, partial=True)
    path = training / "evaluation_groups.jsonl"; raw = path.read_bytes()
    # Even parseable JSON has no durable whole-group seal until its newline.
    path.write_bytes(raw + b'{"schema_version":2}')
    kernel, _, _ = m.kernels()
    groups, accepted, tail, _ = m.prefix(path, contract, tokenizer, kernel)
    assert accepted == raw and len(groups) == 7 and tail == {"size_bytes": 20, "sha256": m.sha(b'{"schema_version":2}')}


def test_full121_original_producer_closure_is_rehashed():
    value = m.verify_source2_inventory(EXP / "source_v2")
    assert value["source_files_verified"] == 121 and not value["native_queries_or_mutations"]


def test_scoped_listing_prevents_hidden_charged_original_attempt():
    cert, args = cost_fixture()
    inventory = json.loads(base64.b64decode(cert["attempt_inventory"]["content_base64"]))
    inventory["attempts"].pop(0)
    cert["attempt_inventory"] = binding(m.encoded(inventory))
    with pytest.raises(ValueError, match="omitted or rewrote"): validate_cost(cert, args)


@pytest.mark.parametrize("key,value", [("script_sha256", "0" * 64), ("automatic_retry", True),
    ("automatic_cancellation", True), ("argv", ["sbatch", "--requeue", "foreign"]),
    ("stdout", "902\n"), ("returncode", 1), ("stderr", "unresolved")])
def test_original_native_attempt_is_not_rewritten_or_substituted(key, value):
    cert, args = cost_fixture(); update_inventory(cert, lambda rows: rows[0].update({key: value}), args)
    with pytest.raises(ValueError): validate_cost(cert, args)


def test_dependency_execution_uses_the_verified_buffer_if_path_changes(tmp_path, monkeypatch):
    path = tmp_path / "source.py"; path.write_bytes(b"value=5\n")
    expected = m.sha(path.read_bytes()); stable = m.stable_bytes
    def moved(path):
        raw = stable(path); Path(path).write_bytes(b"value=7\n"); return raw
    monkeypatch.setattr(m, "stable_bytes", moved)
    module = m.load_module(path, expected, "_bound_fixture")
    assert module.value == 5 and path.read_bytes() == b"value=7\n"


def test_prefix_hash_and_groups_are_the_same_stable_snapshot(tmp_path, monkeypatch):
    _, plan, contract, training, tokenizer, _, _ = actual_saved_fixture(tmp_path)
    path = training / "evaluation_groups.jsonl"; original = path.read_bytes()
    kernel, _, _ = m.kernels(); stable = m.stable_bytes
    def moved(target):
        raw = stable(target)
        if Path(target) == path: path.write_bytes(b"changed after read")
        return raw
    monkeypatch.setattr(m, "stable_bytes", moved)
    groups, prefix, _, observed = m.prefix(path, contract, tokenizer, kernel)
    assert len(groups) == 8 and prefix == original and observed["sha256"] == m.sha(original)
    assert observed["size_bytes"] == len(original)


def test_actual_native_amendment_registration_preserves_original_chronology_without_extra_authority():
    value = m.verify_amendment_registration()
    assert value["native_scope_registered_before_validation"] and not value["receipt_rewritten"]
    assert value["original_registration_utc"] == "2026-10-14T12:16:05.953028+00:00"
    assert not value["operator_or_extra_authority_registered"] and not value["new_job_authorized"]


def new_operator_offline(tmp_path):
    import sys
    sys.path.insert(0, str(EXP))
    import execute_evaluation_rescue_v1 as executor
    _, plan, contract, training, tokenizer, account, base = actual_saved_fixture(tmp_path)
    context = admit_saved(plan, contract, training, tokenizer, account)
    plan_path = tmp_path / "plan.json"; plan_path.write_bytes(m.encoded(plan))
    contract_path = tmp_path / "contract.json"; contract_path.write_bytes(m.encoded(contract))
    output = tmp_path / "registered_offline"
    entry = {"protocol": m.OPERATOR_PROTOCOL, "slot": plan["slot"], "scope": "system_smoke",
        "synthetic_offline_fixture": True, "sources": executor.sources(), "amendment_sha256": m.AMENDMENT_SHA,
        "preparation_semantic_sha256": plan["semantic_sha256"], "whole_actor_sha256": plan["whole_actor_sha256"],
        "contract_sha256": plan["contract_sha256"], "training_or_resume": False, "checkpoint_selection": False,
        "output_directory": str(output.resolve())}
    registry_path = tmp_path / "registry.json"
    registry_path.write_bytes(m.encoded({"protocol": m.OPERATOR_PROTOCOL + "-source-registry", "schema_version": 1, "entries": [entry]}))
    executor.execute(plan_path, plan_sha256=m.identity(plan_path)["sha256"], registry_path=registry_path,
        registry_sha256=m.identity(registry_path)["sha256"], amendment_path=EXP / "planning/evaluation_rescue_amendment_v1/amendment.json",
        contract_path=contract_path, contract_sha256=m.identity(contract_path)["sha256"], training_output=training,
        base=base, output=output, device="cpu")
    return context, plan, output, tokenizer


def test_distinct_new_operator_actual_saved_offline_zero_model_call_custody(tmp_path):
    context, plan, output, tokenizer = new_operator_offline(tmp_path)
    value = m.validate_new_operator_diagnostic(context, plan, output, tokenizer)
    assert value["raw_responses_replayed"] == 36 and value["closed_groups"] == 8 and value["new_model_calls"] == 0
    assert not value["production_admitted"] and value["exact_original_prefix_bytes_preserved"]


@pytest.mark.parametrize("change", ["calls", "checkpoint", "score", "group", "source"])
def test_new_operator_completion_cannot_spoof_work_actor_numerics_or_source(tmp_path, change):
    context, plan, output, tokenizer = new_operator_offline(tmp_path)
    path = output / "operator_complete.json"; value = json.loads(path.read_bytes())
    if change == "calls": value["actual_new_model_calls"] = 1
    if change == "checkpoint": value["checkpoint_selection"] = True
    if change == "source": value["sources"]["executor"]["sha256"] = "0" * 64
    if change == "score":
        result_path = output / "operator_result.json"; result = json.loads(result_path.read_bytes())
        result["raw_completions"][0]["score"]["u10"] = 17
        result_path.write_bytes(m.encoded(result)); value["result"] = m.identity(result_path)
    if change == "group":
        groups_path = output / "operator_evaluation_groups.jsonl"; rows = groups_path.read_bytes().splitlines(keepends=True)
        groups_path.write_bytes(b"".join(rows[1:] + rows[:1])); value["groups"] = m.identity(groups_path)
    sealed(value); path.write_bytes(m.encoded(value))
    with pytest.raises(ValueError): m.validate_new_operator_diagnostic(context, plan, output, tokenizer)
