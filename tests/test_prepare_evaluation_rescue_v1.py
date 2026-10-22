"""Actual saved tiny PEFT fixtures and pure budget/custody guards; no jobs."""
from __future__ import annotations
import base64
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"
spec = importlib.util.spec_from_file_location("evaluation_rescue_preparer", EXP / "prepare_evaluation_rescue_v1.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def encoded(value): return (json.dumps(value, sort_keys=True) + "\n").encode()
def binding(raw, path=None):
    item = {"sha256": m.old_accounting.sha(raw), "size_bytes": len(raw), "content_base64": base64.b64encode(raw).decode()}
    if path is not None: item["native_path"] = path
    return item


def accounting(job="999", state="TIMEOUT"):
    rows = []
    for suffix in ("", ".batch", ".extern"):
        current = "COMPLETED" if suffix == ".extern" else state
        exitcode = "0:0" if current in ("RUNNING", "COMPLETED") else "1:0"
        rows.append(f"{job}{suffix}|{current}|7200|{exitcode}|cpu=80,mem=300000M,node=1,billing=5,gres/gpu=4\n")
    return {"argv": ["sacct", "--noheader", "--parsable2", "--jobs", job, "--format=JobIDRaw,State%40,ElapsedRaw,ExitCode,AllocTRES"],
        "returncode": 0, "stderr": "", "stdout_binding": binding("".join(rows).encode())}


def fixture(tmp_path, task="op", arm="rloo_cardinal_beta005", prefix=None, tail=False, missing_reload=False):
    torch.set_num_threads(1)
    source = EXP / f"validation/v2_peft_cpu_all_arms_v1/{task}"
    original = source / arm; training = tmp_path / "original"
    shutil.copytree(original, training)
    contract = json.loads((source / "contract.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(source / "fixture_sft", local_files_only=True)
    tokenizer.model_input_names = ["input_ids", "attention_mask"]; tokenizer.padding_side = "left"
    progress = training / "evaluation_groups.jsonl"
    groups = [json.loads(line) for line in progress.read_text().splitlines()]
    if prefix is not None: progress.write_bytes(b"".join(encoded(group) for group in groups[:prefix]))
    if tail:
        with progress.open("ab") as stream: stream.write(b'{"interrupted":')
    if missing_reload: (training / "checkpoint_reload_exact.json").unlink()
    authority_raw = (EXP / "collected_status_v2_20261014T0911/status/campaign_authority.json").read_bytes()
    original_authority = binding(authority_raw, m.old_accounting.NAMESPACE + "/control_v2/campaign_authority.json")
    slot = task + "." + arm
    closed, _, _ = m.closed_prefix(progress, contract, tokenizer)
    authority = {"schema_version": 1, "protocol": m.AUTHORITY_PROTOCOL, "synthetic_offline_fixture": True,
        "original_authority": original_authority, "original_reservations": m.old_accounting.reservations(),
        "rescue_reservations": [{"reservation_id": "evalrescue-v1/" + slot, "slot": slot, "gpus": 1,
            "requested_wall_seconds": 7200, "attempt_limit": 1, "purpose": "terminal_evaluation_only",
            "measured_runtime": {"progress_binding": binding(progress.read_bytes()), "complete_groups": len(closed),
                "measured_group_seconds": sum(row["wall_seconds"] for row in closed), "planned_wall_seconds": 3600}}],
        "cumulative_maximum_requested_gpu_hours": 112, "no_training": True, "no_checkpoint_selection": True}
    target_reservation = authority["rescue_reservations"][0]
    authority["rescue_reservations"] = [target_reservation if logical == slot else
        {"reservation_id": "evalrescue-v1/" + logical, "slot": logical, "gpus": 1, "requested_wall_seconds": 7200,
         "attempt_limit": 1, "purpose": "terminal_evaluation_only", "measured_runtime": None}
        for logical in m.old_accounting.ORDER if not logical.endswith(".sft")]
    authority["semantic_sha256"] = m.digest(authority)
    authority_path = tmp_path / "authority.json"; authority_path.write_bytes(encoded(authority))
    ledger_path = tmp_path / "ledger.json"; ledger_path.write_bytes(encoded({"schema_version": 1, "protocol": m.LEDGER_PROTOCOL, "attempts": []}))
    attempt = {"state": "submitted", "study": "rloo-sft-anchor-20261013-phase2", "slot": slot, "job_id": "999",
        "manifest_sha256": m.old_accounting.LAUNCH_SHA, "requested_gpu_hours": 8}
    args = dict(contract=contract, arm=arm, training_output=training, tokenizer=tokenizer,
        training_sha256=m.identity(training / "training_complete.json")["sha256"], parent_accounting=accounting(), parent_job_id="999",
        parent_attempt=binding(encoded(attempt), m.old_accounting.NAMESPACE + "/control_v2/attempts/" + slot + ".json"),
        original_authority_binding=original_authority, revised_authority=authority_path,
        revised_authority_sha256=m.identity(authority_path)["sha256"], cumulative_gpu_hours=112,
        attempt_ledger=ledger_path, attempt_ledger_sha256=m.identity(ledger_path)["sha256"])
    return args, source / "fixture_base", groups


def reseal_authority(args, change):
    path = args["revised_authority"]; authority = json.loads(path.read_bytes()); change(authority)
    authority["semantic_sha256"] = m.digest({key: value for key, value in authority.items() if key != "semantic_sha256"})
    path.write_bytes(encoded(authority)); args["revised_authority_sha256"] = m.identity(path)["sha256"]


@pytest.mark.parametrize("task,arm", [("op", "rloo_cardinal_beta005"), ("cvrp", "foarl")])
def test_saved_actual_tiny_peft_missing_only_rescue_matches_original_full_decode(tmp_path, monkeypatch, task, arm):
    args, base, original = fixture(tmp_path, task, arm, prefix=2, tail=True)
    plan = m.prepare(**args)
    assert plan["mode"] == "one_gpu_evaluation_only" and plan["requested_new_gpus"] == 1
    assert plan["reused_responses"] == 9 and plan["new_responses"] == 27 and plan["terminal_step"] == 4
    assert plan["discarded_partial_tail"] is not None and plan["wasted_inflight_seconds"] is None
    assert not plan["production_executable"] and not plan["training_or_resume_permitted"]
    before = m.identity(args["training_output"] / "evaluation_groups.jsonl")
    monkeypatch.setattr(m.helper3, "make_recovery_trainer", lambda *args, **kwargs: pytest.fail("Inference rescue constructed a trainer"))
    result, complete = m.execute_smoke(plan, args["contract"], args["training_output"], tmp_path / "rescue", args["tokenizer"], base)
    actual = [json.loads(line) for line in (tmp_path / "rescue/evaluation_groups.jsonl").read_text().splitlines()]
    assert [row["raw_completions"] for row in actual] == [row["raw_completions"] for row in original]
    assert m.identity(args["training_output"] / "evaluation_groups.jsonl") == before
    assert complete["new_model_responses"] == 27 and complete["reused_model_responses"] == 9
    assert not complete["original_source2_complete_json_fabricated"] and not (tmp_path / "rescue/complete.json").exists()
    assert result["resources"]["training_responses"] == 144


def test_complete_actual_prefix_closes_without_loading_any_model_or_gpu(tmp_path, monkeypatch):
    args, base, original = fixture(tmp_path)
    plan = m.prepare(**args)
    assert plan["mode"] == "offline_zero_gpu_closure" and plan["cumulative_authority"] is None
    monkeypatch.setattr(m.source2, "loaded_model", lambda *args, **kwargs: pytest.fail("Offline closure loaded a model"))
    monkeypatch.setattr(m.helper3, "_generate_group", lambda *args, **kwargs: pytest.fail("Offline closure called generation"))
    result, complete = m.execute_smoke(plan, args["contract"], args["training_output"], tmp_path / "closure", args["tokenizer"], base)
    assert complete["new_model_responses"] == 0 and complete["reused_model_responses"] == 36
    assert result["resources"]["model_calls_in_this_operator"] == 0
    assert result["resources"]["evaluation_wall_seconds_is_lower_bound"]
    assert result["resources"]["unrecorded_original_loop_overhead_seconds"] is None


def test_missing_original_reload_cannot_admit_existing_prefix_actor_provenance(tmp_path):
    args, base, original = fixture(tmp_path, prefix=2, missing_reload=True)
    with pytest.raises(ValueError, match="no admitted actor provenance"):
        m.prepare(**args)
    assert not (args["training_output"] / "checkpoint_reload_exact.json").exists()


def test_empty_prefix_missing_reload_needs_a_separate_measured_runtime_admission(tmp_path):
    args, _, _ = fixture(tmp_path, prefix=0, missing_reload=True)
    assert m.admit_reload(args["training_output"] / "checkpoint_reload_exact.json", {}, args["training_sha256"], required=False) is None
    with pytest.raises(ValueError, match="measured runtime"):
        m.prepare(**args)


def test_tokenizer_vocabulary_and_input_preparation_are_bound_to_original_actor(tmp_path):
    args, _, _ = fixture(tmp_path, prefix=2)
    args["tokenizer"].add_tokens(["forged"])
    with pytest.raises(ValueError, match="Rescue tokenizer"):
        m.prepare(**args)


def test_offline_closure_requires_original_exact_reload(tmp_path):
    args, _, _ = fixture(tmp_path, missing_reload=True)
    with pytest.raises(ValueError, match="actual original exact reload"):
        m.prepare(**args)


@pytest.mark.parametrize("corruption", ["duplicate", "reorder", "tokens", "text", "seed", "id"])
def test_prefix_byte_and_native_seed_token_identity_guards(tmp_path, corruption):
    args, _, _ = fixture(tmp_path, prefix=2)
    path = args["training_output"] / "evaluation_groups.jsonl"; groups = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == "duplicate": groups.append(groups[0])
    elif corruption == "reorder": groups.reverse()
    elif corruption == "tokens": groups[0]["raw_completions"][0]["completion_token_ids"] = [999999]
    elif corruption == "text": groups[0]["raw_completions"][0]["text"] = "forged"
    elif corruption == "seed": groups[0]["seed"] += 1
    else: groups[0]["instance_id"] = "foreign"
    path.write_bytes(b"".join(encoded(row) for row in groups))
    with pytest.raises(ValueError): m.prepare(**args)


@pytest.mark.parametrize("change", ["drop_original_slot", "drop_prior_failure", "cap_reset", "gpu0", "gpu2", "duplicate_rescue_slot", "attempt2", "training", "no_measurement", "fake_measurement", "wrong_measurement_prefix"])
def test_new_cumulative_authority_retains_all_reservations_and_bounds_actual_measurement(tmp_path, change):
    args, _, _ = fixture(tmp_path, prefix=2)
    def mutate(value):
        row = value["rescue_reservations"][0]
        if change == "drop_original_slot": value["original_reservations"].pop()
        elif change == "drop_prior_failure": value["original_reservations"].pop(0)
        elif change == "cap_reset": value["cumulative_maximum_requested_gpu_hours"] = 92
        elif change in ("gpu0", "gpu2"): row["gpus"] = 0 if change == "gpu0" else 2
        elif change == "duplicate_rescue_slot": value["rescue_reservations"].append(copy.deepcopy(row))
        elif change == "attempt2": row["attempt_limit"] = 2
        elif change == "training": value["no_training"] = False
        elif change == "no_measurement": row["measured_runtime"] = {}
        elif change == "fake_measurement": row["measured_runtime"]["measured_group_seconds"] += 100
        else: row["measured_runtime"]["progress_binding"] = binding(encoded({"wall_seconds": 1.}))
    reseal_authority(args, mutate)
    with pytest.raises((ValueError, KeyError)): m.prepare(**args)


def test_failed_extra_attempt_cannot_be_retried_under_another_job_id(tmp_path):
    args, _, _ = fixture(tmp_path, prefix=2)
    ledger = {"schema_version": 1, "protocol": m.LEDGER_PROTOCOL,
        "attempts": [{"slot": "op.rloo_cardinal_beta005", "kind": "one_gpu_evaluation_only", "job_id": "1000", "state": "TIMEOUT"}]}
    args["attempt_ledger"].write_bytes(encoded(ledger)); args["attempt_ledger_sha256"] = m.identity(args["attempt_ledger"])["sha256"]
    with pytest.raises(ValueError, match="consumes"):
        m.prepare(**args)


def test_live_original_parent_cannot_race_duplicate_evaluation(tmp_path):
    args, _, _ = fixture(tmp_path, prefix=2); args["parent_accounting"] = accounting(state="RUNNING")
    with pytest.raises(ValueError, match="terminal and closed"):
        m.prepare(**args)


@pytest.mark.parametrize("part", ["reference", "rank_actor", "gc", "windows", "terminal", "training_sha"])
def test_original_full_terminal_training_proof_is_required(tmp_path, part):
    args, _, _ = fixture(tmp_path, prefix=2)
    proof_path = args["training_output"] / "training_complete.json"; proof = json.loads(proof_path.read_text())
    if part == "reference": proof["distributed"]["rank_closure"][0]["reference_full_integrity_verified"] = False
    elif part == "rank_actor": proof["distributed"]["rank_closure"][0]["actor_state_sha256"] = "0" * 64
    elif part == "gc": proof["distributed"]["rank_closure"][0]["effective_checkpointing"]["effective_use_reentrant"] = True
    elif part == "windows": (args["training_output"] / "training/actual_prompt_windows.jsonl").write_text("forged\n")
    elif part == "terminal": (args["training_output"] / "terminal/adapter_model.safetensors").write_bytes(b"forged")
    else: args["training_sha256"] = "0" * 64
    if part in ("reference", "rank_actor", "gc"):
        proof_path.write_bytes(encoded(proof)); args["training_sha256"] = m.identity(proof_path)["sha256"]
    with pytest.raises(ValueError): m.prepare(**args)


def test_plan_tamper_or_production_execution_cannot_call_model(tmp_path, monkeypatch):
    args, base, _ = fixture(tmp_path, prefix=2); plan = m.prepare(**args)
    plan["production_executable"] = True
    monkeypatch.setattr(m.source2, "loaded_model", lambda *args, **kwargs: pytest.fail("Tampered plan loaded actor"))
    with pytest.raises(ValueError, match="Only byte-bound"):
        m.execute_smoke(plan, args["contract"], args["training_output"], tmp_path / "out", args["tokenizer"], base)
    plan = m.prepare(**args); production = copy.deepcopy(args["contract"]); production["scope"] = "production"
    with pytest.raises(ValueError, match="Only byte-bound"):
        m.execute_smoke(plan, production, args["training_output"], tmp_path / "out", args["tokenizer"], base)
