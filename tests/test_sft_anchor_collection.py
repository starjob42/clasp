"""Offline guards for immutable read-only native campaign collection."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "experiments/rloo_sft_anchor_20261013/collect_campaign.py"
SPEC = importlib.util.spec_from_file_location("sft_anchor_collection", TOOL)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)
FROZEN = ROOT / "experiments/rloo_sft_anchor_20261013/source_v1/src/clasp/baselines/sft_anchor_pilot.py"
FROZEN_RAW = FROZEN.read_bytes()
scorer = types.ModuleType("test_byte_bound_frozen_anchor_scorer")
scorer.__file__ = str(FROZEN)
exec(compile(FROZEN_RAW, str(FROZEN), "exec"), scorer.__dict__)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def record(native, local, raw, purpose="mutable_status_snapshot", tail=False):
    return {"native_path": native, "local_path": local, "purpose": purpose,
            "sha256": collector.sha(raw), "size_bytes": len(raw), "native_size_bytes": len(raw),
            "snapshot_stable": True, "tail_only": tail, "native_mtime_ns": 1,
            "content_base64": base64.b64encode(raw).decode()}


def make_contract(task):
    if task == "op":
        train_instance = [[[0, 0], [1, 0], [2, 0]], [0, 4, 6], 3]
        validation_instance = [[[0, 0], [3, 4], [6, 4]], [0, 7, 5], 8]
        answer = "Route: [0,1,2]"
    else:
        train_instance = [[[0, 0], [1, 0], [0, 1]], [0, 1, 1], 2]
        validation_instance = [[[0, 0], [3, 0], [0, 4]], [0, 1, 1], 2]
        answer = "Routes: [[0,1,2,0]]"
    rows = [{"instance_id": name, "instance_json": instance, "instruction": name,
             "input": json.dumps(instance), "output": answer}
            for name, instance in (("train", train_instance), ("validation", validation_instance))]
    return scorer.make_contract(task, {"id": "fixture_original_sft", "sha256": "a" * 64},
                                rows[:1], rows[1:], scope="production", horizon=100)


def fixture_bundle(completed_count=0, attempts_count=1, campaign_report=False):
    source = str(collector.NAMESPACE / "source_v1")
    control = str(collector.NAMESPACE / "control_v1")
    contract_values = {task: make_contract(task) for task in ("op", "cvrp")}
    contract_raws = {task: encoded(value) for task, value in contract_values.items()}
    files = {collector.SCORER: FROZEN_RAW, collector.ENTRYPOINT: b"fixture frozen entrypoint\n",
             collector.CONTROLLER: b"fixture frozen controller\n"}
    for task, raw in contract_raws.items():
        files[f"contracts/{task}.json"] = raw
    entries = [{"path": path, "sha256": collector.sha(raw), "size_bytes": len(raw)}
               for path, raw in files.items()]
    source_raw = encoded({"files": entries})
    slots = [{"slot": f"{task}.{arm}", "task": task, "arm": arm, "gpus": 1 if arm == "sft" else 4,
              "wall_seconds": 7200, "requested_gpu_hours": 2 if arm == "sft" else 8}
             for task in ("op", "cvrp") for arm in collector.ARMS]
    manifest = {"schema_version": 1, "study": collector.STUDY, "native_source_root": source,
                "native_control_root": control, "native_python": collector.NATIVE_PYTHON,
                "slots": slots, "maximum_requested_gpu_hours": 84,
                "source_manifest": {"path": "source_manifest.json", "sha256": collector.sha(source_raw)},
                "contracts": {task: {"path": f"contracts/{task}.json", "sha256": collector.sha(contract_raws[task]),
                                     "semantic_sha256": value["contract_sha256"]}
                              for task, value in contract_values.items()}}
    manifest["semantic_sha256"] = collector.canonical(manifest)
    manifest_raw = encoded(manifest)
    manifest_sha = collector.sha(manifest_raw)
    records = [record(control + "/launch_manifest.json", "status/launch_manifest.json", manifest_raw,
                      "bound_launch_manifest"),
               record(source + "/source_manifest.json", "source/source_manifest.json", source_raw,
                      "bound_source_manifest"),
               record(source + "/" + collector.SCORER, "source/frozen_scorer.py", FROZEN_RAW,
                      "bound_frozen_scorer")]
    for task, raw in contract_raws.items():
        records.append(record(source + f"/contracts/{task}.json", f"source/contracts/{task}.json", raw,
                              "bound_task_contract"))
    attempts = {}
    for i, slot in enumerate(collector.ORDER[:attempts_count]):
        arm = slot.split(".")[1]
        attempt = {"slot": slot, "study": collector.STUDY, "manifest_sha256": manifest_sha,
                   "requested_gpu_hours": 2 if arm == "sft" else 8, "state": "submitted", "job_id": str(123 + i)}
        attempts[slot] = attempt
        records.append(record(control + "/attempts/" + slot + ".json", "attempts/" + slot + ".json",
                              encoded(attempt), "submission_attempt_snapshot"))
    status = {"state": "waiting_for_exact_job", "slot": collector.ORDER[0], "scheduler_state": "PENDING"}
    records.append(record(control + "/campaign_status.json", "status/campaign_status.json", encoded(status)))
    owner = {"pid": 88, "proc_start_ticks": "1234", "proc_command": ["python", "run_campaign.py"]}
    records.append(record(control + "/campaign_owners/88-1234.json", "status/campaign_owners/88-1234.json",
                          encoded(owner), "owner_identity_snapshot"))
    records.append(record(control + "/logs/op.rloo_cardinal_beta005-123.out",
                          "logs/op.rloo_cardinal_beta005-123.out.tail", b"pending log\n",
                          "current_log_snapshot", tail=True))
    job_ids = ",".join(item["job_id"] for item in attempts.values())
    accounting = "".join(f"{item['job_id']}|{'COMPLETED' if i < completed_count else 'PENDING'}|0:0\n"
                         for i, item in enumerate(attempts.values()))
    queries = [{"argv": ["sacct", "--noheader", "--parsable2", "--jobs", job_ids,
                          "--format=JobIDRaw,State%40,ExitCode"], "returncode": 0, "stdout": accounting, "stderr": ""},
               {"argv": ["squeue", "--noheader", "--jobs", job_ids, "--format=%i|%T|%R"],
                "returncode": 0, "stdout": "123|PENDING|Priority\n", "stderr": ""}]
    raw_bindings = []
    for slot in collector.ORDER[:completed_count]:
        task, arm = slot.split(".")
        contract = contract_values[task]
        steps = 0 if arm == "sft" else 100
        checkpoint = {"id": slot, "initializer_sha256": contract["initializer_sha256"],
                      "reference_initializer_sha256": contract["initializer_sha256"], "train_seed": 43, "step": steps}
        resources = {"training_optimizer_steps": steps, "training_responses": steps * 36,
                     "training_generated_tokens": steps * 36, "training_wall_seconds": steps * .1,
                     "evaluation_wall_seconds": .5, "checkpoint_reload_exact": True,
                     "prompt_draw_sha256": contract["training"]["prompt_draw_sha256"]}
        raw = [{"instance_id": row["instance_id"], "completion_id": f"{mode}/{index}", "decode": mode,
                "sample_index": index, "text": row["output"], "generated_tokens": 1}
               for row in contract["validation_rows"] for mode, count in (("greedy", 1), ("k8", 8)) for index in range(count)]
        result = scorer.evaluate_arm(contract, arm, raw, checkpoint, resources)
        entry = next(item for item in entries if item["path"] == collector.ENTRYPOINT)
        complete = {"status": "completed_single_arm_model_evaluation", "superiority_established": False,
                    "checkpoint_reload_exact": True, "actual_RL_optimizer_steps": steps,
                    "actual_RL_responses": steps * 36, "actual_evaluation_responses": 9, "device": "cuda:0",
                    "source": {"path": source + "/" + collector.ENTRYPOINT,
                               "sha256": entry["sha256"], "size_bytes": entry["size_bytes"]}}
        binding = {"slot": slot, "job_id": attempts[slot]["job_id"]}
        for name, payload, native in (
            ("complete", complete, f"outputs/{slot}/complete.json"),
            ("result", result, f"outputs/{slot}/{task}/{arm}/result.json")):
            raw = encoded(payload)
            records.append(record(control + "/" + native, f"results/{slot}/{name}.json", raw,
                                  "completed_job_receipt" if name == "complete" else "completed_raw_result"))
            binding[name] = {"path": control + "/" + native, "sha256": collector.sha(raw), "size_bytes": len(raw)}
        raw_bindings.append(binding)
    if campaign_report:
        authority = {"plan": {"prepared_manifest_sha256": manifest_sha,
                              "source_manifest_sha256": manifest["source_manifest"]["sha256"],
                              "controller_sha256": next(item["sha256"] for item in entries if item["path"] == collector.CONTROLLER),
                              "order": list(collector.ORDER)}}
        authority["semantic_sha256"] = collector.canonical(authority)
        report = {"status": "completed_fixed_order_development_campaign", "scope": "production_development_pilot",
                  "superiority_established": False, "no_heldout_or_five_seed_superiority_claim": True,
                  "fixed_order": list(collector.ORDER), "actual_complete_slots": 12, "requested_gpu_hours": 84,
                  "authority_sha256": authority["semantic_sha256"], "raw_result_bindings": raw_bindings}
        records.append(record(control + "/campaign_authority.json", "status/campaign_authority.json", encoded(authority)))
        records.append(record(control + "/campaign_report.json", "campaign_report.json", encoded(report),
                              "completed_campaign_report"))
    bundle = {"schema_version": 1, "study": collector.STUDY, "collected_utc": "2026-10-13T21:12:23+00:00",
              "prepared_manifest_sha256": manifest_sha, "source_proofs": entries,
              "completed_slots": list(collector.ORDER[:completed_count]), "owner_probes": [{"pid": 88, "proc_exists": True}],
              "read_only_queries": queries, "records": records, "native_mutations": False, "superiority_established": False}
    return manifest, manifest_raw, manifest_sha, bundle


def replace_json(bundle, local, edit):
    index = next(i for i, item in enumerate(bundle["records"]) if item["local_path"] == local)
    item = bundle["records"][index]
    value = json.loads(base64.b64decode(item["content_base64"]))
    edit(value)
    bundle["records"][index] = record(item["native_path"], local, encoded(value), item["purpose"])


def test_partial_campaign_collects_status_only_and_preserves_unique_immutable_snapshots(tmp_path):
    manifest, raw, expected, bundle = fixture_bundle()
    packet = tmp_path / "launch_manifest.json"
    packet.write_bytes(raw)
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, json.dumps(bundle), "")
    output = tmp_path / "snapshot_v1"
    receipt = collector.collect_campaign(packet, expected, output, run=run)
    assert receipt["admission"]["status"] == "status_only_campaign_incomplete"
    assert not receipt["admission"]["completed_slots"] and not (output / "results").exists()
    assert not receipt["native_mutations"] and not receipt["superiority_established"]
    assert (output / "status/campaign_status.json").is_file()
    assert (output / "status/campaign_owners/88-1234.json").is_file()
    assert (output / "logs/op.rloo_cardinal_beta005-123.out.tail").read_bytes() == b"pending log\n"
    assert (output / "collection.json").stat().st_mode & 0o222 == 0
    assert calls[0][0][0] == "ssh" and calls[0][0][-2] == "jean-zay"
    command = calls[0][0][-1]
    assert collector.NATIVE_PYTHON + " -I -S -B -c " in command and "sbatch" not in command and "scancel" not in command
    with pytest.raises(ValueError, match="unique output"):
        collector.collect_campaign(packet, expected, output, run=run)
    assert len(calls) == 1


def test_exact_completed_slot_replays_raw_responses_without_superiority_claim():
    manifest, _, expected, bundle = fixture_bundle(completed_count=1)
    admission, decoded = collector.validate_bundle(bundle, manifest, expected)
    assert admission["status"] == "validated_partial_completed_slots"
    assert admission["completed_slots"][0]["actual_optimizer_steps"] == 100
    assert admission["completed_slots"][0]["actual_training_responses"] == 3600
    assert admission["completed_slots"][0]["raw_result_replay_exact"]
    assert not admission["superiority_established"]
    result = json.loads(decoded[f"results/{collector.ORDER[0]}/result.json"][1])
    assert len(result["raw_completions"]) == 9 and all(row["text"] for row in result["raw_completions"])


@pytest.mark.parametrize("field,value", [("actual_RL_optimizer_steps", 99), ("actual_RL_responses", 3564),
                                         ("actual_evaluation_responses", 8), ("device", "cpu"),
                                         ("superiority_established", True)])
def test_mismatched_completion_receipts_rejected(field, value):
    manifest, _, expected, bundle = fixture_bundle(completed_count=1)
    replace_json(bundle, f"results/{collector.ORDER[0]}/complete.json", lambda item: item.update({field: value}))
    with pytest.raises(ValueError, match="Completion/count/device"):
        collector.validate_bundle(bundle, manifest, expected)


def test_wrong_frozen_entrypoint_source_rejected():
    manifest, _, expected, bundle = fixture_bundle(completed_count=1)
    replace_json(bundle, f"results/{collector.ORDER[0]}/complete.json",
                 lambda item: item["source"].update(sha256="b" * 64))
    with pytest.raises(ValueError, match="frozen entrypoint"):
        collector.validate_bundle(bundle, manifest, expected)


@pytest.mark.parametrize("field,value", [("task", "cvrp"), ("arm", "foarl"), ("contract_sha256", "b" * 64)])
def test_wrong_result_task_arm_or_contract_rejected(field, value):
    manifest, _, expected, bundle = fixture_bundle(completed_count=1)
    replace_json(bundle, f"results/{collector.ORDER[0]}/result.json", lambda item: item.update({field: value}))
    with pytest.raises(ValueError, match="task/arm/contract"):
        collector.validate_bundle(bundle, manifest, expected)


def test_rehashed_forged_metrics_fail_independent_raw_replay():
    manifest, _, expected, bundle = fixture_bundle(completed_count=1)
    def forge(item):
        item["metrics"]["k8"]["feasibility_percent"] = 0
        item["result_sha256"] = collector.canonical({key: value for key, value in item.items() if key != "result_sha256"})
    replace_json(bundle, f"results/{collector.ORDER[0]}/result.json", forge)
    with pytest.raises(ValueError, match="independent raw-result replay"):
        collector.validate_bundle(bundle, manifest, expected)


@pytest.mark.parametrize("state,exitcode", [("PENDING", "0:0"), ("FAILED", "1:0"), ("COMPLETED", "1:0")])
def test_result_requires_clean_completed_scheduler_proof(state, exitcode):
    manifest, _, expected, bundle = fixture_bundle(completed_count=1)
    bundle["read_only_queries"][0]["stdout"] = f"123|{state}|{exitcode}\n"
    with pytest.raises(ValueError, match="completed scheduler proof"):
        collector.validate_bundle(bundle, manifest, expected)


def test_completed_parent_with_failed_job_step_is_not_admitted():
    manifest, _, expected, bundle = fixture_bundle(completed_count=1)
    bundle["read_only_queries"][0]["stdout"] += "123.batch|FAILED|1:0\n"
    with pytest.raises(ValueError, match="completed scheduler proof"):
        collector.validate_bundle(bundle, manifest, expected)


def test_snapshot_content_sha_and_frozen_source_mismatch_rejected():
    manifest, _, expected, bundle = fixture_bundle()
    bad = copy.deepcopy(bundle)
    bad["records"][0]["content_base64"] = base64.b64encode(b"changed bytes").decode()
    with pytest.raises(ValueError, match="content SHA/size"):
        collector.validate_bundle(bad, manifest, expected)
    bad = copy.deepcopy(bundle)
    bad["source_proofs"][0]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="full-source verification"):
        collector.validate_bundle(bad, manifest, expected)


def test_mutating_query_and_unsafe_snapshot_paths_rejected():
    manifest, _, expected, bundle = fixture_bundle()
    bundle["read_only_queries"][0]["argv"] = ["scancel", "123"]
    with pytest.raises(ValueError, match="Mutating"):
        collector.validate_bundle(bundle, manifest, expected)
    manifest, _, expected, bundle = fixture_bundle()
    bundle["records"][0]["local_path"] = "../escape.json"
    with pytest.raises(ValueError, match="Unsafe relative"):
        collector.validate_bundle(bundle, manifest, expected)


def test_complete_campaign_report_requires_all_twelve_exact_artifact_bindings():
    manifest, _, expected, bundle = fixture_bundle(completed_count=12, attempts_count=12, campaign_report=True)
    admission, _ = collector.validate_bundle(bundle, manifest, expected)
    assert admission["status"] == "validated_completed_campaign" and admission["campaign_report_admitted"]
    assert len(admission["completed_slots"]) == 12 and not admission["superiority_established"]
    replace_json(bundle, "campaign_report.json", lambda item: item["raw_result_bindings"][0]["result"].update(sha256="b" * 64))
    with pytest.raises(ValueError, match="artifact byte binding"):
        collector.validate_bundle(bundle, manifest, expected)


def test_local_prepared_manifest_sha_must_match_before_any_ssh_call(tmp_path):
    _, raw, _, _ = fixture_bundle()
    packet = tmp_path / "packet.json"
    packet.write_bytes(raw)
    def prohibited(*args, **kwargs):
        raise AssertionError("Unexpected SSH call")
    with pytest.raises(ValueError, match="Local prepared manifest byte SHA"):
        collector.collect_campaign(packet, "f" * 64, tmp_path / "new", run=prohibited)


def test_bound_completed_artifact_must_be_stable_at_snapshot_time():
    manifest, _, expected, bundle = fixture_bundle(completed_count=1)
    next(row for row in bundle["records"] if row["purpose"] == "completed_raw_result")["snapshot_stable"] = False
    with pytest.raises(ValueError, match="changed during collection"):
        collector.validate_bundle(bundle, manifest, expected)
