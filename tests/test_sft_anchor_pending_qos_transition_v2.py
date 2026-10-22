"""Pure/mocked existing-job QoS safeguards; never invokes a scheduler."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments/rloo_sft_anchor_20261013/transition_pending_qos_v2.py"
loader = importlib.util.spec_from_file_location("pending_qos_transition_subject", PATH)
subject = importlib.util.module_from_spec(loader); loader.loader.exec_module(subject)
PREFLIGHT = ROOT / "experiments/rloo_sft_anchor_20261013/validation/native_monitor/qos_transition_preflight_v2.json"


def raw_before():
    return json.loads(json.loads(PREFLIGHT.read_text())["stdout"])["queries"][0]["stdout"]


def before_and_spec():
    before = subject.parse_job(raw_before())
    spec = {"protocol": subject.PROTOCOL + "-spec", "job_id": subject.JOB,
        "old_qos": subject.OLD_QOS, "new_qos": subject.NEW_QOS, "changed_requested_fields": ["QOS"],
        "requested_gpu_hours": 8, "combined_maximum_requested_gpu_hours": 92,
        "fixed_scheduler_fields": {key: before[key] for key in subject.FIXED_FIELDS},
        "running_allocation_tres": {"cpu": "80", "mem": "300000M", "node": "1", "billing": "5", "gres/gpu": "4"}}
    return before, spec


def serialize(record):
    return " ".join(f"{key}={value}" for key, value in record.items()) + "\n"


def running_after(before):
    return {**before, "QOS": subject.NEW_QOS, "JobState": "RUNNING", "Reason": "None",
        "NodeList": "jean-zay-iam09", "NumNodes": "1", "NumCPUs": "80",
        "AllocTRES": "cpu=80,mem=300000M,node=1,billing=5,gres/gpu=4"}


def test_actual_native_before_parses_complete_record_and_preserves_spaceful_submitline():
    before, spec = before_and_spec()
    subject.validate_before(before, spec)
    assert before["JobId"] == "1063203" and before["QOS"] == "qos_gpu_a100-dev"
    assert before["SubmitLine"].startswith("sbatch --parsable /lustre/")
    assert before["NodeList"] == "" and before["ReqTRES"] == "cpu=40,mem=300000M,node=1,billing=4,gres/gpu=4"
    assert subject.parse_job(serialize(before)) == before


@pytest.mark.parametrize("change", ["duplicate", "new_field", "missing_field", "multiline", "prefix"])
def test_parser_rejects_incomplete_or_ambiguous_scheduler_records(change):
    text = raw_before()
    if change == "duplicate": text = text.rstrip() + " JobId=1063203\n"
    elif change == "new_field": text = text.rstrip() + " NewSlurmField=1\n"
    elif change == "missing_field": text = text.replace(" QOS=qos_gpu_a100-dev", "")
    elif change == "multiline": text += text
    else: text = "unexpected prefix " + text
    with pytest.raises(ValueError): subject.parse_job(text)


def test_pending_qos_only_update_and_same_geometry_running_race_are_admitted():
    before, spec = before_and_spec()
    assert subject.validate_after(before, {**before, "QOS": subject.NEW_QOS}, spec) == ["QOS"]
    after = running_after(before)
    changed = subject.validate_after(before, after, spec)
    assert "QOS" in changed and "AllocTRES" in changed


@pytest.mark.parametrize("field,value", [("TimeLimit", "04:00:00"), ("ReqTRES", "cpu=40,mem=300000M,node=1,billing=8,gres/gpu=8"),
    ("TresPerNode", "gres/gpu:8"), ("TresPerTask", "cpu=80"), ("CPUs/Task", "80"), ("MinCPUsNode", "80"),
    ("Account", "vas@h100"), ("UserId", "other(123)"), ("Partition", "gpu_p6"), ("Requeue", "1"),
    ("Command", "/another/script.slurm"), ("Comment", "another-attempt"), ("JobId", "1052261")])
def test_resource_script_owner_or_attempt_changes_are_rejected(field, value):
    before, spec = before_and_spec(); after = {**before, "QOS": subject.NEW_QOS, field: value}
    with pytest.raises(ValueError): subject.validate_after(before, after, spec)


@pytest.mark.parametrize("field,value", [("JobState", "RUNNING"), ("QOS", subject.NEW_QOS), ("JobId", "1052261")])
def test_only_exact_original_pending_job_is_admitted_before_action(field, value):
    before, spec = before_and_spec(); before[field] = value
    with pytest.raises(ValueError): subject.validate_before(before, spec)


@pytest.mark.parametrize("field,value", [("NumCPUs", "160"), ("NumNodes", "2"),
    ("AllocTRES", "cpu=160,mem=300000M,node=1,billing=10,gres/gpu=4"),
    ("AllocTRES", "cpu=80,mem=600000M,node=1,billing=5,gres/gpu=4"),
    ("AllocTRES", "cpu=80,mem=300000M,node=2,billing=5,gres/gpu=4"),
    ("AllocTRES", "cpu=80,mem=300000M,node=1,billing=6,gres/gpu=4"),
    ("AllocTRES", "cpu=80,mem=300000M,node=1,billing=5,gres/gpu=8"),
    ("AllocTRES", "cpu=160,cpu=80,mem=300000M,node=1,billing=5,gres/gpu=4"),
    ("AllocTRES", "cpu=80,mem=300000M,node=1,billing=5,gres/gpu=4,gres/gpu:h100=4")])
def test_actual_running_allocation_cannot_hide_resource_increase(field, value):
    before, spec = before_and_spec(); after = running_after(before); after[field] = value
    with pytest.raises(ValueError): subject.validate_after(before, after, spec)


@pytest.mark.parametrize("field,value", [("NumCPUs", "80"), ("NumNodes", "2"), ("AllocTRES", "cpu=80,mem=300000M,node=1,billing=5,gres/gpu=4"),
    ("RunTime", "00:01:00"), ("NodeList", "jean-zay-iam09")])
def test_pending_readback_cannot_hide_allocation_or_running_work(field, value):
    before, spec = before_and_spec(); after = {**before, "QOS": subject.NEW_QOS, field: value}
    with pytest.raises(ValueError): subject.validate_after(before, after, spec)


@pytest.mark.parametrize("state", ["FAILED", "COMPLETED", "TIMEOUT", "CANCELLED", "COMPLETING"])
def test_terminal_or_unknown_post_action_state_requires_investigation(state):
    before, spec = before_and_spec(); after = {**before, "QOS": subject.NEW_QOS, "JobState": state}
    with pytest.raises(ValueError): subject.validate_after(before, after, spec)


@pytest.mark.parametrize("exception", [subprocess.TimeoutExpired(["scontrol"], 25, output=b"partial update", stderr=b"waiting"), OSError("command unavailable")])
def test_command_timeout_or_oserror_is_a_serializable_ambiguous_action(monkeypatch, exception):
    def fail(*args, **kwargs): raise exception
    monkeypatch.setattr(subject.subprocess, "run", fail)
    result = subject.run(["scontrol", "update", "JobId=1063203", "QOS=qos_gpu_a100-t3"])
    assert result["returncode"] != 0 and result["argv"][1] == "update"
    assert type(exception).__name__ in json.dumps(result)
    if isinstance(exception, subprocess.TimeoutExpired): assert "partial update" in json.dumps(result)


def test_immutable_store_preserves_original_and_rejects_retry(tmp_path):
    path = tmp_path / "intent.json"
    subject.store(path, {"intent": "exact"})
    original = path.read_bytes()
    with pytest.raises(ValueError, match="never retry"):
        subject.store(path, {"intent": "changed"})
    assert path.read_bytes() == original and path.stat().st_mode & 0o777 == 0o440


def mocked_main_fixture(tmp_path, monkeypatch):
    before, spec = before_and_spec()
    source = tmp_path / "source_v2"; source.mkdir()
    files = []
    for index in range(121):
        path = source / f"source_{index}.py"; path.write_text(f"VALUE={index}\n")
        files.append({"path": path.name, **subject.identity(path)})
    manifest = source / "source_manifest.json"; manifest.write_text(json.dumps({"files": files}))
    bindings = {"source_v2/source_manifest.json": subject.identity(manifest)}
    for name in subject.NATIVE_HASHES:
        if name == "source_v2/source_manifest.json": continue
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("immutable native fixture " + name)
        bindings[name] = subject.identity(path)
    monkeypatch.setattr(subject, "NATIVE_HASHES", {key: row["sha256"] for key, row in bindings.items()})
    proc = Path("/proc") / str(os.getpid())
    ticks = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
    command = (proc / "cmdline").read_bytes().rstrip(bytes([0]))
    spec.update(program_binding=subject.identity(PATH), native_bindings=bindings,
        owner={"pid": os.getpid(), "proc_start_ticks": ticks, "command_sha256": hashlib.sha256(command).hexdigest()})
    spec["spec_sha256"] = subject.canonical(spec)
    spec_path = tmp_path / "spec.json"; spec_path.write_text(json.dumps(spec)); spec_path.chmod(0o440)
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(subject.socket, "gethostname", lambda: "jean-zay1")
    monkeypatch.setattr(subject.getpass, "getuser", lambda: "uag72kz")
    monkeypatch.setattr(sys, "argv", [str(PATH), "--spec", str(spec_path), "--spec-sha256", subject.identity(spec_path)["sha256"], "--execute"])
    return before, spec


@pytest.mark.parametrize("failure", ["action_nonzero", "action_timeout", "post_query_failed", "post_parse_bad", "post_resource_changed", "post_source_changed", "post_terminal"])
def test_every_post_intent_failure_seals_ambiguous_receipt_and_never_retries(tmp_path, monkeypatch, failure):
    before, spec = mocked_main_fixture(tmp_path, monkeypatch)
    calls = []; query_count = 0
    def native(argv, **kwargs):
        nonlocal query_count
        calls.append(argv)
        result = {"argv": argv, "returncode": 0, "stdout": "", "stderr": ""}
        if "assoc" in argv: result["stdout"] = f"vas@a100||{subject.OLD_QOS},{subject.NEW_QOS}|{subject.NEW_QOS}\n"
        elif argv[0] == "sacctmgr": result["stdout"] = f"{subject.OLD_QOS}|1.000000|\n{subject.NEW_QOS}|1.000000|\n"
        elif argv[1] == "update":
            if failure == "action_nonzero": result["returncode"] = 1; result["stderr"] = "denied"
            if failure == "action_timeout": raise subprocess.TimeoutExpired(argv, 25, output="ambiguous update")
            if failure == "post_source_changed": (tmp_path / "source_v2/source_1.py").write_text("CHANGED=1\n")
        else:
            query_count += 1
            if query_count == 1: result["stdout"] = serialize(before)
            elif failure == "post_query_failed": result["returncode"] = 1; result["stderr"] = "gone"
            elif failure == "post_parse_bad": result["stdout"] = "invalid readback"
            else:
                after = {**before, "QOS": subject.NEW_QOS}
                if failure == "post_resource_changed": after["TimeLimit"] = "04:00:00"
                if failure == "post_terminal": after["JobState"] = "FAILED"
                result["stdout"] = serialize(after)
        return SimpleNamespace(returncode=result["returncode"], stdout=result["stdout"], stderr=result["stderr"])
    # Every subprocess is fake; retain the program's actual timeout/error wrapper.
    monkeypatch.setattr(subject.subprocess, "run", native)
    with pytest.raises((ValueError, RuntimeError)): subject.main()
    runtime = tmp_path / "qos_transition_v2_runtime"
    assert (runtime / "intent.json").exists() and (runtime / "action.json").exists()
    receipt = json.loads((runtime / "transition_receipt.json").read_text())
    assert receipt["status"] == "requires_investigation"
    assert receipt["statistical_source_modified"] is (True if failure == "post_source_changed" else None)
    assert receipt["script_modified"] is None and receipt["attempt_modified"] is None
    assert len([argv for argv in calls if argv[1] == "update"]) == 1
    with pytest.raises((ValueError, RuntimeError, FileExistsError)): subject.main()
    assert len([argv for argv in calls if argv[1] == "update"]) == 1


@pytest.mark.parametrize("state", ["PENDING", "RUNNING"])
def test_mocked_success_sends_exactly_one_qos_only_argv_and_closes_full_custody(tmp_path, monkeypatch, state):
    before, spec = mocked_main_fixture(tmp_path, monkeypatch)
    calls = []; queries = 0
    def native(argv, **kwargs):
        nonlocal queries
        calls.append(argv); stdout = ""
        if "assoc" in argv: stdout = f"vas@a100||{subject.OLD_QOS},{subject.NEW_QOS}|{subject.NEW_QOS}\n"
        elif argv[0] == "sacctmgr": stdout = f"{subject.OLD_QOS}|1.000000|\n{subject.NEW_QOS}|1.000000|\n"
        elif argv[1] != "update":
            queries += 1
            after = running_after(before) if state == "RUNNING" else {**before, "QOS": subject.NEW_QOS}
            stdout = serialize(before if queries == 1 else after)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    monkeypatch.setattr(subject.subprocess, "run", native)
    subject.main()
    updates = [argv for argv in calls if argv[1] == "update"]
    assert updates == [["scontrol", "update", "JobId=1063203", "QOS=qos_gpu_a100-t3"]]
    assert all(argv[0] not in ("sbatch", "scancel") for argv in calls)
    runtime = tmp_path / "qos_transition_v2_runtime"
    receipt = json.loads((runtime / "transition_receipt.json").read_text())
    assert receipt["status"] == "completed_qos_only_transition" and receipt["post_source_files_verified"] == 121
    assert receipt["statistical_source_modified"] is receipt["script_modified"] is receipt["attempt_modified"] is False
    assert receipt["requested_gpu_hours"] == 8 and receipt["combined_maximum_requested_gpu_hours"] == 92
    assert receipt["intent_binding"] == subject.identity(runtime / "intent.json")
    assert receipt["action_binding"] == subject.identity(runtime / "action.json")
    assert receipt["receipt_sha256"] == subject.canonical({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    assert all((runtime / name).stat().st_mode & 0o777 == 0o440 for name in ("intent.json", "action.json", "transition_receipt.json"))


def test_missing_authority_binding_or_source_inventory_cannot_reach_action(tmp_path, monkeypatch):
    before, spec = mocked_main_fixture(tmp_path, monkeypatch)
    missing = copy.deepcopy(spec)
    missing["native_bindings"].pop(next(key for key in missing["native_bindings"] if "authority" in key))
    with pytest.raises(ValueError, match="Mandatory original"):
        subject.verify_native(missing)
    source = tmp_path / "source_v2/source_manifest.json"
    manifest = json.loads(source.read_text()); manifest["files"][1] = manifest["files"][0]
    source.write_text(json.dumps(manifest))
    spec["native_bindings"]["source_v2/source_manifest.json"] = subject.identity(source)
    monkeypatch.setitem(subject.NATIVE_HASHES, "source_v2/source_manifest.json", subject.identity(source)["sha256"])
    with pytest.raises(ValueError, match="inventory"):
        subject.verify_native(spec)


def test_known_state_dependent_scheduler_inventory_is_checked_without_relaxing_resources():
    before, spec = before_and_spec()
    assert 'SchedNodeList' not in before and 'BatchHost' not in before
    after = running_after(before)
    after['BatchHost'] = 'jean-zay-iam09'
    after.pop('SchedNodeList', None)
    parsed = subject.parse_job(serialize(after))
    assert 'BatchHost' in subject.validate_after(before, parsed, spec)
    pending = {**before, 'QOS': subject.NEW_QOS, 'SchedNodeList': 'jean-zay-iam09'}
    subject.validate_after(before, subject.parse_job(serialize(pending)), spec)
    with pytest.raises(ValueError):
        subject.parse_job(serialize({**after, 'UnreviewedField': 'value'}))


@pytest.mark.parametrize('key,value', [('AllocTRES','cpu=80,mem=300000M,node=1,billing=5,gres/gpu=4'), ('NodeList','jean-zay-iam09'), ('RunTime','00:00:01'), ('ExitCode','1:0'), ('BatchHost','jean-zay-iam09')])
def test_pending_start_race_or_inconsistent_allocation_exits_before_action(key,value):
    before, spec = before_and_spec()
    with pytest.raises(ValueError):
        subject.validate_before({**before,key:value},spec)
