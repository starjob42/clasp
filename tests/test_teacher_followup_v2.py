"""No policy calls: conditional admission and owned-process cleanup tests."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import pytest

SOURCE = Path(__file__).parents[1] / "experiments/rloo_one_shot_20261016/run_teacher_followup_v2.py"
spec = importlib.util.spec_from_file_location("teacher_followup_under_test", SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def original_fixture(tmp_path, status):
    roster = {"path": "/registered/roster.json", "size_bytes": 3, "sha256": "a"*64}
    owner = {"pid": 12, "start_ticks": "99", "roster": roster}
    p = tmp_path / "owner.json"; p.write_text(json.dumps(owner))
    state = {"status": status, "owner": owner, "analysis_results": {}}
    (tmp_path / "status.json").write_text(json.dumps(state))
    return {"original_controller_owner": mod.binding(p), "original_roster": roster}, state


@pytest.mark.parametrize("status", ["running", "waiting_for_idle_local_gpu", "analyzing_complete_op"])
def test_incomplete_original_never_reads_screen(tmp_path, status):
    packet, _ = original_fixture(tmp_path, status)
    # Deliberately unreadable/malformed numerical report must remain unopened.
    (tmp_path / "op_analysis.json").write_bytes(b"not json")
    assert mod.original_condition(packet) == ("wait", None)


def test_same_current_owner_required(tmp_path):
    packet, state = original_fixture(tmp_path, "running")
    state["owner"]["pid"] = 13
    (tmp_path / "status.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="identity"):
        mod.original_condition(packet)


def test_original_pass_does_not_launch_followup(tmp_path):
    packet, state = original_fixture(tmp_path, "running")
    state["analysis_results"] = {"op": {"path": "never-open-this"}}
    (tmp_path / "status.json").write_text(json.dumps(state))
    assert mod.original_condition(packet)[0] == "not_applicable_original_OP_passed"


def test_failed_original_requires_completed_failed_screen(tmp_path, monkeypatch):
    packet, _ = original_fixture(tmp_path, "development_screen_failed_requires_next_candidate")
    packet["old_metadata"] = {"test": "bound separately in production packet"}
    monkeypatch.setattr(mod, "module", lambda *args: types.SimpleNamespace(
        admit_complete_task=lambda *args: {"status": "complete_metadata_admission"}))
    report = {"status": "complete_task_raw_replay", "task": "op", "development_screen_passed": False,
              "superiority_established": False, "roster": packet["original_roster"]}
    p = tmp_path / "op_analysis.json"; p.write_text(json.dumps(report))
    condition, bound = mod.original_condition(packet)
    assert condition == "trigger" and json.loads(mod.read_bound(bound)) == report
    report["development_screen_passed"] = True; p.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="failed OP"):
        mod.original_condition(packet)


def test_old_all_arm_gate_precedes_quality_read(tmp_path, monkeypatch):
    packet, _ = original_fixture(tmp_path, "development_screen_failed_requires_next_candidate")
    packet["old_metadata"] = {"test": "bound separately in production packet"}
    (tmp_path / "op_analysis.json").write_bytes(b"must never be parsed")
    monkeypatch.setattr(mod, "module", lambda *args: types.SimpleNamespace(
        admit_complete_task=lambda *args: {"status": "incomplete_metadata_admission"}))
    with pytest.raises(ValueError, match="before opening quality"):
        mod.original_condition(packet)


@pytest.mark.parametrize("status", ["failed_requires_reconciliation", "development_complete_requires_confirmation"])
def test_original_failure_does_not_create_another_attempt(tmp_path, status):
    packet, _ = original_fixture(tmp_path, status)
    assert mod.original_condition(packet)[0] == "original_requires_reconciliation"


def test_default_dryrun_never_uses_bad_packet(tmp_path):
    result = subprocess.run([sys.executable, "-I", "-B", str(SOURCE), "--packet", str(tmp_path / "missing")],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert json.loads(result.stdout)["GPU_calls"] == 0


def test_cpu_environment_hides_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unrelated-device")
    assert mod.environment(cpu=True)["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "unrelated-device"


def test_launch_requires_pinned_exact_source(tmp_path):
    p = tmp_path / "runner.py"; p.write_text("raise RuntimeError('must never execute')")
    bound = mod.binding(p)
    argv = mod.command(bound, "--execute")
    assert argv[:4] == [mod.PYTHON, "-I", "-B", "-c"]
    p.write_text("changed")
    with pytest.raises(ValueError, match="bytes differ"):
        mod.command(bound, "--execute")


def test_owned_session_cleanup_reaps_parent_and_stops_descendant(tmp_path):
    marker = tmp_path / "child_pid"
    code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-I','-B','-c','import time;time.sleep(60)']);open(sys.argv[1],'w').write(str(p.pid));time.sleep(60)"
    process = subprocess.Popen([sys.executable, "-I", "-B", "-c", code, str(marker)], start_new_session=True)
    try:
        deadline = time.monotonic()+5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert marker.exists()
        descendant = int(marker.read_text())
        mod.stop_own_group(process)
        assert process.poll() is not None
        stat = Path(f"/proc/{descendant}/stat")
        # A reparented short-lived zombie has stopped and consumes no resource.
        assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    finally:
        if process.poll() is None:
            mod.stop_own_group(process)


@pytest.mark.parametrize("field,value", [("maximum_new_requested_GPU_hours", 21.0), ("automatic_retry", True),
    ("study_order", ["rloo_teacher"]), ("publication_eligible", True), ("execution_deadline_utc", "later")])
def test_budget_and_attempt_scope_reject_before_inputs(field, value):
    p = {"protocol": mod.PROTOCOL, "original_roster": {"sha256": mod.ROSTER_SHA}, "plan": {"sha256": mod.PLAN_SHA},
         "maximum_new_requested_GPU_hours": 20.5, "execution_deadline_utc": mod.DEADLINE.isoformat(), "automatic_retry": False,
         "study_order": list(mod.STUDY_ARMS), "publication_eligible": False, "superiority_established": False}
    p[field] = value; p["packet_sha256"] = mod.sha(mod.encoded(p))
    with pytest.raises(ValueError, match="authority"):
        mod.validate_packet(p)
