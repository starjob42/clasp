"""CPU-only handoff guards. Only one freshly owned local child is signalled."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import types

import pytest


SOURCE = (Path(__file__).resolve().parents[1] /
          "experiments/rloo_sft_anchor_20261013/resource_deadline_extension_v2/handoff_controller_v2.py")


@pytest.fixture
def h():
    raw = SOURCE.read_bytes()
    module = types.ModuleType("handoff_guard_fixture")
    module.__file__ = str(SOURCE)
    module.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    exec(compile(raw, str(SOURCE), "exec", dont_inherit=True), module.__dict__)
    return module


def owner(pid=491234):
    return {"pid": pid, "uid": os.getuid(), "hostname": socket.gethostname().split(".")[0],
            "proc_start_ticks": "7654321", "command_sha256": "c" * 64,
            "command_sha256_format": "proc_cmdline_without_terminal_NUL"}


def snapshot(expected, **changes):
    return {**expected, "alive": True, "effective_uid": expected["uid"],
            "all_uids": [expected["uid"]] * 4, "state": "S", "parent_pid": os.getpid(),
            "children": [], "sig_ignored": "0", "sig_caught": "0", "sig_blocked": "0",
            **changes}


def spec(h, expected=None, **changes):
    return h.seal({"protocol": h.PROTOCOL + "-spec", "synthetic_fixture": True,
                   "old_owner": expected or owner(), "first_job_id": "1149471",
                   "old_operator_sha256": h.OLD_OPERATOR_SHA,
                   "old_registration_sha256": h.OLD_REGISTRATION_SHA,
                   "new_registration_sha256": "a" * 64, "handoff_helper_sha256": "b" * 64,
                   "scheduler_mutations_allowed": False, "automatic_retry": False, **changes})


class SimulatedChild:
    def __init__(self, h, *, edit=None, boundary_failure=False, intent_failure=False,
                 never_stops=False, never_dies=False, signal_failure=None):
        self.h = h
        self.expected = owner()
        self.current = snapshot(self.expected)
        self.signals = []
        self.events = []
        self.snapshots = 0
        self.boundaries = 0
        self.edit = edit
        self.boundary_failure = boundary_failure
        self.intent_failure = intent_failure
        self.never_stops = never_stops
        self.never_dies = never_dies
        self.signal_failure = signal_failure

    def read(self, pid):
        assert pid == self.expected["pid"]
        self.snapshots += 1
        result = copy.deepcopy(self.current)
        if self.edit:
            self.edit(self.snapshots, result)
        return result

    def send(self, pid, sig):
        assert pid == self.expected["pid"] != self.h.OWNER["pid"]
        if sig == self.signal_failure:
            raise OSError("synthetic signal ambiguity")
        self.signals.append(sig)
        if sig == signal.SIGSTOP and not self.never_stops:
            self.current["state"] = "T"
        elif sig == signal.SIGCONT:
            self.current["state"] = "S" if self.never_dies else "Z"

    def boundary(self):
        self.boundaries += 1
        if self.boundary_failure and self.boundaries == 2:
            raise ValueError("synthetic first-job boundary changed")
        return {"first_job_id": "1149471", "remaining_five_empty": True,
                "scheduler_mutations": []}

    def write(self, name, value):
        if self.intent_failure and name == "intent":
            raise FileExistsError("existing immutable handoff intent; no retry")
        assert value["receipt_sha256"] == self.h.seal(value)["receipt_sha256"]
        self.events.append((name, copy.deepcopy(value)))

    def execute(self, **kwargs):
        return self.h.quiesce_and_terminate(
            spec(self.h, self.expected), signal_fn=self.send, snapshot=self.read,
            verify_boundary=self.boundary, write_event=self.write,
            sleep=lambda seconds: None, timeout=.002, **kwargs)


def test_mocked_sequence_is_stop_verify_term_continue_and_no_scheduler(h):
    child = SimulatedChild(h)
    result = child.execute()
    assert child.signals == [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT]
    assert [name for name, _ in child.events] == ["intent", "quiesced", "complete"]
    assert child.boundaries == 2
    assert result["synthetic_fixture"] is True
    assert result["status"] == "old_cpu_owner_terminated_no_scheduler_mutations"
    assert result["first_job_id"] == "1149471"
    assert result["scheduler_mutations"] == [] and result["model_calls"] == result["quality_reads"] == 0
    assert signal.SIGKILL not in child.signals


@pytest.mark.parametrize("field,value", [
    ("pid", 491235), ("uid", -1), ("hostname", "another-host"),
    ("proc_start_ticks", "7654322"), ("command_sha256", "d" * 64),
    ("command_sha256_format", "including_trailing_NUL"),
    ("effective_uid", -1), ("all_uids", [0, 0, 0, 0]),
    ("children", [12345]), ("state", "Z"), ("alive", False),
    ("parent_pid", -1),
])
def test_wrong_identity_or_child_before_stop_never_signals(h, field, value):
    child = SimulatedChild(h, edit=lambda n, row: row.update({field: value}))
    with pytest.raises(ValueError):
        child.execute()
    assert child.signals == [] and child.events == []


@pytest.mark.parametrize("field", ["sig_ignored", "sig_caught", "sig_blocked"])
@pytest.mark.parametrize("stage", ["before", "frozen"])
def test_term_disposition_must_be_safe_before_and_after_stop(h, field, stage):
    def change(n, row):
        if (stage == "before" and n == 1) or (stage == "frozen" and n == 2):
            row[field] = format(1 << (signal.SIGTERM - 1), "x")
    child = SimulatedChild(h, edit=change)
    with pytest.raises(ValueError, match="SIGTERM"):
        child.execute()
    assert child.signals == ([] if stage == "before" else [signal.SIGSTOP])
    if stage == "frozen":
        failure = child.events[-1]
        assert failure[0] == "requires_investigation"
        assert failure[1]["old_owner_quiesced"] is True
        assert failure[1]["termination_sent"] is False
        assert failure[1]["automatic_resume"] is False


@pytest.mark.parametrize("field,value", [
    ("proc_start_ticks", "reused"), ("children", [1234]), ("effective_uid", -1),
])
def test_post_stop_identity_or_child_drift_preserves_stopped_state(h, field, value):
    child = SimulatedChild(h, edit=lambda n, row: row.update({field: value}) if n >= 2 else None)
    with pytest.raises(ValueError):
        child.execute()
    assert child.signals == [signal.SIGSTOP] and child.current["state"] == "T"
    assert child.events[-1][0] == "requires_investigation"


def test_second_boundary_failure_preserves_stop_without_term_resume_or_retry(h):
    child = SimulatedChild(h, boundary_failure=True)
    with pytest.raises(ValueError, match="boundary changed"):
        child.execute()
    assert child.signals == [signal.SIGSTOP] and child.current["state"] == "T"
    assert [name for name, _ in child.events] == ["intent", "requires_investigation"]


def test_first_boundary_failure_happens_before_intent_and_every_signal(h):
    child = SimulatedChild(h)
    def reject():
        raise ValueError("initial boundary not closed")
    child.boundary = reject
    with pytest.raises(ValueError, match="initial boundary"):
        child.execute()
    assert child.signals == [] and child.events == []


def test_post_term_exiting_process_may_clear_cmdline_before_becoming_zombie(h):
    def exiting(n, row):
        if n == 4:
            row.update(state="R", command_sha256=hashlib.sha256(b"").hexdigest(), proc_command=[])
    child = SimulatedChild(h, edit=exiting)
    result = child.execute()
    assert child.snapshots == 5
    assert result["final_owner"]["state"] == "Z"
    assert child.signals == [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT]


@pytest.mark.parametrize("field,value", [("command_sha256", "d" * 64),
    ("proc_start_ticks", "reused"), ("all_uids", [-1, -1, -1, -1])])
def test_post_term_live_identity_drift_stops_without_further_signals(h, field, value):
    def exiting(n, row):
        if n == 4:
            row.update(state="R")
            row[field] = value
    child = SimulatedChild(h, edit=exiting)
    with pytest.raises(ValueError):
        child.execute()
    assert child.signals == [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT]
    assert child.events[-1][0] == "requires_investigation"
    assert all(name != "complete" for name, _ in child.events)


def test_existing_intent_denies_all_signals_and_no_automatic_retry(h):
    child = SimulatedChild(h, intent_failure=True)
    with pytest.raises(FileExistsError, match="no retry"):
        child.execute()
    assert child.signals == [] and child.events == []


@pytest.mark.parametrize("field,value", [("proc_start_ticks", "reused"), ("uid", -1), ("pid", 491235)])
def test_terminal_reused_process_is_not_proof_of_original_death(h, field, value):
    child = SimulatedChild(h, edit=lambda n, row: row.update({field: value}) if n >= 4 else None)
    with pytest.raises(ValueError, match="terminal PID"):
        child.execute()
    assert child.signals == [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT]
    assert child.events[-1][0] == "requires_investigation"
    assert all(name != "complete" for name, _ in child.events)


def test_nonquiescent_process_times_out_without_termination(h):
    child = SimulatedChild(h, never_stops=True)
    with pytest.raises(ValueError, match="quiesce"):
        child.execute()
    assert child.signals == [signal.SIGSTOP]
    assert child.events[-1][1]["automatic_resume"] is False


def test_unobserved_death_never_escalates_to_kill_or_retry(h):
    child = SimulatedChild(h, never_dies=True)
    with pytest.raises(ValueError, match="Termination not observed"):
        child.execute()
    assert child.signals == [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT]
    assert child.events[-1][1]["automatic_retry"] is False


def test_ambiguous_stop_failure_records_no_retry_and_no_followup_signal(h):
    child = SimulatedChild(h, signal_failure=signal.SIGSTOP)
    with pytest.raises(OSError, match="signal ambiguity"):
        child.execute()
    assert child.signals == []
    assert child.events[-1][0] == "requires_investigation"


@pytest.mark.parametrize("changes", [
    {"synthetic_fixture": False}, {"first_job_id": "999999"}, {"automatic_retry": True},
    {"scheduler_mutations_allowed": True}, {"old_operator_sha256": "0" * 64},
    {"old_registration_sha256": "0" * 64}, {"new_registration_sha256": False},
    {"handoff_helper_sha256": "bad"},
])
def test_diagnostic_spec_cannot_authorize_native_or_change_first_job(h, changes):
    called = []
    with pytest.raises(ValueError):
        h.quiesce_and_terminate(spec(h, **changes), signal_fn=lambda *args: called.append(args),
            snapshot=lambda pid: snapshot(owner()), verify_boundary=lambda: {}, write_event=lambda *args: None)
    assert called == []


@pytest.mark.parametrize("pid", [True, False, 0, 1, -1, "123"])
def test_process_snapshot_rejects_unsafe_or_untyped_pid(h, pid):
    with pytest.raises(ValueError, match="Positive PID"):
        h.process_snapshot(pid)


def test_native_expected_owner_cannot_be_reused_as_diagnostic(h):
    with pytest.raises(ValueError, match="freshly owned direct child"):
        h.validate_old_owner(snapshot(h.OWNER), h.OWNER, diagnostic=True)


def test_proc_snapshot_reads_blocked_term_and_children_of_every_thread(h, tmp_path):
    pid = 491234
    p = tmp_path / str(pid)
    p.mkdir()
    fields = ["S", str(os.getpid())] + ["0"] * 17 + ["7654321"] + ["0"] * 4
    (p / "stat").write_text(str(pid) + " (a process ) name) " + " ".join(fields))
    (p / "cmdline").write_bytes(b"python\0-c\0pass\0")
    (p / "status").write_text("Uid:\t1610\t1610\t1610\t1610\nSigIgn:\t0\nSigCgt:\t0\nSigBlk:\t4000\n")
    for tid, children in [("491234", "12 13"), ("491235", "13 14")]:
        t = p / "task" / tid
        t.mkdir(parents=True)
        (t / "children").write_text(children)
    value = h.process_snapshot(pid, proc_root=tmp_path)
    assert value["children"] == [12, 13, 14]
    assert value["sig_blocked"] == "4000"
    assert value["proc_start_ticks"] == "7654321"
    assert value["command_sha256"] == hashlib.sha256(b"python\0-c\0pass").hexdigest()


def test_actual_fresh_owned_cpu_child_only_is_terminated(h, tmp_path):
    child = subprocess.Popen([sys.executable, "-I", "-S", "-c",
        "import time;print('ready',flush=True);time.sleep(120)"], stdout=subprocess.PIPE, text=True)
    assert child.pid != h.OWNER["pid"]
    events = []
    observed = []
    before = h.process_snapshot(child.pid)
    try:
        assert child.stdout.readline().strip() == "ready"
        before = h.process_snapshot(child.pid)
        expected = {k: before[k] for k in owner()}
        h.validate_old_owner(before, expected, diagnostic=True)
        def record_signal(pid, sig):
            assert pid == child.pid and pid != h.OWNER["pid"]
            observed.append(sig)
            os.kill(pid, sig)
        result = h.quiesce_and_terminate(spec(h, expected), signal_fn=record_signal,
            verify_boundary=lambda: {"synthetic_fixture": True, "first_job_id": "1149471",
                "remaining_five_empty": True, "scheduler_mutations": []},
            write_event=lambda name, value: events.append({"name": name, "value": value}), timeout=2)
        assert child.wait(timeout=2) == -signal.SIGTERM
        assert observed == [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT]
        assert result["synthetic_fixture"] is True and result["scheduler_mutations"] == []
        assert result["old_owner"] == expected and result["first_job_id"] == "1149471"
        assert [row["name"] for row in events] == ["intent", "quiesced", "complete"]
        evidence = {"scope": "owned_local_CPU_child_diagnostic_only", "before": before,
                    "events": events, "wait_returncode": child.returncode,
                    "native_actions": False, "models": False, "GPU_calls": False}
        path = Path(os.environ.get("HANDOFF_CPU_PROOF_PATH", str(tmp_path / "owned_child_proof.json")))
        path.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
    finally:
        # Cleanup is scoped to this Popen child. Never touch a native/registered PID.
        if child.poll() is None:
            current = h.process_snapshot(child.pid)
            assert current.get("proc_start_ticks") == before["proc_start_ticks"]
            os.kill(child.pid, signal.SIGCONT)
            child.terminate()
            child.wait(timeout=3)
        if child.stdout:
            child.stdout.close()
