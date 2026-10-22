"""No scheduler/process calls: isolated peer guards for the standby wait path."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / (
    "experiments/rloo_sft_anchor_20261013/resource_deadline_extension_v2/standby_timing_controller_v2.py")


@pytest.fixture
def fixture(tmp_path):
    raw = SOURCE.read_bytes()
    m = ModuleType("peer_standby_wait_only")
    m.__file__ = str(SOURCE)
    m.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    exec(compile(raw, str(SOURCE), "exec", dont_inherit=True), m.__dict__)
    m.ROOT = tmp_path
    (tmp_path / "handoff").mkdir()
    spec = {"fixture": "no native authority"}
    expected = {"pid": 1630144, "uid": 304433, "hostname": "jean-zay1", "proc_start_ticks": "22050281"}
    payload = {"spec_sha256": m.sha(m.encoded(spec)), "old_owner": expected,
               "signals": ["SIGSTOP", "SIGTERM", "SIGCONT"]}
    calls = []
    def verify(context):
        calls.append("verify_inputs")
    def load(context):
        calls.append("load_handoff")
        path = tmp_path / "handoff/complete.json"
        b = path.read_bytes()
        return {"path": str(path), "sha256": m.sha(b), "size_bytes": len(b)}
    helper = SimpleNamespace(OWNER=expected, process_snapshot=lambda pid: {"pid": pid, "alive": False})
    return SimpleNamespace(m=m, path=tmp_path / "handoff/complete.json", spec=spec,
        expected=expected, payload=payload, calls=calls, helper=helper,
        operator=SimpleNamespace(verify_inputs=verify, load_handoff=load))


def emit_complete(f, payload=None):
    f.path.write_text(json.dumps(f.payload if payload is None else payload))


def run(f, **kwargs):
    return f.m.wait_for_handoff({}, f.operator, f.helper, f.spec,
        clock=lambda: datetime(2026, 10, 16, tzinfo=timezone.utc), **kwargs)


def test_wait_does_not_enter_operator_and_polls_only_until_exact_handoff(fixture):
    f = fixture
    emissions = []
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        emit_complete(f)
    row = run(f, sleep=sleep, emit=emissions.append)
    assert row["path"] == str(f.path)
    assert sleeps == [10]
    assert emissions == [{"status": "waiting_for_exact_cpu_handoff", "scheduler_mutations": [], "model_calls": 0}]
    assert f.calls == ["verify_inputs", "verify_inputs", "load_handoff", "verify_inputs"]
    assert not any("run" in x or "submit" in x for x in f.calls)


def test_failure_journal_denies_even_if_complete_also_present(fixture):
    f = fixture
    emit_complete(f)
    (f.path.parent / "requires_investigation.json").write_text("{}")
    with pytest.raises(ValueError, match="investigation"):
        run(f)
    assert f.calls == ["verify_inputs"]


def test_old_clock_bounds_wait_and_does_not_infer_success(fixture):
    f = fixture
    with pytest.raises(ValueError, match="waiting deadline"):
        f.m.wait_for_handoff({}, f.operator, f.helper, f.spec,
            clock=lambda: datetime.fromisoformat(f.m.WAIT_DEADLINE))
    assert f.calls == []


@pytest.mark.parametrize("change", [
    {"spec_sha256": "0" * 64}, {"old_owner": {}},
    {"signals": ["SIGTERM", "SIGCONT"]}, {"signals": ["SIGSTOP", "SIGKILL"]},
])
def test_handoff_exact_external_spec_owner_and_signal_order_required(fixture, change):
    f = fixture
    emit_complete(f, {**f.payload, **change})
    with pytest.raises(ValueError, match="external spec"):
        run(f)


@pytest.mark.parametrize("state", ["S", "R", "T"])
def test_live_old_cpu_owner_never_enters_new_operator(fixture, state):
    f = fixture
    emit_complete(f)
    f.helper.process_snapshot = lambda pid: {**f.expected, "alive": True, "state": state}
    with pytest.raises(ValueError, match="still live"):
        run(f)


@pytest.mark.parametrize("field,value", [
    ("pid", 1630145), ("uid", 0), ("hostname", "another-host"),
    ("proc_start_ticks", "reused"),
])
def test_reused_terminal_pid_cannot_satisfy_handoff(fixture, field, value):
    f = fixture
    emit_complete(f)
    f.helper.process_snapshot = lambda pid: {**f.expected, "alive": True, "state": "Z", field: value}
    with pytest.raises(ValueError, match="PID was reused"):
        run(f)


def test_same_identity_zombie_is_terminal_but_not_relabelled_as_new_process(fixture):
    f = fixture
    emit_complete(f)
    f.helper.process_snapshot = lambda pid: {**f.expected, "alive": True, "state": "Z"}
    assert run(f)["path"] == str(f.path)
    assert f.calls[-1] == "verify_inputs"


def test_bound_handoff_bytes_are_checked_before_any_activation(fixture):
    f = fixture
    emit_complete(f)
    f.operator.load_handoff = lambda context: {"path": str(f.path), "sha256": "0" * 64,
                                              "size_bytes": f.path.stat().st_size}
    with pytest.raises(ValueError, match="changed"):
        run(f)


def test_final_input_drift_does_not_return_success(fixture):
    f = fixture
    emit_complete(f)
    count = 0
    def verify(context):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("source drift after handoff")
    f.operator.verify_inputs = verify
    with pytest.raises(ValueError, match="source drift"):
        run(f)


def test_cpu_helpers_execute_only_original_verified_buffer(fixture):
    f = fixture
    raw = b"value = 7\n"
    row = {"path": "/tmp/peer_immutable_buffer.py", "sha256": f.m.sha(raw), "size_bytes": len(raw)}
    loaded = f.m.compile_buffer(row, raw, "diagnostic_captured_source")
    assert loaded.value == 7 and loaded.__executed_source_sha256__ == row["sha256"]
    with pytest.raises(ValueError, match="captured verified source"):
        f.m.compile_buffer(row, b"value = 8\n", "rebound_source")
