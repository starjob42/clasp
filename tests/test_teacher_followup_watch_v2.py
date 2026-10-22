"""Pure metadata/filesystem guards; no model, GPU or live controller calls."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

SOURCE = Path(__file__).parents[1] / "experiments/rloo_one_shot_20261016/watch_teacher_followup_v2.py"
spec = importlib.util.spec_from_file_location("followup_watch_under_test", SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(mod.encoded(value) + b"\n")


def fixture(tmp_path, monkeypatch, phase=None):
    root = tmp_path / "execution"
    owner = {"pid": 12, "start_ticks": "100", "uid": mod.UID}
    launch = {"pid": 12, "start_ticks": "100", "argv_compact_SHA256": "a" * 64}
    code = tmp_path / "source.py";code.write_bytes(b"# no execution\n")
    row = {"path": str(code), "sha256": mod.sha(code.read_bytes()), "size_bytes": code.stat().st_size}
    packet = {"output_root": str(root), "controller": row, "extra_sources": {}}
    pp = tmp_path / "packet.json";write(pp, packet)
    monkeypatch.setattr(mod, "PACKET", pp);monkeypatch.setattr(mod, "PACKET_SHA", mod.sha(pp.read_bytes()))
    state = {"owner": owner, "status": "running" if phase else "waiting_for_original_complete_OP_screen",
             "active_phase": phase, "completed_phases": [], "requested_GPU_hours_launched": 0.,
             "heartbeat_utc": mod.now().isoformat()}
    if phase:
        state["active_parent_pid"] = 13
        write(root / "control" / (phase + ".launch.json"), {"phase": phase, "automatic_retry": False, "argv": ["owned-child"]})
    write(root / "control/status.json", state)
    monkeypatch.setattr(mod, "process_identity", lambda pid, **kw: {"pid": pid, "identity_eligible": True, "start_ticks": "100", "ppid": 12})
    monkeypatch.setattr(mod, "owned_descendants", lambda owner: [{"pid": 14, "identity_eligible": True}])
    monkeypatch.setattr(mod, "gpu_counters", lambda owned: {"owned": sorted(owned)})
    return (packet, launch, owner, {}), root


def test_default_invocation_does_no_process_or_gpu_calls(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(SOURCE)])
    monkeypatch.setattr(mod, "load_target", lambda: pytest.fail("No live target read during dryrun"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("No GPU or process call during dryrun"))
    mod.main()
    assert json.loads(capsys.readouterr().out)["GPU_calls"] == 0


@pytest.mark.parametrize("phase", sorted(mod.PHASES))
def test_phase_output_only_declared_fixed_paths(tmp_path, phase):
    output = mod.phase_output(tmp_path, phase)
    assert output is None or output.is_relative_to(tmp_path)


@pytest.mark.parametrize("phase", ["../unrelated", "study.cvrp.foarl_teacher.seed73", "study.op.unknown.seed73"])
def test_unregistered_phase_cannot_broaden_read_scope(tmp_path, phase):
    with pytest.raises(ValueError, match="Unregistered"):
        mod.phase_output(tmp_path, phase)


def test_window_counter_counts_bytes_without_json_or_quality_read(tmp_path):
    p = tmp_path / "actual_prompt_windows.jsonl"
    p.write_bytes(b"deliberately not JSON\ncoefficient=do-not-decode\n")
    result = mod.window_count(p)
    assert result["count"] == 2 and result["stable_snapshot"] is True
    with pytest.raises(ValueError, match="Only training"):
        mod.window_count(tmp_path / "raw_completions.jsonl")


def test_artifact_stat_never_opens_answer_or_analysis_file(tmp_path, monkeypatch):
    p = tmp_path / "raw_completions.jsonl";p.write_bytes(b"answer hidden")
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("stat-only file opened"))
    assert mod.file_stat(p)["size_bytes"] == 13


def test_linked_window_and_metadata_are_rejected(tmp_path):
    target = tmp_path / "data";target.write_bytes(b"{}\n")
    alias = tmp_path / "actual_prompt_windows.jsonl";alias.symlink_to(target)
    for operation in (mod.window_count, mod.file_stat, mod.stable_bytes):
        with pytest.raises(ValueError):
            operation(alias)


def test_waiting_snapshot_launches_zero_gpu_and_reads_no_quality(tmp_path, monkeypatch):
    target, root = fixture(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail("No endpoint may open while waiting")
    monkeypatch.setattr(mod, "window_count", forbidden)
    monkeypatch.setattr(mod, "file_stat", forbidden)
    result = mod.snapshot(target, {})
    assert result["controller_state"]["requested_GPU_hours_launched"] == 0.
    assert result["GPU_counters"]["owned"] == []
    assert result["quality_files_opened"] is False


@pytest.mark.parametrize("arm", ["foarl_teacher", "teacher_only"])
def test_active_snapshot_only_stats_raw_and_counts_actual_window_paths(tmp_path, monkeypatch, arm):
    phase = "study.op." + arm + ".seed73"
    target, root = fixture(tmp_path, monkeypatch, phase)
    output = root / "runs" / phase
    name = "teacher_only_CE_windows.jsonl" if arm == "teacher_only" else "streamed_teacher_CE_windows.jsonl"
    window = output / ("" if arm == "teacher_only" else "training") / name
    window.parent.mkdir(parents=True, exist_ok=True);window.write_bytes(b"not decoded\n" * 7)
    (output / "raw_completions.jsonl").write_bytes(b"must remain unopened")
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name != "raw_completions.jsonl"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    result = mod.snapshot(target, {})
    assert result["training_window_counts"][name]["count"] == 7
    assert result["GPU_counters"]["owned"] == [13, 14]


def test_changed_status_owner_denied_before_gpu_or_endpoint_read(tmp_path, monkeypatch):
    target, root = fixture(tmp_path, monkeypatch)
    state = json.loads((root / "control/status.json").read_bytes());state["owner"]["pid"] = 99
    write(root / "control/status.json", state)
    monkeypatch.setattr(mod, "gpu_counters", lambda owned: pytest.fail("Changed target reached GPU query"))
    with pytest.raises(ValueError, match="owner"):
        mod.snapshot(target, {})


def test_source_drift_reported_without_execution(tmp_path, monkeypatch):
    target, root = fixture(tmp_path, monkeypatch)
    Path(target[0]["controller"]["path"]).write_bytes(b"changed harmless bytes")
    assert mod.snapshot(target, {})["source_buffers_exact"] is False


def test_only_owned_gpu_pids_are_recorded(monkeypatch):
    replies = iter([
        types.SimpleNamespace(returncode=0, stderr="", stdout=mod.GPU + ", 100, 49140, 30\n"),
        types.SimpleNamespace(returncode=0, stderr="", stdout=mod.GPU + ", 22, 80\n" + mod.GPU + ", 987654, 20\nother-GPU, 55, 50\n")])
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: next(replies))
    result = mod.gpu_counters({22})
    assert result["owned_phase_processes"] == [{"pid": 22, "used_memory_MiB": 80}]
    assert result["other_process_count"] == 1 and "987654" not in mod.encoded(result).decode()


def test_unadmitted_parent_cannot_scan_children():
    with pytest.raises(ValueError, match="admitted parent"):
        mod.owned_descendants({"pid": 1, "identity_eligible": False})


@pytest.mark.parametrize("state", sorted(mod.TERMINAL))
def test_terminal_states_stop_watcher_without_signaling_controller(state):
    assert mod.terminal({"controller_state": {"status": state}})
    assert not mod.terminal({"controller_state": {"status": "waiting_for_original_complete_OP_screen"}})


def test_fixed_poll_and_lifetime_bound():
    assert mod.PERIOD == 30
    assert mod.DEADLINE.isoformat() == "2026-10-20T12:30:00+00:00"
