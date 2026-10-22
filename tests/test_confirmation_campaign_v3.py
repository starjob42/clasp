"""Source-only confirmation guards; synthetic files and owned CPU children."""
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

import pytest

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "experiments/rloo_one_shot_20261016/run_confirmation_campaign_v3.py"


def module():
    raw = SOURCE.read_bytes();m = types.ModuleType("confirmation_controller_diagnostic")
    m.__file__ = str(SOURCE)
    import hashlib
    m.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    exec(compile(raw, str(SOURCE), "exec", dont_inherit=True), m.__dict__)
    return m


def save(m, path, value):
    path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(m.encoded(value) + b"\n")
    return m.binding(path)


def test_default_is_stdlib_only_no_write_no_gpu(tmp_path):
    result = subprocess.run([sys.executable, "-I", "-B", str(SOURCE)], cwd=tmp_path, capture_output=True, check=True, text=True)
    value = json.loads(result.stdout)
    assert value["GPU_calls"] == 0 and value["new_authority_created"] is False and list(tmp_path.iterdir()) == []
    tree = ast.parse(SOURCE.read_bytes())
    names = {n.names[0].name.split(".")[0] for n in tree.body if isinstance(n, ast.Import)}
    assert not names & {"torch", "transformers", "trl", "numpy", "datasets"}


def test_exact_actual_frozen_three_source_packet_admits_without_importing_models():
    m = module();directory = SOURCE.parent / "confirmation_execution_v3/source_extra"
    rows = {name: m.binding(directory / name) for name in m.SOURCE_NAMES}
    assert m.validate_runtime_sources(rows) == directory


def test_exact_reviewed_analyzer_admits_without_quality_read():
    m = module();row = m.binding(SOURCE.with_name("analyze_confirmation_op_v3.py"))
    assert m.validate_analysis_source(row) == row


def test_resealed_unreviewed_analyzer_cannot_read_quality(tmp_path):
    m = module();path = tmp_path / "analyze_confirmation_op_v3.py";path.write_text("# Unreviewed different body\n")
    with pytest.raises(ValueError, match="exact independently reviewed"):
        m.validate_analysis_source(m.binding(path))


@pytest.mark.parametrize("fault", ["changed_resealed_source", "extra_bytecode", "source_symlink"])
def test_unknown_or_expanded_runtime_source_packet_is_closed(tmp_path, fault):
    m = module();original = SOURCE.parent / "confirmation_execution_v3/source_extra"
    for name in m.SOURCE_NAMES:
        (tmp_path / name).write_bytes((original / name).read_bytes())
    target = tmp_path / "run_confirmation_7b_v3.py"
    if fault == "changed_resealed_source":
        target.write_bytes(target.read_bytes() + b"\n# altered source must not acquire policy authority\n")
    elif fault == "extra_bytecode":
        (tmp_path / "hidden.pyc").write_bytes(b"unreviewed bytecode")
    else:
        target.unlink();target.symlink_to(original / target.name)
    rows = {name: m.binding(tmp_path / name) for name in m.SOURCE_NAMES} if fault != "source_symlink" else {
        name: {"path": str(tmp_path / name), "sha256": m.RUNTIME_SHA[name], "size_bytes": len((original / name).read_bytes())}
        for name in m.SOURCE_NAMES}
    with pytest.raises(ValueError):
        m.validate_runtime_sources(rows)


@pytest.mark.parametrize("branch,total,slots,methods", [("pure", 102.5, 28, 5), ("teacher", 182.5, 48, 9)])
def test_complete_roster_budget_and_once_SFT(branch, total, slots, methods, tmp_path):
    m = module();rows = m.fixed_slots(branch, tmp_path)
    assert len(rows) == slots and sum(r["wall_seconds"] for r in rows) / 3600 == total == m.budget(branch)
    assert rows[2]["slot"] == "baseline.op.sft.seed101" and rows[2]["optimizer_updates"] == rows[2]["RL_responses"] == 0
    assert rows[2]["evaluation_rows"] == 256 and all(row["mode"] == "qualification" for row in rows[:2])
    assert len([r for r in rows if r["mode"] == "baseline"]) == 1
    assert len([r for r in rows if r["mode"] == "qualification"]) == 2
    study = [r for r in rows if r["mode"] == "study"]
    assert len(study) == 5 * methods and {r["seed"] for r in study} == set(m.SEEDS)
    assert all(r["evaluation_rows"] == 256 and r["optimizer_updates"] == 100 and r["maximum_attempts"] == 1 for r in study)
    for seed in m.SEEDS:
        group = [r for r in study if r["seed"] == seed]
        assert group[-1]["arm"] == ("rloo_cost" if branch == "pure" else "rloo_teacher")
        assert len({r["arm"] for r in group}) == methods


def test_worker_and_controller_slot_definitions_match_without_model_import(tmp_path):
    m = module();path = SOURCE.with_name("run_confirmation_7b_v3.py");tree = ast.parse(path.read_bytes())
    names = {"fixed_slots", "PURE", "TEACHER", "SEEDS", "MIXED"};nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "fixed_slots":
            nodes.append(node)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
    namespace = {"Path": Path, "need": m.need}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    for branch in ("pure", "teacher"):
        assert m.encoded(namespace["fixed_slots"](branch, tmp_path)) == m.encoded(m.fixed_slots(branch, tmp_path))


@pytest.mark.parametrize("mutation", [lambda rows: rows.pop(), lambda rows: rows.reverse(),
    lambda rows: rows.append(dict(rows[-1])), lambda rows: rows[-1].update(seed=73),
    lambda rows: rows[-1].update(wall_seconds=1), lambda rows: rows[-1].update(maximum_attempts=2),
    lambda rows: rows[-1].update(optimizer_updates=True), lambda rows: rows[-1].update(evaluation_rows=96)])
def test_no_missing_reordered_retried_or_changed_slots(tmp_path, mutation):
    m = module();rows = m.fixed_slots("teacher", tmp_path);mutation(rows)
    with pytest.raises(ValueError):
        m.validate_slot_roster("teacher", tmp_path, rows)


def test_qualifications_use_fixed_source_controls_and_counts(tmp_path):
    m = module()
    for branch, expected in (("pure", ["rloo_cost", "foarl_cost_changed"]), ("teacher", ["rloo_teacher", "foarl_teacher"])):
        rows = [r for r in m.fixed_slots(branch, tmp_path) if r["mode"] == "qualification"]
        assert [r["arm"] for r in rows] == expected
        assert all(r["seed"] == 101 and r["optimizer_updates"] == 1 and r["evaluation_rows"] == 2 and r["RL_responses"] == 9 for r in rows)


@pytest.mark.parametrize("passed,status,expected", [(None, None, ("wait", None)), (True, None, ("selected_development_hint", "pure")),
    (True, "development_pass_requires_fresh_confirmation", ("selected_development_hint", "pure")),
    (False, "running", ("wait", None)), (False, "development_pass_requires_fresh_confirmation", ("selected_development_hint", "teacher")),
    (False, "development_failed_requires_next_candidate", ("no_passing_development_candidate", None)),
    (False, "failed_requires_reconciliation", ("no_passing_development_candidate", None))])
def test_mutually_exclusive_fixed_selection(passed, status, expected):
    assert module().selection_decision(passed, status) == expected


@pytest.mark.parametrize("passed", [0, 1, "true", {}])
def test_selection_requires_typed_boolean(passed):
    with pytest.raises(ValueError):
        module().selection_decision(passed, None)


def hint_fixture(tmp_path, monkeypatch, passed=False):
    m = module();original_owner = {"pid": 10};teacher_owner = {"pid": 20}
    original = save(m, tmp_path / "original/owner.json", original_owner)
    teacher = save(m, tmp_path / "teacher/owner.json", teacher_owner)
    roster = {"path": "/synthetic/roster", "sha256": "a" * 64, "size_bytes": 1}
    report = save(m, tmp_path / "original/op_analysis.json", {"status": "complete_task_raw_replay", "task": "op", "roster": roster,
                   "superiority_established": False, "development_screen_passed": passed})
    save(m, tmp_path / "original/status.json", {"owner": original_owner, "analysis_results": {"op": report}, "status": "running"})
    save(m, tmp_path / "teacher/status.json", {"owner": teacher_owner, "status": "running"})
    packet = {"original_roster": roster, "teacher_controller_owner": teacher}
    master = {"original_controller_owner": original, "old_metadata": {"sha256": m.OLD_METADATA_SHA}}
    calls = []
    monkeypatch.setattr(m, "module", lambda *args: types.SimpleNamespace(admit_complete_task=lambda *args: calls.append("gate") or {"status": "complete_metadata_admission"}))
    return m, packet, master, report, calls


def test_original_metadata_gate_precedes_screen_and_is_cached(tmp_path, monkeypatch):
    m, packet, master, report, calls = hint_fixture(tmp_path, monkeypatch)
    read = m.read_bound
    def guarded(row):
        if row.get("path") == report["path"]:
            assert calls == ["gate"]
        return read(row)
    monkeypatch.setattr(m, "read_bound", guarded);cache = {}
    assert m.development_hint(packet, master, cache) == ("wait", None)
    assert m.development_hint(packet, master, cache) == ("wait", None)
    assert calls == ["gate"]


def test_incomplete_old_metadata_never_opens_quality(tmp_path, monkeypatch):
    m, packet, master, report, _ = hint_fixture(tmp_path, monkeypatch)
    Path(report["path"]).write_bytes(b"invalid JSON must stay unopened")
    monkeypatch.setattr(m, "module", lambda *args: types.SimpleNamespace(admit_complete_task=lambda *args: {"status": "incomplete"}))
    with pytest.raises(ValueError, match="All5"):
        m.development_hint(packet, master)


def test_changed_cached_screen_fails_without_reselection(tmp_path, monkeypatch):
    m, packet, master, report, _ = hint_fixture(tmp_path, monkeypatch);cache = {}
    m.development_hint(packet, master, cache)
    Path(report["path"]).write_bytes(b"changed after metadata gate")
    with pytest.raises(ValueError):
        m.development_hint(packet, master, cache)


def fake_freeze(tmp_path, monkeypatch, branch):
    m = module();root = tmp_path / "execution";root.mkdir();(root / "contracts").mkdir()
    packet = {k: {"synthetic_fixture": True, "name": k} for k in ["plan", "master_manifest", "master_packet", "original_roster", "sources", "inputs", "master_logical"]}
    packet["student_logicals"] = {str(seed): {"path": "/synthetic/student" + str(seed), "sha256": str(seed), "size_bytes": 1} for seed in m.SEEDS}
    monkeypatch.setattr(m, "registered_maxima_context", lambda *args: {"historical_requested_GPU_hours": 239.5})
    monkeypatch.setattr(m, "teacher_inputs", lambda *args: {"synthetic_fixture": True})
    entry = types.SimpleNamespace(fixed_slots=m.fixed_slots, validate_authority=lambda value: None,
        make_contract=lambda authority, slot, student: m.sealed({"protocol": m.EC_PROTOCOL, "authority": authority, "slot": slot, "student_logical": student}, "contract_sha256"))
    args = m.freeze_roster(packet, {}, entry, branch, {"synthetic_fixture": True}, root)
    return m, root, args


@pytest.mark.parametrize("branch,count", [("pure", 28), ("teacher", 48)])
def test_all_ECs_and_registration_exist_before_any_phase(tmp_path, monkeypatch, branch, count):
    m, root, (ab, rb, reg, rows) = fake_freeze(tmp_path, monkeypatch, branch)
    assert len(list((root / "contracts").glob("*.json"))) == count
    assert len(m.verify_frozen_roster(ab, rb, reg)["slots"]) == count
    assert not list(root.rglob("*.launch.json")) and not (root / "runs").exists()
    registration = json.loads(m.read_bound(reg));assert registration["confirmation_policy_calls"] == 0
    assert registration["execution_roster"] == rb


def test_changed_future_EC_blocks_first_policy_call(tmp_path, monkeypatch):
    m, root, (ab, rb, reg, rows) = fake_freeze(tmp_path, monkeypatch, "pure")
    Path(rows[-1]["contract"]["path"]).write_bytes(b"late changed future slot")
    with pytest.raises(ValueError):
        m.verify_frozen_roster(ab, rb, reg)


def state():
    return {"requested_GPU_hours_launched": 0., "maximum_new_requested_GPU_hours": 102.5,
            "prepolicy_registration": {"synthetic_fixture": True}, "completed_phases": []}


def test_no_gpu_phase_before_complete_registration(tmp_path, monkeypatch):
    m = module();s = state();s["prepolicy_registration"] = None
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: pytest.fail("Unregistered phase launched"))
    with pytest.raises(ValueError, match="registration"):
        m.child_phase("synthetic", ["never"], control=tmp_path, state=s, cap=1800, requested_GPU_hours=.5)
    assert not list(tmp_path.iterdir())


def test_failed_child_stops_own_group_and_never_retries(tmp_path, monkeypatch):
    m = module();s = state();calls = [];stopped = []
    process = types.SimpleNamespace(pid=12345, returncode=1, poll=lambda: 1)
    def popen(argv, **kw):
        calls.append((argv, kw));return process
    monkeypatch.setattr(m.subprocess, "Popen", popen);monkeypatch.setattr(m, "stop_own_group", lambda value: stopped.append(value.pid))
    with pytest.raises(ValueError, match="without retry"):
        m.child_phase("synthetic", ["owned"], control=tmp_path, state=s, cap=1800, requested_GPU_hours=.5)
    assert s["requested_GPU_hours_launched"] == .5 and stopped == [12345] and calls[0][1]["start_new_session"] is True
    with pytest.raises(FileExistsError):
        m.child_phase("synthetic", ["owned"], control=tmp_path, state=s, cap=1800, requested_GPU_hours=.5)
    assert len(calls) == 1


def test_post_spawn_heartbeat_failure_cleans_up_child(tmp_path, monkeypatch):
    m = module();s = state();calls = [];stopped = [];process = types.SimpleNamespace(pid=12345)
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(m, "stop_own_group", lambda child: stopped.append(child.pid))
    def beat(*args):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("heartbeat disk unavailable after spawn")
    monkeypatch.setattr(m, "heartbeat", beat)
    with pytest.raises(OSError, match="after spawn"):
        m.child_phase("synthetic", ["owned"], control=tmp_path, state=s, cap=1800, requested_GPU_hours=.5)
    assert stopped == [12345]


def test_CPU_selection_phase_hides_cuda_and_charges_zero(tmp_path, monkeypatch):
    m = module();s = state();seen = []
    def popen(argv, **kw):
        seen.append(kw);return types.SimpleNamespace(pid=12345, returncode=0, poll=lambda: 0)
    monkeypatch.setattr(m.subprocess, "Popen", popen)
    m.child_phase("selection", ["owned"], control=tmp_path, state=s, cap=3600, cpu=True)
    assert seen[0]["env"]["CUDA_VISIBLE_DEVICES"] == "" and s["requested_GPU_hours_launched"] == 0.
    assert s["completed_phases"] == ["selection"]


def test_absolute_deadline_rejects_phase_before_launch(tmp_path, monkeypatch):
    m = module();monkeypatch.setattr(m, "DEADLINE", datetime(2020, 1, 1, tzinfo=timezone.utc))
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: pytest.fail("Expired controller launched"))
    with pytest.raises(ValueError, match="deadline"):
        m.child_phase("expired", ["never"], control=tmp_path, state=state(), cap=1, cpu=True)


def test_own_process_group_cleanup_leaves_other_owned_fixture_alive():
    m = module();argv = [sys.executable, "-I", "-B", "-c", "import time;time.sleep(60)"]
    own = subprocess.Popen(argv, start_new_session=True)
    other = subprocess.Popen(argv, start_new_session=True)
    try:
        m.stop_own_group(own)
        assert own.poll() is not None and other.poll() is None
    finally:
        if own.poll() is None:
            os.killpg(own.pid, signal.SIGKILL);own.wait(timeout=5)
        if other.poll() is None:
            os.killpg(other.pid, signal.SIGTERM);other.wait(timeout=5)


def test_previous_controllers_must_both_finish(tmp_path, monkeypatch):
    m = module();old = save(m, tmp_path / "old.json", {"pid": 10});teacher = save(m, tmp_path / "teacher.json", {"pid": 20})
    monkeypatch.setattr(m, "controller_identity", lambda owner: {"pid": owner["pid"], "exact_alive": owner["pid"] == 10})
    closed, identities = m.previous_controllers_closed({"teacher_controller_owner": teacher}, {"original_controller_owner": old})
    assert closed is False and len(identities) == 2


def test_fixed_history_is_reserved_maxima_not_actual_spend(tmp_path, monkeypatch):
    m = module();rows = {k: save(m, tmp_path / (k + ".json"), {"synthetic_fixture": True, "scope": k})
                         for k in ["historical_native_authority", "original_roster", "master_packet"]}
    monkeypatch.setattr(m, "NATIVE_HISTORY_SHA", rows["historical_native_authority"]["sha256"])
    monkeypatch.setattr(m, "ROSTER_SHA", rows["original_roster"]["sha256"]);monkeypatch.setattr(m, "MASTER_PACKET_SHA", rows["master_packet"]["sha256"])
    for branch, combined in [("pure", 342.), ("teacher", 422.)]:
        context = m.registered_maxima_context(rows, branch)
        assert context["historical_requested_GPU_hours"] == 239.5 and context["combined_requested_GPU_hours"] == combined
        assert context["failed_and_unlaunched_reservations_reclaimed"] is False and context["actual_spent_GPU_hours"] is None


@pytest.mark.parametrize("change", [lambda value: value.update(status="failed"), lambda value: value.pop("status"),
    lambda value: value.update(slot="different"), lambda value: value.update(output="/wrong/output"),
    lambda value: value.update(contract_binding={"wrong": "contract"})])
def test_next_slot_requires_exact_passed_metadata(change):
    m = module();slot = {"slot": "synthetic.slot", "output": "/synthetic/output", "contract": {"synthetic_fixture": True}}
    value = {"status": "complete_confirmation_endpoint_metadata", "slot": slot["slot"], "output": slot["output"], "contract_binding": slot["contract"]}
    assert m.require_completed_metadata(value, slot) == value
    change(value)
    with pytest.raises(ValueError, match="passed endpoint"):
        m.require_completed_metadata(value, slot)


def qualifying_fixture(tmp_path, monkeypatch):
    m, root, (ab, rb, reg, roster) = fake_freeze(tmp_path, monkeypatch, "pure")
    control = root / "control";control.mkdir()
    for slot in roster[:2]:
        output = Path(slot["output"]);output.mkdir(parents=True)
        for name in ("completion.json", "result.json", "training_complete.json", "checkpoint_reload_exact.json"):
            save(m, output / name, {"synthetic_fixture": True, "name": name})
        save(m, control / (slot["slot"] + ".metadata_admission.json"), {"status": "complete_confirmation_endpoint_metadata",
            "slot": slot["slot"], "output": slot["output"], "contract_binding": slot["contract"]})
    return m, root, ab, rb, reg, roster, control


def test_two_phase_registration_binds_both_actual_qualification_receipts(tmp_path, monkeypatch):
    m, root, ab, rb, initial, roster, control = qualifying_fixture(tmp_path, monkeypatch)
    new = m.register_qualifications(ab, rb, initial, roster, control)
    value = m.verify_qualifications_registration(new, ab, rb, initial, roster)
    assert len(value["qualifications"]) == 2 and value["confirmation_policy_calls"] == 0
    assert value["engineering_evaluation_responses"] == 4 and value["engineering_optimizer_updates"] == 2
    assert value["initial_registration"] == initial and new != initial


def test_changed_qualification_artifact_blocks_baseline(tmp_path, monkeypatch):
    m, root, ab, rb, initial, roster, control = qualifying_fixture(tmp_path, monkeypatch)
    new = m.register_qualifications(ab, rb, initial, roster, control)
    (Path(roster[0]["output"]) / "training_complete.json").write_bytes(b"changed actual qualifier")
    with pytest.raises(ValueError):
        m.verify_qualifications_registration(new, ab, rb, initial, roster)


def test_relocated_qualification_metadata_is_rejected(tmp_path, monkeypatch):
    m, root, ab, rb, initial, roster, control = qualifying_fixture(tmp_path, monkeypatch)
    new = m.register_qualifications(ab, rb, initial, roster, control);value = json.loads(m.read_bound(new))
    original = value["qualifications"][0]["metadata_admission"]
    copy = tmp_path / "copied_metadata.json";copy.write_bytes(m.read_bound(original))
    value["qualifications"][0]["metadata_admission"] = m.binding(copy)
    value = m.sealed(value, "receipt_sha256");changed = save(m, tmp_path / "changed_registration.json", value)
    with pytest.raises(ValueError, match="original control"):
        m.verify_qualifications_registration(changed, ab, rb, initial, roster)


def test_baseline_fresh_policy_requires_second_qualification_registration(tmp_path, monkeypatch):
    m = module();s = state()
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: pytest.fail("Fresh policy launched before actual qualification"))
    with pytest.raises(ValueError, match="engineering qualifications"):
        m.child_phase("baseline.op.sft.seed101", ["never"], control=tmp_path, state=s, cap=5400,
                      requested_GPU_hours=1.5, fresh_policy=True)
    assert not list(tmp_path.iterdir())


def test_matching_owned_zombie_is_closed_without_reusing_its_pid():
    m = module();child = subprocess.Popen([sys.executable, "-I", "-B", "-c", "pass"])
    try:
        path = Path("/proc") / str(child.pid)
        ticks = (path / "stat").read_text().rsplit(")", 1)[1].split()[19]
        owner = {"pid": child.pid, "uid": os.getuid(), "hostname": m.socket.gethostname(), "start_ticks": ticks}
        deadline = time.monotonic() + 5
        while (path / "stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
            assert time.monotonic() < deadline
            time.sleep(.01)
        proof = m.controller_identity(owner)
        assert proof["state"] == "Z" and proof["exact_alive"] is False
        assert proof["closed_process_state"] is True and proof["reused_pid"] is False
    finally:
        child.wait(timeout=5)


def test_matching_running_owned_process_still_blocks():
    m = module();path = Path("/proc") / str(os.getpid())
    owner = {"pid": os.getpid(), "uid": os.getuid(), "hostname": m.socket.gethostname(),
             "start_ticks": (path / "stat").read_text().rsplit(")", 1)[1].split()[19]}
    proof = m.controller_identity(owner)
    assert proof["exact_alive"] is True and proof["closed_process_state"] is False


def test_matching_dead_X_state_is_closed_without_labeling_PID_reuse(monkeypatch):
    m = module();raw = Path(f"/proc/{os.getpid()}/stat").read_text();prefix, rest = raw.rsplit(")", 1)
    fields = rest.split();fields[0] = "X";dead_stat = prefix + ") " + " ".join(fields)
    fake_pid = types.SimpleNamespace(exists=lambda: True, stat=lambda: types.SimpleNamespace(st_uid=os.getuid()))
    class FakePid:
        exists = staticmethod(fake_pid.exists)
        stat = staticmethod(fake_pid.stat)
        def __truediv__(self, key):
            assert key == "stat"
            return types.SimpleNamespace(read_text=lambda: dead_stat)
    class FakeProc:
        def __truediv__(self, key):
            assert key == str(os.getpid())
            return FakePid()
    monkeypatch.setattr(m, "Path", lambda value: FakeProc() if value == "/proc" else Path(value))
    owner = {"pid": os.getpid(), "uid": os.getuid(), "hostname": m.socket.gethostname(), "start_ticks": fields[19]}
    proof = m.controller_identity(owner)
    assert proof["state"] == "X" and proof["exact_alive"] is False and proof["reused_pid"] is False


def analysis_fixture(tmp_path, monkeypatch, branch="pure"):
    m, root, (authority, rb, initial, roster) = fake_freeze(tmp_path, monkeypatch, branch)
    control = root / "control";control.mkdir()
    for slot in roster:
        output = Path(slot["output"]);output.mkdir(parents=True)
        result = save(m, output / "result.json", {"synthetic_fixture": True, "quality_summary": None})
        completion = save(m, output / "completion.json", {"synthetic_fixture": True, "status": "complete"})
        save(m, control / (slot["slot"] + ".metadata_admission.json"), {
            "status": "complete_confirmation_endpoint_metadata", "slot": slot["slot"], "output": slot["output"],
            "contract_binding": slot["contract"], "result_binding": result, "completion_binding": completion})
    source = tmp_path / "analyze_confirmation_op_v3.py";source.write_text("# SYNTHETIC TEST ONLY\n")
    packet = {"analysis_source": m.binding(source), "sources": {"run_confirmation_7b_v3.py": {"synthetic_fixture": True}}}
    monkeypatch.setattr(m, "ANALYZER_SHA", packet["analysis_source"]["sha256"])
    s = state();s.update(selected_branch=branch, completed_slots=[row["slot"] for row in roster], requested_GPU_hours_launched=m.budget(branch))
    calls = []
    def phase(name, argv, **kwargs):
        assert name == "independent_confirmation_analysis" and kwargs["cpu"] is True and kwargs["cap"] == 3600
        assert kwargs.get("requested_GPU_hours", 0) == 0
        request_binding = m.binding(control / "confirmation_analysis_request.json")
        request = json.loads(m.read_bound(request_binding))
        assert request["protocol"] == m.ANALYSIS_PROTOCOL and len(request["slots"]) == len(roster)
        assert [row["slot"] for row in request["slots"]] == s["completed_slots"]
        calls.append((request_binding, request, argv))
        report = m.sealed({"protocol": m.ANALYSIS_PROTOCOL + "-result", "status": "complete_confirmation_OP_replay",
            "source": packet["analysis_source"], "request_binding": request_binding, "authority_binding": authority,
            "branch": branch, "metadata_admitted": True, "registered_global_IUT_passed": False,
            "scoped_confirmation_established": False, "superiority_established": False, "publication_eligible": False,
            "model_calls": 0, "GPU_calls": 0, "synthetic_fixture": True}, "receipt_sha256")
        save(m, control / "confirmation_analysis.json", report)
    monkeypatch.setattr(m, "child_phase", phase)
    return m, packet, s, authority, rb, initial, roster, control, calls, phase


@pytest.mark.parametrize("branch", ["pure", "teacher"])
@pytest.mark.parametrize("passed", [False, True])
def test_complete_roster_runs_independent_CPU_analysis_and_retains_both_gate_outcomes(tmp_path, monkeypatch, branch, passed):
    m, packet, s, ab, rb, initial, rows, control, calls, phase = analysis_fixture(tmp_path, monkeypatch, branch)
    if passed:
        def accepted(*args, **kwargs):
            phase(*args, **kwargs)
            path = control / "confirmation_analysis.json";report = json.loads(path.read_bytes())
            report.update(registered_global_IUT_passed=True, scoped_confirmation_established=True)
            save(m, path, m.sealed(report, "receipt_sha256"))
        monkeypatch.setattr(m, "child_phase", accepted)
    report = m.analyze_complete_roster(packet, s, ab, rb, initial, rows, control)
    assert len(calls) == 1 and report["registered_global_IUT_passed"] is passed
    assert s["status"] == "completed_confirmation_scoped_gate_" + ("passed" if passed else "failed") and s["confirmation_quality_analyzed"] is True
    assert s["requested_GPU_hours_launched"] == m.budget(branch) and "superiority_established" not in s


def test_partial_roster_never_opens_metadata_or_starts_analysis(tmp_path, monkeypatch):
    m, packet, s, ab, rb, initial, rows, control, calls, phase = analysis_fixture(tmp_path, monkeypatch)
    s["completed_slots"].pop()
    monkeypatch.setattr(m, "read_bound", lambda *args: pytest.fail("Partial roster opened evidence"))
    with pytest.raises(ValueError, match="Entire fixed roster"):
        m.analyze_complete_roster(packet, s, ab, rb, initial, rows, control)
    assert calls == [] and not (control / "confirmation_analysis_request.json").exists()


def test_changed_endpoint_metadata_blocks_analysis_before_request(tmp_path, monkeypatch):
    m, packet, s, ab, rb, initial, rows, control, calls, phase = analysis_fixture(tmp_path, monkeypatch)
    path = control / (rows[-1]["slot"] + ".metadata_admission.json")
    value = json.loads(path.read_bytes());value["status"] = "failed";save(m, path, value)
    with pytest.raises(ValueError, match="passed endpoint"):
        m.analyze_complete_roster(packet, s, ab, rb, initial, rows, control)
    assert calls == [] and not (control / "confirmation_analysis_request.json").exists()


@pytest.mark.parametrize("change", [lambda r: r.update(status="incomplete"), lambda r: r.update(source={"rebound": True}),
    lambda r: r.update(request_binding={"wrong": True}), lambda r: r.update(authority_binding={"wrong": True}),
    lambda r: r.update(registered_global_IUT_passed=0), lambda r: r.update(scoped_confirmation_established=True),
    lambda r: r.update(model_calls=1), lambda r: r.update(GPU_calls=False), lambda r: r.update(publication_eligible=True)])
def test_exit_zero_or_resealed_report_does_not_bypass_independent_admission(tmp_path, monkeypatch, change):
    m, packet, s, ab, rb, initial, rows, control, calls, phase = analysis_fixture(tmp_path, monkeypatch)
    def altered(*args, **kwargs):
        phase(*args, **kwargs)
        path = control / "confirmation_analysis.json";report = json.loads(path.read_bytes());change(report)
        save(m, path, m.sealed(report, "receipt_sha256"))
    monkeypatch.setattr(m, "child_phase", altered)
    with pytest.raises(ValueError, match="exit alone"):
        m.analyze_complete_roster(packet, s, ab, rb, initial, rows, control)
    assert len(calls) == 1 and s.get("confirmation_quality_analyzed") is not True
