"""Pure diagnostics for closed failure custody; no scheduler/model actions."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"
PATH = EXP / "cvrp_probe_rejected_submission_history_v3.py"
raw = PATH.read_bytes()
m = types.ModuleType("_exact_rejection_validator_fixture")
m.__file__ = str(PATH)
m.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
exec(compile(raw, str(PATH), "exec", dont_inherit=True), m.__dict__)
FAILURE = (EXP / "validation/native_monitor/cvrp_first_probe_preassignment_failure_20261014T1742_bound_v1.json").read_bytes()
NO_JOB = (EXP / "validation/native_monitor/cvrp_first_probe_no_assigned_job_20261014T1743_bound_v1.json").read_bytes()
f = json.loads(FAILURE)
n = json.loads(NO_JOB)
decoded = {k: base64.b64decode(v["content_base64"]) for k, v in f["records"].items()}
action = json.loads(decoded["control/submission_action.json"])
intent = json.loads(decoded["control/submission_intent.json"])
manifest = json.loads(decoded["source_manifest"])


def test_actual_closed_receipts_retain_all_costs_without_creating_retry_authority():
    proof = m.validate_rejected_history(FAILURE, NO_JOB)
    assert proof["old_reserved_gpu_hours"] == 94
    assert proof["original_reserved_gpu_hours"] == 92
    assert proof["rejected_reservation_gpu_hours"] == 2
    assert proof["actual_rejected_gpu_hours"] == proof["accepted_jobs"] == proof["accepted_allocations"] == 0
    assert proof["new_scope_reserved_gpu_hours"] == 96
    assert not proof["source_raw_byte_replay"]
    assert not proof["new_scope_authority_or_retry_authorized"] and not proof["study_admission"]
    assert proof["receipt_sha256"] == m.sha(m.encoded({k: v for k, v in proof.items() if k != "receipt_sha256"}))


@pytest.mark.parametrize("fault", ["failure_changed", "accounting_changed", "synthetic_replacement", "mutable_buffer"])
def test_unregistered_receipt_bytes_cannot_admit_history(fault):
    a, b = FAILURE, NO_JOB
    if fault == "failure_changed": a += b" "
    elif fault == "accounting_changed": b += b" "
    elif fault == "synthetic_replacement": a = b'{"synthetic_fixture":true,"no_assigned_job":true}'
    else: a = bytearray(a)
    with pytest.raises(ValueError): m.validate_rejected_history(a, b)


@pytest.mark.parametrize("field,value", [("returncode", 0), ("returncode", True), ("stdout", "123456\n"),
    ("stderr", "submission failed\n"), ("automatic_retry", True), ("submitted_with_hold_directive", False),
    ("intent_sha256", "0" * 64), ("execution_revision", 2.0)])
def test_rpc_semantics_reject_success_ambiguous_error_or_typed_alias(field, value):
    changed = copy.deepcopy(action); changed[field] = value
    with pytest.raises(ValueError): m._closed_action(changed, intent)


@pytest.mark.parametrize("field,value", [("returncode", False), ("returncode", 1), ("stdout", "123|cvrp-first3-timing-v2|COMPLETED|1|0:0||gres/gpu=4\n"),
    ("stderr", "accounting unavailable"), ("argv", ["sacct"]), ("recorded_utc", "2026-10-14T17:30:00+00:00")])
def test_actual_zero_cost_requires_successful_post_rpc_exact_query(field, value):
    changed = copy.deepcopy(n); changed[field] = value
    with pytest.raises(ValueError): m._scheduler_zero(f, changed, action)


def test_queue_evidence_rejects_any_matching_timing_job_even_when_summary_says_none():
    changed = copy.deepcopy(f)
    queue = next(row for row in changed["queries"] if row["argv"] == m.SQUEUE_QUERY)
    queue["stdout"] += "123456|cvrp-first3-timing-v2|PENDING|JobHeldUser|2026-10-14T17:40:00|N/A\n"
    with pytest.raises(ValueError): m._scheduler_zero(changed, n, action)


@pytest.mark.parametrize("fault", ["raw_corruption", "path_alias", "bool_size", "missing_record", "extra_record"])
def test_bound_native_raw_records_are_not_replaced_by_producer_claims(fault):
    changed = copy.deepcopy(f); row = changed["records"]["control/job.slurm"]
    if fault == "raw_corruption": row["content_base64"] = base64.b64encode(b"changed").decode()
    elif fault == "path_alias": row["native_path"] = m.CONTROL + "/../control_v1/job.slurm"
    elif fault == "bool_size": row["size_bytes"] = True
    elif fault == "missing_record": del changed["records"]["source_manifest"]
    else: changed["records"]["control/attempt.json"] = row
    with pytest.raises(ValueError): m._decode_records(changed)


@pytest.mark.parametrize("fault", ["extra_attempt", "extra_output", "source_hash_drift", "bool_source_size", "policy_calls", "retry"])
def test_whole_native_directory_and_source_inventory_custody(fault):
    changed = copy.deepcopy(f)
    if fault == "extra_attempt": changed["actual_control_regular_files"].append("attempt.json")
    elif fault == "extra_output": changed["actual_control_regular_files"].append("output/result.json")
    elif fault == "source_hash_drift": changed["source_inventory_verified"][0]["sha256"] = "0" * 64
    elif fault == "bool_source_size": changed["source_inventory_verified"][0]["size_bytes"] = True
    elif fault == "policy_calls": changed["new_policy_calls"] = True
    else: changed["no_retry_performed"] = False
    with pytest.raises(ValueError): m._inventory(changed, manifest)


class MemoryReader:
    """Explicitly offline fixture; byte replay does not register native custody."""
    def __init__(self):
        packet = EXP / "cvrp_probe_execution_v2/native_prepared_v2"
        self.files = {m.SOURCE + "/" + p.relative_to(packet / "source_v1").as_posix(): p.read_bytes()
            for p in (packet / "source_v1").rglob("*") if p.is_file()}
        self.files.update({m.CONTROL + "/" + p.name: p.read_bytes()
            for p in (packet / "control_v1").iterdir() if p.is_file()})
        self.files.update({v["native_path"]: decoded[k] for k, v in f["records"].items()})
    def read(self, path): return self.files[path]
    def index(self, root): return sorted(path[len(root) + 1:] for path in self.files if path.startswith(root + "/"))


def test_optional_full_raw_reader_replays_131_source_and10_control_bytes():
    proof = m.validate_rejected_history(FAILURE, NO_JOB, reader=MemoryReader())
    assert proof["source_raw_byte_replay"] and proof["independent_reader_files_replayed"] == 141
    assert not proof["new_scope_authority_or_retry_authorized"]


@pytest.mark.parametrize("fault", ["source_mutation", "extra_cache", "extra_attempt", "control_mutation"])
def test_optional_reader_cannot_omit_or_mutate_historical_files(fault):
    reader = MemoryReader()
    if fault == "source_mutation": reader.files[m.SOURCE + "/" + manifest["files"][0]["path"]] += b"changed"
    elif fault == "extra_cache": reader.files[m.SOURCE + "/__pycache__/unbound.pyc"] = b"cache"
    elif fault == "extra_attempt": reader.files[m.CONTROL + "/attempt.json"] = b"assigned"
    else: reader.files[m.CONTROL + "/pilot_authority.json"] += b"changed"
    with pytest.raises(ValueError): m.validate_rejected_history(FAILURE, NO_JOB, reader=reader)


def test_duplicate_nonfinite_json_and_noncanonical_paths_rejected():
    for value in [b'{"x":0,"x":1}', b'{"x":NaN}']:
        with pytest.raises(ValueError): m._json(value)
    for value in ["../outside", "/absolute", "a//b", "./a"]:
        with pytest.raises(ValueError): m._relative(value)


def test_source_is_pure_and_does_not_import_scheduler_model_or_signing_libraries():
    tree = ast.parse(raw)
    imports = {row.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for row in node.names}
    assert imports <= {"base64", "hashlib", "json", "re"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
        node.func.attr in {"write_text", "write_bytes", "mkdir", "unlink", "run", "Popen", "sign"} for node in ast.walk(tree))
