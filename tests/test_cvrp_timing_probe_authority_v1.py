"""Pure resource-probe guards; no actual authority, policy or scheduler call."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"


def load(name, relative):
    path = EXP / relative; raw = path.read_bytes(); module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


p = load("pure_probe_preparer", "prepare_cvrp_timing_probe_v1.py")
a = load("pure_probe_accounting", "verify_cvrp_timing_probe_accounting_v1.py")
PILOT = (EXP / "collected_native_first_complete_v1/status/campaign_authority.json").read_bytes()
FAILED = (EXP / "validation/native_monitor/prior_failed_1030929_reverification_v2_bound.json").read_bytes()


def raw(value): return (json.dumps(value, sort_keys=True) + "\n").encode()
def seal(value, key):
    value[key] = p.sha(p.encoded({k: v for k, v in value.items() if k != key})); return value
def item(path, body): return {"path": path, "size_bytes": len(body), "sha256": p.sha(body)}


def spec_fixture():
    original = (EXP / "source_v2/source_manifest.json").read_bytes()
    old = json.loads(original); contract_row = next(r for r in old["files"] if r["path"] == p.CONTRACT)
    value = {"protocol": p.PROTOCOL + "-spec", "schema_version": 1,
        "synthetic_fixture": True, "scope": "resource_measurement_only", "native_namespace": p.NAMESPACE,
        "source_root": p.SOURCE, "control_root": p.CONTROL, "output": p.CONTROL + "/output", "python": p.PYTHON,
        "base": p.BASE, "sft": p.SFT, "recipe": copy.deepcopy(p.RECIPE), "resources": copy.deepcopy(p.RESOURCES),
        "heldout_evaluation": False, "checkpoint_saved": False, "study_admission": False,
        "candidate_selection": False, "automatic_submission": False, "production_superiority_claim": False,
        "actor_discarded": True, "first_attempt_only": True,
        "original_source_manifest_binding": item("original_source2_manifest.json", original),
        "contract_binding": contract_row, "contract_semantic_sha256": p.CONTRACT_SEMANTIC_SHA,
        "source_manifest_binding": item("source_manifest.json", b"SYNTHETIC UNEXECUTED MANIFEST")}
    for field, name in (("kernel_binding", p.KERNEL), ("worker_binding", p.WORKER),
                        ("preparer_binding", p.PREPARER), ("accounting_binding", p.ACCOUNTING)):
        value[field] = item(name, b"# SYNTHETIC CLOSED SOURCE\n")
    return seal(value, "spec_sha256")


def authority_fixture():
    authority = p.make_authority_proposal("1" * 64)
    authority["synthetic_fixture"] = True
    return seal(authority, "authority_sha256")


def accounting_fixture(state="COMPLETED", seconds=100, *, allocated=True):
    authority = authority_fixture(); authority_bytes = raw(authority)
    attempt = {"protocol": p.PROTOCOL + "-attempt", "synthetic_fixture": True, "state": "submitted",
        "job_id": "9990001", "attempt_number": 1, "reservation_id": a.RESERVATION, "replacement_or_retry": False,
        "spec_sha256": "1" * 64, "authority_sha256": p.sha(authority_bytes), "source_manifest_sha256": "2" * 64,
        "script_path": a.CONTROL + "/job.slurm", "script_sha256": "3" * 64,
        "submission_argv": ["sbatch", "--parsable", a.CONTROL + "/job.slurm"],
        "step_strategy": "torchrun4_in_batch", "requested_gpus": 4, "requested_wall_seconds": 1800}
    query = {"synthetic_fixture": True, "independent_read_only_collection": True, "returncode": 0,
        "argv": ["sacct", "--noheader", "--parsable2", "--jobs", attempt["job_id"],
            "--format=JobIDRaw,State%40,ElapsedRaw,ExitCode,AllocTRES"], "stderr": ""}
    suffixes = ("", ".batch", ".extern") if allocated else ("",)
    query["stdout"] = "\n".join(f"9990001{x}|{state}|{seconds}|{'0:0' if state in ('COMPLETED','RUNNING','PENDING') else '1:0'}|"
        + ("cpu=80,gres/gpu=4,gres/gpu:a100=4" if allocated else "") for x in suffixes)
    kwargs = {"attempt_raw": raw(attempt), "expected_attempt_sha256": p.sha(raw(attempt)),
        "query_raw": raw(query), "expected_query_sha256": p.sha(raw(query)),
        "expected_spec_sha256": "1" * 64, "expected_authority_sha256": p.sha(authority_bytes),
        "expected_source_manifest_sha256": "2" * 64, "authority_raw": authority_bytes,
        "pilot_authority_raw": PILOT, "prior_failure_raw": FAILED,
        "expected_preparer_sha256": p.sha((EXP / "prepare_cvrp_timing_probe_v1.py").read_bytes()), "synthetic_fixture": True}
    return kwargs, attempt, query, authority


def test_original_actual_failed_lineage_and_all13_reservations_remain94():
    proposal = authority_fixture()
    proof = p.validate_authority(proposal, pilot_authority_raw=PILOT, prior_failure_raw=FAILED)
    assert proof["combined_reserved_requested_gpu_hours"] == 94 and proof["original_reservations_retained"] == 13
    assert proof["prior_failed_job_retained"] == "1030929" and not proof["native_submission_authorized_by_this_validator"]
    assert proposal["native_registered"] is False and proposal["status"] == "source_only_unregistered_proposal"


def test_existing20_rescue_reservations_cannot_be_reset_and_total_is114():
    extras = [{"reservation_id": "evaluation-only/" + str(i), "gpus": 1, "wall_seconds": 7200} for i in range(10)]
    rescue = seal({"synthetic_fixture": True, "original_pilot_authority_sha256": p.PILOT_AUTHORITY_SHA,
        "additional_reservations": extras, "combined_reserved_requested_gpu_hours": 112}, "authority_sha256")
    proposal = p.make_authority_proposal("1" * 64, rescue_reservations=extras, rescue_authority_sha256=p.sha(raw(rescue)))
    proof = p.validate_authority(proposal, pilot_authority_raw=PILOT, prior_failure_raw=FAILED, rescue_authority_raw=raw(rescue))
    assert proof["combined_reserved_requested_gpu_hours"] == 114 and proof["rescue_reservations_retained"] == 10
    proposal["combined_reserved_requested_gpu_hours"] = 94; seal(proposal, "authority_sha256")
    with pytest.raises(ValueError): p.validate_authority(proposal, pilot_authority_raw=PILOT, prior_failure_raw=FAILED, rescue_authority_raw=raw(rescue))


@pytest.mark.parametrize("fault", ["drop_old", "rewrite_failed", "old_bytes", "ceiling", "longer", "more_gpus", "retry", "mutation", "quality", "spec"])
def test_authority_corruptions_reject_even_after_resigning_semantic_seal(fault):
    proposal = authority_fixture(); failed = FAILED; pilot = PILOT
    if fault == "drop_old": proposal["original_reservations"].pop()
    elif fault == "rewrite_failed": failed += b" "
    elif fault == "old_bytes": pilot += b" "
    elif fault == "ceiling": proposal["declared_requested_gpu_hour_ceiling"] = 96
    elif fault == "longer": proposal["probe_reservation"]["wall_seconds"] = 3600
    elif fault == "more_gpus": proposal["probe_reservation"]["gpus"] = 8
    elif fault == "retry": proposal["automatic_retry"] = True
    elif fault == "mutation": proposal["native_campaign_mutations"] = True
    elif fault == "quality": proposal["order_depends_on_quality"] = True
    else: proposal["spec_sha256"] = "invalid"
    seal(proposal, "authority_sha256")
    with pytest.raises(ValueError): p.validate_authority(proposal, pilot_authority_raw=pilot, prior_failure_raw=failed)


@pytest.mark.parametrize("fault", ["horizon", "prefix", "responses", "beta", "lambda", "p", "k", "seed", "wall", "gpus", "qos", "cpu", "mem", "retry", "eval", "save", "study", "actor", "namespace", "base", "contract", "numeric_bool"])
def test_spec_changes_cannot_change_recipe_or_enable_study(fault):
    spec = spec_fixture()
    changes = {"horizon": ("original_horizon", 3), "prefix": ("committed_prefix_updates", 4),
        "responses": ("maximum_RL_responses", 144), "beta": ("beta", 0.), "lambda": ("cardinal_lambda", .5),
        "p": ("top_p", 1.), "k": ("top_k", 0), "seed": ("train_seed", 44)}
    if fault in changes:
        key, value = changes[fault]; spec["recipe"][key] = value
    elif fault in ("wall", "gpus", "qos", "cpu", "mem", "retry", "numeric_bool"):
        key, value = {"wall": ("wall_seconds", 3600), "gpus": ("gpus", 8), "qos": ("qos", "qos_gpu_a100-dev"),
            "cpu": ("cpus_per_task", 8), "mem": ("memory_mb", 30000), "retry": ("automatic_retry", True),
            "numeric_bool": ("maximum_attempts", True)}[fault]; spec["resources"][key] = value
    elif fault in ("eval", "save", "study"):
        spec[{"eval": "heldout_evaluation", "save": "checkpoint_saved", "study": "study_admission"}[fault]] = True
    elif fault == "actor": spec["actor_discarded"] = False
    elif fault == "namespace": spec["native_namespace"] += "other"
    elif fault == "base": spec["base"] = "/other/model"
    else: spec["contract_binding"]["sha256"] = "0" * 64
    seal(spec, "spec_sha256")
    with pytest.raises(ValueError): p.validate_spec(spec)


@pytest.mark.parametrize("state,seconds,complete", [("COMPLETED", 100, True), ("FAILED", 77, True), ("TIMEOUT", 1800, True), ("RUNNING", 33, False)])
def test_raw_actual_cost_kept_for_failure_and_running_lowerbound(state, seconds, complete):
    kwargs, _, _, _ = accounting_fixture(state, seconds)
    report = a.validate_accounting(**kwargs)
    assert report["current_probe_allocated_gpu_seconds"] == seconds * 4
    assert report["allocation_accounting_complete"] is complete and report["actual_cost_is_lower_bound"] is not complete
    assert report["combined_reserved_requested_gpu_hours"] == 94
    assert not report["study_admitted"] and report["cumulative_allocated_gpu_seconds"] is None
    assert not report["native_allocation_accounting_certified"]


def test_cancelled_before_allocation_retains_reservation_with_zero_actual_cost():
    kwargs, _, _, _ = accounting_fixture("CANCELLED", 0, allocated=False)
    report = a.validate_accounting(**kwargs)
    assert report["allocation_accounting_complete"] and report["current_probe_allocated_gpu_seconds"] == 0
    assert report["combined_reserved_requested_gpu_hours"] == 94


def test_failure_termination_grace_retained_without_raising_requested_reservation():
    kwargs, _, _, _ = accounting_fixture("TIMEOUT", 1825)
    with pytest.raises(ValueError): a.validate_accounting(**kwargs)
    grace = {"protocol": p.PROTOCOL + "-termination-grace", "synthetic_fixture": True,
        "job_id": "9990001", "query_sha256": kwargs["expected_query_sha256"], "independent_read_only_collection": True,
        "maximum_overrun_seconds": 30}
    report = a.validate_accounting(**kwargs, termination_grace_raw=raw(grace), expected_grace_sha256=p.sha(raw(grace)))
    assert report["current_probe_allocated_gpu_seconds"] == 7300 and report["combined_reserved_requested_gpu_hours"] == 94


@pytest.mark.parametrize("fault", ["missing_parent", "drop_step", "duplicate", "unrelated", "gpus", "conflict_gpu", "open_step", "exit", "truncated", "argv", "retcode", "retry", "spec", "authority", "source", "attempt_bytes", "query_bytes", "helper", "unknown_state", "grace_success", "native_fixture"])
def test_forged_ambiguous_or_incomplete_accounting_fails_closed(fault):
    kwargs, attempt, query, authority = accounting_fixture()
    if fault == "missing_parent": query["stdout"] = "\n".join(query["stdout"].splitlines()[1:])
    elif fault == "drop_step": query["stdout"] = query["stdout"].splitlines()[0]
    elif fault == "duplicate": query["stdout"] += "\n" + query["stdout"].splitlines()[0]
    elif fault == "unrelated": query["stdout"] += "\n9999|COMPLETED|10|0:0|gres/gpu=4"
    elif fault == "gpus": query["stdout"] = query["stdout"].replace("gres/gpu=4", "gres/gpu=8").replace("gpu:a100=4", "gpu:a100=8")
    elif fault == "conflict_gpu": query["stdout"] = query["stdout"].replace("gpu:a100=4", "gpu:a100=8")
    elif fault == "open_step": query["stdout"] = query["stdout"].replace("9990001.batch|COMPLETED", "9990001.batch|RUNNING")
    elif fault == "exit": query["stdout"] = query["stdout"].replace("0:0", "1:0")
    elif fault == "truncated": query["stdout"] = "9990001|COMPLETED|100"
    elif fault == "argv": query["argv"][4] = "9999"
    elif fault == "retcode": query["returncode"] = 1
    elif fault == "retry": attempt["attempt_number"] = 2
    elif fault in ("spec", "authority", "source"):
        attempt[{"spec": "spec_sha256", "authority": "authority_sha256", "source": "source_manifest_sha256"}[fault]] = "0" * 64
    elif fault == "helper": kwargs["expected_preparer_sha256"] = "0" * 64
    elif fault == "unknown_state": query["stdout"] = query["stdout"].replace("COMPLETED", "UNKNOWN")
    elif fault == "grace_success": query["stdout"] = query["stdout"].replace("|100|", "|1810|")
    elif fault == "native_fixture": kwargs["synthetic_fixture"] = False
    kwargs["attempt_raw"] = raw(attempt); kwargs["expected_attempt_sha256"] = p.sha(raw(attempt))
    kwargs["query_raw"] = raw(query); kwargs["expected_query_sha256"] = p.sha(raw(query))
    if fault == "attempt_bytes": kwargs["attempt_raw"] += b" "
    if fault == "query_bytes": kwargs["query_raw"] += b" "
    with pytest.raises(ValueError): a.validate_accounting(**kwargs)


def test_actual_original121_source_closure_plus_new_unexecuted_metadata(tmp_path):
    source = EXP / "source_v2"; root = tmp_path / "source"; root.mkdir()
    original_raw = (source / "source_manifest.json").read_bytes(); original = json.loads(original_raw)
    rows = copy.deepcopy(original["files"])
    for row in rows:
        path = root / row["path"]; path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source / row["path"], path)
    spec = spec_fixture()
    original_row = item("original_source2_manifest.json", original_raw)
    (root / original_row["path"]).write_bytes(original_raw); rows.append(original_row)
    for field in ("kernel_binding", "worker_binding", "preparer_binding", "accounting_binding"):
        row = spec[field]; path = root / row["path"]; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# SYNTHETIC CLOSED SOURCE\n"); rows.append(row)
    manifest_raw = raw({"protocol": "synthetic_source_only_fixture", "files": rows})
    (root / "source_manifest.json").write_bytes(manifest_raw); spec["source_manifest_binding"] = item("source_manifest.json", manifest_raw)
    seal(spec, "spec_sha256"); spec_path = tmp_path / "spec.json"; spec_path.write_bytes(raw(spec))
    assert p.validated_spec(spec_path, p.sha(raw(spec)), source_root=root) == spec
    extra = root / "unbound.py"; extra.write_text("raise RuntimeError('must never execute')")
    with pytest.raises(ValueError, match="Unbound"): p.validated_spec(spec_path, p.sha(raw(spec)), source_root=root)
    extra.unlink(); (root / p.TRAINER).write_text("altered original")
    with pytest.raises(ValueError): p.validated_spec(spec_path, p.sha(raw(spec)), source_root=root)
