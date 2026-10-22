"""Pure prospective clock/source guards; no scheduler, model or quality calls."""
from datetime import timedelta
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "experiments/rloo_sft_anchor_20261013/resource_deadline_extension_v2/timing_addendum_v2.py"
OLD = ROOT / "experiments/rloo_sft_anchor_20261013/resource_extension_v1/cvrp_wall_resource_v1.py"


def load(path):
    raw = path.read_bytes();m = types.ModuleType("source_only_clock_fixture")
    m.__file__ = str(path);m.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    exec(compile(raw, str(path), "exec", dont_inherit=True), m.__dict__)
    return m


def test_default_is_inert_and_has_no_effect_api(tmp_path):
    r = subprocess.run([sys.executable, "-I", "-B", str(SOURCE)], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(r.stdout)["native_execution_authorized"] is False and not list(tmp_path.iterdir())
    tree = ast.parse(SOURCE.read_bytes())
    imports = {n.names[0].name.split(".")[0] for n in tree.body if isinstance(n, ast.Import)}
    assert not imports & {"torch", "transformers", "trl", "subprocess", "os", "socket"}


def test_default_proposal_preserves_all177_and_only_remaining_five():
    m = load(SOURCE);p = m.make_proposal()
    assert p["remaining_slots"] == list(m.ORDER[7:]) and len(p["remaining_slots"]) == 5
    assert p["cumulative_requested_gpu_hours"] == 177 and p["additional_requested_gpu_hours"] == 0
    assert sum(p["resources"][slot]["effective_requested_gpu_hours"] for slot in m.REMAINING) == 99
    assert sum(p["resources"][slot]["wall_seconds"] for slot in m.REMAINING) == 27*3600
    assert p["first_submitted_job"]["latest_full_wall_start_utc"] == "2026-10-17T05:26:42.646651+00:00"
    assert p["native_execution_authorized"] is p["scientific_admission_granted"] is False


def test_frozen_original_operator_authority_resources_scripts_are_unchanged():
    m = load(SOURCE);assert hashlib.sha256(OLD.read_bytes()).hexdigest() == m.ORIGINAL_OPERATOR_SHA
    old = load(OLD)
    path = ROOT / "experiments/rloo_one_shot_20261016/validation/confirmation_campaign_v3/historical_inputs_v1/historical_native_authority.json"
    raw = path.read_bytes();assert hashlib.sha256(raw).hexdigest() == m.ORIGINAL_AUTHORITY_SHA
    authority = json.loads(raw);assert old.validate_authority(authority) == authority
    for slot in m.ORDER[6:]:
        assert m.resource_spec(slot) == old.resources(slot) and m.SCRIPT_SHA[slot] == old.SCRIPT_SHA[slot]
    assert m.OLD_DEADLINE == old.EXECUTION_DEADLINE and m.ABSOLUTE_DEADLINE == old.ABSOLUTE_DEADLINE


@pytest.mark.parametrize("deadline", ["2026-10-17T11:26:42.646651+00:00", "2026-10-17T11:26:42.646650+00:00",
    "2026-10-28T09:09:58.603382+00:00", "2026-10-27T12:00:00", "2026-10-27T14:00:00+02:00", "2026-10-27T12:00:00Z", True])
def test_only_explicit_forward_execution_clock_within_absolute_ceiling(deadline):
    with pytest.raises(ValueError): load(SOURCE).make_proposal(deadline)


def test_absolute_ceiling_itself_is_not_reset():
    m = load(SOURCE);assert m.make_proposal(m.ABSOLUTE_DEADLINE)["absolute_deadline_utc"] == m.ABSOLUTE_DEADLINE


@pytest.mark.parametrize("change", [lambda v:v.update(additional_requested_gpu_hours=1), lambda v:v.update(additional_requested_gpu_hours=False),
    lambda v:v.update(cumulative_requested_gpu_hours=178), lambda v:v.update(maximum_active_study_jobs=True),
    lambda v:v.update(automatic_retry=True), lambda v:v.update(first_job_reinterpreted=True),
    lambda v:v.update(native_execution_authorized=True), lambda v:v["remaining_slots"].pop(),
    lambda v:v["remaining_slots"].insert(0,"cvrp.rloo_cardinal_beta005"),
    lambda v:v["original_identities"]["operator"].update(sha256="a"*64),
    lambda v:v["resources"]["cvrp.sft"].update(wall_seconds=21600),
    lambda v:v["first_submitted_job"].update(cancel_or_retry_allowed=True)])
def test_resealed_budget_scope_source_or_retry_changes_are_denied(change):
    m = load(SOURCE);p = m.make_proposal();change(p)
    with pytest.raises(ValueError):m.validate_addendum(m.seal(p))


def fixture(slot=None, start=None):
    """Invented timing metadata only, never full execution/quality admission."""
    m = load(SOURCE);slot = m.FIRST_SLOT if slot is None else slot;legacy = slot == m.FIRST_SLOT
    start = ("2026-10-17T05:26:42" if legacy else "2026-10-18T12:00:00") if start is None else start
    when = m.scheduler_stamp(start);job = m.FIRST_JOB if legacy else "9999999"
    source = m.ORIGINAL_OPERATOR_SHA if legacy else "d"*64;registration = m.ORIGINAL_REGISTRATION_SHA if legacy else "e"*64
    q = {"argv":["scontrol","show","job",job],"returncode":0,"stderr":"","time_zone":"UTC",
         "recorded_utc":(when+timedelta(seconds=1)).isoformat(),"stdout":"JobId="+job+" JobState=RUNNING StartTime="+start+
         " TimeLimit="+("03:00:00" if slot=="cvrp.sft" else "06:00:00")}
    startup = {"query":dict(q),"slot":slot,"job_id":job,"recorded_utc":(when+timedelta(seconds=2)).isoformat(),
        "operator_sha256":source,"registration_sha256":registration,"execution_deadline_utc":m.OLD_DEADLINE if legacy else m.PROPOSED_DEADLINE,
        "deadline_check_before_model":True,"model_calls_by_guard":0,"resources":m.resource_spec(slot),
        "original_script_sha256":m.SCRIPT_SHA[slot],"attempt_sha256":m.FIRST_ATTEMPT_SHA if legacy else "f"*64}
    kwargs = {"query":q,"startup":startup,"effective_operator_sha256":source,"effective_registration_sha256":registration,
        "first_job_admitted_utc":None if legacy else "2026-10-17T12:00:00+00:00"}
    return m,m.make_proposal(),slot,job,kwargs


@pytest.mark.parametrize("slot", [None,"cvrp.rloo_scalar_beta0","cvrp.foarl","cvrp.sft","cvrp.rloo_scalar_beta005","cvrp.rloo_cardinal_beta0"])
def test_dual_clock_positive_predicate_never_grants_other_admission(slot):
    m,p,slot,job,kw = fixture(slot);r = m.timestamp_bridge(p,slot,job,**kw)
    assert r["timestamp_predicate_passed"] is True and r["retained_requested_gpu_hours"] == 177
    assert r["additional_requested_gpu_hours"] == 0 and r["quality_summary"] is None
    assert r["native_execution_authorized"] is r["whole_attempt_resource_chain_admitted"] is r["scientific_admission_granted"] is False


def test_first_job_late_start_cannot_be_rescued_by_new_clock():
    m,p,slot,job,kw = fixture(start="2026-10-17T05:26:43")
    with pytest.raises(ValueError,match="complete effective wall"):m.timestamp_bridge(p,slot,job,**kw)


@pytest.mark.parametrize("change", [lambda k:k.update(effective_operator_sha256="a"*64), lambda k:k.update(effective_registration_sha256="a"*64),
    lambda k:k["startup"].update(attempt_sha256="a"*64), lambda k:k["startup"].update(execution_deadline_utc="2026-10-27T12:00:00+00:00")])
def test_original_first_job_never_adopts_new_source_registration_attempt_or_clock(change):
    m,p,slot,job,kw = fixture();change(kw)
    with pytest.raises(ValueError):m.timestamp_bridge(p,slot,job,**kw)


@pytest.mark.parametrize("change", [lambda k:k.update(first_job_admitted_utc=None),lambda k:k.update(first_job_admitted_utc="2026-10-19T12:00:00+00:00"),
    lambda k:k.update(effective_operator_sha256=load(SOURCE).ORIGINAL_OPERATOR_SHA),
    lambda k:k.update(effective_registration_sha256=load(SOURCE).ORIGINAL_REGISTRATION_SHA)])
def test_remaining_slots_require_distinct_source_and_earlier_first_admission(change):
    m,p,slot,job,kw = fixture("cvrp.foarl");change(kw)
    with pytest.raises(ValueError):m.timestamp_bridge(p,slot,job,**kw)


@pytest.mark.parametrize("change", [lambda k:k["query"].update(time_zone="Europe/Paris"), lambda k:k["query"].update(returncode=False),
    lambda k:k["query"].update(stderr="warning"), lambda k:k["query"].update(argv=["scontrol","show","job","1"]),
    lambda k:k["startup"].update(model_calls_by_guard=False), lambda k:k["startup"].update(original_script_sha256="a"*64),
    lambda k:k["startup"]["resources"].update(gpus=1), lambda k:k["startup"].update(query={}),
    lambda k:k["query"].update(stdout=k["query"]["stdout"]+" JobId=9999999")])
def test_raw_query_proof_identity_or_resource_corruption_is_denied(change):
    m,p,slot,job,kw = fixture("cvrp.foarl");change(kw)
    with pytest.raises(ValueError):m.timestamp_bridge(p,slot,job,**kw)


def test_new_job_full_wall_must_fit_new_deadline_too():
    m,p,slot,job,kw = fixture("cvrp.foarl",start="2026-10-27T07:00:00")
    with pytest.raises(ValueError,match="complete effective wall"):m.timestamp_bridge(p,slot,job,**kw)
