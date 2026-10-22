"""Pure synthetic custody guards; never scheduler/model/quality evidence."""
import base64
import copy
from datetime import timedelta
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments/rloo_sft_anchor_20261013/resource_extension_v1/collect_resource_campaign_v1.py"
spec = importlib.util.spec_from_file_location("resource_extension_collection_guards", PATH)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def raw_query(state="COMPLETED", seconds=120, allocation="generic", code="0:0", sft=False):
    job = "4321"
    start = "2026-10-16T13:00:00"
    end = (c.scheduler_utc(start) + timedelta(seconds=seconds)).replace(tzinfo=None).isoformat()
    n, cpu, requested_cpu, mem = (1, 20, 10, "75000M") if sft else (4, 80, 40, "300000M")
    gpu = f"gres/gpu={n}" if allocation == "generic" else f"gres/gpu:a100={n}"
    if allocation == "both":gpu = f"gres/gpu={n},gres/gpu:a100={n}"
    requested = f"cpu={requested_cpu},gres/gpu={n},mem={mem},node=1"
    allocated = f"cpu={cpu},{gpu},mem={mem},node=1"
    lines = [f"{job}{suffix}|{state}|{code}|{seconds}|{start}|{end}|{requested if not suffix else ''}|{allocated}\n"
             for suffix in ("", ".batch", ".extern")]
    return {"argv": ["sacct", "--jobs", job, "--noheader", "--parsable2", "--format=" + c.ACCOUNTING_FORMAT],
            "returncode": 0, "stderr": "", "stdout": "".join(lines),
            "recorded_utc": "2026-10-16T18:00:00+00:00", "time_zone": "UTC", "synthetic_fixture": True}


def allocation(query, sft=False):
    rows, _ = c.decode_accounting(query, ["4321"])
    return c.allocation_receipt("4321", c.resource_spec("cvrp.sft" if sft else c.CVRP_ORDER[0]), rows)


def record(local="resource_extension/fixture.json", native=None, raw=b"{}"):
    return {"local_path": local, "native_path": native or c.EXTENSION + "/fixture.json",
            "snapshot_stable": True, "tail_only": False,
            "sha256": c.digest(raw), "size_bytes": len(raw),
            "content_base64": base64.b64encode(raw).decode(), "synthetic_fixture": True}


def test_six_resource_slots_add_exact81_to_retained96():
    rows = [c.resource_spec(slot) for slot in c.CVRP_ORDER]
    assert sum(r["effective_requested_gpu_hours"] for r in rows) == 123
    assert sum(r["original_reserved_gpu_hours"] for r in rows) == 42
    assert sum(r["additional_requested_gpu_hours"] for r in rows) == 81
    assert 96 + sum(r["additional_requested_gpu_hours"] for r in rows) == 177
    with pytest.raises(ValueError):c.resource_spec("op.foarl")


@pytest.mark.parametrize("descriptor", ["generic", "typed", "both"])
def test_failed_actual_fourGPU_cost_is_retained_without_double_counting(descriptor):
    a = allocation(raw_query("FAILED", allocation=descriptor, code="1:0"))
    assert a["actual_allocated_GPU_seconds"] == 480
    assert a["allocation_accounting_complete"] is True
    assert a["all_parent_steps_completed_zero_exit"] is False
    assert a["effective_requested_gpu_hours"] == 24
    assert a["original_reserved_gpu_hours"] == 8
    assert a["additional_requested_gpu_hours"] == 16


def test_SFT_actual_oneGPU_geometry_and_3h_reservation():
    a = allocation(raw_query(sft=True), sft=True)
    assert a["actual_allocated_GPU_seconds"] == 120
    assert a["all_parent_steps_completed_zero_exit"] is True
    assert a["effective_requested_gpu_hours"] == 3


@pytest.mark.parametrize("fault", ["duplicate", "typed_conflict", "empty_batch", "no_gpu", "wrongCPU", "wrongMemory", "missing_extern"])
def test_ambiguous_or_wrong_allocations_never_become_zero(fault):
    q = raw_query("FAILED", code="1:0")
    rows, _ = c.decode_accounting(q, ["4321"])
    if fault == "duplicate":rows["4321"]["allocated_tres_raw"] += ",gres/gpu=4"
    elif fault == "typed_conflict":rows["4321"]["allocated_tres_raw"] += ",gres/gpu:a100=3"
    elif fault == "empty_batch":rows["4321.batch"]["allocated_tres_raw"] = ""
    elif fault == "no_gpu":rows["4321"]["allocated_tres_raw"] = "cpu=80,mem=300000M,node=1"
    elif fault == "wrongCPU":rows["4321.batch"]["allocated_tres_raw"] = rows["4321.batch"]["allocated_tres_raw"].replace("cpu=80", "cpu=40")
    elif fault == "wrongMemory":rows["4321"]["requested_tres_raw"] = rows["4321"]["requested_tres_raw"].replace("300000M", "150000M")
    else:del rows["4321.extern"]
    with pytest.raises(ValueError):c.allocation_receipt("4321", c.resource_spec(c.CVRP_ORDER[0]), rows)


@pytest.mark.parametrize("fault", ["foreign_job", "foreign_step", "missing_parent", "truncated", "bad_exit", "bad_argv", "bad_rc", "naiveUTC"])
def test_raw_query_custody_rejects_unrelated_unknown_or_self_asserted_records(fault):
    q = raw_query()
    if fault == "foreign_job":q["stdout"] += q["stdout"].splitlines()[0].replace("4321|", "9999|") + "\n"
    elif fault == "foreign_step":q["stdout"] += q["stdout"].splitlines()[0].replace("4321|", "4321.0|") + "\n"
    elif fault == "missing_parent":q["stdout"] = "\n".join(q["stdout"].splitlines()[1:])
    elif fault == "truncated":q["stdout"] = q["stdout"].replace("COMPLETED", "COMPLETED+")
    elif fault == "bad_exit":q["stdout"] = q["stdout"].replace("0:0|", "0|")
    elif fault == "bad_argv":q["argv"][0] = "scontrol"
    elif fault == "bad_rc":q["returncode"] = False
    else:q["recorded_utc"] = "2026-10-16T18:00:00"
    with pytest.raises(ValueError):c.decode_accounting(q, ["4321"])


def test_actual_start_plus_reserved_wall_must_fit_original_deadline():
    q = raw_query()
    q["stdout"] = q["stdout"].replace("2026-10-16T13:00:00", "2026-10-17T07:27:00").replace("2026-10-16T13:02:00", "2026-10-17T07:29:00")
    with pytest.raises(ValueError, match="unchanged execution deadline"):allocation(q)


def test_failure_overrun_cost_preserved_and_grace_still_required():
    a = allocation(raw_query("TIMEOUT", seconds=21601, code="0:15"))
    assert a["actual_allocated_GPU_seconds"] == 86404
    assert a["termination_grace_custody_needed"] is True
    assert a["all_parent_steps_completed_zero_exit"] is False


def test_completed_overrun_cannot_silently_gain_extra_time():
    with pytest.raises(ValueError, match="independent grace custody"):allocation(raw_query(seconds=21601))


def test_pending_empty_snapshot_is_explicit_lower_bound_not_complete_cost():
    q = raw_query("PENDING", seconds=0)
    q["stdout"] = "4321|PENDING|0:0|0|Unknown|Unknown|cpu=40,gres/gpu=4,mem=300000M,node=1|\n"
    a = allocation(q)
    assert a["actual_allocated_GPU_seconds"] == 0 and a["actual_cost_is_lower_bound"] is True
    assert a["allocation_accounting_complete"] is False


def test_real_zero_requires_empty_allocation_zero_elapsed():
    q = raw_query("CANCELLED", seconds=0)
    q["stdout"] = "4321|CANCELLED|0:0|0|Unknown|Unknown||\n"
    a = allocation(q)
    assert a["actual_allocated_GPU_seconds"] == 0 and a["allocation_accounting_complete"] is True
    q["stdout"] = q["stdout"].replace("|0|Unknown", "|120|Unknown")
    with pytest.raises(ValueError, match="truly empty zero-elapsed"):allocation(q)


@pytest.mark.parametrize("fault", ["raw", "sizebool", "tail", "unstable", "copiednative", "dotdot", "duplicate"])
def test_full_raw_record_identity_scope_and_stability_are_mandatory(fault):
    r = record();rows = [r]
    if fault == "raw":r["content_base64"] = base64.b64encode(b"{ }").decode()
    elif fault == "sizebool":r["size_bytes"] = True
    elif fault == "tail":r["tail_only"] = True
    elif fault == "unstable":r["snapshot_stable"] = False
    elif fault == "copiednative":r["native_path"] = c.NAMESPACE + "/copy/fixture.json"
    elif fault == "dotdot":r["local_path"] = "resource_extension/../fixture.json"
    else:rows.append(copy.deepcopy(r))
    with pytest.raises(ValueError):c.decode_records(rows)


def test_incomplete_has_no_quality_summary_candidate_or_paper_claim():
    a = c.incomplete_receipt([c.CVRP_ORDER[0]], [])
    assert a["quality_summary"] is None and a["candidate_selection"] is None
    assert a["eligible_for_paper"] is False and a["campaign_report_admitted"] is False


def test_frozen_kernels_and_preparer_compile_exact_verified_raw_buffers(tmp_path):
    frozen = ROOT / c.KERNEL_RELATIVE
    (tmp_path/frozen.name).write_bytes(frozen.read_bytes())
    prep = frozen.with_name("prepare_admission_patch_v3.py")
    copied = tmp_path/prep.name;copied.write_bytes(prep.read_bytes())
    k = c.load_kernels(tmp_path/frozen.name)
    copied.write_text("raise AssertionError('changed dependency must never execute')\n")
    dependency = k.patch_preparer()
    assert dependency.__executed_source_sha256__ == c.KERNEL_PREPARER_SHA256
    assert k.__executed_source_sha256__ == c.KERNEL_SHA256
    assert k.__dependency_factory_override__ == "same_verified_raw_buffer_only"
    with pytest.raises(ValueError, match="dependency changed"):c.load_kernels(tmp_path/frozen.name)


@pytest.fixture(scope="module")
def complete_fixture():
    from resource_extension_collection_fixture_v1 import fixture_bundle
    return fixture_bundle()


def replace_json(bundle, local, change):
    row = next(r for r in bundle["records"] if r["local_path"] == local)
    value = __import__("json").loads(base64.b64decode(row["content_base64"]))
    change(value)
    raw = c.encoded(value)
    row.update(content_base64=base64.b64encode(raw).decode(),sha256=c.digest(raw),size_bytes=len(raw))
    for folder in c.CHAIN_FOLDERS:
        for member in bundle["directory_census"]["folders"][folder]:
            if member["native_path"] == row["native_path"]:
                member.update(sha256=row["sha256"],size_bytes=row["size_bytes"])


def partial_fixture(complete_fixture):
    op, extension, kwargs = copy.deepcopy(complete_fixture)
    keep = c.CVRP_ORDER[0]
    extension["records"] = [r for r in extension["records"] if not r["local_path"].startswith("results/") and
        not (len(r["local_path"].split("/")) == 3 and r["local_path"].split("/")[1] in c.CHAIN_FOLDERS and
             r["local_path"].split("/")[2] != keep+".json")]
    for folder in c.CHAIN_FOLDERS:
        extension["directory_census"]["folders"][folder] = [row for row in extension["directory_census"]["folders"][folder]
                                                           if row["native_path"].endswith('/'+keep+'.json')]
    query=extension["extension_accounting_query"]
    query["argv"][2]="900001"
    query["stdout"]="\n".join(line for line in query["stdout"].splitlines() if line.split('|')[0].split('.')[0]=='900001')+'\n'
    extension["terminal_artifact_proofs"]={}
    return op,extension,kwargs


def test_full_synthetic12_uses_exact_original_raw_and_durable_kernels_never_claims_paper(complete_fixture):
    op,extension,kwargs=complete_fixture
    admission,_=c.validate_campaign(op,extension,**kwargs)
    assert admission["status"]=="validated_completed_resource_extended_campaign"
    assert len(admission["completed_slots"])==12 and admission["raw_results_replayed_exactly"] is True
    assert admission["scientific_admission_kernel_sha256"]==c.KERNEL_SHA256
    assert admission["eligible_for_paper"] is False and admission["external_publication_registry_adopted"] is False
    assert admission["quality_summary"] is None and admission["candidate_selection"] is None
    report=admission["resource_campaign_report"]
    assert report["cumulative_requested_GPU_hours"]==177 and report["original96_reserved_GPU_hours"]==96
    assert report["additional_reserved_GPU_hours"]==81 and report["old_campaign_report_fabricated"] is False
    assert report["historical_cost"]["historical_actual_GPU_seconds"]==4988
    assert report["synthetic_fixture"] is True and report["allocation_accounting_complete"] is True


def test_incomplete_does_not_call_original_scientific_evaluator_or_return_quality(complete_fixture,monkeypatch):
    op,extension,kwargs=partial_fixture(complete_fixture)
    kernels=c.load_kernels()
    kernels.validate_bundle=lambda *a,**k:pytest.fail('Incomplete must never evaluate quality')
    monkeypatch.setattr(c,'load_kernels',lambda:kernels)
    admission,_=c.validate_campaign(op,extension,**kwargs)
    assert admission["status"]=='status_only_resource_campaign_incomplete'
    assert admission["quality_values_replayed"] is False and admission["quality_summary"] is None
    assert admission["candidate_selection"] is None and admission["actual_cost_is_lower_bound"] is True


def test_incomplete_refuses_eager_raw_quality_collection(complete_fixture):
    op,extension,kwargs=partial_fixture(complete_fixture)
    row=next(r for r in complete_fixture[1]['records'] if r['local_path']==f'results/{c.CVRP_ORDER[0]}/result.json')
    extension['records'].append(copy.deepcopy(row))
    with pytest.raises(ValueError,match='Incomplete extension must not collect'):c.validate_campaign(op,extension,**kwargs)


def test_explicit_synthetic_fixture_cannot_use_live_path_even_if_hashes_are_real(complete_fixture):
    op,extension,kwargs=complete_fixture
    with pytest.raises(ValueError,match='Synthetic collection'):c.validate_campaign(op,extension,**{**kwargs,'diagnostic_fixture':False})


@pytest.mark.parametrize('folder,field,value',[
 ('attempts','resources',{'gpus':4,'cpus':40,'wall_seconds':7200,'effective_requested_gpu_hours':8,'original_reserved_gpu_hours':8,'additional_requested_gpu_hours':0}),
 ('intents','automatic_retry',True),('intents','authority_sha256','0'*64),
 ('submission_actions','returncode',False),('submission_actions','stdout','900999\n'),
 ('assigned_jobs','submission_action_sha256','0'*64),('held_readbacks','fields',{}),
 ('activations','attempt_sha256','0'*64),('release_actions','returncode',1),
 ('release_readbacks','release_action_sha256','0'*64),('startup_proofs','uid',True),
 ('startup_proofs','operator_sha256','0'*64),('startup_proofs','initializer_inventory_sha256','0'*64),
 ('startup_proofs','deadline_check_before_model',False),('startup_proofs','actual_host','other-host'),
 ('startup_proofs','recorded_utc','2026-10-17T12:00:00+00:00')])
def test_actual_control_chain_fields_cannot_be_rebound_or_relabelled(complete_fixture,folder,field,value):
    op,extension,kwargs=partial_fixture(complete_fixture)
    replace_json(extension,f'resource_extension/{folder}/{c.CVRP_ORDER[0]}.json',lambda row:row.update({field:value}))
    with pytest.raises(ValueError):c.validate_campaign(op,extension,**kwargs)


@pytest.mark.parametrize('fault',['omit','stale','copiedroot','extra'])
def test_fresh_independent_directory_census_rejects_omissions_and_stale_custody(complete_fixture,fault):
    op,extension,kwargs=partial_fixture(complete_fixture);census=extension['directory_census']
    if fault=='omit':census['folders']['submission_actions']=[]
    elif fault=='stale':census['recorded_utc']='2026-10-15T20:00:00+00:00'
    elif fault=='copiedroot':census['native_root']=c.NAMESPACE+'/copy'
    else:census['folders']['attempts'].append({'native_path':c.EXTENSION+'/attempts/unknown.json','sha256':'0'*64,'size_bytes':0})
    with pytest.raises(ValueError):c.validate_campaign(op,extension,**kwargs)


def test_orphan_durable_intent_retains_unknown_allocation_and_never_retries(complete_fixture):
    op,extension,kwargs=partial_fixture(complete_fixture);slot=c.CVRP_ORDER[1]
    row=copy.deepcopy(next(r for r in complete_fixture[1]['records'] if r['local_path']==f'resource_extension/intents/{slot}.json'))
    extension['records'].append(row)
    extension['directory_census']['folders']['intents'].append({k:row[k] for k in ('native_path','sha256','size_bytes')})
    extension['directory_census']['folders']['intents'].sort(key=lambda r:r['native_path'])
    admission,_=c.validate_campaign(op,extension,**kwargs)
    assert admission['unresolved_extension_slots']==[slot] and admission['possible_unassigned_allocations'] is True
    assert admission['actual_cost_is_lower_bound'] is True and admission['allocation_accounting_complete'] is False
    assert admission['candidate_selection'] is None


def test_failed_terminal_cost_retained_without_opening_quality_results(complete_fixture):
    op,extension,kwargs=partial_fixture(complete_fixture)
    q=extension['extension_accounting_query'];q['stdout']=q['stdout'].replace('|COMPLETED|0:0|','|FAILED|1:0|')
    admission,_=c.validate_campaign(op,extension,**kwargs)
    rows=[row for row in admission['allocation_receipts'] if row['job_id']=='900001']
    assert rows[0]['actual_allocated_GPU_seconds']==480 and rows[0]['allocation_accounting_complete'] is True
    assert rows[0]['all_parent_steps_completed_zero_exit'] is False and admission['quality_values_replayed'] is False


def test_last_slot_token_mutation_fails_exact_raw_replay_or_unscored_projection(complete_fixture):
    op,extension,kwargs=copy.deepcopy(complete_fixture);slot=c.CVRP_ORDER[-1]
    replace_json(extension,f'results/{slot}/result.json',lambda row:row['raw_completions'][0].update(completion_token_ids=[999]))
    with pytest.raises(ValueError,match='raw-result replay|raw projection'):c.validate_campaign(op,extension,**kwargs)


def test_original_native_OP_attempt_bytes_cannot_be_rewritten(complete_fixture):
    op,extension,kwargs=partial_fixture(complete_fixture)
    row=next(r for r in op['records'] if r['local_path']=='attempts/'+c.OP_ORDER[0]+'.json')
    raw=base64.b64decode(row['content_base64'])+b' '
    row.update(sha256=c.digest(raw),size_bytes=len(raw),content_base64=base64.b64encode(raw).decode())
    with pytest.raises(ValueError):c.validate_campaign(op,extension,**kwargs)


def test_remote_reader_compiles_and_contains_no_scheduler_mutations_or_model_imports():
    compile(c.REMOTE_READER,'<exact-readonly-reader>','exec')
    assert "['sbatch'" not in c.REMOTE_READER and "['scontrol','release'" not in c.REMOTE_READER
    assert 'import torch' not in c.REMOTE_READER and 'import transformers' not in c.REMOTE_READER
    assert c.REMOTE_READER.index('if all_complete:') < c.REMOTE_READER.index("for name in ('result.json'")


def test_action_cli_requires_same_executed_collector_buffer(tmp_path, monkeypatch):
    path=tmp_path/'collector.py';raw=b'original executed collector';path.write_bytes(raw)
    with pytest.raises(ValueError,match='bootstrap identity'):
        c.validate_execution_identity(path)
    pin=c.digest(raw)
    assert c.validate_execution_identity(path,pin)==pin
    path.write_bytes(b'replacement disk collector')
    with pytest.raises(ValueError,match='Executed collector source'):
        c.validate_execution_identity(path,pin)


def test_source_receipt_cannot_late_rebind_executed_module_to_changed_disk(tmp_path, monkeypatch):
    path=tmp_path/'collector.py';path.write_bytes(b'new disk source')
    monkeypatch.setattr(c,'__executed_source_sha256__',c.digest(b'old loaded source'),raising=False)
    with pytest.raises(ValueError,match='Executed collector source'):
        c.validate_execution_identity(path)
