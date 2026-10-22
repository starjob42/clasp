"""Pure additive history closure; all diagnostic submissions are synthetic.

Real old121 source/old92 authority/failed1030929 evidence is read-only. No new
cohort, authority, registered round, policy output or scheduler call is made.
"""
import ast
import copy
import json
from pathlib import Path
import types

import pytest
from test_cvrp_timing_probe_authority_v1 import p, a, PILOT, FAILED, EXP, raw, seal, item, spec_fixture, load

h = load("probe_history_unit", "cvrp_timing_probe_history_v1.py")
b = load("probe_history_bridge_unit", "confirmation_probe_history_bridge_v2.py")


class Evidence:
    def __init__(self): self.files = {}
    def put(self, path, value):
        body = value if type(value) is bytes else raw(value)
        self.files[path] = body
        return item(path, body)
    def __call__(self, binding): return self.files[binding["path"]]


def fixture(*, submitted=True, state="COMPLETED", rescue=False, omit_embedded_manifest=False):
    reader = Evidence(); original_raw = (EXP / "source_v2/source_manifest.json").read_bytes()
    original = json.loads(original_raw); rows = copy.deepcopy(original["files"])
    files = {row["path"]: reader.put(p.SOURCE + "/" + row["path"], (EXP / "source_v2" / row["path"]).read_bytes()) for row in rows}
    old_binding = reader.put(p.SOURCE + "/original_source2_manifest.json", original_raw)
    rows.append(item("original_source2_manifest.json", original_raw)); files["original_source2_manifest.json"] = old_binding
    spec = spec_fixture()
    for field, relative in (("kernel_binding", p.KERNEL), ("worker_binding", p.WORKER),
                            ("preparer_binding", p.PREPARER), ("accounting_binding", p.ACCOUNTING)):
        body = (EXP / Path(relative).name).read_bytes() if field in ("preparer_binding", "accounting_binding") else b"# SYNTHETIC UNEXECUTED KERNEL/WORKER\n"
        row = item(relative, body); spec[field] = row; rows.append(row)
        files[relative] = reader.put(p.SOURCE + "/" + relative, body)
    if omit_embedded_manifest:
        rows=[row for row in rows if row["path"]!="original_source2_manifest.json"]
        files.pop("original_source2_manifest.json")
    source_raw = raw({"protocol": "synthetic_source_only_fixture", "synthetic_fixture": True, "files": rows})
    source_binding = reader.put(p.SOURCE + "/source_manifest.json", source_raw)
    spec["source_manifest_binding"] = item("source_manifest.json", source_raw); seal(spec, "spec_sha256")
    spec_binding = reader.put(p.CONTROL + "/spec.json", spec)
    rescue_binding = None; base = 92
    if rescue:
        extras = [{"reservation_id": "evaluation-only/" + str(i), "gpus": 1,
                   "wall_seconds": 7200, "requested_wall_seconds": 7200} for i in range(10)]
        rescue_value = seal({"synthetic_fixture": True, "original_pilot_authority_sha256": p.PILOT_AUTHORITY_SHA,
            "original_pilot_authority_binding": item("/synthetic/actual-old92.json",PILOT),
            "additional_reservations": extras, "additional_requested_gpu_hours":20,
            "combined_reserved_requested_gpu_hours": 112}, "authority_sha256")
        rescue_binding = reader.put("/synthetic/existing-rescue.json", rescue_value)
        authority = p.make_authority_proposal(spec["spec_sha256"], rescue_reservations=extras, rescue_authority_sha256=rescue_binding["sha256"])
        base = 112
    else: authority = p.make_authority_proposal(spec["spec_sha256"])
    authority["synthetic_fixture"] = True; seal(authority, "authority_sha256")
    authority_binding = reader.put(p.CONTROL + "/authority.json", authority)
    directory = {"protocol": h.PROTOCOL + "-attempt-directory-inventory", "synthetic_fixture": True,
        "native_control_root": p.CONTROL, "independent_read_only_collection": True, "unresolved_intents": [],
        "attempt_files": [], "intent_files": [], "action_files": []}
    listing = {"protocol": h.PROTOCOL + "-source-directory-inventory", "synthetic_fixture": True,
        "native_source_root": p.SOURCE, "independent_read_only_collection": True,
        "regular_files": sorted([row["path"] for row in rows] + ["source_manifest.json"]),
        "symbolic_links": [], "nonregular_files": []}
    record = {"protocol": h.PROTOCOL + "-probe-custody", "synthetic_fixture": True, "performance_admission": False,
        "native_namespace": p.NAMESPACE, "native_source_root": p.SOURCE, "native_control_root": p.CONTROL,
        "reservation_id": a.RESERVATION, "independent_read_only_collection": True,
        "spec_binding": spec_binding, "authority_binding": authority_binding, "source_manifest_binding": source_binding,
        "original_source_manifest_binding": old_binding, "source_file_bindings": files,
        "pilot_authority_binding": reader.put("/synthetic/actual-old92.json", PILOT),
        "prior_failed_proof_binding": reader.put("/synthetic/actual-failed1030929.json", FAILED),
        "rescue_authority_binding": rescue_binding, "state": "closed_unsubmitted_reservation",
        "submission_rpc_started": False, "attempt_binding": None, "query_binding": None,
        "intent_binding": None, "action_binding": None}
    if submitted:
        script = b"# SYNTHETIC UNEXECUTED SBATCH\n"; script_binding = reader.put(p.CONTROL + "/job.slurm", script)
        intent = {"protocol": p.PROTOCOL + "-intent", "synthetic_fixture": True, "reservation_id": a.RESERVATION,
            "spec_sha256": spec["spec_sha256"], "authority_sha256": authority_binding["sha256"],
            "source_manifest_sha256": source_binding["sha256"], "script_sha256": p.sha(script), "automatic_retry": False}
        intent_binding = reader.put(p.CONTROL + "/submission_intent.json", intent)
        attempt = {"protocol": p.PROTOCOL + "-attempt", "synthetic_fixture": True, "state": "submitted", "job_id": "9990001",
            "attempt_number": 1, "reservation_id": a.RESERVATION, "replacement_or_retry": False,
            "spec_sha256": spec["spec_sha256"], "authority_sha256": authority_binding["sha256"],
            "source_manifest_sha256": source_binding["sha256"], "script_path": p.CONTROL + "/job.slurm",
            "script_sha256": p.sha(script), "submission_intent_sha256": intent_binding["sha256"],
            "submission_argv": ["sbatch", "--parsable", p.CONTROL + "/job.slurm"],
            "step_strategy": "torchrun4_in_batch", "requested_gpus": 4, "requested_wall_seconds": 1800}
        action = {"protocol": p.PROTOCOL + "-submission-action", "synthetic_fixture": True,
            "intent_sha256": intent_binding["sha256"], "argv": attempt["submission_argv"],
            "returncode": 0, "stderr": "", "stdout": "9990001\n"}
        query = {"synthetic_fixture": True, "independent_read_only_collection": True, "returncode": 0, "stderr": "",
            "argv": ["sacct", "--noheader", "--parsable2", "--jobs", "9990001", "--format=JobIDRaw,State%40,ElapsedRaw,ExitCode,AllocTRES"],
            "stdout": "\n".join(f"9990001{s}|{state}|100|{'0:0' if state in ('COMPLETED','RUNNING') else '1:0'}|gres/gpu=4,gres/gpu:a100=4" for s in ("", ".batch", ".extern"))}
        directory.update(attempt_files=["attempt.json"], intent_files=["submission_intent.json"], action_files=["submission_action.json"])
        record.update(state="closed_submitted_attempt", submission_rpc_started=True, script_binding=script_binding,
            intent_binding=intent_binding, attempt_binding=reader.put(p.CONTROL + "/attempt.json", attempt),
            action_binding=reader.put(p.CONTROL + "/submission_action.json", action), query_binding=reader.put("/synthetic/query.json", query))
    record["source_directory_inventory_binding"] = reader.put("/synthetic/source-listing.json", listing)
    record["attempt_directory_inventory_binding"] = reader.put("/synthetic/submission-listing.json", directory)
    seal(record, "receipt_sha256")
    return record, reader, base


@pytest.fixture(scope="module")
def closed(): return fixture()


def verify(record, reader, base=92, *, synthetic=True):
    return h.verify_probe_history(record, reader, historical_base_gpu_hours=base,
        expected_preparer_sha256=p.sha((EXP / "prepare_cvrp_timing_probe_v1.py").read_bytes()),
        expected_accounting_sha256=p.sha((EXP / "verify_cvrp_timing_probe_accounting_v1.py").read_bytes()), synthetic_fixture=synthetic)


def alter(record, reader, field, change):
    value = json.loads(reader(record[field])); change(value)
    record[field] = reader.put(record[field]["path"], value)


@pytest.mark.parametrize("submitted,state,base", [(False,"COMPLETED",92),(True,"COMPLETED",92),(True,"FAILED",92),(True,"TIMEOUT",92),(True,"COMPLETED",112)])
def test_reservation_always_adds2_and_preserves_failure_or_unsubmitted_actual_cost(submitted,state,base):
    record, reader, observed_base = fixture(submitted=submitted,state=state,rescue=base==112)
    report = verify(record,reader,observed_base)
    assert report["historical_pilot_reserved_gpu_hours"] == base+2
    assert report["diagnostic_reserved_gpu_hours"] == 2 and report["diagnostic_allocated_gpu_seconds"] == (400 if submitted else 0)
    assert report["study_admission"] is False and report["native_mutations"] is False and report["synthetic_fixture"] is True


@pytest.mark.parametrize("fault", ["old92", "failed", "spec", "source_omission", "source_mutation", "source_copy", "source_extra", "source_cache", "source_link", "manifest_copy",
    "attempt_retry", "attempt_spec", "attempt_source", "attempt_authority", "intent", "action_job", "action_error", "action_argv", "script", "listing", "ambiguous", "query", "job_step", "running", "namespace", "base_reset", "native_fixture"])
def test_cost_only_or_mutated_submission_source_lineage_never_closes_history(closed,fault):
    record, original_reader, base = closed; record=copy.deepcopy(record); reader=Evidence(); reader.files=dict(original_reader.files)
    if fault == "old92": alter(record,reader,"authority_binding",lambda v:(v["original_reservations"].pop(),seal(v,"authority_sha256")))
    elif fault == "failed": record["prior_failed_proof_binding"]=reader.put("/synthetic/wrong-failure",FAILED+b" ")
    elif fault == "spec": alter(record,reader,"spec_binding",lambda v:(v["recipe"].update(original_horizon=3),seal(v,"spec_sha256")))
    elif fault == "source_omission": record["source_file_bindings"].pop(p.TRAINER)
    elif fault == "source_mutation": record["source_file_bindings"][p.TRAINER]=reader.put(p.SOURCE+"/"+p.TRAINER,b"different trainer")
    elif fault == "source_copy": record["source_file_bindings"][p.TRAINER]=reader.put("/copied/"+p.TRAINER,reader(record["source_file_bindings"][p.TRAINER]))
    elif fault in ("source_extra","source_cache","source_link"):
        alter(record,reader,"source_directory_inventory_binding",lambda v:v["symbolic_links" if fault=="source_link" else "regular_files"].append("unbound.pyc" if fault=="source_cache" else "unbound.py"))
    elif fault == "manifest_copy": record["source_manifest_binding"]=reader.put("/copied/manifest",reader(record["source_manifest_binding"]))
    elif fault.startswith("attempt_"):
        key,value={"attempt_retry":("attempt_number",2),"attempt_spec":("spec_sha256","0"*64),"attempt_source":("source_manifest_sha256","0"*64),"attempt_authority":("authority_sha256","0"*64)}[fault]
        alter(record,reader,"attempt_binding",lambda v:v.update({key:value}))
    elif fault == "intent": alter(record,reader,"intent_binding",lambda v:v.update(spec_sha256="0"*64))
    elif fault.startswith("action_"):
        key,value={"action_job":("stdout","9990002"),"action_error":("returncode",1),"action_argv":("argv",["sbatch","other"])}[fault]
        alter(record,reader,"action_binding",lambda v:v.update({key:value}))
    elif fault == "script": record["script_binding"]=reader.put(p.CONTROL+"/job.slurm",b"changed script")
    elif fault in ("listing","ambiguous"):
        alter(record,reader,"attempt_directory_inventory_binding",lambda v:v["attempt_files" if fault=="listing" else "unresolved_intents"].append("hidden.json"))
    elif fault in ("query","job_step","running"):
        alter(record,reader,"query_binding",lambda v:v.update(stdout=v["stdout"].replace("9990001.batch","9990001.0") if fault=="job_step" else v["stdout"].replace("COMPLETED","RUNNING") if fault=="running" else "9990001|COMPLETED|100|0:0|gres/gpu=4"))
    elif fault == "namespace": record["native_namespace"]+="other"
    elif fault == "base_reset": base=90
    seal(record,"receipt_sha256")
    with pytest.raises((ValueError,KeyError,AssertionError)): verify(record,reader,base,synthetic=fault!="native_fixture")


def test_unsubmitted_rpc_ambiguity_cannot_be_hidden():
    record,reader,base=fixture(submitted=False); record["submission_rpc_started"]=True;seal(record,"receipt_sha256")
    with pytest.raises(ValueError): verify(record,reader,base)


def test_original_operator_directory_receipt_is_admitted_without_rewriting_its_bytes(closed):
    record,r,base=closed;record=copy.deepcopy(record);reader=Evidence();reader.files=dict(r.files)
    def original_schema(value):
        value["protocol"]="sft-anchor-cvrp-timing-probe-operator-v1-attempt-directory-inventory"
        value["control_root"]=value.pop("native_control_root")
    alter(record,reader,"attempt_directory_inventory_binding",original_schema);seal(record,"receipt_sha256")
    before=reader(record["attempt_directory_inventory_binding"])
    assert verify(record,reader,base)["historical_pilot_reserved_gpu_hours"]==94
    assert reader(record["attempt_directory_inventory_binding"])==before


def test_effective_manifest_cannot_omit_embedded_original_manifest_even_with_all121_files():
    record,reader,base=fixture(omit_embedded_manifest=True)
    # All semantic and byte bindings are internally rebuilt around the omission;
    # the original121 files and separately bound original manifest remain exact.
    with pytest.raises(ValueError,match="Embedded original manifest"):
        verify(record,reader,base)


def compiled():
    return b.compile_consumers(expected_probe_preparer_sha256=p.sha((EXP/"prepare_cvrp_timing_probe_v1.py").read_bytes()),
        expected_probe_accounting_sha256=p.sha((EXP/"verify_cvrp_timing_probe_accounting_v1.py").read_bytes()),
        expected_probe_history_sha256=p.sha((EXP/"cvrp_timing_probe_history_v1.py").read_bytes()),
        expected_bridge_sha256=p.sha((EXP/"confirmation_probe_history_bridge_v2.py").read_bytes()))


def context_fixture(record,reader,bridge_proof):
    A=load("legacy_history_for_fixture","analyze_chunked_confirmation_v1.py")
    old=json.loads(PILOT)
    baseline=[{"reservation_id":"prior-v1/1030929","gpus":4,"requested_wall_seconds":7200}]
    baseline += [{"reservation_id":"phase2/"+slot,"gpus":1 if slot.endswith(".sft") else 4,"requested_wall_seconds":7200} for slot in old["plan"]["order"]]
    ledger=seal({"protocol":A.PROTOCOL+"-historical-reservation-ledger","synthetic_fixture":True,
        "independent_read_only_collection":True,"all_scoped_attempt_directories_reconciled":True,"unresolved_intents":[],
        "pilot_authority_binding":record["pilot_authority_binding"],"prior_failed_phase_binding":record["prior_failed_proof_binding"],
        "original_pilot_reservations":baseline,"additional_pilot_authorities":[],"prior_formal_rounds":[],
        "cvrp_timing_probe_custody_binding":reader.put("/synthetic/probe-custody.json",record)},"receipt_sha256")
    if record.get("rescue_authority_binding"):
        rescue=json.loads(reader(record["rescue_authority_binding"]))
        ledger["additional_pilot_authorities"]=[{"authority_binding":record["rescue_authority_binding"],
            "additional_reservations":rescue["additional_reservations"]}];seal(ledger,"receipt_sha256")
    binding=reader.put("/synthetic/complete-history.json",ledger)
    identity=bridge_proof["operational_identity"]
    context={"registration_entry":{"historical_resource_ledger_binding":binding,"operational_history_bridge_identity":identity},
        "entries":identity["closed_additive_source_files"],"contract":{"round_number":1},"packet_reader":reader,"synthetic_fixture":True}
    total=json.loads(reader(record["authority_binding"]))["combined_reserved_requested_gpu_hours"]
    authority={"historical_resource_ledger_binding":binding,"historical_pilot_requested_gpu_hours":total,"historical_formal_requested_gpu_hours":0,
        "operational_history_bridge_identity":identity}
    return context,authority,ledger


def test_bridge_exact_two_additive_edits_keep_every_other_function_and_class_ast():
    P,A,proof=compiled()
    for module,key,filename,target in ((P,"preparation","prepare_confirmation_round_v1.py","validate_resource_authority"),
                                      (A,"analysis","analyze_chunked_confirmation_v1.py","historical_resource_custody")):
        assert proof[key]["replaced_resource_expressions"]==2
        expected={node.name:p.sha(ast.dump(node,include_attributes=False).encode()) for node in ast.parse((EXP/filename).read_bytes()).body
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name!=target}
        assert expected==proof[key]["unchanged_function_class_ast_sha256"]
        assert module.__file__==str(EXP/filename) and module.__operational_history_bridge__["bridge_source_sha256"]==p.sha((EXP/"confirmation_probe_history_bridge_v2.py").read_bytes())
    assert not proof["native_registration_created"] and not proof["publication_registry_changed"]


def test_actual_legacy_history_guard_and_new_additive_guard_both_execute(closed):
    record,r,base=closed;reader=Evidence();reader.files=dict(r.files);P,A,proof=compiled();context,authority,ledger=context_fixture(record,reader,proof)
    report=A.historical_resource_custody(context,authority)
    assert report["pilot_reserved_gpu_hours"]==94 and report["formal_reserved_gpu_hours"]==0
    assert report["prior_failed_allocated_gpu_seconds"]==1836
    outer=b.wrap_analysis_receipt(item("/synthetic/original-report",b"{}"),{},context,proof,original_report_raw=b"{}")
    assert not outer["mathematical_outcomes_modified"] and outer["probe_history"]["admission"]["diagnostic_allocated_gpu_seconds"]==400
    authority["historical_pilot_requested_gpu_hours"]=92
    with pytest.raises(ValueError): A.historical_resource_custody(context,authority)
    ledger["original_pilot_reservations"].pop();seal(ledger,"receipt_sha256")
    context["registration_entry"]["historical_resource_ledger_binding"]=authority["historical_resource_ledger_binding"]=reader.put("/synthetic/complete-history.json",ledger)
    with pytest.raises(ValueError): A.historical_resource_custody(context,authority)


def test_new_preparer_hook_requires_same_bound_custody_and_explicit_fixture_flag(closed):
    record,r,base=closed;reader=Evidence();reader.files=dict(r.files);binding=reader.put("/synthetic/probe-custody.json",record)
    P,_,proof=compiled(); sources=proof["operational_identity"]["closed_additive_source_files"]
    source_binding=reader.put("/synthetic/new-formal-source.json",{"synthetic_fixture":True,"files":list(sources.values())})
    proposal={"synthetic_fixture":True,"bindings":{"cvrp_timing_probe_custody":binding,"source_manifest":source_binding},
        "source_file_bindings":{path:reader.put("/synthetic/formal/"+path,(EXP/Path(path).name).read_bytes()) for path in sources}}
    authority={"cvrp_timing_probe_custody_binding":binding,"operational_history_bridge_identity":proof["operational_identity"]}
    assert P._probe_aware_preparation_total(proposal,reader,authority,fixture=True)==94
    with pytest.raises(ValueError):P._probe_aware_preparation_total(proposal,reader,authority,fixture=False)
    with pytest.raises(ValueError):P._probe_aware_preparation_total(proposal,reader,{},fixture=True)


def test_changed_bridge_byte_or_dependency_pin_cannot_compile():
    with pytest.raises(ValueError):b.compile_consumers(expected_probe_preparer_sha256="0"*64,expected_probe_accounting_sha256="0"*64,
        expected_probe_history_sha256="0"*64,expected_bridge_sha256="0"*64)


def test_old_evaluation_only_authority_and_new_probe_authority_are_replayed_from_same_bytes():
    record,reader,base=fixture(rescue=True);_,A,proof=compiled();context,authority,ledger=context_fixture(record,reader,proof)
    report=A.historical_resource_custody(context,authority)
    assert report["pilot_reserved_gpu_hours"]==114 and context["probe_history_v2"]["admission"]["historical_pilot_reserved_gpu_hours"]==114
    ledger["additional_pilot_authorities"]=[];seal(ledger,"receipt_sha256")
    context["registration_entry"]["historical_resource_ledger_binding"]=authority["historical_resource_ledger_binding"]=reader.put("/synthetic/dropped-rescue-history",ledger)
    with pytest.raises(ValueError):A.historical_resource_custody(context,authority)


@pytest.fixture(scope="module")
def original_preparation():
    import test_prepare_confirmation_round_v1 as original
    return original.fixture()


@pytest.mark.parametrize("fault",[None,"missing_timing","failed_timing","missing_slot","unmeasured_step","lost_prior","retry","missing_bridge_source"])
def test_full_original_measured_all12_82slot_preparation_guard_executes_with_new_history(original_preparation,closed,fault):
    proposal,reader=copy.deepcopy(original_preparation);record,r,base=closed;reader.files.update(r.files)
    P,_,proof=compiled();identity=proof["operational_identity"]
    binding=reader.put("/synthetic/probe-custody.json",record);proposal["bindings"]["cvrp_timing_probe_custody"]=binding
    source=json.loads(reader(proposal["bindings"]["source_manifest"]))
    for path,row in identity["closed_additive_source_files"].items():
        source["files"].append(row);proposal["source_file_bindings"][path]=reader.put("/synthetic/formal/"+path,(EXP/Path(path).name).read_bytes())
    proposal["bindings"]["source_manifest"]=reader.put("synthetic/formal-source.json",source)
    authority=json.loads(reader(proposal["bindings"]["resource_authority"]))
    authority["historical_pilot_requested_gpu_hours"]=94;authority["combined_reserved_requested_gpu_hours"]+=2
    authority["declared_requested_gpu_hour_ceiling"]+=2;authority["cvrp_timing_probe_custody_binding"]=binding
    authority["operational_history_bridge_identity"]=identity
    if fault=="missing_slot":authority["planned_slots"].pop()
    if fault=="unmeasured_step":authority["formal_step_overhead_seconds"]=0
    if fault=="lost_prior":authority["prior_failed_phase_binding"]["sha256"]="0"*64
    if fault=="retry":authority["planned_slots"][0]["maximum_attempts"]=2
    if fault=="missing_bridge_source":
        source["files"].pop();proposal["bindings"]["source_manifest"]=reader.put("synthetic/formal-source.json",source)
    seal(authority,"authority_sha256");proposal["bindings"]["resource_authority"]=reader.put("synthetic/resource-authority.json",authority)
    if fault in ("missing_timing","failed_timing"):
        timing=json.loads(reader(proposal["bindings"]["pilot_timing"]))
        if fault=="missing_timing":timing["slots"].pop(P.PILOT_ORDER[0])
        else:timing["slots"][P.PILOT_ORDER[0]]["scheduler_state"]="FAILED"
        proposal["bindings"]["pilot_timing"]=reader.put("synthetic/timing.json",timing)
    selection=json.loads(reader(proposal["bindings"]["selection"]))
    if fault:
        with pytest.raises((ValueError,KeyError)):P.validate_resource_authority(proposal,reader,selection,2,fixture=True)
    else:
        admitted=P.validate_resource_authority(proposal,reader,selection,2,fixture=True)
        assert len(admitted["planned_slots"])==82 and admitted["combined_reserved_requested_gpu_hours"]==94+admitted["new_round_requested_gpu_hours"]
        result={"synthetic_fixture":True,"status":"synthetic_preparation_no_registration_or_claim"}
        outer=b.wrap_preparation_receipt(result,proposal,proof)
        assert outer["original_preparation_result"]==result and not outer["native_submission_authorized"]


@pytest.mark.parametrize("fault",["authority_identity","registered_identity","source_row","missing_custody"])
def test_additive_code_cannot_be_silent_or_omitted_from_formal_registration(closed,fault):
    record,r,_=closed;reader=Evidence();reader.files=dict(r.files);_,A,proof=compiled();context,authority,ledger=context_fixture(record,reader,proof)
    if fault=="authority_identity":authority.pop("operational_history_bridge_identity")
    elif fault=="registered_identity":context["registration_entry"].pop("operational_history_bridge_identity")
    elif fault=="source_row":context["entries"]={}
    else:
        ledger.pop("cvrp_timing_probe_custody_binding");seal(ledger,"receipt_sha256")
        context["registration_entry"]["historical_resource_ledger_binding"]=authority["historical_resource_ledger_binding"]=reader.put("/synthetic/missing-probe.json",ledger)
    with pytest.raises(ValueError):A.historical_resource_custody(context,authority)


@pytest.mark.parametrize("fault",[None,"pin","scope","bytes","fixture"])
def test_immutable_reader_index_binds_actual_native_identity_and_keeps_original_reader_type(tmp_path,fault):
    P,_,_=compiled();body=b"synthetic immutable source";path=tmp_path/"copied-source.py";path.write_bytes(body);path.chmod(0o440)
    binding=item("copied-source.py",body)
    index={"protocol":b.PROTOCOL+"-native-artifact-index","synthetic_fixture":True,"independent_read_only_collection":True,
        "artifacts":[{"binding":binding,"native_path":p.SOURCE+"/src/source.py"}]}
    if fault=="scope":index["artifacts"][0]["native_path"]="/lustre/other/source.py"
    if fault=="bytes":index["artifacts"][0]["binding"]["sha256"]="0"*64
    index_raw=raw(index);index_path=tmp_path/"identity.json";index_path.write_bytes(index_raw);index_path.chmod(0o440)
    index_binding=item("identity.json",index_raw);expected="0"*64 if fault=="pin" else p.sha(index_raw)
    if fault:
        with pytest.raises(ValueError):b.immutable_native_reader(P,tmp_path,index_binding,expected_identity_sha256=expected,synthetic_fixture=fault!="fixture")
    else:
        reader=b.immutable_native_reader(P,tmp_path,index_binding,expected_identity_sha256=expected,synthetic_fixture=True)
        assert isinstance(reader,P.ImmutableReader) and reader(binding)==body
        assert reader.native_identity(binding)["native_path"]==p.SOURCE+"/src/source.py"
        with pytest.raises(ValueError):reader.native_identity({**binding,"sha256":"0"*64})


@pytest.mark.parametrize("original,alias",[({"formal_claim_eligible":False},{"formal_claim_eligible":0}),
                                          ({"draw_count":3},{"draw_count":3.0})])
def test_wrapped_original_raw_report_cannot_alias_boolean_or_integer_types(original,alias):
    body=raw(original);binding=item("synthetic/actual-original-report.json",body)
    context={"probe_history_v2":{"synthetic_fixture":True}}
    valid=b.wrap_analysis_receipt(binding,original,context,{},original_report_raw=body)
    assert valid["original_report"]==original
    assert original==alias  # Demonstrate why ordinary decoded equality is insufficient.
    with pytest.raises(ValueError):b.wrap_analysis_receipt(binding,alias,context,{},original_report_raw=body)
