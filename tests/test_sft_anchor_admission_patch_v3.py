"""Offline guards for an explicitly declared admission-only operational patch.

Real immutable producer bytes and authority are used. Activations/transitions
below are manufactured unit fixtures, never process or scheduler evidence.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_sft_anchor_collection import encoded, record

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"
SNAPSHOT = EXP / "collected_status_v2_20261014T0911/status"


def module(name):
    spec = importlib.util.spec_from_file_location("test_"+name, EXP / (name+".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


preparer = module("prepare_admission_patch_v3")
wrapper = module("run_campaign_admission_v3")
collector = module("collect_campaign_admission_v3")


@pytest.fixture
def patch(tmp_path):
    output=tmp_path/"patch"
    value=preparer.prepare(argparse.Namespace(producer_source_root=EXP/"source_v2",
        producer_prepared_manifest=EXP/"control_v2/launch_manifest.json",
        producer_authority=SNAPSHOT/"campaign_authority.json",
        producer_owner=SNAPSHOT/"campaign_owners/3989259-9534304.json",output=output))
    return output,value["patch_manifest_sha256"],json.loads((output/"patch_manifest.json").read_text())


def transition(manifest,expected):
    value={"schema_version":1,"protocol":preparer.TRANSITION_PROTOCOL,
           "patch_manifest_sha256":expected,"launch_manifest_sha256":preparer.LAUNCH_SHA,
           "source_manifest_sha256":preparer.SOURCE_SHA,
           "authority_sha256":manifest["producer"]["authority"]["sha256"],
           "replaced_owner":preparer.REPLACED_OWNER,"guarded_identity_verified_before_stop":True,
           "old_controller_stopped":True,"post_stop_same_host":"jean-zay1",
           "post_stop_original_identity_present":False,"model_jobs_cancelled_or_retried":False,
           "preserved_job_ids":["1052260"],"synthetic_fixture":True}
    value["semantic_sha256"]=preparer.canonical(value)
    return value


def patch_bundle(patch,activated=False):
    root,expected,manifest=patch
    records=[record(str(preparer.PATCH_ROOT/"patch_manifest.json"),"operational_patch/patch_manifest.json",
                    (root/"patch_manifest.json").read_bytes(),"bound_operational_patch")]
    for item in manifest["files"]:
        records.append(record(str(preparer.PATCH_ROOT/item["path"]),"operational_patch/files/"+item["path"],
                       (root/item["path"]).read_bytes(),"bound_operational_patch_file"))
    records.append(record(str(preparer.NAMESPACE/"control_v2/campaign_authority.json"),
                          "status/campaign_authority.json",(root/"producer_authority.json").read_bytes()))
    info={"manifest_sha256":expected,"native_patch_root":str(preparer.PATCH_ROOT),
          "native_runtime_root":str(preparer.RUNTIME_ROOT),"static_inventory":manifest["files"],
          "activation_receipts":[],"transition_receipt_sha256":None,"report_sidecar_present":False}
    if activated:
        t_raw=encoded(transition(manifest,expected));t_sha=collector.sha(t_raw)
        command=[preparer.NATIVE_PYTHON,"-I","-B",str(preparer.PATCH_ROOT/"run_campaign_admission_v3.py"),
            "--operational-patch-manifest",str(preparer.PATCH_ROOT/"patch_manifest.json"),
            "--operational-patch-manifest-sha256",expected,"--prepared-manifest",str(preparer.NAMESPACE/"control_v2/launch_manifest.json"),
            "--prepared-manifest-sha256",preparer.LAUNCH_SHA,"--producer-source-root",str(preparer.NAMESPACE/"source_v2"),
            "--transition-receipt",str(preparer.RUNTIME_ROOT/"transition_receipt.json"),"--transition-receipt-sha256",t_sha,"--execute"]
        old_owner=json.loads((root/"producer_owner.json").read_text())
        owner={"pid":999,"proc_start_ticks":"123","hostname":"jean-zay1","executable":old_owner["executable"],
               "proc_command":command,"command_sha256":collector.sha("\0".join(command).encode())}
        activation={"schema_version":1,"protocol":"sft-anchor-admission-activation-v3","owner":owner,
            "status":"entered_wrapper_original_controller_lock_required",
            "patch_manifest_sha256":expected,"transition_receipt_sha256":t_sha,
            "authority_sha256":manifest["producer"]["authority"]["sha256"],"effective_code_inventory":manifest["files"],
            "override_only":"collector_module","model_producer_unchanged":True,"combined_requested_gpu_hours":92}
        records.append(record(str(preparer.RUNTIME_ROOT/"transition_receipt.json"),"operational_patch/transition_receipt.json",t_raw,"bound_operational_transition"))
        records.append(record(str(preparer.RUNTIME_ROOT/"activations/999-123.json"),"operational_patch/activations/999-123.json",encoded(activation),"bound_operational_activation"))
        records.append(record(str(preparer.NAMESPACE/"control_v2/campaign_owners/999-123.json"),"status/campaign_owners/999-123.json",
            encoded({**owner,"authority_sha256":json.loads((root/"producer_authority.json").read_text())["semantic_sha256"]})))
        info.update(activation_receipts=["999-123.json"],transition_receipt_sha256=t_sha)
    return {"records":records,"operational_patch":info,"completed_slots":[]}


def check_patch_bundle(bundle,report_present=False):
    return collector.validate_operational_patch(bundle,collector.decode_records(bundle),preparer.LAUNCH_SHA,report_present=report_present)


def mutate_record(bundle,path,change):
    item=next(r for r in bundle["records"] if r["local_path"]==path)
    value=json.loads(collector.decode_records(bundle)[path][1]);change(value)
    bundle["records"][bundle["records"].index(item)]=record(item["native_path"],path,encoded(value),item["purpose"])


def test_prepare_checks_all_original_bytes_and_never_mutates_or_resets_authority(patch):
    root,expected,manifest=patch
    assert preparer.verify_patch(root/"patch_manifest.json",expected)==manifest
    assert manifest["policy"]["combined_requested_gpu_hours"]==92
    assert (root/"producer_authority.json").read_bytes()==(SNAPSHOT/"campaign_authority.json").read_bytes()
    assert (root/"producer_owner.json").read_bytes()==(SNAPSHOT/"campaign_owners/3989259-9534304.json").read_bytes()
    assert {p.name for p in root.glob("*.py")}==set(preparer.CODE_FILES)


def test_copied_wrapper_default_runs_validated_original_plan_without_scheduler_calls(patch):
    root,expected,_=patch
    result=subprocess.run([sys.executable,"-I","-B",str(root/"run_campaign_admission_v3.py"),
        "--operational-patch-manifest",str(root/"patch_manifest.json"),"--operational-patch-manifest-sha256",expected,
        "--prepared-manifest",str(EXP/"control_v2/launch_manifest.json"),"--prepared-manifest-sha256",preparer.LAUNCH_SHA,
        "--producer-source-root",str(EXP/"source_v2")],check=True,capture_output=True,text=True,timeout=30)
    value=json.loads(result.stdout)
    assert value["status"]=="validated_operational_patch_plan_only_no_scheduler_or_process_actions"
    assert value["unchanged_original_plan"]["controller_sha256"]==preparer.PRODUCER_FILES["controller"][1]
    assert value["unchanged_original_plan"]["combined_maximum_requested_gpu_hours"]==92
    assert not (root/"activations").exists()


def test_execute_from_local_paths_fails_before_any_scheduler_or_process_action(patch):
    root,expected,_=patch
    result=subprocess.run([sys.executable,"-I","-B",str(root/"run_campaign_admission_v3.py"),
        "--operational-patch-manifest",str(root/"patch_manifest.json"),"--operational-patch-manifest-sha256",expected,
        "--prepared-manifest",str(EXP/"control_v2/launch_manifest.json"),"--prepared-manifest-sha256",preparer.LAUNCH_SHA,
        "--producer-source-root",str(EXP/"source_v2"),"--execute"],capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and "No such file" in result.stderr
    assert not (root/"activations").exists()


@pytest.mark.parametrize("filename",preparer.CODE_FILES+("producer_authority.json","producer_owner.json"))
def test_changed_closed_patch_bytes_are_rejected(patch,filename):
    root,expected,_=patch;(root/filename).chmod(0o640)
    with (root/filename).open("ab") as stream: stream.write(b"\n ")
    with pytest.raises(ValueError,match="byte inventory"):
        preparer.verify_patch(root/"patch_manifest.json",expected)


def test_extra_patch_python_file_rejected(patch):
    root,expected,_=patch;(root/"extra.py").write_text("raise RuntimeError('never imported')")
    with pytest.raises(ValueError,match="Unbound"):
        preparer.verify_patch(root/"patch_manifest.json",expected)


def test_unbound_bytecode_cannot_override_verified_patch_source(patch):
    root,expected,_=patch;(root/"__pycache__").mkdir()
    (root/"__pycache__/run_campaign_admission_v3.cpython-310.pyc").write_bytes(b"unbound")
    with pytest.raises(ValueError,match="bytecode"):
        preparer.verify_patch(root/"patch_manifest.json",expected)


def test_only_collector_factory_changes_and_effective_module_is_corrected(patch):
    root,expected,manifest=patch
    controller=wrapper.load_module(EXP/"source_v2"/preparer.PRODUCER_FILES["controller"][0],"test_original_controller")
    functions={k:v for k,v in controller.__dict__.items() if callable(v)}
    wrapper.install_override(controller,root,manifest,expected,preparer)
    assert all(controller.__dict__[k] is v for k,v in functions.items() if k!="collector_module")
    assert Path(controller.__file__).resolve()==(EXP/"source_v2"/preparer.PRODUCER_FILES["controller"][0]).resolve()
    assert controller.collector_module().raw_generation_projection({"text":"exact","score":0})=={"text":"exact"}


@pytest.mark.parametrize("key,value",[("patch_manifest_sha256","0"*64),("launch_manifest_sha256","0"*64),
    ("source_manifest_sha256","0"*64),("authority_sha256","0"*64),("guarded_identity_verified_before_stop",False),
    ("old_controller_stopped",False),("post_stop_same_host","jean-zay2"),("post_stop_original_identity_present",True),
    ("model_jobs_cancelled_or_retried",True),("preserved_job_ids",[]),("replaced_owner",{})])
def test_transition_cannot_rebind_owner_jobs_source_or_authority(patch,key,value):
    _,expected,manifest=patch;t=transition(manifest,expected);t[key]=value
    t["semantic_sha256"]=preparer.canonical({k:v for k,v in t.items() if k!="semantic_sha256"})
    with pytest.raises(ValueError,match="guarded transition"):
        preparer.validate_transition(t,manifest,expected)


def test_pending_patch_is_disclosed_without_claiming_activation(patch):
    assert check_patch_bundle(patch_bundle(patch))


def test_actual_owner_proof_must_bind_closed_wrapper_command(patch):
    assert check_patch_bundle(patch_bundle(patch,True))


def test_completed_live_admission_requires_transition_and_actual_lock_owner(patch):
    bundle=patch_bundle(patch);bundle["completed_slots"]=[collector.ORDER[0]]
    with pytest.raises(ValueError,match="actual disclosed"):
        check_patch_bundle(bundle)
    bundle=patch_bundle(patch,True);bundle["completed_slots"]=[collector.ORDER[0]]
    bundle["records"]=[r for r in bundle["records"] if r["local_path"]!="status/campaign_owners/999-123.json"]
    with pytest.raises(ValueError,match="lock owner"):
        check_patch_bundle(bundle)


@pytest.mark.parametrize("target",["authority","code","activation","command","transition"])
def test_collection_patch_guards_reject_tampering(patch,target):
    bundle=patch_bundle(patch,True)
    if target=="authority": mutate_record(bundle,"status/campaign_authority.json",lambda v:v.update(absolute_deadline_unix=999999))
    elif target=="code":
        item=next(r for r in bundle["records"] if r["local_path"]=="operational_patch/files/run_campaign_admission_v3.py")
        bundle["records"][bundle["records"].index(item)]=record(item["native_path"],item["local_path"],b"# changed\n",item["purpose"])
    elif target=="activation": mutate_record(bundle,"operational_patch/activations/999-123.json",lambda v:v.update(override_only="submit_slot"))
    elif target=="command": mutate_record(bundle,"operational_patch/activations/999-123.json",lambda v:v["owner"].update(proc_command=["python","wrong.py"]))
    else: mutate_record(bundle,"operational_patch/transition_receipt.json",lambda v:v.update(model_jobs_cancelled_or_retried=True))
    with pytest.raises(ValueError): check_patch_bundle(bundle)


def test_live_packet_cannot_use_synthetic_escape_hatch():
    with pytest.raises(ValueError,match="explicit closed"):
        collector.validate_operational_patch({"synthetic_fixture":True},{},preparer.LAUNCH_SHA,report_present=False)


def test_full_report_cannot_hide_effective_patched_code(patch):
    with pytest.raises(ValueError,match="effective-code sidecar"):
        check_patch_bundle(patch_bundle(patch,True),report_present=True)


def add_sidecar(bundle,patch):
    _,expected,manifest=patch
    report=encoded({"synthetic_fixture":True,"status":"test_only_no_quality_claim"})
    report_record=record(str(preparer.NAMESPACE/"control_v2/campaign_report.json"),"campaign_report.json",report,"completed_campaign_report")
    bundle["records"].append(report_record)
    activation=next(r for r in bundle["records"] if r["local_path"]=="operational_patch/activations/999-123.json")
    sidecar={"schema_version":1,"protocol":"sft-anchor-admission-report-sidecar-v3",
        "patch_manifest_sha256":expected,"activation_receipt":{"native_path":activation["native_path"],"sha256":activation["sha256"]},
        "original_report":{"native_path":report_record["native_path"],"sha256":report_record["sha256"],"size_bytes":len(report)},
        "effective_code_inventory":manifest["files"],"original_controller_is_base_code_not_exclusive_effective_code":True,
        "model_producer_unchanged":True,"superiority_established":False}
    sidecar["semantic_sha256"]=collector.canonical(sidecar)
    bundle["records"].append(record(str(preparer.RUNTIME_ROOT/"report_sidecar.json"),"operational_patch/report_sidecar.json",encoded(sidecar),"bound_operational_report_sidecar"))
    bundle["operational_patch"]["report_sidecar_present"]=True


def test_full_sidecar_binds_original_report_and_discloses_actual_effective_code(patch):
    bundle=patch_bundle(patch,True);add_sidecar(bundle,patch)
    assert check_patch_bundle(bundle,report_present=True)


@pytest.mark.parametrize("key,value",[("original_controller_is_base_code_not_exclusive_effective_code",False),
    ("original_report",{}),("effective_code_inventory",[]),("patch_manifest_sha256","0"*64),
    ("activation_receipt",{}),("superiority_established",True)])
def test_sidecar_cannot_hide_code_identity_or_claim_quality(patch,key,value):
    bundle=patch_bundle(patch,True);add_sidecar(bundle,patch)
    def change(v):
        v[key]=value;v["semantic_sha256"]=collector.canonical({k:item for k,item in v.items() if k!="semantic_sha256"})
    mutate_record(bundle,"operational_patch/report_sidecar.json",change)
    with pytest.raises(ValueError):check_patch_bundle(bundle,report_present=True)


def test_readonly_remote_reader_with_patch_support_compiles_and_contains_no_model_actions():
    compile(collector.REMOTE_READER,"<readonly-reader>","exec")
    assert "subprocess.run(['sbatch'" not in collector.REMOTE_READER
    assert "subprocess.run(['scancel'" not in collector.REMOTE_READER
    assert "operational_patch/patch_manifest.json" in collector.REMOTE_READER
