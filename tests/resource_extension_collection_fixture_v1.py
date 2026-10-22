"""Wholly synthetic scheduler/model outcomes over real immutable input bytes.

No native action, model, training, policy call, private key or real quality value
is used.  All manufactured outcomes are explicitly marked and cannot publish.
"""
import base64
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import shlex
import types
from unittest.mock import patch

from test_sft_anchor_collection import encoded as pretty, record as old_record
from test_sft_anchor_admission_v3_collection import _durable_slot
from test_sft_anchor_admission_patch_v3 import patch_bundle
from test_resource_extension_collection_v1 import c, record

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"
HERE = EXP / "resource_extension_v1"


def fixture_bundle(extension_count=6, full_results=True):
    raw_operator = (HERE/"cvrp_wall_resource_v1.py").read_bytes()
    operator_sha = c.digest(raw_operator)
    operator = c.compile_operator(raw_operator, operator_sha, diagnostic_fixture=True)
    boundary = json.loads(json.loads((HERE/"six_op_boundary_20261015T192752.json").read_bytes())["stdout"])
    manifest_raw = (EXP/"control_v2/launch_manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    source_root = EXP/"source_v2"
    source_raw = (source_root/"source_manifest.json").read_bytes()
    source_manifest = json.loads(source_raw)
    entries = {r["path"]:r for r in source_manifest["files"]}
    kernels = c.load_kernels()
    scorer_raw = (source_root/kernels.SCORER).read_bytes()
    scorer = types.ModuleType("synthetic_fixture_exact_frozen_scorer")
    exec(compile(scorer_raw,"<frozen-synthetic-fixture-scorer>","exec"),scorer.__dict__)
    contracts = {t:scorer.validate_contract(json.loads((ROOT/b["path"]).read_bytes())) for t,b in manifest["contracts"].items()}
    op_records = [old_record(c.CONTROL+"/launch_manifest.json","status/launch_manifest.json",manifest_raw,"bound_launch_manifest"),
                  old_record(c.SOURCE+"/source_manifest.json","source/source_manifest.json",source_raw,"bound_source_manifest"),
                  old_record(c.SOURCE+"/"+kernels.SCORER,"source/frozen_scorer.py",scorer_raw,"bound_frozen_scorer")]
    prior = manifest["prior_failed_phase"]["proof"]
    op_records.append(old_record(c.SOURCE+"/"+prior["path"],"source/prior_failed_phase.json",(source_root/prior["path"]).read_bytes(),"bound_prior_failed_phase"))
    for task,binding in manifest["contracts"].items():
        op_records.append(old_record(c.SOURCE+"/"+binding["path"],"source/contracts/"+task+".json",(ROOT/binding["path"]).read_bytes(),"bound_task_contract"))
    op_attempts=[]
    for slot in c.OP_ORDER:
        row = boundary["records"]["control_v2/attempts/"+slot+".json"]
        raw = base64.b64decode(row["content_base64"])
        op_attempts.append(json.loads(raw))
        op_records.append(old_record(c.CONTROL+"/attempts/"+slot+".json","attempts/"+slot+".json",raw,"submission_attempt_snapshot"))
    patch_root=EXP/"admission_prepared_v1";patch_raw=(patch_root/"patch_manifest.json").read_bytes();patch_value=json.loads(patch_raw)
    pb=patch_bundle((patch_root,c.digest(patch_raw),patch_value),activated=True)
    op_records+=pb["records"]
    payloads={};trees={};raw_markers={};raw_results={}
    for slot in c.ORDER:
        task,arm=slot.split(".")
        payloads[slot],trees[slot]=_durable_slot(slot,contracts[task],manifest,entries[kernels.ENTRYPOINT])
        raw_markers[slot]=payloads[slot]["complete.json"];raw_results[slot]=payloads[slot]["result.json"]
    for slot in c.OP_ORDER:
        for name,raw in payloads[slot].items():
            task,arm=slot.split(".")
            path=(f"outputs/{slot}/complete.json" if name=="complete.json" else
                  f"outputs/{slot}/{task}/{arm}/"+("training/" if name=="actual_prompt_windows.jsonl" else "")+name)
            op_records.append(old_record(c.CONTROL+"/"+path,f"results/{slot}/{name}",raw,"completed_durable_proof"))
    op_jobs=[a["job_id"] for a in op_attempts]
    op_query={"argv":["sacct","--noheader","--parsable2","--jobs",",".join(op_jobs),"--format=JobIDRaw,State%40,ExitCode"],
              "returncode":0,"stderr":"","stdout":"".join(f"{job}{suffix}|COMPLETED|0:0\n" for job in op_jobs for suffix in ("",".batch",".extern"))}
    op_bundle={"schema_version":2,"study":c.STUDY,"synthetic_fixture":True,"prepared_manifest_sha256":c.ORIGINAL_LAUNCH_SHA256,
               "records":op_records,"source_proofs":source_manifest["files"],"completed_slots":list(c.OP_ORDER),
               "read_only_queries":[op_query],"operational_patch":pb["operational_patch"],
               "terminal_artifact_proofs":{s:trees[s] for s in c.OP_ORDER},"native_mutations":False,"superiority_established":False}
    rows={};by_native={}
    def add(native,local,raw):
        if native in by_native:
            assert by_native[native]==raw
            return {"path":native,"sha256":c.digest(raw),"size_bytes":len(raw)}
        rows[local]=record(local,native,raw);by_native[native]=raw
        return {"path":native,"sha256":c.digest(raw),"size_bytes":len(raw)}
    operator_binding=add(c.EXTENSION+"/cvrp_wall_resource_v1.py","resource_extension/cvrp_wall_resource_v1.py",raw_operator)
    amendment_binding=add(c.EXTENSION+"/amendment.json","resource_extension/amendment.json",(HERE/"amendment.json").read_bytes())
    t0=datetime(2026,10,15,23,0,tzinfo=timezone.utc)
    authority=operator.authority_candidate(amendment_binding,operator_binding,t0.isoformat());authority["synthetic_fixture"]=True
    authority_binding=add(c.EXTENSION+"/authority.json","resource_extension/authority.json",pretty(authority))
    inputs={}
    def inp(key,native,path):inputs[key]=add(native,"resource_extension/inputs/"+key,Path(path).read_bytes())
    inp("original_source_manifest",c.SOURCE+"/source_manifest.json",source_root/"source_manifest.json")
    inp("launch_manifest",c.CONTROL+"/launch_manifest.json",EXP/"control_v2/launch_manifest.json")
    inp("admission_patch_manifest",c.NAMESPACE+"/admission_v1/patch_manifest.json",patch_root/"patch_manifest.json")
    inp("corrected_durable_collector",c.NAMESPACE+"/admission_v1/collect_campaign_admission_v3.py",patch_root/"collect_campaign_admission_v3.py")
    inp("original_authority",c.CONTROL+"/campaign_authority.json",patch_root/"producer_authority.json")
    inp("original_first_running",c.CONTROL+"/campaign_first_running.json",EXP/"collected_native_first_complete_v1/status/campaign_first_running.json")
    inp("probe_terminal_custody",c.EXTENSION+"/inputs/probe_terminal_custody.json",EXP/"validation/native_monitor/actual1074755_terminal_20261015T191744_v4.json")
    inp("initializer_inventory",c.EXTENSION+"/inputs/initializer_inventory.json",EXP/"cvrp_probe_execution_v3/native_prepared_v3/control_v1/initializer_inventory.json")
    inp("six_op_boundary",c.EXTENSION+"/inputs/six_op_boundary.json",HERE/"six_op_boundary_20261015T192752.json")
    inp("prior_rejected_probe",c.EXTENSION+"/inputs/prior_rejected_probe.json",EXP/"validation/native_monitor/cvrp_first_probe_preassignment_failure_20261014T1742_bound_v1.json")
    inp("prior_rejected_probe_no_assigned_job",c.EXTENSION+"/inputs/prior_rejected_probe_no_assigned_job.json",EXP/"validation/native_monitor/cvrp_first_probe_no_assigned_job_20261014T1743_bound_v1.json")
    for e in source_manifest["files"]:add(c.SOURCE+"/"+e["path"],"resource_extension/source_files/"+e["path"],(source_root/e["path"]).read_bytes())
    for name in c.ADMISSION_FILES:add(c.NAMESPACE+"/admission_v1/"+name,"resource_extension/admission_files/"+name,(patch_root/name).read_bytes())
    originals={}
    for slot in c.ORDER:
        originals[slot]=(EXP/"control_v2/job_scripts"/(slot+".slurm")).read_bytes()
        add(c.CONTROL+"/job_scripts/"+slot+".slurm","resource_extension/original_scripts/"+slot+".slurm",originals[slot])
    outer={s:add(c.EXTENSION+"/effective_scripts/"+s+".slurm","resource_extension/effective_scripts/"+s+".slurm",
                 operator.build_outer_script(s,originals[s],operator_binding)) for s in c.CVRP_ORDER}
    registration=operator.registration_candidate(authority_binding,authority,inputs,outer,(t0+timedelta(seconds=1)).isoformat())
    registration["synthetic_fixture"]=True
    reg_raw=pretty(registration);reg_sha=c.digest(reg_raw)
    add(c.EXTENSION+"/registration.json","resource_extension/registration.json",reg_raw)
    def put(folder,slot,value):return add(c.EXTENSION+"/"+folder+"/"+slot+".json","resource_extension/"+folder+"/"+slot+".json",pretty(value))
    def qcontrol(fields,stamp):return {"argv":["scontrol","show","job",fields["JobId"]],"returncode":0,"stderr":"",
        "stdout":"\n".join(k+"="+v for k,v in fields.items())+"\n","recorded_utc":stamp.isoformat(),"time_zone":"UTC"}
    operator_test=types.ModuleType("operator_fixture_field_builder");operator_test.__file__=str(HERE/"test_cvrp_wall_resource_v1.py")
    exec(compile((HERE/"test_cvrp_wall_resource_v1.py").read_bytes(),operator_test.__file__,"exec"),operator_test.__dict__)
    original_times=[("2026-10-14T11:26:01",5759),("2026-10-15T05:12:16",5164),("2026-10-15T06:40:30",5450),
                    ("2026-10-15T08:12:15",1988),("2026-10-15T10:08:06",5657),("2026-10-15T22:57:00",120)]
    starts={a["slot"]:c.scheduler_utc(v[0]) for a,v in zip(op_attempts,original_times)}
    durations={a["slot"]:v[1] for a,v in zip(op_attempts,original_times)}
    all_attempts={a["slot"]:a for a in op_attempts}
    for index,slot in enumerate(c.CVRP_ORDER[:extension_count]):
        instant=datetime(2026,10,16,0,0,tzinfo=timezone.utc)+timedelta(minutes=index*5)
        job=str(900001+index);starts[slot]=instant+timedelta(minutes=1);durations[slot]=120
        argv=operator.effective_sbatch_argv(slot,reg_sha);r=operator.resources(slot)
        prior=[]
        for previous in c.ORDER[:6+index]:
            a=all_attempts[previous];g=1 if previous.endswith('.sft') else 4
            q={"argv":["sacct","--noheader","--parsable2","--jobs",a["job_id"],"--format="+operator.SACCT_FORMAT],
               "returncode":0,"stderr":"","time_zone":"UTC","recorded_utc":(instant-timedelta(seconds=1)).isoformat(),
               "stdout":"".join(f"{a['job_id']}{suffix}|COMPLETED|{durations[previous]}|0:0|cpu=40|gres/gpu={g}\n" for suffix in ('','.batch','.extern'))}
            closure={"slot":previous,"complete":{"path":c.CONTROL+f"/outputs/{previous}/complete.json","sha256":c.digest(raw_markers[previous]),"size_bytes":len(raw_markers[previous])},
                     "result":{"path":c.CONTROL+f"/outputs/{previous}/{previous.split('.')[0]}/{previous.split('.')[1]}/result.json","sha256":c.digest(raw_results[previous]),"size_bytes":len(raw_results[previous])}}
            prior.append({"slot":previous,"job_id":a["job_id"],"accounting":q,"complete_binding":closure["complete"],
                "durable_original_admission":{"slot":previous,"original_durable_closure":closure,"controller_sha256":operator.CONTROLLER_SHA,
                 "launcher_sha256":operator.LAUNCHER_SHA,"collector_sha256":operator.DURABLE_COLLECTOR_SHA,"quality_summary_returned":False}})
        def run_capacity(argv,**kwargs):
            out=("" if argv[0]=='squeue' else "vas@a100||qos_gpu_a100-dev,qos_gpu_a100-t3|qos_gpu_a100-t3\n" if 'assoc' in argv else
                 "qos_gpu_a100-t3|20:00:00|10000|cpu=4096,gres/gpu:a100=256,gres/gpu=256|1.000000\n")
            return types.SimpleNamespace(returncode=0,stderr="",stdout=out)
        with patch.object(operator,'now',return_value=instant-timedelta(seconds=1)):capacity=operator.capacity_snapshot(slot,run_capacity)
        original_binding={"path":c.CONTROL+"/job_scripts/"+slot+".slurm","sha256":c.digest(originals[slot]),"size_bytes":len(originals[slot])}
        intent={"protocol":operator.PROTOCOL+'-intent',"slot":slot,"created_utc":instant.isoformat(),"registration_sha256":reg_sha,
          "authority_sha256":authority_binding['sha256'],"operator_sha256":operator_sha,"resources":r,"original_script_binding":original_binding,
          "effective_script_binding":outer[slot],"prefix_evidence":{"ready":True,"next_slot":slot,"completed_prefix":prior},"capacity":capacity,
          "manifest_sha256":c.ORIGINAL_LAUNCH_SHA256,"argv":argv,"automatic_retry":False,"automatic_cancellation":False}
        ib=put('intents',slot,intent)
        sb=put('submission_actions',slot,{"protocol":operator.PROTOCOL+'-submission-action',"slot":slot,"submission_intent_sha256":ib['sha256'],
          "recorded_utc":instant.isoformat(),"argv":argv,"returncode":0,"stdout":job+'\n',"stderr":"","invocation_error":None})
        put('assigned_jobs',slot,{"protocol":operator.PROTOCOL+'-assigned-job',"slot":slot,"job_id":job,
           "submission_intent_sha256":ib['sha256'],"submission_action_sha256":sb['sha256'],"recorded_utc":instant.isoformat()})
        fields=operator_test.job(slot);fields.update(JobId=job,SubmitLine=shlex.join(argv),
            StdOut=c.CONTROL+'/logs/'+slot+'-'+job+'.out',StdErr=c.CONTROL+'/logs/'+slot+'-'+job+'.err')
        hb=put('held_readbacks',slot,{"protocol":operator.PROTOCOL+'-held-readback',"slot":slot,"job_id":job,
           "submission_intent_sha256":ib['sha256'],"submission_action_sha256":sb['sha256'],"registration_sha256":reg_sha,
           "query":qcontrol(fields,instant+timedelta(seconds=1)),"fields":fields})
        attempt={"protocol":operator.PROTOCOL+'-attempt',"slot":slot,"job_id":job,"state":"submitted","resources":r,
          "manifest_sha256":c.ORIGINAL_LAUNCH_SHA256,"registration_sha256":reg_sha,"authority_sha256":authority_binding['sha256'],
          "submission_intent_sha256":ib['sha256'],"submission_action_sha256":sb['sha256'],"held_readback_sha256":hb['sha256'],
          "created_utc":(instant+timedelta(seconds=2)).isoformat(),"original_script_binding":original_binding,"effective_script_binding":outer[slot]}
        ab=put('attempts',slot,attempt);all_attempts[slot]=attempt
        vb=put('activations',slot,{"protocol":operator.PROTOCOL+'-activation',"slot":slot,"job_id":job,"registration_sha256":reg_sha,
          "authority_sha256":authority_binding['sha256'],"attempt_sha256":ab['sha256'],"held_readback_sha256":hb['sha256'],"created_utc":(instant+timedelta(seconds=3)).isoformat()})
        rb=put('release_intents',slot,{"protocol":operator.PROTOCOL+'-release-intent',"slot":slot,"job_id":job,"registration_sha256":reg_sha,
          "attempt_sha256":ab['sha256'],"activation_sha256":vb['sha256'],"pre_release_query":qcontrol(fields,instant+timedelta(seconds=3)),
          "created_utc":(instant+timedelta(seconds=4)).isoformat(),"argv":['scontrol','release',job]})
        xb=put('release_actions',slot,{"protocol":operator.PROTOCOL+'-release-action',"slot":slot,"job_id":job,
          "release_intent_sha256":rb['sha256'],"recorded_utc":(instant+timedelta(seconds=5)).isoformat(),"argv":['scontrol','release',job],
          "returncode":0,"stdout":"","stderr":"","invocation_error":None})
        post={**fields,'Reason':'Priority','Priority':'1'}
        put('release_readbacks',slot,{"protocol":operator.PROTOCOL+'-release-readback',"slot":slot,"job_id":job,
          "release_action_sha256":xb['sha256'],"query":qcontrol(post,instant+timedelta(seconds=6))})
        running=operator_test.job(slot,True);running.update(JobId=job,SubmitLine=shlex.join(argv),StartTime=starts[slot].replace(tzinfo=None).isoformat(),
          NodeList='jean-zay-iam35',BatchHost='jean-zay-iam35',StdOut=fields['StdOut'],StdErr=fields['StdErr'])
        put('startup_proofs',slot,{"protocol":operator.PROTOCOL+'-startup-proof',"slot":slot,"job_id":job,"registration_sha256":reg_sha,
          "authority_sha256":authority_binding['sha256'],"attempt_sha256":ab['sha256'],"activation_sha256":vb['sha256'],
          "recorded_utc":(starts[slot]+timedelta(seconds=30)).isoformat(),"actual_host":"jean-zay-iam35","uid":304433,
          "query":qcontrol(running,starts[slot]),"initializer_inventory_sha256":operator.INITIALIZER_SHA,
          "original_script_sha256":operator.SCRIPT_SHA[slot],"operator_sha256":operator_sha,"deadline_check_before_model":True,
          "model_calls_by_guard":0,"execution_deadline_utc":c.EXECUTION_DEADLINE,"resources":r})
    def accounting(slots):
        lines=[]
        for slot in slots:
            a=all_attempts[slot];sft=slot.endswith('.sft');g,cpu,mem=(1,10,'75000M') if sft else (4,40,'300000M')
            start=starts[slot];end=start+timedelta(seconds=durations[slot])
            for suffix in ('','.batch','.extern'):
                req=f'cpu={cpu},gres/gpu={g},mem={mem},node=1' if not suffix else ''
                alloc=f'cpu={2*cpu},gres/gpu={g},mem={mem},node=1'
                lines.append(f"{a['job_id']}{suffix}|COMPLETED|0:0|{durations[slot]}|{start.replace(tzinfo=None).isoformat()}|{end.replace(tzinfo=None).isoformat()}|{req}|{alloc}\n")
        return {'argv':['sacct','--jobs',','.join(all_attempts[s]['job_id'] for s in slots),'--noheader','--parsable2','--format='+c.ACCOUNTING_FORMAT],
                'returncode':0,'stderr':'','stdout':''.join(lines),'recorded_utc':'2026-10-16T01:00:00+00:00','time_zone':'UTC','synthetic_fixture':True}
    if full_results:
        for slot in c.CVRP_ORDER[:extension_count]:
            task,arm=slot.split('.')
            for name,raw in payloads[slot].items():
                native=c.CONTROL+(f'/outputs/{slot}/complete.json' if name=='complete.json' else
                       f'/outputs/{slot}/{task}/{arm}/'+('training/' if name=='actual_prompt_windows.jsonl' else '')+name)
                add(native,f'results/{slot}/{name}',raw)
    census={folder:sorted([{'native_path':r['native_path'],'sha256':r['sha256'],'size_bytes':r['size_bytes']}
                         for local,r in rows.items() if local.startswith('resource_extension/'+folder+'/')],key=lambda r:r['native_path']) for folder in c.CHAIN_FOLDERS}
    extension={'protocol':'sft-anchor-resource-extension-campaign-bundle-v1','synthetic_fixture':True,'records':list(rows.values()),
      'original_accounting_query':accounting(c.OP_ORDER),'extension_accounting_query':accounting(c.CVRP_ORDER[:extension_count]) if extension_count else None,
      'directory_census':{'protocol':'sft-anchor-resource-extension-directory-census-v1','native_root':c.EXTENSION,'reader_hostname':'jean-zay1',
          'reader_uid':304433,'recorded_utc':'2026-10-16T01:00:01+00:00','independent_read_only_collection':True,'folders':census},
      'terminal_artifact_proofs':{s:trees[s] for s in c.CVRP_ORDER[:extension_count]} if full_results else {},
      'native_mutations':False,'superiority_established':False}
    return op_bundle,extension,{'expected_registration_sha256':reg_sha,'expected_amendment_sha256':c.AMENDMENT_SHA256,
         'expected_operator_sha256':operator_sha,'diagnostic_fixture':True}
