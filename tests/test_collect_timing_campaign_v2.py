"""Diagnostic custody/real accounting kernels; no native/model/quality calls."""
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT=Path(__file__).parents[1]
DIR=ROOT/'experiments/rloo_sft_anchor_20261013/resource_deadline_extension_v2'
SOURCE=DIR/'collect_timing_campaign_v2.py'
OLD=DIR.parent/'resource_extension_v1/collect_resource_campaign_v1.py'


def load(path):
    raw=path.read_bytes();m=types.ModuleType('diagnostic_dual_clock_custody')
    m.__file__=str(path);m.__executed_source_sha256__=hashlib.sha256(raw).hexdigest()
    exec(compile(raw,str(path),'exec',dont_inherit=True),m.__dict__);return m


def fixture(count=12,first_state='COMPLETED',last_state='COMPLETED',late_first=False):
    c=load(SOURCE);h=load(DIR/'timing_addendum_v2.py');k=c.resource_kernel(c.binding(OLD))
    # These small fixtures isolate the real allocation/clock bodies.  Separate
    # tests below replay the whole unchanged raw-chain verifier.
    c.strict_chain_admission=lambda ctx,kernel,attempt,started:{'slot':attempt['slot'],'started':started,'synthetic_fixture':True}
    first=(DIR/'legacy_inputs_v1/first_attempt.json').read_bytes();a=json.loads(first)
    assert hashlib.sha256(first).hexdigest()==c.FIRST_ATTEMPT_SHA
    originals=[{'slot':s,'job_id':str(9000000+i)} for i,s in enumerate(c.ORDER[:6])]
    extensions=[a]+[{'slot':s,'job_id':str(9100000+i)} for i,s in enumerate(c.ORDER[7:count],7)]
    attempts=originals+extensions;records={};science_calls=[];chain_calls=[]
    legacy=Path(h.ORIGINAL_ROOT);effective=Path(h.NEW_ROOT);control=Path(h.NAMESPACE)/'control_v2'
    records[str(legacy/'attempts'/(c.FIRST_SLOT+'.json'))]=first
    def read(path):
        if path.endswith('/result.json') or path.endswith('/raw_completions.jsonl'):
            pytest.fail('Protected result/raw opened by metadata phase')
        if path not in records:raise FileNotFoundError(path)
        return records[path]
    oldctx={'extension':legacy,'control':control,'registration_sha256':h.ORIGINAL_REGISTRATION_SHA,
        'registration':{'operator_binding':{'sha256':h.ORIGINAL_OPERATOR_SHA}},'read':read}
    newctx={'extension':effective,'registration_sha256':'e'*64,
        'registration':{'operator_binding':{'sha256':'d'*64}},'read':read}
    ctx={'legacy_context':oldctx,'effective_context':newctx,'legacy_operator':types.SimpleNamespace(),
        'clock_helper':h,'addendum':h.make_proposal(),'read':read,
        'source_composition':{'synthetic_fixture':True,'scope':'invented metadata only'},'synthetic_fixture':True}
    first_admission={'recorded_utc':'2026-10-17T12:00:00+00:00','synthetic_fixture':True}
    records[str(effective/'first_admission.json')]=c.encoded(first_admission)+b'\n'
    query_lines=[];timings={}
    for index,attempt in enumerate(attempts):
        slot=attempt['slot'];job=attempt['job_id'];sft=slot.endswith('.sft');g=1 if sft else 4;cpu=10 if sft else 40
        state=first_state if index==6 else last_state if index==11 else 'COMPLETED'
        start=(datetime(2026,10,15,tzinfo=timezone.utc)+timedelta(seconds=index*30) if index<6 else
               datetime(2026,10,16,tzinfo=timezone.utc) if index==6 else datetime(2026,10,18,tzinfo=timezone.utc)+timedelta(seconds=index*30))
        if index==6 and late_first:start=datetime(2026,10,17,6,tzinfo=timezone.utc)
        elapsed=0 if state=='PENDING' else 10;end=start+timedelta(seconds=elapsed)
        times=(start.strftime('%Y-%m-%dT%H:%M:%S'),end.strftime('%Y-%m-%dT%H:%M:%S'));timings[slot]=times
        requested=f'billing={g},cpu={cpu},gres/gpu={g},mem={cpu*7500}M,node=1'
        allocated=f'billing={1 if sft else 5},cpu={cpu*2},gres/gpu={g},mem={cpu*7500}M,node=1'
        exit_code='0:0' if state in ('PENDING','COMPLETED') else '1:0'
        for suffix in ('',) if state=='PENDING' else ('','.batch','.extern'):
            fields=[job+suffix,state,exit_code,str(elapsed),times[0] if elapsed else 'Unknown',times[1] if elapsed else 'Unknown',requested if not suffix else '',allocated if elapsed else '']
            query_lines.append('|'.join(fields))
        if state=='COMPLETED':
            steps=0 if sft else 100
            marker={'status':'completed_single_arm_model_evaluation','operational_version':2,'superiority_established':False,
                    'checkpoint_reload_exact':True,'actual_RL_optimizer_steps':steps,'actual_RL_responses':steps*36,
                    'actual_evaluation_responses':864,'device':'cuda:0'}
            records[str(control/'outputs'/slot/'complete.json')]=c.encoded(marker)+b'\n'
        if index>=6 and elapsed and not late_first:
            when=h.scheduler_stamp(times[0]);query={'argv':['scontrol','show','job',job],'returncode':0,'stderr':'','time_zone':'UTC',
                'recorded_utc':(when+timedelta(seconds=1)).isoformat(),
                'stdout':f'JobId={job} JobState=RUNNING StartTime={times[0]} TimeLimit='+('03:00:00' if sft else '06:00:00')}
            startup={'query':query,'slot':slot,'job_id':job,'recorded_utc':(when+timedelta(seconds=2)).isoformat(),
                'operator_sha256':h.ORIGINAL_OPERATOR_SHA if index==6 else 'd'*64,
                'registration_sha256':h.ORIGINAL_REGISTRATION_SHA if index==6 else 'e'*64,
                'execution_deadline_utc':h.OLD_DEADLINE if index==6 else h.PROPOSED_DEADLINE,
                'deadline_check_before_model':True,'model_calls_by_guard':0,'resources':h.resource_spec(slot),
                'original_script_sha256':h.SCRIPT_SHA[slot],'attempt_sha256':h.FIRST_ATTEMPT_SHA if index==6 else 'f'*64}
            if index>6:startup.update(first_admission_sha256=c.sha(records[str(effective/'first_admission.json')]),
                                      first_job_admitted_utc=first_admission['recorded_utc'])
            path=(legacy if index==6 else effective)/'startup_proofs'/(slot+'.json');records[str(path)]=c.encoded(startup)+b'\n'
    jobs=[a['job_id'] for a in attempts]
    q={'argv':['sacct','--jobs',','.join(jobs),'--noheader','--parsable2','--format='+k.ACCOUNTING_FORMAT],
       'returncode':0,'stderr':'','time_zone':'UTC','recorded_utc':'2026-10-20T12:00:00+00:00','stdout':'\n'.join(query_lines)+'\n'}
    def query(argv):
        assert argv==q['argv'];return q
    operator=types.SimpleNamespace(ledger=lambda context:(originals,extensions,None if count==12 else c.ORDER[count]),
        validate_attempt_chain=lambda context,attempt:chain_calls.append(attempt['slot']),
        load_first_admission=lambda context:json.loads(read(str(effective/'first_admission.json'))),
        verify_inputs=lambda context:None)
    def science(context,kernel,metadata):
        assert metadata['complete'] is True and len(metadata['markers'])==12
        science_calls.append(True)
        return [{'slot':s} for s in c.ORDER],[],{'historical_actual_GPU_seconds':4988,'synthetic_fixture':True}
    return c,h,k,ctx,operator,query,q,records,science,science_calls,chain_calls,timings


def test_default_cold_import_has_no_effects(tmp_path):
    r=subprocess.run([sys.executable,'-I','-B',str(SOURCE)],cwd=tmp_path,capture_output=True,text=True,check=True)
    assert json.loads(r.stdout)['GPU_calls']==0 and not list(tmp_path.iterdir())


def test_pending_first_reads_no_results_or_science_and_retains177():
    c,h,k,ctx,op,query,q,records,science,calls,chains,t=fixture(7,first_state='PENDING')
    report=c.collect_context(ctx,op,k,query=query,science=science)
    assert calls==[] and report['quality_values_replayed'] is False and report['quality_summary'] is None
    assert report['resource_budget']['cumulative_requested_GPU_hours']==177 and report['resource_budget']['additional_requested_GPU_hours']==0
    assert report['allocation_receipts'][-1]['actual_allocated_GPU_seconds']==0
    assert report['synthetic_fixture'] is True and report['publication_registry_adopted'] is False


def test_late_failed_first_keeps_allocated_cost_and_never_adopts_newclock():
    c,h,k,ctx,op,query,q,records,science,calls,chains,t=fixture(7,first_state='FAILED',late_first=True)
    report=c.collect_context(ctx,op,k,query=query,science=science);first=report['allocation_receipts'][-1]
    assert first['actual_allocated_GPU_seconds']==40 and first['full_wall_clock_admitted'] is False
    assert first['effective_execution_deadline_utc']==h.OLD_DEADLINE and first['forensic_cost_clock_grants_execution'] is False
    assert report['missing_startup_proofs'][0]['slot']==c.FIRST_SLOT and calls==[] and report['quality_values_replayed'] is False


def test_all12_marked_diagnostic_metadata_precedes_science_and_costs_are_complete():
    c,h,k,ctx,op,query,q,records,science,calls,chains,t=fixture()
    report=c.collect_context(ctx,op,k,query=query,science=science)
    assert calls==[True] and len(chains)==12 and len(report['allocation_receipts'])==12
    assert report['status']=='validated_completed_dual_clock_native_campaign' and report['synthetic_fixture'] is True
    assert report['actual_total_GPU_seconds']==4988+sum(r['actual_allocated_GPU_seconds'] for r in report['allocation_receipts'])
    assert report['superiority_established'] is False and report['eligible_for_paper'] is False and report['candidate_selection'] is None
    assert report['metadata_readmitted_after_raw_replay'] is True and 'first_admission' not in ctx


@pytest.mark.parametrize('last_state',['FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED'])
def test_failed_last_slot_retains_cost_and_blocks_science(last_state):
    c,h,k,ctx,op,query,q,records,science,calls,chains,t=fixture(last_state=last_state)
    report=c.collect_context(ctx,op,k,query=query,science=science)
    assert calls==[] and report['quality_values_replayed'] is False
    assert report['allocation_receipts'][-1]['actual_allocated_GPU_seconds']==40 and report['allocation_accounting_complete'] is True


def test_last_marker_failure_precedes_any_result_or_science():
    c,h,k,ctx,op,query,q,records,science,calls,chains,t=fixture();path=str(Path(h.NAMESPACE)/'control_v2/outputs'/c.ORDER[-1]/'complete.json')
    marker=json.loads(records[path]);marker['actual_RL_optimizer_steps']=99;records[path]=c.encoded(marker)
    with pytest.raises(ValueError,match='marker'):c.collect_context(ctx,op,k,query=query,science=science)
    assert calls==[]


def test_allocation_overlap_rejected_before_any_science():
    c,h,k,ctx,op,query,q,records,science,calls,chains,t=fixture()
    original_start,original_end=t[c.ORDER[-1]];previous_start,previous_end=t[c.ORDER[-2]]
    q['stdout']=q['stdout'].replace(original_start,previous_start).replace(original_end,previous_end)
    with pytest.raises(ValueError,match='chronology'):c.collect_context(ctx,op,k,query=query,science=science)
    assert calls==[]


def test_missing_completed_startup_reports_cost_but_no_engineering_admission():
    c,h,k,ctx,op,query,q,records,science,calls,chains,t=fixture()
    del records[str(Path(h.NEW_ROOT)/'startup_proofs'/(c.ORDER[-1]+'.json'))]
    report=c.collect_context(ctx,op,k,query=query,science=science)
    assert report['allocation_accounting_complete'] is True and len(report['missing_startup_proofs'])==1
    assert report['quality_values_replayed'] is False and calls==[]


@pytest.mark.parametrize('fault',['missing_gpu','missing_step_allocation','conflicting_typed_gpu','duplicate_tres'])
def test_allocation_ambiguity_never_becomes_zero(fault):
    c,h,k,ctx,op,query,q,records,science,calls,chains,t=fixture(7)
    lines=q['stdout'].splitlines();parts=lines[-3].split('|')
    if fault=='missing_gpu':parts[-1]=parts[-1].replace('gres/gpu=4,','')
    elif fault=='conflicting_typed_gpu':parts[-1]+=',gres/gpu:a100=3'
    elif fault=='duplicate_tres':parts[-1]+=',gres/gpu=4'
    else:
        parts=lines[-2].split('|');parts[-1]='';lines[-2]='|'.join(parts)
    if fault!='missing_step_allocation':lines[-3]='|'.join(parts)
    q['stdout']='\n'.join(lines)+'\n'
    with pytest.raises(ValueError):c.collect_context(ctx,op,k,query=query,science=science)
    assert calls==[]


def test_typed_gpu_allocation_counts_positive_failure_cost():
    c,h,k,ctx,op,query,q,records,science,calls,chains,t=fixture(7,first_state='FAILED')
    q['stdout']=q['stdout'].replace('gres/gpu=4','gres/gpu:a100=4')
    report=c.collect_context(ctx,op,k,query=query,science=science)
    assert report['allocation_receipts'][-1]['actual_allocated_GPU_seconds']==40 and calls==[]


def test_original_first_attempt_cannot_be_replaced():
    c,h,k,ctx,op,query,q,records,science,calls,chains,t=fixture(7)
    path=str(Path(h.ORIGINAL_ROOT)/'attempts'/(c.FIRST_SLOT+'.json'));records[path]+=b' '
    with pytest.raises(ValueError,match='first attempt bytes'):c.collect_context(ctx,op,k,query=query,science=science)
    assert calls==[]


def test_unregistered_native_operator_rejected_before_compilation(monkeypatch):
    c=load(SOURCE);row={'path':str(c.NEW_ROOT/'registration.json'),'sha256':'a'*64,'size_bytes':1}
    registration={'operator_binding':{'path':str(c.NEW_ROOT/'cvrp_timing_operator_v2.py'),'sha256':'b'*64,'size_bytes':1}}
    monkeypatch.setattr(c,'read_bound',lambda binding:c.encoded(registration))
    monkeypatch.setattr(c,'compile_buffer',lambda *args:pytest.fail('Unregistered native code executed'))
    with pytest.raises(ValueError,match='independently closed'):c.collect_native(row)


def test_scientific_phase_itself_rejects_partial_before_loading_kernels():
    c=load(SOURCE)
    with pytest.raises(ValueError,match='All12'):c.scientific_phase({},None,{'complete':False,'markers':{}})


def test_future_attempt_requires_explicit_first_admission_loader_without_context_cache():
    c,h,k,ctx,op,query,q,records,science,calls,chains,t=fixture()
    assert 'first_admission' not in ctx
    op.load_first_admission=lambda context:(_ for _ in ()).throw(ValueError('Actual first closure absent'))
    with pytest.raises(ValueError,match='Actual first closure'):c.collect_context(ctx,op,k,query=query,science=science)
    assert calls==[]


def test_first_admission_loader_and_captured_raw_must_be_identical():
    c,h,k,ctx,op,query,q,records,science,calls,chains,t=fixture()
    op.load_first_admission=lambda context:{'recorded_utc':'2026-10-18T00:00:00+00:00'}
    with pytest.raises(ValueError,match='changed during capture'):c.collect_context(ctx,op,k,query=query,science=science)
    assert calls==[]


@pytest.mark.parametrize('fault',['marker','attempt','startup','first_admission','source'])
def test_after_science_metadata_source_and_markers_are_readmitted(fault):
    c,h,k,ctx,op,query,q,records,science,calls,chains,t=fixture()
    def replay(context,kernel,metadata):
        result=science(context,kernel,metadata)
        if fault=='source':
            op.verify_inputs=lambda context:(_ for _ in ()).throw(ValueError('Operational source changed'))
        elif fault=='attempt':
            path=str(Path(h.ORIGINAL_ROOT)/'attempts'/(c.FIRST_SLOT+'.json'));records[path]+=b' '
        elif fault=='marker':
            path=str(Path(h.NAMESPACE)/'control_v2/outputs'/c.ORDER[0]/'complete.json')
            records[path]=records[path].rstrip()+b'\n\n'
        elif fault=='first_admission':
            path=str(Path(h.NEW_ROOT)/'first_admission.json');records[path]+=b' '
        else:
            path=str(Path(h.NEW_ROOT)/'startup_proofs'/(c.ORDER[-1]+'.json'));records[path]+=b' '
        return result
    with pytest.raises(ValueError):c.collect_context(ctx,op,k,query=query,science=replay)
    assert calls==[True]


@pytest.fixture(scope='module')
def actual_scientific_inputs():
    """Real frozen source/contracts only; never actual model outcome files."""
    c=load(SOURCE);k=c.resource_kernel(c.binding(OLD));actual_loader=k.load_kernels
    k.load_kernels=lambda unused:actual_loader(DIR.parent/'admission_prepared_v1/collect_campaign_admission_v3.py')
    root=DIR.parent/'source_v2';raw=(root/'source_manifest.json').read_bytes()
    manifest=json.loads(raw);native={str(c.SOURCE/'source_manifest.json'):raw}
    native.update({str(c.SOURCE/row['path']):(root/row['path']).read_bytes() for row in manifest['files']})
    launch=(DIR.parent/'control_v2/launch_manifest.json').read_bytes();launch_path=str(c.CONTROL/'launch_manifest.json');native[launch_path]=launch
    def read(path):
        if path not in native:raise FileNotFoundError(path)
        return native[path]
    ctx={'legacy_operator':types.SimpleNamespace(read_bound=c.read_bound),
         'legacy_context':{'read':read,'registration':{'input_bindings':{'launch_manifest':c.binding(Path(launch_path),launch)}}}}
    return c,k,ctx,native


def test_all121_real_frozen_sources_supply_exact_scoring_contract_and_prior_aliases(actual_scientific_inputs):
    c,k,ctx,native=actual_scientific_inputs
    kernels,manifest,entries,decoded,rows=c.scientific_sources(ctx,k)
    assert len(entries)==121 and len(rows)==123 and kernels.__executed_source_sha256__==c.KERNEL_SHA
    scorer=kernels.load_frozen_scorer(decoded,entries,manifest)
    for task in ('op','cvrp'):
        contract=scorer.validate_contract(json.loads(decoded['source/contracts/'+task+'.json'][1]))
        assert len(contract['training_rows'])==128 and len(contract['validation_rows'])==96 and contract['training']['horizon']==100
    assert decoded['source/frozen_scorer.py'][0]['size_bytes']==entries[kernels.SCORER]['size_bytes']
    assert decoded['source/prior_failed_phase.json'][0]['sha256']==manifest['prior_failed_phase']['proof']['sha256']
    kernels.validate_prior_failure(json.loads(decoded['source/prior_failed_phase.json'][1]))


@pytest.mark.parametrize('fault',['scorer','contract','prior','manifest'])
def test_actual_source_layer_rejects_byte_drift_before_any_result(actual_scientific_inputs,fault):
    c,k,ctx,native=actual_scientific_inputs
    copied=dict(native);source_manifest=json.loads(copied[str(c.SOURCE/'source_manifest.json')])
    launch=json.loads(copied[str(c.CONTROL/'launch_manifest.json')])
    name=('src/clasp/trainers/sft_anchored_rloo.py' if fault=='scorer' else
          launch['contracts']['op']['path'] if fault=='contract' else launch['prior_failed_phase']['proof']['path'])
    target=str(c.SOURCE/('source_manifest.json' if fault=='manifest' else name));copied[target]+=b' '
    changed={**ctx,'legacy_context':{**ctx['legacy_context'],'read':lambda path:copied[path]}}
    with pytest.raises(ValueError,match='source'):c.scientific_sources(changed,k)


def test_real_frozen_scoring_replays_all12_synthetic_slots_but_public_science_rejects_synthetic(actual_scientific_inputs):
    from test_sft_anchor_admission_v3_collection import _durable_slot
    c,k,ctx,native=actual_scientific_inputs;kernels,manifest,entries,decoded,rows=c.scientific_sources(ctx,k)
    scorer=kernels.load_frozen_scorer(decoded,entries,manifest)
    contracts={t:scorer.validate_contract(json.loads(decoded['source/contracts/'+t+'.json'][1])) for t in ('op','cvrp')}
    for slot in c.ORDER:
        payload,_=_durable_slot(slot,contracts[slot.split('.')[0]],manifest,entries[kernels.ENTRYPOINT])
        marker=json.loads(payload['complete.json']);result=json.loads(payload['result.json'])
        admitted=kernels.validate_completed_slot(slot,marker,result,contracts[slot.split('.')[0]],manifest,entries,scorer)
        assert admitted['raw_result_replay_exact'] is True
        with pytest.raises(ValueError,match='Synthetic|synthetic'):k.no_synthetic(result)


@pytest.fixture(scope='module')
def full_raw_chain_fixture():
    import base64
    from resource_extension_collection_fixture_v1 import fixture_bundle
    c=load(SOURCE);k=c.resource_kernel(c.binding(OLD));_,ext,kwargs=fixture_bundle(extension_count=1,full_results=False)
    data={r['native_path']:base64.b64decode(r['content_base64']) for r in ext['records']}
    root=c.NAMESPACE/'resource_extension_v1';reg=json.loads(data[str(root/'registration.json')]);authority=json.loads(data[str(root/'authority.json')])
    old=load(DIR.parent/'resource_extension_v1/cvrp_wall_resource_v1.py')
    def context(values):
        parent={'extension':root,'registration':reg,'authority':authority,'authority_sha256':c.sha(values[str(root/'authority.json')]),
                'registration_sha256':kwargs['expected_registration_sha256'],'read':lambda path:values[path]}
        return {'legacy_operator':old,'legacy_context':parent}
    attempt=json.loads(data[str(root/'attempts'/(c.FIRST_SLOT+'.json'))])
    return c,k,data,context,attempt


def test_whole_frozen_raw_chain_body_admits_diagnostic_positive_without_numerical_changes(full_raw_chain_fixture):
    c,k,data,context,attempt=full_raw_chain_fixture
    proof=c.strict_chain_admission(context(data),k,attempt,started=True)
    assert proof['held_chain_verified'] is True and proof['startup_verified'] is True
    assert proof['captured_capacity_queries_replayed'] is True
    assert proof['original_custody_source_sha256']==c.OLD_COLLECTOR_SHA and len(proof['captured_file_bindings'])==10


@pytest.mark.parametrize('fault',['startup_init','startup_UID','startup_authority','startup_cpu','release_rpc','activation','intent_source'])
def test_whole_original_raw_chain_rejects_actual_schema_mutations(full_raw_chain_fixture,fault):
    c,k,data,context,attempt=full_raw_chain_fixture;data=dict(data);root=c.NAMESPACE/'resource_extension_v1'
    folder=('startup_proofs' if fault.startswith('startup') else 'release_actions' if fault=='release_rpc' else
            'activations' if fault=='activation' else 'intents')
    path=str(root/folder/(c.FIRST_SLOT+'.json'));value=json.loads(data[path])
    if fault=='startup_init':value['initializer_inventory_sha256']='0'*64
    elif fault=='startup_UID':value['uid']=304433.0
    elif fault=='startup_authority':value['authority_sha256']='0'*64
    elif fault=='startup_cpu':value['query']['stdout']=value['query']['stdout'].replace('NumCPUs=80','NumCPUs=40')
    elif fault=='release_rpc':value['returncode']=False
    elif fault=='activation':value['attempt_sha256']='0'*64
    else:value['operator_sha256']='0'*64
    data[path]=c.encoded(value)
    with pytest.raises(ValueError):c.strict_chain_admission(context(data),k,attempt,started=True)


def test_captured_capacity_is_replayed_independently_from_full_chain(full_raw_chain_fixture,monkeypatch):
    c,k,data,context,attempt=full_raw_chain_fixture;seen=[];original=k.replay_capacity
    def replay(intent,operator):
        seen.append(intent['slot']);return original(intent,operator)
    monkeypatch.setattr(k,'replay_capacity',replay)
    c.strict_chain_admission(context(data),k,attempt,started=True)
    assert seen==[c.FIRST_SLOT]


@pytest.mark.parametrize('fault',['query_bytes','available_boolean','query_argv','query_time'])
def test_actual_captured_capacity_replay_rejects_rebound_queries(full_raw_chain_fixture,fault):
    c,k,data,context,attempt=full_raw_chain_fixture
    path=str(c.NAMESPACE/'resource_extension_v1/intents'/(c.FIRST_SLOT+'.json'))
    intent=json.loads(data[path]);capacity=intent['capacity']
    if fault=='query_bytes':capacity['policy_query']['stdout']='qos_gpu_a100-t3|20:00:00|1|cpu=1,gres/gpu=1|1.000000\n'
    elif fault=='available_boolean':capacity['available']=1
    elif fault=='query_argv':capacity['rows_query']['argv']=['squeue','--jobs','unbound']
    else:capacity['association_query']['recorded_utc']='2026-10-25T00:00:00+00:00'
    with pytest.raises(ValueError):k.replay_capacity(intent,context(data)['legacy_operator'])
