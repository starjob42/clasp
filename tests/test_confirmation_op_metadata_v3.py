"""Synthetic all-slot custody/strict-replay guards; no policies or models."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/rloo_one_shot_20261016/analyze_confirmation_op_v3.py'
ENTRY = ROOT / 'experiments/rloo_one_shot_20261016/run_confirmation_7b_v3.py'
BASE = ROOT / 'experiments/rloo_one_shot_20261016/execution_v1/source/run_local_7b_v1.py'
SCORER = ROOT / 'src/clasp/baselines/sft_anchor_pilot.py'


def load(path, name):
    raw = path.read_bytes();m = types.ModuleType(name);m.__file__ = str(path)
    m.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    exec(compile(raw, str(path), 'exec', dont_inherit=True), m.__dict__)
    return m


@pytest.fixture
def C():
    return load(SOURCE, 'confirmation_metadata_guard_fixture')


def store(C, path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(C.encoded(value)+b'\n')
    return C.file_binding(path)


@pytest.fixture(params=["pure", "teacher"])
def packet(C, tmp_path, request):
    """Invented all28 metadata; fake weight bytes, never real admission."""
    E = load(ENTRY, 'diagnostic_fixed_roster_only')
    branch = request.param
    slots = E.fixed_slots(branch, tmp_path)
    authority = {'branch': branch, 'execution_root': str(tmp_path), 'slots': slots,
                 'maximum_requested_GPU_hours': 102.5 if branch == 'pure' else 182.5,
                 'execution_deadline_utc': '2026-10-30T12:00:00+00:00'}
    ab = store(C, tmp_path/'authority.json', authority)
    initializer = [{'path': '/diagnostic/fake_initializer', 'sha256': 'a'*64, 'size_bytes': 1}]
    request = {'authority': ab, 'slots': []}
    for slot in slots:
        out = Path(slot['output']);out.mkdir(parents=True)
        cb = store(C, tmp_path/'contracts'/(slot['slot']+'.json'), {'authority': ab, 'slot': slot['slot']})
        record = {'instance_id': 'placeholder_only_no_quality', 'synthetic_fixture': True}
        raw = out/'raw_completions.jsonl';raw.write_bytes(C.encoded(record)+b'\n')
        ib = store(C, out/'input_admission.json', {'initializer_files': initializer})
        result = {'contract_file_binding': cb, 'authority_binding': ab, 'slot': slot,
                  'input_admission_binding': ib, 'raw_completions_binding': C.file_binding(raw),
                  'trained_updates': slot['optimizer_updates'], 'RL_responses': slot['RL_responses'],
                  'teacher_forwards': slot['teacher_forwards'], 'evaluation_responses': slot['evaluation_rows'],
                  'resources': {'training_wall_seconds': 1., 'evaluation_wall_seconds': 2.,
                                'worker_wall_seconds': 4., 'peak_allocated_bytes': 100, 'peak_reserved_bytes': 200}}
        rb = store(C, out/'result.json', result)
        completion = {'status': 'completed', 'parent_elapsed_wall_seconds': 5.}
        pb = store(C, out/'completion.json', completion)
        store(C, out/'attempt.json', {'contract_file_binding': cb, 'authority_binding': ab, 'slot': slot,
              'maximum_wall_seconds': slot['wall_seconds'], 'requested_gpu_hours': slot['wall_seconds']/3600,
              'automatic_retry': False, 'recorded_utc': '2026-10-20T12:00:00+00:00'})
        request['slots'].append({'slot': slot['slot'], 'contract': cb, 'output': str(out), 'result': rb, 'completion': pb})
    verified = []
    def verify_completed(cb, output):
        verified.append(str(output))
        r = json.loads(C.read_bound(C.file_binding(Path(output)/'result.json')))
        p = json.loads(C.read_bound(C.file_binding(Path(output)/'completion.json')))
        C.need(p['status'] == 'completed', 'Failed synthetic endpoint')
        return {'status': 'complete_confirmation_endpoint_metadata', 'contract_binding': cb,
                'slot': r['slot']['slot'], 'output': str(output), 'result': r, 'completion': p,
                'result_binding': C.file_binding(Path(output)/'result.json'),
                'completion_binding': C.file_binding(Path(output)/'completion.json'), 'quality_analyzed': False}
    init_calls = []
    def verify_init(anchor):
        init_calls.append(True);return initializer
    runner = types.SimpleNamespace(verify_completed=verify_completed)
    context = {'authority': authority, 'anchor': {}, 'base': types.SimpleNamespace(verify_initializers=verify_init)}
    reads = []
    def metadata_read(row):
        reads.append(row['path'])
        assert not row['path'].endswith('/raw_completions.jsonl'), 'Raw response opened before full metadata admission'
        return C.read_bound(row)
    return request, runner, context, slots, metadata_read, verified, init_calls, reads


def change_metadata(C, packet, index, filename, mutate):
    request = packet[0];row = request['slots'][index];path = Path(row['output'])/filename
    value = json.loads(path.read_bytes());mutate(value);b = store(C, path, value)
    if filename == 'result.json':row['result'] = b
    if filename == 'completion.json':row['completion'] = b


def test_all28_metadata_first_no_raw_and_initializer_stream_once(C, packet):
    request, runner, context, slots, read, verified, init_calls, reads = packet
    admitted, result = C.admit_all_metadata(request, runner, context, slots, read)
    assert len(admitted) == len(verified) == result['endpoint_count'] == len(slots)
    assert len(init_calls) == 1 and result['quality_read'] is False
    assert result['requested_GPU_hours'] == (102.5 if len(slots)==28 else 182.5)
    assert all(not p.endswith('raw_completions.jsonl') for p in reads)
    assert result.get('scoped_confirmation_established', False) is False


def test_failed_last_parent_denied_before_any_raw(C, packet):
    change_metadata(C, packet, -1, 'completion.json', lambda p:p.update(status='failed'))
    with pytest.raises(ValueError, match='Failed synthetic endpoint'):
        C.admit_all_metadata(*packet[:4], packet[4])
    assert len(packet[5]) == len(packet[3]) and all(not p.endswith('raw_completions.jsonl') for p in packet[7])


@pytest.mark.parametrize('fault', ['source_authority', 'slot', 'boolean_count', 'count', 'synthetic',
                                  'training_wall', 'evaluation_wall', 'worker_wall', 'memory', 'memory_bool'])
def test_corrupted_endpoint_metadata_rejects_before_raw(C, packet, fault):
    def mutate(value):
        if fault == 'source_authority':value['authority_binding'] = dict(value['authority_binding'],sha256='a'*64)
        elif fault == 'slot':value['slot'] = dict(value['slot'],seed=73)
        elif fault == 'boolean_count':value['RL_responses'] = True
        elif fault == 'count':value['trained_updates'] = 99
        elif fault == 'synthetic':value['synthetic_fixture'] = True
        elif fault == 'training_wall':value['resources']['training_wall_seconds'] = 4.
        elif fault == 'evaluation_wall':value['resources']['evaluation_wall_seconds'] = 0
        elif fault == 'worker_wall':value['resources']['worker_wall_seconds'] = 6.
        elif fault == 'memory':value['resources']['peak_reserved_bytes'] = 49*1024**3
        else:value['resources']['peak_allocated_bytes'] = True
    change_metadata(C, packet, -1, 'result.json', mutate)
    with pytest.raises(ValueError): C.admit_all_metadata(*packet[:4], packet[4])
    assert not any(p.endswith('raw_completions.jsonl') for p in packet[7])


@pytest.mark.parametrize('fault', ['authority', 'contract', 'wall', 'retry', 'naive_utc', 'late', 'synthetic'])
def test_first_attempt_cannot_be_rebound_or_retried(C, packet, fault):
    def mutate(value):
        if fault in ('authority','contract'):value[fault+'_binding' if fault=='authority' else 'contract_file_binding']['sha256'] = 'a'*64
        elif fault == 'wall':value['maximum_wall_seconds'] = 14401
        elif fault == 'retry':value['automatic_retry'] = True
        elif fault == 'naive_utc':value['recorded_utc'] = '2026-10-20T12:00:00'
        elif fault == 'late':value['recorded_utc'] = '2026-10-30T12:00:01+00:00'
        else:value['synthetic_fixture'] = True
    change_metadata(C, packet, -1, 'attempt.json', mutate)
    with pytest.raises(ValueError):C.admit_all_metadata(*packet[:4], packet[4])


def test_changed_initializer_inventory_rejected_before_raw(C, packet):
    change_metadata(C, packet, -1, 'input_admission.json', lambda p:p.update(initializer_files=[]))
    # The result still binds the original bytes, hence also catches a raw disk mutation.
    with pytest.raises(ValueError):C.admit_all_metadata(*packet[:4], packet[4])


@pytest.mark.parametrize('fault', ['extra_run','missing_run','symlink'])
def test_exact_run_census_prevents_hidden_or_missing_attempts(C, packet, fault):
    root = Path(packet[2]['authority']['execution_root'])/'runs'
    if fault == 'extra_run':(root/'unregistered_retry').mkdir()
    elif fault == 'symlink':(root/'unregistered_link').symlink_to(next(root.iterdir()),target_is_directory=True)
    else:
        row = packet[0]['slots'][-1]
        Path(row['output']).rename(Path(row['output']).with_name('hidden'))
    with pytest.raises(ValueError):C.admit_all_metadata(*packet[:4], packet[4])
    assert not packet[5]


def test_verified_module_executes_buffer_not_bytecode(C, tmp_path):
    path = tmp_path/'module.py';path.write_text('VALUE="wrong disk"')
    raw = b'VALUE="bound buffer"\n'
    row = {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}
    m = C.compile_buffer(raw,row,'exact_buffer_fixture')
    assert m.VALUE == 'bound buffer' and m.__executed_source_sha256__ == row['sha256']
    with pytest.raises(ValueError):C.compile_buffer(b'VALUE="other"',row,'wrong')


def test_bound_metadata_refuses_symlinks_mutation_and_bool_sizes(C, tmp_path):
    row = store(C,tmp_path/'one.json',{})
    with pytest.raises(ValueError):C.read_bound(dict(row,size_bytes=True))
    alias = tmp_path/'alias';alias.symlink_to(Path(row['path']))
    with pytest.raises(ValueError):C.read_bound(dict(row,path=str(alias)))
    Path(row['path']).write_text('changed')
    with pytest.raises(ValueError):C.read_bound(row)


def physical_fixture(C):
    scorer = load(SCORER,'actual_independent_scorer_fixture')
    candidates = [ROOT/'experiments/rloo_one_shot_20261016/run_local_7b_v1.py',BASE]
    base = load(next(p for p in candidates if p.is_file()),'actual_frozen_raw_token_validator')
    def row(i):
        band = ((10,30),(40,60),(70,100))[i%3];n = band[0]
        instance = [[[j, i] for j in range(n)],[0]+[1]*(n-1),1000]
        return {'task_type':'op','instance_id':'synthetic-input-'+str(i), 'instance_json':instance,
                'semantic_id':scorer.semantic_id('op',instance),'graph_semantic_id':scorer.graph_semantic_id('op',instance),
                'development_band':list(band),'prompt':'synthetic prompt '+str(i),'output':'Route: [0, 1]'}
    rows = [row(i) for i in range(256)]
    tokenizer = type('DiagnosticTokenizer',(),{'eos_token_id':2,'decode':lambda self,ids,skip_special_tokens=True:'Route: [0, 1]', '__len__':lambda self:3})()
    core = types.SimpleNamespace(generation_config=lambda *a,**kw:types.SimpleNamespace(to_dict=lambda:{}),score_completion=scorer.score_completion)
    records = [{'instance_id':r['instance_id'],'semantic_id':r['semantic_id'],'prompt_sha256':hashlib.sha256(r['prompt'].encode()).hexdigest(),
                'completion_token_ids':[1,2], 'text':'Route: [0, 1]', 'generated_tokens':2,'eos_observed':True,
                'truncated':False,'actual_generation_config':{},'use_model_defaults':False,'extra_generation_kwargs':{},
                'generation_calls':1,'sample_index':0,'seed':101,'wall_seconds':.1,
                'strict_score':scorer.score_completion('op','Route: [0, 1]',r['instance_json'],r['output'])} for r in rows]
    return scorer,base,tokenizer,core,rows,records


def test_physical_totals_strata_reference_and_actual_strict_token_helper(C):
    scorer,base,tokenizer,core,rows,records = physical_fixture(C)
    cohort = C.cohort_metadata(rows,scorer)
    scores,work = C.replay_vector(records,rows,seed=101,base=base,tokenizer=tokenizer,core=core,scorer=scorer)
    assert len(scores) == 256 and work['actual_greedy_calls'] == 256 and work['generated_tokens'] == 512
    assert [len(g) for g in C.validate_cohort(cohort)] == [86,85,85]
    assert cohort[0]['total_prize'] == 9 and cohort[0]['reference_objective'] == 1
    assert work['training_or_eval_policy_calls_by_analyzer'] == 0


@pytest.mark.parametrize('fault', ['seed_bool','seed73','sample_bool','second_sample','score','hidden_processor','prompt','EOS','synthetic'])
def test_raw_token_physics_sampler_and_scope_mutations_denied(C,fault):
    scorer,base,tokenizer,core,rows,records = physical_fixture(C)
    r=records[-1]
    if fault=='seed_bool':r['seed']=True
    elif fault=='seed73':r['seed']=73
    elif fault=='sample_bool':r['sample_index']=False
    elif fault=='second_sample':r['sample_index']=1
    elif fault=='score':r['strict_score']['objective']=2
    elif fault=='hidden_processor':r['actual_generation_config']['repetition_penalty']=2.
    elif fault=='prompt':r['prompt_sha256']='a'*64
    elif fault=='EOS':r['completion_token_ids']=[2,1]
    else:r['synthetic_fixture']=True
    with pytest.raises(ValueError):C.replay_vector(records,rows,seed=101,base=base,tokenizer=tokenizer,core=core,scorer=scorer)


def test_complete_original_numerical_function_AST_unchanged(C):
    archived = ROOT/'experiments/rloo_one_shot_20261016/validation/confirmation_op_statistics_v3/analyze_confirmation_op_v3.py'
    old=ast.parse(archived.read_bytes());new=ast.parse(SOURCE.read_bytes())
    names=[n.name for n in old.body if isinstance(n,ast.FunctionDef) and n.name not in ('main','analyze_confirmation_op')]
    oldmap={n.name:ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,ast.FunctionDef)}
    newmap={n.name:ast.dump(n,include_attributes=False) for n in new.body if isinstance(n,ast.FunctionDef)}
    assert all(oldmap[name]==newmap[name] for name in names)
    for key in ('SEEDS','PURE','TEACHER','STRATUM_COUNTS','BOOTSTRAP_SEED','BOOTSTRAP_REPLICATES','FEASIBILITY_MARGIN','ALPHA'):
        oldrow=[n for n in old.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in n.targets)][0]
        newrow=[n for n in new.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in n.targets)][0]
        assert ast.dump(oldrow,include_attributes=False)==ast.dump(newrow,include_attributes=False)


def schema_fixture(C, packet):
    request,runner,context,slots,*_ = packet
    root=Path(context['authority']['execution_root'])
    entry=C.file_binding(ENTRY)
    authority=context['authority']
    authority.update(sources={'run_confirmation_7b_v3.py':entry},plan={'sha256':C.PLAN_SHA256},
                     inputs={'evaluation':{'sha256':C.COHORT_SHA256},'admission':{'sha256':C.INPUT_ADMISSION_SHA256}})
    ab=store(C,root/'authority_for_schema.json',authority)
    value={'protocol':C.REQUEST_PROTOCOL,'entry_source':entry,'authority':ab,'slots':request['slots']}
    value['receipt_sha256']=C.digest(value)
    E=load(ENTRY,'schema_actual_fixed_slots')
    delegate=types.SimpleNamespace(validate_authority=lambda a:dict(context,authority=a),fixed_slots=E.fixed_slots)
    return value,delegate


def test_exact_request_schema_for_all28_and48_before_raw(C,packet):
    request,runner=schema_fixture(C,packet)
    context,slots=C.request_schema(request,runner)
    assert len(slots) == len(packet[3])
    assert context['authority']['branch'] in ('pure','teacher')


@pytest.mark.parametrize('fault',['missing_qual','missing_sft','missing_study','reorder','duplicate','escape','extra_field','bad_seal'])
def test_request_cannot_drop_or_substitute_any_engineering_baseline_or_study_slot(C,packet,fault):
    request,runner=schema_fixture(C,packet)
    if fault=='missing_qual':request['slots'].pop(0)
    elif fault=='missing_sft':request['slots'].pop(2)
    elif fault=='missing_study':request['slots'].pop()
    elif fault=='reorder':request['slots'][0],request['slots'][1]=request['slots'][1],request['slots'][0]
    elif fault=='duplicate':request['slots'][-1]=copy.deepcopy(request['slots'][0])
    elif fault=='escape':request['slots'][-1]['result']['path']=str(Path(request['slots'][-1]['output']).parent/'wrong.json')
    elif fault=='extra_field':request['selected_after_peeking']=True
    else:request['receipt_sha256']='a'*64
    if fault!='bad_seal':request['receipt_sha256']=C.digest({k:v for k,v in request.items() if k!='receipt_sha256'})
    with pytest.raises(ValueError):C.request_schema(request,runner)


def test_public_orchestration_last_metadata_failure_cannot_load_tokenizer_or_raw(C,packet,monkeypatch,tmp_path):
    request,runner,context,slots,read,*_=packet
    # Deliberate test-only substitution supplies metadata context; no successful
    # report can be emitted, and no fabricated actual7B result is constructed.
    common=store(C,tmp_path/'closed_metadata_only.json',{})
    context['authority'].update(plan=common,master_manifest=common,master_packet=common,original_roster=common,selection_receipt=common,master_logical=common,sources={},inputs={})
    context.update(roster={'source_manifest':common},manifest={'extra_sources':{}},buffers={})
    request.update(entry_source=C.file_binding(ENTRY))
    change_metadata(C,packet,-1,'completion.json',lambda v:v.update(status='failed'))
    rb=store(C,tmp_path/'diagnostic_request.json',request)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    real_compile=C.compile_buffer
    monkeypatch.setattr(C,'compile_buffer',lambda raw,b,name:runner if name=='confirmation_closed_metadata_entry' else real_compile(raw,b,name))
    monkeypatch.setattr(C,'request_schema',lambda r,m:(context,slots))
    monkeypatch.setattr(C,'production_replay_context',lambda *a:pytest.fail('Tokenizer/raw phase entered before all metadata'))
    rawpaths={s['output']+'/raw_completions.jsonl' for s in slots}
    original=C.read_bound
    def closed(row):
        assert row['path'] not in rawpaths,'Fresh raw opened before final metadata failure'
        return original(row)
    monkeypatch.setattr(C,'read_bound',closed)
    with pytest.raises(ValueError,match='Failed synthetic endpoint'):C.analyze_confirmation_op(rb)


def registration_fixture(C, packet, monkeypatch):
    """Pure fabricated custody, never admitted by the fixed production entry."""
    request,runner,context,slots,read,*_=packet
    root=Path(context['authority']['execution_root'])
    common=store(C,root/'synthetic_history_only.json',{'scope':'synthetic metadata bytes; no model'})
    historical=C.file_binding(ROOT/'experiments/rloo_one_shot_20261016/validation/confirmation_campaign_v3/historical_inputs_v1/historical_native_authority.json')
    assert historical['sha256']==C.HISTORICAL_NATIVE_AUTHORITY_SHA256
    a=context['authority'];a.update(sources={'run_confirmation_7b_v3.py':C.file_binding(ENTRY)},
        inputs={'training':common,'evaluation':common,'admission':common},selection_receipt=common,
        original_roster=common,master_packet=common)
    ab=store(C,root/'authority.json',a);request['authority']=ab
    students={}
    for seed in C.SEEDS:
        student={'train_seed':seed};student['contract_sha256']=C.digest(student)
        students[str(seed)]=store(C,root/'students'/(str(seed)+'.json'),student)
    for i,row in enumerate(request['slots']):
        slot=slots[i];out=Path(row['output'])
        contract={'protocol':'one-shot-confirmation-OP-7b-execution-v3','authority':ab,'slot':slot['slot'],
                  'student_logical':students[str(slot['seed'])]}
        contract['contract_sha256']=C.digest(contract)
        row['contract']=store(C,Path(row['contract']['path']),contract)
        attempt=json.loads((out/'attempt.json').read_bytes())
        attempt.update(contract_file_binding=row['contract'],authority_binding=ab,
                       recorded_utc='2026-10-20T12:00:02+00:00' if slot['mode']=='qualification' else '2026-10-20T12:00:10+00:00')
        store(C,out/'attempt.json',attempt)
        result=json.loads((out/'result.json').read_bytes())
        result.update(contract_file_binding=row['contract'],authority_binding=ab,
            training_complete_binding=store(C,out/'training_complete.json',{'scope':'fake terminal proof bytes only'}),
            checkpoint_reload_binding=store(C,out/'checkpoint_reload_exact.json',{'scope':'fake reload proof bytes only'}))
        row['result']=store(C,out/'result.json',result)
    admitted,metadata=C.admit_all_metadata(request,runner,context,slots,read)
    resources={'label':'REGISTERED_MAXIMA_INCLUDING_UNLAUNCHED_NOT_ACTUAL_SPEND','historical_requested_GPU_hours':239.5,
               'historical_sources':{'historical_native_authority':historical,'original_roster':common,'master_packet':common},
               'selected_branch':a['branch'],'new_requested_GPU_hours':a['maximum_requested_GPU_hours'],
               'combined_requested_GPU_hours':239.5+a['maximum_requested_GPU_hours'],
               'failed_and_unlaunched_reservations_reclaimed':False,'actual_spent_GPU_hours':None}
    def seal(value,field):return dict(value,**{field:C.digest(value)})
    roster=seal({'protocol':C.CAMPAIGN_PROTOCOL+'-execution-roster','branch':a['branch'],'authority':ab,
        'selection_receipt':common,'slots':[dict(s,contract=r['contract']) for s,r in zip(slots,request['slots'])],
        'automatic_retry':False,'confirmation_policy_calls_at_registration':0,'resources':resources},'roster_sha256')
    rb=store(C,root/'execution_roster.json',roster)
    initial=seal({'protocol':C.CAMPAIGN_PROTOCOL+'-prepolicy-registration','recorded_utc':'2026-10-20T12:00:01+00:00',
        'authority':ab,'execution_roster':rb,'sources':a['sources'],'student_logicals':students,'inputs':a['inputs'],
        'selection_receipt':common,'confirmation_policy_calls':0,'resources':resources},'receipt_sha256')
    ib=store(C,root/'prepolicy_registration.json',initial)
    quals=[]
    for item in admitted[:2]:
        row=item['request_slot'];out=Path(row['output'])
        quals.append({'slot':row['slot'],'contract':row['contract'],'output':row['output'],
            'metadata_admission':store(C,root/'control'/(row['slot']+'.metadata_admission.json'),item['actual_metadata']),
            'artifacts':{'completion.json':row['completion'],'result.json':row['result'],
                         'training_complete.json':item['result']['training_complete_binding'],
                         'checkpoint_reload_exact.json':item['result']['checkpoint_reload_binding']}})
    qualified=seal({'protocol':C.CAMPAIGN_PROTOCOL+'-qualified-prepolicy-registration','recorded_utc':'2026-10-20T12:00:08+00:00',
        'authority':ab,'execution_roster':rb,'initial_registration':ib,'qualifications':quals,
        'engineering_evaluation_responses':4,'engineering_optimizer_updates':2,'confirmation_policy_calls':0,'automatic_retry':False},'receipt_sha256')
    qb=store(C,root/'control/qualification_registration.json',qualified)
    return request,context,admitted,read,root,seal


def test_actual_registration_paths_whole_roster_five_students_and_cumulative_maxima(C,packet,monkeypatch):
    r,ctx,admitted,read,root,seal=registration_fixture(C,packet,monkeypatch)
    result=C.admit_registrations(r,ctx,admitted,read)
    assert result['whole_roster_before_first_qualification'] is True
    assert result['both_qualifications_before_first_fresh_policy'] is True
    assert result['all_five_students_registered'] is True
    assert result['resources']['combined_requested_GPU_hours']==(342 if len(admitted)==28 else 422)
    assert result['resources']['actual_spent_GPU_hours'] is None


@pytest.mark.parametrize('fault',['missing_initial','missing_roster','missing_qualified','late_initial',
    'qualified_before_finish','qualified_after_fresh','naive_initial','wrong_student','wrong_source','wrong_inputs',
    'missing_qualification','rebound_metadata','missing_reload','rebound_result','wrong_roster_EC','history_reclaimed'])
def test_resealed_registration_or_chronology_failures_cannot_admit_fresh_quality(C,packet,monkeypatch,fault):
    r,ctx,admitted,read,root,seal=registration_fixture(C,packet,monkeypatch)
    paths={'initial':root/'prepolicy_registration.json','roster':root/'execution_roster.json','qualified':root/'control/qualification_registration.json'}
    if fault.startswith('missing_') and fault in ('missing_initial','missing_roster','missing_qualified'):
        paths[fault[8:]].unlink()
    else:
        key='initial' if fault in ('late_initial','naive_initial','wrong_student','wrong_source','wrong_inputs') else ('roster' if fault in ('wrong_roster_EC','history_reclaimed') else 'qualified')
        p=paths[key];v=json.loads(p.read_bytes())
        field='roster_sha256' if key=='roster' else 'receipt_sha256';v.pop(field)
        if fault=='late_initial':v['recorded_utc']='2026-10-20T12:00:03+00:00'
        elif fault=='naive_initial':v['recorded_utc']='2026-10-20T12:00:01'
        elif fault=='qualified_before_finish':v['recorded_utc']='2026-10-20T12:00:03+00:00'
        elif fault=='qualified_after_fresh':v['recorded_utc']='2026-10-20T12:00:11+00:00'
        elif fault=='wrong_student':v['student_logicals']['503']=v['student_logicals']['101']
        elif fault=='wrong_source':v['sources']={}
        elif fault=='wrong_inputs':v['inputs']={}
        elif fault=='missing_qualification':v['qualifications'].pop()
        elif fault=='rebound_metadata':v['qualifications'][0]['metadata_admission']=v['qualifications'][1]['metadata_admission']
        elif fault=='missing_reload':v['qualifications'][0]['artifacts'].pop('checkpoint_reload_exact.json')
        elif fault=='rebound_result':v['qualifications'][0]['artifacts']['result.json']=v['qualifications'][1]['artifacts']['result.json']
        elif fault=='wrong_roster_EC':v['slots'][-1]['contract']=v['slots'][0]['contract']
        elif fault=='history_reclaimed':v['resources']['failed_and_unlaunched_reservations_reclaimed']=True
        b=store(C,p,seal(v,field))
        # Preserve pointer coherence after a late initial reseal, so chronology
        # itself (rather than only the changed byte pointer) must reject it.
        if key=='initial':
            q=json.loads(paths['qualified'].read_bytes());q.pop('receipt_sha256');q['initial_registration']=b
            store(C,paths['qualified'],seal(q,'receipt_sha256'))
    with pytest.raises((ValueError,FileNotFoundError)):C.admit_registrations(r,ctx,admitted,read)


def test_all_preexisting_statistical_module_nodes_preserved_except_explicit_admission_switch(C):
    archived=ROOT/'experiments/rloo_one_shot_20261016/validation/confirmation_op_statistics_v3/analyze_confirmation_op_v3.py'
    old=ast.parse(archived.read_bytes());new=ast.parse(SOURCE.read_bytes())
    def key(node):
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):return (type(node).__name__,node.name)
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):return ('assign',node.targets[0].id)
        if isinstance(node,(ast.Import,ast.ImportFrom)):return ('import',ast.dump(node,include_attributes=False))
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):return ('module_doc',)
        return ('other',ast.dump(node,include_attributes=False))
    mapping={key(node):node for node in new.body}
    for node in old.body:
        k=key(node)
        if k in (('module_doc',),('assign','METADATA_ADMISSION_READY'),('FunctionDef','main'),('FunctionDef','analyze_confirmation_op')):
            continue
        assert k in mapping and ast.dump(node,include_attributes=False)==ast.dump(mapping[k],include_attributes=False)
