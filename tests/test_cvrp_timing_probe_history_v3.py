"""Prospective confirmation history: exact source133, synthetic future jobs.

No model calls, native RPC, authority registration or old-source mutation.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import types

import pytest

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/rloo_sft_anchor_20261013'
E=EXP/'cvrp_probe_execution_v3'
SOURCE=E/'source_candidate'
DEP=SOURCE/'experiments/rloo_sft_anchor_20261013'


def load(path,name):
    body=path.read_bytes();m=types.ModuleType(name);m.__file__=str(path);m.__executed_source_sha256__=hashlib.sha256(body).hexdigest()
    exec(compile(body,str(path),'exec',dont_inherit=True),m.__dict__);return m


def sha(body):return hashlib.sha256(body).hexdigest()
def raw(v):return json.dumps(v,sort_keys=True,indent=2).encode()+b'\n'
def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def seal(v,key):v[key]=sha(encoded({k:x for k,x in v.items() if k!=key}));return v

def item(path,body):return {'path':path,'sha256':sha(body),'size_bytes':len(body)}

p=load(DEP/'prepare_cvrp_timing_probe_v3.py','history3_fixture_prep')
a=load(DEP/'verify_cvrp_timing_probe_accounting_v3.py','history3_fixture_accounting')
PILOT=(EXP/'collected_native_first_complete_v1/status/campaign_authority.json').read_bytes()
FAILED=(EXP/'validation/native_monitor/prior_failed_1030929_reverification_v2_bound.json').read_bytes()
FAILURE=(SOURCE/p.FAILURE_PATH).read_bytes()
NOJOB=(SOURCE/p.NO_JOB_PATH).read_bytes()


class Evidence:
    def __init__(self):self.files={};self.identities={}
    def put(self,path,v):
        body=v if type(v) is bytes else raw(v);self.files[path]=body
        self.identities[path]={'native_path':path,'independent_read_only_collection':True}
        return item(path,body)
    def __call__(self,binding):return self.files[binding['path']]
    def native_identity(self,binding):return self.identities[binding['path']]


@pytest.fixture(scope='module')
def modules():
    return load(EXP/'cvrp_timing_probe_history_v3.py','prospective_history3'),load(EXP/'confirmation_probe_history_bridge_v4.py','prospective_bridge4')


def fixture(h,*,submitted=True,state='COMPLETED',rescue=False):
    r=Evidence();source_raw=(SOURCE/'source_manifest.json').read_bytes();manifest=json.loads(source_raw)
    files={row['path']:r.put(p.SOURCE+'/'+row['path'],(SOURCE/row['path']).read_bytes()) for row in manifest['files']}
    spec=json.loads((E/'spec_proposal_v3.json').read_bytes());sb=r.put(p.CONTROL+'/spec.json',spec)
    rescue_binding=None;base=92
    if rescue:
        extras=[{'reservation_id':'evaluation-only/'+slot,'gpus':1,'wall_seconds':7200,'requested_wall_seconds':7200}
            for slot in p.ORDER if not slot.endswith('.sft')]
        rv=seal({'synthetic_fixture':True,'original_pilot_authority_sha256':p.PILOT_AUTHORITY_SHA,
            'original_pilot_authority_binding':item(p.CONTROL+'/pilot_authority.json',PILOT),'additional_reservations':extras,
            'additional_requested_gpu_hours':20,'combined_reserved_requested_gpu_hours':112},'authority_sha256')
        rescue_binding=r.put('/synthetic/existing-rescue.json',rv);base=112
        authority=p.make_authority_proposal(spec['spec_sha256'],rescue_reservations=extras,rescue_authority_sha256=rescue_binding['sha256'])
    else:authority=p.make_authority_proposal(spec['spec_sha256'])
    authority['synthetic_fixture']=True;seal(authority,'authority_sha256');ab=r.put(p.CONTROL+'/authority.json',authority)
    listing={'protocol':h.PROTOCOL+'-source-directory-inventory','synthetic_fixture':True,'native_source_root':p.SOURCE,
        'independent_read_only_collection':True,'regular_files':sorted([row['path'] for row in manifest['files']]+['source_manifest.json']),
        'symbolic_links':[],'nonregular_files':[]}
    directory={'protocol':h.PROTOCOL+'-attempt-directory-inventory','synthetic_fixture':True,'native_control_root':p.CONTROL,
        'independent_read_only_collection':True,'attempt_files':[],'intent_files':[],'action_files':[],'unresolved_intents':[]}
    record={'protocol':h.PROTOCOL+'-probe-custody','synthetic_fixture':True,'performance_admission':False,
        'native_namespace':p.NAMESPACE,'native_source_root':p.SOURCE,'native_control_root':p.CONTROL,'reservation_id':a.RESERVATION,
        'independent_read_only_collection':True,'source_manifest_binding':r.put(p.SOURCE+'/source_manifest.json',source_raw),
        'original_source_manifest_binding':files['original_source2_manifest.json'],'source_file_bindings':files,
        'spec_binding':sb,'authority_binding':ab,'pilot_authority_binding':r.put(p.CONTROL+'/pilot_authority.json',PILOT),
        'prior_failed_proof_binding':r.put(p.CONTROL+'/prior_failed_phase.json',FAILED),'rescue_authority_binding':rescue_binding,
        'rejected_failure_binding':r.put(p.CONTROL+'/rejected_probe_failure.json',FAILURE),'no_assigned_job_binding':r.put(p.CONTROL+'/no_assigned_job.json',NOJOB),
        'state':'closed_unsubmitted_reservation','submission_rpc_started':False,'attempt_binding':None,'query_binding':None,
        'intent_binding':None,'action_binding':None}
    if submitted:
        script=b'# SYNTHETIC NEVER SUBMITTED SCRIPT\n';ib=r.put(p.CONTROL+'/submission_intent.json',{
            'protocol':p.PROTOCOL+'-intent','synthetic_fixture':True,'reservation_id':a.RESERVATION,'spec_sha256':spec['spec_sha256'],
            'authority_sha256':ab['sha256'],'source_manifest_sha256':record['source_manifest_binding']['sha256'],
            'script_sha256':sha(script),'automatic_retry':False,'execution_revision':3})
        attempt={'protocol':p.PROTOCOL+'-attempt','synthetic_fixture':True,'state':'submitted','job_id':'9990001','attempt_number':1,
            'reservation_id':a.RESERVATION,'replacement_or_retry':False,'spec_sha256':spec['spec_sha256'],'authority_sha256':ab['sha256'],
            'source_manifest_sha256':record['source_manifest_binding']['sha256'],'script_path':p.CONTROL+'/job.slurm','script_sha256':sha(script),
            'submission_intent_sha256':ib['sha256'],'submission_argv':['sbatch','--parsable',p.CONTROL+'/job.slurm'],
            'step_strategy':'torchrun4_in_batch','requested_gpus':4,'requested_wall_seconds':1800,'execution_revision':3}
        action={'protocol':p.PROTOCOL+'-submission-action','synthetic_fixture':True,'intent_sha256':ib['sha256'],
            'argv':attempt['submission_argv'],'returncode':0,'stdout':'9990001\n','stderr':''}
        query={'synthetic_fixture':True,'independent_read_only_collection':True,'returncode':0,'stderr':'','recorded_utc':'2026-10-14T18:01:00+00:00',
            'argv':['sacct','--noheader','--parsable2','--jobs','9990001','--format=JobIDRaw,State%40,ElapsedRaw,ExitCode,AllocTRES'],
            'stdout':'\n'.join(f"9990001{x}|{state}|100|{'0:0' if state in ('COMPLETED','RUNNING') else '1:0'}|cpu=80,mem=300000M,gres/gpu=4,gres/gpu:a100=4" for x in ('','.batch','.extern'))}
        directory.update(attempt_files=['attempt.json'],intent_files=['submission_intent.json'],action_files=['submission_action.json'])
        record.update(state='open_submitted_attempt' if state=='RUNNING' else 'closed_submitted_attempt',submission_rpc_started=True,
            script_binding=r.put(p.CONTROL+'/job.slurm',script),intent_binding=ib,attempt_binding=r.put(p.CONTROL+'/attempt.json',attempt),
            action_binding=r.put(p.CONTROL+'/submission_action.json',action),query_binding=r.put(p.CONTROL+'/accounting_query_history_v4.json',query))
        attempt['submission_action_sha256']=record['action_binding']['sha256'];record['attempt_binding']=r.put(p.CONTROL+'/attempt.json',attempt)
    if not submitted:
        census={'synthetic_fixture':True,'independent_read_only_collection':True,'returncode':0,'stdout':'','stderr':'',
            'recorded_utc':'2026-10-14T18:01:00+00:00','argv':['sacct','--user','uag72kz','--name','cvrp-first3-timing-v3',
                '--starttime','2026-10-14','--noheader','--parsable2','--format=JobIDRaw,JobName,State%40,ElapsedRaw,ExitCode,ReqTRES,AllocTRES']}
        record['closure_started_utc']='2026-10-14T18:00:00+00:00'
        record['unsubmitted_job_census_binding']=r.put(p.CONTROL+'/unsubmitted_job_census_history_v4.json',census)
    query_binding=record['query_binding'] if submitted else record['unsubmitted_job_census_binding']
    for snapshot in (listing,directory):snapshot.update(recorded_utc='2026-10-14T18:02:00+00:00',accounting_query_sha256=query_binding['sha256'])
    record['recorded_utc']='2026-10-14T18:03:00+00:00'
    record['source_directory_inventory_binding']=r.put(p.CONTROL+'/source_directory_inventory_history_v4.json',listing)
    record['attempt_directory_inventory_binding']=r.put(p.CONTROL+'/attempt_directory_inventory_history_v4.json',directory)
    seal(record,'receipt_sha256');return record,r,base


def verify(h,record,r,base=92,**kwargs):
    return h.verify_probe_history(record,r,expected_preparer_sha256=sha((DEP/'prepare_cvrp_timing_probe_v3.py').read_bytes()),
        expected_accounting_sha256=sha((DEP/'verify_cvrp_timing_probe_accounting_v3.py').read_bytes()),historical_base_gpu_hours=base,
        synthetic_fixture=kwargs.pop('synthetic_fixture',True),dependency_root=DEP,**kwargs)


def change(record,r,field,fn):
    value=json.loads(r(record[field]));fn(value);record[field]=r.put(record[field]['path'],value)

@pytest.mark.parametrize('submitted,state,rescue',[(False,'COMPLETED',False),(True,'COMPLETED',False),(True,'FAILED',False),(True,'TIMEOUT',False),(True,'COMPLETED',True)])
def test_actual133_source_history_retains_rejected2_and_current2(modules,submitted,state,rescue):
    h,_=modules;record,r,base=fixture(h,submitted=submitted,state=state,rescue=rescue);report=verify(h,record,r,base)
    assert report['historical_pilot_reserved_gpu_hours']==base+4
    assert report['diagnostic_allocated_gpu_seconds']==(400 if submitted else 0)
    assert report['study_admission'] is False and report['native_mutations'] is False
    assert report['synthetic_fixture'] is True and len(record['source_file_bindings'])==133


def test_running_probe_is_explicit_lowerbound_and_default_confirmation_denies(modules):
    h,_=modules;record,r,base=fixture(h,state='RUNNING')
    with pytest.raises(ValueError):verify(h,record,r,base)
    report=verify(h,record,r,base,require_closed=False)
    assert report['historical_pilot_reserved_gpu_hours']==96 and report['diagnostic_allocated_gpu_seconds']==400
    assert not report['diagnostic_accounting']['allocation_accounting_complete']
    assert report['diagnostic_accounting']['actual_cost_is_lower_bound']

@pytest.mark.parametrize('fault',['old92','old_rejected2','reset94','rejected_bytes','nojob_bytes','source_omission','source_mutation','source_copy',
    'source_extra','source_cache','source_link','spec','manifest_copy','attempt_retry','attempt_bool','attempt_source','action_job','action_error',
    'action_bool','query_truncated','query_running','hidden_attempt','ambiguous','script','namespace','base_reset','native_fixture'])
def test_history_source_cost_namespace_and_typed_custody_faults_fail_closed(modules,fault):
    h,_=modules;record,r,base=fixture(h)
    if fault in ('old92','old_rejected2','reset94'):
        def corrupt(v):
            if fault=='old92':v['original_reservations'].pop()
            elif fault=='old_rejected2':v.pop('prior_rejected_probe_reservation')
            else:v['combined_reserved_requested_gpu_hours']=94
            seal(v,'authority_sha256')
        change(record,r,'authority_binding',corrupt)
    elif fault in ('rejected_bytes','nojob_bytes'):
        field='rejected_failure_binding' if fault=='rejected_bytes' else 'no_assigned_job_binding';record[field]=r.put(record[field]['path'],r(record[field])+b' ')
    elif fault=='source_omission':record['source_file_bindings'].pop(p.TRAINER)
    elif fault=='source_mutation':record['source_file_bindings'][p.TRAINER]=r.put(p.SOURCE+'/'+p.TRAINER,b'changed')
    elif fault=='source_copy':record['source_file_bindings'][p.TRAINER]=r.put('/copied/'+p.TRAINER,r(record['source_file_bindings'][p.TRAINER]))
    elif fault in ('source_extra','source_cache','source_link'):
        change(record,r,'source_directory_inventory_binding',lambda v:v['symbolic_links' if fault=='source_link' else 'regular_files'].append('stale.pyc' if fault=='source_cache' else 'unbound.py'))
    elif fault=='spec':change(record,r,'spec_binding',lambda v:(v['recipe'].update(original_horizon=3),seal(v,'spec_sha256')))
    elif fault=='manifest_copy':record['source_manifest_binding']=r.put('/copied/manifest.json',r(record['source_manifest_binding']))
    elif fault.startswith('attempt_'):
        key,value={'attempt_retry':('attempt_number',2),'attempt_bool':('attempt_number',True),'attempt_source':('source_manifest_sha256','0'*64)}[fault]
        change(record,r,'attempt_binding',lambda v:v.update({key:value}))
    elif fault.startswith('action_'):
        key,value={'action_job':('stdout','9990002'),'action_error':('returncode',1),'action_bool':('returncode',False)}[fault]
        change(record,r,'action_binding',lambda v:v.update({key:value}))
    elif fault.startswith('query_'):
        change(record,r,'query_binding',lambda v:v.update(stdout=v['stdout'].replace('COMPLETED','RUNNING') if fault=='query_running' else '9990001|COMPLETED|100|0:0|gres/gpu=4'))
    elif fault in ('hidden_attempt','ambiguous'):
        change(record,r,'attempt_directory_inventory_binding',lambda v:v['attempt_files' if fault=='hidden_attempt' else 'unresolved_intents'].append('hidden.json'))
    elif fault=='script':record['script_binding']=r.put(p.CONTROL+'/job.slurm',b'changed')
    elif fault=='namespace':record['native_namespace']+='unregistered'
    elif fault=='base_reset':base=90
    seal(record,'receipt_sha256')
    with pytest.raises((ValueError,KeyError)):verify(h,record,r,base,synthetic_fixture=fault!='native_fixture')


def test_unsubmitted_reservation_cannot_hide_ambiguous_rpc(modules):
    h,_=modules;record,r,base=fixture(h,submitted=False);record['submission_rpc_started']=True;seal(record,'receipt_sha256')
    with pytest.raises(ValueError):verify(h,record,r,base)


def test_original_operator_directory_schema_is_preserved_byte_for_byte(modules):
    h,_=modules;record,r,base=fixture(h)
    change(record,r,'attempt_directory_inventory_binding',lambda v:(v.update(protocol='sft-anchor-cvrp-timing-probe-operator-v1-attempt-directory-inventory',control_root=v.pop('native_control_root'))))
    seal(record,'receipt_sha256');before=r(record['attempt_directory_inventory_binding'])
    assert verify(h,record,r,base)['historical_pilot_reserved_gpu_hours']==96
    assert r(record['attempt_directory_inventory_binding'])==before


def compiled(b):
    return b.compile_consumers(expected_probe_preparer_sha256=sha((DEP/'prepare_cvrp_timing_probe_v3.py').read_bytes()),
        expected_probe_accounting_sha256=sha((DEP/'verify_cvrp_timing_probe_accounting_v3.py').read_bytes()),
        expected_probe_history_sha256=sha((EXP/'cvrp_timing_probe_history_v3.py').read_bytes()),
        expected_bridge_sha256=sha((EXP/'confirmation_probe_history_bridge_v4.py').read_bytes()),dependency_root=DEP)


def context_fixture(record,r,proof):
    # The exact frozen historical92 guard still runs before the additive proof.
    from test_cvrp_timing_probe_history_v1 import context_fixture as old_context_fixture
    return old_context_fixture(record,r,proof)


def test_bridge_only_two_resource_expressions_change_entire_remaining_AST(modules):
    _,b=modules;P,A,proof=compiled(b)
    for module,key,filename,target in ((P,'preparation','prepare_confirmation_round_v1.py','validate_resource_authority'),
        (A,'analysis','analyze_chunked_confirmation_v1.py','historical_resource_custody')):
        assert proof[key]['replaced_resource_expressions']==2
        old=ast.parse((EXP/filename).read_bytes())
        expected={n.name:sha(ast.dump(n,include_attributes=False).encode()) for n in old.body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!=target}
        assert expected==proof[key]['unchanged_function_class_ast_sha256']
        assert module.__executed_source_sha256__==sha((EXP/filename).read_bytes())
        assert module.__operational_history_bridge__['bridge_source_sha256']==sha((EXP/'confirmation_probe_history_bridge_v4.py').read_bytes())
    # These published classes/functions include slot replay, source checks,
    # onceSFT admission, all20 draw retention and exactalpha majority test.
    assert proof['all12_timing_82slot_20draw_numerical_functions_unchanged']
    assert not proof['native_registration_created'] and not proof['publication_registry_changed']


def test_frozen_baseline_history_guard_and_additive96_guard_execute(modules):
    h,b=modules;record,r,base=fixture(h);P,A,proof=compiled(b);context,authority,ledger=context_fixture(record,r,proof)
    report=A.historical_resource_custody(context,authority)
    assert report['pilot_reserved_gpu_hours']==96 and report['formal_reserved_gpu_hours']==0
    assert report['prior_failed_allocated_gpu_seconds']==1836 and context['probe_history_v4']['admission']['historical_pilot_reserved_gpu_hours']==96
    report_raw=b'{}';outer=b.wrap_analysis_receipt(item('/synthetic/original-report.json',report_raw),{},context,proof,original_report_raw=report_raw)
    assert outer['mathematical_outcomes_modified'] is False
    assert outer['probe_history']['admission']['diagnostic_allocated_gpu_seconds']==400
    authority['historical_pilot_requested_gpu_hours']=92
    with pytest.raises(ValueError):A.historical_resource_custody(context,authority)
    authority['historical_pilot_requested_gpu_hours']=96
    ledger['original_pilot_reservations'].pop();seal(ledger,'receipt_sha256')
    context['registration_entry']['historical_resource_ledger_binding']=authority['historical_resource_ledger_binding']=r.put('/synthetic/complete-history.json',ledger)
    with pytest.raises(ValueError):A.historical_resource_custody(context,authority)


def test_bridge_formal_preparation_requires_exact_same_custody_and_source(modules):
    h,b=modules;record,r,base=fixture(h);binding=r.put('/synthetic/probe-custody.json',record);P,_,proof=compiled(b)
    sources=proof['operational_identity']['closed_additive_source_files']
    source_binding=r.put('/synthetic/formal-source.json',{'synthetic_fixture':True,'files':list(sources.values())})
    files={}
    for path in sources:
        physical=SOURCE/path
        if not physical.is_file():physical=EXP/Path(path).name
        files[path]=r.put('/synthetic/formal/'+path,physical.read_bytes())
    proposal={'synthetic_fixture':True,'bindings':{'cvrp_timing_probe_custody':binding,'source_manifest':source_binding},'source_file_bindings':files}
    authority={'cvrp_timing_probe_custody_binding':binding,'operational_history_bridge_identity':proof['operational_identity']}
    assert P._probe_aware_preparation_total(proposal,r,authority,fixture=True)==96
    with pytest.raises(ValueError):P._probe_aware_preparation_total(proposal,r,authority,fixture=False)
    with pytest.raises(ValueError):P._probe_aware_preparation_total(proposal,r,{},fixture=True)
    removed=next(iter(files));proposal['source_file_bindings'].pop(removed)
    with pytest.raises((ValueError,KeyError)):P._probe_aware_preparation_total(proposal,r,authority,fixture=True)


def test_wrong_dependency_code_pin_cannot_compile_or_promote(modules):
    _,b=modules
    with pytest.raises(ValueError):b.compile_consumers(expected_probe_preparer_sha256='0'*64,expected_probe_accounting_sha256='0'*64,
        expected_probe_history_sha256='0'*64,expected_bridge_sha256='0'*64,dependency_root=DEP)

@pytest.fixture(scope='module')
def original_preparation():
    import test_prepare_confirmation_round_v1 as original
    return original.fixture()

@pytest.mark.parametrize('fault',[None,'missing_timing','failed_timing','missing_slot','unmeasured_step','lost_prior','retry','missing_bridge_source'])
def test_full_frozen_all12_timing_82_slot_preparation_guard_executes(original_preparation,modules,fault):
    h,b=modules;proposal,r=copy.deepcopy(original_preparation);record,proof_reader,base=fixture(h);r.files.update(proof_reader.files)
    P,_,proof=compiled(b);identity=proof['operational_identity'];binding=r.put('/synthetic/probe-custody.json',record)
    proposal['bindings']['cvrp_timing_probe_custody']=binding;source=json.loads(r(proposal['bindings']['source_manifest']))
    for path,row in identity['closed_additive_source_files'].items():
        source['files'].append(row);physical=SOURCE/path
        if not physical.is_file():physical=EXP/Path(path).name
        proposal['source_file_bindings'][path]=r.put('/synthetic/formal/'+path,physical.read_bytes())
    proposal['bindings']['source_manifest']=r.put('synthetic/formal-source.json',source)
    authority=json.loads(r(proposal['bindings']['resource_authority']))
    authority['historical_pilot_requested_gpu_hours']=96;authority['combined_reserved_requested_gpu_hours']+=4
    authority['declared_requested_gpu_hour_ceiling']+=4;authority['cvrp_timing_probe_custody_binding']=binding
    authority['operational_history_bridge_identity']=identity
    if fault=='missing_slot':authority['planned_slots'].pop()
    elif fault=='unmeasured_step':authority['formal_step_overhead_seconds']=0
    elif fault=='lost_prior':authority['prior_failed_phase_binding']['sha256']='0'*64
    elif fault=='retry':authority['planned_slots'][0]['maximum_attempts']=2
    elif fault=='missing_bridge_source':
        source['files'].pop();proposal['bindings']['source_manifest']=r.put('synthetic/formal-source.json',source)
    seal(authority,'authority_sha256');proposal['bindings']['resource_authority']=r.put('synthetic/resource-authority.json',authority)
    if fault in ('missing_timing','failed_timing'):
        timing=json.loads(r(proposal['bindings']['pilot_timing']))
        if fault=='missing_timing':timing['slots'].pop(P.PILOT_ORDER[0])
        else:timing['slots'][P.PILOT_ORDER[0]]['scheduler_state']='FAILED'
        proposal['bindings']['pilot_timing']=r.put('synthetic/timing.json',timing)
    selection=json.loads(r(proposal['bindings']['selection']))
    if fault:
        with pytest.raises((ValueError,KeyError)):P.validate_resource_authority(proposal,r,selection,2,fixture=True)
    else:
        admitted=P.validate_resource_authority(proposal,r,selection,2,fixture=True)
        assert len(admitted['planned_slots'])==82 and admitted['combined_reserved_requested_gpu_hours']==96+admitted['new_round_requested_gpu_hours']
        assert not b.wrap_preparation_receipt({'synthetic_fixture':True},proposal,proof)['native_submission_authorized']

@pytest.mark.parametrize('field',['spec_binding','authority_binding','pilot_authority_binding','prior_failed_proof_binding','rejected_failure_binding','no_assigned_job_binding'])
def test_identical_control_bytes_copied_elsewhere_do_not_prove_native_location(modules,field):
    h,_=modules;record,r,base=fixture(h);record[field]=r.put('/copied/'+field,r(record[field]));seal(record,'receipt_sha256')
    with pytest.raises(ValueError):verify(h,record,r,base)

@pytest.mark.parametrize('field',['require_closed','synthetic_fixture'])
def test_history_scope_flags_require_actual_booleans(modules,field):
    h,_=modules;record,r,base=fixture(h)
    with pytest.raises(ValueError):verify(h,record,r,base,**{field:1})


def test_open_cost_cannot_enter_registered_confirmation_bridge(modules):
    h,b=modules;record,r,base=fixture(h,state='RUNNING');P,A,proof=compiled(b);context,authority,_=context_fixture(record,r,proof)
    with pytest.raises(ValueError):A.historical_resource_custody(context,authority)


def test_116_same_rescue_authority_replayed_by_old_and_new_consumers(modules):
    h,b=modules;record,r,base=fixture(h,rescue=True);_,A,proof=compiled(b);context,authority,ledger=context_fixture(record,r,proof)
    report=A.historical_resource_custody(context,authority)
    assert report['pilot_reserved_gpu_hours']==116 and context['probe_history_v4']['admission']['historical_pilot_reserved_gpu_hours']==116
    ledger['additional_pilot_authorities']=[];seal(ledger,'receipt_sha256')
    context['registration_entry']['historical_resource_ledger_binding']=authority['historical_resource_ledger_binding']=r.put('/synthetic/dropped-rescue.json',ledger)
    with pytest.raises(ValueError):A.historical_resource_custody(context,authority)


def test_complete_consumer_module_AST_changes_only_two_declared_cost_expressions(modules):
    _,b=modules;_,_,proof=compiled(b)
    for key,filename,function in [('preparation','prepare_confirmation_round_v1.py','validate_resource_authority'),
        ('analysis','analyze_chunked_confirmation_v1.py','historical_resource_custody')]:
        tree=ast.parse((EXP/filename).read_bytes());target=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
        replaced=0
        if key=='analysis':
            class CostOnly(ast.NodeTransformer):
                def visit_BinOp(self,node):
                    nonlocal replaced
                    if isinstance(node.op,ast.Add) and isinstance(node.left,ast.Constant) and type(node.left.value) is int and node.left.value==92 and isinstance(node.right,ast.Name) and node.right.id=='extra_hours':
                        replaced+=1
                        return ast.copy_location(ast.parse('_probe_aware_history_total(context, authority, ledger, 92 + extra_hours)',mode='eval').body,node)
                    return self.generic_visit(node)
            CostOnly().visit(target)
        else:
            replacement=ast.parse('_probe_aware_preparation_total(proposal, reader, authority, fixture=fixture)',mode='eval').body
            for node in ast.walk(target):
                if isinstance(node,ast.Compare) and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant) and type(node.comparators[0].value) is int and node.comparators[0].value==92 and isinstance(node.left,ast.Call) and isinstance(node.left.func,ast.Attribute) and isinstance(node.left.func.value,ast.Name) and node.left.func.value.id=='authority' and node.left.args and isinstance(node.left.args[0],ast.Constant) and node.left.args[0].value=='historical_pilot_requested_gpu_hours':
                    node.comparators[0]=copy.deepcopy(replacement);replaced+=1
                elif isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='total':
                    class TotalOnly(ast.NodeTransformer):
                        def visit_Constant(self,value):
                            nonlocal replaced
                            if type(value.value) is int and value.value==92:
                                replaced+=1;return ast.copy_location(copy.deepcopy(replacement),value)
                            return value
                    TotalOnly().visit(node)
        ast.fix_missing_locations(tree)
        assert replaced==2
        assert sha(ast.dump(tree,include_attributes=False).encode())==proof[key]['effective_ast_sha256']

@pytest.mark.parametrize('original,alias',[({'formal_claim_eligible':False},{'formal_claim_eligible':0}),({'draw_count':20},{'draw_count':20.0})])
def test_wrapped_analysis_cannot_alias_typed_original_report(modules,original,alias):
    _,b=modules;body=raw(original);binding=item('/synthetic/original-report.json',body);context={'probe_history_v4':{'synthetic_fixture':True}}
    assert original==alias
    assert b.wrap_analysis_receipt(binding,original,context,{},original_report_raw=body)['original_report']==original
    with pytest.raises(ValueError):b.wrap_analysis_receipt(binding,alias,context,{},original_report_raw=body)


def test_changed_file_cannot_rebind_an_already_executed_bridge_buffer(tmp_path):
    # Only /tmp copies are mutated. The original source packet and source-only
    # helper bytes remain untouched, and no consumers perform policy calls.
    import shutil
    for name in ('confirmation_probe_history_bridge_v4.py','cvrp_timing_probe_history_v3.py','prepare_confirmation_round_v1.py','analyze_chunked_confirmation_v1.py'):
        shutil.copyfile(EXP/name,tmp_path/name)
    path=tmp_path/'confirmation_probe_history_bridge_v4.py';old=load(path,'already_executed_test_bridge')
    original_executed=old.__executed_source_sha256__;path.write_bytes(path.read_bytes()+b'\n# Later bytes are distinct from the operative loaded buffer.\n')
    assert sha(path.read_bytes())!=original_executed
    with pytest.raises(ValueError):old.compile_consumers(expected_probe_preparer_sha256=sha((DEP/'prepare_cvrp_timing_probe_v3.py').read_bytes()),
        expected_probe_accounting_sha256=sha((DEP/'verify_cvrp_timing_probe_accounting_v3.py').read_bytes()),
        expected_probe_history_sha256=sha((tmp_path/'cvrp_timing_probe_history_v3.py').read_bytes()),
        expected_bridge_sha256=sha(path.read_bytes()),dependency_root=DEP)

@pytest.mark.parametrize('filename',['confirmation_probe_history_bridge_v4.py','cvrp_timing_probe_history_v3.py'])
def test_plain_unbound_helper_import_metadata_cannot_create_resource_proof(filename,modules):
    module=load(EXP/filename,'unbound_'+filename.replace('.py',''));module.__dict__.pop('__executed_source_sha256__')
    with pytest.raises(ValueError):
        if filename.startswith('confirmation'):compiled(module)
        else:
            h,_=modules;record,r,base=fixture(h);verify(module,record,r,base)


def test_already_loaded_history_buffer_rejects_mutated_file_before_custody(tmp_path,modules):
    import shutil
    path=tmp_path/'cvrp_timing_probe_history_v3.py';shutil.copyfile(EXP/path.name,path)
    module=load(path,'old_history_buffer');path.write_bytes(path.read_bytes()+b'\n# Later distinct history bytes.\n')
    h,_=modules;record,r,base=fixture(h)
    with pytest.raises(ValueError):verify(module,record,r,base)

@pytest.mark.parametrize('submitted',[True,False])
@pytest.mark.parametrize('field',['source_directory_inventory_binding','attempt_directory_inventory_binding'])
def test_copied_or_fabricated_empty_directory_census_is_not_native_custody(modules,submitted,field):
    h,_=modules;record,r,base=fixture(h,submitted=submitted)
    record[field]=r.put('/copied/'+field,r(record[field]));seal(record,'receipt_sha256')
    with pytest.raises(ValueError):verify(h,record,r,base)


def test_structurally_matching_submission_rpc_bytes_cannot_replace_attempt_pin(modules):
    h,_=modules;record,r,base=fixture(h)
    body=r(record['action_binding']);record['action_binding']=r.put(record['action_binding']['path'],body+b' ')
    assert json.loads(body)==json.loads(r(record['action_binding']))
    seal(record,'receipt_sha256')
    with pytest.raises(ValueError):verify(h,record,r,base)

@pytest.mark.parametrize('fault',['old_filename','stale_timestamp','wrong_query','naive_time','record_earlier'])
def test_prepolicy_or_unbound_current_directory_snapshot_cannot_close_history(modules,fault):
    h,_=modules;record,r,base=fixture(h);field='attempt_directory_inventory_binding'
    if fault=='old_filename':record[field]=r.put(p.CONTROL+'/attempt_directory_inventory.json',r(record[field]))
    elif fault=='record_earlier':record['recorded_utc']='2026-10-14T18:01:30+00:00'
    else:
        key,value={'stale_timestamp':('recorded_utc','2026-10-14T18:00:00+00:00'),
            'wrong_query':('accounting_query_sha256','0'*64),'naive_time':('recorded_utc','2026-10-14T18:02:00')}[fault]
        change(record,r,field,lambda v:v.update({key:value}))
    seal(record,'receipt_sha256')
    with pytest.raises(ValueError):verify(h,record,r,base)

@pytest.mark.parametrize('fault',['missing','assigned_job','old_census','wrong_argv','copied','failed_query'])
def test_unsubmitted_reservation_requires_fresh_empty_own_name_accounting_census(modules,fault):
    h,_=modules;record,r,base=fixture(h,submitted=False);field='unsubmitted_job_census_binding'
    if fault=='missing':record.pop(field)
    elif fault=='copied':record[field]=r.put('/copied/census.json',r(record[field]))
    else:
        key,value={'assigned_job':('stdout','1074755|cvrp-first3-timing-v3|PENDING|0|0:0||'),
            'old_census':('recorded_utc','2026-10-14T17:59:00+00:00'),'wrong_argv':('argv',['sacct','--jobs','9990001']),
            'failed_query':('returncode',1)}[fault]
        change(record,r,field,lambda v:v.update({key:value}))
    seal(record,'receipt_sha256')
    with pytest.raises((ValueError,KeyError)):verify(h,record,r,base)


def test_copied_accounting_query_bytes_do_not_establish_native_snapshot_identity(modules):
    h,_=modules;record,r,base=fixture(h);record['query_binding']=r.put('/copied/accounting_query.json',r(record['query_binding']));seal(record,'receipt_sha256')
    with pytest.raises(ValueError):verify(h,record,r,base)

@pytest.mark.parametrize('copied',[False,True])
def test_bounded_failed_scheduler_grace_retains_actual_cost_and_native_identity(modules,copied):
    h,_=modules;record,r,base=fixture(h,state='TIMEOUT')
    change(record,r,'query_binding',lambda v:v.update(stdout=v['stdout'].replace('|100|','|1801|')))
    for field in ('source_directory_inventory_binding','attempt_directory_inventory_binding'):
        change(record,r,field,lambda v:v.update(accounting_query_sha256=record['query_binding']['sha256']))
    grace={'protocol':p.PROTOCOL+'-termination-grace','synthetic_fixture':True,'job_id':'9990001',
        'query_sha256':record['query_binding']['sha256'],'independent_read_only_collection':True,'maximum_overrun_seconds':30}
    record['termination_grace_binding']=r.put('/copied/grace.json' if copied else p.CONTROL+'/termination_grace_history_v4.json',grace)
    seal(record,'receipt_sha256')
    if copied:
        with pytest.raises(ValueError):verify(h,record,r,base)
    else:
        report=verify(h,record,r,base)
        assert report['diagnostic_allocated_gpu_seconds']==7204 and report['historical_pilot_reserved_gpu_hours']==96
