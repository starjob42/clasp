"""Offline, manufactured registration inputs; no real round/cohort is created.

Two translated toy graphs per task exercise fresh-input custody. Every seed,
selection, timing, scheduler, reference-audit and resource approval here is a
unit fixture. Exact original code/128 training inputs are read-only inputs.
"""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_sft_anchor_admission_v3_collection import make_contract,scorer

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/rloo_sft_anchor_20261013'
spec=importlib.util.spec_from_file_location('source_only_confirmation_preparer',EXP/'prepare_confirmation_round_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class MemoryEvidence:
    def __init__(self):self.files={}
    def put(self,path,value):
        raw=value if isinstance(value,bytes) else (json.dumps(value,sort_keys=True)+'\n').encode()
        self.files[path]=raw
        return {'path':path,'sha256':m.sha(raw),'size_bytes':len(raw)}
    def __call__(self,binding):return self.files[binding['path']]


def sealed(value,key):value[key]=m.canonical({k:v for k,v in value.items() if k!=key});return value


def toy_rows(task,pilot):
    rows=[]
    for index,original in enumerate(pilot['validation_rows'][:2]):
        instance=copy.deepcopy(original['instance_json'])
        instance[0]=[[x+1000+index,y+2000+index] for x,y in instance[0]]
        weight='prize' if task=='op' else 'demand'
        text='\n'.join(f'Node {i}, coordinates: [{x}, {y}], {weight}: {v}, neighbors: [];' for i,((x,y),v) in enumerate(zip(instance[0],instance[1])))
        rows.append({'instance_id':f'synthetic-fresh-{task}-{index}','instance_json':instance,'input':text,
            'instruction':original['instruction'],'output':original['output']})
    return rows


def fixture():
    reader=MemoryEvidence();b={};source_files={};source_entries=[]
    files={'plan':EXP/'planning/selection_and_confirmation_v2.json','selector':EXP/'select_candidate.py',
        'analysis':EXP/'confirmatory_analysis.py','metrics':EXP/'selection_metrics.py','scorer':ROOT/m.SCORER,
        'durable_verifier':EXP/'collect_campaign_admission_v3.py','projection':EXP/'certify_sft_input_overlap.py'}
    for name,path in files.items():b[name]=reader.put('dependencies/'+name+'.py' if name!='plan' else 'dependencies/plan.json',path.read_bytes())
    for relative in (m.ENTRYPOINT,m.SCORER):
        source_files[relative]=reader.put('source/'+relative,(ROOT/relative).read_bytes())
        source_entries.append({'path':relative,'sha256':source_files[relative]['sha256'],'size_bytes':source_files[relative]['size_bytes']})
    b['source_manifest']=reader.put('source/source_manifest.json',{'synthetic_fixture':True,'files':source_entries})
    b['pilot_source_manifest']=reader.put('pilot/source_manifest.json',(EXP/'source_v2/source_manifest.json').read_bytes())
    b['pilot_task_contracts']={};pilots={};rows={};numeric={};contracts={};task_bindings={};refs={};cohorts={};graph_ids={}
    roster=[{'draw_id':f'draw{i:02}','train_seed':100+i} for i in range(20)]
    roster[-1]['train_seed']=100 # A possible independent repeated draw is retained.
    for task in m.TASKS:
        pilots[task]=make_contract(task);b['pilot_task_contracts'][task]=reader.put('pilot/contracts/'+task+'.json',
            (EXP/f'prepared_v1/contracts/{task}_seed43.json').read_bytes())
        rows[task]=toy_rows(task,pilots[task])
        numeric[task]=[{k:v for k,v in row.items() if k!='output'} for row in rows[task]]
    for draw in roster:
        task_bindings[draw['draw_id']]={}
        for task in m.TASKS:
            contract=scorer.make_contract(task,pilots[task]['initializer'],pilots[task]['training_rows'],rows[task],
                train_seed=draw['train_seed'],eval_seed=pilots[task]['eval_seed'],scope='production')
            contracts[draw['draw_id'],task]=contract
            task_bindings[draw['draw_id']][task]=reader.put('contracts/'+draw['draw_id']+'/'+task+'.json',contract)
            cohorts[task]=contract['validation_cohort_sha256'];graph_ids[task]=[r['graph_semantic_id'] for r in contract['validation_rows']]
            refs[task]=[{k:r[k] for k in ('instance_id','reference_id','reference_objective')} for r in contract['validation_rows']]
    b['numeric_inputs']={task:reader.put('numeric/'+task+'.json',numeric[task]) for task in m.TASKS}
    known_graph_bindings={task:[reader.put('history/known_inputs_'+task+'.json',sorted({r['graph_semantic_id'] for field in
        ('training_rows','validation_rows') for r in pilots[task][field]}))] for task in m.TASKS}
    history=sealed({'protocol':m.PROTOCOL+'-alpha-history','synthetic_fixture':True,'development_used_RL_seeds':[43],
                    'development_input_inventory_complete':True,'development_input_graph_bindings':known_graph_bindings,
                    'prior_formal_rounds':[]},'history_sha256')
    b['round_history']=reader.put('registration/history.json',history)
    sampling={'distribution':'uniform_fixed_uint32_complement_with_replacement','independent_draws':True,
        'lower_inclusive':0,'upper_exclusive':2**32,'excluded_previously_used_seeds':[43]}
    entropy=reader.put('registration/synthetic_entropy.bin',b''.join(d['train_seed'].to_bytes(4,'little') for d in roster))
    b['seed_sampling']=reader.put('registration/sampler.json',{'protocol':m.PROTOCOL+'-seed-sampling','synthetic_fixture':True,
        'seed_sampling':sampling,'seed_roster':roster,'entropy_source':'os.getrandom','entropy_binding':entropy,
        'independent_custody_receipt':True,'recorded_utc':'2030-01-01T00:00:01+00:00'})
    snapshots=[];timing={}
    for index,slot in enumerate(m.PILOT_ORDER):
        gpu=1 if slot.endswith('.sft') else 4;steps=0 if gpu==1 else 100
        result={'synthetic_fixture':True,'resources':{'training_wall_seconds':0 if gpu==1 else 1,
            'evaluation_wall_seconds':1,'training_optimizer_steps':steps,'production_fixture':False}}
        result_binding=reader.put('pilot/results/'+slot+'.json',result)
        attempt={'synthetic_fixture':True,'slot':slot,'state':'submitted','job_id':str(999000+index),'manifest_sha256':m.PILOT_LAUNCH_SHA}
        attempt_binding=reader.put('pilot/attempts/'+slot+'.json',attempt)
        snapshots.extend([{'local_path':'results/'+slot+'/result.json','sha256':result_binding['sha256']},
                          {'local_path':'attempts/'+slot+'.json','sha256':attempt_binding['sha256']}])
        job=attempt['job_id'];elapsed=2 if gpu==1 else 3
        query={'synthetic_fixture':True,'argv':['sacct','--noheader','--parsable2','--jobs',job,
            '--format=JobIDRaw,State%40,ElapsedRaw,ExitCode,AllocTRES'],'returncode':0,'independent_read_only_collection':True,
            'stdout':'\n'.join(f'{job}{suffix}|COMPLETED|{elapsed}|0:0|cpu=80,gres/gpu={gpu}' for suffix in ('','.batch','.extern'))}
        timing[slot]={'training_wall_seconds':result['resources']['training_wall_seconds'],'evaluation_wall_seconds':1,
            'allocation_elapsed_seconds':elapsed,'scheduler_state':'COMPLETED','scheduler_exit_code':'0:0','actual_gpus':gpu,
            'evaluated_instances':96,'actual_optimizer_steps':steps,'production_fixture':False,
            'result_binding':result_binding,'attempt_binding':attempt_binding,
            'scheduler_binding':reader.put('pilot/scheduler/'+slot+'.json',query)}
    b['pilot_collection']=reader.put('pilot/collection.json',{'synthetic_fixture':True,'prepared_manifest_sha256':m.PILOT_LAUNCH_SHA,
        'admission':{'status':'validated_completed_campaign','completed_slots':list(m.PILOT_ORDER)},'snapshots':snapshots})
    selection={'protocol':'sft-anchor-complete-development-selection-v1','synthetic_fixture':True,'selected_arm':'rloo_cardinal_beta005',
        'all_development_slots_admitted':12,'confirmation_outcomes_read':False,'superiority_established':False,
        'assessment_requires_separate_production_admission':False,'primary_decoder':'k8','shared_recipe_across_tasks':True,
        'eligible_arm_ranking':['rloo_cardinal_beta005'],'assessments':[{'arm':'rloo_cardinal_beta005','eligible':True,
        'conditions':{'manufactured_test_only':True}}],'recorded_utc':'2030-01-01T00:00:00+00:00',
        'confirmation_policy_order':'must_be_established_by_future_round_registration',
        'bindings':{name:copy.deepcopy(b[name]) for name in ('plan','selector','metrics')}}
    selection['bindings']['collection']=copy.deepcopy(b['pilot_collection'])
    b['selection']=reader.put('registration/synthetic_selection.json',selection)
    fresh={'synthetic_fixture':True,'full_original_SFT500K_graph_exclusion_certified':True,
        'all_prior_tuning_training_confirmation_graphs_excluded':True,'confirmation_model_outcomes_read':False,
        'outcome_based_filtering':False,'references_independently_strict_verified_before_policy_calls':True,
        'validation_cohort_sha256':cohorts}
    b['freshness']=reader.put('registration/freshness.json',fresh)
    registry=sealed({'protocol':m.PROTOCOL+'-prior-input-registry','synthetic_fixture':True,'complete_known_input_history':True,
        'alpha_history_binding':b['round_history'],
        'graph_ids':{task:sorted({r['graph_semantic_id'] for field in ('training_rows','validation_rows') for r in pilots[task][field]})
                     for task in m.TASKS}},'registry_sha256')
    b['prior_input_registry']=reader.put('registration/input_history.json',registry)
    original=json.loads((EXP/'input_exclusion_certificate_v1/certificate.json').read_bytes())['certificate']
    cert=copy.deepcopy(original);cert['synthetic_fixture']=True
    cert['future_numeric_input_bindings']=[{'task':task,**b['numeric_inputs'][task],'rows':2,
        'graph_ids_sha256':m.canonical(graph_ids[task])} for task in m.TASKS]
    for target in cert['targets']:
        target['future_target_rows']=2;target['total_target_graphs']=226
    sealed(cert,'certificate_sha256')
    b['input_exclusion_certificate']=reader.put('registration/synthetic_full_scan_certificate.json',cert)
    request={'synthetic_fixture':True,'projection_source_sha256':m.PROJECTION_SHA,
        'prepared_manifest_sha256':cert['prepared_manifest_sha256'],
        'future_numeric_input_bindings':cert['future_numeric_input_bindings'],
        'future_targets':{task:[{'target_id':f'synthetic_future/{task}/{i}','graph_semantic_id':graph}
                         for i,graph in enumerate(graph_ids[task])] for task in m.TASKS}}
    b['input_scan_request']=reader.put('registration/input_scan_request.json',request)
    b['input_scan_custody']=reader.put('registration/input_scan_custody.json',{'synthetic_fixture':True,
        'protocol':m.PROTOCOL+'-input-scan-custody','independent_read_only_collection':True,'returncode':0,
        'request_sha256':b['input_scan_request']['sha256'],'projection_source_sha256':m.PROJECTION_SHA,
        'certificate_semantic_sha256':cert['certificate_sha256'],
        'native_stdout_binding':reader.put('registration/native_scan_stdout.jsonl',{'event':'input_scan_certificate','certificate':cert}),
        'native_stderr_binding':reader.put('registration/native_scan_stderr.log',b'')})
    b['reference_audit']=reader.put('registration/reference_audit.json',{'synthetic_fixture':True,
        'independent_strict_numeric_verification':True,'reference_kind':'heuristic_feasible_witness','policy_outcomes_read':False,'references':refs})
    b['pilot_timing']=reader.put('registration/synthetic_measured_timing.json',{'protocol':m.PROTOCOL+'-measured-pilot-timing',
        'synthetic_fixture':True,'independent_read_only_collection':True,'pilot_collection_sha256':b['pilot_collection']['sha256'],
        'measured_after_all12_admitted':True,'scope':'actual_full_production_pilot','slots':timing})
    authority_binding=reader.put('pilot/campaign_authority.json',(EXP/'collected_status_v2_20261014T0911/status/campaign_authority.json').read_bytes())
    failure_binding=reader.put('pilot/prior_failed_phase.json',(EXP/'validation/native_monitor/prior_failed_1030929_reverification_v2_bound.json').read_bytes())
    policy_binding=reader.put('policy/own_account.json',(EXP/'planning/native_self_account_resource_policy_v1.json').read_bytes())
    slots=[{'draw_id':d['draw_id'],'task':task,'arm':arm,'gpus':4,'requested_wall_seconds':60,'maximum_attempts':1,'automatic_retry':False}
           for d in roster for task in m.TASKS for arm in ('rloo_cardinal_beta005','foarl')]
    slots.extend({'draw_id':'sft_baseline','task':task,'arm':'sft','gpus':1,'requested_wall_seconds':60,'maximum_attempts':1,'automatic_retry':False} for task in m.TASKS)
    hours=sum(r['gpus']*r['requested_wall_seconds'] for r in slots)/3600
    authority=sealed({'protocol':m.PROTOCOL+'-resource-authority','synthetic_fixture':True,'source_only':True,
        'native_submission_authorized':False,'historical_pilot_requested_gpu_hours':92,'historical_formal_requested_gpu_hours':0,
        'pilot_authority_binding':authority_binding,'prior_failed_phase_binding':failure_binding,'planned_slots':slots,
        'scheduler':{'account':'vas@a100','partition':'gpu_p5','constraint':'a100','qos':'qos_gpu_a100-t3','maximum_wall_seconds':72000},
        'site_policy_binding':policy_binding,
        'measurement_projection_safety_factor':2,'projection_method':'measured_training_plus_500_over_96_evaluation_and_measured_overhead',
        'resource_adequacy_proven':False,'new_round_requested_gpu_hours':hours,'combined_reserved_requested_gpu_hours':92+hours,
        'job_step_strategy':'srun_single_slurm_task_4gpu_torchrun4','paired_training_world_size':4,
        'expected_slurm_steps':['batch','extern','0'],'formal_step_overhead_status':'unmeasured_no_formal_native_attempt',
        'formal_step_overhead_seconds':None,'additional_formal_site_overhead_uncertainty_disclosed':True,
        'declared_requested_gpu_hour_ceiling':92+hours},'authority_sha256')
    b['resource_authority']=reader.put('registration/resource_authority.json',authority)
    commitments={}
    names=('selection','source_manifest','seed_sampling','freshness','input_exclusion_certificate','reference_audit','resource_authority',
           'prior_input_registry','round_history','pilot_timing','input_scan_request','input_scan_custody')
    for name in names:
        second={'selection':1,'source_manifest':3,'seed_sampling':2}.get(name,4)
        commitments[name]=reader.put('custody/'+name+'.json',{'synthetic_fixture':True,'independent_custody_receipt':True,
            'file_sha256':b[name]['sha256'],'observed_utc':f'2030-01-01T00:00:0{second}+00:00'})
    b['pre_registration_custody']=reader.put('custody/pre_registration.json',{'protocol':m.PROTOCOL+'-pre-registration-custody',
        'synthetic_fixture':True,'independent_read_only_collection':True,'confirmation_policy_calls_seen':0,'commitments':commitments,
        'observed_utc':'2030-01-01T00:00:05+00:00'})
    proposal={'schema_version':1,'protocol':m.PROTOCOL+'-proposal','synthetic_fixture':True,'fixture_validation_count':2,
        'round_number':1,'bindings':b,'source_file_bindings':source_files,'seed_sampling':sampling,'seed_roster':roster,
        'original_sft_initializers':m.INITIALIZERS,'task_contract_bindings':task_bindings,
        'native_source_root':'/synthetic/formal/source','native_control_roots':{d['draw_id']:'/synthetic/formal/control/'+d['draw_id'] for d in roster}}
    return proposal,reader


@pytest.fixture(scope='module')
def packet():return fixture()


def cloned(packet):return copy.deepcopy(packet)


def replace(reader,binding,change,seal_key=None):
    v=json.loads(reader(binding));change(v)
    if seal_key:sealed(v,seal_key)
    new=reader.put(binding['path'],v);binding.update(new)
    return v


def test_marked_fixture_prepares_exact20_draws_without_registration_or_claim(packet):
    proposal,reader=packet;result=m.prepare_confirmation(proposal,reader,synthetic_fixture=True)
    contract=result['contract']
    assert len(contract['seed_roster'])==20 and len(contract['task_contract_bindings'])==20
    assert contract['seed_roster'][0]['train_seed']==contract['seed_roster'][-1]['train_seed']
    assert contract['seed_roster'][0]['draw_id']!=contract['seed_roster'][-1]['draw_id']
    assert contract['alpha_fraction']==[1,40] and contract['synthetic_fixture']
    assert not result['actual_registration_created'] and not result['native_submission_authorized']
    assert not result['superiority_established'] and not result['resource_adequacy_proven']
    assert result['separate_registered_formal_collector_and_consumer_still_required']
    assert m.seal(contract,'contract_sha256')==contract


def test_prepared_contract_preserves_frozen_analyzer_schema_for_offline_validation(packet):
    proposal,reader=packet;contract=m.prepare_confirmation(proposal,reader,synthetic_fixture=True)['contract']
    analysis=m.module((EXP/'confirmatory_analysis.py').read_bytes(),'_fixture_legacy_analysis')
    _,_,_,contracts,n=analysis.validate_registration(contract,reader,synthetic_fixture=True)
    assert len(contracts)==40 and n==2 and analysis.PRODUCTION_ADMISSION_READY is False


@pytest.mark.parametrize('count',[19,21])
def test_every_draw_is_retained_no_sample_size_change(packet,count):
    proposal,reader=cloned(packet)
    if count==19:proposal['seed_roster'].pop()
    else:proposal['seed_roster'].append({'draw_id':'draw20','train_seed':999})
    with pytest.raises(ValueError):m.validate_sampler(proposal,reader,json.loads(reader(proposal['bindings']['round_history'])),fixture=True)


@pytest.mark.parametrize('case',['replacement','excluded','duplicate_id','population','extra_entropy','missing_entropy','lost_history'])
def test_seed_draw_custody_cannot_select_replace_or_hide_prior_seeds(packet,case):
    proposal,reader=cloned(packet)
    if case=='replacement':proposal['seed_roster'][0]['train_seed']=777
    elif case=='excluded':proposal['seed_sampling']['excluded_previously_used_seeds']=[]
    elif case=='duplicate_id':proposal['seed_roster'][-1]['draw_id']='draw00'
    elif case=='population':proposal['seed_sampling']['distribution']='uniform_fixed_list'
    elif case in ('extra_entropy','missing_entropy'):
        sampler=json.loads(reader(proposal['bindings']['seed_sampling']));binding=sampler['entropy_binding'];raw=reader(binding)
        binding.update(reader.put(binding['path'],raw+b'\0'*4 if case=='extra_entropy' else raw[:-4]))
        proposal['bindings']['seed_sampling'].update(reader.put(proposal['bindings']['seed_sampling']['path'],sampler))
    else:
        replace(reader,proposal['bindings']['round_history'],lambda v:v.update(development_used_RL_seeds=[]),'history_sha256')
    with pytest.raises(ValueError):m.validate_sampler(proposal,reader,json.loads(reader(proposal['bindings']['round_history'])),fixture=True)


def test_uint32_complement_rejection_keeps_first20_values_and_duplicates():
    values=[43,7,7]+list(range(8,26))
    accepted=m.replay_seed_draws(b''.join(x.to_bytes(4,'little') for x in values),[43])
    assert accepted==values[1:] and accepted[:2]==[7,7]


def test_closed_prior_alpha_history_retains_duplicate_seeds_and_permanent_failures(packet):
    proposal,reader=cloned(packet)
    budget=sealed({'synthetic_fixture':True,'new_round_requested_gpu_hours':10},'authority_sha256')
    budget_binding=reader.put('history/round1_resource.json',budget)
    old=sealed({'synthetic_fixture':True,'protocol':m.CONTRACT_PROTOCOL,'round_number':1,'n_seed_draws':20,
        'alpha_fraction':[1,40],'seed_roster':proposal['seed_roster'],
        'preparation_evidence_bindings':{'resource_authority':budget_binding}},'contract_sha256')
    old_binding=reader.put('history/round1_contract.json',old)
    states=[{'draw_id':d['draw_id'],'state':'terminal'} for d in proposal['seed_roster']]
    states[-1].update(state='declared_permanent_failure',reason='synthetic unit failure',D=0)
    close=sealed({'synthetic_fixture':True,'contract_sha256':old['contract_sha256'],'state':'closed','draw_states':states},'ledger_sha256')
    close_binding=reader.put('history/round1_close.json',close)
    history=sealed({'synthetic_fixture':True,'protocol':m.PROTOCOL+'-alpha-history','development_used_RL_seeds':[43],
        'prior_formal_rounds':[{'round_number':1,'alpha_fraction':[1,40],'state':'closed','previous_round_receipt_sha256':None,
        'registered_contract_binding':old_binding,'training_seeds':[d['train_seed'] for d in proposal['seed_roster']],
        'closed_draw_ledger_binding':close_binding,'resource_authority_binding':budget_binding,'reserved_requested_gpu_hours':10}]},'history_sha256')
    used=m.validate_history(history,reader,2,fixture=True)
    assert used=={43}|{d['train_seed'] for d in proposal['seed_roster']}
    assert len(old['seed_roster'])==20 and len(used)==20
    close['draw_states'][-1]['D']=1;sealed(close,'ledger_sha256')
    close_binding.update(reader.put(close_binding['path'],close))
    sealed(history,'history_sha256')
    with pytest.raises(ValueError,match='Permanent failures'):m.validate_history(history,reader,2,fixture=True)


def test_alpha_history_cannot_skip_or_reset_rounds(packet):
    proposal,reader=packet;history=json.loads(reader(proposal['bindings']['round_history']))
    with pytest.raises(ValueError,match='reset or skip'):m.validate_history(history,reader,2,fixture=True)
    assert m.alpha(1)==[1,40] and m.alpha(2)==[1,80]


@pytest.mark.parametrize('name',['plan','selector','analysis','metrics','scorer','durable_verifier','projection'])
def test_every_prespecified_code_source_is_exact_byte_bound(packet,name):
    proposal,reader=cloned(packet);binding=proposal['bindings'][name]
    binding.update(reader.put(binding['path'],reader(binding)+b'\n'))
    with pytest.raises(ValueError,match='source/byte'):m.validate_source(proposal,reader,fixture=True)


@pytest.mark.parametrize('case',['partial','ineligible','no_candidate','changed_plan','not_admitted','confirmation_exposed'])
def test_selection_requires_all12_eligible_and_prespecified_unexposed_receipt(packet,case):
    proposal,reader=cloned(packet)
    def mutate(v):
        if case=='partial':v['all_development_slots_admitted']=11
        elif case=='ineligible':v['assessments'][0]['eligible']=False
        elif case=='no_candidate':v['selected_arm']=None
        elif case=='changed_plan':v['bindings']['plan']['sha256']='0'*64
        elif case=='not_admitted':v['assessment_requires_separate_production_admission']=True
        else:v['confirmation_outcomes_read']=True
    replace(reader,proposal['bindings']['selection'],mutate)
    with pytest.raises(ValueError):m.validate_selection(proposal,reader,fixture=True)


@pytest.mark.parametrize('case',['row_count','initializers','draw_contract_missing','training_rows','decoder_stream','source_root_collision'])
def test_terminal_recipe_and_all_duplicate_seed_draw_contracts_are_fixed(packet,case):
    proposal,reader=cloned(packet)
    if case=='row_count':proposal['fixture_validation_count']=3
    elif case=='initializers':proposal['original_sft_initializers']={}
    elif case=='draw_contract_missing':proposal['task_contract_bindings'].pop('draw19')
    elif case=='source_root_collision':
        proposal['native_control_roots']['draw19']=proposal['native_control_roots']['draw00']
        with pytest.raises(ValueError):m.prepare_confirmation(proposal,reader,synthetic_fixture=True)
        return
    else:
        binding=proposal['task_contract_bindings']['draw00']['op'];old=json.loads(reader(binding))
        if case=='training_rows':old['training_rows']=old['training_rows'][:-1]
        else:old['eval_seed']+=1
        # Rebuild a valid contract so the independent recipe comparison fires.
        replacement=scorer.make_contract('op',old['initializer'],old['training_rows'],old['validation_rows'],
            train_seed=old['train_seed'],eval_seed=old['eval_seed'],scope='production')
        binding.update(reader.put(binding['path'],replacement))
    with pytest.raises(ValueError):m.validate_task_contracts(proposal,reader,scorer,json.loads(reader(proposal['bindings']['selection'])),fixture=True)


@pytest.mark.parametrize('case',['partial_corpus','wrong_bytes','ambiguity','overlap','wrong_numeric','reference_claim','prior_graph','outcome_filter'])
def test_full_sft_fresh_numeric_and_independent_reference_custody_guards(packet,case):
    proposal,reader=cloned(packet)
    contracts={ (draw['draw_id'],t):json.loads(reader(proposal['task_contract_bindings'][draw['draw_id']][t])) for draw in proposal['seed_roster'] for t in m.TASKS}
    pilots={t:json.loads(reader(proposal['bindings']['pilot_task_contracts'][t])) for t in m.TASKS}
    if case in ('partial_corpus','wrong_bytes','ambiguity','overlap'):
        def mutate(v):
            row=v['corpora'][0]
            if case=='partial_corpus':row['observed']['rows']=499999
            elif case=='wrong_bytes':row['observed']['sha256']='0'*64
            elif case=='ambiguity':row['projection_ambiguities']=1
            else:row['overlapping_corpus_rows']=1
        replace(reader,proposal['bindings']['input_exclusion_certificate'],mutate,'certificate_sha256')
    elif case=='wrong_numeric':replace(reader,proposal['bindings']['numeric_inputs']['op'],lambda v:v[0].update(input='malformed'))
    elif case=='reference_claim':replace(reader,proposal['bindings']['reference_audit'],lambda v:v.update(reference_kind='certified_optimum'))
    elif case=='outcome_filter':replace(reader,proposal['bindings']['freshness'],lambda v:v.update(outcome_based_filtering=True))
    else:
        graph=contracts['draw00','op']['validation_rows'][0]['graph_semantic_id']
        replace(reader,proposal['bindings']['prior_input_registry'],lambda v:v['graph_ids']['op'].append(graph),'registry_sha256')
    with pytest.raises(ValueError):m.validate_freshness(proposal,reader,contracts,2,pilots,fixture=True)


@pytest.mark.parametrize('case',['different_graphs','different_metadata','wrong_request_sha','wrong_output','native_error','missing_target'])
def test_actual_scanned_targets_and_emitted_native_certificate_are_bound(packet,case):
    proposal,reader=cloned(packet)
    contracts={(d['draw_id'],t):json.loads(reader(proposal['task_contract_bindings'][d['draw_id']][t])) for d in proposal['seed_roster'] for t in m.TASKS}
    pilots={t:json.loads(reader(proposal['bindings']['pilot_task_contracts'][t])) for t in m.TASKS}
    request=json.loads(reader(proposal['bindings']['input_scan_request']))
    custody=json.loads(reader(proposal['bindings']['input_scan_custody']))
    if case in ('different_graphs','different_metadata','missing_target'):
        if case=='different_graphs':request['future_targets']['op'][0]['graph_semantic_id']='0'*64
        elif case=='different_metadata':request['future_numeric_input_bindings'][0]['rows']=999
        else:request['future_targets']['op'].pop()
        proposal['bindings']['input_scan_request'].update(reader.put(proposal['bindings']['input_scan_request']['path'],request))
        custody['request_sha256']=proposal['bindings']['input_scan_request']['sha256']
    elif case=='wrong_request_sha':custody['request_sha256']='0'*64
    elif case=='wrong_output':
        binding=custody['native_stdout_binding'];binding.update(reader.put(binding['path'],{'event':'not_completed'}))
    else:
        binding=custody['native_stderr_binding'];binding.update(reader.put(binding['path'],b'failed scan'))
    proposal['bindings']['input_scan_custody'].update(reader.put(proposal['bindings']['input_scan_custody']['path'],custody))
    with pytest.raises(ValueError):m.validate_freshness(proposal,reader,contracts,2,pilots,fixture=True)


@pytest.mark.parametrize('case',['incomplete_inventory','dropped_known_tuning_graph'])
def test_prior_input_registry_cannot_omit_known_tuning_history(packet,case):
    proposal,reader=cloned(packet)
    history=json.loads(reader(proposal['bindings']['round_history']))
    if case=='incomplete_inventory':history['development_input_inventory_complete']=False
    else:
        history['development_input_graph_bindings']['op'].append(reader.put('history/prior_tuning.json',['d'*64]))
    sealed(history,'history_sha256')
    proposal['bindings']['round_history'].update(reader.put(proposal['bindings']['round_history']['path'],history))
    registry=json.loads(reader(proposal['bindings']['prior_input_registry']))
    registry['alpha_history_binding']=copy.deepcopy(proposal['bindings']['round_history']);sealed(registry,'registry_sha256')
    proposal['bindings']['prior_input_registry'].update(reader.put(proposal['bindings']['prior_input_registry']['path'],registry))
    contracts={(d['draw_id'],t):json.loads(reader(proposal['task_contract_bindings'][d['draw_id']][t])) for d in proposal['seed_roster'] for t in m.TASKS}
    pilots={t:json.loads(reader(proposal['bindings']['pilot_task_contracts'][t])) for t in m.TASKS}
    with pytest.raises(ValueError):m.validate_freshness(proposal,reader,contracts,2,pilots,fixture=True)


@pytest.mark.parametrize('case',['unmeasured','missing_arm','fake_equal_compute','bad_gpu','failed_scheduler','wrong_result_time','extra_retry','missing_sft','drop_old_budget','under_wall','under_ceiling','adequacy_claim'])
def test_resource_plan_uses_actual_bound_timing_and_retains_every_reservation(packet,case):
    proposal,reader=cloned(packet)
    if case in ('unmeasured','missing_arm','fake_equal_compute','bad_gpu','wrong_result_time'):
        def change(v):
            if case=='unmeasured':v['measured_after_all12_admitted']=False
            elif case=='missing_arm':v['slots'].pop(m.PILOT_ORDER[0])
            elif case=='fake_equal_compute':v['scope']='assumed_equal_token_compute'
            elif case=='bad_gpu':v['slots'][m.PILOT_ORDER[0]]['actual_gpus']=0
            else:v['slots'][m.PILOT_ORDER[0]]['training_wall_seconds']=0
        replace(reader,proposal['bindings']['pilot_timing'],change)
    elif case=='failed_scheduler':
        timing=json.loads(reader(proposal['bindings']['pilot_timing']));binding=timing['slots'][m.PILOT_ORDER[0]]['scheduler_binding']
        replace(reader,binding,lambda v:v.update(stdout=v['stdout'].replace('COMPLETED','FAILED')))
        proposal['bindings']['pilot_timing'].update(reader.put(proposal['bindings']['pilot_timing']['path'],timing))
    else:
        def change(v):
            if case=='extra_retry':v['planned_slots'][0]['maximum_attempts']=2
            elif case=='missing_sft':v['planned_slots'].pop()
            elif case=='drop_old_budget':v['historical_pilot_requested_gpu_hours']=8
            elif case=='under_wall':v['planned_slots'][0]['requested_wall_seconds']=1
            elif case=='under_ceiling':v['declared_requested_gpu_hour_ceiling']=92
            else:v['resource_adequacy_proven']=True
        replace(reader,proposal['bindings']['resource_authority'],change,'authority_sha256')
    with pytest.raises(ValueError):m.validate_resource_authority(proposal,reader,json.loads(reader(proposal['bindings']['selection'])),2,fixture=True)


@pytest.mark.parametrize('key,value',[('job_step_strategy','torchrun_inside_batch'),('expected_slurm_steps',['batch','extern']),
    ('formal_step_overhead_status','measured'),('formal_step_overhead_seconds',0),
    ('additional_formal_site_overhead_uncertainty_disclosed',False)])
def test_future_srun_step_and_unmeasured_site_overhead_are_explicit(packet,key,value):
    proposal,reader=cloned(packet)
    replace(reader,proposal['bindings']['resource_authority'],lambda v:v.update(**{key:value}),'authority_sha256')
    with pytest.raises(ValueError,match='unmeasured new site'):m.validate_resource_authority(proposal,reader,
        json.loads(reader(proposal['bindings']['selection'])),2,fixture=True)


@pytest.mark.parametrize('case',['wrong_qos','wrong_account','longer_than20h','missing_policy'])
def test_formal_t3_20h_bound_requires_actual_own_account_site_policy(packet,case):
    proposal,reader=cloned(packet)
    def change(v):
        if case=='wrong_qos':v['scheduler']['qos']='qos_gpu_a100-dev'
        elif case=='wrong_account':v['scheduler']['account']='other-account'
        elif case=='longer_than20h':v['planned_slots'][0]['requested_wall_seconds']=72001
        else:v['site_policy_binding']['sha256']='0'*64
    replace(reader,proposal['bindings']['resource_authority'],change,'authority_sha256')
    with pytest.raises(ValueError):m.validate_resource_authority(proposal,reader,
        json.loads(reader(proposal['bindings']['selection'])),2,fixture=True)


@pytest.mark.parametrize('case',['policy_seen','missing_reference','source_before_selection','self_asserted','wrong_sha'])
def test_independent_pre_registration_chronology_is_closed_without_first_call_cycle(packet,case):
    proposal,reader=cloned(packet)
    custody=json.loads(reader(proposal['bindings']['pre_registration_custody']))
    if case=='policy_seen':custody['confirmation_policy_calls_seen']=1
    elif case=='missing_reference':custody['commitments'].pop('reference_audit')
    else:
        name='source_manifest' if case=='source_before_selection' else 'reference_audit'
        binding=custody['commitments'][name]
        replace(reader,binding,lambda v:v.update(**({'observed_utc':'2029-01-01T00:00:00+00:00'} if case=='source_before_selection' else
            {'independent_custody_receipt':False} if case=='self_asserted' else {'file_sha256':'0'*64})))
    proposal['bindings']['pre_registration_custody'].update(reader.put(proposal['bindings']['pre_registration_custody']['path'],custody))
    with pytest.raises(ValueError):m.validate_pre_registration_custody(proposal,reader,json.loads(reader(proposal['bindings']['selection'])),
        json.loads(reader(proposal['bindings']['seed_sampling'])),fixture=True)


def test_fixture_markers_cannot_be_removed_to_prepare_production(packet):
    proposal,reader=cloned(packet)
    with pytest.raises(ValueError,match='Synthetic'):m.prepare_confirmation(proposal,reader)
    proposal.pop('synthetic_fixture')
    with pytest.raises(ValueError):m.prepare_confirmation(proposal,reader)


def test_default_cli_never_reads_outcomes_draws_or_submits():
    result=subprocess.run([sys.executable,'-I','-S','-B',str(EXP/'prepare_confirmation_round_v1.py')],check=True,capture_output=True,text=True,timeout=10)
    row=json.loads(result.stdout)
    assert row['production_execution_enabled'] is False and row['native_mutations'] is False
    assert row['status']=='source_only_no_candidate_roster_cohort_registration_or_jobs'


def test_immutable_reader_rejects_mutable_escaping_or_tampered_bytes(tmp_path):
    reader=m.ImmutableReader(tmp_path);path=tmp_path/'evidence.json';path.write_bytes(b'{}')
    binding={'path':'evidence.json','sha256':m.sha(b'{}'),'size_bytes':2}
    with pytest.raises(ValueError,match='immutable'):reader(binding)
    path.chmod(0o440);assert reader(binding)==b'{}'
    with pytest.raises(ValueError,match='Unsafe'):reader({**binding,'path':'../evidence.json'})
    with pytest.raises(ValueError,match='byte binding'):reader({**binding,'sha256':'0'*64})
