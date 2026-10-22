"""Synthetic metadata and exact source composition; no native/model operations."""
import ast
import hashlib
import json
from pathlib import Path
import types
import copy
import pytest

ROOT=Path(__file__).parents[1]
BASE=ROOT/'experiments/rloo_sft_anchor_20261013'
PATH=BASE/'resource_deadline_extension_v2/cvrp_timing_operator_v2.py'
OLD=BASE/'resource_extension_v1/cvrp_wall_resource_v1.py'
HELPER=BASE/'resource_deadline_extension_v2/timing_addendum_v2.py'


def load(p):
 raw=p.read_bytes();m=types.ModuleType('synthetic_source_operator');m.__file__=str(p)
 m.__executed_source_sha256__=hashlib.sha256(raw).hexdigest();exec(compile(raw,str(p),'exec',dont_inherit=True),m.__dict__);return m


def profile():
 m=load(PATH);h=load(HELPER);raw=OLD.read_bytes();tree,proof=m.composition(raw,h.PROPOSED_DEADLINE)
 p=types.ModuleType('synthetic_profile');p.__file__=str(OLD);p.__executed_source_sha256__=m.LEGACY_SHA
 exec(compile(tree,str(OLD),'exec',dont_inherit=True),p.__dict__)
 return m,h,p,proof


def authority(m,h):
 return dict(protocol=m.PROTOCOL+'-authority',historical_requested_gpu_hours_retained=177,additional_requested_gpu_hours=0,
 cumulative_requested_gpu_hours=177,maximum_active_study_jobs=1,order=list(m.ORDER),remaining_slots=list(m.REMAINING),
 execution_deadline_utc=h.PROPOSED_DEADLINE,absolute_deadline_utc=m.ABSOLUTE_DEADLINE,first_job_execution_deadline_utc=m.OLD_DEADLINE,
 automatic_retry=False,automatic_cancellation=False,rescue_allocations_granted=0,quality_selection_allowed=False,publication_eligible=False,
 superiority_established=False,legacy_authority_sha256=m.LEGACY_AUTHORITY_SHA,legacy_registration_sha256=m.LEGACY_REGISTRATION_SHA,
 first_attempt_sha256=m.FIRST_ATTEMPT_SHA,addendum_sha256=h.make_proposal()['addendum_sha256'],registered_utc='2026-10-16T12:00:00+00:00')


def test_complete_source_AST_exact_declared_operational_delta():
 m,h,p,proof=profile();original=ast.parse(OLD.read_bytes());expected=ast.parse(OLD.read_bytes())
 for node in expected.body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
   k=node.targets[0].id
   if k=='PROTOCOL':node.value=ast.Constant(m.PROTOCOL)
   if k=='EXECUTION_DEADLINE':node.value=ast.Constant(h.PROPOSED_DEADLINE)
   if k=='EXTENSION':node.value=ast.parse("NAMESPACE / 'resource_deadline_extension_v2'",mode='eval').body
  if isinstance(node,ast.FunctionDef) and node.name=='build_outer_script':
   for c in ast.walk(node):
    if isinstance(c,ast.Constant) and c.value=='cvrp_wall_resource_v1.py':c.value='cvrp_timing_operator_v2.py'
 actual,_=m.composition(OLD.read_bytes(),h.PROPOSED_DEADLINE)
 assert ast.dump(actual,include_attributes=False)==ast.dump(expected,include_attributes=False)
 # All other global constants/imports/numerical and resource function bodies retained.
 assert proof['original_source_sha256']==hashlib.sha256(OLD.read_bytes()).hexdigest()
 assert p.SCRIPT_SHA==load(OLD).SCRIPT_SHA and p.resources(m.FIRST_SLOT)==h.resource_spec(m.FIRST_SLOT)
 assert p.EXECUTION_DEADLINE==h.PROPOSED_DEADLINE and load(OLD).EXECUTION_DEADLINE==m.OLD_DEADLINE


@pytest.mark.parametrize('deadline',[None,True,'2026-10-17T11:26:42.646651+00:00','2026-10-29T00:00:00+00:00','2026-10-27T12:00:00'])
def test_invalid_operational_clock_rejects(deadline):
 m=load(PATH)
 with pytest.raises((ValueError,TypeError)):m.composition(OLD.read_bytes(),deadline)


def test_modified_original_source_not_compiled():
 m=load(PATH)
 with pytest.raises(ValueError):m.composition(OLD.read_bytes()+b'\n',load(HELPER).PROPOSED_DEADLINE)


def test_source_loader_executes_verified_buffer_ignoring_pyc(tmp_path):
 m=load(PATH);path=tmp_path/'module.py';path.write_text('answer=999\n');raw=b'answer=7\n'
 x=m.compile_buffer(raw,path,'exact_buffer',m.sha(raw));assert x.answer==7
 with pytest.raises(ValueError):m.compile_buffer(raw,path,'wrong',m.sha(path.read_bytes()))


def test_late_file_source_rebinding_denied(tmp_path):
 p=tmp_path/'operator.py';p.write_bytes(PATH.read_bytes());m=load(p);p.write_bytes(p.read_bytes()+b'\n')
 with pytest.raises(ValueError,match='executed buffer'):m.self_identity()


def test_real_legacy_authority_and_frozen_sources_preserved():
 m=load(PATH);h=load(HELPER);old=load(OLD)
 a=(BASE/'resource_deadline_extension_v2/legacy_inputs_v1/authority.json').read_bytes()
 f=(BASE/'resource_deadline_extension_v2/legacy_inputs_v1/registration.json').read_bytes()
 assert m.sha(a)==m.LEGACY_AUTHORITY_SHA and m.sha(f)==m.LEGACY_REGISTRATION_SHA
 old.validate_authority(json.loads(a));assert m.CLOCK_HELPER_SHA==m.sha(HELPER.read_bytes())
 assert m.LEGACY_COLLECTOR_SHA==m.sha((BASE/'resource_extension_v1/collect_resource_campaign_v1.py').read_bytes())
 assert json.loads(a)['cumulative_requested_gpu_hours']==177 and m.validate_authority(authority(m,h),h.make_proposal())


@pytest.mark.parametrize('key,value',[('historical_requested_gpu_hours_retained',96),('additional_requested_gpu_hours',1),
 ('additional_requested_gpu_hours',False),('cumulative_requested_gpu_hours',178),('maximum_active_study_jobs',True),
 ('automatic_retry',True),('automatic_cancellation',True),('rescue_allocations_granted',False),('quality_selection_allowed',True),
 ('first_job_execution_deadline_utc','2026-10-27T12:00:00+00:00'),('first_attempt_sha256','a'*64),
 ('legacy_registration_sha256','b'*64),('legacy_authority_sha256','c'*64),('remaining_slots',[]),('publication_eligible',True)])
def test_resealed_authority_changes_cannot_grant_extra_budget_retry_or_first_clock(key,value):
 m=load(PATH);h=load(HELPER);a=authority(m,h);a[key]=value
 with pytest.raises(ValueError):m.validate_authority(a,h.make_proposal())


def test_original_script_bytes_numerical_body_and_effective_resources_retained():
 m,h,p,proof=profile();own=dict(path=str(m.NEW_ROOT/'cvrp_timing_operator_v2.py'),sha256='a'*64,size_bytes=7)
 for slot in m.REMAINING:
  original=(BASE/'control_v2/job_scripts'/(slot+'.slurm')).read_bytes()
  new=p.build_outer_script(slot,original,own).decode()
  assert ('--time=03:00:00' if slot=='cvrp.sft' else '--time=06:00:00') in new
  assert '--qos=qos_gpu_a100-t3' in new and '--mem=' not in new
  assert str(m.NEW_ROOT/'cvrp_timing_operator_v2.py') in new
  assert 'train_sft_anchor_pilot' not in new and '--hold' in new
  assert original==(BASE/'control_v2/job_scripts'/(slot+'.slurm')).read_bytes()


def test_first_and_future_attempts_dispatch_distinct_contexts():
 m=load(PATH);events=[];attempt={'slot':m.FIRST_SLOT,'job_id':m.FIRST_JOB}
 m.stable=lambda p:b'first';m.FIRST_ATTEMPT_SHA=m.sha(b'first')
 old=types.SimpleNamespace(validate_attempt_chain=lambda ctx,a:events.append(('old',ctx,a)) or 'legacy')
 ctx={'legacy_operator':old,'legacy_context':{'tag':'old'}}
 assert m.validate_attempt_chain(ctx,attempt)=='legacy';assert events[0][0]=='old'
 with pytest.raises(ValueError):m.validate_attempt_chain(ctx,dict(attempt,job_id='9'))


def test_first_completion_and_startup_cannot_copy_other_native_routes():
 m=load(PATH);value={'protocol':m.PROTOCOL+'-original-first-admission','slot':m.FIRST_SLOT,'job_id':m.FIRST_JOB,
 'attempt_sha256':m.FIRST_ATTEMPT_SHA,'legacy_registration_sha256':m.LEGACY_REGISTRATION_SHA,'legacy_operator_sha256':m.LEGACY_SHA,
 'quality_summary':None,'registration_sha256':'a'*64,'authority_sha256':'b'*64,'recorded_utc':'2026-10-18T00:00:00+00:00',
 'complete_binding':{'path':'/tmp/copied_complete.json'},'startup_binding':{'path':'/tmp/copied_startup.json'}}
 value['receipt_sha256']=m.sha(m.encoded(value));raw=m.encoded(value)
 ctx={'read':lambda p:raw,'registration_sha256':'a'*64,'authority_sha256':'b'*64,
      'registration':{'registered_utc':'2026-10-16T00:00:00+00:00'},'authority':{'execution_deadline_utc':'2026-10-27T12:00:00+00:00'}}
 with pytest.raises(ValueError,match='native routes'):m.load_first_admission(ctx)


def test_first_job_cannot_be_resubmitted_or_released():
 m,h,p,proof=profile();ctx={'registration_sha256':'a'*64,'base_sbatch_argv':p.effective_sbatch_argv}
 m.load_handoff=lambda c:{'sha256':'b'*64}
 with pytest.raises(ValueError):m.effective_sbatch_argv(ctx,m.FIRST_SLOT,'a'*64)
 with pytest.raises(ValueError):m.release_one({},m.FIRST_SLOT)
 for slot in m.REMAINING:
  v=m.effective_sbatch_argv(ctx,slot,'a'*64)
  assert v[0:3]==['sbatch','--parsable','--hold'] and 'CLASP_TIMING_HANDOFF_SHA='+'b'*64 in v[-2]
  assert v[-1]==str(m.NEW_ROOT/'effective_scripts'/(slot+'.slurm'))


def test_native_action_closed_without_actual_owner_and_external_handoff_pin():
 m=load(PATH)
 with pytest.raises(ValueError):m.actual({'synthetic_fixture':True},lambda:None)


def test_current_first_failure_retains_old_no_retry_barrier():
 m,h,p,proof=profile();job=m.FIRST_JOB;slot=m.FIRST_SLOT
 q={'argv':['sacct','--noheader','--parsable2','--jobs',job,'--format='+p.SACCT_FORMAT],
 'returncode':0,'stderr':'','stdout':job+'|TIMEOUT|21600|0:0|gres/gpu=4|gres/gpu=4\n'}
 with pytest.raises(ValueError,match='failed'):p.validate_completed(slot,{'job_id':job},q,None)


def test_default_source_has_no_model_import_or_scheduler_action_at_import():
 m=load(PATH);assert m.REMAINING==load(HELPER).REMAINING
 tree=ast.parse(PATH.read_bytes());imports={q.name.split('.')[0] for n in tree.body if isinstance(n,ast.Import) for q in n.names}
 assert not imports & {'torch','transformers','trl','clasp'}


def handoff_chain(tmp_path):
 m=load(PATH);h=load(BASE/'resource_deadline_extension_v2/handoff_controller_v2.py')
 m.NEW_ROOT=tmp_path;folder=tmp_path/'handoff';folder.mkdir()
 owner=h.OWNER.copy();frozen={**owner,'state':'T','children':[],'sig_ignored':'0','sig_caught':'0','sig_blocked':'0'}
 boundary={'first_query':{'stdout':'metadata fixture only'},'native_mutations':False}
 spec=h.seal({'protocol':h.PROTOCOL+'-spec','old_owner':owner,'new_registration_sha256':'a'*64,'synthetic_fixture':False})
 intent=h.seal({'protocol':h.PROTOCOL+'-intent','spec_sha256':m.sha(m.encoded(spec)),'recorded_utc':'2026-10-16T12:00:00+00:00',
                'boundary':boundary,'synthetic_fixture':False})
 quiesced=h.seal({'protocol':h.PROTOCOL+'-quiesced','recorded_utc':'2026-10-16T12:00:01+00:00','owner':frozen,
                 'signals':['SIGSTOP'],'boundary':boundary,'synthetic_fixture':False})
 complete=h.seal({'protocol':h.PROTOCOL+'-complete','status':'old_cpu_owner_terminated_no_scheduler_mutations',
                 'new_registration_sha256':'a'*64,'old_registration_sha256':m.LEGACY_REGISTRATION_SHA,
                 'old_operator_sha256':m.LEGACY_SHA,'first_job_id':m.FIRST_JOB,'synthetic_fixture':False,'scheduler_mutations':[],
                 'handoff_helper_sha256':'c'*64,'old_owner':owner,'spec_sha256':m.sha(m.encoded(spec)),
                 'signals':['SIGSTOP','SIGTERM','SIGCONT'],'final_owner':{'pid':owner['pid'],'alive':False},
                 'boundary_before':boundary,'boundary_after_quiesce':boundary,'recorded_utc':'2026-10-16T12:00:02+00:00'})
 values=dict(spec=spec,intent=intent,quiesced=quiesced,complete=complete)
 for n,v in values.items():(folder/(n+'.json')).write_bytes(m.encoded(v)+b'\n')
 ctx={'read':m.stable,'registration_sha256':'a'*64,'registration':{'handoff_helper_binding':{'sha256':'c'*64}}}
 return m,h,ctx,values,folder


def test_handoff_join_replays_actual_event_structure_before_new_operator(tmp_path):
 m,h,ctx,values,folder=handoff_chain(tmp_path)
 row=m.load_handoff(ctx);assert row['sha256']==m.sha((folder/'complete.json').read_bytes())
 with pytest.raises(ValueError):m.load_handoff(ctx,'0'*64)


@pytest.mark.parametrize('which,change',[
 ('complete',lambda v:v.update(signals=['SIGTERM','SIGCONT'])),
 ('complete',lambda v:v['old_owner'].update(pid=1630145)),
 ('complete',lambda v:v.update(final_owner={'pid':1630145,'alive':False})),
 ('complete',lambda v:v.update(final_owner={'pid':1630144,'alive':True,'state':'S'})),
 ('complete',lambda v:v.update(boundary_before={'different':True})),
 ('quiesced',lambda v:v['owner'].update(sig_blocked='4000')),
 ('quiesced',lambda v:v['owner'].update(state='S')),
 ('quiesced',lambda v:v['owner'].update(children=[99])),
 ('quiesced',lambda v:v.update(recorded_utc='2026-10-16T12:00:03+00:00')),
 ('intent',lambda v:v.update(spec_sha256='f'*64)),
 ('spec',lambda v:v.update(synthetic_fixture=True)),
 ('complete',lambda v:v['boundary_before'].update(synthetic_fixture=True)),
])
def test_resealed_handoff_missing_identity_source_stop_or_default_term_denied(tmp_path,which,change):
 m,h,ctx,values,folder=handoff_chain(tmp_path);change(values[which]);values[which]=h.seal(values[which])
 (folder/(which+'.json')).write_bytes(m.encoded(values[which])+b'\n')
 with pytest.raises(ValueError):m.load_handoff(ctx)


def test_first_positive_old_metadata_and_original_deadline_not_relabelled(tmp_path):
 m=load(PATH);h=load(HELPER);old=load(OLD)
 m.NEW_ROOT=tmp_path/'new';m.OLD_ROOT=tmp_path/'old';m.CONTROL=tmp_path/'control'
 for path in (m.NEW_ROOT,m.OLD_ROOT/'attempts',m.OLD_ROOT/'startup_proofs',m.CONTROL/'outputs'/m.FIRST_SLOT):path.mkdir(parents=True)
 attempt_raw=(BASE/'resource_deadline_extension_v2/legacy_inputs_v1/first_attempt.json').read_bytes()
 (m.OLD_ROOT/'attempts'/(m.FIRST_SLOT+'.json')).write_bytes(attempt_raw)
 complete={'status':'completed_single_arm_model_evaluation','operational_version':2,'superiority_established':False,
 'checkpoint_reload_exact':True,'actual_RL_optimizer_steps':100,'actual_RL_responses':3600,'actual_evaluation_responses':864,'device':'cuda:0'}
 cp=m.CONTROL/'outputs'/m.FIRST_SLOT/'complete.json';cp.write_bytes(m.encoded(complete)+b'\n')
 q={'argv':['scontrol','show','job',m.FIRST_JOB],'returncode':0,'stderr':'','time_zone':'UTC',
 'recorded_utc':'2026-10-16T13:00:01+00:00','stdout':'JobId='+m.FIRST_JOB+' JobState=RUNNING StartTime=2026-10-16T13:00:00 TimeLimit=06:00:00'}
 startup={'query':q,'slot':m.FIRST_SLOT,'job_id':m.FIRST_JOB,'recorded_utc':'2026-10-16T13:00:02+00:00',
 'operator_sha256':m.LEGACY_SHA,'registration_sha256':m.LEGACY_REGISTRATION_SHA,'execution_deadline_utc':m.OLD_DEADLINE,
 'deadline_check_before_model':True,'model_calls_by_guard':0,'resources':h.resource_spec(m.FIRST_SLOT),
 'original_script_sha256':h.SCRIPT_SHA[m.FIRST_SLOT],'attempt_sha256':m.FIRST_ATTEMPT_SHA}
 sp=m.OLD_ROOT/'startup_proofs'/(m.FIRST_SLOT+'.json');sp.write_bytes(m.encoded(startup)+b'\n')
 aq={'argv':['sacct','--noheader','--parsable2','--jobs',m.FIRST_JOB,'--format='+old.SACCT_FORMAT],
 'returncode':0,'stderr':'','stdout':'\n'.join(x+'|COMPLETED|100|0:0|gres/gpu=4|gres/gpu=4' for x in (m.FIRST_JOB,m.FIRST_JOB+'.batch',m.FIRST_JOB+'.extern'))}
 v={'protocol':m.PROTOCOL+'-original-first-admission','slot':m.FIRST_SLOT,'job_id':m.FIRST_JOB,'attempt_sha256':m.FIRST_ATTEMPT_SHA,
 'legacy_registration_sha256':m.LEGACY_REGISTRATION_SHA,'legacy_operator_sha256':m.LEGACY_SHA,'quality_summary':None,
 'registration_sha256':'a'*64,'authority_sha256':'b'*64,'recorded_utc':'2026-10-16T20:00:00+00:00',
 'complete_binding':m.binding(cp),'startup_binding':m.binding(sp),'accounting':aq,'original_durable_admission':{'unit_fixture_only':True}}
 v['receipt_sha256']=m.sha(m.encoded(v));(m.NEW_ROOT/'first_admission.json').write_bytes(m.encoded(v)+b'\n')
 ctx={'read':m.stable,'registration_sha256':'a'*64,'authority_sha256':'b'*64,'registration':{'registered_utc':'2026-10-16T12:00:00+00:00'},
      'authority':{'execution_deadline_utc':h.PROPOSED_DEADLINE},'legacy_operator':old,'clock_helper':h,'addendum':h.make_proposal()}
 assert m.load_first_admission(ctx)['slot']==m.FIRST_SLOT
 # This is synthetic metadata only; no durable scientific/source admission is granted here.
 startup['execution_deadline_utc']=h.PROPOSED_DEADLINE;sp.write_bytes(m.encoded(startup));v['startup_binding']=m.binding(sp)
 v.pop('receipt_sha256');v['receipt_sha256']=m.sha(m.encoded(v));(m.NEW_ROOT/'first_admission.json').write_bytes(m.encoded(v))
 with pytest.raises(ValueError):m.load_first_admission(ctx)


def test_effective_initializer_input_map_drift_denied_before_any_new_read():
 m=load(PATH);seen=[]
 old=types.SimpleNamespace(verify_inputs=lambda ctx:seen.append('original_source_check'))
 ctx={'legacy_operator':old,'legacy_context':{'registration':{'input_bindings':{'initializer_inventory':{'sha256':'a'*64}}}},
 'registration':{'input_bindings':{}},'read':lambda path:pytest.fail('No new read allowed after drift')}
 with pytest.raises(ValueError,match='input map'):m.verify_inputs(ctx)
 assert seen==['original_source_check']


def test_caller_reader_cannot_authorize_scheduler_actions_even_native_shape():
 m=load(PATH)
 with pytest.raises(ValueError,match='readers'):
  m.actual({'synthetic_fixture':False,'read':lambda path:b'claimed'},lambda:None)


def test_future_startup_export_proof_adds_only_operational_handoff_pin():
 m,h,p,proof=profile();ctx={'registration_sha256':'a'*64,'base_sbatch_argv':p.effective_sbatch_argv}
 m.load_handoff=lambda ctx:{'sha256':'b'*64}
 old_argv=p.effective_sbatch_argv(m.REMAINING[0],'a'*64)
 new_argv=m.effective_sbatch_argv(ctx,m.REMAINING[0],'a'*64)
 assert new_argv[:-2]==old_argv[:-2] and new_argv[-1]==old_argv[-1]
 assert new_argv[-2]==old_argv[-2]+',CLASP_TIMING_HANDOFF_SHA='+'b'*64
