"""Source/statistical continuity and synthetic producer custody; no policy calls."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
from datetime import datetime,timezone,timedelta
import pytest

R=Path(__file__).parents[1];B=R/'experiments/rloo_one_shot_20261016';P=B/'run_confirmation_campaign_reconciled_v4.py';OLD=B/'confirmation_execution_v3/pipeline/run_confirmation_campaign_v3.py'


def load(path):
 raw=Path(path).read_bytes();m=types.ModuleType('explicit_synthetic_reconciliation_guard');m.__file__=str(path)
 m.__executed_source_sha256__=hashlib.sha256(raw).hexdigest();exec(compile(raw,str(path),'exec',dont_inherit=True),m.__dict__);return m


def fixture(tmp_path):
 m=load(P);master_path=B/'teacher_execution_v2/execution_packet.json';master=json.loads(master_path.read_bytes());oldroot=Path(master['output_root'])
 rows={};values={}
 def add(name,value):
  p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(m.encoded(value)+b'\n');return m.binding(p)
 master_binding=m.binding(master_path)
 owner_path=tmp_path/'new_reconciliation/control/owner.json';registration_path=owner_path.with_name('reconciliation_registration.json')
 bp=tmp_path/'bootstrap.py';bp.write_text('import json\nfrom pathlib import Path\ndef load_context(row): return json.loads(Path(row["path"]).read_bytes()), {}\ndef verify_context_extension(new, old):\n a,_=load_context(new);b,_=load_context(old)\n assert all(a[k]==b[k] for k in ("bootstrap","adapter","master_packet","project_manifest","extra_manifest"))\n assert set(b.get("roles",[]))<=set(a.get("roles",[]))\n return {"status":"explicitly_synthetic_extension_only"}\n')
 bootstrap=m.binding(bp);adapter=add('adapter.json',{'scope':'pure metadata fixture, no execution'});controller=add('newcontroller.json',{'scope':'pure metadata fixture, no execution'})
 context=add('context.json',{'scope':'pure metadata fixture, no execution','bootstrap':bootstrap,'adapter':adapter,'master_packet':master_binding,
 'project_manifest':{'sha256':'f48c01a35cbfddca21facced4d57493396b9a40f2a1f146115b890f4362d843a'},'extra_manifest':{'sha256':m.MASTER_MANIFEST_SHA},'roles':['original']})
 m.EXPECTED_RUNTIME_BOOTSTRAP_SHA=bootstrap['sha256'];m.EXPECTED_ADAPTER_SHA=adapter['sha256']
 rp={'protocol':m.RECONCILIATION_PROTOCOL,'master_packet':master_binding,'controller':controller,
 'maximum_total_retained_GPU_hours':20.5,'already_launched_GPU_hours':1.5,'maximum_remaining_GPU_hours':19.0,'additional_GPU_hours':0,
 'execution_deadline_utc':m.TEACHER_DEADLINE,'automatic_retry':False,'output_root':str(tmp_path/'new_reconciliation'),
 'canonical_downstream_root':str(oldroot),'runtime_context':context,'runtime_bootstrap':bootstrap,'adapter':adapter,
 'failed_controller_status':m.binding(oldroot/'control/status.json'),
 'failed_CPU_launch':m.binding(oldroot/'control/CPU_training_preparation.launch.json'),
 'failed_CPU_log':m.binding(oldroot/'control/CPU_training_preparation.log'),
 'old_collection':{k:m.binding(oldroot/rel) for k,rel in [('contract','contracts/collection.json'),('result','runs/collection/result.json'),('completion','runs/collection/completion.json')]}}
 rp['packet_sha256']=m.sha(m.encoded(rp));pb=add('newpacket.json',rp)
 owner={'protocol':m.RECONCILIATION_PROTOCOL,'packet_binding':pb,'source':controller,'automatic_retry':False,'created_utc':'2026-10-16T22:00:00+00:00',
        'pid':99999999,'uid':1610,'hostname':'seacove','start_ticks':'1'}
 owner_binding=add('new_reconciliation/control/owner.json',owner)
 reg={'protocol':m.RECONCILIATION_PROTOCOL+'-registration','packet_binding':pb,'master_packet':master_binding,
 'old_failure_bindings':{k:rp[k] for k in ('failed_controller_status','failed_CPU_launch','failed_CPU_log')},'old_collection':rp['old_collection'],
 'runtime_context':context,'runtime_bootstrap':bootstrap,'adapter':adapter,'remaining_phases':list(m.ALL_RECONCILIATION_PHASES),
 'maximum_total_retained_GPU_hours':20.5,'already_launched_GPU_hours':1.5,'maximum_remaining_GPU_hours':19.0,'additional_GPU_hours':0,'automatic_retry':False,
 'registered_utc':'2026-10-16T22:00:01+00:00'}
 reg['receipt_sha256']=m.sha(m.encoded(reg));rb=add('new_reconciliation/control/reconciliation_registration.json',reg)
 add('new_reconciliation/control/status.json',{'owner':owner,'status':'registered_reconciliation'})
 packet={'master_packet':master_binding,'reconciliation':{'packet':pb,'owner':owner_binding,'registration':rb},
         'runtime_context':context,'runtime_bootstrap':bootstrap,'adapter':adapter}
 return m,packet,rp,owner,reg,add


def test_whole_module_AST_inverse_has_exact_two_operational_edits():
 m=load(P);raw=OLD.read_bytes();tree,proof=m.composition(raw);original=ast.parse(raw)
 for node in tree.body:
  if isinstance(node,ast.FunctionDef) and node.name=='_execute':
   original_node=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='_execute')
   oldguard=next(n for n in original_node.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and len(n.value.args)==2 and isinstance(n.value.args[1],ast.Constant) and n.value.args[1].value=='Exact fixed local controller source/runtime owner required')
   i=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='_operational_source_guard');node.body[i]=copy.deepcopy(oldguard)
  if isinstance(node,ast.FunctionDef) and node.name=='freeze_roster':
   for n in ast.walk(node):
    if isinstance(n,ast.Assign) and isinstance(n.value,ast.Dict) and any(isinstance(x,ast.Name) and x.id=='authority' for x in n.targets):
     assert n.value.keys[-1].value=='operational_handoff';n.value.keys.pop();n.value.values.pop()
 assert ast.dump(tree,include_attributes=False)==ast.dump(original,include_attributes=False)
 assert proof['callback_overrides']==list(m.CALLBACKS) and proof['numerical_and_statistical_functions_changed'] is False


def test_compiled_original_numerical_budget_slot_seed_logic_is_identical():
 m=load(P);tree,proof=m.composition(OLD.read_bytes());legacy=load(OLD);composed=types.ModuleType('synthetic_composition');composed.__file__=str(OLD)
 composed.__executed_source_sha256__=m.LEGACY_SHA;exec(compile(tree,str(OLD),'exec',dont_inherit=True),composed.__dict__)
 for branch in ('pure','teacher'):
  assert composed.fixed_slots(branch,'/tmp/metadata_fixture')==legacy.fixed_slots(branch,'/tmp/metadata_fixture')
  assert composed.budget(branch)==legacy.budget(branch)
 assert composed.SEEDS==legacy.SEEDS==(101,211,307,401,503)
 assert composed.__file__==str(OLD) and composed.__executed_source_sha256__==m.LEGACY_SHA
 assert len(composed.fixed_slots('pure','/tmp/metadata_fixture'))==28
 assert len(composed.fixed_slots('teacher','/tmp/metadata_fixture'))==48


def test_changed_original_controller_source_never_compiled():
 m=load(P)
 with pytest.raises(ValueError):m.composition(OLD.read_bytes()+b'\n')


def test_default_command_has_no_outputs(tmp_path):
 r=subprocess.run([sys.executable,'-I','-B',str(P)],cwd=tmp_path,capture_output=True,text=True,check=True)
 assert json.loads(r.stdout)['new_authority_created'] is False and not list(tmp_path.iterdir())


def test_actual_pinned_old_metadata_and_synthetic_new_owner_positive(tmp_path):
 m,p,rp,owner,reg,add=fixture(tmp_path);v=m.validate_reconciliation(p)
 assert v['state']['status']=='registered_reconciliation' and v['packet']['already_launched_GPU_hours']==1.5
 # No teacher/raw/target/model artifact is loaded by this metadata-only function.
 assert v['control']==Path(rp['output_root'])/'control'


@pytest.mark.parametrize('key,value',[('maximum_total_retained_GPU_hours',19),('already_launched_GPU_hours',0),('maximum_remaining_GPU_hours',20.5),
 ('additional_GPU_hours',1),('additional_GPU_hours',False),('execution_deadline_utc','2026-10-30T12:00:00+00:00'),('automatic_retry',True),
 ('canonical_downstream_root','/tmp/rewritten'),('protocol','oldfailedcontroller'),('synthetic_fixture',True)])
def test_resealed_new_producer_cannot_reclaim_or_relabel_scope(tmp_path,key,value):
 m,p,rp,owner,reg,add=fixture(tmp_path);rp[key]=value;rp.pop('packet_sha256');rp['packet_sha256']=m.sha(m.encoded(rp));p['reconciliation']['packet']=add('newpacket.json',rp)
 with pytest.raises(ValueError):m.validate_reconciliation(p)


@pytest.mark.parametrize('field',["failed_controller_status","failed_CPU_launch","failed_CPU_log"])
def test_original_failed_phase_bytes_are_mandatory(tmp_path,field):
 m,p,rp,owner,reg,add=fixture(tmp_path);rp[field]=add('replaced.json',{'status':'success'});rp.pop('packet_sha256');rp['packet_sha256']=m.sha(m.encoded(rp));p['reconciliation']['packet']=add('newpacket.json',rp)
 with pytest.raises(ValueError):m.validate_reconciliation(p)


@pytest.mark.parametrize('field',['contract','result','completion'])
def test_original_collection_cannot_be_replaced(tmp_path,field):
 m,p,rp,owner,reg,add=fixture(tmp_path);rp['old_collection'][field]=add('recollect.json',{'status':'newcollection'});rp.pop('packet_sha256');rp['packet_sha256']=m.sha(m.encoded(rp));p['reconciliation']['packet']=add('newpacket.json',rp)
 with pytest.raises(ValueError):m.validate_reconciliation(p)


@pytest.mark.parametrize('change',[lambda x:x.update(maximum_remaining_GPU_hours=20.5),lambda x:x.update(additional_GPU_hours=False),
 lambda x:x.update(remaining_phases=list(reversed(x['remaining_phases']))),lambda x:x.update(automatic_retry=True),
 lambda x:x.update(runtime_context={'different':True}),lambda x:x.update(old_collection={}),
 lambda x:x.update(registered_utc='2026-10-16T21:59:00+00:00'),lambda x:x.update(synthetic_fixture=True)])
def test_resealed_registration_drift_rejects_before_quality(tmp_path,change):
 m,p,rp,owner,reg,add=fixture(tmp_path);change(reg);reg.pop('receipt_sha256');reg['receipt_sha256']=m.sha(m.encoded(reg));p['reconciliation']['registration']=add('new_reconciliation/control/reconciliation_registration.json',reg)
 with pytest.raises(ValueError):m.validate_reconciliation(p)


def test_new_status_must_retain_exact_new_owner(tmp_path):
 m,p,rp,owner,reg,add=fixture(tmp_path);add('new_reconciliation/control/status.json',{'owner':{'old':'owner'},'status':'development_pass_requires_fresh_confirmation'})
 with pytest.raises(ValueError):m.validate_reconciliation(p)


def test_new_owner_must_bind_actual_new_packet_not_oldfailedpacket(tmp_path):
 m,p,rp,owner,reg,add=fixture(tmp_path);owner['packet_binding']=p['master_packet'];p['reconciliation']['owner']=add('new_reconciliation/control/owner.json',owner)
 with pytest.raises(ValueError):m.validate_reconciliation(p)


def test_source_rebinding_after_execution_is_rejected(tmp_path):
 q=tmp_path/'v4.py';q.write_bytes(P.read_bytes());m=load(q);q.write_bytes(q.read_bytes()+b'\n')
 with pytest.raises(ValueError):m.self_binding()


@pytest.mark.parametrize('field',['bootstrap','adapter','master_packet','project_manifest','extra_manifest'])
def test_context_cannot_rebind_source_or_master_before_phase(tmp_path,field):
 m,p,rp,owner,reg,add=fixture(tmp_path);context=json.loads(m.read_bound(p['runtime_context']));context[field]={'sha256':'0'*64}
 p['runtime_context']=add('rebound_context.json',context)
 with pytest.raises(ValueError):m.validate_reconciliation(p)


def test_explicit_additive_context_preserves_old_registration(tmp_path):
 m,p,rp,owner,reg,add=fixture(tmp_path);context=json.loads(m.read_bound(p['runtime_context']));context['roles'].append('V4')
 p['runtime_context']=add('extended_context.json',context)
 value=m.validate_reconciliation(p)
 assert value['context_extension']['status']=='explicitly_synthetic_extension_only'
 assert value['registration']['runtime_context']==rp['runtime_context']!=p['runtime_context']


@pytest.mark.parametrize('elapsed,cpu,late',[(2.,True,False),(602.,False,False),(0.,True,True),(-1.,True,False),(float('nan'),True,False)])
def test_successful_exit_does_not_bypass_elapsed_or_UTC_cap(tmp_path,monkeypatch,elapsed,cpu,late):
 m=load(P);monkeypatch.setattr(m.time,'monotonic',lambda:100.+elapsed)
 with pytest.raises(ValueError):m.check_completed_phase(100.,1,cpu,datetime.now(timezone.utc)+timedelta(days=-1 if late else 1))


def test_valid_successful_child_bound(tmp_path,monkeypatch):
 m=load(P);monkeypatch.setattr(m.time,'monotonic',lambda:100.5)
 m.check_completed_phase(100.,1,True,datetime.now(timezone.utc)+timedelta(days=1))
