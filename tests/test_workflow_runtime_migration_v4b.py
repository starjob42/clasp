"""Root-bound zero-call closure and unchanged numerical/runtime bodies."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import pytest

ROOT=Path(__file__).parents[1];B=ROOT/'experiments/rloo_one_shot_20261016';D=B/'cpu_preparation_reconciliation_v3'


def load(path):
 raw=path.read_bytes();m=types.ModuleType('synthetic_operational_migration_guard');m.__file__=str(path);m.__executed_source_sha256__=hashlib.sha256(raw).hexdigest()
 exec(compile(raw,str(path),'exec',dont_inherit=True),m.__dict__);return m


def closure_fixture(tmp_path):
 m=load(B/'run_confirmation_campaign_reconciled_v4b.py');q=D/'runtime_dispatch_reconciliation_v3b/workflow_closure_v4b.json'
 value=json.loads(m.read_bound(m.binding(q)))
 def adopt(value):
  value={k:v for k,v in value.items() if k!='receipt_sha256'};value['receipt_sha256']=m.sha(m.encoded(value));p=tmp_path/'explicitly_synthetic_closure.json';p.write_bytes(m.encoded(value));row=m.binding(p)
  m.PREDECESSOR_CLOSURE_SHA=row['sha256'];return {'operational_predecessor_closure':row}
 return m,value,adopt


def test_actual_pinned_three_predecessor_closure_metadata():
 m=load(B/'run_confirmation_campaign_reconciled_v4b.py');q=D/'runtime_dispatch_reconciliation_v3b/workflow_closure_v4b.json'
 v=m.validate_predecessor_closure({'operational_predecessor_closure':m.binding(q)})
 assert len(v['rows'])==3 and all(r['process']['closed'] for r in v['rows'])


@pytest.mark.parametrize('change',[lambda v:v.update(no_new_GPU_calls=False),lambda v:v.update(no_fresh_confirmation_calls=False),
 lambda v:v.update(no_new_paper_writes=False),lambda v:v.update(rows=v['rows'][:2]),
 lambda v:v['rows'][0]['process'].update(closed=False),lambda v:v['rows'][0]['process'].update(pid=True),
 lambda v:v['rows'][0]['process'].update(final_kernel_state='S'),lambda v:v['rows'][0]['process'].update(observed_utc='2026-10-15T00:00:00+00:00')])
def test_zero_call_identity_and_chronology_are_required_even_for_resealed_fixture(tmp_path,change):
 m,v,adopt=closure_fixture(tmp_path);change(v);p=adopt(v)
 with pytest.raises(ValueError):m.validate_predecessor_closure(p)


def test_caller_cannot_self_promote_resealed_closure(tmp_path):
 m,v,adopt=closure_fixture(tmp_path);v['reason']='synthetic caller'
 p=tmp_path/'copied.json';p.write_bytes(m.encoded(v))
 with pytest.raises(ValueError):m.validate_predecessor_closure({'operational_predecessor_closure':m.binding(p)})


def test_actual_prospective_packet_adopts_prior_CPU_only_without_reexecution():
 m=load(B/'run_confirmation_campaign_reconciled_v4b.py');p=D/'runtime_dispatch_reconciliation_v3b/registration_v3b/execution_packet.json'
 rp=json.loads(m.read_bound(m.binding(p)))
 # This is a prospective registration-field projection, not an actual new
 # owner/registration admission. No CPU search, model or policy runs here.
 m.validate_adopted_CPU(rp,rp)
 assert rp['adopted_completed_phases']==['CPU_training_preparation_reconciled'] and len(rp['remaining_phases'])==8


@pytest.mark.parametrize('change',[lambda v:v.update(operational_revision='silently_repeat_old_CPU'),lambda v:v.update(adopted_completed_phases=[]),
 lambda v:v.update(retained_phase_order=list(reversed(v['retained_phase_order']))),lambda v:v.update(adopted_calibration={})])
def test_adopted_CPU_and_remaining_phase_disposition_cannot_be_rebound(change):
 m=load(B/'run_confirmation_campaign_reconciled_v4b.py');p=D/'runtime_dispatch_reconciliation_v3b/registration_v3b/execution_packet.json'
 rp=json.loads(m.read_bound(m.binding(p)));registration=copy.deepcopy(rp);change(rp)
 with pytest.raises(ValueError):m.validate_adopted_CPU(rp,registration)


def test_complete_original_functions_unchanged_outside_admission_migration():
 for old,new,allowed,added in [
  ('run_confirmation_campaign_reconciled_v4.py','run_confirmation_campaign_reconciled_v4b.py',{'validate_reconciliation','load_campaign','prepare_packet'},{'validate_predecessor_closure','validate_adopted_CPU'}),
  ('watch_reconciled_paper_updates_v4.py','watch_reconciled_paper_updates_v4b.py',{'validate','decision','prepare_packet'},set())]:
  a=ast.parse((B/old).read_bytes());b=ast.parse((B/new).read_bytes());af={n.name:ast.dump(n,include_attributes=False) for n in a.body if isinstance(n,ast.FunctionDef)};bf={n.name:ast.dump(n,include_attributes=False) for n in b.body if isinstance(n,ast.FunctionDef)}
  assert bf.keys()-af.keys()==added
  assert {k for k in af if af[k]!=bf[k]}==allowed
 m=load(B/'run_confirmation_campaign_reconciled_v4b.py');old=load(B/'run_confirmation_campaign_reconciled_v4.py');raw=(B/'confirmation_execution_v3/pipeline/run_confirmation_campaign_v3.py').read_bytes()
 a,p=m.composition(raw);b,q=old.composition(raw)
 assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False) and p==q


@pytest.mark.parametrize('source',['run_confirmation_campaign_reconciled_v4b.py','watch_reconciled_paper_updates_v4b.py'])
def test_default_new_sources_are_inert(tmp_path,source):
 r=subprocess.run([sys.executable,'-I','-B',str(B/source)],cwd=tmp_path,capture_output=True,text=True,check=True)
 assert json.loads(r.stdout)['GPU_calls']==0 and not list(tmp_path.iterdir())
