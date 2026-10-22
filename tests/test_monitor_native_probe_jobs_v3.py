"""No SSH/native action; exercise bounded watch transport and local persistence."""
import hashlib
import json
from pathlib import Path
import subprocess
import types
import pytest
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'experiments/rloo_sft_anchor_20261013/monitor_native_probe_jobs_v3.py'
m=types.ModuleType('readonly_watch_tests');m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)


def result(value):return types.SimpleNamespace(returncode=0,stdout=json.dumps(value),stderr='')
def valid():return {'protocol':m.PROTOCOL,'native_mutations':False,'model_or_policy_calls':0,'benchmark_results_admitted':False,'jobs':list(m.JOBS)}


def test_exact_readonly_ssh_host_scope_and_timeout(monkeypatch):
 def call(argv,**kw):
  assert argv[:-1]==['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','HostName=jean-zay1.idris.fr','-o','HostKeyAlias=jean-zay.idris.fr','-o','StrictHostKeyChecking=yes','jean-zay']
  assert kw['timeout']==50 and '1074755,1063203' in argv[-1]
  assert '--jobs' in argv[-1] and 'sbatch' not in argv[-1] and 'scontrol update' not in argv[-1]
  return result(valid())
 monkeypatch.setattr(m.subprocess,'run',call);r=m.snapshot()
 assert r['ssh_host']=='jean-zay1.idris.fr' and r['remote_program_sha256']==hashlib.sha256(m.REMOTE.encode()).hexdigest()

@pytest.mark.parametrize('fault',['mutation','model','admission','jobs','badjson','sshfailed'])
def test_wrong_or_failed_remote_transport_never_admits_state(monkeypatch,fault):
 v=valid()
 if fault=='mutation':v['native_mutations']=True
 elif fault=='model':v['model_or_policy_calls']=1
 elif fault=='admission':v['benchmark_results_admitted']=True
 elif fault=='jobs':v['jobs']=['foreign']
 r=result(v)
 if fault=='badjson':r.stdout='truncated'
 elif fault=='sshfailed':r.returncode=1;r.stderr='network error'
 monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:r)
 with pytest.raises((ValueError,RuntimeError)):m.snapshot()


def test_local_append_and_atomic_latest_do_not_create_other_files(tmp_path):
 v=valid();m.persist(tmp_path,v);m.persist(tmp_path,{**v,'read_error':'TimeoutExpired','terminal_state_inferred_from_read_error':False})
 assert len((tmp_path/'observations.jsonl').read_text().splitlines())==2
 assert json.loads((tmp_path/'latest.json').read_bytes())['terminal_state_inferred_from_read_error'] is False
 assert sorted(p.name for p in tmp_path.iterdir())==['latest.json','observations.jsonl']


def test_compiled_remote_is_stdlib_only_and_two_day_poll_bound():
 compile(m.REMOTE,'<closed remote>','exec')
 assert m.MAXIMUM_SECONDS==172800 and m.INTERVAL==60
 assert 'torch' not in m.REMOTE and 'transformers' not in m.REMOTE
 assert "'owner'" in m.REMOTE and "getpass.getuser()!='uag72kz'" in m.REMOTE


def test_controller_projection_preserves_actual_slot_heartbeat_job_and_state():
 import ast
 node=next(n for n in ast.parse(m.REMOTE).body if isinstance(n,ast.FunctionDef) and n.name=='controller_projection')
 scope={'hashlib':hashlib};exec(compile(ast.Module(body=[node],type_ignores=[]),'<closed controller projection>','exec'),scope)
 v={'state':'running','slot':'op.rloo_scalar_beta0','recorded_utc':'2026-10-14T18:46:00+00:00','job_id':'1063203',
    'scheduler_state':'PENDING','quality_field':'must be omitted'}
 raw=json.dumps(v).encode();r=scope['controller_projection'](v,raw)
 assert r['current_slot']==r['slot']=='op.rloo_scalar_beta0'
 assert r['heartbeat_utc']==r['recorded_utc']==v['recorded_utc']
 assert r['job_id']=='1063203' and r['scheduler_state']=='PENDING' and 'quality_field' not in r
 assert r['metadata_sha256']==hashlib.sha256(raw).hexdigest()


def test_full_remote_program_projects_real_controller_field_schema(tmp_path,monkeypatch,capsys):
 import getpass,pathlib
 original_path=pathlib.Path
 control=tmp_path/'control';probe=tmp_path/'probe';control.mkdir();probe.mkdir();(control/'campaign_owners').mkdir()
 root=original_path(__file__).resolve().parents[1]
 manifest=(root/'experiments/rloo_sft_anchor_20261013/control_v2/launch_manifest.json').read_bytes()
 (control/'launch_manifest.json').write_bytes(manifest)
 # Recorded field schema comes from the actual collected controller artifact;
 # the pending slot/job below is the explicit root-provided display example.
 original=json.loads((root/'experiments/rloo_sft_anchor_20261013/collected_native_first_complete_v1/status/campaign_status.json').read_bytes())
 assert 'slot' in original and 'recorded_utc' in original
 value={**original,'slot':'op.rloo_scalar_beta0','recorded_utc':'2026-10-14T18:58:31+00:00','job_id':'1063203','scheduler_state':'PENDING'}
 (control/'campaign_status.json').write_text(json.dumps(value))
 def mapped_path(s):
  if s=='/lustre/fsn1/projects/rech/vas/uag72kz/rloo_sft_anchor_20261013/control_v2':return control
  if s=='/lustre/fsn1/projects/rech/vas/uag72kz/rloo_sft_anchor_cvrp_timing_probe_20261014_v2/control_v1':return probe
  return original_path(s)
 with monkeypatch.context() as patch:
  patch.setattr(pathlib,'Path',mapped_path);patch.setattr(getpass,'getuser',lambda:'uag72kz')
  patch.setattr(subprocess,'run',lambda *a,**k:types.SimpleNamespace(returncode=0,stdout='',stderr=''))
  exec(compile(m.REMOTE,'<complete source-bound remote fixture>','exec'),{})
 result=json.loads(capsys.readouterr().out)
 assert result['controller']['current_slot']=='op.rloo_scalar_beta0'
 assert result['controller']['heartbeat_utc']==result['controller']['recorded_utc']=='2026-10-14T18:58:31+00:00'
 assert result['controller']['job_id']=='1063203' and result['controller']['scheduler_state']=='PENDING'
 assert result['native_mutations'] is False and result['model_or_policy_calls']==0
