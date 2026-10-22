"""No SSH/native action; exercise bounded watch transport and local persistence."""
import hashlib
import json
from pathlib import Path
import subprocess
import types
import pytest
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'experiments/rloo_sft_anchor_20261013/monitor_native_probe_jobs_v2.py'
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

