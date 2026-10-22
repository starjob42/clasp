"""No-network metadata watcher guards. No model or actual native calls."""
import ast
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import types

import pytest

ROOT=Path(__file__).parents[1]
SOURCE=ROOT/'experiments/rloo_sft_anchor_20261013/resource_deadline_extension_v2/monitor_timing_campaign_v2.py'


def module():
    raw=SOURCE.read_bytes();m=types.ModuleType('diagnostic_native_watch_v2');m.__file__=str(SOURCE)
    exec(compile(raw,str(SOURCE),'exec',dont_inherit=True),m.__dict__);return m


def scope(m):return {'path':m.REGISTRATION_PATH,'sha256':'a'*64,'size_bytes':123}


def test_cold_cli_is_zero_effect_and_network(tmp_path):
    result=subprocess.run([sys.executable,'-I','-S','-B',str(SOURCE)],cwd=tmp_path,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['status']=='prepared_only_no_network_no_quality_reads' and not list(tmp_path.iterdir())


@pytest.mark.parametrize('fault',['path','hash','bool_size','missing_size'])
def test_native_scope_requires_exact_path_pin_and_size_before_network(fault):
    m=module();row=scope(m)
    if fault=='path':row['path']=m.NAMESPACE+'/unrelated/registration.json'
    elif fault=='hash':row['sha256']='x'*64
    elif fault=='bool_size':row['size_bytes']=True
    else:del row['size_bytes']
    with pytest.raises(ValueError):m.ssh_argv(row)


def test_ssh_route_keeps_trusted_service_host_key_and_ownerhost():
    m=module();argv=m.ssh_argv(scope(m))
    assert argv[:2]==['ssh','-o'] and 'StrictHostKeyChecking=yes' in argv and 'HostName=jean-zay1.idris.fr' in argv
    assert 'HostKeyAlias=jean-zay.idris.fr' in argv and argv[-2]=='jean-zay'
    remote=shlex.split(argv[-1]);assert remote[:4]==['/usr/bin/python3','-I','-S','-B']
    compile(remote[-1],'<diagnostic-native-reader>','exec')
    assert '__SCOPE_LITERAL__' not in remote[-1]


def test_native_program_does_not_execute_any_bound_project_source():
    m=module();tree=ast.parse(m.REMOTE.replace('__SCOPE_LITERAL__',repr(json.dumps(scope(m)))))
    forbidden={'exec','eval','compile','Popen','kill','signal','remove','unlink','mkdir','rename','replace','write_bytes','write_text'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
            assert name not in forbidden
            if name=='open':assert len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value=='rb'
    assert "'result.json'" in m.REMOTE and "'content_opened':False" in m.REMOTE
    assert "queries=[];jobs=','.join(a['job_id'] for a in attempts)" in m.REMOTE
    assert "['squeue','--jobs',jobs" in m.REMOTE and "['sacct','--jobs',jobs" in m.REMOTE


def test_snapshot_uses_one_bounded_readonly_process_and_admits_typed_scope():
    m=module();seen=[];row=scope(m)
    value={'protocol':m.PROTOCOL,'registration_binding':row,'native_mutations':False,'quality_contents_opened':False,'policy_calls':0}
    def run(argv,**kw):seen.append((argv,kw));return types.SimpleNamespace(returncode=0,stdout=json.dumps(value),stderr='')
    assert m.snapshot(row,run=run)==value and len(seen)==1 and seen[0][1]['timeout']==58


@pytest.mark.parametrize('fault',['model_flag','bool_calls','wrong_registration','native_mutation','bad_json','nonzero'])
def test_reader_failure_or_rebound_scope_cannot_be_reported_as_valid(fault):
    m=module();row=scope(m);value={'protocol':m.PROTOCOL,'registration_binding':row,'native_mutations':False,'quality_contents_opened':False,'policy_calls':0};rc=0
    if fault=='model_flag':value['quality_contents_opened']=True
    elif fault=='bool_calls':value['policy_calls']=False
    elif fault=='wrong_registration':value['registration_binding']={**row,'sha256':'b'*64}
    elif fault=='native_mutation':value['native_mutations']=True
    elif fault=='nonzero':rc=1
    raw='notJSON' if fault=='bad_json' else json.dumps(value)
    with pytest.raises((ValueError,RuntimeError)):m.snapshot(row,run=lambda *a,**kw:types.SimpleNamespace(returncode=rc,stdout=raw,stderr='diagnostic'))


def test_local_snapshot_keeps_append_history_and_atomic_latest(tmp_path):
    m=module();m.write_snapshot(tmp_path,{'sequence':1});m.write_snapshot(tmp_path,{'sequence':2})
    assert [json.loads(v)['sequence'] for v in (tmp_path/'observations.jsonl').read_text().splitlines()]==[1,2]
    assert json.loads((tmp_path/'latest.json').read_bytes())['sequence']==2 and not (tmp_path/'latest.pending.json').exists()


def test_one_shot_read_error_is_saved_without_terminal_inference(tmp_path,monkeypatch):
    m=module();folder=tmp_path/'watch';deadline=(datetime.now(timezone.utc)+timedelta(seconds=60)).isoformat()
    monkeypatch.setattr(sys,'argv',[str(SOURCE),'--execute','--registration',m.REGISTRATION_PATH,'--registration-sha256','a'*64,
                                  '--registration-size','123','--output',str(folder),'--deadline-utc',deadline,'--once'])
    monkeypatch.setattr(m,'snapshot',lambda row:(_ for _ in ()).throw(TimeoutError('pure fixture')))
    m.main();value=json.loads((folder/'latest.json').read_bytes())
    assert value['terminal_inferred_from_error'] is False and value['quality_contents_opened'] is False
    assert value['policy_calls']==0 and 'TimeoutError' in value['read_error']


def test_existing_watch_output_is_never_reused(tmp_path,monkeypatch):
    m=module();deadline=(datetime.now(timezone.utc)+timedelta(seconds=60)).isoformat()
    monkeypatch.setattr(sys,'argv',[str(SOURCE),'--execute','--registration',m.REGISTRATION_PATH,'--registration-sha256','a'*64,
                                  '--registration-size','123','--output',str(tmp_path),'--deadline-utc',deadline,'--once'])
    monkeypatch.setattr(m,'snapshot',lambda row:pytest.fail('Network before unique output guard'))
    with pytest.raises(FileExistsError):m.main()
