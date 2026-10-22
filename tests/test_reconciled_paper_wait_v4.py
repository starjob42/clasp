"""CPU-only metadata/writer-order and owned-child cleanup diagnostics."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from datetime import datetime,timezone,timedelta
import pytest

R=Path(__file__).parents[1];B=R/'experiments/rloo_one_shot_20261016';P=B/'watch_reconciled_paper_updates_v4.py';V4=B/'run_confirmation_campaign_reconciled_v4.py'


def load(p):
 raw=Path(p).read_bytes();m=types.ModuleType('synthetic_paper_wait_guard');m.__file__=str(p);m.__executed_source_sha256__=hashlib.sha256(raw).hexdigest()
 exec(compile(raw,str(p),'exec',dont_inherit=True),m.__dict__);return m


def metadata_fixture(tmp_path):
 f=load(R/'tests/test_confirmation_reconciliation_handoff_v4.py');v4,view,rp,owner,reg,add=f.fixture(tmp_path)
 m=load(P);m.HANDOFF_VALIDATOR_SHA=hashlib.sha256(V4.read_bytes()).hexdigest();m.EXPECTED_RUNTIME_BOOTSTRAP_SHA=view['runtime_bootstrap']['sha256'];m.EXPECTED_ADAPTER_SHA=view['adapter']['sha256']
 original_module=m.module
 def fixture_module(row,name):
  value=original_module(row,name)
  if row['sha256']==m.HANDOFF_VALIDATOR_SHA:
   value.EXPECTED_RUNTIME_BOOTSTRAP_SHA=m.EXPECTED_RUNTIME_BOOTSTRAP_SHA;value.EXPECTED_ADAPTER_SHA=m.EXPECTED_ADAPTER_SHA
  return value
 m.module=fixture_module
 old=[]
 for stem in ('teacher_execution_v2','confirmation_execution_v3'):
  pp=B/stem/'paper_update_packet.json';ownerfile=B/stem/'paper_update_watch/owner.json'
  old.append({'packet':m.binding(pp),'owner':m.binding(ownerfile)})
 packet=m.prepare_packet(role='teacher',source=m.binding(P),producer_packet=view['reconciliation']['packet'],producer_owner=view['reconciliation']['owner'],
 prior_writers=old,publisher=m.binding(B/'publish_teacher_paper_v2.py'),renderer=m.binding(B/'render_teacher_op_paper_v2.py'),
 legacy_watcher=m.binding(B/'watch_teacher_paper_update_v2.py'),runtime_bootstrap=view['runtime_bootstrap'],runtime_context=view['runtime_context'],adapter=view['adapter'],
 handoff_validator=m.binding(V4),output=tmp_path/'watch',publication_output=tmp_path/'publication',producer_registration=view['reconciliation']['registration'])
 return m,packet,owner,rp,add


def decision_fixture(tmp_path,role='teacher'):
 m=load(P)
 def add(path,value):
  p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(m.encoded(value));return m.binding(p)
 owner={'pid':99999999,'uid':os.getuid(),'hostname':'synthetic','start_ticks':'1'}
 ownerrow=add('producer/control/owner.json',owner)
 master=add('master.json',{'output_root':str(tmp_path/'canonical')})
 pp={'master_packet':master};status={'owner':owner,'status':'running'};add('producer/control/status.json',status)
 prev={'pid':99999998,'uid':os.getuid(),'hostname':'synthetic','start_ticks':'2'}
 prevowner=add('old/owner.json',prev);add('old/status.json',{'owner':prev,'status':'stop_without_paper_changes'})
 packet={'role':role,'producer_owner':ownerrow,'prior_writers':[{'owner':prevowner}]}
 expected=tmp_path/('canonical/control/teacher_OP_analysis.json' if role=='teacher' else 'producer/control/confirmation_analysis.json')
 expected.parent.mkdir(parents=True,exist_ok=True);expected.write_text('{}');analysis=m.binding(expected)
 return m,packet,owner,pp,status,prev,analysis,add


def test_default_has_no_model_or_paper_calls(tmp_path):
 r=subprocess.run([sys.executable,'-I','-B',str(P)],cwd=tmp_path,capture_output=True,text=True,check=True)
 assert json.loads(r.stdout)['paper_writes']==0 and not list(tmp_path.iterdir())


def test_full_metadata_validation_preserves_actual_old_writer_lineage(tmp_path):
 m,p,owner,rp,add=metadata_fixture(tmp_path);a,b=m.validate(p)
 assert a==owner and b==rp and p['role']=='teacher'


@pytest.mark.parametrize('key,value',[('publisher_timeout_seconds',True),('publisher_timeout_seconds',7201),('snapshot_timeout_seconds',121),
 ('automatic_retry',True),('superiority_established',True),('deadline_utc','2026-10-30T00:00:00+00:00'),('role','anything')])
def test_resealed_caps_or_roles_fail_closed(tmp_path,key,value):
 m,p,owner,rp,add=metadata_fixture(tmp_path);p[key]=value;p=m.seal(p)
 with pytest.raises(ValueError):m.validate(p)


def test_both_actual_stopped_writer_packets_are_required(tmp_path):
 m,p,owner,rp,add=metadata_fixture(tmp_path);p['prior_writers']=p['prior_writers'][:1];p=m.seal(p)
 with pytest.raises(ValueError):m.validate(p)


def test_unknown_publisher_not_executed(tmp_path):
 m,p,owner,rp,add=metadata_fixture(tmp_path);p['publisher']=add('unknown.py',{'no':'execution'});p=m.seal(p)
 with pytest.raises(ValueError):m.validate(p)


def test_default_unfrozen_operational_pin_blocks_actual_watch(tmp_path):
 m,p,owner,rp,add=metadata_fixture(tmp_path);m.EXPECTED_RUNTIME_BOOTSTRAP_SHA=None
 with pytest.raises(ValueError):m.validate(p)


@pytest.mark.parametrize('role',['teacher','confirmation'])
@pytest.mark.parametrize('passed',[True,False])
def test_complete_pass_and_fail_can_publish_only_after_every_writer_is_closed(tmp_path,role,passed):
 m,p,owner,pp,state,prev,analysis,add=decision_fixture(tmp_path,role)
 state['status']=('development_pass_requires_fresh_confirmation' if passed else 'development_failed_requires_next_candidate') if role=='teacher' else ('completed_confirmation_scoped_gate_passed' if passed else 'completed_confirmation_scoped_gate_failed')
 state['analysis' if role=='teacher' else 'confirmation_analysis']=analysis;add('producer/control/status.json',state)
 assert m.decision(p,owner,pp,closed=lambda o:True)==('snapshot_then_publish',analysis)
 assert m.decision(p,owner,pp,closed=lambda o:False)[0]=='wait_for_producer_to_exit'
 assert m.decision(p,owner,pp,closed=lambda o:o==owner)[0]=='wait_for_prior_paper_writer'


@pytest.mark.parametrize('status',['running','registered_reconciliation','waiting_for_idle_GPU'])
def test_incomplete_study_never_opens_quality(tmp_path,status):
 m,p,owner,pp,state,prev,analysis,add=decision_fixture(tmp_path);state['status']=status;add('producer/control/status.json',state)
 m.read_bound=lambda row:pytest.fail('No report or previous quality may be opened')
 assert m.decision(p,owner,pp,closed=lambda o:True)==('wait_for_complete_reconciled_producer',None)


@pytest.mark.parametrize('status',['failed_requires_reconciliation','no_passing_development_candidate','original_requires_reconciliation'])
def test_failed_producer_stops_without_snapshot_or_report_read(tmp_path,status):
 m,p,owner,pp,state,prev,analysis,add=decision_fixture(tmp_path);state['status']=status;add('producer/control/status.json',state)
 m.read_bound=lambda row:pytest.fail('No failed producer report may be opened')
 assert m.decision(p,owner,pp,closed=lambda o:True)==('stop_without_paper_changes',None)


def test_prior_failed_paper_write_requires_reconciliation(tmp_path):
 m,p,owner,pp,state,prev,analysis,add=decision_fixture(tmp_path);state.update(status='development_pass_requires_fresh_confirmation',analysis=analysis);add('producer/control/status.json',state)
 add('old/status.json',{'owner':prev,'status':'failed_requires_reconciliation'})
 with pytest.raises(ValueError):m.decision(p,owner,pp,closed=lambda o:True)


def test_analysis_cannot_escape_actual_canonical_old_downstream_root(tmp_path):
 m,p,owner,pp,state,prev,analysis,add=decision_fixture(tmp_path);state.update(status='development_pass_requires_fresh_confirmation',analysis=add('copied/report.json',{}));add('producer/control/status.json',state)
 with pytest.raises(ValueError):m.decision(p,owner,pp,closed=lambda o:True)


def test_changed_producer_owner_fails_before_report(tmp_path):
 m,p,owner,pp,state,prev,analysis,add=decision_fixture(tmp_path);state['owner']={'different':'owner'};add('producer/control/status.json',state)
 with pytest.raises(ValueError):m.decision(p,owner,pp,closed=lambda o:True)


def test_timed_CPU_child_is_owned_and_reaped_on_first_heartbeat_exception(tmp_path,monkeypatch):
 m=load(P);children=[];real=subprocess.Popen
 def launch(*a,**kw):
  p=real(*a,**kw);children.append(p);return p
 runtime=types.SimpleNamespace(command=lambda *args:[sys.executable,'-I','-B','-c','import time;time.sleep(30)'],verify_receipt=lambda *a:None)
 monkeypatch.setattr(m,'module',lambda *a:runtime);monkeypatch.setattr(m.subprocess,'Popen',launch)
 monkeypatch.setattr(m,'heartbeat',lambda *a:(_ for _ in ()).throw(InterruptedError('synthetic postspawn heartbeat')))
 p={'deadline_utc':m.DEADLINES['teacher'],'runtime_bootstrap':{},'runtime_context':{},'publisher':{}}
 try:
  with pytest.raises(InterruptedError):m.run_cpu(p,[],'synthetic_owned_CPU_only',.5,tmp_path,{})
  assert len(children)==1 and children[0].poll() is not None
 finally:
  for process in children:
   if process.poll() is None:os.killpg(process.pid,9);process.wait()


def test_guarded_teacher_publish_requires_all_three_actual_layer_files(tmp_path,monkeypatch):
 m=load(P);report={'published_files':{}};p=tmp_path/'report.json';p.write_bytes(m.encoded(report));row=m.binding(p)
 legacy=types.SimpleNamespace(verify_publication=lambda *a:row);monkeypatch.setattr(m,'module',lambda *a:legacy)
 packet={'role':'teacher','legacy_watcher':{},'publication_output':str(tmp_path/'published')}
 with pytest.raises(ValueError):m.verify_publication(packet,{}, {}, {})


@pytest.mark.parametrize('late_UTC',[False,True])
def test_zero_exit_after_elapsed_or_UTC_deadline_is_not_success(tmp_path,monkeypatch,late_UTC):
 m=load(P);verified=[];now=datetime.now(timezone.utc);deadline=now+timedelta(days=1)
 class Clock(datetime):
  calls=0
  @classmethod
  def now(cls,tz=None):
   cls.calls+=1
   return deadline+timedelta(seconds=1) if late_UTC and cls.calls>=3 else now
 child=types.SimpleNamespace(pid=99999999,returncode=0,poll=lambda:0)
 runtime=types.SimpleNamespace(command=lambda *args:['never_executed'],verify_receipt=lambda *a:verified.append(True))
 monkeypatch.setattr(m,'module',lambda *a:runtime);monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:child)
 monkeypatch.setattr(m,'stop_group',lambda *a:None)
 monkeypatch.setattr(m,'heartbeat',lambda *a:None);monkeypatch.setattr(m,'datetime',Clock)
 ticks=iter([100.,100.5 if late_UTC else 102.]);monkeypatch.setattr(m.time,'monotonic',lambda:next(ticks))
 packet={'deadline_utc':deadline.isoformat(),'runtime_bootstrap':{},'runtime_context':{},'publisher':{}}
 with pytest.raises(ValueError,match='Completed paper phase exceeded'):m.run_cpu(packet,[],'synthetic_zero_exit',1,tmp_path,{})
 assert not verified
