"""Pure metadata diagnostics; fixture closures cannot authorize real watchers."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT=Path(__file__).parents[1]
BASE=ROOT/'experiments/rloo_one_shot_20261016'
REC=BASE/'cpu_preparation_reconciliation_v3'
SOURCE=BASE/'watch_reconciled_paper_updates_v4c.py'
OLD=BASE/'watch_reconciled_paper_updates_v4b.py'


def load(path=SOURCE):
    raw=path.read_bytes();m=types.ModuleType('synthetic_paper_upgrade')
    m.__file__=str(path);m.__executed_source_sha256__=hashlib.sha256(raw).hexdigest()
    exec(compile(raw,str(path),'exec',dont_inherit=True),m.__dict__)
    return m


def closure_fixture(tmp_path):
    m=load();buffers={};original=m.read_bound
    def put(path,value):
        raw=m.encoded(value);row={'path':str(path),'sha256':m.sha(raw),'size_bytes':len(raw)}
        buffers[str(path)]=raw;return row
    def read(row):
        if row['path'] not in buffers:return original(row)
        raw=buffers[row['path']]
        m.need(m.sha(raw)==row['sha256'] and len(raw)==row['size_bytes'],'Diagnostic byte binding changed')
        return raw
    m.read_bound=read;rows=[]
    for role in m.PAPER_PREDECESSORS:
        stem=role.removesuffix('_watcher')
        packet_path=REC/'handoff_registration_v4b'/f'{stem}_packet.json'
        packet=m.binding(packet_path);prior=json.loads(original(packet))
        owner_path=Path(prior['output'])/'owner.json';owner_row=m.binding(owner_path)
        owner=json.loads(original(owner_row));state={'owner':owner,'status':'failed_requires_reconciliation','error':'InterruptedError: diagnostic'}
        state_row=put(owner_path.with_name('status.json'),state)
        before=m.seal({'recorded_utc':'2026-10-17T00:10:00+00:00','owner':owner,'source':owner['source'],
                       'status':{'owner':owner,'status':'wait_for_complete_reconciled_producer'}},'receipt_sha256')
        before_row=put(tmp_path/f'{role}_before.json',before)
        proc={k:owner[k] for k in ('pid','uid','hostname','start_ticks')}
        proc.update(closed=True,final_kernel_state='missing',observed_utc='2026-10-17T00:10:01+00:00')
        rows.append({'role':role,'packet':packet,'owner':owner_row,'source':owner['source'],
                     'status':state_row,'before_closure':before_row,'process':proc})
    value={'protocol':'one-shot-paper-waiting-runtime-upgrade-closure-v4c','recorded_utc':'2026-10-17T00:10:02+00:00',
           'no_new_GPU_calls':True,'no_fresh_confirmation_calls':True,'no_new_paper_writes':True,
           'unrelated_processes_signalled':[],'rows':rows}
    validator=m.binding(BASE/'run_confirmation_campaign_reconciled_v4b.py')
    def adopt(value):
        row=put(tmp_path/'diagnostic_closure.json',m.seal(value,'receipt_sha256'))
        m.PAPER_PREDECESSOR_CLOSURE_SHA=row['sha256']
        return {'paper_predecessor_closure':row,'handoff_validator':validator}
    return m,value,adopt,put,read


def test_positive_distinct_two_watcher_closure_accepts_interrupted_waiter_status(tmp_path):
    m,v,adopt,_,_=closure_fixture(tmp_path)
    out=m.validate_paper_predecessor_closure(adopt(v))
    assert len(out['rows'])==2
    assert [r['process']['pid'] for r in out['rows']]==[3292718,3294161]


def test_actual_root_two_watcher_closure_is_distinct_from_unchanged_producer_closure():
    m=load();p={'paper_predecessor_closure':m.binding(REC/'paper_runtime_upgrade_v4c/closure.json'),
                'handoff_validator':m.binding(BASE/'run_confirmation_campaign_reconciled_v4b.py')}
    value=m.validate_paper_predecessor_closure(p)
    assert value['training_and_confirmation_controllers_left_running']==[3287518,3292717]
    assert p['paper_predecessor_closure']['sha256']!='295e6428ff09dc0452869771b772d8e57171848601954c628c8c3f2dbdd94e43'


@pytest.mark.parametrize('change',[
    lambda v:v.update(no_new_GPU_calls=False),
    lambda v:v.update(no_new_paper_writes=False),
    lambda v:v.update(no_fresh_confirmation_calls=False),
    lambda v:v.update(unrelated_processes_signalled=[123]),
    lambda v:v.update(rows=v['rows'][:1]),
    lambda v:v['rows'].reverse(),
    lambda v:v['rows'][0]['process'].update(pid=True),
    lambda v:v['rows'][0]['process'].update(start_ticks='wrong'),
    lambda v:v['rows'][0]['process'].update(closed=False),
    lambda v:v['rows'][0]['process'].update(final_kernel_state='S'),
    lambda v:v['rows'][0]['process'].update(observed_utc='2026-10-16T00:00:00+00:00'),
    lambda v:v['rows'][0]['process'].update(observed_utc='2026-10-17T01:00:00+00:00'),
    lambda v:v.update(synthetic_fixture=True),
])
def test_resealed_diagnostic_closure_does_not_waive_identity_zero_actions_or_chronology(tmp_path,change):
    m,v,adopt,_,_=closure_fixture(tmp_path);change(v)
    with pytest.raises(ValueError):m.validate_paper_predecessor_closure(adopt(v))


@pytest.mark.parametrize('key',['analysis','publication','confirmation_analysis','active_child_pid'])
def test_earlier_paper_action_cannot_be_adopted_as_idle_waiting(tmp_path,key):
    m,v,adopt,put,read=closure_fixture(tmp_path);row=v['rows'][0]
    state=json.loads(read(row['status']));state[key]={'unexpected':'paper action'}
    row['status']=put(row['status']['path'],state)
    with pytest.raises(ValueError):m.validate_paper_predecessor_closure(adopt(v))


def validation_fixture(tmp_path,monkeypatch):
    m=load();oldp=json.loads((REC/'handoff_registration_v4b/teacher_paper_packet.json').read_bytes())
    pp=json.loads(m.read_bound(oldp['producer_packet']));calls=[]
    runtime_path=tmp_path/'diagnostic_runtime.py';runtime_path.write_text('# Explicitly synthetic CPU transport fixture\n')
    runtime=m.binding(runtime_path);m.EXPECTED_RUNTIME_BOOTSTRAP_SHA=runtime['sha256']
    context={'bootstrap':runtime,'adapter':oldp['adapter'],'master_packet':pp['master_packet'],
             'project_manifest':{'sha256':'f48c01a35cbfddca21facced4d57493396b9a40f2a1f146115b890f4362d843a'},
             'extra_manifest':{'sha256':'dde13422cf0e977ec0083781e57bd1d9dbe40a582e1338440daabea571bd7d4f'}}
    context_path=tmp_path/'diagnostic_context.json';context_path.write_bytes(m.encoded(context));context_row=m.binding(context_path)
    upgrade_result={'status':'verified_explicit_A462_to_v3c_context_upgrade'}
    def upgrade(new,old):
        calls.append(('upgrade',new,old));return upgrade_result
    transport=types.SimpleNamespace(load_context=lambda row:(context,{}),verify_context_upgrade=upgrade)
    validator=types.SimpleNamespace(validate_reconciliation=lambda view:calls.append(('producer',view)))
    real_module=m.module
    def fixture_module(row,name):
        if row['sha256']==runtime['sha256']:return transport
        if row['sha256']==m.HANDOFF_VALIDATOR_SHA:return validator
        return real_module(row,name)
    monkeypatch.setattr(m,'module',fixture_module)
    oldrows=[]
    for stem in ('teacher','confirmation'):
        p=REC/'handoff_registration_v4b'/f'{stem}_paper_packet.json'
        q=json.loads(p.read_bytes());oldrows.append({'packet':m.binding(p),'owner':m.binding(Path(q['output'])/'owner.json')})
    closure={'rows':oldrows}
    monkeypatch.setattr(m,'validate_paper_predecessor_closure',lambda p:closure)
    p=m.prepare_packet(role='teacher',source=m.binding(SOURCE),producer_packet=oldp['producer_packet'],producer_owner=oldp['producer_owner'],
        prior_writers=oldp['prior_writers']+oldrows,publisher=oldp['publisher'],renderer=oldp['renderer'],legacy_watcher=oldp['legacy_watcher'],
        runtime_bootstrap=runtime,runtime_context=context_row,producer_runtime_bootstrap=pp['runtime_bootstrap'],
        producer_runtime_context=pp['runtime_context'],adapter=oldp['adapter'],handoff_validator=oldp['handoff_validator'],
        output=tmp_path/'new_watch',publication_output=tmp_path/'new_publication',operational_predecessor_closure=oldp['operational_predecessor_closure'],
        paper_predecessor_closure={'diagnostic_only':True},producer_registration=oldp['producer_registration'])
    return m,p,pp,calls,upgrade_result


def test_actual_producer_bindings_remain_A462_while_paper_context_upgrades(tmp_path,monkeypatch):
    m,p,pp,calls,_=validation_fixture(tmp_path,monkeypatch);m.validate(p)
    upgrade=[c for c in calls if c[0]=='upgrade'];producer=[c for c in calls if c[0]=='producer']
    assert len(upgrade)==len(producer)==1
    assert upgrade[0][2]==pp['runtime_context']
    view=producer[0][1]
    assert view['runtime_bootstrap']==pp['runtime_bootstrap'] and view['runtime_context']==pp['runtime_context']
    assert view['runtime_bootstrap']!=p['runtime_bootstrap']
    assert view['operational_predecessor_closure']==p['operational_predecessor_closure']


@pytest.mark.parametrize('key',['producer_runtime_bootstrap','producer_runtime_context'])
def test_paper_runtime_cannot_relabel_actual_producer_source_or_context(tmp_path,monkeypatch,key):
    m,p,pp,calls,_=validation_fixture(tmp_path,monkeypatch)
    p[key]=p[key.removeprefix('producer_')];p=m.seal(p)
    with pytest.raises(ValueError):m.validate(p)
    assert not [c for c in calls if c[0]=='producer']


def test_upgrade_requires_its_specific_verified_status(tmp_path,monkeypatch):
    m,p,pp,calls,result=validation_fixture(tmp_path,monkeypatch);result['status']='self_reported_true'
    with pytest.raises(ValueError):m.validate(p)
    assert not [c for c in calls if c[0]=='producer']


def test_confirmation_requires_the_new_teacher_writer_not_its_closed_predecessor(tmp_path,monkeypatch):
    m,teacher,_,calls,_=validation_fixture(tmp_path,monkeypatch)
    old=json.loads((REC/'handoff_registration_v4b/confirmation_paper_packet.json').read_bytes())
    pp=json.loads(m.read_bound(old['producer_packet']))
    p={**teacher,**{k:old[k] for k in ('role','producer_packet','producer_owner','deadline_utc','publisher','renderer','legacy_watcher','artifact_verifier')}}
    p['producer_runtime_bootstrap']=pp['runtime_bootstrap'];p['producer_runtime_context']=pp['runtime_context']
    p.pop('producer_registration',None);p=m.seal(p)
    with pytest.raises(ValueError,match='wait for the reconciled teacher paper writer'):m.validate(p)
    tp=tmp_path/'new_teacher_packet.json';tp.write_bytes(m.encoded(teacher));teacher_binding=m.binding(tp)
    owner=tmp_path/'new_teacher_owner.json';owner.write_bytes(m.encoded({'source':teacher['source'],'packet_binding':teacher_binding,
        'hostname':'synthetic','pid':12345,'start_ticks':'1'}))
    p['prior_writers'].append({'packet':teacher_binding,'owner':m.binding(owner)});p=m.seal(p)
    m.validate(p)
    # Source/context relabelling of that new writer is denied before snapshot.
    teacher['runtime_context']=teacher['producer_runtime_context'];teacher=m.seal(teacher);tp.write_bytes(m.encoded(teacher))
    teacher_binding=m.binding(tp);o=json.loads(owner.read_bytes());o['packet_binding']=teacher_binding;owner.write_bytes(m.encoded(o))
    p['prior_writers'][-1]={'packet':teacher_binding,'owner':m.binding(owner)};p=m.seal(p)
    with pytest.raises(ValueError,match='same producer and paper runtime'):m.validate(p)


def decision_fixture(tmp_path,monkeypatch,role='teacher'):
    m=load()
    def put(path,value):
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(m.encoded(value));return m.binding(p)
    owner={'pid':100,'uid':1,'hostname':'synthetic','start_ticks':'10'}
    owner_row=put('producer/control/owner.json',owner)
    master=put('master.json',{'output_root':str(tmp_path/'canonical')})
    pp={'master_packet':master};state={'owner':owner,'status':next(iter(m.SUCCESS[role]))}
    expected=tmp_path/('canonical/control/teacher_OP_analysis.json' if role=='teacher' else 'producer/control/confirmation_analysis.json')
    analysis=put(expected,{})
    state['analysis' if role=='teacher' else 'confirmation_analysis']=analysis;put('producer/control/status.json',state)
    oldowner={'pid':99,'uid':1,'hostname':'synthetic','start_ticks':'9'}
    oldrow={'owner':put('old/owner.json',oldowner),'packet':{'sha256':'diagnostic_old'}}
    put('old/status.json',{'owner':oldowner,'status':'failed_requires_reconciliation','error':'InterruptedError'})
    closure={'rows':[{'packet':oldrow['packet']}]}
    monkeypatch.setattr(m,'validate_paper_predecessor_closure',lambda p:closure)
    monkeypatch.setattr(m,'module',lambda *args:types.SimpleNamespace(validate_predecessor_closure=lambda p:None))
    p={'role':role,'producer_owner':owner_row,'prior_writers':[oldrow],'handoff_validator':{}}
    return m,p,owner,pp,analysis,put


def test_only_explicitly_closed_zero_action_writers_can_bypass_interrupted_status(tmp_path,monkeypatch):
    m,p,owner,pp,analysis,_=decision_fixture(tmp_path,monkeypatch)
    assert m.decision(p,owner,pp,closed=lambda o:True)==('snapshot_then_publish',analysis)
    with pytest.raises(ValueError,match='became live again'):
        m.decision(p,owner,pp,closed=lambda o:o==owner)
    monkeypatch.setattr(m,'validate_paper_predecessor_closure',lambda p:{'rows':[]})
    with pytest.raises(ValueError,match='independent reconciliation'):m.decision(p,owner,pp,closed=lambda o:True)


def test_confirmation_snapshot_waits_until_new_teacher_writer_is_terminal_and_closed(tmp_path,monkeypatch):
    m,p,owner,pp,analysis,put=decision_fixture(tmp_path,monkeypatch,'confirmation')
    teacher={'pid':98,'uid':1,'hostname':'synthetic','start_ticks':'8'}
    p['prior_writers'].append({'owner':put('new_teacher/owner.json',teacher),'packet':{'sha256':'new_teacher'}})
    put('new_teacher/status.json',{'owner':teacher,'status':'waiting_for_reconciled_producer'})
    assert m.decision(p,owner,pp,closed=lambda o:True)[0]=='wait_for_prior_paper_writer'
    put('new_teacher/status.json',{'owner':teacher,'status':'completed_admitted_development_paper_update'})
    assert m.decision(p,owner,pp,closed=lambda o:o!=teacher)[0]=='wait_for_prior_paper_writer'
    assert m.decision(p,owner,pp,closed=lambda o:True)==('snapshot_then_publish',analysis)


def test_whole_module_changes_are_only_paper_admission_and_closed_writer_order():
    a=ast.parse(OLD.read_bytes());b=ast.parse(SOURCE.read_bytes())
    af={n.name:ast.dump(n,include_attributes=False) for n in a.body if isinstance(n,ast.FunctionDef)}
    bf={n.name:ast.dump(n,include_attributes=False) for n in b.body if isinstance(n,ast.FunctionDef)}
    assert bf.keys()-af.keys()=={'validate_paper_predecessor_closure'}
    assert {k for k in af if af[k]!=bf[k]}=={'validate','decision','prepare_packet'}
    # Includes timed CPU transport, publication verification, transaction owner,
    # all scientific source pins and all retained deadlines.
    m=load();old=load(OLD)
    assert m.PINS==old.PINS and m.DEADLINES==old.DEADLINES and m.SUCCESS==old.SUCCESS and m.STOP==old.STOP
    allowed={'EXPECTED_RUNTIME_BOOTSTRAP_SHA','EXPECTED_PRODUCER_BOOTSTRAP_SHA','PAPER_PREDECESSOR_CLOSURE_SHA',
             'MIGRATION_PROTOCOL','PAPER_PREDECESSORS','PAPER_PREDECESSOR_SOURCE'}
    def remaining(tree):
        nodes=[]
        for node in tree.body:
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):continue
            if isinstance(node,ast.FunctionDef) and node.name in {'validate','decision','prepare_packet','validate_paper_predecessor_closure'}:continue
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in allowed for t in node.targets):continue
            nodes.append(node)
        return ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)
    assert remaining(a)==remaining(b)


def test_default_source_is_inert(tmp_path):
    r=subprocess.run([sys.executable,'-I','-B',str(SOURCE)],cwd=tmp_path,capture_output=True,text=True,check=True)
    assert json.loads(r.stdout)['paper_writes']==0 and not list(tmp_path.iterdir())
