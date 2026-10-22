"""Pure arithmetic/admission fixtures: these fabricated jobs never existed."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/rloo_sft_anchor_20261013'
spec=importlib.util.spec_from_file_location('pure_recovery_accounting',ROOT/'src/clasp/baselines/sft_anchor_recovery_accounting.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def binding(raw,path=None):
    item={'sha256':m.sha(raw),'size_bytes':len(raw),'content_base64':base64.b64encode(raw).decode()}
    if path is not None:item['native_path']=path
    return item


def encoded(value):return (json.dumps(value,sort_keys=True)+'\n').encode()


def accounting(job,state='RUNNING',elapsed=60,gpus=4):
    rows=[]
    for suffix in ('','.batch','.extern'):
        status='COMPLETED' if state=='FAILED' and suffix=='.extern' else state
        exit_code='1:0' if status=='FAILED' else '0:0'
        rows.append(f'{job}{suffix}|{status}|{elapsed}|{exit_code}|cpu=80,mem=300000M,gres/gpu={gpus}\n')
    return {'argv':['sacct','--noheader','--parsable2','--jobs',job,'--format=JobIDRaw,State%40,ElapsedRaw,ExitCode,AllocTRES'],
            'returncode':0,'stderr':'','stdout_binding':binding(''.join(rows).encode())}


def fixture():
    authority=(EXP/'collected_status_v2_20261014T0911/status/campaign_authority.json').read_bytes()
    slot='op.rloo_cardinal_beta005';job='999'
    attempt={'state':'submitted','study':'rloo-sft-anchor-20261013-phase2','slot':slot,'job_id':job,
             'manifest_sha256':m.LAUNCH_SHA,'requested_gpu_hours':8}
    return {'schema_version':1,'protocol':m.PROTOCOL,'current_job_id':job,
        'reservation_authority':binding(authority,m.NAMESPACE+'/control_v2/campaign_authority.json'),
        'reservations':m.reservations(),'training_proof_binding':{'training_complete_sha256':'a'*64,
        'pilot_contract_sha256':'b'*64,'source_sha256':'c'*64,'task':'op','arm':'rloo_cardinal_beta005','training_parent_job_id':job},
        'jobs':[{'job_id':'1030929','reservation_id':'prior-v1/1030929','accounting':accounting('1030929','FAILED',459)},
                {'job_id':job,'reservation_id':'phase2/'+slot,'accounting':accounting(job),
                 'attempt_binding':binding(encoded(attempt),m.NAMESPACE+'/control_v2/attempts/'+slot+'.json')}]}


def validate(tmp_path,value,**changes):
    path=tmp_path/'certificate.json';path.write_bytes(encoded(value))
    args={'expected_sha256':m.sha(path.read_bytes()),'training_complete_sha256':'a'*64,'pilot_contract_sha256':'b'*64,
          'source_sha256':'c'*64,'task':'op','arm':'rloo_cardinal_beta005','current_job_id':'999','authorized_requested_gpu_hours':92}
    args.update(changes)
    return m.validate_accounting_certificate(path,**args)


def change_stdout(row,change):
    raw=base64.b64decode(row['accounting']['stdout_binding']['content_base64']).decode()
    row['accounting']['stdout_binding']=binding(change(raw).encode())


def test_running_same_reserved_allocation_has_verified_lower_bound_without_new_job_authority(tmp_path):
    value=validate(tmp_path,fixture())
    assert value['accounting_certificate_verified'] and value['total_requested_gpu_hours']==92
    assert value['submitted_requested_gpu_hours']==16
    assert value['total_allocated_gpu_seconds']==4*(459+60)
    assert value['allocation_accounting_complete'] is False
    assert value['allocated_cost_is_lower_bound_running_snapshot'] and not value['new_job_authorized']
    assert not value['native_queries_or_mutations']


@pytest.mark.parametrize('key,value',[('expected_sha256','0'*64),('training_complete_sha256','0'*64),
    ('pilot_contract_sha256','0'*64),('source_sha256','0'*64),('task','cvrp'),('arm','foarl'),('current_job_id','1000'),
    ('authorized_requested_gpu_hours',100),('authorized_requested_gpu_hours',91)])
def test_caller_certificate_and_all_training_identities_are_pinned(tmp_path,key,value):
    with pytest.raises(ValueError):validate(tmp_path,fixture(),**{key:value})


def test_unsubmitted_statistical_slots_cannot_be_dropped_to_hide_recovery_cost(tmp_path):
    value=fixture();value['reservations']=value['reservations'][:2]
    with pytest.raises(ValueError,match='all84h'):validate(tmp_path,value)


def test_new_recovery_job_exceeds_preserved_full_authority(tmp_path):
    value=fixture();value['reservations'].append({'reservation_id':'rescue/1000','gpus':4,'requested_wall_seconds':7200})
    with pytest.raises(ValueError,match='Additional recovery'):validate(tmp_path,value)


def test_failed_training_parent_cannot_be_replaced_inside_the_same_reservation(tmp_path):
    value=fixture();parent=copy.deepcopy(value['jobs'][1]);parent['job_id']='888';parent['accounting']=accounting('888','FAILED',200)
    attempt=json.loads(base64.b64decode(parent['attempt_binding']['content_base64']));attempt['job_id']='888'
    parent['attempt_binding']=binding(encoded(attempt),parent['attempt_binding']['native_path'])
    value['training_proof_binding']['training_parent_job_id']='888';value['jobs'].append(parent)
    with pytest.raises(ValueError,match='two jobs'):validate(tmp_path,value)


@pytest.mark.parametrize('mutate',[lambda v:v['jobs'][1]['accounting'].update(returncode=1),
    lambda v:v['jobs'][1]['accounting'].update(argv=['sacct','--jobs','unrelated']),
    lambda v:v['jobs'][1]['accounting'].update(stderr='unresolved'),
    lambda v:v['jobs'][1]['accounting']['stdout_binding'].update(sha256='0'*64),
    lambda v:change_stdout(v['jobs'][1],lambda s:s.replace('gres/gpu=4','gres/gpu=0')),
    lambda v:change_stdout(v['jobs'][1],lambda s:s.replace('gres/gpu=4','gres/gpu=2')),
    lambda v:change_stdout(v['jobs'][1],lambda s:s.replace('RUNNING','PENDING')),
    lambda v:change_stdout(v['jobs'][1],lambda s:s.replace('RUNNING','COMPLETED')),
    lambda v:change_stdout(v['jobs'][1],lambda s:s.replace('RUNNING','RUNNING+')),
    lambda v:change_stdout(v['jobs'][1],lambda s:s.replace('999.batch','foreign.batch')),
    lambda v:change_stdout(v['jobs'][1],lambda s:s+s.splitlines(True)[0]),
    lambda v:change_stdout(v['jobs'][1],lambda s:s.splitlines(True)[0]),
    lambda v:change_stdout(v['jobs'][1],lambda s:s.replace('|60|','|-60|')),
    lambda v:change_stdout(v['jobs'][1],lambda s:s.replace('cpu=80','cpu=80,cpu=80')),
    lambda v:change_stdout(v['jobs'][1],lambda s:s.replace('0:0','1:0')),
    lambda v:change_stdout(v['jobs'][0],lambda s:s.replace('FAILED','COMPLETED')),
    lambda v:change_stdout(v['jobs'][0],lambda s:s.replace('|459|','|458|')),
    lambda v:v['jobs'].pop(0),lambda v:v['jobs'].pop(1),
    lambda v:v['jobs'].append(copy.deepcopy(v['jobs'][1])),
    lambda v:v.update(synthetic_fixture=True),
    lambda v:v['jobs'][1].update(synthetic_offline_fixture=True)])
def test_raw_scheduler_scope_cost_state_and_closed_lineage_guards(tmp_path,mutate):
    value=fixture();mutate(value)
    with pytest.raises(ValueError):validate(tmp_path,value)


@pytest.mark.parametrize('key,value',[('job_id','1000'),('slot','op.foarl'),('manifest_sha256','0'*64),
    ('state','intent_recorded'),('requested_gpu_hours',0)])
def test_original_attempt_ledger_binding_is_required(tmp_path,key,value):
    cert=fixture();row=cert['jobs'][1];attempt=json.loads(base64.b64decode(row['attempt_binding']['content_base64']))
    attempt[key]=value;row['attempt_binding']=binding(encoded(attempt),row['attempt_binding']['native_path'])
    with pytest.raises(ValueError):validate(tmp_path,cert)


def test_authority_bytes_cannot_be_reset_with_a_new_deadline(tmp_path):
    cert=fixture();authority=json.loads(base64.b64decode(cert['reservation_authority']['content_base64']))
    authority['absolute_deadline_unix']+=10000
    authority['semantic_sha256']=m.canonical({k:v for k,v in authority.items() if k!='semantic_sha256'})
    cert['reservation_authority']=binding(encoded(authority),cert['reservation_authority']['native_path'])
    with pytest.raises(ValueError,match='replace/reset'):validate(tmp_path,cert)


def test_plain_unverified_arithmetic_rows_are_not_a_certificate(tmp_path):
    with pytest.raises(ValueError):validate(tmp_path,[{'id':'dummy','gpus':0,'requested_wall_seconds':0,'elapsed_seconds':0}])
