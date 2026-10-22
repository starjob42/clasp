"""Corrected implicit-memory timing scope; no native submissions or policy reads."""
import ast
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/rloo_sft_anchor_20261013'
E=EXP/'cvrp_probe_execution_v3'
OLD=EXP/'cvrp_probe_execution_v2'
REL=Path('experiments/rloo_sft_anchor_20261013')
O=E/'overrides'
WORKER=O/REL/'run_cvrp_timing_probe_v1.py'
FIXTURE=EXP/'validation/v2_peft_cpu_all_arms_v1/cvrp'


def load(path,name):
    raw=path.read_bytes();m=types.ModuleType(name);m.__file__=str(path)
    m.__executed_source_sha256__=hashlib.sha256(raw).hexdigest()
    exec(compile(raw,str(path),'exec',dont_inherit=True),m.__dict__);return m

m=load(WORKER,'execution3_tests')
h=load(E/'prepare_operator_handoff_v3.py','execution3_handoff_tests')


def raw(value):return json.dumps(value,sort_keys=True,indent=2).encode()+b'\n'
def seal(value,key):value[key]=m.sha(m.encoded({k:v for k,v in value.items() if k!=key}));return value

@pytest.fixture(scope='module')
def packet():
    s=E/'source_candidate';p=E/'spec_proposal_v3.json'
    return s,p,m.load_core(s,p,m.sha(p.read_bytes()))


def history_args():
    return dict(pilot_authority_raw=(EXP/'collected_native_first_complete_v1/status/campaign_authority.json').read_bytes(),
        prior_failure_raw=(EXP/'validation/native_monitor/prior_failed_1030929_reverification_v2_bound.json').read_bytes(),
        rejected_failure_raw=(EXP/m.FAILURE_PATH.split('rloo_sft_anchor_20261013/',1)[1]).read_bytes(),
        no_assigned_job_raw=(EXP/m.NO_JOB_PATH.split('rloo_sft_anchor_20261013/',1)[1]).read_bytes())


def function_asts(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_bytes()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


def test_complete_numerical_kernel_AST_is_execution2():
    old=function_asts(OLD/'overrides/src/clasp/entrypoints/profile_cvrp_timing_probe_v1.py')
    new=function_asts(O/'src/clasp/entrypoints/profile_cvrp_timing_probe_v1.py')
    assert set(old)==set(new)
    for name in old:
        if name!='admit_native_operator':assert old[name]==new[name],name


def test_worker_training_loader_sampler_and_runtime_AST_are_unchanged():
    old=function_asts(OLD/'overrides'/REL/'run_cvrp_timing_probe_v1.py');new=function_asts(WORKER)
    for name in ('native_run','cpu_fixture_run','original_native_tokenizer','execution_environment','file_inventory','validate_initializer',
        'validate_process','BoundProjectLoader','BoundProjectFinder','install_project'):
        assert old[name]==new[name],name


def test_frozen_execution2_and_original121_are_retained(packet):
    core=packet[2];old=json.loads(core['buffers']['original_source2_manifest.json'])
    assert len(old['files'])==121 and len(core['buffers'])==133
    assert all(m.sha(core['buffers'][r['path']])==r['sha256'] for r in old['files'])
    assert m.sha((OLD/'source_candidate/source_manifest.json').read_bytes())=='db5e3e48a4900b0d2b448d3855bb79beb4646a01ffcba7a1ceb894645894045d'
    assert m.sha(core['buffers'][m.HISTORY_PATH])==m.HISTORY_SHA
    assert m.sha(core['buffers'][m.PROMOTION_PATH])==m.PROMOTION_SHA
    assert m.sha(core['buffers'][m.PROXY_PATH])==m.PROXY_SHA
    assert core['spec']['execution_revision']==3 and core['spec']['resources']['memory_mb']==300000


def recorded_job(packet):
    proof=json.loads(json.loads((EXP/'validation/native_monitor/qos_transition_noaction_inventory_v1.json').read_bytes())['stdout'])
    # This independently bound native readback records exact original implicit
    # default geometry, rather than fabricating an allocation from CLI flags.
    def strings(v):
        if isinstance(v,dict):
            for x in v.values():yield from strings(x)
        elif isinstance(v,list):
            for x in v:yield from strings(x)
        elif isinstance(v,str):yield v
    source=next(x for x in strings(proof) if 'MinMemoryCPU=7500M' in x and 'AllocTRES=' in x)
    f=m.parse_job(source.strip());f.update(JobId='9990001',QOS='qos_gpu_a100-t3',TimeLimit='00:30:00',Command=packet[2]['spec']['control_root']+'/job.slurm',SubmitLine='sbatch --parsable '+packet[2]['spec']['control_root']+'/job.slurm')
    return f


def test_actual_recorded_original_default_memory_geometry_is_accepted(packet):
    f=recorded_job(packet)
    assert 'MinMemoryNode' not in f and f['MinMemoryCPU']=='7500M'
    proof=m.validate_job(f,spec=packet[2]['spec'],job_id='9990001')
    assert proof['allocated_tres']['cpu']=='80' and proof['allocated_tres']['billing']=='5'

@pytest.mark.parametrize('fault',['node','percpu','requested80','allocated40','memory','billing','gpu'])
def test_wrong_native_default_or_allocation_is_rejected(packet,fault):
    f=recorded_job(packet)
    if fault=='node':f['MinMemoryNode']='300000M'
    elif fault=='percpu':f['MinMemoryCPU']='7000M'
    elif fault=='requested80':f['ReqTRES']=f['ReqTRES'].replace('cpu=40','cpu=80')
    elif fault=='allocated40':f['AllocTRES']=f['AllocTRES'].replace('cpu=80','cpu=40')
    elif fault=='memory':f['ReqTRES']=f['ReqTRES'].replace('300000M','600000M')
    elif fault=='billing':f['ReqTRES']=f['ReqTRES'].replace('billing=4','billing=5')
    else:f['AllocTRES']=f['AllocTRES'].replace('gres/gpu=4','gres/gpu=3')
    with pytest.raises(ValueError):m.validate_job(f,spec=packet[2]['spec'],job_id='9990001')


def test_cumulative96_retains_rejected2_and_zero_actual_allocation(packet):
    p=packet[2]['preparer'];a=json.loads((E/'authority_proposal_v3.json').read_bytes())
    r=p.validate_authority(a,**history_args())
    assert r['combined_reserved_requested_gpu_hours']==96 and r['original_reservations_retained']==13
    assert r['rejected_probe_history']['actual_rejected_gpu_hours']==0
    assert not r['native_submission_authorized_by_this_validator']

@pytest.mark.parametrize('fault',['drop_old','reset94','bool_old','old_as_retry','failure','nojob','wrong_namespace','float_version'])
def test_rejected_history_and_new_scope_cannot_be_reset(packet,fault):
    p=packet[2]['preparer'];a=json.loads((E/'authority_proposal_v3.json').read_bytes());args=history_args()
    if fault=='drop_old':a.pop('prior_rejected_probe_reservation')
    elif fault=='reset94':a['combined_reserved_requested_gpu_hours']=a['declared_requested_gpu_hour_ceiling']=94
    elif fault=='bool_old':a['prior_rejected_probe_reservation']['automatic_retry']=0
    elif fault=='old_as_retry':a['probe_reservation']['reservation_id']='diagnostic/cvrp-first3-v1'
    elif fault=='failure':args['rejected_failure_raw']+=b' '
    elif fault=='nojob':args['no_assigned_job_raw']+=b' '
    elif fault=='wrong_namespace':a['probe_reservation']['reservation_id']='unregistered'
    else:a['execution_revision']=3.0
    seal(a,'authority_sha256')
    with pytest.raises(ValueError):p.validate_authority(a,**args)


def test_all_existing_rescue_reservations_produce116_without_borrowing(packet):
    p=packet[2]['preparer'];extras=[{'reservation_id':'evaluation-only/'+slot,'gpus':1,'wall_seconds':7200} for slot in p.ORDER if not slot.endswith('.sft')]
    rescue={'original_pilot_authority_sha256':p.PILOT_AUTHORITY_SHA,'additional_reservations':extras,'combined_reserved_requested_gpu_hours':112}
    seal(rescue,'authority_sha256');rr=raw(rescue)
    a=p.make_authority_proposal(packet[2]['spec']['spec_sha256'],rescue_reservations=extras,rescue_authority_sha256=m.sha(rr))
    assert p.validate_authority(a,rescue_authority_raw=rr,**history_args())['combined_reserved_requested_gpu_hours']==116


def test_script_no_memory_flags_and_inherited_mem_options_explicitly_forbidden():
    from test_cvrp_probe_operator_handoff_v3 import context
    ctx=context.__wrapped__();script=h.build_slurm_script(ctx);text=script['script_raw'].decode()
    assert not any(flag in text for flag in ('--mem=','--mem-per-cpu','--mem-per-gpu'))
    assert script['submission_argv']==['sbatch','--parsable',ctx['spec']['control_root']+'/job.slurm']
    assert script['forbidden_inherited_submission_environment']==['SBATCH_MEM_PER_CPU','SBATCH_MEM_PER_NODE','SBATCH_MEM_PER_GPU']


def fixture_registration(packet,tmp_path):
    from test_cvrp_probe_operator_handoff_v3 import context,registration_fixture
    ctx=context.__wrapped__();script,buffers,bindings,attempt=registration_fixture(ctx)
    reg=h.registration_from_buffers(ctx,script,buffers=buffers,bindings=bindings,registered_utc='2026-10-14T18:00:00+00:00')
    for number,(key,body) in enumerate(buffers.items()):
        path=tmp_path/f'{number}.dat';path.write_bytes(body)
        if key in reg:reg[key]=m.binding(path,body)
    ctx['authority_binding']=reg['authority_binding']
    return ctx,reg,buffers


def sign(core,ctx,reg,buffers,changes=None):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
    verifier=load(EXP/'verify_registration_promotion_v1.py','execution3_fixture_verifier')
    proxy=load(O/REL/'subprocess_crypto_verifier_v2.py','execution3_proxy')
    rr=raw(reg);expected=h.signed_expectations(ctx,rr,attempt_raw=buffers['attempt_binding'])
    payload={'protocol':verifier.PROTOCOL,'schema_version':1,'synthetic_fixture':True,'performance_admission':False,**expected};payload.update(changes or {})
    private=Ed25519PrivateKey.generate();cert={'payload':payload,'signer_key_id':'fixture-only','signature_base64':base64.b64encode(private.sign(verifier.encoded(payload))).decode()}
    keys={'fixture-only':{'scope':'fixture_only','purpose':'independent_prepolicy_registration','crypto_environment':verifier.crypto_environment(),
        'public_key_base64':base64.b64encode(private.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)).decode()}}
    return rr,raw(cert),keys,proxy.interpreter_identity('/usr/bin/python3')


def test_actual_fixture_signature_binds_rejection_receipts_and_stays_diagnostic(packet,tmp_path):
    ctx,reg,buffers=fixture_registration(packet,tmp_path);rr,cr,keys,interpreter=sign(packet[2],ctx,reg,buffers)
    admission=m.admit_registration(packet[2],rr,cr,fixture_public_keys=keys,fixture_interpreter_binding=interpreter)
    assert admission['history']['combined_reserved_requested_gpu_hours']==96 and not admission['native_operator_admitted']
    assert admission['signed_expectations']['rejected_failure_sha256']==packet[2]['preparer'].REJECTED_FAILURE_SHA
    with pytest.raises(ValueError):m.native_run(packet[2],admission)
    with pytest.raises(ValueError):m.admit_registration(packet[2],rr,cr)

@pytest.mark.parametrize('field,value',[('rejected_failure_sha256','0'*64),('no_assigned_job_sha256','0'*64),('execution_revision',3.0),('study_admission',0)])
def test_valid_signed_but_wrong_prior_custody_or_typed_scope_cannot_load_model(packet,tmp_path,field,value):
    ctx,reg,buffers=fixture_registration(packet,tmp_path);rr,cr,keys,interpreter=sign(packet[2],ctx,reg,buffers,{field:value})
    with pytest.raises(ValueError):m.admit_registration(packet[2],rr,cr,fixture_public_keys=keys,fixture_interpreter_binding=interpreter)


def test_actual_external_bootstrap_four_CPU_rank_originalH100_prefix_equivalence(packet,tmp_path):
    from test_cvrp_timing_probe_kernel_v1 import contract,same_numerics
    cp=tmp_path/'contract.json';cp.write_bytes(raw(contract()));source,sp,core=packet
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':str(ROOT/'src')}
    prefix=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4']
    control=tmp_path/'control';profile=tmp_path/'profile'
    commands=[prefix+[str(ROOT/'tests/test_cvrp_timing_probe_kernel_v1.py'),'--worker','--mode','control','--contract',str(cp),'--output',str(control)],
        prefix+[str(source/m.BOOTSTRAP_PATH),'--expected-bootstrap-sha256',m.BOOTSTRAP_SHA,'--expected-worker-sha256',core['worker_sha256'],
            '--source-root',str(source),'--spec',str(sp),'--expected-spec-sha256',m.sha(core['spec_raw']),'--cpu-fixture','--fixture-contract',str(cp),
            '--fixture-base',str(FIXTURE/'fixture_base'),'--fixture-sft',str(FIXTURE/'fixture_sft'),'--fixture-output',str(profile)]]
    for number,command in enumerate(commands):
        r=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        (tmp_path/f'execution{number}.log').write_text(r.stdout)
        assert r.returncode==0,r.stdout[-15000:]
    a,b=[json.loads((p/'timing_probe.json').read_bytes()) for p in (control,profile)];same_numerics(a,b)
    assert b['execution_revision']==3 and b['original_horizon']==100 and b['committed_prefix_updates']==3 and len(b['distributed']['rank_closure'])==4
    assert b['responses']==108 and b['evaluation_responses']==0 and not b['model_checkpoint_written']
    retained=E/'validation_v3/actual_cpu_ddp4';retained.mkdir(parents=True,exist_ok=True)
    for p in (control,profile):
        (retained/(p.name+'.json')).write_bytes((p/'timing_probe.json').read_bytes())
    for n in range(2):(retained/f'execution{n}.log').write_bytes((tmp_path/f'execution{n}.log').read_bytes())

@pytest.mark.parametrize('raw_metadata', ['JobId=1 stray', 'SubmitLine=sbatch --parsable /x extra', 'JobId=1 JobId=2', 'SubmitLine=sbatch --parsable /x WorkDir=/x stray'])
def test_scheduler_whitespace_is_bounded_to_exact_submitline(raw_metadata):
    with pytest.raises(ValueError):m.parse_job(raw_metadata)

def test_other_submission_argv_cannot_impersonate_same_allocation(packet):
    fields=recorded_job(packet);fields['SubmitLine']='sbatch --mem=300000M '+packet[2]['spec']['control_root']+'/job.slurm'
    with pytest.raises(ValueError):m.validate_job(fields,spec=packet[2]['spec'],job_id='9990001')

@pytest.mark.parametrize('state,elapsed,complete',[('COMPLETED',100,True),('FAILED',77,True),('TIMEOUT',1800,True),('RUNNING',33,False)])
def test_new_accounting_carries96_ledger_and_actual_failed_cost(packet,state,elapsed,complete):
    from test_cvrp_timing_probe_authority_v1 import accounting_fixture
    kwargs,attempt,query,_=accounting_fixture(state,elapsed)
    p=packet[2]['preparer'];a=load(O/REL/'verify_cvrp_timing_probe_accounting_v3.py','execution3_accounting')
    authority=p.make_authority_proposal('1'*64);authority['synthetic_fixture']=True;seal(authority,'authority_sha256');araw=raw(authority)
    attempt.update(reservation_id=a.RESERVATION,script_path=a.CONTROL+'/job.slurm',authority_sha256=m.sha(araw),
        submission_argv=['sbatch','--parsable',a.CONTROL+'/job.slurm'])
    kwargs.update(attempt_raw=raw(attempt),expected_attempt_sha256=m.sha(raw(attempt)),authority_raw=araw,
        expected_authority_sha256=m.sha(araw),expected_preparer_sha256=m.PREPARER_SHA,**history_args())
    report=a.validate_accounting(**kwargs)
    assert report['combined_reserved_requested_gpu_hours']==96 and report['current_probe_allocated_gpu_seconds']==elapsed*4
    assert report['allocation_accounting_complete'] is complete and not report['study_admitted']
    assert report['cumulative_allocated_gpu_seconds'] is None and not report['native_allocation_accounting_certified']
    kwargs['no_assigned_job_raw']+=b' '
    with pytest.raises(ValueError):a.validate_accounting(**kwargs)

@pytest.mark.parametrize('time',['2026-10-14T16:00:00+00:00','2026-10-14T17:43:05+00:00'])
def test_actual_authority_cannot_predate_independent_no_assigned_job_custody(packet,time):
    p=packet[2]['preparer'];proposal=(E/'authority_proposal_v3.json').read_bytes();spec=packet[2]['spec_raw']
    with pytest.raises(ValueError,match='Fresh authority'):
        h.authority_registration_candidate(proposal_raw=proposal,proposal_binding=h.binding('/local/proposal.json',proposal),
            spec_raw=spec,spec_binding=h.binding(p.CONTROL+'/spec.json',spec),registered_utc=time,**history_args())
