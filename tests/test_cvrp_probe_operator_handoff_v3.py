"""Pure held-script activation tests; no sbatch/release/remote call."""
import copy
import hashlib
import json
from pathlib import Path
import shlex
import types

import pytest

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/rloo_sft_anchor_20261013'
E=EXP/'cvrp_probe_execution_v3'
OLD_E=EXP/'cvrp_probe_execution_v2'
FAILURE=EXP/'validation/native_monitor/cvrp_first_probe_preassignment_failure_20261014T1742_bound_v1.json'
NO_JOB=EXP/'validation/native_monitor/cvrp_first_probe_no_assigned_job_20261014T1743_bound_v1.json'


def load(path,name):
    raw=path.read_bytes();m=types.ModuleType(name);m.__file__=str(path)
    exec(compile(raw,str(path),'exec',dont_inherit=True),m.__dict__);return m

m=load(E/'prepare_operator_handoff_v3.py','handoff_tests')


def seal(value,key):value[key]=m.sha(m.encoded({k:v for k,v in value.items() if k!=key}));return value

@pytest.fixture
def context():
    p=m.helper();spec_raw=(E/'spec_proposal_v3.json').read_bytes()
    authority=json.loads((E/'authority_proposal_v3.json').read_bytes());authority['synthetic_fixture']=True;seal(authority,'authority_sha256')
    authority_raw=m.raw_json(authority)
    return m.admit_pre_submission(spec_raw=spec_raw,spec_binding=m.binding(p.CONTROL+'/spec.json',spec_raw),
        authority_raw=authority_raw,authority_binding=m.binding(p.CONTROL+'/authority.json',authority_raw),
        source_manifest_raw=(E/'source_candidate/source_manifest.json').read_bytes(),
        pilot_authority_raw=(EXP/'collected_native_first_complete_v1/status/campaign_authority.json').read_bytes(),
        prior_failure_raw=(EXP/'validation/native_monitor/prior_failed_1030929_reverification_v2_bound.json').read_bytes(),rejected_failure_raw=FAILURE.read_bytes(),no_assigned_job_raw=NO_JOB.read_bytes(),synthetic_fixture=True)


def submit(context):
    script=m.build_slurm_script(context)
    intent=m.build_submission_intent(context,script,created_utc='2026-10-14T16:00:00+00:00');iraw=m.raw_json(intent)
    ib=m.binding(context['spec']['control_root']+'/submission_intent.json',iraw)
    action={'protocol':m.PREPARATION+'-submission-action','intent_sha256':m.sha(iraw),'argv':script['submission_argv'],
        'returncode':0,'stdout':'9990001\n','stderr':'','completed_utc':'2026-10-14T16:01:00+00:00','synthetic_fixture':True}
    araw=m.raw_json(action);ab=m.binding(context['spec']['control_root']+'/submission_action.json',araw)
    attempt=m.build_submitted_attempt(context,script,intent_raw=iraw,intent_binding=ib,action_raw=araw,action_binding=ab)
    return script,iraw,ib,araw,ab,attempt


def test_exact_held_Slurm_script_carries_caps_and_unchanged_submissionargv(context):
    script=m.build_slurm_script(context);body=script['script_raw'].decode();control=context['spec']['control_root']
    expected=['--hold','--account=vas@a100','--partition=gpu_p5','--constraint=a100','--qos=qos_gpu_a100-t3','--nodes=1',
        '--ntasks=1','--cpus-per-task=40','--time=00:30:00','--gres=gpu:4','--hint=nomultithread','--no-requeue']
    for flag in expected: assert body.count('#SBATCH '+flag+'\n')==1
    assert script['submission_argv']==['sbatch','--parsable',control+'/job.slurm']
    assert 'srun' not in body and 'scontrol' not in body and 'sbatch ' not in body
    assert not script['automatic_submission_or_release']
    command=shlex.split(body[body.index('exec ')+5:])
    assert '--no-python' in command and command.count('-I')==2
    assert '--nproc_per_node=4' in command and command[-2]=='-c'
    assert command[-1]==script['outer_program']


def test_script_has_no_late_job_or_reg_cert_hash_and_cannot_cycle(context):
    script=m.build_slurm_script(context)
    assert '9990001' not in script['script_raw'].decode()
    assert context['authority_binding']['sha256'] in script['script_raw'].decode()
    assert context['spec_binding']['sha256'] in script['script_raw'].decode()
    assert 'activation-'+"'" in script['outer_program']
    # Bootstrap is read/hash/compiled before its operative definitions, with
    # worker source loading delegated to the already-reviewed external bootstrap.
    assert script['outer_program'].index("bootstrap_raw=read(") < script['outer_program'].index("exec(compile(bootstrap_raw")
    assert script['outer_program'].index("exec(compile(bootstrap_raw") < script['outer_program'].index('module.main(argv)')
    compile(script['outer_program'],'<reviewed-outer>','exec')


def test_source_only_proposal_cannot_emit_production_handoff(context):
    p=m.helper()
    with pytest.raises(ValueError):m.admit_pre_submission(spec_raw=(E/'spec_proposal_v3.json').read_bytes(),spec_binding=context['spec_binding'],
        authority_raw=(E/'authority_proposal_v3.json').read_bytes(),authority_binding=m.binding(p.CONTROL+'/authority.json',(E/'authority_proposal_v3.json').read_bytes()),
        source_manifest_raw=(E/'source_candidate/source_manifest.json').read_bytes(),
        pilot_authority_raw=(EXP/'collected_native_first_complete_v1/status/campaign_authority.json').read_bytes(),
        prior_failure_raw=(EXP/'validation/native_monitor/prior_failed_1030929_reverification_v2_bound.json').read_bytes(),rejected_failure_raw=FAILURE.read_bytes(),no_assigned_job_raw=NO_JOB.read_bytes())


def test_successful_unique_held_submission_is_derived_from_raw_action(context):
    script,iraw,ib,araw,ab,attempt=submit(context)
    assert attempt['job_id']=='9990001' and attempt['attempt_number']==1
    assert attempt['step_strategy']=='torchrun4_in_batch' and attempt['requested_wall_seconds']==1800
    assert attempt['submission_intent_sha256']==m.sha(iraw) and attempt['submission_action_sha256']==m.sha(araw)
    assert attempt['synthetic_fixture'] and not attempt['replacement_or_retry']


@pytest.mark.parametrize('fault',['timeout','retcode','partial_stdout','cluster_suffix','stderr','wrong_argv','job_early','wrong_intent','wrong_path'])
def test_ambiguous_failed_or_unrelated_submission_cannot_activate_or_retry(context,fault):
    script,iraw,ib,araw,ab,attempt=submit(context);action=json.loads(araw)
    if fault=='timeout':action['returncode']=None;action['invocation_error']='TimeoutExpired'
    elif fault=='retcode':action['returncode']=1
    elif fault=='partial_stdout':action['stdout']='maybe 9990001'
    elif fault=='cluster_suffix':action['stdout']='9990001;cluster'
    elif fault=='stderr':action['stderr']='warning'
    elif fault=='wrong_argv':action['argv'].append('--hold')
    elif fault=='job_early':action['completed_utc']='2026-10-14T15:00:00+00:00'
    elif fault=='wrong_intent':action['intent_sha256']='0'*64
    else:ab['path']='/copied/action.json'
    changed=m.raw_json(action);ab.update(m.binding(ab['path'],changed))
    with pytest.raises(ValueError):m.build_submitted_attempt(context,script,intent_raw=iraw,intent_binding=ib,action_raw=changed,action_binding=ab)


def env_record(context):
    return {'protocol':m.PROTOCOL+'-readonly-native-environment','independent_read_only_collection':True,'model_or_policy_calls':0,'synthetic_fixture':True,
        'environment':{'executable':context['spec']['python'],'python_version':'SYNTHETIC NATIVE TEMPLATE','libraries':{k:'SYNTHETIC' for k in ['torch','transformers','trl','peft','accelerate']},
            'torch_cuda':'SYNTHETIC','device_name':'NVIDIA A100-SXM4-80GB','requested_base_dtype':'torch.bfloat16','torch_num_threads':1}}


def test_actual_saved_tiny_initializer_full_file_inventory_builder(context):
    old=load(E/'overrides/experiments/rloo_sft_anchor_20261013/run_cvrp_timing_probe_v1.py','inventory_handoff_worker')
    fixture=EXP/'validation/v2_peft_cpu_all_arms_v1/cvrp';contract_raw=(fixture/'contract.json').read_bytes()
    record={'protocol':m.PROTOCOL+'-readonly-initializer-inventory','independent_read_only_collection':True,'model_or_policy_calls':0,
        'base':context['spec']['base'],'sft':context['spec']['sft'],'base_files':old.file_inventory(fixture/'fixture_base'),'sft_files':old.file_inventory(fixture/'fixture_sft'),
        'complete_base_file_set':True,'complete_sft_file_set':True,'links_or_special_files':[],'synthetic_fixture':True}
    r=m.raw_json(record);result=m.initializer_from_collection(r,m.binding('/synthetic/readonly-initializer.json',r),contract_raw=contract_raw,synthetic_fixture=True)
    assert result['initializer_identity_sha256']==json.loads(contract_raw)['initializer']['sha256']
    assert result['synthetic_fixture'] and len(result['base_files'])==6


@pytest.mark.parametrize('fault',['incomplete','links','adapter','duplicate','traversal','bool_size','policy_call'])
def test_full_inventory_collection_rejects_bad_custody(context,fault):
    old=load(E/'overrides/experiments/rloo_sft_anchor_20261013/run_cvrp_timing_probe_v1.py','inventory_handoff_faults')
    f=EXP/'validation/v2_peft_cpu_all_arms_v1/cvrp';cr=(f/'contract.json').read_bytes()
    rec={'protocol':m.PROTOCOL+'-readonly-initializer-inventory','independent_read_only_collection':True,'model_or_policy_calls':0,
        'base':context['spec']['base'],'sft':context['spec']['sft'],'base_files':old.file_inventory(f/'fixture_base'),'sft_files':old.file_inventory(f/'fixture_sft'),
        'complete_base_file_set':True,'complete_sft_file_set':True,'links_or_special_files':[],'synthetic_fixture':True}
    if fault=='incomplete':rec['complete_base_file_set']=False
    elif fault=='links':rec['links_or_special_files']=['linked']
    elif fault=='adapter':next(x for x in rec['sft_files'] if x['path']=='adapter_model.safetensors')['sha256']='0'*64
    elif fault=='duplicate':rec['base_files'].append(rec['base_files'][0])
    elif fault=='traversal':rec['base_files'][0]['path']='../escape'
    elif fault=='bool_size':rec['base_files'][0]['size_bytes']=True
    else:rec['model_or_policy_calls']=1
    raw=m.raw_json(rec)
    with pytest.raises(ValueError):m.initializer_from_collection(raw,m.binding('/synthetic/collect.json',raw),contract_raw=cr,synthetic_fixture=True)


def test_environment_builder_records_original_template_without_fake_dynamicUUID(context):
    rec=env_record(context);raw=m.raw_json(rec);result=m.environment_from_collection(raw,m.binding('/synthetic/env.json',raw))
    assert result['torch_num_threads']==1 and result['requested_base_dtype']=='torch.bfloat16'
    assert 'device_uuid' not in result and 'pid' not in result


@pytest.mark.parametrize('fault',['dtype','thread','bool_thread','interpreter','device','libraries','policy_calls'])
def test_environment_collection_wrong_runtime_denied(context,fault):
    rec=env_record(context)
    if fault=='dtype':rec['environment']['requested_base_dtype']='torch.float32'
    elif fault=='thread':rec['environment']['torch_num_threads']=4
    elif fault=='bool_thread':rec['environment']['torch_num_threads']=True
    elif fault=='interpreter':rec['environment']['executable']='/unregistered/python'
    elif fault=='device':rec['environment']['device_name']='RTX2080'
    elif fault=='libraries':rec['environment']['libraries'].pop('trl')
    else:rec['model_or_policy_calls']=1
    raw=m.raw_json(rec)
    with pytest.raises(ValueError):m.environment_from_collection(raw,m.binding('/synthetic/env.json',raw))


def registration_fixture(context):
    script,iraw,ib,araw,ab,attempt=submit(context);p=m.helper();control=p.CONTROL
    buffers={'script_binding':script['script_raw'],'submission_intent_binding':iraw,'submission_action_binding':araw,'attempt_binding':m.raw_json(attempt),
        'authority_binding':m.raw_json(context['authority']),
        'pilot_authority_binding':(EXP/'collected_native_first_complete_v1/status/campaign_authority.json').read_bytes(),
        'prior_failure_binding':(EXP/'validation/native_monitor/prior_failed_1030929_reverification_v2_bound.json').read_bytes(),
        'rejected_failure_binding':FAILURE.read_bytes(),'no_assigned_job_binding':NO_JOB.read_bytes(),
        'attempt_directory_inventory_binding':m.raw_json({'protocol':m.OPERATOR+'-attempt-directory-inventory','control_root':control,
            'independent_read_only_collection':True,'attempt_files':['attempt.json'],'intent_files':['submission_intent.json'],'action_files':['submission_action.json'],'unresolved_intents':[],'synthetic_fixture':True}),
        'initializer_inventory_binding':m.raw_json({'protocol':m.OPERATOR+'-initializer-inventory','synthetic_fixture':True}),
        'native_environment_collection_binding':m.raw_json(env_record(context))}
    names={'script_binding':'job.slurm','submission_intent_binding':'submission_intent.json','submission_action_binding':'submission_action.json',
        'attempt_binding':'attempt.json','authority_binding':'authority.json','pilot_authority_binding':'pilot_authority.json','prior_failure_binding':'prior_failure.json',
        'rejected_failure_binding':'rejected_failure.json','no_assigned_job_binding':'no_assigned_job.json',
        'attempt_directory_inventory_binding':'attempt_directory_inventory.json','initializer_inventory_binding':'initializer_inventory.json','native_environment_collection_binding':'native_environment.json'}
    bindings={k:m.binding(control+'/'+names[k],v) for k,v in buffers.items()}
    return script,buffers,bindings,attempt


def test_after_unique_heldID_registration_and_activation_do_not_rewrite_script(context):
    script,buffers,bindings,attempt=registration_fixture(context)
    reg=m.registration_from_buffers(context,script,buffers=buffers,bindings=bindings,registered_utc='2026-10-14T16:02:00+00:00');rr=m.raw_json(reg)
    expected=m.signed_expectations(context,rr,attempt_raw=buffers['attempt_binding'])
    cert={'payload':{'protocol':'sft-anchor-independent-registration-promotion-v1','schema_version':1,**expected},'signer_key_id':'UNVERIFIED FIXTURE ONLY','signature_base64':'UNVERIFIED'}
    # This constructor deliberately does not claim signature verification. The
    # exact native proxy remains mandatory before any model, even if allpins fit.
    activation=m.activation_from_certificate(context,script,registration_raw=rr,certificate_raw=m.raw_json(cert),attempt_raw=buffers['attempt_binding'])
    assert activation['path'].endswith('activation-9990001.json')
    assert activation['descriptor']['registration_binding']['path'].endswith('operator_registration-9990001.json')
    assert not activation['descriptor']['activation_authorizes_models_without_independent_signature']
    assert not activation['descriptor']['automatic_release_or_retry']
    assert script['script_binding']==m.binding(context['spec']['control_root']+'/job.slurm',script['script_raw'])


@pytest.mark.parametrize('fault',['extra_attempt','unresolved_intent','changed_script','changed_authority','chronology','changed_history'])
def test_registration_cannot_hide_retry_source_authority_cost_or_before_submit_registration(context,fault):
    script,buffers,bindings,attempt=registration_fixture(context);time='2026-10-14T16:02:00+00:00'
    if fault in ('extra_attempt','unresolved_intent'):
        row=json.loads(buffers['attempt_directory_inventory_binding'])
        if fault=='extra_attempt':row['attempt_files'].append('attempt2.json')
        else:row['unresolved_intents']=['ambiguous']
        buffers['attempt_directory_inventory_binding']=m.raw_json(row)
    elif fault=='changed_script':buffers['script_binding']+=b'\n'
    elif fault=='changed_authority':buffers['authority_binding']+=b' '
    elif fault=='chronology':time='2026-10-14T15:00:00+00:00'
    else:buffers['prior_failure_binding']+=b' '
    for key in buffers:bindings[key]=m.binding(bindings[key]['path'],buffers[key])
    with pytest.raises(ValueError):m.registration_from_buffers(context,script,buffers=buffers,bindings=bindings,registered_utc=time)


def test_activation_projection_rejects_signed_numeric_alias_without_consuming_signature(context):
    script,buffers,bindings,attempt=registration_fixture(context)
    reg=m.registration_from_buffers(context,script,buffers=buffers,bindings=bindings,registered_utc='2026-10-14T16:02:00+00:00');rr=m.raw_json(reg)
    expected=m.signed_expectations(context,rr,attempt_raw=buffers['attempt_binding']);expected['study_admission']=0
    with pytest.raises(ValueError):m.activation_from_certificate(context,script,registration_raw=rr,certificate_raw=m.raw_json({'payload':expected}),attempt_raw=buffers['attempt_binding'])


def test_slurm_login_modules_and_environment_match_original_v2_launcher(context):
    body=m.build_slurm_script(context)['script_raw'].decode()
    original=(EXP/'control_v2/job_scripts/cvrp.rloo_cardinal_beta005.slurm').read_text()
    assert body.startswith('#!/bin/bash -l\n')
    for line in ('umask 027','module purge','module load arch/a100','module load miniforge/24.9.0',
        'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline',
        'export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 NCCL_DEBUG=WARN'):
        assert line+'\n' in original and line+'\n' in body
    assert 'MKL_NUM_THREADS' not in body


@pytest.mark.parametrize('fault',['source','argv','gpus','wall','float_execution','bool_gpus'])
def test_raw_intent_caps_and_source_are_checked_before_activation(context,fault):
    script,iraw,ib,araw,ab,attempt=submit(context);intent=json.loads(iraw)
    if fault=='source':intent['source_manifest_sha256']='0'*64
    elif fault=='argv':intent['submission_argv'].append('--hold')
    elif fault=='gpus':intent['requested_gpus']=8
    elif fault=='wall':intent['requested_wall_seconds']=7200
    elif fault=='float_execution':intent['execution_revision']=3.0
    else:intent['requested_gpus']=True
    changed=m.raw_json(intent);ib=m.binding(ib['path'],changed)
    action=json.loads(araw);action['intent_sha256']=m.sha(changed);araw=m.raw_json(action);ab=m.binding(ab['path'],araw)
    with pytest.raises(ValueError):m.build_submitted_attempt(context,script,intent_raw=changed,intent_binding=ib,action_raw=araw,action_binding=ab)


def test_real_registered_authority_must_predate_exact_intent_even_if_script_fits(context):
    actual=copy.deepcopy(context);actual['synthetic_fixture']=False;actual['authority']['registered_utc']='2026-10-14T16:01:00+00:00'
    with pytest.raises(ValueError):m.build_submission_intent(actual,m.build_slurm_script(actual),created_utc='2026-10-14T16:00:00+00:00')


def test_original_native_contract_is_byte_pinned_for_initializer_collection(context):
    record={'protocol':m.PROTOCOL+'-readonly-initializer-inventory','independent_read_only_collection':True,'model_or_policy_calls':0,
        'base':context['spec']['base'],'sft':context['spec']['sft'],'base_files':[],'sft_files':[],
        'complete_base_file_set':True,'complete_sft_file_set':True,'links_or_special_files':[]}
    raw=m.raw_json(record);cr=(EXP/'validation/v2_peft_cpu_all_arms_v1/cvrp/contract.json').read_bytes()
    with pytest.raises(ValueError,match='original CVRP contract'):m.initializer_from_collection(raw,m.binding('/synthetic/collect.json',raw),contract_raw=cr)


def test_unsigned_expectation_cannot_swap_actual_attempt_after_registration(context):
    script,buffers,bindings,attempt=registration_fixture(context)
    reg=m.registration_from_buffers(context,script,buffers=buffers,bindings=bindings,registered_utc='2026-10-14T16:02:00+00:00');rr=m.raw_json(reg)
    attempt['job_id']='9990002'
    with pytest.raises(ValueError):m.signed_expectations(context,rr,attempt_raw=m.raw_json(attempt))


@pytest.mark.parametrize('fault',['missing_job','wrong_world','wrong_step'])
def test_actual_outer_stdlib_preflight_rejects_wrong_batch_before_any_source_read(context,monkeypatch,fault):
    monkeypatch.setenv('SLURM_JOB_ID','9990001');monkeypatch.setenv('WORLD_SIZE','4');monkeypatch.setenv('SLURM_STEP_ID','batch')
    if fault=='missing_job':monkeypatch.delenv('SLURM_JOB_ID')
    elif fault=='wrong_world':monkeypatch.setenv('WORLD_SIZE','1')
    else:monkeypatch.setenv('SLURM_STEP_ID','0')
    code=m.build_slurm_script(context)['outer_program']
    with pytest.raises(ValueError,match='assigned4-rank'):exec(compile(code,'<actual-outer-fixture>','exec'),{})


def outer_local_fixture(context,tmp_path,monkeypatch):
    # Only transport paths are mapped to tmp; the operative130 source and all
    # source/spec bytes stay exact. The native worker must reject this copied
    # context even if an untrusted late descriptor fits every superficial pin.
    import shutil
    p=m.helper();source=tmp_path/'source';control=tmp_path/'control';control.mkdir()
    shutil.copytree(E/'source_candidate',source)
    p.SOURCE=str(source);p.CONTROL=str(control)
    monkeypatch.setattr(m,'helper',lambda:p)
    ctx=copy.deepcopy(context);ctx['spec']['control_root']=str(control)
    for field,name in (('spec_binding','spec.json'),('authority_binding','authority.json')):
        ctx[field]['path']=str(control/name)
    (control/'spec.json').write_bytes((E/'spec_proposal_v3.json').read_bytes())
    (control/'authority.json').write_bytes(m.raw_json(ctx['authority']))
    script,buffers,bindings,attempt=registration_fixture(ctx)
    for field,raw in buffers.items():Path(bindings[field]['path']).write_bytes(raw)
    reg=m.registration_from_buffers(ctx,script,buffers=buffers,bindings=bindings,registered_utc='2026-10-14T16:02:00+00:00')
    # Empty-key material is deliberately invalid. No actual signer or allocation.
    rr=m.raw_json(reg);expected=m.signed_expectations(ctx,rr,attempt_raw=buffers['attempt_binding'])
    cert={'payload':{'protocol':'sft-anchor-independent-registration-promotion-v1','schema_version':1,**expected},'signer_key_id':'UNVERIFIED','signature_base64':'INVALID'}
    cr=m.raw_json(cert);act=m.activation_from_certificate(ctx,script,registration_raw=rr,certificate_raw=cr,attempt_raw=buffers['attempt_binding'])
    Path(act['registration_path']).write_bytes(rr);Path(act['certificate_path']).write_bytes(cr);Path(act['path']).write_bytes(act['raw'])
    monkeypatch.setenv('SLURM_JOB_ID','9990001');monkeypatch.setenv('WORLD_SIZE','4');monkeypatch.setenv('SLURM_STEP_ID','batch')
    return script,act,control


@pytest.mark.parametrize('fault',['wrong_job','changed_authority','bool_size','changed_reg','changed_script'])
def test_actual_outer_activation_tamper_denied_before_compiling_worker(context,tmp_path,monkeypatch,fault):
    script,act,control=outer_local_fixture(context,tmp_path,monkeypatch)
    desc=copy.deepcopy(act['descriptor'])
    if fault=='wrong_job':desc['job_id']='9990002'
    elif fault=='changed_authority':desc['authority_binding']['sha256']='0'*64
    elif fault=='bool_size':desc['registration_binding']['size_bytes']=True
    elif fault=='changed_reg':Path(act['registration_path']).write_bytes(Path(act['registration_path']).read_bytes()+b' ')
    else:(control/'job.slurm').write_bytes(script['script_raw']+b'\n')
    Path(act['path']).write_bytes(m.raw_json(desc))
    with pytest.raises(ValueError):exec(compile(script['outer_program'],'<actual-outer-tamper>','exec'),{})


def test_correct_unsigned_activation_still_cannot_authorize_copied_native_worker(context,tmp_path,monkeypatch):
    script,act,control=outer_local_fixture(context,tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='Closed timing registration|Fixture and native registration scopes|Copied local source'):
        exec(compile(script['outer_program'],'<actual-outer-valid-pins-untrusted>','exec'),{})


@pytest.mark.parametrize('field',['initializer_inventory_binding','native_environment_collection_binding','attempt_directory_inventory_binding','attempt_binding'])
def test_nested_fixture_markers_cannot_be_released_as_native_context(context,field):
    script,buffers,bindings,attempt=registration_fixture(context)
    ctx=copy.deepcopy(context);ctx['synthetic_fixture']=False
    # A mislabeled fixture must fail before the registration builder could hand
    # its malformed evidence to the signed native worker after an allocation.
    for key in buffers:
        try:record=json.loads(buffers[key])
        except (ValueError,UnicodeError):continue
        record.pop('synthetic_fixture',None);buffers[key]=m.raw_json(record);bindings[key]=m.binding(bindings[key]['path'],buffers[key])
    record=json.loads(buffers[field]);record['nested']=[{'synthetic_fixture':True}]
    buffers[field]=m.raw_json(record);bindings[field]=m.binding(bindings[field]['path'],buffers[field])
    with pytest.raises(ValueError,match='Nested synthetic'):m.registration_from_buffers(ctx,script,buffers=buffers,bindings=bindings,registered_utc='2026-10-14T16:02:00+00:00')


def test_actual_owned_native_read_only_initializer_and_full_runtime_collections_are_admitted_locally():
    # These are pre-existing parent-owned read-only transports, no source, job,
    # policy, registry or authority is mutated or created by this check.
    init_path=OLD_E/'native_initializer_collection_20261014T171319_v2.json'
    env_path=OLD_E/'native_environment_collection_20261014T171803_full_version_modules_v2.json'
    init_transport=json.loads(init_path.read_bytes());env_transport=json.loads(env_path.read_bytes())
    assert init_transport['returncode']==env_transport['returncode']==0
    assert init_transport['stderr']==env_transport['stderr']==''
    init_raw=init_transport['stdout'].encode();env_raw=env_transport['stdout'].encode()
    contract_raw=(E/'source_candidate/experiments/rloo_sft_anchor_20261013/prepared_v1/contracts/cvrp_seed43.json').read_bytes()
    init=m.initializer_from_collection(init_raw,m.binding('/read-only-derived/initializer.json',init_raw),contract_raw=contract_raw)
    env=m.environment_from_collection(env_raw,m.binding('/read-only-derived/runtime.json',env_raw))
    assert len(init['base_files'])+len(init['sft_files'])==33
    assert not init['synthetic_fixture'] and env['torch_num_threads']==1
    assert env['device_name']=='NVIDIA A100-SXM4-80GB'


def test_authority_registration_transition_is_pure_bytebound96_and_predates_intent(context):
    p=m.helper();proposal_raw=(E/'authority_proposal_v3.json').read_bytes();spec_raw=(E/'spec_proposal_v3.json').read_bytes()
    candidate=m.authority_registration_candidate(proposal_raw=proposal_raw,proposal_binding=m.binding('/local/proposal.json',proposal_raw),
        spec_raw=spec_raw,spec_binding=context['spec_binding'],pilot_authority_raw=(EXP/'collected_native_first_complete_v1/status/campaign_authority.json').read_bytes(),
        prior_failure_raw=(EXP/'validation/native_monitor/prior_failed_1030929_reverification_v2_bound.json').read_bytes(),rejected_failure_raw=FAILURE.read_bytes(),no_assigned_job_raw=NO_JOB.read_bytes(),registered_utc='2026-10-14T18:00:00+00:00')
    assert not candidate['registration_applied_by_builder'] and not candidate['native_actions_performed']
    actual=m.admit_pre_submission(spec_raw=spec_raw,spec_binding=context['spec_binding'],authority_raw=candidate['authority_raw'],authority_binding=candidate['authority_binding'],
        source_manifest_raw=(E/'source_candidate/source_manifest.json').read_bytes(),pilot_authority_raw=(EXP/'collected_native_first_complete_v1/status/campaign_authority.json').read_bytes(),
        prior_failure_raw=(EXP/'validation/native_monitor/prior_failed_1030929_reverification_v2_bound.json').read_bytes(),rejected_failure_raw=FAILURE.read_bytes(),no_assigned_job_raw=NO_JOB.read_bytes())
    assert actual['history']['combined_reserved_requested_gpu_hours']==96
    assert m.build_submission_intent(actual,m.build_slurm_script(actual),created_utc='2026-10-14T18:01:00+00:00')['submit_held']
