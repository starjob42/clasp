"""Read-only independent QoS custody; all changed-job records are synthetic."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/rloo_sft_anchor_20261013'
spec=importlib.util.spec_from_file_location('qos_custody_tests',EXP/'verify_qos_transition_v1.py')
verify=importlib.util.module_from_spec(spec);spec.loader.exec_module(verify)
PREFLIGHT=json.loads(json.loads((EXP/'validation/native_monitor/qos_transition_preflight_v1.json').read_bytes())['stdout'])
BEFORE=PREFLIGHT['queries'][0]
COST=PREFLIGHT['queries'][1]


def changed_query(**fields):
    original=verify.parse_job_metadata(BEFORE['stdout'])
    original['QOS']=verify.NEW_QOS
    original.update(fields)
    return {**BEFORE,'stdout':' '.join(k+'='+v for k,v in original.items())+'\n'}


def test_actual_readonly_pending_preflight_and_equal_cost_factors():
    actual=verify.validate_fixed_pending_job(verify.validate_query(BEFORE),verify.OLD_QOS)
    assert actual['JobId']=='1052260' and actual['ReqTRES']=='cpu=40,mem=300000M,node=1,billing=4,gres/gpu=4'
    assert verify.validate_usage_factor(COST)['cost_factor_equal'] is True
    assert PREFLIGHT['native_mutations'] is False


def test_synthetic_qos_only_transition_and_independent_readback():
    after=changed_query(Priority='999999',Reason='Resources')
    checked=verify.compare_pending_metadata(BEFORE,after)
    assert checked['changed_fields']==['Priority','QOS','Reason']
    assert verify.validate_readback(after,changed_query(Priority='999998',LastSchedEval='2030-01-01T00:00:00')) is True
    assert verify.KNOWN_TRANSITIONS=={}


@pytest.mark.parametrize('key,value',[
 ('JobId','1052261'),('JobName','replacement'),('UserId','other(1)'),('Account','vas@h100'),('Partition','gpu_p6'),
 ('Features','h100'),('JobState','RUNNING'),('Requeue','1'),('Restarts','1'),('RunTime','00:00:01'),('TimeLimit','20:00:00'),
 ('NumCPUs','80'),('CPUs/Task','80'),('NumNodes','2'),('NumTasks','4'),('MinMemoryCPU','15000M'),('AllocTRES','gres/gpu=4'),
 ('TresPerNode','gres/gpu:8'),('TresPerTask','cpu=80'),('ReqTRES','cpu=40,mem=300000M,node=1,billing=8,gres/gpu=8'),
 ('SubmitTime','2030-01-01T00:00:00'),('EligibleTime','2030-01-01T00:00:00'),('AccrueTime','2030-01-01T00:00:00'),
 ('Command','/tmp/new-script'),('SubmitLine','sbatch /tmp/new-script'),('Comment','other'),('StdOut','/tmp/new-log'),('StdIn','/tmp/new-input')])
def test_every_job_resource_identity_or_submission_change_rejected(key,value):
    with pytest.raises(ValueError):verify.compare_pending_metadata(BEFORE,changed_query(**{key:value}))


@pytest.mark.parametrize('case',['duplicate','truncated','field_added','unknown_changed','expanded_allowlist','wrong_query'])
def test_complete_metadata_inventory_and_exact_query_required(case):
    after=changed_query()
    if case=='duplicate':after['stdout']+=' JobId=1052260'
    if case=='truncated':after['stdout']='JobId=1052260 QOS='+verify.NEW_QOS
    if case=='field_added':after['stdout']+=' NewField=unexpected'
    if case=='unknown_changed':after=changed_query(Nice='10')
    if case=='wrong_query':after['argv']=['scontrol','update','JobId=1052260','QOS='+verify.NEW_QOS]
    if case=='expanded_allowlist':
        with pytest.raises(ValueError,match='allowlist'):verify.compare_pending_metadata(BEFORE,after,allow_derived={'Nice'})
    else:
        with pytest.raises(ValueError):verify.compare_pending_metadata(BEFORE,after)


@pytest.mark.parametrize('stdout',[
 'qos_gpu_a100-dev|1.000000|\nqos_gpu_a100-t3|2.000000|\n',
 'qos_gpu_a100-dev|NaN|\nqos_gpu_a100-t3|NaN|\n',
 'qos_gpu_a100-dev|1.000000|\nqos_gpu_a100-t3|1.000000|NoDecay\n',
 'qos_gpu_a100-dev|1.000000|\n',
 'qos_gpu_a100-dev|1.000000|\nqos_gpu_a100-dev|1.000000|\n'])
def test_cost_factor_flags_missing_and_duplicates_rejected(stdout):
    with pytest.raises(ValueError):verify.validate_usage_factor({**COST,'stdout':stdout})


def test_owner_must_be_original_same_host_not_bool_nonempty():
    probe={'reader_hostname':verify.OWNER['hostname'],'same_host':True,'original_proc_identity_matches':True,'owner':verify.OWNER}
    assert verify.validate_owner_probe(probe)
    for field,new in [('reader_hostname','jean-zay2'),('same_host',False),('original_proc_identity_matches',False)]:
        with pytest.raises(ValueError):verify.validate_owner_probe({**probe,field:new})
    owner={**verify.OWNER,'pid':2196977}
    with pytest.raises(ValueError):verify.validate_owner_probe({**probe,'owner':owner})


def bindings():
    names={'launch_manifest':verify.LAUNCH_SHA256,'source_manifest':verify.SOURCE_SHA256,'admission_patch_manifest':verify.PATCH_SHA256,
           'attempt':verify.ATTEMPT_SHA256,'script':verify.SCRIPT_SHA256,'authority':'a'*64,'activation':'b'*64,'admission_transition':'c'*64}
    result={name:{'path':'/synthetic/'+name,'sha256':sha,'size_bytes':100} for name,sha in names.items()}
    result['script']['path']=verify.SCRIPT_PATH;result['attempt']['path']=verify.CONTROL+'/attempts/'+verify.SLOT+'.json'
    return result


@pytest.mark.parametrize('name',['launch_manifest','source_manifest','admission_patch_manifest','attempt','script','authority','activation','admission_transition'])
def test_every_producer_attempt_script_and_controller_binding_preserved(name):
    before=bindings();assert verify.validate_bound_producer(before,copy.deepcopy(before))
    after=copy.deepcopy(before);after[name]['sha256']='0'*64
    with pytest.raises(ValueError):verify.validate_bound_producer(before,after)


def test_independent_readback_cannot_change_cost_or_job_resources():
    after=changed_query()
    with pytest.raises(ValueError):verify.validate_readback(after,changed_query(NumCPUs='80'))
    with pytest.raises(ValueError):verify.validate_readback(after,changed_query(QOS=verify.OLD_QOS))


def encode(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
def reseal(value,key):value[key]=verify.canonical({k:v for k,v in value.items() if k!=key});return value


def transition_packet():
    prepared=EXP/'qos_transition_v1_prepared'
    spec=json.loads((prepared/'spec.json').read_bytes());spec['synthetic_fixture']=True;reseal(spec,'spec_sha256')
    raw_spec=encode(spec)
    before=verify.parse_job_metadata(BEFORE['stdout'])
    intent={'protocol':'sft-anchor-pending-qos-transition-v1-preflight','native_mutations':False,
        'spec_file_binding':verify.identity(raw_spec),'spec_sha256':spec['spec_sha256'],'owner':spec['owner'],
        'before_query':BEFORE,'before':before,'association_query':{'argv':['sacctmgr','--noheader','--parsable2','show','assoc','where',
            'user=uag72kz','account=vas@a100','format=Account,Partition,QOS%300,DefaultQOS'],
            'returncode':0,'stderr':'','stdout':'vas@a100||qos_gpu_a100-dev,qos_gpu_a100-t3|qos_gpu_a100-t3\n'},
        'usage_query':COST,'source_files_verified':121}
    action={'argv':['scontrol','update','JobId=1052260','QOS='+verify.NEW_QOS],'returncode':0,'stdout':'','stderr':''}
    after_query=changed_query()
    after=verify.parse_job_metadata(after_query['stdout'])
    receipt={'protocol':'sft-anchor-pending-qos-transition-v1-receipt','synthetic_fixture':True,'job_id':verify.JOB_ID,
        'spec_file_binding':verify.identity(raw_spec),'spec_sha256':spec['spec_sha256'],
        'intent_binding':verify.identity(encode(intent)),'action_binding':verify.identity(encode(action)),
        'post_query':after_query,'after':after,'changed_scheduler_fields':['QOS'],'post_source_files_verified':121,
        'requested_gpu_hours':8,'combined_maximum_requested_gpu_hours':92,'status':'completed_qos_only_transition',
        **{key:False for key in ['statistical_source_modified','script_modified','attempt_modified','controller_restarted',
                                 'job_cancelled_retried_or_resubmitted','superiority_established']}}
    reseal(receipt,'receipt_sha256')
    collected=EXP/'collected_status_admission_v3_20261014T1015'
    native_bytes={
        'control_v2/launch_manifest.json':(EXP/'control_v2/launch_manifest.json').read_bytes(),
        'source_v2/source_manifest.json':(EXP/'source_v2/source_manifest.json').read_bytes(),
        'control_v2/campaign_authority.json':(collected/'status/campaign_authority.json').read_bytes(),
        'control_v2/attempts/'+verify.SLOT+'.json':(collected/'attempts'/f'{verify.SLOT}.json').read_bytes(),
        'control_v2/job_scripts/'+verify.SLOT+'.slurm':(EXP/'control_v2/job_scripts'/f'{verify.SLOT}.slurm').read_bytes(),
        'admission_v1/patch_manifest.json':(EXP/'admission_prepared_v1/patch_manifest.json').read_bytes(),
        'admission_v1_runtime/activations/2196976-9919949.json':(collected/'operational_patch/activations/2196976-9919949.json').read_bytes(),
        'admission_v1_runtime/transition_receipt.json':(collected/'operational_patch/transition_receipt.json').read_bytes()}
    manifest=json.loads(native_bytes['source_v2/source_manifest.json'])
    readback={'protocol':verify.PROTOCOL+'-readback','synthetic_fixture':True,'native_mutations':False,
        'transition_receipt_file_binding':verify.identity(encode(receipt)),
        'owner_probe':{'reader_hostname':verify.OWNER['hostname'],'same_host':True,'original_proc_identity_matches':True,'owner':verify.OWNER},
        'association_query':intent['association_query'],'usage_query':COST,'job_query':after_query,
        'native_bindings':{p:verify.identity(raw) for p,raw in native_bytes.items()},
        'source_proofs':[{'path':r['path'],'sha256':r['sha256'],'size_bytes':r['size_bytes'],
            'native_path':verify.NAMESPACE+'/source_v2/'+r['path']} for r in manifest['files']],
        'unbound_frozen_source_bytecode':[]}
    return {'spec':raw_spec,'program':(prepared/'program.py').read_bytes(),'intent':encode(intent),'action':encode(action),
            'receipt':encode(receipt),'readback':encode(readback),'native_bytes':native_bytes,
            'source_manifest':native_bytes['source_v2/source_manifest.json'],'baseline_allocation':(prepared/'prior_allocation_evidence.json').read_bytes()}


def test_complete_synthetic_packet_chain_cannot_publish():
    packet=transition_packet()
    result=verify.verify_transition_packet(packet,synthetic_fixture=True)
    assert result['status']=='verified_synthetic_diagnostic' and result['source_files_verified']==121
    assert result['publication_custody_admitted'] is False and result['native_actions_by_verifier'] is False
    with pytest.raises(ValueError,match='Unregistered'):verify.verify_transition_packet(packet)


@pytest.mark.parametrize('case',['broken_intent','broken_action','wrong_update','source_drop','source_hash','source_path','owner',
                                 'activation','program','attempt','missing_native','changed_script_flag','wrong_status','bytecode'])
def test_complete_custody_cannot_accept_mutated_or_partial_packet(case):
    packet=transition_packet()
    if case in ('broken_intent','broken_action'):packet[case.split('_')[1]]+=b' '
    elif case in ('activation','attempt','program'):
        if case=='program':packet['program']+=b'#changed'
        else:
            name=next(p for p in packet['native_bytes'] if ('activations/' in p if case=='activation' else '/attempts/' in p))
            packet['native_bytes'][name]+=b' '
    elif case=='missing_native':packet['native_bytes'].pop(next(iter(packet['native_bytes'])))
    elif case in ('source_drop','source_hash','source_path','owner','bytecode'):
        value=json.loads(packet['readback'])
        if case=='source_drop':value['source_proofs'].pop()
        if case=='source_hash':value['source_proofs'][0]['sha256']='0'*64
        if case=='source_path':value['source_proofs'][0]['native_path']='/tmp/different'
        if case=='owner':value['owner_probe']['owner']['pid']+=1
        if case=='bytecode':value['unbound_frozen_source_bytecode']=['unbound.pyc']
        packet['readback']=encode(value)
    elif case=='wrong_update':
        value=json.loads(packet['action']);value['argv'].append('TimeLimit=20:00:00');packet['action']=encode(value)
        receipt=json.loads(packet['receipt']);receipt['action_binding']=verify.identity(packet['action']);reseal(receipt,'receipt_sha256');packet['receipt']=encode(receipt)
    else:
        value=json.loads(packet['receipt'])
        if case=='changed_script_flag':value['script_modified']=True
        if case=='wrong_status':value['status']='invented_success'
        reseal(value,'receipt_sha256');packet['receipt']=encode(value)
    with pytest.raises(ValueError):verify.verify_transition_packet(packet,synthetic_fixture=True)


@pytest.mark.parametrize('returncode,error,flag',[(None,'TimeoutExpired: ambiguous',None),(1,None,None),(0,None,True)])
def test_ambiguous_failed_or_drifted_receipt_preserved_without_admission(returncode,error,flag):
    packet=transition_packet();action=json.loads(packet['action']);action['returncode']=returncode
    if error:action['invocation_error']=error;action['automatic_retry']=False
    packet['action']=encode(action)
    receipt=json.loads(packet['receipt']);receipt['status']='requires_investigation';receipt['action_binding']=verify.identity(packet['action'])
    receipt['statistical_source_modified']=flag;receipt['script_modified']=None;receipt['attempt_modified']=None
    if flag is True:receipt['readback_validation_error']='source drift detected'
    reseal(receipt,'receipt_sha256');packet['receipt']=encode(receipt);packet.pop('readback')
    result=verify.verify_transition_packet(packet,synthetic_fixture=True)
    assert result['publication_custody_admitted'] is False and result['effective_qos'] is None


@pytest.mark.parametrize('case',['valid','cpu','gpu','memory','billing','node','elapsed','request_alias','bad_request_alias'])
def test_running_state_is_bounded_by_original_native_allocation(case):
    before=verify.parse_job_metadata(BEFORE['stdout']);after=verify.parse_job_metadata(changed_query()['stdout'])
    after.update(JobState='RUNNING',NumCPUs='80',NumNodes='1',AllocTRES='billing=5,cpu=80,gres/gpu=4,mem=300000M,node=1',
                 NodeList='jean-zay-iam09',RunTime='00:00:04')
    if case=='cpu':after['NumCPUs']='160'
    if case=='gpu':after['AllocTRES']=after['AllocTRES'].replace('gpu=4','gpu=8')
    if case=='memory':after['AllocTRES']=after['AllocTRES'].replace('300000M','600000M')
    if case=='billing':after['AllocTRES']=after['AllocTRES'].replace('billing=5','billing=6')
    if case=='node':after['NumNodes']='2'
    if case=='elapsed':after['RunTime']='03:00:00'
    if case=='request_alias':after['ReqTRES']='billing=5,cpu=80,gres/gpu=4,mem=300000M,node=1'
    if case=='bad_request_alias':after['ReqTRES']='billing=6,cpu=160,gres/gpu=4,mem=300000M,node=1'
    raw=(EXP/'qos_transition_v1_prepared/prior_allocation_evidence.json').read_bytes()
    if case in ('valid','request_alias'):
        assert 'JobState' in verify.validate_started_job(before,after,raw)
    else:
        with pytest.raises(ValueError):verify.validate_started_job(before,after,raw)
