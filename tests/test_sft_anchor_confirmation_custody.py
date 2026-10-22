"""Source-only incremental/chunked custody tests; no model study or native action."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from test_sft_anchor_confirmatory_analysis import synthetic_round, seal, encoded
from test_sft_anchor_admission_v3_collection import _durable_slot, scorer

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/rloo_sft_anchor_20261013'
SPEC = importlib.util.spec_from_file_location('confirmation_custody_tests', EXP / 'collect_confirmation_round_v1.py')
custody = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(custody)


class ByteProvider:
    """Explicitly synthetic virtual native bytes, with stable read fingerprints."""
    def __init__(self, files): self.files = files
    def fingerprint(self, path):
        raw = self.files[path]
        return {'size_bytes':len(raw), 'synthetic_sha256':hashlib.sha256(raw).hexdigest()}
    def iter_bytes(self,path,chunk_bytes):
        raw = self.files[path]
        for i in range(0,len(raw),chunk_bytes): yield raw[i:i+chunk_bytes]
    def tree_entries(self,root):
        return sorted((p[len(root)+1:],self.fingerprint(p)) for p in self.files if p.startswith(root+'/'))


def native_path(contract, slot, name):
    root = contract['native_control_roots'][slot['draw_id']]
    task,arm=slot['task'],slot['arm']
    if name=='attempt.json': return root+f'/attempts/{task}.{arm}.json'
    if name=='scheduler.json': return '/synthetic/read-only-sacct/'+slot['scheduler']['job_id']+'.json'
    if name=='complete.json': return root+f'/outputs/{task}.{arm}/complete.json'
    return root+f'/outputs/{task}.{arm}/{task}/{arm}/'+('training/' if name=='actual_prompt_windows.jsonl' else '')+name


def repair_terminal(payloads, tree_raw):
    tree={'files':[{'path':'fixture_adapter.safetensors','sha256':hashlib.sha256(tree_raw).hexdigest(),'size_bytes':len(tree_raw)}]}
    tree['tree_sha256']=custody.canonical(tree['files'])
    training=json.loads(payloads['training_complete.json']);training['terminal_artifact_tree']=tree
    payloads['training_complete.json']=encoded(training)
    reload=json.loads(payloads['checkpoint_reload_exact.json']);reload['training_complete_sha256']=hashlib.sha256(payloads['training_complete.json']).hexdigest()
    payloads['checkpoint_reload_exact.json']=encoded(reload)
    result=json.loads(payloads['result.json'])
    result['resources']['training_complete_sha256']=reload['training_complete_sha256']
    result['resources']['checkpoint_reload_exact_sha256']=hashlib.sha256(payloads['checkpoint_reload_exact.json']).hexdigest()
    seal(result,'result_sha256');payloads['result.json']=encoded(result)
    return tree


def capture_slots(packet,store,*,count=4):
    contract,bundle,packet_reader=packet
    files={}
    slots=copy.deepcopy(bundle['slots'][:count])
    terminal_raw=b'SYNTHETIC ADAPTER BYTES ONLY'
    for slot in slots:
        payloads={name:packet_reader(binding) for name,binding in slot['artifacts'].items()}
        tree=repair_terminal(payloads,terminal_raw)
        slot['native_terminal_artifact_proof']=tree
        for name,raw in payloads.items(): files[native_path(contract,slot,name)]=raw
        terminal=native_path(contract,slot,'result.json').rsplit('/',1)[0]+'/terminal'
        files[terminal+'/fixture_adapter.safetensors']=terminal_raw
    source=json.loads(packet_reader(contract['source_manifest_binding']))
    for row in source['files']:
        file={'src/clasp/entrypoints/train_sft_anchor_pilot_v2.py':ROOT/'src/clasp/entrypoints/train_sft_anchor_pilot_v2.py',
              'src/clasp/baselines/sft_anchor_pilot.py':ROOT/'src/clasp/baselines/sft_anchor_pilot.py'}[row['path']]
        files[contract['native_source_root']+'/'+row['path']]=file.read_bytes()
    provider=ByteProvider(files)
    source_chunks={row['path']:store.ingest(provider,contract['native_source_root']+'/'+row['path'],expected=row) for row in source['files']}
    for slot in slots:
        slot['artifacts']={name:store.ingest(provider,native_path(contract,slot,name)) for name in slot['artifacts']}
        terminal=native_path(contract,slot,'result.json').rsplit('/',1)[0]+'/terminal'
        tree,bindings=custody.capture_terminal_tree(store,provider,terminal)
        assert tree==slot['native_terminal_artifact_proof']
        slot['terminal_file_chunks']=bindings
    return slots,source_chunks


@pytest.fixture(scope='module')
def packet(): return synthetic_round()


@pytest.fixture
def captured(packet,tmp_path):
    store=custody.ChunkStore(tmp_path/'snapshot',chunk_bytes=4096)
    slots,sources=capture_slots(packet,store)
    context=custody.prepare_context(packet[0],packet[2],synthetic_fixture=True)
    reader=custody.ChunkReader(store.root)
    custody.verify_source_snapshot(context,sources,reader)
    return context,slots,reader,store


def test_incremental_four_slot_draw_full_replay_and_no_claim(captured):
    context,slots,reader,store=captured
    binding,receipt=custody.collect_draw(context,'draw00',slots,reader,store)
    assert receipt['all_four_slots_admitted'] is True
    assert sum(s['raw_responses_verified'] for s in receipt['slots'])==72
    assert all(s['full_independent_physical_replay'] and not s['all_raw_token_arrays_retained'] for s in receipt['slots'])
    assert not receipt['formal_claim_eligible'] and receipt['synthetic_fixture']
    assert json.loads(reader(binding))==receipt


def test_unregistered_round_never_uses_synthetic_as_claim_evidence(packet):
    with pytest.raises(ValueError,match='No independently registered'):
        custody.prepare_context(packet[0],packet[2],expected_contract_sha256=packet[0]['contract_sha256'])
    contract=copy.deepcopy(packet[0]);contract['synthetic_fixture']=False;seal(contract,'contract_sha256')
    with pytest.raises(ValueError,match='explicit fixture'):
        custody.prepare_context(contract,packet[2],synthetic_fixture=True)
    assert custody.KNOWN_REGISTRATIONS=={} and custody.A.PRODUCTION_ADMISSION_READY is False


def test_chunk_capture_has_no64mb_whole_file_limit(tmp_path):
    root=tmp_path/'native';root.mkdir();file=root/'big.bin'
    block=b'x'*(1024*1024)
    with file.open('wb') as stream:
        for _ in range(65): stream.write(block)
        stream.write(b'end')
    store=custody.ChunkStore(tmp_path/'snap',chunk_bytes=262144)
    provider=custody.LocalReadOnlyProvider([root]);binding=store.ingest(provider,str(file))
    reader=custody.ChunkReader(store.root);manifest=reader.manifest(binding)
    assert manifest['size_bytes']>64*1024*1024
    assert max(x['size_bytes'] for x in manifest['chunks'])<=262144
    assert sum(map(len,reader.iter_bytes(binding)))==manifest['size_bytes']
    with pytest.raises(ValueError,match='streaming reader'):reader.small(binding)


@pytest.mark.parametrize('case',['mutated_source','gap','overlap','corrupt_chunk','writable_chunk','trailing_json','duplicate_key','wrong_semantic_seal'])
def test_chunk_and_streaming_integrity_guards(tmp_path,case):
    raw=b'{"raw_completions":[],"x":1}'
    provider=ByteProvider({'/synthetic/file':raw})
    store=custody.ChunkStore(tmp_path/'snap',chunk_bytes=7)
    if case=='mutated_source':
        original=provider.fingerprint
        calls=0
        def fingerprint(path):
            nonlocal calls
            calls+=1;value=original(path)
            if calls>1:value['synthetic_sha256']='0'*64
            return value
        provider.fingerprint=fingerprint
        with pytest.raises(ValueError,match='mutated'):store.ingest(provider,'/synthetic/file')
        return
    binding=store.ingest(provider,'/synthetic/file');reader=custody.ChunkReader(store.root)
    if case in ('gap','overlap','wrong_semantic_seal'):
        manifest=reader.manifest(binding)
        if case=='gap':manifest['chunks'][1]['offset']+=1
        if case=='overlap':manifest['chunks'][1]['offset']-=1
        if case!='wrong_semantic_seal':seal(manifest,'manifest_sha256')
        else:manifest['size_bytes']+=1
        binding=store.write_receipt(manifest)
        with pytest.raises(ValueError):list(reader.iter_bytes(binding))
    elif case in ('corrupt_chunk','writable_chunk'):
        path=store.root/reader.manifest(binding)['chunks'][0]['path'];path.chmod(0o640)
        if case=='corrupt_chunk':path.write_bytes(b'CORRUPT');path.chmod(0o440)
        with pytest.raises(ValueError):list(reader.iter_bytes(binding))
    else:
        value=raw+b'trailing' if case=='trailing_json' else b'{"raw_completions":[],"x":1,"x":2}'
        with pytest.raises(ValueError):custody.JSONStream([value]).object('raw_completions',lambda r:None)


@pytest.mark.parametrize('case',['raw_text','token_ids','selected','metric','step_failure','gpu_allocation','attempt_retry','rank','reference','prompt','terminal','native_path'])
def test_each_corruption_is_unadmitted_and_never_dropped(captured,case):
    context,slots,reader,store=captured
    slots=copy.deepcopy(slots)
    slot=slots[0]
    name={'raw_text':'result.json','token_ids':'result.json','selected':'result.json','metric':'result.json',
          'step_failure':'scheduler.json','gpu_allocation':'scheduler.json','attempt_retry':'attempt.json',
          'rank':'training_complete.json','reference':'training_complete.json','prompt':'actual_prompt_windows.jsonl',
          'native_path':'complete.json'}.get(case)
    if case=='terminal':
        first=next(iter(slot['terminal_file_chunks']));slot['terminal_file_chunks'].pop(first)
    else:
        original=reader.manifest(slot['artifacts'][name]);raw=reader.small(slot['artifacts'][name])
        if case=='prompt':new=b'{"step":999}\n'
        else:
            value=json.loads(raw)
            if case=='raw_text':value['raw_completions'][0]['text']='corrupted'
            if case=='token_ids':value['raw_completions'][0]['completion_token_ids']=[999,999]
            if case=='selected':value['selected']['k8'][0]['signed_gap']=1
            if case=='metric':value['metrics']['k8']['n']=0
            if case=='step_failure':value['stdout']=value['stdout'].replace('.0|COMPLETED|0:0','.0|FAILED|1:0')
            if case=='gpu_allocation':value['stdout']=value['stdout'].replace('gres/gpu=4','gres/gpu=1')
            if case=='attempt_retry':value['attempt_number']=2
            if case=='rank':value['distributed']['rank_closure'].pop()
            if case=='reference':value['reference']['full_integrity_verified']=False
            if name=='result.json':seal(value,'result_sha256')
            new=encoded(value)
        native=original['native_path']+('/wrong' if case=='native_path' else '')
        slot['artifacts'][name]=store.ingest(ByteProvider({native:new}),native)
    _,receipt=custody.collect_draw(context,'draw00',slots,reader,store)
    assert len(receipt['slots'])==4 and not receipt['all_four_slots_admitted']
    row=next(r for r in receipt['slots'] if r['task']==slot['task'] and r['arm']==slot['arm'])
    assert row['state']=='unadmitted' and row['reason']
    assert not receipt['formal_claim_eligible']


def test_actual_saved_four_rank_peft_groups_stream_replay_still_not_production(tmp_path):
    base=EXP/'validation/v2_peft_cpu_ddp4_v1'
    folder=base/'op/rloo_cardinal_beta005'
    contract=json.loads((base/'op/contract.json').read_bytes())
    store=custody.ChunkStore(tmp_path/'snap',chunk_bytes=127)
    provider=custody.LocalReadOnlyProvider([base])
    result=store.ingest(provider,str(folder/'result.json'))
    groups=store.ingest(provider,str(folder/'evaluation_groups.jsonl'))
    verifier=custody.module_from_bytes((EXP/'collect_campaign_admission_v3.py').read_bytes(),custody.DURABLE_VERIFIER_SHA256,'_actual_four_rank_projection')
    scored=custody.module_from_bytes((ROOT/custody.SCORER).read_bytes(),custody.SCORER_SHA256,'_actual_four_rank_scorer')
    replay=custody.replay_result_stream(custody.ChunkReader(store.root),result,groups,contract,'rloo_cardinal_beta005',scored,verifier)
    assert replay['raw_responses_verified']==36 and replay['metadata']['scope']=='system_smoke'
    assert replay['metadata']['resources']['production_fixture'] is True
    assert custody.A.PRODUCTION_ADMISSION_READY is False and custody.KNOWN_REGISTRATIONS=={}


def test_source_inventory_must_be_full_and_exact(captured):
    context,slots,reader,store=captured
    context['source_verified']=False
    missing=dict(context['source_chunks']);missing.pop(next(iter(missing)))
    with pytest.raises(ValueError,match='Complete registered source'):custody.verify_source_snapshot(context,missing,reader)
    with pytest.raises(ValueError,match='source inventory'):custody.validate_slot(context,slots[0],reader)


def test_open_or_closed_failed_draws_retained_in_twenty_receipts(captured,packet):
    context,slots,reader,store=captured
    receipts={}
    for i in range(20):
        draw=f'draw{i:02}'
        binding,value=custody.collect_draw(context,draw,[],reader,store)
        assert len(value['slots'])==4 and all(r['state']=='missing' for r in value['slots'])
        receipts[draw]=binding
    closure={'state':'closed','no_seed_replacement_or_outcome_based_stopping':True,
             'draw_states':[{'draw_id':f'draw{i:02}','state':'declared_permanent_failure','reason':'synthetic permanent failure'} for i in range(20)]}
    _,value=custody.finalize_round(context,receipts,closure,reader,store,chronology_binding=packet[1]['chronology_binding'])
    assert value['registered_seed_draws']==20 and value['expected_trained_slots']==80
    assert value['raw_responses_verified']==0 and not value['formal_claim_eligible']
    assert not value['registered_closed_custody_complete']
    closure['draw_states'].pop()
    with pytest.raises(ValueError,match='all20'):custody.finalize_round(context,receipts,closure,reader,store,chronology_binding=packet[1]['chronology_binding'])
    closure['state']='open'
    _,value=custody.finalize_round(context,receipts,closure,reader,store,chronology_binding=packet[1]['chronology_binding'])
    assert value['status']=='open_custody_diagnostic' and not value['formal_claim_eligible']


def synthetic500_contract():
    """Manufactured graphs solely for fixed500 streaming tests, never a study cohort."""
    original=json.loads((EXP/'prepared_v1/contracts/op_seed43.json').read_bytes())
    rows=[]
    for i in range(500):
        n=10 if i<167 else 40 if i<334 else 70
        coordinates=[[100000+i+j,2*i] for j in range(n)]
        rows.append({'instance_id':f'SYNTHETIC500-{i}','task_type':'op','instance_json':[coordinates,[0]+[1]*(n-1),1000],
                     'prompt':f'SYNTHETIC INPUT ONLY {i}','output':'Route: [0,1]'})
    return scorer.make_contract('op',original['initializer'],original['training_rows'],rows,train_seed=100,scope='production')


def test_exact500_raw4500_streaming_replays_all_fields_and_reductions(tmp_path):
    contract=synthetic500_contract()
    entry={'sha256':custody.ENTRYPOINT_SHA256,'size_bytes':(ROOT/custody.ENTRYPOINT).stat().st_size}
    payloads,_=_durable_slot('op.rloo_cardinal_beta005',contract,
        {'native_source_root':'/synthetic/source','native_control_root':'/synthetic/control'},entry)
    store=custody.ChunkStore(tmp_path/'snap',chunk_bytes=8192)
    provider=ByteProvider({'/synthetic/result':payloads['result.json'],'/synthetic/groups':payloads['evaluation_groups.jsonl']})
    result=store.ingest(provider,'/synthetic/result');groups=store.ingest(provider,'/synthetic/groups')
    verifier=custody.module_from_bytes((EXP/'collect_campaign_admission_v3.py').read_bytes(),custody.DURABLE_VERIFIER_SHA256,'_full500_projection')
    frozen_scorer=custody.module_from_bytes((ROOT/custody.SCORER).read_bytes(),custody.SCORER_SHA256,'_full500_scorer')
    replay=custody.replay_result_stream(custody.ChunkReader(store.root),result,groups,contract,'rloo_cardinal_beta005',frozen_scorer,verifier,synthetic_fixture=True)
    assert replay['raw_responses_verified']==4500 and len(replay['metadata']['selected']['k8'])==500
    assert replay['metadata']['metrics']['k8']['N']==500 and replay['metadata']['metrics']['k8']['n']==500
    assert replay['metadata']['metrics']['k8']['signed_gap_percent']==0
    assert replay['maximum_candidates_per_decode_group']==8
    assert custody.KNOWN_REGISTRATIONS=={}


def test_mutated_round_contract_after_context_preparation_rejected(captured):
    context,slots,reader,store=captured
    context=copy.copy(context);context['contract']=copy.deepcopy(context['contract'])
    context['contract']['round_number']=2;seal(context['contract'],'contract_sha256')
    with pytest.raises(ValueError,match='mutated'):custody.collect_draw(context,'draw00',slots,reader,store)


def test_finalizer_cannot_trust_forged_or_dropped_slot_receipt(captured,packet):
    context,slots,reader,store=captured
    binding,receipt=custody.collect_draw(context,'draw00',slots,reader,store)
    receipt['slots'][0]['metrics']['k8']['n']=999
    seal(receipt,'receipt_sha256');forged=store.write_receipt(receipt)
    with pytest.raises(ValueError,match='replay differs'):
        custody.finalize_round(context,{'draw00':forged},{'state':'open','draw_states':[]},reader,store,chronology_binding=packet[1]['chronology_binding'])
    receipt['slots'].pop();seal(receipt,'receipt_sha256');forged=store.write_receipt(receipt)
    with pytest.raises(ValueError,match='all four'):
        custody.finalize_round(context,{'draw00':forged},{'state':'open','draw_states':[]},reader,store,chronology_binding=packet[1]['chronology_binding'])
