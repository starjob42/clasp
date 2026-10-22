"""Real original121 bytes plus labelled synthetic full-source composition."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import pytest

ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/rloo_sft_anchor_20261013'
spec=importlib.util.spec_from_file_location('formal_layer',EXP/'formal_source_layer_v2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def binding(path):
    raw=path.read_bytes();return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}


@pytest.fixture
def packet(tmp_path):
    root=tmp_path/'synthetic_effective_source';root.mkdir()
    original_root=EXP/'source_v2';original=json.loads((original_root/'source_manifest.json').read_bytes())
    rows=copy.deepcopy(original['files'])
    for row in rows:
        target=root/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original_root/row['path'],target)
    embedded=root/'original_source2_manifest.json';shutil.copyfile(original_root/'source_manifest.json',embedded)
    extra=[embedded]
    for name in ('formal_srun_launch_v1.py','formal_source2_worker_v1.py','formal_source_layer_v2.py','evaluation_rescue_bootstrap_v1.py',
                 'verify_registration_promotion_v1.py','prepare_confirmation_round_v1.py','confirmatory_analysis.py','collect_confirmation_round_v1.py','select_candidate.py'):
        path=root/'experiments/rloo_sft_anchor_20261013'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(EXP/name,path);extra.append(path)
    for path in extra:rows.append({**binding(path),'path':path.relative_to(root).as_posix()})
    manifest=root/'source_manifest.json';manifest.write_text(json.dumps({'synthetic_fixture':True,'files':rows}))
    ops=[binding(root/'experiments/rloo_sft_anchor_20261013'/n) for n in ('formal_srun_launch_v1.py','formal_source2_worker_v1.py','formal_source_layer_v2.py','evaluation_rescue_bootstrap_v1.py','verify_registration_promotion_v1.py')]
    return root,binding(manifest),binding(embedded),ops


def test_distinct_original_manifest_and_effective_closure_are_executable_without_same_name_conflict(packet):
    root,effective,original,ops=packet
    result=m.resolve(root,effective,original,ops)
    assert result['original_files_verified']==121 and result['effective_files_verified']>121
    assert result['original_manifest_sha256']==m.ORIGINAL_SHA!=result['effective_manifest_sha256']
    assert hashlib.sha256(result['original_buffers'][m.ENTRY]).hexdigest()==m.ENTRY_SHA
    assert 'original_source2_manifest.json' in result['declared_extra_paths']
    # Actual original algorithm module bytes stay separate from extra helpers.
    assert 'experiments/rloo_sft_anchor_20261013/formal_source_layer_v2.py' not in result['original_buffers']


@pytest.mark.parametrize('mutation',['same_name','original_bytes','original_row','missing_extra','unbound_python','cached_python','linked_extra','helper_binding','original_omitted'])
def test_original_effective_or_operational_source_mutations_cannot_be_admitted(packet,mutation):
    root,effective,original,ops=packet
    if mutation=='same_name':original=effective
    elif mutation=='original_bytes':(root/m.ENTRY).write_bytes((root/m.ENTRY).read_bytes()+b'\n# drift\n')
    elif mutation in ('original_row','original_omitted'):
        p=Path(effective['path']);value=json.loads(p.read_bytes())
        if mutation=='original_row':value['files'][0]['sha256']='0'*64
        else:value['files']=[row for row in value['files'] if row['path']!='original_source2_manifest.json']
        p.write_text(json.dumps(value));effective=binding(p)
    elif mutation=='missing_extra':Path(ops[0]['path']).unlink()
    elif mutation=='unbound_python':(root/'unbound.py').write_text('raise RuntimeError("never execute")\n')
    elif mutation=='cached_python':(root/'stale.pyc').write_bytes(b'cached code')
    elif mutation=='linked_extra':
        p=Path(ops[0]['path']);p.unlink();p.symlink_to(EXP/'formal_srun_launch_v1.py')
    elif mutation=='helper_binding':ops[0]['sha256']='0'*64
    with pytest.raises((ValueError,FileNotFoundError)):m.resolve(root,effective,original,ops)


def test_real_crypto_fixture_promotion_follows_source_then_registry_without_hash_cycle(packet):
    import base64
    import types
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
    root,effective,original,ops=packet
    resolved=m.resolve(root,effective,original,ops)
    worker=next(item for item in ops if item['path'].endswith('formal_source2_worker_v1.py'))
    # Immutable source/worker bytes close first; a detached registry can then
    # bind them and the later detached signature binds both existing hashes.
    registry={'synthetic_fixture':True,'performance_admission':False,'job_id':'777',
        'effective_source_manifest_sha256':effective['sha256'],'original_source_manifest_sha256':original['sha256'],
        'worker_sha256':worker['sha256']}
    registry_raw=json.dumps(registry,sort_keys=True,separators=(',',':')).encode()
    expected={**registry,'consumer':'formal_source2_fixture_diagnostic','registry_sha256':hashlib.sha256(registry_raw).hexdigest()}
    rel='experiments/rloo_sft_anchor_20261013/verify_registration_promotion_v1.py'
    raw=resolved['all_buffers'][rel];verifier=types.ModuleType('fixture_promotion');verifier.__file__=str(root/rel)
    exec(compile(raw,verifier.__file__,'exec',dont_inherit=True),verifier.__dict__)
    key=Ed25519PrivateKey.generate();public=key.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)
    payload={'protocol':verifier.PROTOCOL,'schema_version':1,**expected}
    envelope={'payload':payload,'signer_key_id':'ephemeral_fixture_only',
        'signature_base64':base64.b64encode(key.sign(verifier.encoded(payload))).decode()}
    keys={'ephemeral_fixture_only':{'scope':'fixture_only','purpose':'independent_prepolicy_registration',
        'public_key_base64':base64.b64encode(public).decode()}}
    keys['ephemeral_fixture_only']['crypto_environment']=verifier.crypto_environment()
    result=verifier.verify_promotion(envelope,expected,synthetic_fixture=True,fixture_public_keys=keys)
    assert not result['production_registration_authenticated']
    assert hashlib.sha256(m.bound(effective)).hexdigest()==effective['sha256']
    # Default real trusted keys remain empty. Fixture promotion cannot turn
    # the same packet into native authority, even after exact source closure.
    with pytest.raises(ValueError):verifier.verify_promotion(envelope,expected)
