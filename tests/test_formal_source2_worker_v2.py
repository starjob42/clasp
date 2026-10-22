"""Pre-model formal authentication revision; source-only no model execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import types
import pytest
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/rloo_sft_anchor_20261013'


def load(path,name):
 raw=path.read_bytes();m=types.ModuleType(name);m.__file__=str(path);m.__executed_source_sha256__=hashlib.sha256(raw).hexdigest()
 exec(compile(raw,str(path),'exec',dont_inherit=True),m.__dict__);return m

m=load(EXP/'formal_source2_worker_v2.py','formal_worker2_tests')


def test_whole_original_worker_AST_preserved_except_authentication_glue():
 old=ast.parse((EXP/'formal_source2_worker_v1.py').read_bytes());new=ast.parse((EXP/'formal_source2_worker_v2.py').read_bytes())
 new.body=[n for n in new.body if not isinstance(n,ast.Assign) or not any(isinstance(t,ast.Name) and t.id in
    ('AUTHENTICATION_REVISION','CRYPTO_TRANSPORT_SHA','CRYPTO_TRANSPORT_PATH') for t in n.targets)]
 new.body[0]=copy.deepcopy(old.body[0])
 f=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
 cleaned=[]
 for n in f.body:
  if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='need' and any(isinstance(a,ast.Constant) and a.value=='Explicit reviewed formal authentication revision/source binding required' for a in n.value.args):continue
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='transport' for t in n.targets):continue
  if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='transport':
   c=n.value;c.func.value.id='promotion_verifier';c.args=c.args[1:]
   expected=c.args[1];keep=[i for i,k in enumerate(expected.keys) if k.value not in ('operational_authentication_revision','formal_crypto_transport_source_sha256')]
   expected.keys=[expected.keys[i] for i in keep];expected.values=[expected.values[i] for i in keep]
  for child in ast.walk(n):
   if isinstance(child,ast.Tuple):child.elts=[x for x in child.elts if not (isinstance(x,ast.Constant) and x.value=='formal_crypto_transport_binding')]
  cleaned.append(n)
 f.body=cleaned
 assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)


def test_original_source_crypto_and_all_frozen_registries_remain_unchanged():
 expected={'formal_source2_worker_v1.py':'cd2a4a','verify_registration_promotion_v1.py':'9b1f112935adea22d266d3aefcf0b8529d578784deec144bdf407c45b6a76b44'}
 assert hashlib.sha256((EXP/'verify_registration_promotion_v1.py').read_bytes()).hexdigest()==expected['verify_registration_promotion_v1.py']
 assert json.loads((EXP/'planning/formal_execution_source_registry_v1.json').read_bytes())['entries']==[]
 assert json.loads((EXP/'planning/remote_chunk_transport_registry_v2.json').read_bytes())['registrations']=={}
 transport=load(EXP/'formal_crypto_transport_v1.py','formal_worker2_transport')
 assert transport.PRODUCTION_VERIFIERS=={}
 assert m.CRYPTO_TRANSPORT_SHA==hashlib.sha256((EXP/'formal_crypto_transport_v1.py').read_bytes()).hexdigest()
 assert m.PROMOTION_SHA==hashlib.sha256((EXP/'verify_registration_promotion_v1.py').read_bytes()).hexdigest()

@pytest.mark.parametrize('fault',['source','size','expected_source','symlink'])
def test_unbound_authentication_source_cannot_compile_before_models(tmp_path,fault):
 p=tmp_path/'transport.py';p.write_bytes((EXP/'formal_crypto_transport_v1.py').read_bytes())
 b={'path':str(p),'sha256':m.CRYPTO_TRANSPORT_SHA,'size_bytes':p.stat().st_size};expected=m.CRYPTO_TRANSPORT_SHA
 if fault=='source':p.write_bytes(p.read_bytes()+b'\n')
 elif fault=='size':b['size_bytes']+=1
 elif fault=='expected_source':expected='0'*64
 else:
  linked=tmp_path/'linked.py';linked.symlink_to(p);b['path']=str(linked)
 with pytest.raises(ValueError):m.module_from_binding(b,'must_not_compile',expected)


def test_authentication_is_before_any_project_model_import():
 body=(EXP/'formal_source2_worker_v2.py').read_text()
 assert body.index('transport.verify_promotion(')<body.index('from clasp.entrypoints import')<body.index('source2.main()')
 assert 'fixture_public_keys' not in body and 'fixture_interpreter_binding' not in body
 assert 'synthetic_fixture=False' in body and 'formal_crypto_transport_binding' in body


def test_default_empty_registry_rejects_real_worker_before_auth_or_models(tmp_path):
 from test_formal_source2_operational_v1 import fixture
 spec,_,sp,_=fixture(tmp_path);spec['launcher_binding']={'path':str(EXP/'formal_srun_launch_v1.py'),'sha256':m.LAUNCHER_SHA,'size_bytes':(EXP/'formal_srun_launch_v1.py').stat().st_size}
 spec['spec_sha256']=hashlib.sha256(json.dumps({k:v for k,v in spec.items() if k!='spec_sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
 sp.write_text(json.dumps(spec));registry=tmp_path/'empty.json';registry.write_bytes((EXP/'planning/formal_execution_source_registry_v1.json').read_bytes())
 with pytest.raises(ValueError,match='empty'):m.execute(sp,hashlib.sha256(sp.read_bytes()).hexdigest(),registry,hashlib.sha256(registry.read_bytes()).hexdigest(),tmp_path/'no_promotion.json','0'*64)
