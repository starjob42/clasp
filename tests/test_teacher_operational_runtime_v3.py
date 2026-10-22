"""CPU-only operational transport guards over immutable real source closures."""
import ast
import builtins
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT=Path(__file__).parents[1];BASE=ROOT/'experiments/rloo_one_shot_20261016';DIR=BASE/'cpu_preparation_reconciliation_v3';SOURCE=DIR/'run_teacher_operational_v3.py'


def module():
 raw=SOURCE.read_bytes();m=types.ModuleType('operational_transport_fixture');m.__file__=str(SOURCE);m.__executed_source_sha256__=hashlib.sha256(raw).hexdigest()
 exec(compile(raw,str(SOURCE),'exec',dont_inherit=True),m.__dict__);return m


def fixture(tmp_path,code=None,child_code=None):
 m=module();entry=tmp_path/'diagnostic_entry.py';entry.write_text(code or 'def main():\n print("syntheticCPUentry")\n')
 extra=[];entries=[m.binding(entry)]
 if child_code is not None:
  child=tmp_path/'diagnostic_child.py';child.write_text(child_code);extra.append(m.binding(child));entries.append(m.binding(child))
 ctx=m.make_context(bootstrap=m.binding(SOURCE),adapter=m.binding(DIR/'vocabulary_validation_v3.py'),
  master_packet=m.binding(BASE/'teacher_execution_v2/execution_packet.json'),project_manifest=m.binding(BASE/'execution_v1/source_manifest.json'),
  extra_manifest=m.binding(BASE/'teacher_execution_v2/source_manifest.json'),sources=entries,allowed_entries=entries,receipt_root=str(tmp_path/'receipts'))
 path=tmp_path/'context.json';path.write_bytes(m.encoded(ctx)+b'\n');return m,ctx,m.binding(path),m.binding(entry),tmp_path/'receipts/phase.json'


def test_default_cold_import_no_model_no_output(tmp_path):
 r=subprocess.run([sys.executable,'-I','-S','-B',str(SOURCE)],capture_output=True,text=True,cwd=tmp_path,check=True)
 assert json.loads(r.stdout)['status']=='prepared_only_no_entry_or_model_execution' and not list(tmp_path.iterdir())


def test_real107_and12_closure_and_command_pins_are_exact(tmp_path):
 m,c,cb,e,p=fixture(tmp_path);v,buffers=m.load_context(cb)
 assert len(v['sources'])>=125 and len(buffers)==len(v['sources'])
 argv=m.command(e,[],cb,p);assert argv[:4]==[m.PYTHON,'-I','-B','-c']
 assert '--context-sha256' in argv and cb['sha256'] in argv and '--receipt' in argv and '--execute' in argv


@pytest.mark.parametrize('fault',['source','context','entry','receipt_escape','wrong_master','missing_project_row'])
def test_bound_drift_or_escape_rejects_before_entry(tmp_path,fault):
 m,c,cb,e,p=fixture(tmp_path)
 if fault=='source':Path(e['path']).write_text('raise AssertionError("changed")')
 elif fault=='context':Path(cb['path']).write_bytes(Path(cb['path']).read_bytes()+b' ')
 elif fault=='entry':e={**e,'sha256':'0'*64}
 elif fault=='receipt_escape':p=tmp_path/'outside.json'
 elif fault=='wrong_master':
  c['master_packet']['sha256']='0'*64;c=m.seal(c,'context_sha256');Path(cb['path']).write_bytes(m.encoded(c));cb=m.binding(Path(cb['path']))
 else:
  c['sources']=c['sources'][1:];c=m.seal(c,'context_sha256');Path(cb['path']).write_bytes(m.encoded(c));cb=m.binding(Path(cb['path']))
 with pytest.raises(ValueError):m.command(e,[],cb,p)
 assert not p.exists()


def test_actual_fresh_cpu_process_closes_before_after_sidecars(tmp_path):
 m,c,cb,e,p=fixture(tmp_path);r=subprocess.run(m.command(e,[],cb,p),capture_output=True,text=True,check=True,timeout=30)
 assert 'syntheticCPUentry' in r.stdout;v=m.verify_receipt(m.binding(p),e,cb)
 assert v['status']=='completed' and v['guard_transformations']==[] and v['resource_authority_granted'] is False
 assert v['started_receipt_binding']['path']==str(p.with_suffix('.started.json'))


def test_actual_fresh_legacy_boot_child_inherits_context_and_sidecar(tmp_path):
 child=tmp_path/'diagnostic_child.py'
 # This is the exact same BOOT format used by5953/BC288/3b4; no model.
 m=module();child.write_text('def main():\n print("syntheticCPUchild")\n')
 code='import subprocess,sys\nBOOT='+repr(next(b for b in m.LEGACY_BOOTS if '__bounded_worker__' in b))+'\ndef main():\n subprocess.run([sys.executable,"-I","-B","-c",BOOT,'+repr(str(child))+','+repr(m.sha(child.read_bytes()))+'],check=True)\n'
 m,c,cb,e,p=fixture(tmp_path,code=code,child_code=child.read_text());r=subprocess.run(m.command(e,[],cb,p),capture_output=True,text=True,timeout=30)
 assert r.returncode==0,r.stderr
 v=m.verify_receipt(m.binding(p),e,cb);assert 'syntheticCPUchild' in r.stdout and len(v['child_propagation'])==1
 childrow=v['child_propagation'][0]['receipts'][0];cv=json.loads(Path(childrow['path']).read_bytes())
 assert cv['parent_operational_receipt']==str(p) and cv['context_binding']==cb


def test_actual_target_function_full_buffer_is_adapted_before_entry_main(tmp_path):
 target=BASE/'execution_v1/source/run_local_7b_v1.py';raw=target.read_bytes()
 code='from pathlib import Path\nimport types,hashlib\ndef main():\n p=Path('+repr(str(target))+');raw=p.read_bytes();m=types.ModuleType("diagnostic_old_base");m.__file__=str(p);m.__executed_source_sha256__=hashlib.sha256(raw).hexdigest();exec(compile(raw,str(p),"exec",dont_inherit=True),m.__dict__);assert "__vocabulary_validation_v3__" in m.__dict__\n'
 m,c,cb,e,p=fixture(tmp_path,code);r=subprocess.run(m.command(e,[],cb,p),capture_output=True,text=True,timeout=30)
 assert r.returncode==0,r.stderr
 v=m.verify_receipt(m.binding(p),e,cb);proof=v['guard_transformations'][0]
 assert proof['function']=='validate_raw_completion' and proof['original_buffer_was_executed_before_declared_function_replacement'] is True


def test_actual_target_pure_ast_slice_keeps_explicit_slice_provenance(tmp_path):
 target=BASE/'execution_v1/source/run_local_7b_v1.py'
 code='from pathlib import Path\nimport ast\ndef main():\n p=Path('+repr(str(target))+');tree=ast.parse(p.read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="validate_raw_completion");ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),"exec",dont_inherit=True),ns);assert "__vocabulary_validation_v3__" in ns\n'
 m,c,cb,e,p=fixture(tmp_path,code);r=subprocess.run(m.command(e,[],cb,p),capture_output=True,text=True,timeout=30)
 assert r.returncode==0,r.stderr
 proof=m.verify_receipt(m.binding(p),e,cb)['guard_transformations'][0]
 assert proof['known_full_source_function_matched_in_pure_AST_slice'] is True and proof['original_buffer_was_executed_before_declared_function_replacement'] is False


def test_global_exec_and_subprocess_are_restored_on_failure(tmp_path):
 m,c,cb,e,p=fixture(tmp_path,'def main():\n raise RuntimeError("synthetic expected failure")\n');old=builtins.exec;oldp=subprocess.Popen
 with pytest.raises(RuntimeError):m.execute_entry(e,[],cb,p)
 assert builtins.exec is old and subprocess.Popen is oldp
 assert json.loads(p.read_bytes())['status']=='failed'
 with pytest.raises(ValueError):m.verify_receipt(m.binding(p),e,cb)


def test_duplicate_receipt_cannot_repeat_entry(tmp_path):
 m,c,cb,e,p=fixture(tmp_path);m.execute_entry(e,[],cb,p)
 with pytest.raises(ValueError,match='No retry'):m.execute_entry(e,[],cb,p)


def test_unbound_legacy_boot_child_rejects_before_spawn(tmp_path):
 m,c,cb,e,p=fixture(tmp_path);runtime=m.Runtime(cb,e,p)
 try:
  with pytest.raises(ValueError,match='escaped'):runtime.rewrite_child([m.PYTHON,'-I','-B','-c',next(iter(m.LEGACY_BOOTS)),'/tmp/unbound_entry.py','a'*64])
 finally:runtime.close('failed','synthetic fixture only')


def test_popen_non_python_and_non_boot_calls_remain_exact(tmp_path):
 m,c,cb,e,p=fixture(tmp_path);runtime=m.Runtime(cb,e,p)
 try:
  for argv in (['echo','CPU'],[sys.executable,'-c','print(1)']):assert runtime.rewrite_child(argv) is argv
 finally:runtime.close('failed','synthetic fixture only')


def rebind_context(m,path,value):
 path.write_bytes(m.encoded(m.seal(value,'context_sha256'))+b'\n');return m.binding(path)


def test_additive_context_extension_preserves_all_old_roles_and_receipt_roots(tmp_path):
 m,c,cb,e,p=fixture(tmp_path);extra=tmp_path/'additional_reviewed_CPU_entry.py';extra.write_text('def main(): pass\n')
 new=json.loads(m.encoded(c));row=m.binding(extra);new['sources'].append(row);new['allowed_entries'].append(row)
 new['additional_receipt_roots']=[str(tmp_path/'additional_receipts')]
 nb=rebind_context(m,tmp_path/'extended_context.json',new)
 proof=m.verify_context_extension(nb,cb)
 assert proof['old_sources_preserved'] is True and proof['new_source_bindings']==[row]
 assert proof['resource_authority_granted'] is False
 assert m.command(row,[],nb,tmp_path/'additional_receipts/phase.json')


@pytest.mark.parametrize('fault',['remove_source','remove_role','rebind_source','primary_root','adapter','master'])
def test_context_extension_cannot_rewrite_original_custody(tmp_path,fault):
 m,c,cb,e,p=fixture(tmp_path);new=json.loads(m.encoded(c))
 if fault=='remove_source':new['sources'].remove(e)
 elif fault=='remove_role':new['allowed_entries'].remove(e)
 elif fault=='rebind_source':
  for row in new['sources']:
   if row['path']==e['path']:row['sha256']='0'*64
 elif fault=='primary_root':new['receipt_root']=str(tmp_path/'different_primary')
 elif fault=='adapter':new['adapter']['sha256']='0'*64
 else:new['master_packet']['sha256']='0'*64
 nb=rebind_context(m,tmp_path/'bad_extension.json',new)
 with pytest.raises(ValueError):m.verify_context_extension(nb,cb)


@pytest.mark.parametrize('fault',['arbitrary_root','arbitrary_prefix','wrong_sha','new_verifier_teacher_stage'])
def test_paper_alias_policy_is_finite_and_same_byte_only(tmp_path,fault):
 m,c,cb,e,p=fixture(tmp_path);policy=next(v for v in c['paper_stage_aliases'] if v['basename']=='verify_confirmation_artifacts_v3.py')
 if fault=='arbitrary_root':policy['root']=str(tmp_path)
 elif fault=='arbitrary_prefix':policy['directory_prefix']='.unrelated_'
 elif fault=='wrong_sha':policy['source_sha256']='0'*64
 else:policy['directory_prefix']='.rloo9_teacher_stage_'
 nb=rebind_context(m,tmp_path/'bad_alias.json',c)
 with pytest.raises(ValueError,match='paper-stage'):m.load_context(nb)


def test_canonical_qualified_binary_legacy_paper_child_is_propagated(tmp_path):
 m,c,cb,e,p=fixture(tmp_path);runtime=m.Runtime(cb,e,p)
 try:
  canonical=str(Path(m.PYTHON).resolve());assert m.binding(Path(canonical))['sha256']==m.ML_BINARY_SHA
  utility=next(v for v in c['allowed_entries'] if Path(v['path']).name=='verify_confirmation_artifacts_v3.py')
  argv=[canonical,'-I','-B','-c',next(b for b in m.LEGACY_BOOTS if '__bounded_worker__' not in b),utility['path'],utility['sha256'],'--help']
  changed=runtime.rewrite_child(argv)
  assert changed[0]==canonical and changed[1:4]==['-I','-B','-c'] and changed[4]!=argv[4]
  assert runtime.children[0]['template']['entry']==utility
 finally:runtime.close('failed','synthetic transport only; no child spawned')


def test_same_entry_fresh_children_have_distinct_invocation_receipts(tmp_path):
 m=module();child=tmp_path/'diagnostic_child.py';child.write_text('def main(): pass\n')
 boot=next(b for b in m.LEGACY_BOOTS if '__bounded_worker__' in b)
 code='import subprocess,sys\nBOOT='+repr(boot)+'\ndef main():\n for a in ("first","second"):\n  subprocess.run([sys.executable,"-I","-B","-c",BOOT,'+repr(str(child))+','+repr(m.sha(child.read_bytes()))+',a],check=True)\n'
 m,c,cb,e,p=fixture(tmp_path,code,child.read_text());subprocess.run(m.command(e,[],cb,p),capture_output=True,text=True,timeout=30,check=True)
 v=m.verify_receipt(m.binding(p),e,cb);children=v['child_propagation']
 assert len(children)==2 and children[0]['template']['child_invocation_token']!=children[1]['template']['child_invocation_token']
 children[1]['receipts']=children[0]['receipts'];p.write_bytes(m.encoded(m.seal(v))+b'\n')
 with pytest.raises(ValueError,match='another invocation'):m.verify_receipt(m.binding(p),e,cb)


def test_receipt_replays_exact_inverse_AST_and_derived_code_proof(tmp_path):
 target=BASE/'execution_v1/source/run_local_7b_v1.py'
 code='from pathlib import Path\nimport ast\ndef main():\n p=Path('+repr(str(target))+');node=next(n for n in ast.parse(p.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=="validate_raw_completion");exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),"exec",dont_inherit=True),{})\n'
 m,c,cb,e,p=fixture(tmp_path,code);subprocess.run(m.command(e,[],cb,p),capture_output=True,text=True,check=True,timeout=30)
 v=m.verify_receipt(m.binding(p),e,cb);assert v['guard_transformations']
 v['guard_transformations'][0]['derived_function_code_sha256']='0'*64;p.write_bytes(m.encoded(m.seal(v))+b'\n')
 with pytest.raises(ValueError,match='inverse-AST'):m.verify_receipt(m.binding(p),e,cb)


@pytest.mark.parametrize('filename,function',[
 ('collect_training_sft_v1.py','validate_record'),('teacher_candidate_v2.py','_validate_record')])
def test_remaining_exact_target_functions_are_adapted_in_actual_CPU_children(tmp_path,filename,function):
 target=BASE/'teacher_execution_v2/source_extra'/filename
 code='from pathlib import Path\nimport ast\ndef main():\n p=Path('+repr(str(target))+');node=next(n for n in ast.parse(p.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='+repr(function)+');ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),"exec",dont_inherit=True),ns);assert "__vocabulary_validation_v3__" in ns\n'
 m,c,cb,e,p=fixture(tmp_path,code);subprocess.run(m.command(e,[],cb,p),capture_output=True,text=True,check=True,timeout=30)
 v=m.verify_receipt(m.binding(p),e,cb)
 assert v['guard_transformations'][0]['function']==function and v['guard_transformations'][0]['known_full_source_function_matched_in_pure_AST_slice'] is True


@pytest.mark.parametrize('fault',['unbound_entry','borrowed_started_receipt'])
def test_independent_sidecar_verifier_rejects_resealed_unbound_custody(tmp_path,fault):
 m,c,cb,e,p=fixture(tmp_path);m.execute_entry(e,[],cb,p);v=json.loads(p.read_bytes())
 if fault=='unbound_entry':
  entry=tmp_path/'unbound_entry.py';entry.write_text('def main(): pass\n');e=m.binding(entry);v['entry_binding']=e
  started_path=Path(v['started_receipt_binding']['path']);s=json.loads(started_path.read_bytes());s['entry_binding']=e
  started_path.write_bytes(m.encoded(m.seal(s))+b'\n');v['started_receipt_binding']=m.binding(started_path)
 else:
  other=tmp_path/'borrowed.started.json';other.write_bytes(Path(v['started_receipt_binding']['path']).read_bytes());v['started_receipt_binding']=m.binding(other)
 p.write_bytes(m.encoded(m.seal(v))+b'\n')
 with pytest.raises(ValueError,match='closed allowed|exact final'):m.verify_receipt(m.binding(p),e,cb)
