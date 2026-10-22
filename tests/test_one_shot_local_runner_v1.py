"""Source-only custody/resource/decoder guards; no GPU or model execution."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import pytest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/rloo_one_shot_20261016/run_local_7b_v1.py'
spec=importlib.util.spec_from_file_location('one_shot_local_runner_guards',SOURCE)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
AUDIT=ROOT/'experiments/rloo_one_shot_20261016/initializer_audit_v1.json'

def binding(path,rows=None):
 value=r.file_binding(path)
 if rows is not None:value['rows']=rows
 return value

@pytest.fixture
def contract(tmp_path):
 audit=json.loads(AUDIT.read_bytes());init={k:audit[k] for k in ('base_root','base_files')}
 init.update({k:audit['tasks']['op'][k] for k in ('sft_root','sft_files')});init['independent_audit_binding']=binding(AUDIT)
 def rows(prefix):return [{'instance_id':prefix+str(i),'semantic_id':prefix+str(i),'prompt':'x'+str(i),'instance_json':[[[0,0],[1,0]],[0,1],3],
                         'output':'Route: [0, 1], Objective: 1','task_type':'op'} for i in range(3)]
 train=tmp_path/'train.json';evaluation=tmp_path/'eval.json';source=tmp_path/'source.json'
 train.write_text(json.dumps(rows('train')));evaluation.write_text(json.dumps(rows('eval')));source.write_text('{}')
 return r.make_contract(mode='qualification',task='op',arm='one_shot_rloo',seed=19,
  train_binding=binding(train,3),evaluation_binding=binding(evaluation,3),initializer=init,
  source_manifest_binding=binding(source),wall_seconds=1800,cohort_scope='engineering_only')

def reseal(value):value['contract_sha256']=r.sha(r.encoded({k:v for k,v in value.items() if k!='contract_sha256'}));return value

def test_true_B9_single_prompt_group_and_greedy_K1_are_distinct(contract):
 assert contract['recipe']['num_generations']==9 and contract['recipe']['groups_per_update']==1
 assert contract['recipe']['global_responses_per_update']==9 and contract['evaluation_rows']==2
 assert contract['recipe']['gradient_checkpointing'] is True and contract['recipe']['use_reentrant'] is False

@pytest.mark.parametrize('field,value',[('schema_version',True),('seed',False),('seed',2**32),('optimizer_updates',100),
 ('evaluation_rows',3),('wall_seconds',1801),('wall_seconds',False),('gpu_uuid','GPU-other'),('actual_gpu_count',True),
 ('cohort_scope','protected_original_pilot'),('superiority_established',True),('publication_eligible',True)])
def test_qualification_cannot_silently_change_scope_counts_budget_or_types(contract,field,value):
 c=copy.deepcopy(contract);c[field]=value
 with pytest.raises(ValueError):r.validate_contract(reseal(c))

@pytest.mark.parametrize('field,value',[('num_generations',1),('global_responses_per_update',36),('top_p',.9),('top_k',50),
 ('use_reentrant',True),('gradient_checkpointing',False),('beta',0),('learning_rate',1e-4),('max_completion_tokens',1)])
def test_training_recipe_changes_are_rejected_even_when_resealed(contract,field,value):
 c=copy.deepcopy(contract);c['recipe'][field]=value
 with pytest.raises(ValueError):r.validate_contract(reseal(c))

def test_changed_model_audit_cannot_be_self_resealed(contract):
 c=copy.deepcopy(contract);c['initializer']['independent_audit_binding']['sha256']='0'*64
 with pytest.raises(ValueError,match='audit'):r.validate_contract(reseal(c))

def test_all_five_study_controls_have_explicit_estimators(contract):
 assert r.ARM_RECIPES=={'one_shot_rloo':'rloo_cost','one_shot_grpo':'grpo_cost','one_shot_grpo_fixed_horizon':'grpo_cost_fixed_horizon','foarl':'foarl_released','one_shot_sft':'sft'}
 for arm in r.ARMS:
  c=copy.deepcopy(contract);c.update(mode='study',arm=arm,optimizer_updates=0 if arm=='one_shot_sft' else 100,evaluation_rows=96,wall_seconds=1800 if arm=='one_shot_sft' else 14400,cohort_scope='fresh_development')
  assert r.validate_contract(reseal(c))['evaluation_rows']==96

def test_bound_dataset_keeps_full3_rows_but_exact_first2_evaluation(contract):
 train,evaluation=r.verify_datasets(contract)
 assert len(train)==3 and [x['instance_id'] for x in evaluation]==['eval0','eval1']

@pytest.mark.parametrize('fault',['byte','count','duplicate','overlap','task'])
def test_dataset_tamper_and_physical_overlap_fail_before_model_call(contract,fault):
 c=copy.deepcopy(contract);p=Path(c['evaluation']['path']);rows=json.loads(p.read_bytes())
 if fault=='byte':p.write_bytes(p.read_bytes()+b' ')
 else:
  if fault=='count':rows.pop()
  elif fault=='duplicate':rows[1]['instance_id']=rows[0]['instance_id']
  elif fault=='overlap':rows[0]['semantic_id']='train0'
  elif fault=='task':rows[0]['task_type']='cvrp'
  p.write_text(json.dumps(rows));c['evaluation']=binding(p,3)
 with pytest.raises(ValueError):r.verify_datasets(c)

def test_typed_full_model_hash_rejects_changed_bytes_and_symlink(tmp_path):
 p=tmp_path/'weights';p.write_bytes(b'weights');expected={'sha256':r.sha(b'weights'),'size_bytes':7}
 assert r.stream_bound(p,expected)['sha256']==expected['sha256']
 p.write_bytes(b'changed')
 with pytest.raises(ValueError):r.stream_bound(p,expected)
 link=tmp_path/'link';link.symlink_to(p)
 with pytest.raises(ValueError):r.stream_bound(link,expected)

def snapshot(tmp_path):
 root=tmp_path/'snapshot';entry=root/'runner.py';entry.parent.mkdir();entry.write_bytes(SOURCE.read_bytes())
 package=root/'src/clasp';package.mkdir(parents=True);(package/'__init__.py').write_text('value=7\n');(package/'closed.py').write_text('from . import value\nanswer=value*6\n')
 rows=[{'path':p.relative_to(root).as_posix(),**{k:v for k,v in binding(p).items() if k!='path'}} for p in (entry,package/'__init__.py',package/'closed.py')]
 manifest=root/'manifest.json';manifest.write_text(json.dumps({'root':str(root),'files':rows}))
 module=types.ModuleType('snapshot_runner');module.__file__=str(entry);module.__executed_source_sha256__=r.sha(entry.read_bytes())
 exec(compile(entry.read_bytes(),str(entry),'exec'),module.__dict__)
 return root,module,binding(manifest)

def test_all_project_imports_execute_closed_raw_buffers_without_site_fallback(tmp_path):
 root,m,b=snapshot(tmp_path);source,buffers=m.source_buffers(b)
 finder=m.install_project_imports(source,buffers)
 try:
  import clasp.closed
  assert clasp.closed.answer==42 and clasp.closed.__executed_source_sha256__==r.sha(buffers[str(root/'src/clasp/closed.py')])
  with pytest.raises(ValueError,match='Unbound project import'):__import__('clasp.unbound')
 finally:
  sys.meta_path.remove(finder)
  for k in list(sys.modules):
   if k=='clasp' or k.startswith('clasp.'):sys.modules.pop(k)

@pytest.mark.parametrize('fault',['missing_manifest_module','changed_module','bytecode','loaded_old_source'])
def test_source_closure_catches_missing_changed_cached_or_late_rebound_code(tmp_path,fault):
 root,m,b=snapshot(tmp_path)
 if fault=='missing_manifest_module':(root/'src/clasp/extra.py').write_text('value=0')
 elif fault=='changed_module':(root/'src/clasp/closed.py').write_text('value=0')
 elif fault=='bytecode':(root/'src/clasp/extra.pyc').write_bytes(b'cache')
 else:m.__executed_source_sha256__='0'*64
 with pytest.raises(ValueError):m.source_buffers(b)

def gpu_mock(occupied=False):
 def run(argv,**kwargs):
  out=f'3, {r.GPU_UUID}, NVIDIA RTX A6000, 49140, 48675, 0\n' if any('--query-gpu=' in a for a in argv) else (f'{r.GPU_UUID}, 999, unrelated, 2\n' if occupied else '')
  return types.SimpleNamespace(returncode=0,stderr='',stdout=out)
 return run

def test_all_known_GPU_reservation_locks_remain_held_until_worker_closes(tmp_path):
 output=tmp_path/'out';output.mkdir();locks=tmp_path/'locks';locks.mkdir()
 with r.exclusive_gpu(output,run=gpu_mock(),lock_root=locks):
  for p in locks.glob('*.lock'):
   with p.open('rb') as f:
    with pytest.raises(BlockingIOError):r.fcntl.flock(f.fileno(),r.fcntl.LOCK_EX|r.fcntl.LOCK_NB)
 for p in locks.glob('*.lock'):
  with p.open('rb') as f:r.fcntl.flock(f.fileno(),r.fcntl.LOCK_EX|r.fcntl.LOCK_NB)

def test_other_process_or_legacy_reservation_prevents_GPU_use(tmp_path):
 output=tmp_path/'out';output.mkdir();locks=tmp_path/'locks';locks.mkdir()
 with pytest.raises(ValueError,match='another compute process'): 
  with r.exclusive_gpu(output,run=gpu_mock(True),lock_root=locks):pytest.fail('Must not acquire occupied GPU')
 p=locks/'graphopt_iclr_prospective_factorial_ourasi_gpu3.lock'
 with p.open('rb') as f:
  r.fcntl.flock(f.fileno(),r.fcntl.LOCK_EX|r.fcntl.LOCK_NB)
  with pytest.raises(BlockingIOError):
   with r.exclusive_gpu(output,run=gpu_mock(),lock_root=locks):pytest.fail('Must honor legacy reservation')

def test_default_cli_does_not_call_worker_or_GPU(monkeypatch,capsys):
 monkeypatch.setattr(sys,'argv',[str(SOURCE)])
 monkeypatch.setattr(r,'gpu_worker',lambda *a:pytest.fail('No GPU by default'))
 r.main();assert json.loads(capsys.readouterr().out)['status']=='prepared_only_no_model_or_GPU_calls'

def test_timeout_preserves_one_attempt_and_does_not_retry_or_fallback(contract,tmp_path,monkeypatch):
 path=tmp_path/'contract.json';path.write_bytes(r.encoded(contract));out=tmp_path/'attempt';calls=[]
 monkeypatch.setattr(r,'source_buffers',lambda b:(None,{}));monkeypatch.setattr(r,'__executed_source_sha256__',r.sha(SOURCE.read_bytes()),raising=False)
 monkeypatch.setattr(sys,'executable',r.LOCAL_PYTHON)
 monkeypatch.setattr(sys,'argv',[str(SOURCE),'--contract',str(path),'--contract-sha256',r.sha(path.read_bytes()),'--output',str(out),'--execute'])
 def timeout(argv,**kwargs):calls.append((argv,kwargs));raise subprocess.TimeoutExpired(argv,kwargs['timeout'])
 monkeypatch.setattr(r.subprocess,'run',timeout)
 with pytest.raises(RuntimeError,match='TIMEOUT'):r.main()
 proof=json.loads((out/'completion.json').read_bytes())
 assert proof['status']=='failed' and proof['automatic_retry'] is False and len(calls)==1
 assert 0<calls[0][1]['timeout']<=1800 and (out/'attempt.json').exists()
 with pytest.raises(FileExistsError):r.main()
 assert len(calls)==1

def test_artifacts_are_create_once_and_preserve_previous_failure(tmp_path):
 p=tmp_path/'failure.json';r.write_once(p,{'failure':True});before=p.read_bytes()
 with pytest.raises(FileExistsError):r.write_once(p,{'failure':False})
 assert p.read_bytes()==before


def raw_fixture():
 class Tokenizer:
  eos_token_id=2
  def __len__(self):return 3
  def decode(self,ids,skip_special_tokens=True):return ''.join('x' if i==1 else '' for i in ids)
 tokenizer=Tokenizer();config={'do_sample':False,'num_return_sequences':1,'full_default_key':None}
 core=types.SimpleNamespace(generation_config=lambda *a,**k:types.SimpleNamespace(to_dict=lambda:dict(config)),
  score_completion=lambda *a:{'raw_feasible':True,'physical_objective':1.})
 row={'instance_id':'id','semantic_id':'physicalid','prompt':'prompt','task_type':'op','instance_json':[], 'output':'answer'}
 record={'instance_id':'id','semantic_id':'physicalid','prompt_sha256':r.sha(b'prompt'),'completion_token_ids':[1,2],
  'text':'x','generated_tokens':2,'generation_calls':1,'actual_generation_config':dict(config),'use_model_defaults':False,
  'extra_generation_kwargs':{},'eos_observed':True,'truncated':False,'strict_score':{'raw_feasible':True,'physical_objective':1.}}
 return record,row,tokenizer,core

def test_full_greedy_token_text_physical_score_closure():
 assert r.validate_raw_completion(*raw_fixture()) is True

@pytest.mark.parametrize('field,value',[('completion_token_ids',[True,2]),('completion_token_ids',[1,3]),
 ('completion_token_ids',[2,1]),('text','changed'),('generated_tokens',True),('generation_calls',True),
 ('actual_generation_config',{'do_sample':False,'num_return_sequences':1,'full_default_key':.1}),
 ('use_model_defaults',True),('extra_generation_kwargs',{'top_p':.9}),('prompt_sha256','0'*64),
 ('eos_observed',1),('truncated',0),('strict_score',{'raw_feasible':False,'physical_objective':1.})])
def test_raw_and_complete_decoder_mutations_cannot_change_reported_endpoint(field,value):
 record,row,tok,core=raw_fixture();record[field]=value
 with pytest.raises(ValueError):r.validate_raw_completion(record,row,tok,core)

def test_study_cannot_exceed_fixed42_hour_plan_per_arm(contract):
 c=copy.deepcopy(contract);c.update(mode='study',optimizer_updates=100,evaluation_rows=96,cohort_scope='fresh_development',wall_seconds=86400)
 with pytest.raises(ValueError,match='wall authority'):r.validate_contract(reseal(c))

def test_actual_fresh_engineering_rows_construct_real_core_contract_without_model_call(contract):
 code="""
import importlib.util,json
from pathlib import Path
from clasp.entrypoints import train_sft_anchor_one_shot_v1 as core
p=Path('experiments/rloo_one_shot_20261016/run_local_7b_v1.py');spec=importlib.util.spec_from_file_location('runner',p);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
c=json.loads(Path(__import__('sys').argv[1]).read_bytes())
rows=[json.loads(Path('experiments/rloo_one_shot_20261016/data_v1/op_engineering_'+kind+'.json').read_bytes()) for kind in ('train','validation')]
d=r.training_contract(c,rows[0],rows[1][:2],core)
assert d['scope']=='local_qualification' and d['training']['horizon']==1 and d['training']['num_generations']==9
assert d['training']['groups_per_update']==1 and d['training']['global_responses_per_update']==9 and len(d['validation_rows'])==2
print('passed_no_model_or_GPU_calls')
"""
 p=Path(contract['training']['path']).parent/'contract_for_core.json';p.write_bytes(r.encoded(contract))
 env={**__import__('os').environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''}
 completed=subprocess.run(['/tmp/rloo9-kernel-cpu-local-v1/bin/python','-B','-c',code,str(p)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=40)
 assert completed.returncode==0,completed.stderr
 assert completed.stdout.strip()=='passed_no_model_or_GPU_calls'

def fake_parameter(count,dtype='torch.bfloat16',device='cuda:0',trainable=False):
 return types.SimpleNamespace(numel=lambda:count,dtype=dtype,device=device,requires_grad=trainable,is_floating_point=lambda:True)

def test_actual_parameter_dtypes_and_device_counts_are_audited_not_loader_arguments():
 model=types.SimpleNamespace(named_parameters=lambda:[('base.weight',fake_parameter(6_000_000_000)),('lora_A.weight',fake_parameter(100,'torch.float32',trainable=True))])
 d=r.parameter_dtype_receipt(model)
 assert d['actual_BF16_base_weights_verified'] is True and d['trainable_parameters']==100

@pytest.mark.parametrize('parameters',[
 [('base.weight',fake_parameter(6_000_000_000,'torch.float32'))],
 [('base.weight',fake_parameter(5_999_999_999))],
 [('base.weight',fake_parameter(6_000_000_000,device='cpu'))],
 [('base.weight',fake_parameter(6_000_000_000)),('other.weight',fake_parameter(100,'torch.float32'))]])
def test_small_float32_or_cpu_model_cannot_claim_real7B_nativeBF16(parameters):
 with pytest.raises(ValueError):r.parameter_dtype_receipt(types.SimpleNamespace(named_parameters=lambda:parameters))
