"""Source/input/ownership guards and actual tiny numerical constructor parity."""
import ast
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'experiments/rloo_one_shot_20261016'
PATH=HERE/'confirmation_7b_factory_v3.py'


def binding(p):
 p=Path(p).resolve();raw=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}


def load(p,name):
 b=binding(p);m=types.ModuleType(name);m.__file__=b['path'];m.__executed_source_sha256__=b['sha256']
 exec(compile(Path(p).read_bytes(),str(p),'exec',dont_inherit=True),m.__dict__);return m


F=load(PATH,'bound_confirmation_factory_v3_tests')
MASTER=binding(HERE/'teacher_execution_v2/source_manifest.json')
INPUTS={name:binding(HERE/relative) for name,relative in [('training','data_v1/op_train.json'),
 ('evaluation','confirmation_data_v1/op_validation.json'),('admission','confirmation_data_v1/admission.json')]}


def test_import_and_old12_source_loading_are_stdlib_only():
 program="import sys,types,pathlib,hashlib; p=pathlib.Path(sys.argv[1]);r=p.read_bytes();m=types.ModuleType('safe');m.__file__=str(p);m.__executed_source_sha256__=hashlib.sha256(r).hexdigest();exec(compile(r,str(p),'exec'),m.__dict__);m.load_master_context({'path':sys.argv[2],'sha256':sys.argv[3],'size_bytes':int(sys.argv[4])});assert 'torch' not in sys.modules and 'clasp' not in sys.modules"
 r=subprocess.run([sys.executable,'-I','-S','-B','-c',program,str(PATH),MASTER['path'],MASTER['sha256'],str(MASTER['size_bytes'])],capture_output=True,text=True,timeout=20)
 assert r.returncode==0,r.stderr


@pytest.fixture(scope='module')
def context():
 c=F.load_master_context(MASTER);m=F.load_transfer_module(binding(HERE/'confirmation_teacher_transfer_v3.py'),MASTER)
 return c,m


@pytest.mark.parametrize('seed',[101,211,307,401,503])
def test_exact_actual_master12_and_fixed256_numerics_are_admitted_without_policy_calls(context,seed):
 c,m=context;master=c['master_contract'];student=m.make_student_contract(master,seed,json.loads(Path(INPUTS['evaluation']['path']).read_bytes()))
 proof=F.validate_confirmation_numerics(transfer_module=m,master_contract=master,student_contract=student,input_bindings=INPUTS,master_manifest_binding=MASTER)
 assert proof['seed']==seed and proof['horizon']==100 and proof['evaluation_rows']==256
 assert proof['confirmation_policy_authority_granted'] is False and proof['inner_projection_only'] is True
 assert m.V2.__file__==c['sources']['train_teacher_candidate_v2.py']['path']
 assert m.V2.calibration_module().data_module().production_admission_module().__file__==c['sources']['admit_training_collection_v1.py']['path']
 assert len(c['sources'])==12


@pytest.mark.parametrize('fault',['seed','horizon','G','training_order','evaluation_order','evaluation96','LR','scope'])
def test_caller_reseal_cannot_change_fixed_confirmation_numerics(context,fault):
 c,m=context;master=c['master_contract'];student=m.make_student_contract(master,101,json.loads(Path(INPUTS['evaluation']['path']).read_bytes()))
 if fault=='seed':student['train_seed']=73
 elif fault=='horizon':student['training']['horizon']=99
 elif fault=='G':student['training']['groups_per_update']=4
 elif fault=='training_order':student['training_rows'].reverse()
 elif fault=='evaluation_order':student['validation_rows'].reverse()
 elif fault=='evaluation96':student['validation_rows']=master['validation_rows']
 elif fault=='LR':student['training']['learning_rate']=2e-5
 else:student['scope']='production'
 student['contract_sha256']=m.seal(student,'contract_sha256')
 with pytest.raises(ValueError):F.validate_confirmation_numerics(transfer_module=m,master_contract=master,student_contract=student,input_bindings=INPUTS,master_manifest_binding=MASTER)


@pytest.mark.parametrize('fault',['manifest','input_SHA','input_size_bool','master_logical','transfer_source'])
def test_source_and_input_bindings_cannot_be_promoted_or_rebound(context,fault):
 c,m=context;master=copy.deepcopy(c['master_contract']);ib=copy.deepcopy(INPUTS);mb=copy.deepcopy(MASTER)
 student=m.make_student_contract(master,101,json.loads(Path(INPUTS['evaluation']['path']).read_bytes()))
 if fault=='manifest':mb['sha256']='0'*64
 elif fault=='input_SHA':ib['evaluation']['sha256']='0'*64
 elif fault=='input_size_bool':ib['evaluation']['size_bytes']=True
 elif fault=='master_logical':master['train_seed']=101;master['contract_sha256']=m.seal(master,'contract_sha256')
 else:
  with pytest.raises(ValueError):F.load_transfer_module({'path':str(PATH),'sha256':'0'*64,'size_bytes':1},MASTER)
  return
 with pytest.raises(ValueError):F.validate_confirmation_numerics(transfer_module=m,master_contract=master,student_contract=student,input_bindings=ib,master_manifest_binding=mb)


@pytest.fixture
def source_owner(monkeypatch,context):
 c,_=context;source=HERE/'execution_v1/source'
 base=load(source/'run_local_7b_v1.py','native_guard_runtime_metadata')
 core=types.SimpleNamespace(__file__=str(source/'src/clasp/entrypoints/train_sft_anchor_one_shot_v1.py'),__executed_source_sha256__=F.CORE_SHA256,
  one_shot_quality=types.SimpleNamespace(__file__=str(source/'src/clasp/rewards/one_shot_quality.py'),__executed_source_sha256__=F.REWARD_SHA256))
 owner={'pid':os.getpid(),'uid':os.getuid(),'gpu_uuid':base.GPU_UUID,'all_locks_held_until_GPU_lifetime_closed':True}
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES',base.GPU_UUID);monkeypatch.setenv('WORLD_SIZE','1');monkeypatch.setenv('RANK','0');monkeypatch.setenv('LOCAL_RANK','0')
 return c,base,core,owner


@pytest.mark.parametrize('fault',['pid','uid_bool','uuid','lock','visible','world','source','outer_SHA'])
def test_unowned_or_unbound_context_denied_before_parameter_access(source_owner,monkeypatch,fault):
 c,base,core,owner=source_owner;outer='a'*64
 if fault=='pid':owner['pid']=-1
 elif fault=='uid_bool':owner['uid']=True
 elif fault=='uuid':owner['gpu_uuid']='other'
 elif fault=='lock':owner['all_locks_held_until_GPU_lifetime_closed']=False
 elif fault=='visible':monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
 elif fault=='world':monkeypatch.setenv('WORLD_SIZE','4')
 elif fault=='source':base.__executed_source_sha256__=None
 else:outer=True
 with pytest.raises(ValueError):F._native_guard(base_runtime=base,core=core,context=c,actor=None,reference=None,gpu_owner=owner,
  expected_actor_sha256='b'*64,outer_contract_sha256=outer,effective_source_manifest_sha256='c'*64)


@pytest.mark.parametrize('teacher_only',[False,True])
def test_tiny_CPU_metadata_cannot_enter_realBF16_production_guard(source_owner,teacher_only):
 c,base,core,owner=source_owner
 p=types.SimpleNamespace(numel=lambda:1,dtype='torch.bfloat16',device='cpu',requires_grad=True,is_floating_point=lambda:True)
 actor=types.SimpleNamespace(named_parameters=lambda:[('weight',p)])
 with pytest.raises(ValueError,match='Actual6B'):
  F._native_guard(base_runtime=base,core=core,context=c,actor=actor,reference=None if teacher_only else actor,gpu_owner=owner,
   expected_actor_sha256='b'*64,outer_contract_sha256='a'*64,effective_source_manifest_sha256='c'*64)


def test_mixed_path_mandatorily_calls_original5953_validation_and_parent_resource_closure():
 tree=ast.parse(PATH.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_master_calibration')
 calls=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
 assert {n.func.attr for n in calls}>={'validate_contract','verify_result','completed_parent'}
 bad={}
 c=F.load_master_context(MASTER)
 with pytest.raises(ValueError):F.validate_master_calibration(c,bad,recipe='rloo_cost',shared_data={},teacher_targets={},calibration={},expected_actor_sha256='a'*64)


def test_production_teacher_only_calls_exact_original_CE_loop_and_not_seed73_public_wrapper():
 tree=ast.parse(PATH.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_teacher_only')
 calls=[n.func.attr for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
 assert calls.count('_ce_loop')==1 and 'run_teacher_only' not in calls
 assert 'loaded_model' not in calls and 'generate' not in calls


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
 torch=pytest.importorskip('torch');torch.set_num_threads(1)
 from transformers import AutoTokenizer
 from clasp.entrypoints import train_sft_anchor_one_shot_v1 as core
 old=HERE/'validation/confirmation_teacher_transfer_v3/actual_cpu';base=HERE/'validation/teacher_candidate_v2/actual_cpu/op/initializer/base';sft=base.parent/'sft'
 tok=AutoTokenizer.from_pretrained(base);tok.model_input_names=['input_ids','attention_mask']
 m=F.load_transfer_module(binding(HERE/'confirmation_teacher_transfer_v3.py'),MASTER)
 master=json.loads((old/'master_contract.json').read_bytes());shared=json.loads((old/'shared_data.json').read_bytes());targets=json.loads((old/'teacher_targets.json').read_bytes())
 calibrations={r:json.loads((old/('calibration_'+r)/'calibration.json').read_bytes()) for r in F.MIXED_RECIPES}
 return {'torch':torch,'core':core,'m':m,'master':master,'shared':shared,'targets':targets,'calibrations':calibrations,
  'base':base,'sft':sft,'tok':tok,'root':evidence_root(tmp_path_factory)}


def evidence_root(tmp_path_factory):
 declared=os.environ.get('ONE_SHOT_CONFIRMATION_FACTORY_EVIDENCE')
 if declared:
  root=Path(declared);root.mkdir(parents=True,exist_ok=False);return root
 return tmp_path_factory.mktemp('confirmation_factory_v3')


def teacher_kwargs(t,recipe,student):
 c=t['calibrations'][recipe]
 tr=t['m'].make_transfer(t['master'],student,recipe,t['shared'],t['targets'],c,t['tok'])
 return {'master_contract':t['master'],'teacher_transfer':tr,'expected_teacher_transfer_sha256':tr['transfer_sha256'],
  'shared_data':t['shared'],'expected_data_sha256':t['shared']['shared_data_sha256'],'teacher_targets':t['targets'],
  'expected_teacher_targets_sha256':t['targets']['teacher_targets_sha256'],'calibration':c,'expected_calibration_sha256':c['calibration_sha256']}


def numerical_snapshot(t,trainer):
 from transformers import TrainerCallback
 from clasp.models.external_baseline_state import semantic_state_sha256
 class Stop(TrainerCallback):
  def on_step_end(self,args,state,control,**kwargs):control.should_training_stop=True
 trainer.add_callback(Stop());trainer.train();D=t['m'].V2.calibration_module().data_module()
 proof={'step':trainer.state.global_step,'actor':semantic_state_sha256(trainer.model.state_dict()),'reference':trainer.reference_state_sha256(),
  'optimizer':semantic_state_sha256(trainer.optimizer.state_dict()),'scheduler':semantic_state_sha256(trainer.lr_scheduler.state_dict()),
  'RNG':D.rng_sha(D.rng_state()),'windows':trainer.actual_windows,'generated_tokens':trainer.actual_generated_tokens}
 if hasattr(trainer,'teacher_records'):
  proof['CE']=[{k:v for k,v in r.items() if 'wall_seconds' not in k} for r in trainer.teacher_records]
 Path(trainer.args.output_dir,'numeric_snapshot.json').write_text(json.dumps(proof,sort_keys=True,indent=2)+'\n')
 return proof


@pytest.mark.parametrize('recipe',F.PURE_RECIPES)
def test_actual_tiny_pure_constructor_one_H3_window_matches_frozen_core(tiny,recipe,monkeypatch):
 t=tiny;torch=t['torch'];monkeypatch.setenv('WORLD_SIZE','1');monkeypatch.setenv('RANK','0');monkeypatch.delenv('LOCAL_RANK',raising=False)
 student=t['m'].make_student_contract(t['master'],101)
 old=t['core'].make_cpu_trainer(student,recipe,t['root']/('old_'+recipe),t['tok'],t['base'],t['sft'])
 expected=numerical_snapshot(t,old);del old;gc.collect()
 actor=t['core'].v2.loaded_model(t['base'],t['sft'],torch.device('cpu'),trainable=True,dtype=torch.float32)
 ref=t['core'].v2.loaded_model(t['base'],t['sft'],torch.device('cpu'),trainable=False,dtype=torch.float32)
 new=F.make_cpu_diagnostic_trainer(core=t['core'],student_contract=student,recipe=recipe,actor=actor,reference=ref,
  tokenizer=t['tok'],output=t['root']/('new_'+recipe))
 assert numerical_snapshot(t,new)==expected
 assert new.args.max_steps==3 and new.num_generations==9
 del new,actor,ref;gc.collect()


@pytest.mark.parametrize('recipe',F.MIXED_RECIPES)
def test_actual_tiny_mixed_constructor_one_H3_window_matches_frozen_transfer(tiny,recipe,monkeypatch):
 t=tiny;torch=t['torch'];monkeypatch.setenv('WORLD_SIZE','1');monkeypatch.setenv('RANK','0');monkeypatch.delenv('LOCAL_RANK',raising=False)
 student=t['m'].make_student_contract(t['master'],101);kw=teacher_kwargs(t,recipe,student)
 old=t['m'].make_cpu_trainer(student,recipe,t['root']/('old_mixed_'+recipe),t['tok'],t['base'],t['sft'],**kw)
 expected=numerical_snapshot(t,old);del old;gc.collect()
 actor=t['core'].v2.loaded_model(t['base'],t['sft'],torch.device('cpu'),trainable=True,dtype=torch.float32)
 ref=t['core'].v2.loaded_model(t['base'],t['sft'],torch.device('cpu'),trainable=False,dtype=torch.float32)
 new=F.make_cpu_diagnostic_trainer(core=t['core'],student_contract=student,recipe=recipe,actor=actor,reference=ref,
  tokenizer=t['tok'],output=t['root']/('new_mixed_'+recipe),transfer_module=t['m'],teacher_kwargs=kw)
 assert numerical_snapshot(t,new)==expected
 assert new.teacher_forward_calls==1 and new.args.max_steps==3
 del new,actor,ref;gc.collect()


def test_actual_newseed_teacher_only_uses_identical_frozen_loop_noRL_and_clears_gradients(tiny,monkeypatch):
 t=tiny;torch=t['torch'];from transformers import set_seed
 from clasp.models.external_baseline_state import semantic_state_sha256
 student=t['m'].make_student_contract(t['master'],211);context=F.load_master_context(MASTER);helper=context['modules']['teacher_only_7b_v2.py']
 C=t['m'].V2.calibration_module();D=C.data_module()
 old_actor=t['core'].v2.loaded_model(t['base'],t['sft'],torch.device('cpu'),trainable=True,dtype=torch.float32)
 oldout=t['root']/'old_CE';oldout.mkdir();args=t['core'].v2._training_args(student,'rloo_scalar_beta005',oldout,torch.device('cpu'),production_peft=True)
 set_seed(211)
 old=helper.run_cpu_diagnostic(actor=old_actor,tokenizer=t['tok'],rows=student['training_rows'],targets=t['targets']['targets'],
  draws=student['training']['prompt_draws'],args=args,cal=C,util=D.adapter_v1(),anchor=t['core'].v2,
  semantic_state_sha256=semantic_state_sha256,output=oldout,expected_initial_sha256=t['shared']['SFT_identity']['whole_actor_sha256'])
 oldstate=semantic_state_sha256(old_actor.state_dict());oldrng=D.rng_sha(D.rng_state());del old_actor;gc.collect()
 actor=t['core'].v2.loaded_model(t['base'],t['sft'],torch.device('cpu'),trainable=True,dtype=torch.float32)
 newout=t['root']/'new_CE';newout.mkdir()
 monkeypatch.setattr(t['core'],'strict_cost_reward',lambda *a,**k:pytest.fail('Teacher-only must not call RL reward'))
 new=F.run_cpu_teacher_only_diagnostic(transfer_module=t['m'],master_manifest_binding=MASTER,master_contract=t['master'],
  student_contract=student,actor=actor,tokenizer=t['tok'],shared_data=t['shared'],teacher_targets=t['targets'],output=newout,
  expected_initial_sha256=t['shared']['SFT_identity']['whole_actor_sha256'])
 assert semantic_state_sha256(actor.state_dict())==oldstate and D.rng_sha(D.rng_state())==oldrng
 assert new['actual_windows']==old['actual_windows']==student['training']['prompt_draws']
 assert new['updates']==new['teacher_CE_forwards']==new['teacher_CE_backwards']==3 and new['RL_responses']==new['RL_reward_calls']==0
 assert new['all_grads_cleared'] and not new['reference_KL'] and not new['reference_model_loaded_by_helper']
 assert new['confirmation_policy_authority_granted_by_factory'] is False
 for label,proof in [('old_CE',old),('new_CE',new)]:
  (t['root']/label/'CE_complete.json').write_text(json.dumps(proof,sort_keys=True,indent=2)+'\n')
 del actor;gc.collect()


@pytest.mark.parametrize('fault',['different_core','unexecuted_core','unexecuted_reward'])
def test_native_factory_rejects_early_ordinary_or_mismatched_project_imports(context,fault):
 _,m=context
 core=types.SimpleNamespace(__file__=m.core.__file__,__executed_source_sha256__=F.CORE_SHA256,
  one_shot_quality=types.SimpleNamespace(__file__=m.core.one_shot_quality.__file__,__executed_source_sha256__=F.REWARD_SHA256))
 transfer=types.SimpleNamespace(core=core,V2=types.SimpleNamespace(core=core))
 if fault=='different_core':transfer.core=object()
 elif fault=='unexecuted_core':core.__executed_source_sha256__=None
 else:core.one_shot_quality.__executed_source_sha256__=None
 with pytest.raises(ValueError):F.require_native_project_imports(core,transfer)
