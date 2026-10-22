"""CPU-only master-to-student admission and unchanged native-gradient proof."""
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

torch=pytest.importorskip("torch")
from transformers import AutoTokenizer,TrainerCallback
from clasp.entrypoints import train_sft_anchor_one_shot_v1 as core
from clasp.entrypoints import train_sft_anchor_pilot_v2 as anchor
from clasp.models.external_baseline_state import semantic_state_sha256

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'experiments/rloo_one_shot_20261016'
path=HERE/'confirmation_teacher_transfer_v3.py'
spec=importlib.util.spec_from_file_location('teacher_confirmation_transfer_v3_tests',path)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
T=M.V2;C=T.calibration_module();D=C.data_module()


def reseal(x,field='contract_sha256'):
 x[field]=M.seal(x,field);return x


@pytest.fixture(autouse=True)
def world1(monkeypatch):
 torch.set_num_threads(1);monkeypatch.setenv('WORLD_SIZE','1');monkeypatch.setenv('RANK','0');monkeypatch.delenv('LOCAL_RANK',raising=False)
 keys=set(core.reward_environment('foarl'))|set(core.reward_environment('rloo_quality'))
 saved={k:os.environ.get(k) for k in keys}
 yield
 for k,v in saved.items():
  if v is None:os.environ.pop(k,None)
  else:os.environ[k]=v


@pytest.fixture(scope='module')
def f(tmp_path_factory):
 torch.set_num_threads(1)
 declared=os.environ.get('ONE_SHOT_TRANSFER_EVIDENCE')
 root=Path(declared) if declared else tmp_path_factory.mktemp('teacher_confirmation_v3')
 if declared:root.mkdir(parents=True,exist_ok=False)
 old=HERE/'validation/teacher_candidate_v2/actual_cpu/op'
 parent=json.loads((old/'contract.json').read_text())
 base,sft=old/'initializer/base',old/'initializer/sft'
 tok=AutoTokenizer.from_pretrained(base)
 # Match the original in-memory tiny initializer tokenizer; its generic saved
 # WordLevel metadata otherwise defaults to token_type_ids, unsupported by Qwen.
 tok.model_input_names=['input_ids','attention_mask']
 master=core.make_smoke_contract('op',parent['initializer'],parent['training_rows'],parent['validation_rows'],
  horizon=3,max_completion_tokens=4,groups_per_update=1,train_seed=73)
 engine=D.adapter_v1().engine_module()
 excluded={key:[] for key in engine.SPLITS};graphs=copy.deepcopy(excluded)
 excluded['validation']=sorted(r['semantic_id'] for r in master['validation_rows'])
 graphs['validation']=sorted(r['graph_semantic_id'] for r in master['validation_rows'])
 admission=engine.make_training_admission('op',master['training_rows'],
  allowed_training_semantic_ids=sorted(r['semantic_id'] for r in master['training_rows']),
  excluded_semantic_ids_by_split=excluded,excluded_graph_ids_by_split=graphs,
  training_dataset_sha256=core.digest(master['training_rows']))
 context={'split':'training','operation':'supervised_training_target','inference_search':False,'training_dataset_sha256':admission['training_dataset_sha256']}
 data_spec=D.make_spec(master,[r['instance_id'] for r in master['training_rows'][:2]])
 model=anchor.loaded_model(base,sft,torch.device('cpu'),trainable=False,dtype=torch.float32)
 data=D.collect_tiny_cpu_data(master,model,tok,admission=admission,expected_admission_sha256=admission['admission_sha256'],
  context=context,spec=data_spec,expected_spec_sha256=data_spec['spec_sha256']);del model;gc.collect()
 targets=D.prepare_stronger_targets(master,data,data['shared_data_sha256'],tok)
 calibrations={recipe:C.calibrate_cpu(master,recipe,data,data['shared_data_sha256'],targets,targets['teacher_targets_sha256'],
  tok,base,sft,root/('calibration_'+recipe)) for recipe in C.RECIPES}
 for name,value in [('master_contract',master),('shared_data',data),('teacher_targets',targets)]:
  anchor.write_json_atomic(root/(name+'.json'),value)
 return dict(master=master,data=data,targets=targets,calibrations=calibrations,tokenizer=tok,base=base,sft=sft,root=root)


def teacher_kwargs(f,recipe):
 return dict(shared_data=f['data'],expected_data_sha256=f['data']['shared_data_sha256'],
  teacher_targets=f['targets'],expected_teacher_targets_sha256=f['targets']['teacher_targets_sha256'],
  calibration=f['calibrations'][recipe],expected_calibration_sha256=f['calibrations'][recipe]['calibration_sha256'])


def transfer(f,student,recipe):
 return M.make_transfer(f['master'],student,recipe,f['data'],f['targets'],f['calibrations'][recipe],f['tokenizer'])


class Trajectory(TrainerCallback):
 def __init__(self,trainer):self.trainer=trainer;self.pre=[];self.post=[]
 def on_pre_optimizer_step(self,args,state,control,**kwargs):
  pairs=C.parameter_pairs(self.trainer.model,C.parameter_inventory(self.trainer.model))
  g=C.gradient_vector(pairs)
  self.pre.append({'step':state.global_step,'gradient_sha256':semantic_state_sha256(g),'gradient_norm':C.finite_norm(g)})
 def on_step_end(self,args,state,control,**kwargs):
  t=self.trainer
  self.post.append({'step':state.global_step,'actor_sha256':semantic_state_sha256(t.model.state_dict()),
   'reference_sha256':t.reference_state_sha256(),'optimizer_sha256':semantic_state_sha256(t.optimizer.state_dict()),
   'scheduler_sha256':semantic_state_sha256(t.lr_scheduler.state_dict()),'RNG_sha256':D.rng_sha(D.rng_state()),
   'windows':copy.deepcopy(t.actual_windows),'generated_tokens':t.actual_generated_tokens,
   'teacher_CE':[{k:v for k,v in r.items() if 'wall_seconds' not in k} for r in t.teacher_records]})


def run(f,student,recipe,label,*,old=False):
 output=f['root']/label;kw=teacher_kwargs(f,recipe)
 if old:trainer=T.make_cpu_trainer(student,recipe,output,f['tokenizer'],f['base'],f['sft'],**kw)
 else:
  tr=transfer(f,student,recipe)
  trainer=M.make_cpu_trainer(student,recipe,output,f['tokenizer'],f['base'],f['sft'],master_contract=f['master'],
   teacher_transfer=tr,expected_teacher_transfer_sha256=tr['transfer_sha256'],**kw)
  anchor.write_json_atomic(output/'transfer.json',tr)
 callback=Trajectory(trainer);trainer.add_callback(callback);trainer.train();closure=trainer.teacher_closure()
 proof={'pre_optimizer':callback.pre,'post_optimizer':callback.post,'final_rng_sha256':D.rng_sha(D.rng_state()),
  'final_actor_sha256':semantic_state_sha256(trainer.model.state_dict()),'reference_sha256':trainer.reference_state_sha256(),
  'updates':trainer.state.global_step,'RL_responses':9*trainer.state.global_step,'CE_forwards':trainer.teacher_forward_calls,
  'GC':trainer.last_checkpointing_audit,'calibration_sha256':trainer.teacher_calibration_sha256,
  'teacher_targets_sha256':trainer.teacher_targets_sha256,'scope':'tiny_CPU_fixture_only'}
 anchor.write_json_atomic(output/'numeric_trajectory.json',proof)
 anchor.write_json_atomic(output/'teacher_training_complete.json',closure)
 return proof,trainer


@pytest.mark.parametrize('recipe',C.RECIPES)
def test_same_seed73_actual_full_trajectory_and_preoptimizer_gradients_equal_frozen_v2(f,recipe):
 student=M.make_student_contract(f['master'],73)
 assert student==f['master']
 old,t=run(f,student,recipe,'v2_seed73_'+recipe,old=True);del t;gc.collect()
 new,t=run(f,student,recipe,'v3_seed73_'+recipe);del t;gc.collect()
 assert old==new
 assert old['updates']==old['CE_forwards']==3 and old['RL_responses']==27
 assert len(old['pre_optimizer'])==len(old['post_optimizer'])==3
 assert old['GC']['effective_use_reentrant'] is False


@pytest.mark.parametrize('seed',M.PLANNED_SEEDS)
def test_each_new_seed_actual_training_reuses_unchanged_targets_alpha_and_follows_its_own_schedule(f,seed):
 recipe='rloo_cost';student=M.make_student_contract(f['master'],seed)
 immutable=core.digest([f['master'],f['data'],f['targets'],f['calibrations']])
 proof,t=run(f,student,recipe,'v3_seed'+str(seed)+'_'+recipe)
 assert proof['updates']==proof['CE_forwards']==3 and proof['RL_responses']==27
 assert t.teacher_alpha==f['calibrations'][recipe]['alpha']
 assert t.teacher_calibration['core_contract_sha256']==f['master']['contract_sha256']!=student['contract_sha256']
 assert t.actual_windows==[[r[0]]*9 for r in student['training']['prompt_draws']]
 assert [r['instance_id'] for r in t.teacher_records]==[r[0] for r in student['training']['prompt_draws']]
 assert immutable==core.digest([f['master'],f['data'],f['targets'],f['calibrations']])
 assert proof['reference_sha256']==f['data']['SFT_identity']['whole_actor_sha256']
 if seed==101:
  terminal=f['root']/'v3_seed101_terminal'
  model=t.accelerator.unwrap_model(t.model);expected=semantic_state_sha256(model.state_dict())
  model.save_pretrained(terminal);f['tokenizer'].save_pretrained(terminal)
  reloaded=anchor.loaded_model(f['base'],terminal,torch.device('cpu'),trainable=False,dtype=torch.float32)
  assert semantic_state_sha256(reloaded.state_dict())==expected
  anchor.write_json_atomic(f['root']/'v3_seed101_checkpoint_reload_exact.json',
   {'whole_actor_sha256':expected,'bitwise_state_equal':True,'scope':'tiny_CPU_fixture_only',
    'terminal_tree':anchor.artifact_tree(terminal)})
  del reloaded,model
 del t;gc.collect()


@pytest.mark.parametrize('recipe',['foarl_released','foarl_cost_changed'])
def test_same_new_seed_supports_both_foarl_teacher_controls(f,recipe):
 student=M.make_student_contract(f['master'],101)
 proof,t=run(f,student,recipe,'v3_seed101_'+recipe)
 assert proof['updates']==proof['CE_forwards']==3 and proof['RL_responses']==27
 assert t.teacher_alpha==f['calibrations'][recipe]['alpha']
 assert t.actual_windows==[[r[0]]*9 for r in student['training']['prompt_draws']]
 del t;gc.collect()


def test_inherited_compute_backward_and_native_training_methods_are_same_function_objects():
 for name in ['_generate_and_score_completions','_compute_loss','_teacher_backward_if_due','training_step',
              'create_optimizer','_load_optimizer_and_scheduler']:
  assert getattr(M.TransferredTeacherCEMixin,name) is getattr(T.StreamedTeacherCEMixin,name)
  assert name not in M.TransferredTeacherCEMixin.__dict__
 for kind,parent in [(M.TeacherRLOOV3,core.OneShotRLOOTrainer),(M.TeacherFOARLV3,core.OneShotGRPOTrainer)]:
  assert parent in kind.__mro__


@pytest.mark.parametrize('fault',['seed_bool','seed73_changed_horizon','LR','fixed_denominator','reward','sampling',
 'training_order','training_physics','scope','initializer','draw','draw_hash','validation_hash','unknown_field'])
def test_student_reseal_cannot_hide_numerical_or_training_role_changes(f,fault):
 student=M.make_student_contract(f['master'],101)
 if fault=='seed_bool':student['train_seed']=True
 elif fault=='seed73_changed_horizon':student['training']['horizon']=2;student['training']['prompt_draws']=student['training']['prompt_draws'][:2];student['training']['prompt_draw_sha256']=core.digest(student['training']['prompt_draws'])
 elif fault=='LR':student['training']['learning_rate']=2e-5
 elif fault=='fixed_denominator':student['training']['fixed_loss_denominator']=3
 elif fault=='reward':student['recipes']['rloo_cost']['scale_rewards']='group'
 elif fault=='sampling':student['training']['top_p']=.9
 elif fault=='training_order':student['training_rows'].reverse()
 elif fault=='training_physics':student['training_rows'][0]['instance_json'][1][1]+=1
 elif fault=='scope':student['scope']='production'
 elif fault=='initializer':student['initializer']['sha256']='0'*64
 elif fault=='draw':student['training']['prompt_draws'][0]=[student['training_rows'][0]['instance_id']];student['training']['prompt_draw_sha256']=core.digest(student['training']['prompt_draws'])
 elif fault=='draw_hash':student['training']['prompt_draw_sha256']='0'*64
 elif fault=='validation_hash':student['validation_cohort_sha256']='0'*64
 else:student['alpha_is_newly_measured']=True
 reseal(student)
 with pytest.raises((ValueError,RuntimeError)):M.validate_student_contract(f['master'],student)


@pytest.mark.parametrize('fault',['alpha','master','student','source','scope','relabel','calibration_hash'])
def test_transfer_cannot_relabel_or_reseal_old_calibration_as_new_measurement(f,fault):
 student=M.make_student_contract(f['master'],101);tr=transfer(f,student,'rloo_cost')
 if fault=='alpha':tr['alpha']*=2
 elif fault=='master':tr['master_contract_sha256']=student['contract_sha256']
 elif fault=='student':tr['student_contract_sha256']=f['master']['contract_sha256']
 elif fault=='source':tr['sources']['V2_teacher_source_sha256']='0'*64
 elif fault=='scope':tr['scope']='production_training_collection_7b'
 elif fault=='relabel':tr['calibration_measured_again']=True
 else:tr['original_calibration_sha256']='0'*64
 reseal(tr,'transfer_sha256')
 with pytest.raises(ValueError):M.validate_transfer(f['master'],student,'rloo_cost',f['data'],f['targets'],f['calibrations']['rloo_cost'],
  f['tokenizer'],tr,tr['transfer_sha256'])


def test_unchanged_parent_validator_denies_student_while_explicit_transfer_admits(f):
 student=M.make_student_contract(f['master'],211);tr=transfer(f,student,'rloo_cost')
 with pytest.raises(ValueError):D.validate_shared_data(student,f['data'],f['data']['shared_data_sha256'],f['tokenizer'])
 with pytest.raises(ValueError):C.validate_calibration(student,'rloo_cost',f['data'],f['targets'],f['calibrations']['rloo_cost'],
  f['calibrations']['rloo_cost']['calibration_sha256'],f['tokenizer'])
 assert M.validate_transfer(f['master'],student,'rloo_cost',f['data'],f['targets'],f['calibrations']['rloo_cost'],
  f['tokenizer'],tr,tr['transfer_sha256'])==tr


def test_changed_actual_actor_or_teacher_source_is_rejected_before_training(f,tmp_path):
 student=M.make_student_contract(f['master'],101);tr=transfer(f,student,'rloo_cost')
 model=anchor.loaded_model(f['base'],f['sft'],torch.device('cpu'),trainable=True,dtype=torch.float32)
 with torch.no_grad():next(p for p in model.parameters() if p.requires_grad).add_(.1)
 with pytest.raises(ValueError,match='initial model'):
  M.validate_transfer(f['master'],student,'rloo_cost',f['data'],f['targets'],f['calibrations']['rloo_cost'],
   f['tokenizer'],tr,tr['transfer_sha256'],model=model)
 del model;gc.collect()


def test_student_evaluation_projection_changes_no_training_targets_and_can_run_one_engineering_greedy(f):
 validation=f['master']['validation_rows'][::-1][:2]
 student=M.make_student_contract(f['master'],307,validation)
 assert student['validation_rows']==validation and student['training_rows']==f['master']['training_rows']
 tr=transfer(f,student,'rloo_cost');assert tr['original_calibration_sha256']==f['calibrations']['rloo_cost']['calibration_sha256']
 model=anchor.loaded_model(f['base'],f['sft'],torch.device('cpu'),trainable=False,dtype=torch.float32)
 output=f['root']/'engineering_greedy';output.mkdir()
 records=core.collect_greedy_outputs(model,f['tokenizer'],student,output)
 assert len(records)==2 and all(r['generation_calls']==1 and r['sample_index']==0 and
  not r['generation_config']['do_sample'] and r['use_model_defaults'] is False for r in records)
 assert [r['instance_id'] for r in records]==[r['instance_id'] for r in validation]
 del model;gc.collect()


def test_pure_projection_admits_five_seeds_and_exact256_inputs_without_any_policy_call():
 train=json.loads((HERE/'data_v1/op_train.json').read_text());dev=json.loads((HERE/'data_v1/op_validation.json').read_text())
 master=core.make_smoke_contract('op',{'id':'original_SFT_input_only_projection','sha256':'a'*64},train,dev,
  horizon=100,max_completion_tokens=3000,groups_per_update=1,train_seed=73)
 master['scope']='prospective_teacher_development';reseal(master)
 for seed in M.PLANNED_SEEDS:
  student,proof=M.make_confirmation_projection(master,seed)
  assert len(student['validation_rows'])==256 and len(student['training_rows'])==128
  assert student['training']['horizon']==100 and len(student['training']['prompt_draws'])==100
  assert proof['model_calls']==0 and proof['production_registration_ready'] is False
  assert student['training']['fixed_loss_denominator']==3000
 with pytest.raises(ValueError):M.make_confirmation_projection(master,73)


def test_live_alpha_targets_and_student_schedule_drift_denied(f):
 student=M.make_student_contract(f['master'],401);tr=transfer(f,student,'rloo_cost')
 trainer=M.make_cpu_trainer(student,'rloo_cost',f['root']/'guard_only',f['tokenizer'],f['base'],f['sft'],
  master_contract=f['master'],teacher_transfer=tr,expected_teacher_transfer_sha256=tr['transfer_sha256'],**teacher_kwargs(f,'rloo_cost'))
 alpha=trainer.teacher_alpha;trainer.teacher_alpha=alpha*2
 with pytest.raises(ValueError):trainer._assert_contract()
 trainer.teacher_alpha=alpha;trainer.master_teacher_contract['train_seed']=101
 with pytest.raises(ValueError):trainer._assert_contract()
 del trainer;gc.collect()


def external_module(path,*,master_binding=None):
 import types
 raw=M.__file__ and Path(M.__file__).read_bytes();path.write_bytes(raw)
 module=types.ModuleType('external_new_V3_packet');module.__file__=str(path)
 module.__executed_source_sha256__=hashlib.sha256(raw).hexdigest()
 if master_binding is not None:module.__master_teacher_source_binding__=master_binding
 exec(compile(raw,str(path),'exec',dont_inherit=True),module.__dict__)
 return module


def old_master_binding():
 p=Path(T.__file__);raw=p.read_bytes()
 return {'path':str(p.resolve()),'sha256':M.V2_SOURCE_SHA256,'size_bytes':len(raw)}


def test_external_v3_packet_uses_original_master_path_and_never_executes_poison_sibling(tmp_path):
 (tmp_path/'train_teacher_candidate_v2.py').write_text('raise AssertionError("poison sibling executed")\n')
 ext=external_module(tmp_path/'confirmation_teacher_transfer_v3.py',master_binding=old_master_binding())
 assert Path(ext.V2.__file__)==Path(T.__file__)
 assert ext.source_closure()['old_master_source_binding']==old_master_binding()
 assert ext.source_closure()['explicit_old_master_source_binding'] is True
 assert Path(ext.V2.calibration_module().__file__).parent==Path(T.__file__).parent
 assert Path(ext.V2.calibration_module().data_module().__file__).parent==Path(T.__file__).parent
 with pytest.raises(ValueError,match='master source'):
  external_module(tmp_path/'without_master_binding.py')


@pytest.mark.parametrize('fault',['SHA','size_bool','size','relative','wrong_filename'])
def test_explicit_old_master_source_binding_is_typed_and_closed_before_model_work(tmp_path,fault):
 b=old_master_binding()
 if fault=='SHA':b['sha256']='0'*64
 elif fault=='size_bool':b['size_bytes']=True
 elif fault=='size':b['size_bytes']+=1
 elif fault=='relative':b['path']='train_teacher_candidate_v2.py'
 else:b['path']=str(HERE/'calibrate_teacher_v2.py')
 with pytest.raises(ValueError):external_module(tmp_path/'new_V3.py',master_binding=b)


def test_old_master_or_executed_new_source_cannot_be_rebound_after_import(tmp_path):
 original=Path(T.__file__).read_bytes();master=tmp_path/'train_teacher_candidate_v2.py';master.write_bytes(original)
 b={'path':str(master),'sha256':M.V2_SOURCE_SHA256,'size_bytes':len(original)}
 ext=external_module(tmp_path/'new_V3.py',master_binding=b)
 master.write_bytes(original+b'\n')
 with pytest.raises(ValueError,match='V2 numerical teacher'):ext.source_closure()
 master.write_bytes(original)
 Path(ext.__file__).write_bytes(Path(ext.__file__).read_bytes()+b'\n')
 with pytest.raises(ValueError,match='transfer source changed'):ext.source_sha256()


def test_cpu_factory_cannot_promote_resealed_tiny_data_to_production_before_model_load(f,tmp_path,monkeypatch):
 student=M.make_student_contract(f['master'],101);tr=transfer(f,student,'rloo_cost');kw=teacher_kwargs(f,'rloo_cost')
 kw['shared_data']=copy.deepcopy(kw['shared_data']);kw['shared_data']['scope']='production_training_collection_7b'
 reseal(kw['shared_data'],'shared_data_sha256');kw['expected_data_sha256']=kw['shared_data']['shared_data_sha256']
 monkeypatch.setattr(anchor,'loaded_model',lambda *a,**k:pytest.fail('No model loading after kind mismatch'))
 with pytest.raises(ValueError,match='tiny CPU'):
  M.make_cpu_trainer(student,'rloo_cost',tmp_path/'denied',f['tokenizer'],f['base'],f['sft'],
   master_contract=f['master'],teacher_transfer=tr,expected_teacher_transfer_sha256=tr['transfer_sha256'],**kw)


@pytest.mark.parametrize('fault',['teacher_validation_role','calibration_student_relabel','nonplanned_seed'])
def test_master_evidence_role_and_seed_cannot_be_reassigned(f,fault):
 student=M.make_student_contract(f['master'],503);shared=copy.deepcopy(f['data']);cal=copy.deepcopy(f['calibrations']['rloo_cost'])
 if fault=='teacher_validation_role':
  shared['context']['split']='validation';reseal(shared,'shared_data_sha256')
 elif fault=='calibration_student_relabel':
  cal['core_contract_sha256']=student['contract_sha256'];reseal(cal,'calibration_sha256')
 else:
  with pytest.raises(ValueError):M.make_student_contract(f['master'],999)
  return
 with pytest.raises(ValueError):M.make_transfer(f['master'],student,'rloo_cost',shared,f['targets'],cal,f['tokenizer'])
