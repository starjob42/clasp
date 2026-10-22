"""New development evaluator admission; no GPU or real dataset is used."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from test_co_deterministic_launcher import chunk as runtime_chunk  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('foarl_teacher_eval_tested', ROOT/'scripts/evaluate_co_deterministic_dev.py')
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def write(path, value=b'fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    return path


def fixture(root, step=200, method='foarl_teacher'):
    source, train, output = (root/name for name in ('source','train','evaluation'))
    source.mkdir(); train.mkdir(); output.mkdir()
    checkpoint = train/'checkpoints'/f'checkpoint-{step}'
    kl = () if method == 'classic_grpo' else EVAL.KL_NAMES
    files = {name:EVAL.identity(write(checkpoint/name)) for name in (*EVAL.MODEL_NAMES,*EVAL.TOKENIZER_NAMES,*kl)}
    details = write(output/'eval/run_details__fixture.jsonl')
    config_path = write(output/'eval/run_config__fixture.json')
    code_path = write(source/'src/clasp/entrypoints/evaluate.py')
    code_id = EVAL.identity(code_path)
    code_files = [{'path':'entrypoints/evaluate.py','sha256':code_id['sha256'],'size':code_id['size_bytes']}]
    code = {'manifest_sha256':EVAL.digest(code_files),'files':code_files}
    ordered = [{'example_index':i, 'semantic_instance_sha256':str(i),
                'benchmark_row_sha256':'b'+str(i), 'reference_output_sha256':'r'+str(i)} for i in range(100)]
    dataset = {'path':str(EVAL.BENCHMARK/'mis/eval/test.json'),'sha256':'b'*64,'size_bytes':991}
    generation = {**EVAL.GENERATION,'best_of_n':8,'seed_policy':'per_example_sha256_v1',
                  'temperature_schedule':[],'top_p_schedule':[],'op_size_band_top_p_schedule':[],
                  'op_size_band_top_p_min_size':0,'op_size_band_top_p_max_size':0}
    artifact_files = [{'path':name,'sha256':value['sha256'],'size':value['size_bytes']} for name,value in sorted(files.items())]
    launch = {'task':'mis','method':method,'step':step,'output':str(output),
              'training':{'train':str(train),'checkpoint':files},
              'benchmark':{'files':{'mis':{'dataset':dataset}},'ordered_instances':{'mis':ordered}}}
    config = {'stage':'grpo','tasks':['mis'],'split':'eval','run_status':'complete','evaluation_purpose':'development',
              'resolved_model_path':str(checkpoint),'resolved_run_dir':str(train),'data_root':str(EVAL.BENCHMARK),
              'results_dir':str(output/'eval'),'files':{'details':str(details),'run_config':str(config_path)},
              'generation':generation,'code_identity':code,
              'prompting':{'prompt_style_resolved':'jiang_f638','no_reason':True,'clasp_mis_jiang_response_output':'0',
                           'clasp_mis_relabel_order':'','tokenizer_path':str(checkpoint)},
              'limits':{'eval_start_index':0,'max_eval_samples_per_task':100,'prefer_subset_files':False},
              'experiment_provenance':{'generation':{k:v for k,v in generation.items() if k!='best_of_n'},'code':code,
                 'model':{'checkpoint_artifact':{'kind':'local_directory','basename':f'checkpoint-{step}',
                                               'files':artifact_files,'manifest_sha256':EVAL.digest(artifact_files)}},
                 'data':{'files':[{'path':'mis/eval/test.json','sha256':dataset['sha256'],'size':991,'split':'eval','task':'mis'}]}}}
    artifact = SimpleNamespace(artifact_type='native_raw',candidate_budget=8,
        details_snapshot=SimpleNamespace(path=str(details)),companion_snapshot=SimpleNamespace(path=str(config_path)),
        records={i:SimpleNamespace(example_index=i,task='mis',split='eval',candidate_budget=8,
                                  raw={**row,'candidates':[{} for _ in range(8)]}) for i,row in enumerate(ordered)})
    return source, launch, config, artifact


def actual_chunk_command(contract, task, train, diagnostic=False):
    # Evaluate the literal command expression in the independently owned real
    # producer, rather than deriving this fixture from the admission helper.
    tree=ast.parse((ROOT/'scripts/run_co_deterministic_dev_chunk.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.value,ast.List)
              and any(isinstance(t,ast.Name) and t.id=='command' for t in n.targets))
    method=contract['method']
    scope={'RUNTIME':EVAL.RUNTIME,'DATA':EVAL.TRAIN_DATA,'task':task,'sft':Path(contract['initialization']['checkpoint']),
           'output':train,'source':Path(contract['source']['root']),
           'expected_sha':contract['source']['manifest_sha256'],'weight':contract['teacher_weight'],
           'method':method,'diagnostic':diagnostic,'flag_prefix':'classic' if method=='classic_grpo' else 'foarl'}
    # Execute the actual command assignment and its two immediate conditional
    # suffixes, preserving the producer's order for every method and scope.
    additions=sorted([n for n in ast.walk(tree) if isinstance(n,ast.AugAssign)
                     and isinstance(n.target,ast.Name) and n.target.id=='command'],key=lambda n:n.lineno)
    scope['command']=eval(compile(ast.Expression(node.value),'<actual_chunk_command>','eval'),scope)
    if method!='classic_grpo':exec(compile(ast.Module(body=[additions[0]],type_ignores=[]),'<teacher_suffix>','exec'),scope)
    if diagnostic:exec(compile(ast.Module(body=[additions[1]],type_ignores=[]),'<diagnostic_suffix>','exec'),scope)
    return scope['command']


def submission_fixture(root, train, contract, task='mis', method='foarl_teacher',step=200,job='123'):
    exports={'CO_SOURCE':contract['source']['root'],'CO_SOURCE_SHA':contract['source']['manifest_sha256'],
             'CO_TASK':task,'CO_METHOD':method,'CO_OUTPUT':str(train),'CO_STOP':str(step),'CO_DIAGNOSTIC':'0'}
    if step!=100:exports['CO_RESUME']=str(train/'checkpoints'/f'checkpoint-{step-100}')
    stdout=str(root/'logs/chunk-%j.out');stderr=str(root/'logs/chunk-%j.err')
    write(Path(stdout.replace('%j',job)),f'CO_DETERMINISTIC_CHUNK_COMPLETE task={task} method={method} step={step}\n'.encode())
    write(Path(stderr.replace('%j',job)),b'')
    argv=['sbatch','--export=ALL,'+','.join(k+'='+v for k,v in exports.items()),
          '--output='+stdout,'--error='+stderr,str(Path(contract['source']['root'])/'scripts/train_co_deterministic_dev_a100.slurm')]
    return '|'.join((job,'co-deterministic-dev',str(root),shlex.join(argv),stdout,stderr))


class EvaluationAdmissionTest(unittest.TestCase):
    def test_three_methods_have_exact_generic_checkpoint_inventory(self):
        for method in EVAL.METHODS:
            for step in (100,200,400):
                with self.subTest(method=method,step=step), tempfile.TemporaryDirectory() as directory:
                    source,launch,config,artifact=fixture(Path(directory),step,method)
                    EVAL.validate_raw(config,artifact,launch,source)
                    inventory=config['experiment_provenance']['model']['checkpoint_artifact']
                    self.assertEqual(len(inventory['files']),9 if method=='classic_grpo' else 12)
                    if method=='classic_grpo':
                        inventory['files'].append({'path':EVAL.KL_NAMES[0],'sha256':'f'*64,'size':1})
                        inventory['manifest_sha256']=EVAL.digest(inventory['files'])
                        with self.assertRaisesRegex(ValueError,'unknown or duplicate'):
                            EVAL.validate_raw(config,artifact,launch,source)

    def test_raw_100_200_400_checkpoints_accept_all_three_bound_kl_sidecars(self):
        for step in (100,200,400):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as directory:
                source, launch, config, artifact = fixture(Path(directory),step)
                EVAL.validate_raw(config,artifact,launch,source)
                self.assertEqual(len(config['experiment_provenance']['model']['checkpoint_artifact']['files']),12)

    def test_unknown_missing_duplicate_or_changed_kl_file_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            _, launch, config, _ = fixture(Path(directory))
            original=config['experiment_provenance']['model']['checkpoint_artifact']
            for mutation in ('missing','duplicate','root_copy','sha','size','unknown'):
                with self.subTest(mutation=mutation):
                    artifact=copy.deepcopy(original)
                    row=next(x for x in artifact['files'] if x['path']==EVAL.KL_NAMES[0])
                    if mutation=='missing':artifact['files'].remove(row)
                    elif mutation=='duplicate':artifact['files'].append(copy.deepcopy(row))
                    elif mutation=='root_copy':row['path']='../../'+row['path']
                    elif mutation=='sha':row['sha256']='9'*64
                    elif mutation=='size':row['size']+=1
                    else:artifact['files'].append({'path':'extra.safetensors','sha256':'f'*64,'size':1})
                    artifact['manifest_sha256']=EVAL.digest(artifact['files'])
                    with self.assertRaises(ValueError):EVAL.validate_artifact_inventory(artifact,launch['training']['checkpoint'],200)

    def test_wrong_model_data_sampling_prompt_and_code_fail_real_raw_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            source, launch, original, artifact=fixture(Path(directory))
            mutations=[lambda x:x.update(resolved_model_path=str(source)),
                       lambda x:x.update(data_root=str(source)),
                       lambda x:x['generation'].update(num_return_sequences=7),
                       lambda x:x['prompting'].update(no_reason=False),
                       lambda x:x['limits'].update(eval_start_index=1),
                       lambda x:x['experiment_provenance']['data']['files'][0].update(sha256='c'*64)]
            for mutate in mutations:
                config=copy.deepcopy(original);mutate(config)
                with self.assertRaises(ValueError):EVAL.validate_raw(config,artifact,launch,source)
            write(source/'src/clasp/entrypoints/evaluate.py',b'changed scientific code')
            with self.assertRaisesRegex(ValueError,'raw evaluator code differs'):
                EVAL.validate_raw(original,artifact,launch,source)

    def test_missing_instance_wrong_candidate_count_and_row_identity_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            source, launch, config, original=fixture(Path(directory))
            for mutation in ('drop','budget','reference'):
                artifact=copy.deepcopy(original)
                if mutation=='drop':del artifact.records[99]
                elif mutation=='budget':artifact.records[0].raw['candidates'].pop()
                else:artifact.records[0].raw['reference_output_sha256']='wrong'
                with self.assertRaises(ValueError):EVAL.validate_raw(config,artifact,launch,source)

    def test_complete_training_slurm_contract_and_failures(self):
        row='123|COMPLETED|0:0|5000|120|billing=5,cpu=80,gres/gpu=4,mem=300000M,node=1|qos_gpu_a100-dev|vas@a100|gpu_p5'
        self.assertEqual(EVAL.validate_training_allocation(row,'123')['ElapsedRaw'],'5000')
        for bad in ('',row+'\n'+row,row.replace('COMPLETED','FAILED'),row.replace('0:0','1:0'),
                    row.replace('gres/gpu=4','gres/gpu=1'),row.replace('|120|','|121|'),
                    row.replace('cpu=80','cpu=80,cpu=40'),row.replace('123|','124|')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):EVAL.validate_training_allocation(bad,'123')

    def test_only_management_additions_can_differ_between_snapshots(self):
        original={'src/clasp/a.py':'a','scripts/evaluate_a100.slurm':'b','scripts/train.py':'c'}
        EVAL.validate_scientific_equivalence(original,{**original,EVAL.SELF:'d'})
        for changed in ({**original,'src/clasp/a.py':'x'}, {**original,'src/clasp/shadow.py':'x'},
                        {**original,'scripts/evaluate_a100.slurm':'x'}, {**original,'scripts/train.py':'x'},
                        {k:v for k,v in original.items() if k!='scripts/train.py'}):
            with self.assertRaises(ValueError):EVAL.validate_scientific_equivalence(original,changed)

    def test_source_manifest_rejects_untracked_files_symlink_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            required=[EVAL.SELF,EVAL.WRAPPER,'scripts/evaluate_a100.slurm','scripts/clasp_grpo_fresh_dev_artifacts.py',
                      'scripts/compare_development.py','src/clasp/models/foarl_teacher_resume.py','src/clasp/entrypoints/evaluate.py',
                      'src/clasp/models/classic_grpo_resume.py',EVAL.TRAIN_WRAPPER,EVAL.TRAIN_RUNNER,
                      'scripts/run_foarl_teacher_dev_chunk.py']
            for name in required:write(root/name,(ROOT/name).read_bytes())
            lines=[hashlib.sha256((root/name).read_bytes()).hexdigest()+'  '+name for name in required]
            manifest=write(root/'SOURCE_MANIFEST.sha256',('\n'.join(lines)+'\n').encode())
            sha=EVAL.identity(manifest)['sha256']
            EVAL.validate_source(root,sha,management=True)
            extra=write(root/'src/clasp/untracked.py')
            with self.assertRaisesRegex(ValueError,'unmanifested'):EVAL.validate_source(root,sha,management=True)
            extra.unlink();extra.symlink_to(root/'absent')
            with self.assertRaisesRegex(ValueError,'unmanifested'):EVAL.validate_source(root,sha,management=True)
            extra.unlink();write(root/'src/clasp/entrypoints/evaluate.py',b'changed')
            with self.assertRaisesRegex(ValueError,'source byte'):EVAL.validate_source(root,sha,management=True)

    def test_existing_output_and_dangling_links_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'receipt.json'
            EVAL.write_once(path,{'status':'first'})
            with self.assertRaises(FileExistsError):EVAL.write_once(path,{'status':'second'})
            link=Path(directory)/'linked.json';link.symlink_to(Path(directory)/'absent.json')
            with self.assertRaises(FileExistsError):EVAL.write_once(link,{'status':'third'})
            self.assertEqual(EVAL.read_json(path,'receipt_sha256')['status'],'first')

    def test_diagnostic_wrong_task_method_or_horizon_rejected_before_scheduler(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);train=root/'train';train.mkdir()
            contract={'diagnostic':False,'method':'foarl_teacher','seed':43,'fixed_horizon':{'max_steps':400,'world_size':4},
                      'teacher_weight':.05,'initialization':{'checkpoint':str(EVAL.SFT/'runs/mis/seed42/checkpoints/checkpoint-31250'),
                       'files':{'adapter_model.safetensors':{'sha256':EVAL.SFT_HASHES['mis']}}},
                      'data_files':{'mis/train/train.json':{}},'source':{}}
            write(train/'local_development_run_contract.json',contract)
            cases=[('step',None),('diagnostic',True),('method','foarl_control'),('fixed_horizon',{'max_steps':100,'world_size':4}),
                   ('data_files',{'cvrp/train/train.json':{}})]
            for key,value in cases:
                with self.subTest(key=key):
                    changed=copy.deepcopy(contract)
                    if key!='step':changed[key]=value
                    write(train/'local_development_run_contract.json',changed)
                    with patch.object(EVAL,'subprocess') as scheduler:
                        with self.assertRaises(ValueError):EVAL.admit_training(train,'mis','foarl_teacher',5 if key=='step' else 100,'123',root,{})
                        scheduler.check_output.assert_not_called()

    def test_chunk_job_exit_and_current_complete_receipt_are_bound(self):
        # Only the separately tested model/source inspectors are replaced;
        # this executes the real chunk admission and exact receipt comparison.
        with tempfile.TemporaryDirectory() as directory:
            _, launch, _, _ = fixture(Path(directory))
            train=Path(launch['training']['train'])
            contract={'diagnostic':False,'method':'foarl_teacher','seed':43,
                      'fixed_horizon':{'max_steps':400,'world_size':4},'teacher_weight':.05,
                      'initialization':{'checkpoint':str(EVAL.SFT/'runs/mis/seed42/checkpoints/checkpoint-31250'),
                       'files':{'adapter_model.safetensors':{'sha256':EVAL.SFT_HASHES['mis']}}},
                      'data_files':{'mis/train/train.json':{}},
                      'source':{'root':str(ROOT),'manifest_sha256':'a'*64}}
            contract_path=write(train/'local_development_run_contract.json',contract)
            admission={'job_id':'123','stop':200,'contract':EVAL.identity(contract_path)}
            admission.update(command=actual_chunk_command(contract,'mis',train),source_verified_file_count=1,
                             gpu_names=['NVIDIA A100-SXM4-80GB']*4)
            admission_path=write(train/'chunk-200-admission.json',admission)
            exit_path=write(train/'chunk-200-exit.json',{'returncode':0})
            completed=EVAL.write_once(train/'chunk-200-complete.json',
                {'status':'complete_same_contract_resume_inputs_verified','expected_step':200,
                 'files':launch['training']['checkpoint']})
            source_files={'src/clasp/models/foarl_teacher_resume.py':'a'}
            row='123|COMPLETED|0:0|5000|120|billing=5,cpu=80,gres/gpu=4|qos_gpu_a100-dev|vas@a100|gpu_p5'
            binding=submission_fixture(Path(directory),train,contract)
            def runner(command,**kwargs):
                return binding if 'SubmitLine' in command[-1] else row
            inspected={'complete_checkpoint':completed,'deterministic_runtime':{'applied_runtime':{
                'producer_stop':200,'producer_job_id':'123','admission':EVAL.identity(admission_path)}}}
            with patch.object(EVAL,'validate_source',return_value=({'sha256':'a'*64},source_files)), \
                 patch.object(EVAL,'admit_checkpoint',return_value=inspected) as inspector:
                result=EVAL.admit_training(train,'mis','foarl_teacher',200,'123',ROOT,source_files,runner=runner)
                inspector.assert_called_once_with(train,200,contract,ROOT)
                self.assertEqual(set(result['checkpoint']),set(EVAL.MODEL_NAMES+EVAL.TOKENIZER_NAMES+EVAL.KL_NAMES))
                for path,value in ((admission_path,{**admission,'job_id':'124'}),
                                   (admission_path,{**admission,'stop':100}),
                                   (exit_path,{'returncode':1})):
                    prior=path.read_bytes();write(path,value)
                    with self.assertRaises(ValueError):
                        EVAL.admit_training(train,'mis','foarl_teacher',200,'123',ROOT,source_files,runner=runner)
                    write(path,prior)
                inspector.return_value={**inspected,'complete_checkpoint':{**completed,'expected_step':100}}
                with self.assertRaisesRegex(ValueError,'retained chunk completion'):
                    EVAL.admit_training(train,'mis','foarl_teacher',200,'123',ROOT,source_files,runner=runner)

    def test_actual_submission_and_log_aliases_bind_source_arm_step_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);train=root/'train'
            contract={'source':{'root':str(ROOT),'manifest_sha256':'a'*64}}
            raw=submission_fixture(root,train,contract)
            result=EVAL.validate_training_job_binding(raw,'123',train,'mis','foarl_teacher',200,contract)
            self.assertTrue(result['completed_marker_verified'])
            parts=raw.split('|');parts[-2]=parts[-2].replace('%j','123');parts[-1]=parts[-1].replace('%j','123')
            self.assertEqual(EVAL.validate_training_job_binding('|'.join(parts),'123',train,'mis','foarl_teacher',200,contract),result)
            for bad in (raw.replace('CO_TASK=mis','CO_TASK=cvrp'),raw.replace('CO_METHOD=foarl_teacher','CO_METHOD=foarl_control'),
                        raw.replace('CO_STOP=200','CO_STOP=100'),raw.replace('CO_DIAGNOSTIC=0','CO_DIAGNOSTIC=1'),
                        raw.replace('CO_SOURCE_SHA='+'a'*64,'CO_SOURCE_SHA='+'b'*64),
                        raw.replace('checkpoint-100','checkpoint-99'),raw.replace('train_co_deterministic_dev_a100.slurm','other.slurm'),
                        raw.replace('CO_STOP=200','CO_STOP=200,CO_STOP=200'),raw+'\n'+raw):
                with self.assertRaises(ValueError):EVAL.validate_training_job_binding(bad,'123',train,'mis','foarl_teacher',200,contract)
            write(Path(result['stdout']['path']),b'CO_DETERMINISTIC_CHUNK_COMPLETE task=mis method=foarl_teacher step=100\n')
            with self.assertRaisesRegex(ValueError,'completion marker'):
                EVAL.validate_training_job_binding(raw,'123',train,'mis','foarl_teacher',200,contract)

    def test_actual_frozen_chunk_argv_and_source_gpu_binding(self):
        contract={'source':{'root':str(ROOT),'manifest_sha256':'a'*64},'teacher_weight':.05,
                  'method':'foarl_teacher','diagnostic':False,
                  'initialization':{'checkpoint':str(EVAL.SFT/'runs/mis/seed42/checkpoints/checkpoint-31250')}}
        train=Path('/synthetic/train')
        original={'command':actual_chunk_command(contract,'mis',train),'stop':200,
                  'contract':{'path':str(train/'local_development_run_contract.json')},
                  'source_verified_file_count':1,'gpu_names':['NVIDIA A100-SXM4-80GB']*4}
        EVAL.validate_chunk_command(original,contract,'mis','foarl_teacher',200,{'a':'b'})
        for kind in ('command','count','gpu'):
            bad=copy.deepcopy(original)
            if kind=='command':bad['command'][bad['command'].index('--tasks')+1]='cvrp'
            elif kind=='count':bad['source_verified_file_count']=2
            else:bad['gpu_names'][0]='NVIDIA H100'
            with self.assertRaises(ValueError):EVAL.validate_chunk_command(bad,contract,'mis','foarl_teacher',200,{'a':'b'})

    def test_all_methods_diagnostic_and_performance_commands_match_actual_producer(self):
        for method in EVAL.METHODS:
            for diagnostic in (False,True):
                with self.subTest(method=method,diagnostic=diagnostic):
                    contract={'source':{'root':str(ROOT),'manifest_sha256':'a'*64},'method':method,
                              'diagnostic':diagnostic,'teacher_weight':.05 if method=='foarl_teacher' else 0.,
                              'initialization':{'checkpoint':str(EVAL.SFT/'runs/mis/seed42/checkpoints/checkpoint-31250')}}
                    train=Path('/synthetic/train');step=10 if diagnostic else 400
                    admission={'command':actual_chunk_command(contract,'mis',train,diagnostic),'stop':step,
                               'contract':{'path':str(train/'local_development_run_contract.json')},
                               'source_verified_file_count':1,'gpu_names':['NVIDIA A100']*4}
                    EVAL.validate_chunk_command(admission,contract,'mis',method,step,{'a':'b'},diagnostic=diagnostic)
                    changed=copy.deepcopy(admission)
                    changed['command'][changed['command'].index('--co_method')+1]='wrong_method'
                    with self.assertRaises(ValueError):
                        EVAL.validate_chunk_command(changed,contract,'mis',method,step,{'a':'b'},diagnostic=diagnostic)

    def test_actual_shell_passes_exact_dev_generation_and_scrubs_inherited_knobs(self):
        # The sole launcher substitution is a temporary interpreter location;
        # the shell flow, argv, and environment statements execute as written.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir();train=root/'train';train.mkdir()
            runtime=root/'runtime';(runtime/'bin').mkdir(parents=True)
            (runtime/'bin/python').symlink_to(sys.executable)
            helper=source/EVAL.SELF
            write(helper,('import json,os,sys\nfrom pathlib import Path\n'
                          'Path(os.environ["CAPTURE"]+"_"+sys.argv[1]+".json").write_text(json.dumps(sys.argv[1:]))\n').encode())
            generic=source/'scripts/evaluate_a100.slurm'
            selected=['CHECKPOINT_PATH','DATA_ROOT','TASK','SEED','DTYPE','NUM_RETURN_SEQUENCES','GENERATION_BATCH_SIZE',
                      'MAX_NEW_TOKENS','MAX_PROMPT_LENGTH','TEMPERATURE','TOP_P','TOP_K','EVALUATION_PURPOSE',
                      'CLASP_REPAIR_ENABLED','GRPO_FAKE','RESUME_FROM_CHECKPOINT','PYTHONHOME']
            code='import json,os;open(os.environ["CAPTURE"]+"_generic.json","w").write(json.dumps({k:os.environ.get(k) for k in '+repr(selected)+'}))'
            write(generic,('#!/bin/bash\n'+shlex.quote(sys.executable)+' -c '+shlex.quote(code)+'\n').encode())
            wrapper=root/'executed_slurm'
            text=(ROOT/EVAL.WRAPPER).read_text().replace('RUNTIME=/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env','RUNTIME='+str(runtime))
            write(wrapper,text.encode())
            env={**os.environ,'CO_EVAL_SOURCE':str(source),'CO_EVAL_SOURCE_SHA':'a'*64,'CO_TRAIN_ROOT':str(train),
                 'CO_TRAIN_JOB_ID':'123','CO_TASK':'mis','CO_METHOD':'foarl_teacher','CO_STEP':'200',
                 'CO_EVAL_OUTPUT':str(root/'output'),'CAPTURE':str(root/'capture'),
                 'GRPO_FAKE':'wrong','CLASP_REPAIR_ENABLED':'1','RESUME_FROM_CHECKPOINT':'wrong','PYTHONHOME':'wrong'}
            subprocess.run(['bash',str(wrapper)],env=env,capture_output=True,text=True,check=True)
            captured=json.loads((root/'capture_generic.json').read_text())
            self.assertEqual(captured['DATA_ROOT'],str(EVAL.BENCHMARK))
            self.assertEqual(captured['CHECKPOINT_PATH'],str(train/'checkpoints/checkpoint-200'))
            self.assertEqual({k:captured[k] for k in ('SEED','DTYPE','NUM_RETURN_SEQUENCES','GENERATION_BATCH_SIZE','MAX_NEW_TOKENS','MAX_PROMPT_LENGTH')},
                             {'SEED':'271828','DTYPE':'float16','NUM_RETURN_SEQUENCES':'8','GENERATION_BATCH_SIZE':'8','MAX_NEW_TOKENS':'3000','MAX_PROMPT_LENGTH':'29768'})
            self.assertTrue(all(captured[k] is None for k in ('CLASP_REPAIR_ENABLED','GRPO_FAKE','RESUME_FROM_CHECKPOINT','PYTHONHOME')))
            launch=json.loads((root/'capture_launch.json').read_text())
            self.assertEqual(launch[launch.index('--training-job-id')+1],'123')
            self.assertEqual(launch[launch.index('--running-wrapper')+1],str(wrapper))


@pytest.mark.parametrize('method',EVAL.METHODS)
def test_intermediate_checkpoint_uses_original_training_runtime_and_four_rank_evidence(request,monkeypatch,tmp_path,method):
    setup=request.getfixturevalue('runtime_chunk')
    runtime=setup.module
    os.environ.update(CO_METHOD=method,CO_DIAGNOSTIC='1',CO_STOP='10')
    runtime.main()
    contract=runtime.read_json(setup.output/'local_development_run_contract.json')
    management=tmp_path/'management';management.mkdir()
    write(management/'SOURCE_MANIFEST.sha256',b'synthetic management manifest\n')
    files={name:runtime.identity(setup.source/name)['sha256'] for name in runtime.IMPLEMENTATIONS}
    shared_relative='scripts/run_foarl_teacher_dev_chunk.py'
    write(setup.source/shared_relative,(ROOT/shared_relative).read_bytes())
    files[shared_relative]=runtime.identity(setup.source/shared_relative)['sha256']
    monkeypatch.setattr(EVAL,'validate_source',lambda *a,**kw:({'sha256':'a'*64},files))
    from clasp.models import classic_grpo_resume, foarl_teacher_resume
    selected=[]
    def verifier(name,relative,*args):
        selected.append(name)
        return classic_grpo_resume if method=='classic_grpo' else foarl_teacher_resume
    monkeypatch.setattr(EVAL,'_verified_module',verifier)
    load=EVAL.module;paths=[]
    def original_runtime(path,name):
        paths.append(path)
        obj=load(path,name)
        # Only the native specialist directory boundary is substituted. All
        # runtime proof/schema/source/declaration/four-rank readers are real.
        obj.SFT=runtime.SFT;obj.SFT_HASHES=runtime.SFT_HASHES
        return obj
    monkeypatch.setattr(EVAL,'module',original_runtime)
    # A cached dependency outside both admitted roots must be refused before
    # it can supply the original runner's identity/write primitives.
    with pytest.raises(ValueError,match='shared runtime dependency loaded from another source'):
        EVAL.admit_checkpoint(setup.output,5,contract,management)
    monkeypatch.delitem(sys.modules,'run_foarl_teacher_dev_chunk')
    selected.clear()
    result=EVAL.admit_checkpoint(setup.output,5,contract,management)
    assert paths==[setup.source/shared_relative,setup.source/EVAL.TRAIN_RUNNER]
    assert selected==['clasp.models.'+('classic_grpo_resume' if method=='classic_grpo' else 'foarl_teacher_resume')]
    assert result['complete_checkpoint']['step']==5
    assert result['deterministic_runtime']['applied_runtime']['producer_stop']==10
    assert len(result['deterministic_runtime']['applied_runtime']['postrun_per_rank_checks'])==4
    assert not (setup.output/'chunk-5-complete.json').exists()
    (setup.output/'hyperparams.json').write_text('{"later":true}')
    assert EVAL.admit_checkpoint(setup.output,5,contract,management)==result
    (setup.output/'postrun_determinism_step10_rank3.json').unlink()
    with pytest.raises(ValueError,match='regular unlinked file'):
        EVAL.admit_checkpoint(setup.output,5,contract,management)


if __name__=='__main__':
    unittest.main()
