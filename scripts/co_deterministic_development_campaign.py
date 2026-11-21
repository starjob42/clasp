"""Declared six-arm deterministic combinatorial RL development: bounded ticks, one attempt per job.

The original training snapshot and 400-update horizon remain fixed across
chunks. Management/evaluation files may live in a later identical-science
snapshot. This controller never retries, cancels, or starts a background watch.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
SELF = 'scripts/co_deterministic_development_campaign.py'
EVALUATOR = 'scripts/evaluate_co_deterministic_dev.py'
PROTOCOL = 'co-deterministic-six-arm-development-campaign-v1'
PROJECT = Path('/lustre/fsn1/projects/rech/vas/uag72kz')
TRAINING_SOURCE = PROJECT/'co_deterministic_source_20260909_v1/source-snapshot'
TRAINING_SHA = '8fa688e8421f2ef3deda29456ed7b6991a65ab34de492896619df1050d9a81f6'
DIAGNOSTICS = PROJECT/'co_deterministic_diagnostics_20260909_v1'
OUTPUT = PROJECT/'co_deterministic_development400_20260909_v1'
# Actual native declaration created after the strict runtime CPU preflight.
DIAGNOSTIC_PROTOCOL_SHA = '7e452753d15d83043b116726297fc9885b0fb00604aca02de65e3deb8d1e76cc'
DIAGNOSTIC_CASES = {
    'mis_foarl_teacher_full': ('mis','foarl_teacher',10),
    'mis_foarl_teacher_staged': ('mis','foarl_teacher',5),
    'cvrp_foarl_teacher_check': ('cvrp','foarl_teacher',5),
    'mis_classic_grpo_full': ('mis','classic_grpo',10),
    'mis_classic_grpo_staged': ('mis','classic_grpo',5),
    'mis_foarl_control_full': ('mis','foarl_control',10),
}
RESUME_METHODS = ('foarl_teacher','classic_grpo')
TASKS = ('cvrp','mis')
METHODS = ('classic_grpo','foarl_control','foarl_teacher')
STEPS = (100,200,300,400)
EVAL_STEPS = (100,200,400)
TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
JOB = re.compile(r'[1-9][0-9]*')
QUEUE_JOB = re.compile(r'[1-9][0-9]*(?:_[0-9]+)?')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap_identity(path):
    """Hash an input before allowing it to select any imported implementation."""
    path=Path(path).absolute()
    require(path.is_file() and not path.is_symlink(),'bootstrap input is missing or linked')
    before=path.stat();digest=hashlib.sha256()
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(descriptor,'rb') as stream:
        opened=os.fstat(stream.fileno())
        require((before.st_dev,before.st_ino)==(opened.st_dev,opened.st_ino),'bootstrap input changed before open')
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
        after=os.fstat(stream.fileno())
    require((before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns),
            'bootstrap input changed while hashing')
    return {'path':str(path),'sha256':digest.hexdigest(),'size_bytes':before.st_size}


def bootstrap_source(source,sha):
    manifest=source/'SOURCE_MANIFEST.sha256'
    require(re.fullmatch('[a-f0-9]{64}',sha) and bootstrap_identity(manifest)['sha256']==sha,'bootstrap source manifest mismatch')
    entries={}
    for line in manifest.read_text().splitlines():
        match=re.fullmatch(r'([a-f0-9]{64})  (.+)',line)
        require(match is not None,'malformed bootstrap source manifest')
        digest,name=match.groups();relative=Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and relative.as_posix()==name and name not in entries,
                'unsafe/repeated bootstrap source entry')
        entries[name]=digest
    for name in (SELF,EVALUATOR):
        path=source/name
        require(name in entries and path.resolve(strict=True)==path and bootstrap_identity(path)['sha256']==entries[name],
                'bootstrap executable bytes mismatch')
    require(bootstrap_identity(Path(__file__).resolve())['sha256']==entries[SELF],'executing campaign bootstrap is not manifested')


def read_command(argv):
    return subprocess.check_output(argv,text=True,timeout=30)


def submit_command(argv):
    return subprocess.run(argv,text=True,capture_output=True,timeout=60)


def evaluation_module(source):
    return load_module(source/EVALUATOR,'_co_deterministic_campaign_evaluation')


def check_source(source, sha):
    source = Path(source).resolve(strict=True)
    bootstrap_source(source,sha)
    helper = evaluation_module(source)
    manifest, files = helper.validate_source(source,sha,management=True)
    require(helper.identity(Path(__file__).resolve())['sha256'] == files.get(SELF), 'executing campaign controller is not manifested')
    training_manifest, training_files = helper.validate_source(TRAINING_SOURCE,TRAINING_SHA)
    helper.validate_scientific_equivalence(training_files,files)
    require(all(files.get(name)==digest for name,digest in training_files.items()),
            'management changed an original frozen training snapshot input')
    for name in ('scripts/run_co_deterministic_dev_chunk.py','scripts/train_co_deterministic_dev_a100.slurm',
                 'scripts/compare_co_deterministic_state.py','scripts/co_deterministic_diagnostics.py'):
        require(name in training_files and files.get(name) == training_files[name], 'management changed the frozen training/diagnostic producer')
    require((source/'logs').is_dir() and not (source/'logs').is_symlink(), 'pre-provision the management source logs directory before sealing')
    return helper, manifest, training_manifest, training_files


def export(values):
    require(all(isinstance(v,str) and ',' not in v and not any(c.isspace() for c in v) for v in values.values()), 'unsupported export value')
    return '--export=ALL,'+','.join(k+'='+v for k,v in values.items())


def key(role,task,method,step):
    return f'{role}_{task}_{method}_s{step}'


def plans(root, source, sha):
    result = []
    for step in STEPS:
        for task in TASKS:
            for method in METHODS:
                train = root/'runs'/f'{task}_{method}'
                name = key('train',task,method,step)
                values = {'CO_SOURCE':str(TRAINING_SOURCE),'CO_SOURCE_SHA':TRAINING_SHA,
                          'CO_TASK':task,'CO_METHOD':method,'CO_OUTPUT':str(train),'CO_STOP':str(step),
                          'CO_DIAGNOSTIC':'0','CO_RESUME':str(train/'checkpoints'/f'checkpoint-{step-100}') if step>100 else '',
                          'CO_COMPARE_WITH':''}
                command = ['sbatch','--parsable','--no-requeue','--time=02:00:00','--partition=gpu_p5',
                           '--job-name=codet-'+name,'--output='+str(root/'logs'/f'{name}-%j.out'),
                           '--error='+str(root/'logs'/f'{name}-%j.err'),export(values),
                           str(TRAINING_SOURCE/'scripts/train_co_deterministic_dev_a100.slurm')]
                result.append({'key':name,'role':'train','task':task,'method':method,'step':step,
                               'train':str(train),'output':str(train),
                               'depends_on':key('train',task,method,step-100) if step>100 else None,
                               'command':command})
        if step not in EVAL_STEPS:
            continue
        for task in TASKS:
            for method in METHODS:
                train = root/'runs'/f'{task}_{method}'
                name = key('eval',task,method,step)
                dependency = key('train',task,method,step)
                output = root/'evaluations'/f'{task}_{method}'/f'checkpoint-{step}'
                values = {'CO_EVAL_SOURCE':str(source),'CO_EVAL_SOURCE_SHA':sha,
                          'CO_TRAIN_ROOT':str(train),'CO_TRAIN_JOB_ID':'@'+dependency,
                          'CO_TASK':task,'CO_METHOD':method,'CO_STEP':str(step),'CO_EVAL_OUTPUT':str(output)}
                command = ['sbatch','--parsable','--no-requeue','--time=02:00:00','--partition=gpu_p5',
                           '--job-name=codet-'+name,'--output='+str(root/'logs'/f'{name}-%j.out'),
                           '--error='+str(root/'logs'/f'{name}-%j.err'),export(values),str(source/'scripts/evaluate_co_deterministic_dev_a100.slurm')]
                result.append({'key':name,'role':'eval','task':task,'method':method,'step':step,
                               'train':str(train),'output':str(output),'depends_on':dependency,'command':command})
    return result


def accepted_job(receipt):
    require(type(receipt.get('returncode')) is int and receipt['returncode'] == 0, 'submission failed/ambiguous; no retry allowed')
    match = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?\s*',receipt.get('stdout',''))
    require(match is not None,'submission response is ambiguous; no retry allowed')
    return match[1]


def diagnostic_declaration(helper):
    protocol_path = DIAGNOSTICS/'protocol.json'
    require(re.fullmatch('[a-f0-9]{64}',DIAGNOSTIC_PROTOCOL_SHA or '')
            and helper.identity(protocol_path)['sha256']==DIAGNOSTIC_PROTOCOL_SHA,
            'actual native diagnostic protocol must be externally pinned')
    declaration = helper.read_json(protocol_path)
    diagnostic = load_module(TRAINING_SOURCE/'scripts/co_deterministic_diagnostics.py','_frozen_co_deterministic_diagnostics')
    diagnostic.validate_protocol(declaration,TRAINING_SOURCE,TRAINING_SHA,DIAGNOSTICS)
    require(tuple((name,*values) for name,values in DIAGNOSTIC_CASES.items())==diagnostic.CASES
            and diagnostic.RESUME_METHODS==RESUME_METHODS,'diagnostic case inventory differs')
    expected=[];receipts={};jobs=set()
    for name,(task,method,stop) in DIAGNOSTIC_CASES.items():
        command=diagnostic.command(TRAINING_SOURCE,TRAINING_SHA,DIAGNOSTICS,name,task,method,stop)
        expected.append({'key':name,'command':command})
        attempt,submission=DIAGNOSTICS/f'{name}-attempt.json',DIAGNOSTICS/f'{name}-submission.json'
        require(helper.read_json(attempt)['command']==command,'initial diagnostic attempt differs from the fixed gate')
        job=accepted_job(helper.read_json(submission))
        require(job not in jobs,'initial diagnostic jobs must be distinct');jobs.add(job)
        receipts[name]={'attempt':helper.identity(attempt),'submission':helper.identity(submission),'job_id':job}
    require(declaration['plans']==expected,'diagnostic six-arm plan changed')
    resumes={}
    for method in RESUME_METHODS:
        name=f'mis_{method}_staged'
        resumes[method]=diagnostic.command(TRAINING_SOURCE,TRAINING_SHA,DIAGNOSTICS,name,'mis',method,10,
                                          DIAGNOSTICS/name/'checkpoints/checkpoint-5')
    require(declaration['resume_plans']==[{'key':f'mis_{method}_staged-resume10','method':method,'command':resumes[method]}
                                        for method in RESUME_METHODS],'diagnostic continuation plans changed')
    return {'protocol':helper.identity(protocol_path),'initial':receipts,'resume_commands':resumes}


def initialize(args):
    root, source = args.output.absolute(),args.management_source.resolve(strict=True)
    require(root == OUTPUT and root.resolve() == root, 'campaign must use a fresh native project directory')
    helper, manifest, training_manifest, _ = check_source(source,args.management_sha)
    require(not root.exists() and not root.is_symlink(),'campaign already exists')
    benchmark_helper = load_module(source/'scripts/clasp_grpo_fresh_dev_artifacts.py','_co_deterministic_campaign_benchmark')
    benchmark = benchmark_helper.validate_benchmark(helper.BENCHMARK)
    require(benchmark['manifest_sha256'] == helper.BENCHMARK_SHA and benchmark['audit_sha256'] == helper.OVERLAP_SHA,'wrong development benchmark')
    initialization = {}; data_files = {}
    for task in TASKS:
        sft = helper.SFT/'runs'/task/'seed42/checkpoints/checkpoint-31250'
        initialization[task] = {'checkpoint':str(sft),'files':{name:helper.identity(sft/name) for name in helper.MODEL_NAMES}}
        require(initialization[task]['files']['adapter_model.safetensors']['sha256'] == helper.SFT_HASHES[task],'wrong task SFT')
        data_files[task] = {str(path.relative_to(helper.TRAIN_DATA)):helper.identity(path)
                           for path in sorted((helper.TRAIN_DATA/task/'train').glob('*.json'))}
        require(data_files[task],'missing training task data')
    declaration = {'protocol':PROTOCOL,'created_at_utc':now(),'output':str(root),'training_source_manifest':training_manifest,
                   'management_source_manifest':manifest,'controller':helper.identity(source/SELF),
                   'diagnostics':diagnostic_declaration(helper),'initialization':initialization,
                   'data_root':str(helper.TRAIN_DATA),'data_files':data_files,'benchmark':benchmark,
                   'generation':helper.GENERATION,'plans':plans(root,source,args.management_sha),
                   'recipe':{'seed':43,'final_horizon':400,'world_size':4,'global_rollout_batch':36,'gradient_accumulation_steps':3,
                             'teacher_weight':{'classic_grpo':0.,'foarl_control':0.,'foarl_teacher':.05},'beta':{'classic_grpo':0.,'foarl_control':.05,'foarl_teacher':.05},'deterministic_runtime_profile':'co-strict-deterministic-development-20260909-v1','training_stops':list(STEPS),
                             'evaluation_stops':list(EVAL_STEPS),'walltime':'02:00:00','max_active_development_jobs':10},
                   'scope':'reused_development100_single_seed_not_confirmation_or_sota','repair':False,
                   'selection_policy':'all_declared_endpoints_no_outcome_based_recipe_or_horizon_changes',
                   'budget_note':'matched_updates_and_raw_samples; actual tokens/time and teacher overhead must be reported separately',
                   'no_retries':True,'no_cancellation':True,'no_background_watch':True}
    root.mkdir(exist_ok=False)
    (root/'logs').mkdir();(root/'runs').mkdir();(root/'evaluations').mkdir();(root/'terminals').mkdir()
    (root/'.submission.lock').touch(exist_ok=False);(root/'.submission.lock').chmod(0o440)
    for task in TASKS:
        for method in METHODS:(root/'evaluations'/f'{task}_{method}').mkdir()
    return helper.write_once(root/'protocol.json',declaration,'protocol_sha256')


def load_protocol(root, expected_sha):
    root = Path(root).absolute()
    require(re.fullmatch('[a-f0-9]{64}',expected_sha) and bootstrap_identity(root/'protocol.json')['sha256']==expected_sha,
            'campaign declaration file changed before source selection')
    raw = json.loads((root/'protocol.json').read_bytes())
    source = Path(raw['management_source_manifest']['path']).parent
    helper, manifest, training_manifest, training_files = check_source(source,raw['management_source_manifest']['sha256'])
    require(helper.identity(root/'protocol.json')['sha256'] == expected_sha,'campaign declaration file changed')
    protocol = helper.read_json(root/'protocol.json','protocol_sha256')
    require(protocol['protocol'] == PROTOCOL and protocol['output'] == str(root)
            and protocol['management_source_manifest'] == manifest and protocol['training_source_manifest'] == training_manifest,
            'wrong campaign/source')
    require(protocol['plans'] == plans(root,source,manifest['sha256']) and protocol['controller'] == helper.identity(source/SELF),
            'campaign plans/controller changed')
    require(protocol['diagnostics'] == diagnostic_declaration(helper),'diagnostic declaration changed')
    return helper,protocol,source,training_files


def expected_contract(protocol,task,method,diagnostic=False):
    require(task in TASKS and method in METHODS,'unsupported task or method')
    result={'schema_version':1,'stage':'grpo','status':'development_only_not_publication_eligible','method':method,
            'source':{'root':str(TRAINING_SOURCE),'manifest_sha256':TRAINING_SHA},
            'initialization':protocol['initialization'][task],'fixed_horizon':{'max_steps':400,'world_size':4},
            'teacher_weight':.05 if method=='foarl_teacher' else 0.,'seed':43,
            'data_root':protocol['data_root'],'data_files':protocol['data_files'][task],
            'schedule':{'generation_batch_size':36,'gradient_accumulation_steps':3,'steps_per_generation':3,'save_steps':5 if diagnostic else 100},
            'diagnostic':diagnostic}
    if method=='classic_grpo':result['beta']=0.
    return result


def accounting(job, gpus, runner):
    fields = ('JobIDRaw','State','ExitCode','ElapsedRaw','TimelimitRaw','AllocTRES','QOS','Account','Partition')
    raw = runner(['sacct','-X','-P','-n','-j',job,'--format='+','.join(fields)])
    lines = [line.split('|') for line in raw.splitlines() if line.strip()]
    require(len(lines)==1 and len(lines[0])==len(fields),'missing/ambiguous terminal accounting')
    row = dict(zip(fields,lines[0],strict=True))
    require(row['JobIDRaw']==job and row['State']=='COMPLETED' and row['ExitCode']=='0:0','job did not complete successfully')
    entries = [part.split('=',1) for part in row['AllocTRES'].split(',')]
    require(all(len(part)==2 for part in entries) and len({p[0] for p in entries})==len(entries),'invalid allocation TRES')
    tres = dict(entries)
    require(tres.get('gres/gpu')==str(gpus) and row['TimelimitRaw']=='120' and row['QOS']=='qos_gpu_a100-dev'
            and row['Account']=='vas@a100' and row['Partition']=='gpu_p5' and int(row['ElapsedRaw'])>0,'wrong development resources')
    elapsed=int(row['ElapsedRaw'])
    return {'slurm':row,'elapsed_seconds':elapsed,'allocated_gpu_hours':gpus*elapsed/3600,
            'allocated_cpu_hours':int(tres['cpu'])*elapsed/3600,
            'billing_unit_hours':float(tres['billing'])*elapsed/3600}


def submission_binding(helper, argv, job, runner, marker=None):
    fields=('JobIDRaw','JobName','WorkDir','SubmitLine','StdOut','StdErr')
    raw=runner(['sacct','-X','-P','-n','-j',job,'--format=JobIDRaw,JobName%100,WorkDir%1000,SubmitLine%10000,StdOut%1000,StdErr%1000'])
    rows=[line.split('|') for line in raw.splitlines() if line.strip()]
    require(len(rows)==1 and len(rows[0])==len(fields),'missing/ambiguous actual submission binding')
    row=dict(zip(fields,rows[0],strict=True));observed=shlex.split(row['SubmitLine'])
    require(row['JobIDRaw']==job and observed and Path(observed[0]).name=='sbatch'
            and ['sbatch',*observed[1:]]==argv,'actual submitted command differs from the declared attempt')
    options={word.split('=',1)[0]:word.split('=',1)[1] for word in argv[1:-1] if '=' in word}
    require(row['JobName']==options['--job-name'],'actual job label differs')
    result={'job_id':job,'submit_line_sha256':hashlib.sha256(row['SubmitLine'].encode()).hexdigest()}
    for field,option in (('StdOut','--output'),('StdErr','--error')):
        literal=options[option];expanded=literal.replace('%j',job).replace('%x',row['JobName'])
        require(Path(literal).is_absolute() and '%' not in expanded and row[field] in (literal,expanded),'unexpected actual log alias')
        result[field]=helper.identity(expanded)
    if marker:
        with Path(result['StdOut']['path']).open() as stream:
            found=[line.rstrip('\r\n') for line in stream if line.startswith('CO_DETERMINISTIC_CHUNK_COMPLETE ')]
        require(found==[marker] and helper.identity(result['StdOut']['path'])==result['StdOut'],'actual chunk completion marker differs')
    return result


def validate_chunk(helper,protocol,task,method,step,train,job,argv,training_files,runner,diagnostic=False):
    train=Path(train);contract_path=train/'local_development_run_contract.json'
    contract=helper.read_json(contract_path)
    require(contract==expected_contract(protocol,task,method,diagnostic),'actual chunk contract differs from the complete predeclared recipe/data/SFT/source')
    admission_path=train/f'chunk-{step}-admission.json';exit_path=train/f'chunk-{step}-exit.json'
    admission=helper.read_json(admission_path);exit_receipt=helper.read_json(exit_path)
    require(admission['job_id']==job and type(admission['stop']) is int and admission['stop']==step
            and admission['contract']==helper.identity(contract_path)
            and type(exit_receipt['returncode']) is int and exit_receipt['returncode']==0,'chunk admission/exit job or step mismatch')
    helper.validate_chunk_command(admission,contract,task,method,step,training_files,diagnostic=diagnostic)
    allocation=accounting(job,4,runner)
    binding=submission_binding(helper,argv,job,runner,f'CO_DETERMINISTIC_CHUNK_COMPLETE task={task} method={method} step={step}')
    source=Path(protocol['management_source_manifest']['path']).parent
    actual=helper.admit_checkpoint(train,step,contract,source)
    complete_path=train/f'chunk-{step}-complete.json'
    require(actual['complete_checkpoint']==helper.read_json(complete_path,'receipt_sha256'),
            'current checkpoint differs from terminal chunk receipt')
    runtime=actual['deterministic_runtime']['applied_runtime']
    require(runtime['producer_stop']==step and runtime['producer_job_id']==job,
            'endpoint deterministic execution proof belongs to another training job')
    return {'job_id':job,'task':task,'method':method,'step':step,**actual,'allocation':allocation,
            'actual_submission':binding,'admission':helper.identity(admission_path),'exit':helper.identity(exit_path),
            'complete':helper.identity(complete_path)}


def validate_resume_equivalence(helper,receipt,full,staged,*,step=10,method='foarl_teacher'):
    require(type(step) is int and step in (5,10) and method in RESUME_METHODS,'unsupported semantic comparison')
    require(receipt['protocol']=='co-deterministic-semantic-state-comparison-v1'
            and receipt['status']==('passed_exact_step5_prerestart' if step==5 else 'passed_exact_step10_continuation')
            and type(receipt['expected_step']) is int and receipt['expected_step']==step and receipt['method']==method
            and receipt['scope']==('independent_prefix_readiness_not_resume_evidence' if step==5 else 'training_continuation_readiness_not_performance_evidence')
            and type(receipt['compared_leaf_count']) is int and receipt['compared_leaf_count']>0
            and type(receipt['differing_field_count']) is int and receipt['differing_field_count']==0
            and receipt['differing_fields']==[] and receipt['parameter_rng_values_exported'] is False
            and receipt['raw_rollout_or_next_prompt_trace_exported'] is False
            and receipt['live_loaded_optimizer_scheduler_rng_restoration_verified'] is False,
            'native exact semantic optimizer/scheduler/RNG comparison did not pass')
    require(receipt['checkpoints']=={'uninterrupted':full['complete_checkpoint'],'resumed':staged['complete_checkpoint']}
            and receipt['deterministic_runtime']=={'uninterrupted':full['deterministic_runtime'],'resumed':staged['deterministic_runtime']},
            'semantic comparison does not bind both current checkpoints and actual applied runtime proofs')


def diagnostic_gate(helper,protocol,training_files,runner):
    proof=protocol['diagnostics'];results={};jobs=set()
    for name,(task,method,step) in DIAGNOSTIC_CASES.items():
        record=proof['initial'][name];job=record['job_id']
        require(job not in jobs,'one job cannot satisfy two diagnostic cases');jobs.add(job)
        argv=helper.read_json(record['attempt']['path'])['command']
        results[name]=validate_chunk(helper,protocol,task,method,step,DIAGNOSTICS/name,job,argv,training_files,runner,True)
    source=Path(protocol['management_source_manifest']['path']).parent
    resumes={}
    for method in RESUME_METHODS:
        full_name=f'mis_{method}_full';staged_name=f'mis_{method}_staged'
        contract=expected_contract(protocol,'mis',method,True)
        full5=helper.admit_checkpoint(DIAGNOSTICS/full_name,5,contract,source)
        staged5=helper.admit_checkpoint(DIAGNOSTICS/staged_name,5,contract,source)
        for current,producer,name in ((full5,10,full_name),(staged5,5,staged_name)):
            applied=current['deterministic_runtime']['applied_runtime']
            require(applied['producer_stop']==producer and applied['producer_job_id']==results[name]['job_id'],
                    'step-five prefix proof belongs to another actual producer')
        prefix_path=DIAGNOSTICS/f'{method}-step5-prerestart-equivalence.json'
        prefix=helper.read_json(prefix_path,'receipt_sha256')
        validate_resume_equivalence(helper,prefix,full5,staged5,step=5,method=method)
        name=staged_name+'-resume10';attempt=DIAGNOSTICS/f'{name}-attempt.json';submission=DIAGNOSTICS/f'{name}-submission.json'
        require(helper.read_json(attempt)['command']==proof['resume_commands'][method],'staged continuation attempt differs')
        job=accepted_job(helper.read_json(submission))
        require(job not in jobs,'resume must use its own distinct job');jobs.add(job)
        resumed=validate_chunk(helper,protocol,'mis',method,10,DIAGNOSTICS/staged_name,job,
                               proof['resume_commands'][method],training_files,runner,True)
        comparison_path=DIAGNOSTICS/staged_name/'resume_equivalence.json'
        comparison=helper.read_json(comparison_path,'receipt_sha256')
        validate_resume_equivalence(helper,comparison,results[full_name],resumed,step=10,method=method)
        resumes[method]={'resumed':resumed,'attempt':helper.identity(attempt),'submission':helper.identity(submission),
                         'step5_prefix':helper.identity(prefix_path),'step5_prefix_receipt_sha256':prefix['receipt_sha256'],
                         'resume_equivalence':helper.identity(comparison_path),
                         'resume_equivalence_receipt_sha256':comparison['receipt_sha256']}
    require(len(jobs)==8,'exactly eight distinct diagnostic allocations required')
    return {'status':'all_six_diagnostics_two_exact_prefixes_and_two_continuations_verified',
            'initial':results,'resumes':resumes,'scope':'native_correctness_gate_not_performance_evidence'}


def job_state(job,runner):
    raw=runner(['sacct','-X','-P','-n','-j',job,'--format=JobIDRaw,State,ExitCode'])
    rows=[line.split('|') for line in raw.splitlines() if line.strip()]
    if not rows:return {'job_id':job,'state':'UNKNOWN','exit_code':None}
    require(len(rows)==1 and len(rows[0])==3 and rows[0][0]==job,'ambiguous scheduler job state')
    state=rows[0][1].split()[0]
    return {'job_id':job,'state':state,'exit_code':rows[0][2]}


def diagnostic_readiness(helper,runner):
    proof=diagnostic_declaration(helper)
    states={};jobs=set()
    for name,record in proof['initial'].items():
        job=record['job_id'];require(job not in jobs,'diagnostic job reused');jobs.add(job)
        states[name]=job_state(job,runner)
    for method in RESUME_METHODS:
        name=f'mis_{method}_staged-resume10'
        attempt=DIAGNOSTICS/f'{name}-attempt.json';submission=DIAGNOSTICS/f'{name}-submission.json'
        if attempt.exists() or attempt.is_symlink():
            require(helper.read_json(attempt)['command']==proof['resume_commands'][method],'diagnostic resume attempt changed')
            require(submission.is_file() and not submission.is_symlink(),
                    'diagnostic resume submission is unresolved; production remains blocked')
            job=accepted_job(helper.read_json(submission))
            require(job not in jobs,'diagnostic job reused');jobs.add(job)
            states[name]=job_state(job,runner)
        elif submission.exists() or submission.is_symlink():
            raise ValueError('diagnostic resume has no durable attempt')
    for state in states.values():
        require(state['state'] not in TERMINAL or (state['state']=='COMPLETED' and state['exit_code']=='0:0'),
                'a diagnostic failed; production is blocked')
    ready=len(states)==8 and all(item['state']=='COMPLETED' and item['exit_code']=='0:0' for item in states.values())
    return {'scheduler_ready_for_full_evidence_validation':ready,'jobs':states}


def attempts(helper,root,protocol):
    by_key={plan['key']:plan for plan in protocol['plans']};result={}
    attempt_paths=sorted(root.glob('*-attempt.json'))
    attempt_names={path.name.removesuffix('-attempt.json') for path in attempt_paths}
    require(all(path.name.removesuffix('-submission.json') in attempt_names for path in root.glob('*-submission.json')),
            'orphan submission receipt prevents a hidden retry')
    for path in attempt_paths:
        name=path.name.removesuffix('-attempt.json')
        require(name in by_key,'unknown campaign submission attempt')
        record=helper.read_json(path,'attempt_sha256')
        require(record['key']==name and record['protocol_sha256']==protocol['protocol_sha256'],'attempt targets another declaration')
        submission=path.with_name(name+'-submission.json')
        if not submission.is_file():
            require(not submission.is_symlink(),'linked unresolved submission')
            raise ValueError('durable unresolved submission requires review; no further submissions or retry')
        response=helper.read_json(submission,'submission_sha256')
        require(response['attempt']==helper.identity(path),'response belongs to another attempt')
        job=accepted_job(response)
        require(job not in {x['job_id'] for x in result.values()},'one scheduler job cannot satisfy two plans')
        result[name]={'job_id':job,'attempt':record,'attempt_identity':helper.identity(path),'submission':helper.identity(submission)}
    for name,record in result.items():
        require(record['attempt']['command']==resolved_command(by_key[name],result),'submitted command differs from declared dependencies')
    require(all(path.stem in result for path in (root/'terminals').glob('*.json')),
            'terminal receipt has no retained submission authority')
    return result


def resolved_command(plan,submitted):
    argv=list(plan['command'])
    if plan['role']=='eval':
        require(plan['depends_on'] in submitted,'evaluation has no accepted training dependency')
        token='@'+plan['depends_on'];job=submitted[plan['depends_on']]['job_id']
        require(sum(word.count(token) for word in argv)==1,'ambiguous training-job placeholder')
        argv=[word.replace(token,job) for word in argv]
    return argv


def occupancy(submitted,runner):
    raw=runner(['squeue','-r','-u','uag72kz','--qos=qos_gpu_a100-dev','-h','-o','%i'])
    queued=[line.strip() for line in raw.splitlines() if line.strip()]
    require(len(set(queued))==len(queued) and all(QUEUE_JOB.fullmatch(job) for job in queued),'scheduler queue is not fully expanded/unique')
    invisible=set()
    for record in submitted.values():
        job=record['job_id']
        if job not in queued and job_state(job,runner)['state'] not in TERMINAL:invisible.add(job)
    return {'visible_jobs':queued,'accepted_but_not_visible':sorted(invisible),'reserved_job_slots':len(queued)+len(invisible)}


def validate_evaluation(helper,protocol,source,plan,submission,runner):
    output=Path(plan['output']);job=submission['job_id']
    allocation=accounting(job,1,runner)
    binding=submission_binding(helper,submission['attempt']['command'],job,runner)
    index_path=output/'result_index.json';index=helper.read_json(index_path,'index_sha256')
    launch_path=output/'launch_receipt.json';launch=helper.read_json(launch_path,'launch_sha256')
    require(index['status']=='complete_raw_development_100_k8_verified' and index['protocol']==helper.PROTOCOL
            and index['task']==plan['task'] and index['method']==plan['method'] and index['step']==plan['step']
            and index['instances']==100 and index['candidates']==800 and index['candidate_budget']==8
            and index['slurm_job_id']==job and index['launch_receipt']==helper.identity(launch_path),'evaluation result index/job mismatch')
    require(launch['slurm_job_id']==job and launch['source_manifest']==protocol['management_source_manifest']
            and launch['benchmark']==protocol['benchmark'] and launch['output']==str(output)
            and launch['task']==plan['task'] and launch['method']==plan['method'] and launch['step']==plan['step']
            and launch['training']['train']==plan['train'] and index['training_job_id']==launch['training']['job_id']
            and index['checkpoint']==str(Path(plan['train'])/'checkpoints'/f'checkpoint-{plan["step"]}'),
            'actual evaluation launch differs from declared arm/source/benchmark')
    strict=load_module(source/'scripts/compare_development.py','_co_deterministic_campaign_raw')
    require(helper.identity(index['details']['path'])==index['details']
            and helper.identity(index['run_config']['path'])==index['run_config'],'raw artifacts changed after indexing')
    artifact=strict._load_artifact(index['details']['path'],index['run_config']['path'],label='co_deterministic_campaign_raw')
    helper.validate_raw(helper.read_json(index['run_config']['path'],'run_config_sha256'),artifact,launch,source)
    require(artifact.protocol_fingerprint==index['protocol_fingerprint'],'indexed protocol fingerprint differs')
    return {'job_id':job,'task':plan['task'],'method':plan['method'],'step':plan['step'],
            'allocation':allocation,'actual_submission':binding,'index':helper.identity(index_path),
            'launch':helper.identity(launch_path),'details':index['details'],'run_config':index['run_config']}


def refresh(helper,root,protocol,source,training_files,submitted,runner,write=False):
    states={};verified={}
    for plan in protocol['plans']:
        name=plan['key']
        if name not in submitted:
            states[name]={'state':'UNSUBMITTED'};continue
        record=submitted[name];state=job_state(record['job_id'],runner);states[name]=state
        if state['state'] not in TERMINAL:continue
        require(state['state']=='COMPLETED' and state['exit_code']=='0:0','failed campaign job requires review; no retry or further submission')
        if plan['role']=='train':
            current=validate_chunk(helper,protocol,plan['task'],plan['method'],plan['step'],Path(plan['train']),record['job_id'],
                                   record['attempt']['command'],training_files,runner)
        else:
            require(plan['depends_on'] in verified,'evaluation completed without a currently validated training endpoint')
            current=validate_evaluation(helper,protocol,source,plan,record,runner)
            launch=helper.read_json(current['launch']['path'],'launch_sha256')
            train=verified[plan['depends_on']]
            require(launch['training']['job_id']==train['job_id'] and launch['training']['complete_checkpoint']==train['complete_checkpoint']
                    and launch['training']['deterministic_runtime']==train['deterministic_runtime'],
                    'evaluation weights/complete checkpoint differ from the actual declared training job')
        value={'status':'verified_completed_'+plan['role'],'key':name,'protocol_sha256':protocol['protocol_sha256'],
               'attempt':record['attempt_identity'],'submission':record['submission'],'evidence':current}
        terminal=root/'terminals'/f'{name}.json'
        if terminal.exists() or terminal.is_symlink():
            saved=helper.read_json(terminal,'terminal_sha256')
            require(saved=={**value,'terminal_sha256':helper.digest(value)},'cached terminal differs from current inputs')
        elif write:helper.write_once(terminal,value,'terminal_sha256')
        verified[name]=current;states[name]['state']='VERIFIED_COMPLETE'
    return states,verified


def read_only_pair(root,expected_sha,task,step,*,baseline_method='foarl_control',runner=read_command):
    """Re-admit one complete matched milestone, including cumulative chunk cost.

    Other milestones need not have completed. All source, raw, checkpoint and
    native correctness checks are current; no cached terminal/index is created
    or rewritten. Callers may compute paired metrics only after this succeeds.
    """
    require(task in TASKS and type(step) is int and step in EVAL_STEPS
            and baseline_method in ('foarl_control','classic_grpo'),'unsupported development pair')
    root=Path(root).absolute()
    with (root/'.submission.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        helper,protocol,source,training_files=load_protocol(root,expected_sha)
        submitted=attempts(helper,root,protocol)
        gate=diagnostic_gate(helper,protocol,training_files,runner)
        gate_path=root/'diagnostic_gate.json'
        require(helper.read_json(gate_path,'gate_sha256')=={**gate,'gate_sha256':helper.digest(gate)},'retained correctness gate changed')
        by_key={item['key']:item for item in protocol['plans']};arms={}
        for method in (baseline_method,'foarl_teacher'):
            chunks=[]
            for stop in range(100,step+1,100):
                name=key('train',task,method,stop)
                require(name in submitted,'a cumulative training chunk has no accepted job')
                plan,record=by_key[name],submitted[name]
                chunks.append(validate_chunk(helper,protocol,task,method,stop,Path(plan['train']),record['job_id'],
                                             record['attempt']['command'],training_files,runner))
            name=key('eval',task,method,step)
            require(name in submitted,'the matched evaluation has no accepted job')
            evaluation=validate_evaluation(helper,protocol,source,by_key[name],submitted[name],runner)
            launch=helper.read_json(evaluation['launch']['path'],'launch_sha256')
            require(launch['training']['job_id']==chunks[-1]['job_id']
                    and launch['training']['complete_checkpoint']==chunks[-1]['complete_checkpoint']
                    and launch['training']['deterministic_runtime']==chunks[-1]['deterministic_runtime'],
                    'raw evaluation is not the actual declared training endpoint')
            arms[method]={'training':chunks[-1],'training_chunks':chunks,'evaluation':evaluation}
        return {'protocol':protocol,'protocol_identity':helper.identity(root/'protocol.json'),
                'source_manifest':protocol['management_source_manifest'],'diagnostic_gate':helper.identity(gate_path),
                'benchmark':protocol['benchmark'],'task':task,'step':step,'baseline_method':baseline_method,'arms':arms,
                'cost_scope':'training sums all chunks through milestone; evaluation cost is this milestone only'}


def tick(root,expected_sha,*,submit=False,max_new_jobs=5,runner=read_command,submitter=submit_command):
    require(type(max_new_jobs) is int and 0<=max_new_jobs<=5,'a tick may submit at most five jobs')
    root=Path(root).absolute()
    with (root/'.submission.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        helper,protocol,source,training_files=load_protocol(root,expected_sha)
        submitted=attempts(helper,root,protocol)
        states,verified=refresh(helper,root,protocol,source,training_files,submitted,runner,write=submit)
        result={'status':'inspected_declared_campaign','states':states,'submitted_this_tick':[],
                'all_training_verified':sum(name.startswith('train_') for name in verified)==24,
                'all_evaluations_verified':sum(name.startswith('eval_') for name in verified)==18}
        result['diagnostics']=diagnostic_readiness(helper,runner)
        if not submit:return result
        if not result['diagnostics']['scheduler_ready_for_full_evidence_validation']:
            result['status']='waiting_for_declared_diagnostics_and_resume';return result
        # A missing or failed correctness gate never consumes a production attempt.
        gate=diagnostic_gate(helper,protocol,training_files,runner)
        gate_path=root/'diagnostic_gate.json'
        if gate_path.exists() or gate_path.is_symlink():
            require(helper.read_json(gate_path,'gate_sha256')=={**gate,'gate_sha256':helper.digest(gate)},'diagnostic gate bytes/inputs changed')
        else:helper.write_once(gate_path,gate,'gate_sha256')
        for plan in protocol['plans']:
            if len(result['submitted_this_tick'])>=max_new_jobs:break
            name=plan['key']
            if name in submitted or (plan['depends_on'] is not None and plan['depends_on'] not in verified):continue
            output=Path(plan['output'])
            if plan['role']=='eval' or plan['step']==100:
                require(not output.exists() and not output.is_symlink(),'unsubmitted output already exists; refusing a hidden retry')
            if plan['role']=='train':
                for suffix in ('admission','exit','complete'):
                    artifact=output/f'chunk-{plan["step"]}-{suffix}.json'
                    require(not artifact.exists() and not artifact.is_symlink(),'unsubmitted chunk evidence already exists; refusing a hidden retry')
            slots=occupancy(submitted,runner)
            result['queue']=slots
            if slots['reserved_job_slots']>=10:break
            argv=resolved_command(plan,submitted)
            # Every mutable authority is journaled before the scheduler call.
            attempt_path=root/f'{name}-attempt.json'
            helper.write_once(attempt_path,{'key':name,'protocol_sha256':protocol['protocol_sha256'],
                             'command':argv,'created_at_utc':now(),'diagnostic_gate':helper.identity(gate_path),
                             'queue_before':slots},'attempt_sha256')
            response=submitter(argv)  # Timeout/transport error deliberately leaves the durable unresolved attempt.
            helper.write_once(root/f'{name}-submission.json',{'attempt':helper.identity(attempt_path),
                             'returncode':response.returncode,'stdout':response.stdout,'stderr':response.stderr},'submission_sha256')
            job=accepted_job({'returncode':response.returncode,'stdout':response.stdout})
            require(job not in {item['job_id'] for item in submitted.values()},'scheduler returned a reused job ID')
            submitted=attempts(helper,root,protocol)
            result['submitted_this_tick'].append({'key':name,'job_id':job})
        result['status']='bounded_submission_tick_complete'
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    init=commands.add_parser('init');init.add_argument('--management-source',type=Path,required=True)
    init.add_argument('--management-sha',required=True);init.add_argument('--output',type=Path,required=True)
    for action in ('tick','status'):
        sub=commands.add_parser(action);sub.add_argument('--output',type=Path,required=True);sub.add_argument('--protocol-sha',required=True)
        if action=='tick':sub.add_argument('--submit',action='store_true');sub.add_argument('--max-new-jobs',type=int,default=5)
    args=parser.parse_args()
    if args.action=='init':
        result=initialize(args)
        print(json.dumps({'status':'declared_no_jobs','protocol_sha256':result['protocol_sha256'],
                          'protocol_file_sha256':hashlib.sha256((args.output/'protocol.json').read_bytes()).hexdigest(),
                          'plans':len(result['plans'])}))
    else:
        print(json.dumps(tick(args.output,args.protocol_sha,submit=getattr(args,'submit',False),
                              max_new_jobs=getattr(args,'max_new_jobs',0)),sort_keys=True))


if __name__=='__main__':
    main()
