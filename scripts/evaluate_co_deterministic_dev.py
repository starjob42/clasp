"""Launch/index admission for three-method deterministic raw-K8 dev evaluation.

No submission, retry, model execution, repair or statistical estimation occurs
here. The sibling Slurm wrapper invokes the unchanged generic evaluator.
"""
from __future__ import annotations

import argparse
import datetime as dt
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
SELF = 'scripts/evaluate_co_deterministic_dev.py'
WRAPPER = 'scripts/evaluate_co_deterministic_dev_a100.slurm'
BENCHMARK = Path('/lustre/fsn1/projects/rech/vas/uag72kz/clasp_fair_development_benchmark_20260905_v1/data_v3')
BENCHMARK_SHA = '8d2565e29e60a1f5e12fe7fa3635bd2739cc108e7eb2415f112a8941d6ab4752'
OVERLAP_SHA = 'fa10f2a70456223258d83e5d758acdfc552ac5a03681436a044f9969a072a7b7'
RUNTIME = Path('/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env')
TRAIN_DATA = Path('/lustre/fswork/projects/rech/vas/uag72kz/primal_dual/data/data_rl')
SFT = Path('/lustre/fsn1/projects/rech/vas/uag72kz/clasp_jiang_f638_sft500k_sourcev17_production_output_iclr_20260828_v14')
SFT_HASHES = {'cvrp': 'f8f634c3a59f53bf411bb0dfc85af98dbc16d936a53263cdfcd6e74ce88b25b8',
              'mis': 'cb5e979325ba7691f75998326ce79e47038d6462596dbc815f2b54d2b550da15'}
TOKENIZER_NAMES = ('added_tokens.json', 'merges.txt', 'special_tokens_map.json',
                   'tokenizer.json', 'tokenizer_config.json', 'vocab.json')
KL_NAMES = ('exact_peft_kl_reference.json', 'exact_peft_kl_reference_step1.json',
            'exact_peft_kl_reference_post_step.json')
MODEL_NAMES = ('adapter_model.safetensors', 'adapter_config.json', 'checkpoint_fidelity.json')
GENERATION = {'seed': 271828, 'dtype': 'float16', 'num_return_sequences': 8, 'generation_batch_size': 8,
              'do_sample': True, 'temperature': .7, 'top_p': .9, 'top_k': 50, 'repetition_penalty': 1.,
              'max_new_tokens': 3000, 'max_prompt_length': 29768}
PROTOCOL = 'co-deterministic-development-raw-k8-evaluation-v1'
METHODS = ('classic_grpo', 'foarl_control', 'foarl_teacher')
TRAIN_RUNNER = 'scripts/run_co_deterministic_dev_chunk.py'
TRAIN_WRAPPER = 'scripts/train_co_deterministic_dev_a100.slurm'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def identity(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), 'missing or linked input: ' + str(path))
    sha = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            sha.update(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'input changed while hashing')
    return {'path': str(path), 'sha256': sha.hexdigest(), 'size_bytes': before.st_size}


def read_json(path, field=None):
    identity(path)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if field:
        require(value[field] == digest({k:v for k,v in value.items() if k != field}), 'invalid self hash')
    return value


def write_once(path, value, field='receipt_sha256'):
    result = {**value, field: digest(value)}
    with Path(path).open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    Path(path).chmod(0o440)
    return result


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def validate_source(root, expected_sha, *, management=False):
    root = Path(root).resolve(strict=True)
    manifest = identity(root/'SOURCE_MANIFEST.sha256')
    require(re.fullmatch('[a-f0-9]{64}', expected_sha) and manifest['sha256'] == expected_sha, 'source manifest mismatch')
    files = {}
    for line in (root/'SOURCE_MANIFEST.sha256').read_text().splitlines():
        match = re.fullmatch(r'([a-f0-9]{64})  (.+)', line)
        require(match is not None, 'malformed source manifest')
        sha, name = match.groups()
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() == name
                and name not in files, 'unsafe or repeated source name')
        path = root/name
        require(path.resolve(strict=True) == path and identity(path)['sha256'] == sha, 'source byte/path mismatch')
        files[name] = sha
    observed = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    require(observed == set(files) | {'SOURCE_MANIFEST.sha256'}, 'unmanifested source input')
    needed = {'scripts/evaluate_a100.slurm', 'scripts/clasp_grpo_fresh_dev_artifacts.py',
              'scripts/compare_development.py', 'src/clasp/models/foarl_teacher_resume.py',
              'src/clasp/models/classic_grpo_resume.py', 'src/clasp/entrypoints/evaluate.py',
              TRAIN_RUNNER, TRAIN_WRAPPER, 'scripts/run_foarl_teacher_dev_chunk.py'}
    if management:
        needed |= {SELF, WRAPPER}
        require(identity(Path(__file__).resolve())['sha256'] == files.get(SELF), 'executing helper differs from manifest')
    require(needed <= files.keys(), 'required evaluation/checkpoint source missing')
    return manifest, files


def validate_scientific_equivalence(training_files, evaluation_files):
    selected = lambda files: {k:v for k,v in files.items() if k.startswith('src/clasp/') or k == 'scripts/evaluate_a100.slurm'}
    require(selected(training_files) == selected(evaluation_files), 'evaluation snapshot changed trained scientific code')
    require(all(evaluation_files.get(name) == sha for name,sha in training_files.items()),
            'evaluation snapshot changed a frozen training input')


def validate_training_allocation(raw, job):
    require(re.fullmatch('[1-9][0-9]*', job), 'invalid training job id')
    fields = ('JobIDRaw', 'State', 'ExitCode', 'ElapsedRaw', 'TimelimitRaw', 'AllocTRES', 'QOS', 'Account', 'Partition')
    lines = [line.split('|') for line in raw.splitlines() if line.strip()]
    require(len(lines) == 1 and len(lines[0]) == len(fields), 'missing/ambiguous training accounting')
    row = dict(zip(fields, lines[0], strict=True))
    require(row['JobIDRaw'] == job and row['State'] == 'COMPLETED' and row['ExitCode'] == '0:0', 'training job did not complete successfully')
    parts = [item.split('=', 1) for item in row['AllocTRES'].split(',')]
    require(all(len(item) == 2 for item in parts) and len({p[0] for p in parts}) == len(parts), 'malformed allocation')
    tres = dict(parts)
    require(tres.get('gres/gpu') == '4' and row['QOS'] == 'qos_gpu_a100-dev' and row['Account'] == 'vas@a100'
            and row['Partition'] == 'gpu_p5' and row['TimelimitRaw'] == '120' and int(row['ElapsedRaw']) > 0,
            'training allocation differs from four-A100 two-hour development contract')
    return row


def validate_chunk_command(admission, contract, task, method, step, training_files, *, diagnostic=False):
    require(method in METHODS and contract['method'] == method and contract['diagnostic'] is diagnostic,
            'training command method/scope differs')
    source = Path(contract['source']['root'])
    sft = contract['initialization']['checkpoint']
    prefix = 'classic' if method == 'classic_grpo' else 'foarl'
    expected = ['srun','--ntasks=1','--ntasks-per-node=1',str(RUNTIME/'bin/python'),'-B',
                '-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                '-m','clasp.entrypoints.train_co_deterministic_dev','--co_method',method,'--tasks',task,
                '--sft_checkpoint_path',sft,'--tokenizer_path',sft,'--data_root',str(TRAIN_DATA),
                '--output_dir',str(Path(admission['contract']['path']).parent),
                '--'+prefix+'_dev_source_manifest',str(source/'SOURCE_MANIFEST.sha256'),
                '--'+prefix+'_dev_source_manifest_sha256',contract['source']['manifest_sha256'],
                '--eval_ratio','0','--eval-from-train-split','--max_samples_per_task','0',
                '--max_eval_samples_per_task','0','--min_train_samples_per_task','1900',
                '--min_eval_samples_per_task','0','--no-prefer_subset_files','--report_to','none']
    if method != 'classic_grpo':
        expected += ['--foarl_teacher_weight',str(.05 if method=='foarl_teacher' else 0.)]
    if diagnostic:
        expected += ['--classic_grpo_diagnostic' if method == 'classic_grpo' else '--foarl_teacher_diagnostic',
                     '--save_steps','5']
    require(admission['command'] == expected and admission['stop'] == step,
            'training command differs from the frozen chunk recipe')
    require(admission['source_verified_file_count'] == len(training_files)
            and len(admission['gpu_names']) == 4 and all('A100' in item for item in admission['gpu_names']),
            'training source/GPU admission mismatch')


def validate_training_job_binding(raw, job, train, task, method, step, contract):
    fields = ('JobIDRaw','JobName','WorkDir','SubmitLine','StdOut','StdErr')
    lines = [line.split('|') for line in raw.splitlines() if line.strip()]
    require(len(lines) == 1 and len(lines[0]) == len(fields), 'missing/ambiguous training job binding')
    row = dict(zip(fields,lines[0],strict=True))
    require(row['JobIDRaw'] == job and Path(row['WorkDir']).is_absolute(), 'wrong training job/work directory')
    argv = shlex.split(row['SubmitLine'])
    launcher = Path(contract['source']['root'])/TRAIN_WRAPPER
    require(len(argv) >= 2 and Path(argv[0]).name == 'sbatch' and argv[-1] == str(launcher),
            'training job did not submit the declared absolute launcher')
    options = {}
    names = {'--export','--output','--error','--job-name'}
    short = {'-o':'--output','-e':'--error','-J':'--job-name'}
    cursor = 1
    while cursor < len(argv)-1:
        word = argv[cursor]; name, equal, value = word.partition('=')
        name = short.get(name,name)
        if name in names:
            if not equal:
                cursor += 1
                require(cursor < len(argv)-1, 'missing submission option value')
                value = argv[cursor]
            require(name not in options, 'duplicate submission option')
            options[name] = value
        cursor += 1
    require(row['JobName'] == options.get('--job-name','co-deterministic-dev'), 'training job name differs')
    exports = {}
    for item in options.get('--export','').split(','):
        if item in ('ALL','NONE'): continue
        key, separator, value = item.partition('=')
        require(separator and key not in exports, 'training submission needs explicit unambiguous CO exports')
        exports[key] = value
    expected = {'CO_SOURCE':contract['source']['root'],'CO_SOURCE_SHA':contract['source']['manifest_sha256'],
                'CO_TASK':task,'CO_METHOD':method,'CO_OUTPUT':str(train),'CO_STOP':str(step)}
    require(all(exports.get(key) == value for key,value in expected.items()) and exports.get('CO_DIAGNOSTIC','0') == '0',
            'training job exports differ from the declared performance chunk')
    require(exports.get('CO_RESUME','') == ('' if step==100 else str(train/'checkpoints'/f'checkpoint-{step-100}'))
            and exports.get('CO_COMPARE_WITH','') == '',
            'training job resume path differs from the immediately preceding chunk')
    result = {'job_id':job,'job_name':row['JobName'],'work_directory':row['WorkDir'],
              'launcher':str(launcher),'submit_line_sha256':hashlib.sha256(row['SubmitLine'].encode()).hexdigest()}
    for field, option, default in (('StdOut','--output','logs/%x-%j.out'),('StdErr','--error','logs/%x-%j.err')):
        declared = Path(options.get(option,default))
        if not declared.is_absolute(): declared = Path(row['WorkDir'])/declared
        literal = str(declared)
        expanded = literal.replace('%x',row['JobName']).replace('%j',job)
        require('%' not in expanded and row[field] in (literal,expanded), 'unexpected training log path/alias')
        result['stdout' if field=='StdOut' else 'stderr'] = identity(expanded)
    stdout = Path(result['stdout']['path'])
    with stdout.open() as stream:
        markers = [line.rstrip('\r\n') for line in stream if line.startswith('CO_DETERMINISTIC_CHUNK_COMPLETE ')]
    require(markers == [f'CO_DETERMINISTIC_CHUNK_COMPLETE task={task} method={method} step={step}'],
            'actual training stdout lacks the unique matching completion marker')
    require(identity(stdout) == result['stdout'], 'training stdout changed while checking completion')
    result['completed_marker_verified'] = True
    return result


def _verified_module(name, relative, source, training_source, training_files):
    """Permit an already loaded identical scientific module from either snapshot."""
    permitted = {source/relative, training_source/relative}
    cached = sys.modules.get(name)
    if cached is not None:
        path = Path(cached.__file__).resolve()
        require(path in permitted and identity(path)['sha256'] == training_files[relative],
                'checkpoint verifier loaded from another source')
        return cached
    sys.path.insert(0, str(source/'src'))
    path = source/relative
    require(identity(path)['sha256'] == training_files[relative], 'checkpoint verifier source differs')
    return module(path, name)


def admit_checkpoint(train, step, contract, source):
    """Read-only saved-boundary admission, including full10's intermediate5.

    The caller separately binds the producer's successful Slurm job/completion.
    Runtime evidence itself retains its actual producer stop and child exit.
    """
    train, source = Path(train).resolve(strict=True), Path(source).resolve(strict=True)
    method = contract['method']
    require(method in METHODS and type(step) is int and step in
            ((5,10) if contract['diagnostic'] else (100,200,300,400)), 'unsupported saved checkpoint boundary')
    training_source = Path(contract['source']['root']).resolve(strict=True)
    _, training_files = validate_source(training_source, contract['source']['manifest_sha256'])
    _, source_files = validate_source(source, identity(source/'SOURCE_MANIFEST.sha256')['sha256'], management=True)
    validate_scientific_equivalence(training_files, source_files)
    kind = 'classic_grpo_resume' if method == 'classic_grpo' else 'foarl_teacher_resume'
    verifier = _verified_module('clasp.models.'+kind, 'src/clasp/models/'+kind+'.py',
                                source, training_source, training_files)
    validate = getattr(verifier, 'validate_'+kind)
    checkpoint = train/'checkpoints'/f'checkpoint-{step}'
    actual = validate(checkpoint, expected_step=step, expected_contract=contract,
                      run_contract_path=train/'local_development_run_contract.json')
    # Its __file__ guard deliberately requires the ORIGINAL training snapshot;
    # loading a byte-identical management copy would break resumed provenance.
    sys.path.insert(0, str(training_source/'scripts'))
    shared_name = 'run_foarl_teacher_dev_chunk'
    shared_relative = 'scripts/'+shared_name+'.py'
    shared = sys.modules.get(shared_name)
    if shared is not None:
        shared_path = Path(shared.__file__).resolve()
        require(shared_path in {source/shared_relative, training_source/shared_relative}
                and identity(shared_path)['sha256'] == training_files[shared_relative],
                'shared runtime dependency loaded from another source')
    else:
        shared_path = training_source/shared_relative
        require(identity(shared_path)['sha256'] == training_files[shared_relative],
                'shared runtime dependency source differs')
        module(shared_path, shared_name)
    runtime_path = training_source/TRAIN_RUNNER
    require(identity(runtime_path)['sha256'] == training_files[TRAIN_RUNNER], 'runtime verifier source differs')
    runtime = module(runtime_path, '_co_original_training_runtime_'+contract['source']['manifest_sha256'][:12])
    declaration = runtime.read_only_runtime_declaration(train, contract)
    applied = runtime.read_only_runtime_evidence(checkpoint, contract)
    return {'complete_checkpoint':actual, 'deterministic_runtime':{
        'declaration':declaration, 'declaration_file':identity(train/runtime.DECLARATION_NAME),
        'applied_runtime':applied, 'applied_runtime_file':identity(checkpoint/runtime.EVIDENCE_NAME)}}


def admit_training(train, task, method, step, job, source, source_files, *, runner=subprocess.check_output):
    require(task in SFT_HASHES and method in METHODS and type(step) is int
            and step in (100,200,400), 'unsupported task/method/performance checkpoint')
    train = Path(train).resolve(strict=True)
    contract_path = train/'local_development_run_contract.json'
    contract = read_json(contract_path)
    require(contract['diagnostic'] is False and contract['method'] == method and contract['seed'] == 43,
            'diagnostic or wrong training arm cannot enter performance evaluation')
    require(contract['fixed_horizon'] == {'max_steps':400,'world_size':4}
            and contract['teacher_weight'] == (.05 if method == 'foarl_teacher' else 0.)
            and (contract.get('beta') == 0. if method == 'classic_grpo' else 'beta' not in contract),
            'training horizon/objective mismatch')
    sft = SFT/'runs'/task/'seed42/checkpoints/checkpoint-31250'
    require(contract['initialization']['checkpoint'] == str(sft) and
            contract['initialization']['files']['adapter_model.safetensors']['sha256'] == SFT_HASHES[task], 'wrong task SFT initialization')
    require(contract.get('data_files') and all(name.startswith(task+'/train/') for name in contract['data_files']), 'training task/data mismatch')
    training_source = Path(contract['source']['root']).resolve(strict=True)
    training_manifest, training_files = validate_source(training_source, contract['source']['manifest_sha256'])
    validate_scientific_equivalence(training_files, source_files)
    admission_path, exit_path, complete_path = (train/f'chunk-{step}-{suffix}.json' for suffix in ('admission','exit','complete'))
    admission, exit_receipt = read_json(admission_path), read_json(exit_path)
    require(str(admission['job_id']) == job and admission['stop'] == step and admission['contract'] == identity(contract_path),
            'checkpoint chunk differs from the declared training job/contract')
    validate_chunk_command(admission,contract,task,method,step,training_files)
    require(type(exit_receipt['returncode']) is int and exit_receipt['returncode'] == 0, 'training subprocess failed')
    command = ['sacct','-X','-P','-n','-j',job,'--format=JobIDRaw,State,ExitCode,ElapsedRaw,TimelimitRaw,AllocTRES,QOS,Account,Partition']
    allocation = validate_training_allocation(runner(command, text=True, timeout=30), job)
    job_binding = validate_training_job_binding(runner(
        ['sacct','-X','-P','-n','-j',job,'--format=JobIDRaw,JobName%100,WorkDir%1000,SubmitLine%10000,StdOut%1000,StdErr%1000'],
        text=True,timeout=30),job,train,task,method,step,contract)
    checkpoint = train/'checkpoints'/f'checkpoint-{step}'
    admitted = admit_checkpoint(train, step, contract, source)
    actual = admitted['complete_checkpoint']
    require(actual == read_json(complete_path, 'receipt_sha256'), 'retained chunk completion differs from current checkpoint admission')
    proof = admitted['deterministic_runtime']['applied_runtime']
    require(proof['producer_stop'] == step and proof['producer_job_id'] == job
            and proof['admission'] == identity(admission_path), 'runtime producer differs from actual evaluated chunk')
    kl_names = () if method == 'classic_grpo' else KL_NAMES
    names = (*MODEL_NAMES, *TOKENIZER_NAMES, *kl_names)
    checkpoint_files = {name: identity(checkpoint/name) for name in names}
    for name in ('chat_template.jinja',):
        if (checkpoint/name).exists() or (checkpoint/name).is_symlink():
            checkpoint_files[name] = identity(checkpoint/name)
    require(all(checkpoint_files[name] == actual['files'][name] for name in (*MODEL_NAMES,*kl_names)), 'inference/KL files differ from verified checkpoint')
    return {'train': str(train), 'task':task, 'method':method, 'step':step, 'job_id':job,
            'contract':identity(contract_path), 'training_source_manifest':training_manifest,
            'chunk_admission':identity(admission_path), 'chunk_exit':identity(exit_path),
            'chunk_complete':identity(complete_path), 'complete_checkpoint':actual,
            'deterministic_runtime':admitted['deterministic_runtime'],
            'checkpoint':checkpoint_files, 'allocation':allocation,'job_binding':job_binding}


def validate_artifact_inventory(artifact, checkpoint, step):
    require(artifact['kind'] == 'local_directory' and artifact['basename'] == f'checkpoint-{step}'
            and artifact['manifest_sha256'] == digest(artifact['files']), 'model artifact manifest mismatch')
    seen = set()
    for item in artifact['files']:
        name = item['path']
        require(name not in seen and name in checkpoint, 'unknown or duplicate checkpoint inference artifact')
        seen.add(name)
        require(item == {'path':name,'sha256':checkpoint[name]['sha256'],'size':checkpoint[name]['size_bytes']}, 'model inference bytes differ')
    require(seen == set(checkpoint), 'checkpoint inference inventory omitted bound weights/tokenizer/KL proof')


def validate_raw(config, artifact, launch, source):
    task, step = launch['task'], launch['step']
    train = Path(launch['training']['train'])
    checkpoint = train/'checkpoints'/f'checkpoint-{step}'
    output = Path(launch['output'])
    require(config['stage'] == 'grpo' and config['tasks'] == [task] and config['split'] == 'eval'
            and config['run_status'] == 'complete' and config['evaluation_purpose'] == 'development', 'wrong raw run scope/status')
    require(Path(config['resolved_model_path']).resolve() == checkpoint and Path(config['resolved_run_dir']).resolve() == train,
            'raw run resolved wrong checkpoint/training directory')
    require(Path(config['data_root']).resolve() == BENCHMARK and Path(config['results_dir']).resolve().is_relative_to(output), 'wrong raw dataset/output')
    for field, snapshot in (('details',artifact.details_snapshot),('run_config',artifact.companion_snapshot)):
        path = Path(snapshot.path).resolve()
        require(Path(config['files'][field]).resolve() == path and path.is_relative_to(output), 'raw artifact path binding differs')
    generation = config['generation']
    require(all(generation.get(k) == v for k,v in GENERATION.items()) and generation['best_of_n'] == 8
            and generation['seed_policy'] == 'per_example_sha256_v1', 'raw generation mismatch')
    require(all(generation.get(k) == [] for k in ('temperature_schedule','top_p_schedule','op_size_band_top_p_schedule'))
            and generation.get('op_size_band_top_p_min_size') == generation.get('op_size_band_top_p_max_size') == 0,
            'undeclared sampling schedule')
    prompting, limits = config['prompting'], config['limits']
    require(prompting['prompt_style_resolved'] == 'jiang_f638' and prompting['no_reason'] is True
            and prompting['clasp_mis_jiang_response_output'] == '0' and prompting['clasp_mis_relabel_order'] == ''
            and Path(prompting['tokenizer_path']).resolve() == checkpoint, 'prompt/tokenizer representation mismatch')
    require(limits['eval_start_index'] == 0 and limits['max_eval_samples_per_task'] == 100 and limits['prefer_subset_files'] is False,
            'wrong development slice')
    provenance = config['experiment_provenance']
    require(generation == {**provenance['generation'],'best_of_n':8}, 'generation/provenance mismatch')
    validate_artifact_inventory(provenance['model']['checkpoint_artifact'], launch['training']['checkpoint'], step)
    dataset = launch['benchmark']['files'][task]['dataset']
    require(provenance['data']['files'] == [{'path':task+'/eval/test.json','sha256':dataset['sha256'],
            'size':dataset['size_bytes'],'split':'eval','task':task}], 'raw dataset hash differs')
    code = config['code_identity']
    require(code == provenance['code'] and code['manifest_sha256'] == digest(code['files']), 'code manifest mismatch')
    seen = set()
    for item in code['files']:
        name = item['path']; path = source/'src/clasp'/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and name not in seen, 'unsafe/duplicate raw code identity')
        actual = identity(path); seen.add(name)
        require(item == {'path':name,'sha256':actual['sha256'],'size':actual['size_bytes']}, 'raw evaluator code differs')
    require('entrypoints/evaluate.py' in seen, 'missing evaluator source identity')
    ordered = sorted(artifact.records.values(), key=lambda row: row.example_index)
    require(len(ordered) == 100 and artifact.candidate_budget == 8 and artifact.artifact_type == 'native_raw', 'wrong raw candidate/instance count')
    for row, expected in zip(ordered, launch['benchmark']['ordered_instances'][task], strict=True):
        require(all(row.raw.get(k) == v for k,v in expected.items()) and row.task == task and row.split == 'eval'
                and row.candidate_budget == len(row.raw['candidates']) == 8, 'raw instance/reference/K8 mismatch')


def launch(args):
    source = args.source.resolve(strict=True)
    manifest, files = validate_source(source, args.source_sha, management=True)
    require(identity(args.running_wrapper)['sha256'] == files[WRAPPER], 'executed Slurm spool bytes differ from declared wrapper')
    require((source/'logs').is_dir() and not (source/'logs').is_symlink(), 'generic evaluator requires pre-provisioned source logs directory')
    output = args.output.absolute()
    require(not output.exists() and not output.is_symlink(), 'evaluation output already exists')
    job = os.environ.get('SLURM_JOB_ID','')
    require(re.fullmatch('[1-9][0-9]*',job), 'evaluation launch requires an actual Slurm job')
    require(Path(sys.executable).resolve() == (RUNTIME/'bin/python').resolve(), 'wrong evaluation Python')
    benchmark_helper = module(source/'scripts/clasp_grpo_fresh_dev_artifacts.py', '_co_deterministic_benchmark')
    benchmark = benchmark_helper.validate_benchmark(BENCHMARK)
    require(benchmark['manifest_sha256'] == BENCHMARK_SHA and benchmark['audit_sha256'] == OVERLAP_SHA, 'not the fixed inspected development100 benchmark')
    training = admit_training(args.train,args.task,args.method,args.step,args.training_job_id,source,files)
    payload = {'protocol':PROTOCOL,'status':'admitted_co_deterministic_raw_development_evaluation',
               'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'task':args.task,'method':args.method,
               'step':args.step,'output':str(output),'slurm_job_id':job,'source_manifest':manifest,
               'management':{'helper':identity(source/SELF),'wrapper':identity(source/WRAPPER)},
               'runtime_python':identity((RUNTIME/'bin/python').resolve()),'training':training,
               'benchmark':benchmark,'generation':GENERATION,'repair':False,'evaluation_slice':{'split':'eval','start':0,'count':100},
               'scope':'reused_development_data_not_confirmation_or_sota'}
    output.mkdir(parents=False,exist_ok=False)
    return write_once(output/'launch_receipt.json',payload,'launch_sha256')


def index(output):
    output = Path(output).absolute()
    receipt = read_json(output/'launch_receipt.json','launch_sha256')
    require(receipt['protocol'] == PROTOCOL and receipt['output'] == str(output)
            and receipt['slurm_job_id'] == os.environ.get('SLURM_JOB_ID'), 'index job/output differs from actual launch')
    source = Path(receipt['source_manifest']['path']).parent
    _, files = validate_source(source,receipt['source_manifest']['sha256'],management=True)
    details, configs = list(output.glob('eval/**/run_details__*.jsonl')), list(output.glob('eval/**/run_config__*.json'))
    require(len(details) == len(configs) == 1, 'expected one raw details/config pair')
    strict = module(source/'scripts/compare_development.py','_co_deterministic_strict_raw')
    artifact = strict._load_artifact(str(details[0]),str(configs[0]),label='co_deterministic_raw_dev')
    config = read_json(configs[0],'run_config_sha256')
    validate_raw(config,artifact,receipt,source)
    benchmark_helper = module(source/'scripts/clasp_grpo_fresh_dev_artifacts.py','_co_deterministic_benchmark_final')
    require(benchmark_helper.validate_benchmark(BENCHMARK) == receipt['benchmark'], 'benchmark changed during evaluation')
    before = receipt['training']
    after = admit_training(Path(before['train']),receipt['task'],receipt['method'],receipt['step'],before['job_id'],source,files)
    require(after == before, 'training/checkpoint changed during evaluation')
    for item in receipt['management'].values():
        require(identity(item['path']) == item, 'management bytes changed')
    require(identity(receipt['runtime_python']['path']) == receipt['runtime_python'], 'Python runtime changed')
    return write_once(output/'result_index.json', {'protocol':PROTOCOL,'status':'complete_raw_development_100_k8_verified',
        'task':receipt['task'],'method':receipt['method'],'step':receipt['step'],'slurm_job_id':receipt['slurm_job_id'],
        'training_job_id':before['job_id'],'checkpoint':str(Path(before['train'])/'checkpoints'/f"checkpoint-{receipt['step']}"),
        'launch_receipt':identity(output/'launch_receipt.json'),'details':identity(details[0]),'run_config':identity(configs[0]),
        'protocol_fingerprint':artifact.protocol_fingerprint,'instances':100,'candidates':800,'candidate_budget':8,
        'generation_seed':271828,'repair':False,'scope':'reused_development_data_not_confirmation_or_sota',
        'reference_semantics':'gap_to_verified_heuristic_reference' if receipt['task']=='cvrp' else 'gap_to_certified_optimum',
        'final_evaluation_slurm_success_must_be_checked_after_job_exit':True},'index_sha256')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    start = commands.add_parser('launch')
    for name in ('source','train','output','running-wrapper'):
        start.add_argument('--'+name,type=Path,required=True)
    start.add_argument('--source-sha',required=True)
    start.add_argument('--training-job-id',required=True)
    start.add_argument('--task',choices=tuple(SFT_HASHES),required=True)
    start.add_argument('--method',choices=METHODS,required=True)
    start.add_argument('--step',type=int,choices=(100,200,400),required=True)
    finish = commands.add_parser('index'); finish.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = launch(args) if args.command == 'launch' else index(args.output)
    print(json.dumps({k:result[k] for k in ('status','task','method','step')},sort_keys=True))


if __name__ == '__main__':
    main()
