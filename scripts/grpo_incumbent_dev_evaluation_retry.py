#!/usr/bin/env python3
"""Finite, explicit pre-model infrastructure retries; original science/events stay intact.

Exactly three pinned mkdir failures may each receive one successor evaluation.
No other failure is excused. init never submits; all mutations are create-only.
"""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
CAMPAIGN = Path('/lustre/fsn1/projects/rech/vas/uag72kz/grpo_incumbent_foarl_s43_b9_dev_20260907_v1')
TRAINING_SOURCE = Path('/lustre/fsn1/projects/rech/vas/uag72kz/grpo_incumbent_source_20260907_v3/source-snapshot')
TRAINING_SHA = 'b960cd574f4cd605028518ba15b7a55d6277cc319b5e17b64cf8608e74cda10f'
CONTROLLER = 'scripts/grpo_incumbent_dev_campaign.py'
SELF = 'scripts/grpo_incumbent_dev_evaluation_retry.py'
WRAPPER = 'scripts/evaluate_grpo_incumbent_retry_a100.slurm'
TEST = 'tests/test_grpo_incumbent_dev_evaluation_retry.py'
DECLARATION = CAMPAIGN / 'ledgers/development_evaluation_retry_v1.json'
EVENTS = CAMPAIGN / 'ledgers/evaluation_retry_events_v1'
TERMINALS = CAMPAIGN / 'ledgers/evaluation_retry_terminal_v1'
PROVISION = TRAINING_SOURCE.parent / 'readonly_logs_directory_amendment_v1.json'
PROVISION_SHA = 'b23def3960bf58d5347b606087c28923d4f954eeb56ae297d31c09be6ce9863d'
STDERR_SHA = '5906374ca1b07da72b187c3e30d3e90453498beffb423f6c0d9f485f639f21c5'
FAILED = {
    'cvrp/grpo_z': {'job_id': '1847633', 'elapsed_seconds': 11, 'launch_sha256': '1378c4518ce6085706d1fc6e6835ee7ffdf78b2c24f4e9fc1f98e38e2db46cfa', 'launch_size': 78333},
    'cvrp/grpo_verified': {'job_id': '1847860', 'elapsed_seconds': 13, 'launch_sha256': '0bf9755d55fe64248198481b783b0f8a9d54ad2228c8dc8dca97d267cb8d86e9', 'launch_size': 78368},
    'mis/grpo_z': {'job_id': '1847871', 'elapsed_seconds': 8, 'launch_sha256': 'd27ee45bb4831e21bc12ee0c7f1644c17373b29a212a0bc5b96a65219599a02d', 'launch_size': 78328},
}
FAILURE_STATES = {'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'CANCELLED', 'NODE_FAIL', 'BOOT_FAIL', 'PREEMPTED', 'DEADLINE', 'REVOKED'}
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING', 'REQUEUED'}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(block)
    return {'path': str(path), 'sha256': sha.hexdigest(), 'size_bytes': path.stat().st_size}


def checked(path, field='receipt_sha256'):
    value = json.loads(Path(path).read_text())
    require(value.get(field) == digest({k: v for k, v in value.items() if k != field}), 'receipt self hash differs: ' + str(path))
    return value


def create(path, value, field='receipt_sha256'):
    value = {**value, field: digest(value)}
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o440)
    with os.fdopen(descriptor, 'w') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    return value


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def command_output(command, *, text=True):
    return subprocess.check_output(command, text=text, timeout=30)


def original():
    sys.path.insert(0, str(TRAINING_SOURCE / 'src'))
    control = module('retry_unchanged_original_controller', TRAINING_SOURCE / CONTROLLER)
    protocol, source = control.protocol_for(CAMPAIGN)
    require(source == TRAINING_SOURCE and protocol['source_manifest']['sha256'] == TRAINING_SHA, 'wrong original campaign source')
    return control, protocol, source


def validate_management(source, expected_sha, control):
    source = Path(source).resolve(strict=True)
    manifest = control.source_files(source, expected_sha)
    entries = control.manifest_entries(source)
    origin = control.manifest_entries(TRAINING_SOURCE)
    require(len(origin) == 334 and all(entries.get(k) == v for k, v in origin.items()), 'original scientific/management files changed')
    require({SELF, WRAPPER, TEST} <= set(entries), 'retry management/test source missing')
    require(Path(__file__).resolve() == source / SELF, 'retry helper must execute from its declared snapshot')
    return manifest


def validate_provision():
    binding = identity(PROVISION)
    require(binding['sha256'] == PROVISION_SHA, 'unreviewed logs provision receipt')
    p = json.loads(PROVISION.read_text())
    expected = {'status': 'passed_empty_readonly_directory_provision', 'source': str(TRAINING_SOURCE),
                'source_manifest_sha256': TRAINING_SHA, 'verified_source_file_count_before': 334,
                'verified_source_file_count_after': 334, 'all_manifested_source_bytes_unchanged': True,
                'source_file_mode': '0444', 'source_directory_mode': '0555',
                'created_directory': str(TRAINING_SOURCE / 'logs'), 'created_directory_mode': '0555',
                'created_directory_empty': True, 'mkdir_p_existing_directory_passed': True,
                'scientific_arguments_changed': False, 'jobs_submitted': False}
    require(all(p.get(k) == v for k, v in expected.items()), 'logs provision semantics differ')
    logs = TRAINING_SOURCE / 'logs'
    require(logs.is_dir() and not logs.is_symlink() and not list(logs.iterdir()), 'source logs must remain an empty real directory')
    require(stat.S_IMODE(logs.stat().st_mode) == stat.S_IMODE(TRAINING_SOURCE.stat().st_mode) == 0o555,
            'source/logs must remain read-only')
    return binding


def scheduler(control, jobs, runner):
    if not jobs:
        return {}
    return control.scheduler_rows(runner(['sacct', '-X', '-P', '-n', '-j', ','.join(sorted(jobs, key=int)),
                                          '--format=JobIDRaw,State,ExitCode,StdOut%600,ElapsedRaw']))


def validate_failed_attempt(key, control, protocol, row):
    require(key in FAILED, 'failure is outside the finite declared exception')
    expected = FAILED[key]
    job = control.submitted_job(CAMPAIGN, protocol, key, role='eval')
    require(job == expected['job_id'], 'original failed submission differs')
    plan = next(x for x in protocol['planned_eval_commands'] if x['key'] == key)
    require(row and row['job_id'] == job and row['state'] == 'FAILED' and row['exit_code'] == '1:0'
            and row['elapsed_seconds'] == expected['elapsed_seconds'], 'original failure scheduler evidence differs')
    require(row['stdout'] in (plan['stdout'], plan['stdout'].replace('%j', job)), 'original failure stdout binding differs')
    output = Path(plan['output'])
    task, method = key.split('/')
    expected_dirs = ['eval', 'eval/' + method + '_incumbent_seed271828', 'eval/' + method + '_incumbent_seed271828/' + task]
    entries = list(output.rglob('*'))
    require(output.is_dir() and not output.is_symlink() and not any(path.is_symlink() for path in entries)
            and sorted(path.relative_to(output).as_posix() for path in entries if path.is_dir()) == expected_dirs
            and sorted(path.relative_to(output).as_posix() for path in entries if path.is_file()) == ['launch_receipt.json']
            and all(path.is_file() or path.is_dir() for path in entries), 'original failure has undeclared files/directories/model outputs')
    paths = {'launch': output / 'launch_receipt.json', 'stdout': Path(plan['stdout'].replace('%j', job)),
             'stderr': Path(next(x.split('=', 1)[1] for x in plan['command'] if x.startswith('--error=')).replace('%j', job))}
    files = {name: identity(path) for name, path in paths.items()}
    require(all(files[name]['sha256'] == expected['launch_sha256'] and files[name]['size_bytes'] == expected['launch_size']
                for name in ('launch', 'stdout')), 'original launch/stdout bytes differ')
    require(files['stderr']['sha256'] == STDERR_SHA and files['stderr']['size_bytes'] == 150, 'failure is not the exact missing-logs error')
    launch = checked(paths['launch'], 'launch_sha256')
    require(launch['slurm_job_id'] == job and launch['campaign_protocol_sha256'] == protocol['receipt_sha256']
            and launch['source_manifest'] == protocol['source_manifest'] and launch['task'] + '/' + launch['method'] == key,
            'original launch/protocol binding differs')
    return {'job_id': job, 'state': 'FAILED', 'exit_code': '1:0', 'elapsed_seconds': expected['elapsed_seconds'],
            'files': files, 'output_file_paths': ['launch_receipt.json'], 'output_directory_paths': expected_dirs, 'no_model_outputs': True}


def generic_environment(plan, protocol):
    return {'CLASP_PROJECT_ROOT': str(TRAINING_SOURCE), 'CONDA_ENV': protocol['runtime'],
            'DATA_ROOT': protocol['evaluation']['data_root'], 'CHECKPOINT_PATH': plan['checkpoint'],
            'CKPT_LABEL': plan['method'] + '_incumbent_seed271828', 'TASK': plan['task'], 'STAGE': 'grpo',
            'EVALUATION_PURPOSE': 'development', 'EVALUATION_REPRESENTATION_PROFILE': 'jiang_f638_clasp',
            'INFERENCE_PARENT_ROOT': plan['output'] + '/eval', 'EVAL_START_INDEX': '0', 'MAX_EVAL_SAMPLES_PER_TASK': '100',
            'SEED': '271828', 'DTYPE': 'float16', 'NUM_RETURN_SEQUENCES': '8', 'GENERATION_BATCH_SIZE': '8',
            'TEMPERATURE': '0.7', 'TOP_P': '0.9', 'TOP_K': '50', 'REPETITION_PENALTY': '1.0',
            'MAX_NEW_TOKENS': '3000', 'MAX_PROMPT_LENGTH': '29768', 'GPU_ARCH_MODULE': 'arch/a100',
            'CONDA_MODULE': 'miniforge/24.9.0', 'PYTHONPATH': str(TRAINING_SOURCE / 'src'),
            'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'WANDB_MODE': 'offline', 'TOKENIZERS_PARALLELISM': 'false'}


def plans_for(source, protocol):
    result = []
    for old in protocol['planned_eval_commands']:
        if old['key'] not in FAILED:
            continue
        label = old['key'].replace('/', '_')
        output = CAMPAIGN / 'evaluation_retry_v1' / (label + '_raw_k8')
        plan = {k: old[k] for k in ('key', 'task', 'method', 'training_root', 'checkpoint')}
        plan.update(output=str(output), stdout=str(CAMPAIGN / 'logs' / (label + '_retry_v1_%j.out')))
        exports = {'RETRY_SOURCE': str(source), 'RETRY_RUNTIME': protocol['runtime'], 'RETRY_KEY': old['key'], 'RETRY_DECLARATION': str(DECLARATION)}
        plan['command'] = ['sbatch', '--parsable', '--no-requeue', '--job-name=incumbent-retry-' + label, '--output=' + plan['stdout'],
                           '--error=' + str(CAMPAIGN / 'logs' / (label + '_retry_v1_%j.err')),
                           '--export=ALL,' + ','.join(k + '=' + v for k, v in exports.items()), str(source / WRAPPER)]
        plan['generic_command'] = ['/bin/bash', str(TRAINING_SOURCE / 'scripts/evaluate_a100.slurm')]
        plan['generic_environment'] = generic_environment(plan, protocol)
        result.append(plan)
    return result


def initialize(source, source_sha, runner=command_output):
    run = lambda command: runner(command, text=True)
    control, protocol, _ = original()
    source = Path(source).resolve(strict=True)
    management = validate_management(source, source_sha, control)
    provision = validate_provision()
    rows = scheduler(control, {p['job_id'] for p in FAILED.values()}, run)
    failures = {key: validate_failed_attempt(key, control, protocol, rows.get(item['job_id'])) for key, item in FAILED.items()}
    plans = plans_for(source, protocol)
    require(not DECLARATION.exists() and not EVENTS.exists() and not TERMINALS.exists(), 'retry declaration/ledgers already exist')
    require(all(not Path(p['output']).exists() for p in plans), 'retry outputs exist before declaration')
    terminals = {key: control.training_endpoint(CAMPAIGN, key, protocol, TRAINING_SOURCE, runner=run) for key in FAILED}
    payload = {'protocol': 'incumbent-finite-pre-model-evaluation-retry-v1', 'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'campaign': str(CAMPAIGN), 'original_protocol': identity(CAMPAIGN / 'ledgers/protocol.json'),
               'scientific_source': protocol['source_manifest'], 'management_source': management,
               'management': {name: identity(source / relative) for name, relative in (('helper', SELF), ('wrapper', WRAPPER))},
               'provision': provision, 'failed_attempts': failures, 'fixed_exception': FAILED,
               'planned_retries': plans, 'generation': protocol['generation'], 'benchmark': protocol['benchmark'],
               'training_terminal_bindings': {key: identity(CAMPAIGN / 'ledgers/terminal' / (key.replace('/', '_') + '.json')) for key in FAILED},
               'training_job_ids': {key: audit['job_id'] for key, audit in terminals.items()},
               'exactly_one_successor_per_key': True, 'scientific_arguments_changed': False, 'original_events_preserved': True}
    EVENTS.mkdir(); TERMINALS.mkdir()
    result = create(DECLARATION, payload)
    return {'declaration': identity(DECLARATION), 'receipt_sha256': result['receipt_sha256'], 'submitted': False}


def declaration(expected_sha=None):
    binding = identity(DECLARATION)
    if expected_sha is not None:
        require(binding['sha256'] == expected_sha, 'retry declaration differs from explicit pin')
    p = checked(DECLARATION)
    require(p['protocol'] == 'incumbent-finite-pre-model-evaluation-retry-v1' and p['campaign'] == str(CAMPAIGN)
            and p['fixed_exception'] == FAILED and p['exactly_one_successor_per_key'] is True
            and p['scientific_arguments_changed'] is False and p['original_events_preserved'] is True, 'retry scope changed')
    control, protocol, _ = original()
    source = Path(p['management_source']['path']).parent
    require(validate_management(source, p['management_source']['sha256'], control) == p['management_source'], 'retry management changed')
    require(p['management'] == {'helper': identity(source / SELF), 'wrapper': identity(source / WRAPPER)}, 'retry executed management differs')
    require(p['original_protocol'] == identity(CAMPAIGN / 'ledgers/protocol.json') and p['scientific_source'] == protocol['source_manifest'], 'original protocol/source changed')
    require(p['provision'] == validate_provision() and p['planned_retries'] == plans_for(source, protocol)
            and p['generation'] == protocol['generation'] and p['benchmark'] == protocol['benchmark'], 'retry arguments/inputs changed')
    failed_rows = scheduler(control, {item['job_id'] for item in FAILED.values()}, command_output)
    for key, failure in p['failed_attempts'].items():
        row = failed_rows.get(failure['job_id'])
        require(validate_failed_attempt(key, control, protocol, row) == failure, 'preserved failure evidence changed')
        require(identity(p['training_terminal_bindings'][key]['path']) == p['training_terminal_bindings'][key], 'prebound training terminal changed')
    require(set(p['failed_attempts']) == set(FAILED), 'missing failure exception evidence')
    return p, control, protocol, source


def read_events(p):
    result = {key: [] for key in FAILED}
    plans = {plan['key']: plan for plan in p['planned_retries']}
    ids = []
    for path in sorted(EVENTS.glob('*.json')):
        event = checked(path, 'event_sha256')
        require(event['key'] in FAILED and event['kind'] in ('intent', 'submitted', 'rejected')
                and event['declaration'] == identity(DECLARATION) and event['command'] == plans[event['key']]['command'], 'retry event declaration differs')
        result[event['key']].append(event)
    for key, group in result.items():
        intents = [e for e in group if e['kind'] == 'intent']; endings = [e for e in group if e['kind'] != 'intent']
        require(len(intents) <= 1 and len(endings) <= 1 and (not endings or intents and endings[0]['transaction'] == intents[0]['transaction']), 'duplicate/orphan retry attempt')
        for event in endings:
            if event['kind'] == 'submitted':
                require(re.fullmatch(r'[1-9][0-9]*', event['job_id']) and event['job_id'] not in {x['job_id'] for x in FAILED.values()}, 'invalid replacement job')
                ids.append(event['job_id'])
    require(len(ids) == len(set(ids)), 'one replacement job binds multiple arms')
    return result


def submitted(p, key):
    values = [e for e in read_events(p)[key] if e['kind'] == 'submitted']
    require(len(values) == 1, 'one accepted replacement required')
    return values[0]['job_id']


def launch(p, key, wrapper):
    control, protocol, _ = original()
    plan = next(x for x in p['planned_retries'] if x['key'] == key)
    job = submitted(p, key)
    require(os.environ.get('SLURM_JOB_ID') == job and os.environ.get('RETRY_KEY') == key, 'actual replacement job/key differs')
    require(os.environ.get('RETRY_DECLARATION') == str(DECLARATION)
            and Path(os.environ['RETRY_SOURCE']).resolve() == Path(p['management_source']['path']).parent
            and Path(os.environ['RETRY_RUNTIME']).resolve() == Path(protocol['runtime']).resolve(), 'replacement launch paths differ')
    actual = identity(wrapper)
    require(all(actual[k] == p['management']['wrapper'][k] for k in ('sha256', 'size_bytes')), 'replacement spool bytes differ')
    require(Path(sys.executable).resolve() == (Path(protocol['runtime']) / 'bin/python').resolve(), 'replacement Python differs')
    terminal = control.training_endpoint(CAMPAIGN, key, protocol, TRAINING_SOURCE)
    require(terminal['job_id'] == p['training_job_ids'][key], 'replacement initialization changed')
    payload = checked(p['failed_attempts'][key]['files']['launch']['path'], 'launch_sha256')
    payload = {k: v for k, v in payload.items() if k != 'launch_sha256'}
    require(payload['checkpoint'] == {name: terminal['checkpoint'][name] for name in payload['checkpoint']}, 'replacement checkpoint differs from failed launch')
    payload.update(created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), slurm_job_id=job,
                   management=p['management'], spool_execution_path=str(Path(wrapper).resolve()),
                   retry_declaration=identity(DECLARATION), replaced_pre_model_job_id=FAILED[key]['job_id'])
    output = Path(plan['output'])
    require(not output.exists(), 'replacement output already exists')
    output.parent.mkdir(exist_ok=True)
    output.mkdir()
    return create(output / 'launch_receipt.json', payload, 'launch_sha256')


def verify_execution(plan, p, job):
    output = Path(plan['output'])
    launch_receipt = checked(output / 'launch_receipt.json', 'launch_sha256')
    execution = checked(output / 'execution_receipt.json')
    require(launch_receipt['retry_declaration'] == execution['retry_declaration'] == identity(DECLARATION)
            and launch_receipt['replaced_pre_model_job_id'] == FAILED[plan['key']]['job_id']
            and launch_receipt['slurm_job_id'] == execution['slurm_job_id'] == job
            and execution['launch_sha256'] == launch_receipt['launch_sha256'] and execution['returncode'] == 0,
            'replacement actual execution/job/declaration differs')
    require(execution['generic_command'] == plan['generic_command'] and execution['generic_environment'] == plan['generic_environment'], 'replacement executed arguments/environment changed')
    original_launch = checked(p['failed_attempts'][plan['key']]['files']['launch']['path'], 'launch_sha256')
    allowed = {'launch_sha256', 'created_at_utc', 'slurm_job_id', 'management', 'spool_execution_path', 'retry_declaration', 'replaced_pre_model_job_id'}
    require({k: v for k, v in launch_receipt.items() if k not in allowed} == {k: v for k, v in original_launch.items() if k not in allowed}, 'replacement changed scientific launch fields')
    require(launch_receipt['management'] == p['management'], 'replacement management binding differs')
    return launch_receipt, execution


def validate_result(p, key, job):
    control, protocol, _ = original()
    plan = next(x for x in p['planned_retries'] if x['key'] == key)
    require(submitted(p, key) == job, 'replacement result job differs from accepted event')
    launch_receipt, _ = verify_execution(plan, p, job)
    terminal = control.training_endpoint(CAMPAIGN, key, protocol, TRAINING_SOURCE)
    fresh = module('retry_original_fresh_index', TRAINING_SOURCE / 'scripts/clasp_grpo_fresh_dev_artifacts.py')
    fresh.write_new = lambda path, value, field: {**value, field: digest(value)}
    index = fresh.index(Path(plan['output']))
    benchmark = fresh.validate_benchmark(Path(protocol['evaluation']['data_root']))
    require(benchmark == protocol['benchmark'] == p['benchmark'], 'replacement benchmark changed')
    strict = module('retry_original_strict_comparison', TRAINING_SOURCE / 'scripts/compare_development.py')
    artifact = strict._load_artifact(index['details']['path'], index['run_config']['path'], label=key)
    require(artifact.artifact_type == 'native_raw', 'replacement contains assistance or rescoring')
    config = checked(index['run_config']['path'], 'run_config_sha256')
    cross = control.audit_evaluation_arm(plan, launch_receipt, config, index, terminal, job_id=job,
              training=protocol, evaluation={'generation': protocol['generation'], 'management': p['management']},
              benchmark=benchmark, source=TRAINING_SOURCE, records=artifact.records)
    path = Path(plan['output']) / 'result_index.json'
    if path.exists():
        require(checked(path, 'index_sha256') == index, 'replacement index changed')
    return {'status': 'terminal_evaluation_verified', 'key': key, 'job_id': job, 'protocol_sha256': protocol['receipt_sha256'],
            'retry_declaration': identity(DECLARATION), 'replaced_pre_model_job_id': FAILED[key]['job_id'],
            'index_payload': index, 'cross_bindings': cross, 'raw_details': index['details'], 'run_config': index['run_config'],
            'execution_receipt': identity(Path(plan['output']) / 'execution_receipt.json')}


def worker(key, wrapper):
    p, _, _, _ = declaration()
    require(key in FAILED, 'worker outside finite exception')
    launch_receipt = launch(p, key, wrapper)
    plan = next(x for x in p['planned_retries'] if x['key'] == key)
    prefixes = ('CLASP_', 'PD_', 'GRPO_', 'FOARL_', 'GENERATION_', 'WANDB_', 'HF_', 'TRANSFORMERS_', 'TOKENIZERS_',
                'BITSANDBYTES_', 'UNSLOTH_', 'ACCELERATE_', 'NCCL_', 'PYTORCH_')
    names = {'PYTHONPATH', 'PYTHONHOME', 'RESUME_FROM_CHECKPOINT', 'GPU_ARCH_MODULE', 'CONDA_MODULE'}
    environment = {k: v for k, v in os.environ.items() if not k.startswith(prefixes) and k not in names}
    environment.update(plan['generic_environment'])
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    result = subprocess.run(plan['generic_command'], env=environment, check=False)
    execution = {'retry_declaration': identity(DECLARATION), 'slurm_job_id': launch_receipt['slurm_job_id'],
                 'launch_sha256': launch_receipt['launch_sha256'], 'generic_command': plan['generic_command'],
                 'generic_environment': plan['generic_environment'], 'returncode': result.returncode,
                 'started_at_utc': started, 'completed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    create(Path(plan['output']) / 'execution_receipt.json', execution)
    require(result.returncode == 0, 'replacement evaluator failed; no automatic retry')
    audit = validate_result(p, key, launch_receipt['slurm_job_id'])
    index = audit['index_payload']
    return create(Path(plan['output']) / 'result_index.json', {k: v for k, v in index.items() if k != 'index_sha256'}, 'index_sha256')


def all_reservations(p, control, protocol, runner):
    collections = [control.read_events(CAMPAIGN, protocol, role=role) for role in ('train', 'eval')] + [read_events(p)]
    accepted, unresolved = {}, 0
    for collection in collections:
        for key, group in collection.items():
            endings = [e for e in group if e['kind'] == 'submitted']
            if group and not endings:
                unresolved += 1
            for event in endings:
                job = event['job_id']
                require(job not in accepted, 'job ID reused between original/retry ledgers')
                accepted[job] = key
    rows = scheduler(control, set(accepted), runner)
    queued = control.expanded_queue(runner(['squeue', '-r', '-h', '-u', control.USER, '-q', control.QOS, '-o', '%i|%T']))
    require(all(re.fullmatch(r'[1-9][0-9]*(?:_[0-9]+)?', job) for job in queued), 'queue must contain expanded individual job IDs')
    terminal_states = FAILURE_STATES | {'COMPLETED'}
    lagging = {job for job in accepted if job not in rows or rows[job]['state'] not in terminal_states}
    occupied = len(queued | lagging) + unresolved
    for job, row in rows.items():
        if row['state'] in FAILURE_STATES or row['state'] == 'COMPLETED' and row['exit_code'] != '0:0':
            key = accepted[job]
            require(key in FAILED and job == FAILED[key]['job_id'], 'new/unapproved job failure blocks supervisor: ' + job)
            validate_failed_attempt(key, control, protocol, row)
    return occupied, rows, accepted


def await_visibility(p, control, runner, job, key, *, sleep=time.sleep, monotonic=time.monotonic):
    """Keep caller's shared lock after any attempt until visible/terminal or review.

    Unknown IDs cannot be guessed. An operator must resolve the durable attempt
    before terminating this process, because termination releases its flock.
    """
    started = last_alert = monotonic()
    while True:
        evidence = {}
        try:
            queue = control.expanded_queue(runner(['squeue', '-r', '-h', '-u', control.USER, '-q', control.QOS, '-o', '%i|%T']))
            require(all(re.fullmatch(r'[1-9][0-9]*(?:_[0-9]+)?', value) for value in queue), 'queue IDs are not expanded')
            evidence['expanded_queue'] = sorted(queue)
            if job is not None and job in queue:
                result = {'status': 'accepted_retry_visible_in_expanded_queue', 'job_id': job, 'key': key}
                break
            if job is not None:
                rows = scheduler(control, {job}, runner)
                evidence['accounting'] = rows
                if job in rows and rows[job]['state'] in FAILURE_STATES | {'COMPLETED'}:
                    result = {'status': 'accepted_retry_conclusively_terminal', 'job_id': job, 'key': key, 'scheduler': rows[job]}
                    break
        except Exception as error:
            evidence['query_error'] = str(error)
        now = monotonic()
        if now - last_alert >= 120:
            alert = {'status': 'uncertain_retry_attempt_shared_lock_retained_no_resubmission', 'job_id': job, 'key': key,
                     'declaration': identity(DECLARATION), 'elapsed_seconds': now - started, 'evidence': evidence,
                     'operator_action': 'Resolve this exact durable attempt before terminating this process; termination releases the shared lock.'}
            create(CAMPAIGN / 'ledgers' / ('retry_visibility_alert_' + uuid.uuid4().hex + '.json'), alert)
            try:
                print(json.dumps(alert, sort_keys=True), flush=True)
            except BrokenPipeError:
                pass
            last_alert = now
        sleep(5)
    return create(CAMPAIGN / 'ledgers' / ('retry_visibility_' + uuid.uuid4().hex + '.json'),
                  {**result, 'declaration': identity(DECLARATION), 'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()})


def failed_allocation_records(control, runner=command_output):
    fields = ('JobIDRaw', 'State', 'ExitCode', 'Start', 'End', 'ElapsedRaw', 'TimelimitRaw', 'AllocTRES', 'QOS', 'Account', 'Partition')
    jobs = {item['job_id']: key for key, item in FAILED.items()}
    command = ['env', 'TZ=UTC', 'SLURM_TIME_FORMAT=standard', 'sacct', '-X', '-P', '-n', '-j', ','.join(sorted(jobs, key=int)), '--format=' + ','.join(fields)]
    raw = runner(command, text=True)
    rows, seen = [], set()
    for values in csv.reader(raw.splitlines(), delimiter='|'):
        if not values:
            continue
        require(len(values) == len(fields), 'malformed failed-job allocation')
        row = dict(zip(fields, values)); job = row['JobIDRaw']
        require(job in jobs and job not in seen, 'unknown/duplicate failed-job allocation')
        seen.add(job); key = jobs[job]; expected = FAILED[key]
        require(row['State'] == 'FAILED' and row['ExitCode'] == '1:0' and row['TimelimitRaw'] == '120'
                and row['QOS'] == control.QOS and row['Account'] == 'vas@a100' and row['Partition'] == 'gpu_p5', 'wrong failed-job state/resource queue')
        seconds = int(row['ElapsedRaw'])
        require(seconds == expected['elapsed_seconds'] and int((dt.datetime.fromisoformat(row['End']) - dt.datetime.fromisoformat(row['Start'])).total_seconds()) == seconds,
                'failed-job allocation duration changed')
        tres = {}
        for entry in row['AllocTRES'].split(','):
            name, value = entry.split('=', 1)
            require(name not in tres, 'duplicate failed-job TRES')
            tres[name] = value
        require(tres.get('gres/gpu') == '1' and tres.get('cpu') == '16' and tres.get('billing') == '1', 'failed-job actual allocation changed')
        rows.append({**row, 'key': key, 'allocated_gpu_seconds': seconds, 'allocated_cpu_seconds': 16 * seconds,
                     'slurm_billing_unit_seconds': seconds})
    require(seen == set(jobs), 'missing failed-job accounting')
    return {'jobs': rows, 'sacct_command': command, 'sacct_utc': raw,
            'allocated_gpu_hours': sum(row['allocated_gpu_seconds'] for row in rows) / 3600,
            'allocated_cpu_hours': sum(row['allocated_cpu_seconds'] for row in rows) / 3600,
            'slurm_billing_unit_hours': sum(row['slurm_billing_unit_seconds'] for row in rows) / 3600,
            'interpretation': 'Reserved resource time and Slurm billing units; no claim of resource utilization or reconciled charged quota.'}


def retry_tick(p, control, protocol, *, submit, record, max_new_jobs, runner, visibility=None):
    require(type(max_new_jobs) is int and 0 <= max_new_jobs <= 5, 'retry tick budget invalid')
    visibility = visibility or await_visibility
    observations, newly = [], 0
    for plan in p['planned_retries']:
        key = plan['key']; group = read_events(p)[key]
        occupied, rows, accepted = all_reservations(p, control, protocol, runner)
        submitted_events = [e for e in group if e['kind'] == 'submitted']
        if submitted_events:
            job = submitted_events[0]['job_id']; row = rows.get(job)
            if row is None:
                observations.append({'key': key, 'job_id': job, 'state': 'awaiting_accounting_no_retry'}); continue
            require(row['stdout'] in (plan['stdout'], plan['stdout'].replace('%j', job)), 'replacement scheduler stdout differs')
            status = {'key': key, **row}
            if row['state'] == 'COMPLETED' and row['exit_code'] == '0:0':
                audit = validate_result(p, key, job)
                terminal_path = TERMINALS / (key.replace('/', '_') + '.json')
                if terminal_path.exists():
                    require(checked(terminal_path) == {**audit, 'receipt_sha256': digest(audit)}, 'replacement terminal changed')
                elif record:
                    create(terminal_path, audit)
                status['terminal_verified'] = True
            observations.append(status); continue
        if group:
            observations.append({'key': key, 'state': 'prior_attempt_no_retry'}); continue
        if not submit or occupied >= 10 or newly >= max_new_jobs:
            observations.append({'key': key, 'state': 'ready', 'occupied_slots': occupied}); continue
        require(not Path(plan['output']).exists(), 'replacement output exists before submission')
        control.training_endpoint(CAMPAIGN, key, protocol, TRAINING_SOURCE, runner=runner)
        payload = {'key': key, 'declaration': identity(DECLARATION), 'command': plan['command'],
                   'transaction': uuid.uuid4().hex, 'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
        create(EVENTS / (uuid.uuid4().hex + '.json'), {**payload, 'kind': 'intent'}, 'event_sha256')
        try:
            answer = runner(plan['command']).strip()
            require(re.fullmatch(r'[1-9][0-9]*(;[^\s;]+)?', answer), 'ambiguous replacement sbatch result')
            job = answer.split(';', 1)[0]
            require(job not in accepted, 'replacement job ID already bound')
        except Exception as error:
            create(EVENTS / (uuid.uuid4().hex + '.json'), {**payload, 'kind': 'rejected', 'error': str(error)}, 'event_sha256')
            visibility(p, control, runner, None, key)  # Retain the shared lock until operator review.
            raise
        create(EVENTS / (uuid.uuid4().hex + '.json'), {**payload, 'kind': 'submitted', 'job_id': job}, 'event_sha256')
        visibility(p, control, runner, job, key)  # Do not release lock while the accepted slot is invisible.
        newly += 1; observations.append({'key': key, 'job_id': job, 'state': 'submitted'})
    return {'observations': observations, 'newly_submitted': newly}


def supervisor_tick(expected_sha, submit=False, record=False, max_new_jobs=5, runner=command_output):
    require(type(max_new_jobs) is int and 0 <= max_new_jobs <= 5, 'at most five new jobs per supervisor tick')
    run = lambda command: runner(command, text=True)
    with (CAMPAIGN / 'ledgers/controller.lock').open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        p, control, protocol, _ = declaration(expected_sha)
        result, remaining = {}, max_new_jobs
        for role in ('train', 'eval'):
            occupied, _, _ = all_reservations(p, control, protocol, run)
            budget = min(remaining, max(0, 10 - occupied))
            result[role] = control.reconcile(CAMPAIGN, submit=submit, record=record, max_new_jobs=budget, role=role, runner=run)
            remaining -= sum(row['state'] == 'submitted' for row in result[role]['observations'])
            all_reservations(p, control, protocol, run)  # Refuse any new failure before another submission.
        result['retry'] = retry_tick(p, control, protocol, submit=submit, record=record, max_new_jobs=remaining, runner=run)
        if record and result['train']['all_training_verified'] and not (CAMPAIGN / 'ledgers/training_accounting.json').exists():
            control.finalize_training(CAMPAIGN, runner=run)
        originals = [row for row in result['eval']['observations'] if row['key'] not in FAILED]
        result['all_six_evaluations_verified'] = len(originals) == 3 and all(x.get('terminal_verified') for x in originals) and all(x.get('terminal_verified') for x in result['retry']['observations'])
        return result


def resolve_evaluation(key, *, expected_sha=None, runner=command_output):
    """Strict read-only terminal resolver; finite retry keys use only their successor."""
    p, control, protocol, _ = declaration(expected_sha)
    run = lambda command: runner(command, text=True)
    if key in FAILED:
        plan = next(x for x in p['planned_retries'] if x['key'] == key)
        job = submitted(p, key)
        path = TERMINALS / (key.replace('/', '_') + '.json')
    else:
        require(key in {x['key'] for x in protocol['planned_eval_commands']}, 'unknown development evaluation key')
        plan = next(x for x in protocol['planned_eval_commands'] if x['key'] == key)
        job = control.submitted_job(CAMPAIGN, protocol, key, role='eval')
        path = CAMPAIGN / 'ledgers/evaluation_terminal' / (key.replace('/', '_') + '.json')
    row = scheduler(control, {job}, run).get(job)
    require(row and row['state'] == 'COMPLETED' and row['exit_code'] == '0:0'
            and row['stdout'] in (plan['stdout'], plan['stdout'].replace('%j', job)), 'resolved development endpoint not completed/bound')
    audit = validate_result(p, key, job) if key in FAILED else control.evaluation_validation(plan, TRAINING_SOURCE, protocol, job_id=job)
    retained = checked(path)
    require(all(retained.get(k) == v for k, v in audit.items()) and retained['job_id'] == job
            and retained['protocol_sha256'] == protocol['receipt_sha256'], 'resolved development terminal changed')
    index_path = Path(plan['output']) / 'result_index.json'
    require(checked(index_path, 'index_sha256') == audit['index_payload'], 'resolved raw index changed')
    files = {'terminal': identity(path), 'index': identity(index_path), 'launch': identity(Path(plan['output']) / 'launch_receipt.json'),
             'details': audit['raw_details'], 'run_config': audit['run_config']}
    if key in FAILED:
        files['execution'] = audit['execution_receipt']
    return {'key': key, 'job_id': job, 'plan': plan, 'audit': audit, 'files': files,
            'replaced_pre_model_job_id': FAILED[key]['job_id'] if key in FAILED else None}


def admission_bindings(expected_sha):
    p, _, _, _ = declaration(expected_sha)
    files = {'declaration': identity(DECLARATION), 'provision': p['provision'], 'management_source': p['management_source'], **p['management']}
    for key, item in p['failed_attempts'].items():
        for name, value in item['files'].items():
            files[key + '/' + name] = value
    for path in sorted(EVENTS.glob('*.json')):
        files['event/' + path.name] = identity(path)
    return files


def summarize(expected_sha):
    p, control, protocol, _ = declaration(expected_sha)
    accounting = control.finalize_training(CAMPAIGN)
    resolved = [resolve_evaluation(item['key'], expected_sha=expected_sha) for item in protocol['planned_eval_commands']]
    summary = module('retry_original_statistics', TRAINING_SOURCE / 'scripts/summarize_clasp_grpo_fresh_dev.py')
    summary.METHODS = control.METHODS
    indices = [Path(item['files']['index']['path']) for item in resolved]
    loaded, identities, _ = summary.collect(indices)
    rows, comparisons = [], {}
    for task in control.TASKS:
        for method in control.METHODS:
            rows.append(summary.summarize(list(loaded[task, method].records.values()), task, method))
        for left, right in (('grpo_z', 'grpo_verified'), ('grpo_z', 'foarl_f638_dev'), ('foarl_f638_dev', 'grpo_verified')):
            a, b = loaded[task, left], loaded[task, right]
            comparisons[task + '/' + left + '_vs_' + right] = {'strict_comparison': summary.strict.build_receipt(a, b, left_label=left, right_label=right),
                                                              'unconditional_solved_quality': summary.paired_quality(a, b)}
    costs = control.allocation_records({r['job_id']: r['key'] for r in resolved}, gpus=1)
    failed_costs = failed_allocation_records(control)
    payload = {'status': 'all_six_arms_raw_evaluation_and_provenance_verified_with_finite_infrastructure_amendment',
               'source_manifest': protocol['source_manifest'], 'protocol_sha256': protocol['receipt_sha256'],
               'retry_declaration': identity(DECLARATION), 'amendment_bindings': admission_bindings(expected_sha),
               'summary': rows, 'comparisons': comparisons, 'result_indices': identities,
               'resolved_evaluations': resolved, 'training_accounting': accounting,
               'evaluation_accounting': costs, 'excluded_pre_model_failure_accounting': failed_costs,
               'total_evaluation_allocated_gpu_hours_including_failures': costs['allocated_gpu_hours'] + failed_costs['allocated_gpu_hours'],
               'limitations': ['Same original six fixed checkpoints, scientific evaluator and rawK8 development data.',
                              'Three exact pre-model mkdir failures are preserved and charged separately; no outcome-based retries.',
                              'One training seed and reused development100/task; not confirmation/SOTA or literal Jiang reproduction.',
                              'Own-success gaps differ from paired joint-success gaps; CVRP references are verified heuristics.']}
    destination = CAMPAIGN / 'comparison_retry_v1'
    require(not destination.exists(), 'amended comparison already exists')
    destination.mkdir()
    report = create(destination / 'comparison.json', payload)
    columns = ('task', 'method', 'feasible_at_8_percent', 'candidate_feasibility_percent', 'mean_feasible_gap_percent', 'mean_signed_reference_gap_percent', 'unconditional_solved_quality')
    with (destination / 'table.csv').open('x') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows({k: row[k] for k in columns} for row in rows)
    lines = ['| Task | Method | Feasible@8 | Candidate feasibility | Gap on own successes |', '|---|---|---:|---:|---:|']
    for row in rows:
        gap = 'undefined' if row['mean_feasible_gap_percent'] is None else f"{row['mean_feasible_gap_percent']:.4f}%"
        lines.append(f"| {row['task']} | {row['method']} | {row['feasible_at_8_count']}/100 | {row['candidate_feasibility_percent']:.3f}% | {gap} |")
    with (destination / 'table.md').open('x') as handle:
        handle.write('\n'.join(lines) + '\n\nSigned and paired joint-success gaps, all denominators and failed infrastructure allocation are retained in comparison.json.\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'status', 'run', 'watch', 'worker', 'summarize'))
    parser.add_argument('--source', type=Path); parser.add_argument('--source-sha'); parser.add_argument('--declaration-sha')
    parser.add_argument('--key'); parser.add_argument('--running-wrapper', type=Path)
    parser.add_argument('--submit', action='store_true'); parser.add_argument('--max-new-jobs', type=int, default=5)
    parser.add_argument('--hours', type=float, default=6)
    args = parser.parse_args()
    if args.action == 'init':
        require(not args.submit and args.source and args.source_sha, 'init requires source pins and never submits')
        result = initialize(args.source, args.source_sha)
    elif args.action == 'worker':
        require(not args.submit and args.key and args.running_wrapper, 'worker binds already accepted replacement')
        result = worker(args.key, args.running_wrapper)
    else:
        require(args.declaration_sha, 'explicit retry declaration file hash is required')
        if args.action == 'summarize':
            require(not args.submit, 'summary never submits'); result = summarize(args.declaration_sha)
        else:
            require(args.action != 'status' or not args.submit, 'status is read-only')
            require(0 < args.hours <= 6, 'watch must be bounded to six hours')
            deadline = time.monotonic() + args.hours * 3600
            while True:
                result = supervisor_tick(args.declaration_sha, args.submit, args.action != 'status', args.max_new_jobs)
                if args.action != 'watch' or result['all_six_evaluations_verified'] or time.monotonic() >= deadline:
                    break
                print(json.dumps(result, sort_keys=True), flush=True); time.sleep(60)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
