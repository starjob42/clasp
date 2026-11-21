#!/usr/bin/env python3
"""Finite postprocessing admission of complete FOARL outputs; never rerun inference.

The original scientific source, failed output, launch, and scheduler outcome stay
unchanged. Only a validated in-memory checkpoint inventory gains the three
checkpoint-local KL proofs already bound by the original training terminal.
"""
import argparse
import copy
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
CAMPAIGN = Path('/lustre/fsn1/projects/rech/vas/uag72kz/grpo_incumbent_foarl_s43_b9_dev_20260907_v1')
TRAINING_SOURCE = Path('/lustre/fsn1/projects/rech/vas/uag72kz/grpo_incumbent_source_20260907_v3/source-snapshot')
RETRY_SOURCE = Path('/lustre/fsn1/projects/rech/vas/uag72kz/grpo_incumbent_source_20260907_v7/source-snapshot')
RETRY_SOURCE_SHA = '37c1fb23d34ddf36effa8037dc9a3bc884f840682ba3c1d5ce23bf07654031be'
RETRY_HELPER_SHA = '0a70a06d67d567c2668821340c4a7cd6c32440a51ac9f0e351fd57a0a0920a8a'
RETRY_DECLARATION_SHA = '61a15f9d0acd951b12d529dc9f508c8bdb788fd6150f3498fa99767ad158cc79'
SELF = 'scripts/grpo_incumbent_foarl_postprocessing.py'
TEST = 'tests/test_grpo_incumbent_foarl_postprocessing.py'
DECLARATION = CAMPAIGN / 'ledgers/foarl_postprocessing_declaration_v1.json'
SIDECARS = CAMPAIGN / 'evaluation_postprocessing_v1'
COMPARISON = CAMPAIGN / 'comparison_postprocessing_v1'
PROTOCOL = 'incumbent-finite-foarl-postprocessing-admission-v1'
KEYS = ('cvrp/foarl_f638_dev', 'mis/foarl_f638_dev')
PROOFS = ('exact_peft_kl_reference.json', 'exact_peft_kl_reference_post_step.json', 'exact_peft_kl_reference_step1.json')
CHECKPOINT_NAMES = {'adapter_model.safetensors', 'adapter_config.json', 'checkpoint_fidelity.json', 'trainer_state.json',
                    'added_tokens.json', 'merges.txt', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json'}
RETRY_JOBS = {'cvrp/grpo_z': '1849640', 'cvrp/grpo_verified': '1849642', 'mis/grpo_z': '1849644'}
KNOWN_CVRP = {'job_id': '1848933', 'elapsed_seconds': 1879,
              'stderr_sha256': '02f0a0a9827f2afac4aecd429b9790449d514250908d2c1fb4b6c5dc5ea71c46',
              'launch_sha256': '8c0c7a021a7054e661520caf7a551b830658f7a1130bd9c94d7d324b705b1a41',
              'run_config_sha256': '637023deb2db47570e25794daea222ed857d09bf1dec4be4ec0571024700b813',
              'details_sha256': '71418ac17c3e88b2ccb52c5cca3bae05facac531b4ca9937214613952f0ab761',
              'model_artifact_manifest_sha256': '6de9b731e2c48be50a97a73ad0757b6d936f1b305116d862c5377848d44abaf7'}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(chunk)
    return {'path': str(path), 'sha256': sha.hexdigest(), 'size_bytes': path.stat().st_size}


def checked(path, field='receipt_sha256'):
    value = json.loads(Path(path).read_text())
    require(value.get(field) == digest({k: v for k, v in value.items() if k != field}), 'receipt self hash differs: ' + str(path))
    return value


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def command_output(command, *, text=True):
    return subprocess.check_output(command, text=text, timeout=30)


def retry_control():
    require(identity(RETRY_SOURCE / 'SOURCE_MANIFEST.sha256')['sha256'] == RETRY_SOURCE_SHA,
            'immutable v7 retry source differs')
    path = RETRY_SOURCE / 'scripts/grpo_incumbent_dev_evaluation_retry.py'
    require(identity(path)['sha256'] == RETRY_HELPER_SHA, 'immutable v7 retry helper differs')
    bridge = module('postprocessing_immutable_v7_bridge', path)
    require(identity(bridge.DECLARATION)['sha256'] == RETRY_DECLARATION_SHA, 'v7 retry declaration differs')
    p, control, protocol, _ = bridge.declaration(RETRY_DECLARATION_SHA)
    require({key: bridge.submitted(p, key) for key in bridge.FAILED} == RETRY_JOBS,
            'finite three existing replacement jobs changed')
    return bridge, p, control, protocol


def expected_traceback(key):
    path = str(TRAINING_SOURCE / 'scripts/grpo_incumbent_dev_campaign.py')
    return ('Traceback (most recent call last):\n'
            f'  File "{path}", line 1183, in <module>\n    main()\n'
            f'  File "{path}", line 1156, in main\n    result = evaluation_index(output, args.key)\n'
            f'  File "{path}", line 960, in evaluation_index\n    verified = evaluation_validation(item, source, protocol, job_id=job)\n'
            f'  File "{path}", line 943, in evaluation_validation\n    binding = audit_evaluation_arm(item, launch, config, index, terminal, job_id=expected_job,\n'
            f'  File "{path}", line 1086, in audit_evaluation_arm\n'
            '    require(name not in seen and name in terminal["checkpoint"], prefix + "unknown/duplicate checkpoint artifact")\n'
            f'  File "{path}", line 63, in require\n    raise ValueError(message)\n'
            f'ValueError: {key}: unknown/duplicate checkpoint artifact\n')


def checkpoint_audit_view(plan, terminal, config=None):
    """Only call after original training_endpoint independently revalidates KL lineage."""
    require(plan['key'] in KEYS and plan['method'] == 'foarl_f638_dev', 'proof inventory amendment is FOARL-only')
    train = Path(plan['training_root']).resolve(strict=True)
    checkpoint = Path(plan['checkpoint']).resolve(strict=True)
    require(checkpoint == train / 'checkpoints/checkpoint-100', 'proof inventory requires the exact step100 checkpoint')
    require(set(terminal['checkpoint']) == CHECKPOINT_NAMES, 'original checkpoint inventory changed')
    evidence = terminal['exact_reference_evidence']
    expected = {'checkpoints/checkpoint-100/' + name for name in PROOFS} | {PROOFS[0], PROOFS[2]}
    require(evidence['status'] == 'exact_frozen_initial_sft_reference_lineage_verified'
            and set(evidence['files']) == expected, 'exact-reference inventory is incomplete or unexpected')
    additions = {}
    for name in PROOFS:
        bound = evidence['files']['checkpoints/checkpoint-100/' + name]
        path = checkpoint / name
        require(not path.is_symlink() and identity(path) == bound, 'checkpoint-local proof bytes/path differ: ' + name)
        additions[name] = bound
    if config is not None:
        artifact = config['experiment_provenance']['model']['checkpoint_artifact']
        names = [item['path'] for item in artifact['files']]
        require(len(names) == len(set(names)) and set(names) == (CHECKPOINT_NAMES - {'trainer_state.json'}) | set(PROOFS),
                'FOARL raw artifact must contain exactly the original inference files and three verified proofs')
    view = copy.deepcopy(terminal)
    view['checkpoint'].update(additions)
    return view


def failure_metadata(key, control, protocol, row):
    require(key in KEYS, 'unlisted postprocessing exception key')
    plan = next(item for item in protocol['planned_eval_commands'] if item['key'] == key)
    job = control.submitted_job(CAMPAIGN, protocol, key, role='eval')
    require(row and row['job_id'] == job and row['state'] == 'FAILED' and row['exit_code'] == '1:0'
            and 0 < row['elapsed_seconds'] <= 7200
            and row['stdout'] in (plan['stdout'], plan['stdout'].replace('%j', job)),
            'postprocessing exception requires exact submitted FAILED1:0 original job')
    output = Path(plan['output'])
    require(not (output / 'result_index.json').exists()
            and not (CAMPAIGN / 'ledgers/evaluation_terminal' / (key.replace('/', '_') + '.json')).exists(),
            'original failed output must remain without an index or successful terminal')
    stderr_path = Path(plan['stdout'].replace('%j', job)).with_suffix('.err')
    stderr = stderr_path.read_text()
    trace = expected_traceback(key)
    require(stderr.count('Traceback (most recent call last):') == 1 and stderr.endswith(trace),
            'job failed for a different reason or outside the exact original final inventory check')
    configs = list(output.glob('eval/**/run_config__*.json'))
    details = list(output.glob('eval/**/run_details__*.jsonl'))
    require(len(configs) == len(details) == 1, 'one original raw config/details pair required')
    config = checked(configs[0], 'run_config_sha256')
    require(config['run_status'] == 'complete' and config['limits']['eval_start_index'] == 0
            and config['limits']['max_eval_samples_per_task'] == 100 and config['generation']['best_of_n'] == 8,
            'partial or different generation cannot qualify for postprocessing')
    files = {'stderr': identity(stderr_path), 'launch': identity(output / 'launch_receipt.json'),
             'run_config': identity(configs[0]), 'details': identity(details[0])}
    if key == KEYS[0]:
        require(job == KNOWN_CVRP['job_id'] and row['elapsed_seconds'] == KNOWN_CVRP['elapsed_seconds'], 'known CVRP job identity/duration changed')
        require(all(value['sha256'] == KNOWN_CVRP[name + '_sha256'] for name, value in files.items())
                and config['experiment_provenance']['model']['checkpoint_artifact']['manifest_sha256'] == KNOWN_CVRP['model_artifact_manifest_sha256'],
                'known original CVRP artifact or failure evidence changed')
    return {'key': key, 'job_id': job, 'scheduler': row, 'files': files,
            'traceback_sha256': hashlib.sha256(trace.encode()).hexdigest(),
            'scheduler_status_preserved': 'FAILED', 'no_inference_repeat': True}


def validate_post_result(key, control, protocol, row):
    failure = failure_metadata(key, control, protocol, row)
    plan = next(item for item in protocol['planned_eval_commands'] if item['key'] == key)
    output = Path(plan['output']); job = failure['job_id']
    terminal = control.training_endpoint(CAMPAIGN, key, protocol, TRAINING_SOURCE)
    launch = checked(output / 'launch_receipt.json', 'launch_sha256')
    require(launch['campaign_protocol_sha256'] == protocol['receipt_sha256'] and launch['training_job_id'] == terminal['job_id'],
            'postprocessing launch/training protocol differs')
    fresh = module('postprocessing_original_fresh_index', TRAINING_SOURCE / 'scripts/clasp_grpo_fresh_dev_artifacts.py')
    require(fresh.GENERATION == protocol['generation'], 'original scientific generation helper differs')
    fresh.write_new = lambda path, value, field: {**value, field: digest(value)}
    index = fresh.index(output)
    benchmark = fresh.validate_benchmark(Path(protocol['evaluation']['data_root']))
    require(benchmark == protocol['benchmark'], 'postprocessing benchmark changed')
    strict = module('postprocessing_original_raw_parser', TRAINING_SOURCE / 'scripts/compare_development.py')
    artifact = strict._load_artifact(index['details']['path'], index['run_config']['path'], label=key)
    require(artifact.artifact_type == 'native_raw', 'postprocessing only admits original unassisted raw candidates')
    config = checked(index['run_config']['path'], 'run_config_sha256')
    view = checkpoint_audit_view(plan, terminal, config)
    cross = control.audit_evaluation_arm(plan, launch, config, index, view, job_id=job,
             training=protocol, evaluation={'generation': protocol['generation'], 'management': protocol['evaluation_management']},
             benchmark=benchmark, source=TRAINING_SOURCE, records=artifact.records)
    require(failure_metadata(key, control, protocol, row) == failure, 'original artifacts changed during postprocessing')
    return {'status': 'complete_raw_evaluation_verified_after_exact_final_inventory_failure', 'key': key, 'job_id': job,
            'protocol_sha256': protocol['receipt_sha256'], 'scheduler_state': 'FAILED', 'scheduler_exit_code': '1:0',
            'index_payload': index, 'cross_bindings': cross, 'raw_details': index['details'], 'run_config': index['run_config'],
            'failure': failure, 'training_terminal': identity(CAMPAIGN / 'ledgers/terminal' / (key.replace('/', '_') + '.json')),
            'verified_inventory_additions': {name: view['checkpoint'][name] for name in PROOFS}}


def validate_management(source, source_sha, control):
    source = Path(source).resolve(strict=True)
    value = control.source_files(source, source_sha)
    entries = control.manifest_entries(source)
    original = control.manifest_entries(TRAINING_SOURCE)
    require(len(original) == 334 and all(entries.get(k) == v for k, v in original.items()), 'original scientific source changed')
    require({SELF, TEST} <= set(entries) and Path(__file__).resolve() == source / SELF,
            'postprocessing must execute from its declared immutable source')
    return value


def initialize(source, source_sha, runner=command_output):
    bridge, retry, control, protocol = retry_control()
    manifest = validate_management(source, source_sha, control)
    require(not DECLARATION.exists() and not SIDECARS.exists() and not COMPARISON.exists(), 'postprocessing declaration/output exists')
    # MIS is prospective: the one original command is declared before submission.
    events = control.read_events(CAMPAIGN, protocol, role='eval')
    require(not events[KEYS[1]] and not Path(next(p for p in protocol['planned_eval_commands'] if p['key'] == KEYS[1])['output']).exists(),
            'MIS FOARL must be declared before its sole original evaluation submission/output')
    row = bridge.scheduler(control, {KNOWN_CVRP['job_id']}, lambda cmd: runner(cmd, text=True)).get(KNOWN_CVRP['job_id'])
    actual = validate_post_result(KEYS[0], control, protocol, row)
    value = {'protocol': PROTOCOL, 'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'campaign': str(CAMPAIGN),
             'management_source': manifest, 'management': {'helper': identity(Path(source) / SELF), 'test': identity(Path(source) / TEST)},
             'original_protocol': identity(CAMPAIGN / 'ledgers/protocol.json'), 'scientific_source': protocol['source_manifest'],
             'retry_declaration': identity(bridge.DECLARATION), 'retry_bindings': bridge.admission_bindings(RETRY_DECLARATION_SHA),
             'existing_retry_jobs': RETRY_JOBS, 'keys': list(KEYS), 'known_cvrp': KNOWN_CVRP,
             'known_cvrp_audit': actual, 'planned_original_foarl_evaluations': [p for p in protocol['planned_eval_commands'] if p['key'] in KEYS],
             'allowed_checkpoint_inventory_additions': list(PROOFS), 'original_files_and_events_preserved': True,
             'scheduler_failed_status_preserved': True, 'no_inference_retry': True,
             'future_mis_rule': 'one original accepted command; only exact final inventory traceback after complete raw100 K8; no retry',
             'generation': protocol['generation'], 'benchmark': protocol['benchmark']}
    SIDECARS.mkdir()
    return bridge.create(DECLARATION, value)


def declaration(expected_sha=None):
    if expected_sha is not None:
        require(identity(DECLARATION)['sha256'] == expected_sha, 'postprocessing declaration differs from explicit pin')
    p = checked(DECLARATION)
    bridge, retry, control, protocol = retry_control()
    source = Path(p['management_source']['path']).parent
    require(p['protocol'] == PROTOCOL and p['campaign'] == str(CAMPAIGN) and p['keys'] == list(KEYS)
            and p['known_cvrp'] == KNOWN_CVRP and p['existing_retry_jobs'] == RETRY_JOBS
            and p['allowed_checkpoint_inventory_additions'] == list(PROOFS)
            and p['original_files_and_events_preserved'] is p['scheduler_failed_status_preserved'] is p['no_inference_retry'] is True,
            'postprocessing exception scope changed')
    require(validate_management(source, p['management_source']['sha256'], control) == p['management_source']
            and p['management'] == {'helper': identity(source / SELF), 'test': identity(source / TEST)}, 'postprocessing management changed')
    require(p['original_protocol'] == identity(CAMPAIGN / 'ledgers/protocol.json') and p['scientific_source'] == protocol['source_manifest']
            and p['retry_declaration'] == identity(bridge.DECLARATION)
            and p['retry_bindings'] == bridge.admission_bindings(RETRY_DECLARATION_SHA), 'postprocessing origin/retry bindings changed')
    require(p['planned_original_foarl_evaluations'] == [x for x in protocol['planned_eval_commands'] if x['key'] in KEYS]
            and p['generation'] == protocol['generation'] and p['benchmark'] == protocol['benchmark'], 'postprocessing scientific arguments changed')
    for bound in p['known_cvrp_audit']['failure']['files'].values():
        require(identity(bound['path']) == bound, 'prebound CVRP original failure/output changed')
    return p, bridge, retry, control, protocol


def all_reservations(p, bridge, retry, control, protocol, runner):
    collections = [control.read_events(CAMPAIGN, protocol, role=role) for role in ('train', 'eval')] + [bridge.read_events(retry)]
    accepted, unresolved = {}, 0
    for collection in collections:
        for key, group in collection.items():
            endings = [e for e in group if e['kind'] == 'submitted']
            unresolved += int(bool(group) and not endings)
            for event in endings:
                require(event['job_id'] not in accepted, 'one job appears in multiple original/retry ledgers')
                accepted[event['job_id']] = key
    rows = bridge.scheduler(control, set(accepted), runner)
    queue = control.expanded_queue(runner(['squeue', '-r', '-h', '-u', control.USER, '-q', control.QOS, '-o', '%i|%T']))
    require(all(re.fullmatch(r'[1-9][0-9]*(?:_[0-9]+)?', value) for value in queue), 'queue IDs are not expanded')
    terminal_states = bridge.FAILURE_STATES | {'COMPLETED'}
    lagging = {job for job in accepted if job not in rows or rows[job]['state'] not in terminal_states}
    for job, row in rows.items():
        if row['state'] in bridge.FAILURE_STATES or row['state'] == 'COMPLETED' and row['exit_code'] != '0:0':
            key = accepted[job]
            if key in bridge.FAILED and job == bridge.FAILED[key]['job_id']:
                bridge.validate_failed_attempt(key, control, protocol, row)
            else:
                failure_metadata(key, control, protocol, row)
    return len(queue | lagging) + unresolved, rows, accepted


def retain_post_result(p, bridge, control, protocol, key, row, *, record):
    actual = validate_post_result(key, control, protocol, row)
    if key == KEYS[0]:
        require(actual == p['known_cvrp_audit'], 'prebound CVRP postprocessing audit changed')
    destination = SIDECARS / key.replace('/', '_')
    index_path, terminal_path = destination / 'result_index.json', destination / 'terminal.json'
    if record:
        destination.mkdir(exist_ok=True)
    if index_path.exists():
        require(checked(index_path, 'index_sha256') == actual['index_payload'], 'postprocessing index sidecar changed')
    elif record:
        bridge.create(index_path, {k: v for k, v in actual['index_payload'].items() if k != 'index_sha256'}, 'index_sha256')
    payload = {**actual, 'postprocessing_declaration': identity(DECLARATION)}
    if terminal_path.exists():
        require(checked(terminal_path) == {**payload, 'receipt_sha256': digest(payload)}, 'postprocessing terminal sidecar changed')
    elif record:
        bridge.create(terminal_path, payload)
    return actual


def supervisor_tick(expected_sha, *, submit=False, record=False, max_new_jobs=5, runner=command_output):
    require(type(max_new_jobs) is int and 0 <= max_new_jobs <= 5, 'at most five new jobs per tick')
    run = lambda command: runner(command, text=True)
    with (CAMPAIGN / 'ledgers/controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        p, bridge, retry, control, protocol = declaration(expected_sha)
        result, remaining = {}, max_new_jobs
        for role in ('train', 'eval'):
            occupied, _, _ = all_reservations(p, bridge, retry, control, protocol, run)
            before = control.read_events(CAMPAIGN, protocol, role=role)
            # Every other original arm already has one immutable attempt.
            require(all(group or role == 'eval' and key == KEYS[1] for key, group in before.items()),
                    'only the sole original MIS FOARL evaluation may remain unsubmitted')
            budget = min(remaining, max(0, 10 - occupied))
            if role == 'eval' and submit and budget and not before[KEYS[1]]:
                # Revalidate the complete known raw output and exact KL lineage
                # again immediately before the only prospective submission.
                row = bridge.scheduler(control, {KNOWN_CVRP['job_id']}, run).get(KNOWN_CVRP['job_id'])
                require(validate_post_result(KEYS[0], control, protocol, row) == p['known_cvrp_audit'],
                        'known CVRP postprocessing evidence changed before MIS submission')
            try:
                result[role] = control.reconcile(CAMPAIGN, submit=submit, record=record, max_new_jobs=budget, role=role, runner=run)
            except Exception:
                after = control.read_events(CAMPAIGN, protocol, role=role)
                for key in after:
                    if not before[key] and after[key]:
                        accepted = [e for e in after[key] if e['kind'] == 'submitted']
                        bridge.await_visibility(retry, control, run, accepted[0]['job_id'] if accepted else None, key)
                raise
            for row in result[role]['observations']:
                if row['state'] == 'submitted':
                    require(role == 'eval' and row['key'] == KEYS[1], 'undeclared new job')
                    bridge.await_visibility(retry, control, run, row['job_id'], row['key'])
                    remaining -= 1
            all_reservations(p, bridge, retry, control, protocol, run)
        _, rows, _ = all_reservations(p, bridge, retry, control, protocol, run)
        result['retry'] = []
        for key, job in RETRY_JOBS.items():
            row = rows.get(job); status = {'key': key, 'job_id': job, 'state': row['state'] if row else 'accounting_unavailable'}
            if row and row['state'] == 'COMPLETED' and row['exit_code'] == '0:0':
                audit = bridge.validate_result(retry, key, job)
                path = bridge.TERMINALS / (key.replace('/', '_') + '.json')
                if path.exists():
                    require(bridge.checked(path) == {**audit, 'receipt_sha256': bridge.digest(audit)}, 'existing v7 replacement terminal changed')
                elif record:
                    bridge.create(path, audit)
                status['terminal_verified'] = True
            result['retry'].append(status)
        result['postprocessing'] = []
        for key in KEYS:
            events = control.read_events(CAMPAIGN, protocol, role='eval')[key]
            jobs = [e['job_id'] for e in events if e['kind'] == 'submitted']
            row = rows.get(jobs[0]) if jobs else None
            status = {'key': key, 'job_id': jobs[0] if jobs else None, 'state': row['state'] if row else 'waiting_for_original_job'}
            if row and row['state'] in bridge.FAILURE_STATES:
                retain_post_result(p, bridge, control, protocol, key, row, record=record)
                status['postprocessing_verified'] = True
            result['postprocessing'].append(status)
        if record and result['train']['all_training_verified'] and not (CAMPAIGN / 'ledgers/training_accounting.json').exists():
            control.finalize_training(CAMPAIGN, runner=run)
        ordinary = [row for row in result['eval']['observations'] if row['key'] not in bridge.FAILED and row['key'] not in KEYS]
        result['all_six_evaluations_verified'] = len(ordinary) == 1 and all(r.get('terminal_verified') for r in ordinary)
        result['all_six_evaluations_verified'] &= all(r.get('terminal_verified') for r in result['retry']) and all(r.get('postprocessing_verified') for r in result['postprocessing'])
        return result


def resolve_evaluation(key, *, expected_sha=None, runner=command_output):
    p, bridge, retry, control, protocol = declaration(expected_sha)
    if key not in KEYS:
        return bridge.resolve_evaluation(key, expected_sha=RETRY_DECLARATION_SHA, runner=runner)
    job = control.submitted_job(CAMPAIGN, protocol, key, role='eval')
    row = bridge.scheduler(control, {job}, lambda cmd: runner(cmd, text=True)).get(job)
    audit = retain_post_result(p, bridge, control, protocol, key, row, record=False)
    destination = SIDECARS / key.replace('/', '_')
    terminal = checked(destination / 'terminal.json')
    require(terminal == {**audit, 'postprocessing_declaration': identity(DECLARATION),
                        'receipt_sha256': digest({**audit, 'postprocessing_declaration': identity(DECLARATION)})},
            'postprocessing terminal is absent or changed')
    require(checked(destination / 'result_index.json', 'index_sha256') == audit['index_payload'], 'postprocessing index is absent or changed')
    plan = next(x for x in protocol['planned_eval_commands'] if x['key'] == key)
    return {'key': key, 'job_id': job, 'plan': plan, 'audit': audit,
            'files': {'terminal': identity(destination / 'terminal.json'), 'index': identity(destination / 'result_index.json'),
                      'launch': identity(Path(plan['output']) / 'launch_receipt.json'),
                      'details': audit['raw_details'], 'run_config': audit['run_config'], 'failed_stderr': audit['failure']['files']['stderr']},
            'replaced_pre_model_job_id': None, 'scheduler_state': 'FAILED', 'postprocessing_verified': True}


def admission_bindings(expected_sha):
    p, bridge, _, _, _ = declaration(expected_sha)
    return {'postprocessing_declaration': identity(DECLARATION), 'postprocessing_source': p['management_source'],
            **{'postprocessing_' + key: value for key, value in p['management'].items()},
            **{'v7/' + key: value for key, value in bridge.admission_bindings(RETRY_DECLARATION_SHA).items()}}


def postprocessing_allocation_records(resolved, control, runner=command_output):
    jobs = {r['job_id']: r['key'] for r in resolved if r['key'] in KEYS}
    require(set(jobs.values()) == set(KEYS), 'two exact postprocessed original FOARL allocations required')
    fields = ('JobIDRaw', 'State', 'ExitCode', 'Start', 'End', 'ElapsedRaw', 'TimelimitRaw', 'AllocTRES', 'QOS', 'Account', 'Partition')
    command = ['env', 'TZ=UTC', 'SLURM_TIME_FORMAT=standard', 'sacct', '-X', '-P', '-n', '-j', ','.join(sorted(jobs, key=int)), '--format=' + ','.join(fields)]
    raw = runner(command, text=True); records, seen = [], set()
    for values in csv.reader(raw.splitlines(), delimiter='|'):
        require(len(values) == len(fields), 'malformed postprocessing allocation')
        row = dict(zip(fields, values)); job = row['JobIDRaw']
        require(job in jobs and job not in seen and row['State'] == 'FAILED' and row['ExitCode'] == '1:0'
                and row['TimelimitRaw'] == '120' and row['QOS'] == control.QOS and row['Account'] == 'vas@a100' and row['Partition'] == 'gpu_p5',
                'postprocessing allocation scheduler/resource mismatch')
        seen.add(job); tres = {}
        for part in row['AllocTRES'].split(','):
            name, value = part.split('=', 1)
            require(name not in tres, 'duplicate postprocessing TRES')
            tres[name] = value
        require(tres.get('gres/gpu') == '1' and int(tres['cpu']) == 16 and float(tres['billing']) == 1., 'postprocessing actual allocation differs')
        seconds = int(row['ElapsedRaw'])
        require(0 < seconds <= 7200 and (dt.datetime.fromisoformat(row['End']) - dt.datetime.fromisoformat(row['Start'])).total_seconds() == seconds,
                'postprocessing allocation duration differs')
        audit = next(r['audit'] for r in resolved if r['job_id'] == job)
        require(seconds == audit['failure']['scheduler']['elapsed_seconds'], 'postprocessing failed job accounting differs from admitted evidence')
        records.append({**row, 'key': jobs[job], 'allocated_gpus': 1, 'allocated_gpu_seconds': seconds, 'allocated_gpu_hours': seconds / 3600,
                        'allocated_cpu_seconds': int(tres['cpu']) * seconds, 'billing_unit_seconds': float(tres['billing']) * seconds})
    require(seen == set(jobs), 'missing postprocessing allocation')
    return {'jobs': records, 'sacct_command': command, 'sacct_utc': raw,
            'allocated_gpu_hours': sum(r['allocated_gpu_seconds'] for r in records) / 3600}


def summarize(expected_sha):
    p, bridge, _, control, protocol = declaration(expected_sha)
    accounting = control.finalize_training(CAMPAIGN)
    resolved = [resolve_evaluation(item['key'], expected_sha=expected_sha) for item in protocol['planned_eval_commands']]
    summary = module('postprocessing_unchanged_statistics', TRAINING_SOURCE / 'scripts/summarize_clasp_grpo_fresh_dev.py')
    summary.METHODS = control.METHODS
    loaded, identities, _ = summary.collect([Path(r['files']['index']['path']) for r in resolved])
    rows, comparisons = [], {}
    for task in control.TASKS:
        for method in control.METHODS:
            rows.append(summary.summarize(list(loaded[task, method].records.values()), task, method))
        for left, right in (('grpo_z', 'grpo_verified'), ('grpo_z', 'foarl_f638_dev'), ('foarl_f638_dev', 'grpo_verified')):
            a, b = loaded[task, left], loaded[task, right]
            comparisons[task + '/' + left + '_vs_' + right] = {'strict_comparison': summary.strict.build_receipt(a, b, left_label=left, right_label=right),
                                                              'unconditional_solved_quality': summary.paired_quality(a, b)}
    success = control.allocation_records({r['job_id']: r['key'] for r in resolved if r['key'] not in KEYS}, gpus=1)
    post_cost = postprocessing_allocation_records(resolved, control)
    failed = bridge.failed_allocation_records(control)
    payload = {'status': 'six_raw_arms_verified_with_preserved_pre_model_failures_and_postprocessing_only_foarl_admission',
               'source_manifest': protocol['source_manifest'], 'protocol_sha256': protocol['receipt_sha256'],
               'postprocessing_declaration': identity(DECLARATION), 'retry_declaration': identity(bridge.DECLARATION),
               'amendment_bindings': admission_bindings(expected_sha), 'summary': rows, 'comparisons': comparisons,
               'result_indices': identities, 'resolved_evaluations': resolved, 'training_accounting': accounting,
               'successful_evaluation_accounting': success, 'postprocessed_failed_foarl_allocation_accounting': post_cost,
               'excluded_pre_model_failure_accounting': failed,
               'total_evaluation_allocated_gpu_hours_including_failures': success['allocated_gpu_hours'] + post_cost['allocated_gpu_hours'] + failed['allocated_gpu_hours'],
               'limitations': ['All scientific evaluator/model/data/generation bytes unchanged; raw K8 outputs receive no assistance.',
                              'Three pre-model failures remain excluded with costs; two complete FOARL raw outputs retain FAILED1:0 scheduler status and full allocation cost.',
                              'Only three previously validated checkpoint-local KL proof files are added to an in-memory audit view; no originals change and no FOARL inference repeats.',
                              'One training seed and reused development100/task; not confirmation, SOTA, or literal Jiang reproduction.',
                              'Own-success gaps and paired joint-success gaps have distinct denominators; CVRP references are verified heuristics.']}
    require(not COMPARISON.exists(), 'postprocessing comparison already exists')
    COMPARISON.mkdir()
    report = bridge.create(COMPARISON / 'comparison.json', payload)
    columns = ('task', 'method', 'feasible_at_8_percent', 'candidate_feasibility_percent', 'mean_feasible_gap_percent', 'mean_signed_reference_gap_percent', 'unconditional_solved_quality')
    with (COMPARISON / 'table.csv').open('x') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows({k: row[k] for k in columns} for row in rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'status', 'run', 'watch', 'summarize'))
    parser.add_argument('--source', type=Path); parser.add_argument('--source-sha'); parser.add_argument('--declaration-sha')
    parser.add_argument('--submit', action='store_true'); parser.add_argument('--max-new-jobs', type=int, default=5)
    parser.add_argument('--hours', type=float, default=6)
    args = parser.parse_args()
    if args.action == 'init':
        require(not args.submit and args.source and args.source_sha, 'init only creates the reviewed management declaration')
        result = initialize(args.source, args.source_sha)
    elif args.action == 'summarize':
        require(not args.submit and args.declaration_sha, 'summary requires the explicit declaration and never submits')
        result = summarize(args.declaration_sha)
    else:
        require(args.declaration_sha and (args.action != 'status' or not args.submit), 'status is read-only; an explicit declaration hash is required')
        require(0 < args.hours <= 6, 'watch is bounded to six hours')
        deadline = time.monotonic() + args.hours * 3600
        while True:
            result = supervisor_tick(args.declaration_sha, submit=args.submit, record=args.action != 'status', max_new_jobs=args.max_new_jobs)
            if args.action != 'watch' or result['all_six_evaluations_verified'] or time.monotonic() >= deadline:
                break
            print(json.dumps(result, sort_keys=True), flush=True); time.sleep(60)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
