#!/usr/bin/env python3
"""Prospectively bind and evaluate six fixed endpoints on the new heldout300 data.

Management only: the original evaluator and training source stay unchanged.
Development evaluator purpose is explicit; this single-seed report is neither a
built-in confirmation run nor a SOTA/paper-reproduction claim. Init never submits.
"""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True
SELF = 'scripts/grpo_incumbent_heldout300_campaign.py'
WRAPPER = 'scripts/evaluate_grpo_incumbent_heldout300_a100.slurm'
STATISTICS = 'scripts/grpo_incumbent_heldout300_statistics.py'
TRAINING_SOURCE = Path('/lustre/fsn1/projects/rech/vas/uag72kz/grpo_incumbent_source_20260907_v3/source-snapshot')
TRAINING_SOURCE_SHA = 'b960cd574f4cd605028518ba15b7a55d6277cc319b5e17b64cf8608e74cda10f'
TRAINING_CAMPAIGN = Path('/lustre/fsn1/projects/rech/vas/uag72kz/grpo_incumbent_foarl_s43_b9_dev_20260907_v1')
PLAN_SHA = '51cebb1d8e2085e7c9a7bf5a37274f34dfbdbf28d14ac5ab27d06c26c3b53dc1'
BUILDER = 'scripts/build_grpo_incumbent_holdout_successor_20260907.py'
BUILDER_SHA = '9bdc676704457daf4d70e526ed9eeee0218935d70454e847b59f0f9c39fac692'
RETRY_SOURCE = Path('/lustre/fsn1/projects/rech/vas/uag72kz/grpo_incumbent_source_20260907_v7/source-snapshot')
RETRY_SOURCE_SHA = '37c1fb23d34ddf36effa8037dc9a3bc884f840682ba3c1d5ce23bf07654031be'
RETRY_HELPER = 'scripts/grpo_incumbent_dev_evaluation_retry.py'
RETRY_HELPER_SHA = '0a70a06d67d567c2668821340c4a7cd6c32440a51ac9f0e351fd57a0a0920a8a'
RETRY_DECLARATION_SHA = '61a15f9d0acd951b12d529dc9f508c8bdb788fd6150f3498fa99767ad158cc79'
POSTPROCESSING_SOURCE = Path('/lustre/fsn1/projects/rech/vas/uag72kz/grpo_incumbent_source_20260907_v9/source-snapshot')
POSTPROCESSING_SOURCE_SHA = '3c899b08ef859eb50409172fcc83a31d879cc152c8130792107074cd4951a8ce'
POSTPROCESSING_HELPER = 'scripts/grpo_incumbent_foarl_postprocessing.py'
POSTPROCESSING_HELPER_SHA = 'da282229ca7c7544616eedfaf8a5528cf31083d33f5db14967ee1777cb1805d2'
POSTPROCESSING_DECLARATION_SHA = 'ffca44aa1db341d1b3ebb3f1e97bab57a4dc66f8fc7e633a31c51656321cb14b'
HELDOUT_OUTPUT = Path('/lustre/fsn1/projects/rech/vas/uag72kz/grpo_incumbent_heldout300_s43_rawk8_20260907_v1')
AUDIT_WRAPPER_SHA = 'bd7fce18120f9ba4776ec75995137d025bfcc8625381fb84b67794ed44e6d993'
AUDIT_SLURM_SHA = '323d0f99ca888146e5d5bc69acac1b18462a7b221a7b289d469374b9967be2ad'
RUNTIME = Path('/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env')
TASKS = ('cvrp', 'mis')
METHODS = ('grpo_z', 'grpo_verified', 'foarl_f638_dev')
GENERATION = {'seed': 271828, 'dtype': 'float16', 'num_return_sequences': 8, 'generation_batch_size': 8,
              'do_sample': True, 'temperature': .7, 'top_p': .9, 'top_k': 50, 'repetition_penalty': 1.,
              'max_new_tokens': 3000, 'max_prompt_length': 29768}
ENVIRONMENT = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'WANDB_MODE': 'offline',
    'TOKENIZERS_PARALLELISM': 'false', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
    'CLASP_TSP_PROMPT_KNN': '0', 'CLASP_TSP_INPUT_ORDER': '', 'CLASP_TSP_RELABEL_ORDER': '',
    'CLASP_TSP_ROUTE_CANONICALIZE': '', 'CLASP_TSP_LABEL_TTA_SCHEDULE': '', 'CLASP_TSP_PROMPT_KNN_DISTANCES': '0',
    'CLASP_TSP_ROUTE_ONLY_OUTPUT': '0', 'CLASP_MIS_RELABEL_ORDER': '', 'CLASP_MIS_JIANG_RESPONSE_OUTPUT': '0',
    'CLASP_MIS_PROMPT_NODE_DEGREES': '0', 'CLASP_MIS_PROMPT_LOW_DEGREE_K': '32', 'CLASP_OP_STRICT_BUDGET_HINT': '0',
    'CLASP_OP_PROMPT_BUDGET_SCALE': '1.0', 'CLASP_OP_PROMPT_BUDGET_SCALE_MIN_SIZE': '0',
    'CLASP_OP_PROMPT_BUDGET_SCALE_MAX_SIZE': '0', 'CLASP_OP_PROMPT_BUDGET_SCALE_SIZE_RANGES': '',
    'CLASP_OP_SIZE_BAND_TOP_P_SCHEDULE': '', 'CLASP_OP_SIZE_BAND_TOP_P_MIN_SIZE': '0', 'CLASP_OP_SIZE_BAND_TOP_P_MAX_SIZE': '0'}
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING', 'REQUEUED'}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return {'path': str(path), 'sha256': sha.hexdigest(), 'size_bytes': path.stat().st_size}


def checked(path, field='receipt_sha256'):
    value = json.loads(Path(path).read_text())
    require(value.get(field) == digest({k: v for k, v in value.items() if k != field}), 'invalid self-hash: ' + str(path))
    return value


def write_new(path, value, field='receipt_sha256'):
    value = {**value, field: digest(value)}
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    Path(path).chmod(0o440)
    return value


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def old_control():
    require(identity(TRAINING_SOURCE / 'SOURCE_MANIFEST.sha256')['sha256'] == TRAINING_SOURCE_SHA, 'original training source pin differs')
    sys.path.insert(0, str(TRAINING_SOURCE / 'src'))
    return module('heldout_original_campaign_control', TRAINING_SOURCE / 'scripts/grpo_incumbent_dev_campaign.py')


def validate_source(source, sha, control):
    source = Path(source).resolve(strict=True)
    observed = control.source_files(source, sha)
    require(all(name in control.manifest_entries(source) for name in (SELF, WRAPPER, STATISTICS,
            'tests/test_grpo_incumbent_heldout300_campaign.py', 'tests/test_grpo_incumbent_heldout300_statistics.py')),
            'heldout management/test files are not all in the sealed snapshot')
    require(identity(__file__) == identity(source / SELF), 'executing heldout manager is outside its declared snapshot')
    scientific = lambda entries: {name: value for name, value in entries.items()
                                  if name.startswith('src/clasp/') and name.endswith('.py') or name == control.LAUNCHER}
    require(scientific(control.manifest_entries(source)) == scientific(control.manifest_entries(TRAINING_SOURCE)),
            'heldout scientific code differs from the completed performance campaign')
    return observed


def validate_holdout(root, expected_file_sha, source):
    root = Path(root).resolve(strict=True)
    heldout_file = identity(root / 'heldout_manifest.json')
    require(heldout_file['sha256'] == expected_file_sha, 'heldout manifest differs from prospective pin')
    heldout = checked(root / 'heldout_manifest.json', 'heldout_manifest_sha256')
    plan = checked(root / 'plan.json', 'plan_sha256')
    built = checked(root / 'build_manifest.json', 'manifest_sha256')
    audit = checked(root / 'overlap_audit.json', 'audit_sha256')
    require(plan['plan_sha256'] == PLAN_SHA and heldout['plan_sha256'] == built['plan_sha256'] == audit['plan_sha256'] == PLAN_SHA,
            'heldout whole-dataset successor plan binding differs')
    require(heldout['status'] == 'frozen_new_two_task_holdout_requires_model_predeclaration'
            and heldout['claim_eligibility'] == 'not_confirmation_or_sota'
            and heldout['model_outputs_examined'] is False
            and heldout['model_evaluation_allowed_only_after_separate_preoutput_binding'] is True,
            'heldout admission scope differs')
    require(heldout['build_manifest_sha256'] == audit['build_manifest_sha256'] == built['manifest_sha256']
            and heldout['overlap_audit_sha256'] == audit['audit_sha256'] and heldout['tasks'] == built['tasks'],
            'heldout build/audit linkage differs')
    require(audit['status'] == 'passed_zero_semantic_overlap' and audit['total_overlaps'] == 0, 'heldout overlap audit failed')
    labels = {(task, label) for task in TASKS for label in ('sft_train', 'rl_train', 'public_eval', 'inspected_dev100')}
    declared = {(row['task'], row['label']): row for row in plan['forbidden_sources']}
    require(len(plan['forbidden_sources']) == len(audit['sources']) == 8 and set(declared) == labels
            and {(row['task'], row['label']) for row in audit['sources']} == labels, 'all eight forbidden corpora must be audited')
    for row in audit['sources']:
        target = declared[row['task'], row['label']]
        require(not row['overlaps'] and row['rows'] == target['expected_rows']
                and all(row[k] == target[k] for k in ('sha256', 'size_bytes')), 'forbidden corpus identity/count differs')
    builder_path = root / 'implementation_sources' / BUILDER
    builder_identity = identity(builder_path)
    require(builder_identity['sha256'] == BUILDER_SHA
            and {k: builder_identity[k] for k in ('sha256', 'size_bytes')} == plan['source_files'][BUILDER],
            'retained heldout verifier source differs')
    for name, expected in plan['source_files'].items():
        current = identity(root / 'implementation_sources' / name)
        require({k: current[k] for k in ('sha256', 'size_bytes')} == expected, 'retained source identity differs')
    builder = module('heldout_retained_reference_verifier', builder_path)
    builder.validate_build(root)
    for module_name, relative in (('clasp.data.benchmark', 'src/clasp/data/benchmark.py'), ('clasp.rewards.co_tasks', 'src/clasp/rewards/co_tasks.py')):
        actual = identity(sys.modules[module_name].__file__)
        require({k: actual[k] for k in ('sha256', 'size_bytes')} == plan['source_files'][relative], 'loaded reference verifier bytes differ from plan')
    fresh = module('heldout_identity_functions', source / 'scripts/clasp_grpo_fresh_dev_artifacts.py')
    snapshots, ordered = {}, {}
    for task in TASKS:
        entry = heldout['tasks'][task]
        require(entry['instances'] == 300, 'heldout task count is not 300')
        for relative, expected in entry['files'].items():
            require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'unsafe heldout artifact path')
            actual = identity(root / relative)
            require({k: actual[k] for k in ('sha256', 'size_bytes')} == expected, 'heldout artifact changed')
        rows = json.loads((root / task / 'eval/test.json').read_text())
        proofs = json.loads((root / task / 'reference_evidence.json').read_text())
        require(len(rows) == len(proofs) == 300, '300 rows and reference proofs required')
        ordered[task] = []
        for i, (row, proof) in enumerate(zip(rows, proofs, strict=True)):
            require(proof['evidence_sha256'] == digest({k: v for k, v in proof.items() if k != 'evidence_sha256'}), 'invalid reference self-hash')
            semantic = fresh.semantic_id(task, row)
            require(proof['index'] == i and proof['semantic_instance_sha256'] == semantic, 'reference belongs to another instance')
            text, observed = builder.canonical_reference(task, row, row['output'], proof['native_evidence'])
            require(text == row['output'] and all(proof[k] == v for k, v in observed.items()), 'reference failed strict native replay')
            require(observed['strict_verifier_feasible'] and (task != 'mis' or observed['optimality_certified']), 'missing certified MIS optimum or feasible CVRP reference')
            ordered[task].append({'example_index': i, 'semantic_instance_sha256': semantic,
                                  'benchmark_row_sha256': fresh.row_identity(row), 'reference_output_sha256': digest(str(row['output']))})
        require(len({row['semantic_instance_sha256'] for row in ordered[task]}) == 300, 'duplicate heldout instance')
        snapshots[task] = {'dataset': identity(root / task / 'eval/test.json'), 'reference_evidence': identity(root / task / 'reference_evidence.json')}
    return {'manifest': heldout_file, 'manifest_sha256': heldout['heldout_manifest_sha256'],
            'overlap_audit': identity(root / 'overlap_audit.json'), 'audit_sha256': audit['audit_sha256'],
            'plan': identity(root / 'plan.json'), 'plan_sha256': PLAN_SHA, 'build_manifest': identity(root / 'build_manifest.json'),
            'files': snapshots, 'ordered_instances': ordered}


def retry_control():
    """Import the actual immutable v7 resolver; its __file__ is part of its contract."""
    require(identity(RETRY_SOURCE / 'SOURCE_MANIFEST.sha256')['sha256'] == RETRY_SOURCE_SHA,
            'development retry source manifest differs')
    require(identity(RETRY_SOURCE / RETRY_HELPER)['sha256'] == RETRY_HELPER_SHA,
            'development retry helper differs')
    bridge = module('heldout_immutable_v7_development_retry', RETRY_SOURCE / RETRY_HELPER)
    require(Path(bridge.__file__).resolve() == RETRY_SOURCE / RETRY_HELPER,
            'development resolver loaded from another management snapshot')
    return bridge


def postprocessing_control():
    """Import the actual v9 postprocessing resolver, retaining its v7 dependency."""
    require(re.fullmatch(r'[a-f0-9]{64}', POSTPROCESSING_DECLARATION_SHA), 'native postprocessing declaration is not pinned')
    require(identity(POSTPROCESSING_SOURCE / 'SOURCE_MANIFEST.sha256')['sha256'] == POSTPROCESSING_SOURCE_SHA,
            'development postprocessing source manifest differs')
    path = POSTPROCESSING_SOURCE / POSTPROCESSING_HELPER
    require(identity(path)['sha256'] == POSTPROCESSING_HELPER_SHA, 'development postprocessing helper differs')
    resolver = module('heldout_immutable_v9_postprocessing', path)
    require(Path(resolver.__file__).resolve() == path, 'postprocessing resolver loaded from another management snapshot')
    return resolver


def original_event_bindings(control, protocol):
    bindings = {}
    for role, directory in (('train', 'events'), ('eval', 'evaluation_events')):
        groups = control.read_events(TRAINING_CAMPAIGN, protocol, role=role)
        require(len(groups) == 6 and all(len(group) == 2 and {e['kind'] for e in group} == {'intent', 'submitted'}
                                       for group in groups.values()), 'all original accepted event pairs are required')
        for path in sorted((TRAINING_CAMPAIGN / 'ledgers' / directory).glob('*.json')):
            require(path.is_file() and not path.is_symlink(), 'original submission evidence must be regular files')
            bindings[role + '/' + path.name] = identity(path)
    require(len(bindings) == 24, 'original submission-event inventory differs')
    return bindings


def verify_development_admission(admission):
    """Rehash original and replacement development proofs after model predeclaration."""
    development = admission['development_evaluation_admission']
    resolver = postprocessing_control()
    require(development['declaration']['sha256'] == POSTPROCESSING_DECLARATION_SHA
            and development['retry_declaration']['sha256'] == RETRY_DECLARATION_SHA
            and resolver.admission_bindings(POSTPROCESSING_DECLARATION_SHA) == development['amendment_bindings'],
            'development amendment/source/failure/event evidence changed')
    control = old_control()
    protocol, source = control.protocol_for(TRAINING_CAMPAIGN)
    require(source == TRAINING_SOURCE and original_event_bindings(control, protocol) == development['original_event_bindings'],
            'original submission events changed after heldout admission')
    for file in (development['comparison'], *development['files'].values()):
        require(identity(file['path']) == file, 'development launch/execution/index/terminal/raw evidence changed')


def admit_models(control):
    protocol, source = control.protocol_for(TRAINING_CAMPAIGN)
    require(source == TRAINING_SOURCE and protocol['source_manifest']['sha256'] == TRAINING_SOURCE_SHA, 'wrong original campaign source')
    accounting = control.signed(TRAINING_CAMPAIGN / 'ledgers/training_accounting.json')
    require(accounting['status'] == 'all_six_training_jobs_verified' and accounting['protocol_sha256'] == protocol['receipt_sha256'],
            'all six fixed training endpoints must be finalized before heldout admission')
    bridge = retry_control()
    resolver = postprocessing_control()
    amendment, _, _, _, _ = resolver.declaration(POSTPROCESSING_DECLARATION_SHA)
    require(Path(amendment['management_source']['path']).parent == POSTPROCESSING_SOURCE
            and amendment['management_source']['sha256'] == POSTPROCESSING_SOURCE_SHA
            and amendment['retry_declaration'] == identity(bridge.DECLARATION), 'development amendment uses another source')
    audits, eval_jobs, resolved, files, original_jobs = {}, {}, {}, {}, {}
    for plan in protocol['planned_train_commands']:
        key = plan['key']
        audits[key] = control.training_endpoint(TRAINING_CAMPAIGN, key, protocol, source)
        original_job = control.submitted_job(TRAINING_CAMPAIGN, protocol, key, role='eval')
        evaluation = resolver.resolve_evaluation(key, expected_sha=POSTPROCESSING_DECLARATION_SHA)
        require(evaluation['key'] == key, 'development resolver did not verify the exact arm')
        if key in resolver.KEYS:
            require(evaluation['audit']['status'] == 'complete_raw_evaluation_verified_after_exact_final_inventory_failure'
                    and evaluation.get('postprocessing_verified') is True and evaluation.get('scheduler_state') == 'FAILED'
                    and evaluation['audit']['scheduler_state'] == 'FAILED' and evaluation['audit']['scheduler_exit_code'] == '1:0'
                    and evaluation['job_id'] == original_job and evaluation['replaced_pre_model_job_id'] is None,
                    'FOARL must retain its original failed job with complete strictly verified raw outputs')
        else:
            require(evaluation['audit']['status'] == 'terminal_evaluation_verified', 'development resolver did not verify the exact arm')
        if key in bridge.FAILED:
            require(original_job == evaluation['replaced_pre_model_job_id'] == bridge.FAILED[key]['job_id']
                    and evaluation['job_id'] != original_job, 'failed and replacement development jobs differ from declaration')
        elif key not in resolver.KEYS:
            require(evaluation['job_id'] == original_job and evaluation['replaced_pre_model_job_id'] is None,
                    'healthy original evaluation was replaced')
        require(evaluation['job_id'] not in eval_jobs, 'development job reused across arms')
        eval_jobs[evaluation['job_id']] = key
        original_jobs[original_job] = key
        resolved[key] = evaluation
        for name, binding in evaluation['files'].items():
            files[key + '/' + name] = binding
    expected = {(task + '/' + method) for task in TASKS for method in METHODS}
    require(set(audits) == set(resolved) == expected and len(eval_jobs) == len(original_jobs) == 6, 'all six fixed methods/tasks required')
    require(accounting['terminal_audits'] == [identity(TRAINING_CAMPAIGN / 'ledgers/terminal' / (plan['key'].replace('/', '_') + '.json'))
                                            for plan in protocol['planned_train_commands']], 'original finalized terminal metadata changed')
    pairings = [control.compare_actual_configuration(task, audits[task + '/grpo_z'], audits[task + '/' + method], protocol, source)
                for task in TASKS for method in METHODS[1:]]
    for task in TASKS:
        for name in ('added_tokens.json', 'merges.txt', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json'):
            require(len({audits[task + '/' + method]['checkpoint'][name]['sha256'] for method in METHODS}) == 1,
                    'paired tokenizer bytes differ before heldout admission')
    comparison_path = TRAINING_CAMPAIGN / 'comparison_postprocessing_v1/comparison.json'
    comparison = checked(comparison_path)
    require(comparison['status'] == 'six_raw_arms_verified_with_preserved_pre_model_failures_and_postprocessing_only_foarl_admission'
            and comparison['protocol_sha256'] == protocol['receipt_sha256']
            and comparison['source_manifest'] == protocol['source_manifest']
            and comparison['retry_declaration'] == identity(bridge.DECLARATION)
            and comparison['postprocessing_declaration'] == identity(resolver.DECLARATION)
            and comparison['resolved_evaluations'] == [resolved[plan['key']] for plan in protocol['planned_eval_commands']],
            'amended development comparison differs from current strict endpoints')
    amendment_bindings = resolver.admission_bindings(POSTPROCESSING_DECLARATION_SHA)
    require(comparison['amendment_bindings'] == amendment_bindings, 'development comparison amendment bindings changed')
    successful_allocations = control.allocation_records({job: key for job, key in eval_jobs.items() if key not in resolver.KEYS}, gpus=1)
    postprocessed_allocations = resolver.postprocessing_allocation_records(list(resolved.values()), control)
    failed_allocations = bridge.failed_allocation_records(control)
    require(comparison['successful_evaluation_accounting'] == successful_allocations
            and comparison['postprocessed_failed_foarl_allocation_accounting'] == postprocessed_allocations
            and comparison['excluded_pre_model_failure_accounting'] == failed_allocations,
            'development comparison allocation accounting changed')
    return {'training_protocol': identity(TRAINING_CAMPAIGN / 'ledgers/protocol.json'),
            'training_accounting': identity(TRAINING_CAMPAIGN / 'ledgers/training_accounting.json'),
            'terminal_audits': audits, 'paired_actual_configurations': pairings,
            'training_allocations': control.allocation_records({audit['job_id']: key for key, audit in audits.items()}, gpus=4),
            'development_evaluation_allocations': successful_allocations,
            'postprocessed_failed_foarl_allocations': postprocessed_allocations,
            'excluded_pre_model_failure_allocations': failed_allocations,
            'development_evaluation_admission': {'declaration': identity(resolver.DECLARATION), 'retry_declaration': identity(bridge.DECLARATION),
                'amendment_bindings': amendment_bindings, 'original_event_bindings': original_event_bindings(control, protocol),
                'comparison': identity(comparison_path), 'files': files, 'original_job_ids': original_jobs,
                'resolved_job_ids': eval_jobs, 'resolved_evaluations': resolved}}


def validate_native_audit(bundle, manifest_sha, job_id, benchmark, runner=subprocess.check_output):
    bundle = Path(bundle).resolve(strict=True)
    require(re.fullmatch(r'[1-9][0-9]*', job_id), 'native audit requires an explicitly bound accepted job ID')
    manifest_file = identity(bundle / 'bundle_manifest.json')
    require(manifest_file['sha256'] == manifest_sha, 'native audit bundle pin differs')
    manifest = json.loads((bundle / 'bundle_manifest.json').read_text())
    start = json.loads((bundle / 'execution/start.json').read_text())
    end = json.loads((bundle / 'execution/exit.json').read_text())
    require(manifest['protocol'] == start['protocol'] == 'grpo-incumbent-native-holdout-overlap-audit-v1'
            and manifest['plan_sha256'] == start['plan_sha256'] == PLAN_SHA
            and manifest['build_manifest_sha256'] == start['build_manifest_sha256']
            == checked(benchmark['build_manifest']['path'], 'manifest_sha256')['manifest_sha256'], 'native audit plan/build scope differs')
    require(manifest['files']['run_native_holdout_audit_v1.py']['sha256'] == AUDIT_WRAPPER_SHA
            and manifest['files']['audit_native_holdout_v1.slurm']['sha256'] == AUDIT_SLURM_SHA,
            'native overlap wrapper/Slurm source differs from reviewed implementation')
    require(start['slurm']['SLURM_JOB_ID'] == job_id and start['bundle_manifest_sha256'] == end['bundle_manifest_sha256'] == manifest_sha,
            'native audit execution/job binding differs')
    require(start['python_flags'] == {'isolated': 1, 'no_site': 1, 'dont_write_bytecode': 1}, 'native audit runtime isolation differs')
    require(end['status'] == 'passed' and end['exit_code'] == 0 and end['frozen_inputs_unchanged'] is True
            and end['sources'] == 8 and end['rows'] == 1006000 and end['total_overlaps'] == 0
            and end['audit_sha256'] == benchmark['audit_sha256'], 'native eight-source audit did not finish successfully')
    require(end['overlap_receipt_file'] == {k: benchmark['overlap_audit'][k] for k in ('sha256', 'size_bytes')},
            'heldout overlap receipt differs from the successful native audit')
    profile = start['resource_profile']
    fixed = {'name': 'one_reserved_a100_cpu_only_scan_v3', 'requested_cpus_per_task': 2,
             'reserved_gpus': 1, 'cpu_scan_threads': 1, 'cuda_visible_devices': '', 'gpu_compute_requested': False}
    require(all(profile.get(k) == v and type(profile[k]) is type(v) for k, v in fixed.items())
            and end['resource_profile'] == profile, 'native audit reserved-A100 CPU-only profile differs')
    require(start['slurm']['SLURM_CPUS_PER_TASK'] == '2' and start['slurm']['SLURM_JOB_ACCOUNT'] == 'vas@a100'
            and start['slurm']['SLURM_JOB_PARTITION'] == 'gpu_p5', 'native audit requested allocation differs')
    job_profile, node_profile = profile['slurm_job'], profile['slurm_node']
    require(job_profile['JobId'] == job_id and job_profile['Account'] == 'vas@a100'
            and job_profile['QOS'] == 'qos_gpu_a100-dev' and job_profile['Partition'] == 'gpu_p5'
            and job_profile['CPUs/Task'] == '2' and job_profile['NumTasks'] == job_profile['NumNodes'] == '1',
            'native audit scontrol job/profile differs')
    require(job_profile['NodeList'] == node_profile['NodeName']
            and 'a100' in node_profile['AvailableFeatures'].split(','), 'native audit node is not the declared A100 allocation')

    def tres_fields(text):
        parts = [part.split('=', 1) for part in text.split(',')]
        require(all(len(part) == 2 and all(part) for part in parts) and len({p[0] for p in parts}) == len(parts),
                'native audit allocation TRES are malformed')
        return dict(parts)

    def memory_quantity(text):
        match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGTPE]?)', text)
        require(match is not None, 'native audit memory allocation is malformed')
        return float(match[1]) * 1024 ** ('KMGTPE'.index(match[2]) + 1 if match[2] else 0)

    start_tres = tres_fields(job_profile['AllocTRES'])
    allocated_cpus = profile['actual_allocated_cpus']
    require(type(allocated_cpus) is int and allocated_cpus >= 2 and int(start_tres['cpu']) == allocated_cpus
            and job_profile['NumCPUs'] == str(allocated_cpus) and start_tres['gres/gpu'] == '1'
            and profile['actual_allocated_memory'] == start_tres['mem'], 'native audit actual allocation profile differs')
    command = ['env', 'TZ=UTC', 'SLURM_TIME_FORMAT=standard', 'sacct', '-X', '-P', '-n', '-j', job_id,
               '--format=JobIDRaw,State,ExitCode,ElapsedRaw,TimelimitRaw,AllocTRES,QOS,Account,Partition']
    rows = [row for row in csv.reader(runner(command, text=True).splitlines(), delimiter='|') if row]
    require(len(rows) == 1 and len(rows[0]) == 9, 'native audit allocation unavailable or ambiguous')
    job, state, code, elapsed, limit, tres, qos, account, partition = rows[0]
    require(job == job_id and state == 'COMPLETED' and code == '0:0' and limit == '120'
            and 0 < int(elapsed) <= 7200 and qos == 'qos_gpu_a100-dev' and account == 'vas@a100' and partition == 'gpu_p5',
            'native audit reserved-A100 job did not complete under its declared budget')
    terminal_tres = tres_fields(tres)
    require(terminal_tres['gres/gpu'] == '1' and int(terminal_tres['cpu']) == allocated_cpus
            and memory_quantity(terminal_tres['mem']) == memory_quantity(start_tres['mem'])
            and float(terminal_tres['billing']) == float(start_tres['billing']),
            'native audit start/terminal allocation differs')
    billing_units = float(terminal_tres['billing'])
    require(math.isfinite(billing_units) and billing_units > 0, 'native audit Slurm billing units are invalid')
    wrapper_seconds = end['elapsed_seconds']
    require(type(wrapper_seconds) in (float, int) and math.isfinite(wrapper_seconds) and 0 < wrapper_seconds <= int(elapsed) + 2
            and end['full_job_allocation_cost_requires_slurm_terminal_accounting'] is True
            and math.isclose(end['reserved_gpu_seconds_during_wrapper'], wrapper_seconds)
            and math.isclose(end['reserved_gpu_hours_during_wrapper'], wrapper_seconds / 3600)
            and math.isclose(end['allocated_cpu_seconds_during_wrapper'], allocated_cpus * wrapper_seconds),
            'native audit wrapper allocation-cost evidence differs')
    files = {name: identity(bundle / name) for name in ('bundle_manifest.json', 'run_native_holdout_audit_v1.py',
             'audit_native_holdout_v1.slurm', 'execution/start.json', 'execution/exit.json', 'execution/runtime_imports.json', 'data/overlap_audit.json')}
    for name in ('run_native_holdout_audit_v1.py', 'audit_native_holdout_v1.slurm'):
        require({k: files[name][k] for k in ('sha256', 'size_bytes')} == manifest['files'][name], 'native audit implementation bytes changed')
    require(all(files['data/overlap_audit.json'][k] == benchmark['overlap_audit'][k] for k in ('sha256', 'size_bytes')),
            'native and evaluation overlap receipt bytes differ')
    return {'status': 'successful_native_reserved_a100_cpu_overlap_audit_bound', 'job_id': job_id, 'files': files,
            'resource_profile': profile, 'sacct_command': command, 'sacct_row': rows[0],
            'allocated_cpus': allocated_cpus, 'allocated_cpu_seconds': allocated_cpus * int(elapsed),
            'allocated_gpu_count': 1, 'allocated_gpu_seconds': int(elapsed), 'allocated_gpu_hours': int(elapsed) / 3600,
            'slurm_billing_units': billing_units, 'slurm_billing_unit_seconds': billing_units * int(elapsed),
            'slurm_billing_unit_hours': billing_units * int(elapsed) / 3600,
            'cost_interpretation': 'Reserved/allocated resources, not measured utilization; allocated GPU-hours and Slurm billing-unit hours are distinct from independently reconciled project charges.'}


def evaluator_argv(plan, data):
    return [str(RUNTIME / 'bin/python'), '-m', 'clasp.entrypoints.evaluate', '--evaluation_purpose', 'development',
            '--checkpoint_path', plan['checkpoint'], '--stage', 'grpo', '--data_root', str(data), '--tasks', plan['task'],
            '--eval_start_index', '0', '--max_eval_samples_per_task', '300', '--no-prefer_subset_files', '--seed', '271828',
            '--dtype', 'float16', '--num_return_sequences', '8', '--generation_batch_size', '8', '--do_sample',
            '--temperature', '0.7', '--top_p', '0.9', '--top_k', '50', '--repetition_penalty', '1.0',
            '--max_new_tokens', '3000', '--max_prompt_length', '29768', '--prompt_style', 'auto', '--force_no_reason_prompt',
            '--save_results', '--save_details', '--inference_root', str(Path(plan['output']) / 'eval'),
            '--eval_name', plan['key'].replace('/', '_') + '_heldout300_seed271828_raw_k8']


def planned_jobs(output, source, data, admission):
    result = []
    for task in TASKS:
        for method in METHODS:
            key = task + '/' + method
            audit = admission['terminal_audits'][key]
            checkpoint = str(Path(audit['checkpoint']['adapter_model.safetensors']['path']).parent)
            plan = {'key': key, 'task': task, 'method': method, 'checkpoint': checkpoint,
                    'training_root': str(Path(checkpoint).parent.parent), 'training_job_id': audit['job_id'],
                    'output': str(output / 'evaluations' / key.replace('/', '_')),
                    'stdout': str(output / 'logs' / (key.replace('/', '_') + '_%j.out'))}
            env = {'HELDOUT_CAMPAIGN': str(output), 'HELDOUT_SOURCE': str(source), 'HELDOUT_KEY': key}
            require(all(',' not in v and not any(c.isspace() for c in v) for v in env.values()), 'unsupported export path')
            plan['command'] = ['sbatch', '--parsable', '--job-name=heldout300-' + key.replace('/', '-'),
                               '--output=' + plan['stdout'], '--error=' + plan['stdout'].removesuffix('.out') + '.err',
                               '--export=ALL,' + ','.join(k + '=' + v for k, v in env.items()), str(source / WRAPPER)]
            plan['evaluator_argv'] = evaluator_argv(plan, data)
            result.append(plan)
    return result


def initialize(output, source, source_sha, data, heldout_sha, audit_bundle, audit_manifest_sha, audit_job_id):
    output, source, data = Path(output).resolve(), Path(source).resolve(strict=True), Path(data).resolve(strict=True)
    require(output == HELDOUT_OUTPUT and not output.exists(), 'heldout campaign must use the fixed never-created output root')
    control = old_control()
    source_manifest = validate_source(source, source_sha, control)
    benchmark = validate_holdout(data, heldout_sha, source)
    native_audit = validate_native_audit(audit_bundle, audit_manifest_sha, audit_job_id, benchmark)
    admission = admit_models(control)
    plan = checked(data / 'plan.json', 'plan_sha256')
    payload = {'status': 'six_fixed_step100_models_bound_before_first_heldout_output', 'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'claim_eligibility': 'not_confirmation_or_sota', 'output_root': str(output), 'data_root': str(data),
               'source_manifest': source_manifest, 'training_source_manifest': identity(TRAINING_SOURCE / 'SOURCE_MANIFEST.sha256'),
               'management': {name: identity(source / name) for name in (SELF, WRAPPER, STATISTICS)},
               'runtime_python': identity(RUNTIME / 'bin/python'), 'benchmark': benchmark, 'model_admission': admission,
               'native_overlap_audit': native_audit,
               'generation': GENERATION, 'environment': ENVIRONMENT, 'statistical_plan': plan['analysis'],
               'claim_scope': 'Raw one-seed heldout generalization on two tasks; no SOTA, built-in confirmation, or literal FOARL paper reproduction.',
               'model_selection': 'All three predeclared methods at seed43 step100; no heldout tuning, replacement, retry or adaptive selection.',
               'planned_jobs': planned_jobs(output, source, data, admission)}
    output.mkdir()
    for directory in ('logs', 'evaluations', 'ledgers', 'ledgers/events', 'ledgers/terminal'):
        (output / directory).mkdir()
    return write_new(output / 'ledgers/protocol.json', payload)


def protocol_for(output):
    output = Path(output).resolve(strict=True)
    p = checked(output / 'ledgers/protocol.json')
    require(output == HELDOUT_OUTPUT and p['output_root'] == str(output)
            and p['claim_eligibility'] == 'not_confirmation_or_sota', 'heldout campaign binding differs')
    source = Path(p['source_manifest']['path']).parent
    validate_source(source, p['source_manifest']['sha256'], old_control())
    require(p['generation'] == GENERATION and p['environment'] == ENVIRONMENT, 'heldout decoding/environment changed')
    require(p['statistical_plan'] == checked(Path(p['data_root']) / 'plan.json', 'plan_sha256')['analysis'], 'heldout statistical predeclaration changed')
    require(p['management'] == {name: identity(source / name) for name in (SELF, WRAPPER, STATISTICS)}
            and p['runtime_python'] == identity(RUNTIME / 'bin/python'), 'heldout management/runtime changed')
    require(p['planned_jobs'] == planned_jobs(output, source, Path(p['data_root']), p['model_admission']), 'heldout command/argv changed')
    return p, source


def verify_bound_inputs(p, source, key=None):
    require(validate_holdout(Path(p['data_root']), p['benchmark']['manifest']['sha256'], source) == p['benchmark'], 'heldout benchmark changed')
    admission = p['model_admission']
    verify_development_admission(admission)
    for file in p['native_overlap_audit']['files'].values():
        require(identity(file['path']) == file, 'native overlap-audit execution evidence changed')
    for name in ('training_protocol', 'training_accounting'):
        require(identity(admission[name]['path']) == admission[name], 'original campaign admission changed')
    for name, audit in admission['terminal_audits'].items():
        if key is not None and name != key:
            continue
        require(checked(TRAINING_CAMPAIGN / 'ledgers/terminal' / (name.replace('/', '_') + '.json')) == audit,
                'predeclared model differs from its original verified terminal ledger')
        for file in (*audit['checkpoint'].values(), audit['contract'], audit['hyperparams'], audit['policy_consistency']):
            require(identity(file['path']) == file, 'predeclared model/metadata bytes changed: ' + name)
        if audit['exact_reference_evidence'] is not None:
            for file in audit['exact_reference_evidence']['files'].values():
                require(identity(file['path']) == file, 'predeclared FOARL reference evidence changed')


def read_events(output, p):
    plans = {plan['key']: plan for plan in p['planned_jobs']}
    grouped = {key: [] for key in plans}
    for path in sorted((output / 'ledgers/events').glob('*.json')):
        event = checked(path, 'event_sha256')
        require(event['key'] in plans and event['kind'] in ('intent', 'submitted', 'rejected')
                and event['protocol_sha256'] == p['receipt_sha256'] and event['command'] == plans[event['key']]['command'], 'heldout submission binding differs')
        grouped[event['key']].append(event)
    ids = []
    for key, events in grouped.items():
        if not events:
            continue
        intents = [e for e in events if e['kind'] == 'intent']
        endings = [e for e in events if e['kind'] != 'intent']
        require(len(intents) == 1 and len(endings) <= 1 and all(e['transaction'] == intents[0]['transaction'] for e in events), 'multiple or inconsistent heldout attempts')
        for event in endings:
            if event['kind'] == 'submitted':
                require(re.fullmatch(r'[1-9][0-9]*', event['job_id']), 'invalid heldout job ID')
                ids.append(event['job_id'])
    require(len(ids) == len(set(ids)), 'one job cannot bind multiple heldout arms')
    return grouped


def submitted_job(output, p, key):
    accepted = [e for e in read_events(output, p)[key] if e['kind'] == 'submitted']
    require(len(accepted) == 1, 'exactly one accepted heldout submission is required')
    return accepted[0]['job_id']


def launch(output, key, running_wrapper):
    p, source = protocol_for(output)
    plan = next(row for row in p['planned_jobs'] if row['key'] == key)
    job = submitted_job(output, p, key)
    require(os.environ.get('SLURM_JOB_ID') == job, 'actual heldout job differs from accepted submission')
    require(Path(os.environ['HELDOUT_CAMPAIGN']).resolve() == output and Path(os.environ['HELDOUT_SOURCE']).resolve() == source
            and os.environ['HELDOUT_KEY'] == key, 'actual heldout worker environment differs')
    wrapper = identity(running_wrapper)
    require(all(wrapper[k] == p['management'][WRAPPER][k] for k in ('sha256', 'size_bytes')), 'actual Slurm spool wrapper bytes differ')
    require(Path(sys.executable).resolve() == Path(p['runtime_python']['path']), 'worker uses another Python runtime')
    verify_bound_inputs(p, source, key)
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0), 'heldout inference requires exactly one visible A100')
    root = Path(plan['output'])
    require(not root.exists(), 'heldout arm output already exists')
    audit = p['model_admission']['terminal_audits'][key]
    payload = {'status': 'validated_predeclared_heldout300_raw_launch', 'task': plan['task'], 'method': plan['method'], 'stage': 'grpo',
               'slurm_job_id': job, 'training_job_id': audit['job_id'], 'heldout_protocol_sha256': p['receipt_sha256'],
               'generation': GENERATION, 'evaluation_slice': {'split': 'eval', 'start': 0, 'count': 300}, 'repair': False,
               'benchmark': p['benchmark'], 'source_manifest': p['source_manifest'], 'training_contract': audit['contract'],
               'checkpoint': audit['checkpoint'], 'management': p['management'], 'runtime_python': p['runtime_python'],
               'evaluator_argv': plan['evaluator_argv'], 'environment': ENVIRONMENT, 'spool_execution_path': str(Path(running_wrapper).resolve()),
               'reference_semantics': {'cvrp': 'gap_to_verified_heuristic_reference', 'mis': 'gap_to_certified_optimum'},
               'claim_eligibility': 'not_confirmation_or_sota', 'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    root.mkdir()
    return write_new(root / 'launch_receipt.json', payload, 'launch_sha256')


def worker(output, key, running_wrapper):
    receipt = launch(output, key, running_wrapper)
    p, source = protocol_for(output)
    plan = next(row for row in p['planned_jobs'] if row['key'] == key)
    prefixes = ('CLASP_', 'PD_', 'GRPO_', 'FOARL_', 'GENERATION_', 'WANDB_', 'HF_', 'TRANSFORMERS_', 'TOKENIZERS_',
                'BITSANDBYTES_', 'UNSLOTH_', 'ACCELERATE_', 'NCCL_', 'PYTORCH_')
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes) and k not in ('PYTHONPATH', 'PYTHONHOME')}
    env.update(ENVIRONMENT, PYTHONPATH=str(source / 'src'))
    started = time.monotonic()
    code = subprocess.run(plan['evaluator_argv'], cwd=source, env=env, check=False).returncode
    write_new(Path(plan['output']) / 'execution_receipt.json', {'status': 'evaluator_process_returned', 'returncode': code,
              'elapsed_seconds': time.monotonic() - started, 'slurm_job_id': receipt['slurm_job_id'],
              'launch_sha256': receipt['launch_sha256'], 'heldout_protocol_sha256': p['receipt_sha256'],
              'evaluator_argv': plan['evaluator_argv'], 'environment': ENVIRONMENT})
    require(code == 0, 'heldout evaluator failed; no automatic retry')
    return index_arm(output, key, write=True)


def index_arm(output, key, *, write=False):
    p, source = protocol_for(output)
    plan = next(row for row in p['planned_jobs'] if row['key'] == key)
    root, job = Path(plan['output']), submitted_job(output, p, key)
    launch_receipt = checked(root / 'launch_receipt.json', 'launch_sha256')
    execution = checked(root / 'execution_receipt.json')
    require(launch_receipt['heldout_protocol_sha256'] == execution['heldout_protocol_sha256'] == p['receipt_sha256']
            and launch_receipt['slurm_job_id'] == execution['slurm_job_id'] == job
            and execution['launch_sha256'] == launch_receipt['launch_sha256'] and execution['returncode'] == 0,
            'heldout actual process/launch/job binding differs')
    require(execution['evaluator_argv'] == launch_receipt['evaluator_argv'] == plan['evaluator_argv']
            and execution['environment'] == launch_receipt['environment'] == ENVIRONMENT, 'actual heldout argv/environment differs')
    verify_bound_inputs(p, source, key)
    details = list(root.glob('eval/**/run_details__*.jsonl'))
    configs = list(root.glob('eval/**/run_config__*.json'))
    require(len(details) == len(configs) == 1, 'exactly one raw heldout result is required')
    config = checked(configs[0], 'run_config_sha256')
    strict = module('heldout_strict_raw_comparator', source / 'scripts/compare_development.py')
    artifact = strict._load_artifact(details[0], configs[0], label=key)
    require(artifact.artifact_type == 'native_raw', 'heldout assistance/rescoring/repair is not allowed')
    payload = {'status': 'complete_bound_raw_heldout300', 'task': plan['task'], 'method': plan['method'], 'candidate_budget': 8,
               'generation_seed': 271828, 'repair': False, 'evaluation_slice': {'split': 'eval', 'start': 0, 'count': 300},
               'heldout_protocol_sha256': p['receipt_sha256'], 'slurm_job_id': job,
               'launch_receipt': identity(root / 'launch_receipt.json'), 'execution_receipt': identity(root / 'execution_receipt.json'),
               'details': identity(details[0]), 'run_config': identity(configs[0]),
               'reference_semantics': 'gap_to_verified_heuristic_reference' if plan['task'] == 'cvrp' else 'gap_to_certified_optimum'}
    temporary_index = {**payload, 'index_sha256': digest(payload)}
    terminal = p['model_admission']['terminal_audits'][key]
    if plan['method'] == 'foarl_f638_dev':
        terminal = postprocessing_control().checkpoint_audit_view(plan, terminal, config)
    audit_raw300(plan, launch_receipt, config, temporary_index, terminal, job_id=job,
                 training={'source_manifest': p['source_manifest'], 'evaluation': {'data_root': p['data_root']}},
                 evaluation={'generation': GENERATION, 'management': p['management']}, benchmark=p['benchmark'], source=source, records=artifact.records)
    path = root / 'result_index.json'
    if path.exists():
        require(checked(path, 'index_sha256') == temporary_index, 'heldout result changed after indexing')
    elif write:
        write_new(path, payload, 'index_sha256')
    return temporary_index, artifact


def reconcile(output, submit=False, record=False, max_new_jobs=5, runner=subprocess.check_output):
    require(type(max_new_jobs) is int and 0 <= max_new_jobs <= 5, 'at most five new heldout jobs per tick')
    p, source = protocol_for(output)
    control = old_control()
    events = read_events(output, p)
    accepted = {e['job_id']: key for key, values in events.items() for e in values if e['kind'] == 'submitted'}
    rows = control.scheduler_rows(runner(['sacct', '-X', '-P', '-n', '-j', ','.join(accepted),
                      '--format=JobIDRaw,State,ExitCode,StdOut%600,ElapsedRaw'], text=True)) if accepted else {}
    queue = control.expanded_queue(runner(['squeue', '-r', '-h', '-u', 'uag72kz', '-q', 'qos_gpu_a100-dev', '-o', '%i|%T'], text=True)) if submit else set()
    pending = {job for job in accepted if job not in rows or rows[job]['state'] in ACTIVE}
    unresolved = sum(bool(values) and not any(e['kind'] == 'submitted' for e in values) for values in events.values())
    new, observations = set(), []
    for plan in p['planned_jobs']:
        key = plan['key']
        submitted = [e for e in events[key] if e['kind'] == 'submitted']
        if submitted:
            job = submitted[0]['job_id']; row = rows.get(job)
            if row is None:
                observations.append({'key': key, 'job_id': job, 'state': 'awaiting_accounting_no_retry'}); continue
            require(row['stdout'] in (plan['stdout'], plan['stdout'].replace('%j', job)), 'heldout scheduler stdout differs')
            result = {'key': key, **row}
            if row['state'] == 'COMPLETED' and row['exit_code'] == '0:0':
                receipt_path = output / 'ledgers/terminal' / (key.replace('/', '_') + '.json')
                if receipt_path.exists():
                    terminal = checked(receipt_path)
                    require(terminal['job_id'] == job and terminal['protocol_sha256'] == p['receipt_sha256'], 'cached heldout terminal differs')
                else:
                    index, _ = index_arm(output, key)
                    terminal = {'status': 'terminal_raw_heldout300_verified', 'key': key, 'job_id': job,
                                'protocol_sha256': p['receipt_sha256'], 'result_index': index}
                    if record:
                        write_new(receipt_path, terminal)
                result['terminal_verified'] = True
            observations.append(result); continue
        if events[key]:
            observations.append({'key': key, 'state': 'prior_attempt_no_retry'}); continue
        occupied = len(queue | pending | new) + unresolved
        if not submit or occupied >= 10 or len(new) >= max_new_jobs:
            observations.append({'key': key, 'state': 'ready', 'occupied_slots': occupied}); continue
        require(not Path(plan['output']).exists(), 'heldout output exists before first submission')
        verify_bound_inputs(p, source, key)
        event = {'key': key, 'command': plan['command'], 'protocol_sha256': p['receipt_sha256'],
                 'transaction': uuid.uuid4().hex, 'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
        directory = output / 'ledgers/events'
        write_new(directory / (uuid.uuid4().hex + '.json'), {**event, 'kind': 'intent'}, 'event_sha256')
        try:
            answer = runner(plan['command'], text=True).strip()
            require(re.fullmatch(r'[1-9][0-9]*(;[^\s;]+)?', answer), 'ambiguous heldout sbatch result')
            job = answer.split(';')[0]
            require(job not in accepted and job not in new, 'heldout job ID already bound')
        except Exception as error:
            write_new(directory / (uuid.uuid4().hex + '.json'), {**event, 'kind': 'rejected', 'error': str(error)}, 'event_sha256')
            raise
        write_new(directory / (uuid.uuid4().hex + '.json'), {**event, 'kind': 'submitted', 'job_id': job}, 'event_sha256')
        new.add(job)
        observations.append({'key': key, 'state': 'submitted', 'job_id': job})
    return {'observations': observations, 'all_six_verified': len(observations) == 6 and all(row.get('terminal_verified') for row in observations)}


def summarize(output):
    p, source = protocol_for(output)
    state = reconcile(output)
    require(state['all_six_verified'], 'all six completed heldout evaluations are required')
    control = old_control()
    verify_bound_inputs(p, source)
    strict = module('heldout_final_pair_comparator', source / 'scripts/compare_development.py')
    statistics = module('heldout_plan_v2_statistics', source / STATISTICS)
    artifacts, arms, indices = {}, {}, {}
    for plan in p['planned_jobs']:
        index, artifact = index_arm(output, plan['key'])
        task, method = plan['task'], plan['method']
        artifacts[task, method] = artifact
        indices[plan['key']] = index
        records = []
        for record in sorted(artifact.records.values(), key=lambda row: row.example_index):
            signed = None
            if record.pass_at_k:
                objective = float(record.raw['best_objective_value']); reference = float(record.raw['ground_truth_objective'])
                require(math.isfinite(objective) and math.isfinite(reference) and reference > 0, 'invalid heldout objective/reference')
                signed = (objective - reference) / reference if task == 'cvrp' else (reference - objective) / reference
            count = int(round(record.candidate_feasibility * 8))
            require(math.isclose(count / 8, record.candidate_feasibility, abs_tol=1e-12), 'candidate feasibility is not a K8 fraction')
            records.append({'example_index': record.example_index, 'feasible': bool(record.pass_at_k),
                            'candidate_count': 8, 'candidate_feasible_count': count, 'best_gap_signed': signed,
                            'best_gap_clipped': float(record.best_feasible_gap) if record.pass_at_k else None})
        arms[task, method] = records
    for task in TASKS:
        for baseline in ('grpo_z', 'foarl_f638_dev'):
            strict._validate_pairing(artifacts[task, baseline], artifacts[task, 'grpo_verified'])
    report = statistics.build_statistics(arms)
    report.update(protocol=identity(output / 'ledgers/protocol.json'), indices=indices,
                  allocation=control.allocation_records({submitted_job(output, p, plan['key']): plan['key'] for plan in p['planned_jobs']}, gpus=1),
                  heldout_benchmark=p['benchmark'])
    destination = output / 'comparison'
    destination.mkdir(exist_ok=False)
    receipt = write_new(destination / 'comparison.json', report)
    with (destination / 'table.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report['summary'][0])); writer.writeheader(); writer.writerows(report['summary'])
    lines = ['| Task | Method | Feas@8 | Feasible candidates | Signed gap on own successes |', '| --- | --- | ---: | ---: | ---: |']
    for row in report['summary']:
        gap = '—' if row['mean_signed_reference_gap_percent'] is None else f"{row['mean_signed_reference_gap_percent']:.4f}% (n={row['own_success_gap_denominator']})"
        lines.append(f"| {row['task'].upper()} | {row['method']} | {row['feasible_at_8_count']}/300 | {row['candidate_feasible_count']}/2400 | {gap} |")
    lines += ['', 'One-seed heldout generalization; no confirmation/SOTA or literal paper-reproduction claim.',
              'CVRP gaps use frozen heuristic references and preserve negative values. Paired joint gaps, denominators and multiplicity-adjusted statistics are in comparison.json.', '']
    with (destination / 'table.md').open('x') as handle:
        handle.write('\n'.join(lines))
    return receipt



# Retained raw cross-bindings from frozen performance controller; only evaluation
# instance/candidate counts change to300/2400. Checkpoint horizon remains100.
# Original function SHA256: 99ddf89914d9f39c32474fdacc3834e19d0c6109dab0b93e1861ff953cd27af4
def audit_raw300(plan, launch, config, index, terminal, *, job_id, training, evaluation, benchmark, source, records):
    """Cross-bind already byte-validated receipts; return a compact arm receipt."""
    key, task, method = (plan['key'], plan['task'], plan['method'])
    prefix = key + ': '
    checkpoint = Path(plan['checkpoint']).resolve(strict=True)
    train = Path(plan['training_root']).resolve(strict=True)
    output = Path(plan['output']).resolve(strict=True)
    require(checkpoint == train / 'checkpoints/checkpoint-100', prefix + 'planned checkpoint is not step 100')
    require(launch['slurm_job_id'] == job_id, prefix + 'launch job differs from submission ledger')
    require((launch['task'], launch['method'], launch['stage']) == (task, method, 'grpo') and (index['task'], index['method']) == (task, method), prefix + 'method/task mismatch')
    require(launch['source_manifest'] == training['source_manifest'], prefix + 'launch source mismatch')
    require(launch['training_contract'] == terminal['contract'], prefix + 'launch training contract mismatch')
    require(launch['benchmark'] == benchmark, prefix + 'launch benchmark mismatch')
    require(launch['management'] == evaluation['management'], prefix + 'launch management mismatch')
    for name in ('adapter_model.safetensors', 'adapter_config.json', 'checkpoint_fidelity.json'):
        require(launch['checkpoint'][name] == terminal['checkpoint'][name] and Path(terminal['checkpoint'][name]['path']).resolve() == checkpoint / name, prefix + 'launch checkpoint hash/path mismatch: ' + name)
    expected_slice = {'split': 'eval', 'start': 0, 'count': 300}
    require(launch['generation'] == evaluation['generation'] and launch['repair'] is False and (index['repair'] is False) and (launch['evaluation_slice'] == index['evaluation_slice'] == expected_slice), prefix + 'launch generation/repair/slice mismatch')
    reference = 'gap_to_verified_heuristic_reference' if task == 'cvrp' else 'gap_to_certified_optimum'
    require(index['reference_semantics'] == launch['reference_semantics'][task] == reference, prefix + 'reference semantics mismatch')
    require(config['stage'] == 'grpo' and config['tasks'] == [task] and (config['split'] == 'eval') and (config['run_status'] == 'complete') and (config['evaluation_purpose'] == 'development'), prefix + 'run task/stage/status mismatch')
    require(Path(config['resolved_model_path']).resolve(strict=True) == checkpoint, prefix + 'raw run model differs from planned checkpoint')
    require(Path(config['resolved_run_dir']).resolve(strict=True) == train, prefix + 'raw run training directory mismatch')
    require(Path(config['data_root']).resolve(strict=True) == Path(training['evaluation']['data_root']).resolve(strict=True), prefix + 'raw run data root mismatch')
    require(Path(config['results_dir']).resolve(strict=True).is_relative_to(output), prefix + 'raw run output belongs to a different evaluation')
    for name, field in (('run_config', 'run_config'), ('details', 'details')):
        require(Path(config['files'][name]).resolve(strict=True) == Path(index[field]['path']).resolve(strict=True) and Path(index[field]['path']).resolve(strict=True).is_relative_to(output), prefix + 'raw artifact path linkage mismatch: ' + name)
    provenance = config['experiment_provenance']
    generation = config['generation']
    require(all((generation.get(k) == v for k, v in evaluation['generation'].items())) and generation['seed_policy'] == 'per_example_sha256_v1' and (generation['best_of_n'] == 8), prefix + 'raw generation differs from declaration')
    require(generation == {**provenance['generation'], 'best_of_n': 8}, prefix + 'generation/provenance mismatch')
    require(all((generation.get(k) == [] for k in ('temperature_schedule', 'top_p_schedule', 'op_size_band_top_p_schedule'))) and generation.get('op_size_band_top_p_min_size') == generation.get('op_size_band_top_p_max_size') == 0, prefix + 'undeclared sampling schedule')
    prompting, limits = (config['prompting'], config['limits'])
    require(prompting['prompt_style_resolved'] == 'jiang_f638' and prompting['no_reason'] is True and (prompting['clasp_mis_jiang_response_output'] == '0') and (prompting['clasp_mis_relabel_order'] == '') and (Path(prompting['tokenizer_path']).resolve(strict=True) == checkpoint), prefix + 'prompt/tokenizer configuration mismatch')
    require(limits['eval_start_index'] == 0 and limits['max_eval_samples_per_task'] == 300 and (limits['prefer_subset_files'] is False), prefix + 'raw evaluation slice mismatch')
    artifact = provenance['model']['checkpoint_artifact']
    require(artifact['kind'] == 'local_directory' and artifact['basename'] == 'checkpoint-100' and (artifact['manifest_sha256'] == digest(artifact['files'])), prefix + 'model artifact manifest mismatch')
    seen = set()
    for file in artifact['files']:
        name = file['path']
        require(name not in seen and name in terminal['checkpoint'], prefix + 'unknown/duplicate checkpoint artifact')
        seen.add(name)
        expected = terminal['checkpoint'][name]
        require(file['sha256'] == expected['sha256'] and file['size'] == expected['size_bytes'], prefix + 'raw run checkpoint bytes differ from training: ' + name)
    require({'adapter_model.safetensors', 'adapter_config.json', 'checkpoint_fidelity.json'} <= seen, prefix + 'raw model provenance lacks checkpoint identity')
    data_files = provenance['data']['files']
    dataset = benchmark['files'][task]['dataset']
    require(data_files == [{'path': task + '/eval/test.json', 'sha256': dataset['sha256'], 'size': dataset['size_bytes'], 'split': 'eval', 'task': task}], prefix + 'raw run dataset hash differs from benchmark')
    code = config['code_identity']
    require(code == provenance['code'] and code['manifest_sha256'] == digest(code['files']), prefix + 'raw code provenance mismatch')
    seen_code = set()
    for file in code['files']:
        path = (source / 'src/clasp' / file['path']).resolve(strict=True)
        require(path.is_relative_to(source / 'src/clasp') and path not in seen_code, prefix + 'code path escaped/duplicated')
        seen_code.add(path)
        actual = identity(path)
        require(file['sha256'] == actual['sha256'] and file['size'] == actual['size_bytes'], prefix + 'raw evaluator code differs from frozen source')
    require(source / 'src/clasp/entrypoints/evaluate.py' in seen_code, prefix + 'missing evaluator code identity')
    ordered = sorted(records.values(), key=lambda row: row.example_index)
    require(len(ordered) == 300, prefix + 'expected 300 raw records')
    for row, expected in zip(ordered, benchmark['ordered_instances'][task], strict=True):
        require(all((row.raw.get(k) == v for k, v in expected.items())) and row.raw['task'] == task and (row.raw['split'] == 'eval') and (row.raw['num_return_sequences'] == len(row.raw['candidates']) == 8), prefix + 'raw record differs from benchmark/K8')
    return {'key': key, 'job_id': job_id, 'checkpoint': str(checkpoint), 'instances': 300, 'candidates': 2400, 'launch_sha256': launch['launch_sha256'], 'index_sha256': index['index_sha256'], 'run_config_sha256': config['run_config_sha256'], 'details': index['details'], 'model_artifact_manifest_sha256': artifact['manifest_sha256'], 'status': 'all_cross_bindings_verified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'status', 'run', 'watch', 'worker', 'index', 'summarize'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path); parser.add_argument('--source-sha')
    parser.add_argument('--data', type=Path); parser.add_argument('--heldout-manifest-sha')
    parser.add_argument('--native-audit-bundle', type=Path); parser.add_argument('--native-audit-manifest-sha'); parser.add_argument('--native-audit-job-id')
    parser.add_argument('--key'); parser.add_argument('--running-wrapper', type=Path)
    parser.add_argument('--submit', action='store_true'); parser.add_argument('--max-new-jobs', type=int, default=5)
    parser.add_argument('--hours', type=float, default=6)
    args = parser.parse_args()
    if args.action == 'init':
        require(not args.submit and args.source and args.source_sha and args.data and args.heldout_manifest_sha
                and args.native_audit_bundle and args.native_audit_manifest_sha and args.native_audit_job_id,
                'init requires source/data hashes, successful native audit bundle/job and never submits')
        result = initialize(args.output, args.source, args.source_sha, args.data, args.heldout_manifest_sha,
                            args.native_audit_bundle, args.native_audit_manifest_sha, args.native_audit_job_id)
    else:
        output = args.output.resolve(strict=True)
        if args.action == 'status':
            require(not args.submit, 'status is read-only'); result = reconcile(output)
        elif args.action == 'worker':
            require(not args.submit and args.key and args.running_wrapper, 'worker requires an already submitted key/spool')
            result = worker(output, args.key, args.running_wrapper)[0]
        elif args.action == 'index':
            require(not args.submit and args.key, 'index never submits'); result = index_arm(output, args.key, write=True)[0]
        elif args.action == 'summarize':
            require(not args.submit, 'summary never submits'); result = summarize(output)
        else:
            require(0 < args.hours <= 6, 'heldout watch must be bounded to at most six hours')
            deadline = time.monotonic() + args.hours * 3600
            while True:
                with (output / 'ledgers/controller.lock').open('a') as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    result = reconcile(output, args.submit, True, args.max_new_jobs)
                if args.action == 'run' or result['all_six_verified'] or time.monotonic() >= deadline:
                    break
                require(not any(row['state'] in {'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'CANCELLED', 'NODE_FAIL', 'BOOT_FAIL', 'PREEMPTED'}
                                for row in result['observations']), 'failed heldout arm requires review; no automatic retry')
                print(json.dumps(result, sort_keys=True), flush=True); time.sleep(60)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
