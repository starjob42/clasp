"""Bounded CPU admission checks for the actual full-budget profile and datasets.

Calls the real entrypoint argument/fidelity checks and canonical data loader;
does not instantiate a language model, allocate a GPU, or submit scheduler work.
Each method/task runs in a fresh process with an isolated writable cache.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys

from common import (CONTRACT_NAME, COUNTS, METHODS, PROTOCOL, identity,
                    load_declaration, read, require, run_contract, signed,
                    verify, write_once)

CAMPAIGN_TASKS = ('op', 'jssp', 'cvrp', 'mis', 'tsp')


def declared_scope(d):
    """Task/method coverage: full campaign, or the declared continuation scope."""
    if d.get("scope") is None:
        return CAMPAIGN_TASKS, METHODS
    return tuple(d["scope"]["tasks"]), tuple(d["scope"]["methods"])


def check_runtime_and_source(d):
    require(not os.environ.get('SLURM_JOB_ID'), 'CPU preflight runs outside a GPU allocation')
    require(Path(sys.executable).resolve() == Path(d['runtime_python']).resolve(), 'Wrong native CPU interpreter')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    source = Path(d['source']['root'])
    require(source.resolve() == source and Path(__file__).resolve().is_relative_to(source), 'Run the frozen preflight inside its declared source')
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / 'src'))
    from clasp.baselines.external_baseline_dev import verify_source_manifest
    result = verify_source_manifest(source / 'SOURCE_MANIFEST.sha256', d['source']['manifest_sha256'])
    return result


def task_inputs(d, task, *, require_initializer=True):
    spec = d['tasks'][task]
    initialization = spec.get('initialization') if require_initializer else None
    require(not require_initializer or initialization is not None, 'Initializer remains gated/pending')
    if require_initializer and task == 'tsp':
        require(Path(initialization['checkpoint']).name == 'checkpoint-31250', 'TSP preflight requires the completed full-SFT initializer')
    for name, bound in (initialization or {}).get('files', {}).items():
        require(bound['path'] == str(Path(initialization['checkpoint']) / name), 'Initializer file escaped checkpoint')
        verify(bound)
    files = {path.relative_to(spec['training_data_root']).as_posix(): identity(path)
             for split in ('train', 'eval')
             for path in sorted((Path(spec['training_data_root']) / task / split).glob('*.json'))}
    require(files and files == spec['training_data_files'], 'Native loader inventory differs from declaration')
    require(spec['effective_train_size'] == COUNTS[task], 'Declared retained dataset size differs')
    for bound in spec['evaluation']['data_files']:
        verify(bound)
    return initialization, files


def worker(d, binding, output, task, method, *, dataset_only=False):
    verified_source = check_runtime_and_source(d)
    initialization, input_files = task_inputs(d, task, require_initializer=not dataset_only)
    case = output / task / method
    case.mkdir(parents=True, exist_ok=False)
    from run_chunk import command, environment
    from clasp.entrypoints.train_budget_matched_grpo import parse_args
    profiles = {}
    main_args, main_runtime = None, None
    for kind in (() if dataset_only else ('diagnostic', 'train')):
        job = {'task': task, 'method': method, 'kind': kind, 'step': 5 if kind == 'diagnostic' else 50, 'prior_step': 0}
        contract = run_contract(d, binding, job, initialization)
        contract_path = case / (kind + '_' + CONTRACT_NAME)
        write_once(contract_path, contract)
        # Use the real worker's environment construction, redirecting its cache
        # to this preflight tree. The contract/argv still use the future run's
        # declared output path, which parse_args only reads and validates.
        cache_contract = {**contract, 'output_dir': str(case / (kind + '_cache_owner'))}
        environment(d, cache_contract, contract_path, job)
        os.environ.update(CUDA_VISIBLE_DEVICES='', WORLD_SIZE='4', OMP_NUM_THREADS='2',
                          OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        for name in ('RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
            os.environ.pop(name, None)
        invocation = command(d, contract, contract_path)
        marker = invocation.index('clasp.entrypoints.train_budget_matched_grpo')
        with contextlib.redirect_stdout(sys.stderr):
            args, runtime = parse_args(invocation[marker + 1:])
        require(runtime.contract == contract and runtime.contract_binding == identity(contract_path),
                'Entrypoint selected a different run contract')
        profiles[kind] = {'status': 'actual_entrypoint_parse_passed_no_model_instantiated',
                          'contract': identity(contract_path), 'runtime': runtime.metadata(),
                          'initialization_validation': args.budget_task_identity_binding,
                          'resolved_batch': args.resolved_generation_batch_size,
                          'effective_optimization_batch': args.effective_train_batch_size}
        if kind == 'train':
            main_args, main_runtime = args, runtime
    if dataset_only:
        from clasp.baselines.budget_matched_dev import build_runtime
        job = {'task': task, 'method': method, 'kind': 'train', 'step': 50, 'prior_step': 0}
        contract = run_contract(d, binding, job, {'pending_full_sft': True})
        contract_path = case / ('dataset_only_' + CONTRACT_NAME)
        write_once(contract_path, contract)
        environment(d, {**contract, 'output_dir': str(case / 'dataset_cache_owner'),
                       'initialization': {'checkpoint': str(case / 'unused_pending_initializer')}}, contract_path, job)
        os.environ.update(CUDA_VISIBLE_DEVICES='', WORLD_SIZE='4', OMP_NUM_THREADS='2',
                          OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        main_runtime = build_runtime(verified_source, contract, identity(contract_path))
        from clasp.entrypoints.train_foarl_f638_dev_20260907 import seal_dev_loader_environment
        seal_dev_loader_environment()
    require(main_runtime is not None, 'Missing dataset-validation profile')
    # run_grpo validates the unwrapped TRL class before selecting its opt-in
    # checkpoint/advantage wrapper. Import the identical base without building
    # a trainer or model, and exercise that source/version gate on the CPU.
    from trl import GRPOTrainer
    trainer_implementation = main_runtime.validate_trainer_implementation(GRPOTrainer)
    from datasets import disable_caching
    from clasp.data.benchmark import semantic_instance_sha256
    from clasp.data.loaders import load_mixed_grpo_dataset
    disable_caching()
    with contextlib.redirect_stdout(sys.stderr):
        train, evaluation = load_mixed_grpo_dataset(
            tasks=[task], data_root=d['tasks'][task]['training_data_root'], eval_ratio=0.0, eval_from_train_split=True,
            seed=43, max_samples_per_task=0, max_eval_samples_per_task=0,
            min_train_samples_per_task=1900, min_eval_samples_per_task=0, prefer_subset_files=False,
            no_reason=True, prompt_style='jiang_f638')
    require(len(train) == COUNTS[task] and len(evaluation) == 0, 'Actual retained train/eval count differs')
    validation = {'train': main_runtime.validate_dataset(train, label='train'),
                  'eval': main_runtime.validate_dataset(evaluation, label='eval')}
    train_ids = [semantic_instance_sha256(task, row) for row in train]
    spec = d['tasks'][task]['evaluation']
    eval_rows = []
    for bound in spec['data_files']:
        value = read(bound['path'])
        require(isinstance(value, list), 'Evaluation data must be a JSON row array')
        eval_rows.extend(value)
    selected = eval_rows[spec.get('start', 0):spec.get('start', 0) + spec['count']]
    eval_ids = [semantic_instance_sha256(task, row) for row in selected]
    require(len(eval_ids) == 100 and len(set(eval_ids)) == 100, 'Expected a unique DEV100 evaluation slice')
    require(not (set(train_ids) & set(eval_ids)), 'Training overlaps the frozen development cohort')
    import hashlib
    from common import canonical
    result = {'protocol': PROTOCOL, 'status': ('passed_dataset_only_initializer_pending' if dataset_only
                                              else 'passed_actual_profiles_and_dataset_cpu_only'),
              'task': task, 'method': method, 'declaration': binding, 'source': d['source'],
              'source_files_verified': verified_source['verified_file_count'], 'initialization': initialization,
              'input_files': input_files, 'profiles': profiles, 'dataset_validation': validation,
              'trainer_implementation': trainer_implementation,
              'ordered_training_semantic_ids_sha256': hashlib.sha256(canonical(train_ids)).hexdigest(),
              'training_unique_instances': len(set(train_ids)), 'training_rows': len(train_ids),
              'development_rows': len(eval_ids), 'train_dev_overlap': 0,
              'gpu_used': False, 'language_model_instantiated': False, 'scheduler_mutations': False,
              'policy_improvement_claimed': False, 'entrypoint_validation_deferred': dataset_only}
    write_once(case / 'result.json', signed(result))
    return identity(case / 'result.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', required=True)
    parser.add_argument('--declaration-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-tsp', action='store_true')
    parser.add_argument('--task', choices=COUNTS)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--dataset-only', action='store_true')
    args = parser.parse_args(argv)
    d, binding = load_declaration(args.declaration, args.declaration_sha)
    output = args.output.absolute()
    require(output.resolve() == output and output.parent == Path(d['root']).parent and output != Path(d['root']),
            'Use a separate sibling CPU-preflight directory')
    require((args.task is None) == (args.method is None), 'Worker needs both task and method')
    if args.task:
        require(output.is_dir(), 'Parent must create the preflight output')
        receipt = worker(d, binding, output, args.task, args.method, dataset_only=args.dataset_only)
        print(receipt, flush=True)
        return 0
    verified_source = check_runtime_and_source(d)
    output.mkdir(exist_ok=False)
    tasks, methods = declared_scope(d)
    write_once(output / 'launch.json', signed({'protocol': PROTOCOL, 'declaration': binding,
               'implementation': identity(__file__), 'tasks': list(tasks), 'methods': list(methods),
               'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'gpu_used': False}))
    reports = {}
    for task in tasks:
        task_hash = None
        for method in methods:
            invocation = [d['runtime_python'], '-B', str(Path(__file__).absolute()),
                          '--declaration', binding['path'], '--declaration-sha', binding['sha256'],
                          '--output', str(output), '--task', task, '--method', method]
            if task == 'tsp' and not args.include_tsp:
                invocation.append('--dataset-only')
            with (output / (task + '_' + method + '.log')).open('x') as log:
                result = subprocess.run(invocation, cwd=d['source']['root'], stdout=log, stderr=subprocess.STDOUT,
                                        timeout=1800, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
            require(result.returncode == 0, 'CPU preflight failed: ' + task + '/' + method)
            report = output / task / method / 'result.json'
            value = read(report, sealed=True)
            current_hash = value['ordered_training_semantic_ids_sha256']
            require(task_hash in (None, current_hash), 'Methods see different training order/cohorts')
            task_hash = current_hash
            reports[task + '/' + method] = identity(report)
            print('CPU_PREFLIGHT_PASSED ' + task + '/' + method, flush=True)
    import hashlib
    from common import canonical
    numerical_sha = hashlib.sha256(canonical({k: v for k, v in d.items() if k != 'cpu_preflight'})).hexdigest()
    write_once(output / 'complete.json', signed({'protocol': PROTOCOL,
               'status': 'passed', 'declaration': binding, 'reports': reports,
               'numerical_declaration_sha256': numerical_sha, 'source': d['source'], 'source_binding': verified_source,
               'deferred_tasks': [] if args.include_tsp else ['tsp'],
               'deferred_reason': None if args.include_tsp else 'All TSP methods wait for the same completed certified full-SFT initializer.',
               'finished_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'gpu_used': False,
               'training_or_inference_performed': False, 'scheduler_mutations': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
