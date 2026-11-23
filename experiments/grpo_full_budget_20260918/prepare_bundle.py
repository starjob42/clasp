"""Create a new immutable local source package and native draft declaration."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

from common import (HORIZONS, METHODS, PROTOCOL, PROTOCOL_V2, continuation_graph,
                    graph, identity, read, write_once)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--tsp-hyperparameters', type=Path,
                        help='JSON file containing the native hyperparams.json identity')
    parser.add_argument('--continuation-tasks', default='',
                        help='Comma-separated continuation scope; empty means the full campaign')
    parser.add_argument('--continuation-methods', default='',
                        help='Comma-separated continuation scope; empty means the full campaign')
    parser.add_argument('--initialization-attestation', default=None,
                        help='Required when a declared initializer lacks trainer_state.json')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    local = Path(__file__).resolve().parent
    if bool(args.continuation_tasks) != bool(args.continuation_methods):
        parser.error('--continuation-tasks and --continuation-methods must be given together')
    continuation = bool(args.continuation_tasks)
    if not continuation and args.tsp_hyperparameters is None:
        parser.error('--tsp-hyperparameters is required for a full campaign')
    if continuation and args.initialization_attestation is None:
        parser.error('--initialization-attestation is required for a continuation')
    source = local / f'source_v{args.version}'
    native = Path('/lustre/fsn1/projects/rech/vas/uag72kz/grpo_full_budget_20260918')
    native_source = native / source.name
    source.mkdir(exist_ok=False)
    paths = list((repo / 'src').rglob('*.py'))
    paths += [repo / 'scripts' / name for name in ('run_external_baseline_dev_chunk.py', 'audit_sft_reload_equivalence.py')]
    paths += list(local.glob('*.py')) + list(local.glob('*.slurm'))
    paths += list((repo / 'tests').glob('test_budget_matched_*.py'))
    lines = []
    for path in sorted(set(paths)):
        relative = path.relative_to(repo)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        target.chmod(0o440)
        lines.append(hashlib.sha256(target.read_bytes()).hexdigest() + '  ' + relative.as_posix())
    manifest = source / 'SOURCE_MANIFEST.sha256'
    manifest.write_text('\n'.join(lines) + '\n')
    manifest.chmod(0o440)
    archive = local / (source.name + '.tar.gz')
    with tarfile.open(archive, 'x:gz') as stream:
        for member in sorted(source.rglob('*')):
            if member.is_file():
                raw = member.read_bytes()
                info = tarfile.TarInfo((Path(source.name) / member.relative_to(source)).as_posix())
                info.size, info.mode = len(raw), 0o440
                stream.addfile(info, io.BytesIO(raw))
    if continuation:
        # A continuation inherits the fully resolved task bindings of the
        # original campaign instead of re-deriving them from earlier trees.
        parent = read(local / 'launch_declaration_v1.json')
        tasks = copy.deepcopy(parent['tasks'])
        cohort = dict(parent['cohort_declaration'])
        prior = {'runtime_python': parent['runtime_python'], 'runtime': parent['runtime'],
                 'evaluation': parent['evaluation']}
        scope = {'tasks': [t.strip() for t in args.continuation_tasks.split(',')],
                 'methods': [m.strip() for m in args.continuation_methods.split(',')]}
        claim_scope = 'continuation_single_seed_inspected_development_cohorts'
        protocol = PROTOCOL_V2
        jobs = continuation_graph(scope)
        extra = {'scope': scope, 'initialization_attestation': args.initialization_attestation,
                 'continuation_of': {'declaration': parent['source'],
                                     'root': parent['root'],
                                     'reason': 'FOARL MIS/OP adapters and receipts pruned 2026-11-13/14; '
                                               'restart from the shared step-31250 SFT initializers'},
                 'evaluation_policy': 'training-only continuation; arms are scored with the same '
                                      'standalone matched-protocol evaluator as the published rows '
                                      '(DEV100, raw8, seed 271828, jiang_f638)'}
    else:
        parent = read(repo / 'experiments/grpo_quality_signal_20260916/launch_declaration_v3.json')
        tasks = copy.deepcopy(parent['tasks'])
        tasks['tsp'].pop('initialization')
        tsp_source = read(repo / 'research/grpo_budget_fairness_20260918/tsp_sft_continuation_source.json')
        tasks['tsp']['initialization_pending'] = {
            'checkpoint': '/lustre/fsn1/projects/rech/vas/uag72kz/clasp_tsp_stability_lr1e4_sourcev18_production_gate_iclr_20260831_v1/runs/tsp/seed42/checkpoints/checkpoint-31250',
            'jobs': ['1703800', '1703801', '1703802', '1703803'],
            'source': {'root': tsp_source['source_root'], 'manifest_sha256': tsp_source['manifest_sha256']},
            'hyperparameters': read(args.tsp_hyperparameters)}
        cohort_path = repo / 'experiments/grpo_quality_signal_20260916/launch_declaration_v3.json'
        cohort = {**identity(cohort_path), 'path': '/lustre/fsn1/projects/rech/vas/uag72kz/grpo_quality_signal_20260916/launch_declaration_v3.json'}
        scope, extra = None, {}
        claim_scope = 'prospective_single_seed_inspected_development_cohorts'
        protocol = PROTOCOL
        jobs = graph()
    declaration = {'protocol': protocol, 'schema_version': 1,
        'claim_scope': claim_scope,
        'policy_improvement_established': False, 'root': str(native / f'campaign_v{args.version}'),
        'source': {'root': str(native_source), 'manifest_sha256': identity(manifest)['sha256']},
        'runtime_python': prior['runtime_python'], 'runtime': prior['runtime'],
        'user': 'uag72kz', 'seed': 43, 'methods': list(METHODS), 'horizons': HORIZONS,
        'tasks': tasks, 'cohort_declaration': cohort, 'evaluation': prior['evaluation'],
        'jobs': jobs, 'budget': {'maximum_jobs': len(jobs), 'maximum_outstanding': 2,
            'maximum_a100_reservation_hours': sum(j['gpus'] * j['walltime_seconds'] / 3600 for j in jobs.values()),
            'controller_duration_days': 60, 'automatic_retry': False},
        **extra}
    if continuation:
        declaration['comparison'] = {**parent['comparison'], 'fresh_full_horizon_from_shared_sft': True,
                                     'continuation_scope': scope}
    declaration_path = local / f'draft_declaration_v{args.version}.json'
    write_once(declaration_path, declaration)
    package = {'source': declaration['source'], 'archive': identity(archive),
               'draft_declaration': {**identity(declaration_path), 'native_path': str(native / declaration_path.name)},
               'source_file_count': len(lines), 'numerical_changes_to_prior_campaigns': False}
    write_once(local / f'package_v{args.version}.json', package)
    print(json.dumps(package, indent=2))


if __name__ == '__main__':
    main()
