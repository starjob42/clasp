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

from common import HORIZONS, METHODS, PROTOCOL, graph, identity, read, write_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--tsp-hyperparameters', type=Path, required=True,
                        help='JSON file containing the native hyperparams.json identity')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    local = Path(__file__).resolve().parent
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
    prior = read(repo / 'experiments/grpo_quality_signal_20260916/launch_declaration_v3.json')
    tasks = copy.deepcopy(prior['tasks'])
    tasks['tsp'].pop('initialization')
    tsp_source = read(repo / 'research/grpo_budget_fairness_20260918/tsp_sft_continuation_source.json')
    tasks['tsp']['initialization_pending'] = {
        'checkpoint': '/lustre/fsn1/projects/rech/vas/uag72kz/clasp_tsp_stability_lr1e4_sourcev18_production_gate_iclr_20260831_v1/runs/tsp/seed42/checkpoints/checkpoint-31250',
        'jobs': ['1703800', '1703801', '1703802', '1703803'],
        'source': {'root': tsp_source['source_root'], 'manifest_sha256': tsp_source['manifest_sha256']},
        'hyperparameters': read(args.tsp_hyperparameters)}
    jobs = graph()
    cohort_path = repo / 'experiments/grpo_quality_signal_20260916/launch_declaration_v3.json'
    cohort = {**identity(cohort_path), 'path': '/lustre/fsn1/projects/rech/vas/uag72kz/grpo_quality_signal_20260916/launch_declaration_v3.json'}
    declaration = {'protocol': PROTOCOL, 'schema_version': 1,
        'claim_scope': 'prospective_single_seed_inspected_development_cohorts',
        'policy_improvement_established': False, 'root': str(native / f'campaign_v{args.version}'),
        'source': {'root': str(native_source), 'manifest_sha256': identity(manifest)['sha256']},
        'runtime_python': prior['runtime_python'], 'runtime': prior['runtime'],
        'user': 'uag72kz', 'seed': 43, 'methods': list(METHODS), 'horizons': HORIZONS,
        'tasks': tasks, 'cohort_declaration': cohort, 'evaluation': prior['evaluation'],
        'jobs': jobs, 'budget': {'maximum_jobs': len(jobs), 'maximum_outstanding': 2,
            'maximum_a100_reservation_hours': sum(j['gpus'] * j['walltime_seconds'] / 3600 for j in jobs.values()),
            'controller_duration_days': 60, 'automatic_retry': False},
        'comparison': {'matches_released_prompt_exposure': True, 'released_rollout_group_size': 8,
            'study_rollout_group_size': 9, 'study_responses_per_update': 36,
            'released_response_budget_multiplier': 1.125, 'official_reproduction_claimed': False,
            'equal_tokens_or_compute_claimed': False, 'fresh_full_horizon_from_shared_sft': True,
            'mis_unique_training_instances': 2900, 'mis_prompt_exposures': 3200,
            'mis_reintroduce_excluded_overlap': False, 'no_guaranteed_win': True,
            'confirmation_requires_fresh_cohort_and_multiple_training_seeds': True}}
    declaration_path = local / f'draft_declaration_v{args.version}.json'
    write_once(declaration_path, declaration)
    package = {'source': declaration['source'], 'archive': identity(archive),
               'draft_declaration': {**identity(declaration_path), 'native_path': str(native / declaration_path.name)},
               'source_file_count': len(lines), 'numerical_changes_to_prior_campaigns': False}
    write_once(local / f'package_v{args.version}.json', package)
    print(json.dumps(package, indent=2))


if __name__ == '__main__':
    main()
