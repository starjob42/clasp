"""The archive excludes unplanned files and verifies executable bytes before import."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def packager():
    path = Path(__file__).resolve().parents[1] / (
        'analysis_outputs/grpo_incumbent_20260907_v1/package_native_holdout_successor_bundle_v1.py')
    spec = importlib.util.spec_from_file_location('successor_packager_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('changed', ['builder.py', 'helper.py'])
def test_changed_live_source_rejected_before_any_executable_import(packager, tmp_path, monkeypatch, changed):
    for name in ('builder.py', 'helper.py'):
        (tmp_path / name).write_text('raise RuntimeError("must not execute")\n')
    plan = {'plan_sha256': 'a' * 64,
            'source_files': {name: packager.identity(tmp_path / name) for name in ('builder.py', 'helper.py')}}
    plan_file = tmp_path / 'plan.json'
    plan_file.write_text(json.dumps(plan))
    monkeypatch.setattr(packager, 'REPO', tmp_path)
    monkeypatch.setattr(packager, 'BUILDER', 'builder.py')
    monkeypatch.setattr(packager, 'PLAN_SHA', plan['plan_sha256'])
    monkeypatch.setattr(packager, 'PLAN_FILE_SHA', packager.identity(plan_file)['sha256'])
    (tmp_path / changed).write_text('modified\n')
    called = []
    monkeypatch.setattr(packager.importlib.util, 'spec_from_file_location', lambda *args: called.append(args))
    with pytest.raises(AssertionError):
        packager.load_verified_builder(plan_file)
    assert not called


@pytest.fixture
def synthetic_data(packager, tmp_path):
    for name in ('plan.json', 'build_declaration.json', 'build_manifest.json'):
        (tmp_path / name).write_text('{}\n')
    retained = tmp_path / 'implementation_sources/helper.py'
    retained.parent.mkdir()
    retained.write_text('# retained synthetic helper\n')
    plan = {'source_files': {'helper.py': packager.identity(retained)}}
    attempts = []
    for index in range(300):
        directory = tmp_path / f'cvrp/lkh/instance_{index:03d}'
        directory.mkdir(parents=True)
        problem = directory / 'problem.vrp'
        problem.write_text(f'synthetic problem {index}\n')
        for seed in (101, 202):
            parameter = directory / f'seed{seed}.par'
            parameter.write_text(f'synthetic seed {seed}\n')
            attempts.append({'seed': seed, 'problem': {'path': str(problem.relative_to(tmp_path)), **packager.identity(problem)},
                             'parameter': {'path': str(parameter.relative_to(tmp_path)), **packager.identity(parameter)},
                             'raw_log': None, 'solution': None})
    attempt_file = tmp_path / 'cvrp/lkh_attempts.json'
    attempt_file.write_text(json.dumps(attempts))
    built = {'tasks': {'cvrp': {'files': {'cvrp/lkh_attempts.json': packager.identity(attempt_file)}},
                       'mis': {'files': {}}}}
    return tmp_path, plan, built


def test_inventory_retains_every_bound_lkh_artifact_without_optional_failed_outputs(packager, synthetic_data):
    data, plan, built = synthetic_data
    inventory = packager.data_input_inventory(data, plan, built)
    assert len(inventory) == 905  # Three metadata, one retained source, one attempt file, 900 artifacts.
    assert inventory['cvrp/lkh/instance_299/seed202.par'].is_file()


@pytest.mark.parametrize('mutation', ['extra', 'missing', 'modified', 'symlink'])
def test_inventory_rejects_unplanned_missing_changed_or_linked_payloads(packager, synthetic_data, mutation):
    data, plan, built = synthetic_data
    if mutation == 'extra':
        (data / 'unplanned_training_corpus.json').write_text('[]\n')
    elif mutation == 'missing':
        (data / 'cvrp/lkh/instance_000/seed101.par').unlink()
    elif mutation == 'modified':
        (data / 'cvrp/lkh/instance_000/seed101.par').write_text('changed\n')
    else:
        (data / 'unexpected_link').symlink_to(data / 'plan.json')
    with pytest.raises(AssertionError):
        packager.data_input_inventory(data, plan, built)


def test_attempt_cannot_authorize_an_arbitrary_path(packager, synthetic_data):
    data, plan, built = synthetic_data
    path = data / 'cvrp/lkh_attempts.json'
    attempts = json.loads(path.read_text())
    attempts[0]['problem']['path'] = '../../private.json'
    path.write_text(json.dumps(attempts))
    built['tasks']['cvrp']['files']['cvrp/lkh_attempts.json'] = packager.identity(path)
    with pytest.raises(AssertionError):
        packager.data_input_inventory(data, plan, built)
