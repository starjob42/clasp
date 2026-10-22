"""CPU contract and reward semantics for the prospective longer-budget study."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from clasp.baselines import budget_matched_dev as b
from clasp.entrypoints.train_budget_matched_grpo import build_parser, resolve_declared_defaults
from clasp.config import finalize_grpo_args


def make_runtime(task='op', method='rloo_quality', diagnostic=False):
    horizon = b.HORIZONS[task]
    boundaries = [5, 10] if diagnostic else [v for v in range(50, horizon + 1, 50) if not (task == 'mis' and v == 750)]
    declaration = {'protocol': b.PROTOCOL, 'task': task, 'method': method, 'horizon': horizon,
                   'chunk_boundaries': boundaries, 'source': {'root': '/tmp/budget-source', 'manifest_sha256': 'a' * 64},
                   'expected_train_count': b.TRAIN_COUNTS[task], 'seed': 43, 'diagnostic': diagnostic,
                   'fixed_horizon': {'max_steps': horizon, 'world_size': 4}}
    source = {'status': 'passed_frozen_manifest_successor_source', 'source_root': '/tmp/budget-source',
              'manifest': {'path': '/tmp/budget-source/SOURCE_MANIFEST.sha256', 'sha256': 'a' * 64}}
    bound = {'path': '/tmp/budget-contract.json', 'sha256': 'b' * 64, 'size_bytes': 100}
    return b.build_runtime(source, declaration, bound)


def args_for(runtime):
    return SimpleNamespace(**{**b.COMMON_DEFAULTS, 'tasks': [runtime.task], 'budget_method': runtime.method,
                             'budget_diagnostic': runtime.diagnostic, 'max_steps': runtime.contract['horizon'],
                             'save_steps': 5 if runtime.diagnostic else 50,
                             'grpo_beta': runtime.algorithm_config()['beta'],
                             'grpo_loss_type': runtime.algorithm_config()['loss_type'],
                             'verify_gt_rewards': runtime.method != 'foarl'})


def env_for(monkeypatch, runtime):
    for name, value in b.reward_environment(runtime.method).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv('WORLD_SIZE', '4')
    monkeypatch.setenv('GRPO_DATA_SEED', '43')
    monkeypatch.setenv('GRPO_STRICT_OPTIMIZER_RESUME', '1')
    monkeypatch.setenv('GRPO_STOP_AT_GLOBAL_STEP', str(runtime.contract['chunk_boundaries'][0]))
    for name in b.KL_ENVIRONMENT:
        if runtime.method == 'foarl':
            monkeypatch.setenv(name, '/tmp/fixed-sft' if name.endswith('PATH') else '/tmp/fixed-sft-receipt.json')
        else:
            monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize('task', b.HORIZONS)
@pytest.mark.parametrize('method', b.METHODS)
def test_full_horizons_share_exact_recipe(task, method, monkeypatch):
    runtime = make_runtime(task, method)
    env_for(monkeypatch, runtime)
    runtime.validate_args(args_for(runtime))
    config = runtime.algorithm_config()
    resolved = SimpleNamespace(**config, num_generations=9, max_steps=b.HORIZONS[task],
                               generation_batch_size=36, gradient_accumulation_steps=3,
                               steps_per_generation=3, ignore_data_skip=False, eval_strategy='no')
    runtime.validate_grpo_config(resolved)
    metadata = runtime.metadata()
    assert metadata['common_budget']['responses'] == 36 * b.HORIZONS[task]
    assert metadata['common_budget']['prompt_exposures'] == 4 * b.HORIZONS[task]
    assert metadata['common_budget']['training_sampling'] == {'temperature': .7, 'top_p': .9, 'top_k': 50}
    assert metadata['policy_improvement_established'] is False
    assert runtime.requires_explicit_fixed_reference == (method == 'foarl')
    assert config['loss_type'] == ('dr_grpo' if method.startswith('rloo_') else 'grpo')
    assert config['scale_rewards'] == ('none' if method.startswith('rloo_') else 'group')
    if method == 'foarl':
        assert runtime.reward_functions() is b.FOARL_F638_REWARD_FUNCS
    if method == 'rloo_legacy':
        assert runtime.reward_contract() is None
        assert runtime.reward_functions() == runtime._legacy_runtime().reward_functions()


@pytest.mark.parametrize('name,value', [('max_steps', 100), ('generation_temperature', 1),
                                       ('grpo_loss_type', 'grpo'), ('num_generations', 8),
                                       ('save_steps', 100), ('learning_rate', 2e-6)])
def test_explicit_numerical_drift_rejected(name, value, monkeypatch):
    runtime = make_runtime()
    env_for(monkeypatch, runtime)
    args = args_for(runtime)
    setattr(args, name, value)
    with pytest.raises(ValueError, match='numerical recipe'):
        runtime.validate_args(args)


def test_diagnostics_keep_full_horizon_and_cannot_become_main(monkeypatch):
    runtime = make_runtime('tsp', diagnostic=True)
    env_for(monkeypatch, runtime)
    runtime.validate_args(args_for(runtime))
    assert runtime.contract['horizon'] == 800
    assert runtime.resume_contract() != make_runtime('tsp').resume_contract()
    monkeypatch.setenv('GRPO_STOP_AT_GLOBAL_STEP', '50')
    with pytest.raises(ValueError, match='chunk boundary'):
        runtime.validate_args(args_for(runtime))


def test_contract_mutation_schedule_rebase_and_kl_leak_fail(monkeypatch):
    runtime = make_runtime()
    env_for(monkeypatch, runtime)
    monkeypatch.setenv('GRPO_REBASE_LOADED_SCHEDULER_AT_STEP', '100')
    with pytest.raises(ValueError, match='rebasing'):
        runtime.validate_args(args_for(runtime))
    monkeypatch.delenv('GRPO_REBASE_LOADED_SCHEDULER_AT_STEP')
    monkeypatch.setenv('CLASP_KL_REFERENCE_MODEL_PATH', '/tmp/leaked-reference')
    with pytest.raises(ValueError, match='Beta-zero'):
        runtime.validate_args(args_for(runtime))
    runtime.contract['expected_train_count'] = 4999
    with pytest.raises(ValueError, match='mutated'):
        runtime.metadata()


def test_foarl_requires_explicit_reference(monkeypatch):
    runtime = make_runtime(method='foarl')
    env_for(monkeypatch, runtime)
    monkeypatch.delenv('CLASP_KL_REFERENCE_RECEIPT')
    with pytest.raises(ValueError, match='initial-SFT'):
        runtime.validate_args(args_for(runtime))


def test_contract_hash_and_mis_resume_boundary(tmp_path):
    contract = dict(make_runtime('mis').contract)
    file = tmp_path / 'contract.json'
    raw = json.dumps(contract).encode()
    file.write_bytes(raw)
    value, bound = b.read_contract(file, hashlib.sha256(raw).hexdigest())
    assert value == contract and bound['size_bytes'] == len(raw)
    with pytest.raises(ValueError, match='byte hash'):
        b.read_contract(file, '0' * 64)
    contract['chunk_boundaries'].insert(-1, 750)
    with pytest.raises(ValueError, match='epoch boundary'):
        b.validate_contract(contract)


def test_source_and_config_drift_fail():
    runtime = make_runtime()
    runtime.source_binding['manifest']['sha256'] = 'c' * 64
    with pytest.raises(ValueError, match='source differs'):
        runtime.metadata()
    runtime = make_runtime()
    with pytest.raises(ValueError, match='cannot express'):
        runtime.grpo_config_overrides({'beta': object()})


def test_cli_omitted_horizon_defaults_do_not_overwrite_explicit_values():
    parser = build_parser()
    argv = ['--tasks', 'op', '--sft_checkpoint_path', '/tmp/sft', '--budget_contract', '/tmp/contract',
            '--budget_contract_sha256', 'a' * 64, '--budget_method', 'rloo_quality', '--output_dir', '/tmp/output']
    args = resolve_declared_defaults(parser.parse_args(argv), make_runtime().contract)
    assert args.max_steps == 1250 and args.save_steps == 50 and args.grpo_beta == 0
    assert args.generation_temperature == .7 and args.verify_gt_rewards is True
    args = resolve_declared_defaults(parser.parse_args(argv + ['--max_steps', '100']), make_runtime().contract)
    assert args.max_steps == 100


@pytest.mark.parametrize('method', b.METHODS)
def test_finalized_cli_preserves_profile_contract(method, monkeypatch):
    # Shared finalize_grpo_args exports environment settings; isolate the whole
    # mapping so successive method parametrizations behave like fresh workers.
    monkeypatch.setattr(b.os, 'environ', dict(b.os.environ))
    runtime = make_runtime(method=method)
    env_for(monkeypatch, runtime)
    raw = build_parser().parse_args(['--tasks', 'op', '--sft_checkpoint_path', '/tmp/sft',
                                    '--budget_contract', '/tmp/contract', '--budget_contract_sha256', 'a' * 64,
                                    '--budget_method', method, '--output_dir', '/tmp/output'])
    args = finalize_grpo_args(resolve_declared_defaults(raw, runtime.contract))
    runtime.validate_args(args)
    assert args.resolved_generation_batch_size == 36
    assert args.resolved_steps_per_generation in (None, 3)
    assert args.effective_train_batch_size == 36


FIXTURES = {
    'tsp': ([[0, 0], [1, 0], [1, 1], [0, 1]], 'Route: [0, 1, 2, 3, 0], Objective: 4'),
    'cvrp': ([[[0, 0], [1, 0], [1, 1], [0, 1]], [0, 1, 1, 1], 3], 'Routes: [[0, 1, 2, 3, 0]], Objective: 4'),
    'op': ([[[0, 0], [1, 0], [1, 1], [0, 1]], [0, 1, 2, 3], 4], 'Route: [0, 1, 2, 3], Objective: 6'),
    'mis': ({'num_nodes': 5, 'edges': [[0, 1], [1, 2], [2, 3]]}, 'Set: [0, 2, 4], Objective: 3'),
    'jssp': ([[0, 2, 1, 1], [0, 1, 1, 2]], 'Schedule: [[1, 0], [1, 0]], Makespan: 4'),
}


@pytest.mark.parametrize('task', b.HORIZONS)
def test_all_five_classic_heads_preserve_strict_rewards_and_candidate_calibration(task, monkeypatch):
    instance, answer = FIXTURES[task]
    kwargs = {'task_type': [task, task], 'answer': [answer, answer],
              'instance_json': [json.dumps(instance)] * 2, 'input': ['', '']}
    classic = make_runtime(task, 'classic_grpo')
    env_for(monkeypatch, classic)
    values = classic.reward_functions()[0](['prompt'] * 2, [answer, 'unparseable'], **kwargs)
    expected = b.STRICT_HEADS[task](['prompt'] * 2, [answer, 'unparseable'],
                                 **kwargs, apply_constraint_gate=True, allow_infeasible_partial_credit=False)
    assert values == expected
    assert values[0] > 0 and values[1] == 0
    candidate = make_runtime(task)
    env_for(monkeypatch, candidate)
    values = candidate.reward_functions()[0](['prompt'] * 2, [answer, 'unparseable'], **kwargs)
    assert values == [1.5, -1.0]


class Dataset(list):
    column_names = ['input', 'prompt', 'answer', 'instance_json', 'task_type']


def test_exact_retained_training_count_and_no_inloop_eval(monkeypatch):
    runtime = make_runtime('mis')
    env_for(monkeypatch, runtime)
    instance, answer = FIXTURES['mis']
    row = {'input': 'Graph', 'prompt': 'Solve graph', 'answer': answer, 'task_type': 'mis', 'instance_json': json.dumps(instance)}
    data = Dataset([row] * 2900)
    assert runtime.validate_dataset(data, label='train')['rows'] == 2900
    with pytest.raises(ValueError, match='Retained training count'):
        runtime.validate_dataset(Dataset(data[:-1]), label='train')
    with pytest.raises(ValueError, match='in-loop'):
        runtime.validate_dataset(Dataset([row]), label='eval')
    assert runtime.validate_dataset(Dataset(), label='eval')['rows'] == 0
