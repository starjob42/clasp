"""Opt-in feasible-quality ablation on the calibrated RLOO full-horizon recipe.

The full task horizon is declared before the first optimizer update. Chunking
never changes the schedule, rewards or method. This profile compares complete
recipes and does not claim an official FOARL reproduction or a performance win.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from clasp.baselines.external_baseline_dev import (
    COMMON_DEFAULTS as EXTERNAL_DEFAULTS,
    REWARD_ENVIRONMENT as GRADED_ENVIRONMENT,
    ExternalBaselineRuntime,
    verify_source_manifest,
)
from clasp.baselines.foarl_f638_dev_20260907 import COMMON_PROMPT_SHA256
from clasp.baselines.llmcosolver_f638_training import (
    FOARL_F638_REWARD_FUNCS,
    validate_foarl_f638_dataset,
    validate_trl_foarl_implementation,
)
from clasp.baselines.quality_signal_dev import QualitySignalRuntime
from clasp.rewards.feasible_quality import make_advantage_contract, validate_advantage_contract

PROTOCOL = 'feasible-quality-grpo-development-20260919-v1'
PROFILE_ID = PROTOCOL
METHODS = ('rloo_quality_control', 'rloo_quality_cardinal', 'rloo_quality_scalar', 'rloo_quality_midrank')
ADVANTAGE_METHODS = dict(zip(METHODS, ('control', 'cardinal', 'scalar_reweight', 'midrank')))
HORIZONS = {'cvrp': 500, 'tsp': 800, 'mis': 800, 'jssp': 800, 'op': 1250}
TRAIN_COUNTS = {'cvrp': 2000, 'tsp': 3200, 'mis': 2900, 'jssp': 3200, 'op': 5000}
COMMON_DEFAULTS = {
    **EXTERNAL_DEFAULTS, 'max_steps': None, 'save_steps': None,
    'generation_temperature': 0.7, 'generation_top_p': 0.9, 'generation_top_k': 50,
    'save_total_limit': 3, 'eval_from_train_split': True, 'max_samples_per_task': 0,
    'grpo_beta': None, 'grpo_loss_type': None, 'verify_gt_rewards': None,
}
STRICT_ENVIRONMENT = {**GRADED_ENVIRONMENT, 'GRPO_GRADED_REWARD': '0'}
KL_ENVIRONMENT = ('CLASP_KL_REFERENCE_MODEL_PATH', 'CLASP_KL_REFERENCE_RECEIPT')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_identity(path):
    path = Path(path).absolute()
    require(path.is_file() and path.resolve(strict=True) == path, 'Missing or linked budget contract/input')
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'Budget input changed while reading')
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}, raw


def validate_contract(contract):
    required = {'protocol', 'task', 'method', 'horizon', 'chunk_boundaries', 'source', 'expected_train_count', 'advantage_contract'}
    require(isinstance(contract, dict) and required <= contract.keys(), 'Incomplete budget contract')
    require(contract['protocol'] == PROTOCOL and contract['task'] in HORIZONS
            and contract['method'] in METHODS, 'Unknown budget task/method/protocol')
    require(validate_advantage_contract(contract['advantage_contract']) ==
            make_advantage_contract(ADVANTAGE_METHODS[contract['method']]),
            'Quality advantage differs from the declared method')
    task, horizon = contract['task'], contract['horizon']
    require(type(horizon) is int and horizon == HORIZONS[task], 'Task full horizon differs')
    require(type(contract['expected_train_count']) is int and contract['expected_train_count'] == TRAIN_COUNTS[task],
            'Retained training count differs from the prospective study')
    require(contract.get('seed', 43) == 43 and type(contract.get('seed', 43)) is int, 'Budget study fixes RL seed43')
    diagnostic = contract.get('diagnostic', False)
    require(type(diagnostic) is bool, 'Diagnostic mode must be boolean')
    boundaries = contract['chunk_boundaries']
    require(isinstance(boundaries, list) and boundaries and all(type(v) is int for v in boundaries)
            and boundaries == sorted(set(boundaries)), 'Chunk boundaries must be distinct increasing integers')
    if diagnostic:
        require(boundaries == [5, 10], 'Diagnostics keep the full horizon and stop only at5/10')
    else:
        require(boundaries[-1] == horizon and all(0 < v <= horizon and v % 50 == 0 for v in boundaries),
                'Main chunk boundaries must be declared50-step checkpoints ending at the full horizon')
        require(task != 'mis' or 750 not in boundaries, 'MIS must cross its retained-data epoch boundary in one700→800 chunk')
    if 'fixed_horizon' in contract:
        require(contract['fixed_horizon'] == {'max_steps': horizon, 'world_size': 4}, 'Changed fixed-horizon declaration')
    require(contract.get('stage', 'grpo') == 'grpo' and contract.get('schema_version', 1) == 1,
            'Wrong budget contract stage/schema')
    source = contract['source']
    require(isinstance(source, dict) and set(source) == {'root', 'manifest_sha256'}
            and isinstance(source['root'], str) and Path(source['root']).is_absolute()
            and '..' not in Path(source['root']).parts
            and isinstance(source['manifest_sha256'], str)
            and re.fullmatch('[0-9a-f]{64}', source['manifest_sha256']) is not None, 'Invalid budget source binding')
    if 'receipt_sha256' in contract:
        require(contract['receipt_sha256'] == digest({k: v for k, v in contract.items() if k != 'receipt_sha256'}),
                'Invalid budget contract self-hash')
    return copy.deepcopy(contract)


def read_contract(path, expected_sha256):
    bound, raw = file_identity(path)
    require(bound['sha256'] == expected_sha256, 'Budget contract byte hash differs')
    return validate_contract(json.loads(raw)), bound


def reward_environment(method):
    require(method in METHODS, 'Unknown budget method')
    return dict(GRADED_ENVIRONMENT)


def seal_reward_environment(method):
    expected = reward_environment(method)
    drift = {name: os.environ[name] for name, value in expected.items() if name in os.environ and os.environ[name] != value}
    require(not drift, 'Inherited reward settings differ: ' + str(drift))
    if method != 'foarl':
        require(not any(os.environ.get(name, '').strip() for name in KL_ENVIRONMENT), 'Beta-zero methods forbid an explicit KL reference')
    os.environ.update(expected)
    return expected


@dataclass(frozen=True)
class FeasibleQualityRuntime:
    contract: Mapping[str, Any]
    contract_binding: Mapping[str, Any]
    source_binding: Mapping[str, Any]
    feasible_quality_extension: bool = True
    budget_matched_extension: bool = True
    verified_teacher_extension: bool = True
    teacher_weight: float = 0.0
    advantage_estimator: str = 'rloo'
    baseline_initial: float = 0.0
    baseline_scale: float = 1.0
    num_generations: int = 9
    world_size: int = 4
    seed: int = 43

    def __post_init__(self):
        validated = validate_contract(dict(self.contract))
        object.__setattr__(self, 'contract', validated)
        object.__setattr__(self, 'contract_binding', copy.deepcopy(dict(self.contract_binding)))
        object.__setattr__(self, 'source_binding', copy.deepcopy(dict(self.source_binding)))
        object.__setattr__(self, '_contract_digest', digest(validated))

    @property
    def task(self):
        return self.contract['task']

    @property
    def method(self):
        return self.contract['method']

    @property
    def diagnostic(self):
        return self.contract.get('diagnostic', False)

    @property
    def external_baseline_extension(self):
        return True

    @property
    def requires_explicit_fixed_reference(self):
        return False

    def _check_binding(self):
        require(digest(self.contract) == self._contract_digest, 'Budget contract mutated after validation')
        require(self.source_binding.get('status') == 'passed_frozen_manifest_successor_source'
                and self.source_binding.get('source_root') == self.contract['source']['root']
                and self.source_binding.get('manifest', {}).get('sha256') == self.contract['source']['manifest_sha256'],
                'Budget runtime source differs from its contract')
        require(self.world_size == 4 and self.seed == 43 and self.num_generations == 9 and self.teacher_weight == 0
                and self.baseline_initial == 0 and self.baseline_scale == 1 and self.advantage_estimator == 'rloo',
                'Budget runtime constants differ')

    def algorithm_config(self):
        candidate = self.external_baseline_extension
        return {'beta': 0.05 if self.method == 'foarl' else 0.0,
                'epsilon': 0.2, 'epsilon_high': 0.2, 'loss_type': 'dr_grpo' if candidate else 'grpo',
                'scale_rewards': 'none' if candidate else 'group', 'num_iterations': 1,
                'sync_ref_model': False, 'reward_weights': [1.0, 1.0] if self.method == 'foarl' else [1.0],
                'importance_sampling_level': 'token', 'mask_truncated_completions': False,
                'use_liger_loss': False, 'use_vllm': False}

    def _quality_runtime(self):
        return QualitySignalRuntime(task=self.task, advantage_estimator='rloo', num_generations=9,
                                    source_binding=self.source_binding, diagnostic=self.diagnostic)

    def _legacy_runtime(self):
        return ExternalBaselineRuntime(task=self.task, advantage_estimator='rloo', num_generations=9,
                                       source_binding=self.source_binding, diagnostic=self.diagnostic)

    def reward_contract(self):
        return self._quality_runtime().reward_contract()

    def advantage_contract(self):
        return make_advantage_contract(ADVANTAGE_METHODS[self.method])

    def resume_contract(self):
        self._check_binding()
        return {'protocol': PROTOCOL, 'method': self.method, 'task': self.task, 'seed': 43,
                'horizon': self.contract['horizon'], 'contract_file': copy.deepcopy(dict(self.contract_binding)),
                'declaration': copy.deepcopy(dict(self.contract)),
                'source': copy.deepcopy(self.contract['source']), 'algorithm_config': self.algorithm_config(),
                'reward_contract': self.reward_contract(), 'advantage_contract': self.advantage_contract()}

    def validate_args(self, args):
        self._check_binding()
        require(tuple(args.tasks) == (self.task,) and args.budget_method == self.method,
                'Budget CLI task/method differs from the declaration')
        require(bool(args.budget_diagnostic) == self.diagnostic, 'Budget diagnostic mode differs')
        expected = {**COMMON_DEFAULTS, 'max_steps': self.contract['horizon'],
                    'save_steps': 5 if self.diagnostic else 50,
                    'grpo_beta': self.algorithm_config()['beta'],
                    'grpo_loss_type': self.algorithm_config()['loss_type'],
                    'verify_gt_rewards': self.method != 'foarl'}
        drift = {key: {'expected': value, 'observed': getattr(args, key, None)}
                 for key, value in expected.items() if getattr(args, key, None) != value}
        require(not drift, 'Budget common numerical recipe differs: ' + str(drift))
        require(os.environ.get('WORLD_SIZE', '1') == '4' and os.environ.get('GRPO_DATA_SEED', '43') == '43',
                'Budget study requires four ranks and matching seed43 data order')
        require(os.environ.get('GRPO_STRICT_OPTIMIZER_RESUME') == '1', 'Budget chunks require strict optimizer resume')
        stop = os.environ.get('GRPO_STOP_AT_GLOBAL_STEP', '')
        require(stop.isdigit() and int(stop) in self.contract['chunk_boundaries'], 'Stop is not a declared chunk boundary')
        require(not any(value for name, value in os.environ.items()
                        if name == 'GRPO_REBASE_LOADED_SCHEDULER_AT_STEP' or name.startswith('GRPO_EXTENSION_')),
                'Fixed budget forbids schedule extensions/rebasing')
        require(all(os.environ.get(name) == value for name, value in reward_environment(self.method).items()),
                'Budget reward environment differs')
        if self.requires_explicit_fixed_reference:
            require(all(os.environ.get(name, '').strip() and Path(os.environ[name]).is_absolute() for name in KL_ENVIRONMENT),
                    'FOARL requires the explicit fixed initial-SFT KL reference and receipt')
        else:
            require(not any(os.environ.get(name, '').strip() for name in KL_ENVIRONMENT), 'Beta-zero methods forbid an explicit KL reference')
        for field in ('data_root', 'output_dir'):
            if field in self.contract:
                require(str(Path(getattr(args, field)).absolute()) == self.contract[field], 'Budget ' + field + ' differs')
        if 'initialization' in self.contract:
            require(str(Path(args.sft_checkpoint_path).absolute()) == self.contract['initialization']['checkpoint'],
                    'Budget SFT checkpoint differs')
        require(getattr(args, 'resolved_generation_batch_size', 36) == 36
                and getattr(args, 'resolved_steps_per_generation', None) in (None, 3), 'Budget rollout buffer differs')
        from clasp.data.prompts import JIANG_F638_INFERENCE_TEMPLATE
        require(hashlib.sha256(JIANG_F638_INFERENCE_TEMPLATE.encode()).hexdigest() == COMMON_PROMPT_SHA256,
                'Budget common Jiang prompt changed')

    def grpo_config_overrides(self, grpo_signature):
        config = self.algorithm_config()
        require(not (set(config) - set(grpo_signature)), 'Installed GRPOConfig cannot express the budget recipe')
        return config

    def validate_grpo_config(self, config):
        expected = {**self.algorithm_config(), 'num_generations': 9, 'max_steps': self.contract['horizon'],
                    'generation_batch_size': 36, 'gradient_accumulation_steps': 3, 'steps_per_generation': 3,
                    'ignore_data_skip': False, 'eval_strategy': 'no'}
        for name, value in expected.items():
            require(getattr(config, name, None) == value, 'Resolved budget GRPOConfig differs: ' + name)

    def reward_functions(self):
        return self._quality_runtime().reward_functions()

    def validate_dataset(self, dataset, *, label):
        self._check_binding()
        result = self._quality_runtime().validate_dataset(dataset, label=label)
        if label == 'train':
            require(len(dataset) == self.contract['expected_train_count'], 'Retained training count differs from contract')
        else:
            require(len(dataset) == 0, 'Budget training forbids in-loop evaluation data')
        return result

    def validate_trainer_implementation(self, trainer_class):
        if self.external_baseline_extension:
            return ExternalBaselineRuntime.validate_trainer_implementation(self, trainer_class)
        return validate_trl_foarl_implementation(trainer_class)

    def metadata(self):
        self._check_binding()
        return {'profile_id': PROFILE_ID, 'method_id': self.method, 'task': self.task,
                'study_scope': 'distributed_diagnostic' if self.diagnostic else 'prospective_development_comparison',
                'reproduction_claim': 'not_an_official_reproduction', 'policy_improvement_established': False,
                'seed': 43, 'world_size': 4, 'num_generations': 9, 'max_steps': self.contract['horizon'],
                'source_binding': copy.deepcopy(dict(self.source_binding)), 'contract_binding': copy.deepcopy(dict(self.contract_binding)),
                'algorithm_config': self.algorithm_config(), 'reward_contract': self.reward_contract(),
                'advantage_contract': self.advantage_contract(),
                'reward_heads': [head.__name__ for head in self.reward_functions()],
                'reward_environment': reward_environment(self.method), 'prompt_sha256': COMMON_PROMPT_SHA256,
                'no_reason': True, 'teacher_weight': 0.0,
                'common_budget': {'updates': self.contract['horizon'], 'responses_per_update': 36,
                                  'responses': self.contract['horizon'] * 36,
                                  'prompt_exposures': self.contract['horizon'] * 4,
                                  'expected_train_count': self.contract['expected_train_count'],
                                  'training_sampling': {'temperature': 0.7, 'top_p': 0.9, 'top_k': 50},
                                  'equal_tokens_or_compute_claimed': False, 'published_B8_candidate_multiplier': 1.125},
                'continuation': {'fixed_scheduler_horizon': self.contract['horizon'], 'strict_optimizer_resume': True,
                                 'chunk_boundaries': list(self.contract['chunk_boundaries']),
                                 'checkpoint_save_total_limit': 3,
                                 'unevaluated_milestones_require_external_launch_gate': True,
                                 'rank_owned_optimizer_scheduler_state': True,
                                 'diagnostic_policy_must_not_be_promoted': self.diagnostic},
                'resume_contract': self.resume_contract()}


def build_runtime(source_binding, contract, contract_binding):
    return FeasibleQualityRuntime(contract=contract, contract_binding=contract_binding, source_binding=source_binding)
