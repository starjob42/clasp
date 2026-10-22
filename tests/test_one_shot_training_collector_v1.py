"""Training-only collector scope, native token custody and single-attempt guards."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'experiments/rloo_one_shot_20261016/collect_training_sft_v1.py'
SPEC = importlib.util.spec_from_file_location('training_collector_under_test', PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
ROSTER = ROOT / 'experiments/rloo_one_shot_20261016/execution_v1/execution_roster.json'


def reseal(value):
    value['contract_sha256'] = M.sha(M.encode({k:v for k,v in value.items() if k != 'contract_sha256'}))


@pytest.fixture
def contract(tmp_path):
    return M.make_contract(M.binding(ROSTER), 'op', M.binding(PATH), output=tmp_path/'once')


def test_real_training_contract_binds_only_registered_training(contract):
    _, _, rows, validation, _ = M.validate_contract(contract)
    assert len(rows) == 128 and len(validation) == 96
    assert not {r['semantic_id'] for r in rows} & {r['semantic_id'] for r in validation}
    assert len(set(contract['calibration_instance_ids'])) == 8
    assert set(contract['calibration_instance_ids']) <= {r['instance_id'] for r in rows}
    assert contract['maximum_policy_responses'] == 200


@pytest.mark.parametrize('field,value', [('training', 'validation'), ('seed', 74), ('optimizer_updates', 1),
                                      ('wall_seconds', 5401), ('split', 'validation'), ('maximum_attempts', 2),
                                      ('calibration_instance_ids', [])])
def test_resealed_scope_or_budget_changes_rejected(contract, field, value):
    if field == 'training':
        roster = json.loads(ROSTER.read_text())
        anchor = json.loads(Path(roster['slots'][2]['contract']['path']).read_text())
        value = anchor['evaluation']
    contract[field] = value
    reseal(contract)
    with pytest.raises(ValueError):
        M.validate_contract(contract)


def test_wrong_output_cannot_start_another_attempt(contract, tmp_path, monkeypatch):
    path = tmp_path/'contract.json'
    path.write_bytes(M.encode(contract))
    monkeypatch.setattr(sys, 'argv', [str(PATH), '--execute', '--contract', str(path),
                                    '--contract-sha256', M.sha(path.read_bytes()), '--output', str(tmp_path/'different')])
    monkeypatch.setattr(M.subprocess, 'run', lambda *a, **kw: pytest.fail('Started a subprocess'))
    with pytest.raises(ValueError, match='single bound attempt'):
        M.main()
    assert not (tmp_path/'different').exists()


class Tokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def __len__(self):
        return 10

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [4, 5]

    def decode(self, ids, *, skip_special_tokens):
        return 'good' if ids == [3, 9] else 'bad'


class Configuration:
    def __init__(self, sample):
        self.sample = sample

    def to_dict(self):
        return {'do_sample':self.sample, 'num_return_sequences':1, 'num_beams':1, 'max_new_tokens':3000}


class Core:
    @staticmethod
    def generation_config(tokenizer, cap, *, sample):
        assert cap == 3000
        return Configuration(sample)

    @staticmethod
    def score_completion(*args):
        return {'feasible':True, 'objective':1.0}


def record(row, *, group=None, sample=0, call=0):
    calibration = group is not None
    return {'instance_id':row['instance_id'], 'semantic_id':row['semantic_id'],
            'prompt_sha256':M.sha(row['prompt'].encode()), 'split':'training',
            'role':'gradient_calibration' if calibration else 'teacher_incumbent',
            'sample_index':sample, 'group_size':9 if calibration else 1, 'generation_calls_for_group':1,
            'generation_call_index':128+group if calibration else call, 'calibration_group_index':group,
            'global_stream_seed':100073 if calibration else 73,
            'RNG_before_group':{'CPU_sha256':'a'*64, 'CUDA_sha256':'b'*64},
            'RNG_after_group':{'CPU_sha256':'a'*64, 'CUDA_sha256':'b'*64},
            'prompt_token_ids':[4,5], 'prompt_attention_mask':[1,1], 'prompt_token_count':2,
            'padded_completion_token_ids':[3,9,0], 'completion_token_ids':[3,9], 'completion_mask':[1,1,0],
            'text':'good', 'generated_tokens':2, 'eos_observed':True, 'truncated':False,
            'actual_generation_config':Configuration(calibration).to_dict(),
            'use_model_defaults':False, 'extra_generation_kwargs':{},
            'strict_score':Core.score_completion()}


@pytest.fixture
def replay():
    rows = [{'instance_id':str(i), 'semantic_id':hashlib.sha256(str(i).encode()).hexdigest(),
             'prompt':'prompt'+str(i), 'instance_json':[[0]*(i%10+2),[],[]], 'output':'witness'} for i in range(128)]
    ids = M.calibration_ids(rows)
    by_id = {r['instance_id']:r for r in rows}
    greedy = [record(row, call=index) for index,row in enumerate(rows)]
    calibration = [record(by_id[identifier], group=group, sample=sample)
                   for group,identifier in enumerate(ids) for sample in range(9)]
    return greedy, calibration, rows, {'task':'op', 'calibration_instance_ids':ids}


def test_complete_training_only_token_and_physics_replay(replay):
    M.validate_collection_records(*replay, Tokenizer(), Core)


@pytest.mark.parametrize('field,value', [('prompt_token_ids',[1,4,5]), ('prompt_attention_mask',[1,0]),
                                      ('padded_completion_token_ids',[3,9,2]), ('completion_mask',[1,1,1]),
                                      ('generated_tokens',True), ('group_size',True), ('generation_calls_for_group',True),
                                      ('eos_observed',1), ('truncated',0), ('split','validation'),
                                      ('strict_score',{'feasible':True,'objective':2.0})])
def test_corrupt_token_role_or_physics_is_rejected(replay, field, value):
    replay[0][0][field] = value
    with pytest.raises(ValueError):
        M.validate_collection_records(*replay, Tokenizer(), Core)


def test_calibration_rng_reset_rejected(replay):
    for r in replay[1][9:18]:
        r['RNG_before_group']['CPU_sha256'] = 'c'*64
    with pytest.raises(ValueError, match='stream was reset'):
        M.validate_collection_records(*replay, Tokenizer(), Core)


def test_reordered_calibration_siblings_rejected(replay):
    replay[1][0], replay[1][1] = replay[1][1], replay[1][0]
    with pytest.raises(ValueError):
        M.validate_collection_records(*replay, Tokenizer(), Core)


def test_incomplete_training_collection_rejected(replay):
    replay[0].pop()
    with pytest.raises(ValueError, match='Incomplete'):
        M.validate_collection_records(*replay, Tokenizer(), Core)
