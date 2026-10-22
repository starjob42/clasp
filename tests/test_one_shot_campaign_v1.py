"""Controller failures must not create duplicate attempts or start model calls."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('campaign_under_test', ROOT / 'experiments/rloo_one_shot_20261016/run_campaign.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
ROSTER = ROOT / 'experiments/rloo_one_shot_20261016/execution_v1/execution_roster.json'


def store(path, value):
    path.write_text(json.dumps(value))
    return M.binding(path)


def test_real_roster_dry_run_cannot_launch_or_create_control(tmp_path, monkeypatch):
    monkeypatch.setattr(M.subprocess, 'Popen', lambda *a, **k: pytest.fail('Unexpected child'))
    output = tmp_path / 'control'
    result = M.run(M.binding(ROSTER), output, {}, {}, {})
    assert result == {'status': 'prepared_only', 'model_calls': 0, 'new_local_requested_gpu_hours': 42}
    assert not output.exists()


@pytest.mark.parametrize('mutation', ['budget', 'duplicate', 'retry'])
def test_bad_roster_rejected_before_execution(tmp_path, mutation):
    value = json.loads(ROSTER.read_text())
    if mutation == 'budget':
        value['slots'][0]['wall_seconds'] += 1
    elif mutation == 'duplicate':
        value['slots'][1]['slot'] = value['slots'][0]['slot']
    else:
        value['slots'][0]['maximum_attempts'] = 2
    with pytest.raises(ValueError):
        M.run(store(tmp_path / 'roster.json', value), tmp_path / 'control', {}, {}, {})


def test_bound_command_uses_fixed_runner_and_contract():
    roster = json.loads(ROSTER.read_text())
    slot = roster['slots'][2]
    argv = M.command(roster, slot)
    assert argv[:4] == [M.PYTHON, '-I', '-B', '-c']
    assert argv[5:7] == [roster['runner']['path'], roster['runner']['sha256']]
    assert argv[-3:] == ['--output', slot['output'], '--execute']
    assert '--contract-sha256' in argv


def test_changed_contract_rejected_before_command(tmp_path):
    roster = json.loads(ROSTER.read_text())
    slot = copy.deepcopy(roster['slots'][2])
    contract = tmp_path / 'contract.json'
    slot['contract'] = store(contract, {})
    contract.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='changed'):
        M.command(roster, slot)


def test_actual_qualification_system_metadata_admits():
    # No model loading, no completion text decoding or scoring.
    proofs = M.verify_qualifications(json.loads(ROSTER.read_text()))
    assert len(proofs) == 2


def test_failed_qualification_never_admits(tmp_path):
    roster = json.loads(ROSTER.read_text())
    slot = roster['slots'][0]
    source = Path(slot['output'])
    for name in ('result.json', 'completion.json'):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    completion = json.loads((tmp_path / 'completion.json').read_text())
    completion['status'] = 'failed'
    (tmp_path / 'completion.json').write_text(json.dumps(completion))
    slot['output'] = str(tmp_path)
    with pytest.raises(ValueError, match='Qualification failed'):
        M.verify_qualifications(roster)


@pytest.mark.parametrize('free,util,apps', [(46000, 0, ''), (49000, 1, ''), (49000, 0, 'GPU-test\n')])
def test_nonidle_gpu_rejected_without_reservation_probes(monkeypatch, free, util, apps):
    replies = iter([SimpleNamespace(returncode=0, stdout=f'GPU-test, {free}, {util}'),
                    SimpleNamespace(returncode=0, stdout=apps)])
    monkeypatch.setattr(M.subprocess, 'run', lambda *a, **k: next(replies))
    monkeypatch.setattr(M.os, 'open', lambda *a, **k: pytest.fail('Probe while nonidle'))
    assert M.gpu_is_idle('GPU-test') is False


def test_existing_attempt_intent_cannot_be_overwritten(tmp_path):
    path = tmp_path / 'attempt.json'
    M.write_once(path, {'attempt': 1})
    with pytest.raises(FileExistsError):
        M.write_once(path, {'attempt': 2})
    assert json.loads(path.read_text()) == {'attempt': 1}
