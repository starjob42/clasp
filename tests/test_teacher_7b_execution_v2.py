"""No-GPU guards for the prospective teacher study parent and source closure."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'experiments/rloo_one_shot_20261016/run_teacher_7b_v2.py'
SPEC = importlib.util.spec_from_file_location('teacher_execution_under_test',PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_default_dryrun_cannot_spawn_or_read_contract(monkeypatch,capsys):
    monkeypatch.setattr(sys,'argv',[str(PATH)])
    monkeypatch.setattr(M.subprocess,'run',lambda *a,**k:pytest.fail('Unexpected subprocess'))
    monkeypatch.setattr(M,'read_bound',lambda *a,**k:pytest.fail('Unexpected input consumption'))
    M.main()
    result = json.loads(capsys.readouterr().out)
    assert result['model_calls'] == result['GPU_calls'] == 0
    assert result['superiority_established'] is False


@pytest.fixture
def parent(tmp_path):
    path = tmp_path/'contract.json';path.write_text('{"kind":"synthetic_parent_guard_fixture"}')
    bound = M.binding(path)
    return {'status':'completed','error':None,'contract_file_sha256':bound['sha256'],
            'parent_elapsed_wall_seconds':10.0,'actual_GPU_hour_upper_bound':10/3600,
            'requested_gpu_hours':.5,'automatic_retry':False},bound


def test_completed_parent_resource_arithmetic(parent):
    value,bound = parent
    M.completed_parent(value,cap=1800,contract_binding=bound)


@pytest.mark.parametrize('field,value',[('status','failed'),('error','timeout'),('parent_elapsed_wall_seconds',True),
                                      ('parent_elapsed_wall_seconds',0),('parent_elapsed_wall_seconds',1801),
                                      ('parent_elapsed_wall_seconds',float('nan')),
                                      ('actual_GPU_hour_upper_bound',float('inf')),('actual_GPU_hour_upper_bound',1.0),
                                      ('requested_gpu_hours',4.0),('automatic_retry',True)])
def test_failed_overbudget_or_malformed_parent_is_rejected(parent,field,value):
    record,bound = parent;record[field] = value
    with pytest.raises(ValueError,match='resource proof'):
        M.completed_parent(record,cap=1800,contract_binding=bound)


def test_calibration_parent_elapsed_field_supported(parent):
    record,bound = parent
    record['actual_wall_seconds'] = record.pop('parent_elapsed_wall_seconds')
    record['requested_gpu_hours'] = 2.0
    M.completed_parent(record,cap=7200,contract_binding=bound)


@pytest.fixture
def extras(tmp_path,monkeypatch):
    values = {}
    for name in M.EXTRA_FILES:
        path = tmp_path/name;path.write_text('# synthetic source closure fixture only\n')
        values[name] = M.binding(path)
    monkeypatch.setattr(M,'__file__',str(tmp_path/PATH.name))
    return {'extra_sources':values},tmp_path


def test_complete_twelve_sibling_source_buffers(extras):
    contract,_ = extras
    result = M.extra_buffers(contract)
    assert set(result) == M.EXTRA_FILES and len(result) == 12


def test_omitted_source_cannot_enter_runtime(extras):
    contract,_ = extras;contract['extra_sources'].pop('teacher_candidate_v1.py')
    with pytest.raises(ValueError,match='source closure'):
        M.extra_buffers(contract)


def test_changed_source_cannot_enter_runtime(extras):
    contract,root = extras;(root/'train_teacher_candidate_v2.py').write_text('# changed')
    with pytest.raises(ValueError,match='bytes differ'):
        M.extra_buffers(contract)


def test_source_path_escape_cannot_enter_runtime(extras,tmp_path):
    contract,root = extras
    outside = root/'nested/teacher_candidate_v1.py';outside.parent.mkdir();outside.write_text('# fixture')
    contract['extra_sources']['teacher_candidate_v1.py'] = M.binding(outside)
    with pytest.raises(ValueError,match='sibling source'):
        M.extra_buffers(contract)


def test_stale_bytecode_cannot_enter_runtime(extras):
    contract,root = extras
    path = root/'__pycache__/unbound.pyc';path.parent.mkdir();path.write_bytes(b'unbound')
    with pytest.raises(ValueError,match='Unbound bytecode'):
        M.extra_buffers(contract)


@pytest.mark.parametrize('arm,mode,wall,updates,eval_rows',[
    ('rloo_teacher','qualification',1801,1,2),('rloo_teacher','study',14401,100,96),
    ('teacher_only','qualification',1800,1,2),('foarl_cost_teacher','qualification',1800,1,2),
    ('rloo_teacher','study',14400,101,96),('rloo_teacher','study',14400,100,95)])
def test_unregistered_attempt_budget_or_qualification_rejected_before_inputs(arm,mode,wall,updates,eval_rows,monkeypatch):
    value = {'protocol':M.PROTOCOL,'mode':mode,'arm':arm,'task':'op','seed':73,
             'optimizer_updates':updates,'evaluation_rows':eval_rows,'wall_seconds':wall,
             'attempt_index':1,'maximum_attempts':1,'automatic_retry':False,
             'superiority_established':False,'publication_eligible':False}
    value['contract_sha256'] = M.sha(M.encoded(value))
    monkeypatch.setattr(M,'read_bound',lambda *a,**k:pytest.fail('Read later inputs before numerical rejection'))
    with pytest.raises(ValueError):
        M.validate_contract(value)
