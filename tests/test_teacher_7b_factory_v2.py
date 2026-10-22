"""The production factory must reject absent source/lease and tiny CPU models."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'experiments/rloo_one_shot_20261016/teacher_7b_factory_v2.py'
SPEC = importlib.util.spec_from_file_location('prospective_factory', PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
SOURCE = ROOT / 'experiments/rloo_one_shot_20261016/execution_v1/source'


@pytest.fixture
def source_context(monkeypatch):
    runtime = SOURCE/'run_local_7b_v1.py'
    base = SimpleNamespace(__file__=str(runtime), __executed_source_sha256__=M.BASE_RUNTIME_SHA,
                           GPU_UUID='GPU-025e10e6-263f-d814-6dd5-added86fc8af')
    core = SimpleNamespace(__file__=str(SOURCE/'src/clasp/entrypoints/train_sft_anchor_one_shot_v1.py'),
                           __executed_source_sha256__=M.CORE_SHA,
                           one_shot_quality=SimpleNamespace(__file__=str(SOURCE/'src/clasp/rewards/one_shot_quality.py'),
                                                           __executed_source_sha256__=M.REWARD_SHA))
    owner = {'pid':os.getpid(), 'gpu_uuid':base.GPU_UUID, 'all_locks_held_until_GPU_lifetime_closed':True}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',base.GPU_UUID)
    return base,core,owner


def test_missing_executed_source_rejected_before_parameter_access(source_context):
    base,core,owner = source_context
    base.__executed_source_sha256__ = None
    with pytest.raises(ValueError,match='source buffers'):
        M.require_loaded_production_context(base,core,None,None,owner,expected_actor_sha256='a'*64)


@pytest.mark.parametrize('field,value', [('pid',-1), ('gpu_uuid','GPU-different'),
                                      ('all_locks_held_until_GPU_lifetime_closed',False)])
def test_unowned_context_rejected_before_model_access(source_context,field,value):
    base,core,owner = source_context
    owner[field] = value
    with pytest.raises(ValueError,match='UUID lease'):
        M.require_loaded_production_context(base,core,None,None,owner,expected_actor_sha256='a'*64)


def test_wrong_visible_device_rejected_before_model_access(source_context,monkeypatch):
    base,core,owner = source_context
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1,2,3')
    with pytest.raises(ValueError,match='UUID lease'):
        M.require_loaded_production_context(base,core,None,None,owner,expected_actor_sha256='a'*64)


def test_tiny_cpu_parameters_cannot_qualify_as_production(source_context):
    base,core,owner = source_context
    namespace = {'__file__':base.__file__, '__name__':'verified_runtime_guard'}
    exec(compile(Path(base.__file__).read_bytes(),base.__file__,'exec',dont_inherit=True),namespace)
    base.parameter_dtype_receipt = namespace['parameter_dtype_receipt']
    param = SimpleNamespace(numel=lambda:1, dtype='torch.bfloat16', device='cpu', requires_grad=True,
                            is_floating_point=lambda:True)
    tiny = SimpleNamespace(named_parameters=lambda:[('weight',param)])
    with pytest.raises(ValueError,match='Actual6B'):
        M.require_loaded_production_context(base,core,tiny,tiny,owner,expected_actor_sha256='a'*64)
