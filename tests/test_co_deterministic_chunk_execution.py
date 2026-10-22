"""Exercise actual chunk argv/environment and fail-closed launch boundaries.

Only machine interfaces and the separately tested checkpoint inspector are
stubbed. These tests neither load a model nor invoke Slurm or GPUs.
"""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from clasp.baselines import foarl_f638_dev_20260907 as source_guard

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def chunk(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "run_foarl_teacher_dev_chunk", load_script("run_foarl_teacher_dev_chunk"))
    module = load_script("run_co_deterministic_dev_chunk")
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    slurm = source / "scripts/train_co_deterministic_dev_a100.slurm"
    slurm.write_bytes((ROOT / "scripts/train_co_deterministic_dev_a100.slurm").read_bytes())
    spool = tmp_path / "spool.slurm"
    spool.write_bytes(slurm.read_bytes())
    (source / "SOURCE_MANIFEST.sha256").write_text("synthetic pinned manifest\n")
    sft_root, data = tmp_path / "sft", tmp_path / "data"
    sft = sft_root / "runs/mis/seed42/checkpoints/checkpoint-31250"
    sft.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json"):
        (sft / name).write_text(name)
    (data / "mis/train").mkdir(parents=True)
    (data / "mis/train/train.json").write_text('[{"synthetic":true}]\n')
    output = tmp_path / "teacher"
    binding = {"status": "passed_frozen_manifest_successor_source", "verified_file_count": 17,
               "required_files": {"src/clasp/entrypoints/train_grpo.py": {"sha256": "a" * 64}}}
    monkeypatch.setattr(module, "RUNTIME", Path(sys.executable).parent.parent)
    monkeypatch.setattr(module, "DATA", data)
    monkeypatch.setattr(module, "SFT", sft_root)
    monkeypatch.setattr(module, "SFT_HASHES", {"mis": module.identity(sft / "adapter_model.safetensors")["sha256"]})
    # Replace the mapping itself so scrubbed inherited variables are restored.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ.update(CO_SOURCE=str(source), CO_SOURCE_SHA="a" * 64, CO_TASK="mis",
                      CO_OUTPUT=str(output), CO_STOP="100",
                      CO_METHOD="classic_grpo", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="43", CO_DIAGNOSTIC="0", CO_RESUME="", CO_COMPARE_WITH="", SLURM_JOB_ID="12345",
                      WORLD_SIZE="4", RESUME_FROM_CHECKPOINT="/poisoned/resume",
                      CLASP_KL_REFERENCE_MODEL_PATH="/poisoned/reference",
                      CLASP_KL_REFERENCE_RECEIPT="/poisoned/receipt", GRPO_GRADED_REWARD="1",
                      GRPO_STOP_AT_GLOBAL_STEP="999", GRPO_VERIFIED_IMPROVEMENT="1",
                      FOARL_UNDECLARED_CONTROL="poison", PD_UNDECLARED_CONTROL="poison")
    monkeypatch.setattr(sys, "argv", ["chunk", "--executed-slurm", str(spool)])
    os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
    source_checks, validations, commands = [], [], []
    seals = []

    def verify(path, digest):
        source_checks.append((Path(path), digest))
        return binding

    def inspect(checkpoint, **kwargs):
        validations.append((Path(checkpoint), kwargs))
        contract = json.loads(kwargs["run_contract_path"].read_text())
        if contract != kwargs["expected_contract"]:
            raise ValueError("existing arm contract differs")
        Path(checkpoint).mkdir(parents=True, exist_ok=True)
        return {"status": "synthetic_checkpoint_inspected", "step": kwargs["expected_step"]}

    def execute(argv, **kwargs):
        # Independent descriptor cannot enter the arm while a child runs.
        with (output / ".chunk.lock").open("a") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        commands.append({"argv": argv, "environment": dict(os.environ), "kwargs": kwargs})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(source_guard, "verify_dev_source_manifest", verify)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "2.9.1")
    monkeypatch.setattr(module, "validate_checkpoint", lambda method, checkpoint, step, contract, path:
        inspect(checkpoint, expected_step=step, expected_contract=contract, run_contract_path=path))
    def declaration(output, contract):
        observed = json.loads((output / "local_development_run_contract.json").read_text())
        if observed != contract:
            raise ValueError("Existing original contract changed")
        return {"contract": observed}
    monkeypatch.setattr(module, "write_runtime_declaration", lambda output, contract:
        module.write_once(output / module.DECLARATION_NAME, declaration(output, contract)))
    monkeypatch.setattr(module, "read_only_runtime_declaration", declaration)
    evidence = []
    def save_evidence(output, checkpoint, stop, contract):
        assert json.loads((output / f"chunk-{stop}-exit.json").read_text())["returncode"] == 0
        evidence.append((Path(checkpoint), stop))
        return {"synthetic_applied_flags": True}
    monkeypatch.setattr(module, "save_runtime_evidence", save_evidence)
    monkeypatch.setattr(module, "read_only_runtime_evidence", lambda *a: {"synthetic_applied_flags": True})
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "NVIDIA A100\n" * 4)
    monkeypatch.setattr(module.subprocess, "run", execute)
    return SimpleNamespace(module=module, source=source, spool=spool, sft=sft, data=data,
                           output=output, binding=binding, source_checks=source_checks,
                           validations=validations, seals=seals, commands=commands, evidence=evidence)


@pytest.mark.parametrize('method', ['foarl_control', 'foarl_teacher', 'classic_grpo'])
@pytest.mark.parametrize('diagnostic,stop', [(False, 100), (True, 5), (True, 10)])
def test_actual_method_routing_and_environment(chunk, method, diagnostic, stop):
    os.environ.update(CO_METHOD=method, CO_DIAGNOSTIC=str(int(diagnostic)), CO_STOP=str(stop))
    chunk.module.main()
    child = next(c for c in chunk.commands if c['argv'][0] == 'srun')
    argv, env = child['argv'], child['environment']
    entry = argv.index('clasp.entrypoints.train_co_deterministic_dev')
    assert argv[entry + 1:entry + 3] == ['--co_method', method]
    prefix = 'classic' if method == 'classic_grpo' else 'foarl'
    assert argv[argv.index('--' + prefix + '_dev_source_manifest_sha256') + 1] == 'a' * 64
    assert argv[argv.index('--sft_checkpoint_path') + 1] == str(chunk.sft)
    assert env['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8' and env['PYTHONHASHSEED'] == '43'
    assert env.get('CUDA_LAUNCH_BLOCKING') is None
    assert env['GRPO_STOP_AT_GLOBAL_STEP'] == str(stop)
    assert env['GRPO_STRICT_OPTIMIZER_RESUME'] == '1'
    assert env['CLASP_GRADIENT_CONSISTENCY_CHECKS'] == str(int(diagnostic))
    assert 'RESUME_FROM_CHECKPOINT' not in env
    assert 'FOARL_UNDECLARED_CONTROL' not in env and 'PD_UNDECLARED_CONTROL' not in env
    if method == 'classic_grpo':
        assert not any(k.startswith('CLASP_KL_REFERENCE_') for k in env)
        assert env['GRPO_GRADED_REWARD'] == '0'
        assert env['CLASP_OBJECTIVE_QUALITY_MODE'] == 'exp'
        assert '--foarl_teacher_weight' not in argv
    else:
        assert env['CLASP_KL_REFERENCE_MODEL_PATH'] == str(chunk.sft)
        assert env['CLASP_KL_REFERENCE_RECEIPT'] == str(chunk.sft / 'checkpoint_fidelity.json')
        assert float(argv[argv.index('--foarl_teacher_weight') + 1]) == (.05 if method == 'foarl_teacher' else 0.)
    assert chunk.source_checks == [(chunk.source / 'SOURCE_MANIFEST.sha256', 'a' * 64)]
    contract = json.loads((chunk.output / 'local_development_run_contract.json').read_text())
    assert contract['fixed_horizon'] == {'max_steps': 400, 'world_size': 4}
    assert contract['method'] == method
    assert contract['schedule']['save_steps'] == (5 if diagnostic else 100)
    assert ('beta' in contract) == (method == 'classic_grpo')
    assert len(chunk.evidence) == (2 if diagnostic and stop == 10 else 1)
    assert (chunk.output / f'chunk-{stop}-complete.json').is_file()
    assert chunk.seals == []  # Callback owns provisional sealing; launcher admits it.


@pytest.mark.parametrize('method', ['foarl_control', 'foarl_teacher', 'classic_grpo'])
def test_resume_preserves_original_contract_and_runtime_admission(chunk, method):
    os.environ['CO_METHOD'] = method
    chunk.module.main()
    before = (chunk.output / 'local_development_run_contract.json').read_bytes()
    checkpoint = chunk.output / 'checkpoints/checkpoint-100'
    os.environ.update(CO_STOP='200', CO_RESUME=str(checkpoint))
    chunk.commands.clear()
    chunk.module.main()
    assert chunk.commands[0]['environment']['RESUME_FROM_CHECKPOINT'] == str(checkpoint)
    assert (chunk.output / 'local_development_run_contract.json').read_bytes() == before
    count = len(chunk.commands)
    with pytest.raises(FileExistsError):
        chunk.module.main()
    assert len(chunk.commands) == count


@pytest.mark.parametrize('failure', ['manifest', 'slurm', 'gpu', 'sft', 'environment'])
def test_launch_rejects_wrong_source_hardware_or_prestart_flags(chunk, monkeypatch, failure):
    if failure == 'manifest':
        monkeypatch.setattr(source_guard, 'verify_dev_source_manifest', lambda *a: (_ for _ in ()).throw(ValueError('source')))
    elif failure == 'slurm':
        chunk.spool.write_text('changed script')
    elif failure == 'gpu':
        monkeypatch.setattr(chunk.module.subprocess, 'check_output', lambda *a, **k: 'NVIDIA A100\n' * 3)
    elif failure == 'sft':
        (chunk.sft / 'adapter_model.safetensors').write_text('changed weights')
    else:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    with pytest.raises(ValueError):
        chunk.module.main()
    assert not chunk.output.exists() and chunk.commands == []


def test_failed_child_cannot_create_applied_runtime_or_checkpoint_completion(chunk, monkeypatch):
    monkeypatch.setattr(chunk.module.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=7))
    with pytest.raises(SystemExit) as exc:
        chunk.module.main()
    assert exc.value.code == 7 and chunk.evidence == []
    assert not (chunk.output / 'chunk-100-complete.json').exists()
    assert json.loads((chunk.output / 'chunk-100-exit.json').read_text())['returncode'] == 7


@pytest.mark.parametrize('method', ['foarl_teacher', 'classic_grpo'])
def test_only_diagnostic_resume_compares_step10_states(chunk, tmp_path, method):
    os.environ.update(CO_METHOD=method, CO_DIAGNOSTIC='1', CO_STOP='5')
    chunk.module.main()
    full = tmp_path / 'full/checkpoints/checkpoint-10'
    full.mkdir(parents=True)
    os.environ.update(CO_STOP='10', CO_RESUME=str(chunk.output / 'checkpoints/checkpoint-5'), CO_COMPARE_WITH=str(full))
    chunk.commands.clear()
    chunk.module.main()
    assert len(chunk.commands) == 2
    assert chunk.commands[1]['argv'] == [sys.executable, '-B', str(chunk.source / 'scripts/compare_co_deterministic_state.py'),
        '--uninterrupted', str(full), '--resumed', str(chunk.output / 'checkpoints/checkpoint-10'), '--step', '10',
        '--output', str(chunk.output / 'resume_equivalence.json')]
    assert chunk.commands[1]['kwargs'] == {'check': True}


def test_a100_slurm_exports_flags_before_launcher_startup():
    text = (ROOT / 'scripts/train_co_deterministic_dev_a100.slurm').read_text()
    for directive in ('--account=vas@a100', '--partition=gpu_p5', '--constraint=a100', '--gres=gpu:4',
                      '--nodes=1', '--cpus-per-task=40', '--hint=nomultithread', '--time=02:00:00',
                      '--qos=qos_gpu_a100-dev', '--no-requeue'):
        assert text.splitlines().count('#SBATCH ' + directive) == 1
    assert text.index('CUBLAS_WORKSPACE_CONFIG=:4096:8') < text.index('exec "${RUNTIME}/bin/python"')
    assert 'scripts/run_co_deterministic_dev_chunk.py --executed-slurm "${BASH_SOURCE[0]}"' in text
