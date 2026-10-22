"""Deterministic execution admission without changing any scientific runtime."""
from __future__ import annotations

import copy
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from clasp.baselines import co_deterministic_dev as deterministic
from clasp.baselines.classic_grpo_400_dev import REWARD_ENVIRONMENT
from clasp.entrypoints import train_co_deterministic_dev as entry

NATIVE_TORCH_PIN = deterministic.TORCH_VERSION


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    previous = deterministic.determinism_state()
    rng = torch.random.get_rng_state().clone()
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items()
        if not k.startswith(("CLASP_", "GRPO_", "PD_", "FOARL_", "CUBLAS_", "CUDA_LAUNCH_"))})
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=deterministic.CUBLAS_WORKSPACE_CONFIG,
                      PYTHONHASHSEED="43",
                      WORLD_SIZE="4", GRPO_DATA_SEED="43", GRPO_STOP_AT_GLOBAL_STEP="100",
                      GRPO_STRICT_OPTIMIZER_RESUME="1", **REWARD_ENVIRONMENT)
    # Exercise actual CPU backend controls locally, without pretending this is
    # native2.9.1 kernel/reproducibility evidence. The native pin itself and its
    # rejection behavior are tested separately below.
    monkeypatch.setattr(deterministic, "TORCH_VERSION", str(torch.__version__).split("+")[0])
    yield
    torch.use_deterministic_algorithms(previous["deterministic_algorithms"],
                                       warn_only=previous["deterministic_warn_only"])
    torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
    torch.backends.cuda.enable_flash_sdp(previous["flash_sdp_enabled"])
    torch.backends.cuda.enable_mem_efficient_sdp(previous["efficient_sdp_enabled"])
    torch.backends.cuda.enable_math_sdp(previous["math_sdp_enabled"])
    torch.backends.cuda.enable_cudnn_sdp(previous["cudnn_sdp_enabled"])
    torch.random.set_rng_state(rng)


def test_native_version_pin_and_rejection_before_setting_flags(monkeypatch):
    assert NATIVE_TORCH_PIN == "2.9.1"
    before = deterministic.determinism_state()
    monkeypatch.setattr(deterministic, "TORCH_VERSION", "0.0.0")
    with pytest.raises(ValueError, match="pinned PyTorch version"):
        deterministic.apply_strict_determinism()
    assert deterministic.determinism_state() == before


def test_python_hash_seed_must_be_predeclared(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED")
    with pytest.raises(ValueError, match="PYTHONHASHSEED=43"):
        deterministic.apply_strict_determinism()


def test_manual_strict_flags_preserve_rng_environment_and_cudnn_sdp_flag():
    before_rng = torch.random.get_rng_state().clone()
    before_environment = dict(os.environ)
    before = deterministic.determinism_state()
    receipt = deterministic.apply_strict_determinism()
    state = deterministic.validate_strict_determinism()
    assert state["deterministic_algorithms"] is True
    assert state["deterministic_warn_only"] is False
    assert state["cudnn_deterministic"] is True
    assert state["cudnn_benchmark"] is False
    assert all(state[key] for key in ("flash_sdp_enabled", "efficient_sdp_enabled", "math_sdp_enabled"))
    assert state["cudnn_sdp_enabled"] == before["cudnn_sdp_enabled"]
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert dict(os.environ) == before_environment
    assert receipt["applied_before_cuda_initialization"] is True
    assert receipt["hf_full_determinism"] is False
    assert receipt["exact_continuation_proved"] is False


@pytest.mark.parametrize("value", [None, "", ":16:8", ":4096:2", "invalid"])
def test_workspace_must_be_predeclared_before_even_importing_torch(value, monkeypatch):
    if value is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG")
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = value
    monkeypatch.setattr(deterministic, "_torch", lambda: pytest.fail("torch imported before environment check"))
    with pytest.raises(ValueError, match="before Python child startup"):
        deterministic.apply_strict_determinism()


def test_already_initialized_cuda_rejected_before_setting_flags(monkeypatch):
    before = deterministic.determinism_state()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="before CUDA initialization"):
        deterministic.apply_strict_determinism()
    assert deterministic.determinism_state() == before


def test_debug_blocking_environment_rejected():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    with pytest.raises(ValueError, match="debugging"):
        deterministic.apply_strict_determinism()


@pytest.mark.parametrize("drift", ["disabled", "warn_only", "cudnn_nondeterministic", "benchmark",
                                  "flash", "efficient", "math", "workspace"])
def test_execution_drift_is_rejected(drift):
    deterministic.apply_strict_determinism()
    actions = {
        "disabled": lambda: torch.use_deterministic_algorithms(False),
        "warn_only": lambda: torch.use_deterministic_algorithms(True, warn_only=True),
        "cudnn_nondeterministic": lambda: setattr(torch.backends.cudnn, "deterministic", False),
        "benchmark": lambda: setattr(torch.backends.cudnn, "benchmark", True),
        "flash": lambda: torch.backends.cuda.enable_flash_sdp(False),
        "efficient": lambda: torch.backends.cuda.enable_mem_efficient_sdp(False),
        "math": lambda: torch.backends.cuda.enable_math_sdp(False),
        "workspace": lambda: os.environ.update(CUBLAS_WORKSPACE_CONFIG=":16:8"),
    }
    actions[drift]()
    with pytest.raises(ValueError):
        deterministic.validate_strict_determinism()


def actual_base(method, task="mis"):
    from clasp.baselines.classic_grpo_400_dev import build_runtime as classic
    from clasp.baselines.foarl_teacher_dev import build_runtime as foarl
    from clasp.config import finalize_grpo_args
    from clasp.entrypoints.train_classic_grpo_400_dev import build_parser as classic_parser
    from clasp.entrypoints.train_foarl_teacher_dev import build_parser as foarl_parser
    binding = {"status": "passed_frozen_manifest_successor_source",
               "required_files": {"src/clasp/entrypoints/train_grpo.py": {"sha256": "a" * 64}}}
    common = ["--tasks", task, "--sft_checkpoint_path", "/synthetic/" + task]
    if method == "classic_grpo":
        args = classic_parser().parse_args(common + ["--classic_dev_source_manifest", "/source/SOURCE_MANIFEST.sha256",
                                                    "--classic_dev_source_manifest_sha256", "a" * 64])
        base = classic(binding, task)
    else:
        weight = .05 if method == "foarl_teacher" else 0.
        args = foarl_parser().parse_args(common + ["--foarl_teacher_weight", str(weight),
            "--foarl_dev_source_manifest", "/source/SOURCE_MANIFEST.sha256",
            "--foarl_dev_source_manifest_sha256", "a" * 64])
        base = foarl(binding, weight)
    return finalize_grpo_args(args), base


@pytest.mark.parametrize("method", deterministic.METHODS)
@pytest.mark.parametrize("task", ["mis", "cvrp"])
def test_actual_method_hooks_rewards_config_and_budgets_remain_identical(method, task):
    initialization = deterministic.apply_strict_determinism()
    args, base = actual_base(method, task)
    before_args = copy.deepcopy(vars(args))
    base_metadata = base.metadata()
    runtime = deterministic.wrap_runtime(base, method, initialization)
    runtime.validate_args(args)
    assert vars(args) == before_args
    assert runtime.reward_functions() == base.reward_functions()
    assert runtime.teacher_weight == base.teacher_weight
    assert runtime.requires_explicit_fixed_reference == base.requires_explicit_fixed_reference
    assert runtime.verified_teacher_extension == base.verified_teacher_extension
    signature = {key: None for key in ("beta", "epsilon", "epsilon_high", "loss_type", "scale_rewards",
        "num_iterations", "sync_ref_model", "reward_weights", "importance_sampling_level",
        "mask_truncated_completions", "use_liger_loss")}
    overrides = runtime.grpo_config_overrides(signature)
    assert overrides == base.grpo_config_overrides(signature)
    config = SimpleNamespace(**{**overrides, "num_generations": 9, "max_steps": 400,
        "generation_batch_size": 36, "gradient_accumulation_steps": 3, "steps_per_generation": 3,
        "ignore_data_skip": False, "full_determinism": False})
    before_config = copy.deepcopy(vars(config))
    runtime.validate_grpo_config(config)
    assert vars(config) == before_config
    metadata = runtime.metadata()
    # The launcher binds this exact initialization to rank-zero postrun proof.
    assert metadata["determinism"] == initialization
    assert metadata["method_id"] == method
    assert metadata["base_profile_id"] == base_metadata["profile_id"]
    assert metadata["profile_id"] == deterministic.PROFILE_ID
    projected = {k: v for k, v in metadata.items() if k not in ("base_profile_id", "execution_successor", "determinism")}
    projected["profile_id"] = base_metadata["profile_id"]
    assert projected == base_metadata
    assert base.metadata() == base_metadata
    assert args.max_steps == 400 and args.seed == 43
    assert args.generation_batch_size == 36 and args.gradient_accumulation_steps == 3
    # The actual original validators still reject any scientific drift.
    args.learning_rate = 1e-4
    with pytest.raises((ValueError, RuntimeError)):
        runtime.validate_args(args)


def test_method_weight_mismatch_and_missing_initialization_are_rejected():
    receipt = deterministic.apply_strict_determinism()
    _, base = actual_base("foarl_teacher")
    with pytest.raises(ValueError, match="scientific runtime differ"):
        deterministic.wrap_runtime(base, "foarl_control", receipt)
    with pytest.raises(ValueError, match="initialization receipt"):
        deterministic.wrap_runtime(base, "foarl_teacher", {})


def test_hf_full_determinism_is_rejected_at_args_and_resolved_config():
    receipt = deterministic.apply_strict_determinism()
    args, base = actual_base("classic_grpo")
    runtime = deterministic.wrap_runtime(base, "classic_grpo", receipt)
    args.full_determinism = True
    with pytest.raises(ValueError, match="HF debugging"):
        runtime.validate_args(args)
    with pytest.raises(ValueError, match="full_determinism=False"):
        runtime.validate_grpo_config(SimpleNamespace(full_determinism=True))


@pytest.mark.parametrize("method", deterministic.METHODS)
def test_entry_enables_determinism_before_unchanged_parser_and_delegates_argv(method, monkeypatch):
    args, base = actual_base(method)
    calls = []
    def parse(values):
        deterministic.validate_strict_determinism()
        calls.append(values)
        return args, base
    target = ("clasp.entrypoints.train_classic_grpo_400_dev" if method == "classic_grpo"
              else "clasp.entrypoints.train_foarl_teacher_dev")
    monkeypatch.setitem(sys.modules, target, SimpleNamespace(parse_args=parse))
    torch.use_deterministic_algorithms(False)
    result, runtime = entry.parse_args(["--co_method", method, "--synthetic-passthrough", "literal"])
    assert result is args and runtime.base is base
    assert calls == [["--synthetic-passthrough", "literal"]]
    assert args.co_method == method
    assert args.co_deterministic_runtime_receipt["profile_id"] == deterministic.PROFILE_ID
    old_name = "classic_runtime_receipt" if method == "classic_grpo" else "foarl_teacher_runtime_receipt"
    assert getattr(args, old_name) == args.co_deterministic_runtime_receipt


def test_entry_rejects_environment_before_existing_parser(monkeypatch):
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG")
    def forbidden(*args):
        pytest.fail("old parser called before deterministic environment admission")
    monkeypatch.setitem(sys.modules, "clasp.entrypoints.train_foarl_teacher_dev", SimpleNamespace(parse_args=forbidden))
    with pytest.raises(ValueError):
        entry.parse_args(["--co_method", "foarl_teacher"])


def test_entry_runs_common_training_with_proxy_and_checks_final_flags(monkeypatch):
    from clasp.models import classic_grpo_save_lineage, co_determinism_receipt
    args, base = actual_base("classic_grpo")
    receipt = deterministic.apply_strict_determinism()
    runtime = deterministic.wrap_runtime(base, "classic_grpo", receipt)
    monkeypatch.setattr(entry, "parse_args", lambda argv=None: (args, runtime))
    calls = []
    publications = []
    @contextmanager
    def installed(module, method):
        assert method == "classic_grpo"
        yield
    monkeypatch.setattr(classic_grpo_save_lineage, "install_classic_save_lineage", installed)
    monkeypatch.setattr(co_determinism_receipt, "write_postrun_receipt", lambda *values: publications.append(values))
    def train(arg, *, runtime_profile):
        calls.append((arg, runtime_profile))
        return "synthetic-output"
    monkeypatch.setitem(sys.modules, "clasp.entrypoints.train_grpo", SimpleNamespace(run_grpo=train))
    assert entry.main([]) == "synthetic-output"
    assert calls == [(args, runtime)]
    assert publications == [(args, runtime)]
    def drift(*args, **kwargs):
        torch.use_deterministic_algorithms(False)
    monkeypatch.setitem(sys.modules, "clasp.entrypoints.train_grpo", SimpleNamespace(run_grpo=drift))
    with pytest.raises(ValueError, match="settings changed"):
        entry.main([])
    assert publications == [(args, runtime)]
