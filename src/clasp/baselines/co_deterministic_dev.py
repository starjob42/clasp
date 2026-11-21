"""Strict deterministic execution around unchanged combinatorial RL profiles.

This successor changes execution determinism only. It neither resets RNG state
nor changes rewards, optimizer settings, teacher weighting or the fixed horizon.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Mapping

PROFILE_ID = "co-strict-deterministic-development-20260909-v1"
METHODS = ("foarl_control", "foarl_teacher", "classic_grpo")
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
TORCH_VERSION = "2.9.1"


def _torch():
    # Keep module import safe before launcher environment validation.
    import torch
    return torch


def validate_environment(environment=None):
    environment = os.environ if environment is None else environment
    if environment.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise ValueError("CUBLAS_WORKSPACE_CONFIG=:4096:8 must be supplied before Python child startup")
    if environment.get("PYTHONHASHSEED") != "43":
        raise ValueError("PYTHONHASHSEED=43 must be supplied before Python child startup")
    if environment.get("CUDA_LAUNCH_BLOCKING") not in (None, "0"):
        raise ValueError("This execution profile does not admit CUDA_LAUNCH_BLOCKING debugging")
    return {"CUBLAS_WORKSPACE_CONFIG": environment["CUBLAS_WORKSPACE_CONFIG"],
            "PYTHONHASHSEED": environment["PYTHONHASHSEED"],
            "CUDA_LAUNCH_BLOCKING": environment.get("CUDA_LAUNCH_BLOCKING")}


def determinism_state(torch_module=None):
    torch = _torch() if torch_module is None else torch_module
    cuda = torch.backends.cuda
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "flash_sdp_enabled": bool(cuda.flash_sdp_enabled()),
        "efficient_sdp_enabled": bool(cuda.mem_efficient_sdp_enabled()),
        "math_sdp_enabled": bool(cuda.math_sdp_enabled()),
        "cudnn_sdp_enabled": bool(cuda.cudnn_sdp_enabled()),
    }


def validate_strict_determinism(torch_module=None):
    validate_environment()
    torch = _torch() if torch_module is None else torch_module
    if str(torch.__version__).split("+")[0] != TORCH_VERSION:
        raise ValueError("Strict native SDPA execution requires the pinned PyTorch version")
    state = determinism_state(torch)
    expected = {"deterministic_algorithms": True, "deterministic_warn_only": False,
                "cudnn_deterministic": True, "cudnn_benchmark": False,
                "flash_sdp_enabled": True, "efficient_sdp_enabled": True, "math_sdp_enabled": True}
    if any(state[name] != value for name, value in expected.items()):
        raise ValueError("Strict deterministic execution settings changed")
    return state


def apply_strict_determinism(torch_module=None):
    """Apply before model/CUDA initialization; unsupported operations must raise."""
    environment = validate_environment()
    torch = _torch() if torch_module is None else torch_module
    if str(torch.__version__).split("+")[0] != TORCH_VERSION:
        raise ValueError("Strict native SDPA execution requires the pinned PyTorch version")
    if torch.cuda.is_initialized():
        raise ValueError("Deterministic execution must be established before CUDA initialization")
    before = determinism_state(torch)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Preserve fused deterministic kernels where supported. In the pinned
    # native PyTorch2.9.1 implementation strict mode selects deterministic
    # Flash backward / efficient split count and rejects cuDNN SDPA selection.
    # Do not force all attention to the slower math implementation.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    after = validate_strict_determinism(torch)
    return {"profile_id": PROFILE_ID, "torch_version": str(torch.__version__),
            "environment": environment, "applied_before_cuda_initialization": True,
            "manual_strict_algorithms": True, "hf_full_determinism": False,
            "initial_flags": before, "training_flags": after,
            "cudnn_sdp_flag_unchanged": before["cudnn_sdp_enabled"] == after["cudnn_sdp_enabled"],
            "sdpa_dispatch": "pinned_torch_strict_deterministic_dispatch",
            "unsupported_nondeterministic_operations": "raise_without_warn_only_fallback",
            "seed_or_rng_state_reset_by_this_profile": False,
            "exact_continuation_proved": False}


@dataclass(frozen=True)
class DeterministicCORuntime:
    """Delegate every scientific method hook, adding execution-state admission."""

    base: Any
    method: str
    initialization_receipt: Mapping[str, Any]

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError("Unsupported deterministic combinatorial method")
        expected_weight = .05 if self.method == "foarl_teacher" else 0.
        if (self.base.teacher_weight != expected_weight
                or self.base.metadata().get("method_id") != self.method):
            raise ValueError("Selected method and underlying scientific runtime differ")
        receipt = self.initialization_receipt
        if (receipt.get("profile_id") != PROFILE_ID
                or receipt.get("applied_before_cuda_initialization") is not True
                or receipt.get("hf_full_determinism") is not False):
            raise ValueError("Missing strict deterministic initialization receipt")
        validate_strict_determinism()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def validate_args(self, args):
        validate_strict_determinism()
        if bool(getattr(args, "full_determinism", False)):
            raise ValueError("Use manual strict determinism, not the HF debugging profile")
        self.base.validate_args(args)

    def validate_grpo_config(self, config):
        validate_strict_determinism()
        if bool(getattr(config, "full_determinism", False)):
            raise ValueError("GRPOConfig must retain full_determinism=False")
        self.base.validate_grpo_config(config)

    def metadata(self):
        state = validate_strict_determinism()
        receipt = copy.deepcopy(self.base.metadata())
        receipt["base_profile_id"] = receipt["profile_id"]
        receipt["profile_id"] = PROFILE_ID
        receipt["execution_successor"] = "new_strict_deterministic_source_not_historical_parity"
        receipt["determinism"] = {**copy.deepcopy(dict(self.initialization_receipt)), "training_flags": state}
        return receipt


def wrap_runtime(base, method, initialization_receipt):
    return DeterministicCORuntime(base=base, method=method,
                                  initialization_receipt=copy.deepcopy(dict(initialization_receipt)))
