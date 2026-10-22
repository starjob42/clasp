"""Compare validated own step-10 diagnostic checkpoints without exporting values.

The native RNG pickle contains Python/NumPy state, so it is loaded only after
same-contract, checkpoint-fidelity and complete-file admission of both own runs.
No model, corpus or completion is loaded. Tensor comparisons are byte-exact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from clasp.models.foarl_teacher_resume import (
    _canonical,
    _identity,
    _load_json,
    _require,
    validate_foarl_teacher_resume,
)

PROTOCOL = "foarl-teacher-exact-resume-comparison-v1"


class Comparison:
    """Keep field names/counts only; never retain differing state values."""

    def __init__(self):
        self.leaves = 0
        self.differences = []

    def difference(self, path):
        self.differences.append(path)

    def compare(self, left, right, path):
        import numpy as np
        import torch

        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            self.leaves += 1
            if not (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)):
                self.difference(path + ".type")
            elif left.dtype != right.dtype or left.shape != right.shape or left.layout != right.layout:
                self.difference(path + ".tensor_metadata")
            else:
                _require(left.layout == torch.strided, "Only dense diagnostic state tensors are supported")
                a = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
                b = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
                if not torch.equal(a, b):
                    self.difference(path + ".tensor_bytes")
            return
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            self.leaves += 1
            if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
                self.difference(path + ".type")
            elif left.dtype != right.dtype or left.shape != right.shape:
                self.difference(path + ".array_metadata")
            elif left.tobytes(order="C") != right.tobytes(order="C"):
                self.difference(path + ".array_bytes")
            return
        if type(left) is not type(right):
            self.leaves += 1
            self.difference(path + ".type")
        elif isinstance(left, Mapping):
            _require(all(type(key) in (str, int) for key in left)
                     and all(type(key) in (str, int) for key in right),
                     "Unsupported state dictionary key type")
            if set(left) != set(right):
                self.difference(path + ".keys")
            for key in sorted(set(left) & set(right), key=lambda value: (type(value).__name__, str(value))):
                self.compare(left[key], right[key], f"{path}[{key}]")
        elif isinstance(left, (tuple, list)):
            if len(left) != len(right):
                self.difference(path + ".length")
            for index, (a, b) in enumerate(zip(left, right)):
                self.compare(a, b, f"{path}[{index}]")
        else:
            self.leaves += 1
            _require(type(left) in (str, int, float, bool, bytes, type(None)),
                     "Unsupported state value type")
            if isinstance(left, float):
                import struct

                equal = struct.pack("!d", left) == struct.pack("!d", right)
            else:
                equal = left == right
            if not equal:
                self.difference(path)


def _checkpoint(path):
    root = Path(path).absolute()
    if root.name != "checkpoint-10":
        root = root / "checkpoints/checkpoint-10"
    _require(root.name == "checkpoint-10" and root.parent.name == "checkpoints",
             "Only step-10 diagnostic checkpoints are supported")
    contract_path = root.parent.parent / "local_development_run_contract.json"
    contract = _load_json(contract_path)
    _require(contract.get("diagnostic") is True, "Only declared diagnostic runs may be compared")
    return root, contract_path, contract


def _check_optimizer(optimizer, scheduler):
    import torch

    _require(isinstance(optimizer, dict) and isinstance(optimizer.get("state"), dict)
             and optimizer["state"] and isinstance(optimizer.get("param_groups"), list)
             and optimizer["param_groups"], "Optimizer state is missing moments/parameter groups")
    parameters = []
    for group in optimizer["param_groups"]:
        _require(isinstance(group, dict) and isinstance(group.get("params"), list),
                 "Optimizer parameter group is malformed")
        parameters.extend(group["params"])
    _require(len(parameters) == len(set(parameters)) and set(parameters) == set(optimizer["state"]),
             "Optimizer moments do not cover every saved parameter exactly once")
    for state in optimizer["state"].values():
        _require(isinstance(state, dict) and "step" in state and len(state) > 1,
                 "Optimizer parameter state lacks step/moments")
        step = state["step"]
        if isinstance(step, torch.Tensor):
            _require(step.numel() == 1, "Optimizer step must be scalar")
            step = step.item()
        _require(type(step) in (int, float) and step == 10, "Optimizer moments are not at step 10")
    _require(isinstance(scheduler, dict) and scheduler.get("last_epoch") == 10
             and scheduler.get("_step_count") == 11, "Scheduler is not at diagnostic step 10")
    _require(scheduler.get("_last_lr") == [group["lr"] for group in optimizer["param_groups"]],
             "Scheduler and optimizer learning rates disagree")


def _check_rng(rng):
    import numpy as np
    import torch

    _require(isinstance(rng, dict) and set(rng) == {"python", "numpy", "cpu", "cuda"},
             "Each native RNG file must contain Python, NumPy, Torch CPU and CUDA states")
    _require(isinstance(rng["python"], tuple) and len(rng["python"]) == 3,
             "Python RNG state is malformed")
    _require(isinstance(rng["numpy"], tuple) and len(rng["numpy"]) == 5
             and isinstance(rng["numpy"][1], np.ndarray), "NumPy RNG state is malformed")
    _require(isinstance(rng["cpu"], torch.Tensor) and rng["cpu"].dtype == torch.uint8,
             "Torch CPU RNG state is malformed")
    _require(isinstance(rng["cuda"], (list, tuple)) and len(rng["cuda"]) == 4
             and all(isinstance(value, torch.Tensor) and value.dtype == torch.uint8
                     for value in rng["cuda"]), "Four visible CUDA RNG states are required on each rank")


def compare_checkpoints(uninterrupted, resumed):
    first, first_contract_path, contract = _checkpoint(uninterrupted)
    second, second_contract_path, second_contract = _checkpoint(resumed)
    _require(first.resolve() != second.resolve(), "Diagnostic comparison requires two distinct runs")
    _require(_canonical(contract) == _canonical(second_contract), "Diagnostic original contracts differ")
    admitted = [validate_foarl_teacher_resume(
        path, expected_step=10, expected_contract=contract, run_contract_path=contract_path)
        for path, contract_path in ((first, first_contract_path), (second, second_contract_path))]
    # No pickle/safetensors is loaded until BOTH independent own runs are admitted.
    import numpy as np
    import torch
    from safetensors.torch import load_file

    comparison = Comparison()
    comparison.compare(_load_json(first / "adapter_config.json"),
                       _load_json(second / "adapter_config.json"), "adapter_config")
    comparison.compare(load_file(str(first / "adapter_model.safetensors"), device="cpu"),
                       load_file(str(second / "adapter_model.safetensors"), device="cpu"), "adapter")
    optimizers, schedulers = [], []
    for path in (first, second):
        optimizer = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
        scheduler = torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True)
        _check_optimizer(optimizer, scheduler)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    comparison.compare(optimizers[0], optimizers[1], "optimizer")
    comparison.compare(schedulers[0], schedulers[1], "scheduler")
    del optimizers, schedulers
    for rank in range(4):
        name = f"rng_state_{rank}.pth"
        # Trusted, self-produced, admitted checkpoints only; NumPy/Python RNG
        # pickles need full loading in the exact native runtime.
        a = torch.load(first / name, map_location="cpu", weights_only=False)
        b = torch.load(second / name, map_location="cpu", weights_only=False)
        _check_rng(a)
        _check_rng(b)
        comparison.compare(a, b, f"rng_rank_{rank}")
    for path, receipt in zip((first, second), admitted):
        _require({name: _identity(path / name) for name in receipt["files"]} == receipt["files"],
                 "Diagnostic checkpoint changed during state comparison")
        _require(_identity(Path(receipt["run_contract"]["path"])) == receipt["run_contract"],
                 "Diagnostic contract changed during state comparison")
    result = {
        "protocol": PROTOCOL,
        "status": "passed_exact_step10_continuation" if not comparison.differences else "failed_state_difference",
        "scope": "training_continuation_readiness_not_performance_evidence",
        "checkpoints": {"uninterrupted": admitted[0], "resumed": admitted[1]},
        "runtime": {"torch": torch.__version__, "numpy": np.__version__},
        "compared_leaf_count": comparison.leaves,
        "differing_field_count": len(comparison.differences),
        "differing_fields": comparison.differences,
        "parameter_rng_values_exported": False,
        "independent_next_prompt_or_rollout_equality_proved": False,
        "limitation": "Exact endpoint state equality is checked; no raw rollout or next-prompt trace is exported.",
    }
    result["receipt_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    # Reserve the output before expensive validation; a failed attempt is retained.
    with output.open("x") as handle:
        try:
            result = compare_checkpoints(args.uninterrupted, args.resumed)
            code = 0 if result["status"] == "passed_exact_step10_continuation" else 1
        except Exception as exc:
            # Exception text can contain deserialized values. Emit only the class.
            result = {"protocol": PROTOCOL, "status": "failed_admission_or_comparison",
                      "exception_class": type(exc).__name__, "parameter_rng_values_exported": False}
            result["receipt_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
            code = 1
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    output.chmod(0o440)
    print(json.dumps({"status": result["status"], "output": _identity(output)}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
