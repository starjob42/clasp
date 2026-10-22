"""Value-free semantic equality of admitted deterministic step-5/10 states.

Step five establishes independent-run equality before restart. Step ten then
checks continuation endpoints. This script neither runs training nor submits a
job; scheduler/launch admission belongs to the diagnostic campaign controller.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path

from compare_foarl_teacher_resume import Comparison, _check_rng

from clasp.models.classic_grpo_resume import validate_classic_grpo_resume
from clasp.models.foarl_teacher_resume import (
    _absolute,
    _canonical,
    _identity,
    _load_json,
    _require,
    validate_foarl_teacher_resume,
)

PROTOCOL = "co-deterministic-semantic-state-comparison-v1"
METHODS = ("classic_grpo", "foarl_control", "foarl_teacher")
SUCCESS = {5: "passed_exact_step5_prerestart", 10: "passed_exact_step10_continuation"}


def _checkpoint(path, expected_step):
    _require(type(expected_step) is int and expected_step in SUCCESS, "Only diagnostic steps five and ten are supported")
    root = _absolute(path, "Checkpoint/run")
    if root.name != f"checkpoint-{expected_step}":
        _require(not root.name.startswith("checkpoint-"), "Checkpoint boundary differs from requested step")
        root = root / "checkpoints" / f"checkpoint-{expected_step}"
    _require(root.parent.name == "checkpoints", "Wrong checkpoint directory")
    contract_path = root.parent.parent / "local_development_run_contract.json"
    _identity(contract_path)
    contract = _load_json(contract_path)
    _require(contract.get("diagnostic") is True and contract.get("method") in METHODS,
             "Only supported declared diagnostic methods may be compared")
    return root, contract_path, contract


def _check_optimizer(optimizer, scheduler, expected_step):
    """The old step-10 validator's checks, directly parameterized by boundary.

    Never rewrite or normalize optimizer/scheduler values before comparison.
    """
    import torch

    _require(isinstance(optimizer, dict) and isinstance(optimizer.get("state"), dict)
             and optimizer["state"] and isinstance(optimizer.get("param_groups"), list)
             and optimizer["param_groups"], "Optimizer state is missing moments/parameter groups")
    parameters = []
    for group in optimizer["param_groups"]:
        _require(isinstance(group, dict) and isinstance(group.get("params"), list),
                 "Optimizer parameter group is malformed")
        parameters.extend(group["params"])
    _require(all(type(parameter) is int for parameter in parameters)
             and len(parameters) == len(set(parameters)) and set(parameters) == set(optimizer["state"]),
             "Optimizer moments do not cover every saved parameter exactly once")
    for state in optimizer["state"].values():
        _require(isinstance(state, dict) and "step" in state and len(state) > 1,
                 "Optimizer parameter state lacks step/moments")
        step = state["step"]
        if isinstance(step, torch.Tensor):
            _require(step.numel() == 1, "Optimizer step must be scalar")
            step = step.item()
        _require(type(step) in (int, float) and step == expected_step,
                 "Optimizer moments are not at the requested diagnostic boundary")
    _require(isinstance(scheduler, dict) and type(scheduler.get("last_epoch")) is int
             and scheduler["last_epoch"] == expected_step and type(scheduler.get("_step_count")) is int
             and scheduler["_step_count"] == expected_step + 1, "Scheduler is not at the requested diagnostic boundary")
    _require(scheduler.get("_last_lr") == [group["lr"] for group in optimizer["param_groups"]],
             "Scheduler and optimizer learning rates disagree")


def _runtime_inputs(root, contract, expected_step):
    # Late import keeps admission utilities free of training/runtime imports.
    from run_co_deterministic_dev_chunk import (
        read_only_runtime_declaration,
        read_only_runtime_evidence,
    )

    run = root.parent.parent
    declaration = read_only_runtime_declaration(run, contract)
    evidence = read_only_runtime_evidence(root, contract)
    return {"declaration": declaration,
            "declaration_file": _identity(run / "deterministic_runtime_declaration.json"),
            "applied_runtime": evidence,
            "applied_runtime_file": _identity(root / "co_deterministic_runtime_evidence.json")}


def _admit(root, contract_path, contract, expected_step):
    validator = validate_classic_grpo_resume if contract["method"] == "classic_grpo" else validate_foarl_teacher_resume
    checkpoint = validator(root, expected_step=expected_step, expected_contract=contract,
                           run_contract_path=contract_path)
    return {"checkpoint": checkpoint, "deterministic_runtime": _runtime_inputs(root, contract, expected_step)}


def compare_checkpoints(uninterrupted, resumed, *, expected_step):
    first, first_contract_path, contract = _checkpoint(uninterrupted, expected_step)
    second, second_contract_path, other_contract = _checkpoint(resumed, expected_step)
    _require(first.resolve() != second.resolve(), "Two distinct diagnostic runs are required")
    _require(_canonical(contract) == _canonical(other_contract), "Diagnostic original contracts differ")
    pairs = ((first, first_contract_path), (second, second_contract_path))
    admitted = [_admit(path, contract_path, contract, expected_step) for path, contract_path in pairs]
    declarations = [value["deterministic_runtime"]["declaration"] for value in admitted]
    semantic = [{key: value for key, value in declaration.items() if key not in {"receipt_sha256", "run_contract"}}
                for declaration in declarations]
    _require(_canonical(semantic[0]) == _canonical(semantic[1]), "Deterministic execution declarations differ")
    # BOTH method-specific state inventories and applied runtime evidence must
    # pass before any pickle or safetensors is deserialized.
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
        _check_optimizer(optimizer, scheduler, expected_step)
        optimizers.append(optimizer); schedulers.append(scheduler)
    comparison.compare(optimizers[0], optimizers[1], "optimizer")
    comparison.compare(schedulers[0], schedulers[1], "scheduler")
    del optimizers, schedulers
    for rank in range(4):
        name = f"rng_state_{rank}.pth"
        a = torch.load(first / name, map_location="cpu", weights_only=False)
        b = torch.load(second / name, map_location="cpu", weights_only=False)
        for state in (a, b):
            _check_rng(state)
            _require(all(value.ndim == 1 and value.numel() > 0 for value in [state["cpu"], *state["cuda"]]),
                     "RNG byte vectors must be nonempty and one-dimensional")
        comparison.compare(a, b, f"rng_rank_{rank}")
    # Re-run full method/runtime admission, including retained and current byte
    # identities, rather than trusting only the pre-deserialization inventory.
    after = [_admit(path, contract_path, contract, expected_step) for path, contract_path in pairs]
    _require(after == admitted, "Diagnostic inputs changed during state comparison")
    result = {
        "protocol": PROTOCOL,
        "status": SUCCESS[expected_step] if not comparison.differences else "failed_state_difference",
        "scope": "independent_prefix_readiness_not_resume_evidence" if expected_step == 5
                 else "training_continuation_readiness_not_performance_evidence",
        "expected_step": expected_step, "method": contract["method"],
        "checkpoints": {"uninterrupted": admitted[0]["checkpoint"], "resumed": admitted[1]["checkpoint"]},
        "deterministic_runtime": {"uninterrupted": admitted[0]["deterministic_runtime"],
                                  "resumed": admitted[1]["deterministic_runtime"]},
        "runtime": {"torch": torch.__version__, "numpy": np.__version__},
        "compared_leaf_count": comparison.leaves,
        "differing_field_count": len(comparison.differences), "differing_fields": comparison.differences,
        "parameter_rng_values_exported": False, "raw_rollout_or_next_prompt_trace_exported": False,
        "live_loaded_optimizer_scheduler_rng_restoration_verified": False,
        "limitation": "Exact saved semantic states are compared; no live restoration or next-prompt trace is inferred.",
    }
    result["receipt_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--step", type=int, choices=tuple(SUCCESS), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    with output.open("x") as handle:
        try:
            # Nested diagnostics can contain state values; retain only this
            # comparator's explicit result, never exception text or streams.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = compare_checkpoints(args.uninterrupted, args.resumed, expected_step=args.step)
            code = 0 if result["status"] == SUCCESS[args.step] else 1
        except Exception as error:
            result = {"protocol": PROTOCOL, "status": "failed_admission_or_comparison",
                      "expected_step": args.step, "exception_class": type(error).__name__,
                      "parameter_rng_values_exported": False}
            result["receipt_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
            code = 1
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    output.chmod(0o440)
    print(json.dumps({"status": result["status"], "output": _identity(output)}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
