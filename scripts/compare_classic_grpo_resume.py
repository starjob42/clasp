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
from pathlib import Path

from compare_foarl_teacher_resume import Comparison, _check_optimizer, _check_rng, _checkpoint

from clasp.models.classic_grpo_resume import validate_classic_grpo_resume
from clasp.models.foarl_teacher_resume import (
    _canonical,
    _identity,
    _load_json,
    _require,
)

PROTOCOL = "classic-grpo-exact-resume-comparison-v1"


def compare_checkpoints(uninterrupted, resumed):
    first, first_contract_path, contract = _checkpoint(uninterrupted)
    second, second_contract_path, second_contract = _checkpoint(resumed)
    _require(first.resolve() != second.resolve(), "Diagnostic comparison requires two distinct runs")
    _require(_canonical(contract) == _canonical(second_contract), "Diagnostic original contracts differ")
    admitted = [validate_classic_grpo_resume(
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
