"""Run an existing CO development method with strict deterministic execution.

Supply --co_method and all arguments of its existing method entry point.
The launcher must export CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python startup.
"""
from __future__ import annotations

import argparse
import importlib
import sys

from clasp.baselines.co_deterministic_dev import (
    METHODS,
    apply_strict_determinism,
    validate_strict_determinism,
    wrap_runtime,
)


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--co_method", choices=METHODS, required=True)
    selected, remaining = parser.parse_known_args(argv)
    # This must precede the existing entry imports: those may load model and
    # adapter utilities while finalizing their unchanged argument contracts.
    initialization = apply_strict_determinism()
    if selected.co_method == "classic_grpo":
        from clasp.entrypoints.train_classic_grpo_400_dev import parse_args as parse_base
    else:
        from clasp.entrypoints.train_foarl_teacher_dev import parse_args as parse_base
    args, base = parse_base(remaining)
    runtime = wrap_runtime(base, selected.co_method, initialization)
    runtime.validate_args(args)
    args.co_method = selected.co_method
    args.co_deterministic_runtime_receipt = runtime.metadata()
    # Avoid retaining an apparently nondeterministic top-level profile receipt
    # beside the new runtime. Its original profile ID remains explicit inside.
    name = "classic_runtime_receipt" if selected.co_method == "classic_grpo" else "foarl_teacher_runtime_receipt"
    setattr(args, name, runtime.metadata())
    return args, runtime


def main(argv=None):
    args, runtime = parse_args(argv)
    training = importlib.import_module("clasp.entrypoints.train_grpo")
    from clasp.models.classic_grpo_save_lineage import install_classic_save_lineage
    from clasp.models.co_determinism_receipt import write_postrun_receipt
    with install_classic_save_lineage(training, runtime.method):
        result = training.run_grpo(args, runtime_profile=runtime)
    validate_strict_determinism()
    write_postrun_receipt(args, runtime)
    return result


if __name__ == "__main__":
    main()
