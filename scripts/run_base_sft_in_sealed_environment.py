#!/usr/bin/env python3
"""Revalidate the sealed base-SFT environment inside a Slurm step, then exec.

This wrapper deliberately runs in the same Slurm step and Python interpreter as
``torch.distributed.run``.  The verified Python prefix and its parent are
non-writable, so successful verification closes the package/GPU check-to-use
gap before the training subprocesses are spawned.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from base_sft_resume_fidelity import (  # type: ignore[import-not-found]
    RECEIPT_FILENAME,
    resume_state_identity,
)
from freeze_base_sft_campaign import verify_campaign_runtime  # type: ignore[import-not-found]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--campaign-contract", type=Path, required=True)
    parser.add_argument("--immutable-input-trust-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--forbidden-source-registry", type=Path, required=True)
    parser.add_argument("--reserved-source-registry", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-receipt", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command.pop(0)
    if not command:
        raise ValueError("sealed environment wrapper requires a command after --")
    verify_campaign_runtime(
        campaign_path=args.campaign_contract,
        immutable_input_trust_root=args.immutable_input_trust_root,
        source_root=args.source_root,
        base_model_path=args.base_model_path,
        forbidden_source_registry=args.forbidden_source_registry,
        reserved_source_registry=args.reserved_source_registry,
        remote_environment_receipt=args.receipt,
        sft_data_root=args.data_root,
        output_root=args.output_root,
        tasks=[args.task],
        require_output_absent=False,
        require_accelerator=True,
    )
    if (args.resume_checkpoint is None) != (args.resume_receipt is None):
        raise ValueError("resume checkpoint and receipt must be supplied together")
    if args.resume_checkpoint is not None:
        expected_receipt = (
            args.resume_checkpoint.expanduser().resolve(strict=True) / RECEIPT_FILENAME
        )
        supplied_receipt = args.resume_receipt.expanduser().resolve(strict=True)
        if supplied_receipt != expected_receipt:
            raise ValueError(
                f"resume receipt path mismatch: supplied={supplied_receipt}, "
                f"expected={expected_receipt}"
            )
        resume_state_identity(
            campaign_path=args.campaign_contract,
            run_contract_path=args.run_contract,
            checkpoint=args.resume_checkpoint,
            require_accelerator=True,
        )
    executable = str(Path(sys.executable).resolve(strict=True))
    os.execv(executable, [executable, *command])
    return 127  # pragma: no cover - os.execv either replaces the process or raises


if __name__ == "__main__":
    raise SystemExit(main())
