#!/usr/bin/env python3
"""Revalidate the frozen Jiang-f638 campaign in-process, then exec one chunk.

Fresh allocations use ``trainer.train()``. Continuations are admitted only
after the worker validates an exact, create-once single-H100 checkpoint receipt
and passes its canonical path explicitly to the dedicated trainer.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

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
    executable = str(Path(sys.executable).resolve(strict=True))
    os.execv(executable, [executable, *command])
    return 127  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
