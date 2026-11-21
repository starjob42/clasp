#!/usr/bin/env python3
"""Revalidate the sealed publication environment in the GRPO Slurm step, then exec.

The wrapper runs under the exact interpreter that will launch ``torchrun``.
Successful verification binds the complete read-only Python prefix and all four
A100 devices immediately before the distributed training subprocesses are
created.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from freeze_remote_environment import environment_identity  # type: ignore[import-not-found]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command.pop(0)
    if not command:
        raise ValueError("sealed GRPO wrapper requires a command after --")
    environment_identity(args.receipt, require_accelerator=True)
    executable = str(Path(sys.executable).resolve(strict=True))
    os.execv(executable, [executable, *command])
    return 127  # pragma: no cover - os.execv either replaces the process or raises


if __name__ == "__main__":
    raise SystemExit(main())
