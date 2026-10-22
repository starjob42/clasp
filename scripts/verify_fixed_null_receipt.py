#!/usr/bin/env python3
"""Independently replay every row in a fixed-null score receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clasp.set_grpo.fixed_null_replay import verify_fixed_null_receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--task")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        identity = verify_fixed_null_receipt(
            receipt_path=args.receipt,
            training_rows_path=args.training_rows,
            expected_task=args.task,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[FixedNullReplayError] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(identity, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
