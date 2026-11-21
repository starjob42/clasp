#!/usr/bin/env python3
"""Build one immutable fixed-null score receipt from explicit frozen inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clasp.set_grpo.fixed_null_construction import build_fixed_null_receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-spec", type=Path, required=True)
    parser.add_argument("--null-candidate", type=Path, required=True)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = build_fixed_null_receipt(
            construction_spec_path=args.construction_spec,
            null_candidate_path=args.null_candidate,
            training_rows_path=args.training_rows,
            output_path=args.output,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[FixedNullConstructionError] {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output.expanduser().resolve()),
                "receipt_sha256": payload["receipt_sha256"],
                "record_count": len(payload["records"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
