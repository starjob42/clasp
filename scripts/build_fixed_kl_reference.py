#!/usr/bin/env python3
"""Build or live-verify a full step-zero policy for biased-LoRA KL."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from clasp.models.fixed_kl_reference import (
    SUPPORTED_DTYPES,
    FixedKLReferenceError,
    build_fixed_kl_reference,
    verify_fixed_kl_reference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--source-adapter-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--output-model-path", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument(
        "--claim-scope",
        choices=("development", "publication"),
        required=True,
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(sorted(SUPPORTED_DTYPES)),
        default="bfloat16",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="freshly reconstruct merge/state/tokenizer/logit evidence for an existing receipt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_only:
            payload = verify_fixed_kl_reference(
                args.output_model_path,
                args.receipt_path,
                claim_scope=args.claim_scope,
            )
        else:
            missing = [
                name
                for name, value in (
                    ("--base-model-path", args.base_model_path),
                    ("--source-adapter-path", args.source_adapter_path),
                    ("--tokenizer-path", args.tokenizer_path),
                )
                if value is None
            ]
            if missing:
                raise FixedKLReferenceError("building requires explicit " + ", ".join(missing))
            payload = build_fixed_kl_reference(
                base_model_path=args.base_model_path,
                source_adapter_path=args.source_adapter_path,
                tokenizer_path=args.tokenizer_path,
                output_model_path=args.output_model_path,
                receipt_path=args.receipt_path,
                claim_scope=args.claim_scope,
                dtype=args.dtype,
            )
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    except (FileExistsError, FixedKLReferenceError) as exc:
        print(f"[FixedKLReferenceError] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
