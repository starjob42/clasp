"""Entry point: standard (non-constrained) TRL GRPO (``clasp-grpo-standard``).

The unconstrained baseline. It reuses the GRPO parser with ``use_primal_dual`` forced off,
and refuses to run if the constraint flags are passed. ``--sft_checkpoint_path`` accepts
either an SFT output directory or a ``checkpoint-*`` directory and resolves it to a usable
initialization path.
"""
from __future__ import annotations

import argparse
import os

from clasp.config import build_grpo_parser, finalize_grpo_args
from clasp.models.training_fidelity import verify_publication_input_adapter
from clasp.runtime import (
    find_latest_checkpoint_dir,
    list_resumable_checkpoint_dirs,
    output_dir_has_saved_model,
    print_primary,
)

try:
    from torch.distributed.elastic.multiprocessing.errors import record as _elastic_record
except ImportError:
    def _elastic_record(fn):
        return fn


def _resolve_existing_directory(path: str) -> str:
    resolved_path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isdir(resolved_path):
        raise ValueError(
            "SFT checkpoint path must be an existing directory. "
            f"Received: {resolved_path}"
        )
    return resolved_path


def _resolve_sft_init_model_path(raw_path: str) -> str:
    candidate_root = _resolve_existing_directory(raw_path)
    normalized_leaf = os.path.basename(os.path.normpath(candidate_root))

    if output_dir_has_saved_model(candidate_root):
        return candidate_root

    if normalized_leaf.startswith("checkpoint-"):
        return candidate_root

    resumable_checkpoint_dirs = list_resumable_checkpoint_dirs(candidate_root, require_trainer_state=False)
    if resumable_checkpoint_dirs:
        return resumable_checkpoint_dirs[0]

    latest_checkpoint_dir = find_latest_checkpoint_dir(candidate_root)
    if latest_checkpoint_dir is not None:
        return latest_checkpoint_dir

    raise ValueError(
        "Could not resolve a usable SFT initialization path. "
        "Provide either an SFT output directory with saved model artifacts or a "
        "directory that contains checkpoint-* folders. "
        f"Received: {candidate_root}"
    )


def build_normal_grpo_parser() -> argparse.ArgumentParser:
    parser = build_grpo_parser()
    parser.description = "Run standard (non-primal-dual) TRL GRPO for mixed CO tasks."
    parser.add_argument(
        "--sft_checkpoint_path",
        type=str,
        default=None,
        help=(
            "Optional SFT initialization directory for GRPO. "
            "Can be an SFT output directory (with saved model/checkpoints) or a direct "
            "checkpoint-* directory."
        ),
    )
    parser.set_defaults(use_primal_dual=False)
    return parser


def parse_normal_grpo_args():
    args = finalize_grpo_args(build_normal_grpo_parser().parse_args())
    if bool(args.use_primal_dual):
        raise ValueError(
            "clasp.entrypoints.train_grpo_standard only supports standard GRPO. "
            "Remove --use_primal_dual or pass --no-use_primal_dual."
        )

    raw_sft_checkpoint_path = str(args.sft_checkpoint_path or "").strip()
    if raw_sft_checkpoint_path:
        args.model_path = _resolve_sft_init_model_path(raw_sft_checkpoint_path)
        print_primary(f"[NormalGRPO] Initializing GRPO from SFT path: {args.model_path}")
    else:
        print_primary(f"[NormalGRPO] Initializing GRPO from --model_path: {args.model_path}")

    verify_publication_input_adapter(
        args.model_path,
        purpose="standard-GRPO initialization adapter",
    )

    return args


@_elastic_record
def main() -> None:
    args = parse_normal_grpo_args()
    from clasp.entrypoints.train_grpo import run_grpo

    print_primary(
        "[Runtime] clasp.entrypoints.train_grpo_standard active: "
        f"output_root={args.output_root}, report_to={args.report_to}"
    )
    run_grpo(args)


if __name__ == "__main__":
    main()
