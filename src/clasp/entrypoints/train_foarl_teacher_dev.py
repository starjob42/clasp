"""Train the FOARL control or its verified-teacher successor on a fixed horizon."""
from __future__ import annotations

from clasp.baselines.foarl_f638_dev_20260907 import verify_dev_source_manifest
from clasp.baselines.foarl_teacher_dev import COMMON_DEFAULTS, build_runtime
from clasp.config import finalize_grpo_args
from clasp.entrypoints.train_foarl_f638_corrected import (
    _elastic_record, build_foarl_f638_corrected_parser,
)
from clasp.entrypoints.train_foarl_f638_dev_20260907 import finalize_dev_args


def build_parser():
    parser = build_foarl_f638_corrected_parser()
    parser.add_argument("--foarl_dev_source_manifest", required=True)
    parser.add_argument("--foarl_dev_source_manifest_sha256", required=True)
    parser.add_argument("--foarl_teacher_weight", type=float, choices=(0.0, 0.05), required=True)
    parser.add_argument("--foarl_teacher_diagnostic", action="store_true")
    parser.set_defaults(**COMMON_DEFAULTS, wandb_project="foarl-teacher-dev-20260909")
    return parser


def parse_args(argv=None):
    args = finalize_grpo_args(build_parser().parse_args(argv))
    binding = verify_dev_source_manifest(
        args.foarl_dev_source_manifest, args.foarl_dev_source_manifest_sha256)
    runtime = build_runtime(binding, args.foarl_teacher_weight, args.foarl_teacher_diagnostic)
    args = finalize_dev_args(args, runtime)
    args.foarl_teacher_runtime_receipt = runtime.metadata()
    return args, runtime


@_elastic_record
def main():
    args, runtime = parse_args()
    from clasp.entrypoints.train_grpo import run_grpo
    run_grpo(args, runtime_profile=runtime)


if __name__ == "__main__":
    main()
