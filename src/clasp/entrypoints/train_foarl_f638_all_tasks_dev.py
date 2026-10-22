"""All-task fresh-100-update FOARL development entrypoint; five-step diagnostics allowed."""
from __future__ import annotations

from clasp.baselines.foarl_f638_all_tasks_dev import (
    build_all_tasks_foarl_runtime, verify_all_tasks_source_manifest,
)
from clasp.config import finalize_grpo_args
from clasp.entrypoints.train_foarl_f638_corrected import _elastic_record
from clasp.entrypoints.train_foarl_f638_dev_20260907 import (
    build_parser as build_previous_parser, finalize_dev_args,
)


def build_parser():
    parser = build_previous_parser()
    parser.description = "TSP/CVRP/OP/MIS/JSSP FOARL-f638 common-budget development control"
    parser.set_defaults(save_total_limit=1, wandb_project="grpo-all-co-controls-dev-20260910")
    return parser


def parse_args(argv=None):
    args = finalize_grpo_args(build_parser().parse_args(argv))
    binding = verify_all_tasks_source_manifest(
        args.foarl_dev_source_manifest, args.foarl_dev_source_manifest_sha256,
    )
    runtime = build_all_tasks_foarl_runtime(binding)
    args = finalize_dev_args(args, runtime)
    args.foarl_dev_runtime_receipt = runtime.metadata()
    return args, runtime


@_elastic_record
def main():
    args, runtime = parse_args()
    from clasp.entrypoints.train_grpo import run_grpo
    run_grpo(args, runtime_profile=runtime)


if __name__ == "__main__":
    main()
