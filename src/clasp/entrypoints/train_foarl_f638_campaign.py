"""Executable entrypoint for the two approved corrected-FOARL campaign lanes."""

from __future__ import annotations

import argparse
from typing import Sequence

from clasp.baselines.foarl_f638_campaign import (
    FOARL_F638_CAMPAIGN_METHODS,
    FOARL_F638_MATCHED_B9,
    FOARL_F638_NATIVE,
    get_foarl_f638_campaign_profile,
)
from clasp.config import finalize_grpo_args
from clasp.entrypoints.train_foarl_f638_corrected import (
    build_foarl_f638_corrected_parser,
    finalize_foarl_f638_args_namespace,
)
from clasp.runtime import print_primary
from clasp.training_source_identity import (
    FOARL_F638_CORRECTED_TRAINER_VERSION,
    require_train_grpo_source,
)

try:
    from torch.distributed.elastic.multiprocessing.errors import record as _elastic_record
except ImportError:

    def _elastic_record(fn):
        return fn


def build_foarl_f638_campaign_parser(method: str) -> argparse.ArgumentParser:
    profile = get_foarl_f638_campaign_profile(method)
    parser = build_foarl_f638_corrected_parser()
    parser.description = (
        f"Run {profile.method_id}; {profile.attribution}. "
        "This lane is never an official Jiang et al. reproduction."
    )
    parser.add_argument(
        "--foarl_campaign_lane",
        choices=FOARL_F638_CAMPAIGN_METHODS,
        required=True,
    )
    defaults = {
        "num_generations": profile.semantics.num_generations,
        "generation_batch_size": profile.generation_batch_size,
        "per_device_train_batch_size": profile.per_device_train_batch_size,
        "per_device_eval_batch_size": profile.per_device_eval_batch_size,
        "gradient_accumulation_steps": profile.gradient_accumulation_steps,
        "optim": profile.optimizer,
        "weight_decay": profile.weight_decay,
        "warmup_ratio": profile.warmup_ratio,
        "lr_scheduler_type": profile.lr_scheduler_type,
        "eval_from_train_split": profile.eval_from_train_split,
        "max_eval_samples_per_task": profile.max_eval_samples_per_task,
        "min_eval_samples_per_task": profile.min_eval_samples_per_task,
        "wandb_project": "foarl_f638_publication",
    }
    if method == FOARL_F638_NATIVE:
        defaults.update(
            max_samples_per_task=0,
            min_train_samples_per_task=1,
            save_steps=50,
            eval_steps=50,
        )
    elif method == FOARL_F638_MATCHED_B9:
        defaults.update(
            max_samples_per_task=0,
            min_train_samples_per_task=1_900,
            save_steps=400,
            eval_steps=100,
        )
    parser.set_defaults(**defaults)
    return parser


def _selected_method(argv: Sequence[str] | None) -> str:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument(
        "--foarl_campaign_lane",
        choices=FOARL_F638_CAMPAIGN_METHODS,
        required=True,
    )
    selected, _ = selector.parse_known_args(argv)
    return str(selected.foarl_campaign_lane)


def parse_foarl_f638_campaign_args(argv: list[str] | None = None):
    method = _selected_method(argv)
    profile = get_foarl_f638_campaign_profile(method)
    parser = build_foarl_f638_campaign_parser(method)
    args = finalize_grpo_args(parser.parse_args(argv))
    if args.foarl_campaign_lane != method:
        raise ValueError("FOARL campaign lane changed while resolving arguments")
    if int(args.max_steps) != profile.final_step(str(args.tasks[0])):
        raise ValueError(
            f"{method}/{args.tasks[0]} requires max_steps="
            f"{profile.final_step(str(args.tasks[0]))}, got {args.max_steps}"
        )
    if profile.required_train_seed is not None and int(args.seed) != profile.required_train_seed:
        raise ValueError(
            f"{method} requires released seed={profile.required_train_seed}, got {args.seed}"
        )
    return finalize_foarl_f638_args_namespace(
        args,
        runtime_profile=profile.runtime,
        purpose=method,
    )


@_elastic_record
def main() -> None:
    args = parse_foarl_f638_campaign_args()
    profile = get_foarl_f638_campaign_profile(args.foarl_campaign_lane)
    trainer_source = require_train_grpo_source(FOARL_F638_CORRECTED_TRAINER_VERSION)
    from clasp.entrypoints.train_grpo import run_grpo

    print_primary(
        "[FOARL campaign] "
        f"lane={profile.method_id}, classification={profile.attribution}, "
        "reproduction_claim=not_an_official_reproduction, "
        f"profile={profile.semantics.profile_id}, "
        f"trainer_source={trainer_source['version']}@{trainer_source['sha256']}"
    )
    run_grpo(args, runtime_profile=profile.runtime)


if __name__ == "__main__":
    main()
