"""Executable 100-update matched FOARL development port; never a reproduction."""

from __future__ import annotations

import argparse
import os
from typing import Any

from clasp.baselines.foarl_f638_dev_20260907 import (
    DEV_DEFAULTS,
    METHOD_ID,
    build_foarl_f638_dev_runtime,
    diagnostic_updates,
    verify_dev_source_manifest,
)
from clasp.config import finalize_grpo_args
from clasp.entrypoints.train_foarl_f638_corrected import (
    _REPRESENTATION_ENV_DEFAULTS,
    _elastic_record,
    build_foarl_f638_corrected_parser,
    verify_foarl_f638_checkpoint_task_binding,
)
from clasp.runtime import print_primary


def build_parser() -> argparse.ArgumentParser:
    parser = build_foarl_f638_corrected_parser()
    parser.description = "FOARL-f638 matched B9 development port, not an official reproduction."
    parser.add_argument("--foarl_dev_source_manifest", required=True)
    parser.add_argument("--foarl_dev_source_manifest_sha256", required=True)
    parser.set_defaults(**DEV_DEFAULTS, wandb_project="grpo-incumbent-dev-20260907")
    diagnostic = diagnostic_updates()
    if diagnostic is not None:
        parser.set_defaults(max_steps=diagnostic, save_steps=diagnostic)
    return parser


def parse_args(argv: list[str] | None = None) -> tuple[Any, Any]:
    args = finalize_grpo_args(build_parser().parse_args(argv))
    binding = verify_dev_source_manifest(
        args.foarl_dev_source_manifest,
        args.foarl_dev_source_manifest_sha256,
    )
    runtime = build_foarl_f638_dev_runtime(binding)
    args = finalize_dev_args(args, runtime)
    args.foarl_dev_runtime_receipt = runtime.metadata()
    return args, runtime


def seal_dev_loader_environment() -> dict[str, str]:
    """Keep the common output representation in all three development arms."""

    expected = {**_REPRESENTATION_ENV_DEFAULTS, "CLASP_MIS_JIANG_RESPONSE_OUTPUT": "0"}
    drift = {
        name: {"expected": value, "observed": os.environ[name]}
        for name, value in expected.items()
        if name in os.environ and os.environ[name] != value
    }
    if drift:
        raise ValueError(f"FOARL dev common representation environment differs: {drift}")
    os.environ.update(expected)
    return expected


def finalize_dev_args(args: Any, runtime: Any) -> Any:
    """Retain corrected-SFT gates with this profile's explicit common prompt."""

    runtime.validate_args(args)
    from clasp.entrypoints.train_grpo_standard import _resolve_sft_init_model_path
    from clasp.models.loading import (
        JIANG_LORA_TARGET_MODULES,
        lora_target_modules_from_env,
        peft_checkpoint_bias_policy,
    )
    from clasp.models.training_fidelity import verify_publication_input_adapter

    args.model_path = _resolve_sft_init_model_path(args.sft_checkpoint_path)
    args.foarl_task_identity_binding = verify_foarl_f638_checkpoint_task_binding(
        args.model_path, str(args.tasks[0]).lower()
    )
    args.foarl_loader_environment = seal_dev_loader_environment()
    if peft_checkpoint_bias_policy(args.model_path) != "all":
        raise ValueError("FOARL dev requires the common corrected bias=all SFT adapter")
    if lora_target_modules_from_env() != list(JIANG_LORA_TARGET_MODULES):
        raise ValueError("FOARL dev requires the exact seven Jiang LoRA target modules")
    verify_publication_input_adapter(args.model_path, purpose=f"{METHOD_ID} initialization adapter")
    args.foarl_prompt_template_validation = {
        "status": "PASS",
        "sha256": runtime.semantics.prompt_sha256,
        "no_reason": True,
        "scope": "common_development_prompt_not_released_RL_prompt",
    }
    return args


@_elastic_record
def main() -> None:
    args, runtime = parse_args()
    # The complete frozen tree was verified before importing the GPU runtime.
    from clasp.entrypoints.train_grpo import run_grpo

    print_primary(
        f"[FOARL dev] method={METHOD_ID}, scope={runtime.metadata()['study_scope']}, "
        f"source_manifest={args.foarl_dev_source_manifest_sha256}, "
        "reproduction_claim=not_an_official_reproduction"
    )
    run_grpo(args, runtime_profile=runtime)


if __name__ == "__main__":
    main()
