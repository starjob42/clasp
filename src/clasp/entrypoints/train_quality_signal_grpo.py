"""Isolated B9 reward-strength and invalid-diagnostic development ablation."""
from __future__ import annotations

from clasp.baselines.external_baseline_dev import seal_reward_environment, verify_source_manifest
from clasp.baselines.quality_signal_dev import PROFILE_ID, REWARD_MODES, build_runtime
from clasp.config import finalize_grpo_args
from clasp.entrypoints.train_external_baseline_grpo import build_parser as external_parser
from clasp.entrypoints.train_foarl_f638_corrected import (
    _elastic_record,
    verify_foarl_f638_checkpoint_task_binding,
)
from clasp.entrypoints.train_foarl_f638_dev_20260907 import seal_dev_loader_environment


def build_parser():
    parser = external_parser()
    parser.description = __doc__
    parser.add_argument("--quality_reward", choices=REWARD_MODES,
                        default="reference_sigmoid_v2")
    parser.set_defaults(num_generations=9, wandb_project="grpo-quality-signal-dev-20260916")
    return parser


def parse_args(argv=None):
    raw = build_parser().parse_args(argv)
    binding = verify_source_manifest(raw.external_dev_source_manifest,
                                     raw.external_dev_source_manifest_sha256)
    seal_reward_environment()
    loader_environment = seal_dev_loader_environment()
    args = finalize_grpo_args(raw)
    runtime = build_runtime(binding, args)
    runtime.validate_args(args)
    from clasp.entrypoints.train_grpo_standard import _resolve_sft_init_model_path
    from clasp.models.loading import (
        JIANG_LORA_TARGET_MODULES,
        lora_target_modules_from_env,
        peft_checkpoint_bias_policy,
    )
    from clasp.models.training_fidelity import verify_publication_input_adapter

    args.model_path = _resolve_sft_init_model_path(args.sft_checkpoint_path)
    args.external_baseline_task_identity_binding = verify_foarl_f638_checkpoint_task_binding(
        args.model_path, str(args.tasks[0]))
    if peft_checkpoint_bias_policy(args.model_path) != "all":
        raise ValueError("Quality-signal experiment requires corrected bias=all SFT")
    if lora_target_modules_from_env() != list(JIANG_LORA_TARGET_MODULES):
        raise ValueError("Quality-signal experiment requires all seven Jiang LoRA targets")
    verify_publication_input_adapter(args.model_path, purpose=f"{PROFILE_ID} initialization adapter")
    args.external_baseline_loader_environment = loader_environment
    args.external_baseline_runtime_receipt = runtime.metadata()
    return args, runtime


@_elastic_record
def main():
    args, runtime = parse_args()
    from clasp.entrypoints.train_grpo import run_grpo
    run_grpo(args, runtime_profile=runtime)


if __name__ == "__main__":
    main()
