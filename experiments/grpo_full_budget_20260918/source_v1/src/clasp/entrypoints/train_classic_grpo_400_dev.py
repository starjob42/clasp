"""Fresh strict-objective classic GRPO control on the common 400-step horizon."""
from __future__ import annotations

from clasp.baselines.classic_grpo_400_dev import (
    COMMON_DEFAULTS,
    METHOD_ID,
    build_runtime,
    seal_reward_environment,
)
from clasp.baselines.foarl_f638_dev_20260907 import verify_dev_source_manifest
from clasp.config import build_grpo_parser, finalize_grpo_args
from clasp.entrypoints.train_foarl_f638_corrected import (
    _elastic_record,
    verify_foarl_f638_checkpoint_task_binding,
)
from clasp.entrypoints.train_foarl_f638_dev_20260907 import seal_dev_loader_environment


def build_parser():
    parser = build_grpo_parser()
    parser.description = "Classic strict-objective beta-zero400 development control"
    for action in parser._actions:
        if action.dest == "model_path":
            action.required = False
            action.default = None
        elif action.dest == "tasks":
            action.required = True
            action.default = None
    parser.add_argument("--sft_checkpoint_path", required=True)
    parser.add_argument("--classic_dev_source_manifest", required=True)
    parser.add_argument("--classic_dev_source_manifest_sha256", required=True)
    parser.add_argument("--classic_grpo_diagnostic", action="store_true")
    parser.set_defaults(**COMMON_DEFAULTS, dtype="bfloat16", prompt_style="jiang_f638",
                        use_primal_dual=False, use_clasp=False, full_finetune=False,
                        verify_gt_rewards=True, grpo_beta=0.0, grpo_epsilon=0.2,
                        grpo_epsilon_high=0.2, grpo_loss_type="grpo", num_train_epochs=1,
                        wandb_project="classic-grpo400-dev-20260909")
    return parser


def parse_args(argv=None):
    raw = build_parser().parse_args(argv)
    binding = verify_dev_source_manifest(raw.classic_dev_source_manifest,
                                         raw.classic_dev_source_manifest_sha256)
    seal_reward_environment()
    loader_environment = seal_dev_loader_environment()
    args = finalize_grpo_args(raw)
    runtime = build_runtime(binding, str(args.tasks[0]), args.classic_grpo_diagnostic)
    runtime.validate_args(args)
    from clasp.entrypoints.train_grpo_standard import _resolve_sft_init_model_path
    from clasp.models.loading import (
        JIANG_LORA_TARGET_MODULES,
        lora_target_modules_from_env,
        peft_checkpoint_bias_policy,
    )
    from clasp.models.training_fidelity import verify_publication_input_adapter

    args.model_path = _resolve_sft_init_model_path(args.sft_checkpoint_path)
    args.classic_task_identity_binding = verify_foarl_f638_checkpoint_task_binding(
        args.model_path, str(args.tasks[0]))
    if peft_checkpoint_bias_policy(args.model_path) != "all":
        raise ValueError("Classic control requires the common bias=all SFT adapter")
    if lora_target_modules_from_env() != list(JIANG_LORA_TARGET_MODULES):
        raise ValueError("Classic control requires the seven common Jiang LoRA targets")
    verify_publication_input_adapter(args.model_path, purpose=f"{METHOD_ID} initialization adapter")
    args.classic_loader_environment = loader_environment
    args.classic_runtime_receipt = runtime.metadata()
    return args, runtime


@_elastic_record
def main():
    args, runtime = parse_args()
    from clasp.entrypoints.train_grpo import run_grpo
    run_grpo(args, runtime_profile=runtime)


if __name__ == "__main__":
    main()
