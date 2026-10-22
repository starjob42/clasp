"""Opt-in graded-reward RLOO/external baseline for TSP, CVRP, OP, MIS or JSSP."""
from __future__ import annotations

from clasp.baselines.external_baseline_dev import (
    COMMON_DEFAULTS,
    PROFILE_ID,
    build_runtime,
    seal_reward_environment,
    verify_source_manifest,
)
from clasp.config import build_grpo_parser, finalize_grpo_args
from clasp.entrypoints.train_foarl_f638_corrected import (
    _elastic_record,
    verify_foarl_f638_checkpoint_task_binding,
)
from clasp.entrypoints.train_foarl_f638_dev_20260907 import seal_dev_loader_environment


def build_parser():
    parser = build_grpo_parser()
    parser.description = (
        "TSP/CVRP/OP/MIS/JSSP specialist: raw-policy graded reward with RLOO "
        "or a pre-batch external baseline"
    )
    for action in parser._actions:
        if action.dest == "model_path":
            action.required = False
            action.default = None
        elif action.dest == "tasks":
            action.required = True
            action.default = None
    parser.add_argument("--sft_checkpoint_path", required=True)
    parser.add_argument("--external_dev_source_manifest", required=True)
    parser.add_argument("--external_dev_source_manifest_sha256", required=True)
    parser.add_argument("--advantage_estimator", choices=("rloo", "external"), required=True)
    parser.add_argument("--external_baseline_initial", type=float, default=0.0)
    parser.add_argument("--external_baseline_scale", type=float, default=1.0)
    parser.add_argument("--external_baseline_diagnostic", action="store_true")
    parser.set_defaults(**COMMON_DEFAULTS, wandb_project="grpo-external-baseline-dev-20260910")
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
        raise ValueError("External-baseline experiment requires the common corrected bias=all SFT")
    if lora_target_modules_from_env() != list(JIANG_LORA_TARGET_MODULES):
        raise ValueError("External-baseline experiment requires the seven Jiang LoRA targets")
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
