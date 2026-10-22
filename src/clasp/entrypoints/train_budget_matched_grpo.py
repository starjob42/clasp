"""Prospective full-budget classic GRPO, FOARL, legacy or calibrated RLOO."""
from __future__ import annotations

from pathlib import Path

from clasp.baselines.budget_matched_dev import (
    COMMON_DEFAULTS,
    METHODS,
    PROFILE_ID,
    build_runtime,
    read_contract,
    require,
    seal_reward_environment,
    verify_source_manifest,
)
from clasp.config import build_grpo_parser, finalize_grpo_args
from clasp.entrypoints.train_foarl_f638_corrected import _elastic_record, verify_foarl_f638_checkpoint_task_binding
from clasp.entrypoints.train_foarl_f638_dev_20260907 import seal_dev_loader_environment


def build_parser():
    parser = build_grpo_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == 'model_path':
            action.required, action.default = False, None
        elif action.dest == 'tasks':
            action.required, action.default = True, None
    parser.add_argument('--sft_checkpoint_path', required=True)
    parser.add_argument('--budget_contract', required=True)
    parser.add_argument('--budget_contract_sha256', required=True)
    parser.add_argument('--budget_method', choices=METHODS, required=True)
    parser.add_argument('--budget_diagnostic', action='store_true')
    parser.set_defaults(**COMMON_DEFAULTS, wandb_project='grpo-budget-matched-dev-20260918')
    return parser


def resolve_declared_defaults(raw, contract):
    """Only fill omitted values; explicit contradictory CLI settings fail later."""
    values = {'max_steps': contract['horizon'], 'save_steps': 5 if contract.get('diagnostic', False) else 50,
              'grpo_beta': 0.05 if raw.budget_method == 'foarl' else 0.0,
              'grpo_loss_type': 'dr_grpo' if raw.budget_method in ('rloo_legacy', 'rloo_quality') else 'grpo',
              'verify_gt_rewards': raw.budget_method != 'foarl'}
    for name, value in values.items():
        if getattr(raw, name, None) is None:
            setattr(raw, name, value)
    return raw


def parse_args(argv=None):
    raw = build_parser().parse_args(argv)
    contract, contract_binding = read_contract(raw.budget_contract, raw.budget_contract_sha256)
    source = contract['source']
    source_binding = verify_source_manifest(Path(source['root']) / 'SOURCE_MANIFEST.sha256',
                                            source['manifest_sha256'])
    seal_reward_environment(raw.budget_method)
    loader_environment = seal_dev_loader_environment()
    args = finalize_grpo_args(resolve_declared_defaults(raw, contract))
    runtime = build_runtime(source_binding, contract, contract_binding)
    runtime.validate_args(args)
    from clasp.entrypoints.train_grpo_standard import _resolve_sft_init_model_path
    from clasp.models.loading import JIANG_LORA_TARGET_MODULES, lora_target_modules_from_env, peft_checkpoint_bias_policy
    from clasp.models.training_fidelity import verify_publication_input_adapter
    args.model_path = _resolve_sft_init_model_path(args.sft_checkpoint_path)
    args.budget_task_identity_binding = verify_foarl_f638_checkpoint_task_binding(args.model_path, runtime.task)
    require(peft_checkpoint_bias_policy(args.model_path) == 'all', 'Budget study requires corrected bias=all SFT')
    require(lora_target_modules_from_env() == list(JIANG_LORA_TARGET_MODULES), 'Budget study requires all seven Jiang LoRA targets')
    verify_publication_input_adapter(args.model_path, purpose=PROFILE_ID + ' initialization adapter')
    args.budget_loader_environment = loader_environment
    args.budget_runtime_receipt = runtime.metadata()
    return args, runtime


@_elastic_record
def main():
    args, runtime = parse_args()
    from clasp.entrypoints.train_grpo import run_grpo
    run_grpo(args, runtime_profile=runtime)


if __name__ == '__main__':
    main()
