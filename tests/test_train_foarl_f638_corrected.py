from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from clasp.baselines.llmcosolver_f638_training import (
    FOARL_F638_CORRECTED_RUNTIME,
    FOARL_F638_REWARD_HEAD_ORDER,
    FOARL_F638_SEMANTICS,
    validate_foarl_f638_prompt_template,
    validate_trl_foarl_implementation,
)
from clasp.entrypoints.train_foarl_f638_corrected import (
    build_foarl_f638_corrected_parser,
    seal_foarl_f638_loader_environment,
    verify_foarl_f638_checkpoint_task_binding,
)
from clasp.training_source_identity import (
    FOARL_F638_CORRECTED_TRAINER_VERSION,
    FOARL_TEACHER_DEV_TRAINER_VERSION,
    TSP_FCDF_BETA_PILOT_TRAINER_SHA256,
    require_train_grpo_source,
)


ROOT = Path(__file__).resolve().parents[1]


def _canonical_args(**overrides):
    values = {
        "tasks": ["tsp"],
        "use_primal_dual": False,
        "use_clasp": False,
        "full_finetune": False,
        "verify_gt_rewards": False,
        "num_generations": 8,
        "prompt_style": "jiang_f638",
        "no_reason": False,
        "grpo_beta": 0.05,
        "grpo_epsilon": 0.2,
        "grpo_epsilon_high": 0.2,
        "grpo_loss_type": "grpo",
        "foarl_advantage_epsilon": 1e-4,
        "foarl_scale_rewards": "group",
        "foarl_num_iterations": 1,
        "foarl_token_aggregation": "completion_token_mean_then_sequence_mean",
        "foarl_sync_ref_model": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_immutable_semantics_receipt_separates_advantage_and_clipping_epsilons():
    metadata = FOARL_F638_SEMANTICS.metadata()
    assert metadata["method_id"] == "foarl_f638_corrected"
    assert metadata["reward_heads"] == ["optimality", "feasibility"]
    assert metadata["reward_weights"] == [1.0, 1.0]
    assert metadata["num_generations"] == 8
    assert metadata["beta"] == 0.05
    assert metadata["advantage_normalization"] == {
        "scope": "group",
        "std_estimator": "sample_unbiased",
        "epsilon": 1e-4,
    }
    assert metadata["clip_epsilon"] == 0.2
    assert metadata["num_iterations"] == 1
    assert metadata["loss_type"] == "grpo"
    assert metadata["sync_ref_model"] is False
    assert metadata["reference"] == "explicit_frozen_corrected_sft_step_zero"
    assert metadata["shared_trainer_source"]["version"] == (
        FOARL_F638_CORRECTED_TRAINER_VERSION
    )
    assert metadata["shared_trainer_source"]["sha256"] != (
        TSP_FCDF_BETA_PILOT_TRAINER_SHA256
    )
    assert "not an official reproduction" in metadata["classification"]
    assert FOARL_F638_REWARD_HEAD_ORDER == ("optimality", "feasibility")


def test_dedicated_parser_has_exact_defaults_and_does_not_require_ambiguous_model_path():
    parser = build_foarl_f638_corrected_parser()
    args = parser.parse_args(
        ["--sft_checkpoint_path", "/tmp/corrected-sft", "--tasks", "tsp"]
    )
    assert args.model_path is None
    assert args.sft_checkpoint_path == "/tmp/corrected-sft"
    assert args.use_primal_dual is False
    assert args.use_clasp is False
    assert args.prompt_style == "jiang_f638"
    assert args.no_reason is False
    assert args.num_generations == 8
    assert args.grpo_beta == 0.05
    assert args.grpo_epsilon == 0.2
    assert args.grpo_epsilon_high == 0.2
    assert args.grpo_loss_type == "grpo"
    assert args.foarl_advantage_epsilon == 1e-4
    assert args.foarl_scale_rewards == "group"
    assert args.foarl_num_iterations == 1
    assert args.foarl_sync_ref_model is False
    assert args.per_device_train_batch_size == 8
    assert args.per_device_eval_batch_size == 8
    assert args.gradient_accumulation_steps == 8
    assert args.learning_rate == 1e-6
    assert args.weight_decay == 0.02
    assert args.max_prompt_length == 20_000
    assert args.max_completion_length == 1_000
    assert args.lora_r == args.lora_alpha == 64
    assert args.save_steps == 50


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tasks", ["mvc"], "FOARL tasks"),
        ("tasks", ["tsp", "cvrp"], "exactly one task-specialist"),
        ("use_primal_dual", True, "forbids primal-dual"),
        ("use_clasp", True, "forbids --use_clasp"),
        ("full_finetune", True, "corrected SFT adapter"),
        ("verify_gt_rewards", True, "must not substitute"),
        ("num_generations", 9, "num_generations=8"),
        ("prompt_style", "alpaca", "prompt_style=jiang_f638"),
        ("no_reason", True, "reasoning-enabled"),
        ("grpo_beta", 0.0, "grpo_beta"),
        ("grpo_epsilon", 1e-4, "PPO clipping"),
        ("grpo_loss_type", "dr_grpo", "grpo_loss_type=grpo"),
        ("foarl_advantage_epsilon", 1e-5, "foarl_advantage_epsilon"),
        ("foarl_scale_rewards", "batch", "scale_rewards=group"),
        ("foarl_num_iterations", 2, "num_iterations=1"),
        ("foarl_token_aggregation", "global_token_mean", "completion-token mean"),
        ("foarl_sync_ref_model", True, "sync_ref_model=false"),
    ],
)
def test_runtime_args_fail_closed_on_every_semantic_drift(field, value, message):
    with pytest.raises(ValueError, match=message):
        FOARL_F638_CORRECTED_RUNTIME.validate_args(_canonical_args(**{field: value}))


def test_live_prompt_template_and_versioned_shared_trainer_are_bound(monkeypatch):
    assert validate_foarl_f638_prompt_template()["status"] == "PASS"
    # A successor entrypoint must not be mislabeled as the historical source.
    assert require_train_grpo_source(FOARL_TEACHER_DEV_TRAINER_VERSION)["status"] == "PASS"
    with pytest.raises(RuntimeError, match="source identity mismatch"):
        require_train_grpo_source(FOARL_F638_CORRECTED_TRAINER_VERSION)

    from clasp.data import prompts

    monkeypatch.setattr(prompts, "JIANG_F638_RL_TEMPLATE", "drifted {} {}")
    with pytest.raises(RuntimeError, match="prompt template drift"):
        validate_foarl_f638_prompt_template()


def test_loader_environment_rejects_inherited_representation_drift(monkeypatch):
    monkeypatch.setenv("CLASP_TSP_INPUT_ORDER", "morton")
    with pytest.raises(ValueError, match="representation-altering"):
        seal_foarl_f638_loader_environment("tsp")

    monkeypatch.delenv("CLASP_TSP_INPUT_ORDER")
    sealed = seal_foarl_f638_loader_environment("mis")
    assert sealed["CLASP_LORA_TARGET_MODULES"] == "jiang"
    assert sealed["CLASP_LORA_BIAS"] == "all"
    assert sealed["CLASP_MIS_JIANG_RESPONSE_OUTPUT"] == "1"
    assert sealed["CLASP_TSP_PROMPT_KNN"] == "0"


def test_checkpoint_task_binding_rejects_cross_task_specialist(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    receipt = model / "hyperparams.json"
    receipt.write_text(
        '{"stage":"sft","tasks":["tsp"],"args":{"task":"tsp","tasks":["tsp"]}}',
        encoding="utf-8",
    )
    bound = verify_foarl_f638_checkpoint_task_binding(str(model), "tsp")
    assert bound["status"] == "BOUND"
    assert bound["receipts"][0]["task"] == "tsp"
    with pytest.raises(ValueError, match="disagrees with requested specialist"):
        verify_foarl_f638_checkpoint_task_binding(str(model), "cvrp")


def test_runtime_config_overrides_are_complete_and_resolved_values_are_revalidated():
    expected = {
        "beta": 0.05,
        "epsilon": 0.2,
        "epsilon_high": 0.2,
        "loss_type": "grpo",
        "scale_rewards": "group",
        "num_iterations": 1,
        "sync_ref_model": False,
        "reward_weights": [1.0, 1.0],
        "importance_sampling_level": "token",
        "mask_truncated_completions": False,
        "use_liger_loss": False,
    }
    overrides = FOARL_F638_CORRECTED_RUNTIME.grpo_config_overrides(
        {name: object() for name in expected}
    )
    assert overrides == expected
    FOARL_F638_CORRECTED_RUNTIME.validate_grpo_config(
        SimpleNamespace(num_generations=8, **overrides)
    )

    with pytest.raises(RuntimeError, match="missing reward_weights"):
        FOARL_F638_CORRECTED_RUNTIME.grpo_config_overrides(
            {name: object() for name in expected if name != "reward_weights"}
        )
    with pytest.raises(RuntimeError, match="scale_rewards='batch'"):
        FOARL_F638_CORRECTED_RUNTIME.validate_grpo_config(
            SimpleNamespace(num_generations=8, **{**overrides, "scale_rewards": "batch"})
        )


class _CompatibleTrainer:
    def compute_loss(self):
        per_token_loss = self.per_token_loss
        completion_mask = self.completion_mask
        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        return loss

    def _generate_and_score_completions(self):
        rewards = self.rewards
        advantages = self.advantages
        std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        advantages = advantages / (std_rewards + 1e-4)
        return advantages


class _GlobalTokenMeanTrainer(_CompatibleTrainer):
    def compute_loss(self):
        per_token_loss = self.per_token_loss
        completion_mask = self.completion_mask
        if self.loss_type == "grpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        return loss


class _PopulationStdTrainer(_CompatibleTrainer):
    def _generate_and_score_completions(self):
        rewards = self.rewards
        advantages = self.advantages
        std_rewards = rewards.view(-1, self.num_generations).std(dim=1, correction=0)
        advantages = advantages / (std_rewards + 1e-4)
        return advantages


def test_trainer_differential_guard_accepts_exact_trl024_and_rejects_both_nearby_formulas():
    receipt = validate_trl_foarl_implementation(_CompatibleTrainer, trl_version="0.24.0")
    assert receipt["status"] == "PASS"
    assert receipt["loss_reduction"] == "completion_token_mean_then_sequence_mean"
    assert receipt["advantage_normalization"]["std_estimator"] == "sample_unbiased"
    with pytest.raises(RuntimeError, match="per-sequence-token-mean"):
        validate_trl_foarl_implementation(_GlobalTokenMeanTrainer, trl_version="0.24.0")
    with pytest.raises(RuntimeError, match="sample-std"):
        validate_trl_foarl_implementation(_PopulationStdTrainer, trl_version="0.24.0")
    with pytest.raises(RuntimeError, match=r"TRL 0\.24\.x"):
        validate_trl_foarl_implementation(_CompatibleTrainer, trl_version="0.25.0")


def test_live_installed_trl_implements_and_resolves_the_exact_profile(tmp_path):
    trl = pytest.importorskip("trl")
    from trl import GRPOConfig, GRPOTrainer

    receipt = validate_trl_foarl_implementation(GRPOTrainer)
    assert receipt["trl_version"] == "0.24.0"
    signature = inspect.signature(GRPOConfig.__init__).parameters
    overrides = FOARL_F638_CORRECTED_RUNTIME.grpo_config_overrides(signature)
    config = GRPOConfig(
        output_dir=str(tmp_path / "output"),
        eval_strategy="no",
        report_to="none",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        num_generations=8,
        generation_batch_size=8,
        max_completion_length=16,
        remove_unused_columns=False,
        use_cpu=True,
        bf16=False,
        fp16=False,
        **overrides,
    )
    FOARL_F638_CORRECTED_RUNTIME.validate_grpo_config(config)
    assert trl.__version__ == "0.24.0"


def test_trl024_formula_is_numerically_differential_to_pinned_helper_when_torch_available():
    torch = pytest.importorskip("torch")
    from clasp.baselines.llmcosolver_f638_objective import sequence_mean_grpo_loss

    policy = torch.tensor([[-1.0, -2.0, -3.0], [-0.4, -0.8, -1.2]], requires_grad=True)
    reference = torch.tensor([[-1.2, -2.1, -2.5], [-0.7, -0.9, -1.1]])
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]])
    advantages = torch.tensor([1.5, -0.5])
    expected, _, _ = sequence_mean_grpo_loss(
        policy,
        reference,
        mask,
        advantages,
        beta=0.05,
    )
    kl = torch.exp(reference - policy) - (reference - policy) - 1
    per_token_loss = -(
        torch.exp(policy - policy.detach()) * advantages.unsqueeze(1) - 0.05 * kl
    )
    actual = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    global_token_mean = (per_token_loss * mask).sum() / mask.sum()
    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(actual, global_token_mean)


def test_console_entrypoint_is_registered():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        'clasp-foarl-f638-corrected = "clasp.entrypoints.train_foarl_f638_corrected:main"'
        in pyproject
    )


def test_shared_grpo_runtime_wires_profile_without_replacing_model_or_checkpoint_path():
    source = (ROOT / "src/clasp/entrypoints/train_grpo.py").read_text(encoding="utf-8")
    assert "def run_grpo(args, *, runtime_profile=None)" in source
    assert "runtime_profile.validate_trainer_implementation(GRPOTrainer)" in source
    assert '"train": runtime_profile.validate_dataset(train_dataset, label="train")' in source
    assert "profile_grpo_kwargs = runtime_profile.grpo_config_overrides(grpo_signature)" in source
    assert "runtime_profile.validate_grpo_config(grpo_args)" in source
    assert '"runtime_profile": runtime_profile_receipt' in source
    assert "reward_funcs = list(runtime_profile.reward_functions())" in source
    assert "runtime_profile_fixed_reference_required" in source
    # The vertical slice deliberately reuses the already-audited loading and
    # exact-reference injection instead of introducing a second trainer stack.
    assert "load_model_and_tokenizer(" in source
    assert "resolve_explicit_fixed_kl_reference(" in source
    assert "trainer.ref_model = prepared_kl_reference" in source
