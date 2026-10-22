"""Use the installed TRL generation context to test checkpointing restoration."""
from types import SimpleNamespace

import pytest
from transformers import Qwen2Config, Qwen2ForCausalLM
from trl import GRPOTrainer
from trl.models.utils import unwrap_model_for_generation

from clasp.trainers.primal_dual_grpo import PrimalDualGRPOTrainer


def _trainer():
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=13, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
    ))
    kwargs = {"use_reentrant": False, "preserve_rng_state": False}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
    trainer = object.__new__(PrimalDualGRPOTrainer)
    trainer.model = model
    trainer.args = SimpleNamespace(gradient_checkpointing=True, gradient_checkpointing_kwargs=kwargs)
    trainer.accelerator = SimpleNamespace(
        unwrap_model=lambda model: model, state=SimpleNamespace(deepspeed_plugin=None))
    return trainer


def _keywords(trainer):
    return trainer.model.model.layers[0]._gradient_checkpointing_func.keywords


def test_actual_trl_context_loses_kwargs_and_trainer_restores_them():
    trainer = _trainer()
    before = dict(_keywords(trainer))
    with unwrap_model_for_generation(trainer.model, trainer.accelerator):
        assert not trainer.model.is_gradient_checkpointing
    # This assertion intentionally documents the pinned TRL 0.24 defect. An
    # upstream behavior change requires reconsidering the compatibility patch.
    assert _keywords(trainer)["use_reentrant"] is True
    trainer._restore_configured_gradient_checkpointing()
    assert _keywords(trainer) == before


@pytest.mark.parametrize("generation_raises", [False, True])
def test_actual_trainer_hook_restores_before_scoring_and_on_generation_failure(
    monkeypatch, generation_raises
):
    trainer = _trainer()
    restored_budgets = []
    trainer._validate_fixed_kl_reference = lambda: None
    trainer._extract_task_types = lambda inputs: []
    trainer._set_clasp_generation_budget = lambda tasks: "budget"
    trainer._restore_generation_budget = restored_budgets.append

    def generate(self, inputs):
        with unwrap_model_for_generation(self.model, self.accelerator):
            pass
        assert _keywords(self)["use_reentrant"] is True
        if generation_raises:
            raise RuntimeError("generation failed")
        return {}

    def after_generation():
        assert _keywords(trainer) == {"use_reentrant": False, "preserve_rng_state": False}
        raise RuntimeError("reached scoring after restoration")

    monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", generate)
    trainer._get_local_primal_dual_components = after_generation
    with pytest.raises(RuntimeError, match=("generation failed" if generation_raises else "reached scoring")):
        trainer._generate_and_score_completions([])
    assert _keywords(trainer) == {"use_reentrant": False, "preserve_rng_state": False}
    assert restored_budgets == ["budget"]


def test_disabled_training_checkpointing_is_not_enabled_by_restoration():
    trainer = _trainer()
    trainer.args.gradient_checkpointing = False
    trainer.model.gradient_checkpointing_disable()
    trainer._restore_configured_gradient_checkpointing()
    assert not trainer.model.is_gradient_checkpointing
