"""Rank-owned controls, immutable resume identity, and untouched algorithms."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from clasp.models.external_baseline_state import semantic_state_sha256
from clasp.trainers.budget_matched_grpo import (
    CONTRACT_FILENAME,
    budget_matched_trainer_class,
)


def contract(**changes):
    value = {"protocol": "budget-test-v1", "method": "classic_grpo", "task": "mis",
             "seed": 43, "horizon": 800, "contract_file": {"sha256": "a" * 64},
             "source": {"sha256": "b" * 64}}
    value.update(changes)
    return value


class BaseTrainer:
    def __init__(self, directory, *, rank=0, world_size=1):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.args = SimpleNamespace(output_dir=str(directory), max_steps=800,
                                    ignore_data_skip=False)
        self.accelerator = SimpleNamespace(
            process_index=rank, num_processes=world_size, is_main_process=rank == 0,
            wait_for_everyone=lambda: None)
        self.state = SimpleNamespace(global_step=50)
        self.parameter = torch.nn.Parameter(torch.tensor([1.0 + rank], dtype=torch.float64))
        self.optimizer = torch.optim.AdamW([self.parameter], lr=0.003)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: 1 - step / 800)
        self.loads = self.saves = 0

    def _get_output_dir(self, trial):
        return self.args.output_dir

    def _save_checkpoint(self, model, trial):
        self.saves += 1
        directory = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
        directory.mkdir(exist_ok=True)
        (directory / "trainer_state.json").write_text(json.dumps(
            {"global_step": self.state.global_step, "max_steps": self.args.max_steps}))

    def _load_optimizer_and_scheduler(self, checkpoint):
        self.loads += 1
        return "base-return"

    def compute_loss(self, *args):
        return "original-loss"

    def _get_train_sampler(self, *args):
        return "original-sampler"


def make(directory, *, binding=None, rank=0, world_size=1, external=False):
    cls = budget_matched_trainer_class(BaseTrainer, external_baseline=external)
    obj = cls(directory, rank=rank, world_size=world_size,
              budget_resume_contract=contract() if binding is None else binding)
    if world_size > 1:
        # Simulate successful collectives while keeping each rank's moments distinct.
        obj._budget_gather = lambda value: [value] * world_size
    return obj


def step(trainer):
    trainer.optimizer.zero_grad()
    trainer.parameter.square().sum().backward()
    trainer.optimizer.step()
    trainer.lr_scheduler.step()


def test_controls_restore_each_rank_exactly_and_preserve_base_return(tmp_path):
    saved = []
    for rank in range(4):
        original = make(tmp_path / "original", rank=rank, world_size=4)
        step(original)
        original._save_checkpoint(None, None)
        saved.append(semantic_state_sha256(original.optimizer.state_dict()))
    assert len(set(saved)) == 4
    checkpoint = tmp_path / "original" / "checkpoint-50"
    for rank in range(4):
        resumed = make(tmp_path / "resumed", rank=rank, world_size=4)
        assert resumed._load_optimizer_and_scheduler(checkpoint) == "base-return"
        assert semantic_state_sha256(resumed.optimizer.state_dict()) == saved[rank]
        receipt = json.loads((tmp_path / "resumed" / f"budget_resume_50_rank{rank}.json").read_text())
        assert receipt["rank"] == rank and receipt["contract"] == contract()
        assert (tmp_path / "resumed" / f"external_resume_50_rank{rank}.json").exists()


@pytest.mark.parametrize("changes", [
    {"method": "foarl"}, {"source": {"sha256": "c" * 64}},
    {"contract_file": {"sha256": "d" * 64}}, {"seed": 42}, {"task": "cvrp"},
])
def test_changed_run_identity_refused_before_base_load(tmp_path, changes):
    original = make(tmp_path / "original")
    original._save_checkpoint(None, None)
    resumed = make(tmp_path / "resumed", binding=contract(**changes))
    with pytest.raises(RuntimeError, match="contract differs"):
        resumed._load_optimizer_and_scheduler(tmp_path / "original" / "checkpoint-50")
    assert resumed.loads == 0


@pytest.mark.parametrize("changes", [{"max_steps": 100}, {"global_step": 0}, {"global_step": 801}])
def test_changed_checkpoint_step_or_horizon_refused_before_base_load(tmp_path, changes):
    original = make(tmp_path / "original")
    original._save_checkpoint(None, None)
    checkpoint = tmp_path / "original" / "checkpoint-50"
    path = checkpoint / "trainer_state.json"
    state = json.loads(path.read_text())
    state.update(changes)
    path.write_text(json.dumps(state))
    resumed = make(tmp_path / "resumed")
    with pytest.raises(RuntimeError, match="resume contract validation"):
        resumed._load_optimizer_and_scheduler(checkpoint)
    assert resumed.loads == 0


def test_missing_rank_state_refuses_canonical_fallback(tmp_path):
    original = make(tmp_path / "original")
    original._save_checkpoint(None, None)
    checkpoint = tmp_path / "original" / "checkpoint-50"
    (checkpoint / "optimizer_rank0.pt").unlink()
    resumed = make(tmp_path / "resumed")
    with pytest.raises(RuntimeError, match="rank-owned restoration"):
        resumed._load_optimizer_and_scheduler(checkpoint)
    assert not (tmp_path / "resumed" / "budget_resume_50_rank0.json").exists()


def test_candidate_does_not_duplicate_existing_rank_hooks(tmp_path, monkeypatch):
    import clasp.trainers.budget_matched_grpo as module

    def prohibited(*args, **kwargs):
        raise AssertionError("Candidate rank-state hook must remain owned by its base")

    monkeypatch.setattr(module, "save_rank_state", prohibited)
    monkeypatch.setattr(module, "restore_rank_state", prohibited)
    candidate = make(tmp_path, external=True)
    candidate._save_checkpoint(None, None)
    candidate._load_optimizer_and_scheduler(tmp_path / "checkpoint-50")
    assert candidate.saves == candidate.loads == 1
    assert (tmp_path / "checkpoint-50" / CONTRACT_FILENAME).is_file()


def test_other_rank_failure_propagates_and_no_seal_is_written(tmp_path):
    original = make(tmp_path, world_size=4)
    original._budget_gather = lambda value: [None, "simulated rank 1 write failure", None, None]
    with pytest.raises(RuntimeError, match="rank 1 write failure"):
        original._save_checkpoint(None, None)
    assert not (tmp_path / "checkpoint-50" / CONTRACT_FILENAME).exists()


def test_algorithms_are_inherited_without_overrides_and_wrapper_is_opt_in(tmp_path):
    wrapped = budget_matched_trainer_class(BaseTrainer, external_baseline=False)
    assert wrapped.compute_loss is BaseTrainer.compute_loss
    assert wrapped._get_train_sampler is BaseTrainer._get_train_sampler
    with pytest.raises(ValueError, match="exactly once"):
        budget_matched_trainer_class(wrapped, external_baseline=False)
    assert not hasattr(BaseTrainer, "_budget_payload")


def test_horizon_mismatch_and_data_skip_rejected(tmp_path):
    with pytest.raises(ValueError, match="Trainer horizon"):
        make(tmp_path, binding=contract(horizon=100))

    class SkipTrainer(BaseTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.args.ignore_data_skip = True

    wrapped = budget_matched_trainer_class(SkipTrainer, external_baseline=False)
    with pytest.raises(ValueError, match="ignore_data_skip"):
        wrapped(tmp_path, budget_resume_contract=contract())


def test_native_trainer_algorithms_and_signatures_are_preserved():
    pytest.importorskip("trl")
    from clasp.trainers.external_baseline_grpo import ExternalBaselineGRPOTrainer
    from clasp.trainers.verified_teacher_grpo import VerifiedTeacherGRPOTrainer

    for base, external in ((VerifiedTeacherGRPOTrainer, False), (ExternalBaselineGRPOTrainer, True)):
        wrapped = budget_matched_trainer_class(base, external_baseline=external)
        assert wrapped.compute_loss is base.compute_loss
        assert wrapped._get_train_sampler is base._get_train_sampler
        assert wrapped._generate_and_score_completions is base._generate_and_score_completions


def test_native_control_checkpoint_roundtrip_uses_real_trainer_hooks(tmp_path):
    pytest.importorskip("trl")
    from test_external_baseline_trainer import tiny_trainer
    from clasp.trainers.verified_teacher_grpo import VerifiedTeacherGRPOTrainer

    seed = tiny_trainer(tmp_path / "source")
    wrapped = budget_matched_trainer_class(VerifiedTeacherGRPOTrainer, external_baseline=False)
    control = wrapped(
        model=seed.model, args=seed.args, train_dataset=seed.train_dataset,
        processing_class=seed.processing_class, reward_funcs=seed.reward_funcs,
        verified_teacher_weight=0.0, budget_resume_contract=contract(horizon=2))
    control.create_optimizer_and_scheduler(2)
    loss = sum(parameter.square().sum() for parameter in control.model.parameters()
               if parameter.requires_grad)
    loss.backward()
    control.optimizer.step()
    control.lr_scheduler.step()
    control.state.global_step = 1
    control.state.max_steps = 2
    expected = semantic_state_sha256(control.optimizer.state_dict())
    control._save_checkpoint(control.model, None)
    checkpoint = tmp_path / "source" / "checkpoint-1"
    assert (checkpoint / "rng_state.pth").exists()
    assert (checkpoint / "optimizer_rank0.pt").exists()
    seed.args.output_dir = str(tmp_path / "resumed")
    Path(seed.args.output_dir).mkdir()
    resumed = wrapped(
        model=seed.model, args=seed.args, train_dataset=seed.train_dataset,
        processing_class=seed.processing_class, reward_funcs=seed.reward_funcs,
        verified_teacher_weight=0.0, budget_resume_contract=contract(horizon=2))
    resumed.create_optimizer_and_scheduler(2)
    resumed._load_optimizer_and_scheduler(str(checkpoint))
    assert semantic_state_sha256(resumed.optimizer.state_dict()) == expected
    assert (tmp_path / "resumed" / "budget_resume_1_rank0.json").exists()
