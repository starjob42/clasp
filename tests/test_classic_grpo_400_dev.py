"""Independent legacy-reward and differentiable stock-PPO control oracles."""
from __future__ import annotations

import copy
import math
import os
import statistics
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from trl import GRPOTrainer

from clasp.baselines.classic_grpo_400_dev import (
    CONFIG,
    REWARD_ENVIRONMENT,
    build_runtime,
    classic_cvrp_strict_objective_reward,
    classic_mis_strict_objective_reward,
    seal_reward_environment,
)
from clasp.config import finalize_grpo_args
from clasp.entrypoints import train_classic_grpo_400_dev as entry
from clasp.rewards.heads import build_primal_dual_manager
from clasp.trainers.primal_dual_grpo import compute_scalarized_group_advantages
from clasp.trainers.verified_teacher_grpo import VerifiedTeacherGRPOTrainer


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    # finalize_grpo_args mutates process environment; isolate every case.
    clean = {k: v for k, v in os.environ.items()
             if not k.startswith(("CLASP_", "GRPO_", "PD_", "FOARL_"))}
    monkeypatch.setattr(os, "environ", clean)
    for name, value in {**REWARD_ENVIRONMENT, "WORLD_SIZE": "4", "GRPO_DATA_SEED": "43",
                        "GRPO_STRICT_OPTIMIZER_RESUME": "1", "GRPO_STOP_AT_GLOBAL_STEP": "100"}.items():
        monkeypatch.setenv(name, value)


def parsed(task="mis", diagnostic=False):
    argv = ["--tasks", task, "--sft_checkpoint_path", "/synthetic/" + task,
            "--classic_dev_source_manifest", "/synthetic/SOURCE_MANIFEST.sha256",
            "--classic_dev_source_manifest_sha256", "a" * 64]
    if diagnostic:
        argv += ["--classic_grpo_diagnostic", "--save_steps", "5"]
    return finalize_grpo_args(entry.build_parser().parse_args(argv))


def binding():
    return {"status": "passed_frozen_manifest_successor_source", "manifest": {"sha256": "a" * 64}}


def synthetic(task):
    if task == "mis":
        instance = {"num_nodes": 5, "edges": [[0, 1], [1, 2], [3, 4]]}
        answer = "Set: [0, 2], Objective: 2"
        completions = ["Set: [0, 2, 3], Objective: 3", "Set: [1, 4], Objective: 2",
                       "Set: [1], Objective: 1", "Set: [0, 1], Objective: 2",
                       "Set: [0, 0], Objective: 2", "Set: [5], Objective: 1",
                       "Objective: 2", "Set: [0, 2]", "Set: [0, 2], Objective: 900"]
        expected = [2 + math.tanh(1), 2, 2 * math.exp(-1.5), 0, 0, 0, 0, 2, 2]
        fn = classic_mis_strict_objective_reward
    else:
        points = [[0., 0.], [1., 0.], [0., 1.], [2., 0.]]
        instance = [points, [0, 1, 1, 1], 3]
        answer = "Routes: [[0, 1, 2, 3, 0]], Objective: 999"
        routes = [[[0, 2, 1, 3, 0]], [[0, 1, 2, 3, 0]], [[0, 1, 0], [0, 2, 0], [0, 3, 0]]]
        distance = lambda rs: sum(math.dist(points[a], points[b]) for r in rs for a, b in zip(r, r[1:]))
        reference = distance(routes[1])
        gaps = [(distance(r) - reference) / reference for r in routes]
        expected = [2 * math.exp(-3 * g) if g >= 0 else 2 + math.tanh(-2 * g) for g in gaps]
        completions = [f"Routes: {r}, Objective: 999" for r in routes]
        completions += ["Routes: [[0, 1, 2, 0]], Objective: 5", "Routes: [[0, 1, 1, 2, 3, 0]], Objective: 7",
                        "Routes: [[0, 999, 0]], Objective: 2", "Objective: 7",
                        "Routes: [[0, 1, 2, 3, 0]]", "Routes: [[0, 1, 2, 3, 0]], Objective: -9"]
        expected += [0, 0, 0, 0, 2, 2]
        fn = classic_cvrp_strict_objective_reward
    kwargs = {"task_type": [task] * 9, "answer": [answer] * 9,
              "instance_json": [instance] * 9, "input": [""] * 9}
    return fn, [""] * 9, completions, kwargs, expected


@pytest.mark.parametrize("task", ["cvrp", "mis"])
def test_one_head_matches_independent_formula_and_old_zero_dual_manager(task):
    fn, prompts, completions, kwargs, expected = synthetic(task)
    old = parsed(task)
    for key, value in {"pd_component_profile": "co_feasibility_lock_all", "pd_lambda_init": 0.,
                       "pd_dual_lr": 0., "pd_dual_lr_up": 0., "pd_dual_lr_down": 0.,
                       "pd_objective_weight": 1., "pd_structure_weight": 0.,
                       "pd_cost_limit": 0., "pd_constraint_target": 1.,
                       "pd_soft_objective_for_infeasible": False, "pd_sync_across_ranks": False}.items():
        setattr(old, key, value)
    manager = build_primal_dual_manager(old, [task])
    before = copy.deepcopy((prompts, completions, kwargs))
    actual = fn(prompts, completions, **kwargs)
    legacy = manager(prompts, completions, **kwargs)
    assert actual == pytest.approx(expected, abs=1e-13)
    assert actual == legacy
    assert (prompts, completions, kwargs) == before
    assert actual == fn(prompts, completions, **kwargs,
                        allow_infeasible_partial_credit=True, apply_constraint_gate=False)
    assert len(build_runtime(binding(), task).reward_functions()) == 1


@pytest.mark.parametrize("values", [[0.] * 9, [2.] * 9, [0., 0., 0., 1., 2., 1.5, 2.1, 0., 1.8]])
def test_legacy_standardization_matches_sample_std_epsilon_oracle(values):
    r = torch.tensor(values, dtype=torch.float32)
    actual = compute_scalarized_group_advantages(r, torch.zeros_like(r), 9, std_correction=1)
    oracle = [(x - statistics.mean(values)) / (statistics.stdev(values) + 1e-4) for x in values]
    torch.testing.assert_close(actual, torch.tensor(oracle), atol=3e-7, rtol=1e-6)


@pytest.mark.parametrize("old_policy", [False, True])
def test_beta_zero_stock_loss_and_gradient_match_independent_clipped_ragged_oracle(old_policy):
    obj = object.__new__(VerifiedTeacherGRPOTrainer)
    obj.model = SimpleNamespace(training=True)
    obj.args = SimpleNamespace(delta=None, report_to=[])
    obj._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    obj.accelerator = SimpleNamespace(gather=lambda x: x)
    obj.use_liger_loss = obj.use_vllm = False
    obj.beta = 0.; obj.top_entropy_quantile = 1.
    obj.importance_sampling_level = "token"
    obj.epsilon_low = obj.epsilon_high = .2
    obj.loss_type = "grpo"; obj.current_gradient_accumulation_steps = 3
    obj.grpo_verified_improvement = False
    obj._compute_improvement_distillation_losses = lambda *a: pytest.fail("weight zero ran teacher loss")
    raw = torch.arange(9 * 3 * 5, dtype=torch.float64).reshape(9, 3, 5) / 71
    parameter = raw.clone().requires_grad_()
    targets = torch.tensor([[1, 3, 2]] * 9)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]] * 3)
    advantages = torch.tensor([1., -1., .4, -.6, 0., 1.4, -.2, .7, -1.1], dtype=torch.float64)
    def logps(*args, **kwargs):
        return parameter.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1), torch.ones_like(mask)
    obj._get_per_token_logps_and_entropies = logps
    batch = {"prompt_ids": torch.ones(9, 2, dtype=torch.long), "prompt_mask": torch.ones(9, 2),
             "completion_ids": targets, "completion_mask": mask, "advantages": advantages}
    current = logps()[0]
    ratios = torch.tensor([[1.5, .5, 1.], [.5, 1.5, 1.], [1., 1.1, .9]] * 3, dtype=torch.float64)
    old = current.detach() - ratios.log() if old_policy else current.detach()
    if old_policy:
        batch["old_per_token_logps"] = old
    stock = GRPOTrainer.compute_loss(obj, obj.model, batch)
    actual = obj.compute_loss(obj.model, batch)
    terms = []
    for i in range(9):
        tokens = []
        for t in range(3):
            if mask[i, t]:
                rho = (current[i, t] - old[i, t]).exp()
                bounded = torch.minimum(rho, rho.new_tensor(1.2)) if advantages[i] >= 0 else torch.maximum(rho, rho.new_tensor(.8))
                tokens.append(-advantages[i] * bounded)
        terms.append(torch.stack(tokens).mean())
    oracle = torch.stack(terms).mean() / 3
    assert torch.equal(actual, stock)
    torch.testing.assert_close(actual, oracle, atol=1e-15, rtol=0)
    ga = torch.autograd.grad(actual, parameter, retain_graph=True)[0]
    go = torch.autograd.grad(oracle, parameter)[0]
    torch.testing.assert_close(ga, go, atol=1e-15, rtol=0)
    assert "ref_per_token_logps" not in batch


@pytest.mark.parametrize("task", ["cvrp", "mis"])
@pytest.mark.parametrize("diagnostic,stop", [(False, 100), (False, 200), (False, 300), (False, 400), (True, 5), (True, 10)])
def test_profile_uses_existing_stock_teacher_zero_hook_and_fixed_horizon(task, diagnostic, stop, monkeypatch):
    monkeypatch.setenv("GRPO_STOP_AT_GLOBAL_STEP", str(stop))
    runtime = build_runtime(binding(), task, diagnostic)
    args = parsed(task, diagnostic)
    runtime.validate_args(args)
    assert runtime.verified_teacher_extension and runtime.teacher_weight == 0.
    assert not runtime.requires_explicit_fixed_reference
    assert runtime.metadata()["method_id"] == "classic_grpo"
    assert runtime.metadata()["beta"] == 0.
    cfg = SimpleNamespace(**CONFIG, num_generations=9, max_steps=400, generation_batch_size=36,
                          gradient_accumulation_steps=3, steps_per_generation=3, ignore_data_skip=False)
    runtime.validate_grpo_config(cfg)
    assert runtime.grpo_config_overrides(dict.fromkeys(CONFIG)) == CONFIG
    cfg.ignore_data_skip = True
    with pytest.raises(ValueError, match="ignore_data_skip"):
        runtime.validate_grpo_config(cfg)


@pytest.mark.parametrize("field,value", [("grpo_beta", .05), ("use_primal_dual", True), ("use_clasp", True),
                                        ("max_steps", 100), ("no_reason", False), ("seed", 44),
                                        ("num_generations", 8), ("verify_gt_rewards", False), ("full_finetune", True)])
def test_profile_rejects_algorithm_or_budget_drift(field, value):
    args = parsed(); setattr(args, field, value)
    with pytest.raises(ValueError, match="recipe mismatch"):
        build_runtime(binding(), "mis").validate_args(args)


@pytest.mark.parametrize("name,value", [("GRPO_GRADED_REWARD", "1"), ("GRPO_VERIFIED_IMPROVEMENT", "1"),
                                       ("CLASP_OBJECTIVE_GAP_SLOPE", "1.0"),
                                       ("CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT", "0.5"),
                                       ("CLASP_KL_REFERENCE_MODEL_PATH", "/wrong/reference")])
def test_reward_environment_drift_is_rejected_before_reward_or_resealing(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError): seal_reward_environment()
    fn, prompts, completions, kwargs, _ = synthetic("mis")
    with pytest.raises(ValueError): fn(prompts, completions, **kwargs)


def test_reward_alignment_and_task_mismatch_fail():
    fn, prompts, completions, kwargs, _ = synthetic("mis")
    with pytest.raises(ValueError): fn(prompts[:-1], completions, **kwargs)
    with pytest.raises(ValueError): fn(prompts, completions, **{**kwargs, "answer": []})
    with pytest.raises(ValueError): fn(prompts, completions, **{**kwargs, "task_type": ["cvrp"] * 9})


def test_invalid_source_fails_entry_before_loader_or_training(monkeypatch):
    def reject(*args):
        raise ValueError("unbound source")
    monkeypatch.setattr(entry, "verify_dev_source_manifest", reject)
    monkeypatch.setattr(entry, "seal_reward_environment", lambda: pytest.fail("source rejection was too late"))
    with pytest.raises(ValueError, match="unbound source"):
        entry.parse_args(["--tasks", "mis", "--sft_checkpoint_path", "/synthetic/mis",
                          "--classic_dev_source_manifest", "/synthetic/SOURCE_MANIFEST.sha256",
                          "--classic_dev_source_manifest_sha256", "a" * 64])


def test_installed_stock_trl_normalization_and_reduction_pass_shared_source_audit():
    assert build_runtime(binding(), "mis").validate_trainer_implementation(GRPOTrainer)["status"] == "PASS"


def test_dataset_requires_complete_specialist_schema_and_first_epoch_budget():
    class Dataset(list):
        column_names = ["prompt", "task_type", "answer", "instance_json", "input"]
    row = {"prompt": "Solve this graph", "task_type": "mis", "answer": "Set: [0, 2], Objective: 2",
           "instance_json": {"num_nodes": 3, "edges": [[0, 1], [1, 2]]}, "input": "graph"}
    runtime = build_runtime(binding(), "mis")
    data = Dataset([row] * 1600)
    assert runtime.validate_dataset(data, label="train")["rows"] == 1600
    with pytest.raises(ValueError, match="at least1600"):
        runtime.validate_dataset(Dataset([row] * 1599), label="train")
    with pytest.raises(ValueError, match="another specialist"):
        build_runtime(binding(), "cvrp").validate_dataset(data, label="train")
    data.column_names = [c for c in data.column_names if c != "input"]
    with pytest.raises(ValueError, match="input column"):
        runtime.validate_dataset(data, label="train")
