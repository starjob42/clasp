"""Synthetic custody/strict replay guards; never run a model or read live quality."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/rloo_one_shot_20261016"
PATH = HERE / "analyze_teacher_op_v2.py"


def module(path, name):
    raw = path.read_bytes();value = types.ModuleType(name)
    value.__file__ = str(path);value.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    exec(compile(raw, str(path), "exec", dont_inherit=True), value.__dict__)
    return value


@pytest.fixture
def A():
    return module(PATH, "synthetic_teacher_analysis")


class Tokens:
    eos_token_id = 1
    def __len__(self):
        return 512
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - 2) for i in ids if i != 1)


class Fixture:
    def __init__(self, root, A, monkeypatch):
        self.root, self.A, self.reads = root, A, []
        self.tokenizer = Tokens()
        self.base = module(HERE / "run_local_7b_v1.py", "synthetic_closed_base")
        self.metadata = module(HERE / "analysis_admission_v1.py", "synthetic_metadata_primitives")
        self.scorer = module(ROOT / "src/clasp/baselines/sft_anchor_pilot.py", "actual_strict_scorer")
        self.config = {"do_sample": False, "num_return_sequences": 1, "num_beams": 1, "max_new_tokens": 3000,
                       "repetition_penalty": 1., "suppress_tokens": None}
        self.core = types.SimpleNamespace(generation_config=lambda *a, **k: types.SimpleNamespace(to_dict=lambda: self.config),
                                         score_completion=self.scorer.score_completion)
        self.rows = [{"instance_id": "evaluation-" + str(i), "semantic_id": "semantic-" + str(i), "task_type": "op",
                      "prompt": "Synthetic OP prompt " + str(i), "instance_json": [[[0, 0], [1, 0], [0, 1]], [0, 5, 7], 10],
                      "output": "Route: [0, 1]"} for i in range(96)]
        self.train = [{**self.rows[0], "instance_id": "training-0", "semantic_id": "training-semantic"}]
        self.logical = {"task": "op", "scope": "prospective_teacher_development", "training_rows": self.train,
                        "training": {"prompt_draws": [["training-0"] for _ in range(100)]}}
        self.shared = {"scope": "production_training_collection_7b", "synthetic_fixture": True,
                       "logical_contract": self.logical, "SFT_identity": {"whole_actor_sha256": "1" * 64}, "shared_data_sha256": "2" * 64}
        self.targets = {"targets": [{"instance_id": "training-0", "target_text_sha256": "3" * 64}], "teacher_targets_sha256": "4" * 64}
        self.inventory = [{"name": "layer.lora_A.default.weight", "shape": [1, 1], "numel": 1, "dtype": "torch.float32"}]
        self.alpha = {recipe: {"alpha": 1., "calibration_sha256": str(i + 5) * 64, "parameter_inventory": self.inventory}
                      for i, recipe in enumerate(("rloo_cost", "foarl_released", "foarl_cost_changed"))}
        source = root / "source";source.mkdir()
        extras = {}
        for name in A.EXTRA_SOURCE_PINS:
            path = source / name;path.write_bytes((HERE / name).read_bytes());extras[name] = A.binding(path)
        monkeypatch.setattr(A, "EXTRA_SOURCE_PINS", {name: row["sha256"] for name, row in extras.items()})
        self.anchor = {"initializer": {"base_root": str(root / "base"), "sft_root": str(root / "sft"),
                                       "base_files": [], "sft_files": []}}
        self.roster = {"source_manifest": self.write(root / "manifest.json", {"synthetic_fixture": True})}
        roster = self.write(root / "roster.json", self.roster)
        plan = self.write(root / "plan.json", {"synthetic_fixture": True})
        monkeypatch.setattr(A, "ROSTER_SHA", roster["sha256"]);monkeypatch.setattr(A, "PLAN_SHA", plan["sha256"])
        self.prior = {"status": "complete_task_raw_replay", "development_screen_passed": False,
                      "synthetic_fixture": True, "quality_analyzed": True}
        self.request = {"protocol": A.PROTOCOL, "synthetic_fixture": True, "original_roster": roster, "plan": plan,
                        "extra_sources": extras, "slots": [],
                        "prior_failed_screen": self.write(root / "control_v1/op_analysis.json", self.prior)}
        for key, filename, pin in (("old_analyzer", "analyze_task.py", "OLD_ANALYZER_SHA"),
                                   ("old_metadata", "analysis_admission_v1.py", "OLD_METADATA_SHA"),
                                   ("old_screen_addendum", "development_screen_sft_addendum_v1.json", "OLD_SCREEN_SHA")):
            path = HERE / "pipeline_v1" / filename
            if not path.exists():
                path = self.root / filename;self.write(path, {"synthetic_fixture": True})
            row = A.binding(path);self.request[key] = row;monkeypatch.setattr(A, pin, row["sha256"])
        data = {"logical_contract": self.write(root / "logical.json", self.logical),
                "shared_data": self.write(root / "shared.json", self.shared),
                "teacher_targets": self.write(root / "targets.json", self.targets),
                "evaluation": self.write(root / "evaluation.json", self.rows),
                "calibrations": {recipe: self.write(root / (recipe + ".json"), value) for recipe, value in self.alpha.items()}}
        for arm in A.ORDER:
            output = root / arm;output.mkdir()
            contract = {"synthetic_fixture": True, "mode": "study", "arm": arm, "output": str(output),
                        "contract_sha256": hashlib.sha256(arm.encode()).hexdigest(), "attempt_id": "study.op." + arm + ".seed73",
                        "extra_sources": extras, "roster": roster, "plan": plan, "failed_OP_screen": self.request["prior_failed_screen"], **data}
            slot = {"arm": arm, "output": str(output), "contract": self.write(root / (arm + ".contract.json"), contract)}
            self.request["slots"].append(slot)
            self.write(output / "attempt.json", {"contract_file_binding": slot["contract"], "contract_sha256": contract["contract_sha256"],
                       "attempt_id": contract["attempt_id"], "attempt_index": 1, "automatic_retry": False,
                       "maximum_wall_seconds": 14400, "requested_gpu_hours": 4.0})
            tree = output / "terminal";tree.mkdir()
            for name in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "tokenizer_config.json"):
                (tree / name).write_bytes(b"explicit synthetic terminal bytes")
            files = [{"path": p.name, "sha256": A.sha(p.read_bytes()), "size_bytes": p.stat().st_size} for p in sorted(tree.iterdir())]
            terminal = {"files": files, "tree_sha256": A.digest(files)}
            teacher_only = arm == "teacher_only";recipe = {"rloo_teacher": "rloo_cost", "foarl_teacher": "foarl_released", "foarl_cost_teacher": "foarl_cost_changed"}.get(arm)
            training = {"synthetic_fixture": True, "initial_whole_actor_sha256": "1" * 64, "whole_actor_sha256": "a" * 64,
                        "actor_changed": True, "terminal_tree": terminal, "trained_updates": 100, "RL_responses": 0 if teacher_only else 900,
                        "teacher_forwards": 100, "reference_integrity": None if teacher_only else True,
                        "checkpoint_policy": "terminal_only_no_optimizer_resume", "training_wall_seconds": 20.,
                        "GC": {"effective_use_reentrant": False, "status": "passed", "frozen_embedding_weights": True},
                        "actual_windows": [["training-0"] * (1 if teacher_only else 9) for _ in range(100)],
                        "teacher_target_tokens_including_Objective_and_EOS": 200, "teacher_forward_tokens": 300,
                        "teacher_records": [self.teacher_record(i, teacher_only, self.alpha.get(recipe)) for i in range(100)]}
            if teacher_only:
                training.update(scope="production_training_collection_7b", BF16_autocast=True, reference_KL=False,
                                reference_model_loaded_by_helper=False, all_grads_cleared=True, teacher_CE_backwards=100,
                                parameter_inventory=self.inventory, optimizer_settings={"learning_rate": 1e-5, "adam_beta1": .9, "adam_beta2": .999,
                                "adam_epsilon": 1e-8, "weight_decay": 0., "max_grad_norm": 1., "lr_scheduler_type": "constant", "warmup_ratio": 0., "optim": "adamw_torch"})
            else:
                training.update(reference_sha256="1" * 64, calibration_sha256=self.alpha[recipe]["calibration_sha256"], alpha=1.,
                                shared_data_sha256=self.shared["shared_data_sha256"], teacher_targets_sha256=self.targets["teacher_targets_sha256"])
            proof = self.write(output / "training_complete.json", training)
            reloaded = self.write(output / "checkpoint_reload_exact.json", {"bitwise_state_equal": True, "whole_actor_sha256": "a" * 64,
                                  "training_complete_binding": proof, "terminal_tree": terminal})
            model = self.write(output / "model_admission.json", {"initial_whole_actor_sha256": "1" * 64, "runtime": A.PACKAGES,
                               "parameter_counts_by_dtype": {"torch.bfloat16": 6_000_000_001, "torch.float32": 1},
                               "devices": ["cuda:0"], "trainable_parameters": 1, "actual_BF16_base_weights_verified": True,
                               "FP32_adapter_parameter_names": ["layer.lora_A.default.weight"]})
            inputs = self.write(output / "input_admission.json", {"contract_file_binding": slot["contract"], "source_manifest": self.roster["source_manifest"],
                                "extra_sources": extras, "new_policy_calls_before_admission": 0, "initializer_files": []})
            owner = self.write(output / "gpu_owner.json", {"protocol": "one-shot-exclusive-gpu-owner-v1", "gpu_uuid": self.base.GPU_UUID,
                               "pid": os.getpid(), "uid": os.getuid(), "all_locks_held_until_GPU_lifetime_closed": True})
            raw = self.raw(output / "raw_completions.jsonl", better=arm == "rloo_teacher")
            result = {"synthetic_fixture": True, "protocol": "one-shot-calibrated-teacher-OP-7b-execution-v2", "status": "complete_teacher_development_arm",
                      "arm": arm, "task": "op", "contract": contract, "contract_file_binding": slot["contract"], "extra_sources": extras,
                      "shared_data_binding": data["shared_data"], "targets_binding": data["teacher_targets"], "calibrations": data["calibrations"],
                      "checkpoint_reload_exact": True, "superiority_established": False, "quality_summary": None,
                      "trained_updates": 100, "RL_responses": 0 if teacher_only else 900, "teacher_forwards": 100, "evaluation_responses": 96,
                      "reference_integrity": None if teacher_only else True, "initial_whole_actor_sha256": "1" * 64, "whole_actor_sha256": "a" * 64,
                      "training_complete_binding": proof, "checkpoint_reload_binding": reloaded, "model_admission_binding": model,
                      "input_admission_binding": inputs, "gpu_owner_binding": owner, "raw_completions_binding": raw,
                      "resources": {"runtime": A.PACKAGES, "GPU_uuid": self.base.GPU_UUID, "physical_gpus": 1,
                                    "worker_wall_seconds": 25., "training_wall_seconds": 20., "evaluation_wall_seconds": 3.,
                                    "peak_allocated_bytes": 100, "peak_reserved_bytes": 200}}
            slot["result"] = self.write(output / "result.json", result)
            slot["completion"] = self.write(output / "completion.json", {"status": "completed", "error": None, "automatic_retry": False,
                                       "contract_file_sha256": slot["contract"]["sha256"], "parent_elapsed_wall_seconds": 30.,
                                       "actual_GPU_hour_upper_bound": 30. / 3600, "requested_gpu_hours": 4.0})
        old_results = []
        for arm in A.OLD_ORDER:
            output = root / arm;output.mkdir()
            proof = self.write(output / "training_complete.json", {"initial_whole_actor_sha256": "1" * 64})
            old_results.append(({"arm": arm}, {"contract": {**self.anchor, "evaluation": data["evaluation"]},
                               "training_complete_binding": proof, "raw_completions_binding": self.raw(output / "raw_completions.jsonl"),
                               "resources": {"synthetic_fixture": True}}))
        actual_runner = module(HERE / "run_teacher_7b_v2.py", "actual_teacher_metadata_utilities")
        self.context = {"runner": types.SimpleNamespace(PROTOCOL=actual_runner.PROTOCOL, RECIPES=actual_runner.RECIPES,
                        completed_parent=actual_runner.completed_parent,
                        validate_contract=lambda contract: (self.roster, self.anchor, self.logical, self.rows, self.base, {})),
                        "metadata": self.metadata, "original": {"results": old_results}, "base": self.base,
                        "old": types.SimpleNamespace(analyze=self.old_analyze)}
        self.old_replays = 0
        monkeypatch.setattr(A, "load_context", lambda request, read: self.context)
        monkeypatch.setattr(A, "replay_context", lambda context, anchor: (self.tokenizer, self.core, self.scorer))
        # Full raw/physical metadata path is real; speed only the eight numeric
        # bootstraps in the synthetic end-to-end guard. A separate test below
        # executes the exact frozen50k routine on nonconstant paired values.
        real_bootstrap = self.scorer.paired_bootstrap
        def small_bootstrap(left, right, *, seed, replicates):
            assert seed == A.BOOTSTRAP_SEED and replicates == 50000
            value = real_bootstrap(left, right, seed=seed, replicates=100)
            value["bootstrap_replicates"] = replicates
            return value
        self.scorer.paired_bootstrap = small_bootstrap
        original_read = A.read_bound
        def track(row):
            self.reads.append(row["path"])
            return original_read(row)
        monkeypatch.setattr(A, "read_bound", track)

    def old_analyze(self, *a, **k):
        self.old_replays += 1
        return self.prior

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(self.A.encoded(value) + b"\n")
        return self.A.binding(path)

    def raw(self, path, better=False):
        text = "Route: [0, 1, 2]" if better else "Route: [0, 1]"
        records = []
        for row in self.rows:
            ids = [ord(c) + 2 for c in text] + [1]
            records.append({"synthetic_fixture": True, "instance_id": row["instance_id"], "semantic_id": row["semantic_id"],
                            "prompt_sha256": self.A.sha(row["prompt"].encode()), "completion_token_ids": ids, "text": text,
                            "generated_tokens": len(ids), "eos_observed": True, "truncated": False,
                            "actual_generation_config": self.config, "use_model_defaults": False, "extra_generation_kwargs": {},
                            "generation_calls": 1, "sample_index": 0, "seed": 73, "wall_seconds": .01,
                            "strict_score": self.scorer.score_completion("op", text, row["instance_json"], row["output"])})
        path.write_bytes(b"".join(self.A.encoded(r) + b"\n" for r in records));return self.A.binding(path)

    def teacher_record(self, i, only, alpha):
        value = {"step": i, "instance_id": "training-0", "semantic_id": "training-semantic", "target_sha256": "3" * 64,
                 "CE": 1., "CE_temperature": 1., "split": "training", "target_tokens_including_Objective_and_EOS": 2,
                 "forward_tokens": 3, "masked_prompt_tokens": 1}
        if only:
            value.update(CE_forwards=1, CE_backwards=1, optimizer_steps=1, CE_multiplier=1., target_truncated=False,
                         BF16_autocast=True, reference_KL=False, RL_responses=0, RL_reward_calls=0, learning_rate=1e-5,
                         CE_reduction="mean_target_tokens_including_Objective_and_EOS")
        else:
            value.update(alpha=1., rho=1., CE_backward_multiplier=1., calibration_sha256=alpha["calibration_sha256"],
                         teacher_forward_calls=1, after_native_RL_backward_microbatch=9, CE_target_truncated=False, CE_RNG_restored=True)
        return value

    def request_binding(self):
        return self.write(self.root / "request.json", self.request)

    def mutate_proof(self, arm, name, change):
        slot = next(s for s in self.request["slots"] if s["arm"] == arm)
        result = json.loads(self.A.read_bound(slot["result"]))
        key = {"training_complete.json": "training_complete_binding", "model_admission.json": "model_admission_binding",
               "checkpoint_reload_exact.json": "checkpoint_reload_binding", "gpu_owner.json": "gpu_owner_binding",
               "input_admission.json": "input_admission_binding"}[name]
        value = json.loads(self.A.read_bound(result[key]));change(value)
        result[key] = self.write(Path(slot["output"]) / name, value)
        if name == "training_complete.json":
            reload = json.loads(self.A.read_bound(result["checkpoint_reload_binding"]))
            reload["training_complete_binding"] = result[key]
            reload["terminal_tree"] = value["terminal_tree"]
            result["checkpoint_reload_binding"] = self.write(Path(slot["output"]) / "checkpoint_reload_exact.json", reload)
        slot["result"] = self.write(Path(slot["output"]) / "result.json", result)


@pytest.fixture
def fixture(tmp_path, A, monkeypatch):
    return Fixture(tmp_path, A, monkeypatch)


def test_default_cli_is_stdlib_only_and_zero_call():
    result = subprocess.run([sys.executable, "-I", "-B", str(PATH)], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["GPU_calls"] == 0


def test_complete_synthetic_custody_and_strict_physical_replay(fixture):
    A = fixture.A
    result = A._analyze(fixture.request_binding(), synthetic_fixture=True)
    assert result["status"] == "synthetic_teacher_OP_guard_fixture"
    assert result["synthetic_screen_formula_result"] is True
    assert result["development_screen_passed"] is False
    assert result["publication_eligible"] is result["superiority_established"] is False
    assert len(result["summaries"]) == 9 and len(result["rloo_teacher_minus_control_comparisons"]) == 8
    assert result["summaries"]["rloo_teacher"]["own_feasible_signed_gap_percent"]["mean"] < 0
    assert result["reserved_local_GPU_hours"]["combined"] == 62.5


def test_public_api_rejects_synthetic_request(fixture):
    with pytest.raises(ValueError, match="synthetic"):
        fixture.A.analyze_teacher_op(fixture.request_binding())


def test_missing_fourth_arm_never_reads_results_or_quality(fixture):
    fixture.request["slots"][-1]["completion"] = None
    result = fixture.A._analyze(fixture.request_binding(), synthetic_fixture=True)
    assert result["missing_slots"] == ["rloo_teacher"]
    assert not any(p.endswith(("result.json", "raw_completions.jsonl", "training_complete.json")) for p in fixture.reads)


def test_failed_fourth_parent_rejects_before_results_or_quality(fixture):
    slot = fixture.request["slots"][-1]
    slot["completion"] = fixture.write(Path(slot["output"]) / "completion.json", {"status": "failed", "error": "timeout", "automatic_retry": False})
    fixture.reads.clear()
    with pytest.raises(ValueError, match="All four teacher parents"):
        fixture.A._analyze(fixture.request_binding(), synthetic_fixture=True)
    assert not any(p.endswith(("result.json", "raw_completions.jsonl", "training_complete.json")) for p in fixture.reads)


@pytest.mark.parametrize("name,change", [
    ("training_complete.json", lambda x: x.update(trained_updates=99)),
    ("training_complete.json", lambda x: x.update(RL_responses=899)),
    ("training_complete.json", lambda x: x.update(reference_integrity=False)),
    ("training_complete.json", lambda x: x.update(reference_sha256="b" * 64)),
    ("training_complete.json", lambda x: x["actual_windows"][-1].pop()),
    ("training_complete.json", lambda x: x["teacher_records"][-1].update(after_native_RL_backward_microbatch=8)),
    ("training_complete.json", lambda x: x["teacher_records"][-1].update(CE_RNG_restored=False)),
    ("training_complete.json", lambda x: x["teacher_records"][-1].update(alpha=True)),
    ("training_complete.json", lambda x: x["GC"].update(effective_use_reentrant=True)),
    ("model_admission.json", lambda x: x.update(actual_BF16_base_weights_verified=False)),
    ("model_admission.json", lambda x: x.update(parameter_counts_by_dtype={"torch.float32": 6_000_000_001})),
    ("checkpoint_reload_exact.json", lambda x: x.update(whole_actor_sha256="b" * 64)),
    ("checkpoint_reload_exact.json", lambda x: x.update(bitwise_state_equal=False)),
    ("input_admission.json", lambda x: x.update(new_policy_calls_before_admission=1)),
    ("input_admission.json", lambda x: x.update(initializer_files=[{"path": "unbound"}])),
    ("gpu_owner.json", lambda x: x.update(all_locks_held_until_GPU_lifetime_closed=False)),
])
def test_metadata_corruptions_reject_before_any_quality(fixture, name, change):
    fixture.mutate_proof("rloo_teacher", name, change);fixture.reads.clear()
    with pytest.raises(ValueError):
        fixture.A._analyze(fixture.request_binding(), synthetic_fixture=True)
    assert fixture.old_replays == 0
    assert not any(p.endswith("raw_completions.jsonl") for p in fixture.reads)


def test_actual_terminal_file_is_streamed_before_quality(fixture):
    (Path(fixture.request["slots"][-1]["output"]) / "terminal/adapter_model.safetensors").write_bytes(b"mutated")
    fixture.reads.clear()
    with pytest.raises(ValueError, match="terminal artifact hashes"):
        fixture.A._analyze(fixture.request_binding(), synthetic_fixture=True)
    assert fixture.old_replays == 0 and not any(p.endswith("raw_completions.jsonl") for p in fixture.reads)


@pytest.mark.parametrize("field,value", [("worker_wall_seconds", 31.), ("evaluation_wall_seconds", float("inf")),
                                        ("physical_gpus", True), ("peak_allocated_bytes", True), ("peak_reserved_bytes", 2**50)])
def test_resource_faults_before_quality(fixture, field, value):
    slot = fixture.request["slots"][-1];result = json.loads(fixture.A.read_bound(slot["result"]))
    result["resources"][field] = value
    if value == float("inf"):
        # JSON Infinity is parsed by stdlib but finite() must reject it.
        path = Path(slot["result"]["path"]);path.write_text(json.dumps(result));slot["result"] = fixture.A.binding(path)
    else:
        slot["result"] = fixture.write(Path(slot["output"]) / "result.json", result)
    fixture.reads.clear()
    with pytest.raises(ValueError):
        fixture.A._analyze(fixture.request_binding(), synthetic_fixture=True)
    assert fixture.old_replays == 0 and not any(p.endswith("raw_completions.jsonl") for p in fixture.reads)


def test_teacher_only_no_rl_and_exact_native_optimizer(fixture):
    fixture.mutate_proof("teacher_only", "training_complete.json", lambda x: x["optimizer_settings"].update(weight_decay=.01))
    with pytest.raises(ValueError, match="optimizer/inventory"):
        fixture.A._analyze(fixture.request_binding(), synthetic_fixture=True)
    assert fixture.old_replays == 0


@pytest.mark.parametrize("mutation", [lambda r: r.update(seed=True), lambda r: r.update(generated_tokens=True),
                                     lambda r: r.update(generation_calls=2), lambda r: r["actual_generation_config"].update(repetition_penalty=2.),
                                     lambda r: r.update(text="Route: [0, 1.9]"), lambda r: r["strict_score"].update(feasible=1)])
def test_raw_token_seed_sampler_physical_typed_replay(fixture, mutation):
    raw = [json.loads(line) for line in fixture.A.read_bound(fixture.request["slots"][-1]["result"]).splitlines()]
    result = raw[0];records = [json.loads(line) for line in fixture.A.read_bound(result["raw_completions_binding"]).splitlines()]
    mutation(records[0])
    with pytest.raises(ValueError):
        fixture.A.replay_arm(records, fixture.rows, fixture.base, fixture.tokenizer, fixture.core, fixture.scorer)


def test_zero_joint_support_is_reported_and_fails_screen(A):
    scorer = module(ROOT / "src/clasp/baselines/sft_anchor_pilot.py", "zero_support_scorer")
    selected = {}
    for arm in (*A.ORDER, *A.OLD_ORDER):
        selected[arm] = [{"instance_id": "a", "reference_id": "ref", "feasible": arm == "rloo_teacher",
                          "signed_gap": -.2 if arm == "rloo_teacher" else None, "clipped_gap": 0. if arm == "rloo_teacher" else None,
                          "u10": 1. if arm == "rloo_teacher" else 0.}]
    summaries, effects, screen = A.summaries_and_comparisons(selected, scorer)
    assert screen is False and all(x["zero_joint_support"] and x["n_joint"] == 0 for x in effects.values())
    assert summaries["foarl_teacher"]["own_feasible_signed_gap_percent"]["mean"] is None


def test_exact_frozen_50k_paired_bootstrap_has_pointwise_support(A):
    scorer = module(ROOT / "src/clasp/baselines/sft_anchor_pilot.py", "exact_50k_scorer")
    left = [{"instance_id": str(i), "reference_id": str(i), "feasible": True, "u10": 1., "signed_gap": g, "clipped_gap": max(g, 0.)}
            for i, g in enumerate((-.3, .1, .2))]
    right = [{**row, "u10": .5, "signed_gap": row["signed_gap"] + .1, "clipped_gap": max(row["signed_gap"] + .1, 0.)} for row in left]
    effect = scorer.paired_bootstrap(left, right, seed=A.BOOTSTRAP_SEED, replicates=A.BOOTSTRAP_REPLICATES)
    assert effect["bootstrap_replicates"] == 50000 and effect["n_joint"] == 3
    assert effect["joint_signed_gap_delta_pp"]["estimate"] == pytest.approx(-10.)
    assert effect["u10_delta_pp"]["ci95"] == [50., 50.]
    assert "not_training_seed_or_familywise" in effect["uncertainty_scope"]


def test_symlink_evidence_is_rejected(tmp_path, A):
    target = tmp_path / "target";target.write_bytes(b"{}")
    linked = tmp_path / "linked";linked.symlink_to(target)
    with pytest.raises(ValueError, match="Canonical"):
        A.binding(linked)


def test_loaded_old_analysis_cannot_rebind_later_source(tmp_path, A):
    path = tmp_path / PATH.name;path.write_bytes(PATH.read_bytes());old = module(path, "old_analysis")
    path.write_bytes(path.read_bytes() + b"\n# later bytes\n")
    with pytest.raises(ValueError, match="externally pinned"):
        old.self_binding()


@pytest.mark.parametrize("name", ["hidden.py", "nested/hidden.py", "__pycache__/stale.pyc"])
def test_full_source_closure_rejects_extra_code_before_old_gate(fixture, name):
    path = fixture.root / "source" / name;path.parent.mkdir(exist_ok=True);path.write_bytes(b"unbound")
    # Use the real implementation, not the explicitly synthetic context hook.
    real = module(PATH, "source_gate_under_test")
    real.EXTRA_SOURCE_PINS = fixture.A.EXTRA_SOURCE_PINS
    with pytest.raises(ValueError, match="Unbound Python"):
        real.load_context(fixture.request, real.read_bound)


def test_current_source_layer_constants_bind_actual_readonly_files(A):
    # Metadata only. This test never opens any current training/evaluation output.
    for name, expected in A.EXTRA_SOURCE_PINS.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected


def test_old_actor_identity_difference_rejects_before_old_quality(fixture):
    proof = fixture.context["original"]["results"][0][1]["training_complete_binding"]
    fixture.context["original"]["results"][0][1]["training_complete_binding"] = fixture.write(Path(proof["path"]), {"initial_whole_actor_sha256": "b" * 64})
    with pytest.raises(ValueError, match="different original SFT"):
        fixture.A._analyze(fixture.request_binding(), synthetic_fixture=True)
    assert fixture.old_replays == 0


def test_production_rejects_synthetic_custody_even_when_request_flag_removed(fixture):
    fixture.request["synthetic_fixture"] = False
    with pytest.raises(ValueError, match="Synthetic"):
        fixture.A.analyze_teacher_op(fixture.request_binding())
    assert fixture.old_replays == 0
