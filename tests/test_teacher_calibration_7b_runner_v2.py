"""Prospective production-runner guards; no model, CUDA or policy calls."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/rloo_one_shot_20261016"
PATH = HERE / "run_teacher_calibration_7b_v2.py"
SPEC = importlib.util.spec_from_file_location("prospective_calibration_runner", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def seal(value):
    value["contract_sha256"] = runner.sha(runner.encoded({k: v for k, v in value.items() if k != "contract_sha256"}))
    return value


@pytest.fixture
def declared(tmp_path):
    extras = tmp_path / "source"; extras.mkdir()
    rows = {}
    for name in runner.SOURCES:
        raw = (HERE / name).read_bytes(); path = extras / name; path.write_bytes(raw)
        rows[name] = runner.binding(path)
    def dummy(name):
        return {"path": str(tmp_path / name), "sha256": "1" * 64, "size_bytes": 1}
    bindings = {key: dummy(key + ".json") for key in runner.BINDING_KEYS}
    bindings.update(roster=runner.binding(HERE / "execution_v1/execution_roster.json"),
        plan=runner.binding(HERE / "prospective_teacher_op_v2_plan.json"),
        logical_contract=runner.binding(HERE / "teacher_data_v1/op/logical_contract.json"),
        old_analyzer=runner.binding(HERE / "pipeline_v1/analyze_task.py"),
        old_metadata=runner.binding(HERE / "pipeline_v1/analysis_admission_v1.py"),
        old_screen_addendum=runner.binding(HERE / "pipeline_v1/development_screen_addendum_v1.json"))
    value = runner.make_contract(**bindings, extra_sources=rows, output=str(tmp_path / "output"))
    return value, extras


def test_default_dry_run_has_no_input_model_or_process_calls(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(PATH)])
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run attempted execution or evidence read")
    monkeypatch.setattr(runner, "stable_bytes", forbidden)
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    before = set(sys.modules)
    runner.main()
    assert json.loads(capsys.readouterr().out)["status"] == "prepared_only_no_model_or_GPU_calls"
    assert not ({"torch", "transformers", "trl", "peft", "accelerate"} & (set(sys.modules) - before))


@pytest.mark.parametrize("field,value", [
    ("task", "cvrp"), ("seed", 74), ("seed", True), ("recipes", list(reversed(runner.RECIPES))),
    ("wall_seconds", 7201), ("wall_seconds", 7200.0), ("requested_gpu_hours", 2),
    ("attempt_index", 2), ("maximum_attempts", 2), ("automatic_retry", True),
    ("optimizer_updates", 1), ("optimizer_updates", False), ("new_generation_calls", 1),
    ("heldout_policy_calls", 1), ("GPU_UUID", "unrelated_gpu"),
    ("publication_eligible", True), ("superiority_established", True), ("output", "relative")])
def test_fixed_authority_fields_reject_resealed_changes(declared, field, value):
    contract, _ = declared; bad = copy.deepcopy(contract); bad[field] = value; seal(bad)
    with pytest.raises(ValueError):
        runner.validate_contract(bad)


def test_unknown_authority_extra_source_and_source_pin_rejected(declared):
    contract, _ = declared
    for mutation in ("unknown", "missing", "foreign", "collector", "metadata", "factory"):
        bad = copy.deepcopy(contract)
        if mutation == "unknown": bad["allow_tiny_alpha"] = True
        elif mutation == "missing": bad["extra_sources"].pop("teacher_only_7b_v2.py")
        elif mutation == "foreign": bad["extra_sources"]["unbound.py"] = dict(next(iter(bad["extra_sources"].values())))
        else:
            name = {"collector": "admit_training_collection_v1.py", "metadata": "analysis_admission_v1.py", "factory": "teacher_7b_factory_v2.py"}[mutation]
            bad["extra_sources"][name]["sha256"] = "0" * 64
        seal(bad)
        with pytest.raises(ValueError): runner.validate_contract(bad)


def test_real_frozen107_and_arm_agnostic_H100_projection_before_model_imports(declared, monkeypatch):
    contract, extras = declared
    monkeypatch.setattr(runner, "__file__", str(extras / PATH.name))
    monkeypatch.setattr(runner, "__executed_source_sha256__", contract["extra_sources"][PATH.name]["sha256"], raising=False)
    before = set(sys.modules)
    base, roster, anchor, logical, source, buffers, new = runner.source_context(contract)
    assert len(buffers) == 107 and len(new) == 12
    assert logical["scope"] == "prospective_teacher_development" and "local_execution_contract_sha256" not in logical
    assert logical["training"]["horizon"] == 100 and logical["training"]["groups_per_update"] == 1
    assert logical["train_seed"] == 73
    assert not ({"torch", "transformers", "trl", "peft", "accelerate"} & (set(sys.modules) - before))


@pytest.mark.parametrize("fault", ["extra_python", "pyc", "pyo", "changed_source"])
def test_closed_extra_snapshot_rejects_drift_before_quality(declared, monkeypatch, fault):
    contract, extras = declared
    monkeypatch.setattr(runner, "__file__", str(extras / PATH.name))
    monkeypatch.setattr(runner, "__executed_source_sha256__", contract["extra_sources"][PATH.name]["sha256"], raising=False)
    if fault == "extra_python": (extras / "fallback.py").write_text("raise RuntimeError('unbound')")
    elif fault in ("pyc", "pyo"): (extras / ("cached." + fault)).write_bytes(b"not admitted")
    else: (extras / "teacher_candidate_v2.py").write_bytes(b"changed")
    with pytest.raises(ValueError): runner.source_context(contract)


def test_prior_gate_replays_only_exact_analyzer_with_GPU_hidden(declared, tmp_path):
    contract, _ = declared; output = tmp_path / "gate"; output.mkdir()
    report = {"protocol": "independent-one-shot-development-replay-v1", "status": "complete_task_raw_replay",
              "task": "op", "quality_analyzed": True, "development_screen_passed": False,
              "superiority_established": False, "roster": contract["roster"], "synthetic_fixture": True}
    path = tmp_path / "prior_fixture.json"; path.write_text(json.dumps(report)); contract["prior_failed_screen"] = runner.binding(path)
    calls = []
    def process(argv, **kwargs):
        calls.append(argv)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "" and kwargs["timeout"] == 25
        assert argv[5] == contract["old_analyzer"]["path"] and argv[6] == runner.ANALYZER_SHA
        Path(argv[argv.index("--output") + 1]).write_text(json.dumps(report))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    binding = runner.prior_failed_screen(contract, output, 25, run=process)
    assert len(calls) == 1 and binding["path"] == str(output / "prior_screen_replay.json")


@pytest.mark.parametrize("fault", ["incomplete", "successful_screen", "nonzero_exit", "different_result"])
def test_condition_failure_stops_before_supplied_quality_or_worker(declared, tmp_path, monkeypatch, fault):
    contract, _ = declared; output = tmp_path / "gate"; output.mkdir()
    report = {"status": "complete_task_raw_replay", "task": "op", "quality_analyzed": True,
              "development_screen_passed": False, "superiority_established": False, "roster": contract["roster"]}
    if fault == "incomplete": report["status"] = "incomplete_task_no_quality_analysis"
    if fault == "successful_screen": report["development_screen_passed"] = True
    reads = []
    def read(row):
        reads.append(row["path"])
        return b'{"different": true}'
    monkeypatch.setattr(runner, "read_bound", read)
    def process(argv, **kwargs):
        Path(argv[argv.index("--output") + 1]).write_text(json.dumps(report))
        return types.SimpleNamespace(returncode=1 if fault == "nonzero_exit" else 0, stdout="", stderr="")
    with pytest.raises(ValueError): runner.prior_failed_screen(contract, output, 25, run=process)
    assert len(reads) == (1 if fault == "different_result" else 0)


def test_bound_raw_source_loader_rejects_uncaptured_transitive_bytes(tmp_path):
    raw = b"COLLECTOR_SHA256='collector'\nMETADATA_HELPER_SHA256='" + runner.METADATA_SHA.encode() + b"'\ndef compile_buffer(raw,path,name):\n return raw\n"
    path = tmp_path / "admit_training_collection_v1.py"
    rows = {"admit_training_collection_v1.py": {"path": str(path)},
            "collect_training_sft_v1.py": {"path": str(tmp_path / "collect_training_sft_v1.py"), "sha256": "collector"}}
    project = str(tmp_path / "frozen_project.py")
    loader = runner.ClosedHelpers({"extra_sources": rows}, {"admit_training_collection_v1.py": raw,
        "collect_training_sft_v1.py": b"closed collector"}, {project: b"closed scorer"})
    module = loader.load("admit_training_collection_v1.py")
    assert module.compile_buffer(b"closed scorer", project, "fixture") == b"closed scorer"
    with pytest.raises(ValueError, match="escaped"):
        module.compile_buffer(b"changed scorer", project, "fixture")
    with pytest.raises(ValueError, match="escaped"):
        module.compile_buffer(b"closed scorer", str(tmp_path / "unbound.py"), "fixture")


def test_write_once_and_typed_byte_binding_preserve_failure_artifacts(tmp_path):
    path = tmp_path / "attempt.json"; first = runner.write_once(path, {"attempt": 1})
    with pytest.raises(FileExistsError): runner.write_once(path, {"attempt": 2})
    assert json.loads(runner.read_bound(first)) == {"attempt": 1}
    path.write_bytes(b"changed")
    with pytest.raises(ValueError): runner.read_bound(first)
    linked = tmp_path / "linked"; linked.symlink_to(path)
    with pytest.raises(ValueError): runner.binding(linked)


def test_actual_bounded_stdlib_child_is_killed_and_reaped_without_models(tmp_path):
    pid_path = tmp_path / "child.pid"
    program = "import os,pathlib,time;pathlib.Path(" + repr(str(pid_path)) + ").write_text(str(os.getpid()));time.sleep(5)"
    with (tmp_path / "child.log").open("xb") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            runner.run_bounded_child([sys.executable, "-I", "-B", "-c", program], log, .3)
    pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError): os.kill(pid, 0)


@pytest.fixture
def shaped_native_result(declared, tmp_path):
    """Metadata-only structural fixture, never an actual7B qualification."""
    contract, _ = declared; output = Path(contract["output"]); output.mkdir()
    own_binding = {"path": str(tmp_path / "contract.json"), "sha256": "3" * 64, "size_bytes": 1}
    logical = json.loads(runner.read_bound(contract["logical_contract"]))
    adapter = {"source_binding": contract["extra_sources"]["run_teacher_calibration_7b_v2.py"],
               "transformation": runner.MEMORY_TRANSFORM,
               "frozen_calibration_source_sha256": contract["extra_sources"]["calibrate_teacher_v2.py"]["sha256"]}
    identity = {"shape_only_not_actual_GPU_measurement": True}
    inventory = [{"name": "shape_only.lora_A.weight", "shape": [1, 1], "dtype": "torch.float32", "numel": 1}]
    dtype = {"parameter_counts_by_dtype": {"torch.bfloat16": 7_000_000_000, "torch.float32": 100},
             "trainable_parameters": 100, "actual_BF16_base_weights_verified": True, "devices": ["cuda:0"]}
    reference = copy.deepcopy(dtype); reference["trainable_parameters"] = 0
    loaded = {"runtime": runner.PACKAGES, "actual_loaded_context": {"whole_actor_sha256": "7" * 64,
        "actor_dtype": dtype, "reference_dtype": reference}, "actual_optimizer_inventory": inventory,
        "calibration_actor_identity": identity}
    model_binding = runner.write_once(output / "model_load_admission.json", loaded)
    records = [{"instance_id": "shape_" + str(i), "full_window_gradient_equality_required": False,
        "restored_active_CUDA_RNG_devices": [0], "native_RL_microbatches": 9, "optimizer_updates": 0,
        "whole_actor_sha256": "7" * 64, "whole_reference_sha256": "7" * 64,
        "actor_reference_optimizer_scheduler_unchanged": True, "model_and_generation_config_unchanged": True,
        "mode_and_RNG_restored": True, "all_grads_cleared": True, "captured_before_clipping_and_optimizer": True,
        "operational_memory_adapter": {"identity": adapter, "forward_wrappers_restored": True,
            "native_CPU_snapshots": 1, "independent_calls": 1, "native_gradient_bytes_released": 123, "CPU_accumulator_bytes": 246}}
        for i in range(8)]
    alphas = {}
    for recipe in runner.RECIPES:
        environment = {"GRPO_GRADED_REWARD": "1" if recipe == "rloo_cost" else "0", "GRPO_VERIFIED_IMPROVEMENT": "0"}
        recipe_records = copy.deepcopy(records)
        for record in recipe_records: record["reward_environment"] = dict(environment)
        value = {"model_kind": "actual_7b_native_gradient_calibration", "scope": "production_training_collection_7b",
            "recipe": recipe, "core_contract_sha256": logical["contract_sha256"], "optimizer_updates": 0,
            "source_sha256": contract["extra_sources"]["calibrate_teacher_v2.py"]["sha256"],
            "data_source_sha256": contract["extra_sources"]["teacher_candidate_v2.py"]["sha256"],
            "records": recipe_records, "parameter_inventory": inventory, "calibration_actor_identity": identity,
            "operational_memory_adapter": adapter, "recipe_reward_environment": environment}
        value["calibration_sha256"] = runner.sha(runner.encoded(value))
        alphas[recipe] = runner.write_once(output / (recipe + ".calibration.json"), value)
    result = {"status": "complete_actual_7b_native_calibration", "model_kind": "actual_7b_native_gradient_calibration",
        "contract": contract, "contract_file_binding": own_binding, "extra_sources": contract["extra_sources"],
        "shared_data_binding": contract["shared_data"], "targets_binding": contract["teacher_targets"],
        "optimizer_updates": 0, "new_generation_calls": 0, "heldout_policy_calls": 0,
        "original_actor_and_reference_unchanged": True, "reference_integrity": True, "initial_actor_sha256": "7" * 64,
        "fixed_training_prompt_count": 8, "captured_recipes": list(runner.RECIPES), "native_microbatches_per_prompt": 9,
        "publication_eligible": False, "superiority_established": False, "calibrations": alphas,
        "model_load_admission_binding": model_binding,
        "operational_memory_adapter": adapter,
        "resources": {"physical_gpus": 1, "gpu_uuid": runner.GPU_UUID,
            "worker_wall_seconds_including_input_hashing": 25., "peak_allocated_bytes": 123, "peak_reserved_bytes": 456}}
    runner.write_once(output / "result.json", result)
    return contract, own_binding, output, result


def test_parent_checks_complete_bytebound_structural_result_without_GPU(shaped_native_result):
    contract, own, output, value = shaped_native_result
    assert runner.verify_result(contract, own, output) == value


@pytest.mark.parametrize("fault", ["tiny_alpha", "cpu_reference", "lost_group", "changed_alpha", "lost_model_binding", "fallback_source", "optimizer_step"])
def test_parent_rejects_cpu_alpha_missing_proofs_and_changed_bytes(shaped_native_result, fault):
    contract, own, output, result = shaped_native_result
    recipe = runner.RECIPES[0]; path = Path(result["calibrations"][recipe]["path"])
    value = json.loads(path.read_text())
    if fault in ("tiny_alpha", "cpu_reference", "lost_group"):
        if fault == "tiny_alpha": value["model_kind"] = "actual_tiny_cpu_only"
        elif fault == "cpu_reference": value["records"][0]["full_window_gradient_equality_required"] = True
        else: value["records"].pop()
        value["calibration_sha256"] = runner.sha(runner.encoded({k: v for k, v in value.items() if k != "calibration_sha256"}))
        path.write_bytes(runner.encoded(value)); result["calibrations"][recipe] = runner.binding(path)
    elif fault == "changed_alpha": path.write_bytes(b"changed")
    elif fault == "lost_model_binding": result.pop("model_load_admission_binding")
    elif fault == "fallback_source": result["extra_sources"] = {}
    else: result["optimizer_updates"] = 1
    (output / "result.json").write_bytes(runner.encoded(result))
    with pytest.raises(ValueError): runner.verify_result(contract, own, output)


def reference_gradient_functions(pairs):
    """Compile only two pure tensor functions from exact frozen A5 bytes."""
    import ast
    import torch
    path = HERE / "calibrate_teacher_v2.py"; raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "a5ea942619ac6187327239dc9d9249e79c7ee47aaff14834a6a7edfefda4210f"
    nodes = [node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef) and
             node.name in {"gradient_vector", "independent_micro_policy_gradient"}]
    scope = {"torch": torch, "need": runner.need, "parameter_pairs": lambda model, inventory: pairs}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def tensor_trainer(dtype, loss_type, *, nonzero_kl=False):
    import torch
    pairs = [("lora_A", torch.tensor([.12, -.08], dtype=torch.float32, requires_grad=True)),
             ("bias", torch.tensor([.04, .07], dtype=dtype, requires_grad=True)),
             ("unused_bias", torch.tensor([.02], dtype=dtype, requires_grad=True))]
    batch = {"prompt_ids": torch.arange(9).reshape(9, 1), "completion_ids": torch.ones(9, 3, dtype=torch.long),
             "prompt_mask": torch.ones(9, 1, dtype=torch.long),
             "completion_mask": torch.tensor([[1, 1, 1] if i % 2 else [1, 1, 0] for i in range(9)]),
             "advantages": torch.tensor([-2., -.5, .8, 1.4, .2, -.7, .4, 1.1, -.3])}
    def logps(model, ids, mask, keep, batch_size):
        a, b = pairs[0][1], pairs[1][1]
        x = ids[:, 0].float().unsqueeze(1) / 10 + 1
        position = torch.arange(keep).float().unsqueeze(0) + 1
        value = (a[0]*x + a[1]*position + b.float()[0]*x.square()/4 + b.float()[1]*position.square()/10).tanh()*.1
        return value, None
    trainer = types.SimpleNamespace(calibration_batch=batch, model=object(), loss_type=loss_type,
        max_completion_length=3000, beta=.05, _get_per_token_logps_and_entropies=logps)
    ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], 1)
    batch["ref_per_token_logps"] = logps(None, ids, None, 3, 1)[0].detach() + (.03 if nonzero_kl else 0.)
    return pairs, trainer


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("loss_type", ["dr_grpo", "grpo"])
@pytest.mark.parametrize("nonzero_kl", [False, True])
def test_streamed_CPU_native_dtype_sum_is_bitwise_equal_to_frozen_reference(dtype_name, loss_type, nonzero_kl):
    import torch
    dtype = getattr(torch, dtype_name); pairs, trainer = tensor_trainer(dtype, loss_type, nonzero_kl=nonzero_kl)
    reference = reference_gradient_functions(pairs)
    before_rng = torch.get_rng_state().clone(); before = [param.detach().clone() for _, param in pairs]
    expected = reference["independent_micro_policy_gradient"](trainer, [])
    cal = types.SimpleNamespace(parameter_pairs=lambda model, inventory: pairs, gradient_vector=reference["gradient_vector"])
    audit = {"independent_calls": 0, "CPU_accumulator_bytes": 0}
    actual = runner.streamed_independent_micro_gradients(trainer, [], cal, audit, diagnostic_cpu=True)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert all(torch.equal(param, saved) and param.grad is None for (_, param), saved in zip(pairs, before))
    assert torch.equal(torch.get_rng_state(), before_rng) and audit["independent_calls"] == 1


def test_streaming_releases_policy_gradient_tuple_before_KL(monkeypatch):
    import torch
    import weakref
    pairs, trainer = tensor_trainer(torch.bfloat16, "grpo", nonzero_kl=True)
    reference = reference_gradient_functions(pairs)
    cal = types.SimpleNamespace(parameter_pairs=lambda model, inventory: pairs, gradient_vector=reference["gradient_vector"])
    original = torch.autograd.grad; policy_refs = []; calls = 0
    def observed(*args, **kwargs):
        nonlocal calls, policy_refs
        if calls % 2:
            assert all(ref() is None for ref in policy_refs)
        values = original(*args, **kwargs)
        if calls % 2 == 0:
            policy_refs = [weakref.ref(value) for value in values if value is not None]
        calls += 1
        return values
    monkeypatch.setattr(torch.autograd, "grad", observed)
    runner.streamed_independent_micro_gradients(trainer, [], cal,
        {"independent_calls": 0, "CPU_accumulator_bytes": 0}, diagnostic_cpu=True)
    assert calls == 18


def test_native_snapshot_CPU_copy_then_clear_preserves_values_and_denies_unqualified_CPU():
    import torch
    pairs, _ = tensor_trainer(torch.bfloat16, "grpo")
    reference = reference_gradient_functions(pairs)
    for _, param in pairs: param.grad = torch.ones_like(param)
    expected = reference["gradient_vector"](pairs)
    cal = types.SimpleNamespace(gradient_vector=reference["gradient_vector"])
    runner.install_streamed_reference_adapter(cal)
    with pytest.raises(ValueError, match="CUDA"): cal.gradient_vector(pairs)
    assert all(param.grad is not None for _, param in pairs)
    cal = types.SimpleNamespace(gradient_vector=reference["gradient_vector"])
    audit = runner.install_streamed_reference_adapter(cal, diagnostic_cpu=True)
    assert torch.equal(cal.gradient_vector(pairs), expected) and all(param.grad is None for _, param in pairs)
    assert audit["native_CPU_snapshots"] == 1 and audit["native_gradient_bytes_released"] > 0


def test_independent_loss_expressions_match_frozen_A5_AST():
    import ast
    old = ast.parse((HERE / "calibrate_teacher_v2.py").read_bytes())
    new = ast.parse(PATH.read_bytes())
    old_fn = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "independent_micro_policy_gradient")
    new_fn = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "streamed_independent_micro_gradients")
    def expressions(fn):
        return {node.targets[0].id: ast.dump(node.value, include_attributes=False) for node in ast.walk(fn)
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and
                node.targets[0].id in {"token_mask", "weighted", "policy", "delta", "k3", "kl"}}
    assert expressions(old_fn) == expressions(new_fn)


@pytest.mark.parametrize("fail", [False, True])
def test_forward_wrappers_restore_exact_instance_state_across24_prompts(fail):
    class Model:
        def __init__(self, base=None): self.base = base or self
        def get_base_model(self): return self.base
        def forward(self): return 1
    actor = Model(Model()); reference = Model(Model())
    old = object(); reference._original_forward = old
    original = [dict(model.__dict__) for model in (actor, reference, actor.base, reference.base)]
    for _ in range(24):
        proof = None
        try:
            with runner.preserve_forward_wrappers(actor, reference) as proof:
                for model in (actor, reference, actor.base, reference.base):
                    model._original_forward = model.forward
                    model.forward = lambda: 99
                    model._is_accelerate_prepared = True
                if fail: raise RuntimeError("diagnostic capture failure")
        except RuntimeError:
            assert fail
        assert proof["restored"] is True
        assert [model.__dict__ for model in (actor, reference, actor.base, reference.base)] == original
        assert actor.forward() == reference.forward() == 1


@pytest.mark.parametrize("fault", ["missing_provenance", "changed_adapter_source", "bool_snapshot_count"])
def test_parent_requires_truthful_source_bound_memory_override(shaped_native_result, fault):
    contract, own, output, result = shaped_native_result
    if fault == "missing_provenance":
        result.pop("operational_memory_adapter")
    else:
        recipe = runner.RECIPES[0]; path = Path(result["calibrations"][recipe]["path"])
        value = json.loads(path.read_text())
        if fault == "changed_adapter_source":
            value["operational_memory_adapter"]["source_binding"]["sha256"] = "0" * 64
        else:
            value["records"][0]["operational_memory_adapter"]["native_CPU_snapshots"] = True
        value["calibration_sha256"] = runner.sha(runner.encoded({k: v for k, v in value.items() if k != "calibration_sha256"}))
        path.write_bytes(runner.encoded(value)); result["calibrations"][recipe] = runner.binding(path)
    (output / "result.json").write_bytes(runner.encoded(result))
    with pytest.raises(ValueError): runner.verify_result(contract, own, output)


@pytest.mark.parametrize("recipe", runner.RECIPES)
def test_calibration_installs_the_same_frozen_per_method_environment_as_factory(monkeypatch, recipe):
    import ast
    source = HERE / "execution_v1/source/src/clasp/baselines/external_baseline_dev.py"
    node = next(node for node in ast.parse(source.read_bytes()).body if isinstance(node, ast.Assign) and
                any(isinstance(target, ast.Name) and target.id == "REWARD_ENVIRONMENT" for target in node.targets))
    graded = ast.literal_eval(node.value)
    strict = {**graded, "GRPO_GRADED_REWARD": "0"}
    calls = []
    def environment(method):
        calls.append(method)
        return dict(graded if method == "rloo_quality" else strict)
    core = types.SimpleNamespace(RECIPES={"rloo_cost": {"method": "rloo"}, "foarl_released": {"method": "grpo"},
                                          "foarl_cost_changed": {"method": "grpo"}}, reward_environment=environment)
    for key in graded: monkeypatch.setenv(key, "inherited_drift")
    actual = runner.install_recipe_reward_environment(core, recipe)
    expected = graded if recipe == "rloo_cost" else strict
    assert actual == expected and all(os.environ[key] == value for key, value in expected.items())
    assert calls == ["rloo_quality" if recipe == "rloo_cost" else "foarl"]
