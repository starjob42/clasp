"""Exercise adaptive orchestration with CPU generation doubles and real selection."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/rloo_decoder_search_20261014/worker.py"
MODULE_SPEC = importlib.util.spec_from_file_location("decoder_search_worker", SOURCE)
worker = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(worker)


def plan_at(tmp_path):
    datasets = {}
    development = [{"instance_id": f"dev-{i}", "semantic_id": f"dev-semantic-{i}"}
                   for i in range(4)]
    for split, rows in (("screen", development[:2]), ("coarse", development[:3]),
                        ("dev", development),
                        ("test", [{"instance_id": f"test-{i}", "semantic_id": f"test-semantic-{i}"}
                                  for i in range(4)])):
        path = tmp_path / "data" / (split + ".json")
        worker.write(path, rows)
        datasets[split] = {"relative_path": str(path.relative_to(tmp_path)), "sha256": worker.sha(path)}
    models = {}
    for method in ("rloo", "foarl"):
        models[method] = {}
        for checkpoint in ("a", "b", "c"):
            directory = tmp_path / method / checkpoint
            directory.mkdir(parents=True)
            weights = directory / "adapter_model.safetensors"
            weights.write_text(f"test-double {method} {checkpoint}")
            models[method][checkpoint] = {"snapshot_path": str(directory),
                                           "files": {weights.name: worker.sha(weights)}}
    base_path = tmp_path / "base"
    base_path.mkdir()
    worker.write(base_path / "config.json", {"test_double": True})
    worker.write(base_path / "tokenizer.json", {"test_double": True})
    source_paths = [SOURCE, SOURCE.parent / "evaluate.py", SOURCE.parent / "analyze.py",
                    ROOT / "src/clasp/baselines/sft_anchor_pilot.py"]
    plan = {"native_root": str(tmp_path), "datasets": {"cvrp": datasets},
            "models": {"cvrp": models}, "base": {"path": str(base_path), "files": {"config.json": worker.sha(base_path / "config.json")}},
            "tokenizer_files": {"tokenizer.json": worker.sha(base_path / "tokenizer.json")}, "dtype": "bfloat16", "max_prompt_tokens": 4096,
            "source_root": str(ROOT), "source_files": {str(path.relative_to(ROOT)): worker.sha(path) for path in source_paths},
            "development_sampling_seeds": [11], "test_sampling_seeds": [21, 22, 23],
            "top_p_grid": [.7, .9, .95, 1.], "top_k_grid": [0, 20, 50],
            "temperature_refinement": [.4, .55, .7, .85, 1.],
            "comparison_scope": "fixed selected checkpoints and declared decoders"}
    plan["plan_sha256"] = worker.digest(plan)
    return plan


def synthetic_result(binding, config, gap, analysis):
    configuration = worker.effective(config)
    material = {"schema_version": 1, "phase": binding["phase"], "task": binding["task"],
                "method": binding["method"], "checkpoint_id": binding["checkpoint_id"],
                "config_id": config["config_id"], "configuration": configuration,
                "dataset_sha256": binding["dataset_sha256"], "sampling_seeds": binding["sampling_seeds"],
                "plan_sha256": binding["plan_sha256"], "binding_sha256": binding["binding_sha256"],
                "engine_sha256": worker.sha(SOURCE.parent / "evaluate.py"),
                "scorer": {"path": str(ROOT / "src/clasp/baselines/sft_anchor_pilot.py"),
                           "sha256": worker.sha(ROOT / "src/clasp/baselines/sft_anchor_pilot.py")},
                "cohort_sha256": "f" * 64, "backend": {"backend": "test-double"},
                "utility_definition": "F*sigmoid(-8*signed_gap_fraction)",
                "u10_definition": "F*max(0,1-max(0,signed_gap_fraction)/0.1)",
                "seed_policy": "SHA256(replicate_seed,task,semantic_id); excludes method/config",
                "selection": "strict feasible recomputed objective; candidate-index tie-break"}
    fingerprint = worker.digest(material)
    rows = []
    for instance in json.loads(Path(binding["dataset_path"]).read_text()):
        for seed in binding["sampling_seeds"]:
            row = {**instance, "replicate_seed": seed, "feasible": True,
                   "gap_signed_percent": gap, "gap_clipped_percent": max(0, gap),
                   "utility": analysis.quality(gap, True),
                   "u10": max(0, 1 - max(0, gap) / 10), "candidate_count": 8,
                   "candidates": [{} for _ in range(8)], "evaluation_fingerprint": fingerprint}
            row["row_sha256"] = worker.digest(row)
            rows.append(row)
    return {**material, "evaluation_fingerprint": fingerprint, "rows": rows,
            "aggregate": {}, "expected_rows": len(rows), "complete": True, "stop_reason": "complete"}


def scores(binding, config):
    stage = Path(binding["output_dir"]).parents[1].name
    method, checkpoint = binding["method"], binding["checkpoint_id"]
    p, k, temp = config["top_p"], config["top_k"], config["temperature"]
    if stage == "01_checkpoint_screen":
        # The methods have different best checkpoints.
        best = "b" if method == "rloo" else "c"
        return 1 if checkpoint == best else 5
    if stage == "02_decoder_grid":
        return .5 if (p, k) == (.95, 20) else 1 if (p, k) == (1., 50) else 5
    if stage == "03_temperature_refinement":
        target = (.55, .95, 20) if method == "rloo" else (.85, 1., 50)
        return .1 if (temp, p, k) == target else 2
    # The shared baseline has a different method ranking from tuned winners.
    if (temp, p, k) == (.7, .7, 0):
        return 50 if method == "rloo" else 10
    return 1 if method == "rloo" else 2


class FakeEngine:
    def __init__(self, search, *, interrupt_stage=None):
        self.search, self.calls, self.interrupt_stage = search, [], interrupt_stage

    def __call__(self, argv, check):
        assert check is False
        assert "--resume" in argv
        binding = json.loads(Path(argv[argv.index("--binding") + 1]).read_text())
        stage = Path(binding["output_dir"]).parents[1].name
        self.calls.append((stage, binding))
        if binding["phase"] == "test":
            frozen_path = self.search.control / "frozen_selection.json"
            assert frozen_path.is_file(), "Test generation must follow durable selection"
            frozen = json.loads(frozen_path.read_text())
            snapshot = dict(frozen)
            seal = snapshot.pop("frozen_selection_sha256")
            assert seal == worker.digest(snapshot)
            allowed = {(row["method"], row["checkpoint_id"], row["config_id"])
                       for row in [*frozen["selected"].values(), *frozen["diagnostic_candidates"]]}
            assert all((binding["method"], binding["checkpoint_id"], row["config_id"]) in allowed
                       for row in binding["configurations"])
        if stage == self.interrupt_stage:
            return SimpleNamespace(returncode=75)
        directory = Path(binding["output_dir"])
        for config in binding["configurations"]:
            result = synthetic_result(binding, config, scores(binding, config), self.search.analysis)
            worker.write(directory / (config["config_id"] + ".json"), result)
            material = {key: value for key, value in result.items()
                        if key not in ("evaluation_fingerprint", "rows", "aggregate", "expected_rows", "complete", "stop_reason")}
            worker.write(directory / (config["config_id"] + ".binding.json"),
                         {**material, "evaluation_fingerprint": result["evaluation_fingerprint"]})
        return SimpleNamespace(returncode=0)


def make_search(tmp_path):
    plan = plan_at(tmp_path)
    search = worker.Search(plan, "cvrp", 10000)
    search.control.mkdir(parents=True)
    return search


def test_stages_carry_method_specific_checkpoints_and_only_frozen_test_candidates(tmp_path, monkeypatch):
    search = make_search(tmp_path)
    engine = FakeEngine(search)
    monkeypatch.setattr(worker.subprocess, "run", engine)
    search.run()
    stages = [stage for stage, _ in engine.calls]
    assert stages == ["01_checkpoint_screen"] * 6 + ["02_decoder_grid"] * 2 + [
        "03_temperature_refinement"] * 2 + ["04_frozen_test"] * 2
    for stage, binding in engine.calls:
        if stage != "01_checkpoint_screen":
            assert binding["checkpoint_id"] == ("b" if binding["method"] == "rloo" else "c")
        if stage == "02_decoder_grid":
            assert len(binding["configurations"]) == 12
            assert binding["configurations"][0] == search.baseline
        if stage == "03_temperature_refinement":
            effective = {(row["temperature"], row["top_p"], row["top_k"]) for row in binding["configurations"]}
            assert effective == {(.7, .9, 50)} | {
                (temperature, p, k) for temperature in search.plan["temperature_refinement"]
                for p, k in ((.95, 20), (1., 50))}
    frozen = json.loads((search.control / "frozen_selection.json").read_text())
    assert frozen["selected"]["rloo"]["configuration"]["temperature"] == .55
    assert frozen["selected"]["foarl"]["configuration"]["temperature"] == .85
    result = json.loads((search.control / "final_result.json").read_text())
    assert result["paired_contrast"]["paired_signed_gap_difference_percent"] == -1
    assert result["diagnostic_comparisons"][0]["contrast"]["paired_signed_gap_difference_percent"] == 40
    assert not result["absolute_optimum_established"]
    assert not result["supports_training_algorithm_superiority"]


def test_allocation_interruption_cannot_freeze_or_inspect_test_and_restart_reuses_screen(tmp_path, monkeypatch):
    search = make_search(tmp_path)
    engine = FakeEngine(search, interrupt_stage="02_decoder_grid")
    monkeypatch.setattr(worker.subprocess, "run", engine)
    with pytest.raises(TimeoutError, match="partial progress"):
        search.run()
    assert not (search.control / "frozen_selection.json").exists()
    assert not (search.control / "04_frozen_test").exists()
    resumed_engine = FakeEngine(search)
    monkeypatch.setattr(worker.subprocess, "run", resumed_engine)
    search.run()
    assert all(stage != "01_checkpoint_screen" for stage, _ in resumed_engine.calls)
    assert (search.control / "final_result.json").is_file()


def test_resume_keeps_results_and_selection_immutable_and_avoids_new_inference(tmp_path, monkeypatch):
    search = make_search(tmp_path)
    engine = FakeEngine(search)
    monkeypatch.setattr(worker.subprocess, "run", engine)
    search.run()
    before = {str(path.relative_to(search.control)): worker.sha(path)
              for path in search.control.rglob("*.json")}
    engine.calls.clear()
    search.run()
    assert engine.calls == []
    after = {str(path.relative_to(search.control)): worker.sha(path)
             for path in search.control.rglob("*.json")}
    assert before == after


def test_modified_dataset_is_rejected_before_generation(tmp_path, monkeypatch):
    search = make_search(tmp_path)
    engine = FakeEngine(search)
    monkeypatch.setattr(worker.subprocess, "run", engine)
    binding = search.plan["datasets"]["cvrp"]["screen"]
    (tmp_path / binding["relative_path"]).write_text("[]\n")
    with pytest.raises(ValueError, match="Dataset content changed"):
        search.run()
    assert engine.calls == []


def test_existing_stage_binding_cannot_be_repurposed_for_another_checkpoint(tmp_path, monkeypatch):
    search = make_search(tmp_path)
    engine = FakeEngine(search)
    monkeypatch.setattr(worker.subprocess, "run", engine)
    search.stage("screen_only", "screen", [("rloo", "a", [search.baseline])], seeds=[11])
    search.plan["models"]["cvrp"]["rloo"]["a"]["snapshot_path"] = str(tmp_path / "different")
    with pytest.raises(ValueError, match="Refusing to change frozen artifact"):
        search.stage("screen_only", "screen", [("rloo", "a", [search.baseline])], seeds=[11])


@pytest.mark.parametrize("damage", ["another_binding", "row_changed", "missing_header", "changed_header",
                                     "wrong_denominator", "engine_changed", "scorer_changed"])
def test_complete_cached_outputs_must_retain_their_engine_and_row_seals(tmp_path, monkeypatch, damage):
    search = make_search(tmp_path)
    engine = FakeEngine(search)
    monkeypatch.setattr(worker.subprocess, "run", engine)
    models = [("rloo", "a", [search.baseline])]
    search.stage("cached", "screen", models, seeds=[11])
    directory = search.control / "cached" / "rloo" / "a"
    result_path = directory / (search.baseline["config_id"] + ".json")
    header_path = directory / (search.baseline["config_id"] + ".binding.json")
    result = json.loads(result_path.read_text())
    if damage == "another_binding":
        result["binding_sha256"] = "a" * 64
    elif damage == "row_changed":
        result["rows"][0]["gap_signed_percent"] += 1
    elif damage == "missing_header":
        header_path.unlink()
    elif damage == "changed_header":
        header_path.write_text("{}\n")
    elif damage == "wrong_denominator":
        result["expected_rows"] += 1
    elif damage == "engine_changed":
        result["engine_sha256"] = "b" * 64
    elif damage == "scorer_changed":
        result["scorer"]["sha256"] = "c" * 64
    result_path.write_text(json.dumps(result))
    engine.calls.clear()
    with pytest.raises(ValueError, match="Cached"):
        search.stage("cached", "screen", models, seeds=[11])
    assert engine.calls == []


def test_complete_cache_requires_unchanged_snapshot_files_even_without_inference(tmp_path, monkeypatch):
    search = make_search(tmp_path)
    engine = FakeEngine(search)
    monkeypatch.setattr(worker.subprocess, "run", engine)
    models = [("rloo", "a", [search.baseline])]
    search.stage("cached", "screen", models, seeds=[11])
    model = search.plan["models"]["cvrp"]["rloo"]["a"]
    (Path(model["snapshot_path"]) / "adapter_model.safetensors").write_text("different weights")
    engine.calls.clear()
    with pytest.raises(ValueError, match="Cached artifact identity mismatch"):
        search.stage("cached", "screen", models, seeds=[11])
    assert engine.calls == []
