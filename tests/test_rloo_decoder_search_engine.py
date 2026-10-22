"""CPU-only controls for fresh sampling, strict scoring and crash-safe resume."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments/rloo_decoder_search_20261014/evaluate.py"
SPEC = importlib.util.spec_from_file_location("rloo_decoder_search_engine", PATH)
engine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(engine)


def config(identifier="T070_P090_K050", **overrides):
    return {"config_id": identifier, "temperature": 0.7, "top_p": 0.9, "top_k": 50,
            "num_return_sequences": 8, "max_new_tokens": 3000, **overrides}


def example(task="tsp", identifier="a"):
    instances = {
        "tsp": [[0, 0], [3, 0], [3, 4], [0, 4]],
        "cvrp": [[[0, 0], [1, 0], [2, 0]], [0, 1, 1], 2],
        "op": [[[0, 0], [3, 4], [0, 8]], [0, 9, 4], 13],
        "mis": {"num_nodes": 4, "edges": [[0, 1], [1, 2]]},
        "jssp": [[0, 3, 1, 2], [1, 4, 0, 1]],
    }
    references = {"tsp": "Route: [0,1,3,2,0], Objective: -9999",
                  "cvrp": "Routes: [[0,1,0],[0,2,0]], Objective: -9999",
                  "op": "Route: [0,1], Objective: -9999",
                  "mis": "Set: [0,2], Objective: -9999",
                  "jssp": "Schedule: [[0,1],[0,1]], Makespan: -9999"}
    return {"instance_id": identifier, "semantic_id": "semantic-" + identifier,
            "instruction": "solve " + task, "input": "numeric instance " + identifier,
            "instance_json": instances[task], "output": references[task]}


class MockBackend:
    metadata = {"backend": "mock", "synthetic_fixture": True}

    def __init__(self, texts=None):
        self.calls = []
        self.texts = texts or ["Route: [0,1,2,3,0]"] * 8

    def render_prompt(self, row):
        return row["instruction"] + "\n" + row["input"]

    def generate_candidates(self, prompt, configuration, seed):
        self.calls.append((prompt, configuration, seed))
        return ([{"raw_text": text, "completion_tokens": 5} for text in self.texts],
                {"prompt_tokens": 10, "generation_time_sec": 0.01})


def evaluate(tmp_path, backend=None, **overrides):
    return engine.evaluate_config(
        **{"task": "tsp", "method": "rloo", "checkpoint_id": "frozen-checkpoint",
           "configuration": config(), "examples": [example()], "sampling_seeds": [42],
           "backend": backend or MockBackend(), "phase": "development", "dataset_sha256": "a" * 64,
           "output_dir": tmp_path, "plan_sha256": "b" * 64, "binding_sha256": "c" * 64,
           **overrides})


@pytest.mark.parametrize("task,text,objective,reference,gap", [
    ("tsp", "Route: [0,1,2,3,0], Objective: 9999", 14, 16, -12.5),
    ("cvrp", "Routes: [[0,1,2,0]], Objective: 9999", 4, 6, -100 / 3),
    ("op", "Route: [0,1,2], Objective: 9999", 13, 9, -400 / 9),
    ("mis", "Set: [0,2,3], Objective: 9999", 3, 2, -50),
    ("jssp", "Schedule: [[0,1],[1,0]], Makespan: 9999", 6, 10, -40),
])
def test_all_tasks_recompute_reference_and_keep_negative_signed_gap(task, text, objective,
                                                                  reference, gap):
    scorer = engine.StrictScorer(task)
    prepared = scorer.prepare(example(task))
    score = scorer.score(text, prepared)
    assert score["feasible"] is True
    assert score["objective"] == objective
    assert score["reference_objective"] == reference
    assert score["gap_signed_percent"] == pytest.approx(gap)
    assert score["gap_clipped_percent"] == 0
    assert score["utility"] > 0.5
    assert score["u10"] == 1


@pytest.mark.parametrize("task,text", [
    ("tsp", "Route: [0,True,2,3,0]"),
    ("cvrp", "Routes: [[0,1.0,2,0]]"),
    ("op", "Route: [0,1,1]"),
    ("mis", "Set: [0,1]"),
    ("jssp", "Schedule: [[False,1],[1,0]]"),
])
def test_strict_infeasibility_has_no_gap_and_zero_utility(task, text):
    scorer = engine.StrictScorer(task)
    score = scorer.score(text, scorer.prepare(example(task)))
    assert score["feasible"] is False
    assert score["objective"] is None
    assert score["gap_signed_percent"] is None
    assert score["gap_clipped_percent"] is None
    assert score["utility"] == 0
    assert score["u10"] == 0


def test_infeasible_reference_aborts_before_any_generation(tmp_path):
    row = {**example(), "output": "Route: [0,1,1,3,0], Objective: 14"}
    backend = MockBackend()
    with pytest.raises(ValueError, match="Reference witness"):
        evaluate(tmp_path, backend, examples=[row])
    assert not backend.calls


def test_best8_uses_physical_objective_and_stable_tie_break(tmp_path):
    backend = MockBackend(["Route: [0,1,1,3,0], Objective: 0",
                           "Route: [0,1,3,2,0], Objective: 0",
                           "Route: [0,1,2,3,0], Objective: 9999",
                           "Route: [0,1,2,3,0], Objective: 0"] + ["invalid"] * 4)
    result = evaluate(tmp_path, backend)
    row = result["rows"][0]
    assert result["complete"] is True
    assert row["selected_candidate_index"] == 2
    assert row["candidate_count"] == 8
    assert len(row["candidates"]) == 8
    assert row["objective"] == 14
    assert row["gap_signed_percent"] == -12.5
    assert result["aggregate"]["candidate_feasibility_percent"] == 37.5
    assert result["aggregate"]["selected_feasibility_percent"] == 100
    assert result["aggregate"]["generated_tokens"] == 40
    assert json.loads((tmp_path / (config()["config_id"] + ".json")).read_text()) == result


def test_all_infeasible_stays_in_all_instance_denominator(tmp_path):
    result = evaluate(tmp_path, MockBackend(["invalid"] * 8))
    row = result["rows"][0]
    assert row["selected_candidate_index"] is None
    assert row["feasible"] is False
    assert row["utility"] == 0
    assert result["aggregate"]["mean_all_instance_utility"] == 0
    assert result["aggregate"]["mean_selected_signed_gap_percent"] is None


def test_each_configuration_fresh_generates_and_uses_common_seed(tmp_path):
    backend = MockBackend()
    first = evaluate(tmp_path, backend)
    second = evaluate(tmp_path, backend,
                      configuration=config("T100_P090_K000", temperature=1.0, top_k=0))
    assert len(backend.calls) == 2
    assert backend.calls[0][2] == backend.calls[1][2]
    assert backend.calls[0][1] != backend.calls[1][1]
    assert first["evaluation_fingerprint"] != second["evaluation_fingerprint"]


def test_resume_exact_completed_rows_skips_generation(tmp_path):
    backend = MockBackend()
    result = evaluate(tmp_path, backend, sampling_seeds=[42, 43])
    assert len(backend.calls) == 2
    assert len(result["rows"]) == 2
    assert result == evaluate(tmp_path, backend, sampling_seeds=[42, 43])
    assert len(backend.calls) == 2


def test_resume_recovers_crash_truncated_final_row_only(tmp_path):
    backend = MockBackend()
    result = evaluate(tmp_path, backend)
    rows_path = tmp_path / (config()["config_id"] + ".rows.jsonl")
    with rows_path.open("ab") as stream:
        stream.write(b'{"incomplete":')
    assert evaluate(tmp_path, backend) == result
    assert len(backend.calls) == 1
    assert rows_path.read_bytes().endswith(b"\n")
    assert b"incomplete" not in rows_path.read_bytes()


def test_resume_rejects_different_binding_and_duplicate_records(tmp_path):
    evaluate(tmp_path)
    with pytest.raises(ValueError, match="binding differs"):
        evaluate(tmp_path, dataset_sha256="e" * 64)
    rows_path = tmp_path / (config()["config_id"] + ".rows.jsonl")
    rows_path.write_bytes(rows_path.read_bytes() * 2)
    with pytest.raises(ValueError, match="Duplicate resume row"):
        evaluate(tmp_path)


def test_resume_rejects_changed_candidate_evidence(tmp_path):
    evaluate(tmp_path)
    rows_path = tmp_path / (config()["config_id"] + ".rows.jsonl")
    row = json.loads(rows_path.read_text())
    row["candidates"][0]["raw_text"] = "Route: [0,1,1,3,0]"
    rows_path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="content seal mismatch"):
        evaluate(tmp_path)


def test_deadline_partial_is_resumable_and_never_marked_complete(tmp_path):
    backend = MockBackend()
    partial = evaluate(tmp_path, backend, deadline_at=time.monotonic() - 1)
    assert partial["complete"] is False
    assert partial["stop_reason"] == "deadline"
    assert partial["rows"] == []
    assert not backend.calls
    completed = evaluate(tmp_path, backend)
    assert completed["complete"] is True
    assert len(backend.calls) == 1


def test_run_binding_loads_one_backend_for_multiple_fresh_configurations(tmp_path, monkeypatch):
    rows_path = tmp_path / "dataset.json"
    rows_path.write_text(json.dumps([example()]))
    binding = {"task": "tsp", "method": "rloo", "checkpoint_id": "one-checkpoint",
               "phase": "development", "dataset_path": str(rows_path),
               "dataset_sha256": engine.file_sha256(rows_path), "sampling_seeds": [42],
               "configurations": [config(), config("T100_P090_K000", temperature=1, top_k=0)],
               "output_dir": str(tmp_path / "results"), "source_root": str(ROOT),
               "plan_sha256": "b" * 64, "binding_sha256": "c" * 64}
    loaded = []
    backend = MockBackend()

    def loader(value):
        loaded.append(value)
        return backend

    monkeypatch.setattr(engine, "HFBackend", loader)
    results = engine.run_binding(binding)
    assert len(loaded) == 1
    assert len(backend.calls) == 2
    assert all(result["complete"] for result in results)
    assert len(results) == 2
    assert json.loads((tmp_path / "results/invocation.json").read_text())["complete"] is True


def test_cli_exits75_for_resumable_partial(monkeypatch):
    monkeypatch.setattr(engine, "load_binding", lambda path: {"sealed": True})
    monkeypatch.setattr(engine, "run_binding", lambda *args, **kwargs:
                        [{"config_id": "test", "complete": False}])
    with pytest.raises(SystemExit) as error:
        engine.main(["--binding", "sealed.json", "--deadline-seconds", "120"])
    assert error.value.code == 75


@pytest.mark.parametrize("changes", [
    {"top_k": -1}, {"top_k": True}, {"top_p": 0}, {"top_p": 1.1},
    {"temperature": float("nan")}, {"num_return_sequences": 16}, {"max_new_tokens": 0},
    {"unregistered_parameter": 1},
])
def test_invalid_decoder_configuration_rejected(changes):
    with pytest.raises(ValueError):
        engine.validate_configuration(config(**changes))


def test_explicit_generation_config_disables_hidden_sampler_defaults():
    settings = engine.generation_settings(config(top_k=0),
                                          SimpleNamespace(pad_token_id=1, eos_token_id=2,
                                                          bos_token_id=3))
    assert settings["num_return_sequences"] == 8
    assert settings["do_sample"] is True
    assert settings["top_k"] == 0
    assert settings["forced_eos_token_id"] is None
    assert settings["suppress_tokens"] is None
    assert settings["repetition_penalty"] == 1
    assert settings["num_beams"] == 1
    assert "config_id" not in settings


def test_stable_seed_separates_instance_replicate_task_but_not_config():
    assert engine.sampling_seed(42, "tsp", "a") == engine.sampling_seed(42, "tsp", "a")
    assert len({engine.sampling_seed(42, "tsp", "a"),
                engine.sampling_seed(43, "tsp", "a"),
                engine.sampling_seed(42, "tsp", "b"),
                engine.sampling_seed(42, "mis", "a")}) == 4


def test_file_hash_verification_rejects_tampering_and_escape(tmp_path):
    file = tmp_path / "adapter_model.safetensors"
    file.write_bytes(b"test adapter")
    engine.verify_files(tmp_path, {file.name: engine.file_sha256(file)}, "checkpoint")
    with pytest.raises(ValueError, match="identity mismatch"):
        engine.verify_files(tmp_path, {file.name: "0" * 64}, "checkpoint")
    with pytest.raises(ValueError, match="escapes"):
        engine.verify_files(tmp_path, {"../model": "0" * 64}, "checkpoint")


def test_import_has_no_gpu_or_ml_dependency_side_effects():
    script = ("import importlib.util,sys; "
              f"s=importlib.util.spec_from_file_location('probe',{str(PATH)!r}); "
              "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
              "assert not any(k in sys.modules for k in ['torch','transformers','peft','datasets'])")
    subprocess.run([sys.executable, "-c", script], check=True)
