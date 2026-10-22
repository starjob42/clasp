"""The analysis must not inspect partial-task quality or infer a win."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/rloo_one_shot_20261016/analyze_task.py"
SPEC = importlib.util.spec_from_file_location("one_shot_analysis_under_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ARMS = ["one_shot_sft", "foarl", "one_shot_grpo", "one_shot_grpo_fixed_horizon", "one_shot_rloo"]


def bound(path, value):
    path.write_text(json.dumps(value))
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def roster(tmp_path):
    plan = bound(tmp_path / "plan.json", {"development": {"arms_in_fixed_order": ARMS}})
    slots = [{"mode": "study", "task": "op", "arm": arm, "slot": "study.op." + arm,
              "output": str(tmp_path / arm)} for arm in ARMS]
    value = {"protocol": "one-shot-first-attempt-execution-roster-v1", "campaign_plan": plan, "slots": slots}
    return value, bound(tmp_path / "roster.json", value)


def test_incomplete_task_never_imports_model_or_quality_dependencies(tmp_path, monkeypatch):
    value, binding = roster(tmp_path)
    # A completed candidate with malformed raw output must remain unread while
    # any control is missing. The model libraries are unnecessary at this gate.
    candidate = Path(value["slots"][-1]["output"])
    candidate.mkdir()
    (candidate / "completion.json").write_text('{"status":"completed"}')
    (candidate / "result.json").write_text("INVALID PARTIAL QUALITY JSON")
    (candidate / "raw_completions.jsonl").write_text("INVALID RAW QUALITY")
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in ("clasp", "torch", "transformers", "tokenizers"):
            raise AssertionError("Incomplete-task gate imported quality/model dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    result = MODULE.analyze(binding, "op")
    assert result["status"] == "incomplete_task_no_quality_analysis"
    assert result["quality_analyzed"] is False
    assert result["superiority_established"] is False
    assert len(result["missing_slots"]) == 4


def test_completion_without_durable_result_cannot_open_quality_gate(tmp_path):
    value, binding = roster(tmp_path)
    for slot in value["slots"]:
        output = Path(slot["output"])
        output.mkdir()
        (output / "completion.json").write_text("INVALID JSON MUST STAY UNREAD")
    result = MODULE.analyze(binding, "op")
    assert len(result["missing_slots"]) == 5
    assert result["quality_analyzed"] is False


def test_reduced_control_roster_is_rejected(tmp_path):
    value, _ = roster(tmp_path)
    value["slots"].pop(3)
    binding = bound(tmp_path / "roster.json", value)
    with pytest.raises(ValueError, match="five-arm"):
        MODULE.analyze(binding, "op")


def test_tampered_roster_is_rejected_before_quality(tmp_path):
    _, binding = roster(tmp_path)
    Path(binding["path"]).write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        MODULE.analyze(binding, "op")


def test_tampered_plan_is_rejected_before_quality(tmp_path):
    value, binding = roster(tmp_path)
    Path(value["campaign_plan"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        MODULE.analyze(binding, "op")


def test_failed_complete_endpoint_does_not_pass_gate(tmp_path):
    value, _ = roster(tmp_path)
    for slot in value["slots"]:
        output = Path(slot["output"])
        output.mkdir()
        slot["contract"] = bound(tmp_path / (slot["arm"] + ".json"), {})
        (output / "completion.json").write_text('{"status":"failed"}')
        (output / "result.json").write_text('{"status":"completed_development_arm"}')
    binding = bound(tmp_path / "roster.json", value)
    with pytest.raises(ValueError, match="Incomplete or unmatched"):
        MODULE.analyze(binding, "op")
