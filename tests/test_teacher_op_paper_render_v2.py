"""Explicitly synthetic format checks; no live development outcomes are read."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments/rloo_one_shot_20261016/render_teacher_op_paper_v2.py"


def module(path=PATH):
    raw = path.read_bytes();value = types.ModuleType("teacher_render_diagnostic")
    value.__file__ = str(path);value.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    exec(compile(raw, str(path), "exec", dont_inherit=True), value.__dict__)
    return value


def fixture_analysis(R):
    summaries = {}
    for i, arm in enumerate(R.OLD + R.NEW):
        n = 96 - i
        summaries[arm] = {"N": 96, "n_feasible": n, "feasibility_percent": 100. * n / 96,
                          "own_feasible_signed_gap_percent": {"mean": -12.345, "median": -10.123, "best": -140., "worst": 99.999},
                          "all_instance_u10_percent": 82.25}
    contrasts = {}
    for arm in (*R.OLD, "foarl_teacher", "foarl_cost_teacher", "teacher_only"):
        contrasts[arm] = {"N": 96, "n_joint": 80, "joint_instance_ids": ["synthetic-" + str(i) for i in range(80)],
                          "zero_joint_support": False, "bootstrap_replicates": 50000, "bootstrap_seed": 2026101602,
                          "feasibility_delta_pp": {"estimate": 1.234, "ci95": [-3.456, 5.678]},
                          "joint_signed_gap_delta_pp": {"estimate": -12.345, "ci95": [-45.678, 3.456]},
                          "u10_delta_pp": {"estimate": .123, "ci95": [-5.678, 12.345]}}
    return {"synthetic_fixture": True, "status": "synthetic_teacher_OP_guard_fixture", "task": "op", "training_seed": 73,
            "quality_analyzed": True, "superiority_established": False, "publication_eligible": False,
            "bootstrap_replicates": 50000, "bootstrap_seed": 2026101602,
            "summaries": summaries, "rloo_teacher_minus_control_comparisons": contrasts}


def test_default_cli_writes_nothing(tmp_path):
    result = subprocess.run([sys.executable, "-I", "-B", str(PATH)], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["status"] == "prepared_only_no_artifacts_or_manuscript_changes"
    assert list(tmp_path.iterdir()) == []


def test_all_nine_rows_eight_contrasts_and_scope_are_explicit():
    R = module();analysis = fixture_analysis(R);texts = R._layout(analysis, synthetic_fixture=True)
    assert len(texts) == 3
    for text in texts.values():
        assert "SYNTHETIC LAYOUT FIXTURE; NO MODEL RESULTS" in text
        assert "textbf{RLOO" not in text and "best result" not in text.lower()
    own = texts["teacher_op_development_summary_table.tex"]
    pairs = texts["teacher_op_development_contrast_table.tex"]
    for arm in R.OLD + R.NEW:
        assert R.LABELS[arm] in own
    assert "-12.35" in own and "-140.00" in own and "not certified optima" in own
    for arm in (*R.OLD, "foarl_teacher", "foarl_cost_teacher", "teacher_only"):
        assert R.LABELS[arm] in pairs
    assert "pointwise 95" in pairs and "training-seed variation" in pairs
    section = texts["teacher_op_development_section.tex"]
    assert "historical RLOO9" in section and "no RL responses" in section
    assert "neither untouched confirmation nor superiority" in section
    assert "REGISTERED CONFIRMATION" not in section


def test_zero_support_prints_undefined_never_zero_effect():
    R = module();analysis = fixture_analysis(R)
    own = analysis["summaries"]["foarl_teacher"];own.update(n_feasible=0, feasibility_percent=0.)
    own["own_feasible_signed_gap_percent"] = {key: None for key in ("mean", "median", "best", "worst")}
    own["all_instance_u10_percent"] = 0.
    pair = analysis["rloo_teacher_minus_control_comparisons"]["foarl_teacher"]
    pair.update(n_joint=0, joint_instance_ids=[], zero_joint_support=True)
    pair["joint_signed_gap_delta_pp"] = {"estimate": None, "ci95": None}
    texts = R._layout(analysis, synthetic_fixture=True)
    assert "0/96 & -- & --" in texts["teacher_op_development_summary_table.tex"]
    assert "no jointly feasible instance is undefined" in texts["teacher_op_development_contrast_table.tex"]


@pytest.mark.parametrize("change", [lambda x: x["summaries"].pop("foarl"),
                                    lambda x: x["rloo_teacher_minus_control_comparisons"].pop("teacher_only"),
                                    lambda x: x.update(superiority_established=True),
                                    lambda x: x.update(bootstrap_replicates=100),
                                    lambda x: x["summaries"]["one_shot_sft"].update(n_feasible=True),
                                    lambda x: x["summaries"]["foarl"].update(all_instance_u10_percent=True),
                                    lambda x: x["summaries"]["foarl"]["own_feasible_signed_gap_percent"].update(mean=float("nan")),
                                    lambda x: x["rloo_teacher_minus_control_comparisons"]["teacher_only"].update(n_joint=0),
                                    lambda x: x["rloo_teacher_minus_control_comparisons"]["foarl"]["u10_delta_pp"].update(ci95=[1., -1.])])
def test_incomplete_or_misleading_numerical_layout_rejected(change):
    R = module();analysis = fixture_analysis(R);change(analysis)
    with pytest.raises(ValueError):
        R._layout(analysis, synthetic_fixture=True)


def test_synthetic_layout_cannot_use_actual_route():
    R = module()
    with pytest.raises(ValueError, match="separately marked"):
        R._layout(fixture_analysis(R), synthetic_fixture=False)


def test_public_build_rejects_synthetic_receipt_before_replay(tmp_path):
    R = module();path = tmp_path / "synthetic.json";raw = R.encoded(fixture_analysis(R));path.write_bytes(raw)
    with pytest.raises(ValueError, match="known completed actual"):
        R.build_layer({"path": str(path), "sha256": R.sha(raw), "size_bytes": len(raw)})
    assert list(tmp_path.iterdir()) == [path]


def test_renderer_source_cannot_be_rebound_after_execution(tmp_path):
    path = tmp_path / PATH.name;path.write_bytes(PATH.read_bytes());R = module(path)
    path.write_bytes(path.read_bytes() + b"\n# changed later\n")
    with pytest.raises(ValueError, match="pinned renderer"):
        R.build_layer({})


def test_known_analyzer_pin_matches_closed_source():
    R = module();path = PATH.parent / "analyze_teacher_op_v2.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == R.ANALYZER_SHA
