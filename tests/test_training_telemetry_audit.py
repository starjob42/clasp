import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "analysis_outputs/clasp_grpo_fair_dev_20260905/current/audit_training_telemetry.py"
SPEC = importlib.util.spec_from_file_location("training_telemetry_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_log_parser_accepts_nan_diagnostics_and_ignores_incomplete_final_record():
    raw = (b"startup\n{'loss': 0.0, 'grad_norm': 0.2, 'unrelated': nan}\n"
           b"{'loss': 0.0, 'grad_norm': 0.3")
    rows, incomplete = audit.parse_rows(raw)
    assert len(rows) == 1
    assert rows[0]["grad_norm"] == 0.2
    assert audit.math.isnan(rows[0]["unrelated"])
    assert incomplete == 1


def test_log_parser_does_not_execute_expressions(tmp_path):
    target = tmp_path / "must-not-exist"
    raw = f"{{'loss': 0, 'grad_norm': __import__('pathlib').Path({str(target)!r}).touch()}}\n".encode()
    with pytest.raises(ValueError):
        audit.parse_rows(raw)
    assert not target.exists()


def test_common_record_limit_excludes_later_diagnostics():
    raw = b"{'loss': 0, 'grad_norm': 0.2}\n{'loss': 0, 'grad_norm': 2.0}\n"
    hyperparams = {"trainer_config": {"max_grad_norm": 1.0}}
    result = audit.summarize_arm(raw, hyperparams, "cvrp", record_limit=1)
    assert result["captured_record_count"] == 2
    assert result["analyzed_record_count"] == 1
    assert result["grad_norm"]["max"] == 0.2
    assert result["gradient_norm_above_clip_threshold_count"] == 0
    with pytest.raises(ValueError, match="non-finite"):
        audit.numeric_summary([float("nan")])
