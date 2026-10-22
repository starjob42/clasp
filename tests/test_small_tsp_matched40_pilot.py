from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


launcher = load_module(
    "run_small_tsp_matched40_pilot",
    "scripts/run_small_tsp_matched40_pilot.py",
)
analysis = load_module(
    "analyze_small_tsp_matched40_pilot",
    "scripts/analyze_small_tsp_matched40_pilot.py",
)


def option_value(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def test_contract_keeps_policy_and_data_seeds_separate() -> None:
    contract = launcher.load_contract()
    assert contract["data_seed"] == 20260815
    for seed, gpu in ((101, 0), (202, 1), (303, 2)):
        env = launcher.build_environment(contract, "dual_scalar", seed, os.environ)
        assert env["GRPO_DATA_SEED"] == "20260815"
        assert env["PYTHONHASHSEED"] == str(seed)
        assert env["CUDA_VISIBLE_DEVICES"] == str(gpu)


def test_arm_commands_share_budget_decoder_and_token_surrogate() -> None:
    contract = launcher.load_contract()
    commands = {
        arm: launcher.build_train_argv(
            contract,
            arm,
            101,
            launcher.output_dir(contract, arm, 101),
        )
        for arm in contract["arms"]
    }
    for command in commands.values():
        assert option_value(command, "--num_generations") == "9"
        assert option_value(command, "--generation_batch_size") == "9"
        assert option_value(command, "--per_device_train_batch_size") == "3"
        assert option_value(command, "--gradient_accumulation_steps") == "3"
        assert option_value(command, "--max_steps") == "40"
        assert option_value(command, "--grpo_beta") == "0"
        assert option_value(command, "--grpo_loss_type") == "grpo"
        assert option_value(command, "--generation_temperature") == "0.7"
        assert option_value(command, "--generation_top_p") == "0.9"
    canonical = commands["canonical"]
    no_controller = commands["canonical_no_controller"]
    dual = commands["dual_scalar"]
    standard = commands["standard_grpo"]
    assert "--clasp_canonical_set" in canonical
    assert option_value(canonical, "--clasp_deployment_set_size") == "8"
    assert "--clasp_canonical_set" in no_controller
    assert "--pd_group_scalarization" in dual
    assert "--no-use_clasp" in dual
    assert standard[2] == "clasp.entrypoints.train_grpo_standard"
    assert "--no-use_primal_dual" in standard
    assert contract["arms"]["canonical"]["environment"]["CLASP_BON_DUAL_LR"] == "0.1"
    assert (
        contract["arms"]["canonical_no_controller"]["environment"][
            "CLASP_BON_DUAL_LR"
        ]
        == "0"
    )


def test_parser_diff_matches_completed_canonical_exactly() -> None:
    contract = launcher.load_contract()
    result = launcher.validate_resolutions(contract, 101)
    assert result["matched_field_mismatches"] == {}
    assert result["completed_canonical_reference_diff"]["differences"] == {}
    assert result["resolved"]["canonical"]["clasp_deployment_set_size"] == 8
    assert result["resolved"]["dual_scalar"]["pd_group_scalarization"] is True
    assert result["resolved"]["standard_grpo"]["use_primal_dual"] is False


def test_create_once_run_receipt_cannot_be_reused(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    receipt = launcher.create_run_receipt(run_dir, {"arm": "dual_scalar", "seed": 101})
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert len(payload["receipt_sha256"]) == 64
    assert receipt.stat().st_mode & 0o222 == 0
    with pytest.raises(RuntimeError, match="already exists"):
        launcher.require_fresh_output(run_dir)
    with pytest.raises(RuntimeError, match="appeared"):
        launcher.create_run_receipt(run_dir, {"arm": "dual_scalar", "seed": 101})


def test_gpu_gate_accepts_only_idle_seed_mapped_2080ti() -> None:
    contract = launcher.load_contract()

    def good(*args, **kwargs):
        return subprocess.CompletedProcess(
            [],
            0,
            "0, NVIDIA GeForce RTX 2080 Ti, 11264, 16\n"
            "1, NVIDIA GeForce RTX 2080 Ti, 11264, 20\n",
            "",
        )

    assert launcher.validate_gpu(contract, 101, good)["physical_index"] == 0

    def busy(*args, **kwargs):
        return subprocess.CompletedProcess(
            [], 0, "0, NVIDIA GeForce RTX 2080 Ti, 11264, 2048\n", ""
        )

    with pytest.raises(RuntimeError, match="not idle enough"):
        launcher.validate_gpu(contract, 101, busy)


def make_record(feasible_gaps: list[float | None]) -> analysis.EvaluationRecord:
    assert len(feasible_gaps) == 9
    candidates = tuple(
        analysis.Candidate(feasible=gap is not None, gap=gap) for gap in feasible_gaps
    )
    return analysis.EvaluationRecord(
        example_index=0,
        semantic_hash="a" * 64,
        candidates=candidates,
        generated_tokens=90,
        generation_time_sec=9.0,
    )


def test_exact_leave_one_out_k8_not_bo9() -> None:
    one_feasible = make_record([0.05, None, None, None, None, None, None, None, None])
    row = analysis.leave_one_out_instance(one_feasible, [0.01, 0.05])
    assert row["pass_at_8"] == pytest.approx(8 / 9)
    assert row["failure_at_8"] == pytest.approx(1 / 9)
    assert row["success_at_gap_threshold"]["0.01"] == 0.0
    assert row["success_at_gap_threshold"]["0.05"] == pytest.approx(8 / 9)
    assert row["bo9_diagnostic"]["pass"] == 1.0

    two_feasible = make_record([0.02, 0.2, None, None, None, None, None, None, None])
    row = analysis.leave_one_out_instance(two_feasible, [0.05])
    assert row["pass_at_8"] == 1.0
    # Omitting the 0.02 candidate leaves best gap 0.2; the other eight subsets succeed.
    assert row["success_at_gap_threshold"]["0.05"] == pytest.approx(8 / 9)


def test_sealed_sft_b9_recomputes_exact_failure_at_8() -> None:
    paths = list(
        (
            ROOT
            / "experiments/additional_evidence/small_tsp_canonical_v1/"
            "eval_sft_b9_seed314159"
        ).rglob("run_details__*.jsonl")
    )
    assert len(paths) == 1
    records = analysis.load_records(paths[0])
    summary, rows = analysis.summarize_records(records, [0.0, 0.05])
    assert len(rows) == 31
    assert summary["failure_at_8"] == pytest.approx(0.07168458781362007)
    assert summary["bo9_secondary_diagnostic"]["pass_at_9"] == pytest.approx(29 / 31)
    assert summary["estimand"].startswith("uniform average")
