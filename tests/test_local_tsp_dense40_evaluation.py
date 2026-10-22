from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_local_tsp_dense40_evaluation",
    ROOT / "scripts" / "run_local_tsp_dense40_evaluation.py",
)
assert SPEC is not None and SPEC.loader is not None
lane = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lane
SPEC.loader.exec_module(lane)


def _seal(payload: dict, field: str) -> dict:
    sealed = dict(payload)
    sealed[field] = hashlib.sha256(lane.canonical_json_bytes(payload)).hexdigest()
    return sealed


def _option_value(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def _write_complete_training_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project = tmp_path / "project"
    run = project / "outputs" / "tsp_k8_development_20260814" / lane.TRAIN_RUN.name
    run.mkdir(parents=True)
    launcher = project / "scripts" / "run_local_tsp_k8_development.py"
    launcher.parent.mkdir()
    launcher.write_bytes(b"audited launcher")
    launcher_hash = hashlib.sha256(launcher.read_bytes()).hexdigest()

    sft = project / "models" / "local_tsp_sft500k_seed42_instruct"
    sft.mkdir(parents=True)
    sft_files = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"sft"}
    for name, payload in sft_files.items():
        (sft / name).write_bytes(payload)
    sft_manifest = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in sft_files.items()
    }
    train = project / "data" / "data_rl" / "tsp" / "train" / "train_rl.json"
    train.parent.mkdir(parents=True)
    train.write_text("[]", encoding="utf-8")
    train_hash = hashlib.sha256(train.read_bytes()).hexdigest()

    argv = [
        "/venv/python",
        "-m",
        "clasp.entrypoints.train_grpo",
        "--tasks",
        "tsp",
        "--model_path",
        "models/local_tsp_sft500k_seed42_instruct",
        "--tokenizer_path",
        "models/local_tsp_sft500k_seed42_instruct",
        "--data_root",
        "data/data_rl",
        "--output_dir",
        str(run.relative_to(project)),
        "--seed",
        "42",
        "--max_steps",
        "40",
        "--max_prompt_length",
        "29768",
        "--max_completion_length",
        "3000",
        "--num_generations",
        "8",
        "--generation_batch_size",
        "8",
        "--generation_temperature",
        "0.7",
        "--generation_top_p",
        "0.9",
        "--generation_top_k",
        "50",
        "--clasp_bon_success_credit",
        "dense_rank",
        "--clasp_proper_dual_gradient_mode",
        "signed",
        "--no_reason",
        "--eval-from-train-split",
        "--no-prefer_subset_files",
        "--use_clasp",
        "--clasp_strict_defaults",
        "--pd_group_scalarization",
        "--no-clasp_self_replay",
        "--no-clasp_adaptive_length",
        "--pd_require_state_on_resume",
    ]
    contract_payload = {
        "schema_version": 1,
        "protocol": "clasp-local-tsp-k8-development-v1",
        "status": "development_only_not_publication_eligible",
        "mode": "dense40",
        "task": "tsp",
        "method": "clasp_dense",
        "seed": 42,
        "run_dir": str(run.resolve()),
        "safety": {
            "create_once": True,
            "resume_allowed": False,
            "overwrite_allowed": False,
            "existing_run_adoption_allowed": False,
        },
        "limitations": {
            "development_only": True,
            "publication_eligible": False,
            "sota_claim_eligible": False,
        },
        "launcher": {"path": str(launcher.resolve()), "sha256": launcher_hash},
        "artifacts": {
            "sft_model": {
                "canonical_path": str(sft.resolve()),
                "files": [
                    {
                        "path": name,
                        "size_bytes": (sft / name).stat().st_size,
                        "sha256": digest,
                    }
                    for name, digest in sorted(sft_manifest.items())
                ],
            },
            "rl_train": {
                "canonical_path": str(train.resolve()),
                "rows": 3200,
                "size_bytes": train.stat().st_size,
                "sha256": train_hash,
            },
        },
        "argv": argv,
        "cli_resolution": {
            "tasks": ["tsp"],
            "max_steps": 40,
            "num_generations": 8,
            "generation_batch_size": 8,
            "effective_train_batch_size": 32,
            "use_clasp": True,
            "dense_credit": "dense_rank",
            "proper_dual_mode": "signed",
            "group_scalarization": True,
            "component_profile": "co_feasibility_lock_all",
        },
        "dataset_partition": {
            "source_rows": 3200,
            "expected_train_rows": 3040,
            "expected_eval_rows": 100,
            "external_eval_loaded": False,
        },
    }
    contract = run / "local_development_run_contract.json"
    contract.write_text(json.dumps(_seal(contract_payload, "contract_sha256")), encoding="utf-8")
    contract.chmod(0o440)

    monkeypatch.setattr(lane, "PROJECT_ROOT", project)
    contract_kwargs = {
        "contract_path": contract,
        "launcher_path": launcher,
        "sft_model": sft,
        "train_data": train,
        "expected_sft_manifest": sft_manifest,
        "expected_launcher_sha256": launcher_hash,
        "expected_train_sha256": train_hash,
    }
    return run, contract_kwargs


def _finish_training_tree(run: Path) -> Path:
    checkpoint = run / "checkpoints" / "checkpoint-40"
    checkpoint.mkdir(parents=True)
    history = [{"loss": 0.1, "grad_norm": 1.0, "step": 40}]
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"dense")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    (checkpoint / "rng_state.pth").write_bytes(b"rng")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 40, "max_steps": 40, "log_history": history}),
        encoding="utf-8",
    )
    pd_state = {
        "tsp": {
            "cost_names": ["parse_failure"],
            "cost_limits": [0.0],
            "cost_weights": [1.0],
            "dual_lambdas": [0.1],
        },
        "__clasp_bon__": {
            "lambdas": {"tsp": 0.1},
            "failure_ema": {"tsp": 0.2},
            "observed_groups": {"tsp": 40},
        },
    }
    (checkpoint / "multitask_primal_dual_state.json").write_text(
        json.dumps(pd_state), encoding="utf-8"
    )
    (run / "adapter_model.safetensors").write_bytes(b"final")
    (run / "grpo_trainer_state.json").write_text(
        json.dumps({"global_step": 40, "max_steps": 40, "log_history": history}),
        encoding="utf-8",
    )
    (run / "grpo_log_history.json").write_text(json.dumps(history), encoding="utf-8")
    (run / "stage_status.json").write_text(
        json.dumps(
            {
                "stage": "grpo",
                "completed": True,
                "global_step": 40,
                "max_steps": 40,
                "has_saved_model": True,
                "latest_checkpoint_dir": str(checkpoint.resolve()),
            }
        ),
        encoding="utf-8",
    )
    (run / "hyperparams.json").write_text(
        json.dumps(
            {
                "stage": "grpo",
                "tasks": ["tsp"],
                "base_model_name": "Qwen2.5-7B-Instruct",
                "args": {
                    "max_steps": 40,
                    "max_prompt_length": 29768,
                    "max_completion_length": 3000,
                    "num_generations": 8,
                    "generation_batch_size": 8,
                    "seed": 42,
                    "no_reason": True,
                    "prompt_style": "alpaca",
                    "use_clasp": True,
                    "clasp_bon_success_credit": "dense_rank",
                    "clasp_proper_dual_gradient_mode": "signed",
                },
                "dataset": {"train_size": 3040, "eval_size": 100},
                "distributed": {"world_size": 1},
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def test_completion_gate_refuses_early_and_requires_full_step40_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, kwargs = _write_complete_training_tree(tmp_path, monkeypatch)
    with pytest.raises(lane.TrainingNotComplete, match="stage_status"):
        lane.verify_complete_training_run(run, **kwargs)

    checkpoint = _finish_training_tree(run)
    receipt = lane.verify_complete_training_run(run, **kwargs)
    assert receipt["checkpoint"]["canonical_path"] == str(checkpoint.resolve())
    assert receipt["checkpoint"]["file_count"] == 7

    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(lane.LocalEvaluationError, match="full resumable state: optimizer.pt"):
        lane.verify_complete_training_run(run, **kwargs)


def test_completion_gate_resolves_trainer_relative_checkpoint_from_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, kwargs = _write_complete_training_tree(tmp_path, monkeypatch)
    checkpoint = _finish_training_tree(run)
    status_path = run / "stage_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["latest_checkpoint_dir"] = str(checkpoint.relative_to(lane.PROJECT_ROOT))
    status_path.write_text(json.dumps(status), encoding="utf-8")

    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    receipt = lane.verify_complete_training_run(run, **kwargs)

    assert receipt["checkpoint"]["canonical_path"] == str(checkpoint.resolve())


def test_training_contract_rejects_mutable_or_changed_sft_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, kwargs = _write_complete_training_tree(tmp_path, monkeypatch)
    contract = kwargs["contract_path"]
    contract.chmod(0o640)
    with pytest.raises(lane.LocalEvaluationError, match="writable"):
        lane.verify_local_training_contract(run, **kwargs)
    contract.chmod(0o440)
    (kwargs["sft_model"] / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(lane.LocalEvaluationError, match="SFT model hash/size mismatch"):
        lane.verify_local_training_contract(run, **kwargs)


def test_historical_eval_is_copied_once_and_is_the_only_staged_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [{"id": index} for index in range(3)]
    source = tmp_path / "source.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    root = tmp_path / "staged"
    destination = root / "tsp" / "eval" / "test.json"
    monkeypatch.setattr(lane, "SOURCE_EVAL_DATA", source)
    monkeypatch.setattr(lane, "EVAL_DATA_ROOT", root)
    monkeypatch.setattr(lane, "STAGED_EVAL_DATA", destination)
    monkeypatch.setattr(lane, "EVAL_SOURCE_ROWS", 3)
    monkeypatch.setattr(lane, "EVAL_START", 0)
    monkeypatch.setattr(lane, "EVAL_STOP", 2)
    monkeypatch.setattr(lane, "EVAL_DATA_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(
        lane,
        "EVAL_SLICE_SHA256",
        hashlib.sha256(lane.canonical_json_bytes(rows[:2])).hexdigest(),
    )

    dry = lane.stage_or_verify_eval_data(execute=False)
    assert dry["staged"] is None
    staged = lane.stage_or_verify_eval_data(execute=True)
    assert staged["staged_now"] is True
    assert staged["selected_interval"] == [0, 2]
    assert stat.S_IMODE(destination.stat().st_mode) == 0o440

    destination.chmod(0o640)
    with pytest.raises(lane.LocalEvaluationError, match="staged eval copy is writable"):
        lane.stage_or_verify_eval_data(execute=False)
    destination.chmod(0o440)
    extra = root / "tsp" / "train" / "leak.json"
    extra.parent.mkdir()
    extra.write_text("[]", encoding="utf-8")
    with pytest.raises(lane.LocalEvaluationError, match="must contain only"):
        lane.stage_or_verify_eval_data(execute=False)


def test_eval_commands_are_exactly_matched_except_stage_model_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sft = tmp_path / "sft"
    checkpoint = tmp_path / "checkpoint-40"
    data_root = tmp_path / "eval"
    monkeypatch.setattr(lane, "SFT_MODEL", sft)
    monkeypatch.setattr(lane, "EVAL_DATA_ROOT", data_root)
    left = lane.build_eval_command("sft", sft, tmp_path / "raw-sft")
    right = lane.build_eval_command("grpo", checkpoint, tmp_path / "raw-dense")
    ignored = {"--stage", "--model_path", "--inference_root"}

    def scientific_options(command: list[str]) -> dict[str, str | bool]:
        result: dict[str, str | bool] = {}
        index = 3
        while index < len(command):
            option = command[index]
            if option in {
                "--force_no_reason_prompt",
                "--do_sample",
                "--no-prefer_subset_files",
                "--no-update_best_run",
                "--save_details",
                "--save_results",
                "--no-resume",
            }:
                result[option] = True
                index += 1
            else:
                if option not in ignored:
                    result[option] = command[index + 1]
                index += 2
        return result

    assert scientific_options(left) == scientific_options(right)
    assert _option_value(left, "--tokenizer_path") == str(sft.resolve())
    assert _option_value(right, "--tokenizer_path") == str(sft.resolve())
    assert _option_value(left, "--max_eval_samples_per_task") == "200"
    assert _option_value(left, "--max_prompt_length") == "29768"
    assert _option_value(left, "--max_new_tokens") == "3000"
    assert _option_value(left, "--num_return_sequences") == "8"
    assert _option_value(left, "--seed") == "271828"
    assert _option_value(left, "--temperature") == "0.7"
    assert _option_value(left, "--top_p") == "0.9"
    assert _option_value(left, "--top_k") == "50"


def test_lane_contract_and_final_receipts_are_create_once_and_self_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane_root = tmp_path / "lane"
    monkeypatch.setattr(lane, "LANE_ROOT", lane_root)
    payload = {"schema_version": 1, "protocol": lane.PROTOCOL, "development_only": True}
    path = lane.reserve_lane(payload)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert (
        stored["contract_sha256"] == hashlib.sha256(lane.canonical_json_bytes(payload)).hexdigest()
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o440
    with pytest.raises(lane.LocalEvaluationError, match="already exists"):
        lane.reserve_lane(payload)

    final = lane._write_self_hashed_create_once(
        lane_root / "evaluation_receipt.json", payload, "receipt_sha256"
    )
    final_payload = json.loads(final.read_text(encoding="utf-8"))
    assert (
        final_payload["receipt_sha256"]
        == hashlib.sha256(lane.canonical_json_bytes(payload)).hexdigest()
    )
    with pytest.raises(lane.LocalEvaluationError, match="already exists"):
        lane._write_self_hashed_create_once(final, payload, "receipt_sha256")


def test_limitations_cannot_be_mistaken_for_publication_evidence() -> None:
    assert lane.LIMITATIONS["development_only"] is True
    assert lane.LIMITATIONS["publication_eligible"] is False
    assert lane.LIMITATIONS["sota_claim_eligible"] is False
    assert "Instruct" in lane.LIMITATIONS["instruction_tuned_initialization"]
    assert "A6000" in lane.LIMITATIONS["single_gpu"]
    assert "PEFT 0.19.1" in lane.LIMITATIONS["peft"]
    assert "src/clasp/entrypoints/repair.py" in lane.CODE_FILES
