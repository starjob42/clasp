#!/usr/bin/env python3
"""Create-once post-training evaluation for the contracted local TSP dense40 run.

This lane is intentionally DEVELOPMENT-ONLY.  It will not reserve an output
directory, stage evaluation data, or start inference until the fixed step-40
training run is finalized with a complete optimizer checkpoint.  Once started,
the lane is single-use: a failed or partial directory is retained for audit and
cannot be resumed or adopted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

SCHEMA_VERSION = 1
PROTOCOL = "clasp-local-tsp-dense40-posttrain-evaluation-v1"
ERROR_PREFIX = "[LocalDense40EvaluationError]"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = (PROJECT_ROOT.parent / "my_idea_grpo" / ".venv" / "bin" / "python").absolute()
TRAIN_RUN = (
    PROJECT_ROOT / "outputs" / "tsp_k8_development_20260814" / "contracted_clasp_dense40_seed42_v1"
)
CHECKPOINT = TRAIN_RUN / "checkpoints" / "checkpoint-40"
TRAIN_CONTRACT = TRAIN_RUN / "local_development_run_contract.json"
TRAIN_LAUNCHER = PROJECT_ROOT / "scripts" / "run_local_tsp_k8_development.py"
SFT_MODEL = PROJECT_ROOT / "models" / "local_tsp_sft500k_seed42_instruct"
TRAIN_DATA = PROJECT_ROOT / "data" / "data_rl" / "tsp" / "train" / "train_rl.json"
SOURCE_EVAL_DATA = PROJECT_ROOT.parent / "hybrid_gnn_llm" / "data_rl" / "tsp" / "eval" / "test.json"
EVAL_DATA_ROOT = PROJECT_ROOT / "data" / "development_historical_tsp_eval_20260814"
STAGED_EVAL_DATA = EVAL_DATA_ROOT / "tsp" / "eval" / "test.json"
LANE_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "sota_20260814"
    / "local_development"
    / "tsp_dense40_posttrain_k8_seed271828_v1"
)

EXPECTED_TRAIN_LAUNCHER_SHA256 = "849dbbd93510d593bf52276d8f2405eb6bbdfaea20209a9e7804ee8356141ba3"
TRAIN_DATA_SHA256 = "90c5a48909ac9df191c727237474ec262db4a9c654f3555a19b744304dc0a288"
EVAL_DATA_SHA256 = "f169771f0ccfe0e74cf247067edcea8f89a357ab4823352e3ff55201cf8b9597"
EVAL_SLICE_SHA256 = "c9beea11e14f7b31897c515e3fb561fa93e470e258ea62a69478f3e06603437c"
EVAL_SOURCE_ROWS = 300
EVAL_START = 0
EVAL_STOP = 200

SFT_MANIFEST = {
    "README.md": "37b7d9d614273fa4a8a1bf3d96becb7c5c30ca22a4fe1f34014c14d780f84591",
    "adapter_config.json": "77a1bbe21c4ed225bcd14a78cc9386b819fb6369f4624ea8cd2dc7fc2702f7fd",
    "adapter_model.safetensors": "9bf9ea50a5f7e9ff57399ea9bcca487785867d95bc9fdc11dadb7c723cef5173",
    "added_tokens.json": "58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b",
    "chat_template.jinja": "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f",
    "hyperparams.json": "7908d8124f65858be515101f265d630ebcca54d0090665a9e648673a30a0038f",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "sft_log_history.json": "ef0cb7a880baf47872cedadec2e0fb6a7a3c29adc2d71871b0f3cdc72dfbcc7f",
    "sft_trainer_state.json": "23ffb305c35a17bc74f3b0851d76b567e92e4fe9b9a689f6bfaa286a84e73be2",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    "stage_status.json": "0b8d662c308caa6af0e2316518ce5adfcf6ca19a105b2474b09a5c081d6a0aeb",
    "tokenizer.json": "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa",
    "tokenizer_config.json": "e6711f2a711bd7e5607373a1770048709e4494bc08ab8fb6b54659943ce55845",
    "training_args.bin": "0f8a99ac79db71f82b6597316ffdaf8d34d99d0eeb2e0fbe4efb1788279e30c8",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}

EXPECTED_RUNTIME = {
    "python": "3.11.2",
    "torch": "2.10.0+cu128",
    "transformers": "4.57.6",
    "trl": "0.24.0",
    "peft": "0.19.1",
    "datasets": "4.3.0",
    "bitsandbytes": "0.49.2",
    "cuda": "12.8",
}

CODE_FILES = (
    "scripts/run_local_tsp_k8_development.py",
    "scripts/run_local_tsp_dense40_evaluation.py",
    "scripts/compare_development.py",
    "src/clasp/config.py",
    "src/clasp/runtime.py",
    "src/clasp/data/benchmark.py",
    "src/clasp/data/loaders.py",
    "src/clasp/data/prompts.py",
    "src/clasp/entrypoints/evaluate.py",
    "src/clasp/entrypoints/repair.py",
    "src/clasp/models/loading.py",
    "src/clasp/rewards/co_tasks.py",
)

EVALUATION_ENV = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "3",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONHASHSEED": "271828",
    "PYTHONPATH": "src",
    "CLASP_TSP_PROMPT_KNN": "0",
    "CLASP_TSP_PROMPT_KNN_DISTANCES": "0",
    "CLASP_TSP_ROUTE_ONLY_OUTPUT": "0",
}

LIMITATIONS = {
    "development_only": True,
    "publication_eligible": False,
    "sota_claim_eligible": False,
    "instruction_tuned_initialization": (
        "The specialist and CLASP trajectory start from Qwen2.5-7B-Instruct, not the "
        "approved non-instruction Qwen2.5-7B publication base."
    ),
    "single_gpu": (
        "Training and evaluation use one local RTX A6000; this is not bit-equivalent to "
        "the matched four-A100 publication design."
    ),
    "peft": (
        "The local runtime uses PEFT 0.19.1, outside the publication environment contract "
        "peft>=0.18,<0.19."
    ),
    "benchmark": (
        "The first 200 rows of the audited historical TSP eval file are development data, "
        "not the frozen independent confirmation benchmark."
    ),
    "scope": "One TSP training seed only; no multi-task or training-seed inference.",
}


class LocalEvaluationError(RuntimeError):
    """Raised when the create-once development lane cannot proceed safely."""


class TrainingNotComplete(LocalEvaluationError):
    """Raised only while the contracted training run has not reached step 40."""


def fail(message: str) -> NoReturn:
    raise LocalEvaluationError(f"{ERROR_PREFIX} {message}")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} JSON from {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be one JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        fail(f"artifact must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        fail(f"cannot fingerprint {path}: {exc}")
    signature = lambda value: (  # noqa: E731 - compact stable-file identity helper
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if signature(before) != signature(after) or not stat.S_ISREG(after.st_mode):
        fail(f"artifact changed while it was fingerprinted: {path}")
    return digest.hexdigest()


def artifact_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    digest = sha256_file(path)
    label = str(path.relative_to(relative_to)) if relative_to is not None else str(path.resolve())
    return {"path": label, "size_bytes": path.stat().st_size, "sha256": digest}


def manifest_directory(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        fail(f"artifact tree must be a regular non-symlink directory: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            fail(f"symlinks are forbidden in artifact trees: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            fail(f"non-regular artifact-tree entry: {path}")
        files.append(artifact_identity(path, relative_to=root))
    if not files:
        fail(f"artifact tree is empty: {root}")
    aggregate = hashlib.sha256(canonical_json_bytes(files)).hexdigest()
    return {
        "canonical_path": str(root.resolve()),
        "file_count": len(files),
        "files": files,
        "manifest_sha256": aggregate,
    }


def code_manifest() -> dict[str, Any]:
    files = [
        artifact_identity(PROJECT_ROOT / name, relative_to=PROJECT_ROOT) for name in CODE_FILES
    ]
    return {
        "files": files,
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }


def _validate_self_hash(payload: Mapping[str, Any], field: str, *, label: str) -> str:
    stored = payload.get(field)
    if not isinstance(stored, str) or len(stored) != 64:
        fail(f"{label}.{field} is not a SHA-256 digest")
    unsigned = dict(payload)
    unsigned.pop(field, None)
    observed = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if stored != observed:
        fail(f"{label}.{field} mismatch: stored={stored}, observed={observed}")
    return stored


def _argv_value(argv: Sequence[str], option: str) -> str:
    if argv.count(option) != 1:
        fail(f"training contract must contain exactly one {option}")
    index = argv.index(option)
    if index + 1 >= len(argv):
        fail(f"training contract option has no value: {option}")
    return str(argv[index + 1])


def _verify_sft_manifest(
    root: Path,
    contract_files: object,
    *,
    expected: Mapping[str, str],
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        fail(f"SFT model must be a copied non-symlink directory: {root}")
    actual_names = sorted(path.name for path in root.iterdir())
    if actual_names != sorted(expected):
        fail("SFT model file set differs from the contracted audited specialist")
    if not isinstance(contract_files, list):
        fail("training contract SFT manifest is missing")
    by_name = {
        str(entry.get("path")): entry
        for entry in contract_files
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    if set(by_name) != set(expected) or len(by_name) != len(contract_files):
        fail("training contract SFT manifest file set is invalid")
    receipts = []
    for name, expected_hash in sorted(expected.items()):
        path = root / name
        observed = sha256_file(path)
        entry = by_name[name]
        if (
            observed != expected_hash
            or entry.get("sha256") != expected_hash
            or entry.get("size_bytes") != path.stat().st_size
        ):
            fail(f"SFT model hash/size mismatch for {name}")
        receipts.append(artifact_identity(path, relative_to=root))
    return {
        "canonical_path": str(root.resolve()),
        "files": receipts,
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(receipts)).hexdigest(),
    }


def verify_local_training_contract(
    run_dir: Path = TRAIN_RUN,
    *,
    contract_path: Path | None = None,
    launcher_path: Path = TRAIN_LAUNCHER,
    sft_model: Path = SFT_MODEL,
    train_data: Path = TRAIN_DATA,
    expected_sft_manifest: Mapping[str, str] = SFT_MANIFEST,
    expected_launcher_sha256: str = EXPECTED_TRAIN_LAUNCHER_SHA256,
    expected_train_sha256: str = TRAIN_DATA_SHA256,
) -> dict[str, Any]:
    run_dir = run_dir.absolute()
    contract_path = contract_path or run_dir / "local_development_run_contract.json"
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise TrainingNotComplete(f"contracted training run is not present: {run_dir}")
    if contract_path.is_symlink() or not contract_path.is_file():
        raise TrainingNotComplete(
            f"immutable local training contract is not present: {contract_path}"
        )
    if stat.S_IMODE(contract_path.stat().st_mode) & 0o222:
        fail(f"local training contract is writable, hence not immutable: {contract_path}")
    contract = load_json_object(contract_path, "local training contract")
    contract_hash = _validate_self_hash(contract, "contract_sha256", label="training_contract")

    expected_fields = {
        "schema_version": 1,
        "protocol": "clasp-local-tsp-k8-development-v1",
        "status": "development_only_not_publication_eligible",
        "mode": "dense40",
        "task": "tsp",
        "method": "clasp_dense",
        "seed": 42,
        "run_dir": str(run_dir.resolve()),
    }
    for field, expected in expected_fields.items():
        if contract.get(field) != expected:
            fail(f"local training contract {field} drift: expected={expected!r}")
    if contract.get("safety") != {
        "create_once": True,
        "resume_allowed": False,
        "overwrite_allowed": False,
        "existing_run_adoption_allowed": False,
    }:
        fail("local training contract create-once safety declaration drifted")
    limitations = contract.get("limitations")
    if not isinstance(limitations, dict) or not (
        limitations.get("development_only") is True
        and limitations.get("publication_eligible") is False
        and limitations.get("sota_claim_eligible") is False
    ):
        fail("local training contract is not explicitly development-only")

    observed_launcher_hash = sha256_file(launcher_path)
    launcher = contract.get("launcher")
    if not isinstance(launcher, dict) or not (
        Path(str(launcher.get("path", ""))).resolve() == launcher_path.resolve()
        and launcher.get("sha256") == expected_launcher_sha256
        and observed_launcher_hash == expected_launcher_sha256
    ):
        fail("training launcher bytes differ from the immutable contracted launcher")

    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict):
        fail("local training contract has no artifact bindings")
    sft_contract = artifacts.get("sft_model")
    if (
        not isinstance(sft_contract, dict)
        or Path(str(sft_contract.get("canonical_path", ""))).resolve() != sft_model.resolve()
    ):
        fail("local training contract SFT canonical path drifted")
    sft_receipt = _verify_sft_manifest(
        sft_model,
        sft_contract.get("files"),
        expected=expected_sft_manifest,
    )

    train_contract = artifacts.get("rl_train")
    observed_train_hash = sha256_file(train_data)
    if not isinstance(train_contract, dict) or not (
        Path(str(train_contract.get("canonical_path", ""))).resolve() == train_data.resolve()
        and train_contract.get("rows") == 3200
        and train_contract.get("sha256") == expected_train_sha256
        and train_contract.get("size_bytes") == train_data.stat().st_size
        and observed_train_hash == expected_train_sha256
    ):
        fail("local training contract RL-train data hash/path/row binding drifted")

    argv = contract.get("argv")
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        fail("local training contract argv is malformed")
    required_values = {
        "--tasks": "tsp",
        "--model_path": "models/local_tsp_sft500k_seed42_instruct",
        "--tokenizer_path": "models/local_tsp_sft500k_seed42_instruct",
        "--data_root": "data/data_rl",
        "--output_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "--seed": "42",
        "--max_steps": "40",
        "--max_prompt_length": "29768",
        "--max_completion_length": "3000",
        "--num_generations": "8",
        "--generation_batch_size": "8",
        "--generation_temperature": "0.7",
        "--generation_top_p": "0.9",
        "--generation_top_k": "50",
        "--clasp_bon_success_credit": "dense_rank",
        "--clasp_proper_dual_gradient_mode": "signed",
    }
    for option, expected in required_values.items():
        if _argv_value(argv, option) != expected:
            fail(f"local training contract option drift: {option} != {expected}")
    for flag in (
        "--no_reason",
        "--eval-from-train-split",
        "--no-prefer_subset_files",
        "--use_clasp",
        "--clasp_strict_defaults",
        "--pd_group_scalarization",
        "--no-clasp_self_replay",
        "--no-clasp_adaptive_length",
    ):
        if argv.count(flag) != 1:
            fail(f"local training contract is missing unique required flag {flag}")
    forbidden_resume_options = {
        "--resume",
        "--resume_from_checkpoint",
        "--resume-from-checkpoint",
    }
    if any(
        value in forbidden_resume_options
        or any(value.startswith(f"{option}=") for option in forbidden_resume_options)
        for value in argv
    ):
        fail("local training contract argv unexpectedly contains a resume source")
    resolution = contract.get("cli_resolution")
    if resolution != {
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
    }:
        fail("local training contract resolved CLI drifted")
    partition = contract.get("dataset_partition")
    if not isinstance(partition, dict) or any(
        partition.get(key) != expected
        for key, expected in {
            "source_rows": 3200,
            "expected_train_rows": 3040,
            "expected_eval_rows": 100,
            "external_eval_loaded": False,
        }.items()
    ):
        fail("local training contract train-derived partition drifted")
    return {
        "contract": artifact_identity(contract_path),
        "contract_sha256": contract_hash,
        "launcher": artifact_identity(launcher_path),
        "sft_model": sft_receipt,
        "rl_train": artifact_identity(train_data),
    }


def _require_fields(payload: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for name, value in expected.items():
        if payload.get(name) != value:
            fail(f"{label}.{name} mismatch: expected={value!r}, observed={payload.get(name)!r}")


def verify_complete_training_run(
    run_dir: Path = TRAIN_RUN,
    **contract_kwargs: Any,
) -> dict[str, Any]:
    contract = verify_local_training_contract(run_dir, **contract_kwargs)
    status_path = run_dir / "stage_status.json"
    if not status_path.is_file():
        raise TrainingNotComplete("contracted dense40 run has no completed stage_status.json yet")
    status = load_json_object(status_path, "dense40 stage status")
    completion_fields = {
        "stage": "grpo",
        "completed": True,
        "global_step": 40,
        "max_steps": 40,
        "has_saved_model": True,
    }
    if any(status.get(name) != value for name, value in completion_fields.items()):
        raise TrainingNotComplete(
            "contracted dense40 run is not COMPLETE at global_step=max_steps=40"
        )

    checkpoint = run_dir / "checkpoints" / "checkpoint-40"
    latest = status.get("latest_checkpoint_dir")
    if not isinstance(latest, str):
        fail("stage_status.latest_checkpoint_dir is not the terminal checkpoint-40")
    latest_path = Path(latest)
    if not latest_path.is_absolute():
        # The contracted trainer runs from PROJECT_ROOT and therefore records a
        # project-relative checkpoint path.  Resolve it against that same fixed
        # origin so invoking this evaluator from another working directory cannot
        # produce a false mismatch after training has completed.
        latest_path = PROJECT_ROOT / latest_path
    if latest_path.resolve() != checkpoint.resolve():
        fail("stage_status.latest_checkpoint_dir is not the terminal checkpoint-40")
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        fail("terminal checkpoint-40 is missing or is a symlink")
    required = (
        "adapter_config.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "multitask_primal_dual_state.json",
    )
    for name in required:
        path = checkpoint / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            fail(f"terminal checkpoint lacks full resumable state: {name}")
    model_files = [
        checkpoint / name
        for name in ("adapter_model.safetensors", "adapter_model.bin")
        if (checkpoint / name).is_file()
    ]
    if len(model_files) != 1:
        fail("terminal checkpoint must contain exactly one PEFT adapter weight artifact")
    trainer_state = load_json_object(checkpoint / "trainer_state.json", "checkpoint trainer state")
    _require_fields(trainer_state, {"global_step": 40, "max_steps": 40}, "trainer_state")
    if not isinstance(trainer_state.get("log_history"), list):
        fail("checkpoint trainer_state.log_history is missing")

    pd_state = load_json_object(
        checkpoint / "multitask_primal_dual_state.json", "checkpoint primal-dual state"
    )
    task_state = pd_state.get("tsp")
    bon_state = pd_state.get("__clasp_bon__")
    if not isinstance(task_state, dict) or not isinstance(bon_state, dict):
        fail("checkpoint primal-dual state lacks TSP or CLASP BoN controller state")
    for name in ("cost_names", "cost_limits", "cost_weights", "dual_lambdas"):
        if not isinstance(task_state.get(name), list):
            fail(f"checkpoint TSP primal-dual state lacks {name}")
    if not all(
        len(task_state[name]) == len(task_state["cost_names"])
        for name in ("cost_limits", "cost_weights", "dual_lambdas")
    ):
        fail("checkpoint TSP primal-dual vectors are misaligned")
    for name in ("lambdas", "failure_ema", "observed_groups"):
        values = bon_state.get(name)
        if not isinstance(values, dict) or "tsp" not in values:
            fail(f"checkpoint CLASP BoN state lacks tsp {name}")

    root_state = load_json_object(run_dir / "grpo_trainer_state.json", "root trainer state")
    _require_fields(root_state, {"global_step": 40, "max_steps": 40}, "grpo_trainer_state")
    try:
        root_history = json.loads((run_dir / "grpo_log_history.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read root GRPO log history: {exc}")
    if not isinstance(root_history, list) or root_state.get("log_history") != root_history:
        fail("root trainer state and GRPO log history disagree")
    if root_history[: len(trainer_state["log_history"])] != trainer_state["log_history"]:
        fail("checkpoint-40 log history is not a prefix of the finalized root history")

    hparams = load_json_object(run_dir / "hyperparams.json", "dense40 hyperparameters")
    _require_fields(
        hparams,
        {
            "stage": "grpo",
            "tasks": ["tsp"],
            "base_model_name": "Qwen2.5-7B-Instruct",
        },
        "hyperparams",
    )
    args = hparams.get("args")
    dataset = hparams.get("dataset")
    distributed = hparams.get("distributed")
    if (
        not isinstance(args, dict)
        or not isinstance(dataset, dict)
        or not isinstance(distributed, dict)
    ):
        fail("dense40 hyperparameters lack args/dataset/distributed contracts")
    _require_fields(
        args,
        {
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
        "hyperparams.args",
    )
    _require_fields(dataset, {"train_size": 3040, "eval_size": 100}, "hyperparams.dataset")
    _require_fields(distributed, {"world_size": 1}, "hyperparams.distributed")
    root_models = [
        run_dir / name
        for name in ("adapter_model.safetensors", "adapter_model.bin")
        if (run_dir / name).is_file()
    ]
    if len(root_models) != 1:
        fail("finalized dense40 run must contain exactly one root adapter weight artifact")
    return {
        "training_contract": contract,
        "stage_status": artifact_identity(status_path),
        "hyperparameters": artifact_identity(run_dir / "hyperparams.json"),
        "root_trainer_state": artifact_identity(run_dir / "grpo_trainer_state.json"),
        "root_log_history": artifact_identity(run_dir / "grpo_log_history.json"),
        "root_model": artifact_identity(root_models[0]),
        "checkpoint": manifest_directory(checkpoint),
    }


def wait_for_complete_training(timeout_seconds: int, poll_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + max(0, timeout_seconds)
    while True:
        try:
            return verify_complete_training_run()
        except TrainingNotComplete:
            if timeout_seconds <= 0 or time.monotonic() >= deadline:
                raise
            time.sleep(min(max(1, poll_seconds), 60, max(0.0, deadline - time.monotonic())))


def _eval_payload(path: Path) -> tuple[bytes, list[Any], str]:
    if path.is_symlink() or not path.is_file():
        fail(f"historical TSP eval must be a regular non-symlink file: {path}")
    try:
        payload = path.read_bytes()
        rows = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot load historical TSP eval data {path}: {exc}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EVAL_DATA_SHA256:
        fail(f"historical TSP eval SHA-256 mismatch: observed={digest}")
    if not isinstance(rows, list) or len(rows) != EVAL_SOURCE_ROWS:
        fail(f"historical TSP eval must contain exactly {EVAL_SOURCE_ROWS} rows")
    slice_hash = hashlib.sha256(canonical_json_bytes(rows[EVAL_START:EVAL_STOP])).hexdigest()
    if slice_hash != EVAL_SLICE_SHA256:
        fail("historical TSP eval [0,200) slice identity drifted")
    return payload, rows, slice_hash


def _validate_staged_tree(root: Path, destination: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        fail(f"staged eval root is missing or is a symlink: {root}")
    actual_files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            fail(f"staged eval tree contains a symlink: {path}")
        if path.is_file():
            actual_files.append(path.resolve())
        elif not path.is_dir():
            fail(f"staged eval tree contains a non-regular entry: {path}")
    if actual_files != [destination.resolve()]:
        fail("staged eval root must contain only tsp/eval/test.json")


def stage_or_verify_eval_data(*, execute: bool) -> dict[str, Any]:
    payload, rows, slice_hash = _eval_payload(SOURCE_EVAL_DATA)
    staged = STAGED_EVAL_DATA
    staged_now = False
    if staged.exists() or staged.is_symlink():
        _eval_payload(staged)
    elif execute:
        staged.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
        except FileExistsError:
            fail(f"staged eval destination appeared during create-once copy: {staged}")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            staged.chmod(0o440)
            staged_now = True
        except Exception as exc:
            fail(f"failed while staging audited eval bytes; preserve destination for audit: {exc}")
    if staged.exists():
        _validate_staged_tree(EVAL_DATA_ROOT, staged)
        _eval_payload(staged)
        if stat.S_IMODE(staged.stat().st_mode) & 0o222:
            fail(f"staged eval copy is writable, hence not immutable: {staged}")
        identity: dict[str, Any] | None = artifact_identity(staged)
    else:
        identity = None
    return {
        "source": artifact_identity(SOURCE_EVAL_DATA),
        "staged": identity,
        "staged_path": str(staged.resolve(strict=False)),
        "staged_now": staged_now,
        "source_rows": len(rows),
        "selected_interval": [EVAL_START, EVAL_STOP],
        "selected_rows": EVAL_STOP - EVAL_START,
        "selected_slice_sha256": slice_hash,
    }


def build_environment(environ: Mapping[str, str]) -> dict[str, str]:
    result = {
        name: value
        for name, value in environ.items()
        if not name.startswith(("CLASP_", "PD_", "GENERATION_", "GRPO_", "WANDB_"))
    }
    for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        result.pop(name, None)
    result.update(EVALUATION_ENV)
    return result


def build_eval_command(stage: str, model_path: Path, inference_root: Path) -> list[str]:
    if stage not in {"sft", "grpo"}:
        fail(f"unsupported evaluation stage: {stage}")
    return [
        str(PYTHON),
        "-m",
        "clasp.entrypoints.evaluate",
        "--tasks",
        "tsp",
        "--stage",
        stage,
        "--model_path",
        str(model_path.resolve()),
        "--tokenizer_path",
        str(SFT_MODEL.resolve()),
        "--data_root",
        str(EVAL_DATA_ROOT.resolve()),
        "--split",
        "eval",
        "--evaluation_purpose",
        "development",
        "--prompt_style",
        "alpaca",
        "--force_no_reason_prompt",
        "--dtype",
        "bfloat16",
        "--seed",
        "271828",
        "--no-prefer_subset_files",
        "--max_eval_samples_per_task",
        "200",
        "--eval_start_index",
        "0",
        "--size_bucket",
        "all",
        "--max_prompt_length",
        "29768",
        "--max_new_tokens",
        "3000",
        "--num_return_sequences",
        "8",
        "--generation_batch_size",
        "8",
        "--do_sample",
        "--temperature",
        "0.7",
        "--top_p",
        "0.9",
        "--top_k",
        "50",
        "--repetition_penalty",
        "1",
        "--inference_root",
        str(inference_root.resolve()),
        "--eval_name",
        "contracted_eval200_k8_seed271828_v1",
        "--no-update_best_run",
        "--save_details",
        "--save_results",
        "--no-resume",
    ]


def build_repair_command(details: Path, config: Path, output: Path, system_name: str) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "clasp.entrypoints.repair",
        "--details",
        str(details.resolve()),
        "--source_config",
        str(config.resolve()),
        "--output",
        str(output.resolve()),
        "--system_name",
        system_name,
        "--tsp_max_passes",
        "12",
    ]


def build_compare_command(
    left: Path,
    left_companion: Path,
    left_label: str,
    right: Path,
    right_companion: Path,
    right_label: str,
    output: Path,
) -> list[str]:
    return [
        str(PYTHON),
        str((PROJECT_ROOT / "scripts" / "compare_development.py").resolve()),
        "--left",
        str(left.resolve()),
        "--left-companion",
        str(left_companion.resolve()),
        "--left-label",
        left_label,
        "--right",
        str(right.resolve()),
        "--right-companion",
        str(right_companion.resolve()),
        "--right-label",
        right_label,
        "--output",
        str(output.resolve()),
    ]


def render_command(command: Sequence[str]) -> str:
    env_text = " ".join(
        f"{name}={shlex.quote(value)}" for name, value in sorted(EVALUATION_ENV.items())
    )
    return env_text + " " + " ".join(shlex.quote(value) for value in command)


def probe_runtime() -> dict[str, str]:
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        fail(f"audited local Python is missing or not executable: {PYTHON}")
    code = (
        "import json,platform,torch,transformers,trl,peft,datasets,bitsandbytes;"
        "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
        "'transformers':transformers.__version__,'trl':trl.__version__,'peft':peft.__version__,"
        "'datasets':datasets.__version__,'bitsandbytes':bitsandbytes.__version__,"
        "'cuda':torch.version.cuda or 'none'}))"
    )
    completed = subprocess.run(
        [str(PYTHON), "-c", code],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        fail(f"cannot probe local runtime: {completed.stderr.strip()}")
    try:
        observed = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        fail(f"runtime probe returned invalid JSON: {exc}")
    if observed != EXPECTED_RUNTIME:
        fail(f"runtime version drift: expected={EXPECTED_RUNTIME}, observed={observed}")
    return observed


def validate_gpu3() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        fail(f"nvidia-smi failed: {completed.stderr.strip()}")
    selected = None
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4 and fields[0] == "3":
            selected = fields
            break
    if selected is None:
        fail("physical GPU index 3 is unavailable")
    try:
        total_mib, used_mib = int(selected[2]), int(selected[3])
    except ValueError:
        fail(f"invalid nvidia-smi memory values for GPU 3: {selected}")
    if "A6000" not in selected[1] or total_mib < 45_000 or used_mib > 1_024:
        fail(f"GPU 3 must be an idle 48 GB RTX A6000; observed={selected}")
    return {
        "physical_index": 3,
        "name": selected[1],
        "memory_total_mib": total_mib,
        "memory_used_mib_at_launch": used_mib,
        "cuda_visible_devices": "3",
    }


def _write_self_hashed_create_once(path: Path, payload: dict[str, Any], field: str) -> Path:
    sealed = dict(payload)
    sealed[field] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    except FileExistsError:
        fail(f"create-once receipt already exists: {path}")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                sealed,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                default=str,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o440)
    except Exception as exc:
        fail(f"receipt creation failed; preserve partial lane for audit: {exc}")
    return path


def reserve_lane(payload: dict[str, Any]) -> Path:
    if LANE_ROOT.exists() or LANE_ROOT.is_symlink():
        fail(f"create-once evaluation lane already exists and cannot be adopted: {LANE_ROOT}")
    LANE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    try:
        LANE_ROOT.mkdir(mode=0o750)
    except FileExistsError:
        fail(f"evaluation lane appeared during reservation: {LANE_ROOT}")
    return _write_self_hashed_create_once(
        LANE_ROOT / "evaluation_lane_contract.json", payload, "contract_sha256"
    )


def _run_command(name: str, command: Sequence[str], env: Mapping[str, str]) -> dict[str, Any]:
    log_path = LANE_ROOT / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(
                list(command),
                cwd=PROJECT_ROOT,
                env=dict(env),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
    except FileExistsError:
        fail(f"create-once command log already exists: {log_path}")
    if completed.returncode != 0:
        fail(f"pipeline command {name} failed with code {completed.returncode}; see {log_path}")
    return {
        "name": name,
        "argv": list(command),
        "rendered": render_command(command),
        "returncode": int(completed.returncode),
        "log": artifact_identity(log_path, relative_to=LANE_ROOT),
    }


def validate_raw_evaluation(root: Path, label: str) -> dict[str, Any]:
    configs = sorted(root.rglob("run_config__*.json"))
    details_files = sorted(root.rglob("run_details__*.jsonl"))
    if len(configs) != 1 or len(details_files) != 1:
        fail(f"{label} evaluation did not produce exactly one config/details pair")
    config_path, details_path = configs[0], details_files[0]
    config = load_json_object(config_path, f"{label} run config")
    _validate_self_hash(config, "run_config_sha256", label=f"{label}_run_config")
    _require_fields(
        config,
        {
            "run_status": "complete",
            "evaluation_purpose": "development",
            "tasks": ["tsp"],
            "split": "eval",
        },
        f"{label}.run_config",
    )
    generation = config.get("generation")
    limits = config.get("limits")
    prompting = config.get("prompting")
    if not all(isinstance(value, dict) for value in (generation, limits, prompting)):
        fail(f"{label} run config lacks generation/limits/prompting contracts")
    _require_fields(
        generation,
        {
            "seed": 271828,
            "seed_policy": "per_example_sha256_v1",
            "dtype": "bfloat16",
            "max_new_tokens": 3000,
            "max_prompt_length": 29768,
            "num_return_sequences": 8,
            "generation_batch_size": 8,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "best_of_n": 8,
        },
        f"{label}.generation",
    )
    _require_fields(
        limits,
        {
            "max_eval_samples_per_task": 200,
            "eval_start_index": 0,
            "size_bucket": "all",
            "prefer_subset_files": False,
            "resume": False,
        },
        f"{label}.limits",
    )
    _require_fields(
        prompting,
        {
            "no_reason": True,
            "prompt_style_resolved": "alpaca",
            "tokenizer_path": str(SFT_MODEL.resolve()),
        },
        f"{label}.prompting",
    )
    provenance = config.get("experiment_provenance")
    data = provenance.get("data") if isinstance(provenance, dict) else None
    files = data.get("files") if isinstance(data, dict) else None
    if (
        not isinstance(files, list)
        or len(files) != 1
        or not (
            files[0].get("path") == "tsp/eval/test.json"
            and files[0].get("sha256") == EVAL_DATA_SHA256
        )
    ):
        fail(f"{label} provenance is not bound only to the audited TSP eval file")
    raw = details_path.read_bytes()
    record_count = sum(1 for line in raw.splitlines() if line.strip())
    details_artifact = config.get("details_artifact")
    if not isinstance(details_artifact, dict) or not (
        details_artifact.get("record_count") == 200
        and details_artifact.get("record_count") == record_count
        and details_artifact.get("size_bytes") == len(raw)
        and details_artifact.get("sha256") == hashlib.sha256(raw).hexdigest()
    ):
        fail(f"{label} details artifact is incomplete or disagrees with its run config")
    protocol_fingerprint = config.get("protocol_fingerprint")
    if not isinstance(protocol_fingerprint, str) or len(protocol_fingerprint) != 64:
        fail(f"{label} protocol fingerprint is invalid")
    return {
        "details_path": details_path,
        "config_path": config_path,
        "details": artifact_identity(details_path, relative_to=LANE_ROOT),
        "config": artifact_identity(config_path, relative_to=LANE_ROOT),
        "experiment_fingerprint": config.get("experiment_fingerprint"),
        "protocol_fingerprint": protocol_fingerprint,
        "records": record_count,
    }


def lane_artifact_manifest() -> dict[str, Any]:
    final_receipt = LANE_ROOT / "evaluation_receipt.json"
    files = []
    for path in sorted(LANE_ROOT.rglob("*")):
        if path == final_receipt:
            continue
        if path.is_symlink():
            fail(f"evaluation lane contains a symlink: {path}")
        if path.is_file():
            files.append(artifact_identity(path, relative_to=LANE_ROOT))
        elif not path.is_dir():
            fail(f"evaluation lane contains a non-regular entry: {path}")
    return {
        "file_count": len(files),
        "files": files,
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }


def _pipeline_plan() -> dict[str, list[str]]:
    raw_sft_root = LANE_ROOT / "raw" / "sft"
    raw_dense_root = LANE_ROOT / "raw" / "dense40"
    return {
        "eval_sft": build_eval_command("sft", SFT_MODEL, raw_sft_root),
        "eval_dense40": build_eval_command("grpo", CHECKPOINT, raw_dense_root),
    }


def execute_pipeline(
    training: dict[str, Any],
    eval_data: dict[str, Any],
    runtime: dict[str, str],
    code: dict[str, Any],
    gpu: dict[str, Any],
) -> Path:
    env = build_environment(os.environ)
    planned = _pipeline_plan()
    lane_contract_payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "reserved_development_only",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "lane_root": str(LANE_ROOT.resolve()),
        "training": training,
        "evaluation_data": eval_data,
        "runtime": runtime,
        "code": code,
        "gpu": gpu,
        "evaluation_environment": dict(sorted(EVALUATION_ENV.items())),
        "planned_commands": {
            name: {"argv": command, "rendered": render_command(command)}
            for name, command in planned.items()
        },
        "create_once": {
            "resume": False,
            "overwrite": False,
            "partial_lane_reuse": False,
        },
        "limitations": LIMITATIONS,
    }
    contract_path = reserve_lane(lane_contract_payload)
    command_receipts = []
    command_receipts.append(_run_command("01_eval_sft", planned["eval_sft"], env))
    sft_raw = validate_raw_evaluation(LANE_ROOT / "raw" / "sft", "sft")
    command_receipts.append(_run_command("02_eval_dense40", planned["eval_dense40"], env))
    dense_raw = validate_raw_evaluation(LANE_ROOT / "raw" / "dense40", "dense40")
    if sft_raw["protocol_fingerprint"] != dense_raw["protocol_fingerprint"]:
        fail("SFT and dense40 raw evaluations do not have an identical protocol fingerprint")

    repair_dir = LANE_ROOT / "repair"
    repair_dir.mkdir()
    sft_repair = repair_dir / "sft_repair.jsonl"
    dense_repair = repair_dir / "dense40_repair.jsonl"
    repair_commands = {
        "03_repair_sft": build_repair_command(
            sft_raw["details_path"],
            sft_raw["config_path"],
            sft_repair,
            "contracted_tsp_sft_repair_v1",
        ),
        "04_repair_dense40": build_repair_command(
            dense_raw["details_path"],
            dense_raw["config_path"],
            dense_repair,
            "contracted_tsp_dense40_repair_v1",
        ),
    }
    for name, command in repair_commands.items():
        command_receipts.append(_run_command(name, command, env))
    sft_repair_summary = repair_dir / "sft_repair__summary.json"
    dense_repair_summary = repair_dir / "dense40_repair__summary.json"
    for path in (sft_repair, sft_repair_summary, dense_repair, dense_repair_summary):
        if not path.is_file():
            fail(f"repair lane did not produce expected artifact: {path}")

    comparisons = LANE_ROOT / "comparisons"
    comparisons.mkdir()
    compare_commands = {
        "05_compare_raw_sft_dense40": build_compare_command(
            sft_raw["details_path"],
            sft_raw["config_path"],
            "sft_raw_k8",
            dense_raw["details_path"],
            dense_raw["config_path"],
            "clasp_dense40_raw_k8",
            comparisons / "raw_sft_vs_dense40.json",
        ),
        "06_compare_repaired_sft_dense40": build_compare_command(
            sft_repair,
            sft_repair_summary,
            "sft_repaired_k8",
            dense_repair,
            dense_repair_summary,
            "clasp_dense40_repaired_k8",
            comparisons / "repaired_sft_vs_dense40.json",
        ),
        "07_compare_sft_raw_repair": build_compare_command(
            sft_raw["details_path"],
            sft_raw["config_path"],
            "sft_raw_k8",
            sft_repair,
            sft_repair_summary,
            "sft_repaired_k8",
            comparisons / "sft_raw_vs_repair.json",
        ),
        "08_compare_dense40_raw_repair": build_compare_command(
            dense_raw["details_path"],
            dense_raw["config_path"],
            "clasp_dense40_raw_k8",
            dense_repair,
            dense_repair_summary,
            "clasp_dense40_repaired_k8",
            comparisons / "dense40_raw_vs_repair.json",
        ),
    }
    for name, command in compare_commands.items():
        command_receipts.append(_run_command(name, command, env))

    # A multi-hour evaluation must fail closed if any trusted input or code was
    # changed while it ran.
    training_after = verify_complete_training_run()
    eval_after = stage_or_verify_eval_data(execute=False)
    code_after = code_manifest()
    if training_after != training or eval_after["staged"] != eval_data["staged"]:
        fail("training or staged evaluation artifacts changed during the evaluation lane")
    if code_after != code:
        fail("evaluation/repair/comparison code changed during the evaluation lane")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete_development_only_not_publication_eligible",
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "lane_contract": artifact_identity(contract_path, relative_to=LANE_ROOT),
        "training": training,
        "evaluation_data": eval_data,
        "runtime": runtime,
        "code": code,
        "gpu": gpu,
        "matched_protocol_fingerprint": sft_raw["protocol_fingerprint"],
        "raw_evaluations": {"sft": sft_raw, "dense40": dense_raw},
        "commands": command_receipts,
        "artifacts": lane_artifact_manifest(),
        "comparisons": [
            "comparisons/raw_sft_vs_dense40.json",
            "comparisons/repaired_sft_vs_dense40.json",
            "comparisons/sft_raw_vs_repair.json",
            "comparisons/dense40_raw_vs_repair.json",
        ],
        "limitations": LIMITATIONS,
    }
    return _write_self_hashed_create_once(
        LANE_ROOT / "evaluation_receipt.json", payload, "receipt_sha256"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create-once DEVELOPMENT-ONLY TSP dense40 post-training evaluation."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Stage audited eval bytes, reserve the lane, and run GPU evaluation/repair/comparison.",
    )
    parser.add_argument(
        "--acknowledge-development-only",
        action="store_true",
        help="Required with --execute; results cannot support publication or SOTA claims.",
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=int,
        default=0,
        help="Optionally wait for COMPLETE step-40 state; zero refuses immediately.",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.wait_timeout_seconds < 0 or args.poll_seconds < 1:
        fail("wait timeout must be >=0 and poll interval must be >=1")
    if args.execute and not args.acknowledge_development_only:
        fail("--execute requires --acknowledge-development-only")
    if LANE_ROOT.exists() or LANE_ROOT.is_symlink():
        fail(f"create-once evaluation lane already exists and cannot be adopted: {LANE_ROOT}")

    training = wait_for_complete_training(args.wait_timeout_seconds, args.poll_seconds)
    eval_data = stage_or_verify_eval_data(execute=args.execute)
    runtime = probe_runtime()
    code = code_manifest()
    plan = _pipeline_plan()
    summary = {
        "protocol": PROTOCOL,
        "status": "ready_development_only" if eval_data["staged"] else "ready_to_stage",
        "lane_root": str(LANE_ROOT),
        "training_checkpoint": str(CHECKPOINT),
        "eval_source_sha256": EVAL_DATA_SHA256,
        "eval_interval": [EVAL_START, EVAL_STOP],
        "development_only": True,
        "publication_eligible": False,
        "commands": {name: render_command(command) for name, command in plan.items()},
    }
    if not args.execute:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if eval_data["staged"] is None:
        fail("audited eval data could not be staged")
    gpu = validate_gpu3()
    receipt = execute_pipeline(training, eval_data, runtime, code, gpu)
    print(f"[LocalDense40Evaluation] receipt={receipt}")
    print("[LocalDense40Evaluation] development_only=true publication_eligible=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LocalEvaluationError, TrainingNotComplete) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
