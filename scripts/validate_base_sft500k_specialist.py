#!/usr/bin/env python3
"""Prepare, resume, and certify one frozen Qwen2.5-7B-base SFT cell.

The script is dependency-light and is intended to run on both login and compute
nodes.  Unlike the historical missing-specialist validator, it accepts exactly
the complete five-task campaign and requires a frozen campaign receipt on every
invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audit_lora_persistence import (  # type: ignore[import-not-found]
    LORA_BIAS_POLICY,
    audit_adapter_persistence,
    audit_qwen25_base_bias_scope,
    validate_audit_receipt,
)
from audit_sft_reload_equivalence import (  # type: ignore[import-not-found]
    RECEIPT_FILENAME as RELOAD_EQUIVALENCE_FILENAME,
)
from audit_sft_reload_equivalence import validate_reload_receipt
# Retained as an importable historical diagnostic surface for forensic
# inspection of old four-rank artifacts. Publication v9 uses the separate,
# exact single-H100 receipt below.
import base_sft_resume_fidelity as _legacy_resume_fidelity  # type: ignore[import-not-found]  # noqa: F401,E402
from base_sft_single_h100_resume_fidelity import (  # type: ignore[import-not-found]
    RECEIPT_FILENAME as SINGLE_H100_RESUME_RECEIPT_FILENAME,
)
from base_sft_single_h100_resume_fidelity import resume_state_identity
from freeze_base_sft_campaign import (  # type: ignore[import-not-found]
    BASE_SFT_TRAINING_DESIGN,
    MINIMUM_CELL_FREE_BYTES,
    MINIMUM_MATRIX_FREE_BYTES,
    TASKS,
    expected_run_contract,
    sha256_json,
    verify_campaign_runtime,
)
from seal_completed_sft_run import (  # type: ignore[import-not-found]
    require_exact_sealed_modes,
    validate_completed_run_artifacts,
    validate_seal_receipt,
)
from prepare_sota_run import (  # type: ignore[import-not-found]
    JIANG_LORA_MODULES,
    validate_qwen25_7b_base,
)

from clasp.models.publication_fidelity import publication_fidelity_identity  # noqa: E402
from clasp.entrypoints.train_sft_jiang_f638 import (  # noqa: E402
    FIXED_RECORDED_CLI_ARGS,
)

TARGET_STEP = 31_250
WORLD_SIZE = 1
CHUNK_BOUNDARIES = (5_000, 10_000, 15_000, 20_000, 25_000, 30_000, TARGET_STEP)
RUN_CONTRACT_FILENAME = "base_sft_run_contract.json"
CHECKPOINT_RE = re.compile(r"checkpoint-([1-9][0-9]*)")
MODEL_FILES = ("adapter_model.safetensors", "adapter_model.bin")
REQUIRED_CHECKPOINT_FILES = (
    "adapter_config.json",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _canonical(path: Path, *, label: str, must_exist: bool) -> Path:
    raw = path.expanduser().absolute()
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {raw}")
    resolved = raw.resolve(strict=must_exist)
    if raw != resolved:
        raise ValueError(f"{label} must be canonical: supplied={raw}, resolved={resolved}")
    return resolved


def _write_create_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace base-SFT run contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to replace base-SFT run contract: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _expected_run_contract(
    *,
    campaign: Mapping[str, Any],
    campaign_file_sha256: str,
    task: str,
    run_dir: Path,
) -> dict[str, Any]:
    expected = expected_run_contract(
        campaign=campaign,
        campaign_file_sha256=campaign_file_sha256,
        task=task,
    )
    if expected["run_dir"] != str(run_dir):
        raise ValueError("authoritative run contract targets a different run directory")
    return expected


def _initialize_or_validate_run_contract(
    *,
    campaign: Mapping[str, Any],
    campaign_file_sha256: str,
    task: str,
    run_dir: Path,
    initialize: bool,
) -> None:
    expected = _expected_run_contract(
        campaign=campaign,
        campaign_file_sha256=campaign_file_sha256,
        task=task,
        run_dir=run_dir,
    )
    contract_path = run_dir / RUN_CONTRACT_FILENAME
    if not run_dir.exists():
        if not initialize:
            return
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_create_once(contract_path, expected)
        if os.path.samefile(contract_path, Path(expected["authoritative_path"])):
            raise ValueError("mutable run-contract copy must not hardlink sealed authority")
        return
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError(f"base-SFT run path is not a regular directory: {run_dir}")
    if not contract_path.is_file() or contract_path.is_symlink():
        raise ValueError(f"base-SFT run directory lacks {RUN_CONTRACT_FILENAME}")
    if stat.S_IMODE(contract_path.stat().st_mode) & 0o222:
        raise ValueError("base-SFT run contract is writable")
    payload = _load_object(contract_path)
    stored_hash = payload.get("run_contract_sha256")
    unsigned = dict(payload)
    unsigned.pop("run_contract_sha256", None)
    if stored_hash != sha256_json(unsigned):
        raise ValueError("base-SFT run-contract self-hash mismatch")
    if payload != expected:
        raise ValueError("base-SFT run contract differs from the frozen campaign cell")
    authority_path = Path(expected["authoritative_path"])
    if contract_path.read_bytes() != authority_path.read_bytes():
        raise ValueError("mutable run-contract copy differs from sealed authority bytes")
    if os.path.samefile(contract_path, authority_path):
        raise ValueError("mutable run-contract copy must use a distinct inode from authority")


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}={actual!r}, expected {expected!r}")


def _validate_adapter(path: Path, base_model_path: Path) -> None:
    config = _load_object(path / "adapter_config.json")
    expected = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 64,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "bias": LORA_BIAS_POLICY,
    }
    for key, value in expected.items():
        _require_equal(config.get(key), value, f"adapter.{key}")
    _require_equal(
        set(config.get("target_modules") or []),
        JIANG_LORA_MODULES,
        "adapter.target_modules",
    )
    configured_base = Path(str(config.get("base_model_name_or_path", ""))).resolve()
    _require_equal(configured_base, base_model_path, "adapter.base_model_name_or_path")
    audit_adapter_persistence(path)


def _validate_checkpoint_fidelity(path: Path) -> dict[str, Any]:
    """Require the live, exact publication receipt produced by train_sft."""

    try:
        return publication_fidelity_identity(path)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} lacks live publication checkpoint fidelity: {exc}") from exc


def _validate_hyperparams(
    *,
    run_dir: Path,
    task: str,
    base_model_path: Path,
    data_root: Path,
) -> None:
    payload = _load_object(run_dir / "hyperparams.json")
    _require_equal(payload.get("stage"), "sft", "stage")
    _require_equal(payload.get("tasks"), [task], "tasks")
    _require_equal(
        Path(str(payload.get("model_path", ""))).resolve(),
        base_model_path,
        "model_path",
    )
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("hyperparams.json has no dataset object")
    _require_equal(dataset.get("train_size"), 500_000, "dataset.train_size")
    _require_equal(dataset.get("eval_size"), 0, "dataset.eval_size")
    _require_equal(dataset.get("train_task_counts"), {task: 500_000}, "dataset counts")

    stage = payload.get("stage_hyperparameters")
    if not isinstance(stage, dict):
        raise ValueError("hyperparams.json has no stage_hyperparameters object")
    expected_stage = BASE_SFT_TRAINING_DESIGN["stage_hyperparameters"]
    stage_only_fields = set(expected_stage) - {
        "logging_steps",
        "save_steps",
        "eval_steps",
        "save_total_limit",
    }
    for key in sorted(stage_only_fields):
        _require_equal(stage.get(key), expected_stage[key], f"stage_hyperparameters.{key}")
    _require_equal(
        Path(str(stage.get("tokenizer_path", ""))).resolve(),
        base_model_path,
        "stage_hyperparameters.tokenizer_path",
    )
    _require_equal(
        stage.get("tokenization_alignment_train_mismatches"),
        0,
        "training tokenization mismatches",
    )
    _require_equal(
        stage.get("tokenization_alignment_eval_mismatches"),
        0,
        "evaluation tokenization mismatches",
    )

    args = payload.get("args")
    if not isinstance(args, dict):
        raise ValueError("hyperparams.json has no args object")
    exact_args = {
        "tasks": [task],
        **FIXED_RECORDED_CLI_ARGS,
    }
    for key, value in exact_args.items():
        _require_equal(args.get(key), value, f"args.{key}")
    _require_equal(Path(str(args.get("data_root", ""))).resolve(), data_root, "args.data_root")
    _require_equal(
        Path(str(args.get("model_path", ""))).resolve(), base_model_path, "args.model_path"
    )
    _require_equal(
        Path(str(args.get("tokenizer_path", ""))).resolve(),
        base_model_path,
        "args.tokenizer_path",
    )
    distributed = payload.get("distributed")
    if not isinstance(distributed, dict):
        raise ValueError("hyperparams.json has no distributed object")
    _require_equal(distributed.get("world_size"), WORLD_SIZE, "distributed.world_size")


def _strict_checkpoint_step(
    checkpoint: Path,
    base_model_path: Path,
    *,
    campaign_path: Path,
    run_contract_path: Path,
    require_live_accelerator: bool,
) -> int | None:
    match = CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None:
        return None
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError(f"checkpoint path is not a regular directory: {checkpoint}")
    step = int(match.group(1))
    if step not in CHUNK_BOUNDARIES:
        raise ValueError(f"checkpoint step {step} is not an approved v9 chunk boundary")
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"checkpoint {checkpoint} is missing: {', '.join(missing)}")
    if not any((checkpoint / name).is_file() for name in MODEL_FILES):
        raise ValueError(f"checkpoint {checkpoint} has no adapter model artifact")
    has_single_process_rng = (checkpoint / "rng_state.pth").is_file()
    has_legacy_four_rank_rng = all(
        (checkpoint / f"rng_state_{rank}.pth").is_file() for rank in range(4)
    )
    missing_rng = (
        []
        if has_single_process_rng or has_legacy_four_rank_rng
        else ["rng_state.pth"]
    )
    if missing_rng:
        raise ValueError(f"checkpoint {checkpoint} is missing: {', '.join(missing_rng)}")
    state = _load_object(checkpoint / "trainer_state.json")
    _require_equal(int(state.get("global_step", -1)), step, "trainer_state.global_step")
    _require_equal(int(state.get("max_steps", -1)), TARGET_STEP, "trainer_state.max_steps")
    _validate_adapter(checkpoint, base_model_path)
    _validate_checkpoint_fidelity(checkpoint)
    resume_identity = resume_state_identity(
        campaign_path=campaign_path,
        run_contract_path=run_contract_path,
        checkpoint=checkpoint.resolve(strict=True),
        require_accelerator=require_live_accelerator,
    )
    _require_equal(resume_identity.get("global_step"), step, "resume receipt global_step")
    _require_equal(resume_identity.get("world_size"), WORLD_SIZE, "resume receipt world_size")
    return step


def _is_finalized(run_dir: Path, checkpoint: Path, base_model_path: Path) -> bool:
    if not any((run_dir / name).is_file() for name in MODEL_FILES):
        return False
    try:
        status = _load_object(run_dir / "stage_status.json")
        _validate_adapter(run_dir, base_model_path)
        _validate_checkpoint_fidelity(run_dir)
        validate_audit_receipt(
            run_dir / "adapter_persistence_receipt.json",
            base_model_path=base_model_path,
            adapter_path=run_dir,
        )
        validate_reload_receipt(
            run_dir / RELOAD_EQUIVALENCE_FILENAME,
            base_model_path=base_model_path,
            adapter_path=run_dir,
        )
        validate_completed_run_artifacts(run_dir, base_model_path)
        validate_seal_receipt(run_dir, base_model_path)
        require_exact_sealed_modes(run_dir)
    except ValueError:
        return False
    return (
        status.get("stage") == "sft"
        and status.get("completed") is True
        and status.get("has_saved_model") is True
        and int(status.get("global_step", -1)) == TARGET_STEP
        and int(status.get("max_steps", -1)) == TARGET_STEP
        and Path(str(status.get("latest_checkpoint_dir", ""))).resolve() == checkpoint
    )


def resolve_state(
    *,
    campaign_path: Path,
    immutable_input_trust_root: Path | None = None,
    source_root: Path,
    base_model_path: Path,
    forbidden_source_registry: Path,
    reserved_source_registry: Path | None = None,
    remote_environment_receipt: Path | None = None,
    data_root: Path,
    output_root: Path,
    task: str,
    run_dir: Path,
    initialize_run_binding: bool,
    require_live_accelerator: bool = False,
) -> str:
    if task not in TASKS:
        raise ValueError(f"task must be one of {list(TASKS)}, got {task!r}")
    campaign, campaign_file_sha256 = verify_campaign_runtime(
        campaign_path=campaign_path,
        immutable_input_trust_root=immutable_input_trust_root,
        source_root=source_root,
        base_model_path=base_model_path,
        forbidden_source_registry=forbidden_source_registry,
        reserved_source_registry=reserved_source_registry,
        remote_environment_receipt=remote_environment_receipt,
        sft_data_root=data_root,
        output_root=output_root,
        tasks=[task],
        require_output_absent=False,
        require_accelerator=require_live_accelerator,
    )
    base = _canonical(base_model_path, label="base model", must_exist=True)
    data = _canonical(data_root, label="SFT data root", must_exist=True)
    output = _canonical(output_root, label="base-SFT output root", must_exist=True)
    run = _canonical(run_dir, label="base-SFT run directory", must_exist=False)
    expected_run = output / campaign["run_relative_template"].format(task=task)
    if run != expected_run:
        raise ValueError(f"run directory is not the frozen campaign cell: {run} != {expected_run}")
    validate_qwen25_7b_base(base)
    audit_qwen25_base_bias_scope(base)
    _initialize_or_validate_run_contract(
        campaign=campaign,
        campaign_file_sha256=campaign_file_sha256,
        task=task,
        run_dir=run,
        initialize=initialize_run_binding,
    )
    if not run.exists():
        return "FRESH"
    hyperparams = run / "hyperparams.json"
    if not hyperparams.is_file():
        unexpected = sorted(
            candidate.name for candidate in run.iterdir() if candidate.name != RUN_CONTRACT_FILENAME
        )
        if unexpected:
            raise ValueError(
                "uninitialized base-SFT run contains unexpected artifacts: " + ", ".join(unexpected)
            )
        return "FRESH"
    _validate_hyperparams(
        run_dir=run,
        task=task,
        base_model_path=base,
        data_root=data,
    )
    checkpoint_root = run / "checkpoints"
    if not checkpoint_root.is_dir() or checkpoint_root.is_symlink():
        raise ValueError(f"checkpoint directory is missing or unsafe: {checkpoint_root}")
    candidates: list[tuple[int, Path]] = []
    entries = sorted(checkpoint_root.iterdir(), key=lambda item: item.name)
    unexpected = [entry.name for entry in entries if CHECKPOINT_RE.fullmatch(entry.name) is None]
    if unexpected:
        raise ValueError(
            "checkpoint directory contains unexpected entries: " + ", ".join(unexpected)
        )
    run_contract_path = run / RUN_CONTRACT_FILENAME
    for checkpoint in entries:
        step = _strict_checkpoint_step(
            checkpoint,
            base,
            campaign_path=campaign_path,
            run_contract_path=run_contract_path,
            require_live_accelerator=require_live_accelerator,
        )
        if step is not None:
            candidates.append((step, checkpoint.resolve()))
    if not candidates:
        raise ValueError("no optimizer-resumable checkpoint satisfies the base-SFT contract")
    step, checkpoint = max(candidates)
    expected_steps = [boundary for boundary in CHUNK_BOUNDARIES if boundary <= step]
    observed_steps = sorted(candidate_step for candidate_step, _ in candidates)
    if observed_steps != expected_steps:
        raise ValueError(
            "checkpoint inventory is not the exact certified boundary prefix: "
            f"observed={observed_steps}, expected={expected_steps}"
        )
    if _is_finalized(run, checkpoint, base):
        return f"COMPLETE:{run}"
    allowed_partial_root = {RUN_CONTRACT_FILENAME, "hyperparams.json", "checkpoints"}
    root_entries = {entry.name for entry in run.iterdir()}
    unexpected_root = sorted(root_entries - allowed_partial_root)
    if unexpected_root:
        raise ValueError(
            "partial SFT cell contains terminal or stray root artifacts: "
            + ", ".join(unexpected_root)
        )
    receipt = checkpoint / SINGLE_H100_RESUME_RECEIPT_FILENAME
    if not receipt.is_file() or receipt.is_symlink():
        raise ValueError("latest partial checkpoint lacks its single-H100 resume receipt")
    return f"RESUME:{checkpoint}"


def verify_campaign_only(args: argparse.Namespace) -> str:
    campaign, _ = verify_campaign_runtime(
        campaign_path=args.campaign_contract,
        immutable_input_trust_root=args.immutable_input_trust_root,
        source_root=args.source_root,
        base_model_path=args.base_model_path,
        forbidden_source_registry=args.forbidden_source_registry,
        reserved_source_registry=args.reserved_source_registry,
        remote_environment_receipt=args.remote_environment_receipt,
        sft_data_root=args.data_root,
        output_root=args.output_root,
        tasks=TASKS,
        require_output_absent=True,
        require_accelerator=args.require_live_accelerator,
        minimum_free_bytes=(
            MINIMUM_MATRIX_FREE_BYTES if getattr(args, "require_matrix_quota", False) else 0
        ),
    )
    return f"VALID:{campaign['campaign_id']}:{campaign['campaign_sha256']}"


def verify_inputs_only(args: argparse.Namespace) -> str:
    if args.task is None:
        raise ValueError("input-only validation requires --task")
    campaign, _ = verify_campaign_runtime(
        campaign_path=args.campaign_contract,
        immutable_input_trust_root=args.immutable_input_trust_root,
        source_root=args.source_root,
        base_model_path=args.base_model_path,
        forbidden_source_registry=args.forbidden_source_registry,
        reserved_source_registry=args.reserved_source_registry,
        remote_environment_receipt=args.remote_environment_receipt,
        sft_data_root=args.data_root,
        output_root=args.output_root,
        tasks=[args.task],
        require_output_absent=False,
        require_accelerator=args.require_live_accelerator,
        minimum_free_bytes=(
            MINIMUM_CELL_FREE_BYTES if getattr(args, "require_cell_quota", False) else 0
        ),
    )
    return f"INPUTS_VALID:{campaign['campaign_id']}:{campaign['campaign_sha256']}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-contract", type=Path, required=True)
    parser.add_argument("--immutable-input-trust-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--forbidden-source-registry", type=Path, required=True)
    parser.add_argument("--reserved-source-registry", type=Path, required=True)
    parser.add_argument("--remote-environment-receipt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-campaign-only", action="store_true")
    mode.add_argument("--verify-inputs-only", action="store_true")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--initialize-run-binding", action="store_true")
    parser.add_argument("--require-live-accelerator", action="store_true")
    quota = parser.add_mutually_exclusive_group()
    quota.add_argument("--require-matrix-quota", action="store_true")
    quota.add_argument("--require-cell-quota", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_campaign_only:
            if args.task is not None or args.run_dir is not None or args.initialize_run_binding:
                raise ValueError("campaign-only validation does not accept run-cell arguments")
            print(verify_campaign_only(args))
            return 0
        if args.verify_inputs_only:
            if args.run_dir is not None or args.initialize_run_binding:
                raise ValueError("input-only validation does not inspect or initialize a run cell")
            print(verify_inputs_only(args))
            return 0
        if args.task is None or args.run_dir is None:
            raise ValueError("run validation requires --task and --run-dir")
        print(
            resolve_state(
                campaign_path=args.campaign_contract,
                immutable_input_trust_root=args.immutable_input_trust_root,
                source_root=args.source_root,
                base_model_path=args.base_model_path,
                forbidden_source_registry=args.forbidden_source_registry,
                reserved_source_registry=args.reserved_source_registry,
                remote_environment_receipt=args.remote_environment_receipt,
                data_root=args.data_root,
                output_root=args.output_root,
                task=args.task,
                run_dir=args.run_dir,
                initialize_run_binding=args.initialize_run_binding,
                require_live_accelerator=args.require_live_accelerator,
            )
        )
    except (FileExistsError, ValueError) as exc:
        print(f"[BaseSFT500kError] {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
