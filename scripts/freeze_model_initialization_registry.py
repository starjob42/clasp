#!/usr/bin/env python3
"""Freeze the exact base and five matched specialist SFT starts before RL training."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Sequence

from prepare_sota_run import (  # type: ignore[import-not-found]
    TASKS,
    build_model_initialization_registry,
)


def _parse_specialist(value: str) -> tuple[str, dict[str, Any]]:
    parts = value.split(",", 3)
    if len(parts) != 4 or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "specialist must use TASK,MODEL_PATH,SFT_DATA_PATH,SFT_SEED"
        )
    task, model_path, sft_data_path, raw_seed = (part.strip() for part in parts)
    if task not in TASKS:
        raise argparse.ArgumentTypeError(f"unsupported specialist task {task!r}")
    if not raw_seed.isdecimal() or (len(raw_seed) > 1 and raw_seed.startswith("0")):
        raise argparse.ArgumentTypeError("specialist SFT seed must be canonical decimal")
    return task, {
        "model_path": Path(model_path),
        "sft_data_path": Path(sft_data_path),
        "sft_seed": int(raw_seed),
    }


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _write_create_once(path: Path, payload: object) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite model-initialization registry: {destination}"
            ) from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--forbidden-source-registry", type=Path, required=True)
    parser.add_argument(
        "--base-sft-campaign",
        type=Path,
        required=True,
        help="Frozen five-task non-instruction base-SFT campaign receipt.",
    )
    parser.add_argument(
        "--specialist",
        action="append",
        required=True,
        type=_parse_specialist,
        help="Required exactly once per task: TASK,MODEL_PATH,SFT_DATA_PATH,SFT_SEED.",
    )
    parser.add_argument(
        "--frozen-at-utc",
        help="Canonical YYYY-MM-DDTHH:MM:SSZ timestamp; defaults to current UTC.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    specialists: dict[str, dict[str, Any]] = {}
    for task, spec in args.specialist:
        if task in specialists:
            raise ValueError(f"duplicate specialist specification for {task}")
        specialists[task] = spec

    # Operational publication freezing is allowed only for the distinct
    # all-five-task base campaign.  This deliberately excludes the historical
    # Instruct specialists even if their local hyperparameter files otherwise
    # resemble the matched design.
    from freeze_base_sft_campaign import (
        TASKS as BASE_SFT_TASKS,
    )
    from freeze_base_sft_campaign import (
        load_campaign,
        verify_campaign_runtime,
    )
    from validate_base_sft500k_specialist import resolve_state as resolve_base_sft_state

    if set(specialists) != set(BASE_SFT_TASKS):
        raise ValueError("base-SFT campaign specialists must exactly cover all five tasks")
    campaign, _campaign_file_sha256 = load_campaign(args.base_sft_campaign)
    data_roots = {
        Path(spec["sft_data_path"]).expanduser().resolve().parent.parent
        for spec in specialists.values()
    }
    if len(data_roots) != 1:
        raise ValueError("base-SFT specialist data paths do not share one frozen data root")
    data_root = next(iter(data_roots))
    output_root = Path(campaign["output_root"])
    source_root = Path(campaign["source_snapshot"]["canonical_path"])
    reserved_source_registry = Path(
        campaign["sft_reserved_source_registry"]["canonical_path"]
    )
    verify_campaign_runtime(
        campaign_path=args.base_sft_campaign,
        source_root=source_root,
        base_model_path=args.base_model_path,
        forbidden_source_registry=args.forbidden_source_registry,
        reserved_source_registry=reserved_source_registry,
        sft_data_root=data_root,
        output_root=output_root,
        tasks=BASE_SFT_TASKS,
        require_output_absent=False,
    )
    for task in BASE_SFT_TASKS:
        spec = specialists[task]
        model_path = Path(spec["model_path"]).expanduser().resolve()
        expected_model_path = output_root / campaign["run_relative_template"].format(task=task)
        if model_path != expected_model_path:
            raise ValueError(
                f"{task} specialist is outside its frozen base-SFT campaign cell: "
                f"actual={model_path}, expected={expected_model_path}"
            )
        state = resolve_base_sft_state(
            campaign_path=args.base_sft_campaign,
            source_root=source_root,
            base_model_path=args.base_model_path,
            forbidden_source_registry=args.forbidden_source_registry,
            reserved_source_registry=reserved_source_registry,
            data_root=data_root,
            output_root=output_root,
            task=task,
            run_dir=model_path,
            initialize_run_binding=False,
        )
        if not state.startswith("COMPLETE:"):
            raise ValueError(f"{task} base-SFT campaign cell is not complete: {state}")
    registry = build_model_initialization_registry(
        base_model_path=args.base_model_path,
        specialists=specialists,
        forbidden_source_registry=args.forbidden_source_registry,
        frozen_at_utc=args.frozen_at_utc,
    )
    _write_create_once(args.output, registry)
    print(f"frozen model-initialization registry: {args.output.expanduser().absolute()}")
    print(f"registry_sha256: {registry['registry_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
