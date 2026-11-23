"""Exact, rank-owned optimizer/scheduler state for the external-baseline pilot.

Canonical Trainer files remain available for adapter tooling. Continuations
restore each rank's own files and verify their tensor contents after restoration.
No claim of bitwise future trajectory equivalence is made.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


def semantic_state_sha256(state):
    """Hash state types, tensor dtype/shape/bytes, and scalar values, excluding device."""
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous().cpu()
            digest.update(json.dumps(["tensor", str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict[")
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                visit(key)
                visit(value[key])
            digest.update(b"]")
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode() + b"[")
            for item in value:
                visit(item)
            digest.update(b"]")
        elif value is None or type(value) in (str, bool, int, float):
            digest.update(json.dumps([type(value).__name__, value], allow_nan=False).encode())
        else:
            raise TypeError(f"Unsupported optimizer/scheduler state type: {type(value).__name__}")

    visit(state)
    return digest.hexdigest()


def _write_json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_rank_state(directory, optimizer, scheduler, *, rank, world_size, step):
    directory = Path(directory)
    receipt = {"schema_version": 1, "status": "passed", "rank": rank,
               "world_size": world_size, "global_step": step, "states": {}}
    for name, owner in (("optimizer", optimizer), ("scheduler", scheduler)):
        value = owner.state_dict()
        before = semantic_state_sha256(value)
        path = directory / f"{name}_rank{rank}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(value, temporary)
        os.replace(temporary, path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        after = semantic_state_sha256(saved)
        if before != after or before != semantic_state_sha256(owner.state_dict()):
            raise RuntimeError(f"Rank {rank} {name} saved-to-live state differs")
        receipt["states"][name] = {"file": path.name, "semantic_sha256": after,
                                   "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    _write_json_atomic(directory / f"external_rank_state_{rank}.json", receipt)
    return receipt


def restore_rank_state(directory, optimizer, scheduler, *, rank, world_size, step):
    directory = Path(directory)
    receipt = json.loads((directory / f"external_rank_state_{rank}.json").read_text())
    for name, expected in (("schema_version", 1), ("status", "passed"), ("rank", rank),
                           ("world_size", world_size), ("global_step", step)):
        if receipt.get(name) != expected:
            raise ValueError(f"Rank-owned resume receipt differs: {name}")
    for name, owner in (("optimizer", optimizer), ("scheduler", scheduler)):
        binding = receipt["states"][name]
        path = directory / f"{name}_rank{rank}.pt"
        if binding["file"] != path.name:
            raise ValueError("Rank-owned state path differs")
        if hashlib.sha256(path.read_bytes()).hexdigest() != binding["file_sha256"]:
            raise ValueError(f"Rank {rank} {name} file changed since checkpoint")
        value = torch.load(path, map_location="cpu", weights_only=False)
        if semantic_state_sha256(value) != binding["semantic_sha256"]:
            raise ValueError(f"Rank {rank} {name} saved contents differ")
        owner.load_state_dict(value)
        if semantic_state_sha256(owner.state_dict()) != binding["semantic_sha256"]:
            raise RuntimeError(f"Rank {rank} {name} restoration changed saved values")
    return {"schema_version": 1, "status": "passed", "rank": rank,
            "world_size": world_size, "global_step": step,
            "checkpoint": str(directory), "states": receipt["states"]}
