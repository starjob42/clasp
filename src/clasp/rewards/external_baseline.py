"""Past-only pooled baselines for single- or multiple-rollout score estimators.

The caller snapshots globally ordered (task, problem-size) keys *before*
generation, computes advantages with that immutable snapshot, then updates once
with the complete globally gathered rewards. This module performs no gathering,
centering, stochastic sampling, policy update, or current-batch normalization.
Identical global inputs and checkpoint state produce identical state hashes on
every rank. Callers must still check distributed agreement and checkpoint this
state alongside the policy/optimizer at the same update boundary.

The baseline pools all past completions in an inclusive size bucket, backing
off to the task mean, global mean, then an explicit initializer. It therefore
does not require repeated exact prompts. The scale is one fixed positive scalar
for every task/sample. ``baseline_instance_size`` defines input-only size:
TSP cities, CVRP customers (excluding the depot), OP nodes (including its start
node), MIS graph vertices, and JSSP operations. No solution enters this feature.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

SCHEMA_VERSION = 1
METHOD = "past_only_task_size_mean_v1"
DEFAULT_SIZE_BOUNDARIES = (25, 50, 75, 100)
BaselineKey = tuple[str, int]
SUPPORTED_CO_TASKS = ("tsp", "cvrp", "op", "mis", "jssp")
TASK_SIZE_SEMANTICS = {
    "tsp": "cities", "cvrp": "customers_excluding_depot",
    "op": "nodes_including_start", "mis": "graph_vertices", "jssp": "operations",
}


def baseline_instance_size(task: str, row: Mapping[str, Any]) -> int:
    """Extract a positive size from structured instance inputs, never answers.

    Prepared common-contract JSSP instances are rectangular rows of alternating
    machine,duration values, so their operation count is half the scalar count.
    Dictionary layouts are also accepted for callers using structured metadata.
    Full numerical/feasibility validation remains the task reward's responsibility.
    """
    if task not in SUPPORTED_CO_TASKS or not isinstance(row, Mapping):
        raise ValueError("Baseline requires a supported CO task and an input row")
    instance = row.get("instance_json", row.get("instance"))
    if isinstance(instance, str):
        try:
            instance = json.loads(instance)
        except (ValueError, TypeError) as exc:
            raise ValueError("Baseline instance_json must contain structured JSON") from exc
    size = None
    if task in ("tsp", "cvrp", "op"):
        if isinstance(instance, list):
            coords = instance if task == "tsp" else (instance[0] if instance else None)
        elif isinstance(instance, Mapping):
            coords = next((instance[name] for name in ("coords", "coordinates", "nodes", "locations")
                           if isinstance(instance.get(name), list)), None)
        else:
            coords = None
        if isinstance(coords, list):
            size = len(coords) - int(task == "cvrp")
    elif task == "mis" and isinstance(instance, Mapping):
        for name in ("num_nodes", "n", "nodes"):
            value = instance.get(name)
            if isinstance(value, list):
                size = len(value)
                break
            if isinstance(value, Integral) and not isinstance(value, bool):
                size = value
                break
    elif task == "jssp":
        if isinstance(instance, list) and instance and all(
            isinstance(job, list) and job and len(job) % 2 == 0 for job in instance
        ):
            size = sum(len(job) // 2 for job in instance)
        elif isinstance(instance, Mapping):
            jobs = instance.get("jobs")
            if isinstance(jobs, list) and jobs and all(isinstance(job, list) and job for job in jobs):
                size = sum(len(job) for job in jobs)
            else:
                n = instance.get("num_jobs", instance.get("n_jobs"))
                m = instance.get("num_machines", instance.get("n_machines"))
                if all(isinstance(v, Integral) and not isinstance(v, bool) and v > 0 for v in (n, m)):
                    size = n * m
    if size is None:
        raise ValueError(f"{task.upper()} baseline requires structured input-derived instance size")
    return _integer(size, f"{task} input-derived problem size", 1)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _sequence(value: Any, name: str) -> Sequence:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _keys(keys: Sequence[BaselineKey]) -> tuple[BaselineKey, ...]:
    result = []
    for key in _sequence(keys, "keys"):
        if len(_sequence(key, "key")) != 2:
            raise ValueError("each key must contain task and problem size")
        task, size = key
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a nonempty string")
        result.append((task.strip().lower(), _integer(size, "problem size", 1)))
    if not result:
        raise ValueError("a baseline snapshot requires a nonempty global batch")
    return tuple(result)


def _hash(payload: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("baseline state must contain finite JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BaselineSnapshot:
    """Immutable pre-generation predictions; one value per globally gathered row."""

    revision: int
    keys: tuple[BaselineKey, ...]
    values: tuple[float, ...]
    scale: float
    state_sha256: str


def external_advantages(snapshot: BaselineSnapshot, rewards: Sequence[float]) -> list[float]:
    """Return (reward - pre-batch baseline)/fixed scale without recentering.

    A snapshot remains usable for an already-collected batch after tracker
    advancement. The caller must retain the original row ordering. Only
    ``tracker.update`` checks whether the snapshot belongs to the current state.
    """
    if not isinstance(snapshot, BaselineSnapshot):
        raise ValueError("a BaselineSnapshot is required")
    keys = _keys(snapshot.keys)
    values = _sequence(snapshot.values, "snapshot values")
    rewards = _sequence(rewards, "rewards")
    if len(values) != len(keys) or len(rewards) != len(keys):
        raise ValueError("rewards, keys and baseline values must have equal lengths")
    scale = _number(snapshot.scale, "scale")
    if scale <= 0:
        raise ValueError("scale must be positive")
    result = [(_number(r, "reward") - _number(b, "baseline")) / scale
              for r, b in zip(rewards, values)]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("external advantage became non-finite")
    return result


class ExternalBaselineTracker:
    """Checkpointable completion-weighted task/size running means.

    No current reward enters its own baseline. B9 and B1 use the same estimator
    convention, but naturally expose different numbers of distinct instances.
    ``update`` is atomic and rejects stale/replayed or altered snapshots.
    Baseline predictions are not claimed to be variance optimal.
    """

    def __init__(self, *, size_boundaries=DEFAULT_SIZE_BOUNDARIES,
                 initial_value: float = 0.0, scale: float = 1.0):
        boundaries = tuple(_integer(v, "size boundary", 1)
                           for v in _sequence(size_boundaries, "size boundaries"))
        if tuple(sorted(set(boundaries))) != boundaries:
            raise ValueError("size boundaries must be strictly increasing")
        self._boundaries = boundaries
        self._initial_value = _number(initial_value, "initial value")
        self._scale = _number(scale, "scale")
        if self._scale <= 0:
            raise ValueError("scale must be positive")
        self._revision = 0
        # One canonical record per task/bucket avoids redundant global/task state.
        self._cells: dict[tuple[str, int], tuple[int, float]] = {}

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def seen_samples(self) -> int:
        return sum(count for count, _ in self._cells.values())

    @property
    def state_sha256(self) -> str:
        return self.state_dict()["sha256"]

    def _pooled_mean(self, task: str | None = None) -> float:
        cells = [value for (name, _), value in sorted(self._cells.items())
                 if task is None or task == name]
        count = sum(n for n, _ in cells)
        if not count:
            return self._initial_value if task is None else self._pooled_mean()
        return math.fsum(total for _, total in cells) / count

    def snapshot(self, keys: Sequence[BaselineKey]) -> BaselineSnapshot:
        keys = _keys(keys)
        values = []
        for task, size in keys:
            cell = self._cells.get((task, bisect.bisect_left(self._boundaries, size)))
            values.append(cell[1] / cell[0] if cell else self._pooled_mean(task))
        return BaselineSnapshot(self._revision, keys, tuple(values), self._scale,
                                self.state_sha256)

    advantages = staticmethod(external_advantages)

    def update(self, snapshot: BaselineSnapshot, rewards: Sequence[float]) -> None:
        if not isinstance(snapshot, BaselineSnapshot):
            raise ValueError("a BaselineSnapshot is required")
        if snapshot.revision != self._revision or snapshot.state_sha256 != self.state_sha256:
            raise ValueError("stale, replayed or foreign baseline snapshot")
        if snapshot != self.snapshot(snapshot.keys):
            raise ValueError("altered baseline snapshot")
        # Validate all rows and subtraction/scale before changing any state.
        external_advantages(snapshot, rewards)
        grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
        for (task, size), reward in zip(snapshot.keys, rewards):
            grouped[(task, bisect.bisect_left(self._boundaries, size))].append(float(reward))
        cells = dict(self._cells)
        try:
            for key, values in sorted(grouped.items()):
                count, total = cells.get(key, (0, 0.0))
                total = math.fsum([total, *sorted(values)])
                if not math.isfinite(total):
                    raise ValueError("baseline reward sum became non-finite")
                cells[key] = (count + len(values), total)
            # Ensure future global/task backoff predictions remain representable.
            math.fsum(total for _, total in cells.values())
            for task in {task for task, _ in cells}:
                math.fsum(total for (name, _), (_, total) in cells.items() if name == task)
        except OverflowError as exc:
            raise ValueError("baseline reward sum overflow") from exc
        self._cells = cells
        self._revision += 1

    def state_dict(self) -> dict[str, Any]:
        """Return independent JSON data, including a canonical content checksum."""
        payload = {
            "schema_version": SCHEMA_VERSION, "method": METHOD,
            "revision": self._revision,
            "config": {"size_boundaries": list(self._boundaries),
                       "initial_value": self._initial_value, "scale": self._scale},
            "cells": [{"task": task, "bucket": bucket, "count": count, "reward_sum": total}
                      for (task, bucket), (count, total) in sorted(self._cells.items())],
        }
        return {**payload, "sha256": _hash(payload)}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> ExternalBaselineTracker:
        """Validate and restore a checkpoint; malformed state is never repaired."""
        fields = {"schema_version", "method", "revision", "config", "cells", "sha256"}
        if not isinstance(state, Mapping) or set(state) != fields:
            raise ValueError("invalid baseline state fields")
        payload = {key: value for key, value in state.items() if key != "sha256"}
        if state["sha256"] != _hash(payload):
            raise ValueError("baseline state checksum mismatch")
        if (type(state["schema_version"]) is not int or state["schema_version"] != SCHEMA_VERSION
                or state["method"] != METHOD):
            raise ValueError("unsupported baseline state schema/method")
        config = state["config"]
        if not isinstance(config, Mapping) or set(config) != {"size_boundaries", "initial_value", "scale"}:
            raise ValueError("invalid baseline configuration fields")
        restored = cls(**config)
        revision = _integer(state["revision"], "revision", 0)
        previous = None
        for cell in _sequence(state["cells"], "cells"):
            if not isinstance(cell, Mapping) or set(cell) != {"task", "bucket", "count", "reward_sum"}:
                raise ValueError("invalid baseline cell fields")
            task = _keys([(cell["task"], 1)])[0][0]
            bucket = _integer(cell["bucket"], "bucket", 0)
            if bucket > len(restored._boundaries):
                raise ValueError("baseline bucket exceeds configured boundaries")
            key = task, bucket
            if previous is not None and key <= previous:
                raise ValueError("baseline cells must be unique and canonically ordered")
            previous = key
            restored._cells[key] = (_integer(cell["count"], "count", 1),
                                    _number(cell["reward_sum"], "reward sum"))
        if (revision == 0) != (restored.seen_samples == 0) or revision > restored.seen_samples:
            raise ValueError("baseline revision and sample counts disagree")
        restored._revision = revision
        try:
            restored._pooled_mean()
            for task, _ in restored._cells:
                restored._pooled_mean(task)
        except OverflowError as exc:
            raise ValueError("baseline reward sum overflow") from exc
        if restored.state_sha256 != state["sha256"]:
            raise ValueError("baseline state is not canonically encoded")
        return restored
