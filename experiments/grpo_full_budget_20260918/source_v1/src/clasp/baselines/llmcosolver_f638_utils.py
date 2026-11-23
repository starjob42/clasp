"""Minimal numerical helpers used by the pinned LLMCoSolver reward port.

The implementations preserve the operational behavior of ``utils.py`` from
LLMCoSolver commit ``f6388991d243c0aa91d9ee7e2fa18ed1c9e72477``.  Keeping
them here avoids importing the upstream module's unrelated dataset stack.
"""

from __future__ import annotations

import os
import pickle

import numpy as np


def get_makespan(instance: np.ndarray, schedule: list[list[int]]) -> float | str:
    """Decode the upstream machine-based JSSP representation exactly."""

    n, two_m = instance.shape
    m = two_m // 2
    job_next = [0] * n
    job_finish = [0] * n
    machine_finish = [0] * m
    machine_ptr = [0] * m
    scheduled_ops = 0
    total_ops = n * m

    while scheduled_ops < total_ops:
        available = []
        for machine in range(m):
            if machine_ptr[machine] < n:
                job = schedule[machine][machine_ptr[machine]]
                op_idx = job_next[job]
                if op_idx < m and instance[job, 2 * op_idx] == machine:
                    start_time = max(job_finish[job], machine_finish[machine])
                    available.append((start_time, machine, job))

        if not available:
            return "infeasible"

        available.sort(key=lambda x: x[0])
        start_time, machine, job = available[0]
        op_idx = job_next[job]
        proc_time = instance[job, 2 * op_idx + 1]
        finish_time = start_time + proc_time
        job_finish[job] = finish_time
        machine_finish[machine] = finish_time
        job_next[job] += 1
        machine_ptr[machine] += 1
        scheduled_ops += 1

    return max(job_finish)


def load_pkl_dataset(filename: str, disable_print: bool = False):
    """Load a pickle with the upstream extension convention."""

    actual_filename = filename if os.path.splitext(filename)[1] == ".pkl" else filename + ".pkl"
    with open(actual_filename, "rb") as handle:
        data = pickle.load(handle)
    if not disable_print:
        print(f">> Load {len(data)} data ({type(data)}) from {filename}")
    return data


def compute_euclidean_distance_matrix(locations):
    """Build the same dense Euclidean matrix as the upstream implementation."""

    if isinstance(locations, list):
        locations = np.array(locations)
    num_nodes = locations.shape[0]
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(locations[i] - locations[j])
    return distance_matrix


def calculate_total_distance(route, distance_matrix):
    """Use the upstream route sum, including its unconditional closing edge."""

    total_distance = 0
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        total_distance += distance_matrix[from_node][to_node]
    total_distance += distance_matrix[route[-1]][route[0]]
    return total_distance
