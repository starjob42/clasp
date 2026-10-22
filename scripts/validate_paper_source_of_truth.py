"""Validate the machine-readable source of truth used by the revised paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "canonical_baseline",
    "tuned_validation",
    "algorithm_identity",
    "artifact_policy",
}
TASKS = ("tsp", "op", "jssp", "cvrp", "mis")


def load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    missing = REQUIRED_TOP_LEVEL_KEYS - manifest.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"manifest is missing required keys: {names}")
    return manifest


def validate_complete_result(result: dict, *, label: str) -> None:
    if result.get("status") == "complete":
        if result.get("instances") != 96:
            raise ValueError(f"{label} must contain 96 instances")
        if not isinstance(result.get("feasible"), int):
            raise ValueError(f"{label} must contain an integer feasible count")
        if not isinstance(result.get("gap_percent"), (int, float)):
            raise ValueError(f"{label} must contain a numeric gap")


def validate_manifest(manifest: dict) -> None:
    baseline = manifest["canonical_baseline"]
    for method in ("clasp", "foarl"):
        results = baseline[method]
        if set(results) != set(TASKS):
            raise ValueError(f"canonical {method} results must cover {TASKS}")
        for task, result in results.items():
            if result.get("instances") != 100:
                raise ValueError(f"canonical {method}/{task} must contain 100 instances")

    tuned = manifest["tuned_validation"]
    clasp_results = tuned["clasp_best_completed_by_task"]
    if set(clasp_results) != set(TASKS):
        raise ValueError(f"tuned CLASP results must cover {TASKS}")
    for task, result in clasp_results.items():
        validate_complete_result(result, label=f"tuned CLASP/{task}")

    foarl_results = tuned["foarl_same_settings_jobs"]
    if set(foarl_results) != set(TASKS):
        raise ValueError(f"same-slice FOARL results must cover {TASKS}")
    for task, result in foarl_results.items():
        if result.get("status") == "complete":
            validate_complete_result(result, label=f"same-slice FOARL/{task}")

    if tuned["mis_stage2"]["status"] != "complete":
        if clasp_results["mis"]["status"] == "complete":
            raise ValueError(
                "MIS cannot be marked complete while the stage-two search is pending"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("final_clasp/revised/paper_results_source_of_truth.json"),
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    validate_manifest(manifest)
    print(f"validated {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
