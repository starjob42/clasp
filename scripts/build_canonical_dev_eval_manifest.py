#!/usr/bin/env python3
"""Build a frozen manifest for a fixed development evaluation cohort.

The goal is one immutable evaluation set shared by every baseline and challenger.
The manifest pins the dataset bytes, the exact selected rows, the decode/prompt
protocol, and the cached baseline artifacts, so challengers never require a new
baseline inference run unless the manifest itself changes.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def one(pattern: str) -> str:
    matches = sorted(glob.glob(pattern, recursive=True))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one match for {pattern}, found {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--tasks", nargs="+", default=["cvrp", "jssp", "mis", "op", "tsp"])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tasks: dict[str, object] = {}
    protocol: dict[str, object] | None = None
    baseline_verifier: dict[str, object] | None = None

    for task in args.tasks:
        source = os.path.join(args.data_root, task, "eval", "test.json")
        with open(source, encoding="utf-8") as handle:
            rows = json.load(handle)

        summary_path = one(os.path.join(args.baseline_root, task, "**", "run_summary*.json"))
        details_path = one(os.path.join(args.baseline_root, task, "**", "run_details*.jsonl"))
        config_path = one(os.path.join(args.baseline_root, task, "**", "run_config*.json"))

        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
        with open(details_path, encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        records.sort(key=lambda record: record["example_index"])
        selected = records[: args.count]
        expected_indices = list(range(args.count))
        observed_indices = [record["example_index"] for record in selected]
        if observed_indices != expected_indices:
            raise RuntimeError(f"{task}: selected indices are not {expected_indices[0]}..{expected_indices[-1]}")

        instance_hashes = [record["instance_sha256"] for record in selected]
        semantic_hashes = [record["semantic_instance_sha256"] for record in selected]

        this_protocol = {
            key: summary["generation"][key]
            for key in (
                "dtype",
                "max_new_tokens",
                "max_prompt_length",
                "num_return_sequences",
                "generation_batch_size",
                "do_sample",
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "seed",
                "seed_policy",
            )
        }
        this_protocol.update(
            {
                "prompt_template": summary["prompting"]["prompt_template"],
                "prompt_style_resolved": summary["prompting"]["prompt_style_resolved"],
                "eval_start_index": summary["limits"]["eval_start_index"],
                "max_eval_samples_per_task": summary["limits"]["max_eval_samples_per_task"],
                "evaluation_purpose": summary["limits"]["evaluation_purpose"],
            }
        )
        if protocol is None:
            protocol = this_protocol
        elif protocol != this_protocol:
            raise RuntimeError(f"{task}: protocol differs from the first task")

        if baseline_verifier is None:
            baseline_verifier = summary["verifier_identity"]
        elif baseline_verifier != summary["verifier_identity"]:
            raise RuntimeError(f"{task}: verifier identity differs from the first task")

        overall = summary["overall"]
        tasks[task] = {
            "source": source,
            "available_examples": len(rows),
            "file_sha256": sha256_file(source),
            "file_size_bytes": os.path.getsize(source),
            "selected_example_indices": [expected_indices[0], expected_indices[-1]],
            "selected_rows": len(selected),
            "ordered_instance_sha256": instance_hashes,
            "ordered_instance_sha256_sha256": sha256_json(instance_hashes),
            "ordered_semantic_instance_sha256": semantic_hashes,
            "ordered_semantic_instance_sha256_sha256": sha256_json(semantic_hashes),
            "baseline": {
                "label": "foarl_f638_dev_seed43_step100",
                "model_path": summary["resolved_model_path"],
                "run_config": config_path,
                "run_config_sha256": sha256_file(config_path),
                "run_summary": summary_path,
                "run_summary_sha256": sha256_file(summary_path),
                "run_details": details_path,
                "run_details_sha256": sha256_file(details_path),
                "protocol_fingerprint": summary["protocol_fingerprint"],
                "metrics": {
                    "examples": overall["examples"],
                    "candidate_feasibility_rate": overall["candidate_feasibility_rate"],
                    "best_of_n_instance_feasibility_rate": overall[
                        "best_of_n_instance_feasibility_rate"
                    ],
                    "mean_best_feasible_gap": overall["mean_best_feasible_gap"],
                    "median_best_feasible_gap": overall["median_best_feasible_gap"],
                },
            },
        }

    manifest = {
        "schema_version": 1,
        "name": "canonical_dev_eval_v1",
        "status": "frozen",
        "purpose": "development_only",
        "data_root": os.path.abspath(args.data_root),
        "split": "eval",
        "tasks": tasks,
        "protocol": protocol,
        "baseline_verifier_identity": baseline_verifier,
        "baseline_label": "foarl_f638_dev_seed43_step100",
        "baseline_eval_root": os.path.abspath(args.baseline_root),
        "notes": [
            "All challengers must evaluate with EVAL_START_INDEX=0 and "
            "MAX_EVAL_SAMPLES_PER_TASK=100 against this exact data_root.",
            "The baseline artifacts in this manifest were produced once with the "
            "frozen eval snapshot correct_clasp_v8_20261113_v2 on this exact "
            "cohort; do not rerun the baseline unless the manifest changes.",
            "Baseline and challenger summaries may carry different protocol "
            "fingerprints because protocol material includes the checkpoint "
            "tokenizer artifact. tokenizer.json, merges.txt, added_tokens.json, "
            "special_tokens_map.json and vocab.json are byte-identical across the "
            "FOARL and CLASP checkpoints; verify that identity before pairing "
            "per-instance details.",
        ],
    }
    manifest["manifest_sha256"] = sha256_json(manifest)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    compact = {
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks": {
            task: {
                "file_sha256": data["file_sha256"],
                "rows": data["available_examples"],
                "baseline_gap": data["baseline"]["metrics"]["mean_best_feasible_gap"],
                "baseline_bon_feasibility": data["baseline"]["metrics"][
                    "best_of_n_instance_feasibility_rate"
                ],
                "baseline_candidate_feasibility": data["baseline"]["metrics"][
                    "candidate_feasibility_rate"
                ],
            }
            for task, data in tasks.items()
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
