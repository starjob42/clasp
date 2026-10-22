"""Entry point: SFT/RL leakage audit (``clasp-audit-data``).

Reports exact-row and structured-instance overlap between every SFT and RL split pair, per
task. Run this before any long training job; ``--fail_on_overlap`` turns a non-zero overlap
into a non-zero exit status so it can gate a launcher.
"""
from __future__ import annotations

import json
import os
from typing import Any

from clasp.config import parse_data_audit_args

try:
    from clasp.data.loaders import (  # type: ignore[import-not-found]
        coerce_json_string,
        list_split_json_files,
        normalize_rows,
        read_json_rows,
        stable_row_signature,
    )
except ModuleNotFoundError:
    def coerce_json_string(value) -> str:
        if isinstance(value, str):
            try:
                json.loads(value)
                return value
            except Exception:
                return json.dumps(value, ensure_ascii=True)
        try:
            return json.dumps(value, ensure_ascii=True)
        except Exception:
            return json.dumps(str(value), ensure_ascii=True)


    def read_json_rows(file_path: str) -> list[dict]:
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"Expected top-level JSON list in {file_path}")
        return data


    def list_split_json_files(
        data_root: str,
        task: str,
        split: str,
        *,
        prefer_subset_files: bool = True,
        required: bool = True,
    ) -> list[str]:
        split_dir = os.path.join(data_root, task, split)
        if not os.path.isdir(split_dir):
            if required:
                raise FileNotFoundError(f"Missing split directory: {split_dir}")
            return []

        candidates = sorted(
            os.path.join(split_dir, name)
            for name in os.listdir(split_dir)
            if name.endswith(".json")
        )
        if not candidates:
            if required:
                raise FileNotFoundError(f"No .json files found under: {split_dir}")
            return []

        if prefer_subset_files:
            by_name = {os.path.basename(path): path for path in candidates}
            preferred_names = (
                ("train_subset.json", "train_small.json")
                if split == "train"
                else ("eval_subset.json", "test_subset.json", "eval_small.json", "test_small.json")
            )
            for name in preferred_names:
                if name in by_name:
                    return [by_name[name]]

            subset_candidates = [
                path for path in candidates if os.path.basename(path).endswith("_subset.json")
            ]
            if subset_candidates:
                return subset_candidates

        return candidates


    def stable_row_signature(row: dict) -> str:
        payload = json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        import hashlib
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


    def normalize_rows(rows: list[dict], json_file: str) -> list[dict]:
        normalized_rows = []
        invalid_rows: list[dict[str, object]] = []
        invalid_count = 0
        for row_idx, row in enumerate(rows):
            if not isinstance(row, dict):
                invalid_count += 1
                if len(invalid_rows) < 5:
                    invalid_rows.append(
                        {
                            "row_index": row_idx,
                            "issue": f"expected dict, found {type(row).__name__}",
                        }
                    )
                continue
            instruction = str(row.get("instruction", ""))
            input_text = str(row.get("input", ""))
            output_text = str(row.get("output", ""))

            missing_fields = []
            if not instruction.strip():
                missing_fields.append("instruction")
            if not input_text.strip():
                missing_fields.append("input")
            if not output_text.strip():
                missing_fields.append("output")

            if missing_fields:
                invalid_count += 1
                if len(invalid_rows) < 5:
                    invalid_rows.append(
                        {
                            "row_index": row_idx,
                            "missing_fields": list(missing_fields),
                        }
                    )
                continue

            instance_source = row.get("instance_json", row.get("instance"))
            normalized_rows.append(
                {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text,
                    "instance_json": coerce_json_string(instance_source),
                }
            )

        if invalid_count > 0:
            raise ValueError(
                f"Found {invalid_count} invalid row(s) in {json_file}. "
                "Every example must contain non-empty 'instruction', 'input', and 'output' fields. "
                f"Examples: {invalid_rows!r}"
            )

        if not normalized_rows:
            raise ValueError(f"No valid dict rows found in JSON dataset: {json_file}")
        return normalized_rows


def _deduplicate_by_signature(rows: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    exact_rows: dict[str, dict] = {}
    instance_rows: dict[str, dict] = {}
    for row in rows:
        exact_signature = stable_row_signature(row)
        exact_rows.setdefault(exact_signature, row)

        instance_signature = stable_row_signature(
            {
                "instruction": row["instruction"],
                "input": row["input"],
                "instance_json": row.get("instance_json", ""),
            }
        )
        instance_rows.setdefault(instance_signature, row)
    return exact_rows, instance_rows


def _row_preview(row: dict | None) -> dict[str, str] | None:
    if row is None:
        return None
    return {
        "instruction": str(row.get("instruction", ""))[:120].replace("\n", "\\n"),
        "input": str(row.get("input", ""))[:160].replace("\n", "\\n"),
        "output": str(row.get("output", ""))[:120].replace("\n", "\\n"),
    }


def _load_split_index(
    *,
    data_root: str,
    task: str,
    split: str,
    prefer_subset_files: bool,
) -> dict[str, Any]:
    json_files = list_split_json_files(
        data_root,
        task,
        split,
        prefer_subset_files=prefer_subset_files,
        required=(split == "train"),
    )
    if not json_files:
        return {
            "task": task,
            "split": split,
            "sources": [],
            "num_rows": 0,
            "exact_rows": {},
            "instance_rows": {},
        }

    merged_rows: list[dict] = []
    for json_file in json_files:
        merged_rows.extend(normalize_rows(read_json_rows(json_file), json_file))

    exact_rows, instance_rows = _deduplicate_by_signature(merged_rows)
    return {
        "task": task,
        "split": split,
        "sources": list(json_files),
        "num_rows": len(exact_rows),
        "exact_rows": exact_rows,
        "instance_rows": instance_rows,
    }


def _pair_key(sft_split: str, grpo_split: str) -> str:
    return f"sft_{sft_split}__grpo_{grpo_split}"


def audit_sft_vs_grpo_overlap(
    *,
    tasks: list[str],
    sft_data_root: str,
    grpo_data_root: str,
    prefer_subset_files: bool,
) -> dict[str, Any]:
    stage_indexes = {
        "sft": {
            task: {
                split: _load_split_index(
                    data_root=sft_data_root,
                    task=task,
                    split=split,
                    prefer_subset_files=prefer_subset_files,
                )
                for split in ("train", "eval")
            }
            for task in tasks
        },
        "grpo": {
            task: {
                split: _load_split_index(
                    data_root=grpo_data_root,
                    task=task,
                    split=split,
                    prefer_subset_files=prefer_subset_files,
                )
                for split in ("train", "eval")
            }
            for task in tasks
        },
    }

    total_exact = 0
    total_instance = 0
    pair_totals: dict[str, dict[str, int]] = {}
    task_reports: dict[str, Any] = {}

    for task in tasks:
        pair_reports = {}
        for sft_split in ("train", "eval"):
            for grpo_split in ("train", "eval"):
                sft_index = stage_indexes["sft"][task][sft_split]
                grpo_index = stage_indexes["grpo"][task][grpo_split]

                exact_overlap = sorted(
                    set(sft_index["exact_rows"]).intersection(grpo_index["exact_rows"])
                )
                instance_overlap = sorted(
                    set(sft_index["instance_rows"]).intersection(grpo_index["instance_rows"])
                )

                pair_key = _pair_key(sft_split, grpo_split)
                exact_count = len(exact_overlap)
                instance_count = len(instance_overlap)
                total_exact += exact_count
                total_instance += instance_count
                pair_totals.setdefault(pair_key, {"exact_row_overlap": 0, "instance_overlap": 0})
                pair_totals[pair_key]["exact_row_overlap"] += exact_count
                pair_totals[pair_key]["instance_overlap"] += instance_count

                pair_reports[pair_key] = {
                    "exact_row_overlap": exact_count,
                    "instance_overlap": instance_count,
                    "sample_exact_rows": [
                        _row_preview(sft_index["exact_rows"].get(signature))
                        for signature in exact_overlap[:3]
                    ],
                    "sample_instance_rows": [
                        _row_preview(sft_index["instance_rows"].get(signature))
                        for signature in instance_overlap[:3]
                    ],
                }

        task_reports[task] = {
            "sft": {
                split: {
                    "num_rows": stage_indexes["sft"][task][split]["num_rows"],
                    "sources": stage_indexes["sft"][task][split]["sources"],
                }
                for split in ("train", "eval")
            },
            "grpo": {
                split: {
                    "num_rows": stage_indexes["grpo"][task][split]["num_rows"],
                    "sources": stage_indexes["grpo"][task][split]["sources"],
                }
                for split in ("train", "eval")
            },
            "pairs": pair_reports,
        }

    return {
        "tasks": list(tasks),
        "sft_data_root": sft_data_root,
        "grpo_data_root": grpo_data_root,
        "prefer_subset_files": bool(prefer_subset_files),
        "total_exact_row_overlap": total_exact,
        "total_instance_overlap": total_instance,
        "has_overlap": (total_exact > 0 or total_instance > 0),
        "pair_totals": pair_totals,
        "task_reports": task_reports,
    }


def _print_human_report(report: dict[str, Any]) -> None:
    print("SFT-vs-GRPO overlap audit", flush=True)
    print(
        f"tasks={','.join(report['tasks'])} "
        f"total_exact_row_overlap={report['total_exact_row_overlap']} "
        f"total_instance_overlap={report['total_instance_overlap']}",
        flush=True,
    )
    for task, task_report in report["task_reports"].items():
        print(
            f"[{task}] "
            f"sft(train={task_report['sft']['train']['num_rows']}, eval={task_report['sft']['eval']['num_rows']}) "
            f"grpo(train={task_report['grpo']['train']['num_rows']}, eval={task_report['grpo']['eval']['num_rows']})",
            flush=True,
        )
        for pair_key, pair_report in task_report["pairs"].items():
            print(
                f"  {pair_key}: exact_row_overlap={pair_report['exact_row_overlap']} "
                f"instance_overlap={pair_report['instance_overlap']}",
                flush=True,
            )


def main() -> None:
    args = parse_data_audit_args()
    report = audit_sft_vs_grpo_overlap(
        tasks=args.tasks,
        sft_data_root=args.sft_data_root,
        grpo_data_root=args.grpo_data_root,
        prefer_subset_files=args.prefer_subset_files,
    )
    _print_human_report(report)
    if args.report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)), exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=True)
        print(f"[Audit] Wrote JSON report to: {args.report_json}", flush=True)
    if args.fail_on_overlap and report["has_overlap"]:
        raise SystemExit(1)


__all__ = ["audit_sft_vs_grpo_overlap", "main"]


if __name__ == "__main__":
    main()
