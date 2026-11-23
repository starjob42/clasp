"""Recompute the matched-budget CLASP-vs-FOARL table from committed evidence.

Every row is read from an immutable artifact copied verbatim out of the Jean Zay
campaign trees (`evidence/`). The script fails closed if the evaluation protocol
differs between rows, so the published Markdown table cannot silently drift from
the numbers that were actually measured.

Usage:
    python experiments/grpo_full_budget_20260918/verify_matched_table.py [--markdown]

Exit code 0 means every row matched its recorded expectation and the shared
evaluation protocol is identical across artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"

PROTOCOL = {
    "data_root_suffix": "/clasp_foarl_v3_reproduction_20261018_v1/data_eval",
    "seed": 271828,
    "num_return_sequences": 8,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_new_tokens": 3000,
    "eval_start_index": 0,
    "eval_examples": 100,
}

# (label, relative evidence path, checkpoint step, expected boN feasibility, expected gap)
CLASP_ROWS = [
    ("CVRP", "clasp/cvrp_v11_step500_run_summary.json", 500, 0.95, 0.038814583895832463),
    ("JSSP", "clasp/jssp_v8_step800_run_summary.json", 800, 1.0, 0.06439931895768153),
    ("TSP", "clasp/tsp_v8_step800_run_summary.json", 800, 1.0, 0.006286115912965584),
    ("MIS", "clasp/mis_v15_step800_run_summary.json", 800, 0.94, 0.020355597861033036),
    ("OP", "clasp/op_v14_step1250_run_summary.json", 1250, 0.84, 0.01259127475472288),
]

FOARL_ROWS = [
    ("CVRP", "foarl/cvrp_step500_run_summary.json", 500, 0.95, 0.03819515140541658),
    ("JSSP", "foarl/jssp_step650_jiang_run_summary.json", 650, 1.0, 0.0664195709404245),
    ("TSP", "foarl/tsp_step650_jiang_run_summary.json", 650, 1.0, 0.0073028760522326815),
]

# The MIS FOARL run was pruned before its declared horizon; only the campaign's
# signed step-100/step-400 task reports survive. Step 400 is the best evidence.
FOARL_REPORT_ROWS = [
    ("MIS", "foarl/mis_step400_campaign_report.json", 400, 0.90, 0.024563890542562715),
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(relative):
    path = EVIDENCE / relative
    require(path.is_file(), f"missing evidence: {path}")
    return json.loads(path.read_text())


def check_protocol(summary, relative):
    generation, limits = summary["generation"], summary["limits"]
    for key, expected in (
        ("seed", PROTOCOL["seed"]),
        ("num_return_sequences", PROTOCOL["num_return_sequences"]),
        ("temperature", PROTOCOL["temperature"]),
        ("top_p", PROTOCOL["top_p"]),
        ("top_k", PROTOCOL["top_k"]),
        ("max_new_tokens", PROTOCOL["max_new_tokens"]),
    ):
        require(generation[key] == expected, f"{relative}: generation {key} differs")
    require(limits["eval_start_index"] == PROTOCOL["eval_start_index"], f"{relative}: eval start differs")
    require(limits["max_eval_samples_per_task"] == PROTOCOL["eval_examples"], f"{relative}: eval count differs")
    require(summary["data_root"].endswith(PROTOCOL["data_root_suffix"]), f"{relative}: data root differs")
    require(summary["run_status"] == "complete", f"{relative}: run is not complete")


def run_summary_row(label, relative, step, expected_bon, expected_gap, method):
    summary = load(relative)
    check_protocol(summary, relative)
    overall = summary["overall"]
    require(overall["examples"] == 100, f"{relative}: wrong example count")
    require(abs(overall["best_of_n_instance_feasibility_rate"] - expected_bon) < 1e-9,
            f"{relative}: BoN feasibility differs")
    require(abs(overall["mean_best_feasible_gap"] - expected_gap) < 1e-6,
            f"{relative}: mean gap differs")
    return {
        "task": label,
        "method": method,
        "step": step,
        "checkpoint": summary["checkpoint_name"],
        "prompt_style": summary["prompting"]["prompt_style_resolved"],
        "candidate_feasibility": overall["candidate_feasibility_rate"],
        "bon_feasibility": overall["best_of_n_instance_feasibility_rate"],
        "mean_gap": overall["mean_best_feasible_gap"],
        "evidence": relative,
        "sha256": summary["summary_sha256"],
    }


def campaign_report_row(label, relative, step, expected_bon, expected_gap):
    report = load(relative)
    require(report["task"] == label.lower(), f"{relative}: task differs")
    require(report["step"] == step, f"{relative}: step differs")
    require(report["status"] == "complete_four_methods_within_task_milestone", f"{relative}: incomplete milestone")
    row = next(item for item in report["summary"] if item["method"] == "foarl")
    require(abs(row["feasible_at_8_percent"] / 100 - expected_bon) < 1e-9, f"{relative}: BoN feasibility differs")
    require(abs(row["mean_signed_reference_gap_percent"] / 100 - expected_gap) < 1e-6, f"{relative}: gap differs")
    return {
        "task": label,
        "method": "foarl",
        "step": step,
        "checkpoint": f"checkpoint-{step}",
        "prompt_style": "jiang_f638",
        "candidate_feasibility": row["candidate_feasibility_percent"] / 100,
        "bon_feasibility": row["feasible_at_8_percent"] / 100,
        "mean_gap": row["mean_signed_reference_gap_percent"] / 100,
        "evidence": relative,
        "sha256": report["receipt_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true", help="print the Markdown table only")
    args = parser.parse_args()

    clasp = [run_summary_row(*row, method="clasp") for row in CLASP_ROWS]
    foarl = [run_summary_row(*row, method="foarl") for row in FOARL_ROWS]
    foarl += [campaign_report_row(*row) for row in FOARL_REPORT_ROWS]
    foarl.sort(key=lambda item: item["task"])

    for row in clasp:
        require(row["prompt_style"] == "jiang_f638", "CLASP rows must use the jiang_f638 prompt")

    table = {}
    for row in clasp + foarl:
        table.setdefault(row["task"], {})[row["method"]] = row

    if args.markdown:
        print("| Task | CLASP step | CLASP BoN feas / gap | FOARL step | FOARL BoN feas / gap | gap(FOARL-CLASP) pp |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for task in ("CVRP", "JSSP", "TSP", "MIS", "OP"):
            cell = table[task]
            left, right = cell.get("clasp"), cell.get("foarl")
            if left and right:
                delta = (right["mean_gap"] - left["mean_gap"]) * 100
                delta_text = f"{delta:+.3f}"
            else:
                delta_text = "n/a"
            fmt = lambda r: (f"{r['step']}", f"{r['bon_feasibility']:.1%} / {r['mean_gap']:.3%}") if r else ("-", "-")
            (ls, lg), (rs, rg) = fmt(left), fmt(right)
            print(f"| {task} | {ls} | {lg} | {rs} | {rg} | {delta_text} |")
        return 0

    print(json.dumps({"protocol": PROTOCOL, "table": table}, indent=2, sort_keys=True))
    print("\nMATCHED_TABLE_VERIFIED rows=%d" % sum(len(v) for v in table.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
