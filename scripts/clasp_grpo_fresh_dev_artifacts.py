#!/usr/bin/env python3
"""Validate frozen fresh-development inputs and bind raw evaluation artifacts.

Only the Python standard library is required. Native solver receipts are checked;
this tool neither solves instances nor repairs model outputs.
"""
import argparse
import ast
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re

SOURCE_MANIFEST_SHA256 = "fbeaa2711c6dd18b51b82fc84f0c8a6890d3f58cfc690e829bdcf875e81c6248"
METHODS = {"sft": "sft", "grpo_z": "grpo_z", "v3": "clasp_grpo_anchor_v3", "v4": "clasp_grpo_anchor_v4"}
SFT_HASHES = {
    "cvrp": "f8f634c3a59f53bf411bb0dfc85af98dbc16d936a53263cdfcd6e74ce88b25b8",
    "mis": "cb5e979325ba7691f75998326ce79e47038d6462596dbc815f2b54d2b550da15",
}
GENERATION = {"seed": 271828, "dtype": "float16", "num_return_sequences": 8, "generation_batch_size": 8,
              "do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.0,
              "max_new_tokens": 3000, "max_prompt_length": 29768}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return {"path": str(path), "sha256": sha.hexdigest(), "size_bytes": path.stat().st_size}


def checked_payload(value, field):
    unsigned = {key: item for key, item in value.items() if key != field}
    require(value.get(field) == digest(unsigned), "invalid self hash: " + field)
    return value


def checked_json(path, field):
    return checked_payload(json.loads(Path(path).read_text()), field)


def write_new(path, payload, field):
    payload = dict(payload, **{field: digest(payload)})
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o440)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def structured(row):
    value = row["instance_json"]
    return json.loads(value) if isinstance(value, str) else value


def number(value):
    value = float(value)
    require(math.isfinite(value), "nonfinite instance number")
    rounded = round(value)
    return int(rounded) if abs(value - rounded) <= 1e-12 else value


def semantic_id(task, row):
    """Structured-only equivalent of the frozen CVRP/MIS identity functions."""
    instance = structured(row)
    if task == "cvrp":
        coords, demands, capacity = instance
        value = {"task": task, "coordinates": [[number(x), number(y)] for x, y in coords],
                 "demands": [number(item) for item in demands], "vehicle_capacity": number(capacity)}
    else:
        edges = {tuple(sorted((int(edge[0]), int(edge[1])))) for edge in instance["edges"] if edge[0] != edge[1]}
        value = {"task": task, "num_nodes": int(instance["num_nodes"]), "edges": [list(edge) for edge in sorted(edges)]}
    return digest(value)


def row_identity(row):
    return digest({"instruction": str(row.get("instruction", "")), "input": str(row.get("input", "")),
                   "output": str(row.get("output", "")), "instance_json": structured(row)})


def parse_reference_output(task, output):
    label = "Routes" if task == "cvrp" else "Set"
    match = re.fullmatch(r"\s*" + label + r":\s*(\[.*\])\s*,\s*Objective:\s*"
                         r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*", str(output), re.S)
    require(match is not None, "malformed rendered reference output")
    try:
        solution = ast.literal_eval(match.group(1))
        objective = float(match.group(2))
    except (ValueError, SyntaxError, TypeError) as error:
        raise ValueError("malformed rendered reference solution") from error
    require(isinstance(solution, list) and math.isfinite(objective), "invalid rendered reference output")
    return solution, objective


def validate_reference(task, row, proof):
    checked_payload(proof, "evidence_sha256")
    require(proof["task"] == task and proof["strict_verifier_feasible"] is True, "unverified reference")
    instance = structured(row)
    rendered_solution, rendered_objective = parse_reference_output(task, row["output"])
    if task == "mis":
        require(proof["native_status"] == "OPTIMAL" and proof["native_status_code"] == 4
                and proof["optimality_certified"] is True and proof["reference_quality"] == "certified_optimum",
                "MIS requires native optimality certification for every instance")
        objective = proof["objective"]
        require(type(objective) is int and objective > 0 and proof["best_bound"] == objective,
                "MIS incumbent and bound do not certify the same integer optimum")
        stats = proof["native_response_stats"]
        require(re.search(r"^status: OPTIMAL$", stats, re.M), "MIS native response lacks OPTIMAL status")
        for field in ("objective", "best_bound"):
            match = re.search(r"^" + field + r":\s*([-+0-9.eE]+)$", stats, re.M)
            require(match is not None and float(match.group(1)) == objective, "MIS native bound/objective mismatch")
        selected = proof["reference_set"]
        require(len(selected) == len(set(selected)) == objective
                and all(type(vertex) is int and 0 <= vertex < instance["num_nodes"] for vertex in selected),
                "MIS incumbent set is invalid")
        chosen = set(selected)
        require(not any(a in chosen and b in chosen for a, b in instance["edges"]), "MIS incumbent is not independent")
        require(all(type(vertex) is int for vertex in rendered_solution)
                and len(rendered_solution) == len(set(rendered_solution))
                and set(rendered_solution) == chosen, "rendered MIS set differs from the certified reference")
        require(rendered_objective == len(rendered_solution) == objective,
                "rendered MIS objective differs from its verifier cardinality")
    else:
        require(proof["optimality_certified"] is False and proof["reference_quality"] == "verified_heuristic_reference",
                "CVRP references must retain their heuristic label")
        coords, demands, capacity = instance
        routes = proof["reference_routes"]
        require(rendered_solution == routes, "rendered CVRP routes differ from the verified reference")
        customers = []
        distance = 0.0
        for route in routes:
            require(len(route) >= 3 and route[0] == route[-1] == 0 and 0 not in route[1:-1], "invalid CVRP reference route")
            require(all(type(vertex) is int and 0 <= vertex < len(coords) for vertex in route), "invalid CVRP node")
            require(sum(demands[vertex] for vertex in route[1:-1]) <= capacity, "CVRP reference exceeds capacity")
            customers.extend(route[1:-1])
            # Match the frozen shared verifier's per-route float64 hypot
            # accumulation, including its outer sum across routes.
            route_distance = 0.0
            for a, b in zip(route, route[1:]):
                route_distance += math.hypot(float(coords[a][0]) - float(coords[b][0]),
                                             float(coords[a][1]) - float(coords[b][1]))
            distance += route_distance
        require(sorted(customers) == list(range(1, len(coords))), "CVRP reference does not cover each customer once")
        tolerance = max(5e-13, 8 * math.ulp(distance))  # 12 printed decimals plus float64 rounding.
        require(math.isclose(distance, proof["objective"], rel_tol=0, abs_tol=tolerance), "CVRP reference objective mismatch")
        require(math.isclose(rendered_objective, distance, rel_tol=0, abs_tol=tolerance),
                "rendered CVRP objective differs from its shared-verifier distance")


def validate_benchmark(root):
    root = Path(root).resolve(strict=True)
    manifest = checked_json(root / "development_manifest.json", "manifest_sha256")
    audit = checked_json(root / "training_overlap_audit.json", "audit_sha256")
    require(manifest["purpose"] == "fresh_development_only", "unexpected benchmark purpose")
    require(audit["status"] == "passed_zero_semantic_overlap" and audit["total_overlaps"] == 0,
            "a completed zero-overlap audit is required")
    require(audit["development_manifest_sha256"] == manifest["manifest_sha256"], "overlap audit targets a different dataset manifest")
    expected_sources = {(task, label) for task in ("cvrp", "mis") for label in ("sft_train", "rl_train", "public_eval")}
    require(len(audit["sources"]) == 6 and {(item["task"], item["label"]) for item in audit["sources"]} == expected_sources,
            "overlap audit does not cover all six forbidden sources")
    require(all(item["rows"] > 0 and not item["overlaps"] and len(item["source"]["sha256"]) == 64
                for item in audit["sources"]), "incomplete source scan in overlap audit")
    identities, snapshots = {}, {}
    for task in ("cvrp", "mis"):
        dataset_path = root / task / "eval/test.json"
        evidence_path = root / task / "reference_evidence.json"
        snapshots[task] = {"dataset": identity(dataset_path), "reference_evidence": identity(evidence_path)}
        for kind in snapshots[task]:
            require(snapshots[task][kind]["sha256"] == manifest["tasks"][task][kind]["sha256"], "benchmark file hash mismatch: " + task + "/" + kind)
        rows, evidence = json.loads(dataset_path.read_text()), json.loads(evidence_path.read_text())
        require(len(rows) == len(evidence) == manifest["tasks"][task]["instances"] == 100, "expected 100 instances and reference proofs per task")
        identities[task] = []
        for index, (row, proof) in enumerate(zip(rows, evidence)):
            semantic = semantic_id(task, row)
            require(proof["index"] == index and proof["semantic_instance_sha256"] == semantic, "reference proof/instance mismatch")
            validate_reference(task, row, proof)
            identities[task].append({"example_index": index, "semantic_instance_sha256": semantic,
                                     "benchmark_row_sha256": row_identity(row), "reference_output_sha256": digest(str(row["output"]))})
        semantic_ids = [item["semantic_instance_sha256"] for item in identities[task]]
        require(len(set(semantic_ids)) == 100 and sorted(semantic_ids) == manifest["tasks"][task]["semantic_instance_sha256"],
                "benchmark semantic identity set mismatch")
    return {"manifest": identity(root / "development_manifest.json"), "overlap_audit": identity(root / "training_overlap_audit.json"),
            "manifest_sha256": manifest["manifest_sha256"], "audit_sha256": audit["audit_sha256"],
            "files": snapshots, "ordered_instances": identities}


def validate_checkpoint(source, checkpoint, train, task, method):
    source, checkpoint = Path(source).resolve(strict=True), Path(checkpoint).resolve(strict=True)
    manifest = identity(source / "SOURCE_MANIFEST.sha256")
    require(manifest["sha256"] == SOURCE_MANIFEST_SHA256, "evaluation requires the exact frozen v30 source manifest")
    for line in (source / "SOURCE_MANIFEST.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = (source / relative.lstrip("*")).resolve(strict=True)
        require(path.is_relative_to(source) and identity(path)["sha256"] == expected,
                "frozen source file differs from its manifest")
    spec = importlib.util.spec_from_file_location("frozen_fair_artifacts", source / "scripts/clasp_grpo_fair_dev_artifacts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_source(source)
    files = {name: identity(checkpoint / name) for name in ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json")}
    if method == "sft":
        require(checkpoint.name == "checkpoint-31250" and files["adapter_model.safetensors"]["sha256"] == SFT_HASHES[task],
                "SFT checkpoint differs from the certified training initialization")
        require(not train, "SFT evaluation does not accept an RL training root")
        contract = None
    else:
        require(train and checkpoint == Path(train).resolve(strict=True) / "checkpoints/checkpoint-100", "RL checkpoint/training root mismatch")
        module.validate_training(train, task, METHODS[method], source)
        contract = identity(Path(train) / "local_development_run_contract.json")
    return {"source_manifest": manifest, "checkpoint": files, "training_contract": contract}


def launch(args):
    output = args.output.resolve()
    require(not output.exists(), "fresh evaluation output already exists")
    benchmark = validate_benchmark(args.data)
    checkpoint = validate_checkpoint(args.source, args.checkpoint, args.train, args.task, args.method)
    wrapper = args.management / "scripts/evaluate_clasp_grpo_fresh_dev_a100.slurm"
    require(identity(args.running_wrapper)["sha256"] == identity(wrapper)["sha256"], "submitted wrapper differs from the management copy")
    payload = {"schema_version": 1, "status": "validated_fresh_development_launch", "task": args.task,
               "method": args.method, "stage": "sft" if args.method == "sft" else "grpo",
               "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "generation": GENERATION,
               "evaluation_slice": {"split": "eval", "start": 0, "count": 100}, "repair": False,
               "representation": "jiang_f638_sft" if args.method == "sft" else "jiang_f638_clasp",
               "benchmark": benchmark, **checkpoint,
               "management": {"wrapper": identity(wrapper), "helper": identity(Path(__file__))},
               "runtime_python": identity(args.runtime / "bin/python"),
               "reference_semantics": {"cvrp": "gap_to_verified_heuristic_reference", "mis": "gap_to_certified_optimum"},
               "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    output.mkdir()
    return write_new(output / "launch_receipt.json", payload, "launch_sha256")


def index(output):
    output = Path(output)
    receipt = checked_json(output / "launch_receipt.json", "launch_sha256")
    details, configs = list(output.glob("eval/**/run_details__*.jsonl")), list(output.glob("eval/**/run_config__*.json"))
    require(len(details) == len(configs) == 1, "expected one raw evaluation run")
    config = checked_json(configs[0], "run_config_sha256")
    require(config["run_status"] == "complete", "evaluation did not complete")
    require(config["stage"] == receipt["stage"] and config["tasks"] == [receipt["task"]], "evaluation stage/task mismatch")
    expected_checkpoint = Path(receipt["checkpoint"]["adapter_model.safetensors"]["path"]).parent
    require(Path(config["resolved_model_path"]).resolve() == expected_checkpoint, "evaluation checkpoint differs from the launch receipt")
    require(all(config["generation"].get(key) == value for key, value in GENERATION.items()), "evaluation decoding mismatch")
    require(config["prompting"]["prompt_style_resolved"] == "jiang_f638" and config["prompting"]["no_reason"] is True,
            "evaluation representation mismatch")
    require(config["prompting"]["clasp_mis_jiang_response_output"] == "0"
            and config["prompting"]["clasp_mis_relabel_order"] == "", "evaluation MIS representation changed")
    require(config["limits"]["eval_start_index"] == 0 and config["limits"]["max_eval_samples_per_task"] == 100
            and config["limits"]["prefer_subset_files"] is False, "evaluation slice mismatch")
    detail_identity = identity(details[0])
    require(detail_identity["sha256"] == config["details_artifact"]["sha256"], "detail/config linkage mismatch")
    records = [json.loads(line) for line in details[0].read_text().splitlines() if line.strip()]
    require(len(records) == 100, "evaluation must contain exactly 100 instances")
    expected = receipt["benchmark"]["ordered_instances"][receipt["task"]]
    for record, target in zip(records, expected):
        checked_payload(record, "record_sha256")
        require(record["task"] == receipt["task"] and record["split"] == "eval", "record task/split mismatch")
        require(record["num_return_sequences"] == len(record["candidates"]) == 8, "record candidate budget mismatch")
        require(all(record.get(key) == value for key, value in target.items()), "record benchmark instance/reference mismatch")
        require(record["protocol_fingerprint"] == config["protocol_fingerprint"], "record evaluation protocol mismatch")
    # Bind completion to the exact code, inputs, and weights checked at launch.
    for file in [receipt["source_manifest"], *receipt["checkpoint"].values(),
                 *receipt["management"].values(), receipt["benchmark"]["manifest"], receipt["benchmark"]["overlap_audit"]]:
        require(identity(file["path"])["sha256"] == file["sha256"], "launch-bound input changed during evaluation")
    for task_files in receipt["benchmark"]["files"].values():
        for file in task_files.values():
            require(identity(file["path"])["sha256"] == file["sha256"], "benchmark changed during evaluation")
    return write_new(output / "result_index.json", {
        "schema_version": 1, "status": "complete_fresh_development_only", "task": receipt["task"], "method": receipt["method"],
        "candidate_budget": 8, "generation_seed": 271828, "repair": False,
        "evaluation_slice": receipt["evaluation_slice"], "launch_receipt": identity(output / "launch_receipt.json"),
        "details": detail_identity, "run_config": identity(configs[0]), "reference_semantics": receipt["reference_semantics"][receipt["task"]],
    }, "index_sha256")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("launch")
    for name in ("source", "management", "runtime", "data", "checkpoint", "output", "running-wrapper"):
        start.add_argument("--" + name, type=Path, required=True)
    start.add_argument("--train", type=Path)
    start.add_argument("--task", choices=("cvrp", "mis"), required=True)
    start.add_argument("--method", choices=METHODS, required=True)
    finish = commands.add_parser("index")
    finish.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = launch(args) if args.command == "launch" else index(args.output)
    print(json.dumps({key: result[key] for key in ("status", "task", "method")}, sort_keys=True))


if __name__ == "__main__":
    main()
