"""Bounded milestone evaluation on the original, immutable DEV100/raw-K8 cohort.

The numerical evaluator and all 800 candidate rechecks are the canonical ones.
Only checkpoint admission, launcher provenance and the four-arm report change.
Checkpoint files are verified while present. Later reports replay the raw outputs
against their preserved policy hashes and signed admission/index snapshots, so
normal checkpoint rotation does not erase an earlier milestone comparison.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import statistics
import sys

PROTOCOL = "budget-matched-grpo-development-20260918-v1"
EVALUATION_PROTOCOL = "budget-matched-grpo-dev100-raw8-20260918-v1"
METHODS = ("classic_grpo", "foarl", "rloo_legacy", "rloo_quality")
HORIZONS = {"cvrp": 500, "mis": 800, "jssp": 800, "tsp": 800, "op": 1250}
DIRECTORY = "experiments/grpo_full_budget_20260918"
HELPER = "experiments/grpo_external_baseline_all_tasks_20260910/all_tasks_evaluation.py"
COHORT_SHA = "ba256066362b43eefbc90bb692a46890261dd822357ff52674a5c0dc27043ca3"
CANONICAL_MANIFEST_SHA = "3be76406526bddf94a76ffe42eba1f303976099adb3b120162d320b62f87f79c"
SHARED_FILES = tuple("src/clasp/" + name for name in (
    "entrypoints/evaluate.py", "data/loaders.py", "data/prompts.py", "data/benchmark.py",
    "models/loading.py", "models/checkpoint_fidelity.py", "rewards/co_tasks.py", "runtime.py"))
EXPECTED_EVALUATION = {"count_per_task": 100, "max_completion_tokens": 3000,
    "no_reason": True, "prompt_style": "jiang_f638", "raw_candidates": 8,
    "repair": False, "seed": 271828, "temperature": 0.7, "top_k": 50, "top_p": 0.9}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def signed(value):
    return {**value, "receipt_sha256": digest(value)}


def verify_signed(value):
    require(value == signed({k: v for k, v in value.items() if k != "receipt_sha256"}),
            "Invalid receipt self-hash")
    return value


def identity(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and path.resolve() == path,
            "Missing, relative or linked input: " + str(path))
    before = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    after = path.stat()
    fields = ("st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(all(getattr(before, k) == getattr(after, k) for k in fields),
            "Input changed while hashing")
    return {"path": str(path), "sha256": sha.hexdigest(), "size_bytes": after.st_size}


def check(binding):
    require(identity(binding["path"]) == binding, "Bound input changed: " + binding["path"])
    return binding


def read(path, *, sealed=False):
    identity(path)
    value = json.loads(Path(path).read_text())
    return verify_signed(value) if sealed else value


def write_once(path, value):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, "Relative or linked output")
    payload = signed(value)
    if path.exists():
        require(read(path, sealed=True) == payload, "Existing output differs: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o440)
    return identity(path)


def manifest(source):
    root = Path(source["root"])
    require(root.is_absolute() and root.resolve() == root, "Linked source root")
    path = root / "SOURCE_MANIFEST.sha256"
    require(identity(path)["sha256"] == source["manifest_sha256"], "Source manifest changed")
    names = {}
    for line in path.read_text().splitlines():
        fields = line.split("  ")
        require(len(fields) == 2, "Malformed source manifest")
        sha, name = fields
        require(name and name not in names and not Path(name).is_absolute()
                and ".." not in Path(name).parts, "Unsafe/duplicate manifest member")
        bound = identity(root / name)
        require(bound["sha256"] == sha, "Frozen source changed: " + name)
        names[name] = bound
    require(names, "Empty source manifest")
    return names


def module(path, namespace=""):
    name = "budget_evaluation_" + hashlib.sha256((str(path) + namespace).encode()).hexdigest()
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
    return sys.modules[name]


def validate_scope(value):
    require(value["task"] in HORIZONS and value["method"] in METHODS, "Unknown task/method")
    require(type(value["step"]) is int and type(value["horizon"]) is int
            and value["horizon"] == HORIZONS[value["task"]]
            and value["step"] in (100, 400, value["horizon"])
            and 0 < value["step"] <= value["horizon"], "Undeclared evaluation milestone/horizon")


def context(value):
    require(value["protocol"] == EVALUATION_PROTOCOL and value["schema_version"] == 1,
            "Unknown evaluation contract")
    validate_scope(value)
    for key in ("campaign_declaration", "cohort_declaration", "terminal_audit"):
        check(value[key])
    require(value["cohort_declaration"]["sha256"] == COHORT_SHA,
            "Only the pinned launch_declaration_v3 DEV100 cohort is admitted")
    cohort = read(value["cohort_declaration"]["path"])
    require(cohort["protocol"] == "quality-signal-native-launch-20260917-v1"
            and cohort["evaluation"] == EXPECTED_EVALUATION
            and cohort["canonical_source"]["manifest_sha256"] == CANONICAL_MANIFEST_SHA,
            "Canonical cohort/decoder declaration changed")
    campaign = read(value["campaign_declaration"]["path"])
    require(campaign["protocol"] == PROTOCOL and campaign["source"] == value["source"]
            and campaign["methods"] == list(METHODS) and campaign["horizons"] == HORIZONS
            and campaign["seed"] == 43, "Campaign source/method/budget differs")
    require(value["runtime_python"] == campaign["runtime_python"] == cohort["runtime_python"],
            "Evaluation runtime differs")
    current, canonical = manifest(value["source"]), manifest(cohort["canonical_source"])
    for name in SHARED_FILES:
        require(name in current and name in canonical
                and current[name]["sha256"] == canonical[name]["sha256"],
                "Canonical evaluation dependency changed: " + name)
    require({HELPER, "scripts/compare_development.py"} <= set(canonical),
            "Canonical numerical audit helpers missing")
    require({DIRECTORY + "/budget_evaluation.py", DIRECTORY + "/evaluate_a100.slurm",
             DIRECTORY + "/common.py", DIRECTORY + "/campaign.py"} <= set(current),
            "Frozen evaluation management incomplete")
    require(Path(__file__).resolve() == Path(value["source"]["root"]) / DIRECTORY / "budget_evaluation.py",
            "Executing evaluator is not the frozen source")
    common = module(Path(value["source"]["root"]) / DIRECTORY / "common.py")
    common.validate_declaration(campaign)
    output = Path(value["output_root"])
    expected = Path(campaign["root"]) / "evaluations" / value["task"] / value["method"] / f"step{value['step']}"
    require(output == expected and output.is_absolute() and output.resolve() == output
            and not output.is_relative_to(Path(value["source"]["root"])),
            "Evaluation output is outside its declared arm/milestone")
    require(value["bundle"] == value["source"], "Evaluation wrapper bundle differs")
    return campaign, cohort, common


def validate_checkpoint(value, campaign, common, *, archived_index=None):
    """Check the full-budget chunk receipt, never the historical step-100 adapter."""
    task, method, step, horizon = (value[k] for k in ("task", "method", "step", "horizon"))
    validate_scope(value)
    check(value["terminal_audit"])
    complete = read(value["terminal_audit"]["path"], sealed=True)
    require(complete["protocol"] == PROTOCOL and complete["status"] == "verified_budget_training",
            "Unsuccessful training chunk")
    material = verify_signed(complete["checkpoint"])
    require(material["protocol"] == PROTOCOL and material["status"] == "verified_budget_checkpoint",
            "Unsuccessful checkpoint material")
    for item in (complete, material):
        require(all(type(item[key]) is int for key in ("step", "max_steps", "world_size"))
                and (item["task"], item["method"], item["step"], item["max_steps"], item["world_size"])
                == (task, method, step, horizon, 4), "Checkpoint task/method/milestone/horizon differs")
    require(complete["contract"] == material["contract"], "Chunk/checkpoint run contract differs")
    check(complete["contract"])
    contract = read(complete["contract"]["path"])
    require(contract["protocol"] == PROTOCOL and contract["stage"] == "grpo"
            and contract["task"] == task and contract["method"] == method
            and contract["horizon"] == horizon and contract["seed"] == 43
            and contract["diagnostic"] is False
            and contract["source"] == material["source"] == value["source"]
            and contract["campaign_declaration"] == value["campaign_declaration"]
            and contract["initialization"] == material["initialization"],
            "Training recipe/source/initialization differs")
    name = common.job_key(task, method, "train", step)
    job = campaign["jobs"][name]
    require(contract == common.run_contract(campaign, value["campaign_declaration"], job,
                                           contract["initialization"]), "Run contract differs from campaign")
    output = Path(contract["output_dir"])
    require(Path(complete["contract"]["path"]) == output / common.CONTRACT_NAME
            and Path(value["terminal_audit"]["path"]) == output / f"chunk-{step}-complete.json",
            "Training receipt belongs to another output")
    check(complete["admission"])
    require(Path(complete["admission"]["path"]) == output / f"chunk-{step}-admission.json",
            "Training admission is outside declared chunk")
    admission = read(complete["admission"]["path"], sealed=True)
    exit_receipt = read(output / f"chunk-{step}-exit.json")
    require(admission["protocol"] == PROTOCOL and admission["job"] == name
            and admission["contract"] == complete["contract"]
            and admission["job_id"] == complete["job_id"] == exit_receipt["job_id"]
            and str(complete["job_id"]).isdigit() and exit_receipt["returncode"] == 0,
            "Training admission/exit differs")
    checkpoint = output / "checkpoints" / f"checkpoint-{step}"
    files = material["files"]
    required = {"adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json",
                "trainer_state.json", "optimizer.pt", "scheduler.pt", "budget_resume_contract.json"}
    required.update(f"rng_state_{rank}.pth" for rank in range(4))
    required.update(f"external_rank_state_{rank}.json" for rank in range(4))
    required.update(f"{owner}_rank{rank}.pt" for rank in range(4) for owner in ("optimizer", "scheduler"))
    if method.startswith("rloo_"):
        required.add("external_baseline_state.json")
    kl = {"exact_peft_kl_reference.json", "exact_peft_kl_reference_step1.json", "exact_peft_kl_reference_post_step.json"}
    require((kl <= set(files)) if method == "foarl" else not kl.intersection(files),
            "FOARL/beta-zero KL inventory differs")
    require(required <= set(files) and all(Path(bound["path"]) == checkpoint / name
            and Path(name).name == name for name, bound in files.items()), "Incomplete or foreign policy inventory")
    rank_state = material["rank_state"]
    require(set(rank_state["receipts"]) == {str(rank) for rank in range(4)}
            and all(files.get(name) == bound for name, bound in rank_state["files"].items()),
            "Checkpoint rank inventory differs")
    for rank in range(4):
        row = rank_state["receipts"][str(rank)]
        require(row["schema_version"] == 1 and row["status"] == "passed" and row["rank"] == rank
                and row["world_size"] == 4 and row["global_step"] == step,
                "Checkpoint rank receipt differs")
        require(set(row["states"]) == {"optimizer", "scheduler"}, "Missing rank optimizer/scheduler state")
        for owner in ("optimizer", "scheduler"):
            state = row["states"][owner]
            filename = f"{owner}_rank{rank}.pt"
            require(state["file"] == filename and state["file_sha256"] == files[filename]["sha256"]
                    and rank_state["files"].get(filename) == files[filename],
                    "Rank state file hash differs")
    require(job["prior_step"] > 0 and isinstance(complete["live_rank_restoration"], dict)
            and set(complete["live_rank_restoration"]) == {str(rank) for rank in range(4)},
            "Missing resumed-rank restoration evidence")
    for rank in range(4):
        restored = complete["live_rank_restoration"][str(rank)]
        prior = job["prior_step"]
        require(Path(restored["file"]["path"]) == output / "checkpoints" / f"external_resume_{prior}_rank{rank}.json",
                "Restoration receipt belongs to another chunk")
        check(restored["file"])
        row = read(restored["file"]["path"])
        require(row == restored["receipt"] and row["status"] == "passed"
                and row["rank"] == rank and row["world_size"] == 4 and row["global_step"] == prior
                and row["checkpoint"] == str(output / "checkpoints" / f"checkpoint-{prior}"),
                "Live rank restoration differs")
    if archived_index is None:
        for bound in files.values():
            check(bound)
        require(read(checkpoint / common.SEAL_NAME, sealed=True) == material, "Checkpoint inventory seal differs")
        state = read(checkpoint / "trainer_state.json")
        require(type(state["global_step"]) is int and state["global_step"] == step
                and state["max_steps"] == horizon, "Live trainer state milestone/horizon differs")
        resume = read(checkpoint / "budget_resume_contract.json")
        resumed_contract = resume["contract"]
        require(resume["schema_version"] == 1 and resume["status"] == "passed"
                and resume["global_step"] == step and resume["world_size"] == 4
                and resume["contract_sha256"] == digest(resumed_contract)
                and resumed_contract["protocol"] == PROTOCOL
                and resumed_contract["method"] == method and resumed_contract["task"] == task
                and resumed_contract["seed"] == 43 and resumed_contract["horizon"] == horizon
                and resumed_contract["contract_file"] == complete["contract"]
                and resumed_contract["declaration"] == contract
                and resumed_contract["source"] == value["source"], "Live resume contract differs")
        for rank in range(4):
            require(read(checkpoint / f"external_rank_state_{rank}.json") == rank_state["receipts"][str(rank)],
                    "Embedded rank state differs from live checkpoint")
        fidelity = module(Path(value["source"]["root"]) / "src/clasp/models/checkpoint_fidelity.py")
        require(fidelity.verify_checkpoint_fidelity(checkpoint)["receipt_sha256"]
                == material["checkpoint_fidelity_receipt_sha256"], "Checkpoint fidelity differs")
    else:
        verify_signed(archived_index)
        require(archived_index["status"] == "complete_raw_evaluation_canonical_audit"
                and archived_index["lineage_snapshot"] == complete
                and archived_index["run_contract_snapshot"] == contract,
                "Archived checkpoint lacks matching original complete audit")
    return complete, contract, str(checkpoint)


def prepare_contract(*, source, campaign_declaration, cohort_declaration, runtime_python,
                     task, method, step, horizon, terminal_audit, output_root, output_path):
    """Create the immutable per-arm contract; return its path/hash/size identity."""
    value = {"protocol": EVALUATION_PROTOCOL, "schema_version": 1, "source": source,
             "bundle": source, "campaign_declaration": campaign_declaration,
             "cohort_declaration": cohort_declaration, "runtime_python": runtime_python,
             "task": task, "method": method, "step": step, "horizon": horizon,
             "terminal_audit": terminal_audit, "output_root": str(output_root)}
    campaign, _, common = context(value)
    validate_checkpoint(value, campaign, common)
    return write_once(output_path, value)


def load_contract(path, sha):
    bound = identity(path)
    require(bound["sha256"] == sha, "Evaluation contract byte hash differs")
    value = read(path, sealed=True)
    return value, bound, context(value)


def adapted_tools(value, binding, cohort, complete, contract, checkpoint):
    """Reuse canonical generation/re-scoring while replacing only scope adapters."""
    tools = module(Path(cohort["canonical_source"]["root"]) / HELPER, binding["sha256"])
    tools.PROTOCOL = EVALUATION_PROTOCOL
    source = Path(value["source"]["root"])
    frozen_wrapper = source / DIRECTORY / "evaluate_a100.slurm"
    task, method = value["task"], value["method"]
    spec = copy.deepcopy(cohort["tasks"][task]["evaluation"])
    require((spec["count"], spec["start"], spec["split"]) == (100, 0, "eval"), "Changed DEV100 slice")
    spec.update(source=value["source"], runtime_python=value["runtime_python"])
    arm = {"checkpoint": checkpoint, "checkpoint_files": complete["checkpoint"]["files"],
           "lineage_receipt": value["terminal_audit"], "output": value["output_root"],
           "training_recipe": {"task": task, "method": method, "seed": 43,
               "updates": value["step"], "horizon": value["horizon"],
               "responses": value["step"] * 36, "prompt_groups": value["step"] * 4,
               "rollouts_per_prompt": 9, "run_contract": complete["contract"],
               "initialization": contract["initialization"], "source": value["source"],
               "checkpoint_audit": value["terminal_audit"]}}

    def task_tools(current):
        require(current == spec, "Evaluation spec changed")
        sys.path.insert(0, str(source / "src"))
        core = importlib.import_module("clasp.entrypoints.evaluate")
        require(Path(core.__file__).resolve() == source / "src/clasp/entrypoints/evaluate.py",
                "Another numerical evaluator is already imported; use a fresh process")
        strict = module(Path(cohort["canonical_source"]["root"]) / "scripts/compare_development.py")
        return core, strict

    def lineage(t, m, candidate):
        require((t, m) == (task, method) and candidate == arm, "Evaluation arm changed")
        check(value["terminal_audit"])
        require(read(value["terminal_audit"]["path"], sealed=True) == complete,
                "Checkpoint receipt changed during evaluation")
        return complete

    def validate_wrapper(execution, bundle, expected_source):
        require(bundle == expected_source == value["source"]
                and execution["source"] == expected_source
                and execution["evaluation_contract"] == binding,
                "Wrapper source/contract differs")
        require(execution["frozen_slurm"] == identity(frozen_wrapper)
                and execution["source_manifest"] == identity(source / "SOURCE_MANIFEST.sha256"),
                "Frozen wrapper bytes changed")
        require(all(execution["executed_slurm"][key] == execution["frozen_slurm"][key]
                    for key in ("sha256", "size_bytes")), "Actual Slurm script differs from frozen wrapper")
        allocation = execution["allocation"]
        require(allocation["campaign_declaration"] == value["campaign_declaration"]
                and allocation["job"] == f"{task}_{method}_eval_{value['step']}"
                and str(allocation["job_id"]).isdigit(), "Evaluation allocation binding differs")

    def wrapper(executed_slurm, bundle, expected_source):
        require(executed_slurm is not None, "Actual Slurm script required")
        require(os.environ["FB_DECLARATION"] == value["campaign_declaration"]["path"]
                and os.environ["FB_DECLARATION_SHA"] == value["campaign_declaration"]["sha256"]
                and os.environ["BUDGET_EVAL_SOURCE"] == str(source)
                and os.environ["BUDGET_EVAL_CONTRACT"] == binding["path"]
                and os.environ["BUDGET_EVAL_CONTRACT_SHA"] == binding["sha256"],
                "Scheduler environment differs from evaluation contract")
        name = f"{task}_{method}_eval_{value['step']}"
        require(os.environ["FB_JOB"] == name, "Wrong declared evaluation job")
        campaign = read(value["campaign_declaration"]["path"])
        # campaign.py uses sibling common.py; only the manifest-bound directory is added.
        sys.path.insert(0, str(source / DIRECTORY))
        scheduler = module(source / DIRECTORY / "campaign.py")
        scheduler.verify_allocation(campaign, name, campaign["jobs"][name], os.environ["SLURM_JOB_ID"])
        execution = {"executed_slurm": identity(executed_slurm),
            "frozen_slurm": identity(frozen_wrapper), "source_manifest": identity(source / "SOURCE_MANIFEST.sha256"),
            "source": expected_source, "evaluation_contract": binding,
            "allocation": {"campaign_declaration": value["campaign_declaration"],
                           "job": name, "job_id": os.environ["SLURM_JOB_ID"]}}
        validate_wrapper(execution, bundle, expected_source)
        return execution

    original_command = tools.command

    def command(t, m, current, candidate):
        argv = original_command(t, m, current, candidate)
        argv[argv.index("--eval_name") + 1] = f"{t}_{m}_step{value['step']}_of{value['horizon']}_raw8"
        return argv

    def save_index(t, m, current, candidate, benchmark, evidence, registry):
        require((t, m) == (task, method) and current == spec and candidate == arm and registry == binding,
                "Audit index scope changed")
        _, strict = task_tools(spec)
        artifact = strict._load_artifact(evidence["details"]["path"], evidence["run_config"]["path"], label=t + "/" + m)
        result = {"protocol": EVALUATION_PROTOCOL, "status": "complete_raw_evaluation_canonical_audit",
            "task": t, "method": m, "step": value["step"], "horizon": value["horizon"],
            "evaluation_contract": binding, "registry": binding,
            "evaluation_source": value["source"], "dataset": benchmark, "reference": spec["reference"],
            "evidence": evidence, "lineage_snapshot": complete, "run_contract_snapshot": contract,
            "summary": tools.summarize(list(artifact.records.values()), t, m, spec["reference"]["kind"])}
        write_once(Path(arm["output"]) / "result_index.json", result)
        return signed(result)

    tools.task_tools = task_tools
    tools.validate_lineage = lineage
    tools.wrapper_execution = wrapper
    tools.validate_wrapper_execution = validate_wrapper
    tools.command = command
    tools.save_arm_index = save_index
    return tools, spec, arm


def audit_contract(path, sha, *, run=False, executed_slurm=None):
    value, binding, (campaign, cohort, common) = load_contract(path, sha)
    require(Path(sys.executable).resolve() == Path(value["runtime_python"]).resolve(),
            "Use the declared native evaluation runtime")
    index_path = Path(value["output_root"]) / "result_index.json"
    archived = read(index_path, sealed=True) if index_path.exists() and not run else None
    if archived:
        require(archived["evaluation_contract"] == binding, "Archived index contract differs")
    complete, contract, checkpoint = validate_checkpoint(value, campaign, common, archived_index=archived)
    tools, spec, arm = adapted_tools(value, binding, cohort, complete, contract, checkpoint)
    os.environ.update(tools.REPRESENTATION)
    if run:
        tools.run_evaluation(value["task"], value["method"], spec, arm, binding,
                             executed_slurm=executed_slurm, bundle=value["source"])
    core, strict = tools.task_tools(spec)
    benchmark = tools.dataset_contract(value["task"], spec, core)
    artifact, evidence = tools.audit_arm(value["task"], value["method"], spec, arm, core, benchmark, binding)
    # Canonical raw replay checks all candidates; also bind the exact command and allocation job.
    launch = read(Path(value["output_root"]) / "launch.json", sealed=True)
    require(launch["protocol"] == EVALUATION_PROTOCOL
            and launch["command"] == tools.command(value["task"], value["method"], spec, arm)
            and launch["execution"]["allocation"]["job_id"] == launch["job_id"],
            "Original evaluation command/allocation changed")
    tools.save_arm_index(value["task"], value["method"], spec, arm, benchmark, evidence, binding)
    return {"value": value, "binding": binding, "index": identity(index_path),
            "index_value": read(index_path, sealed=True), "artifact": artifact,
            "tools": tools, "strict": strict, "spec": spec, "dataset": benchmark}


def report_task(contract_bindings, output_path):
    require(len(contract_bindings) == len(METHODS), "Four evaluation contract bindings required")
    values = []
    for binding in contract_bindings:
        check(binding)
        values.append(read(binding["path"], sealed=True))
    first = values[0]
    require({value["method"] for value in values} == set(METHODS), "Missing or duplicate method")
    for value in values:
        validate_scope(value)
        require(all(value[key] == first[key] for key in
                    ("task", "step", "horizon", "source", "campaign_declaration", "cohort_declaration", "runtime_python")),
                "Mixed task/milestone/horizon/source/cohort comparison")
    results = {value["method"]: audit_contract(binding["path"], binding["sha256"])
               for value, binding in zip(values, contract_bindings, strict=True)}
    initializations = [result["index_value"]["run_contract_snapshot"]["initialization"] for result in results.values()]
    require(all(init == initializations[0] for init in initializations), "Methods start from different SFT policies")
    tools, strict = results[METHODS[0]]["tools"], results[METHODS[0]]["strict"]
    kind = results[METHODS[0]]["spec"]["reference"]["kind"]
    artifacts = {method: results[method]["artifact"] for method in METHODS}
    for method in METHODS:
        require(results[method]["dataset"] == results[METHODS[0]]["dataset"], "Cohort rows/reference differ")
        strict._validate_pairing(artifacts[METHODS[0]], artifacts[method])
    comparisons = {left + "_vs_" + right: {
        "strict_comparison": strict.build_receipt(artifacts[left], artifacts[right], left_label=left, right_label=right),
        **tools.paired_quality(artifacts[left], artifacts[right], kind, strict)}
        for left, right in itertools.combinations(METHODS, 2)}
    common = [key for key in artifacts[METHODS[0]].records
              if all(artifacts[method].records[key].pass_at_k for method in METHODS)]
    thresholds = set(tools.GAP_THRESHOLDS.values()) | {
        tools.signed_gap(row, kind) for artifact in artifacts.values() for row in artifact.records.values() if row.pass_at_k}
    cdf = [{"task": first["task"], "method": method, "signed_gap_threshold": threshold,
            "instances": 100, "success_fraction": sum(row.pass_at_k and tools.signed_gap(row, kind) <= threshold + 1e-12
                for row in artifacts[method].records.values()) / 100}
           for threshold in sorted(thresholds) for method in METHODS]
    rows = [results[method]["index_value"]["summary"] for method in METHODS]
    payload = {"protocol": EVALUATION_PROTOCOL, "status": "complete_four_methods_within_task_milestone",
        "task": first["task"], "step": first["step"], "horizon": first["horizon"],
        "is_terminal": first["step"] == first["horizon"],
        "campaign_declaration": first["campaign_declaration"], "cohort_declaration": first["cohort_declaration"],
        "source": first["source"], "initialization": initializations[0],
        "arm_indexes": {method: results[method]["index"] for method in METHODS},
        "evaluation_contracts": {method: results[method]["binding"] for method in METHODS},
        "dataset": results[METHODS[0]]["dataset"], "reference": results[METHODS[0]]["spec"]["reference"],
        "summary": rows, "paired_comparisons": comparisons, "quality_cdf": cdf,
        "all_four_joint_success": {"count": len(common), "mean_signed_gap": {
            method: statistics.fmean(tools.signed_gap(artifacts[method].records[key], kind) for key in common)
                    if common else None for method in METHODS}},
        "units": {"summary_feasibility_and_gap": "percent",
                  "paired_differences_and_95pct_CIs": "right minus left, fraction; multiply by 100 for percentage points"},
        "limitations": ["One seed43 on inspected DEV100; no guarantee of improvement or confirmation claim.",
            "Feasible@1 is sample zero of the eight generated candidates, not greedy decoding.",
            "Gap is signed relative to the declared reference; own-success and paired joint-success denominators differ.",
            "Failed outputs have no gap; the all-instance quality CDF retains all 100 instances.",
            "Paired instance bootstrap intervals do not measure training-seed variation and are not multiplicity adjusted.",
            "This is a common B9 prompt-budget comparison, not exact reproduction of the released FOARL recipe.",
            "Checkpoint files were checked at admission; retained raw policy hashes and audit snapshots survive checkpoint rotation."]}
    receipt = write_once(output_path, payload)
    table = Path(output_path).with_suffix(".csv")
    if not table.exists():
        with table.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "audit", "report-task"))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--contract-sha")
    parser.add_argument("--contracts", nargs="+", type=Path)
    parser.add_argument("--executed-slurm", type=Path)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        require(args.request and args.output, "Prepare requires request JSON and contract output")
        result = {"status": "evaluation_contract_prepared", "contract":
                  prepare_contract(**read(args.request), output_path=args.output)}
    elif args.action == "report-task":
        require(args.contracts and args.output, "Report requires four contracts and output JSON")
        result = {"status": "complete_four_methods_within_task_milestone", "report":
                  report_task([identity(path) for path in args.contracts], args.output)}
    else:
        require(args.contract and args.contract_sha, "Pinned evaluation contract required")
        audited = audit_contract(args.contract, args.contract_sha, run=args.action == "run",
                                 executed_slurm=args.executed_slurm)
        result = {"status": "complete_raw_evaluation_canonical_audit", "index": audited["index"]}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
