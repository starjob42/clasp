"""Declare and explicitly submit strict-deterministic CO correctness diagnostics.

The six initial jobs and two possible continuations are declared before any
submission. Full/staged semantic state must already match at step5 before a
continuation is admitted. This helper never retries or chooses a policy.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

QOS = "qos_gpu_a100-dev"
PROJECT = Path("/lustre/fsn1/projects/rech/vas/uag72kz")
PROTOCOL = "co-strict-deterministic-correctness-diagnostics-20260909-v1"
RUNTIME_PROFILE = "co-strict-deterministic-development-20260909-v1"
CASES = (
    ("mis_foarl_teacher_full", "mis", "foarl_teacher", 10),
    ("mis_foarl_teacher_staged", "mis", "foarl_teacher", 5),
    ("cvrp_foarl_teacher_check", "cvrp", "foarl_teacher", 5),
    ("mis_classic_grpo_full", "mis", "classic_grpo", 10),
    ("mis_classic_grpo_staged", "mis", "classic_grpo", 5),
    ("mis_foarl_control_full", "mis", "foarl_control", 10),
)
RESUME_METHODS = ("foarl_teacher", "classic_grpo")
REQUIRED = {
    "scripts/co_deterministic_diagnostics.py",
    "scripts/run_co_deterministic_dev_chunk.py",
    "scripts/train_co_deterministic_dev_a100.slurm",
    "scripts/compare_co_deterministic_state.py",
    "scripts/run_foarl_teacher_dev_chunk.py",
    "scripts/compare_foarl_teacher_resume.py",
    "scripts/probe_foarl_teacher_tokenizer.py",
    "src/clasp/baselines/co_deterministic_dev.py",
    "src/clasp/entrypoints/train_co_deterministic_dev.py",
    "src/clasp/models/classic_grpo_resume.py",
    "src/clasp/models/foarl_teacher_resume.py",
}


def write_once(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    Path(path).chmod(0o440)


def active_jobs():
    return subprocess.check_output(["squeue", "-r", "-u", "uag72kz", "--qos=" + QOS,
                                    "-h", "-o", "%i"], text=True).splitlines()


def submit(root, key, argv):
    # Durable attempt exists before submission; any ambiguous outcome requires
    # inspection, never automatic retry. Serialize admission across invocations.
    with (root / ".submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queued = set(active_jobs())
        invisible = set()
        unresolved = 0
        for attempt in root.glob("*-attempt.json"):
            prior = attempt.with_name(attempt.name.removesuffix("-attempt.json") + "-submission.json")
            if not prior.is_file():
                unresolved += 1
                continue
            receipt = json.loads(prior.read_text())
            job = receipt["stdout"].strip().split(";")[0]
            if receipt["returncode"] or not job.isdigit():
                # An error does not prove a scheduler side effect was absent.
                unresolved += 1
            elif job not in queued:
                states = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", job,
                                                  "-o", "JobIDRaw,State"], text=True).splitlines()
                terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
                rows = [row.split("|") for row in states if row.split("|")[0] == job]
                if len(rows) != 1 or rows[0][1].split()[0] not in terminal:
                    invisible.add(job)
        if len(queued) + len(invisible) + unresolved >= 10:
            raise RuntimeError("No development job slot available, including unresolved submissions")
        write_once(root / f"{key}-attempt.json", {"command": argv,
                   "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        result = subprocess.run(argv, text=True, capture_output=True)
        receipt = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        write_once(root / f"{key}-submission.json", receipt)
        print(json.dumps({"key": key, **receipt}), flush=True)
        job = result.stdout.strip().split(";")[0]
        if result.returncode or not job.isdigit():
            raise RuntimeError("Failed or ambiguous submission; no retry performed")


def verify_source(source, expected_sha):
    """Verify source bytes before importing a checkpoint validator from that tree."""
    source = Path(source).resolve(strict=True)
    if Path(__file__).resolve() != source / "scripts/co_deterministic_diagnostics.py":
        raise ValueError("Executing diagnostic helper must belong to the declared source")
    manifest = source / "SOURCE_MANIFEST.sha256"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("Source manifest must be a regular file")
    raw = manifest.read_bytes()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Externally pinned source manifest differs")
    found = set()
    for line in raw.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError("Malformed source manifest")
        digest, relative = match.groups()
        locator = PurePosixPath(relative)
        if locator.is_absolute() or ".." in locator.parts or str(locator) != relative or relative in found:
            raise ValueError("Unsafe or duplicate source path")
        path = source / relative
        if path.is_symlink() or path.resolve() != path or not path.is_file():
            raise ValueError("Source entry must be an unlinked regular file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Source entry differs: " + relative)
        found.add(relative)
    required = REQUIRED | {path.relative_to(source).as_posix() for path in (source / "src/clasp").rglob("*.py")}
    if not required <= found:
        raise ValueError("Manifest omits required diagnostic or Python source")
    return source


def load_json(path):
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError("Receipt must be an unlinked regular file: " + str(path))
    return json.loads(path.read_text())


def command(source, sha, root, name, task, method, stop, resume=None):
    values = {"CO_SOURCE": str(source), "CO_SOURCE_SHA": sha, "CO_TASK": task,
              "CO_METHOD": method, "CO_OUTPUT": str(root / name), "CO_STOP": str(stop),
              "CO_DIAGNOSTIC": "1", "CO_RESUME": str(resume) if resume else "",
              "CO_COMPARE_WITH": str(root / f"mis_{method}_full/checkpoints/checkpoint-10") if resume else ""}
    if any("," in value or any(c.isspace() for c in value) for value in values.values()):
        raise ValueError("Unsupported export path")
    return ["sbatch", "--parsable", "--no-requeue", "--time=02:00:00", "--job-name=codet-" + name,
            "--output=" + str(root / "logs" / f"{name}-s{stop}-%j.out"),
            "--error=" + str(root / "logs" / f"{name}-s{stop}-%j.err"),
            "--export=ALL," + ",".join(f"{k}={v}" for k, v in values.items()),
            str(source / "scripts/train_co_deterministic_dev_a100.slurm")]


def declared_plans(source, sha, root):
    return [{"key": name, "command": command(source, sha, root, name, task, method, stop)}
            for name, task, method, stop in CASES]


def declared_resumes(source, sha, root):
    return [{"key": f"mis_{method}_staged-resume10", "method": method,
             "command": command(source, sha, root, f"mis_{method}_staged", "mis", method, 10,
                                root / f"mis_{method}_staged/checkpoints/checkpoint-5")}
            for method in RESUME_METHODS]


def protocol_material(source, sha, root):
    return {"protocol": PROTOCOL, "scope": "correctness_only_no_performance_claim",
            "source": str(source), "source_manifest_sha256": sha,
            "runtime_profile": RUNTIME_PROFILE, "cublas_workspace_config": ":4096:8",
            "plans": declared_plans(source, sha, root), "resume_plans": declared_resumes(source, sha, root),
            "fixed_horizon": 400, "native_gpus_per_job": 4, "walltime": "02:00:00",
            "preresume_gate": "exact_semantic_model_optimizer_scheduler_all_rank_rng_at_step5",
            "no_automatic_retry": True, "cross_campaign_submission_authority": "one_sequential_operator"}


def validate_protocol(protocol, source, sha, root):
    material = {k: v for k, v in protocol.items()
                if k not in ("created_at_utc", "preexisting_dev_jobs_untouched")}
    if material != protocol_material(source, sha, root):
        raise ValueError("Diagnostic protocol differs from its prospective plans")


def require_completed_job(root, name, expected_command):
    attempt = load_json(root / f"{name}-attempt.json")
    if attempt.get("command") != expected_command:
        raise ValueError("Original diagnostic attempt command differs")
    receipt = load_json(root / f"{name}-submission.json")
    job = receipt["stdout"].strip().split(";")[0]
    if receipt["returncode"] != 0 or not job.isdigit():
        raise ValueError("Initial submission is failed or ambiguous")
    raw = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", job,
        "--format=JobIDRaw,User,State,ExitCode,SubmitLine%10000"], text=True).splitlines()
    if len(raw) != 1:
        raise ValueError("Missing unique scheduler record")
    row = raw[0].split("|")
    if len(row) != 5 or row[:4] != [job, "uag72kz", "COMPLETED", "0:0"]:
        raise ValueError("Initial job must have completed successfully under the authorized owner")
    actual = shlex.split(row[4])
    if not actual or Path(actual[0]).name != "sbatch" or ["sbatch", *actual[1:]] != expected_command:
        raise ValueError("Actual scheduler submission differs from the declaration")
    return job


def source_module(source, name, relative):
    """Import only after complete external source verification by main()."""
    import importlib
    sys.path[:0] = [str(source / "scripts"), str(source / "src")]
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != source / relative:
        raise ValueError("Admission module came from another source")
    return module


def validate_checkpoint(checkpoint, step, contract, contract_path, source):
    method = contract.get("method")
    if method == "classic_grpo":
        module = source_module(source, "clasp.models.classic_grpo_resume",
                               "src/clasp/models/classic_grpo_resume.py")
        validate = module.validate_classic_grpo_resume
    elif method in ("foarl_control", "foarl_teacher"):
        module = source_module(source, "clasp.models.foarl_teacher_resume",
                               "src/clasp/models/foarl_teacher_resume.py")
        validate = module.validate_foarl_teacher_resume
    else:
        raise ValueError("Unsupported method")
    return validate(checkpoint, expected_step=step, expected_contract=contract,
                    run_contract_path=contract_path)


def admit_resume(root, source, sha, method):
    if method not in RESUME_METHODS:
        raise ValueError("No continuation declared for this method")
    protocol = load_json(root / "protocol.json")
    validate_protocol(protocol, source, sha, root)
    contracts, jobs = [], set()
    for suffix, step in (("full", 10), ("staged", 5)):
        name = f"mis_{method}_{suffix}"
        expected = command(source, sha, root, name, "mis", method, step)
        job = require_completed_job(root, name, expected)
        if job in jobs:
            raise ValueError("Initial diagnostic jobs must be distinct")
        jobs.add(job)
        output = root / name
        contract_path = output / "local_development_run_contract.json"
        contract = load_json(contract_path)
        if (contract.get("source") != {"root": str(source), "manifest_sha256": sha}
                or contract.get("method") != method or contract.get("diagnostic") is not True):
            raise ValueError("Initial chunk belongs to another source, method, or scope")
        admission = load_json(output / f"chunk-{step}-admission.json")
        if admission.get("job_id") != job or admission.get("stop") != step:
            raise ValueError("Chunk admission belongs to another job or boundary")
        if load_json(output / f"chunk-{step}-exit.json").get("returncode") != 0:
            raise ValueError("Training subprocess failed")
        observed = validate_checkpoint(output / "checkpoints" / f"checkpoint-{step}", step,
                                       contract, contract_path, source)
        if load_json(output / f"chunk-{step}-complete.json") != observed:
            raise ValueError("Retained completion differs from the live checkpoint")
        contracts.append(contract)
    if contracts[0] != contracts[1]:
        raise ValueError("Full and staged runs have different immutable contracts")
    module = source_module(source, "compare_co_deterministic_state",
                           "scripts/compare_co_deterministic_state.py")
    comparison = module.compare_checkpoints(
        root / f"mis_{method}_full/checkpoints/checkpoint-5",
        root / f"mis_{method}_staged/checkpoints/checkpoint-5", expected_step=5)
    proof_path = root / f"{method}-step5-prerestart-equivalence.json"
    if proof_path.exists():
        if load_json(proof_path) != comparison:
            raise ValueError("Existing semantic step5 proof changed; no retry admitted")
    else:
        write_once(proof_path, comparison)
    if comparison.get("status") != "passed_exact_step5_prerestart":
        raise ValueError("Initial semantic states differ: continuation cannot isolate restoration")
    return next(plan["command"] for plan in protocol["resume_plans"] if plan["method"] == method)


def submit_initial(root, source, sha, max_new_jobs):
    protocol = load_json(root / "protocol.json")
    validate_protocol(protocol, source, sha, root)
    if type(max_new_jobs) is not int or not 1 <= max_new_jobs <= 5:
        raise ValueError("One explicit batch may submit 1 to 5 jobs")
    submitted = 0
    for plan in protocol["plans"]:
        attempt = root / (plan["key"] + "-attempt.json")
        if attempt.exists():
            prior = root / (plan["key"] + "-submission.json")
            if not prior.is_file():
                raise ValueError("Unresolved prior attempt; inspect without automatic retry")
            receipt = load_json(prior)
            if receipt.get("returncode") != 0 or not receipt.get("stdout", "").strip().split(";")[0].isdigit():
                raise ValueError("Failed or ambiguous prior attempt; no automatic retry")
            continue
        if submitted == max_new_jobs or len(active_jobs()) >= 10:
            break
        submit(root, plan["key"], plan["command"])
        submitted += 1
    return submitted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "submit", "resume"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-jobs", type=int, default=1)
    parser.add_argument("--method", choices=RESUME_METHODS)
    args = parser.parse_args(argv)
    source = verify_source(args.source, args.source_sha)
    root = args.output.absolute()
    if root.parent != PROJECT or root.resolve() != root or root.is_symlink():
        raise ValueError("Output must be a direct unlinked child of the authorized native project")
    if args.action == "init":
        root.mkdir(exist_ok=False)
        (root / "logs").mkdir()
        protocol = {**protocol_material(source, args.source_sha, root),
                    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "preexisting_dev_jobs_untouched": active_jobs()}
        write_once(root / "protocol.json", protocol)
        print(json.dumps({"status": "declared_without_submission", "initial_plans": len(CASES),
                          "conditional_resumes": len(RESUME_METHODS)}), flush=True)
    elif args.action == "submit":
        count = submit_initial(root, source, args.source_sha, args.max_new_jobs)
        print(json.dumps({"new_jobs_submitted": count}), flush=True)
    else:
        if args.method is None:
            parser.error("resume requires --method")
        argv = admit_resume(root, source, args.source_sha, args.method)
        submit(root, f"mis_{args.method}_staged-resume10", argv)


if __name__ == "__main__":
    main()
