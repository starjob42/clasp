"""One-attempt classic-GRPO correctness diagnostics under a fixed 400-step horizon.

Only init (two MIS jobs) and resume (one MIS 5-to-10 continuation) are admitted.
The operator must serialize submissions across other campaigns; this lock owns
only this fresh diagnostic root. No inference or performance selection occurs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

QOS = "qos_gpu_a100-dev"
PROJECT = Path("/lustre/fsn1/projects/rech/vas/uag72kz")
PROTOCOL = "classic-grpo-400-correctness-diagnostics-v1"
CASES = (("mis_classic_full", 10), ("mis_classic_staged", 5))
REQUIRED = {
    "scripts/classic_grpo_400_diagnostics.py",
    "scripts/train_classic_grpo_400_dev_a100.slurm",
    "scripts/run_classic_grpo_400_dev_chunk.py",
    "scripts/run_foarl_teacher_dev_chunk.py",
    "scripts/compare_classic_grpo_resume.py",
    "scripts/compare_foarl_teacher_resume.py",
    "scripts/probe_foarl_teacher_tokenizer.py",
    "src/clasp/baselines/classic_grpo_400_dev.py",
    "src/clasp/entrypoints/train_classic_grpo_400_dev.py",
    "src/clasp/models/classic_grpo_resume.py",
    "src/clasp/trainers/verified_teacher_grpo.py",
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
    if Path(__file__).resolve() != source / "scripts/classic_grpo_400_diagnostics.py":
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


def command(source, sha, root, name, stop, resume=None):
    values = {"CG_SOURCE": str(source), "CG_SOURCE_SHA": sha, "CG_TASK": "mis",
              "CG_OUTPUT": str(root / name), "CG_STOP": str(stop), "CG_DIAGNOSTIC": "1",
              "CG_RESUME": str(resume) if resume else "",
              "CG_COMPARE_WITH": str(root / "mis_classic_full/checkpoints/checkpoint-10") if resume else ""}
    if any("," in value or any(c.isspace() for c in value) for value in values.values()):
        raise ValueError("Unsupported export path")
    return ["sbatch", "--parsable", "--no-requeue", "--time=02:00:00", "--job-name=cgdiag-" + name,
            "--output=" + str(root / "logs" / f"{name}-s{stop}-%j.out"),
            "--error=" + str(root / "logs" / f"{name}-s{stop}-%j.err"),
            "--export=ALL," + ",".join(f"{k}={v}" for k, v in values.items()),
            str(source / "scripts/train_classic_grpo_400_dev_a100.slurm")]


def declared_plans(source, sha, root):
    return [{"key": name, "command": command(source, sha, root, name, stop)} for name, stop in CASES]


def declared_resume(source, sha, root):
    return command(source, sha, root, "mis_classic_staged", 10,
                   root / "mis_classic_staged/checkpoints/checkpoint-5")


def load_json(path):
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError("Receipt must be an unlinked regular file: " + str(path))
    return json.loads(path.read_text())


def validate_protocol(protocol, source, sha, root):
    expected = {
        "protocol": PROTOCOL, "scope": "correctness_only_no_performance_claim",
        "source": str(source), "source_manifest_sha256": sha,
        "plans": declared_plans(source, sha, root), "resume_plan": declared_resume(source, sha, root),
        "fixed_horizon": 400, "native_gpus_per_job": 4, "walltime": "02:00:00",
        "no_automatic_retry": True, "cross_campaign_submission_authority": "one_sequential_operator",
    }
    material = {k: v for k, v in protocol.items() if k not in ("created_at_utc", "preexisting_dev_jobs_untouched")}
    if material != expected:
        raise ValueError("Diagnostic protocol differs from the declared classic MIS plans")


def validate_checkpoint(checkpoint, step, contract, contract_path, source):
    # No project imports occur until the complete source inventory is verified.
    sys.path.insert(0, str(source / "src"))
    from clasp.models.classic_grpo_resume import validate_classic_grpo_resume
    module = sys.modules[validate_classic_grpo_resume.__module__]
    if Path(module.__file__).resolve() != source / "src/clasp/models/classic_grpo_resume.py":
        raise ValueError("Checkpoint validator was imported from another source")
    return validate_classic_grpo_resume(checkpoint, expected_step=step,
                                       expected_contract=contract, run_contract_path=contract_path)


def admit_resume(root, source, sha):
    protocol = load_json(root / "protocol.json")
    validate_protocol(protocol, source, sha, root)
    contracts, jobs = [], set()
    for name, step in CASES:
        attempt = load_json(root / f"{name}-attempt.json")
        if attempt.get("command") != command(source, sha, root, name, step):
            raise ValueError("Original diagnostic attempt command differs")
        receipt = load_json(root / f"{name}-submission.json")
        job = receipt["stdout"].strip().split(";")[0]
        if receipt["returncode"] != 0 or not job.isdigit() or job in jobs:
            raise ValueError("Initial submission is failed, ambiguous, or duplicated")
        jobs.add(job)
        accounting = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", job,
                                              "-o", "JobIDRaw,State,ExitCode"], text=True).splitlines()
        if accounting != [job + "|COMPLETED|0:0"]:
            raise ValueError("Both full and staged jobs must complete successfully before resuming")
        output = root / name
        contract_path = output / "local_development_run_contract.json"
        contract = load_json(contract_path)
        if (contract.get("source") != {"root": str(source), "manifest_sha256": sha}
                or contract.get("method") != "classic_grpo" or contract.get("diagnostic") is not True):
            raise ValueError("Initial chunk belongs to another source, method, or scope")
        admission = load_json(output / f"chunk-{step}-admission.json")
        if admission.get("job_id") != job or admission.get("stop") != step:
            raise ValueError("Chunk admission belongs to another scheduler job or boundary")
        if load_json(output / f"chunk-{step}-exit.json").get("returncode") != 0:
            raise ValueError("Initial training subprocess failed")
        retained = load_json(output / f"chunk-{step}-complete.json")
        observed = validate_checkpoint(output / "checkpoints" / f"checkpoint-{step}", step,
                                       contract, contract_path, source)
        if retained != observed:
            raise ValueError("Retained chunk completion differs from the current checkpoint")
        contracts.append(contract)
    if contracts[0] != contracts[1]:
        raise ValueError("Full and staged runs have different immutable contracts")
    # The resumed launcher repeats complete input admission under its chunk lock.
    return protocol["resume_plan"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "resume"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = verify_source(args.source, args.source_sha)
    root = args.output.absolute()
    if root.parent != PROJECT or root.resolve() != root or root.is_symlink():
        raise ValueError("Diagnostic output must be a direct unlinked child of the authorized native project")
    if args.action == "init":
        preexisting = active_jobs()
        if len(preexisting) > 8:
            raise RuntimeError("Two diagnostic jobs exceed the current development quota")
        root.mkdir(exist_ok=False)
        (root / "logs").mkdir()
        plans = declared_plans(source, args.source_sha, root)
        write_once(root / "protocol.json", {
            "protocol": PROTOCOL, "scope": "correctness_only_no_performance_claim",
            "source": str(source), "source_manifest_sha256": args.source_sha, "plans": plans,
            "resume_plan": declared_resume(source, args.source_sha, root),
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "fixed_horizon": 400, "native_gpus_per_job": 4, "walltime": "02:00:00",
            "no_automatic_retry": True, "cross_campaign_submission_authority": "one_sequential_operator",
            "preexisting_dev_jobs_untouched": preexisting})
        for plan in plans:
            submit(root, plan["key"], plan["command"])
    else:
        resume = admit_resume(root, source, args.source_sha)
        submit(root, "mis_classic_staged-resume10", resume)


if __name__ == "__main__":
    main()
