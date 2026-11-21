"""One-attempt native diagnostic admission; no training performance selection."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path
import subprocess

from run_foarl_teacher_dev_chunk import write_once

QOS = "qos_gpu_a100-dev"


def active_jobs():
    return subprocess.check_output(["squeue", "-r", "-u", "uag72kz", "--qos=" + QOS,
                                    "-h", "-o", "%i"], text=True).splitlines()


def command(source, sha, root, name, task, method, stop, resume=None):
    values = {"FT_SOURCE": str(source), "FT_SOURCE_SHA": sha, "FT_TASK": task,
              "FT_METHOD": method, "FT_OUTPUT": str(root / name), "FT_STOP": str(stop),
              "FT_DIAGNOSTIC": "1", "FT_RESUME": str(resume) if resume else "",
              "FT_COMPARE_WITH": str(root / "mis_teacher_full/checkpoints/checkpoint-10") if resume else ""}
    if any("," in value or any(c.isspace() for c in value) for value in values.values()):
        raise ValueError("Unsupported export path")
    return ["sbatch", "--parsable", "--no-requeue", "--time=02:00:00", "--job-name=ftdiag-" + name,
            "--output=" + str(root / "logs" / f"{name}-s{stop}-%j.out"),
            "--error=" + str(root / "logs" / f"{name}-s{stop}-%j.err"),
            "--export=ALL," + ",".join(f"{k}={v}" for k, v in values.items()),
            str(source / "scripts/train_foarl_teacher_dev_a100.slurm")]


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "resume"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    from clasp.baselines.foarl_f638_dev_20260907 import verify_dev_source_manifest
    verify_dev_source_manifest(source / "SOURCE_MANIFEST.sha256", args.source_sha)
    root = args.output.absolute()
    if root.parent != Path("/lustre/fsn1/projects/rech/vas/uag72kz"):
        raise ValueError("Diagnostic output must remain in the authorized native project")
    if args.action == "init":
        if len(active_jobs()) > 6:
            raise RuntimeError("Four diagnostic jobs exceed the current development quota")
        root.mkdir(exist_ok=False)
        (root / "logs").mkdir()
        cases = [("mis_control_full", "mis", "foarl_control", 10),
                 ("mis_teacher_full", "mis", "foarl_teacher", 10),
                 ("mis_teacher_staged", "mis", "foarl_teacher", 5),
                 ("cvrp_teacher_check", "cvrp", "foarl_teacher", 5)]
        plans = [{"key": name, "command": command(source, args.source_sha, root, name, task, method, stop)}
                 for name, task, method, stop in cases]
        write_once(root / "protocol.json", {
            "scope": "correctness_only_no_performance_claim", "source": str(source),
            "source_manifest_sha256": args.source_sha, "plans": plans,
            "resume_plan": command(source, args.source_sha, root, "mis_teacher_staged", "mis",
                                   "foarl_teacher", 10, root / "mis_teacher_staged/checkpoints/checkpoint-5"),
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "fixed_horizon": 400, "native_gpus_per_job": 4, "walltime": "02:00:00",
            "no_automatic_retry": True, "preexisting_dev_jobs_untouched": active_jobs()})
        for plan in plans:
            submit(root, plan["key"], plan["command"])
    else:
        protocol = json.loads((root / "protocol.json").read_text())
        if protocol["source"] != str(source) or protocol["source_manifest_sha256"] != args.source_sha:
            raise ValueError("Diagnostic source changed")
        expected_resume = command(source, args.source_sha, root, "mis_teacher_staged", "mis",
                                  "foarl_teacher", 10, root / "mis_teacher_staged/checkpoints/checkpoint-5")
        if protocol["resume_plan"] != expected_resume:
            raise ValueError("Resume plan differs from the declared staged MIS diagnostic")
        if not (root / "mis_teacher_staged/chunk-5-complete.json").is_file():
            raise ValueError("Staged checkpoint has not passed chunk completion validation")
        for key in ("mis_teacher_staged", "mis_teacher_full"):
            previous = json.loads((root / f"{key}-submission.json").read_text())["stdout"].strip().split(";")[0]
            accounting = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", previous,
                                                  "-o", "JobIDRaw,State,ExitCode"], text=True).splitlines()
            if accounting != [previous + "|COMPLETED|0:0"]:
                raise ValueError("Previous and full stages must complete successfully before resuming")
        submit(root, "mis_teacher_staged-resume10", protocol["resume_plan"])


if __name__ == "__main__":
    main()
