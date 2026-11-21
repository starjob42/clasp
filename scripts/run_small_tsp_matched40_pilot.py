#!/usr/bin/env python3
"""Fail-closed launcher for the small-TSP matched 40-step development pilot.

The static JSON contract binds the data, initialization, implementation, runtime,
method-only diffs, and seed-to-GPU mapping.  ``preflight`` is read-only.  ``launch``
creates one immutable run receipt and refuses existing output, resume, overwrite, or
adoption.  Nothing produced by this script is publication or SOTA evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "experiments/additional_evidence/small_tsp_canonical_v1/matched40_v1.contract.json"
)
PROTOCOL = "clasp-small-tsp-matched40-development-v1"
RUN_RECEIPT = "matched40_run_contract.json"
ERROR_PREFIX = "[SmallTSPMatched40Error]"
SCRUB_PREFIXES = ("CLASP_", "PD_", "GRPO_", "GENERATION_", "WANDB_", "CUDA_")
SCRUB_NAMES = {
    "RESUME_FROM_CHECKPOINT",
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
}
MATCHED_RESOLVED_KEYS = (
    "tasks",
    "model_path",
    "tokenizer_path",
    "data_root",
    "prompt_style",
    "no_reason",
    "dtype",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "eval_ratio",
    "eval_from_train_split",
    "max_samples_per_task",
    "max_eval_samples_per_task",
    "min_train_samples_per_task",
    "min_eval_samples_per_task",
    "prefer_subset_files",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "effective_train_batch_size",
    "learning_rate",
    "optim",
    "weight_decay",
    "warmup_ratio",
    "lr_scheduler_type",
    "max_steps",
    "max_prompt_length",
    "max_completion_length",
    "num_generations",
    "resolved_generation_batch_size",
    "generation_temperature",
    "generation_top_p",
    "generation_top_k",
    "generation_repetition_penalty",
    "grpo_beta",
    "grpo_epsilon",
    "grpo_epsilon_high",
    "grpo_loss_type",
    "verify_gt_rewards",
    "report_to",
)
RESOLUTION_FIELDS = MATCHED_RESOLVED_KEYS + (
    "use_primal_dual",
    "use_clasp",
    "pd_group_scalarization",
    "pd_constraint_target",
    "pd_cost_limit",
    "pd_dual_lr",
    "pd_violation_ema_beta",
    "pd_lambda_max",
    "clasp_canonical_set",
    "clasp_deployment_set_size",
    "clasp_canonical_recovery_scale",
    "clasp_canonical_recovery_temperature",
    "clasp_canonical_max_subsets",
    "clasp_adaptive_length",
    "clasp_proper_dual",
    "clasp_self_replay",
)

# The completed seed-101 receipt predates these parser fields.  They were added
# later with inert ``False`` defaults, so absence in that immutable receipt is
# interpreted as the exact versioned default rather than as a wildcard.  A
# future default change (or a true value on either side) still fails the full
# parser comparison below.
POST_HOC_PARSER_DEFAULTS_VERSION = "clasp-small-tsp-matched40-parser-defaults-v1"
POST_HOC_PARSER_DEFAULTS: dict[str, object] = {
    "clasp_fixed_null_set": False,
    "pd_separate_z": False,
}


def fail(message: str) -> NoReturn:
    raise RuntimeError(f"{ERROR_PREFIX} {message}")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        fail(f"cannot fingerprint {path}: {exc}")
    identity = lambda item: (  # noqa: E731 - compact immutable stat projection
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after):
        fail(f"artifact changed while being fingerprinted: {path}")
    return digest.hexdigest()


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} from {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object: {path}")
    return value


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path.absolute() if path.is_absolute() else (PROJECT_ROOT / path).absolute()


def load_contract() -> dict[str, Any]:
    contract = load_json_object(CONTRACT_PATH, label="matched40 contract")
    if (
        contract.get("schema_version") != 1
        or contract.get("protocol") != PROTOCOL
        or contract.get("status") != "READY_DEVELOPMENT_ONLY"
        or contract.get("development_only") is not True
        or contract.get("publication_eligible") is not False
        or contract.get("sota_eligible") is not False
    ):
        fail("contract status/scope is not the approved development-only schema")
    if set(contract.get("arms", {})) != {
        "canonical",
        "canonical_no_controller",
        "dual_scalar",
        "standard_grpo",
    }:
        fail("contract arm inventory drift")
    if contract.get("seeds") != [101, 202, 303]:
        fail("contract seed inventory drift")
    if contract.get("data_seed") != 20260815:
        fail("contract must keep the completed canonical data seed fixed at 20260815")
    return contract


def verify_bindings(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    observed: list[dict[str, Any]] = []
    bindings = contract.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        fail("contract has no artifact bindings")
    for binding in bindings:
        if not isinstance(binding, dict):
            fail("contract binding is not an object")
        path = resolve_path(str(binding.get("path", "")))
        if not path.is_file():
            fail(f"bound artifact is missing: {path}")
        actual = sha256_file(path)
        expected = str(binding.get("sha256", ""))
        if actual != expected:
            fail(f"SHA-256 drift for {path}: expected={expected}, observed={actual}")
        observed.append(
            {
                "path": str(path),
                "resolved_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": actual,
            }
        )

    data_manifest = load_json_object(
        resolve_path(str(contract["paths"]["data_root"])) / "manifest.json",
        label="small-TSP data manifest",
    )
    if not (
        data_manifest.get("development_only") is True
        and data_manifest.get("publication_eligible") is False
        and data_manifest.get("instance_overlap_count") == 0
        and data_manifest.get("train", {}).get("rows") == 90
        and data_manifest.get("eval", {}).get("rows") == 31
    ):
        fail("small-TSP data manifest no longer proves a 90/31 disjoint development split")

    adapter_root = resolve_path(str(contract["paths"]["sft_adapter"]))
    adapter = load_json_object(adapter_root / "adapter_config.json", label="SFT adapter config")
    status = load_json_object(adapter_root / "stage_status.json", label="SFT stage status")
    expected_modules = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    if not (
        adapter.get("r") == 16
        and adapter.get("lora_alpha") == 16
        and adapter.get("bias") == "none"
        and set(adapter.get("target_modules", [])) == expected_modules
    ):
        fail("SFT adapter is not the contracted r16/a16 bias-none Jiang module set")
    if not (
        status.get("completed") is True
        and status.get("global_step") == 80
        and status.get("max_steps") == 80
        and status.get("has_saved_model") is True
    ):
        fail("SFT initialization is not the completed 80-step saved adapter")
    return observed


def probe_runtime(
    contract: Mapping[str, Any],
    run: Any = subprocess.run,
) -> dict[str, Any]:
    python = resolve_path(str(contract["paths"]["python"]))
    if not python.is_file() or not os.access(python, os.X_OK):
        fail(f"contracted Python is missing or not executable: {python}")
    code = (
        "import inspect,json,platform,torch,transformers,trl,peft,datasets,bitsandbytes;"
        "from trl import GRPOConfig;"
        "p=inspect.signature(GRPOConfig.__init__).parameters;"
        "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
        "'transformers':transformers.__version__,'trl':trl.__version__,"
        "'peft':peft.__version__,'datasets':datasets.__version__,"
        "'bitsandbytes':bitsandbytes.__version__,'cuda':torch.version.cuda or 'none',"
        "'trl_defaults':{'importance_sampling_level':p['importance_sampling_level'].default,"
        "'scale_rewards':p['scale_rewards'].default}},sort_keys=True))"
    )
    completed = run(
        [str(python), "-c", code],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        fail(f"runtime probe failed: {completed.stderr.strip()}")
    try:
        observed = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        fail(f"runtime probe returned invalid JSON: {exc}")
    if observed != contract.get("expected_runtime"):
        fail(f"runtime drift: expected={contract.get('expected_runtime')}, observed={observed}")
    return observed


def output_dir(contract: Mapping[str, Any], arm: str, seed: int) -> Path:
    arm_spec = contract["arms"][arm]
    root = resolve_path(str(contract["paths"]["output_root"]))
    return root / str(arm_spec["output_template"]).format(seed=seed)


def build_environment(
    contract: Mapping[str, Any],
    arm: str,
    seed: int,
    inherited: Mapping[str, str],
    *,
    run_contract_path: Path | None = None,
) -> dict[str, str]:
    result = {
        key: value
        for key, value in inherited.items()
        if key not in SCRUB_NAMES and not key.startswith(SCRUB_PREFIXES)
    }
    result.update({str(k): str(v) for k, v in contract["common_environment"].items()})
    result.update({str(k): str(v) for k, v in contract["arms"][arm]["environment"].items()})
    gpu = int(contract["gpu_by_seed"][str(seed)])
    result.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONHASHSEED": str(seed),
            "GRPO_DATA_SEED": str(contract["data_seed"]),
        }
    )
    if run_contract_path is not None:
        result["CLASP_TRAINING_RUN_CONTRACT_PATH"] = str(run_contract_path.absolute())
    return result


def build_train_argv(
    contract: Mapping[str, Any], arm: str, seed: int, run_dir: Path
) -> list[str]:
    python = resolve_path(str(contract["paths"]["python"]))
    return [
        str(python),
        "-m",
        str(contract["arms"][arm]["entrypoint"]),
        *[str(value) for value in contract["shared_argv"]],
        "--seed",
        str(seed),
        "--output_dir",
        str(run_dir.relative_to(PROJECT_ROOT)),
        *[str(value) for value in contract["arms"][arm]["argv"]],
    ]


def resolve_cli(
    contract: Mapping[str, Any], arm: str, seed: int, *, full: bool = False
) -> dict[str, Any]:
    run_dir = output_dir(contract, arm, seed)
    command = build_train_argv(contract, arm, seed, run_dir)
    parser_argv = command[3:]
    if arm == "standard_grpo":
        imports = (
            "from clasp.config import finalize_grpo_args;"
            "from clasp.entrypoints.train_grpo_standard import build_normal_grpo_parser;"
            "a=finalize_grpo_args(build_normal_grpo_parser().parse_args(sys.argv[1:]));"
        )
    else:
        imports = (
            "from clasp.config import build_grpo_parser,finalize_grpo_args;"
            "a=finalize_grpo_args(build_grpo_parser().parse_args(sys.argv[1:]));"
        )
    code = "import json,sys;" + imports + "print(json.dumps(vars(a),sort_keys=True))"
    env = build_environment(contract, arm, seed, os.environ)
    completed = subprocess.run(
        [command[0], "-c", code, *parser_argv],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        fail(f"{arm} CLI no longer parses: {completed.stderr.strip()}")
    try:
        namespace = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        fail(f"{arm} CLI resolution returned invalid JSON: {exc}")
    # run_grpo records these two runtime-resolved values back into the argparse
    # namespace after parser finalization.  Mirror that deterministic step for
    # byte-for-field comparison against the already completed canonical run.
    namespace["pd_importance_sampling_level"] = env["PD_IMPORTANCE_SAMPLING_LEVEL"]
    namespace["pd_allow_token_importance_ablation"] = (
        env["PD_ALLOW_TOKEN_IMPORTANCE_ABLATION"] == "1"
    )
    if full:
        return namespace
    return {key: namespace.get(key) for key in RESOLUTION_FIELDS}


def validate_resolutions(contract: Mapping[str, Any], seed: int) -> dict[str, Any]:
    full_resolved = {
        arm: resolve_cli(contract, arm, seed, full=True) for arm in contract["arms"]
    }
    resolved = {
        arm: {key: namespace.get(key) for key in RESOLUTION_FIELDS}
        for arm, namespace in full_resolved.items()
    }
    anchor = resolved["canonical"]
    mismatches: dict[str, dict[str, Any]] = {}
    for key in MATCHED_RESOLVED_KEYS:
        values = {arm: value.get(key) for arm, value in resolved.items()}
        if any(value != values["canonical"] for value in values.values()):
            mismatches[key] = values
    if mismatches:
        fail(f"matched CLI fields drifted across arms: {mismatches}")
    expected_common = {
        "num_generations": 9,
        "resolved_generation_batch_size": 9,
        "effective_train_batch_size": 9,
        "max_steps": 40,
        "max_prompt_length": 1536,
        "max_completion_length": 160,
        "grpo_beta": 0.0,
        "grpo_loss_type": "grpo",
    }
    if any(anchor.get(key) != value for key, value in expected_common.items()):
        fail(f"resolved common protocol drift: expected={expected_common}, observed={anchor}")
    expected_methods = {
        "canonical": (True, True, True, False, 8),
        "canonical_no_controller": (True, True, True, False, 8),
        "dual_scalar": (True, False, False, True, 0),
        "standard_grpo": (False, False, False, False, 0),
    }
    for arm, expected in expected_methods.items():
        actual = (
            resolved[arm]["use_primal_dual"],
            resolved[arm]["use_clasp"],
            resolved[arm]["clasp_canonical_set"],
            resolved[arm]["pd_group_scalarization"],
            resolved[arm]["clasp_deployment_set_size"],
        )
        if actual != expected:
            fail(f"{arm} method resolution drift: expected={expected}, observed={actual}")
    if contract["arms"]["canonical"]["environment"]["CLASP_BON_DUAL_LR"] != "0.1":
        fail("controller-on canonical dual LR drift")
    if contract["arms"]["canonical_no_controller"]["environment"]["CLASP_BON_DUAL_LR"] != "0":
        fail("controller-off canonical must freeze the multiplier with dual_lr=0")
    if not (
        contract["common_environment"]["PD_IMPORTANCE_SAMPLING_LEVEL"] == "token"
        and contract["common_environment"]["PD_ALLOW_TOKEN_IMPORTANCE_ABLATION"] == "1"
    ):
        fail("matched constrained token-level importance-ratio acknowledgement drift")

    reference_path = resolve_path(
        str(contract["paths"]["completed_canonical_seed101_hyperparams"])
    )
    reference = load_json_object(reference_path, label="completed canonical hyperparams").get(
        "args"
    )
    if not isinstance(reference, dict):
        fail("completed canonical hyperparams lacks an args object")
    reference_values = {
        key: (
            reference[key]
            if key in reference
            else POST_HOC_PARSER_DEFAULTS.get(key)
        )
        for key in set(reference) | set(full_resolved["canonical"])
    }
    reference_differences = {
        key: {
            "completed": reference_values[key],
            "contracted": full_resolved["canonical"].get(key),
        }
        for key in sorted(set(reference) | set(full_resolved["canonical"]))
        if key not in {"output_dir", "seed"}
        and reference_values[key] != full_resolved["canonical"].get(key)
    }
    if reference_differences:
        fail(
            "contracted canonical CLI differs from the completed seed-101 canonical run: "
            f"{reference_differences}"
        )

    diff_keys = sorted(
        key
        for key in RESOLUTION_FIELDS
        if len({json.dumps(values.get(key), sort_keys=True) for values in resolved.values()}) > 1
    )
    return {
        "resolved": resolved,
        "matched_keys": list(MATCHED_RESOLVED_KEYS),
        "matched_field_mismatches": {},
        "method_diff": {
            key: {arm: values.get(key) for arm, values in resolved.items()}
            for key in diff_keys
        },
        "known_nonmatches": contract["known_nonmatches"],
        "completed_canonical_reference_diff": {
            "reference": str(reference_path),
            "ignored_bookkeeping_fields": ["output_dir", "seed"],
            "post_hoc_parser_defaults": {
                "version": POST_HOC_PARSER_DEFAULTS_VERSION,
                "values": dict(POST_HOC_PARSER_DEFAULTS),
                "applied_to_missing_historical_keys": sorted(
                    key for key in POST_HOC_PARSER_DEFAULTS if key not in reference
                ),
            },
            "differences": reference_differences,
        },
    }


def validate_gpu(
    contract: Mapping[str, Any],
    seed: int,
    run: Any = subprocess.run,
) -> dict[str, Any]:
    expected_index = int(contract["gpu_by_seed"][str(seed)])
    completed = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        fail(f"nvidia-smi failed: {completed.stderr.strip()}")
    selected: list[str] | None = None
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4 and fields[0] == str(expected_index):
            selected = fields
            break
    if selected is None:
        fail(f"contracted physical GPU {expected_index} is missing")
    _, name, total_raw, used_raw = selected
    try:
        total_mib, used_mib = int(total_raw), int(used_raw)
    except ValueError:
        fail(f"nvidia-smi returned invalid memory values: {selected}")
    if "2080 Ti" not in name or not 10_500 <= total_mib <= 12_000:
        fail(f"GPU {expected_index} is not the contracted RTX 2080 Ti: {selected}")
    if used_mib > 1_024:
        fail(f"GPU {expected_index} is not idle enough: used={used_mib} MiB")
    return {
        "physical_index": expected_index,
        "name": name,
        "memory_total_mib": total_mib,
        "memory_used_mib_at_launch": used_mib,
    }


def require_fresh_output(run_dir: Path) -> None:
    if run_dir.exists() or run_dir.is_symlink():
        fail(
            "create-once output already exists; resume, overwrite, and adoption are forbidden: "
            f"{run_dir}"
        )


def write_json_exclusive(path: Path, payload: Mapping[str, Any], *, mode: int = 0o440) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    path.chmod(mode)


def create_run_receipt(
    run_dir: Path,
    payload: Mapping[str, Any],
) -> Path:
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(mode=0o750)
    except FileExistsError:
        fail(f"output appeared during create-once reservation: {run_dir}")
    unsigned = dict(payload)
    unsigned["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    path = run_dir / RUN_RECEIPT
    try:
        write_json_exclusive(path, unsigned)
    except BaseException as exc:
        fail(f"run was reserved but its immutable receipt could not be written: {exc}")
    return path


def render_command(command: Sequence[str], environment: Mapping[str, str]) -> str:
    visible = {
        key: value
        for key, value in environment.items()
        if key.startswith(SCRUB_PREFIXES) or key == "PYTHONHASHSEED"
    }
    assignments = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(visible.items())
    )
    argv = " ".join(shlex.quote(value) for value in command)
    return f"{assignments} {argv}".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "launch"))
    parser.add_argument(
        "--arm",
        choices=("all", "canonical", "canonical_no_controller", "dual_scalar", "standard_grpo"),
        default="all",
    )
    parser.add_argument("--seed", type=int, choices=(101, 202, 303), default=101)
    parser.add_argument(
        "--acknowledge-development-only",
        action="store_true",
        help="Required for launch; acknowledges that these runs cannot support publication claims.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = load_contract()
    if args.action == "launch" and args.arm == "all":
        fail("launch requires one explicit --arm")
    if args.action == "launch" and not args.acknowledge_development_only:
        fail("launch requires --acknowledge-development-only")

    arm_names = list(contract["arms"]) if args.arm == "all" else [args.arm]
    for arm in arm_names:
        require_fresh_output(output_dir(contract, arm, args.seed))
    bindings = verify_bindings(contract)
    runtime = probe_runtime(contract)
    config_diff = validate_resolutions(contract, args.seed)
    plans = []
    for arm in arm_names:
        run_dir = output_dir(contract, arm, args.seed)
        command = build_train_argv(contract, arm, args.seed, run_dir)
        environment = build_environment(contract, arm, args.seed, os.environ)
        plans.append(
            {
                "arm": arm,
                "seed": args.seed,
                "gpu": int(contract["gpu_by_seed"][str(args.seed)]),
                "output_dir": str(run_dir),
                "command": render_command(command, environment),
            }
        )
    if args.action == "preflight":
        print(
            json.dumps(
                {
                    "status": "preflight_passed_development_only",
                    "contract": str(CONTRACT_PATH),
                    "contract_sha256": sha256_file(CONTRACT_PATH),
                    "plans": plans,
                    "config_diff": config_diff,
                    "limitations": contract["limitations"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    arm = args.arm
    run_dir = output_dir(contract, arm, args.seed)
    gpu = validate_gpu(contract, args.seed)
    command = build_train_argv(contract, arm, args.seed, run_dir)
    temporary_env = build_environment(contract, arm, args.seed, os.environ)
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "authorized_development_only_not_publication_eligible",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "arm": arm,
        "seed": args.seed,
        "gpu": gpu,
        "contract": {
            "path": str(CONTRACT_PATH),
            "sha256": sha256_file(CONTRACT_PATH),
        },
        "launcher": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "bindings": bindings,
        "runtime": runtime,
        "resolved_cli": config_diff["resolved"][arm],
        "config_diff": config_diff,
        "argv": command,
        "environment": {
            key: value
            for key, value in sorted(temporary_env.items())
            if key.startswith(SCRUB_PREFIXES) or key == "PYTHONHASHSEED"
        },
        "safety": {
            "create_once": True,
            "resume_allowed": False,
            "overwrite_allowed": False,
            "existing_run_adoption_allowed": False,
        },
        "limitations": contract["limitations"],
    }
    receipt = create_run_receipt(run_dir, payload)
    environment = build_environment(
        contract,
        arm,
        args.seed,
        os.environ,
        run_contract_path=receipt,
    )
    print(f"[SmallTSPMatched40] receipt={receipt}", flush=True)
    print(f"[SmallTSPMatched40] arm={arm} seed={args.seed} development_only=true", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
    outcome = {
        "schema_version": 1,
        "protocol": f"{PROTOCOL}-outcome",
        "arm": arm,
        "seed": args.seed,
        "returncode": int(completed.returncode),
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_contract_sha256": sha256_file(receipt),
    }
    write_json_exclusive(run_dir / "matched40_launcher_outcome.json", outcome)
    return int(completed.returncode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
