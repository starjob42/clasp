#!/usr/bin/env python3
"""Fail-closed launcher for the local TSP K=8 CLASP development experiment.

This is deliberately separate from every publication/remote launcher.  It runs
one create-once trajectory and never resumes, overwrites, or adopts an existing
directory.  The resulting evidence is development-only and is not eligible for
publication claims.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, NoReturn, Sequence

SCHEMA_VERSION = 1
PROTOCOL = "clasp-local-tsp-k8-development-v1"
ERROR_PREFIX = "[LocalDevelopmentLaunchError]"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = (PROJECT_ROOT.parent / "my_idea_grpo" / ".venv" / "bin" / "python").absolute()
SFT_MODEL = PROJECT_ROOT / "models" / "local_tsp_sft500k_seed42_instruct"
DATA_ROOT = PROJECT_ROOT / "data" / "data_rl"
TRAIN_DATA = DATA_ROOT / "tsp" / "train" / "train_rl.json"
BASE_MODEL = (
    PROJECT_ROOT.parent / "my_idea_grpo" / "seacove_models" / "base" / "Qwen2.5-7B-Instruct"
).resolve()
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tsp_k8_development_20260814"


SFT_MANIFEST = {
    "README.md": "37b7d9d614273fa4a8a1bf3d96becb7c5c30ca22a4fe1f34014c14d780f84591",
    "adapter_config.json": "77a1bbe21c4ed225bcd14a78cc9386b819fb6369f4624ea8cd2dc7fc2702f7fd",
    "adapter_model.safetensors": "9bf9ea50a5f7e9ff57399ea9bcca487785867d95bc9fdc11dadb7c723cef5173",
    "added_tokens.json": "58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b",
    "chat_template.jinja": "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f",
    "hyperparams.json": "7908d8124f65858be515101f265d630ebcca54d0090665a9e648673a30a0038f",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "sft_log_history.json": "ef0cb7a880baf47872cedadec2e0fb6a7a3c29adc2d71871b0f3cdc72dfbcc7f",
    "sft_trainer_state.json": "23ffb305c35a17bc74f3b0851d76b567e92e4fe9b9a689f6bfaa286a84e73be2",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    "stage_status.json": "0b8d662c308caa6af0e2316518ce5adfcf6ca19a105b2474b09a5c081d6a0aeb",
    "tokenizer.json": "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa",
    "tokenizer_config.json": "e6711f2a711bd7e5607373a1770048709e4494bc08ab8fb6b54659943ce55845",
    "training_args.bin": "0f8a99ac79db71f82b6597316ffdaf8d34d99d0eeb2e0fbe4efb1788279e30c8",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
BASE_MANIFEST = {
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "README.md": "f366f33bbf6bcadbb7d87f0a21a7b65584a56b8d58b0743c77c88bee625b93a6",
    "config.json": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
    "generation_config.json": "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model-00001-of-00004.safetensors": "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
    "model-00002-of-00004.safetensors": "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
    "model-00003-of-00004.safetensors": "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
    "model-00004-of-00004.safetensors": "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd",
    "model.safetensors.index.json": "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
TRAIN_DATA_SHA256 = "90c5a48909ac9df191c727237474ec262db4a9c654f3555a19b744304dc0a288"
EXPECTED_TRAIN_SOURCE_ROWS = 3_200
EXPECTED_RUNTIME = {
    "python": "3.11.2",
    "torch": "2.10.0+cu128",
    "transformers": "4.57.6",
    "trl": "0.24.0",
    "peft": "0.19.1",
    "datasets": "4.3.0",
    "bitsandbytes": "0.49.2",
    "cuda": "12.8",
}


@dataclass(frozen=True)
class Mode:
    output_name: str
    max_steps: int
    max_source_rows: int
    max_eval_samples: int
    min_train_samples: int
    min_eval_samples: int
    expected_train_rows: int
    expected_eval_rows: int


MODES = {
    # This path intentionally differs from the earlier successful, unmanaged
    # gate at outputs/.../k8_memory_gate.  Existing evidence is never adopted.
    "memory_gate": Mode(
        output_name="contracted_memory_gate_v1",
        max_steps=1,
        max_source_rows=160,
        max_eval_samples=8,
        min_train_samples=32,
        min_eval_samples=8,
        expected_train_rows=152,
        expected_eval_rows=8,
    ),
    "dense40": Mode(
        output_name="contracted_clasp_dense40_seed42_v1",
        max_steps=40,
        max_source_rows=0,
        max_eval_samples=100,
        min_train_samples=1900,
        min_eval_samples=100,
        expected_train_rows=3040,
        expected_eval_rows=100,
    ),
}


EXPERIMENT_ENV = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "3",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_MODE": "disabled",
    "PYTHONHASHSEED": "42",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONPATH": "src",
    "CLASP_LORA_TARGET_MODULES": "jiang",
    "CLASP_LORA_BIAS": "lora_only",
    "CLASP_TSP_PROMPT_KNN": "0",
    "CLASP_TSP_PROMPT_KNN_DISTANCES": "0",
    "CLASP_TSP_ROUTE_ONLY_OUTPUT": "0",
    "CLASP_OBJECTIVE_QUALITY_MODE": "exp",
    "CLASP_OBJECTIVE_GAP_SLOPE": "3.0",
    "CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT": "0.0",
    "CLASP_BON_ALIGNED": "1",
    "CLASP_FRONTIER_CDF": "0",
    "CLASP_BON_TARGET_FAILURE_RATE": "0.005",
    "CLASP_BON_TARGET_FAILURE_RATES": "tsp:0.005",
    "CLASP_BON_SUCCESS_CREDIT": "dense_rank",
    "CLASP_BON_DENSE_QUALITY_SCALE": "1.0",
    "CLASP_BON_DENSE_FEASIBILITY_SCALE": "1.0",
    "CLASP_BON_DENSE_VIOLATION_MIX": "0.5",
    "CLASP_BON_RECOVERY_SCALE": "1.0",
    "CLASP_COMPONENT_DUAL_RECOVERY_SCALE": "1.0",
    "CLASP_PROPER_DUAL_GRADIENT_MODE": "signed",
    "CLASP_AUX_GROUPING_STRICT": "1",
    "PD_IMPORTANCE_SAMPLING_LEVEL": "sequence",
    "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION": "0",
    "GRPO_DATA_SEED": "42",
    "GRPO_STRICT_OPTIMIZER_RESUME": "1",
}

LIMITATIONS = {
    "development_only": True,
    "publication_eligible": False,
    "sota_claim_eligible": False,
    "instruction_tuned_initialization": (
        "Uses Qwen2.5-7B-Instruct, not the approved non-instruction Qwen2.5-7B base."
    ),
    "runtime_contract_mismatch": (
        "Uses local PEFT 0.19.1; the publication environment requires PEFT >=0.18,<0.19."
    ),
    "single_gpu_mismatch": (
        "Uses one RTX A6000 with generation_batch_size=8 and gradient accumulation; "
        "it is not bit-equivalent to the four-A100 publication trajectory."
    ),
    "scope": "TSP only; no multi-task or confirmation evidence is produced.",
}


def fail(message: str) -> NoReturn:
    raise RuntimeError(f"{ERROR_PREFIX} {message}")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        fail(f"cannot fingerprint {path}: {exc}")

    def identity(file_stat):
        return (
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
        )

    if identity(before) != identity(after):
        fail(f"artifact changed while it was fingerprinted: {path}")
    return digest.hexdigest()


def require_plain_directory(path: Path, label: str) -> Path:
    absolute = path.absolute()
    if absolute.is_symlink() or not absolute.is_dir():
        fail(f"{label} must be a copied, non-symlink directory: {absolute}")
    return absolute.resolve()


def verify_manifest(
    root: Path, expected: Mapping[str, str], *, label: str, exact: bool
) -> dict[str, object]:
    root = require_plain_directory(root, label)
    if exact:
        actual_names = sorted(path.name for path in root.iterdir())
        if actual_names != sorted(expected):
            fail(f"{label} file set mismatch: expected={sorted(expected)}, actual={actual_names}")
    receipts = []
    for relative, expected_hash in sorted(expected.items()):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            fail(f"{label} file must be copied and regular: {path}")
        observed = sha256_file(path)
        if observed != expected_hash:
            fail(
                f"{label} SHA-256 mismatch for {relative}: "
                f"expected={expected_hash}, observed={observed}"
            )
        receipts.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": observed})
    return {"canonical_path": str(root), "files": receipts}


def load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} JSON from {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object: {path}")
    return value


def verify_artifacts() -> dict[str, object]:
    sft = verify_manifest(SFT_MODEL, SFT_MANIFEST, label="SFT model", exact=True)
    adapter = load_json_object(SFT_MODEL / "adapter_config.json", "adapter config")
    if Path(str(adapter.get("base_model_name_or_path", ""))).resolve() != BASE_MODEL:
        fail("adapter base_model_name_or_path does not equal the contracted Instruct base")
    expected_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if not (
        adapter.get("r") == 64
        and adapter.get("lora_alpha") == 64
        and adapter.get("bias") == "lora_only"
        and set(adapter.get("target_modules", [])) == expected_modules
    ):
        fail("SFT adapter is not the audited Jiang-style r64/a64 lora_only artifact")
    status = load_json_object(SFT_MODEL / "stage_status.json", "SFT stage status")
    if not (
        status.get("completed") is True
        and status.get("global_step") == 31_250
        and status.get("max_steps") == 31_250
        and status.get("has_saved_model") is True
    ):
        fail("SFT stage status is not the completed 31,250-step specialist")
    base = verify_manifest(BASE_MODEL, BASE_MANIFEST, label="Instruct base model", exact=True)

    train_dir = require_plain_directory(TRAIN_DATA.parent, "RL train data")
    if sorted(path.name for path in train_dir.iterdir()) != [TRAIN_DATA.name]:
        fail("copied TSP train directory must contain only train_rl.json")
    if TRAIN_DATA.is_symlink() or not TRAIN_DATA.is_file():
        fail(f"RL train data must be a copied, non-symlink file: {TRAIN_DATA}")
    observed_data_hash = sha256_file(TRAIN_DATA)
    if observed_data_hash != TRAIN_DATA_SHA256:
        fail(
            f"RL train SHA-256 mismatch: expected={TRAIN_DATA_SHA256}, "
            f"observed={observed_data_hash}"
        )
    eval_dir = DATA_ROOT / "tsp" / "eval"
    if eval_dir.exists():
        fail(
            "local development data root must not contain tsp/eval; confirmation and "
            "historical evaluation data are outside this launcher"
        )
    try:
        rows = json.loads(TRAIN_DATA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot parse copied RL train data: {exc}")
    if not isinstance(rows, list) or len(rows) != EXPECTED_TRAIN_SOURCE_ROWS:
        fail(
            f"RL train data must contain exactly {EXPECTED_TRAIN_SOURCE_ROWS} rows; "
            f"observed={len(rows) if isinstance(rows, list) else '<non-list>'}"
        )
    return {
        "sft_model": sft,
        "base_model": base,
        "rl_train": {
            "canonical_path": str(TRAIN_DATA.resolve()),
            "size_bytes": TRAIN_DATA.stat().st_size,
            "sha256": observed_data_hash,
            "rows": len(rows),
        },
    }


def build_train_args(mode: Mode, run_dir: Path) -> list[str]:
    common = [
        "-m",
        "clasp.entrypoints.train_grpo",
        "--tasks",
        "tsp",
        "--model_path",
        str(SFT_MODEL.relative_to(PROJECT_ROOT)),
        "--tokenizer_path",
        str(SFT_MODEL.relative_to(PROJECT_ROOT)),
        "--data_root",
        str(DATA_ROOT.relative_to(PROJECT_ROOT)),
        "--output_root",
        str(OUTPUT_ROOT.relative_to(PROJECT_ROOT)),
        "--output_dir",
        str(run_dir.relative_to(PROJECT_ROOT)),
        "--prompt_style",
        "alpaca",
        "--no_reason",
        "--dtype",
        "bfloat16",
        "--lora_r",
        "64",
        "--lora_alpha",
        "64",
        "--lora_dropout",
        "0",
        "--seed",
        "42",
        "--eval_ratio",
        "0.05",
        "--eval-from-train-split",
        "--max_samples_per_task",
        str(mode.max_source_rows),
        "--max_eval_samples_per_task",
        str(mode.max_eval_samples),
        "--min_train_samples_per_task",
        str(mode.min_train_samples),
        "--min_eval_samples_per_task",
        str(mode.min_eval_samples),
        "--no-prefer_subset_files",
        "--per_device_train_batch_size",
        "2",
        "--per_device_eval_batch_size",
        "8",
        "--gradient_accumulation_steps",
        "16",
        "--learning_rate",
        "1e-6",
        "--optim",
        "adamw_8bit",
        "--weight_decay",
        "0.01",
        "--warmup_ratio",
        "0.03",
        "--lr_scheduler_type",
        "linear",
        "--num_train_epochs",
        "1",
        "--max_steps",
        str(mode.max_steps),
        "--max_prompt_length",
        "29768",
        "--max_completion_length",
        "3000",
        "--num_generations",
        "8",
        "--generation_batch_size",
        "8",
        "--generation_temperature",
        "0.7",
        "--generation_top_p",
        "0.9",
        "--generation_top_k",
        "50",
        "--generation_repetition_penalty",
        "1",
        "--grpo_beta",
        "0",
        "--grpo_epsilon",
        "0.2",
        "--grpo_epsilon_high",
        "0.2",
        "--grpo_loss_type",
        "dr_grpo",
        "--verify_gt_rewards",
        "--reward_check_samples_per_task",
        "5",
        "--logging_steps",
        "1",
        "--save_steps",
        str(mode.max_steps),
        "--eval_steps",
        str(mode.max_steps),
        "--save_total_limit",
        "1",
        "--report_to",
        "none",
        "--use_primal_dual",
        "--pd_constraint_target",
        "1",
        "--pd_cost_limit",
        "0",
        "--pd_lambda_init",
        "0",
        "--pd_dual_lr",
        "0.05",
        "--pd_dual_lr_up",
        "0.05",
        "--pd_dual_lr_down",
        "0.05",
        "--pd_dual_update_interval",
        "1",
        "--pd_violation_ema_beta",
        "0.1",
        "--pd_lambda_max",
        "100",
        "--pd_violation_tolerance",
        "0",
        "--pd_objective_weight",
        "1",
        "--pd_structure_weight",
        "0.2",
        "--pd_sync_across_ranks",
        "--pd_require_state_on_resume",
        "--pd_strict_constraint_range",
        "--pd_force_scale_rewards_none",
        "--no-pd_soft_objective_for_infeasible",
        "--no-pd_normalize_primal_advantage",
        "--pd_primal_advantage_scale",
        "1",
        "--pd_penalty_scale",
        "1",
        "--pd_positive_constraint_penalty",
        "--pd_group_scalarization",
        "--pd_component_profile",
        "co_feasibility_lock_all",
        "--no-pd_enable_schedule",
        "--use_clasp",
        "--clasp_strict_defaults",
        "--no-clasp_raw_feasibility_targeting",
        "--clasp_bon_success_credit",
        "dense_rank",
        "--clasp_component_dual_recovery_scale",
        "1",
        "--clasp_feasibility_warmup_steps",
        "0",
        "--clasp_feasibility_warmup_targets",
        "tsp:0.995",
        "--no-clasp_self_replay",
        "--no-clasp_validation_gate_checkpoints",
        "--no-clasp_pareto_anchor",
        "--clasp_proper_dual",
        "--clasp_dual_update_interval",
        "4",
        "--clasp_proper_dual_gradient_mode",
        "signed",
        "--no-clasp_adaptive_length",
        "--clasp_mixed_adaptive_length_mode",
        "error",
    ]
    return common


def build_environment(environ: Mapping[str, str]) -> dict[str, str]:
    result = {
        key: value
        for key, value in environ.items()
        if not key.startswith(("CLASP_", "PD_", "GENERATION_", "GRPO_", "WANDB_"))
    }
    for name in (
        "RESUME_FROM_CHECKPOINT",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        result.pop(name, None)
    result.update(EXPERIMENT_ENV)
    return result


def probe_runtime(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, str]:
    # A virtualenv's python entry point is normally a symlink; resolving it
    # would bypass the virtualenv and silently select the system environment.
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        fail(f"audited Python interpreter is missing or not executable: {PYTHON}")
    code = (
        "import json,platform,torch,transformers,trl,peft,datasets,bitsandbytes;"
        "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
        "'transformers':transformers.__version__,'trl':trl.__version__,'peft':peft.__version__,"
        "'datasets':datasets.__version__,'bitsandbytes':bitsandbytes.__version__,"
        "'cuda':torch.version.cuda or 'none'}))"
    )
    completed = run(
        [str(PYTHON), "-c", code],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        fail(f"cannot probe audited runtime: {completed.stderr.strip()}")
    try:
        observed = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        fail(f"runtime probe returned invalid JSON: {exc}")
    if observed != EXPECTED_RUNTIME:
        fail(f"runtime version drift: expected={EXPECTED_RUNTIME}, observed={observed}")
    return observed


def validate_gpu3(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
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
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            rows.append(fields)
    selected = next((row for row in rows if row[0] == "3"), None)
    if selected is None:
        fail("physical GPU index 3 is not present")
    index, name, total_raw, used_raw = selected
    try:
        total_mib, used_mib = int(total_raw), int(used_raw)
    except ValueError:
        fail(f"nvidia-smi returned invalid memory values for GPU 3: {selected}")
    if "A6000" not in name or total_mib < 45_000:
        fail(f"physical GPU 3 must be the 48 GB RTX A6000; observed={selected}")
    if used_mib > 1_024:
        fail(f"physical GPU 3 is not idle enough for a fresh launch: used={used_mib} MiB")
    return {
        "physical_index": int(index),
        "name": name,
        "memory_total_mib": total_mib,
        "memory_used_mib_at_launch": used_mib,
        "cuda_visible_devices": "3",
        "world_size": 1,
    }


def validate_cli(train_args: Sequence[str], env: Mapping[str, str]) -> dict[str, object]:
    parser_args = list(train_args[2:])
    code = (
        "import json,sys;from clasp.config import build_grpo_parser,finalize_grpo_args;"
        "a=finalize_grpo_args(build_grpo_parser().parse_args(sys.argv[1:]));"
        "print(json.dumps({'tasks':a.tasks,'max_steps':a.max_steps,"
        "'num_generations':a.num_generations,'generation_batch_size':a.resolved_generation_batch_size,"
        "'effective_train_batch_size':a.effective_train_batch_size,'use_clasp':a.use_clasp,"
        "'dense_credit':a.clasp_bon_success_credit,'proper_dual_mode':a.clasp_proper_dual_gradient_mode,"
        "'group_scalarization':a.pd_group_scalarization,'component_profile':a.pd_component_profile}))"
    )
    completed = subprocess.run(
        [str(PYTHON), "-c", code, *parser_args],
        cwd=PROJECT_ROOT,
        env=dict(env),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        fail(f"contracted CLI no longer parses: {completed.stderr.strip()}")
    try:
        resolved = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        fail(f"CLI validation returned invalid JSON: {exc}")
    expected = {
        "tasks": ["tsp"],
        "max_steps": int(parser_args[parser_args.index("--max_steps") + 1]),
        "num_generations": 8,
        "generation_batch_size": 8,
        "effective_train_batch_size": 32,
        "use_clasp": True,
        "dense_credit": "dense_rank",
        "proper_dual_mode": "signed",
        "group_scalarization": True,
        "component_profile": "co_feasibility_lock_all",
    }
    if resolved != expected:
        fail(f"contracted CLI resolution drift: expected={expected}, observed={resolved}")
    return resolved


def render_command(env: Mapping[str, str], command: Sequence[str]) -> str:
    assignments = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(EXPERIMENT_ENV.items())
    )
    return f"{assignments} " + " ".join(shlex.quote(part) for part in command)


def require_fresh_run_dir(run_dir: Path) -> None:
    if run_dir.exists() or run_dir.is_symlink():
        fail(
            f"create-once run directory already exists; resume, overwrite, and adoption are "
            f"forbidden: {run_dir}"
        )


def create_contract_once(run_dir: Path, payload: dict[str, object]) -> Path:
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(mode=0o750)
    except FileExistsError:
        fail(f"run directory appeared during launch; refusing race: {run_dir}")
    unsigned = dict(payload)
    unsigned["contract_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    contract_path = run_dir / "local_development_run_contract.json"
    try:
        descriptor = os.open(contract_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(unsigned, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        contract_path.chmod(0o440)
    except Exception as exc:
        fail(
            f"run directory was reserved but immutable contract creation failed; "
            f"leave it untouched for audit: {exc}"
        )
    return contract_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create-once local TSP K=8 CLASP development launcher."
    )
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create the immutable run contract and execute training; otherwise verify/print only.",
    )
    parser.add_argument(
        "--acknowledge-development-only",
        action="store_true",
        help="Required with --execute: acknowledge that this cannot support publication/SOTA claims.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = MODES[args.mode]
    run_dir = OUTPUT_ROOT / mode.output_name
    require_fresh_run_dir(run_dir)
    if args.execute and not args.acknowledge_development_only:
        fail("--execute requires --acknowledge-development-only")

    artifacts = verify_artifacts()
    runtime = probe_runtime()
    env = build_environment(os.environ)
    train_args = build_train_args(mode, run_dir)
    command = [str(PYTHON), *train_args]
    cli_resolution = validate_cli(train_args, env)

    summary = {
        "protocol": PROTOCOL,
        "mode": args.mode,
        "run_dir": str(run_dir),
        "source_rows": mode.max_source_rows or EXPECTED_TRAIN_SOURCE_ROWS,
        "expected_train_rows": mode.expected_train_rows,
        "expected_eval_rows": mode.expected_eval_rows,
        "development_only": True,
        "publication_eligible": False,
        "command": render_command(env, command),
    }
    if not args.execute:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    gpu = validate_gpu3()
    launcher_hash = sha256_file(Path(__file__).resolve())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "development_only_not_publication_eligible",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": args.mode,
        "task": "tsp",
        "method": "clasp_dense",
        "seed": 42,
        "run_dir": str(run_dir.resolve()),
        "launcher": {"path": str(Path(__file__).resolve()), "sha256": launcher_hash},
        "artifacts": artifacts,
        "runtime": runtime,
        "gpu": gpu,
        "dataset_partition": {
            "source": "copied RL train only",
            "source_rows": mode.max_source_rows or EXPECTED_TRAIN_SOURCE_ROWS,
            "eval_ratio": 0.05,
            "expected_train_rows": mode.expected_train_rows,
            "expected_eval_rows": mode.expected_eval_rows,
            "external_eval_loaded": False,
        },
        "environment": dict(sorted(EXPERIMENT_ENV.items())),
        "argv": command,
        "cli_resolution": cli_resolution,
        "safety": {
            "create_once": True,
            "resume_allowed": False,
            "overwrite_allowed": False,
            "existing_run_adoption_allowed": False,
        },
        "limitations": LIMITATIONS,
    }
    contract_path = create_contract_once(run_dir, payload)
    print(f"[LocalDevelopmentLaunch] contract={contract_path}", flush=True)
    print(f"[LocalDevelopmentLaunch] mode={args.mode} development_only=true", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
