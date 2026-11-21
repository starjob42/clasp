#!/usr/bin/env python3
"""Freeze a self-hashed source and semantic contract for LLMCoSolver f6388991."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

UPSTREAM_COMMIT = "f6388991d243c0aa91d9ee7e2fa18ed1c9e72477"
UPSTREAM_ARCHIVE_SHA256 = "35a6ffdd5b165cdd5b35976a63596d41c818b6994bef09e3d09e9645c97caaf5"

UPSTREAM_FILES = {
    "LICENSE": "b141a9ad01cc6e3c9316aa6e50d96a06fc227c6dd87df92b1c213b0abe3b07a6",
    "README.md": "09b2bfea2a6af3d56457d72d4d34c8039d61319fa5ff5d0e17436fe34eda4050",
    "requirements.txt": "b8c309502184dbb01f01c3d016c70d5d69d04ce03ec770ae7be720424a9a22b6",
    "main_train.py": "a37901e49fcea759f9fb960e67a9c38a6f73aad4da8f79df222aa8e1bde84f13",
    "rl_train.py": "c6b7703e8f621e136e78724a0d4788795aeda58188f51f3351981d6409ccf583",
    "rewards.py": "879f73a0b4f6ae5d4227a8d4ce8dbf1aa5cc966b526c6d76bba5a2c29a2380c5",
    "utils.py": "88b5c9c06fad5025e8d3caf6ed24d0a0b8f97dacefc537b3a60d27c0bd512db1",
    "eval.py": "6f60157115e69fe0550fb43a54ad506281859c473b7d63eb6b14714a944f72c3",
    "Envs/TSPEnv/TSPEnv.py": "7ef914ef627ae731121e2634394d9f1293dbbdbbae46165ba5c2ad201cf0b87b",
    "Envs/CVRPEnv/CVRPEnv.py": "36491447ee6ff1bbec1c911a8f9a0e74620778fe566e708ad970c4ac239baac0",
    "Envs/OPEnv/OPEnv.py": "b9b24145c393aad9d7a63716bc6f96ddf3cb6a74ebe57e6cb6a10a86636d77e2",
    "Envs/MVCEnv/MVCEnv.py": "c248a5f342e203157abb097c20ca3d1e9ec1a1b9005268ef2ce101f83a754ada",
    "Envs/MISEnv/MISEnv.py": "d91cd63c4d67603fd7cecc68f9e6a07982df9d8781adf2882a171982b0fdade7",
    "Envs/JSSPEnv/JSSPEnv.py": "310a741749ef355e86e61a3eeeb8cc1004555895c31a5084f4f297fa6c7d1e05",
}

STACK = {
    "trl": {
        "version": "0.15.2",
        "files": {
            "trainer/grpo_trainer.py": (
                "5ba2c35e866459c7b8f8c453ef2daebb18e8aea6e468c218afa55965d55a29b0"
            ),
        },
        "license": "Apache-2.0",
    },
    "unsloth": {
        "version": "2025.3.19",
        "files": {
            "models/rl.py": "e5082d7650df80dc13ab03aa2c2cb03cc379d47263800e24f61e6b11bd183a30",
            "models/rl_replacements.py": (
                "7e0539e1bec4593e3f1fc955fed9dd821a3d91f743f6c8eead36e55dfa504910"
            ),
        },
        "license": "Apache-2.0",
    },
    "unsloth_zoo": {
        "version": "2025.3.17",
        "files": {
            "rl_replacements.py": (
                "ef89558f62bc1bb65e95cb1c8a28ae25c68f11a2f435e437d7d1e7e9f2a3fc4f"
            ),
        },
        "license": "LGPL-3.0-or-later",
    },
    "peft": {
        "version": "0.17.0",
        "files": {
            "utils/save_and_load.py": (
                "849282f01e69b7055cc499e67f279cde3f489421aea212f59f701f45755de26b"
            ),
            "tuners/lora/model.py": (
                "e7beb9549d6f12f06edb9561191e041ce3e92aab6f098bc6b3650e7ab2d7e2b3"
            ),
            "tuners/lora/layer.py": (
                "7c458f3d26f0c3a7cb90f6de5133814619a87ef858a839d79f94d53e90f0bc2a"
            ),
            "peft_model.py": "2307ebeb101baff53b19ce5a0f109885d99b64ab147d67cba19719b6097c3e10",
        },
        "license": "Apache-2.0",
    },
    "transformers": {
        "version": "4.50.3",
        "files": {
            "models/qwen2/modeling_qwen2.py": (
                "6a119c45ee2a6bb3fdaa6a268d712167dc3dd779aafdd94f87028783696fe616"
            ),
        },
        "license": "Apache-2.0",
    },
}

REPOSITORY_SEAL_PATHS = (
    "src/clasp/baselines/__init__.py",
    "src/clasp/baselines/llmcosolver_f638_utils.py",
    "src/clasp/baselines/llmcosolver_f638_rewards.py",
    "src/clasp/baselines/llmcosolver_f638_objective.py",
    "third_party/llmcosolver_f6388991/LICENSE",
    "third_party/llmcosolver_f6388991/README.md",
    "third_party/llmcosolver_f6388991/rewards.py",
    "third_party/llmcosolver_f6388991/utils.py",
    "scripts/probe_llmcosolver_f638_stack.py",
    "scripts/freeze_llmcosolver_f638_contract.py",
    "experiments/sota_20260814/llmcosolver_f638_stack_probe.json",
    "tests/test_llmcosolver_f638_rewards.py",
    "tests/test_llmcosolver_f638_objective.py",
    "tests/test_llmcosolver_f638_contract.py",
    "tests/test_llmcosolver_f638_stack_probe.py",
    "docs/llmcosolver_f638_corrected_baseline.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def self_hash(value: dict, field: str) -> str:
    payload = dict(value)
    payload[field] = None
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def normalized_source_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").rstrip(b"\n") + b"\n"


def verify_hashes(root: Path, expected: dict[str, str], label: str) -> dict[str, str]:
    observed = {}
    for relative, expected_hash in expected.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} file: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{label} hash mismatch for {relative}: expected {expected_hash}, got {actual_hash}"
            )
        observed[relative] = actual_hash
    return observed


def package_version(package_root: Path, distribution_name: str) -> str:
    for metadata_path in sorted(package_root.parent.glob("*.dist-info/METADATA")):
        text = metadata_path.read_text(encoding="utf-8")
        name_match = re.search(r"(?m)^Name:\s*(.+?)\s*$", text)
        version_match = re.search(r"(?m)^Version:\s*(.+?)\s*$", text)
        if not name_match or not version_match:
            continue
        if name_match.group(1).strip().lower().replace(
            "-", "_"
        ) == distribution_name.lower().replace("-", "_"):
            return version_match.group(1).strip()
    raise ValueError(
        f"Could not find dist-info metadata for {distribution_name} beside {package_root}"
    )


def validate_probe(path: Path) -> dict:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = receipt.get("receipt_sha256")
    if expected_hash != self_hash(receipt, "receipt_sha256"):
        raise ValueError("Pinned-stack probe receipt self-hash mismatch")
    expected_packages = {key: STACK[key]["version"] for key in ("peft", "transformers", "trl")}
    expected_packages["torch"] = "2.7.1"
    if receipt.get("runtime", {}).get("packages") != expected_packages:
        raise ValueError("Pinned-stack probe package versions do not match the contract")
    expected_conclusion = {
        "disable_adapter_reference_is_fixed": False,
        "disable_adapter_reference_is_foundation_base": False,
        "lora_only_checkpoint_persists_live_bias": False,
        "reference_includes_live_trainable_qkv_bias": True,
    }
    if receipt.get("conclusion") != expected_conclusion:
        raise ValueError("Pinned-stack probe did not establish the required reference semantics")
    observations = receipt.get("observations", {})
    if observations.get("trainable_bias_count") != 3:
        raise ValueError("Probe did not expose Qwen2 q/k/v trainable biases")
    if observations.get("saved_bias_tensor_count") != 0:
        raise ValueError("Probe unexpectedly persisted lora_only bias tensors")
    if observations.get("disabled_logits_changed_after_bias_mutation") is not True:
        raise ValueError("Probe did not demonstrate evolving disabled-adapter logits")
    return receipt


def validate_upstream_semantic_anchors(upstream_root: Path) -> None:
    rl_source = (upstream_root / "rl_train.py").read_text(encoding="utf-8")
    required_rl_fragments = (
        "reward_funcs = [optimality_reward_func_cvrp, feasibility_reward_func_cvrp]",
        "reward_funcs = [optimality_reward_func_op, feasibility_reward_func_op]",
        "reward_funcs = [optimality_reward_func_tsp, feasibility_reward_func_tsp]",
        "reward_funcs = [optimality_reward_func_mvc, feasibility_reward_func_mvc]",
        "reward_funcs = [optimality_reward_func_mis, feasibility_reward_func_mis]",
        "reward_funcs = [optimality_reward_func_jssp, feasibility_reward_func_jssp]",
        "parser.add_argument('--beta', type=float, default=0.05",
        "parser.add_argument('--bias', type=str, default='lora_only'",
        "parser.add_argument('--num_generations', type=int, default=8",
        "parser.add_argument('--seed', type=int, default=42",
    )
    for fragment in required_rl_fragments:
        if fragment not in rl_source:
            raise ValueError(f"Missing pinned rl_train semantic anchor: {fragment}")

    requirements = {}
    for line in (upstream_root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line:
            package, version = line.split("==", 1)
            requirements[package.strip().lower()] = version.strip()
    expected_versions = {
        "peft": "0.17.0",
        "torch": "2.7.1",
        "transformers": "4.50.3",
        "trl": "0.15.2",
        "unsloth": "2025.3.19",
        "unsloth_zoo": "2025.3.17",
    }
    for package, version in expected_versions.items():
        if requirements.get(package) != version:
            raise ValueError(
                f"requirements.txt mismatch for {package}: {requirements.get(package)!r} != {version!r}"
            )


def build_contract(args: argparse.Namespace) -> dict:
    repository_root = args.repository_root.resolve()
    upstream_root = args.upstream_root.resolve()
    archive = args.archive.resolve()
    if sha256_file(archive) != UPSTREAM_ARCHIVE_SHA256:
        raise ValueError("Upstream archive hash does not match pinned commit archive")
    upstream_hashes = verify_hashes(upstream_root, UPSTREAM_FILES, "upstream")
    validate_upstream_semantic_anchors(upstream_root)

    roots = {
        "trl": args.trl_root.resolve(),
        "unsloth": args.unsloth_root.resolve(),
        "unsloth_zoo": args.unsloth_zoo_root.resolve(),
        "peft": args.peft_root.resolve(),
        "transformers": args.transformers_root.resolve(),
    }
    stack_projection = {}
    for package, specification in STACK.items():
        observed_version = package_version(roots[package], package)
        if observed_version != specification["version"]:
            raise ValueError(
                f"{package} version mismatch: {observed_version!r} != {specification['version']!r}"
            )
        stack_projection[package] = {
            "version": observed_version,
            "license": specification["license"],
            "files": verify_hashes(roots[package], specification["files"], package),
        }

    vendor_root = repository_root / "third_party/llmcosolver_f6388991"
    for source_name in ("rewards.py", "utils.py", "LICENSE"):
        if normalized_source_bytes(upstream_root / source_name) != normalized_source_bytes(
            vendor_root / source_name
        ):
            raise ValueError(f"Vendored {source_name} differs from newline-normalized upstream")

    probe_path = args.probe_receipt.resolve()
    probe = validate_probe(probe_path)
    repository_hashes = {}
    for relative in REPOSITORY_SEAL_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing repository contract dependency: {relative}")
        repository_hashes[relative] = sha256_file(path)

    contract = {
        "schema_version": 1,
        "contract_type": "llmcosolver_f638_source_and_semantics",
        "contract_sha256": None,
        "status": "HOLD_NON_EXECUTABLE",
        "upstream": {
            "name": "LLMCoSolver",
            "repository": "https://github.com/Summer142857/LLMCoSolver",
            "commit": UPSTREAM_COMMIT,
            "archive_sha256": UPSTREAM_ARCHIVE_SHA256,
            "root_license": "MIT",
            "files": upstream_hashes,
        },
        "pinned_stack": stack_projection,
        "vendored_oracle": {
            "policy": "upstream source with CRLF and EOF-newline normalization only",
            "license_path": "third_party/llmcosolver_f6388991/LICENSE",
            "rewards_path": "third_party/llmcosolver_f6388991/rewards.py",
            "utils_path": "third_party/llmcosolver_f6388991/utils.py",
        },
        "scope": {
            "official_rl_reward_tasks": ["cvrp", "op", "tsp", "mvc", "mis", "jssp"],
            "clasp_publication_intersection": ["tsp", "cvrp", "op", "mis", "jssp"],
            "mvc_status": "official reward exists; CLASP publication benchmark has no MVC lane",
            "pfsp_status": "environment/evaluation code exists, but rl_train has no PFSP reward mapping",
        },
        "reward_and_loss_semantics": {
            "reward_heads": ["optimality", "feasibility"],
            "head_count": 2,
            "head_aggregation": "sum with implicit unit weights",
            "beta": 0.05,
            "num_generations": 8,
            "advantage_normalization": (
                "within-prompt mean and unbiased sample std over 8 generations, epsilon=1e-4"
            ),
            "loss": (
                "one-iteration on-policy GRPO surrogate plus reverse-KL estimator; token mean "
                "within each completion followed by sequence mean"
            ),
            "current_trl_compatibility": "loss_type=grpo with num_iterations=1, not dr_grpo",
            "prompt": {
                "style": "plain Alpaca with explicit <reasoning></reasoning> permission",
                "utils_literal_sha256": (
                    "8e90f30020283d017f2b7a9c3109d10fde262488a8f36fd16f0da17ba984a606"
                ),
            },
            "differential_oracle_required": True,
            "publication_evaluator": "CLASP strict evaluator; never upstream training rewards",
        },
        "adapter_and_reference_variants": {
            "A_official_code_operational": {
                "classification": "official-code reproduction/context only",
                "executable": False,
                "adapter_bias": "lora_only",
                "reference_mechanism": "same live PEFT model under disable_adapter()",
                "lora_reference": "disabled",
                "qkv_bias_reference": "live trainable base_layer.bias; evolves with policy",
                "fixed_reference": False,
                "checkpoint_fidelity": (
                    "invalid for publication: PEFT 0.17.0 saves zero of the live lora_only q/k/v biases"
                ),
                "resume_equivalence": False,
                "strict_publication_reference": False,
            },
            "B_corrected_frozen_step_zero_sft": {
                "classification": (
                    "corrected semantics-preserving reward/loss port; not byte-identical FOARL reproduction"
                ),
                "executable": False,
                "adapter_bias": "all (capacity-equivalent persistence correction on Qwen2)",
                "reference_mechanism": "separately loaded, frozen step-zero SFT policy",
                "lora_reference": "frozen step-zero adapter active",
                "qkv_bias_reference": "frozen step-zero values",
                "fixed_reference": True,
                "checkpoint_fidelity": "must pass CLASP exact no-merge post-step verifier",
                "resume_equivalence": True,
                "strict_publication_reference": True,
            },
        },
        "pinned_stack_probe": {
            "path": str(args.probe_receipt.resolve().relative_to(repository_root)),
            "receipt_sha256": probe["receipt_sha256"],
            "observations": probe["observations"],
            "conclusion": probe["conclusion"],
        },
        "model_data_and_budget_evidence": {
            "sft_defaults": {
                "base_model": "unsloth/Qwen2.5-7B",
                "max_seq_length": 20000,
                "lora_r": 64,
                "lora_alpha": 64,
                "bias": "lora_only",
                "per_device_batch": 4,
                "gradient_accumulation": 4,
                "epochs": 1,
                "learning_rate": 0.0002,
                "warmup_steps": 20,
                "trainer_seed": 42,
                "lora_random_state": 42,
                "shuffle_seed": 42,
                "save_steps": 5000,
            },
            "rl_defaults": {
                "default_checkpoint": (
                    "output_alpha64_r64_cvrp_gamma_train_embed_tok_False_seq20000_b4_ep1/"
                    "checkpoint-31250"
                ),
                "max_prompt_length": 20000,
                "max_completion_length": 1000,
                "lora_r": 64,
                "lora_alpha": 64,
                "per_device_batch": 8,
                "gradient_accumulation": 8,
                "epochs": 1,
                "learning_rate": 0.000001,
                "weight_decay": 0.02,
                "trainer_seed": 42,
                "lora_random_state": 42,
                "save_steps": 50,
            },
            "evaluation_defaults": {
                "num_samples": 100,
                "vanilla_max_new_tokens": 3000,
                "best_of_n": 8,
                "best_of_n_temperature": 0.7,
                "best_of_n_top_p": 0.9,
            },
            "unresolved_publication_requirements": [
                "upstream SFT, RL, and evaluation data artifacts are absent from the source archive",
                "per-task row counts, row hashes, split identities, and exact reference objectives are not pinned",
                "the default checkpoint-31250 string is not proof of every task's initialization budget",
                "the source has no canonical RL max_steps; one epoch depends on the missing train_rl.json size",
                "generator entry-point seeds differ by task and do not freeze the released paper datasets",
                "matching compute/token/checkpoint budgets therefore remains unresolved",
            ],
            "matched_baseline_status": "UNRESOLVED_DATA_AND_BUDGET_PROVENANCE",
        },
        "classification_guards": {
            "current_clasp_method_grpo": (
                "three strict reward heads with dr_grpo; non-FOARL and unmatched"
            ),
            "no_sota_claim_from_this_contract": True,
            "no_training_or_evaluation_authorized": True,
            "both_reference_variants_hold": True,
        },
        "repository_source_projection": repository_hashes,
    }
    contract["contract_sha256"] = self_hash(contract, "contract_sha256")
    return contract


def parse_args() -> argparse.Namespace:
    repository_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=repository_default)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--trl-root", type=Path, required=True)
    parser.add_argument("--unsloth-root", type=Path, required=True)
    parser.add_argument("--unsloth-zoo-root", type=Path, required=True)
    parser.add_argument("--peft-root", type=Path, required=True)
    parser.add_argument("--transformers-root", type=Path, required=True)
    parser.add_argument(
        "--probe-receipt",
        type=Path,
        default=repository_default / "experiments/sota_20260814/llmcosolver_f638_stack_probe.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = build_contract(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(args.output)
    print(contract["contract_sha256"])


if __name__ == "__main__":
    main()
