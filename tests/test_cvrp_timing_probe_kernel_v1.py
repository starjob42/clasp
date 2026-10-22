"""Actual tiny PEFT prefix equivalence; no native allocation or heldout calls."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from transformers import AutoTokenizer
from clasp.baselines.sft_anchor_pilot import make_contract
from clasp.entrypoints import profile_cvrp_timing_probe_v1 as m

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "experiments/rloo_sft_anchor_20261013/validation/v2_peft_cpu_all_arms_v1/cvrp"

def contract():
    original = json.loads((FIXTURE / "contract.json").read_bytes())
    return make_contract("cvrp", original["initializer"], original["training_rows"], original["validation_rows"],
        train_seed=original["train_seed"], eval_seed=original["eval_seed"], horizon=100,
        scope="system_smoke", max_completion_tokens=1)

def tokenizer():
    value = AutoTokenizer.from_pretrained(FIXTURE / "fixture_base", local_files_only=True)
    value.padding_side = "left"; value.model_input_names = ["input_ids", "attention_mask"]
    return value

@pytest.mark.parametrize("fault", ["task", "horizon", "temperature", "top_p", "top_k", "learning_rate", "responses", "groups"])
def test_probe_never_changes_the_h100_numeric_recipe(fault):
    value = contract()
    if fault == "task": value["task"] = "op"
    elif fault == "horizon": value["training"]["horizon"] = 3
    else: value["training"][{"responses": "responses_per_step", "groups": "prompt_groups_per_step"}.get(fault, fault)] = .95
    value["contract_sha256"] = m.v2.digest({key: member for key, member in value.items() if key != "contract_sha256"})
    with pytest.raises(ValueError): m.admit_contract(value, fixture=True)

def same_numerics(left, right):
    need = ("optimizer_steps", "responses", "actual_windows_count", "actual_windows_sha256", "generated_tokens",
        "initial_actor_state_sha256", "actor_state_sha256", "reference_state_sha256", "optimizer_state_sha256", "rng_state_sha256",
        "lr_scheduler_state_sha256", "actor_parameter_dtype_counts", "effective_checkpointing", "checkpointing_stage_counts")
    assert len(left["distributed"]["rank_closure"]) == len(right["distributed"]["rank_closure"])
    for a, b in zip(left["distributed"]["rank_closure"], right["distributed"]["rank_closure"]):
        assert {key: a[key] for key in need} == {key: b[key] for key in need}
        assert a["optimizer_steps"] == 3 and a["responses"] == 108 and a["actual_windows_count"] == 3
        assert a["reference_full_integrity_verified"]
    for result in (left, right):
        assert result["original_horizon"] == 100 and result["committed_prefix_updates"] == 3
        assert result["actor_discarded"] and not result["model_checkpoint_written"]
        assert result["evaluation_responses"] == result["heldout_policy_calls"] == 0
        assert not result["study_admission"] and not result["timing_representativeness_or_full_budget_adequacy_proven"]

def test_actual_tiny_peft_original_h100_first3_instrumented_equals_stock_single_rank(tmp_path, monkeypatch):
    torch.set_num_threads(1); monkeypatch.setenv("WORLD_SIZE", "1"); monkeypatch.setenv("RANK", "0")
    values = []
    for instrumented in (False, True):
        values.append(m.run_probe(contract(), tmp_path / str(instrumented), tokenizer(), torch.device("cpu"),
            base=FIXTURE / "fixture_base", sft=FIXTURE / "fixture_sft", source_admission={"synthetic_fixture": True},
            fixture=True, instrumented=instrumented))
    same_numerics(*values)
    assert len(values[1]["distributed"]["rank_closure"][0]["step_timings"]) == 3
    assert len(values[1]["distributed"]["rank_closure"][0]["stage_measurements"]) > 3
    assert not any(path.name.startswith("checkpoint-") for path in tmp_path.rglob("*"))

def test_actual_tiny_four_rank_original_prefix_preserved_with_passive_instrumentation(tmp_path):
    contract_path = tmp_path / "contract.json"; contract_path.write_text(json.dumps(contract()))
    values = []
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": str(ROOT / "src")}
    for mode in ("control", "profile"):
        output = tmp_path / mode
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
            str(Path(__file__)), "--worker", "--mode", mode, "--contract", str(contract_path), "--output", str(output)]
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180, text=True)
        (tmp_path / (mode + ".log")).write_text(result.stdout)
        assert result.returncode == 0, result.stdout[-10000:]
        values.append(json.loads((output / "timing_probe.json").read_bytes()))
    same_numerics(*values)
    assert values[1]["distributed"]["world_size"] == 4
    assert len({row["actor_state_sha256"] for row in values[1]["distributed"]["rank_closure"]}) == 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--worker", action="store_true"); parser.add_argument("--mode")
    parser.add_argument("--contract", type=Path); parser.add_argument("--output", type=Path); args = parser.parse_args()
    torch.set_num_threads(1); torch.distributed.init_process_group(backend="gloo")
    m.run_probe(json.loads(args.contract.read_bytes()), args.output, tokenizer(), torch.device("cpu"),
        base=FIXTURE / "fixture_base", sft=FIXTURE / "fixture_sft", source_admission={"synthetic_fixture": True},
        fixture=True, instrumented=args.mode == "profile")

@pytest.mark.parametrize("fault", ["contract", "base", "sft", "output", "source", "executed_buffer", "producer_boolean"])
def test_native_kernel_cannot_reuse_admission_for_another_contract_initializer_or_output(monkeypatch, fault):
    value = {"contract_sha256": "777f6a911e2bca723ccc992d5e07978f7930fd2eeec89e592c3bd963c6e7fcef"}
    digest = m.hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
    monkeypatch.setattr(m, "__executed_source_sha256__", digest, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "9990001")
    expected = {"consumer": m.PROTOCOL, "kernel_sha256": digest, "source2_sha256": m.SOURCE2_SHA, "job_id": "9990001",
        "contract_semantic_sha256": value["contract_sha256"], "base": "/original/base", "sft": "/original/sft", "output": "/original/output",
        "native_source_root": str(Path(m.__file__).resolve().parents[3]), "committed_prefix_updates": 3, "original_horizon": 100,
        "study_admission": False, "policy_outcomes_read": False}
    admission = {"protocol": "sft-anchor-cvrp-timing-probe-native-admission-v1", "signed_expectations": expected}
    actual = {"contract": value, "base": expected["base"], "sft": expected["sft"], "output": expected["output"]}
    if fault == "contract": actual["contract"] = {"contract_sha256": "0" * 64}
    elif fault in ("base", "sft", "output"): actual[fault] = "/unregistered"
    elif fault == "source": expected["native_source_root"] = "/copied/source"
    elif fault == "executed_buffer": monkeypatch.delattr(m, "__executed_source_sha256__")
    else: admission = {"native_operator_admitted": True}
    with pytest.raises(ValueError): m.admit_native_operator(admission, **actual)
