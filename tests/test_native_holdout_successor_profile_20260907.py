"""The CPU-only scan may reserve one A100 without confusing requested/allocated CPUs."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def profile(monkeypatch):
    path = Path(__file__).resolve().parents[1] / (
        "analysis_outputs/grpo_incumbent_20260907_v1/run_native_holdout_successor_audit_v1.py"
    )
    spec = importlib.util.spec_from_file_location("a100_cpu_only_wrapper_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, value in {
        "SLURM_JOB_ID": "123", "SLURM_JOB_ACCOUNT": "vas@a100",
        "SLURM_JOB_PARTITION": "gpu_p5", "SLURM_CPUS_PER_TASK": "2", "CUDA_VISIBLE_DEVICES": "",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SLURM_JOB_QOS", raising=False)
    records = {
        "job": {"JobId": "123", "Account": "vas@a100", "QOS": "qos_gpu_a100-dev", "Partition": "gpu_p5",
                "CPUs/Task": "2", "NumTasks": "1", "NumNodes": "1", "NumCPUs": "4", "NodeList": "a100node",
                "AllocTRES": "billing=1,cpu=4,gres/gpu=1,mem=15000M,node=1",
                "ReqTRES": "cpu=2,gres/gpu=1,mem=15000M,node=1"},
        "node": {"NodeName": "a100node", "AvailableFeatures": "Tesla,a100,mps", "CPUTot": "128",
                 "ThreadsPerCore": "2", "Gres": "gpu:8", "CfgTRES": "cpu=128,gres/gpu=8"},
    }

    def run(argv, **kwargs):
        assert argv[:2] == ["scontrol", "show"] and kwargs["timeout"] == 30
        return SimpleNamespace(stdout=" ".join(f"{key}={value}" for key, value in records[argv[2]].items()))

    monkeypatch.setattr(module.subprocess, "run", run)
    return module, records


@pytest.mark.parametrize("allocated", [2, 4, 16])
def test_requested_two_cpus_does_not_assume_logical_allocation(profile, allocated):
    module, records = profile
    records["job"]["AllocTRES"] = f"billing=1,cpu={allocated},gres/gpu=1,mem=15000M,node=1"
    result = module.allocated_a100_cpu_only_profile()
    assert result["requested_cpus_per_task"] == 2
    assert result["actual_allocated_cpus"] == allocated
    assert result["actual_allocated_memory"] == "15000M"
    assert result["reserved_gpus"] == 1 and result["cpu_scan_threads"] == 1
    assert result["gpu_compute_requested"] is False and result["cuda_visible_devices"] == ""


@pytest.mark.parametrize("key,value", [
    ("SLURM_JOB_ACCOUNT", "vas@cpu"), ("SLURM_JOB_PARTITION", "cpu_p1"),
    ("SLURM_CPUS_PER_TASK", "4"), ("CUDA_VISIBLE_DEVICES", "0"),
])
def test_wrong_environment_rejected_before_scan(profile, monkeypatch, key, value):
    module, _ = profile
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        module.allocated_a100_cpu_only_profile()


@pytest.mark.parametrize("key,value", [
    ("Account", "vas@cpu"), ("QOS", "qos_gpu_a100-t3"), ("Partition", "cpu_p1"),
    ("CPUs/Task", "4"), ("NumTasks", "2"), ("NumNodes", "2"),
    ("AllocTRES", "cpu=1,gres/gpu=1"), ("AllocTRES", "cpu=4,gres/gpu=2"),
    ("AllocTRES", "cpu=4,gres/gpu=0"),
])
def test_actual_scheduler_profile_must_match(profile, key, value):
    module, records = profile
    records["job"][key] = value
    with pytest.raises(ValueError):
        module.allocated_a100_cpu_only_profile()


def test_actual_node_must_have_a100_feature(profile):
    module, records = profile
    records["node"]["AvailableFeatures"] = "Tesla,v100"
    with pytest.raises(ValueError, match="A100 feature"):
        module.allocated_a100_cpu_only_profile()


def test_v3_script_has_exact_cpu_only_a100_resource_contract():
    base = Path(__file__).resolve().parents[1] / "analysis_outputs/grpo_incumbent_20260907_v1"
    script = (base / "audit_native_holdout_v3.slurm").read_text()
    assert all(line in script.splitlines() for line in (
        "#SBATCH --account=vas@a100", "#SBATCH --partition=gpu_p5", "#SBATCH --qos=qos_gpu_a100-dev",
        "#SBATCH --gres=gpu:1", "#SBATCH --cpus-per-task=2", "#SBATCH --time=02:00:00",
        'export CUDA_VISIBLE_DEVICES=""',
    ))
    assert "115m" in script and "#SBATCH --mem" not in script


def test_v4_adds_only_explicit_a100_constraint_to_preserved_v3():
    base = Path(__file__).resolve().parents[1] / "analysis_outputs/grpo_incumbent_20260907_v1"
    previous = (base / "audit_native_holdout_v3.slurm").read_bytes()
    current = (base / "audit_native_holdout_v4.slurm").read_bytes()
    directive = b"#SBATCH --constraint=a100\n"
    assert directive not in previous and current.count(directive) == 1
    assert current.replace(directive, b"") == previous
    assert b"#SBATCH --partition=gpu_p5\n" + directive in current


def test_successor_wrapper_only_changes_bound_builder_path():
    base = Path(__file__).resolve().parents[1] / "analysis_outputs/grpo_incumbent_20260907_v1"
    original = (base / "run_native_holdout_audit_v3.py").read_text()
    expected = original.replace("source/scripts/build_grpo_incumbent_holdout_20260907.py",
                                "source/scripts/build_grpo_incumbent_holdout_successor_20260907.py")
    assert original != expected
    assert (base / "run_native_holdout_successor_audit_v1.py").read_text() == expected
