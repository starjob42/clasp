from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_jiang_f638_sft_environment_h100.slurm"


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_dev_gate_is_exactly_two_hours_on_one_h100_dev_cell() -> None:
    text = _text()
    assert "#SBATCH --constraint=h100" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --time=02:00:00" in text
    assert "#SBATCH --qos=qos_gpu_h100-dev" in text
    assert re.search(r"^#SBATCH --job-name=.*_iclr$", text, re.MULTILINE)
    assert "${SLURM_JOB_ID:?the development gate must run as a Slurm job}" in text


def test_dev_gate_requires_isolated_create_once_iclr_paths() -> None:
    text = _text()
    assert "BASE_SFT_SMOKE_OUTPUT_ROOT" in text
    assert "BASE_SFT_SMOKE_CACHE_ROOT" in text
    assert '"${safe_root}" == *_iclr*' in text
    assert "development and protected paths overlap" in text
    assert 'mkdir "${run_dir}"' in text
    assert 'mkdir "${job_cache}"' in text
    assert 'receipt_path.open("x"' in text
    assert "stat.S_IRUSR | stat.S_IRGRP" in text
    assert "dev_gate_success_iclr.json" in text


def test_dev_gate_uses_short_node_local_tmpdir_for_multiprocess_ipc() -> None:
    text = _text()
    assert 'mktemp -d "/tmp/c_iclr_${job_id}.XXXXXX"' in text
    assert 'export TMPDIR="${node_tmpdir}"' in text
    assert 'trap \'rm -rf -- "${node_tmpdir}"\' EXIT' in text
    assert '${job_cache}/tmp_iclr' not in text


def test_dev_gate_audits_complete_executable_dependency_inventory() -> None:
    text = _text()
    assert 'expected_versions["pip"] = "25.1.1"' in text
    assert "requirements.jiang-f638-runtime-closure.txt" in text
    assert "runtime_closure_versions == expected_runtime_closure_versions" in text
    for requirement in (
        '"annotated-types": "0.7.0"',
        '"cffi": "1.17.1"',
        '"distro": "1.9.0"',
        '"platformdirs": "4.3.8"',
        '"pycparser": "2.22"',
        '"pydantic": "2.11.7"',
        '"pydantic-core": "2.33.2"',
        '"typing-inspection": "0.4.1"',
    ):
        assert requirement in text
    assert "metadata.distributions()" in text
    assert "duplicate distribution" in text
    assert "len(expected_versions) == 177" in text
    assert "len(expected_versions) == 186" in text
    assert "assert observed_versions == expected_versions" in text
    assert '"missing": sorted(set(expected_versions) - set(observed_versions))' in text
    assert '"unexpected": sorted(set(observed_versions) - set(expected_versions))' in text


def test_dev_gate_executes_exact_masked_lora_bias_optimizer_update() -> None:
    text = _text()
    assert "import unsloth  # Must precede Transformers/TRL" in text
    assert 'bias="all"' in text
    assert "enable_qwen25_targeted_bias_training(model)" in text
    assert 'optim="adamw_8bit"' in text
    assert "per_device_train_batch_size=4" in text
    assert "gradient_accumulation_steps=4" in text
    assert "audit_effective_sft_config(smoke_trainer.args)" in text
    assert "build_strict_response_collator(loaded_tokenizer)" in text
    assert 'for key in ("input_ids", "attention_mask")' in text
    assert 'assert "input_ids" in collator_feature' in text
    assert "data_collator([collator_feature])" in text
    assert "data_collator([smoke_trainer.train_dataset[0]])" not in text
    assert "loaded_tokenizer.pad_token_id is not None" in text
    assert "train_result = smoke_trainer.train()" in text
    assert "smoke_trainer.state.global_step) == 2" in text
    assert '"value_changing_optimizer_updates": 1' in text
    assert "changed_bias_keys" in text
    assert "changed_lora_keys" in text


def test_dev_gate_persists_and_independently_reloads_the_policy() -> None:
    text = _text()
    assert "smoke_trainer.save_model(str(adapter_dir))" in text
    assert "loaded_tokenizer.save_pretrained(str(adapter_dir))" in text
    assert "audit_and_write_checkpoint_fidelity(" in text
    assert "scripts/audit_lora_persistence.py" in text
    assert "scripts/audit_sft_reload_equivalence.py" in text
    assert "verify_checkpoint_fidelity(adapter_dir)" in text
    assert "validate_audit_receipt(" in text
    assert "validate_reload_receipt(" in text


def test_dev_gate_wandb_identity_and_storage_are_iclr_offline() -> None:
    text = _text()
    assert "export WANDB_MODE=offline" in text
    assert 'export WANDB_DIR="${wandb_dir}"' in text
    assert 'WANDB_PROJECT="clasp_jiang_f638_sft_dev2h_iclr"' in text
    assert 'WANDB_RUN_GROUP="jiang_f638_sft_dev2h_iclr_job${job_id}"' in text
    assert 'WANDB_NAME="jiang_f638_sft_dev2h_iclr_job${job_id}"' in text
    assert 'report_to="wandb"' in text
    assert 'wandb_dir.rglob("*.wandb")' in text
    assert '"_iclr" in evidence["wandb"][field]' in text


def test_embedded_python_programs_compile() -> None:
    text = _text()
    programs = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", text, flags=re.DOTALL)
    assert len(programs) == 2
    for index, program in enumerate(programs):
        compile(program, f"{SCRIPT}:heredoc-{index}", "exec")
