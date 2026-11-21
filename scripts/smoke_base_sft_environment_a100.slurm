#!/bin/bash -l
# Smoke-check a candidate base-SFT environment before it is sealed.
#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=sft-env-smoke

set -euo pipefail
umask 027

environment_prefix="$(readlink -f "${BASE_SFT_CONDA_ENV:?BASE_SFT_CONDA_ENV is required}")"
base_model="$(readlink -f "${BASE_SFT_BASE_MODEL_PATH:?BASE_SFT_BASE_MODEL_PATH is required}")"
cache_root="$(readlink -m "${BASE_SFT_SMOKE_CACHE_ROOT:?BASE_SFT_SMOKE_CACHE_ROOT is required}")"

module purge
module load "${GPU_ARCH_MODULE:-arch/a100}"
module load "${CONDA_MODULE:-miniforge/24.9.0}"
conda activate "${environment_prefix}"

python_bin="$(readlink -f "${CONDA_PREFIX}/bin/python")"
[[ "${CONDA_PREFIX}" == "${environment_prefix}" && -x "${python_bin}" ]] || {
    echo "candidate environment activation resolved an unexpected prefix" >&2
    exit 2
}

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${cache_root}/huggingface"
export HF_HOME="${cache_root}/huggingface"

"${python_bin}" - "${environment_prefix}" "${base_model}" <<'PY'
import importlib.metadata as metadata
import json
import sys

import torch
import unsloth  # Import first, matching the supported patch order.
import accelerate
import bitsandbytes
import datasets
import peft
import safetensors
import transformers
import trl
from transformers import AutoConfig, AutoTokenizer

expected_prefix, base_model = sys.argv[1:]
assert sys.prefix == expected_prefix, (sys.prefix, expected_prefix)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
name = torch.cuda.get_device_name(0)
assert "A100" in name, name

expected_versions = {
    "torch": "2.9.1",
    "transformers": "4.57.3",
    "peft": "0.18.0",
    "trl": "0.24.0",
    "unsloth": "2026.1.4",
    "unsloth-zoo": "2026.1.4",
    "bitsandbytes": "0.49.0",
    "datasets": "4.3.0",
    "accelerate": "1.12.0",
    "safetensors": "0.7.0",
}
observed_versions = {name: metadata.version(name) for name in expected_versions}
assert observed_versions == expected_versions, observed_versions

config = AutoConfig.from_pretrained(
    base_model,
    trust_remote_code=True,
    local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True,
    local_files_only=True,
)
probe = tokenizer("### Instruction:\nSolve the instance.\n\n### Response:\n", add_special_tokens=False)
assert config.model_type == "qwen2"
assert tokenizer.is_fast
assert tokenizer.eos_token_id is not None
assert probe["input_ids"]

print(json.dumps({
    "status": "candidate_sft_environment_smoke_passed",
    "python": sys.executable,
    "prefix": sys.prefix,
    "gpu": name,
    "cuda_runtime": torch.version.cuda,
    "model_type": config.model_type,
    "tokenizer_class": type(tokenizer).__name__,
    "eos_token_id": tokenizer.eos_token_id,
    "versions": observed_versions,
}, sort_keys=True))
PY
