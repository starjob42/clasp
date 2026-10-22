import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

SCRIPT = Path(__file__).resolve().parents[1] / "analysis_outputs/clasp_grpo_fair_dev_20260905/current/audit_adapter_endpoint_deltas.py"
SPEC = importlib.util.spec_from_file_location("adapter_endpoint_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_aggregate_relative_l2_uses_concatenated_parameters():
    group = audit.blank()
    audit.add(group, np.array([3.0, 4.0]), np.array([0.0, 0.0]))
    audit.add(group, np.array([12.0]), np.array([12.0]))
    result = audit.finish(group)
    assert result["reference_l2"] == 13
    assert result["net_endpoint_delta_l2"] == 5
    assert result["relative_net_endpoint_delta_l2"] == pytest.approx(5 / 13)
    assert result["changed_element_fraction"] == pytest.approx(2 / 3)


def test_cpu_safetensors_comparison_retains_small_f32_change_and_bf16_bias(tmp_path):
    reference, endpoint = tmp_path / "before.safetensors", tmp_path / "after.safetensors"
    before = {"layer.lora_A.weight": torch.tensor([1.0, 2.0]),
              "layer.lora_B.weight": torch.zeros(2),
              "layer.base_layer.bias": torch.tensor([1.0, -2.0], dtype=torch.bfloat16)}
    after = {key: value.clone() for key, value in before.items()}
    after["layer.lora_A.weight"][0] = torch.nextafter(torch.tensor(1.0), torch.tensor(float("inf")))
    after["layer.lora_B.weight"] = torch.tensor([3.0, 4.0])
    save_file(before, reference)
    save_file(after, endpoint)
    before_meta, _ = audit.inventory(reference)
    after_meta, _ = audit.inventory(endpoint)
    result = audit.compare(reference, endpoint, before_meta, after_meta)
    assert result["groups"]["lora_A"]["changed_elements"] == 1
    assert result["groups"]["lora_A"]["net_endpoint_delta_l2"] == 2 ** -23
    assert result["groups"]["lora_B"]["net_endpoint_delta_l2"] == 5
    assert result["groups"]["lora_B"]["relative_net_endpoint_delta_l2"] is None
    assert result["groups"]["bias"]["changed_elements"] == 0
    assert result["bias_initial_storage_resolution"]["next_representable_increment_median"] == 2 ** -7
    after_meta["layer.lora_A.weight"]["dtype"] = "BF16"
    with pytest.raises(ValueError, match="dtype/shape"):
        audit.compare(reference, endpoint, before_meta, after_meta)
