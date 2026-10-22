"""Actual tiny producer records vs scorer enrichment, never production evidence."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "experiments/rloo_sft_anchor_20261013/validation"
SPEC = importlib.util.spec_from_file_location("actual_admission_v3", ROOT /
    "experiments/rloo_sft_anchor_20261013/collect_campaign_admission_v3.py")
ADMISSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADMISSION)
FIXTURES = sorted(path for name in ("v2_peft_cpu_all_arms_v1", "v2_peft_cuda_ddp2_v1",
                                   "v2_peft_foarl_cuda_ddp2_v1")
                  for path in (VALIDATION / name).rglob("result.json"))
assert len(FIXTURES) == 14


def actual_records(path):
    result = json.loads(path.read_text())
    groups = [json.loads(line) for line in path.with_name("evaluation_groups.jsonl").read_text().splitlines()]
    raw = [record for group in groups for record in group["raw_completions"]]
    return result, raw


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: str(p.relative_to(VALIDATION)))
def test_actual_generation_projection_preserves_every_original_field(path):
    result, generated = actual_records(path)
    assert len(generated) == 36
    assert generated != result["raw_completions"]
    projected = [ADMISSION.raw_generation_projection(row) for row in result["raw_completions"]]
    assert projected == generated
    assert all(set(scored) - set(original) == ADMISSION.SCORER_ENRICHMENT_FIELDS
               for scored, original in zip(result["raw_completions"], generated))
    assert all(projected[i] == generated[i] and set(projected[i]) == set(generated[i])
               for i in range(len(generated)))


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: str(p.relative_to(VALIDATION)))
@pytest.mark.parametrize("field", ["completion_token_ids", "text", "generation_seed", "completion_id"])
def test_actual_generation_corruption_survives_projection_and_is_rejected(path, field):
    result, generated = actual_records(path)
    mutated = copy.deepcopy(result["raw_completions"])
    if field == "completion_token_ids":
        mutated[0][field] = list(mutated[0][field]) + [987654]
    elif field == "generation_seed":
        mutated[0][field] += 1
    else:
        mutated[0][field] += "-corrupt"
    with pytest.raises(ValueError, match="Durable generation rows differ"):
        ADMISSION.require(generated == [ADMISSION.raw_generation_projection(row) for row in mutated],
                          "Durable generation rows differ from the exact raw projection of the scored result")


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: str(p.relative_to(VALIDATION)))
def test_actual_tiny_result_stays_ineligible_for_production_admission(path):
    result, _ = actual_records(path)
    task, arm = result["task"], result["arm"]
    contract = json.loads((path.parent.parent / "contract.json").read_text())
    complete_path = (path.parent.parent.parent / "complete.json")
    complete = json.loads(complete_path.read_text())
    assert contract["scope"] == "system_smoke"
    with pytest.raises(ValueError, match="fixed production recipe"):
        ADMISSION.validate_completed_slot(f"{task}.{arm}", complete, result, contract, {}, {}, None)
    with pytest.raises(ValueError, match="never a tiny fixture"):
        ADMISSION.validate_durable_proofs(f"{task}.{arm}", result, contract, {}, {}, {}, None)
