from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPRESENTATION_ENV = {
    "clasp_tsp_prompt_knn": "CLASP_TSP_PROMPT_KNN",
    "clasp_tsp_prompt_knn_distances": "CLASP_TSP_PROMPT_KNN_DISTANCES",
    "clasp_tsp_input_order": "CLASP_TSP_INPUT_ORDER",
    "clasp_tsp_relabel_order": "CLASP_TSP_RELABEL_ORDER",
    "clasp_tsp_route_canonicalize": "CLASP_TSP_ROUTE_CANONICALIZE",
    "clasp_tsp_route_only_output": "CLASP_TSP_ROUTE_ONLY_OUTPUT",
    "clasp_tsp_label_tta_schedule": "CLASP_TSP_LABEL_TTA_SCHEDULE",
    "clasp_mis_relabel_order": "CLASP_MIS_RELABEL_ORDER",
    "clasp_mis_jiang_response_output": "CLASP_MIS_JIANG_RESPONSE_OUTPUT",
    "clasp_mis_prompt_node_degrees": "CLASP_MIS_PROMPT_NODE_DEGREES",
    "clasp_mis_prompt_low_degree_k": "CLASP_MIS_PROMPT_LOW_DEGREE_K",
}

OP_PROMPT_ENV = {
    "op_strict_budget_hint": ("CLASP_OP_STRICT_BUDGET_HINT", "0", False),
    "op_prompt_budget_scale": ("CLASP_OP_PROMPT_BUDGET_SCALE", "1.0", 1.0),
    "op_prompt_budget_scale_min_size": (
        "CLASP_OP_PROMPT_BUDGET_SCALE_MIN_SIZE",
        "0",
        0,
    ),
    "op_prompt_budget_scale_max_size": (
        "CLASP_OP_PROMPT_BUDGET_SCALE_MAX_SIZE",
        "0",
        0,
    ),
    "op_prompt_budget_scale_size_ranges": (
        "CLASP_OP_PROMPT_BUDGET_SCALE_SIZE_RANGES",
        "",
        [],
    ),
}

OP_GENERATION_ENV = {
    "op_size_band_top_p_schedule": ("CLASP_OP_SIZE_BAND_TOP_P_SCHEDULE", "", []),
    "op_size_band_top_p_min_size": ("CLASP_OP_SIZE_BAND_TOP_P_MIN_SIZE", "0", 0),
    "op_size_band_top_p_max_size": ("CLASP_OP_SIZE_BAND_TOP_P_MAX_SIZE", "0", 0),
}


def _exported_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    pattern = re.compile(r'^export\s+([A-Z0-9_]+)=(?:"([^"]*)"|([^\s#]+))\s*$')
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.fullmatch(line)
        if match:
            values[match.group(1)] = (
                match.group(2) if match.group(2) is not None else match.group(3)
            )
    return values


def test_evaluation_launchers_pin_frozen_representation_contract() -> None:
    registry = json.loads(
        (ROOT / "experiments" / "sota_20260814" / "lane_contracts.json").read_text(encoding="utf-8")
    )
    prompting = registry["native_contract"]["prompting_exact"]
    generation = registry["native_contract"]["generation_exact"]
    for launcher_name in ("evaluate.slurm", "evaluate_a100.slurm"):
        launcher_path = ROOT / "scripts" / launcher_name
        exported = _exported_values(launcher_path)
        launcher_text = launcher_path.read_text(encoding="utf-8")
        assert 'EVALUATION_REPRESENTATION_PROFILE:-native' in launcher_text
        assert '"jiang_f638_sft"' in launcher_text
        assert "export CLASP_MIS_JIANG_RESPONSE_OUTPUT=0" in launcher_text
        static_representation_env = {
            field_name: env_name
            for field_name, env_name in REPRESENTATION_ENV.items()
            if env_name
            not in {
                "CLASP_TSP_PROMPT_KNN_DISTANCES",
                "CLASP_TSP_ROUTE_ONLY_OUTPUT",
                "CLASP_MIS_JIANG_RESPONSE_OUTPUT",
            }
        }
        assert {
            env_name: exported.get(env_name)
            for env_name in static_representation_env.values()
        } == {
            env_name: str(prompting[field_name])
            for field_name, env_name in static_representation_env.items()
        }
        assert "export CLASP_MIS_JIANG_RESPONSE_OUTPUT=1" in launcher_text
        assert 'export CLASP_TSP_PROMPT_KNN_DISTANCES=""' in launcher_text
        assert 'export CLASP_TSP_ROUTE_ONLY_OUTPUT=""' in launcher_text
        assert "export CLASP_TSP_PROMPT_KNN_DISTANCES=0" in launcher_text
        assert "export CLASP_TSP_ROUTE_ONLY_OUTPUT=0" in launcher_text
        assert 'MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-0}"' in launcher_text
        for field_name, (
            env_name,
            expected_env_value,
            expected_contract_value,
        ) in OP_PROMPT_ENV.items():
            assert prompting[field_name] == expected_contract_value
            assert exported.get(env_name) == expected_env_value
        for field_name, (
            env_name,
            expected_env_value,
            expected_contract_value,
        ) in OP_GENERATION_ENV.items():
            assert generation[field_name] == expected_contract_value
            assert exported.get(env_name) == expected_env_value


def test_generic_confirmation_launcher_requires_one_explicit_specialist_task() -> None:
    launcher = (ROOT / "scripts" / "evaluate.slurm").read_text(encoding="utf-8")
    assert 'TASKS_WAS_SET="${TASKS+x}"' in launcher
    for variable in ("DATA_ROOT", "CHECKPOINT_PATH", "INFERENCE_ROOT", "TASKS"):
        assert f'{variable}_EXPLICIT_VALUE="${{{variable}-}}"' in launcher
        assert (
            f'[[ "${{{variable}_WAS_SET}}" == x && '
            f'-n "${{{variable}_EXPLICIT_VALUE}}" ]]' in launcher
        )
    assert launcher.count("tsp|cvrp|op|mis|jssp)") == 1
    assert "confirmation TASKS must be exactly one of tsp,cvrp,op,mis,jssp" in launcher
    assert 'CAMPAIGN_MANIFEST="${CAMPAIGN_MANIFEST:-}"' in launcher
    assert '--campaign_manifest "${CAMPAIGN_MANIFEST}"' in launcher
    assert "  --resume \\\n" in launcher


def test_generic_confirmation_launcher_rejects_explicit_empty_required_values(
    tmp_path: Path,
) -> None:
    launcher = ROOT / "scripts" / "evaluate.slurm"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for command_name in ("module", "conda"):
        command = fake_bin / command_name
        command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        command.chmod(0o755)
    required = {
        "DATA_ROOT": "/nonexistent/confirmation-data",
        "CHECKPOINT_PATH": "/nonexistent/checkpoint-400",
        "INFERENCE_ROOT": "/nonexistent/confirmation-output",
        "TASKS": "tsp",
    }
    expected_messages = {
        "DATA_ROOT": "confirmation requires an explicit DATA_ROOT",
        "CHECKPOINT_PATH": "confirmation requires an explicit CHECKPOINT_PATH",
        "INFERENCE_ROOT": "confirmation requires an explicit INFERENCE_ROOT",
        "TASKS": "confirmation requires an explicit single specialist TASKS value",
    }
    for empty_variable, expected_message in expected_messages.items():
        environment = {
            **os.environ,
            **required,
            empty_variable: "",
            "CLASP_PROJECT_ROOT": str(ROOT),
            "EVALUATION_PURPOSE": "confirmation",
            "BENCHMARK_MANIFEST": "/nonexistent/benchmark.json",
            "CHECKPOINT_SELECTION_RECEIPT": "/nonexistent/selection.json",
            "CAMPAIGN_MANIFEST": "/nonexistent/campaign.json",
            "MAX_EVAL_SAMPLES_PER_TASK": "0",
            "EVAL_START_INDEX": "0",
            "PREFER_SUBSET_FILES": "false",
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        }
        completed = subprocess.run(
            ["bash", str(launcher)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 2
        assert expected_message in completed.stderr
