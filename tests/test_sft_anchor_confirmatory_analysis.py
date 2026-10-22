"""Prospective formal-analysis guards, wholly synthetic and never publishable.

Inputs use the frozen strict scorer and two supplied development input rows,
solely to exercise the schema. All twenty paired outcomes, scheduler responses,
checkpoint trees and custody records below are manufactured test fixtures.
No confirmation cohort, candidate selection, training run or native action is
created. Production remains disabled pending independent formal collection.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

from test_sft_anchor_admission_v3_collection import _durable_slot, encoded, make_contract, scorer

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"
SPEC = importlib.util.spec_from_file_location("prospective_confirmation", EXP / "confirmatory_analysis.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class MemoryEvidence:
    def __init__(self):
        self.files = {}

    def put(self, path, value):
        raw = value if isinstance(value, bytes) else encoded(value)
        self.files[path] = raw
        return {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}

    def __call__(self, binding):
        raw = self.files[binding["path"]]
        assert hashlib.sha256(raw).hexdigest() == binding["sha256"]
        assert len(raw) == binding["size_bytes"]
        return raw


def seal(value, key):
    value[key] = analysis.canonical({k: v for k, v in value.items() if k != key})
    return value


def synthetic_round(*, duplicate_seed=False):
    """Exact frozen scorer+durable admission, fixture-only two-row cohorts."""
    reader = MemoryEvidence()
    roster = [{"draw_id": f"draw{i:02}", "train_seed": 100 + i} for i in range(20)]
    if duplicate_seed:
        roster[-1]["train_seed"] = roster[0]["train_seed"]
    entry_raw = (ROOT / analysis.ENTRYPOINT).read_bytes()
    score_raw = (ROOT / analysis.SCORER).read_bytes()
    entries = [{"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
               for path, raw in ((analysis.ENTRYPOINT, entry_raw), (analysis.SCORER, score_raw))]
    source_binding = reader.put("source/source_manifest.json", {"synthetic_fixture": True, "files": entries})
    scorer_binding = reader.put("source/scorer.py", score_raw)
    verifier_binding = reader.put("source/admission.py", (EXP / "collect_campaign_admission_v3.py").read_bytes())
    seed_sampling = {"distribution": "uniform_fixed_uint32_complement_with_replacement", "independent_draws": True,
        "lower_inclusive": 0, "upper_exclusive": 2 ** 32, "excluded_previously_used_seeds": [43]}
    sampler_binding = reader.put("registration/sampler.json", {"synthetic_fixture": True,
        "seed_sampling": seed_sampling, "seed_roster": roster})
    selection = {"synthetic_fixture": True, "selected_arm": "rloo_cardinal_beta005",
        "all_development_slots_admitted": 12, "confirmation_outcomes_read": False,
        "superiority_established": False, "confirmation_policy_order": "must_be_established_by_future_round_registration",
        "recorded_utc": "2030-01-01T00:00:00+00:00"}
    selection_binding = reader.put("registration/synthetic_selection.json", selection)
    plan_binding = reader.put("registration/plan.json", (EXP / "planning/selection_and_confirmation_v2.json").read_bytes())
    history_binding = reader.put("registration/history.json", {"synthetic_fixture": True,
        "development_used_RL_seeds": [43], "prior_formal_rounds": []})
    contracts, bindings, cohorts = {}, {}, {}
    for draw in roster:
        bindings[draw["draw_id"]] = {}
        for task in analysis.TASKS:
            original = make_contract(task)
            contract = scorer.make_contract(task, original["initializer"], original["training_rows"],
                original["validation_rows"][:2], train_seed=draw["train_seed"],
                eval_seed=original["eval_seed"], scope="production", cohort_provenance=original["cohort_provenance"])
            contracts[draw["draw_id"], task] = contract
            cohorts[task] = contract["validation_cohort_sha256"]
            bindings[draw["draw_id"]][task] = reader.put(f"contracts/{draw['draw_id']}/{task}.json", contract)
    freshness_binding = reader.put("registration/synthetic_freshness.json", {"synthetic_fixture": True,
        "full_original_SFT500K_graph_exclusion_certified": True,
        "all_prior_tuning_training_confirmation_graphs_excluded": True,
        "confirmation_model_outcomes_read": False, "references_independently_strict_verified_before_policy_calls": True,
        "validation_cohort_sha256": cohorts})
    contract = seal({"protocol": analysis.CONTRACT_PROTOCOL, "schema_version": 1, "synthetic_fixture": True,
        "frozen_before_confirmation_policy_calls": True, "primary_decoder": "k8",
        "candidate_arm": "rloo_cardinal_beta005", "task_weights": {"op": .5, "cvrp": .5},
        "n_seed_draws": 20, "round_number": 1, "alpha_fraction": [1, 40],
        "seed_sampling": seed_sampling, "seed_roster": roster,
        "scorer_binding": scorer_binding, "durable_verifier_binding": verifier_binding,
        "plan_binding": plan_binding, "selection_binding": selection_binding, "seed_sampling_binding": sampler_binding,
        "round_history_binding": history_binding, "source_manifest_binding": source_binding,
        "freshness_binding": freshness_binding, "fixture_validation_count": 2,
        "original_sft_initializers": analysis.ORIGINAL_SFT_INITIALIZERS,
        "task_contract_bindings": bindings, "native_source_root": "/synthetic/formal/source",
        "native_control_roots": {d["draw_id"]: "/synthetic/formal/control/" + d["draw_id"] for d in roster}}, "contract_sha256")
    commitments = {}
    for name, second in (("selection", 1), ("seed_roster", 2), ("source", 3), ("round_registration", 4), ("first_policy_call", 5)):
        row = {"synthetic_fixture": True, "independent_custody_receipt": True,
               "observed_utc": f"2030-01-01T00:00:0{second}+00:00"}
        if name in ("selection", "seed_roster", "source"):
            row["file_sha256"] = {"selection": selection_binding, "seed_roster": sampler_binding,
                                  "source": source_binding}[name]["sha256"]
        elif name == "round_registration":
            row["registered_contract_sha256"] = contract["contract_sha256"]
        commitments[name] = reader.put("chronology/" + name + ".json", row)
    chronology_binding = reader.put("chronology/audit.json", {"protocol": analysis.PROTOCOL + "-prepolicy-chronology",
        "synthetic_fixture": True, "independent_read_only_collection": True,
        "selection_receipt_sha256": selection_binding["sha256"], "source_manifest_sha256": source_binding["sha256"],
        "seed_sampling_receipt_sha256": sampler_binding["sha256"], "commitments": commitments})
    slots = []
    for i, draw in enumerate(roster):
        for task in analysis.TASKS:
            for arm_index, arm in enumerate((contract["candidate_arm"], "foarl")):
                task_contract = contracts[draw["draw_id"], task]
                roots = {"native_source_root": contract["native_source_root"],
                         "native_control_root": contract["native_control_roots"][draw["draw_id"]]}
                payloads, tree = _durable_slot(task + "." + arm, task_contract, roots, entries[0])
                complete = json.loads(payloads["complete.json"])
                complete["actual_evaluation_responses"] = 18
                payloads["complete.json"] = encoded(complete)
                job = str(900000 + i * 4 + int(task == "cvrp") * 2 + arm_index)
                payloads["attempt.json"] = encoded({"protocol": analysis.PROTOCOL + "-attempt",
                    "synthetic_fixture": True, "state": "submitted", "job_id": job, "attempt_number": 1,
                    "draw_id": draw["draw_id"], "train_seed": draw["train_seed"], "task": task, "arm": arm,
                    "round_contract_sha256": contract["contract_sha256"], "native_control_root": roots["native_control_root"],
                    "requested_gpus": 4, "replacement_or_retry": False})
                payloads["scheduler.json"] = encoded({"synthetic_fixture": True, "independent_read_only_collection": True,
                    "argv": ["sacct", "--noheader", "--parsable2", "--jobs", job, "--format=JobIDRaw,State%40,ExitCode,AllocTRES"],
                    "returncode": 0, "stdout": "\n".join(f"{job}{suffix}|COMPLETED|0:0|gres/gpu=4" for suffix in ("", ".batch", ".extern", ".0"))})
                artifact_bindings = {name: reader.put(f"results/{draw['draw_id']}/{task}/{arm}/{name}", raw)
                                     for name, raw in payloads.items()}
                slots.append({"synthetic_fixture": True, "draw_id": draw["draw_id"], "task": task, "arm": arm,
                    "state": "complete", "scheduler": {"job_id": job, "state": "COMPLETED", "exit_code": "0:0"},
                    "artifacts": artifact_bindings, "native_terminal_artifact_proof": tree})
    bundle = seal({"protocol": analysis.PROTOCOL + "-collection", "synthetic_fixture": True,
        "contract_sha256": contract["contract_sha256"], "native_mutations": False, "source_proofs": entries,
        "chronology_binding": chronology_binding, "slots": slots,
        "round_close": {"state": "closed", "no_seed_replacement_or_outcome_based_stopping": True,
            "draw_states": [{"draw_id": d["draw_id"], "state": "terminal"} for d in roster]}}, "collection_sha256")
    return contract, bundle, reader


@pytest.fixture(scope="module")
def complete_round():
    return synthetic_round()


def run(packet):
    contract, bundle, reader = packet
    return analysis.analyze_round(contract, bundle, reader, synthetic_fixture=True)


@pytest.mark.parametrize("round_number,threshold", [(1, 15), (2, 16), (3, 16), (4, 17)])
def test_exact_fixed20_thresholds_and_alpha_spending(round_number, threshold):
    below = analysis.exact_global_test(threshold - 1, round_number=round_number)
    at = analysis.exact_global_test(threshold, round_number=round_number)
    assert below["n_registered_draws"] == at["n_registered_draws"] == 20
    assert below["reject_null_numerically"] is False and at["reject_null_numerically"] is True
    assert at["minimum_successes"] == threshold
    assert Fraction(*at["alpha_fraction"]) == Fraction(1, 20 * 2 ** round_number)
    assert Fraction(*at["p_value_fraction"]) == sum(Fraction(math.comb(20, j), 2 ** 20) for j in range(threshold, 21))
    assert below["theta_one_sided_lower_bound"] < .5 < at["theta_one_sided_lower_bound"]


def test_exact_confidence_endpoints_and_tail_monotonicity():
    assert analysis.exact_global_test(0)["theta_one_sided_lower_bound"] == 0
    assert analysis.exact_global_test(20)["theta_one_sided_lower_bound"] == pytest.approx(.025 ** (1 / 20), abs=2e-15)
    values = [analysis.exact_global_test(k) for k in range(21)]
    assert all(a["p_value"] > b["p_value"] and a["theta_one_sided_lower_bound"] < b["theta_one_sided_lower_bound"]
               for a, b in zip(values, values[1:]))
    assert math.fsum(float(analysis.round_alpha(i)) for i in range(1, 100)) == pytest.approx(.05)
    with pytest.raises(ValueError, match="twenty seed draws"):
        analysis.exact_global_test(15, n=19)


def row(identifier, gap=0.0, feasible=True):
    return {"instance_id": identifier, "semantic_id": "sem" + identifier, "reference_id": "ref" + identifier,
            "feasible": feasible, "signed_gap": gap, "clipped_gap": max(0, gap),
            "u10": max(0, min(1, 1 - 10 * max(0, gap))) if feasible else 0}


def favorable_pairs():
    return {task: ([row("a", -.2), row("b", 0)], [row("a", -.1), row("b", .1)]) for task in analysis.TASKS}


def test_whole_seed_gate_preserves_negative_reference_improvement():
    result = analysis.seed_gate(favorable_pairs())
    assert result["D"] == 1 and result["macro_q_sigma_delta_pp"] > 0
    assert analysis.quality(row("a", -.2)) > analysis.quality(row("a", -.1)) > .5


@pytest.mark.parametrize("case", ["feasibility", "joint_empty", "gap_reverse", "all_gaps_tie", "quality_reverse", "task_missing", "paired_ids"])
def test_each_whole_gate_guard_is_required(case):
    pairs = favorable_pairs()
    if case == "feasibility": pairs["op"][0][1]["feasible"] = False
    if case == "joint_empty":
        for r in pairs["op"][0]: r["feasible"] = False
    if case == "gap_reverse": pairs["cvrp"][0][0]["signed_gap"] = .4
    if case == "all_gaps_tie": pairs = {task: ([row("a", 0)], [row("a", 0)]) for task in analysis.TASKS}
    if case == "quality_reverse": pairs = {task: ([row("a", .1)], [row("a", -.1)]) for task in analysis.TASKS}
    if case == "task_missing": del pairs["op"]
    if case == "paired_ids": pairs["op"][0][0]["reference_id"] = "wrong"
    if case in ("task_missing", "paired_ids"):
        with pytest.raises(ValueError): analysis.seed_gate(pairs)
    else:
        assert analysis.seed_gate(pairs)["D"] == 0


def test_all80_slots_full_raw_replay_stays_synthetic_and_fixed20(complete_round):
    report = run(complete_round)
    assert len(report["seed_draws"]) == report["global_test"]["n_registered_draws"] == 20
    assert report["raw_candidates_replayed"] == 80 * 18
    assert not any(d["admission_errors"] for d in report["seed_draws"])
    assert report["formal_claim_eligible"] is False
    assert report["conditional_majority_seed_gate_probability_established"] is False
    assert report["expected_mean_improvement_established"] is False
    assert report["population_feasibility_noninferiority_established"] is False


def test_allowed_duplicate_seed_values_have_distinct_draw_ids():
    contract, bundle, reader = synthetic_round(duplicate_seed=True)
    result = analysis.validate_registration(contract, reader, synthetic_fixture=True)
    assert len(result[3]) == 40
    assert contract["seed_roster"][-1]["train_seed"] == contract["seed_roster"][0]["train_seed"]


@pytest.mark.parametrize("case", ["missing", "failed", "unadmitted", "source", "raw", "rank", "prompt", "allocation", "step_failure", "attempt"])
def test_any_slot_or_admission_failure_retains_zero_and_all20(complete_round, case):
    contract, bundle, reader = copy.deepcopy(complete_round)
    first = bundle["slots"][0]
    if case == "missing":
        bundle["slots"].pop(0)
        bundle["round_close"]["draw_states"][0].update(state="declared_permanent_failure", reason="synthetic missing slot")
    elif case in ("failed", "unadmitted"): first["state"] = case
    elif case == "source": bundle["source_proofs"][0]["sha256"] = "0" * 64
    else:
        key = {"raw": "result.json", "rank": "training_complete.json", "prompt": "actual_prompt_windows.jsonl",
               "allocation": "scheduler.json", "step_failure": "scheduler.json", "attempt": "attempt.json"}[case]
        value = json.loads(reader(first["artifacts"][key])) if key.endswith(".json") else None
        if case == "raw": value["raw_completions"][0]["text"] = "raw corruption"
        if case == "rank": value["distributed"]["rank_closure"].pop()
        if case == "allocation": value["stdout"] = value["stdout"].replace("gres/gpu=4", "gres/gpu=1")
        if case == "step_failure": value["stdout"] = value["stdout"].replace(".0|COMPLETED|0:0", ".0|FAILED|1:0")
        if case == "attempt": value["attempt_number"] = 2
        raw = encoded(value) if value is not None else b'{"step":999}\n'
        first["artifacts"][key] = reader.put(first["artifacts"][key]["path"], raw)
    seal(bundle, "collection_sha256")
    report = run((contract, bundle, reader))
    assert len(report["seed_draws"]) == 20 and report["global_test"]["n_registered_draws"] == 20
    assert report["seed_draws"][0]["D"] == 0 and report["seed_draws"][0]["admission_errors"]
    assert report["seed_replacement_performed"] is False


def test_open_round_has_no_interim_claim_even_with_all_complete(complete_round):
    contract, bundle, reader = copy.deepcopy(complete_round)
    bundle["round_close"]["state"] = "open"
    seal(bundle, "collection_sha256")
    report = run((contract, bundle, reader))
    assert report["formal_claim_eligible"] is False


@pytest.mark.parametrize("case", ["duplicate_draw", "foreign_slot", "reset_alpha", "used_seed", "full_uint32", "initializer", "dropped_history", "closed19", "self_timing"])
def test_registration_and_closure_cannot_drop_or_relabel_draws(complete_round, case):
    contract, bundle, reader = copy.deepcopy(complete_round)
    if case == "duplicate_draw": contract["seed_roster"][-1]["draw_id"] = contract["seed_roster"][0]["draw_id"]
    if case == "foreign_slot": bundle["slots"][0]["draw_id"] = "replacement"
    if case == "reset_alpha": contract["alpha_fraction"] = [1, 20]
    if case == "used_seed": contract["seed_roster"][0]["train_seed"] = 43
    if case == "full_uint32": contract["seed_sampling"]["distribution"] = "uniform_uint32_with_replacement"
    if case == "initializer": contract["original_sft_initializers"]["op"] = "0" * 64
    if case == "dropped_history": contract["round_number"] = 2; contract["alpha_fraction"] = [1, 80]
    if case == "closed19": bundle["round_close"]["draw_states"].pop()
    if case == "self_timing":
        audit = json.loads(reader(bundle["chronology_binding"]))
        audit["independent_read_only_collection"] = False
        bundle["chronology_binding"] = reader.put(bundle["chronology_binding"]["path"], audit)
    seal(contract, "contract_sha256"); bundle["contract_sha256"] = contract["contract_sha256"]
    seal(bundle, "collection_sha256")
    with pytest.raises(ValueError): run((contract, bundle, reader))


def test_production_remains_closed_to_all_synthetic_inputs(complete_round):
    contract, bundle, reader = complete_round
    assert analysis.PRODUCTION_ADMISSION_READY is False
    with pytest.raises(ValueError, match="dedicated formal-round collector"):
        analysis.analyze_round(contract, bundle, reader, expected_contract_sha256=contract["contract_sha256"])


@pytest.mark.parametrize("case,successes", [("closed", 15), ("one_failed", 14), ("permanent_failure", 14), ("open", 15)])
def test_failure_retention_and_no_interim_claim_at_actual_binomial_boundary(monkeypatch, case, successes):
    """Pure seed-gate oracle; independent raw admission is covered above."""
    reader = MemoryEvidence()
    selection = reader.put("selection.json", {"synthetic_fixture": True})
    draws = [{"draw_id": f"d{i}", "train_seed": i} for i in range(20)]
    contract = {"candidate_arm": "rloo_cardinal_beta005", "round_number": 1,
                "seed_roster": draws, "contract_sha256": "synthetic", "selection_binding": selection}
    slots = [{"draw_id": d["draw_id"], "task": task, "arm": arm, "state": "complete"}
             for d in draws for task in analysis.TASKS for arm in (contract["candidate_arm"], "foarl")]
    close = {"state": "open" if case == "open" else "closed", "no_seed_replacement_or_outcome_based_stopping": True,
             "draw_states": [{"draw_id": d["draw_id"], "state": "terminal"} for d in draws]}
    if case == "one_failed": slots[0]["state"] = "failed"
    if case == "permanent_failure": close["draw_states"][0].update(state="declared_permanent_failure", reason="synthetic technical failure")
    bundle = seal({"protocol": analysis.PROTOCOL + "-collection", "synthetic_fixture": True,
        "contract_sha256": "synthetic", "native_mutations": False, "source_proofs": [], "slots": slots,
        "round_close": close, "chronology_binding": {}}, "collection_sha256")
    monkeypatch.setattr(analysis, "validate_registration", lambda *a, **k: (None, None, {},
        {(d["draw_id"], task): {} for d in draws for task in analysis.TASKS}, 1))
    monkeypatch.setattr(analysis, "validate_chronology", lambda *a, **k: None)
    def admitted(slot, **kwargs):
        if slot["state"] != "complete": raise ValueError("synthetic failed slot")
        gap = -.1 if kwargs["arm"] == contract["candidate_arm"] and kwargs["draw"]["train_seed"] < 15 else 0.0
        return {"selected": {mode: [row("fixed", gap)] for mode in ("greedy", "k8")}}
    monkeypatch.setattr(analysis, "admitted_slot", admitted)
    report = analysis.analyze_round(contract, bundle, reader, synthetic_fixture=True)
    assert report["global_test"]["successes"] == successes
    assert report["global_test"]["n_registered_draws"] == len(report["seed_draws"]) == 20
    assert report["global_test"]["reject_null_numerically"] is (successes >= 15)
    assert report["formal_claim_eligible"] is False and report["conditional_majority_seed_gate_probability_established"] is False


def test_actual_tiny_unscored_groups_match_only_exact_four_field_projection():
    """Actual integration evidence is still ineligible for formal policy claims."""
    raw = (EXP / "collect_campaign_admission_v3.py").read_bytes()
    verifier = analysis.module_from_bytes(raw, analysis.DURABLE_VERIFIER_SHA256, "_formal_actual_projection_check")
    folder = EXP / "validation/v2_peft_foarl_cuda_ddp2_v1/op/foarl"
    result = json.loads((folder / "result.json").read_bytes())
    groups = [json.loads(line) for line in (folder / "evaluation_groups.jsonl").read_bytes().splitlines()]
    generated = [r for group in groups for r in group["raw_completions"]]
    assert len(generated) == len(result["raw_completions"]) == 36
    assert generated != result["raw_completions"]
    assert generated == [verifier.raw_generation_projection(r) for r in result["raw_completions"]]
    assert result["scope"] == "system_smoke"
