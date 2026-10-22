"""Pure synthetic matrices only; no fresh confirmation/model outcomes are read."""
import builtins
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments/rloo_one_shot_20261016/analyze_confirmation_op_v3.py"


@pytest.fixture
def C():
    raw = PATH.read_bytes();module = types.ModuleType("confirmation_statistics_fixture")
    module.__file__ = str(PATH);module.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    exec(compile(raw, str(PATH), "exec", dont_inherit=True), module.__dict__)
    return module


def matrices(C, branch="pure"):
    cohort = [{"instance_id": "synthetic-" + str(i), "semantic_id": "graph-" + str(i), "stratum": i % 3,
               "total_prize": 20 + i % 5, "reference_objective": 5} for i in range(256)]
    def vector(prize):
        return [{"instance_id": row["instance_id"], "reference_id": row["semantic_id"], "feasible": True, "objective": prize}
                for row in cohort]
    methods, candidate, _ = C.recipe(branch)
    trained = [{"seed": seed, "methods": {arm: vector(8 if arm == candidate else 6) for arm in methods}} for seed in C.SEEDS]
    return {"branch": branch, "cohort": cohort, "trained_by_seed": trained,
            "sft_once": {"seed": 101, "arm": "sft", "records": vector(6)}}


def components(C, branch="pure", effect=Fraction(1, 10)):
    _, _, controls = C.recipe(branch)
    effects = {arm: {key: [effect] * 5 for key in ("utility", "joint_gap", "feasibility_NI")} for arm in controls}
    flags = {arm: [True] * 5 for arm in controls}
    return effects, flags


def test_default_is_stdlib_zero_call_and_no_input_reads(C, monkeypatch, capsys):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("numpy", "torch", "transformers", "clasp"):
            pytest.fail("Default imported a numerical/model dependency")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded);monkeypatch.setattr(sys, "argv", [str(PATH)])
    C.main();value = json.loads(capsys.readouterr().out)
    assert value["filesystem_quality_reads"] == value["model_calls"] == 0
    assert value["metadata_admission_ready"] is True


def test_actual_registration_path_remains_closed_without_reading_binding(C, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        C.analyze_confirmation_op({"path": "/does/not/exist"})


@pytest.mark.parametrize("effects,k,p", [([1] * 5, 5, Fraction(1, 32)), ([1] * 4 + [0], 4, Fraction(6, 32)),
                                        ([1] * 4 + [None], 4, Fraction(6, 32)), ([None] * 5, 0, Fraction(1))])
def test_exact_sign_fixed_denominator_ties_and_failures(C, effects, k, p):
    value = C.exact_sign_component(effects)
    assert value["n_seeds"] == 5 and value["favorable_seeds"] == k
    assert value["p_exact"] == {"numerator": str(p.numerator), "denominator": str(p.denominator)}


@pytest.mark.parametrize("effects", [[1] * 4, [1] * 6, [True] * 5, [float("nan")] * 5, [float("inf")] * 5])
def test_sign_cannot_drop_or_replace_or_cast_seed_effects(C, effects):
    with pytest.raises(ValueError):
        C.exact_sign_component(effects)


def test_all_controls_endpoints_required_global_IUT(C):
    effects, flags = components(C)
    value = C.sign_iut("pure", effects, flags)
    assert value["statistical_gate_passed"] is True and value["global_p_value"] == .03125
    effects["sft"]["joint_gap"][-1] = 0
    value = C.sign_iut("pure", effects, flags)
    assert value["statistical_gate_passed"] is False and value["global_p_value"] == .1875
    # No favorable component is promoted after the required conjunction fails.
    assert value["primary_rule"].startswith("all-controls/all-components")


def test_empirical_no_loss_guard_is_stronger_than_NI_sign_component(C):
    effects, flags = components(C)
    effects["sft"]["feasibility_NI"] = [Fraction(1, 50) - Fraction(1, 256)] * 5
    flags["sft"][2] = False
    value = C.sign_iut("pure", effects, flags)
    assert value["global_p_value"] == .03125 and value["all_empirical_no_feasibility_loss"] is False
    assert value["statistical_gate_passed"] is False


@pytest.mark.parametrize("fault", ["control", "endpoint", "flag", "flag_type"])
def test_IUT_roster_and_safeguards_cannot_be_filtered(C, fault):
    effects, flags = components(C)
    if fault == "control": effects.pop("foarl_cost_changed")
    elif fault == "endpoint": effects["sft"].pop("utility")
    elif fault == "flag": flags["sft"].pop()
    else: flags["sft"][0] = 1
    with pytest.raises(ValueError): C.sign_iut("pure", effects, flags)


def test_exact_effect_arithmetic_retains_a_mathematical_zero(C):
    cohort = [{"instance_id": str(i), "semantic_id": str(i), "total_prize": total, "reference_objective": 1}
              for i, total in enumerate((3, 5, 15))]
    def scores(prizes):
        return [{"instance_id": str(i), "reference_id": str(i), "feasible": True, "objective": p} for i, p in enumerate(prizes)]
    left = C.scored_vector(scores((2, 2, 2)), cohort);right = C.scored_vector(scores((1, 1, 10)), cohort)
    pair = C.paired_effects(left, right, cohort)
    assert pair["mean_utility_delta"] == 0  # 1/6 +1/10 -4/15 ==0 exactly
    assert C.exact_sign_component([pair["mean_utility_delta"]] * 5)["favorable_seeds"] == 0


def test_Uphys_improvement_continues_past_heuristic_and_U10_saturation(C):
    args = matrices(C);cohort = args["cohort"]
    candidate = C.scored_vector(args["trained_by_seed"][0]["methods"]["rloo_cost"], cohort)
    sft = C.scored_vector(args["sft_once"]["records"], cohort)
    assert all(a["U10"] == b["U10"] == 1 and a["Uphys"] > b["Uphys"] for a, b in zip(candidate, sft))
    assert all(a["gap"] < 0 for a in candidate)


@pytest.mark.parametrize("change", [lambda r: r.update(feasible=1), lambda r: r.update(objective=float("nan")),
                                    lambda r: r.update(objective=21), lambda r: r.update(feasible=False, objective=1),
                                    lambda r: r.update(reference_id="wrong"), lambda r: r.update(u10=.5)])
def test_score_type_physical_bound_and_derived_value_denials(C, change):
    args = matrices(C);records = args["sft_once"]["records"];change(records[0])
    with pytest.raises(ValueError): C.scored_vector(records, args["cohort"])


@pytest.mark.parametrize("fault", ["seed73", "reorder", "missing_method", "extra_GRPO_teacher", "repeat_SFT", "SFT_seed", "cohort_count", "strata", "duplicate_id"])
def test_complete_fixed_matrix_admission_before_bootstrap(C, monkeypatch, fault):
    args = matrices(C)
    if fault == "seed73": args["trained_by_seed"][0]["seed"] = 73
    elif fault == "reorder": args["trained_by_seed"].reverse()
    elif fault == "missing_method": args["trained_by_seed"][3]["methods"].pop("foarl_released")
    elif fault == "extra_GRPO_teacher": args["trained_by_seed"][0]["methods"]["grpo_teacher"] = []
    elif fault == "repeat_SFT": args["sft_once"] = [args["sft_once"]] * 5
    elif fault == "SFT_seed": args["sft_once"]["seed"] = 73
    elif fault == "cohort_count": args["cohort"].pop()
    elif fault == "strata": args["cohort"][0]["stratum"] = 1
    else: args["cohort"][-1]["instance_id"] = args["cohort"][0]["instance_id"]
    monkeypatch.setattr(C, "_bootstrap", lambda *a, **k: pytest.fail("Incomplete matrix reached bootstrap"))
    with pytest.raises(ValueError): C.compute_statistics(**args)


def tiny_pairs(C, sparse=False):
    rows = []
    for seed in range(5):
        rows.append({"utility_delta": [Fraction(i, 10) for i in range(6)], "feasibility_delta": [Fraction(0)] * 6,
                     "gap_delta": [-Fraction(i + 1, 10) if not sparse or i == 0 else Fraction(0) for i in range(6)],
                     "joint_mask": [not sparse or i == 0 for i in range(6)]})
    return {"control": rows}


def test_bootstrap_matches_independent_shared_graph_reference_and_index_hash(C):
    pairs = tiny_pairs(C);groups = [[0, 3], [1, 4], [2, 5]];reps = 300
    value = C._bootstrap(pairs, groups, replicates=reps, chunk_size=17)
    rng = random.Random(C.BOOTSTRAP_SEED);samples = [];hasher = hashlib.sha256()
    for _ in range(reps):
        ids = [group[rng.randrange(len(group))] for group in groups for _ in group]
        hasher.update(struct.pack("<" + "I" * len(ids), *ids))
        samples.append(sum(ids) / 60.)
    expected = [C._percentile(samples, 1), C._percentile(samples, 39)]
    assert value["resample_index_sha256"] == hasher.hexdigest()
    assert value["comparisons"]["control"]["mean_Uphys_delta_ci95"] == pytest.approx(expected)
    # Identical seed vectors do not shrink graph uncertainty by sqrt(5).
    assert value["graph_sampling_units"] == 6 and value["trained_seed_count"] == 5
    assert value["all_seeds_share_indices"] and value["all_methods_share_indices"] and not value["seed_resampling"]
    other = C._bootstrap(pairs, groups, replicates=reps, chunk_size=1)
    assert other["comparisons"] == value["comparisons"] and other["resample_index_sha256"] == value["resample_index_sha256"]


def test_undefined_joint_replicates_never_become_unconditional_CI(C):
    value = C._bootstrap(tiny_pairs(C, sparse=True), [[0, 3], [1, 4], [2, 5]], replicates=400)
    row = value["comparisons"]["control"]
    assert 0 < row["undefined_joint_gap_replicates"] < 400
    assert row["defined_joint_gap_replicates"] + row["undefined_joint_gap_replicates"] == 400
    assert row["mean_joint_gap_delta_ci95"] is None
    assert row["conditional_defined_only_joint_gap_quantiles"] == pytest.approx([-.1, -.1])


def test_one_seed_without_joint_support_makes_every_replicate_undefined(C):
    pairs = tiny_pairs(C);pairs["control"][2]["joint_mask"] = [False] * 6
    pairs["control"][2]["gap_delta"] = [Fraction(0)] * 6
    value = C._bootstrap(pairs, [[0, 3], [1, 4], [2, 5]], replicates=100)
    row = value["comparisons"]["control"]
    assert row["undefined_joint_gap_replicates"] == 100 and row["defined_joint_gap_replicates"] == 0
    assert row["mean_joint_gap_delta_ci95"] is row["conditional_defined_only_joint_gap_quantiles"] is None


def test_complete_synthetic_pure_50k_uses_onceSFT_and_fixed5_seed_rule(C):
    value = C.compute_statistics(**matrices(C))
    assert value["status"] == "complete_statistics_from_unadmitted_matrices"
    assert value["global_IUT"]["statistical_gate_passed"] is True
    assert value["metadata_admitted"] is value["superiority_established"] is value["publication_eligible"] is False
    assert value["supplied_trained_score_rows"] == 6400 and value["supplied_once_SFT_rows"] == 256
    assert value["SFT_supplied_vectors"] == 1 and value["SFT_reused_in_seed_pairs"] == 5
    assert len(value["controls"]) == 5 and value["graph_bootstrap"]["replicates"] == 50000
    assert value["graph_bootstrap"]["stratum_counts"] == [86, 85, 85]
    assert all(x["undefined_joint_gap_replicates"] == 0 for x in value["graph_bootstrap"]["comparisons"].values())
    assert all(seed["sft"]["SFT_vector_reused"] for seed in value["summaries"].values())
    summary = value["summaries"]["101"]["rloo_cost"]
    assert [row["N"] for row in summary["strata"]] == [86, 85, 85]
    assert summary["own_feasible_signed_gap"]["maximum"] < 0
    assert summary["own_feasible_recomputed_prize"]["mean"] == 8.
    assert value["comparisons"]["sft"]["per_seed"][0]["joint_signed_gap_delta_distribution"]["maximum"] < 0


def test_teacher_branch_exact_roster_and_controls_without_optional_tier(C):
    methods, candidate, controls = C.recipe("teacher")
    assert len(methods) == 9 and len(controls) == 9 and candidate == "rloo_teacher" and "sft" in controls
    assert "grpo_teacher" not in methods
    effects, flags = components(C, "teacher");assert C.sign_iut("teacher", effects, flags)["statistical_gate_passed"]


def test_teacher_full_matrix_path_retains_every_required_method_and_pair(C, monkeypatch):
    # Arithmetic/admission path on all11520 supplied synthetic scores, with
    # bootstrap implementation separately exercised at50k in the pure case.
    seen = []
    def inspect(pairs, groups, **kwargs):
        seen.append((pairs, groups, kwargs))
        assert len(pairs) == 9 and all(len(rows) == 5 for rows in pairs.values())
        return {"diagnostic_stub": True}
    monkeypatch.setattr(C, "_bootstrap", inspect)
    value = C.compute_statistics(**matrices(C, "teacher"))
    assert value["supplied_trained_score_rows"] == 11520 and len(value["controls"]) == 9 and len(seen) == 1
    assert value["global_IUT"]["statistical_gate_passed"] is True
    assert value["metadata_admitted"] is value["publication_eligible"] is False


def test_invalid_routes_have_zero_unconditional_utility_not_imputed_gap(C):
    args = matrices(C);records = args["sft_once"]["records"]
    records[0].update(feasible=False, objective=None)
    value = C.scored_vector(records, args["cohort"])[0]
    assert value["Uphys"] == value["U10"] == 0 and value["gap"] is value["P"] is None


def test_negative_or_zero_reference_scale_rejected_before_scores(C):
    args = matrices(C);args["cohort"][0]["reference_objective"] = 0
    with pytest.raises(ValueError, match="Positive feasible reference"):
        C.validate_cohort(args["cohort"])


def test_rational_small_positive_sign_does_not_underflow_to_false_tie(C):
    value = C.exact_sign_component([Fraction(1, 10**400)] * 5)
    assert value["favorable_seeds"] == 5


def test_nonfinite_derived_gap_rejected_not_clipped(C):
    cohort = [{"instance_id": "a", "semantic_id": "b", "total_prize": 1e308, "reference_objective": 1e-308}]
    records = [{"instance_id": "a", "reference_id": "b", "feasible": True, "objective": 1e308}]
    with pytest.raises(ValueError, match="range"):
        C.scored_vector(records, cohort)


def test_source_constants_match_frozen_plan_input_bindings_without_opening_graphs(C):
    plan = PATH.parent / "prospective_confirmation_v2_plan.json"
    assert hashlib.sha256(plan.read_bytes()).hexdigest() == C.PLAN_SHA256
    value = json.loads(plan.read_bytes())
    assert value["input_bindings"]["confirmation_256"]["sha256"] == C.COHORT_SHA256
    assert value["fixed_training"]["planned_training_seeds"] == list(C.SEEDS)
    assert value["statistics"]["paired_graph_bootstrap"]["replicates"] == C.BOOTSTRAP_REPLICATES
    assert value["statistics"]["paired_graph_bootstrap"]["seed"] == C.BOOTSTRAP_SEED
