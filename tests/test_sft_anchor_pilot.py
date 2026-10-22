"""Independent numerical and experiment-admission controls for the new pilot."""
import copy
import json

import pytest

from clasp.baselines import sft_anchor_pilot as pilot

OP = [[[0, 0], [3, 4], [6, 4]], [0, 7, 5], 8]
CVRP = [[[0, 0], [3, 0], [0, 4]], [0, 1, 1], 2]


def row(task, identifier, instance, output):
    return {"id": identifier, "task_type": task, "instruction": "Solve " + task,
            "input": json.dumps(instance), "instance_json": json.dumps(instance), "output": output}


def contract(task="op", **kwargs):
    if task == "op":
        train = row(task, "train", [[[0, 0], [1, 0], [2, 0]], [0, 4, 6], 3], "Route: [0,1,2]")
        validation = row(task, "validation", OP, "Route: [0,1]")
    else:
        train = row(task, "train", [[[0, 0], [1, 0], [0, 1]], [0, 1, 1], 2], "Routes: [[0,1,2,0]]")
        validation = row(task, "validation", CVRP, "Routes: [[0,1,2,0]]")
    defaults = {"scope": "system_smoke", "horizon": 2}
    return pilot.make_contract(task, {"id": "toy-sft", "sha256": "a" * 64}, [train],
                               [validation], **{**defaults, **kwargs})


def completions(c, text=None):
    task = c["task"]
    text = text or ("Route: [0,1,2]" if task == "op" else "Routes: [[0,1,2,0]]")
    return [{"instance_id": row["instance_id"], "completion_id": f"{row['instance_id']}/{mode}/{i}",
             "decode": mode, "sample_index": i, "text": text, "generated_tokens": 12}
            for row in c["validation_rows"] for mode, n in (("greedy", 1), ("k8", 8)) for i in range(n)]


def evaluate(c, arm="rloo_cardinal_beta005", texts=None):
    steps = 0 if arm == "sft" else c["training"]["horizon"]
    checkpoint = {"id": arm + "-terminal", "initializer_sha256": c["initializer_sha256"],
                  "reference_initializer_sha256": c["initializer_sha256"],
                  "train_seed": c["train_seed"], "step": steps}
    resources = {"training_optimizer_steps": steps, "training_responses": 36 * steps,
                 "training_generated_tokens": 111 * steps, "training_wall_seconds": 0.1 * steps,
                 "evaluation_wall_seconds": 0.2, "prompt_draw_sha256": c["training"]["prompt_draw_sha256"]}
    return pilot.evaluate_arm(c, arm, texts or completions(c), checkpoint, resources)


def test_known_op_open_path_prize_and_negative_gap_ignore_printed_claim():
    score = pilot.score_completion("op", "Route: [0,1,2], Objective: -999", OP, "Route: [0,1], Objective: 999")
    assert score["feasible"] and score["distance"] == 8 and score["objective"] == 12
    assert score["reference_objective"] == 7 and score["signed_gap"] == pytest.approx(-5 / 7)
    assert score["clipped_gap"] == 0 and score["u10"] == 1


def test_known_cvrp_geometry_coverage_and_quality_utility():
    score = pilot.score_completion("cvrp", "Routes: [[0,1,0],[0,2,0]], Objective: 0", CVRP,
                                   "Routes: [[0,1,2,0]], Objective: 999")
    assert score["feasible"] and score["objective"] == 14 and score["reference_objective"] == 12
    assert score["signed_gap"] == pytest.approx(1 / 6) and score["u10"] == 0
    improved = pilot.score_completion("cvrp", "[[0,1,2,0]]", CVRP, "[[0,1,0],[0,2,0]]")
    assert improved["signed_gap"] == pytest.approx(-1 / 7) and improved["u10"] == 1


@pytest.mark.parametrize("text", ["[0]", "[0,1,1]", "[0,1,0]", "[1,0]", "[0,-1]", "[0,3]",
                                     "[False,1]", "[0,1.0]", "['0',1]", "[[0,1]]", "[0,1", "[]"])
def test_op_malformed_or_infeasible_is_not_repaired(text):
    score = pilot.score_completion("op", text, OP, "[0,1]")
    assert not score["feasible"] and score["objective"] is None and score["u10"] == 0


@pytest.mark.parametrize("text", ["[[0,1,0]]", "[[0,1,2,1,0]]", "[[1,2,0]]", "[[0,1,0,2,0]]",
                                     "[[0,1,2,0],[0]]", "[[0,1,3,0]]", "[[0,True,2,0]]",
                                     "[[0,1.0,2,0]]", "[[0,'1',2,0]]", "[0,1,2,0]", "[]"])
def test_cvrp_malformed_route_constraints_are_not_repaired(text):
    score = pilot.score_completion("cvrp", text, CVRP, "[[0,1,2,0]]")
    assert not score["feasible"] and score["objective"] is None


def test_budget_and_capacity_use_raw_units():
    over_budget = [OP[0], OP[1], 8 - 2e-9]
    assert not pilot.score_completion("op", "[0,1,2]", over_budget, "[0,1]")["feasible"]
    within_tolerance = [OP[0], OP[1], 8 - 5e-10]
    assert pilot.score_completion("op", "[0,1,2]", within_tolerance, "[0,1]")["feasible"]
    capacity = [CVRP[0], CVRP[1], 1]
    assert not pilot.score_completion("cvrp", "[[0,1,2,0]]", capacity, "[[0,1,0],[0,2,0]]")["feasible"]
    # Preserve the established strict CVRP boundary predicate for unused vehicles.
    assert pilot.score_completion("cvrp", "[[0,1,2,0],[0,0]]", CVRP, "[[0,1,2,0]]")["feasible"]


def test_reference_is_positive_feasible_and_recomputed():
    for task, instance, reference in (("op", OP, "[0]"), ("op", OP, "[0,1,0]"),
                                      ("cvrp", CVRP, "[[0,1,0]]")):
        with pytest.raises(ValueError, match="Reference witness"):
            pilot.score_completion(task, "garbage", instance, reference)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -1])
def test_invalid_instance_metadata_fails_admission(bad):
    with pytest.raises(ValueError):
        pilot.numeric_instance("op", [OP[0], OP[1], bad])


def test_semantic_identity_normalizes_spelling_without_merging_large_integers():
    float_spelling = [[[0.0, -0.0], [3.0, 4.0], [6.0, 4.0]], [0.0, 7.0, 5.0], 8.0]
    assert pilot.semantic_id("op", OP) == pilot.semantic_id("op", json.dumps(float_spelling))
    first, second = copy.deepcopy(OP), copy.deepcopy(OP)
    first[0][1][0] = 9007199254740992
    second[0][1][0] = 9007199254740993
    assert pilot.semantic_id("op", first) != pilot.semantic_id("op", second)


def test_semantic_overlap_ignores_changed_ids_and_prompts():
    first = row("op", "train", OP, "[0,1]")
    second = {**first, "id": "renamed", "instruction": "Different prose"}
    with pytest.raises(ValueError, match="semantic overlap"):
        pilot.validate_cohorts([first], [second], task="op")
    with pytest.raises(ValueError, match="duplicate mathematical"):
        pilot.validate_cohorts([first, second], [contract()["validation_rows"][0]], task="op")


def test_supplied_previously_used_validation_cohort_is_excluded():
    c = contract()
    with pytest.raises(ValueError, match="previously used cohort"):
        pilot.validate_cohorts(c["training_rows"], c["validation_rows"], task="op",
                              forbidden_rows=c["validation_rows"])


def test_changed_limits_or_customer_numbers_do_not_evade_graph_overlap():
    train = row("op", "train", OP, "[0,1]")
    changed = [OP[0], OP[1], 9]
    val = row("op", "validation", changed, "[0,1]")
    with pytest.raises(ValueError, match="graph semantic overlap"):
        pilot.validate_cohorts([train], [val], task="op")
    permuted = [list(reversed(OP[0][1:])), list(reversed(OP[1][1:]))]
    instance = [[OP[0][0]] + permuted[0], [0] + permuted[1], 20]
    assert pilot.graph_semantic_id("op", OP) == pilot.graph_semantic_id("op", instance)


def test_production_fixes_budget_and_all_arms_share_sampler():
    c = contract(scope="production", horizon=100)
    assert pilot.validate_contract(c) == c and set(c["arms"]) == set(pilot.ARMS)
    assert len(c["training"]["prompt_draws"]) == 100
    assert all(len(draw) == 4 for draw in c["training"]["prompt_draws"])
    assert c["training"]["responses_per_step"] == 36
    assert c["training"]["fixed_loss_denominator"] == 3000
    assert c == contract(scope="production", horizon=100)
    with pytest.raises(ValueError, match="Production fixes"):
        contract(scope="production", horizon=4)
    with pytest.raises(ValueError, match="Production fixes"):
        contract(scope="production", horizon=100, max_completion_tokens=1)
    changed = copy.deepcopy(c)
    changed["training"]["prompt_draws"][0][0] = "different"
    changed["contract_sha256"] = pilot.digest({k: v for k, v in changed.items() if k != "contract_sha256"})
    with pytest.raises(ValueError, match="numerical recipe"):
        pilot.validate_contract(changed)


@pytest.mark.parametrize("task", pilot.TASKS)
def test_end_to_end_all_six_arms_preserve_raw_ids_metrics_and_resources(task):
    c = contract(task)
    outputs = [evaluate(c, arm) for arm in pilot.ARMS]
    for result in outputs:
        assert result["resources"]["evaluation_responses"] == 9
        assert result["resources"]["evaluation_generated_tokens"] == 108
        assert result["metrics"]["k8"]["N"] == result["metrics"]["k8"]["n"] == 1
        assert result["metrics"]["k8"]["feasibility_percent"] == 100
        assert len(result["raw_completions"]) == 9
        assert result["raw_completions"][0]["reference_id"]
        assert result["raw_completions"][0]["text"]
        assert result["selected"]["k8"][0]["selected_completion_id"].endswith("k8/0")
        assert not result["superiority_established"]
    compared = pilot.compare_results(outputs, bootstrap_reps=100)
    assert compared["groups"][0]["matched"] and not compared["groups"][0]["missing_arms"]
    assert len(compared["groups"][0]["comparisons"]) == 16
    assert not compared["superiority_established"]
    assert any("system smoke" in reason for reason in compared["reasons"])


def test_missing_duplicate_and_wrong_cohort_completions_rejected():
    c = contract()
    raw = completions(c)
    with pytest.raises(ValueError, match="Incomplete evaluation"):
        evaluate(c, texts=raw[:-1])
    with pytest.raises(ValueError, match="unique"):
        evaluate(c, texts=raw + [raw[0]])
    changed = copy.deepcopy(raw)
    changed[0]["instance_id"] = "unmatched"
    with pytest.raises(ValueError, match="cohort/decoder"):
        evaluate(c, texts=changed)


def test_k8_selects_raw_quality_and_runtime_failure_is_ineligible():
    c = contract()
    raw = completions(c, "[0,1]")
    raw[1]["text"] = "[0,1,2]"
    raw[1]["runtime_complete"] = False
    raw[2]["text"] = "[0,1,2]"
    result = evaluate(c, texts=raw)
    assert result["selected"]["k8"][0]["selected_completion_id"].endswith("k8/1")
    assert not result["raw_completions"][1]["score"]["feasible"]
    assert result["metrics"]["greedy"]["signed_gap_percent"] == 0
    assert result["metrics"]["k8"]["signed_gap_percent"] < 0


def test_missing_runtime_token_receipt_rejected():
    c = contract()
    raw = completions(c)
    del raw[0]["generated_tokens"]
    with pytest.raises(ValueError, match="generated-token count"):
        evaluate(c, texts=raw)


def test_positive_kl_reference_must_be_common_sft():
    c = contract()
    result = evaluate(c)
    checkpoint = {**result["checkpoint"], "reference_initializer_sha256": "wrong"}
    with pytest.raises(ValueError, match="frozen SFT reference"):
        pilot.evaluate_arm(c, result["arm"], completions(c), checkpoint, result["resources"])


def test_actual_update_and_prompt_receipts_reject_drift():
    c = contract()
    result = evaluate(c)
    for field, value in (("training_responses", 1), ("prompt_draw_sha256", "wrong")):
        resources = {**result["resources"], field: value}
        with pytest.raises(ValueError):
            pilot.evaluate_arm(c, result["arm"], completions(c), result["checkpoint"], resources)


def selected(identifier, feasible, gap):
    return {"instance_id": identifier, "reference_id": identifier + "-reference", "feasible": feasible,
            "signed_gap": gap if feasible else None, "clipped_gap": max(0, gap) if feasible else None,
            "u10": max(0, 1 - max(0, gap) / .1) if feasible else 0}


def test_paired_effects_use_joint_gaps_and_all_instance_utility():
    left = [selected("a", True, -.02), selected("b", True, .04), selected("c", False, 0)]
    right = [selected("a", True, .01), selected("b", False, 0), selected("c", True, .02)]
    effect = pilot.paired_bootstrap(left, right, seed=7, replicates=200)
    assert effect == pilot.paired_bootstrap(left, right, seed=7, replicates=200)
    assert effect["N"] == 3 and effect["n_joint"] == 1
    assert effect["feasibility_delta_pp"]["estimate"] == 0
    assert effect["joint_signed_gap_delta_pp"]["estimate"] == pytest.approx(-3)
    assert effect["joint_signed_gap_delta_pp"]["ci95"] == pytest.approx([-3, -3])
    assert effect["joint_clipped_gap_delta_pp"]["estimate"] == pytest.approx(-1)
    assert effect["u10_delta_pp"]["estimate"] == pytest.approx(-10 / 3)


def test_no_joint_support_is_missing_not_zero():
    effect = pilot.paired_bootstrap([selected("a", True, 0)], [selected("a", False, 0)], replicates=100)
    assert effect["n_joint"] == 0
    assert effect["joint_signed_gap_delta_pp"] == {"estimate": None, "ci95": None}


def test_incomplete_single_seed_and_unmatched_protocol_never_claim_superiority():
    c = contract()
    result = evaluate(c)
    comparison = pilot.compare_results([result], bootstrap_reps=100)
    assert not comparison["superiority_established"]
    assert comparison["groups"][0]["missing_arms"]
    assert any("fewer than five" in reason for reason in comparison["reasons"])
    control = evaluate(c, "foarl")
    control["validation_cohort_sha256"] = "different"
    control["result_sha256"] = pilot.digest({k: v for k, v in control.items() if k != "result_sha256"})
    comparison = pilot.compare_results([result, control], bootstrap_reps=100)
    assert not comparison["groups"][0]["matched"] and not comparison["groups"][0]["comparisons"]


def test_tampered_result_refused():
    result = evaluate(contract())
    result["metrics"]["k8"]["feasibility_percent"] = 0
    with pytest.raises(ValueError, match="Result digest"):
        pilot.compare_results([result], bootstrap_reps=100)
