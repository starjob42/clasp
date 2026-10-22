"""Pure synthetic training-only teacher tests; no models or protected data."""
import copy
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random

import pytest

from clasp.baselines.sft_anchor_pilot import _raw_score,digest,graph_semantic_id,semantic_id

PATH = Path(__file__).resolve().parents[1]/"experiments/rloo_one_shot_20261016/teacher_candidate_v1.py"
spec = importlib.util.spec_from_file_location("one_shot_teacher_candidate_v1",PATH)
t = importlib.util.module_from_spec(spec);spec.loader.exec_module(t)


def row(task,instance,identifier="synthetic-training"):
    return {"task_type":task,"instance_id":identifier,"semantic_id":semantic_id(task,instance),
            "instance_json":copy.deepcopy(instance),"prompt":"Solve this synthetic training instance", "split":"training"}


def setup(task,instance,**limit_changes):
    r = row(task,instance);excluded = {split:[] for split in t.SPLITS}
    limits = {**t.LIMITS,"max_wall_seconds":10.0,**limit_changes}
    a = t.make_training_admission(task,[r],allowed_training_semantic_ids=[r["semantic_id"]],
        excluded_semantic_ids_by_split=excluded,excluded_graph_ids_by_split=excluded,
        training_dataset_sha256=digest([r]),limits=limits)
    c = {"split":"training","operation":"supervised_training_target","inference_search":False,
         "training_dataset_sha256":a["training_dataset_sha256"]}
    return r,a,c


def teach(r,a,c,text,**kwargs):
    return t.generate_teacher(r,text,admission=a,expected_admission_sha256=a["admission_sha256"],context=c,**kwargs)


def check_retention(result,instance):
    score = _raw_score(result["task"],result["target_text"],instance)
    assert score["feasible"] and score["solution"] == result["target_solution"]
    assert score["objective"] == result["target_objective"]
    if result["task"] == "op": assert score["objective"] >= result["incumbent_objective"]
    else: assert score["objective"] <= result["incumbent_objective"]
    assert result["move_evaluations"] <= result["limits"]["max_move_evaluations"]
    assert result["passes"] <= result["limits"]["max_passes"]
    assert result["model_calls"] == 0 and not result["inference_search"]
    assert not result["learned_gain_established"] and not result["trainer_integration"]
    assert result["result_sha256"] == digest({k:v for k,v in result.items() if k != "result_sha256"})


@pytest.mark.parametrize("instance,text,kind",[
    ([[[0,0],[1,0],[-1,0],[0,1]],[0,1,1,1],3],"Routes: [[0, 1, 2, 3, 0]]","two_opt"),
    ([[[0,0],[1,0],[2,0]],[0,1,1],2],"Routes: [[0, 1, 0], [0, 2, 0]]","relocate"),
    ([[[0,0],[1,0],[10,0],[1,1],[10,1]],[0,1,1,1,1],2],"Routes: [[0, 1, 2, 0], [0, 3, 4, 0]]","swap"),
])
def test_cvrp_actual_strict_local_improvement_and_each_neighborhood(instance,text,kind):
    r,a,c = setup("cvrp",instance)
    result = teach(r,a,c,text,clock=lambda:0.)
    check_retention(result,instance)
    assert result["improved"] and result["target_objective"] < result["incumbent_objective"]-1e-8
    assert kind in [move["move"] for move in result["accepted_internal_moves"]]
    assert all(move["after_objective"] < move["before_objective"] for move in result["accepted_internal_moves"])


@pytest.mark.parametrize("instance,text,kind,objective",[
    ([[[0,0],[1,0],[2,0]],[0,2,7],3],"Route: [0, 1]","insertion",9),
    ([[[0,0],[2,0],[-1,0]],[0,1,7],2],"Route: [0, 1]","removal_replacement",7),
])
def test_op_genuine_prize_improvement_insertion_and_replacement(instance,text,kind,objective):
    r,a,c = setup("op",instance)
    result = teach(r,a,c,text,clock=lambda:0.)
    check_retention(result,instance)
    assert result["improved"] and result["target_objective"] == objective
    assert kind in [move["move"] for move in result["accepted_internal_moves"]]


def test_op_open_path_two_opt_and_zero_prize_removal_enable_feasible_insertions():
    instance = [[[0,0],[1,1],[-1,1],[1,-1],[1,-1.5]],[0,1,1,1,7],6.3]
    route = [0,1,2,3];original = _raw_score("op",t.format_solution("op",route,0),instance)
    assert original["feasible"]
    assert all(not _raw_score("op",t.format_solution("op",candidate,0),instance)["feasible"]
               for _,candidate in t._op_insert(route,instance[1]))
    shorter = [candidate for _,candidate in t._op_two_opt(route)
               if _raw_score("op",t.format_solution("op",candidate,0),instance)["distance"] < original["distance"]-1e-8]
    assert shorter and any(_raw_score("op",t.format_solution("op",candidate,0),instance)["feasible"]
                          for candidate in shorter for _,candidate in t._op_insert(candidate,instance[1]))
    # Removing a zero-prize detour preserves prize and frees the open-path budget.
    empty_prize = [[[0,0],[0,3],[1,0]],[0,0,4],7]
    before = _raw_score("op","Route: [0, 1, 2]",empty_prize)
    after = _raw_score("op","Route: [0, 2]",empty_prize)
    assert before["feasible"] and after["feasible"] and before["objective"] == after["objective"]
    assert after["distance"] < before["distance"] and [0,2] in [s for _,s in t._op_remove([0,1,2])]


def test_equal_prize_shorter_internal_op_path_does_not_create_a_fake_quality_target():
    instance = [[[0,0],[1,1],[-1,1],[1,-1]],[0,1,1,1],7]
    r,a,c = setup("op",instance)
    result = teach(r,a,c,"Route: [0, 1, 2, 3]",clock=lambda:0.)
    check_retention(result,instance)
    assert result["accepted_internal_moves"] and not result["improved"]
    assert result["target_solution"] == result["incumbent_solution"] == [0,1,2,3]


@pytest.mark.parametrize("task,instance,text",[
    ("op",[[[0,0],[1,0]],[0,3],1],"Route: [0, 1]\nObjective: 999999"),
    ("cvrp",[[[0,0],[1,0]],[0,1],1],"Routes: [[0, 1, 0]]\nObjective: -100"),
    ("op",[[[0,0],[0,0]],[0,0],1],"Route: [0, 1]"),
    ("cvrp",[[[0,0],[0,0]],[0,1],1],"Routes: [[0, 1, 0]]"),
])
def test_optimal_or_degenerate_feasible_incumbent_retention_recomputes_printed_objective(task,instance,text):
    r,a,c = setup(task,instance);result = teach(r,a,c,text,clock=lambda:0.)
    check_retention(result,instance)
    assert not result["improved"] and result["incumbent_solution_retained"]
    assert result["target_solution"] == result["incumbent_solution"]
    assert result["target_text"].endswith("Objective: "+format(result["target_objective"],".17g"))


@pytest.mark.parametrize("task,text",[
    ("op","Route: [0, True]"),("op","Route: [0, 1.0]"),("op",'Route: [0, "1"]'),
    ("op","Route: [0, 1, 1]"),("op","Route: [1, 0]"),("op","Route: [0, 99]"),("op","invalid"),
    ("cvrp","Routes: [[0, 1, 0]]"),("cvrp","Routes: [[0, 1, 0, 2, 0]]"),
    ("cvrp","Routes: [[0, 1, 2.0, 0]]"),("cvrp","Routes: [[0, 1, 1, 2, 0]]"),
])
def test_invalid_incumbents_are_rejected_instead_of_repaired(task,text):
    instance = [[[0,0],[1,0],[2,0]],[0,1,1],3]
    r,a,c = setup(task,instance)
    with pytest.raises(ValueError,match="strictly feasible typed"):
        teach(r,a,c,text,clock=lambda:0.)


def test_all_candidate_capacity_and_budget_checks_are_full_strict_replay(monkeypatch):
    instance = [[[0,0],[1,0],[2,0]],[0,1,1],1]
    r,a,c = setup("cvrp",instance);real = t._raw_score;calls = []
    def observed(task,text,physical):
        value = real(task,text,physical);calls.append(value);return value
    monkeypatch.setattr(t,"_raw_score",observed)
    result = teach(r,a,c,"Routes: [[0, 1, 0], [0, 2, 0]]",clock=lambda:0.)
    check_retention(result,instance)
    assert any("capacity_exceeded" in call["violations"] for call in calls)
    assert len(calls) == result["move_evaluations"]+2
    assert result["incumbent_solution_retained"]
    op = [[[0,0],[1,0],[10,0]],[0,1,9],1]
    r,a,c = setup("op",op);result = teach(r,a,c,"Route: [0, 1]",clock=lambda:0.)
    check_retention(result,op)
    assert result["target_solution"] == [0,1]


@pytest.mark.parametrize("split",["validation","test","engineering_eval","evaluation","engineering_training"])
def test_nontraining_request_context_is_denied_even_with_allowed_training_id(split):
    r,a,c = setup("op",[[[0,0],[1,0]],[0,1],1]);c["split"] = split
    with pytest.raises(ValueError,match="training-only context"):teach(r,a,c,"Route: [0,1]")


@pytest.mark.parametrize("change",[{"inference_search":True},{"operation":"greedy_evaluation"},
                                  {"evaluation":True},{"training_dataset_sha256":"0"*64}])
def test_teacher_cannot_be_used_as_inference_or_rebound_dataset(change):
    r,a,c = setup("op",[[[0,0],[1,0]],[0,1],1]);c.update(change)
    with pytest.raises(ValueError,match="training-only context"):teach(r,a,c,"Route: [0,1]")


@pytest.mark.parametrize("change",[{"split":"test"},{"metadata":{"dataset_role":"validation"}},
                                  {"semantic_id":"0"*64},{"prompt":"different prompt"},{"instance_id":"copied-id"}])
def test_nontraining_or_copied_row_context_is_denied(change):
    r,a,c = setup("op",[[[0,0],[1,0]],[0,1],1]);r.update(change)
    with pytest.raises(ValueError):teach(r,a,c,"Route: [0,1]")


def test_complete_allowed_and_excluded_physical_id_sets_are_required():
    instance = [[[0,0],[1,0]],[0,1],1];r,a,c = setup("op",instance)
    excluded = {split:[] for split in t.SPLITS};args = dict(allowed_training_semantic_ids=[r["semantic_id"]],
        excluded_semantic_ids_by_split=excluded,excluded_graph_ids_by_split=excluded,training_dataset_sha256=digest([r]))
    for field,value in [("allowed_training_semantic_ids",[]),("allowed_training_semantic_ids",["short"]),
                        ("excluded_semantic_ids_by_split",{})]:
        with pytest.raises(ValueError):t.make_training_admission("op",[r],**{**args,field:value})
    for split in t.SPLITS:
        forbidden = {**excluded,split:[r["semantic_id"]]}
        with pytest.raises(ValueError,match="overlaps"):t.make_training_admission("op",[r],**{**args,"excluded_semantic_ids_by_split":forbidden})
    changed_budget = copy.deepcopy(instance);changed_budget[2] = 2
    assert semantic_id("op",instance) != semantic_id("op",changed_budget)
    assert graph_semantic_id("op",instance) == graph_semantic_id("op",changed_budget)
    forbidden_graph = {**excluded,"validation":[graph_semantic_id("op",changed_budget)]}
    with pytest.raises(ValueError,match="overlaps"):t.make_training_admission("op",[r],**{**args,"excluded_graph_ids_by_split":forbidden_graph})


def test_resealed_but_physically_forged_admission_cannot_authorize_a_row():
    r,a,c = setup("op",[[[0,0],[1,0]],[0,1],1]);original_sha = a["admission_sha256"]
    a["training_rows"][0]["instance_json"][1][1] = 99
    with pytest.raises(ValueError,match="identity differs"):
        t.generate_teacher(r,"Route: [0,1]",admission=a,expected_admission_sha256=original_sha,context=c)
    a["training_rows_sha256"] = digest(a["training_rows"])
    a["admission_sha256"] = digest({k:v for k,v in a.items() if k != "admission_sha256"})
    with pytest.raises(ValueError,match="physical row identity"):teach(r,a,c,"Route: [0,1]")


@pytest.mark.parametrize("key,value",[("max_move_evaluations",True),("max_move_evaluations",100001),
    ("max_passes",0),("max_wall_seconds",float("nan")),("max_wall_seconds",31),("max_nodes",201),
    ("max_move_evaluations",10**500)])
def test_declared_cpu_bounds_are_typed_finite_and_hard_capped(key,value):
    with pytest.raises(ValueError):setup("op",[[[0,0],[1,0]],[0,1],1],**{key:value})


def test_fixed_move_budget_and_elapsed_boundary_retain_incumbent():
    instance = [[[0,0],[1,0],[2,0]],[0,1,1],2]
    r,a,c = setup("cvrp",instance,max_move_evaluations=1)
    result = teach(r,a,c,"Routes: [[0,1,0],[0,2,0]]",clock=lambda:0.)
    check_retention(result,instance)
    assert result["move_evaluations"] == 1 and result["stop_reason"] == "move_evaluation_budget"
    times = iter([0.,11.,12.])
    result = teach(r,a,c,"Routes: [[0,1,0],[0,2,0]]",clock=lambda:next(times))
    check_retention(result,instance)
    assert result["move_evaluations"] == 0 and result["stop_reason"] == "wall_boundary" and not result["improved"]


@pytest.mark.parametrize("task,instance,text",[
    ("op",[[[0,0],[1,0],[2,0]],[0,1e308,1e308],3],"Route: [0,1]"),
    ("cvrp",[[[0,0],[1,0],[2,0]],[0,1e308,1e308],1e308],"Routes: [[0,1,0],[0,2,0]]"),
])
def test_overflowing_candidate_cannot_destroy_an_existing_finite_feasible_incumbent(task,instance,text):
    r,a,c = setup(task,instance)
    result = teach(r,a,c,text,clock=lambda:0.)
    check_retention(result,instance)
    assert result["numeric_candidate_rejections"] > 0 and result["incumbent_solution_retained"]


@pytest.mark.parametrize("task",["op","cvrp"])
def test_deterministic_bounded_teacher_and_randomized_retention_across_small_instances(task):
    rng = random.Random(6101601)
    for index in range(12):
        coords = [[0,0]]+[[rng.randrange(-5,6),rng.randrange(-5,6)] for _ in range(5)]
        if task == "op":
            instance = [coords,[0]+[rng.randrange(0,10) for _ in range(5)],100]
            text = "Route: "+json.dumps([0,1,3])
        else:
            instance = [coords,[0,1,1,1,1,1],3]
            ids = list(range(1,6));rng.shuffle(ids)
            text = "Routes: "+json.dumps([[0,*ids[:3],0],[0,*ids[3:],0]])
        r,a,c = setup(task,instance,max_move_evaluations=100,max_passes=3)
        result = teach(r,a,c,text,clock=lambda:0.)
        assert result == teach(r,a,c,text,clock=lambda:0.)
        check_retention(result,instance)
        for move in result["accepted_internal_moves"]:
            if task == "op": assert move["after_objective"] >= move["before_objective"]
            else: assert move["after_objective"] < move["before_objective"]


def test_full_typed_physical_input_and_scope_are_required():
    for task,instance in [("tsp",[[[0,0],[1,0]],[0,1],1]),("op",[[[0,0],[True,0]],[0,1],1]),
        ("op",[[[0,0],[1,0]],[1,1],1]),("cvrp",[[[0,0],[1,0]],[0,2],1])]:
        with pytest.raises(ValueError):setup(task,instance)
