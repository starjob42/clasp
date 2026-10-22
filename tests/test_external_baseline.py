"""Past-only credit, distributed determinism and exact checkpoint restoration."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import replace
from fractions import Fraction

import pytest

from clasp.rewards.external_baseline import ExternalBaselineTracker, external_advantages


def observe(tracker, keys, rewards):
    snapshot = tracker.snapshot(keys)
    result = tracker.advantages(snapshot, rewards)
    tracker.update(snapshot, rewards)
    return result


def rehash(state):
    payload = {key: value for key, value in state.items() if key != "sha256"}
    state["sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()
    return state


def test_unique_prompt_epoch_uses_previous_bucket_data_without_current_reward_leakage():
    tracker = ExternalBaselineTracker()
    first = tracker.snapshot([("cvrp", 12), ("cvrp", 20)])
    assert first.values == (0.0, 0.0)
    assert tracker.advantages(first, [-0.5, -0.5]) == [-0.5, -0.5]
    assert tracker.advantages(first, [1.5, 1.5]) == [1.5, 1.5]
    assert tracker.revision == 0
    tracker.update(first, [-0.5, -0.5])
    next_batch = tracker.snapshot([("cvrp", 22)])
    assert next_batch.values == (-0.5,)
    assert tracker.advantages(next_batch, [1.5]) == [2.0]
    # Previously collected credit stays defined after advancement.
    assert tracker.advantages(first, [-0.5, -0.5]) == [-0.5, -0.5]


def test_inclusive_size_buckets_and_task_global_backoff_are_completion_weighted():
    tracker = ExternalBaselineTracker(size_boundaries=(25, 50))
    observe(tracker, [("cvrp", 25), ("cvrp", 20), ("cvrp", 26)], [-0.5, 1.5, 2.0])
    probe = tracker.snapshot([("cvrp", 24), ("cvrp", 50), ("cvrp", 80), ("mis", 10)])
    assert probe.values == (0.5, 2.0, 1.0, 1.0)
    observe(tracker, [("mis", 5)], [-1.0])
    probe = tracker.snapshot([("mis", 60), ("cvrp", 80), ("new_task", 1)])
    assert probe.values == (-1.0, 1.0, 0.5)


def test_explicit_initializer_and_one_constant_scale_apply_across_tasks():
    tracker = ExternalBaselineTracker(initial_value=0.5, scale=2.0)
    snapshot = tracker.snapshot([("mis", 10), ("cvrp", 50)])
    assert snapshot.scale == 2.0
    assert tracker.advantages(snapshot, [-1, 1.5]) == [-0.75, 0.5]
    # No sample-dependent centering: the mean of these values need not be zero.
    assert sum(tracker.advantages(snapshot, [-1, 1.5])) != 0


def test_checkpoint_roundtrip_reproduces_predictions_and_future_updates_exactly():
    original = ExternalBaselineTracker(initial_value=0.25, size_boundaries=(16, 64))
    observe(original, [("mis", 10), ("mis", 100), ("cvrp", 20)], [1.1, -0.3, 1.5])
    saved = json.loads(json.dumps(original.state_dict()))
    restored = ExternalBaselineTracker.from_state_dict(saved)
    assert restored.state_dict() == original.state_dict()
    for keys, rewards in [([("mis", 50), ("cvrp", 80)], [-0.2, 1.2]),
                          ([("mis", 10), ("mis", 10)], [1.7, -0.1])]:
        left, right = original.snapshot(keys), restored.snapshot(keys)
        assert left == right
        assert original.advantages(left, rewards) == restored.advantages(right, rewards)
        original.update(left, rewards)
        restored.update(right, rewards)
        assert original.state_dict() == restored.state_dict()
    assert restored.seen_samples == 7
    assert restored.revision == 3
    # state_dict is independent of live tracker state.
    saved["config"]["size_boundaries"].append(999)
    saved["cells"][0]["reward_sum"] = 999
    assert original.state_dict() == restored.state_dict()


def test_rank_replicas_and_permuted_global_arrivals_have_same_canonical_state():
    pairs = [(("mis", 10), -0.1), (("mis", 10), 1.4),
             (("cvrp", 50), -1.0), (("cvrp", 80), 1.5),
             (("mis", 10), 0.3), (("cvrp", 50), 1.6)]
    states = []
    for seed in range(4):
        rows = list(pairs)
        random.Random(seed).shuffle(rows)
        tracker = ExternalBaselineTracker()
        observe(tracker, [key for key, _ in rows], [reward for _, reward in rows])
        states.append(tracker.state_dict())
    assert all(state == states[0] for state in states)


def test_replayed_foreign_and_changed_snapshots_fail_without_state_mutation():
    tracker = ExternalBaselineTracker()
    snapshot = tracker.snapshot([("mis", 10)])
    before = tracker.state_dict()
    with pytest.raises(ValueError, match="altered"):
        tracker.update(replace(snapshot, values=(1.5,)), [1.5])
    assert tracker.state_dict() == before
    foreign = ExternalBaselineTracker(initial_value=0.5).snapshot([("mis", 10)])
    with pytest.raises(ValueError, match="foreign"):
        tracker.update(foreign, [1.5])
    assert tracker.state_dict() == before
    tracker.update(snapshot, [1.5])
    after = tracker.state_dict()
    with pytest.raises(ValueError, match="stale, replayed"):
        tracker.update(snapshot, [-1])
    assert tracker.state_dict() == after


@pytest.mark.parametrize("rewards", [[math.nan], [math.inf], [True], [], [-1, 2], "1"])
def test_invalid_global_reward_batch_leaves_checkpoint_state_unchanged(rewards):
    tracker = ExternalBaselineTracker()
    before = tracker.state_dict()
    with pytest.raises(ValueError):
        tracker.update(tracker.snapshot([("mis", 10)]), rewards)
    assert tracker.state_dict() == before


def test_overflow_in_later_bucket_does_not_partially_update_earlier_bucket():
    tracker = ExternalBaselineTracker()
    keys = [("cvrp", 10), ("mis", 50), ("mis", 50)]
    snapshot = tracker.snapshot(keys)
    before = tracker.state_dict()
    with pytest.raises(ValueError, match="overflow"):
        tracker.update(snapshot, [1.5, 1e308, 1e308])
    assert tracker.state_dict() == before


@pytest.mark.parametrize("keys", [[], [('', 5)], [('mis', 0)], [('mis', True)],
                                    [('mis', 5.0)], [('mis',)], 'mis', [(1, 5)]])
def test_missing_or_ambiguous_state_keys_are_rejected(keys):
    with pytest.raises(ValueError):
        ExternalBaselineTracker().snapshot(keys)


@pytest.mark.parametrize("kwargs", [{"scale": 0}, {"scale": -1}, {"scale": math.nan},
                                      {"scale": True}, {"initial_value": math.inf},
                                      {"size_boundaries": (25, 25)},
                                      {"size_boundaries": (50, 25)},
                                      {"size_boundaries": (True,)}])
def test_invalid_declared_baseline_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ExternalBaselineTracker(**kwargs)


def test_fresh_empty_checkpoint_restores_and_empty_boundaries_pool_within_task():
    tracker = ExternalBaselineTracker(size_boundaries=())
    restored = ExternalBaselineTracker.from_state_dict(tracker.state_dict())
    assert restored.state_dict() == tracker.state_dict()
    observe(restored, [("mis", 1), ("mis", 1000)], [1.0, -1.0])
    assert restored.snapshot([("mis", 50)]).values == (0.0,)


def test_checksum_and_internal_invariants_both_protect_checkpoint_loading():
    tracker = ExternalBaselineTracker()
    observe(tracker, [("mis", 10)], [1.5])
    state = tracker.state_dict()
    bad = copy.deepcopy(state)
    bad["cells"][0]["reward_sum"] = -1.0
    with pytest.raises(ValueError, match="checksum"):
        ExternalBaselineTracker.from_state_dict(bad)
    for field, value in [("bucket", 99), ("count", 0), ("count", True)]:
        bad = copy.deepcopy(state)
        bad["cells"][0][field] = value
        with pytest.raises(ValueError):
            ExternalBaselineTracker.from_state_dict(rehash(bad))
    bad = copy.deepcopy(state)
    bad["cells"].append(copy.deepcopy(bad["cells"][0]))
    with pytest.raises(ValueError, match="unique"):
        ExternalBaselineTracker.from_state_dict(rehash(bad))
    for revision in (0, 2, True):
        bad = copy.deepcopy(state)
        bad["revision"] = revision
        with pytest.raises(ValueError):
            ExternalBaselineTracker.from_state_dict(rehash(bad))
    bad = copy.deepcopy(state)
    bad["config"]["scale"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        ExternalBaselineTracker.from_state_dict(rehash(bad))


def test_exact_categorical_external_baseline_preserves_the_reward_gradient():
    # Softmax-logit scores are 1{action=j}-p_j. This independent analytic check
    # validates fixed scaling/centering semantics without torch or a model.
    probabilities = [Fraction(1, 5), Fraction(3, 10), Fraction(1, 2)]
    rewards = [Fraction(-1), Fraction(3, 2), Fraction(7, 4)]
    expected_reward = sum(p * r for p, r in zip(probabilities, rewards))
    tracker = ExternalBaselineTracker(initial_value=0.75, scale=2.0)
    snapshot = tracker.snapshot([("mis", 10)] * 3)
    advantages = external_advantages(snapshot, [float(r) for r in rewards])
    observed = [sum(float(p) * a * (int(i == j) - float(probabilities[j]))
                    for i, (p, a) in enumerate(zip(probabilities, advantages)))
                for j in range(3)]
    target = [float(p * (r - expected_reward) / 2) for p, r in zip(probabilities, rewards)]
    assert observed == pytest.approx(target, abs=1e-15)


def test_equal_failure_rewards_keep_realized_credit_but_not_a_fake_expected_direction():
    tracker = ExternalBaselineTracker()
    snapshot = tracker.snapshot([("mis", 10), ("mis", 10)])
    advantages = tracker.advantages(snapshot, [-1, -1])
    assert advantages == [-1, -1]
    # Every possible action has the same reward; the expected gradient is zero.
    expected_score = 0.25 * advantages[0] * (1 - 0.25) + 0.75 * advantages[1] * -0.25
    assert expected_score == 0


def test_response_length_mean_would_break_the_external_baseline_cancellation():
    tracker = ExternalBaselineTracker(initial_value=0.5)
    snapshot = tracker.snapshot([("mis", 10), ("mis", 10)])
    a_short, a_long = tracker.advantages(snapshot, [1.0, 1.0])
    # Parameter is the logit of the long completion; each has probability1/2.
    fixed_horizon_gradient = (0.5 * a_short * -0.5 + 0.5 * a_long * 0.5) / 2
    response_mean_gradient = 0.5 * a_short * -0.5 / 1 + 0.5 * a_long * 0.5 / 2
    assert fixed_horizon_gradient == 0
    assert response_mean_gradient == -1 / 16
