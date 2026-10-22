import math

import pytest
import torch

from clasp.set_grpo import compute_canonical_clasp_set_advantages, enumerate_k_subsets


@pytest.mark.parametrize("batch_size,set_size", [(2, 1), (4, 2), (6, 3), (7, 6)])
def test_safe_enumeration_exactly_matches_small_legacy_order(batch_size, set_size):
    expected = torch.combinations(torch.arange(batch_size), r=set_size)
    actual = enumerate_k_subsets(batch_size, set_size)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.long
    assert actual.device == expected.device


def test_b18_k8_cardinality_lexicographic_order_and_complete_unique_rows():
    actual = enumerate_k_subsets(18, 8, device=torch.device("cpu"))
    rows = [tuple(row) for row in actual.tolist()]
    assert actual.shape == (math.comb(18, 8), 8) == (43_758, 8)
    assert rows == sorted(rows)
    assert len(set(rows)) == math.comb(18, 8)
    assert all(0 <= row[0] < row[-1] < 18 and all(a < b for a, b in zip(row, row[1:])) for row in rows)
    assert rows[0] == tuple(range(8))
    assert rows[-1] == tuple(range(10, 18))


def test_cached_indices_cannot_be_corrupted_by_returned_tensor_mutation():
    first = enumerate_k_subsets(18, 8)
    first.fill_(-1)
    second = enumerate_k_subsets(18, 8)
    assert first.data_ptr() != second.data_ptr()
    assert second[0].tolist() == list(range(8))
    assert second[-1].tolist() == list(range(10, 18))
    assert bool((second >= 0).all())


def test_requested_device_is_not_inherited_from_previous_cached_result():
    cpu = enumerate_k_subsets(5, 3, device="cpu")
    meta = enumerate_k_subsets(5, 3, device="meta")
    assert cpu.device.type == "cpu"
    assert meta.device.type == "meta"
    assert meta.dtype == torch.long
    assert meta.shape == cpu.shape


def test_full_b18_credit_matches_failure_counts_and_retains_small_top_ties():
    objective = torch.full((3, 18), 0.5, dtype=torch.float64)
    flags = torch.ones_like(objective)
    flags[0, 9:] = 0
    objective[0] = flags[0]
    objective[1, :2] = 1
    objective[2, :11] = 1
    result = compute_canonical_clasp_set_advantages(
        objective, 1 - flags, flags,
        rollout_batch_size=18, deployment_set_size=8, group_lambdas=1.0,
    )
    # With nine feasible and nine bad candidates, a failing subset is still
    # possible. These counts are independent of the subset enumerator.
    expected_feasible = -math.comb(9, 8) / math.comb(17, 8)
    expected_bad = math.comb(8, 7) / math.comb(17, 7) - math.comb(8, 8) / math.comb(17, 8)
    torch.testing.assert_close(result.failure_advantages[0, :9], torch.full((9,), expected_feasible, dtype=torch.float64))
    torch.testing.assert_close(result.failure_advantages[0, 9:], torch.full((9,), expected_bad, dtype=torch.float64))
    assert result.deployment_failure_estimates[0].item() == pytest.approx(math.comb(9, 8) / math.comb(18, 8))
    # Two tied winners no longer make every K-subset's max identical; eleven
    # tied winners still do, because only seven lower values remain.
    expected_tied_winner = 0.5 * math.comb(16, 8) / math.comb(17, 8)
    torch.testing.assert_close(result.quality_advantages[1, :2], torch.full((2,), expected_tied_winner, dtype=torch.float64))
    torch.testing.assert_close(result.quality_advantages[2], torch.zeros(18, dtype=torch.float64), rtol=0, atol=1e-12)
    torch.testing.assert_close(result.mean_reduced_policy_coefficients.sum(1), torch.zeros(3, dtype=torch.float64), rtol=0, atol=1e-11)
