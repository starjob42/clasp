"""Prospective plan-v2 statistics for six fixed, unassisted heldout300 arms.

Inputs are independently validated per-instance records, never aggregate means.
CVRP's signed heuristic-reference gap is primary; clipped legacy gaps remain
descriptive. Bootstrap units are whole paired instances or the explicit joint
success cohort for the conditional gap contrast.
"""
import math
import statistics

import numpy as np

TASKS = ('cvrp', 'mis')
METHODS = ('grpo_z', 'grpo_verified', 'foarl_f638_dev')
INSTANCES = 300
CANDIDATES = 8
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 2026090703


def require(value, message):
    if not value:
        raise ValueError(message)


def validated_rows(rows, task):
    require(task in TASKS and len(rows) == INSTANCES, 'exactly 300 task instances required')
    require(all(type(row['example_index']) is int for row in rows)
            and [row['example_index'] for row in rows] == list(range(INSTANCES)), 'instance order or denominator differs')
    for row in rows:
        require(type(row['feasible']) is bool and type(row['candidate_feasible_count']) is int
                and 0 <= row['candidate_feasible_count'] <= CANDIDATES
                and type(row['candidate_count']) is int and row['candidate_count'] == CANDIDATES,
                'candidate feasibility count or K8 budget differs')
        require(row['feasible'] == (row['candidate_feasible_count'] > 0), 'instance and candidate feasibility disagree')
        signed, clipped = row['best_gap_signed'], row['best_gap_clipped']
        if row['feasible']:
            require(isinstance(signed, (int, float)) and not isinstance(signed, bool) and math.isfinite(signed)
                    and isinstance(clipped, (int, float)) and not isinstance(clipped, bool) and math.isfinite(clipped)
                    and math.isclose(max(0., signed), clipped, rel_tol=1e-10, abs_tol=1e-12), 'invalid signed/clipped feasible gap')
            require(task != 'mis' or signed >= -1e-12, 'MIS result contradicts its certified optimum')
        else:
            require(signed is None and clipped is None, 'failure must have a missing gap, never a zero gap')
    return rows


def summarize_rows(rows, task, method):
    validated_rows(rows, task)
    successful = [row for row in rows if row['feasible']]
    return {'task': task, 'method': method, 'instances': INSTANCES, 'candidate_count': INSTANCES * CANDIDATES,
            'feasible_at_8_count': len(successful), 'feasible_at_8_percent': 100 * len(successful) / INSTANCES,
            'candidate_feasible_count': sum(row['candidate_feasible_count'] for row in rows),
            'candidate_feasibility_percent': 100 * sum(row['candidate_feasible_count'] for row in rows) / (INSTANCES * CANDIDATES),
            'own_success_gap_denominator': len(successful),
            'mean_signed_reference_gap_percent': 100 * statistics.fmean(row['best_gap_signed'] for row in successful) if successful else None,
            'mean_clipped_legacy_gap_percent': 100 * statistics.fmean(row['best_gap_clipped'] for row in successful) if successful else None,
            'better_than_reference_instances': sum(row['best_gap_signed'] < -1e-12 for row in successful),
            'gap_reference': 'verified_heuristic_reference' if task == 'cvrp' else 'certified_optimum'}


def exact_paired_feasibility_pvalue(wins, losses):
    require(type(wins) is int and type(losses) is int and wins >= 0 and losses >= 0, 'invalid discordant counts')
    n = wins + losses
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2 ** n)) if n else 1.


def holm_adjust(pvalues):
    require(all(math.isfinite(p) and 0 <= p <= 1 for p in pvalues), 'invalid exact p-value')
    result, previous = [None] * len(pvalues), 0.
    for rank, index in enumerate(sorted(range(len(pvalues)), key=lambda i: pvalues[i])):
        previous = max(previous, min(1., (len(pvalues) - rank) * pvalues[index]))
        result[index] = previous
    return result


def bootstrap_contrast(deltas, rng, *, simultaneous=True):
    if not deltas:
        return {'denominator': 0, 'right_minus_left': None, 'paired_bootstrap_95pct_ci': None,
                'bonferroni_eight_contrasts_99_375pct_ci': None}
    values = np.asarray(deltas, dtype=np.float64)
    require(np.isfinite(values).all(), 'nonfinite paired contrast')
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))].mean(axis=1)
    return {'denominator': len(deltas), 'right_minus_left': float(values.mean()),
            'paired_bootstrap_95pct_ci': [float(v) for v in np.quantile(draws, [.025, .975])],
            'bonferroni_eight_contrasts_99_375pct_ci': [float(v) for v in np.quantile(draws, [.003125, .996875])] if simultaneous else None}


def paired_contrast(left, right, task, left_method, right_method='grpo_verified'):
    validated_rows(left, task); validated_rows(right, task)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    pairs = list(zip(left, right, strict=True))
    wins = sum(not a['feasible'] and b['feasible'] for a, b in pairs)
    losses = sum(a['feasible'] and not b['feasible'] for a, b in pairs)
    joint = [(a, b) for a, b in pairs if a['feasible'] and b['feasible']]
    return {'task': task, 'left_method': left_method, 'right_method': right_method,
            'direction': 'modified minus baseline; positive feasibility and negative gap favor modified',
            'paired_instance_count': INSTANCES, 'joint_success_count': len(joint),
            'feasibility_discordant_wins': wins, 'feasibility_discordant_losses': losses,
            'feasibility_ties': INSTANCES - wins - losses,
            'exact_two_sided_conditional_binomial_pvalue': exact_paired_feasibility_pvalue(wins, losses),
            'feasible_at_8': bootstrap_contrast([int(b['feasible']) - int(a['feasible']) for a, b in pairs], rng),
            'paired_joint_signed_gap': bootstrap_contrast([b['best_gap_signed'] - a['best_gap_signed'] for a, b in joint], rng),
            'candidate_feasibility_secondary': bootstrap_contrast([(b['candidate_feasible_count'] - a['candidate_feasible_count']) / CANDIDATES
                                                                  for a, b in pairs], rng, simultaneous=False),
            'joint_gap_wins': sum(b['best_gap_signed'] < a['best_gap_signed'] - 1e-12 for a, b in joint),
            'joint_gap_losses': sum(b['best_gap_signed'] > a['best_gap_signed'] + 1e-12 for a, b in joint),
            'joint_gap_ties': sum(abs(b['best_gap_signed'] - a['best_gap_signed']) <= 1e-12 for a, b in joint),
            'joint_gap_left_mean': statistics.fmean(a['best_gap_signed'] for a, b in joint) if joint else None,
            'joint_gap_right_mean': statistics.fmean(b['best_gap_signed'] for a, b in joint) if joint else None}


def build_statistics(arms):
    require(set(arms) == {(task, method) for task in TASKS for method in METHODS}, 'all six predeclared arms required')
    rows = [summarize_rows(arms[task, method], task, method) for task in TASKS for method in METHODS]
    comparisons = [paired_contrast(arms[task, baseline], arms[task, 'grpo_verified'], task, baseline)
                   for task in TASKS for baseline in ('grpo_z', 'foarl_f638_dev')]
    adjusted = holm_adjust([row['exact_two_sided_conditional_binomial_pvalue'] for row in comparisons])
    for row, pvalue in zip(comparisons, adjusted, strict=True):
        row['holm_four_feasibility_tests_adjusted_pvalue'] = pvalue
    return {'scope': 'one_seed_two_task_heldout_generalization_not_confirmation_or_sota', 'summary': rows, 'comparisons': comparisons,
            'statistics': {'bootstrap_replicates': BOOTSTRAP_REPLICATES, 'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_rng': 'numpy.random.PCG64',
                           'feasibility_resampling_unit': 'paired whole instance with all eight candidates',
                           'gap_resampling_unit': 'paired whole instance within the explicit jointly successful cohort',
                           'units': 'table percentages; contrasts and intervals are fractions',
                           'primary_gap': 'signed directional reference gap; negative CVRP values preserved',
                           'multiplicity': 'Approximate percentile-bootstrap intervals at 99.375% for eight primary contrasts; Holm adjustment for four exact feasibility tests'},
            'limitations': ['One training seed and two task distributions; no training-seed robustness or broad SOTA claim.',
                            'CVRP reference routes are verified heuristics, not certified optima.',
                            'Conditional gaps require their own and joint success denominators; failed instances retain missing gaps.',
                            'Percentile-bootstrap intervals are approximate and do not guarantee exact simultaneous coverage.',
                            'The common-budget FOARL development port is not a literal reproduction of the published training pipeline.']}
