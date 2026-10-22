import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('heldout_statistics', Path(__file__).resolve().parents[1] / 'scripts/grpo_incumbent_heldout300_statistics.py')
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)


def rows(successes=range(300), gap=.2, candidates=8):
    successful = set(successes)
    return [{'example_index': i, 'feasible': i in successful, 'candidate_count': 8,
             'candidate_feasible_count': candidates if i in successful else 0,
             'best_gap_signed': gap if i in successful else None,
             'best_gap_clipped': max(0., gap) if i in successful else None} for i in range(300)]


class HeldoutStatisticsTests(unittest.TestCase):
    def test_exact_paired_feasibility_test_and_holm_known_values(self):
        self.assertEqual(stats.exact_paired_feasibility_pvalue(0, 0), 1.)
        self.assertEqual(stats.exact_paired_feasibility_pvalue(5, 0), .0625)
        self.assertEqual(stats.exact_paired_feasibility_pvalue(5, 1), .21875)
        self.assertEqual(stats.holm_adjust([.01, .03, .04, .2]), [.04, .09, .09, .2])

    def test_joint_gap_excludes_newly_solved_hard_cohort_and_keeps_all_feasibility_rows(self):
        left, right = rows(range(100), .2, 4), rows(range(50, 150), .9, 1)
        for row in right[50:100]:
            row['best_gap_signed'] = row['best_gap_clipped'] = .1
        pair = stats.paired_contrast(left, right, 'cvrp', 'grpo_z')
        self.assertEqual(pair['paired_instance_count'], 300)
        self.assertEqual(pair['joint_success_count'], 50)
        self.assertEqual(pair['feasibility_discordant_wins'], 50)
        self.assertEqual(pair['feasibility_discordant_losses'], 50)
        self.assertEqual(pair['feasible_at_8']['right_minus_left'], 0.)
        self.assertAlmostEqual(pair['paired_joint_signed_gap']['right_minus_left'], -.1)
        own = stats.summarize_rows(right, 'cvrp', 'grpo_verified')
        self.assertEqual(own['candidate_count'], 2400)
        self.assertEqual(own['candidate_feasible_count'], 100)
        self.assertEqual(own['own_success_gap_denominator'], 100)
        self.assertAlmostEqual(own['mean_signed_reference_gap_percent'], 50.)

    def test_negative_cvrp_reference_gap_is_preserved_in_primary_contrast(self):
        pair = stats.paired_contrast(rows(gap=0.), rows(gap=-.1), 'cvrp', 'grpo_z')
        self.assertAlmostEqual(pair['paired_joint_signed_gap']['right_minus_left'], -.1)
        for bound in pair['paired_joint_signed_gap']['bonferroni_eight_contrasts_99_375pct_ci']:
            self.assertAlmostEqual(bound, -.1)
        own = stats.summarize_rows(rows(gap=-.1), 'cvrp', 'grpo_verified')
        self.assertEqual(own['better_than_reference_instances'], 300)
        self.assertEqual(own['mean_clipped_legacy_gap_percent'], 0.)
        self.assertAlmostEqual(own['mean_signed_reference_gap_percent'], -10.)

    def test_all_failure_gap_is_missing_and_must_never_be_zero(self):
        failed = rows([])
        own = stats.summarize_rows(failed, 'mis', 'grpo_z')
        self.assertIsNone(own['mean_signed_reference_gap_percent'])
        self.assertEqual(own['feasible_at_8_count'], 0)
        failed[0]['best_gap_signed'] = failed[0]['best_gap_clipped'] = 0.
        with self.assertRaisesRegex(ValueError, 'missing gap'):
            stats.validated_rows(failed, 'mis')

    def test_count_candidate_or_certified_optimum_contradictions_fail(self):
        with self.assertRaisesRegex(ValueError, '300'):
            stats.validated_rows(rows()[:299], 'cvrp')
        invalid = rows(); invalid[0]['candidate_count'] = 7
        with self.assertRaisesRegex(ValueError, 'K8'):
            stats.validated_rows(invalid, 'cvrp')
        for field, value, message in (('example_index', False, 'instance order'),
                                      ('candidate_count', 8., 'K8'),
                                      ('candidate_feasible_count', True, 'K8')):
            with self.subTest(field=field):
                invalid = rows(); invalid[0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    stats.validated_rows(invalid, 'cvrp')
        with self.assertRaisesRegex(ValueError, 'certified optimum'):
            stats.validated_rows(rows(gap=-.01), 'mis')

    def test_all_six_arms_and_four_primary_comparisons_are_required(self):
        arms = {(task, method): rows() for task in stats.TASKS for method in stats.METHODS}
        result = stats.build_statistics(arms)
        self.assertEqual(len(result['summary']), 6)
        self.assertEqual(len(result['comparisons']), 4)
        self.assertTrue(all(row['holm_four_feasibility_tests_adjusted_pvalue'] == 1 for row in result['comparisons']))
        self.assertEqual(result['statistics']['bootstrap_replicates'], 10000)
        self.assertEqual(result['statistics']['bootstrap_seed'], 2026090703)
        del arms['mis', 'foarl_f638_dev']
        with self.assertRaisesRegex(ValueError, 'all six'):
            stats.build_statistics(arms)
