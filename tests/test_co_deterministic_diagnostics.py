"""Failure-oriented admission tests for the prospective deterministic diagnostics."""
import importlib.util
import json
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/co_deterministic_diagnostics.py'
spec = importlib.util.spec_from_file_location('co_diag_test_module', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Diagnostics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'campaign'
        self.root.mkdir()
        self.source = Path(self.tmp.name) / 'source'
        self.sha = 'a' * 64
        self.protocol = m.protocol_material(self.source, self.sha, self.root)
        self.put(self.root / 'protocol.json', self.protocol)

    @staticmethod
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def initial_pair(self, method='foarl_teacher'):
        for suffix, step, job in [('full', 10, '123'), ('staged', 5, '456')]:
            name = f'mis_{method}_{suffix}'
            output = self.root / name
            self.put(self.root / f'{name}-attempt.json', {'command': m.command(
                self.source, self.sha, self.root, name, 'mis', method, step)})
            self.put(self.root / f'{name}-submission.json', {'returncode': 0, 'stdout': job + '\n', 'stderr': ''})
            self.put(output / 'local_development_run_contract.json', {
                'source': {'root': str(self.source), 'manifest_sha256': self.sha},
                'method': method, 'diagnostic': True})
            self.put(output / f'chunk-{step}-admission.json', {'job_id': job, 'stop': step})
            self.put(output / f'chunk-{step}-exit.json', {'returncode': 0})
            self.put(output / f'chunk-{step}-complete.json', {'validated': step})

    def accounting(self, args, **kwargs):
        job = args[args.index('-j') + 1]
        for path in self.root.glob('*-submission.json'):
            if json.loads(path.read_text())['stdout'].strip() == job:
                attempt = json.loads(path.with_name(path.name.replace('-submission', '-attempt')).read_text())
                return '|'.join([job, 'uag72kz', 'COMPLETED', '0:0', shlex.join(attempt['command'])]) + '\n'
        raise AssertionError('unknown job')

    def admit(self, method='foarl_teacher', comparison=None):
        comparison = comparison or {'status': 'passed_exact_step5_prerestart', 'differing_field_count': 0}
        with patch.object(m.subprocess, 'check_output', side_effect=self.accounting), \
             patch.object(m, 'validate_checkpoint', side_effect=lambda checkpoint, step, *args: {'validated': step}), \
             patch.object(m, 'source_module', return_value=SimpleNamespace(compare_checkpoints=lambda *args, **kwargs: comparison)):
            return m.admit_resume(self.root, self.source, self.sha, method)

    def test_six_initial_two_resumes_no_evaluation(self):
        self.assertEqual(len(self.protocol['plans']), 6)
        self.assertEqual(len(self.protocol['resume_plans']), 2)
        for plan in self.protocol['plans']:
            self.assertIn('--time=02:00:00', plan['command'])
            self.assertIn('--no-requeue', plan['command'])
            self.assertIn('CO_DIAGNOSTIC=1', ' '.join(plan['command']))
        self.assertEqual(self.protocol['fixed_horizon'], 400)

    def test_protocol_mutation_rejected(self):
        for field in ['source_manifest_sha256', 'runtime_profile', 'preresume_gate', 'fixed_horizon', 'plans']:
            with self.subTest(field=field):
                changed = {**self.protocol, field: None}
                with self.assertRaises(ValueError):
                    m.validate_protocol(changed, self.source, self.sha, self.root)

    def test_resume_requires_full_semantic_prerestart_equality(self):
        self.initial_pair()
        result = self.admit()
        self.assertEqual(result, self.protocol['resume_plans'][0]['command'])
        proof = json.loads((self.root / 'foarl_teacher-step5-prerestart-equivalence.json').read_text())
        self.assertEqual(proof['status'], 'passed_exact_step5_prerestart')
        self.assertEqual(self.admit(), result)

    def test_classic_also_requires_step5_comparison(self):
        self.initial_pair('classic_grpo')
        self.assertEqual(self.admit('classic_grpo'), self.protocol['resume_plans'][1]['command'])

    def test_different_step5_state_blocks_resume_and_retains_failure(self):
        self.initial_pair()
        with self.assertRaisesRegex(ValueError, 'states differ'):
            self.admit(comparison={'status': 'failed_state_difference', 'differing_field_count': 1})
        self.assertTrue((self.root / 'foarl_teacher-step5-prerestart-equivalence.json').is_file())
        with self.assertRaisesRegex(ValueError, 'proof changed'):
            self.admit()

    def test_step10_success_is_not_step5_evidence(self):
        self.initial_pair()
        with self.assertRaises(ValueError):
            self.admit(comparison={'status': 'passed_exact_step10_continuation'})

    def test_failed_child_rejects_even_completed_scheduler(self):
        self.initial_pair()
        self.put(self.root / 'mis_foarl_teacher_staged/chunk-5-exit.json', {'returncode': 1})
        with self.assertRaisesRegex(ValueError, 'subprocess failed'):
            self.admit()

    def test_changed_live_checkpoint_rejected(self):
        self.initial_pair()
        self.put(self.root / 'mis_foarl_teacher_staged/chunk-5-complete.json', {'validated': -1})
        with self.assertRaisesRegex(ValueError, 'live checkpoint'):
            self.admit()

    def test_changed_source_or_method_or_job_rejected(self):
        for field, value in [('source', {}), ('method', 'classic_grpo'), ('diagnostic', False)]:
            with self.subTest(field=field):
                self.initial_pair()
                path = self.root / 'mis_foarl_teacher_staged/local_development_run_contract.json'
                contract = json.loads(path.read_text())
                self.put(path, {**contract, field: value})
                with self.assertRaises(ValueError):
                    self.admit()
        self.initial_pair()
        self.put(self.root / 'mis_foarl_teacher_staged/chunk-5-admission.json', {'job_id': '999', 'stop': 5})
        with self.assertRaisesRegex(ValueError, 'another job'):
            self.admit()

    def test_actual_scheduler_binding_not_only_local_receipt(self):
        self.initial_pair()
        name = 'mis_foarl_teacher_full'
        expected = m.command(self.source, self.sha, self.root, name, 'mis', 'foarl_teacher', 10)
        base = ['123', 'uag72kz', 'COMPLETED', '0:0', shlex.join(expected)]
        for index, value in [(0, '999'), (1, 'another'), (2, 'RUNNING'), (3, '1:0'), (4, 'sbatch other.slurm')]:
            with self.subTest(index=index):
                row = base.copy()
                row[index] = value
                with patch.object(m.subprocess, 'check_output', return_value='|'.join(row) + '\n'), self.assertRaises(ValueError):
                    m.require_completed_job(self.root, name, expected)

    def test_submit_respects_quota_and_explicit_limit(self):
        with patch.object(m, 'active_jobs', return_value=['old'] * 5), patch.object(m, 'submit') as submit:
            self.assertEqual(m.submit_initial(self.root, self.source, self.sha, 2), 2)
            self.assertEqual(submit.call_count, 2)
        with patch.object(m, 'active_jobs', return_value=['old'] * 10), patch.object(m, 'submit') as submit:
            self.assertEqual(m.submit_initial(self.root, self.source, self.sha, 5), 0)
            submit.assert_not_called()

    def test_unresolved_or_failed_attempt_cannot_be_retried(self):
        plan = self.protocol['plans'][0]
        self.put(self.root / (plan['key'] + '-attempt.json'), {'command': plan['command']})
        with patch.object(m, 'submit') as submit:
            with self.assertRaisesRegex(ValueError, 'Unresolved'):
                m.submit_initial(self.root, self.source, self.sha, 5)
            submit.assert_not_called()
        self.put(self.root / (plan['key'] + '-submission.json'), {'returncode': 1, 'stdout': ''})
        with patch.object(m, 'submit') as submit:
            with self.assertRaisesRegex(ValueError, 'Failed or ambiguous'):
                m.submit_initial(self.root, self.source, self.sha, 5)
            submit.assert_not_called()

    def test_symlink_receipt_rejected(self):
        target = self.root / 'target.json'
        self.put(target, {})
        linked = self.root / 'linked.json'
        linked.symlink_to(target)
        with self.assertRaises(ValueError):
            m.load_json(linked)

    def test_export_injection_rejected(self):
        for bad in ['a,b', 'a b', 'a\nb']:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.command(self.source, self.sha, self.root / bad, 'test', 'mis', 'classic_grpo', 5)


if __name__ == '__main__':
    unittest.main()
