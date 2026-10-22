"""One-shot admission/uncertainty regressions: no Slurm, remote writes or jobs."""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'analysis_outputs/grpo_incumbent_20260907_v1/submit_native_holdout_successor_a100_once_v1.py'
spec = importlib.util.spec_from_file_location('holdout_successor_submit_once_test', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.native = self.root / 'audit'
        self.campaign = self.root / 'performance'
        self.native.mkdir()
        (self.campaign / 'ledgers/events').mkdir(parents=True)
        (self.campaign / 'ledgers/evaluation_events').mkdir()
        (self.campaign / 'ledgers/controller.lock').touch()
        for name, value in [('NATIVE_ROOT', self.native), ('CAMPAIGN', self.campaign)]:
            patcher = patch.object(helper, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for native_name, local_name in [('run_native_holdout_audit_v1.py', 'run_native_holdout_successor_audit_v1.py'),
                                        ('audit_native_holdout_v1.slurm', 'audit_native_holdout_v4.slurm')]:
            (self.native / native_name).write_bytes((HELPER.parent / local_name).read_bytes())
        inputs = [helper.identity(self.native / name) for name in ('run_native_holdout_audit_v1.py', 'audit_native_holdout_v1.slurm')]
        manifest = {'files': {Path(row['path']).name: {key: row[key] for key in ('sha256', 'size_bytes')} for row in inputs}}
        (self.native / 'bundle_manifest.json').write_text(json.dumps(manifest))
        manifest_id = helper.identity(self.native / 'bundle_manifest.json')
        self.request = {'protocol': helper.PROTOCOL, 'native_root': str(self.native), 'performance_campaign': str(self.campaign),
                        'bundle_manifest': manifest_id, 'inputs': inputs, 'submission_helper': helper.identity(HELPER),
                        'observation_directory': str(self.native / 'submission_receipts')}
        self.request['command'] = ['sbatch', '--parsable', '--no-requeue', '--job-name=clasp-holdout-successor-audit-v1',
                                  '--comment=holdout-successor-audit-v1-' + manifest_id['sha256'][:12], '--chdir=' + str(self.native),
                                  '--export=HOLDOUT_AUDIT_BUNDLE=' + str(self.native) + ',HOLDOUT_AUDIT_MANIFEST_SHA256=' + manifest_id['sha256'],
                                  '--output=' + str(self.native / 'audit_%j.out'), '--error=' + str(self.native / 'audit_%j.err'),
                                  str(self.native / 'audit_native_holdout_v1.slurm')]
        self.request_path = self.native / 'request.json'
        self.make_bound_performance()
        self.request['retry_bridge'] = self.make_bound_retry_bridge()
        self.save_request()
        self.events = {'accepted_job_ids': [], 'unresolved_intents': 0}
        self.commands = []
        self.queue_responses = ['']
        self.accounting_response = ''
        self.sbatch_response = '990001;cluster\n'
        self.elapsed = 0
        self.alerts = []
        patcher = patch.object(helper, 'announce', side_effect=self.alerts.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def save_request(self):
        self.request_path.write_text(json.dumps(self.request))
        self.request_sha = hashlib.sha256(self.request_path.read_bytes()).hexdigest()

    def assert_lock_held(self):
        with (self.campaign / 'ledgers/controller.lock').open('r+') as handle:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def runner(self, command):
        self.assert_lock_held()
        self.commands.append(command)
        if command[0] == 'squeue':
            self.assertEqual(command, ['squeue', '-r', '-h', '-u', 'uag72kz', '-q', 'qos_gpu_a100-dev', '-o', '%i|%T'])
            return self.queue_responses.pop(0) if len(self.queue_responses) > 1 else self.queue_responses[0]
        if command[0] == 'sacct':
            return self.accounting_response
        if command[0] == 'sbatch':
            self.assertTrue((self.native / 'submission_receipts/submission_attempt.json').is_file())
            self.assertEqual(command, self.request['command'])
            if isinstance(self.sbatch_response, Exception):
                raise self.sbatch_response
            return self.sbatch_response
        raise AssertionError(command)

    def sleep(self, seconds):
        self.assert_lock_held()
        self.elapsed += seconds

    def invoke(self, **kwargs):
        return helper.submit_once(self.request_path, self.request_sha, runner=self.runner, event_reader=lambda: self.events,
                                  monotonic=lambda: self.elapsed, sleep=kwargs.pop('sleep', self.sleep), **kwargs)

    def assert_no_attempt(self):
        self.assertFalse((self.native / 'submission_receipts/submission_attempt.json').exists())
        self.assertFalse(any(command[0] == 'sbatch' for command in self.commands))

    def test_actual_reviewed_wrapper_request_and_single_submission_visibility(self):
        self.queue_responses = ['', '', '990001|PENDING\n']
        result = self.invoke()
        self.assertEqual(result['status'], 'accepted_job_visible_in_expanded_queue')
        self.assertEqual(result['job_id'], '990001')
        self.assertEqual(sum(command[0] == 'sbatch' for command in self.commands), 1)
        helper.signed(self.native / 'submission_receipts/submission_attempt.json')
        helper.signed(self.native / 'submission_receipts/submission_accepted.json')
        helper.signed(self.native / 'submission_receipts/submission_visibility.json')
        count = len(self.commands)
        with self.assertRaisesRegex(ValueError, 'prior attempt'):
            self.invoke()
        self.assertEqual(len(self.commands), count)

    def test_full_expanded_unrelated_array_waits_without_intent(self):
        self.queue_responses = [''.join(f'870000_{i}|PENDING\n' for i in range(10))]
        result = self.invoke()
        self.assertEqual(result['admission']['occupied_slots'], 10)
        self.assertEqual(result['status'], 'waiting_for_queue_slot_no_submission_attempt')
        self.assert_no_attempt()
        # Waiting is safe to call again and creates a new observation only.
        self.invoke()
        self.assertEqual(len(list((self.native / 'submission_receipts').glob('*.json'))), 2)

    def test_lagging_and_rejected_unresolved_intents_across_roles_reserve_slots(self):
        self.events = {'accepted_job_ids': ['101', '102', '103'], 'unresolved_intents': 2}
        self.queue_responses = [''.join(f'870000_{i}|PENDING\n' for i in range(5))]
        self.accounting_response = '101|RUNNING|0:0|Unknown\n102|FAILED|1:0|Unknown\n'
        result = self.invoke()
        self.assertEqual(result['admission']['occupied_slots'], 10)
        self.assertEqual(result['admission']['accepted_not_visible_and_not_conclusively_terminal'], ['101', '102', '103'])
        self.assert_no_attempt()

    def test_conclusive_terminal_before_visibility_releases_after_one_submit(self):
        self.events = {'accepted_job_ids': ['101'], 'unresolved_intents': 0}
        self.accounting_response = '101|COMPLETED|0:0|2026-09-07T09:00:00\n'
        original_runner = self.runner
        def runner(command):
            if command[0] == 'sacct' and '990001' in command:
                self.assert_lock_held()
                return '990001|FAILED|1:0|2026-09-07T10:01:00\n'
            return original_runner(command)
        with patch.object(self, 'runner', side_effect=runner):
            result = self.invoke()
        self.assertEqual(result['status'], 'accepted_job_conclusively_terminal')
        self.assertEqual(result['accounting']['990001']['state'], 'FAILED')
        self.assertEqual(helper.signed(self.native / 'submission_receipts/submission_attempt.json')['admission']['occupied_slots'], 0)

    def test_busy_shared_lock_observes_without_queries_or_intent(self):
        with (self.campaign / 'ledgers/controller.lock').open('r+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.invoke()
        self.assertEqual(result['status'], 'waiting_for_shared_lock_no_submission_attempt')
        self.assertEqual(self.commands, [])
        self.assert_no_attempt()

    def test_visibility_timeout_alert_is_durable_and_lock_stays_held(self):
        def sleep(seconds):
            self.sleep(60)
            if self.elapsed == 180:
                self.assertTrue(any(row['status'] == 'uncertain_submission_shared_lock_retained_no_retry' for row in self.alerts))
                self.queue_responses = ['990001|RUNNING\n']
        result = self.invoke(sleep=sleep)
        self.assertEqual(result['status'], 'accepted_job_visible_in_expanded_queue')
        self.assertEqual(sum(command[0] == 'sbatch' for command in self.commands), 1)
        self.assertEqual(len(list((self.native / 'submission_receipts').glob('observation_*.json'))), 1)

    def test_ambiguous_or_timeout_sbatch_never_retries_and_operator_interruption_is_recorded(self):
        for response in ['990001\n990002\n', subprocess.TimeoutExpired(['sbatch'], 30), '101\n']:
            with self.subTest(response=str(response)):
                self.events = {'accepted_job_ids': ['101'], 'unresolved_intents': 0}
                self.sbatch_response = response
                def interrupt(_):
                    self.assert_lock_held()
                    raise KeyboardInterrupt('explicit mocked operator intervention')
                with self.assertRaises(KeyboardInterrupt):
                    self.invoke(sleep=interrupt)
                self.assertEqual(sum(command[0] == 'sbatch' for command in self.commands), 1)
                self.assertTrue(any(row['status'] == 'uncertain_sbatch_response_shared_lock_retained_no_retry' for row in self.alerts))
                self.assertTrue(any(row['status'] == 'operator_interruption_releases_lock_with_unresolved_attempt_no_retry' for row in self.alerts))
                with self.assertRaisesRegex(ValueError, 'prior attempt'):
                    self.invoke()
                # Isolated temporary receipt cleanup is test-only, never a helper operation.
                for path in (self.native / 'submission_receipts').iterdir():
                    path.unlink()
                self.commands.clear()
                self.alerts.clear()

    def test_changed_request_or_script_or_helper_or_command_rejected_before_attempt(self):
        original = copy.deepcopy(self.request)
        for change in ('hash', 'script', 'helper', 'command', 'escape'):
            with self.subTest(change=change):
                self.request = copy.deepcopy(original)
                if change == 'hash':
                    self.request['extra'] = True
                    self.request_path.write_text(json.dumps(self.request))
                elif change == 'script':
                    script = self.native / 'audit_native_holdout_v1.slurm'
                    before = script.read_bytes()
                    script.write_bytes(before + b'# changed\n')
                else:
                    if change == 'helper':
                        self.request['submission_helper']['sha256'] = '0' * 64
                    elif change == 'command':
                        self.request['command'].insert(2, '--wrap=touch /tmp/unapproved')
                    else:
                        self.request['observation_directory'] = '/tmp/elsewhere'
                    self.save_request()
                with self.assertRaises(ValueError):
                    self.invoke()
                self.assert_no_attempt()
                if change == 'script':
                    script.write_bytes(before)
                self.request = copy.deepcopy(original)
                self.save_request()

    def test_script_mutation_during_admission_rejected_inside_lock(self):
        def reader():
            with (self.native / 'audit_native_holdout_v1.slurm').open('a') as handle:
                handle.write('# changed after initial validation\n')
            return self.events
        with self.assertRaisesRegex(ValueError, 'changed audit input'):
            helper.submit_once(self.request_path, self.request_sha, runner=self.runner, event_reader=reader)
        self.assert_no_attempt()

    def test_query_error_or_unexpanded_array_cannot_admit(self):
        for response in ['101_[0-9]|PENDING\n', '101_1|PENDING\n101_1|PENDING\n', '101.batch|RUNNING\n']:
            with self.subTest(response=response):
                self.queue_responses = [response]
                with self.assertRaises(ValueError):
                    self.invoke()
                self.assert_no_attempt()
        with patch.object(self, 'runner', side_effect=RuntimeError('squeue unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                self.invoke()
        self.assert_no_attempt()

    def test_unknown_or_missing_terminal_end_still_reserves(self):
        events = {'accepted_job_ids': ['101', '102', '103'], 'unresolved_intents': 1}
        rows = helper.parse_accounting('101|COMPLETED|0:0|Unknown\n102|FUTURE_UNKNOWN|0:0|2026-09-07T00:00:00\n103|CANCELLED by 12|0:15|2026-09-07T00:00:00\n', set(events['accepted_job_ids']))
        result = helper.occupancy(events, {'104_0': 'PENDING'}, rows)
        self.assertEqual(result['occupied_slots'], 4)
        self.assertEqual(result['accepted_not_visible_and_not_conclusively_terminal'], ['101', '102'])

    def make_bound_performance(self):
        if (self.campaign / 'ledgers/protocol.json').exists():
            return helper.signed(self.campaign / 'ledgers/protocol.json')
        source = self.campaign / 'source'
        (source / 'scripts').mkdir(parents=True)
        controller = source / 'scripts/grpo_incumbent_dev_campaign.py'
        controller.write_bytes((ROOT / 'scripts/grpo_incumbent_dev_campaign.py').read_bytes())
        manifest = source / 'SOURCE_MANIFEST.sha256'
        manifest.write_text(helper.CONTROLLER_SHA + '  scripts/grpo_incumbent_dev_campaign.py\n')
        patcher = patch.object(helper, 'SOURCE_SHA', helper.identity(manifest)['sha256'])
        patcher.start()
        self.addCleanup(patcher.stop)
        plans = [{'key': str(i), 'command': ['sbatch', str(i)]} for i in range(6)]
        protocol = {'output_root': str(self.campaign), 'controller': helper.identity(controller), 'source_manifest': helper.identity(manifest),
                    'planned_train_commands': plans, 'planned_eval_commands': plans}
        return helper.write_new(self.campaign / 'ledgers/protocol.json', protocol)

    def make_bound_retry_bridge(self):
        source = self.root / 'retry_source'
        (source / 'scripts').mkdir(parents=True)
        path = source / 'scripts/grpo_incumbent_dev_evaluation_retry.py'
        # Relocate only the campaign constant in the real reviewed reader. This
        # fixture never calls its native/controller/submission entrypoints.
        original = (ROOT / 'scripts/grpo_incumbent_dev_evaluation_retry.py').read_text()
        native = '/lustre/fsn1/projects/rech/vas/uag72kz/grpo_incumbent_foarl_s43_b9_dev_20260907_v1'
        self.assertEqual(original.count("CAMPAIGN = Path('" + native + "')"), 1)
        original = original.replace("CAMPAIGN = Path('" + native + "')", 'CAMPAIGN = Path(' + repr(str(self.campaign)) + ')')
        # Executing any broader bridge API fails, including during import.
        path.write_text(original + '\n' +
                        'def forbidden(*args, **kwargs):\n    raise AssertionError("broader bridge API executed")\n' +
                        'original = declaration = supervisor_tick = launch = worker = forbidden\n')
        manifest = source / 'SOURCE_MANIFEST.sha256'
        manifest.write_text(helper.identity(path)['sha256'] + '  scripts/grpo_incumbent_dev_evaluation_retry.py\n')
        spec = importlib.util.spec_from_file_location('fixture_retry_reader', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.retry_keys = list(module.FAILED)
        declaration = self.campaign / 'ledgers/development_evaluation_retry_v1.json'
        events = self.campaign / 'ledgers/evaluation_retry_events_v1'
        events.mkdir()
        value = {'protocol': 'incumbent-finite-pre-model-evaluation-retry-v1',
                 'campaign': str(self.campaign), 'original_protocol': helper.identity(self.campaign / 'ledgers/protocol.json'),
                 'management_source': helper.identity(manifest), 'management': {'helper': helper.identity(path)},
                 'fixed_exception': module.FAILED, 'failed_attempts': {key: {} for key in module.FAILED},
                 'planned_retries': [{'key': key, 'command': ['sbatch', '--retry-key=' + key]} for key in module.FAILED],
                 'exactly_one_successor_per_key': True, 'scientific_arguments_changed': False, 'original_events_preserved': True}
        helper.write_new(declaration, value)
        return {'helper': helper.identity(path), 'source_manifest': helper.identity(manifest),
                'declaration': helper.identity(declaration), 'event_directory': str(events)}

    def retry_event(self, key, kind, *, job=None, command=None):
        binding = self.request['retry_bridge']
        value = {'key': key, 'kind': kind, 'command': command or ['sbatch', '--retry-key=' + key],
                 'transaction': key, 'declaration': binding['declaration']}
        if job is not None:
            value['job_id'] = job
        value['event_sha256'] = helper.digest(value)
        path = Path(binding['event_directory']) / (key.replace('/', '_') + '_' + kind + '.json')
        path.write_text(json.dumps(value))
        return path

    def event(self, protocol, directory, key, kind, *, job=None, command=None):
        value = {'key': str(key), 'kind': kind, 'command': command or ['sbatch', str(key)], 'transaction': str(key),
                 'protocol_sha256': protocol['receipt_sha256']}
        if job is not None:
            value['job_id'] = job
        value['event_sha256'] = helper.digest(value)
        (self.campaign / 'ledgers' / directory / f'{key}_{kind}.json').write_text(json.dumps(value))

    def test_actual_pinned_controller_event_reader_both_roles_and_rejected_reservation(self):
        protocol = self.make_bound_performance()
        for directory, key, kind, job in [('events', 0, 'submitted', '101'), ('evaluation_events', 1, 'submitted', '102'),
                                          ('evaluation_events', 2, 'rejected', None), ('events', 3, None, None)]:
            self.event(protocol, directory, key, 'intent')
            if kind:
                self.event(protocol, directory, key, kind, job=job)
        events = helper.performance_events(self.request['retry_bridge'])
        self.assertEqual(events['accepted_job_ids'], ['101', '102'])
        self.assertEqual(events['unresolved_intents'], 2)
        self.event(protocol, 'evaluation_events', 1, 'submitted', job='101')
        with self.assertRaisesRegex(ValueError, 'multiple performance'):
            helper.performance_events(self.request['retry_bridge'])

    def test_signed_but_changed_event_command_is_rejected(self):
        protocol = self.make_bound_performance()
        self.event(protocol, 'events', 0, 'intent', command=['sbatch', 'unreviewed'])
        with self.assertRaisesRegex(ValueError, 'event declaration mismatch'):
            helper.performance_events(self.request['retry_bridge'])

    def test_controller_and_protocol_source_identity_are_checked_before_import(self):
        protocol = self.make_bound_performance()
        Path(protocol['controller']['path']).write_text('raise RuntimeError("must not execute")\n')
        with self.assertRaisesRegex(ValueError, 'controller changed'):
            helper.performance_events(self.request['retry_bridge'])

    def test_actual_original_and_retry_readers_reserve_invisible_and_unresolved_under_lock(self):
        protocol = self.make_bound_performance()
        self.event(protocol, 'events', 0, 'intent')
        self.event(protocol, 'events', 0, 'submitted', job='101')
        self.retry_event(self.retry_keys[0], 'intent')
        self.retry_event(self.retry_keys[0], 'submitted', job='201')
        self.retry_event(self.retry_keys[1], 'intent')
        self.retry_event(self.retry_keys[1], 'rejected')
        self.retry_event(self.retry_keys[2], 'intent')
        self.queue_responses = [''.join(f'870000_{i}|PENDING\n' for i in range(6))]
        self.accounting_response = '101|RUNNING|0:0|Unknown\n'
        with patch.object(helper.subprocess, 'check_output', side_effect=AssertionError('bridge queried native scheduler')):
            result = helper.submit_once(self.request_path, self.request_sha, runner=self.runner)
        self.assertEqual(result['status'], 'waiting_for_queue_slot_no_submission_attempt')
        self.assertEqual(result['admission']['occupied_slots'], 10)
        self.assertEqual(result['admission']['accepted_not_visible_and_not_conclusively_terminal'], ['101', '201'])
        self.assertEqual(result['performance_events']['retry_bridge']['accepted_job_ids'], ['201'])
        self.assertEqual(result['performance_events']['retry_bridge']['unresolved_intents'], 2)
        self.assertEqual(len(result['performance_events']['retry_bridge']['event_files']), 5)
        self.assert_no_attempt()

    def test_duplicate_job_across_original_and_retry_is_rejected_before_queue(self):
        protocol = self.make_bound_performance()
        self.event(protocol, 'evaluation_events', 0, 'intent')
        self.event(protocol, 'evaluation_events', 0, 'submitted', job='201')
        self.retry_event(self.retry_keys[0], 'intent')
        self.retry_event(self.retry_keys[0], 'submitted', job='201')
        with self.assertRaisesRegex(ValueError, 'across original and retry'):
            helper.submit_once(self.request_path, self.request_sha, runner=self.runner)
        self.assertEqual(self.commands, [])
        self.assert_no_attempt()

    def test_retry_initialization_and_pinned_bytes_required_before_attempt(self):
        binding = self.request['retry_bridge']
        for name in ('helper', 'source_manifest', 'declaration'):
            with self.subTest(name=name):
                path = Path(binding[name]['path'])
                previous = path.read_bytes()
                path.write_bytes(previous + b'\n')
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    self.invoke()
                path.write_bytes(previous)
                self.assert_no_attempt()
        declaration = Path(binding['declaration']['path'])
        previous = declaration.read_bytes()
        declaration.unlink()
        with self.assertRaisesRegex(ValueError, 'expected regular'):
            self.invoke()
        declaration.write_bytes(previous)
        Path(binding['event_directory']).rmdir()
        with self.assertRaisesRegex(ValueError, 'initialized real retry'):
            self.invoke()
        self.assert_no_attempt()

    def test_resigned_scope_protocol_and_source_membership_mutations_rejected(self):
        original_request = copy.deepcopy(self.request)
        binding = self.request['retry_bridge']
        declaration = Path(binding['declaration']['path'])
        original_bytes = declaration.read_bytes()
        manifest = Path(binding['source_manifest']['path'])
        original_manifest = manifest.read_bytes()
        for change in ('scope', 'keys', 'original_protocol', 'source_membership'):
            with self.subTest(change=change):
                value = helper.read_json(declaration)
                value.pop('receipt_sha256')
                if change == 'scope':
                    value['scientific_arguments_changed'] = True
                elif change == 'keys':
                    value['planned_retries'] = value['planned_retries'][:2]
                elif change == 'original_protocol':
                    value['original_protocol']['sha256'] = '0' * 64
                else:
                    manifest.write_text('0' * 64 + '  scripts/grpo_incumbent_dev_evaluation_retry.py\n')
                    value['management_source'] = helper.identity(manifest)
                    self.request['retry_bridge']['source_manifest'] = helper.identity(manifest)
                declaration.unlink()
                helper.write_new(declaration, value)
                self.request['retry_bridge']['declaration'] = helper.identity(declaration)
                self.save_request()
                with self.assertRaises(ValueError):
                    helper.submit_once(self.request_path, self.request_sha, runner=self.runner)
                self.assert_no_attempt()
                self.request = copy.deepcopy(original_request)
                declaration.write_bytes(original_bytes)
                manifest.write_bytes(original_manifest)
                self.save_request()

    def test_retry_event_rejects_signed_wrong_command_and_orphan_submission(self):
        for change in ('command', 'orphan'):
            with self.subTest(change=change):
                key = self.retry_keys[0]
                if change == 'command':
                    path = self.retry_event(key, 'intent', command=['sbatch', 'unreviewed'])
                else:
                    path = self.retry_event(key, 'submitted', job='201')
                with self.assertRaises(ValueError):
                    helper.submit_once(self.request_path, self.request_sha, runner=self.runner)
                self.assert_no_attempt()
                path.unlink()

    def test_subprocess_removes_sbatch_overrides_and_uses_no_shell(self):
        completed = subprocess.CompletedProcess(['sbatch'], 0, '990001\n', '')
        with patch.dict(helper.os.environ, {'SBATCH_ACCOUNT': 'other', 'SBATCH_GRES': 'gpu:4'}), patch.object(helper.subprocess, 'run', return_value=completed) as run:
            self.assertEqual(helper.run_command(['sbatch', '--parsable']), '990001\n')
        kwargs = run.call_args.kwargs
        self.assertFalse(any(key.startswith('SBATCH_') for key in kwargs['env']))
        self.assertNotIn('shell', kwargs)
        self.assertEqual(kwargs['timeout'], 30)


if __name__ == '__main__':
    unittest.main()
