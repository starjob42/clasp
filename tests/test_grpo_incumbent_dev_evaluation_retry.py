"""Model-free tests of finite retry admission, shared reservations, argv and raw binding."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


retry = load('finite_retry_tests', ROOT / 'scripts/grpo_incumbent_dev_evaluation_retry.py')
old = load('finite_retry_old_parser', ROOT / 'scripts/grpo_incumbent_dev_campaign.py')


class FailedAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.output = self.root / 'eval'; self.output.mkdir()
        self.key, self.job = 'cvrp/grpo_z', '1847633'
        (self.output / 'eval/grpo_z_incumbent_seed271828/cvrp').mkdir(parents=True)
        launch = {'slurm_job_id': self.job, 'campaign_protocol_sha256': 'original', 'source_manifest': {'sha256': 'science'}, 'task': 'cvrp', 'method': 'grpo_z'}
        retry.create(self.output / 'launch_receipt.json', launch, 'launch_sha256')
        self.stdout = self.root / (self.job + '.out'); self.stdout.write_bytes((self.output / 'launch_receipt.json').read_bytes())
        self.stderr = self.root / (self.job + '.err'); self.stderr.write_text('missing logs')
        self.plan = {'key': self.key, 'output': str(self.output), 'stdout': str(self.root / '%j.out'), 'command': ['--error=' + str(self.root / '%j.err')]}
        self.protocol = {'receipt_sha256': 'original', 'source_manifest': {'sha256': 'science'}, 'planned_eval_commands': [self.plan]}
        self.control = SimpleNamespace(submitted_job=lambda *args, **kwargs: self.job)
        self.row = {'job_id': self.job, 'state': 'FAILED', 'exit_code': '1:0', 'elapsed_seconds': 11, 'stdout': str(self.stdout)}
        expected = {self.key: {'job_id': self.job, 'elapsed_seconds': 11, 'launch_sha256': retry.identity(self.stdout)['sha256'], 'launch_size': self.stdout.stat().st_size}}
        self.addCleanup(patch.stopall)
        patch.object(retry, 'FAILED', expected).start()
        patch.object(retry, 'STDERR_SHA', retry.identity(self.stderr)['sha256']).start()
        # Native error size is fixed to150; fixture uses the same size and digest.
        self.stderr.write_text('missing logs'.ljust(150))
        retry.STDERR_SHA = retry.identity(self.stderr)['sha256']

    def check(self):
        return retry.validate_failed_attempt(self.key, self.control, self.protocol, self.row)

    def test_exact_pre_model_failure_is_admitted(self):
        result = self.check()
        self.assertTrue(result['no_model_outputs'])
        self.assertEqual(result['output_file_paths'], ['launch_receipt.json'])

    def test_wrong_job_exit_duration_or_success_never_qualifies(self):
        for field, value in [('job_id', '999'), ('state', 'COMPLETED'), ('exit_code', '0:0'), ('elapsed_seconds', 12)]:
            with self.subTest(field=field):
                row = dict(self.row); row[field] = value
                with self.assertRaisesRegex(ValueError, 'scheduler'):
                    retry.validate_failed_attempt(self.key, self.control, self.protocol, row)

    def test_any_model_output_or_changed_log_rejects(self):
        (self.output / 'run_details__x.jsonl').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'model outputs'):
            self.check()
        (self.output / 'run_details__x.jsonl').unlink()
        self.stderr.write_text('different'.ljust(150))
        with self.assertRaisesRegex(ValueError, 'missing-logs error'):
            self.check()

    def test_unlisted_key_and_mutated_launch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            retry.validate_failed_attempt('mis/foarl_f638_dev', self.control, self.protocol, self.row)
        self.stdout.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'launch/stdout'):
            self.check()


class FixedEvidenceTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'analysis_outputs/grpo_incumbent_20260907_v1/all_pre_model_eval_failure_proofs_v1.json').exists(), 'retained local native evidence is outside the code-only snapshot')
    def test_compiled_native_exception_matches_three_retained_receipts(self):
        proof = json.loads((ROOT / 'analysis_outputs/grpo_incumbent_20260907_v1/all_pre_model_eval_failure_proofs_v1.json').read_text())
        self.assertEqual(set(retry.FAILED), {'cvrp/grpo_z', 'cvrp/grpo_verified', 'mis/grpo_z'})
        self.assertEqual(sum(v['elapsed_seconds'] for v in retry.FAILED.values()), 32)
        for key, expected in retry.FAILED.items():
            actual = proof['jobs'][key.replace('/', '_')]
            self.assertEqual(actual['job_id'], expected['job_id'])
            self.assertEqual(actual['launch']['sha256'], expected['launch_sha256'])
            self.assertEqual(actual['launch']['size_bytes'], expected['launch_size'])
            self.assertEqual(actual['stdout']['sha256'], expected['launch_sha256'])
            self.assertEqual(actual['stderr']['sha256'], retry.STDERR_SHA)
            self.assertEqual(actual['output_file_paths'], ['launch_receipt.json'])

    def test_actual_original_shell_environment_matches_new_generic_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / 'source'; runtime = root / 'runtime'; campaign = root / 'campaign'
            (source / 'scripts').mkdir(parents=True); (runtime / 'bin').mkdir(parents=True); (campaign / 'ledgers').mkdir(parents=True)
            capture = root / 'capture.json'
            python = runtime / 'bin/python'
            python.write_text('#!/bin/bash\nif [[ "$1" == "-" ]]; then exec ' + sys.executable + ' "$@"; fi\nexit 0\n'); python.chmod(0o755)
            plan = {'key': 'cvrp/grpo_z', 'task': 'cvrp', 'method': 'grpo_z', 'checkpoint': str(root / 'checkpoint-100'), 'output': str(root / 'evaluation')}
            protocol = {'runtime': str(runtime), 'evaluation': {'data_root': str(root / 'data')}, 'planned_eval_commands': [plan]}
            (campaign / 'ledgers/protocol.json').write_text(json.dumps(protocol))
            with patch.object(retry, 'TRAINING_SOURCE', source):
                expected = retry.generic_environment(plan, protocol)
            code = 'import os,json;from pathlib import Path;Path(' + repr(str(capture)) + ').write_text(json.dumps({k:os.environ.get(k) for k in ' + repr(list(expected)) + '}))'
            import shlex
            (source / 'scripts/evaluate_a100.slurm').write_text('#!/bin/bash\n' + shlex.quote(sys.executable) + ' -c ' + shlex.quote(code) + '\n')
            env = {**os.environ, 'INCUMBENT_CAMPAIGN': str(campaign), 'INCUMBENT_SOURCE': str(source), 'INCUMBENT_RUNTIME': str(runtime),
                   'INCUMBENT_OUTPUT': plan['output'], 'INCUMBENT_TASK': 'cvrp', 'INCUMBENT_METHOD': 'grpo_z', 'CLASP_FAKE_REPAIR': '1'}
            subprocess.run(['bash', str(ROOT / 'scripts/evaluate_grpo_incumbent_dev_a100.slurm')], env=env, check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(capture.read_text()), expected)

    def test_retry_wrapper_preserves_original_gpu_resource_directives(self):
        original = (ROOT / 'scripts/evaluate_grpo_incumbent_dev_a100.slurm').read_text().splitlines()
        actual = (ROOT / retry.WRAPPER).read_text().splitlines()
        prefixes = ('#SBATCH --account', '#SBATCH --constraint', '#SBATCH --nodes', '#SBATCH --gres', '#SBATCH --ntasks', '#SBATCH --cpus', '#SBATCH --hint', '#SBATCH --time', '#SBATCH --qos')
        self.assertEqual([s for s in actual if s.startswith(prefixes)], [s for s in original if s.startswith(prefixes)])


class ReservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); (self.root / 'events').mkdir(); (self.root / 'terminal').mkdir()
        self.decl = self.root / 'declaration.json'; self.decl.write_text('{}')
        self.addCleanup(patch.stopall)
        patch.object(retry, 'EVENTS', self.root / 'events').start(); patch.object(retry, 'TERMINALS', self.root / 'terminal').start()
        patch.object(retry, 'DECLARATION', self.decl).start()
        self.p = {'planned_retries': [{'key': key, 'command': ['sbatch', key], 'output': str(self.root / key.replace('/', '_')), 'stdout': str(self.root / (key.replace('/', '_') + '_%j.out'))} for key in retry.FAILED]}
        self.accepted = []
        self.control = SimpleNamespace(read_events=lambda *a, **k: {}, scheduler_rows=old.scheduler_rows,
                      expanded_queue=old.expanded_queue, USER='uag72kz', QOS='qos_gpu_a100-dev', training_endpoint=lambda *a, **k: {})

    def runner(self, command):
        if command[0] == 'squeue':
            return ''.join(str(700 + i) + '|PENDING\n' for i in range(9))
        if command[0] == 'sacct':
            return ''  # Deliberately invisible accepted submissions must reserve slots.
        if command[0] == 'sbatch':
            self.accepted.append(command); return str(900 + len(self.accepted))
        raise AssertionError(command)

    def test_accepted_but_invisible_retry_reserves_tenth_slot_without_duplicate(self):
        first = retry.retry_tick(self.p, self.control, {}, submit=True, record=True, max_new_jobs=5, runner=self.runner, visibility=lambda *args: None)
        self.assertEqual(len(self.accepted), 1)
        self.assertEqual(first['newly_submitted'], 1)
        second = retry.retry_tick(self.p, self.control, {}, submit=True, record=True, max_new_jobs=5, runner=self.runner, visibility=lambda *args: None)
        self.assertEqual(len(self.accepted), 1)
        self.assertEqual(second['observations'][0]['state'], 'awaiting_accounting_no_retry')

    def test_unresolved_intent_blocks_slot_and_never_retries(self):
        plan = self.p['planned_retries'][0]
        retry.create(retry.EVENTS / 'intent.json', {'key': plan['key'], 'kind': 'intent', 'declaration': retry.identity(self.decl),
                     'command': plan['command'], 'transaction': 'lost-ack'}, 'event_sha256')
        result = retry.retry_tick(self.p, self.control, {}, submit=True, record=True, max_new_jobs=5, runner=self.runner, visibility=lambda *args: None)
        self.assertEqual(result['observations'][0]['state'], 'prior_attempt_no_retry')
        self.assertEqual(self.accepted, [])

    def test_ambiguous_sbatch_is_durable_rejection_not_retry(self):
        def ambiguous(command):
            if command[0] == 'sbatch':
                return 'Submitted batch job 123'
            return self.runner(command)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            retry.retry_tick(self.p, self.control, {}, submit=True, record=True, max_new_jobs=5, runner=ambiguous, visibility=lambda *args: None)
        events = retry.read_events(self.p)[self.p['planned_retries'][0]['key']]
        self.assertEqual({e['kind'] for e in events}, {'intent', 'rejected'})
        retry.retry_tick(self.p, self.control, {}, submit=True, record=True, max_new_jobs=5, runner=self.runner, visibility=lambda *args: None)
        self.assertEqual(self.accepted, [])

    def test_duplicate_or_rebound_event_is_rejected(self):
        plan = self.p['planned_retries'][0]
        base = {'key': plan['key'], 'kind': 'intent', 'declaration': retry.identity(self.decl), 'command': plan['command'], 'transaction': 'one'}
        retry.create(retry.EVENTS / 'first.json', base, 'event_sha256')
        retry.create(retry.EVENTS / 'second.json', {**base, 'transaction': 'two'}, 'event_sha256')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            retry.read_events(self.p)

    def test_original_roles_are_capped_by_combined_reservations(self):
        campaign = self.root / 'campaign'; (campaign / 'ledgers').mkdir(parents=True)
        calls = []
        def reconcile(*a, **kw):
            calls.append((kw['role'], kw['max_new_jobs']))
            rows = [{'state': 'submitted', 'key': 'new'}] if kw['role'] == 'train' else []
            return {'observations': rows, 'all_training_verified': False}
        self.control.reconcile = reconcile
        with patch.object(retry, 'CAMPAIGN', campaign), patch.object(retry, 'declaration', return_value=(self.p, self.control, {}, self.root)), \
                patch.object(retry, 'all_reservations', side_effect=[(9, {}, {}), (10, {}, {}), (10, {}, {}), (10, {}, {})]), \
                patch.object(retry, 'retry_tick', return_value={'observations': [], 'newly_submitted': 0}) as tick:
            retry.supervisor_tick('pin', submit=True, record=False)
        self.assertEqual(calls, [('train', 1), ('eval', 0)])
        self.assertEqual(tick.call_args.kwargs['max_new_jobs'], 4)


class VisibilityAndAllocationTests(unittest.TestCase):
    def test_unknown_state_still_reserves_an_absent_queue_job(self):
        control = SimpleNamespace(read_events=lambda *a, **k: {'cvrp/grpo_z': [{'kind': 'submitted', 'job_id': '1999'}]} if k['role'] == 'eval' else {},
                                  scheduler_rows=old.scheduler_rows, expanded_queue=old.expanded_queue, USER='uag72kz', QOS='qos_gpu_a100-dev')
        def runner(command):
            return '1999|REQUEUE_FED|0:0|/log|0\n' if command[0] == 'sacct' else ''
        with patch.object(retry, 'read_events', return_value={}):
            occupied, _, _ = retry.all_reservations({}, control, {}, runner)
        self.assertEqual(occupied, 1)

    def test_unexpanded_array_is_rejected_instead_of_counted_as_one(self):
        control = SimpleNamespace(read_events=lambda *a, **k: {}, scheduler_rows=old.scheduler_rows,
                                  expanded_queue=old.expanded_queue, USER='uag72kz', QOS='qos_gpu_a100-dev')
        with patch.object(retry, 'read_events', return_value={}):
            with self.assertRaisesRegex(ValueError, 'expanded'):
                retry.all_reservations({}, control, {}, lambda command: '879217_[0-4]|PENDING\n')

    def test_visible_barrier_polls_read_only_then_returns_under_callers_lock(self):
        control = SimpleNamespace(scheduler_rows=old.scheduler_rows, expanded_queue=old.expanded_queue, USER='uag72kz', QOS='qos_gpu_a100-dev')
        calls, sleeps = [], []
        def runner(command):
            calls.append(command)
            if command[0] == 'squeue':
                return '' if len([x for x in calls if x[0] == 'squeue']) == 1 else '999|PENDING\n'
            return ''
        with patch.object(retry, 'identity', return_value={'sha256': 'p'}), patch.object(retry, 'create', side_effect=lambda path, value: value):
            result = retry.await_visibility({}, control, runner, '999', 'cvrp/grpo_z', sleep=lambda n: sleeps.append(n), monotonic=lambda: 0)
        self.assertEqual(result['status'], 'accepted_retry_visible_in_expanded_queue')
        self.assertEqual(sleeps, [5])
        self.assertFalse(any(x[0] == 'sbatch' for x in calls))

    def test_ambiguous_unknown_job_cannot_be_released_by_some_other_visible_job(self):
        control = SimpleNamespace(expanded_queue=old.expanded_queue, USER='uag72kz', QOS='qos_gpu_a100-dev')
        def interrupted(_):
            raise RuntimeError('operator interruption required')
        with self.assertRaisesRegex(RuntimeError, 'operator interruption'):
            retry.await_visibility({}, control, lambda cmd: '999|PENDING\n', None, 'cvrp/grpo_z', sleep=interrupted, monotonic=lambda: 0)

    def test_scheduler_timeout_produces_durable_uncertainty_alert_without_resubmission(self):
        control = SimpleNamespace(expanded_queue=old.expanded_queue, USER='uag72kz', QOS='qos_gpu_a100-dev')
        calls, alerts = [], []
        def runner(command):
            calls.append(command)
            raise subprocess.TimeoutExpired(command, 30)
        times = iter([0, 121])
        def sleep(_):
            raise RuntimeError('end bounded synthetic observation')
        with patch.object(retry, 'identity', return_value={'sha256': 'declaration'}), \
                patch.object(retry, 'create', side_effect=lambda path, value: alerts.append(value) or value):
            with self.assertRaisesRegex(RuntimeError, 'bounded synthetic'):
                retry.await_visibility({}, control, runner, '999', 'cvrp/grpo_z', sleep=sleep, monotonic=lambda: next(times))
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['status'], 'uncertain_retry_attempt_shared_lock_retained_no_resubmission')
        self.assertIn('timed out', alerts[0]['evidence']['query_error'])
        self.assertFalse(any(cmd[0] == 'sbatch' for cmd in calls))

    def test_default_native_command_runner_has_bounded_timeout(self):
        with patch.object(retry.subprocess, 'check_output', return_value='ok') as run:
            self.assertEqual(retry.command_output(['squeue']), 'ok')
        self.assertEqual(run.call_args.kwargs, {'text': True, 'timeout': 30})

    def test_failed_allocation_preserves_exact32_gpu_seconds_and_cpu_billing_cost(self):
        rows = []
        for item in retry.FAILED.values():
            seconds = item['elapsed_seconds']
            rows.append(f"{item['job_id']}|FAILED|1:0|2026-09-07T10:00:00|2026-09-07T10:00:{seconds:02d}|{seconds}|120|billing=1,cpu=16,gres/gpu=1,mem=60000M,node=1|qos_gpu_a100-dev|vas@a100|gpu_p5")
        raw = '\n'.join(rows)
        def runner(command, text=True):
            self.assertEqual(command[:3], ['env', 'TZ=UTC', 'SLURM_TIME_FORMAT=standard'])
            return raw
        result = retry.failed_allocation_records(SimpleNamespace(QOS='qos_gpu_a100-dev'), runner=runner)
        self.assertAlmostEqual(result['allocated_gpu_hours'], 32 / 3600)
        self.assertAlmostEqual(result['allocated_cpu_hours'], 512 / 3600)
        self.assertAlmostEqual(result['slurm_billing_unit_hours'], 32 / 3600)
        for text in [raw.replace('|FAILED|', '|COMPLETED|', 1), raw.replace('gres/gpu=1', 'gres/gpu=4', 1),
                     raw.replace('|120|', '|25|', 1), raw.replace('cpu=16', 'cpu=8', 1)]:
            with self.assertRaises(ValueError):
                retry.failed_allocation_records(SimpleNamespace(QOS='qos_gpu_a100-dev'), runner=lambda *a, **k: text)


class ExecutionBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name); self.output = root / 'retry'; self.output.mkdir()
        self.decl = root / 'declaration.json'; self.decl.write_text('{}')
        self.addCleanup(patch.stopall); patch.object(retry, 'DECLARATION', self.decl).start()
        self.key, self.job = 'cvrp/grpo_z', '999'
        original = {'slurm_job_id': '1847633', 'management': {'helper': 'old'}, 'generation': {'seed': 271828}, 'checkpoint': 'original'}
        old_path = root / 'old_launch.json'; retry.create(old_path, original, 'launch_sha256')
        self.p = {'failed_attempts': {self.key: {'files': {'launch': retry.identity(old_path)}}}, 'management': {'helper': 'new'}}
        self.plan = {'key': self.key, 'output': str(self.output), 'generic_command': ['/bin/bash', '/frozen/evaluate'], 'generic_environment': {'SEED': '271828'}}
        self.launch = {**original, 'slurm_job_id': self.job, 'management': self.p['management'], 'retry_declaration': retry.identity(self.decl), 'replaced_pre_model_job_id': '1847633'}
        self.execution = {'retry_declaration': retry.identity(self.decl), 'slurm_job_id': self.job, 'returncode': 0,
                          'generic_command': self.plan['generic_command'], 'generic_environment': self.plan['generic_environment']}

    def check(self):
        for name in ['launch_receipt.json', 'execution_receipt.json']:
            path = self.output / name
            if path.exists(): path.unlink()
        launch = retry.create(self.output / 'launch_receipt.json', self.launch, 'launch_sha256')
        execution = {**self.execution, 'launch_sha256': launch['launch_sha256']}
        retry.create(self.output / 'execution_receipt.json', execution)
        return retry.verify_execution(self.plan, self.p, self.job)

    def test_declared_successor_preserves_every_scientific_launch_field(self):
        self.check()

    def test_job_seed_checkpoint_environment_and_failed_process_reject(self):
        mutations = [('launch', 'slurm_job_id', '1000'), ('launch', 'generation', {'seed': 42}), ('launch', 'checkpoint', 'other'),
                     ('execution', 'generic_environment', {'SEED': '42'}), ('execution', 'returncode', 1)]
        for obj, key, value in mutations:
            with self.subTest(obj=obj, key=key):
                target = self.launch if obj == 'launch' else self.execution
                previous = copy.deepcopy(target[key]); target[key] = value
                with self.assertRaises(ValueError): self.check()
                target[key] = previous


class WorkerAndRawAuditIntegrationTests(unittest.TestCase):
    def test_actual_worker_command_and_scrubbed_environment_are_captured(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root / 'output'; output.mkdir()
            declaration = root / 'declaration.json'; declaration.write_text('{}')
            plan = {'key': 'cvrp/grpo_z', 'task': 'cvrp', 'method': 'grpo_z', 'checkpoint': '/original/checkpoint-100', 'output': str(output)}
            protocol = {'runtime': '/original/runtime', 'evaluation': {'data_root': '/original/data'}}
            plan.update(generic_command=['/bin/bash', str(retry.TRAINING_SOURCE / 'scripts/evaluate_a100.slurm')],
                        generic_environment=retry.generic_environment(plan, protocol))
            p = {'planned_retries': [plan]}
            captured = {}
            def run(command, **kwargs):
                captured.update(command=command, **kwargs)
                return SimpleNamespace(returncode=0)
            with patch.object(retry, 'DECLARATION', declaration), patch.object(retry, 'declaration', return_value=(p, None, protocol, root)), \
                    patch.object(retry, 'launch', return_value={'slurm_job_id': '999', 'launch_sha256': 'launch'}), \
                    patch.object(retry.subprocess, 'run', side_effect=run), \
                    patch.object(retry, 'validate_result', return_value={'index_payload': {'index_sha256': 'discarded', 'test': 'result'}}), \
                    patch.dict(os.environ, {'PYTHONHOME': '/wrong', 'PYTHONPATH': '/wrong', 'CLASP_ASSISTANCE': '1', 'GENERATION_REPAIR': '1', 'KEEP_ME': 'kept'}, clear=True):
                retry.worker(plan['key'], root / 'spool')
            self.assertEqual(captured['command'], plan['generic_command'])
            self.assertTrue(all(captured['env'][key] == value for key, value in plan['generic_environment'].items()))
            self.assertNotIn('PYTHONHOME', captured['env']); self.assertNotIn('CLASP_ASSISTANCE', captured['env'])
            self.assertNotIn('GENERATION_REPAIR', captured['env']); self.assertEqual(captured['env']['KEEP_ME'], 'kept')
            receipt = retry.checked(output / 'execution_receipt.json')
            self.assertEqual(receipt['generic_environment'], plan['generic_environment'])
            self.assertEqual(receipt['generic_command'], captured['command'])
            self.assertEqual(receipt['returncode'], 0)

    def test_validate_result_reaches_retained_native_cross_binding_auditor(self):
        fixtures = load('retry_real_cross_binding_fixture', ROOT / 'tests/test_grpo_incumbent_dev_campaign.py')
        fixture = fixtures.CrossBindingTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        plan, launch, config, index, terminal = fixture.args
        kw = fixture.kwargs; source = kw['source']
        (Path(plan['output']) / 'execution_receipt.json').write_text('{}')
        p = {'planned_retries': [plan], 'benchmark': kw['benchmark'], 'management': kw['evaluation']['management']}
        protocol = {**kw['training'], 'benchmark': kw['benchmark'], 'generation': kw['evaluation']['generation'], 'receipt_sha256': 'original'}
        control = SimpleNamespace(training_endpoint=lambda *a, **k: terminal, audit_evaluation_arm=old.audit_evaluation_arm)
        fresh = SimpleNamespace(index=lambda output: index, validate_benchmark=lambda data: kw['benchmark'])
        artifact = SimpleNamespace(artifact_type='native_raw', records=kw['records'])
        strict = SimpleNamespace(_load_artifact=lambda *a, **k: artifact)
        def module(name, path):
            return fresh if Path(path).name == 'clasp_grpo_fresh_dev_artifacts.py' else strict
        declaration = source / 'declaration.json'; declaration.write_text('{}')
        with patch.object(retry, 'DECLARATION', declaration), patch.object(retry, 'TRAINING_SOURCE', source), \
                patch.object(retry, 'original', return_value=(control, protocol, source)), \
                patch.object(retry, 'submitted', return_value=kw['job_id']), \
                patch.object(retry, 'verify_execution', return_value=(launch, {})), \
                patch.object(retry, 'module', side_effect=module), patch.object(retry, 'checked', return_value=config):
            result = retry.validate_result(p, plan['key'], kw['job_id'])
            self.assertEqual(result['cross_bindings']['instances'], 100)
            config['resolved_model_path'] = plan['training_root']
            with self.assertRaisesRegex(ValueError, 'raw run model'):
                retry.validate_result(p, plan['key'], kw['job_id'])
            config['resolved_model_path'] = plan['checkpoint']
            config['experiment_provenance']['data']['files'][0]['sha256'] = 'different-data'
            with self.assertRaisesRegex(ValueError, 'dataset hash'):
                retry.validate_result(p, plan['key'], kw['job_id'])


if __name__ == '__main__':
    unittest.main()
