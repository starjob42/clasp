"""Finite metadata exception, actual retained raw auditor, queue and cost tests."""
import copy
import fcntl
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


post = load('foarl_post_tests', ROOT / 'scripts/grpo_incumbent_foarl_postprocessing.py')
old = load('foarl_post_original', ROOT / 'scripts/grpo_incumbent_dev_campaign.py')
bridge = load('foarl_post_retry', ROOT / 'scripts/grpo_incumbent_dev_evaluation_retry.py')


class CheckpointViewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.checkpoint = self.root / 'checkpoints/checkpoint-100'
        self.checkpoint.mkdir(parents=True)
        self.plan = {'key': post.KEYS[0], 'method': 'foarl_f638_dev', 'training_root': str(self.root), 'checkpoint': str(self.checkpoint)}
        files = {}
        for name in post.CHECKPOINT_NAMES:
            path = self.checkpoint / name; path.write_text(name); files[name] = post.identity(path)
        evidence = {}
        for name in post.PROOFS:
            path = self.checkpoint / name; path.write_text(name)
            evidence[str(path.relative_to(self.root))] = post.identity(path)
        for name in (post.PROOFS[0], post.PROOFS[2]):
            path = self.root / name; path.write_text(name); evidence[name] = post.identity(path)
        self.terminal = {'checkpoint': files, 'exact_reference_evidence': {'status': 'exact_frozen_initial_sft_reference_lineage_verified', 'files': evidence}}
        artifact_files = [{'path': path.name, 'sha256': post.identity(path)['sha256'], 'size': path.stat().st_size}
                          for path in sorted(self.checkpoint.iterdir()) if path.name != 'trainer_state.json']
        self.config = {'experiment_provenance': {'model': {'checkpoint_artifact': {'files': artifact_files}}}}

    def test_exact_three_proof_view_never_mutates_original_terminal(self):
        original = copy.deepcopy(self.terminal)
        view = post.checkpoint_audit_view(self.plan, self.terminal, self.config)
        self.assertEqual(self.terminal, original)
        self.assertEqual(set(view['checkpoint']) - set(original['checkpoint']), set(post.PROOFS))
        self.assertEqual(len(self.config['experiment_provenance']['model']['checkpoint_artifact']['files']), 12)

    def test_wrong_method_checkpoint_or_missing_proof_rejects(self):
        for field, value in [('key', 'cvrp/grpo_z'), ('method', 'grpo_z'), ('checkpoint', str(self.root))]:
            with self.subTest(field=field):
                changed = {**self.plan, field: value}
                with self.assertRaises(ValueError):
                    post.checkpoint_audit_view(changed, self.terminal, self.config)
        self.terminal['exact_reference_evidence']['files'].pop('checkpoints/checkpoint-100/' + post.PROOFS[0])
        with self.assertRaisesRegex(ValueError, 'inventory'):
            post.checkpoint_audit_view(self.plan, self.terminal, self.config)

    def test_root_copy_cannot_substitute_for_checkpoint_local_proof(self):
        self.terminal['exact_reference_evidence']['files']['checkpoints/checkpoint-100/' + post.PROOFS[0]] = self.terminal['exact_reference_evidence']['files'][post.PROOFS[0]]
        with self.assertRaisesRegex(ValueError, 'bytes/path'):
            post.checkpoint_audit_view(self.plan, self.terminal, self.config)

    def test_mutated_proof_unknown_artifact_duplicate_or_missing_name_reject(self):
        artifact = self.config['experiment_provenance']['model']['checkpoint_artifact']
        original = copy.deepcopy(artifact['files'])
        for values in [original + [original[0]], original[:-1], original + [{'path': 'unverified.json'}]]:
            with self.subTest(values=len(values)):
                artifact['files'] = values
                with self.assertRaisesRegex(ValueError, 'exactly'):
                    post.checkpoint_audit_view(self.plan, self.terminal, self.config)
        artifact['files'] = original
        (self.checkpoint / post.PROOFS[0]).write_text('mutated')
        with self.assertRaisesRegex(ValueError, 'bytes/path'):
            post.checkpoint_audit_view(self.plan, self.terminal, self.config)


class FailedMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.output = self.root / 'output'; (self.output / 'eval').mkdir(parents=True)
        self.key, self.job = post.KEYS[1], '999001'
        self.plan = {'key': self.key, 'output': str(self.output), 'stdout': str(self.root / '%j.out')}
        self.protocol = {'planned_eval_commands': [self.plan]}
        self.control = SimpleNamespace(submitted_job=lambda *a, **k: self.job)
        self.stderr = self.root / (self.job + '.err'); self.stderr.write_text('warnings and progress\n' + post.expected_traceback(self.key))
        (self.output / 'launch_receipt.json').write_text('{}')
        self.config = {'run_status': 'complete', 'limits': {'eval_start_index': 0, 'max_eval_samples_per_task': 100}, 'generation': {'best_of_n': 8}}
        bridge.create(self.output / 'eval/run_config__fixed.json', self.config, 'run_config_sha256')
        (self.output / 'eval/run_details__fixed.jsonl').write_text('{}\n')
        self.row = {'job_id': self.job, 'state': 'FAILED', 'exit_code': '1:0', 'elapsed_seconds': 800, 'stdout': str(self.root / (self.job + '.out'))}

    def test_only_exact_final_traceback_admits_complete_metadata_and_preserves_failed(self):
        value = post.failure_metadata(self.key, self.control, self.protocol, self.row)
        self.assertEqual(value['scheduler_status_preserved'], 'FAILED')
        self.assertTrue(value['no_inference_repeat'])

    def test_other_error_extra_traceback_partial_k_or_wrong_scheduler_rejects(self):
        for ending in ['ValueError: another error\n', post.expected_traceback(self.key) * 2, post.expected_traceback(self.key) + 'unexpected tail\n']:
            self.stderr.write_text(ending)
            with self.assertRaisesRegex(ValueError, 'different reason'):
                post.failure_metadata(self.key, self.control, self.protocol, self.row)
        self.stderr.write_text(post.expected_traceback(self.key))
        for field, value in [('job_id', 'other'), ('state', 'COMPLETED'), ('exit_code', '0:0'), ('elapsed_seconds', 7201)]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'submitted'):
                post.failure_metadata(self.key, self.control, self.protocol, {**self.row, field: value})
        for field, value in [('run_status', 'interrupted'), ('generation', {'best_of_n': 1}), ('limits', {'eval_start_index': 0, 'max_eval_samples_per_task': 99})]:
            changed = {**self.config, field: value}
            path = self.output / 'eval/run_config__fixed.json'
            path.chmod(0o640)
            path.write_text(json.dumps({**changed, 'run_config_sha256': post.digest(changed)}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'partial or different'):
                post.failure_metadata(self.key, self.control, self.protocol, self.row)

    def test_original_success_index_must_not_be_added_to_failed_output(self):
        (self.output / 'result_index.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'without an index'):
            post.failure_metadata(self.key, self.control, self.protocol, self.row)

    @unittest.skipUnless((ROOT / 'analysis_outputs/grpo_incumbent_20260907_v1/foarl_postprocessing_metadata_20260907T111046107040Z_6138f0bd.json').exists(), 'native evidence is outside code-only snapshot')
    def test_exact_native_witness_matches_compiled_finite_exception(self):
        path = ROOT / 'analysis_outputs/grpo_incumbent_20260907_v1/foarl_postprocessing_metadata_20260907T111046107040Z_6138f0bd.json'
        witness = post.checked(path)['native']
        self.assertEqual(witness['exact_traceback'], post.expected_traceback(post.KEYS[0]))
        self.assertEqual(witness['duplicate_artifact_names'], [])
        for name in ('stderr', 'launch', 'run_config'):
            self.assertEqual(witness[name]['sha256'], post.KNOWN_CVRP[name + '_sha256'])
        self.assertEqual(witness['raw_details_identity']['sha256'], post.KNOWN_CVRP['details_sha256'])
        for proof in witness['unknown_artifact_proofs']:
            self.assertTrue(proof['bytes_match_training_evidence'] and proof['bytes_match_artifact'])
            self.assertLess(proof['proof_mtime_utc'], witness['terminal_mtime_utc'])


class ActualRawAuditorTests(unittest.TestCase):
    def setUp(self):
        fixtures = load('foarl_original_raw_fixture', ROOT / 'tests/test_grpo_incumbent_dev_campaign.py')
        fixture = fixtures.CrossBindingTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.plan, self.launch, self.config, self.index, self.terminal = fixture.args
        self.kw = fixture.kwargs; self.source = self.kw['source']
        self.plan.update(key=post.KEYS[0], method='foarl_f638_dev')
        self.launch['method'] = self.index['method'] = 'foarl_f638_dev'
        self.launch.update(campaign_protocol_sha256='original', training_job_id='777')
        self.terminal['job_id'] = '777'
        self.campaign = self.source.parent / 'campaign'
        (self.campaign / 'ledgers/terminal').mkdir(parents=True)
        (self.campaign / 'ledgers/terminal/cvrp_foarl_f638_dev.json').write_text('{}')
        checkpoint = Path(self.plan['checkpoint']); train = Path(self.plan['training_root'])
        for name in post.CHECKPOINT_NAMES - set(self.terminal['checkpoint']):
            (checkpoint / name).write_text(name); self.terminal['checkpoint'][name] = post.identity(checkpoint / name)
        evidence = {}
        for name in post.PROOFS:
            path = checkpoint / name; path.write_text(name); evidence[str(path.relative_to(train))] = post.identity(path)
        for name in (post.PROOFS[0], post.PROOFS[2]):
            path = train / name; path.write_text(name); evidence[name] = post.identity(path)
        self.terminal['exact_reference_evidence'] = {'status': 'exact_frozen_initial_sft_reference_lineage_verified', 'files': evidence}
        artifact = self.config['experiment_provenance']['model']['checkpoint_artifact']
        artifact['files'] = [{'path': path.name, 'sha256': post.identity(path)['sha256'], 'size': path.stat().st_size}
                             for path in sorted(checkpoint.iterdir()) if path.name != 'trainer_state.json']
        artifact['manifest_sha256'] = post.digest(artifact['files'])
        self.protocol = {**self.kw['training'], 'receipt_sha256': 'original', 'planned_eval_commands': [self.plan],
                         'benchmark': self.kw['benchmark'], 'generation': self.kw['evaluation']['generation'], 'evaluation_management': self.kw['evaluation']['management']}
        self.control = SimpleNamespace(training_endpoint=lambda *a, **k: self.terminal, audit_evaluation_arm=old.audit_evaluation_arm)
        self.fresh = SimpleNamespace(GENERATION=self.protocol['generation'], index=lambda *a: self.index, validate_benchmark=lambda *a: self.kw['benchmark'])
        self.artifact = SimpleNamespace(artifact_type='native_raw', records=self.kw['records'])
        self.strict = SimpleNamespace(_load_artifact=lambda *a, **k: self.artifact)
        self.failure = {'job_id': self.kw['job_id']}
        self.addCleanup(patch.stopall)
        patch.object(post, 'CAMPAIGN', self.campaign).start(); patch.object(post, 'TRAINING_SOURCE', self.source).start()
        patch.object(post, 'failure_metadata', return_value=self.failure).start()
        patch.object(post, 'module', side_effect=lambda name, path: self.fresh if Path(path).name == 'clasp_grpo_fresh_dev_artifacts.py' else self.strict).start()
        patch.object(post, 'checked', side_effect=lambda path, field: self.launch if field == 'launch_sha256' else self.config).start()

    def check(self):
        return post.validate_post_result(self.plan['key'], self.control, self.protocol, {})

    def test_original_auditor_fails_and_validated_view_passes_without_raw_change(self):
        before = copy.deepcopy(self.terminal)
        with self.assertRaisesRegex(ValueError, 'unknown/duplicate'):
            old.audit_evaluation_arm(self.plan, self.launch, self.config, self.index, self.terminal, **self.kw)
        result = self.check()
        self.assertEqual(result['cross_bindings']['instances'], 100)
        self.assertEqual(result['cross_bindings']['candidates'], 800)
        self.assertEqual(result['scheduler_state'], 'FAILED')
        self.assertEqual(self.terminal, before)

    def test_actual_raw_crossauditor_still_rejects_model_data_job_and_counts(self):
        self.config['resolved_model_path'] = self.plan['training_root']
        with self.assertRaisesRegex(ValueError, 'raw run model'):
            self.check()
        self.config['resolved_model_path'] = self.plan['checkpoint']
        self.config['experiment_provenance']['data']['files'][0]['sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'dataset hash'):
            self.check()
        self.config['experiment_provenance']['data']['files'][0]['sha256'] = 'dataset'
        self.launch['slurm_job_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'launch job'):
            self.check()
        self.launch['slurm_job_id'] = self.kw['job_id']
        self.artifact.records.pop(99)
        with self.assertRaisesRegex(ValueError, '100 raw records'):
            self.check()

    def test_assisted_or_changed_proof_bytes_never_admitted(self):
        self.artifact.artifact_type = 'assisted'
        with self.assertRaisesRegex(ValueError, 'unassisted'):
            self.check()
        self.artifact.artifact_type = 'native_raw'
        (Path(self.plan['checkpoint']) / post.PROOFS[1]).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'bytes/path'):
            self.check()


class QueueAndCostTests(unittest.TestCase):
    def setUp(self):
        self.control = SimpleNamespace(QOS='qos_gpu_a100-dev', USER='uag72kz', expanded_queue=old.expanded_queue)

    def test_unknown_states_and_external_queue_reserve_slots(self):
        original = {post.KEYS[0]: [{'kind': 'submitted', 'job_id': '101'}]}
        self.control.read_events = lambda *a, role: original if role == 'eval' else {}
        retry_events = {'mis/grpo_z': [{'kind': 'submitted', 'job_id': '102'}]}
        rows = {'101': {'state': 'UNKNOWN', 'exit_code': '0:0'}, '102': {'state': 'COMPLETING', 'exit_code': '0:0'}}
        fake = SimpleNamespace(read_events=lambda p: retry_events, scheduler=lambda *a: rows, FAILURE_STATES=bridge.FAILURE_STATES)
        occupied, _, _ = post.all_reservations({}, fake, {}, self.control, {}, lambda cmd: '100|PENDING\n')
        self.assertEqual(occupied, 3)
        with self.assertRaisesRegex(ValueError, 'expanded'):
            post.all_reservations({}, fake, {}, self.control, {}, lambda cmd: '100_[1-5]|PENDING\n')

    def test_only_known_failure_lanes_reach_their_exact_validators(self):
        self.control.read_events = lambda *a, role: {'mis/grpo_verified': [{'kind': 'submitted', 'job_id': '103'}]} if role == 'eval' else {}
        fake = SimpleNamespace(read_events=lambda p: {}, scheduler=lambda *a: {'103': {'state': 'FAILED', 'exit_code': '1:0'}},
                               FAILURE_STATES=bridge.FAILURE_STATES, FAILED=bridge.FAILED)
        with self.assertRaisesRegex(ValueError, 'unlisted'):
            post.all_reservations({}, fake, {}, self.control, {}, lambda cmd: '')

    def test_failed_foarl_costs_remain_failed_and_fully_counted(self):
        resolved = [{'job_id': '1001', 'key': post.KEYS[0], 'audit': {'failure': {'scheduler': {'elapsed_seconds': 1879}}}},
                    {'job_id': '1002', 'key': post.KEYS[1], 'audit': {'failure': {'scheduler': {'elapsed_seconds': 800}}}}]
        raw = ('1001|FAILED|1:0|2026-09-07T10:29:18|2026-09-07T11:00:37|1879|120|billing=1,cpu=16,gres/gpu=1,mem=60000M,node=1|qos_gpu_a100-dev|vas@a100|gpu_p5\n'
               '1002|FAILED|1:0|2026-09-07T11:00:00|2026-09-07T11:13:20|800|120|billing=1,cpu=16,gres/gpu=1,mem=60000M,node=1|qos_gpu_a100-dev|vas@a100|gpu_p5\n')
        result = post.postprocessing_allocation_records(resolved, self.control, runner=lambda *a, **k: raw)
        self.assertAlmostEqual(result['allocated_gpu_hours'], 2679 / 3600)
        self.assertEqual(sum(r['allocated_cpu_seconds'] for r in result['jobs']), 16 * 2679)
        self.assertTrue(all(r['State'] == 'FAILED' for r in result['jobs']))
        for modified in [raw.replace('FAILED', 'COMPLETED', 1), raw.replace('gres/gpu=1', 'gres/gpu=2', 1), raw.replace('1879|120', '1878|120', 1),
                         raw.replace('billing=1', 'billing=7,billing=1', 1)]:
            with self.subTest(modified=modified[:20]), self.assertRaises(ValueError):
                post.postprocessing_allocation_records(resolved, self.control, runner=lambda *a, **k: modified)


class SidecarAndSupervisorTests(unittest.TestCase):
    def test_sidecars_are_separate_create_only_and_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); original = root / 'original'; original.mkdir()
            raw = original / 'raw.jsonl'; raw.write_text('retained original bytes')
            before = post.identity(raw)
            declaration = root / 'declaration.json'; declaration.write_text('{}')
            actual = {'key': post.KEYS[0], 'index_payload': {'key': 'raw', 'index_sha256': post.digest({'key': 'raw'})}}
            p = {'known_cvrp_audit': actual}
            sidecars = root / 'sidecars'; sidecars.mkdir()
            with patch.object(post, 'SIDECARS', sidecars), patch.object(post, 'DECLARATION', declaration), \
                    patch.object(post, 'validate_post_result', return_value=actual):
                post.retain_post_result(p, bridge, None, {}, post.KEYS[0], {}, record=True)
                self.assertEqual(post.identity(raw), before)
                self.assertEqual(list(original.iterdir()), [raw])
                post.retain_post_result(p, bridge, None, {}, post.KEYS[0], {}, record=False)
                terminal = sidecars / post.KEYS[0].replace('/', '_') / 'terminal.json'
                value = post.checked(terminal); value['key'] = 'changed'; value.pop('receipt_sha256')
                terminal.chmod(0o640); terminal.write_text(json.dumps({**value, 'receipt_sha256': post.digest(value)}))
                with self.assertRaisesRegex(ValueError, 'sidecar changed'):
                    post.retain_post_result(p, bridge, None, {}, post.KEYS[0], {}, record=False)

    def test_only_original_mis_submission_occurs_after_full_known_cvrp_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'ledgers').mkdir()
            keys = [task + '/' + method for task in ('cvrp', 'mis') for method in ('grpo_z', 'grpo_verified', 'foarl_f638_dev')]
            events = {role: {key: [{'kind': 'submitted', 'job_id': str(100 + i)}] for i, key in enumerate(keys)} for role in ('train', 'eval')}
            events['eval'][post.KEYS[1]] = []
            markers = []
            def reconcile(*args, role, submit, max_new_jobs, **kwargs):
                observations = [{'key': key, 'state': 'COMPLETED', 'terminal_verified': True} for key in keys]
                if role == 'eval':
                    self.assertTrue(submit); self.assertEqual(max_new_jobs, 1)
                    markers.append('original_mis_submit')
                    events['eval'][post.KEYS[1]] = [{'kind': 'submitted', 'job_id': '201'}]
                    observations[-1] = {'key': post.KEYS[1], 'state': 'submitted', 'job_id': '201'}
                return {'observations': observations, 'all_training_verified': True}
            def visible(*args):
                self.assertEqual(args[-2:], ('201', post.KEYS[1]))
                with (root / 'ledgers/controller.lock').open('a') as competing:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(competing.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                markers.append('visibility_before_unlock')
            control = SimpleNamespace(read_events=lambda *a, role: copy.deepcopy(events[role]), reconcile=reconcile)
            fake_bridge = SimpleNamespace(scheduler=lambda *a: {post.KNOWN_CVRP['job_id']: {'job_id': post.KNOWN_CVRP['job_id']}},
                         await_visibility=visible, FAILURE_STATES=bridge.FAILURE_STATES, FAILED=bridge.FAILED)
            rows = {job: {'state': 'RUNNING'} for job in post.RETRY_JOBS.values()}
            rows.update({'102': {'state': 'FAILED'}, '201': {'state': 'RUNNING'}})
            p = {'known_cvrp_audit': {'validated': True}}
            def full_validation(*args):
                markers.append('full_cvrp_raw_and_kl_validation'); return p['known_cvrp_audit']
            with patch.object(post, 'CAMPAIGN', root), patch.object(post, 'declaration', return_value=(p, fake_bridge, {}, control, {})), \
                    patch.object(post, 'all_reservations', return_value=(9, rows, {})), \
                    patch.object(post, 'validate_post_result', side_effect=full_validation), \
                    patch.object(post, 'retain_post_result', return_value={}):
                value = post.supervisor_tick('pinned', submit=True, record=False, max_new_jobs=5)
            self.assertEqual(markers, ['full_cvrp_raw_and_kl_validation', 'original_mis_submit', 'visibility_before_unlock'])
            self.assertFalse(value['all_six_evaluations_verified'])
            self.assertEqual(events['eval'][post.KEYS[1]][0]['job_id'], '201')

    def test_full_queue_forwards_zero_budget_and_missing_other_arm_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'ledgers').mkdir()
            keys = [task + '/' + method for task in ('cvrp', 'mis') for method in ('grpo_z', 'grpo_verified', 'foarl_f638_dev')]
            events = {role: {key: [{'kind': 'submitted', 'job_id': str(100 + i)}] for i, key in enumerate(keys)} for role in ('train', 'eval')}
            events['eval'][post.KEYS[1]] = []
            calls = []
            def reconcile(*args, role, submit, max_new_jobs, **kwargs):
                calls.append((role, submit, max_new_jobs))
                return {'observations': [{'key': key, 'state': 'RUNNING'} for key in keys], 'all_training_verified': False}
            control = SimpleNamespace(read_events=lambda *a, role: copy.deepcopy(events[role]), reconcile=reconcile)
            fake_bridge = SimpleNamespace(FAILURE_STATES=bridge.FAILURE_STATES, FAILED=bridge.FAILED)
            with patch.object(post, 'CAMPAIGN', root), patch.object(post, 'declaration', return_value=({}, fake_bridge, {}, control, {})), \
                    patch.object(post, 'all_reservations', return_value=(10, {}, {})), \
                    patch.object(post, 'validate_post_result', side_effect=AssertionError('no inference admission when queue full')):
                post.supervisor_tick('pinned', submit=True, record=False)
                self.assertEqual(calls, [('train', True, 0), ('eval', True, 0)])
                calls.clear(); events['train']['cvrp/grpo_z'] = []
                with self.assertRaisesRegex(ValueError, 'only the sole original'):
                    post.supervisor_tick('pinned', submit=True, record=False)
                self.assertEqual(calls, [])

    def test_ambiguous_original_mis_attempt_retains_visibility_lock_and_is_never_repeated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'ledgers').mkdir()
            keys = [task + '/' + method for task in ('cvrp', 'mis') for method in ('grpo_z', 'grpo_verified', 'foarl_f638_dev')]
            events = {role: {key: [{'kind': 'submitted', 'job_id': str(100 + i)}] for i, key in enumerate(keys)} for role in ('train', 'eval')}
            events['eval'][post.KEYS[1]] = []
            attempted, visibility = [], []
            def reconcile(*args, role, submit, max_new_jobs, **kwargs):
                if role == 'eval' and not events[role][post.KEYS[1]] and submit and max_new_jobs:
                    attempted.append(post.KEYS[1])
                    events[role][post.KEYS[1]] = [{'kind': 'intent'}, {'kind': 'rejected'}]
                    raise ValueError('ambiguous sbatch reply')
                return {'observations': [{'key': key, 'state': 'RUNNING'} for key in keys], 'all_training_verified': False}
            def await_visibility(*args):
                job, key = args[-2:]; visibility.append((job, key))
                with (root / 'ledgers/controller.lock').open('a') as competing:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(competing.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Real helper remains here until explicit operator review; this
                # model-free stand-in returns only to inspect the outer raise.
            control = SimpleNamespace(read_events=lambda *a, role: copy.deepcopy(events[role]), reconcile=reconcile)
            fake_bridge = SimpleNamespace(scheduler=lambda *a: {post.KNOWN_CVRP['job_id']: {}}, await_visibility=await_visibility,
                                          FAILURE_STATES=bridge.FAILURE_STATES, FAILED=bridge.FAILED)
            p = {'known_cvrp_audit': {'valid': True}}
            with patch.object(post, 'CAMPAIGN', root), patch.object(post, 'declaration', return_value=(p, fake_bridge, {}, control, {})), \
                    patch.object(post, 'all_reservations', return_value=(9, {}, {})), \
                    patch.object(post, 'validate_post_result', return_value=p['known_cvrp_audit']):
                with self.assertRaisesRegex(ValueError, 'ambiguous sbatch'):
                    post.supervisor_tick('pinned', submit=True, record=False)
                post.supervisor_tick('pinned', submit=True, record=False)
            self.assertEqual(attempted, [post.KEYS[1]])
            self.assertEqual(visibility, [(None, post.KEYS[1])])


if __name__ == '__main__':
    unittest.main()
