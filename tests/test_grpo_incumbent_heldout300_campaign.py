"""No models or native jobs: reject heldout count, assistance and binding drift."""
import copy
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
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


control = load('heldout_campaign', ROOT / 'scripts/grpo_incumbent_heldout300_campaign.py')
original_tests = load('heldout_original_fixture', ROOT / 'tests/test_grpo_incumbent_dev_campaign.py')
CURRENT_AUDIT_SLURM_SHA = control.AUDIT_SLURM_SHA


class HeldoutRawBindingTests(unittest.TestCase):
    def setUp(self):
        fixture = original_tests.CrossBindingTests()
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.args, self.kwargs = fixture.args, fixture.kwargs
        self.args[1]['evaluation_slice']['count'] = 300
        self.args[3]['evaluation_slice']['count'] = 300
        self.args[2]['limits']['max_eval_samples_per_task'] = 300
        expected = [{'example_index': i} for i in range(300)]
        self.kwargs['benchmark']['ordered_instances']['cvrp'] = expected
        self.kwargs['records'] = {i: SimpleNamespace(example_index=i, raw={'example_index': i, 'task': 'cvrp', 'split': 'eval',
                                 'num_return_sequences': 8, 'candidates': [{}] * 8}) for i in range(300)}

    def check(self):
        return control.audit_raw300(*self.args, **self.kwargs)

    def test_exact_300_instance_2400_candidate_binding_keeps_step100_checkpoint(self):
        result = self.check()
        self.assertEqual((result['instances'], result['candidates']), (300, 2400))
        self.assertTrue(result['checkpoint'].endswith('checkpoint-100'))

    def test_old_100_slice_cannot_masquerade_as_heldout300(self):
        self.args[2]['limits']['max_eval_samples_per_task'] = 100
        with self.assertRaisesRegex(ValueError, 'slice'):
            self.check()

    def test_dropped_instance_or_candidate_is_rejected(self):
        del self.kwargs['records'][299]
        with self.assertRaisesRegex(ValueError, '300 raw records'):
            self.check()
        self.setUp()
        self.kwargs['records'][0].raw['candidates'] = [{}] * 7
        with self.assertRaisesRegex(ValueError, 'benchmark/K8'):
            self.check()

    def test_assistance_and_alternative_model_or_job_are_rejected(self):
        for target, field, value, error in ((1, 'repair', True, 'repair'), (1, 'slurm_job_id', '1999999', 'launch job'),
                                          (2, 'resolved_model_path', self.args[0]['training_root'], 'raw run model')):
            with self.subTest(field=field):
                previous = self.args[target][field]; self.args[target][field] = value
                with self.assertRaisesRegex(ValueError, error):
                    self.check()
                self.args[target][field] = previous

    def test_code_bytes_changed_after_receipt_cannot_pass(self):
        source = self.kwargs['source'] / 'src/clasp/entrypoints/evaluate.py'
        source.write_text('different evaluation implementation')
        with self.assertRaisesRegex(ValueError, 'evaluator code'):
            self.check()


class HeldoutFoarlIndexIntegrationTests(unittest.TestCase):
    def setUp(self):
        fixtures = load('heldout_postprocessing_actual_raw_fixture', ROOT / 'tests/test_grpo_incumbent_foarl_postprocessing.py')
        fixture = fixtures.ActualRawAuditorTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.fixtures = fixtures
        self.plan, self.launch, self.config, self.terminal = fixture.plan, fixture.launch, fixture.config, fixture.terminal
        self.source = fixture.source; self.root = Path(self.plan['output']); (self.root / 'eval').mkdir()
        for name, filename in [('details', 'run_details__fixed.jsonl'), ('run_config', 'run_config__fixed.json')]:
            destination = self.root / 'eval' / filename
            Path(self.config['files'][name]).rename(destination)
            self.config['files'][name] = str(destination)
        self.records = {i: SimpleNamespace(example_index=i, raw={'example_index': i, 'task': 'cvrp', 'split': 'eval',
                        'num_return_sequences': 8, 'candidates': [{}] * 8}) for i in range(300)}
        benchmark = fixture.kw['benchmark']; benchmark['ordered_instances']['cvrp'] = [{'example_index': i} for i in range(300)]
        self.launch['evaluation_slice']['count'] = 300
        self.launch.update(heldout_protocol_sha256='heldout', environment=control.ENVIRONMENT)
        self.launch['generation'] = control.GENERATION
        self.config['limits']['max_eval_samples_per_task'] = 300
        self.config['generation'].update(control.GENERATION)
        self.config['experiment_provenance']['generation'] = {k: v for k, v in self.config['generation'].items() if k != 'best_of_n'}
        self.plan['evaluator_argv'] = control.evaluator_argv(self.plan, Path(self.config['data_root']))
        self.launch['evaluator_argv'] = self.plan['evaluator_argv']
        execution = {'heldout_protocol_sha256': 'heldout', 'slurm_job_id': self.launch['slurm_job_id'],
                     'launch_sha256': self.launch['launch_sha256'], 'returncode': 0,
                     'evaluator_argv': self.plan['evaluator_argv'], 'environment': control.ENVIRONMENT}
        (self.root / 'launch_receipt.json').write_text('synthetic launch bytes')
        (self.root / 'execution_receipt.json').write_text('synthetic execution bytes')
        self.p = {'receipt_sha256': 'heldout', 'planned_jobs': [self.plan], 'source_manifest': self.launch['source_manifest'],
                  'data_root': self.config['data_root'], 'management': self.launch['management'], 'benchmark': benchmark,
                  'model_admission': {'terminal_audits': {self.plan['key']: self.terminal}}}
        self.addCleanup(patch.stopall)
        patch.object(control, 'protocol_for', return_value=(self.p, self.source)).start()
        patch.object(control, 'submitted_job', return_value=self.launch['slurm_job_id']).start()
        patch.object(control, 'verify_bound_inputs').start()
        patch.object(control, 'checked', side_effect=lambda path, *a: self.launch if Path(path).name == 'launch_receipt.json'
                     else execution if Path(path).name == 'execution_receipt.json' else self.config).start()
        self.artifact = SimpleNamespace(artifact_type='native_raw', records=self.records)
        patch.object(control, 'module', return_value=SimpleNamespace(_load_artifact=lambda *a, **k: self.artifact)).start()
        self.resolver_loader = patch.object(control, 'postprocessing_control', return_value=fixtures.post).start()

    def test_actual_heldout_index_admits_exact_bound_foarl_proofs_and_300_rows(self):
        before = copy.deepcopy(self.terminal)
        result, artifact = control.index_arm(self.root, self.plan['key'])
        self.assertEqual(len(artifact.records), 300)
        self.assertEqual(result['evaluation_slice']['count'], 300)
        self.assertEqual(self.terminal, before)
        self.resolver_loader.assert_called_once()
        self.records.pop(299)
        with self.assertRaisesRegex(ValueError, '300 raw records'):
            control.index_arm(self.root, self.plan['key'])

    def test_nonfoarl_heldout_index_does_not_use_a_proof_exception(self):
        original_key = self.plan['key']; self.plan.update(key='cvrp/grpo_z', method='grpo_z')
        self.launch['method'] = 'grpo_z'
        self.p['model_admission']['terminal_audits'] = {self.plan['key']: self.terminal}
        artifact = self.config['experiment_provenance']['model']['checkpoint_artifact']
        artifact['files'] = [f for f in artifact['files'] if f['path'] not in self.fixtures.post.PROOFS]
        artifact['manifest_sha256'] = control.digest(artifact['files'])
        result, _ = control.index_arm(self.root, self.plan['key'])
        self.assertEqual(result['method'], 'grpo_z')
        self.assertNotEqual(self.plan['key'], original_key)
        self.resolver_loader.assert_not_called()


class HeldoutManagementTests(unittest.TestCase):
    def test_planned_argv_has_exact_count_budget_sampler_and_unassisted_evaluator(self):
        plan = {'checkpoint': '/model/checkpoint-100', 'task': 'mis', 'key': 'mis/foarl_f638_dev', 'output': '/new/heldout'}
        argv = control.evaluator_argv(plan, Path('/new/data'))
        for key, value in (('--max_eval_samples_per_task', '300'), ('--eval_start_index', '0'), ('--seed', '271828'),
                           ('--num_return_sequences', '8'), ('--generation_batch_size', '8'), ('--dtype', 'float16'),
                           ('--temperature', '0.7'), ('--top_p', '0.9'), ('--top_k', '50'), ('--max_new_tokens', '3000'),
                           ('--max_prompt_length', '29768'), ('--evaluation_purpose', 'development')):
            self.assertEqual(argv[argv.index(key) + 1], value)
        self.assertIn('--force_no_reason_prompt', argv)
        self.assertIn('--no-prefer_subset_files', argv)
        self.assertFalse(any('repair' in arg or 'rescore' in arg for arg in argv))

    def test_changed_prebound_model_bytes_fail_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); files = {}
            for name in ('model', 'contract', 'hp', 'policy', 'protocol', 'accounting'):
                path = root / name; path.write_text(name); files[name] = control.identity(path)
            p = {'data_root': str(root), 'benchmark': {'manifest': {'sha256': 'benchmark'}}, 'native_overlap_audit': {'files': {}},
                 'model_admission': {'training_protocol': files['protocol'], 'training_accounting': files['accounting'],
                    'terminal_audits': {'cvrp/grpo_z': {'checkpoint': {'adapter_model.safetensors': files['model']},
                        'contract': files['contract'], 'hyperparams': files['hp'], 'policy_consistency': files['policy'],
                        'exact_reference_evidence': None}}}}
            with patch.object(control, 'validate_holdout', return_value=p['benchmark']), \
                    patch.object(control, 'checked', return_value=p['model_admission']['terminal_audits']['cvrp/grpo_z']), \
                    patch.object(control, 'verify_development_admission'):
                control.verify_bound_inputs(p, root)
                (root / 'model').write_text('a substituted checkpoint')
                with self.assertRaisesRegex(ValueError, 'model/metadata bytes changed'):
                    control.verify_bound_inputs(p, root)

    def test_queue_cap_and_orphan_submission_never_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('ledgers', 'ledgers/events', 'ledgers/terminal'):
                (root / name).mkdir()
            admission = {'terminal_audits': {task + '/' + method: {'job_id': str(2000 + i), 'checkpoint': {
                          'adapter_model.safetensors': {'path': '/models/' + task + '/' + method + '/checkpoints/checkpoint-100/adapter_model.safetensors'}}}
                          for i, (task, method) in enumerate((t, m) for t in control.TASKS for m in control.METHODS)}}
            plans = control.planned_jobs(root, root, root / 'data', admission)
            p = {'receipt_sha256': 'protocol', 'planned_jobs': plans}
            event = {'key': plans[0]['key'], 'command': plans[0]['command'], 'protocol_sha256': 'protocol', 'transaction': 'first', 'kind': 'intent'}
            control.write_new(root / 'ledgers/events/intent.json', event, 'event_sha256')
            submitted = []
            def runner(command, text=True):
                if command[0] == 'squeue':
                    return ''.join(f'879217_{i}|PENDING\n' for i in range(5))
                if command[0] == 'sbatch':
                    submitted.append(command); return str(1950000 + len(submitted))
                if command[0] == 'sacct':
                    return ''
                raise AssertionError(command)
            with patch.object(control, 'protocol_for', return_value=(p, root)), patch.object(control, 'old_control', return_value=original_tests.control), \
                    patch.object(control, 'verify_bound_inputs'):
                result = control.reconcile(root, submit=True, record=True, runner=runner)
            self.assertEqual(result['observations'][0]['state'], 'prior_attempt_no_retry')
            self.assertEqual(len(submitted), 4)  # five unrelated jobs + unresolved intent + four accepted = ten
            self.assertNotIn(plans[0]['command'], submitted)


class NativeAuditBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('data', 'execution'):
            (self.root / name).mkdir()
        self.job = '1999998'
        for name in ('run_native_holdout_audit_v1.py', 'audit_native_holdout_v1.slurm'):
            (self.root / name).write_text('synthetic reviewed native audit implementation: ' + name)
        for field, name in (('AUDIT_WRAPPER_SHA', 'run_native_holdout_audit_v1.py'), ('AUDIT_SLURM_SHA', 'audit_native_holdout_v1.slurm')):
            patcher = patch.object(control, field, control.identity(self.root / name)['sha256'])
            patcher.start(); self.addCleanup(patcher.stop)
        (self.root / 'data/overlap_audit.json').write_text('{"native_audit":true}\n')
        (self.root / 'execution/runtime_imports.json').write_text('{"imports":"pinned"}\n')
        built = control.write_new(self.root / 'data/build_manifest.json', {'plan_sha256': control.PLAN_SHA}, 'manifest_sha256')
        self.benchmark = {'build_manifest': control.identity(self.root / 'data/build_manifest.json'), 'audit_sha256': 'audit-self',
                          'overlap_audit': control.identity(self.root / 'data/overlap_audit.json')}
        manifest = {'protocol': 'grpo-incumbent-native-holdout-overlap-audit-v1', 'plan_sha256': control.PLAN_SHA,
                    'build_manifest_sha256': built['manifest_sha256'], 'files': {}}
        for name in ('run_native_holdout_audit_v1.py', 'audit_native_holdout_v1.slurm'):
            identity = control.identity(self.root / name)
            manifest['files'][name] = {k: identity[k] for k in ('sha256', 'size_bytes')}
        (self.root / 'bundle_manifest.json').write_text(json.dumps(manifest))
        self.sha = control.identity(self.root / 'bundle_manifest.json')['sha256']
        self.start = {'protocol': manifest['protocol'], 'plan_sha256': control.PLAN_SHA,
                      'build_manifest_sha256': built['manifest_sha256'],
                      'slurm': {'SLURM_JOB_ID': self.job, 'SLURM_CPUS_PER_TASK': '2',
                                'SLURM_JOB_ACCOUNT': 'vas@a100', 'SLURM_JOB_PARTITION': 'gpu_p5'},
                      'bundle_manifest_sha256': self.sha, 'python_flags': {'isolated': 1, 'no_site': 1, 'dont_write_bytecode': 1}}
        self.profile = {'name': 'one_reserved_a100_cpu_only_scan_v3', 'requested_cpus_per_task': 2,
                        'reserved_gpus': 1, 'cpu_scan_threads': 1, 'cuda_visible_devices': '', 'gpu_compute_requested': False,
                        'actual_allocated_cpus': 16, 'actual_allocated_memory': '60000M',
                        'slurm_job': {'JobId': self.job, 'Account': 'vas@a100', 'QOS': 'qos_gpu_a100-dev', 'Partition': 'gpu_p5',
                                      'NumCPUs': '16', 'CPUs/Task': '2', 'NumTasks': '1', 'NumNodes': '1', 'NodeList': 'r0n0',
                                      'AllocTRES': 'cpu=16,mem=60000M,node=1,billing=1,gres/gpu=1',
                                      'ReqTRES': 'cpu=2,mem=60000M,node=1,billing=1,gres/gpu=1'},
                        'slurm_node': {'NodeName': 'r0n0', 'AvailableFeatures': 'a100,x86_64', 'CPUTot': '128',
                                       'ThreadsPerCore': '2', 'Gres': 'gpu:a100:8', 'CfgTRES': 'cpu=128,gres/gpu=8'}}
        self.start['resource_profile'] = self.profile
        self.end = {'status': 'passed', 'exit_code': 0, 'frozen_inputs_unchanged': True, 'sources': 8, 'rows': 1006000,
                    'total_overlaps': 0, 'audit_sha256': 'audit-self', 'bundle_manifest_sha256': self.sha,
                    'overlap_receipt_file': {k: self.benchmark['overlap_audit'][k] for k in ('sha256', 'size_bytes')},
                    'resource_profile': self.profile, 'elapsed_seconds': 180.25,
                    'reserved_gpu_seconds_during_wrapper': 180.25, 'reserved_gpu_hours_during_wrapper': 180.25 / 3600,
                    'allocated_cpu_seconds_during_wrapper': 16 * 180.25,
                    'full_job_allocation_cost_requires_slurm_terminal_accounting': True}
        self.slurm = self.job + '|COMPLETED|0:0|200|120|cpu=16,mem=60000M,node=1,billing=1,gres/gpu=1|qos_gpu_a100-dev|vas@a100|gpu_p5\n'

    def check(self):
        (self.root / 'execution/start.json').write_text(json.dumps(self.start))
        (self.root / 'execution/exit.json').write_text(json.dumps(self.end))
        return control.validate_native_audit(self.root, self.sha, self.job, self.benchmark, runner=lambda *a, **kw: self.slurm)

    def test_exact_successful_eight_source_native_audit_passes(self):
        self.assertEqual(self.check()['status'], 'successful_native_reserved_a100_cpu_overlap_audit_bound')

    def test_old_failed_dataset_plan_is_rejected_even_after_consistent_receipt_rehash(self):
        old_plan = '7b6d9bb9867d4a3f39d6b5fcb0900bd05db570d52c68d60edd83f0af16dc7503'
        manifest_path = self.root / 'bundle_manifest.json'
        manifest = json.loads(manifest_path.read_text()); manifest['plan_sha256'] = old_plan
        manifest_path.write_text(json.dumps(manifest)); self.sha = control.identity(manifest_path)['sha256']
        self.start['plan_sha256'] = old_plan
        self.start['bundle_manifest_sha256'] = self.end['bundle_manifest_sha256'] = self.sha
        with self.assertRaisesRegex(ValueError, 'plan/build scope'):
            self.check()

    def test_failed_native_job_cannot_be_admitted_from_a_passed_overlap_json(self):
        self.slurm = self.slurm.replace('|COMPLETED|0:0|', '|FAILED|1:0|')
        with self.assertRaisesRegex(ValueError, 'reserved-A100 job did not complete'):
            self.check()

    def test_native_audit_implementation_changed_after_binding_fails(self):
        (self.root / 'run_native_holdout_audit_v1.py').write_text('changed audit semantics')
        with self.assertRaisesRegex(ValueError, 'implementation bytes changed'):
            self.check()

    def test_missing_source_scan_wrong_job_or_nonisolated_runtime_fails(self):
        for target, key, value, error in ((self.end, 'sources', 6, 'eight-source'),
                                          (self.start['slurm'], 'SLURM_JOB_ID', '1999997', 'execution/job'),
                                          (self.start['python_flags'], 'isolated', 0, 'isolation')):
            with self.subTest(key=key):
                previous = target[key]; target[key] = value
                with self.assertRaisesRegex(ValueError, error):
                    self.check()
                target[key] = previous

    def use_real_slurm(self, path):
        (self.root / 'audit_native_holdout_v1.slurm').write_bytes(path.read_bytes())
        manifest = json.loads((self.root / 'bundle_manifest.json').read_text())
        observed = control.identity(self.root / 'audit_native_holdout_v1.slurm')
        manifest['files']['audit_native_holdout_v1.slurm'] = {k: observed[k] for k in ('sha256', 'size_bytes')}
        (self.root / 'bundle_manifest.json').write_text(json.dumps(manifest))
        self.sha = control.identity(self.root / 'bundle_manifest.json')['sha256']
        self.start['bundle_manifest_sha256'] = self.end['bundle_manifest_sha256'] = self.sha

    def test_reviewed_v4_admits_requested_two_cpus_with_scheduler_allocated_sixteen(self):
        path = ROOT / 'analysis_outputs/grpo_incumbent_20260907_v1/audit_native_holdout_v4.slurm'
        self.assertNotIn(b'#SBATCH --mem', path.read_bytes())
        self.assertEqual(path.read_bytes().count(b'#SBATCH --constraint=a100\n'), 1)
        self.use_real_slurm(path)
        with patch.object(control, 'AUDIT_SLURM_SHA', CURRENT_AUDIT_SLURM_SHA):
            result = self.check()
        self.assertEqual(result['allocated_cpus'], 16)
        self.assertEqual(result['allocated_cpu_seconds'], 3200)
        self.assertEqual(result['allocated_gpu_seconds'], 200)
        self.assertEqual(result['slurm_billing_unit_seconds'], 200.)

    def test_old_scripts_fail_even_with_consistent_manifest_and_job(self):
        for version in (1, 2, 3):
            with self.subTest(version=version):
                self.use_real_slurm(ROOT / f'analysis_outputs/grpo_incumbent_20260907_v1/audit_native_holdout_v{version}.slurm')
                with patch.object(control, 'AUDIT_SLURM_SHA', CURRENT_AUDIT_SLURM_SHA):
                    with self.assertRaisesRegex(ValueError, 'reviewed implementation'):
                        self.check()

    def test_slurm_bytes_changed_after_manifest_binding_fail(self):
        (self.root / 'audit_native_holdout_v1.slurm').write_text('unreviewed resource directives')
        with self.assertRaisesRegex(ValueError, 'implementation bytes changed'):
            self.check()

    def test_cpu_gpu_and_billing_allocation_costs_remain_distinct(self):
        self.profile['slurm_job']['AllocTRES'] = self.profile['slurm_job']['AllocTRES'].replace('billing=1', 'billing=2')
        self.slurm = self.slurm.replace('billing=1', 'billing=2')
        result = self.check()
        self.assertEqual(result['allocated_cpu_seconds'], 3200)
        self.assertEqual(result['allocated_gpu_seconds'], 200)
        self.assertEqual(result['slurm_billing_unit_seconds'], 400.)
        self.assertAlmostEqual(result['allocated_gpu_hours'], 200 / 3600)
        self.assertAlmostEqual(result['slurm_billing_unit_hours'], 400 / 3600)

    def test_memory_units_and_tres_order_can_differ_without_changing_allocation(self):
        self.profile['actual_allocated_memory'] = '61440M'
        self.profile['slurm_job']['AllocTRES'] = self.profile['slurm_job']['AllocTRES'].replace('mem=60000M', 'mem=61440M')
        self.slurm = self.slurm.replace('cpu=16,mem=60000M,node=1,billing=1,gres/gpu=1',
                                        'billing=1,gres/gpu=1,mem=60G,node=1,cpu=16')
        self.assertEqual(self.check()['allocated_cpus'], 16)

    def test_wrong_device_visibility_threads_requested_cpus_or_node_are_rejected(self):
        mutations = [(self.profile, 'cuda_visible_devices', '0'), (self.profile, 'gpu_compute_requested', True),
                     (self.profile, 'cpu_scan_threads', 4), (self.profile, 'reserved_gpus', True),
                     (self.start['slurm'], 'SLURM_CPUS_PER_TASK', '8'),
                     (self.profile['slurm_node'], 'AvailableFeatures', 'h100,x86_64'),
                     (self.profile['slurm_job'], 'JobId', '1999997')]
        for target, field, value in mutations:
            with self.subTest(field=field):
                previous = target[field]; target[field] = value
                with self.assertRaises(ValueError):
                    self.check()
                target[field] = previous

    def test_start_terminal_allocation_changes_are_rejected(self):
        original = self.slurm
        for old, new in (('cpu=16', 'cpu=20'), ('gres/gpu=1', 'gres/gpu=2'), ('mem=60000M', 'mem=70000M'),
                         ('billing=1', 'billing=2'), ('qos_gpu_a100-dev', 'qos_gpu_a100-t3')):
            with self.subTest(field=old):
                self.slurm = original.replace(old, new)
                with self.assertRaises(ValueError):
                    self.check()
        self.slurm = original

    def test_profile_and_wrapper_cost_projection_must_agree(self):
        self.end['allocated_cpu_seconds_during_wrapper'] = 2 * self.end['elapsed_seconds']
        with self.assertRaisesRegex(ValueError, 'allocation-cost'):
            self.check()
        self.end['allocated_cpu_seconds_during_wrapper'] = 16 * self.end['elapsed_seconds']
        self.end['resource_profile'] = {**self.profile, 'reserved_gpus': 2}
        with self.assertRaisesRegex(ValueError, 'CPU-only profile'):
            self.check()


class ImmutableRetryImportTests(unittest.TestCase):
    def test_retry_resolver_imports_only_its_original_v7_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'v7'; (source / 'scripts').mkdir(parents=True)
            helper = source / control.RETRY_HELPER; helper.write_text('original immutable helper')
            manifest = source / 'SOURCE_MANIFEST.sha256'; manifest.write_text('pinned inventory')
            with patch.object(control, 'RETRY_SOURCE', source), \
                    patch.object(control, 'RETRY_SOURCE_SHA', control.identity(manifest)['sha256']), \
                    patch.object(control, 'RETRY_HELPER_SHA', control.identity(helper)['sha256']), \
                    patch.object(control, 'module', return_value=SimpleNamespace(__file__=str(helper))) as load:
                control.retry_control()
                self.assertEqual(load.call_args.args[1], helper)
                load.return_value = SimpleNamespace(__file__=str(Path(temporary) / 'v8' / control.RETRY_HELPER))
                with self.assertRaisesRegex(ValueError, 'another management snapshot'):
                    control.retry_control()
                helper.write_text('a modified resolver')
                with self.assertRaisesRegex(ValueError, 'helper differs'):
                    control.retry_control()

    def test_initialization_refuses_any_other_output_before_loading_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(control, 'old_control') as loader:
                with self.assertRaisesRegex(ValueError, 'fixed never-created output'):
                    control.initialize(root / 'other', root, 'source', root, 'data', root, 'audit', '123')
            loader.assert_not_called()
            self.assertFalse((root / 'other').exists())

    def test_source_plan_builder_and_wrapper_have_successor_pins(self):
        self.assertEqual(control.PLAN_SHA, '51cebb1d8e2085e7c9a7bf5a37274f34dfbdbf28d14ac5ab27d06c26c3b53dc1')
        self.assertEqual(control.identity(ROOT / control.BUILDER)['sha256'], control.BUILDER_SHA)
        wrapper = ROOT / 'analysis_outputs/grpo_incumbent_20260907_v1/run_native_holdout_successor_audit_v1.py'
        if wrapper.exists():
            self.assertEqual(control.identity(wrapper)['sha256'], control.AUDIT_WRAPPER_SHA)
        self.assertEqual(control.RETRY_DECLARATION_SHA, '61a15f9d0acd951b12d529dc9f508c8bdb788fd6150f3498fa99767ad158cc79')

    def test_postprocessing_resolver_imports_only_its_immutable_v9_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'v9'; (source / 'scripts').mkdir(parents=True)
            helper = source / control.POSTPROCESSING_HELPER; helper.write_text('immutable v9 helper')
            manifest = source / 'SOURCE_MANIFEST.sha256'; manifest.write_text('v9 inventory')
            with patch.object(control, 'POSTPROCESSING_SOURCE', source), \
                    patch.object(control, 'POSTPROCESSING_SOURCE_SHA', control.identity(manifest)['sha256']), \
                    patch.object(control, 'POSTPROCESSING_HELPER_SHA', control.identity(helper)['sha256']), \
                    patch.object(control, 'POSTPROCESSING_DECLARATION_SHA', 'a' * 64), \
                    patch.object(control, 'module', return_value=SimpleNamespace(__file__=str(helper))) as loader:
                control.postprocessing_control()
                self.assertEqual(loader.call_args.args[1], helper)
                loader.return_value = SimpleNamespace(__file__=str(Path(temporary) / 'v10' / control.POSTPROCESSING_HELPER))
                with self.assertRaisesRegex(ValueError, 'another management snapshot'):
                    control.postprocessing_control()
                helper.write_text('changed helper')
                with self.assertRaisesRegex(ValueError, 'helper differs'):
                    control.postprocessing_control()
            with patch.object(control, 'POSTPROCESSING_DECLARATION_SHA', 'PENDING'):
                with self.assertRaisesRegex(ValueError, 'not pinned'):
                    control.postprocessing_control()


class DevelopmentRevalidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = {}
        for name in ('helper', 'declaration', 'retry_declaration', 'provision', 'failed_error', 'failed_launch', 'retry_event', 'original_event',
                     'terminal', 'index', 'launch', 'execution', 'details', 'run_config', 'comparison'):
            path = self.root / name; path.write_text(name); self.paths[name] = path
        self.amendment_names = ('helper', 'declaration', 'retry_declaration', 'provision', 'failed_error', 'failed_launch', 'retry_event')
        self.development = {'declaration': control.identity(self.paths['declaration']),
            'retry_declaration': control.identity(self.paths['retry_declaration']),
            'amendment_bindings': {name: control.identity(self.paths[name]) for name in self.amendment_names},
            'original_event_bindings': {'original': control.identity(self.paths['original_event'])},
            'comparison': control.identity(self.paths['comparison']),
            'files': {name: control.identity(self.paths[name]) for name in ('terminal', 'index', 'launch', 'execution', 'details', 'run_config')}}
        self.bridge = SimpleNamespace(admission_bindings=lambda sha: {name: control.identity(self.paths[name]) for name in self.amendment_names})
        self.native = SimpleNamespace(protocol_for=lambda root: ({}, control.TRAINING_SOURCE))

    def check(self):
        with patch.object(control, 'POSTPROCESSING_DECLARATION_SHA', self.development['declaration']['sha256']), \
                patch.object(control, 'RETRY_DECLARATION_SHA', self.development['retry_declaration']['sha256']), \
                patch.object(control, 'postprocessing_control', return_value=self.bridge), \
                patch.object(control, 'old_control', return_value=self.native), \
                patch.object(control, 'original_event_bindings', side_effect=lambda *a: {'original': control.identity(self.paths['original_event'])}):
            control.verify_development_admission({'development_evaluation_admission': self.development})

    def test_all_prebound_development_inputs_are_revalidated(self):
        self.check()
        for name in self.paths:
            with self.subTest(name=name):
                self.paths[name].write_text('mutated ' + name)
                with self.assertRaises(ValueError):
                    self.check()
                self.paths[name].write_text(name)

    def test_new_retry_or_original_event_cannot_be_hidden_after_predeclaration(self):
        with patch.object(control, 'POSTPROCESSING_DECLARATION_SHA', self.development['declaration']['sha256']), \
                patch.object(control, 'RETRY_DECLARATION_SHA', self.development['retry_declaration']['sha256']), \
                patch.object(control, 'postprocessing_control', return_value=SimpleNamespace(admission_bindings=lambda sha: {**self.development['amendment_bindings'], 'new_event': {'sha256': 'new'}})):
            with self.assertRaisesRegex(ValueError, 'event evidence changed'):
                control.verify_development_admission({'development_evaluation_admission': self.development})


class SuccessorModelAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.campaign = self.root / 'campaign'; self.source = self.root / 'science'
        (self.campaign / 'ledgers/terminal').mkdir(parents=True); self.source.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(control, 'TRAINING_CAMPAIGN', self.campaign).start()
        patch.object(control, 'TRAINING_SOURCE', self.source).start()
        self.plans = [{'key': task + '/' + method} for task in control.TASKS for method in control.METHODS]
        self.protocol = {'receipt_sha256': 'original', 'source_manifest': {'sha256': control.TRAINING_SOURCE_SHA},
                         'planned_train_commands': self.plans, 'planned_eval_commands': self.plans}
        (self.campaign / 'ledgers/protocol.json').write_text('original protocol bytes')
        declaration = self.root / 'declaration'; declaration.write_text('immutable amendment')
        self.declaration = control.identity(declaration)
        patch.object(control, 'RETRY_DECLARATION_SHA', self.declaration['sha256']).start()
        post_declaration = self.root / 'post_declaration'; post_declaration.write_text('immutable postprocessing amendment')
        self.post_declaration = control.identity(post_declaration)
        patch.object(control, 'POSTPROCESSING_DECLARATION_SHA', self.post_declaration['sha256']).start()
        self.post_keys = ('cvrp/foarl_f638_dev', 'mis/foarl_f638_dev')
        self.failed = {'cvrp/grpo_z': {'job_id': '1847633'}, 'cvrp/grpo_verified': {'job_id': '1847860'}, 'mis/grpo_z': {'job_id': '1847871'}}
        self.original_jobs, self.audits, self.resolved = {}, {}, {}
        names = ('added_tokens.json', 'merges.txt', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json')
        for i, plan in enumerate(self.plans):
            key = plan['key']; self.original_jobs[key] = self.failed[key]['job_id'] if key in self.failed else str(4000 + i)
            audit = {'job_id': str(2000 + i), 'checkpoint': {name: {'sha256': key.split('/')[0] + name} for name in names}}
            self.audits[key] = control.write_new(self.campaign / 'ledgers/terminal' / (key.replace('/', '_') + '.json'), audit)
            terminal = self.root / (key.replace('/', '_') + '_dev_terminal'); terminal.write_text(key)
            self.resolved[key] = {'key': key, 'job_id': str(8000 + i) if key in self.failed else self.original_jobs[key],
                 'replaced_pre_model_job_id': self.original_jobs[key] if key in self.failed else None,
                 'audit': {'status': 'terminal_evaluation_verified'}, 'files': {'terminal': control.identity(terminal)}}
            if key in self.post_keys:
                self.resolved[key].update(postprocessing_verified=True, scheduler_state='FAILED')
                self.resolved[key]['audit'] = {'status': 'complete_raw_evaluation_verified_after_exact_final_inventory_failure',
                                               'scheduler_state': 'FAILED', 'scheduler_exit_code': '1:0'}
        self.accounting = {'status': 'all_six_training_jobs_verified', 'protocol_sha256': 'original',
                           'terminal_audits': [control.identity(self.campaign / 'ledgers/terminal' / (p['key'].replace('/', '_') + '.json')) for p in self.plans]}
        control.write_new(self.campaign / 'ledgers/training_accounting.json', self.accounting)
        self.cost = {'allocated_gpu_hours': 2.0}; self.failed_cost = {'allocated_gpu_hours': 32 / 3600}
        self.post_cost = {'allocated_gpu_hours': 2679 / 3600}
        self.bindings = {'declaration': self.declaration, 'postprocessing_declaration': self.post_declaration}
        self.bridge = SimpleNamespace(DECLARATION=declaration, FAILED=self.failed,
             declaration=lambda sha: ({'management_source': {'sha256': control.RETRY_SOURCE_SHA}}, None, None, control.RETRY_SOURCE),
             resolve_evaluation=lambda key, **kwargs: self.resolved[key], admission_bindings=lambda sha: self.bindings,
             failed_allocation_records=lambda native: self.failed_cost)
        self.resolver = SimpleNamespace(DECLARATION=post_declaration, KEYS=self.post_keys,
             declaration=lambda sha: ({'management_source': {'path': str(control.POSTPROCESSING_SOURCE / 'SOURCE_MANIFEST.sha256'),
                                       'sha256': control.POSTPROCESSING_SOURCE_SHA}, 'retry_declaration': self.declaration}, None, None, None, None),
             resolve_evaluation=lambda key, **kwargs: self.resolved[key], admission_bindings=lambda sha: self.bindings,
             postprocessing_allocation_records=lambda *args: self.post_cost)
        self.native = SimpleNamespace(protocol_for=lambda root: (self.protocol, self.source), signed=control.checked,
              training_endpoint=lambda root, key, p, s: self.audits[key], submitted_job=lambda root, p, key, **kwargs: self.original_jobs[key],
              compare_actual_configuration=lambda *a: {'matched': True}, allocation_records=lambda *a, **k: self.cost)
        patch.object(control, 'retry_control', return_value=self.bridge).start()
        patch.object(control, 'postprocessing_control', return_value=self.resolver).start()
        patch.object(control, 'original_event_bindings', return_value={'all_original_events': 'unchanged'}).start()
        comparison = {'status': 'six_raw_arms_verified_with_preserved_pre_model_failures_and_postprocessing_only_foarl_admission',
                      'protocol_sha256': 'original', 'source_manifest': self.protocol['source_manifest'], 'retry_declaration': self.declaration,
                      'postprocessing_declaration': self.post_declaration,
                      'resolved_evaluations': [self.resolved[p['key']] for p in self.plans], 'amendment_bindings': self.bindings,
                      'successful_evaluation_accounting': self.cost, 'postprocessed_failed_foarl_allocation_accounting': self.post_cost,
                      'excluded_pre_model_failure_accounting': self.failed_cost}
        (self.campaign / 'comparison_postprocessing_v1').mkdir()
        control.write_new(self.campaign / 'comparison_postprocessing_v1/comparison.json', comparison)

    def test_six_endpoints_preserve_both_failed_and_successful_jobs_and_costs(self):
        result = control.admit_models(self.native)
        self.assertEqual(len(result['terminal_audits']), 6)
        self.assertEqual(result['excluded_pre_model_failure_allocations']['allocated_gpu_hours'], 32 / 3600)
        development = result['development_evaluation_admission']
        self.assertEqual(set(development['original_job_ids']) & set(development['resolved_job_ids']), {'4002', '4004', '4005'})
        self.assertTrue(set(item['job_id'] for item in self.failed.values()) <= set(development['original_job_ids']))
        self.assertTrue(set(item['job_id'] for item in self.failed.values()).isdisjoint(development['resolved_job_ids']))
        self.assertEqual(development['declaration'], self.post_declaration)
        self.assertEqual(development['retry_declaration'], self.declaration)
        self.assertEqual(result['postprocessed_failed_foarl_allocations'], self.post_cost)

    def test_original_healthy_job_cannot_be_replaced(self):
        self.resolved['mis/grpo_verified']['job_id'] = '9000'
        with self.assertRaisesRegex(ValueError, 'healthy original evaluation was replaced'):
            control.admit_models(self.native)

    def test_wrong_failed_job_scope_or_unverified_endpoint_is_rejected(self):
        self.resolved['cvrp/grpo_z']['replaced_pre_model_job_id'] = '1840000'
        with self.assertRaisesRegex(ValueError, 'failed and replacement'):
            control.admit_models(self.native)
        self.resolved['cvrp/grpo_z']['replaced_pre_model_job_id'] = '1847633'
        self.resolved['cvrp/grpo_z']['audit']['status'] = 'partial'
        with self.assertRaisesRegex(ValueError, 'did not verify'):
            control.admit_models(self.native)

    def test_final_comparison_and_failed_allocation_must_match_current_audits(self):
        self.failed_cost['allocated_gpu_hours'] = 0
        with self.assertRaisesRegex(ValueError, 'allocation accounting changed'):
            control.admit_models(self.native)

    def test_foarl_cannot_be_reclassified_as_successful_or_repeated_inference(self):
        key = self.post_keys[0]
        self.resolved[key]['scheduler_state'] = 'COMPLETED'
        with self.assertRaisesRegex(ValueError, 'FOARL must retain'):
            control.admit_models(self.native)
        self.resolved[key]['scheduler_state'] = 'FAILED'
        self.resolved[key]['job_id'] = 'new-inference-job'
        with self.assertRaisesRegex(ValueError, 'FOARL must retain'):
            control.admit_models(self.native)

    def test_postprocessing_failed_full_allocation_cannot_be_dropped(self):
        self.post_cost['allocated_gpu_hours'] = 0
        with self.assertRaisesRegex(ValueError, 'allocation accounting changed'):
            control.admit_models(self.native)
