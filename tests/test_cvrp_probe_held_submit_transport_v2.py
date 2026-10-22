"""Actual durable files and mocked RPC boundary; never a native submission."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'experiments/rloo_sft_anchor_20261013/cvrp_probe_execution_v2/native_prepared_v2'
PROGRAM = PACKET.parent / 'submit_held_native_v2.py'
PROGRAM_SHA = '28b03eb26a2cd69b6de9a6d0e2675729d32ab2a40576d0684b5e596859ee1445'


class DurableHeldTransport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rloo-held-rpc-fixture-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for stem in ('source_v1', 'control_v1'):
            shutil.copytree(PACKET / stem, self.root / stem)
        raw = PROGRAM.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), PROGRAM_SHA)
        self.module = types.ModuleType('_fixture_held_transport')
        self.module.__file__ = str(PROGRAM)
        exec(compile(raw, str(PROGRAM), 'exec', dont_inherit=True), self.module.__dict__)
        self.module.CONTROL = str(self.root / 'control_v1')
        self.module.SOURCE = str(self.root / 'source_v1')
        self.request = json.loads((PACKET / 'held_submission_request_v2.json').read_bytes())
        old_namespace = self.request['source_root'].rsplit('/', 1)[0]
        self.request['source_root'] = self.module.SOURCE
        self.request['control_root'] = self.module.CONTROL
        for row in self.request['staged_bindings']:
            row['path'] = row['path'].replace(old_namespace, str(self.root), 1)
        self.request['intent_binding']['path'] = self.module.CONTROL + '/submission_intent.json'
        intent = json.loads(base64.b64decode(self.request['intent_base64']))
        intent['submission_argv'] = ['sbatch', '--parsable', self.module.CONTROL + '/job.slurm']
        intent_raw = json.dumps(intent, sort_keys=True, indent=2).encode() + b'\n'
        self.request['intent_base64'] = base64.b64encode(intent_raw).decode()
        self.request['intent_binding'].update(sha256=hashlib.sha256(intent_raw).hexdigest(), size_bytes=len(intent_raw))
        self.calls = []
        self.rpc_result = subprocess.CompletedProcess([], 0, '424242\n', '')
        def mocked_rpc(argv, **kwargs):
            self.calls.append((argv, kwargs))
            if isinstance(self.rpc_result, BaseException):
                raise self.rpc_result
            return self.rpc_result
        self.module.subprocess = types.SimpleNamespace(run=mocked_rpc, TimeoutExpired=subprocess.TimeoutExpired)

    def rebind(self, relative):
        path = self.root / relative
        raw = path.read_bytes()
        row = next(row for row in self.request['staged_bindings'] if row['path'] == str(path))
        row.update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))

    def reject_without_rpc(self):
        with self.assertRaises((ValueError, KeyError)):
            self.module.run(self.request)
        self.assertEqual(self.calls, [])

    def test_success_persists_actual_intent_action_and_denies_duplicate(self):
        result = self.module.run(self.request)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result['action']['stdout'], '424242\n')
        self.assertFalse(result['native_job_released'])
        self.assertFalse(result['automatic_retry'])
        self.assertTrue((self.root / 'control_v1/submission_intent.json').is_file())
        action_raw = (self.root / 'control_v1/submission_action.json').read_bytes()
        self.assertEqual(hashlib.sha256(action_raw).hexdigest(), result['action_binding']['sha256'])
        with self.assertRaises(ValueError):
            self.module.run(self.request)
        self.assertEqual(len(self.calls), 1)

    def test_rpc_timeout_retains_ambiguous_cost_lineage_and_cannot_retry(self):
        self.rpc_result = subprocess.TimeoutExpired(['sbatch'], 25, output=b'')
        result = self.module.run(self.request)
        self.assertIsNone(result['action']['returncode'])
        self.assertTrue(result['action']['submission_ambiguous_no_retry'])
        with self.assertRaises(ValueError):
            self.module.run(self.request)
        self.assertEqual(len(self.calls), 1)

    def test_rpc_exception_retains_ambiguity_and_cannot_retry(self):
        self.rpc_result = OSError('diagnostic transport failure')
        result = self.module.run(self.request)
        self.assertTrue(result['action']['submission_ambiguous_no_retry'])
        with self.assertRaises(ValueError):
            self.module.run(self.request)
        self.assertEqual(len(self.calls), 1)

    def test_nonzero_rpc_is_recorded_once(self):
        self.rpc_result = subprocess.CompletedProcess([], 1, '', 'diagnostic refusal')
        result = self.module.run(self.request)
        self.assertEqual(result['action']['returncode'], 1)
        with self.assertRaises(ValueError):
            self.module.run(self.request)
        self.assertEqual(len(self.calls), 1)

    def test_uncertain_success_stderr_is_preserved_without_release(self):
        self.rpc_result = subprocess.CompletedProcess([], 0, '424242\n', 'diagnostic warning')
        result = self.module.run(self.request)
        self.assertEqual(result['action']['stderr'], 'diagnostic warning')
        self.assertFalse(result['native_job_released'])
        self.assertEqual(len(self.calls), 1)

    def test_duplicate_source_members_cannot_hide_required_files(self):
        self.request['staged_bindings'][-1] = copy.deepcopy(self.request['staged_bindings'][0])
        self.reject_without_rpc()

    def test_dotdot_path_denied_before_read_or_rpc(self):
        self.request['staged_bindings'][0]['path'] = self.module.SOURCE + '/../control_v1/spec.json'
        self.reject_without_rpc()

    def test_rebound_modified_source_still_denied_by_frozen_manifest(self):
        relative = 'source_v1/src/clasp/entrypoints/train_sft_anchor_pilot_v2.py'
        with (self.root / relative).open('ab') as stream:
            stream.write(b'\n# diagnostic tamper\n')
        self.rebind(relative)
        self.reject_without_rpc()

    def test_rebound_modified_script_with_hold_still_denied(self):
        relative = 'control_v1/job.slurm'
        path = self.root / relative
        path.write_bytes(path.read_bytes().replace(b'#SBATCH --gres=gpu:4', b'#SBATCH --gres=gpu:8'))
        self.rebind(relative)
        self.reject_without_rpc()

    def test_rebound_modified_authority_still_denied(self):
        relative = 'control_v1/authority.json'
        path = self.root / relative
        value = json.loads(path.read_bytes())
        value['registered_utc'] = '2000-01-01T00:00:00+00:00'
        path.write_text(json.dumps(value))
        self.rebind(relative)
        self.reject_without_rpc()

    def test_source_link_denied_before_rpc(self):
        path = self.root / 'control_v1/spec.json'
        target = self.root / 'diagnostic-linked-spec.json'
        path.rename(target)
        path.symlink_to(target)
        self.reject_without_rpc()

    def test_intent_directory_sync_failure_prevents_rpc_and_prevents_retry(self):
        def fail_sync(_):
            raise OSError('diagnostic directory durability failure')
        self.module.sync_directory = fail_sync
        with self.assertRaises(OSError):
            self.module.run(self.request)
        self.assertEqual(self.calls, [])
        self.assertTrue((self.root / 'control_v1/submission_intent.json').is_file())
        with self.assertRaises(ValueError):
            self.module.run(self.request)
        self.assertEqual(self.calls, [])

    def test_fixed_script_changed_after_intent_denies_rpc_without_retry(self):
        original = self.module.sync_directory
        def mutate_after_intent(path):
            original(path)
            script = self.root / 'control_v1/job.slurm'
            script.write_bytes(script.read_bytes() + b'\n# diagnostic post-intent tamper\n')
        self.module.sync_directory = mutate_after_intent
        with self.assertRaises(ValueError):
            self.module.run(self.request)
        self.assertEqual(self.calls, [])
        self.assertTrue((self.root / 'control_v1/submission_intent.json').is_file())

    def test_action_directory_sync_failure_preserves_intent_and_blocks_second_rpc(self):
        original = self.module.sync_directory
        count = []
        def fail_second(path):
            count.append(path)
            if len(count) == 2:
                raise OSError('diagnostic action durability failure')
            original(path)
        self.module.sync_directory = fail_second
        with self.assertRaises(OSError):
            self.module.run(self.request)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue((self.root / 'control_v1/submission_action.json').is_file())
        with self.assertRaises(ValueError):
            self.module.run(self.request)
        self.assertEqual(len(self.calls), 1)


if __name__ == '__main__':
    unittest.main()
