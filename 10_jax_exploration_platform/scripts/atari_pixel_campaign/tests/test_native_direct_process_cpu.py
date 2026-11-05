"""Check native role startup and real own-child cleanup on exact owned CPU, without tools or GPUs."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import direct_profile_process as direct
import diagnose_direct_nsight_process as diagnostic
import native_process_evidence as evidence
import linux_pidfd_process_control as native
import nsight_process_identity as metadata


class ActualNativeCpuTests(unittest.TestCase):
    """Use only actual own CPU processes and explicitly temporary unit/lock fixtures."""
    def setUp(self):
        """Require the existing owned CPU allocation before any actual process-control fixture."""
        self.assertEqual(os.environ.get('SLURM_JOB_ID'), '6783003')
        self.assertTrue(os.environ.get('SLURM_STEP_ID', '').isdecimal())
        self.assertFalse(os.environ.get('SLURM_STEP_GPUS'))

    def plan(self, root):
        """Construct a labelled temporary metadata plan with actual current source hashes."""
        unit = root / 'data/unit';(unit / 'code').mkdir(parents=True)
        import hashlib
        plan = {'kind': direct.DIAGNOSTIC, 'python': sys.executable,
            'helpers': {'files': {direct.PREFIX + name: hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest()
                for name in diagnostic.required_helpers(True)}}}
        plan['native_process_control'] = evidence.declaration(plan)
        metadata.publish(unit / 'measurement_plan.json', plan)
        return unit, plan

    def test_actual_begin_calls_capture_and_portable_contract_reader(self):
        """The pure contract/call readers accept real own-self evidence and copied public bytes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);unit, plan = self.plan(root)
            owner = metadata.ProcReader().snapshot(os.getpid(), os.getuid())
            runtime = evidence.bootstrap(unit, plan, 'producer', metadata, owner)
            saved = metadata.read_json(unit / 'native_process_target_contract.json')
            contract = evidence.validate_contract(saved, plan, metadata)
            summary = evidence.validate_calls(runtime.provider.evidence(), owner, {owner['pid']: owner}, contract)
            self.assertEqual(summary['calls'], 3)
            self.assertEqual(len(runtime.provider.descriptor_events), 3)
            # Before: shared captures under the unit; after: explicit copies used through exact path mapping.
            portable = root / 'portable_public_files';portable.mkdir();mapping = {}
            for original in {ref['path'] for ref in saved['captured_files'].values()}:
                copied = portable / Path(original).name;shutil.copyfile(original, copied);mapping[original] = str(copied)
            self.assertEqual(evidence.validate_contract(saved, plan, metadata, mapping), contract)
            with patch.object(native, 'current_contract', side_effect=AssertionError('saved reader made native calls')):
                evidence.validate_contract(saved, plan, metadata, mapping)
                evidence.validate_calls(runtime.provider.evidence(), owner, {owner['pid']: owner}, contract)
            first = Path(next(iter(mapping.values())));first.write_bytes(b'explicit corrupted portable fixture')
            with self.assertRaises(ValueError):evidence.validate_contract(saved, plan, metadata, mapping)

    def test_actual_parent_handle_and_reused_descriptor_generations(self):
        """The native adapter keeps actual parent identity and separates sequential reuse of descriptor numbers."""
        contract, unused = native.current_contract();provider = native.NativePidfdProvider(contract)
        observer = native.NativeProcessObserver(metadata, provider)
        owner = metadata.ProcReader().snapshot(os.getpid(), os.getuid())
        for unused in range(3):
            handle = observer.parent_handle({'parent': owner})
            observer.close_process_handle(handle)
        calls = provider.evidence()
        summary = evidence.validate_calls(calls, owner, {owner['pid']: owner}, contract)
        self.assertEqual(summary['calls'], 9)
        generations = [event['open_generation'] for event in calls['descriptor_events'] if event['operation'] == 'pidfd_open']
        self.assertEqual(generations, [0, 3, 5, 7])
        malformed = copy.deepcopy(calls);malformed['descriptor_events'][-1]['open_generation'] = 0
        with self.assertRaises(ValueError):evidence.validate_calls(malformed, owner, {owner['pid']: owner}, contract)

    def test_failed_role_bootstrap_retains_partial_failure_without_tool_work(self):
        """A deliberately refused own-provider declaration publishes failure and no successful begin."""
        with tempfile.TemporaryDirectory() as temporary:
            unit, plan = self.plan(Path(temporary))
            owner = metadata.ProcReader().snapshot(os.getpid(), os.getuid())
            module = evidence.load_provider(plan)
            with patch.object(module, 'current_contract', side_effect=ValueError('explicit synthetic startup refusal')):
                with self.assertRaisesRegex(ValueError, 'synthetic startup refusal'):
                    evidence.bootstrap(unit, plan, 'producer', metadata, owner)
            failure = metadata.read_json(unit / 'native_process_producer_failure.json')
            self.assertFalse(failure['completed'])
            self.assertIsNone(failure['provider'])
            self.assertFalse((unit / 'native_process_producer_begin.json').exists())

    def regression(self, mode):
        """Run one fresh selected-Python CPU worker and retain its exact real process evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, '-I', '-S', '-B',
                str(SCRIPTS / 'tests/native_direct_tool_cpu_worker.py'), temporary, mode],
                capture_output=True, timeout=30, check=False, close_fds=True, pass_fds=())
            preserved = os.environ.get('NATIVE_DIRECT_CPU_REVIEW_OUTPUT')
            if preserved:
                target = Path(preserved) / mode;target.mkdir(parents=True, exist_ok=False)
                for name, raw in [('worker_stdout.raw', result.stdout), ('worker_stderr.raw', result.stderr)]:
                    (target / name).write_bytes(raw)
                for source in Path(temporary).rglob('*.json'):
                    if 'code' in source.parts:continue
                    destination = target / source.relative_to(temporary);destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
            self.assertEqual(result.returncode, 0, result.stderr.decode() + result.stdout.decode())
            record = json.loads(result.stdout)
            self.assertTrue(record['passed'])
            self.assertTrue(record['payload_exit_observed_by_independent_pidfd'])
            self.assertTrue(record['temporary_lock_reacquired_only_after_verified_descendant_exit'])
            self.assertEqual(record['native_provider']['open_handle_count'], 0)
            self.assertTrue(record['kernel_terminal']['actual_echild'])
            self.assertFalse(record['standing_gpu_lock_opened'])
            self.assertFalse(record['nsight_executed'])

    def test_actual_startup_cleanup_keeps_temporary_lock_until_child_exit(self):
        """A real startup failure cleans its positively owned CPU payload before temporary lock release."""
        self.regression('startup_timeout')

    def test_actual_orphan_cleanup_keeps_temporary_lock_until_kernel_terminal(self):
        """A real double-forked own CPU orphan is adopted and waited before temporary lock release."""
        self.regression('orphan_root_failure')


if __name__ == '__main__':
    unittest.main()
