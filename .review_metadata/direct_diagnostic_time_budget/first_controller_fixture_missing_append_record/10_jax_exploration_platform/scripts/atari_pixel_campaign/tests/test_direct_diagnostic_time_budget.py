"""Check one shared diagnostic deadline with synthetic clocks and bounded own-process CPU timing."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import direct_profile_process as direct
import direct_tool_descendants as scope_source
import diagnose_direct_nsight_process as diagnostic
from test_nsight_process_identity import ProcessObservationTests, M

DOMAIN = {'hostname': 'serval02', 'boot_id': '11111111-1111-1111-1111-111111111111',
    'clock': 'monotonic', 'implementation': 'explicit synthetic monotonic clock'}


def deadline(start=100.):
    """Build a labelled synthetic time-only object, never an actual target or process qualification."""
    return direct.DiagnosticDeadline({'started_monotonic': start, 'clock_domain': dict(DOMAIN)},
        {'path': '/explicit/synthetic/origin.json', 'sha256': 'a' * 64})


class DeadlineArithmeticTests(unittest.TestCase):
    """Check finite absolute deadlines without waiting for actual three-minute intervals."""
    def test_common_work_limit_is_not_restarted_by_each_tool(self):
        """A later version/capture request receives only time left from the same original start."""
        limit = deadline()
        for now, expected in [(100., 15.), (330., 10.), (339., 1.)]:
            with self.subTest(now=now), patch.object(direct.time, 'monotonic', return_value=now):
                self.assertEqual(limit.timeout(15), expected)
                self.assertEqual(limit.total_end, 400.)
        with patch.object(direct.time, 'monotonic', return_value=340.), self.assertRaises(TimeoutError):limit.remaining('work')

    def test_cleanup_end_is_retained_once_and_capped_by_original_total(self):
        """Repeated cleanup checks cannot turn a sixty-second reservation into a fresh wait."""
        limit = deadline()
        with patch.object(direct.time, 'monotonic', return_value=350.):self.assertEqual(limit.begin_cleanup(), 400.)
        with patch.object(direct.time, 'monotonic', return_value=390.):
            self.assertEqual(limit.begin_cleanup(), 400.)
            self.assertEqual(limit.timeout(30, phase='cleanup'), 10.)
        with patch.object(direct.time, 'monotonic', return_value=400.):
            self.assertEqual(limit.remaining('cleanup', allow_zero=True), 0.)
            with self.assertRaises(TimeoutError):limit.remaining('cleanup')

    def test_saved_snapshot_refuses_extended_domain_or_cleanup_window(self):
        """Saved arithmetic cannot extend a deadline or borrow another boot's clock."""
        limit = deadline()
        with patch.object(direct.time, 'monotonic', return_value=330.):
            good = limit.snapshot();limit.verify_snapshot(good, phase='work')
        for field, value in [('total_end_monotonic', 700.), ('observed_monotonic', 401.), ('started_monotonic', True)]:
            changed = copy.deepcopy(good);changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):limit.verify_snapshot(changed, phase='total')
        changed = copy.deepcopy(good);changed['clock_domain']['boot_id'] = 'another boot'
        with self.assertRaises(ValueError):limit.verify_snapshot(changed, phase='work')
        changed = copy.deepcopy(good);changed.update(cleanup_started_monotonic=300., cleanup_end_monotonic=400.)
        with self.assertRaises(ValueError):limit.verify_snapshot(changed, phase='total', cleanup_required=True)

    def test_policy_is_diagnostic_only_and_strictly_typed(self):
        """A Boolean duration or formal plan cannot acquire the diagnostic's shared budget."""
        plan = {'kind': direct.DIAGNOSTIC, 'diagnostic_time_budget': dict(direct.DIAGNOSTIC_BUDGET), 'maximum_active_seconds': 240}
        self.assertEqual(direct.diagnostic_budget(plan), direct.DIAGNOSTIC_BUDGET)
        self.assertIsNone(direct.diagnostic_budget({}))
        for field, value in [('work_seconds', True), ('total_seconds', 600), ('cleanup_seconds', 120)]:
            changed = copy.deepcopy(plan);changed['diagnostic_time_budget'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):direct.diagnostic_budget(changed)
        plan['kind'] = 'selected_configuration_profile'
        with self.assertRaises(ValueError):direct.diagnostic_budget(plan)

    def test_scope_wait_uses_existing_absolute_work_end(self):
        """A request for another300 seconds cannot extend a scope already near its240-second work end."""
        clock = [335.]
        scope = scope_source.OwnedToolScope.__new__(scope_source.OwnedToolScope)
        scope.deadline = deadline();scope.scan = lambda: clock.__setitem__(0, 341.)
        scope.record_wait = Mock()
        child = SimpleNamespace(poll=lambda: None)
        with patch.object(direct.time, 'monotonic', side_effect=lambda: clock[0]), self.assertRaises(ValueError):
            scope.wait(child, timeout=300)
        scope.record_wait.assert_not_called()

    def test_expired_cleanup_keeps_descriptor_and_does_not_signal_or_report_success(self):
        """Stop this synthetic fixture after one loop, before the real source's continued custody observation."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scope = scope_source.OwnedToolScope.__new__(scope_source.OwnedToolScope)
            scope.directory, scope.deadline = directory, deadline()
            scope.pid, scope.birth, scope.failed, scope.closed = 100, '10', False, False
            scope.read_errors, scope.budget_inconclusive_published = [], False
            scope.grace_seconds = 120
            scope.check_descriptor = Mock(return_value={'explicit_synthetic_custody': True})
            scope.scan, scope.signal_known, scope.closure, scope.progress = Mock(), Mock(), Mock(return_value=None), Mock()
            scope.observer = SimpleNamespace(publish=M.publish, close_process_handle=Mock())
            class StopFixture(Exception):
                """End only the synthetic test loop without making the production source claim closure."""
            with patch.object(direct.time, 'monotonic', return_value=401.), \
                    patch.object(scope_source, 'child_subreaper', return_value={'value': 1}), \
                    patch.object(scope_source.time, 'sleep', side_effect=StopFixture):
                with self.assertRaises(StopFixture):scope.finish(False)
            self.assertFalse(scope.closed)
            scope.signal_known.assert_not_called();scope.observer.close_process_handle.assert_not_called()
            record = M.read_json(directory / 'tool_descendant_budget_inconclusive.json')
            self.assertFalse(record['completed']);self.assertFalse(record['descriptor_release_attempted'])
            self.assertTrue(record['no_new_tool_or_signal_requests_after_expiry'])


class OriginAndControllerTests(ProcessObservationTests):
    """Use raw-checked private process fixtures for controller ordering, without actual child launches."""
    def setUp(self):
        """Retain the existing positive-identity fixture and a temporary diagnostic unit."""
        super().setUp()
        self.directory = self.root / 'data/unit';(self.directory / 'code').mkdir(parents=True)
        source = self.directory / 'code/diagnose_direct_nsight_process.py';source.write_bytes((SCRIPTS / source.name).read_bytes())
        self.plan = {'kind': direct.DIAGNOSTIC, 'python': '/synthetic/python', 'diagnostic_time_budget': dict(direct.DIAGNOSTIC_BUDGET),
            'maximum_active_seconds': 240, 'helpers': {'files': {direct.PREFIX + source.name: M.receipt(source)['sha256']}}}
        self.command = [self.plan['python'], '-I', '-S', '-B', str(source), '--controller-unit', str(self.directory)]
        (self.root / '100/cmdline').write_bytes(b'\0'.join(value.encode() for value in self.command) + b'\0')
        self.parent = self.reader.snapshot(100, 888, self.cgroup)
        M.publish(self.directory / 'measurement_plan.json', self.plan)
        self.clock = [160.]

    def test_origin_is_durable_before_popen_and_has_no_future_producer_identity(self):
        """The real publisher writes only verified parent/source/clock fields before any child exists."""
        with patch.object(direct, 'OWNER', 888), patch.object(direct, 'generic', return_value=M), \
                patch.object(direct, 'monotonic_clock_domain', return_value=DOMAIN), \
                patch.object(direct.time, 'monotonic', side_effect=lambda: self.clock[0]):
            limit = direct.publish_diagnostic_origin(self.directory, self.plan, self.parent, 100.)
            self.assertEqual(limit.remaining('total'), 240.)
        record = M.read_json(self.directory / 'diagnostic_controller_time_origin.json')
        self.assertEqual(record['controller']['pid'], 100)
        self.assertNotIn('producer', record);self.assertNotIn('producer_pid', record)
        self.assertEqual(record['started_monotonic'], 100.)

    def test_live_role_refuses_a_different_host_or_boot_domain(self):
        """A child cannot reuse correct origin bytes with a different current monotonic domain."""
        with patch.object(direct, 'OWNER', 888), patch.object(direct, 'generic', return_value=M), \
                patch.object(direct, 'monotonic_clock_domain', return_value=DOMAIN), \
                patch.object(direct.time, 'monotonic', side_effect=lambda: self.clock[0]):
            direct.publish_diagnostic_origin(self.directory, self.plan, self.parent, 100.)
            with patch.object(direct, 'monotonic_clock_domain', return_value={**DOMAIN, 'hostname': 'different-host'}):
                with self.assertRaises(ValueError):direct.read_diagnostic_deadline(self.directory, self.plan)

    def test_controller_timeout_has_no_extra_wait_or_holder_signal(self):
        """A late producer remains inconclusive; controller neither kills it nor adds a cleanup wait."""
        source = self.directory / 'code/diagnose_direct_nsight_process.py'
        command = [self.plan['python'], '-I', '-S', '-B', str(source), '--producer-unit', str(self.directory)]
        (self.root / '101/cmdline').write_bytes(b'\0'.join(value.encode() for value in command) + b'\0')
        producer = self.reader.snapshot(101, 888, self.cgroup)
        child = Mock(pid=101);child.poll.return_value = None
        def spawned(*arguments, **keywords):
            self.assertTrue((self.directory / 'diagnostic_controller_time_origin.json').exists())
            return child
        def waited(timeout):
            self.assertEqual(timeout, 240.)
            self.clock[0] = 401.
            raise subprocess.TimeoutExpired(command, timeout)
        child.wait.side_effect = waited
        with patch.object(direct, 'OWNER', 888), patch.object(direct, 'generic', return_value=M), \
                patch.object(direct, 'context', return_value=self.parent), patch.object(direct, 'no_standing_descriptor', return_value={}), \
                patch.object(direct, 'monotonic_clock_domain', return_value=DOMAIN), \
                patch.object(direct.time, 'monotonic', side_effect=lambda: self.clock[0]), \
                patch.object(diagnostic, 'read_plan', return_value=(self.directory, self.plan)), \
                patch.object(diagnostic, 'launch_review'), patch.object(diagnostic, 'append_producer', return_value={'pid': 101}), \
                patch.object(diagnostic.subprocess, 'Popen', side_effect=spawned), \
                patch.object(M, 'ProcReader', return_value=self.reader), \
                patch.object(diagnostic, 'controller_total_limit', side_effect=lambda value: __import__('contextlib').nullcontext()):
            with self.assertRaises(subprocess.TimeoutExpired):diagnostic.controller(self.directory, entry_monotonic=100.)
        child.wait.assert_called_once();child.terminate.assert_not_called();child.kill.assert_not_called()
        record = M.read_json(self.directory / 'controller_budget_inconclusive.json')
        self.assertFalse(record['completed']);self.assertFalse(record['producer_signal_sent'])
        self.assertEqual(producer['pid'], record['producer_pid'])


class ActualControllerTimerTests(unittest.TestCase):
    """Test only the current process's short real timer on owned CPU; no tool or GPU descriptor exists."""
    def test_actual_own_controller_alarm_returns_promptly_and_restores_handler(self):
        """The controller's total timer interrupts its own wait without signalling any other process."""
        self.assertEqual(os.environ.get('SLURM_JOB_ID'), '6783003')
        previous = signal.getsignal(signal.SIGALRM)
        limit = deadline(time.monotonic() - 299.95)
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            with diagnostic.controller_total_limit(limit):time.sleep(1)
        self.assertLess(time.monotonic() - started, .5)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))

    def test_actual_owned_cpu_cleanup_uses_one_synthetic_early_work_deadline(self):
        """Use the real cleanup code with a five-second remaining work budget and a temporary CPU lock."""
        self.assertEqual(os.environ.get('SLURM_JOB_ID'), '6783003')
        with tempfile.TemporaryDirectory() as temporary:
            started = time.monotonic()
            result = subprocess.run([sys.executable, '-I', '-S', '-B',
                str(SCRIPTS / 'tests/direct_diagnostic_budget_cpu_worker.py'), temporary, 'startup_timeout'],
                capture_output=True, timeout=30, check=False, close_fds=True, pass_fds=())
            preserved = os.environ.get('DIAGNOSTIC_BUDGET_CPU_REVIEW_OUTPUT')
            if preserved:
                target = Path(preserved);target.mkdir(parents=True, exist_ok=False)
                (target / 'stdout.raw').write_bytes(result.stdout);(target / 'stderr.raw').write_bytes(result.stderr)
                for source in Path(temporary).rglob('*.json'):
                    if 'code' in source.parts:continue
                    destination = target / source.relative_to(temporary);destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
            self.assertEqual(result.returncode, 0, result.stderr.decode() + result.stdout.decode())
            record = json.loads(result.stdout)
            self.assertTrue(record['passed'])
            self.assertTrue(record['kernel_terminal']['actual_echild'])
            self.assertTrue(record['temporary_lock_retained_until_all_descendants_exited'])
            self.assertTrue(record['temporary_lock_reacquired_only_after_verified_descendant_exit'])
            self.assertFalse(record['actual_target_qualification'])
            self.assertFalse(record['standing_gpu_lock_opened']);self.assertFalse(record['nsight_executed'])
            self.assertLess(time.monotonic() - started, 15.)


if __name__ == '__main__':
    unittest.main()
