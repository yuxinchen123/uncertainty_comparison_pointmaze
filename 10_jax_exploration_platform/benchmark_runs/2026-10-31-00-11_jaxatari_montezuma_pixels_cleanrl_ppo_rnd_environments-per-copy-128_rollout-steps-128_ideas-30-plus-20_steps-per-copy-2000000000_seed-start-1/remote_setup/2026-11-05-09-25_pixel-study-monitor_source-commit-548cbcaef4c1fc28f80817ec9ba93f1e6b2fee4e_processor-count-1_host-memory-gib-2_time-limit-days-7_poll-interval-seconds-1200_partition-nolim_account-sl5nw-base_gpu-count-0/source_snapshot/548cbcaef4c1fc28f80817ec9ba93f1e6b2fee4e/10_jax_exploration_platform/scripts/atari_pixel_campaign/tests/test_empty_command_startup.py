"""Test empty-command waiting with synthetic clocks and fresh own CPU children."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('empty_startup_metadata_controller', SCRIPTS / 'direct_common_timing_controller.py')
core = importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(core)


def with_command(value, command):
    result = copy.deepcopy(value)
    raw = b'\0'.join(item.encode() for item in command) + b'\0' if command else b''
    result.update(command=command, command_bytes_hex=raw.hex(), command_sha256=core.digest(raw))
    return result


class EmptyCommandStartupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.unit = Path(self.temporary.name)
        self.actual = core.read_process(os.getpid(), os.getuid())
        self.expected = ['explicit_synthetic_metadata_child']
        self.empty = with_command(self.actual, [])
        self.exact = with_command(self.actual, self.expected)
        self.clock = [1.]
        self.sleeps, self.reads = [], []
        self.child = SimpleNamespace(pid=self.actual['pid'], poll=lambda: None)

    def run_case(self, next_value, *, initial=None, maximum=.03):
        def sleep(seconds):
            self.sleeps.append(seconds);self.clock[0] += seconds
        def observe(pid, uid):
            self.reads.append((pid, uid))
            return next_value() if callable(next_value) else copy.deepcopy(next_value)
        return core.wait_for_exact_child_startup(self.child, initial or self.empty, self.expected,
            self.actual['affinity'], self.actual['parent_pid'], os.getuid(), self.unit,
            started_monotonic=1., maximum_seconds=maximum, clock=lambda:self.clock[0], sleeper=sleep, observer=observe)

    def saved(self):
        return json.loads((self.unit / 'direct_controller_startup_observation_sequence.json').read_bytes())

    def test_empty_then_exact_retains_first_empty_and_requires_positive_identity(self):
        self.assertEqual(self.run_case(self.exact), self.exact)
        proof = self.saved()
        self.assertEqual([item['raw_command_is_empty'] for item in proof['observations']], [True, False])
        self.assertTrue(proof['exact_command_observed'])
        self.assertFalse(proof['registration_performed_by_this_observer'])

    def test_initial_exact_never_polls_another_observation_or_sleeps(self):
        self.assertEqual(self.run_case(self.empty, initial=self.exact), self.exact)
        self.assertEqual(self.sleeps, []);self.assertEqual(self.reads, [])

    def test_unexpected_nonempty_command_refuses_without_wait(self):
        wrong = with_command(self.actual, ['unexpected'])
        with self.assertRaisesRegex(ValueError, 'unexpected nonempty'):
            self.run_case(self.exact, initial=wrong)
        self.assertEqual(self.sleeps, []);self.assertEqual(self.reads, [])
        self.assertFalse(self.saved()['exact_command_observed'])

    def test_empty_then_wrong_command_refuses(self):
        with self.assertRaisesRegex(ValueError, 'unexpected nonempty'):
            self.run_case(with_command(self.actual, ['unexpected']))
        self.assertFalse(self.saved()['exact_command_observed'])

    def test_trailing_empty_argument_is_not_hidden_by_decoding(self):
        changed = copy.deepcopy(self.exact)
        raw = bytes.fromhex(changed['command_bytes_hex']) + b'\0'
        changed.update(command_bytes_hex=raw.hex(), command_sha256=core.digest(raw))
        with self.assertRaisesRegex(ValueError, 'exact expected bytes'):
            self.run_case(self.exact, initial=changed)
        self.assertEqual(self.sleeps, []);self.assertEqual(self.reads, [])
        proof = self.saved()
        self.assertFalse(proof['exact_command_observed'])
        self.assertFalse(proof['observations'][0]['expected_raw_command_matches'])

    def test_each_changed_identity_refuses_before_registration(self):
        changes = [('pid', self.actual['pid'] + 1), ('process_start_time', str(int(self.actual['process_start_time'])+1)),
            ('parent_pid', self.actual['parent_pid']+1), ('affinity', self.actual['affinity']+[100000]), ('uid', 0)]
        for key, value in changes:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                self.unit = Path(temporary);self.clock[0] = 1.
                changed = {**self.exact, key:value}
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    self.run_case(changed)
                self.assertFalse(self.saved()['exact_command_observed'])
        with tempfile.TemporaryDirectory() as temporary:
            self.unit = Path(temporary);self.clock[0] = 1.
            changed = copy.deepcopy(self.exact);changed['uid_bytes'] = 'Uid:\t0\t0\t0\t0\n'
            changed['uid_sha256'] = core.digest(changed['uid_bytes'].encode())
            with self.assertRaisesRegex(ValueError, 'identity changed'):self.run_case(changed)

    def test_exit_while_empty_precedes_another_proc_read(self):
        values = iter([None, 1]);self.child.poll = lambda:next(values)
        with self.assertRaisesRegex(ValueError, 'exited'):
            self.run_case(self.exact)
        self.assertEqual(self.reads, [])
        self.assertEqual(self.saved()['observations'][0]['after_empty_poll']['actual_popen_returncode'], 1)

    def test_empty_timeout_and_late_exact_both_refuse(self):
        with self.assertRaisesRegex(ValueError, 'deadline'):self.run_case(self.empty)
        self.assertFalse(self.saved()['exact_command_observed'])
        with tempfile.TemporaryDirectory() as temporary:
            self.unit = Path(temporary);self.clock[0] = 1.
            def late():
                self.clock[0] = 1.04
                return self.exact
            with self.assertRaisesRegex(ValueError, 'deadline'):self.run_case(late)
            proof = self.saved()
            self.assertFalse(proof['exact_command_observed'])
            self.assertEqual(proof['observations'][-1]['observed_child']['command'], self.expected)

    def test_false_empty_raw_bytes_and_more_than_thirty_seconds_refuse(self):
        changed = with_command(self.actual, ['nonempty']);changed['command'] = []
        with self.assertRaisesRegex(ValueError, 'identity changed'):self.run_case(self.exact, initial=changed)
        with self.assertRaises(ValueError):self.run_case(self.exact, maximum=31)

    def test_actual_selected_python_isolated_CPU_child_passes_exact_guard_and_wait(self):
        original = sorted(os.sched_getaffinity(0));chosen = core.chosen_affinity(original, 1)
        child = None
        try:
            os.sched_setaffinity(0, chosen)
            command = ['/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python', '-I', '-S', '-B', '-c',
                'import select,sys; r,_,_=select.select([sys.stdin.buffer],[],[],3); sys.exit(0 if r and sys.stdin.buffer.read(1)==b"R" else 2)']
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env={'PATH':'/usr/local/bin:/usr/bin:/bin','LANG':'C.UTF-8'}, close_fds=True)
            started = time.monotonic();initial = core.read_process(child.pid, os.getuid())
            exact = core.wait_for_exact_child_startup(child, initial, command, chosen, os.getpid(), os.getuid(),
                self.unit, started_monotonic=started)
            self.assertEqual(exact['command'], command)
            child.stdin.write(b'R');child.stdin.flush();child.stdin.close();child.stdin = None
            stdout, stderr = child.communicate(timeout=5)
            self.assertEqual((child.returncode, stdout, stderr), (0, b'', b''))
        finally:
            if child is not None and child.returncode is None:
                child.stdin.close();child.stdin = None;child.communicate(timeout=5)
            os.sched_setaffinity(0, original)


if __name__ == '__main__':
    unittest.main()
