"""Classify slow kernel-closure and custody checks using their final retained deadline snapshot."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_direct_diagnostic_time_budget as tested

direct, scope_source = tested.direct, tested.scope_source


class LateDiagnosticCompletionTests(unittest.TestCase):
    """Mock verified closure/custody only; use no actual process-control or GPU descriptor."""
    def exercise(self, closure_time, custody_time, expected_within):
        """Retain the exact source terminal produced after a synthetic slow closure or custody read."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tested.M.publish(directory / 'tool_descendant_scope_begin.json', {'explicit_synthetic_begin': True})
            scope = scope_source.OwnedToolScope.__new__(scope_source.OwnedToolScope)
            scope.directory, scope.deadline = directory, tested.deadline()
            scope.pid, scope.birth, scope.failed, scope.closed = 100, '10', False, False
            scope.read_errors, scope.budget_inconclusive_published = [], False
            scope.grace_seconds, scope.active, scope.previous = 120, True, {'value': 0}
            scope.request_counts_before = dict(scope_source._REQUEST_COUNTS)
            scope.waits, scope.nodes, scope.events, scope.interventions = {}, {}, [], []
            scope.handles, scope.producer_handle = {}, 777
            clock, native_flag, custody_reads = [350.], [1], [0]
            def custody():
                custody_reads[0] += 1
                if custody_reads[0] > 1:clock[0] = custody_time
                return {'explicit_synthetic_verified_custody': True}
            def closure():
                clock[0] = closure_time
                return {'explicit_synthetic_verified_kernel_closure': True}
            def subreaper(value=None):
                if value is not None:native_flag[0] = value
                return {'value': native_flag[0], 'returncode': 0, 'errno': 0}
            scope.check_descriptor, scope.closure = custody, closure
            scope.scan, scope.progress, scope.signal_known = Mock(), Mock(), Mock()
            scope.observer = SimpleNamespace(publish=tested.M.publish, receipt=tested.M.receipt,
                close_process_handle=Mock())
            with patch.object(direct.time, 'monotonic', side_effect=lambda: clock[0]), \
                    patch.object(scope_source, 'child_subreaper', side_effect=subreaper):
                scope.finish(False)
            terminal = tested.M.read_json(directory / 'tool_descendant_scope_terminal.json')
            self.assertIs(terminal['within_diagnostic_budget'], expected_within)
            self.assertEqual(terminal['diagnostic_deadline']['observed_monotonic'], custody_time)
            self.assertTrue(scope.closed)
            scope.signal_known.assert_not_called()
            scope.observer.close_process_handle.assert_called_once_with(777)
            self.assertEqual(scope.budget_inconclusive_published, not expected_within)
            if not expected_within:
                self.assertTrue(scope.failed)
                inconclusive = tested.M.read_json(directory / 'tool_descendant_budget_inconclusive.json')
                self.assertFalse(inconclusive['completed'])
                self.assertTrue(inconclusive['kernel_closure_verified'])
                with self.assertRaises(RuntimeError):scope.__exit__(None, None, None)
            else:
                self.assertFalse(scope.failed)
            output = os.environ.get('LATE_DIAGNOSTIC_COMPLETION_REVIEW_OUTPUT')
            if output:
                path = Path(output);path.mkdir(parents=True, exist_ok=True)
                value = {'synthetic_time_closure_custody_only': True, 'actual_process_control_or_GPU_descriptor_operations': 0,
                    'closure_monotonic': closure_time, 'custody_monotonic': custody_time, 'terminal': terminal,
                    'completed_within_budget': expected_within, 'scientific_credit': 0, 'admission_granted': False}
                target = path / ('closure-' + str(closure_time) + '_custody-' + str(custody_time) + '.json')
                with target.open('x') as stream:json.dump(value, stream, indent=2, sort_keys=True);stream.write('\n')

    def test_exact_rejected_late_closure_is_inconclusive(self):
        """The review399 closure returning401 after deadline400 cannot claim within-budget completion."""
        self.exercise(401., 401., False)

    def test_final_custody_read_crossing_deadline_is_inconclusive(self):
        """Closure before the limit does not hide a final custody read that returns after expiry."""
        self.exercise(399., 401., False)

    def test_exact_deadline_return_is_not_within_the_remaining_time_window(self):
        """A remaining-time value of zero remains expired at the terminal classification boundary."""
        self.exercise(400., 400., False)

    def test_fast_verified_closure_and_custody_remain_within_budget(self):
        """The positive case keeps the final399 snapshot and closes only after verified closure."""
        self.exercise(399., 399., True)


if __name__ == '__main__':
    unittest.main()
