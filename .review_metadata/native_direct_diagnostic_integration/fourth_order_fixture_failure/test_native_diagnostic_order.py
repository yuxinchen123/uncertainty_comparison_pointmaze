"""Check diagnostic startup ordering with labelled metadata-only role mocks."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose_direct_nsight_process as diagnostic


class NativeDiagnosticOrderTests(unittest.TestCase):
    """Refuse tool work before the explicit native bootstrap and keep default guard ordering."""
    def run_role(self, role, native):
        """Record function order without starting a process, library, tool, lock or actual role."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = {'native_process_control': {'explicit_synthetic_selection': True}} if native else {}
            (directory / 'measurement_plan.json').write_text(json.dumps(plan))
            events = []
            runtime = SimpleNamespace(observer=object(), complete=Mock(), failure=Mock(), files={'failure': directory / 'failure'})
            def read(value):
                events.append('read_plan');return directory, plan
            def start(value, actual_plan, actual_role):
                events.append('bootstrap_' + actual_role);return runtime
            def guard():
                events.append('guard')
            def work(*arguments, **keywords):
                events.append('work');return 0
            library = SimpleNamespace(prepare_native_control=lambda: events.append('prepare_prctl'))
            argv = ['explicit_metadata_fixture', '--' + role + '-unit', str(directory)]
            with patch.object(sys, 'argv', argv), patch.object(diagnostic, 'read_plan', side_effect=read), \
                    patch.object(diagnostic, 'native_role', side_effect=start) as bootstrap, \
                    patch.object(diagnostic, 'metadata_guard', side_effect=guard), \
                    patch.object(diagnostic.direct, 'context'), patch.object(diagnostic.direct, 'adjacent_helper', return_value=library), \
                    patch.object(diagnostic, 'producer', side_effect=work), patch.object(diagnostic, 'probe', side_effect=work), \
                    patch.object(diagnostic.direct, 'run_waiter', side_effect=work):
                diagnostic.main()
            self.assertLess(events.index('guard'), events.index('work'))
            if native:
                actual_role = 'producer' if role == 'producer' else 'waiter' if role == 'direct-owned-waiter' else 'capture'
                self.assertLess(events.index('bootstrap_' + actual_role), events.index('guard'))
                self.assertEqual(bootstrap.call_count, 1)
            else:
                bootstrap.assert_not_called()
                if role != 'producer':self.assertLess(events.index('guard'), events.index('read_plan'))

    def test_native_provider_precedes_guard_once_in_each_actual_role(self):
        """Every explicit role bootstraps once before the library guard and its later work."""
        for role in ('producer', 'direct-owned-waiter', 'direct-owned-capture'):
            with self.subTest(role=role):self.run_role(role, True)

    def test_default_child_guard_order_and_absent_native_bootstrap_are_preserved(self):
        """The default roles keep the original APIs and guard-before-child-reader order."""
        for role in ('producer', 'direct-owned-waiter', 'direct-owned-capture'):
            with self.subTest(role=role):self.run_role(role, False)


if __name__ == '__main__':
    unittest.main()
