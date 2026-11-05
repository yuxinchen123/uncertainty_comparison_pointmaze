"""Reject releases and saved native role graphs that cross their original work/total deadline."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import direct_profile_process as direct
import diagnose_direct_nsight_process as diagnostic
import native_process_evidence as evidence
import test_native_process_role_evidence as roles
from test_direct_diagnostic_time_budget import deadline


class ReleaseBoundaryTests(unittest.TestCase):
    """Mock only supporting ownership evidence; exercise real deadline predicates without a tool or device."""
    def test_late_existing_release_is_refused_before_acknowledgment(self):
        """An already present release cannot bypass the final work-deadline check after the wait loop."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary);(directory / 'capture_release.json').write_text('explicit synthetic late release')
            plan = {'kind': direct.DIAGNOSTIC, 'diagnostic_time_budget': dict(direct.DIAGNOSTIC_BUDGET), 'maximum_active_seconds': 240}
            observer = Mock();observer.read_json.return_value = {'producer': {'pid': 100}}
            with patch.object(direct, 'runtime_observer', return_value=observer), \
                    patch.object(direct, 'read_diagnostic_deadline', return_value=deadline()), \
                    patch.object(direct, 'claim_self', return_value={'explicit_synthetic_claim': True}), \
                    patch.object(direct, 'producer_handle', return_value=777), \
                    patch.object(direct, 'adjacent_helper') as acknowledgment, \
                    patch.object(direct.time, 'monotonic', return_value=341.):
                with self.assertRaises(TimeoutError):direct.wait_release(directory, plan)
            acknowledgment.assert_not_called();observer.close_process_handle.assert_called_once_with(777)

    def test_late_payload_does_not_publish_or_complete(self):
        """A late release return cannot start payload work or create a successful native terminal."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = {'kind': direct.DIAGNOSTIC, 'diagnostic_time_budget': dict(direct.DIAGNOSTIC_BUDGET), 'maximum_active_seconds': 240}
            observer = Mock()
            with patch.object(diagnostic, 'read_plan', return_value=(directory, plan)), \
                    patch.object(direct, 'generic', return_value=observer), patch.object(direct, 'wait_release'), \
                    patch.object(direct, 'read_diagnostic_deadline', return_value=deadline()), \
                    patch.object(direct.time, 'monotonic', return_value=341.):
                with self.assertRaises(TimeoutError):diagnostic.probe(directory)
            observer.publish.assert_not_called()


class NativeBudgetRoleTests(unittest.TestCase):
    """Use the existing raw-checked synthetic role graph with an explicitly prevalidated time-origin object."""
    def setUp(self):
        """Retain all role/raw/call predicates; patch only the independent contract-byte gate when reading."""
        self.case = roles.NativeSavedRoleTests('test_three_roles_keep_exact_owned_opens_wait_references_and_chronology')
        self.case.setUp();self.addCleanup(self.case.doCleanups)
        case = self.case
        case.plan['diagnostic_time_budget'] = dict(direct.DIAGNOSTIC_BUDGET);case.plan['maximum_active_seconds'] = 240
        case.save('measurement_plan.json', case.plan)
        self.limit = deadline(0.)
        for role in evidence.ROLES:
            begin, terminal = case.begins[role], case.terminals[role]
            begin['plan'] = roles.M.receipt(case.directory / 'measurement_plan.json')
            with patch.object(direct.time, 'monotonic', return_value=begin['published_at_seconds']):
                begin['diagnostic_deadline'] = self.limit.snapshot()
            paths = evidence.role_files(case.directory, role);case.save(paths['begin'].name, begin)
            terminal['begin'] = roles.M.receipt(paths['begin'])
            with patch.object(direct.time, 'monotonic', return_value=terminal['completed_at_seconds']):
                terminal['diagnostic_deadline'] = self.limit.snapshot()
            case.save(paths['terminal'].name, terminal)
        payload = roles.M.read_json(case.directory / 'probe_result.json')
        with patch.object(direct.time, 'monotonic', return_value=payload['payload_finished_at_seconds']):
            payload['diagnostic_deadline'] = self.limit.snapshot()
        case.save('probe_result.json', payload)
        case.terminals['capture']['references']['probe_result.json'] = roles.M.receipt(case.directory / 'probe_result.json')
        case.save(evidence.role_files(case.directory, 'capture')['terminal'].name, case.terminals['capture'])

    def read(self):
        """Read real graph predicates using a synthetic predecessor origin and patched contract-byte gate."""
        case = self.case
        with patch.object(evidence, 'validate_contract', return_value=case.contract):
            return evidence.validate_saved(case.directory, case.plan, roles.M, case.intent, case.observation,
                case.launch, case.waited, diagnostic_deadline=self.limit)

    def test_complete_budgeted_roles_and_payload_are_accepted_without_native_calls(self):
        """All three retained snapshots and payload time remain within the same synthetic predecessor budget."""
        self.assertTrue(self.read()['passed'])

    def test_payload_after_work_deadline_refuses_even_with_updated_hash(self):
        """A fresh payload hash cannot permit work after240 seconds on the shared clock."""
        case = self.case;payload = roles.M.read_json(case.directory / 'probe_result.json')
        payload['diagnostic_deadline']['observed_monotonic'] = 241.
        case.save('probe_result.json', payload)
        case.terminals['capture']['references']['probe_result.json'] = roles.M.receipt(case.directory / 'probe_result.json')
        case.save(evidence.role_files(case.directory, 'capture')['terminal'].name, case.terminals['capture'])
        with self.assertRaises(ValueError):self.read()

    def test_late_terminal_or_changed_clock_domain_refuses(self):
        """A native terminal cannot exceed300 seconds or substitute a different boot's clock."""
        case = self.case;paths = evidence.role_files(case.directory, 'producer')
        original = copy.deepcopy(case.terminals['producer'])
        changed = copy.deepcopy(original);changed['diagnostic_deadline']['observed_monotonic'] = 301.
        case.save(paths['terminal'].name, changed)
        with self.assertRaises(ValueError):self.read()
        changed = copy.deepcopy(original);changed['diagnostic_deadline']['clock_domain']['hostname'] = 'different-host'
        case.save(paths['terminal'].name, changed)
        with self.assertRaises(ValueError):self.read()


if __name__ == '__main__':
    unittest.main()
