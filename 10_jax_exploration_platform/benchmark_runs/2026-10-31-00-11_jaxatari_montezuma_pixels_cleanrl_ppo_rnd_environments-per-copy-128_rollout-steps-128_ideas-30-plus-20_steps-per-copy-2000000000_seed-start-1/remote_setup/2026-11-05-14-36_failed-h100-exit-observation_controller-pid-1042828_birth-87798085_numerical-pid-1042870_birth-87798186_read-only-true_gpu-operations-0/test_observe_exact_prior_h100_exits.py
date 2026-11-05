"""Use private synthetic proc files; never inspect the recorded target PIDs on this CPU node."""
import ast
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('read_only_prior_exit_observer_fixture', ROOT / 'observe_exact_prior_h100_exits.py')
observer = importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(observer)


def stat_text(pid, birth):
    """Make an explicitly synthetic Linux stat record with one declared process birth."""
    fields = ['S', '1', str(pid), str(pid), *(['0'] * 15), str(birth), *(['0'] * 30)]
    return str(pid) + ' (synthetic owned process) ' + ' '.join(fields) + '\n'


class PriorExitObserverTests(unittest.TestCase):
    """Require truthful absence, present and unknown outcomes without acting on any process."""
    def setUp(self):
        """Create a private filesystem fixture rather than a live proc namespace."""
        self.directory = tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.proc = Path(self.directory.name)

    def present(self, role, birth=None, status=None):
        """Write only clearly synthetic status/stat data for one fixed allowed role."""
        pid, expected = observer.IDENTITIES[role]
        path = self.proc / str(pid);path.mkdir()
        (path / 'status').write_text(status or 'Name:\tsynthetic\nUid:\t' + '\t'.join([str(observer.OWNER_UID)] * 4) + '\n')
        (path / 'stat').write_text(stat_text(pid, expected if birth is None else birth))
        return path

    def test_absent_roles_keep_real_current_observer_identity_and_utc(self):
        """Synthetic absence rows retain the literal expected identities and actual CPU observer fields."""
        rows = [observer.observe_process(role, proc_root=self.proc) for role in observer.IDENTITIES]
        record = observer.observation_record(rows)
        self.assertEqual(record['kind'], 'owned_prior_direct_process_exit_observation')
        self.assertEqual(record['hostname'], os.uname().nodename.split('.')[0])
        self.assertEqual((record['observer_pid'], record['observer_uid']), (os.getpid(), os.getuid()))
        self.assertEqual(datetime.fromisoformat(record['at_utc']).utcoffset(), timezone.utc.utcoffset(None))
        self.assertTrue(record['all_original_processes_exited']);self.assertTrue(record['all_expected_processes_absent'])
        for row in rows:
            self.assertEqual((row['expected_pid'], row['expected_birth']), observer.IDENTITIES[row['role']])
            self.assertEqual(row['observation'], 'absent');self.assertIs(row['original_process_exited'], True)

    def test_present_same_birth_is_not_an_exit(self):
        """A still-present exact own identity cannot become an absence row."""
        self.present('controller')
        row = observer.observe_process('controller', proc_root=self.proc)
        self.assertEqual(row['observation'], 'present_same_birth');self.assertIs(row['original_process_exited'], False)

    def test_reused_own_birth_is_reported_without_claiming_absence(self):
        """An owned reused PID proves a different birth while remaining unacceptable to the absent-only reader."""
        self.present('controller', birth='87798086')
        row = observer.observe_process('controller', proc_root=self.proc)
        self.assertEqual(row['observation'], 'different_owned_birth');self.assertIs(row['original_process_exited'], True)
        other = observer.observe_process('numerical_child', proc_root=self.proc)
        self.assertFalse(observer.observation_record([row, other])['all_expected_processes_absent'])

    def test_foreign_directory_owner_refuses_before_any_status_read(self):
        """A present PID under another owner is left unverified without reading its process files."""
        self.present('controller')
        with patch.object(observer.os, 'fstat', return_value=SimpleNamespace(st_uid=observer.OWNER_UID + 1)), \
            patch.object(observer, 'bounded', side_effect=AssertionError('no foreign process metadata')) as read:
            row = observer.observe_process('controller', proc_root=self.proc)
        read.assert_not_called();self.assertEqual(row['observation'], 'unverified');self.assertIsNone(row['original_process_exited'])

    def test_missing_or_duplicate_user_identifier_fields_remain_unknown(self):
        """Incomplete or duplicated owner fields cannot prove absence or reuse."""
        path = self.present('controller', status='Uid:\t1\t2\t3\n')
        self.assertEqual(observer.observe_process('controller', proc_root=self.proc)['observation'], 'unverified')
        (path / 'status').write_text('Uid:\t2618919\t2618919\t2618919\t2618919\n' * 2)
        self.assertEqual(observer.observe_process('controller', proc_root=self.proc)['observation'], 'unverified')

    def test_birth_change_during_read_remains_unknown(self):
        """A changing positive identity is refused instead of reclassified as a completed original process."""
        self.present('controller');pid, birth = observer.IDENTITIES['controller']
        status='Uid:\t2618919\t2618919\t2618919\t2618919\n'
        with patch.object(observer, 'bounded', side_effect=[status, stat_text(pid,birth), stat_text(pid,'87798086'), status]):
            row = observer.observe_process('controller', proc_root=self.proc)
        self.assertEqual(row['observation'], 'unverified');self.assertIsNone(row['original_process_exited'])

    def test_unavailable_proc_root_is_unknown(self):
        """A missing namespace cannot be mistaken for two absent prior processes."""
        row = observer.observe_process('controller', proc_root=self.proc / 'not_a_proc_namespace')
        self.assertEqual(row['observation'], 'unverified');self.assertIsNone(row['original_process_exited'])

    def test_actual_own_cpu_proc_namespace_is_readable(self):
        """Validate only this real CPU observer process, without reading either target PID."""
        observer.verify_own_proc_namespace()

    def test_unknown_role_and_runtime_arguments_refuse(self):
        """Only the literal two roles and argument-free runtime operation are supported."""
        with patch.object(observer.os, 'open', side_effect=AssertionError('no undeclared process read')):
            with self.assertRaises(ValueError):observer.observe_process('another_process', proc_root=self.proc)
        with patch.object(observer.sys, 'argv', ['observer.py', '--pid', '999']), patch.object(observer, 'observe_process') as read:
            with self.assertRaises(ValueError):observer.main()
        read.assert_not_called()

    def test_wrong_host_refuses_before_prior_identity_reads(self):
        """A local CPU invocation cannot emit a target-host exit proof."""
        with patch.object(observer.sys, 'argv', ['observer.py']), patch.object(observer.os, 'uname', return_value=SimpleNamespace(nodename='synthetic_wrong_host')), patch.object(observer, 'observe_process') as read:
            with self.assertRaises(ValueError):observer.main()
        read.assert_not_called()

    def test_source_contains_no_process_actions_or_global_enumeration(self):
        """The standalone observer has no signal, tool, GPU, lock or process-enumeration API call."""
        tree = ast.parse((ROOT / 'observe_exact_prior_h100_exits.py').read_text())
        names = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(names & {'kill', 'killpg', 'system', 'Popen', 'run', 'flock', 'listdir', 'scandir', 'glob', 'rglob'})
        self.assertEqual(observer.IDENTITIES, {'controller': (1042828, '87798085'), 'numerical_child': (1042870, '87798186')})


if __name__ == '__main__':unittest.main()
