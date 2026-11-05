"""Reject raw pidfd inode/flags contradictions in live and pure saved descriptor readers."""
import ast
import copy
import hashlib
import io
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import native_process_evidence as evidence
import linux_pidfd_process_control as native
from test_native_process_evidence import synthetic_calls


class PureFdinfoTests(unittest.TestCase):
    """Use only explicitly synthetic metadata for parser and saved-call negative cases."""
    def setUp(self):
        """Retain one canonical synthetic pidfd record without issuing a native call."""
        self.raw = 'pos:\t0\nflags:\t02000002\nmnt_id:\t5\nino:\t77\nPid:\t101\nNSpid:\t101\n'

    def test_live_and_saved_parsers_have_identical_function_source(self):
        """Keep the two pure copies identical without adding a provider dependency to saved imports."""
        functions = []
        for name in ('native_process_evidence.py', 'linux_pidfd_process_control.py'):
            text = (SCRIPTS / name).read_text()
            node = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)
                and node.name == 'parse_pidfd_fdinfo')
            functions.append(ast.get_source_segment(text, node))
        self.assertEqual(*functions)

    def test_positive_complete_raw_fields_and_exited_pid_policy(self):
        """Both copies join canonical inode and octal CLOEXEC fields while retaining expected-Pid/-1 behavior."""
        for module in (native, evidence):
            with self.subTest(module=module.__name__):
                self.assertEqual(module.parse_pidfd_fdinfo(self.raw, 77, 101),
                    {'inode': 77, 'pid': 101, 'flags': 0o2000002, 'close_on_exec': True})
                self.assertEqual(module.parse_pidfd_fdinfo(self.raw.replace('Pid:\t101\n', 'Pid:\t-1\n'), 77, 101)['pid'], -1)
                with self.assertRaises(ValueError):module.parse_pidfd_fdinfo(self.raw, 77, 102)

    def test_duplicate_and_malformed_raw_fields_are_refused(self):
        """A duplicate is refused even if its second spelling would otherwise be ignored by a numeric regex."""
        values = [self.raw + key + ':\t' + value + '\n' for key, value in
            [('ino', '77'), ('ino', 'invalid'), ('Pid', '101'), ('Pid', 'unknown'), ('flags', '02000002'), ('flags', 'True')]]
        values += [self.raw.replace('ino:\t77', 'ino :\t77'), self.raw.replace('Pid:', ' Pid:'),
            self.raw + 'malformed line\n', self.raw.replace('\n', '\r\n'), self.raw + '\0\n', self.raw[:-1]]
        for raw in values:
            for module in (native, evidence):
                with self.subTest(module=module.__name__, raw=raw), self.assertRaises(ValueError):
                    module.parse_pidfd_fdinfo(raw, 77, 101)

    def test_raw_numbers_flags_and_expected_types_are_strict(self):
        """Boolean, signed, noncanonical and nonoctal fields never become valid descriptor identities."""
        for key, old, bad in [('ino', '77', 'True'), ('ino', '77', '077'), ('ino', '77', '-1'),
                ('Pid', '101', '+101'), ('Pid', '101', '0101'), ('Pid', '101', '0'),
                ('flags', '02000002', '0o2000002'), ('flags', '02000002', '02000008'),
                ('flags', '02000002', '00000002')]:
            changed = self.raw.replace(key + ':\t' + old, key + ':\t' + bad)
            for module in (native, evidence):
                with self.subTest(key=key, bad=bad, module=module.__name__), self.assertRaises(ValueError):
                    module.parse_pidfd_fdinfo(changed, 77, 101)
        for inode, pid in [(True, 101), (77, True), ('77', 101), (77, '101'), (0, 101), (77, 0)]:
            for module in (native, evidence):
                with self.subTest(inode=inode, pid=pid), self.assertRaises(ValueError):
                    module.parse_pidfd_fdinfo(self.raw, inode, pid)

    def changed_events(self, mutate):
        """Refresh raw hashes after a labelled mutation so the real parser must detect the contradiction."""
        value, owner, allowed, contract = synthetic_calls()
        for event in value['descriptor_events']:
            mutate(event['identity'])
            event['identity']['fdinfo_sha256'] = hashlib.sha256(event['identity']['raw_fdinfo'].encode()).hexdigest()
        return value, owner, allowed, contract

    def test_reproduced_saved_raw_inode_summary_mismatch_is_refused(self):
        """The exact independently reproduced mismatch cannot pass after its raw hash is refreshed."""
        def alter(identity):
            identity['raw_fdinfo'] = identity['raw_fdinfo'].replace('ino:\t77', 'ino:\t88')
        with self.assertRaises(ValueError):evidence.validate_calls(*self.changed_events(alter))

    def test_reproduced_saved_missing_cloexec_is_refused(self):
        """A saved false-inheritance summary cannot override raw flags lacking CLOEXEC."""
        def alter(identity):
            identity['raw_fdinfo'] = identity['raw_fdinfo'].replace('02000002', '00000002')
        with self.assertRaises(ValueError):evidence.validate_calls(*self.changed_events(alter))

    def test_saved_flags_summaries_and_duplicate_raw_fields_are_refused(self):
        """Raw fields, typed summaries and explicit noninheritance must agree in every generation."""
        mutations = [lambda identity: identity.update(flags=True), lambda identity: identity.update(flags=2),
            lambda identity: identity.update(inheritable=0), lambda identity: identity.update(inheritable=True),
            lambda identity: identity.update(raw_fdinfo=identity['raw_fdinfo'] + 'ino:\t77\n'),
            lambda identity: identity.update(raw_fdinfo=identity['raw_fdinfo'] + 'flags:\t02000002\n')]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                evidence.validate_calls(*self.changed_events(mutate))


class ActualOwnFdinfoTests(unittest.TestCase):
    """Inspect one positively retained own-self descriptor on the exact owned CPU allocation."""
    def setUp(self):
        """Require owned compute before opening the actual self pidfd used by these fixtures."""
        self.assertEqual(os.environ.get('SLURM_JOB_ID'), '6783003')
        self.assertTrue(os.environ.get('SLURM_STEP_ID', '').isdecimal())
        contract, unused = native.current_contract()
        self.provider = native.NativePidfdProvider(contract)
        self.handle = self.provider.open_process_handle(os.getpid())
        self.addCleanup(self.provider.close_process_handle, self.handle)

    def test_actual_fdinfo_matches_fstat_and_noninheritance(self):
        """The real descriptor proves both its raw inode and its raw CLOEXEC bit."""
        actual = self.provider.descriptor_identity(self.handle, os.getpid())
        self.assertEqual(actual['inode'], os.fstat(self.handle).st_ino)
        self.assertIs(actual['inheritable'], False)
        self.assertIs(os.get_inheritable(self.handle), False)
        parsed = evidence.parse_pidfd_fdinfo(actual['raw_fdinfo'], actual['inode'], os.getpid())
        self.assertEqual(parsed['flags'], actual['flags'])
        self.assertTrue(parsed['flags'] & os.O_CLOEXEC)

    def test_live_reader_rejects_changed_inode_flags_duplicates_and_typed_raw_fields(self):
        """Substitute only this own descriptor's fdinfo text while retaining its actual fstat inode."""
        raw = self.provider.descriptor_identity(self.handle, os.getpid())['raw_fdinfo']
        original = Path.open
        values = [re.sub(r'(?m)^ino:\s*\d+$', 'ino:\t1', raw),
            re.sub(r'(?m)^flags:\s*[0-7]+$', 'flags:\t00000002', raw),
            raw + 'Pid:\t' + str(os.getpid()) + '\n', raw + 'ino:\tinvalid\n',
            re.sub(r'(?m)^Pid:\s*\d+$', 'Pid:\tTrue', raw)]
        for changed in values:
            def substituted(path, *arguments, **keywords):
                if str(path) == '/proc/self/fdinfo/' + str(self.handle):return io.StringIO(changed)
                return original(path, *arguments, **keywords)
            with self.subTest(raw=changed), patch.object(Path, 'open', new=substituted), self.assertRaises(ValueError):
                self.provider.descriptor_identity(self.handle, os.getpid())


if __name__ == '__main__':
    unittest.main()
