"""Exercise exact Git command forms used by the frozen historical metadata readers."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import direct_common_confirmation as guard
METADATA = SCRIPTS.parents[2] / '.review_metadata/historical_git_read_only_branch_name'


class HistoricalReadOnlyGitFormsTests(unittest.TestCase):
    """Keep the three actual historical forms precise and every existing safety check active."""

    def setUp(self):
        """Use only synthetic source names and the exact fixed metadata environment."""
        directory = tempfile.TemporaryDirectory(prefix='historical_git_shape_fixture_')
        self.addCleanup(directory.cleanup)
        self.root = directory.name
        self.commit = 'a' * 40
        self.environment = guard.metadata_environment({})

    def command(self, arguments):
        """Build the same fixed Git prefix as the unchanged old readers."""
        return ['git', '-C', self.root, *arguments]

    def test_three_exact_historical_forms_are_recognized(self):
        """Only the approved branch, blob-name and porcelain aliases join existing forms."""
        for arguments in (['rev-parse', '--abbrev-ref', 'HEAD'],
                ['rev-parse', self.commit + ':src/record.py'], ['status', '--porcelain=v1', '-uall']):
            with self.subTest(arguments=arguments):
                guard.source_inspection_git(self.command(arguments), {self.root})

    def test_nearby_options_revision_expressions_and_extra_arguments_are_refused(self):
        """Literal branch/object queries do not permit arbitrary revisions, options or output programs."""
        invalid = (['rev-parse', '--abbrev-ref=loose', 'HEAD'], ['rev-parse', '--abbrev-ref', 'HEAD', '--verify'],
            ['rev-parse', '--abbrev-ref', 'another'], ['rev-parse', 'HEAD:src/record.py'],
            ['rev-parse', self.commit + '^:src/record.py'], ['rev-parse', self.commit + ':../private'],
            ['rev-parse', self.commit + ':/absolute'], ['rev-parse', self.commit + ':src/'],
            ['rev-parse', self.commit + ':src/record.py', '--verify'],
            ['status', '--porcelain=v1', '-uall', '--ignored'], ['status', '--porcelain=v1', '-uno'],
            ['status', '--porcelain=v1', '--', 'helper.py'], ['diff', '--ext-diff'])
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                guard.source_inspection_git(self.command(arguments), {self.root})

    def test_actual_executable_cwd_environment_root_and_configuration_checks_remain(self):
        """A newly permitted argument form still uses the complete same inspection boundary."""
        arguments = self.command(['rev-parse', '--abbrev-ref', 'HEAD'])
        with mock.patch.object(guard, 'verify_repository_configuration') as verify:
            guard.metadata_event_guard('subprocess.Popen', ('git', arguments, None, self.environment), {self.root})
        verify.assert_called_once_with(self.root, self.environment)
        wrong = [('/another/executable', arguments, None, self.environment),
            ('git', arguments, '/another/directory', self.environment),
            ('git', arguments, None, {**self.environment, 'PATH': '/another/bin'})]
        for values in wrong:
            with self.subTest(values=values), self.assertRaises(ValueError):
                guard.metadata_event_guard('subprocess.Popen', values, {self.root})
        with self.assertRaises(ValueError):
            guard.metadata_event_guard('subprocess.Popen', ('git', arguments, None, self.environment), set())
        with mock.patch.object(guard, 'verify_repository_configuration', side_effect=ValueError('synthetic filter refusal')), \
                self.assertRaisesRegex(ValueError, 'synthetic filter refusal'):
            guard.metadata_event_guard('subprocess.Popen', ('git', arguments, None, self.environment), {self.root})

    def old_function(self, filename, name, namespace):
        """Compile only one unchanged old metadata function from its actual hash-pinned source."""
        inventory = json.loads((METADATA / 'old_twenty_six_git_call_inventory.json').read_bytes())
        record = next(value for path, value in inventory['old_source_files'].items() if path.endswith('/' + filename))
        path = Path(record['path'])
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), record['sha256'])
        node = next(node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        return namespace[name]

    def test_unchanged_infrastructure_git_receipt_uses_the_new_exact_branch_form(self):
        """The actual old function's five calls pass the policy without executing Git or source validation."""
        contents = b'# synthetic metadata only\n'
        captured = []
        def output(command, text=False):
            """Record real old function call shapes against synthetic replies only."""
            guard.metadata_event_guard('subprocess.Popen', ('git', command, None, self.environment), {self.root})
            captured.append(command)
            arguments = command[3:]
            if arguments == ['rev-parse', '--show-toplevel']:
                return self.root
            if arguments == ['rev-parse', 'HEAD']:
                return self.commit
            if arguments == ['rev-parse', '--abbrev-ref', 'HEAD']:
                return 'HEAD'
            if arguments[:1] == ['status']:
                return ''
            return contents
        evidence = types.SimpleNamespace(git_root=lambda root, commit: Path(root), file=lambda path: contents)
        namespace = {'Path': Path, 'subprocess': types.SimpleNamespace(check_output=output),
            'digest': lambda raw: hashlib.sha256(raw).hexdigest()}
        function = self.old_function('infrastructure_retry.py', 'git_receipt', namespace)
        with mock.patch.object(guard, 'verify_repository_configuration', return_value={}):
            value = function(self.root, self.commit, ['src/record.py'], evidence=evidence)
        self.assertEqual(value['files']['src/record.py'], hashlib.sha256(contents).hexdigest())
        self.assertIn(self.command(['rev-parse', '--abbrev-ref', 'HEAD']), captured)
        self.assertEqual(len(captured), 5)

    def test_unchanged_async_blob_name_query_keeps_the_recorded_git_blob_gate(self):
        """The actual old validate_core function still compares the literal blob name."""
        expected_blob = 'b' * 40
        calls = []
        def output(command, text=False):
            """Exercise only the old call shape against a declared synthetic blob name."""
            guard.metadata_event_guard('subprocess.Popen', ('git', command, None, self.environment), {self.root})
            calls.append(command)
            return expected_blob
        evidence = types.SimpleNamespace(git_root=lambda root, commit: Path(root))
        namespace = {'subprocess': types.SimpleNamespace(check_output=output),
            'infra': types.SimpleNamespace(validate_git_receipt=mock.Mock())}
        function = self.old_function('pregather_async_retry.py', 'validate_core', namespace)
        record = {'source_root': self.root, 'source_commit': self.commit,
            'files': {'src/record.py': {'sha256': 'c' * 64, 'git_blob': expected_blob}}}
        with mock.patch.object(guard, 'verify_repository_configuration', return_value={}):
            function(record, evidence)
            record['files']['src/record.py']['git_blob'] = 'd' * 40
            with self.assertRaisesRegex(ValueError, 'Git object changed'):
                function(record, evidence)
        self.assertTrue(all(command == self.command(['rev-parse', self.commit + ':src/record.py']) for command in calls))

    def test_unchanged_observer_source_identity_keeps_exact_status_and_source_checks(self):
        """The actual old identity function emits the fixed porcelain alias without executing it."""
        contents = b'# synthetic source bytes\n'
        calls = []
        def output(command, text=False):
            """Use fixed metadata replies and the actual updated audit predicate."""
            guard.metadata_event_guard('subprocess.Popen', ('git', command, None, self.environment), {self.root})
            calls.append(command)
            if command[3:] == ['rev-parse', '--show-toplevel']:
                return self.root
            if command[3:] == ['rev-parse', 'HEAD']:
                return self.commit
            if command[3:4] == ['status']:
                return ''
            return contents
        def run(command, **kwargs):
            """Supply only the unchanged detached-branch native return in this synthetic fixture."""
            guard.metadata_event_guard('subprocess.Popen', ('git', command, None, self.environment), {self.root})
            return subprocess.CompletedProcess(command, 1)
        def require(condition, message):
            """Keep the old function's source/identity failure behavior in the fixture."""
            if not condition:
                raise ValueError(message)
        namespace = {'Path': Path, 're': __import__('re'), 'require': require,
            'subprocess': types.SimpleNamespace(check_output=output, run=run), 'ordinary': lambda path: contents,
            'PREFIX': 'metadata/', 'FILENAME': 'fixture.py', 'sha256': lambda raw: hashlib.sha256(raw).hexdigest()}
        function = self.old_function('uva_active_configuration_observation.py', 'source_identity', namespace)
        with mock.patch.object(guard, 'verify_repository_configuration', return_value={}):
            result = function(Path(self.root), self.commit)
        self.assertEqual(result['source_commit'], self.commit)
        self.assertIn(self.command(['status', '--porcelain=v1', '-uall']), calls)


if __name__ == '__main__':
    unittest.main(verbosity=2)
