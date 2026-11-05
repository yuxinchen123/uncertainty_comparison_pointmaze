"""Adversarial metadata fixtures for exact old-source direct consumer isolation."""
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import direct_historical_consumer as historical
import direct_common_confirmation as guard
import measure_selected as selected


def source_digest(path):
    """Identify the exact test source dependency without executing numerical code."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_plan():
    """Construct synthetic current metadata with the real bridge bytes and no scientific claims."""
    files = {historical.PREFIX + "fixture_helper_" + str(index) + ".py": str(index).zfill(64) for index in range(26)}
    files[historical.PREFIX + historical.NAME] = source_digest(SCRIPTS / historical.NAME)
    files[historical.PREFIX + historical.GUARD] = source_digest(SCRIPTS / historical.GUARD)
    return {"kind": "selected_configuration_measurement", "mode": "confirmation", "hardware_class": "h100",
        "gpu_uuid": "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3",
        "source_commit": "d28e6735e79788f618acb0ebfde8609db1f776af",
        "direct_historical_consumer_protocol": copy.deepcopy(historical.PROTOCOL), "direct_binding": {},
        "common_correctness_inputs": {}, "direct_qualification_inputs": {},
        "helpers": {"source_root": "/synthetic/current/helpers", "source_commit": "a" * 40, "files": files}}


class HistoricalProtocolTests(unittest.TestCase):
    """Keep the current capability explicit and distinct from historical operations."""

    def test_exact_current_protocol_is_recognized(self):
        """The synthetic current role requires all twenty-eight declared files."""
        plan = protocol_plan()
        self.assertIs(historical.contract(plan), plan)

    def test_old_registered_helper_count_cannot_bypass_current_protocol(self):
        """An old twenty-six-helper unit remains an old read-only input only."""
        plan = protocol_plan()
        del plan['helpers']['files'][historical.PREFIX + historical.NAME]
        del plan['helpers']['files'][historical.PREFIX + historical.GUARD]
        with self.assertRaisesRegex(ValueError, 'twenty-eight-helper'):
            historical.contract(plan)

    def test_boolean_schema_is_not_integer_schema(self):
        """Boolean equality cannot substitute for an integer capability version."""
        plan = protocol_plan()
        plan['direct_historical_consumer_protocol']['schema_version'] = True
        with self.assertRaises(ValueError):
            historical.contract(plan)

    def test_cache_or_extra_helper_is_not_silently_enabled(self):
        """A future twenty-nine-file cache needs a separate reviewed protocol."""
        plan = protocol_plan()
        plan['helpers']['files'][historical.PREFIX + 'future_cache.py'] = 'f' * 64
        with self.assertRaises(ValueError):
            historical.contract(plan)
        plan = protocol_plan()
        plan['direct_historical_consumer_protocol']['digest_cache'] = 'enabled'
        with self.assertRaises(ValueError):
            historical.contract(plan)

    def test_final_or_other_target_never_uses_historical_operational_bypass(self):
        """Neither final admission nor another physical target receives this capability."""
        for change in ({'final_protocol': {}}, {'hardware_class': 'jaguar03_rtx_a4500'},
                {'gpu_uuid': 'GPU-another-target'}, {'mode': 'packing-point'}):
            with self.subTest(change=change):
                plan = protocol_plan()
                plan.update(change)
                with self.assertRaises(ValueError):
                    historical.contract(plan)

    def test_duplicate_or_nonfinite_process_metadata_is_refused(self):
        """Input keys and finite numeric meaning remain exact across process messages."""
        for raw in (b'{"operation":1,"operation":2}', b'{"value":NaN}'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    historical.document(raw)


class ClosedOldNamespaceTests(unittest.TestCase):
    """Exercise exact lazy old source loading without touching real historical units."""

    def setUp(self):
        """Give each source fixture an isolated ordinary source directory and module names."""
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.finders = list(sys.meta_path)
        self.names = {'historical_fixture_alpha', 'historical_fixture_beta'}
        self.original_modules = {name: sys.modules.get(name) for name in self.names}

    def tearDown(self):
        """Restore only the fixture's own module bindings and loader list."""
        sys.meta_path[:] = self.finders
        for name, value in self.original_modules.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        self.directory.cleanup()

    def source(self, name, raw):
        """Create an explicit synthetic staged source with exact raw bytes and digest."""
        path = self.root / (name + '.py')
        path.write_bytes(raw)
        return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'raw': raw}

    def test_old_adjacent_style_import_observes_only_the_exact_old_source(self):
        """An already reserved old name executes its pinned bytes when its file is inspected."""
        sources = {'historical_fixture_alpha': self.source('historical_fixture_alpha', b'value = 7\n')}
        loader = historical.install_old_modules(sources)
        module = sys.modules['historical_fixture_alpha']
        self.assertEqual(module.__file__, sources['historical_fixture_alpha']['path'])
        self.assertEqual(module.value, 7)
        self.assertEqual(loader.loaded['historical_fixture_alpha']['sha256'], sources['historical_fixture_alpha']['sha256'])

    def test_transitive_import_keeps_a_separate_exact_old_global(self):
        """An old helper importing another old helper cannot receive a current value."""
        sources = {
            'historical_fixture_alpha': self.source('historical_fixture_alpha', b'import historical_fixture_beta\nvalue = historical_fixture_beta.value\n'),
            'historical_fixture_beta': self.source('historical_fixture_beta', b'value = 11\n')}
        historical.install_old_modules(sources)
        self.assertEqual(sys.modules['historical_fixture_alpha'].value, 11)

    def test_ambient_module_with_same_name_is_refused_before_loading(self):
        """A pre-existing current module is never reused as an old global."""
        sys.modules['historical_fixture_alpha'] = importlib.util.module_from_spec(
            importlib.util.spec_from_loader('historical_fixture_alpha', loader=None))
        sources = {'historical_fixture_alpha': self.source('historical_fixture_alpha', b'value = 7\n')}
        with self.assertRaisesRegex(ValueError, 'already replaced or loaded'):
            historical.install_old_modules(sources)

    def test_buffer_substitution_is_refused(self):
        """The closed loader refuses source bytes that differ from its declared SHA256."""
        source = self.source('historical_fixture_alpha', b'value = 7\n')
        source['raw'] = b'value = 99\n'
        historical.install_old_modules({'historical_fixture_alpha': source})
        with self.assertRaisesRegex(ValueError, 'source buffer changed'):
            unused = sys.modules['historical_fixture_alpha'].value

    def test_late_staged_source_substitution_is_refused(self):
        """A staged helper changed after closure pinning is refused before its code executes."""
        source = self.source('historical_fixture_alpha', b'value = 7\n')
        historical.install_old_modules({'historical_fixture_alpha': source})
        Path(source['path']).write_bytes(b'value = 99\n')
        with self.assertRaisesRegex(ValueError, 'staged source changed'):
            unused = sys.modules['historical_fixture_alpha'].value

    def test_numerical_import_finder_refuses_gpu_and_array_packages(self):
        """The old namespace cannot initialize a numerical or GPU package."""
        finder = historical.ExactOldLoader({})
        for name in ('numpy', 'jax', 'jax_cuda12_plugin', 'nvidia.cuda_runtime'):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    finder.find_spec(name)


class ReadOnlyAuditTests(unittest.TestCase):
    """Keep every old consumer read-only with a narrow local Git process allowlist."""

    def test_declared_read_only_git_is_allowed(self):
        """Source checks may inspect one exact declared local root."""
        historical.readonly_audit(guard, {'/declared/source'}, 'subprocess.Popen',
            ('git', ['git', '-C', '/declared/source', 'show', 'a' * 40 + ':source.py'], None, guard.metadata_environment({})))

    def test_substituted_actual_git_executable_and_external_diff_are_refused(self):
        """Exact-looking arguments cannot conceal another executable or an external Git command."""
        for executable, command in (('/another/executable', ['git', '-C', '/declared/source', 'show', 'a' * 40 + ':source.py']),
                ('git', ['git', '-C', '/declared/source', 'diff', '--ext-diff'])):
            with self.subTest(executable=executable, command=command), self.assertRaises(ValueError):
                historical.readonly_audit(guard, {'/declared/source'}, 'subprocess.Popen',
                    (executable, command, None, guard.metadata_environment({})))

    def test_python_child_is_forbidden_inside_historical_replay(self):
        """The old reader cannot become a second launcher."""
        with self.assertRaisesRegex(ValueError, 'cannot start Python'):
            historical.readonly_audit(guard, set(), 'subprocess.Popen',
                ('/usr/bin/python3', ['/usr/bin/python3', '-c', 'print(1)'], None, None))

    def test_process_execution_and_file_metadata_mutation_are_refused(self):
        """Alternate process APIs and metadata writes grant no historical read-only capability."""
        for event in ('os.exec', 'os.posix_spawn', 'os.fork', 'os.chown', 'os.utime', 'os.truncate'):
            with self.subTest(event=event), self.assertRaises(ValueError):
                historical.readonly_audit(guard, set(), event, ())

    def test_writes_and_changed_roots_are_refused(self):
        """Neither a filesystem mutation nor a foreign Git root is a permitted read."""
        with self.assertRaises(ValueError):
            historical.readonly_audit(guard, set(), 'open', ('/tmp/fixture', 'wb', 0))
        with self.assertRaises(ValueError):
            historical.readonly_audit(guard, set(), 'os.mkdir', ('/tmp/fixture', 0o700, -1))
        with self.assertRaises(ValueError):
            historical.readonly_audit(guard, {'/declared/source'}, 'subprocess.Popen',
                ('git', ['git', '-C', '/foreign/source', 'status'], None, None))

    def test_later_native_library_loads_and_numerical_imports_are_refused(self):
        """The explicit earlier ctypes bootstrap grants no later library access."""
        with self.assertRaises(RuntimeError):
            historical.readonly_audit(guard, set(), 'ctypes.dlopen', ('libcuda.so',))
        for name in ('numpy', 'nvidia'):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    historical.readonly_audit(guard, set(), 'import', (name, None, None, None, None))


class ReturnedProofIdentityTests(unittest.TestCase):
    """Keep a synthetic proof bound to its actual expected child, inputs and role."""

    def setUp(self):
        """Build metadata fixtures only; these are not an actual historical validation."""
        self.plan = protocol_plan()
        self.observed = {'pid': 71, 'parent_pid': 70, 'birth_counter': '123', 'all_uids': [2618919] * 4,
            'command': ['synthetic read-only child'], 'cgroup': 'synthetic fixture'}
        self.inputs = {'result': {'path': '/synthetic/result.json', 'sha256': '2' * 64},
            'review': {'path': '/synthetic/review.json', 'sha256': '3' * 64}}
        self.old_plan = {'path': '/synthetic/measurement_plan.json', 'sha256': '4' * 64}
        self.old_sources = {'direct_measurement': {'path': '/synthetic/code/direct_measurement.py', 'sha256': '5' * 64}}
        self.payload = {'operation': 'correctness', 'plan_sha256': '6' * 64,
            'challenge': '7' * 64, 'parent': {'pid': 70, 'birth_counter': '122'}}
        self.proof = {'schema_version': 1, 'kind': 'exact_old_direct_consumer_read_only_proof', 'passed': True,
            'operation': 'correctness', 'plan_sha256': '6' * 64, 'challenge': '7' * 64,
            'parent': self.payload['parent'], 'child': self.observed, 'old_helper_commit': historical.OLD_COMMIT,
            'old_consumer_returned_normally': True, 'old_inputs': self.inputs, 'old_plan': self.old_plan,
            'old_staged_files': self.old_sources, 'current_helper_closure': self.plan['helpers'],
            'old_consumer_return_sha256': '8' * 64, 'actually_loaded_old_modules': self.old_sources,
            'numerical_imports': [], 'gpu_operations': 0, 'scientific_credit': 0,
            'later_ctypes_load_or_symbol_events': 0, 'admission_granted': False, 'operational_admission': False}

    def validate(self, proof):
        """Exercise the same returned-proof validation used only after the real Popen wait."""
        return historical.validate_returned_proof(proof, self.payload, self.observed,
            self.inputs, self.old_plan, self.old_sources, self.plan)

    def test_exact_synthetic_identity_is_recognized(self):
        """Only the supplied expected metadata identities match this synthetic fixture."""
        self.assertIs(self.validate(self.proof), self.proof)

    def test_changed_child_parent_or_challenge_is_refused(self):
        """A correct-looking result from another process or request cannot substitute."""
        for field, value in [('child', {**self.observed, 'birth_counter': 'another birth'}),
                ('parent', {'pid': 99}), ('challenge', 'f' * 64), ('operation', 'qualification')]:
            with self.subTest(field=field):
                proof = copy.deepcopy(self.proof)
                proof[field] = value
                with self.assertRaises(ValueError):
                    self.validate(proof)

    def test_changed_old_or_current_sources_are_refused(self):
        """Neither a different original helper nor another current closure is accepted."""
        for field, value in [('old_helper_commit', 'f' * 40), ('old_staged_files', {}),
                ('current_helper_closure', {}), ('actually_loaded_old_modules', {'foreign': {'path': '/foreign'}}),
                ('old_plan', {'path': '/another/plan.json', 'sha256': '4' * 64})]:
            with self.subTest(field=field):
                proof = copy.deepcopy(self.proof)
                proof[field] = value
                with self.assertRaises(ValueError):
                    self.validate(proof)

    def test_boolean_counts_or_new_admission_are_refused(self):
        """Boolean zero and scientific or operational credit are different meanings."""
        for field, value in [('schema_version', True), ('gpu_operations', False),
                ('scientific_credit', False), ('later_ctypes_load_or_symbol_events', False),
                ('admission_granted', True), ('operational_admission', True), ('numerical_imports', ['numpy'])]:
            with self.subTest(field=field):
                proof = copy.deepcopy(self.proof)
                proof[field] = value
                with self.assertRaises(ValueError):
                    self.validate(proof)


class ActualOwnChildCleanupTests(unittest.TestCase):
    """An early identity failure must reap only the real own read-only child."""

    def test_identity_failure_reaps_the_exact_metadata_child(self):
        """The actual helper child receives no fake plan before its failed identity check."""
        class MetadataGuard:
            """Use no scientific gate in this process-cleanup unit fixture."""
            def metadata_compute_context(self, plan):
                """The fixture itself runs in the parent-owned metadata test step."""
                return {'synthetic': True}

            def contract(self, plan):
                """Skip only the unrelated complete scientific-plan fixture setup."""
                return None

            def metadata_environment(self, original):
                """Keep the exact real metadata-child environment in this cleanup fixture."""
                return guard.metadata_environment(original)
        real_process = historical.process
        def process_or_fail(pid):
            """Observe the real parent and reject its actual new child before stdin release."""
            if pid == __import__('os').getpid():
                return real_process(pid)
            raise RuntimeError('declared child identity fixture failure')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'measurement_plan.json').write_text('{}\n')
            records = (root, {'helpers': {'files': {}}}, {}, {'result': {}, 'review': {}})
            with mock.patch.object(historical, 'current_guard', return_value=MetadataGuard()), \
                    mock.patch.object(historical, 'current_closure', return_value={}), \
                    mock.patch.object(historical, 'selected_records', return_value=records), \
                    mock.patch.object(historical, 'process', side_effect=process_or_fail):
                with self.assertRaises(ValueError) as captured:
                    historical.request(protocol_plan(), 'correctness')
            message = str(captured.exception)
            self.assertIn('declared child identity fixture failure', message)
            wait = json.loads(message.split('historical consumer failed: ', 1)[1])
            self.assertEqual(wait['returncode'], -9)
            self.assertTrue(wait['exact_popen_child_reaped'])
            self.assertIsNone(wait['actual_child'])
            self.assertEqual(wait['stdout_base64'], '')


class DependencyAndReaderSyntaxTests(unittest.TestCase):
    """Keep ordinary dependencies unchanged while staging both required common helpers."""

    def test_new_common_stages_historical_dependency_and_ordinary_does_not(self):
        """The explicit common capability stages exactly its complete two-file addition."""
        ordinary = set(selected.required_helpers(direct=True, backend_image=True))
        common = set(selected.required_helpers(direct=True, common=True, backend_image=True))
        self.assertEqual(len(ordinary), 26)
        self.assertEqual(len(common), 28)
        self.assertEqual(common - ordinary, {guard.NAME, historical.NAME})
        self.assertNotIn(historical.NAME, selected.required_helpers(configured=True, common=True))

    def test_incomplete_or_ambiguous_target_union_is_refused(self):
        """A common flag cannot omit its backend or select two target policies."""
        for options in ({'direct': True, 'common': True}, {'direct': True, 'configured': True, 'common': True}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    selected.required_helpers(**options)

    def test_original_qualification_reader_body_is_retained_after_one_statement(self):
        """The new metadata dispatch adds no removal or rewrite of an old reader gate."""
        current = SCRIPTS / 'direct_measurement.py'
        tree = ast.parse(current.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name == 'verify_qualification_inputs')
        old = copy.deepcopy(function)
        del old.body[1]
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / 'original.py'
            original.write_text(ast.unparse(ast.Module(body=[old], type_ignores=[])) + '\n')
            proof = guard.qualification_reader_extension(original, current)
            self.assertTrue(proof['entire_original_body_retained'])
            self.assertEqual(proof['current_body_statements'], proof['original_body_statements'] + 1)

    def test_an_old_gate_change_is_not_hidden_by_the_dispatch(self):
        """A changed original statement remains a hard source-contract failure."""
        current = SCRIPTS / 'direct_measurement.py'
        tree = ast.parse(current.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name == 'verify_qualification_inputs')
        old = copy.deepcopy(function)
        del old.body[1]
        old.body.append(ast.parse('raise ValueError("changed old gate")').body[0])
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / 'changed.py'
            original.write_text(ast.unparse(ast.Module(body=[old], type_ignores=[])) + '\n')
            with self.assertRaises(ValueError):
                guard.qualification_reader_extension(original, current)


if __name__ == '__main__':
    unittest.main(verbosity=2)
