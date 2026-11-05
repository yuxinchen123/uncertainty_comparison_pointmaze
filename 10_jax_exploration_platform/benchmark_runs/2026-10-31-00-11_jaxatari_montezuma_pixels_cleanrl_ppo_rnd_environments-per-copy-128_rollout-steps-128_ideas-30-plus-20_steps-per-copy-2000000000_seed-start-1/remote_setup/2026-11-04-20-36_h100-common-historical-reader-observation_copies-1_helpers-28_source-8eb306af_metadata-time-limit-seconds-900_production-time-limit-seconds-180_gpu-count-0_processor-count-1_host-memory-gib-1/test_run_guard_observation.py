"""Metadata-only fixtures for the new read-only input snapshot wrapper."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

PACKET = Path(__file__).resolve().parent
specification = importlib.util.spec_from_file_location('snapshot_wrapper_under_test', PACKET / 'run_guard_observation.py')
wrapper = importlib.util.module_from_spec(specification)
specification.loader.exec_module(wrapper)
observer_specification = importlib.util.spec_from_file_location('unchanged_observer_process_reader', PACKET / 'observe_guard_phases.py')
observer = importlib.util.module_from_spec(observer_specification)
observer_specification.loader.exec_module(observer)


class SnapshotWrapperTests(unittest.TestCase):
    """Use temporary metadata only, never the actual guard or existing scientific inputs."""

    def setUp(self):
        """Give each fixture a fresh output directory and one ordinary input file."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / 'existing_input.json'
        self.input.write_text('{"value":1}\n')
        self.reference = {'path': str(self.input), 'sha256': hashlib.sha256(self.input.read_bytes()).hexdigest()}

    def test_snapshot_reads_without_rewriting_the_input(self):
        """Hashing retains the input's exact bytes, inode and modification time."""
        before = self.input.stat()
        value = wrapper.snapshot({'synthetic': self.reference})
        self.assertEqual(value['synthetic']['reference'], self.reference)
        self.assertEqual(before.st_ino, self.input.stat().st_ino)
        self.assertEqual(before.st_mtime_ns, self.input.stat().st_mtime_ns)
        self.assertEqual(self.input.read_bytes(), b'{"value":1}\n')

    def test_changed_bytes_and_symlink_inputs_are_refused(self):
        """A changed file or replacement path cannot be treated as the earlier input."""
        self.input.write_text('{"value":2}\n')
        with self.assertRaises(ValueError):
            wrapper.snapshot({'synthetic': self.reference})
        link = self.root / 'replacement.json'
        link.symlink_to(self.input)
        with self.assertRaises(ValueError):
            wrapper.snapshot({'synthetic': {**self.reference, 'path': str(link)}})

    def test_oversized_metadata_is_refused_before_reading(self):
        """A sparse oversized fixture stays outside the explicit metadata-only snapshot scope."""
        with self.input.open('wb') as stream:
            stream.truncate(16 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(ValueError, 'ownership or size'):
            wrapper.snapshot({'synthetic': self.reference})

    def test_output_names_are_local_and_never_reused(self):
        """Only fixed new output names are writable and an earlier result is preserved."""
        with mock.patch.object(wrapper, 'PACKET', self.root):
            with self.assertRaises(ValueError):
                wrapper.save('../existing_input.json', b'changed')
            wrapper.save('wrapper_intent.json', {'value': 1})
            with self.assertRaises(FileExistsError):
                wrapper.save('wrapper_intent.json', {'value': 2})
        self.assertEqual(self.input.read_bytes(), b'{"value":1}\n')

    def fixture_main(self, action):
        """Replace only the observer call with a declared synthetic action; do not invoke its real parent."""
        for name in ('observe_guard_phases.py', 'run_guard_observation.py'):
            (self.root / name).write_bytes((PACKET / name).read_bytes())
        for name in ('unregistered_guard_input_plan.json', 'diagnostic_unit_identity.json', 'source_publication_binding.json'):
            (self.root / name).write_text('{"synthetic":true}\n')
        (self.root / 'consumed_input_references.json').write_text(json.dumps({'references_by_role': {'synthetic': self.reference}}))
        manifest = {'observer_source': {'sha256': hashlib.sha256((self.root / 'observe_guard_phases.py').read_bytes()).hexdigest()},
            'wrapper_source': {'sha256': hashlib.sha256((PACKET / 'run_guard_observation.py').read_bytes()).hexdigest()},
            'diagnostic_timeout_seconds': 900, 'production_timeout_seconds_unchanged': 180, 'registered': False}
        (self.root / 'manifest.yaml').write_text(json.dumps(manifest))
        fake = types.SimpleNamespace(process=observer.process, parent=action)
        specification = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda module: None))
        with mock.patch.object(wrapper, 'PACKET', self.root), \
                mock.patch.object(wrapper.importlib.util, 'spec_from_file_location', return_value=specification), \
                mock.patch.object(wrapper.importlib.util, 'module_from_spec', return_value=fake), \
                mock.patch.object(wrapper.sys, 'argv', ['synthetic_snapshot_fixture']):
            wrapper.main()

    def test_normal_synthetic_observer_keeps_inputs_and_records_zero_credit(self):
        """The wrapper reports unchanged metadata and no production-unit proof."""
        self.fixture_main(lambda: None)
        result = json.loads((self.root / 'input_snapshot_comparison.json').read_bytes())
        self.assertTrue(result['all_inputs_unchanged'])
        self.assertFalse(result['borrowed_identity_is_production_unit_proof'])
        self.assertFalse(result['production_guard_qualified'])
        self.assertEqual(self.input.read_bytes(), b'{"value":1}\n')

    def test_observer_failure_still_records_after_snapshot_and_reraises(self):
        """A synthetic observer failure is preserved after the required final input snapshot."""
        def fail():
            """Exercise only failure capture, never the actual metadata observer."""
            raise RuntimeError('synthetic observer failure')
        with self.assertRaisesRegex(RuntimeError, 'synthetic observer failure'):
            self.fixture_main(fail)
        result = json.loads((self.root / 'wrapper_return.json').read_bytes())
        self.assertEqual(result['observer_failure']['type'], 'RuntimeError')
        self.assertTrue((self.root / 'input_snapshot_after.json').exists())

    def test_changed_input_during_observer_is_refused_and_recorded(self):
        """A temporary metadata mutation cannot be accepted as unchanged input."""
        with self.assertRaisesRegex(ValueError, 'bytes or identity changed'):
            self.fixture_main(lambda: self.input.write_text('{"value":2}\n'))
        result = json.loads((self.root / 'input_snapshot_comparison.json').read_bytes())
        self.assertFalse(result['all_inputs_unchanged'])
        self.assertTrue(json.loads((self.root / 'input_snapshot_after.json').read_bytes())['snapshot_failed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
