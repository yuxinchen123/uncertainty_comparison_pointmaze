"""Independently run synthetic metadata controls inside the existing owned CPU allocation."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
import unittest
from unittest.mock import patch

W = Path(__file__).resolve().parents[1]
S = W / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
R = W / '.review_metadata'
FREEZE = R / 'direct_h100_profile_source_freeze_2026-11-04_author_6.json'
freeze = json.loads(FREEZE.read_text())
assert os.environ['SLURM_JOB_ID'] == '6783003'
assert os.environ['SLURM_STEP_ID'].isdecimal()
assert all(hashlib.sha256((W / p).read_bytes()).hexdigest() == h for p, h in freeze['files'].items())
sys.path.insert(0, str(S / 'tests'))
sys.path.insert(0, str(S))
import test_direct_profile_prerequisites as fixtures
import direct_profile_prerequisites as adapter
import direct_measurement as direct


class IndependentControls(unittest.TestCase):
    """Check source and refusal cases independently of the author's declared tests."""
    def setUp(self):
        """Use only synthetic metadata and absent result files from the fixture builder."""
        self.fixture = fixtures.DirectProfileMetadataTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_accepted_backend_location_is_checked_before_opening_result(self):
        """A prior accepted plan cannot turn its backend reference into a scientific-result read."""
        f = self.fixture
        f.previous['direct_backend_image']['expected'] = f.inputs['confirmation']
        previous_path = Path(f.inputs['confirmation_plan']['path'])
        f.plan['direct_profile_prerequisites']['confirmation_plan'] = f.write(previous_path, f.previous)
        original = adapter.read
        def read_metadata(path, *arguments):
            """Fail if the producer tries opening the deliberately absent scientific file."""
            self.assertNotEqual(str(path), f.inputs['confirmation']['path'])
            return original(path, *arguments)
        with patch.object(adapter, 'read', side_effect=read_metadata):
            with self.assertRaisesRegex(ValueError, 'reserved unit filename'):
                adapter.validate_metadata(f.plan)

    def test_explicit_historical_read_flag_cannot_admit_a_new_profile(self):
        """The old read-only flag never bypasses the new unconditional direct admission stop."""
        f = self.fixture
        f.write(f.unit / 'measurement_plan.json', f.plan)
        with self.assertRaisesRegex(ValueError, 'remain disabled'):
            direct.verify_saved_measurement(f.unit, {}, historical_profile_read_only=True)

    def test_current_raw_history_context_refusal_has_no_import(self):
        """The scientific history reader stays unavailable before the actual owned-compute check."""
        with patch.object(adapter, 'compute_context', side_effect=ValueError('independent unallocated refusal')), \
                patch.object(adapter.importlib, 'import_module') as imports:
            with self.assertRaisesRegex(ValueError, 'independent unallocated refusal'):
                adapter.validate_current_history(self.fixture.plan)
            imports.assert_not_called()

    def test_preparation_binding_source_has_no_raw_history_call(self):
        """The new builder cannot invoke the raw validator directly or through the old builder."""
        tree = ast.parse((S / adapter.NAME).read_text())
        builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'configure_binding')
        calls = [node for node in ast.walk(builder) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr == 'validate_history' for node in calls))
        self.assertTrue(any(isinstance(node.func, ast.Attribute) and node.func.attr == 'validate_plan' for node in calls))
        old = ast.parse((S / 'direct_measurement.py').read_text())
        original_builder = next(node for node in old.body if isinstance(node, ast.FunctionDef) and node.name == 'configure_plan')
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == 'validate_history' for node in ast.walk(original_builder)))


suite = unittest.TestSuite()
loader = unittest.defaultTestLoader
for name in ('test_direct_profile_prerequisites.DirectProfileMetadataTests',
             'test_direct_profile_prerequisites.DirectProfileBackendTests',
             'test_profile_selected', 'test_profile_owned_processes',
             'test_profile_common_confirmation_helpers', 'test_direct_common_confirmation', 'test_direct_measurement'):
    suite.addTests(loader.loadTestsFromName(name))
suite.addTests(loader.loadTestsFromTestCase(IndependentControls))
output = io.StringIO()
started = time.time()
result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
raw = output.getvalue().encode()
label = 'direct_h100_profile_independent_' + os.environ['SLURM_JOB_ID'] + '_' + os.environ['SLURM_STEP_ID']
(R / (label + '_unittest.raw')).write_bytes(raw)
unchanged = all(hashlib.sha256((W / p).read_bytes()).hexdigest() == h for p, h in freeze['files'].items())
receipt = {'kind': 'independent_direct_profile_metadata_source_review', 'source_freeze': {'path': str(FREEZE),
    'sha256': hashlib.sha256(FREEZE.read_bytes()).hexdigest()}, 'base': freeze['base'],
    'job_id': os.environ['SLURM_JOB_ID'], 'step_id': os.environ['SLURM_STEP_ID'], 'hostname': socket.gethostname(),
    'pid': os.getpid(), 'started_at_seconds': started, 'finished_at_seconds': time.time(),
    'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
    'source_unchanged_before_and_after': unchanged, 'unittest_output': {'path': str(R / (label + '_unittest.raw')),
    'sha256': hashlib.sha256(raw).hexdigest()}, 'gpu_operations': 0, 'scientific_credit': 0,
    'admission_granted': False, 'actual_direct_process_descriptor_cuda_qualification': False,
    'approved_disabled_source': result.wasSuccessful() and unchanged,
    'scope': 'Source and metadata only. Actual direct profiling remains unconditionally disabled; no process/FD/CUDA qualification or job action is approved.'}
(R / (label + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'receipt': str(R / (label + '.json')), 'tests_run': result.testsRun,
    'failures': len(result.failures), 'errors': len(result.errors), 'source_unchanged': unchanged,
    'approved_disabled_source': receipt['approved_disabled_source'], 'output_tail': output.getvalue()[-1800:]}))
raise SystemExit(0 if result.wasSuccessful() and unchanged else 1)
