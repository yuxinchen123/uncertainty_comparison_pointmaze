"""Review frozen timing metadata without running a numerical or remote process."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


WORKTREE = Path(__file__).resolve().parents[2]
REVIEW = Path(__file__).resolve().parent
SOURCE = WORKTREE / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
FREEZE = WORKTREE / '.review_metadata/direct_common_timing_controller/source_freeze.json'
HANDOFF = WORKTREE / '.review_metadata/direct_common_timing_controller/author_handoff.json'
EXPECTED_FREEZE = '1bbdf561344c8a00852794436b7c74fa32f3f80465607b81d1a5cd93d75ea327'
EXPECTED_HANDOFF = 'ec83df3d7d772848ec9810e8a79bad1932ccec7af3ddc87136d576272db12ac0'


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_frozen():
    assert checksum(FREEZE) == EXPECTED_FREEZE
    assert checksum(HANDOFF) == EXPECTED_HANDOFF
    freeze = json.loads(FREEZE.read_bytes())
    result = {}
    for relative, expected in freeze['files'].items():
        path = WORKTREE / relative
        assert path.is_file() and not path.is_symlink()
        assert checksum(path) == expected['sha256'] and path.stat().st_size == expected['bytes']
        result[str(path)] = {'sha256': checksum(path), 'mode': stat.S_IMODE(path.stat().st_mode)}
    for name, expected in freeze['protected_original_numerical_files'].items():
        assert checksum(name) == expected
    return result


started = time.monotonic()
assert os.environ.get('SLURM_JOB_ID') == '6783003'
assert '/job_6783003/' in Path('/proc/self/cgroup').read_text()
before = capture_frozen()
sys.path.insert(0, str(SOURCE / 'tests'))
sys.path.insert(0, str(SOURCE))
import test_direct_common_timing_controller as original_tests
control = original_tests.control


class IndependentTimingBoundaryTests(unittest.TestCase):
    """Check that copy two cannot select the first serial position."""

    def test_copy_two_cannot_use_first_serial_position(self):
        helper = original_tests.TimingControllerMetadataTests()
        with tempfile.TemporaryDirectory(prefix='common_timing_order_review_') as temporary:
            packet = Path(temporary)
            path, request, refs = helper.predecessor(packet)
            real_document = control.document
            prior_plan = real_document(refs['plan']['path'])
            request['configuration'] = copy.deepcopy(prior_plan['configuration'])
            request['reference_configuration'] = copy.deepcopy(prior_plan['reference_configuration'])
            request['copy_seeds'] = [1, 2]
            assert request['configuration']['learner']['n_copies'] == 2
            assert request['serial_order'] == 1

            def read(name):
                if str(name) == '/synthetic/reader-only-snapshot-not-admitted':
                    return {'accepted': {'correctness': {'2': {key: refs[key] for key in ('plan', 'result')}}}}
                return real_document(name)

            with patch.object(control, 'PACKET', packet), patch.object(control, 'RUN', original_tests.RUN), patch.object(control, 'document', side_effect=read):
                # All predecessor bytes are genuine. Only the new request's count/order join is under test.
                with self.assertRaises(ValueError):
                    control.check_serial_predecessor_inputs(request, control.receipt(path))


suite = unittest.TestSuite()
suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(original_tests.TimingControllerMetadataTests))
suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentTimingBoundaryTests))
result = unittest.TextTestRunner(verbosity=2).run(suite)
after = capture_frozen()
assert before == after
assert not {'jax', 'jaxlib', 'numpy', 'torch', 'cupy', 'flax', 'optax', 'scipy', 'jaxtari', 'exploration_platform'}.intersection(sys.modules)
record = {
    'kind': 'independent_frozen_common_timing_controller_review',
    'passed': result.wasSuccessful(),
    'source_freeze_sha256': EXPECTED_FREEZE,
    'handoff_sha256': EXPECTED_HANDOFF,
    'actual_job_id': os.environ['SLURM_JOB_ID'],
    'actual_step_id': os.environ['SLURM_STEP_ID'],
    'actual_pid': os.getpid(),
    'tests_run': result.testsRun,
    'failures': [{'test': str(test), 'traceback': trace} for test, trace in result.failures],
    'errors': [{'test': str(test), 'traceback': trace} for test, trace in result.errors],
    'elapsed_seconds': time.monotonic() - started,
    'frozen_files_before': before,
    'frozen_files_unchanged': before == after,
    'original_numerical_files_verified_unchanged': 13,
    'numerical_imports': [],
    'actual_packet_or_timing_unit_prepared': False,
    'remote_actions': 0,
    'gpu_operations': 0,
    'native_process_control_operations': 0,
    'scientific_credit': 0,
}
destination = REVIEW / ('review_step_' + os.environ['SLURM_STEP_ID'] + '.json')
with destination.open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps({'passed': record['passed'], 'tests_run': result.testsRun,
                  'step': os.environ['SLURM_STEP_ID'], 'record': str(destination)}))
raise SystemExit(0 if result.wasSuccessful() else 1)
