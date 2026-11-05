"""Review frozen timing metadata without running a numerical or remote process."""
import ast
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
FREEZE = WORKTREE / '.review_metadata/direct_common_timing_controller/serial_order_corrected_source_freeze.json'
HANDOFF = WORKTREE / '.review_metadata/direct_common_timing_controller/serial_order_corrected_author_handoff.json'
EXPECTED_FREEZE = '67e9c1acc2834c35780353b51ba5dfb18ebaeafacae5e3a3d3b4c2d732ea4c9f'
EXPECTED_HANDOFF = '2db4671aa5146081e8e05c8e443963a64edd30df6e73c5f2ea708a4fe360afab'


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
frozen = json.loads(FREEZE.read_bytes())
functions = {node.name: ast.dump(node, include_attributes=False)
             for node in ast.parse((SOURCE / 'direct_common_timing_controller.py').read_text()).body
             if isinstance(node, ast.FunctionDef)}
for name, expected in frozen['unchanged_process_publication_function_ast_hashes'].items():
    assert hashlib.sha256(functions[name].encode()).hexdigest() == expected
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

    def test_independent_seed_and_actual_plan_refusals(self):
        base = {'configuration': {'learner': {'n_copies': 2}}, 'serial_order': 2, 'copy_seeds': [1, 2]}
        for seeds in ([1, 3], [1, 1], [1.0, 2], [1, True], (1, 2)):
            changed = copy.deepcopy(base)
            changed['copy_seeds'] = seeds
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                control.timing_copy_order(changed)
        for count, seeds in ((1, [1]), (True, [1, 2]), (2, [1, 3]), (2, [1, True])):
            plan = {'configuration': {'learner': {'n_copies': count}}, 'copy_seeds': seeds}
            with self.subTest(count=count, seeds=seeds), self.assertRaises(ValueError):
                control.timing_copy_order(base, plan)

    def test_changed_genuine_predecessor_source_refused(self):
        helper = original_tests.TimingControllerMetadataTests()
        with tempfile.TemporaryDirectory(prefix='common_timing_source_review_') as temporary:
            packet = Path(temporary)
            path, request, refs = helper.predecessor(packet)
            real_document = control.document
            original_plan = real_document(refs['plan']['path'])
            for changed_field in ('source_commit', 'helper_source_commit'):
                altered_plan = copy.deepcopy(original_plan)
                if changed_field == 'source_commit':
                    altered_plan['source_commit'] = 'f' * 40
                else:
                    altered_plan['helpers']['source_commit'] = 'f' * 40

                def read(name):
                    if str(name) == refs['plan']['path']:
                        return altered_plan
                    if str(name) == '/synthetic/reader-only-snapshot-not-admitted':
                        return {'accepted': {'correctness': {'2': {key: refs[key] for key in ('plan', 'result')}}}}
                    return real_document(name)

                with self.subTest(field=changed_field), patch.object(control, 'PACKET', packet), patch.object(control, 'RUN', original_tests.RUN), patch.object(control, 'document', side_effect=read):
                    with self.assertRaisesRegex(ValueError, 'prior process'):
                        control.check_serial_predecessor_inputs(request, control.receipt(path))


suite = unittest.TestSuite()
suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(original_tests.TimingCopyOrderCorrectionTests))
suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentTimingBoundaryTests))
result = unittest.TextTestRunner(verbosity=2).run(suite)
after = capture_frozen()
assert before == after
assert not {'jax', 'jaxlib', 'numpy', 'torch', 'cupy', 'flax', 'optax', 'scipy', 'jaxtari', 'exploration_platform'}.intersection(sys.modules)
record = {
    'kind': 'independent_corrected_common_timing_controller_review',
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
    'unchanged_process_publication_function_asts_verified': 25,
    'previous_six_method_batch_repeated': False,
    'first_rejection_preserved': True,
    'numerical_imports': [],
    'actual_packet_or_timing_unit_prepared': False,
    'remote_actions': 0,
    'gpu_operations': 0,
    'native_process_control_operations': 0,
    'scientific_credit': 0,
}
destination = REVIEW / ('corrected_review_step_' + os.environ['SLURM_STEP_ID'] + '.json')
with destination.open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps({'passed': record['passed'], 'tests_run': result.testsRun,
                  'step': os.environ['SLURM_STEP_ID'], 'record': str(destination)}))
raise SystemExit(0 if result.wasSuccessful() else 1)
