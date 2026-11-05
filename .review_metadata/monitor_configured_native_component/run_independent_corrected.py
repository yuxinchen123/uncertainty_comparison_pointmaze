"""Review only the two corrected monitor predicates and their exact held source."""
import ast
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

W = Path("/p/rlprojects/RND_worktrees/2026-11-05-08-12_monitor-configured-native-diagnostic_owned-ledger-visible_unknown-kinds-explicit_gpu-actions-0")
D = W / '.review_metadata/monitor_configured_native_component'
S = W / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
freeze = json.loads((D / 'author_source_freeze_2.json').read_text())

def reference(path):
    """Bind one actual small source or proof record."""
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def check():
    """Keep the corrected source, tests and original monitor baseline unchanged."""
    for name, value in freeze['files'].items():
        path = W / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == value['sha256']
        assert oct(stat.S_IMODE(path.stat().st_mode)) == value['mode']

assert os.environ['SLURM_JOB_ID'] == '6821437' and os.environ['SLURM_STEP_ID'].isdecimal()
started = time.monotonic();check()
before = (D / 'independent_preserved_initial_rejection/source/10_jax_exploration_platform/scripts/atari_pixel_campaign/monitor.py').read_text()
expected = before.replace('raw = stream.read()\n', 'raw = stream.read(1024 * 1024 + 1)\n')
expected = expected.replace('if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):',
    'if len(raw) != before.st_size or len(raw) > 1024 * 1024 or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):')
expected = expected.replace('failure = document(unit / "failure.json")[0] if (unit / "failure.json").exists() else None',
    'failure_present = (unit / "failure.json").exists()\n    failure = document(unit / "failure.json")[0] if failure_present else None')
expected = expected.replace('    if failure:\n        return {**base, "state": "failed", "progress": "component failure record retained; zero study credit"}',
    '    if failure_present:\n        return {**base, "state": "failed", "progress": "component failure record retained; zero study credit"}')
assert (S / 'monitor.py').read_text() == expected

# Reuse the already completed eleven-method pass; run only the changed predicates and two affected displays.
sys.path.insert(0, str(S))
spec = importlib.util.spec_from_file_location('independent_corrected_monitor_tests', S / 'tests/test_monitor_native_component.py')
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
names = ['test_component_metadata_reads_are_bounded_and_require_exact_snapshot_size',
    'test_any_present_failure_marker_prevents_completed_display',
    'test_actual_small_metadata_component_is_visible_with_zero_authority',
    'test_missing_progress_discovery_regression_and_no_duplicate_after_progress']
suite = unittest.defaultTestLoader.loadTestsFromNames(['NativeComponentMonitorTests.' + name for name in names], module)
result = unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful()

# Independently extract the exact nested reader and exercise bounded EOF and changed-size behavior.
tree = ast.parse((S / 'monitor.py').read_bytes())
display = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'configured_native_component_display')
reader = next(node for node in display.body if isinstance(node, ast.FunctionDef) and node.name == 'document')
namespace = {'Path': Path, 'stat': stat, 'os': os, 'hashlib': hashlib, 'json': json}
exec(compile(ast.fix_missing_locations(ast.Module(body=[reader], type_ignores=[])), '<exact-corrected-monitor-reader>', 'exec'), namespace)

class SyntheticStream:
    """Record the real requested bound without allocating a large payload."""
    def __init__(self, raw):
        self.raw = raw
    def __enter__(self):
        return self
    def __exit__(self, *arguments):
        return False
    def read(self, size=-1):
        observed.append(size);assert size == 1024 * 1024 + 1
        return self.raw

observed = []
with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / 'synthetic_small_metadata.json';path.write_bytes(b'{}')
    with patch.object(os, 'fdopen', return_value=SyntheticStream(b'{}')):
        assert namespace['document'](path)[0] == {}
    with patch.object(os, 'fdopen', return_value=SyntheticStream(b'{} ')):
        try:
            namespace['document'](path)
        except ValueError as error:
            assert 'changed during its read' in str(error)
        else:
            raise AssertionError('the changed-size read was accepted')
assert observed == [1024 * 1024 + 1] * 2
check()
receipt = {'kind': 'independent_corrected_native_component_monitor_source_review', 'approved': True, 'passed': True,
    'source_freeze': reference(D / 'author_source_freeze_2.json'), 'source_unchanged': True,
    'previous_eleven_case_pass_reused': reference(D / 'independent_initial_findings.json'),
    'preserved_rejected_source': reference(D / 'independent_preserved_initial_rejection/preservation_manifest.json'),
    'changed_predicates_and_actual_positive_methods': result.testsRun, 'independent_bounded_reader_cases': 2,
    'all_other_source_text_unchanged_since_initial_review': True,
    'actual_metadata_job_visible': '6825842', 'unknown_owned_kind_stays_unknown': True,
    'scientific_credit': 0, 'admission_authority': False, 'actual_current_monitor_deployment': False,
    'actual_scheduler_native_target_or_gpu_operations': False,
    'job_id': os.environ['SLURM_JOB_ID'], 'step_id': os.environ['SLURM_STEP_ID'],
    'elapsed_seconds': time.monotonic() - started,
    'required_before_monitor_handover': 'root prepares and publishes one replacement, then verifies actual runtime and first full report before old step drain/cancellation'}
output = D / 'independent_corrected_source_review.json'
with output.open('x') as stream:
    json.dump(receipt, stream, sort_keys=True, indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'review': reference(output), 'methods': result.testsRun, 'reader_cases': 2,
    'seconds': receipt['elapsed_seconds'], 'step_id': receipt['step_id']}))
