"""Inspect only held monitor metadata code and explicit private failure fixtures."""
import ast
import contextlib
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
freeze = json.loads((D / 'author_source_freeze_1.json').read_text())

def check():
    """Keep all three held source/test/baseline files exact throughout this review."""
    for name, value in freeze['files'].items():
        path = W / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == value['sha256']
        assert oct(stat.S_IMODE(path.stat().st_mode)) == value['mode']

def reference(path):
    """Record one actual small reviewed file without making a future hash."""
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

assert os.environ.get('SLURM_JOB_ID') == '6821437' and os.environ.get('SLURM_STEP_ID', '').isdecimal()
check(); started = time.monotonic()
sys.path.insert(0, str(S))
spec = importlib.util.spec_from_file_location('independent_native_monitor_tests', S / 'tests/test_monitor_native_component.py')
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
assert result.wasSuccessful()

# Extract the exact nested reader; no changed runtime source or extra scheduler context is substituted.
tree = ast.parse((S / 'monitor.py').read_bytes())
display = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'configured_native_component_display')
reader = next(node for node in display.body if isinstance(node, ast.FunctionDef) and node.name == 'document')
namespace = {'Path': Path, 'stat': stat, 'os': os, 'hashlib': hashlib, 'json': json}
exec(compile(ast.fix_missing_locations(ast.Module(body=[reader], type_ignores=[])), '<exact-held-monitor-reader>', 'exec'), namespace)

class UnboundedReadObserved(Exception):
    """Stop the demonstration before it reads or allocates an oversized payload."""

class SyntheticStream:
    """Expose only the size argument actually requested by the held reader."""
    def __enter__(self):
        return self
    def __exit__(self, *arguments):
        return False
    def read(self, size=-1):
        observed_sizes.append(size)
        if size < 0:
            raise UnboundedReadObserved('held metadata reader called read without a size bound')
        return b'{}'

observed_sizes = []
with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / 'synthetic_small_metadata.json';path.write_bytes(b'{}')
    with patch.object(os, 'fdopen', return_value=SyntheticStream()):
        try:
            namespace['document'](path)
        except UnboundedReadObserved:
            pass
        else:
            raise AssertionError('the held reader unexpectedly supplied a size bound')
assert observed_sizes == [-1]

# A syntactically valid empty failure file must not disappear because its JSON value is false.
fixture = module.NativeComponentMonitorTests(methodName='test_actual_small_metadata_component_is_visible_with_zero_authority')
fixture.setUp()
try:
    fixture.write(fixture.unit / 'failure.json', {})
    empty_failure_row = fixture.row()
    assert empty_failure_row['state'] == 'completed'
finally:
    fixture.doCleanups()
check()
receipt = {'kind': 'independent_monitor_metadata_initial_findings', 'approved': False, 'passed': False,
    'author_cases_reproduced': result.testsRun, 'author_cases_all_passed': result.wasSuccessful(),
    'findings': [
        {'field': 'bounded ordinary metadata read', 'reproduced': True, 'observed_stream_read_sizes': observed_sizes,
            'requested_correction': 'read at most the fixed metadata limit plus one byte and verify exact size; do not allow a file growth race to allocate without a bound',
            'payload_read_or_allocated_by_reproducer': False},
        {'field': 'failure marker presence', 'reproduced': True, 'synthetic_empty_failure_row_state': empty_failure_row['state'],
            'requested_correction': 'a present failure file must prevent completed display even when parsed JSON is empty or null'}],
    'source_freeze': reference(D / 'author_source_freeze_1.json'), 'source_unchanged': True,
    'job_id': os.environ['SLURM_JOB_ID'], 'step_id': os.environ['SLURM_STEP_ID'],
    'elapsed_seconds': time.monotonic() - started, 'actual_scheduler_target_native_or_gpu_operations': False,
    'scientific_credit': 0, 'persistent_monitor_not_changed': True}
output = D / 'independent_initial_findings.json'
with output.open('x') as stream:
    json.dump(receipt, stream, sort_keys=True, indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'review': reference(output), 'tests': result.testsRun, 'findings': 2, 'seconds': receipt['elapsed_seconds'],
    'step_id': receipt['step_id']}))
