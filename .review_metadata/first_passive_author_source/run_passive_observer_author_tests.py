"""Run only the frozen passive metadata cases inside an already owned CPU allocation."""

import ast
import builtins
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
FILES = [SCRIPTS / 'passive_installed_slurm_observation.py',
    SCRIPTS / 'tests/test_passive_installed_slurm_observation.py',
    SCRIPTS / 'uva_active_configuration_observation.py']
BEFORE = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in FILES}
ORIGINAL_IMPORT = builtins.__import__
BLOCKED = []


def guarded_import(name, *args, **kwargs):
    """Reject numerical, GPU and training imports before any module body executes."""
    if name.split('.')[0] in {'jax', 'jaxlib', 'numpy', 'torch', 'pynvml', 'jaxtari',
            'exploration_platform', 'cupy', 'numba', 'matplotlib', 'scipy'}:
        BLOCKED.append({'kind': 'numerical_import', 'name': name})
        raise AssertionError('numerical or GPU import forbidden in metadata tests: ' + name)
    return ORIGINAL_IMPORT(name, *args, **kwargs)


def audit(event, args):
    """Reject actual network, subprocess, library, device and foreign-process operations."""
    if event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'socket.connect',
            'socket.getaddrinfo', 'ctypes.dlopen'}:
        BLOCKED.append({'kind': 'forbidden_audit_event', 'event': event})
        raise AssertionError('live command, network or library operation forbidden: ' + event)
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if path.startswith(('/dev/nvidia', '/dev/dri')) or __import__('re').match(r'^/proc/[0-9]+(?:/|$)', path):
            BLOCKED.append({'kind': 'forbidden_path', 'path': path})
            raise AssertionError('device or foreign process read forbidden: ' + path)


# The exact job and numeric step identify this author execution without a guessed step number.
assert os.environ.get('SLURM_JOB_ID') == '6781231'
assert os.environ.get('SLURM_STEP_ID', '').isdigit()
assert os.environ.get('SLURM_CPUS_PER_TASK') == '1'
builtins.__import__ = guarded_import
sys.addaudithook(audit)
sys.path.insert(0, str(SCRIPTS / 'tests'))
stream = io.StringIO()
started = time.monotonic()
suite = unittest.defaultTestLoader.loadTestsFromName('test_passive_installed_slurm_observation')
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
elapsed = time.monotonic() - started
after = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in FILES}
record = {'schema_version': 1, 'kind': 'passive_installed_slurm_observer_author_metadata_tests',
    'at_utc': datetime.now(timezone.utc).isoformat(), 'job_id': os.environ['SLURM_JOB_ID'],
    'step_id': os.environ['SLURM_STEP_ID'], 'hostname': os.uname().nodename,
    'cpu_affinity': sorted(os.sched_getaffinity(0)), 'worktree': str(ROOT),
    'base': 'fb3071f3772ed836144f5876dbdda1113a5339d5', 'source_hashes_before': BEFORE,
    'source_hashes_after': after, 'source_unchanged': BEFORE == after,
    'test_count': result.testsRun, 'expected_test_count': 29,
    'failures': len(result.failures), 'errors': len(result.errors), 'seconds': elapsed,
    'stdout': stream.getvalue(), 'blocked_attempts': BLOCKED,
    'numerical_modules_loaded': [name for name in sys.modules if name.split('.')[0] in
        {'jax', 'jaxlib', 'numpy', 'torch', 'pynvml', 'jaxtari', 'exploration_platform', 'cupy', 'numba', 'matplotlib', 'scipy'}],
    'passed': result.wasSuccessful() and result.testsRun == 29 and BEFORE == after and not BLOCKED,
    'live_controller_query': False, 'slurmd_executed': False, 'GPU_or_device_access': False,
    'environment_transitions': 0, 'case_credit': 0, 'idea_credit': 0,
    'physical_gres_mapping_verified': False, 'kernel_device_enforcement_verified': False,
    'admission_granted': False}
path = ROOT / '.review_metadata' / ('passive_installed_slurm_author_' + record['job_id'] + '_' + record['step_id'] + '.json')
with path.open('x') as output:
    json.dump(record, output, indent=2, allow_nan=False)
    output.write('\n')
    output.flush()
    os.fsync(output.fileno())
print(json.dumps({'passed': record['passed'], 'test_count': result.testsRun, 'receipt': str(path),
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'failures': len(result.failures),
    'errors': len(result.errors), 'seconds': elapsed}, indent=2))
if not record['passed']:
    print(stream.getvalue())
    raise SystemExit(1)
