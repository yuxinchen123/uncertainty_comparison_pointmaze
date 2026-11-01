"""Run only synthetic wrapper metadata on an already owned zero-GPU CPU allocation."""

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
FILES = [SCRIPTS / name for name in ('run_passive_installed_slurm_observation.py',
    'tests/test_run_passive_installed_slurm_observation.py', 'passive_installed_slurm_observation.py',
    'uva_active_configuration_observation.py')]
before = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in FILES}
expected = sum(isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
    for node in ast.walk(ast.parse(FILES[1].read_text())))
original_import = builtins.__import__
blocked, git_calls = [], []


def guarded_import(name, *args, **kwargs):
    """Refuse numerical or GPU modules before their module bodies run."""
    if name.split('.')[0] in {'jax', 'jaxlib', 'numpy', 'torch', 'pynvml', 'jaxtari',
            'exploration_platform', 'cupy', 'numba', 'matplotlib', 'scipy'}:
        blocked.append({'event': 'numerical_import', 'name': name})
        raise AssertionError('numerical import forbidden: ' + name)
    return original_import(name, *args, **kwargs)


def audit(event, args):
    """Permit local Git fixture/source operations and reject all live commands and GPU/network access."""
    if event == 'subprocess.Popen':
        command = args[1]
        allowed = (isinstance(command, (list, tuple)) and command[0] == 'git'
            and not set(command) & {'push', 'pull', 'fetch', 'clone', 'remote', 'ls-remote', 'submodule'})
        if allowed:
            git_calls.append(command)
            return
        blocked.append({'event': event, 'command': repr(command)})
        raise AssertionError('non-Git subprocess forbidden in metadata tests')
    if event in {'os.system', 'os.posix_spawn', 'socket.connect', 'socket.getaddrinfo', 'ctypes.dlopen'}:
        blocked.append({'event': event})
        raise AssertionError('live library, network or command operation forbidden: ' + event)
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if path.startswith(('/dev/nvidia', '/dev/dri')) or __import__('re').match(r'^/proc/[0-9]+(?:/|$)', path):
            blocked.append({'event': 'forbidden_path', 'path': path})
            raise AssertionError('device or foreign-process path forbidden')


# Read the actual allocation identifiers before tests temporarily install synthetic environment fixtures.
job, step = os.environ.get('SLURM_JOB_ID'), os.environ.get('SLURM_STEP_ID')
assert job == '6781231' and step.isdigit() and os.environ.get('SLURM_CPUS_PER_TASK') == '1'
builtins.__import__ = guarded_import
sys.addaudithook(audit)
sys.path.insert(0, str(SCRIPTS / 'tests'))
stream = io.StringIO()
started = time.monotonic()
suite = unittest.defaultTestLoader.loadTestsFromName('test_run_passive_installed_slurm_observation')
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
after = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in FILES}
record = {'schema_version': 1, 'kind': 'passive_wrapper_author_metadata_tests',
    'at_utc': datetime.now(timezone.utc).isoformat(), 'job_id': job, 'step_id': step,
    'hostname': os.uname().nodename, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
    'worktree': str(ROOT), 'base': 'f5b353822d3383cac3082c2442e7e9e5a0dd7279',
    'source_hashes_before': before, 'source_hashes_after': after,
    'tests': result.testsRun, 'expected_tests': expected, 'failures': len(result.failures),
    'errors': len(result.errors), 'seconds': time.monotonic() - started,
    'stdout': stream.getvalue(), 'blocked': blocked, 'local_git_commands': len(git_calls),
    'passed': result.wasSuccessful() and result.testsRun == expected and before == after and not blocked,
    'real_scheduler_queries': 0, 'slurmd_executed': False, 'GPU_or_device_access': False,
    'environment_transitions': 0, 'case_credit': 0, 'idea_credit': 0, 'admission_granted': False}
path = ROOT / '.review_metadata' / ('passive_wrapper_author_' + job + '_' + step + '.json')
with path.open('x') as output:
    json.dump(record, output, indent=2, allow_nan=False)
    output.write('\n')
    output.flush()
    os.fsync(output.fileno())
print(json.dumps({'passed': record['passed'], 'tests': result.testsRun, 'receipt': str(path),
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'seconds': record['seconds']}, indent=2))
if not record['passed']:
    print(stream.getvalue())
    raise SystemExit(1)
