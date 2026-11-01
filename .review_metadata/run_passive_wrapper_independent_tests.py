"""Run only synthetic wrapper metadata on an already owned zero-GPU CPU allocation."""

import ast
import re
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
EXPECTED_HASHES = {
    'run_passive_installed_slurm_observation.py': '6fb870494ea065c1c66cf37a4141e2b766b74b0bb7cbd2cff931047770b7b011',
    'tests/test_run_passive_installed_slurm_observation.py': 'c2d327b4b3aa9a9b0b0b5155f21e1abaee9af9eb7d5017f8c41ede39df3021b5',
    'passive_installed_slurm_observation.py': '97ce634c07cd7a2b045d4253938db21c844abd93f3bb7b9d4dbebfd5f33bf842',
    'uva_active_configuration_observation.py': '8cc5a3544e426e3bd3e78707f6c77dc249a1d19165a593e157d2097daa07a648',
}
assert {name: hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest() for name in EXPECTED_HASHES} == EXPECTED_HASHES
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
            and (command[1:3] == ['init', '-q'] or (len(command) > 3 and command[1] == '-C'
                and str(command[2]).startswith('/tmp/')
                and command[3] in {'add', '-c', 'checkout', 'rev-parse', 'status', 'show', 'symbolic-ref'})))
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
process_stat = Path('/proc/self/stat').read_text()
process_birth = process_stat[process_stat.rfind(')') + 2:].split()[19]
process_cgroup = Path('/proc/self/cgroup').read_text()
os.environ.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null', GIT_CONFIG_COUNT='1',
    GIT_CONFIG_KEY_0='core.hooksPath', GIT_CONFIG_VALUE_0='/dev/null', GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS='0')
builtins.__import__ = guarded_import
sys.addaudithook(audit)
sys.path.insert(0, str(SCRIPTS / 'tests'))
stream = io.StringIO()
started = time.monotonic()
suite = unittest.defaultTestLoader.loadTestsFromName('test_run_passive_installed_slurm_observation')
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)

# Check extra operational boundaries without altering either frozen source file.
import test_run_passive_installed_slurm_observation as fixtures
from unittest.mock import patch
checks = []

def temporary_case():
    """Create one independent synthetic fixture and register its own cleanup."""
    case = fixtures.PassiveWrapperTests('test_standard_preparation_has_exact_three_sources_and_null_identity')
    case.setUp()
    return case

case = temporary_case()
try:
    unit = case.prepare()
    case.bind(unit)
    fixture = case.allocation()
    fixture['job'] = fixture['job'].replace('NumCPUs=2', 'NumCPUs=1').replace('AllocTRES=cpu=2', 'AllocTRES=cpu=1')
    stack, call = case.execution(unit, fixture)
    topology = {'capture_status': 'completed', 'returncode': 0, 'stderr_byte_count': 0,
        'filter_status': 'parsed', 'fields': [{'name': 'NodeName', 'value': 'jaguar03'},
            {'name': 'ThreadsPerCore', 'value': '1'}]}
    with stack, patch.object(fixtures.wrapper.passive, 'controller_query', return_value=topology):
        actual = fixtures.wrapper.run(unit)
    assert actual['completed'] is True and call.call_count == 1
    checks.append({'name': 'one_actual_cpu_with_one_thread_topology_passes', 'passed': True})
finally:
    case.doCleanups()

case = temporary_case()
try:
    unit = case.prepare()
    case.bind(unit)
    path = unit / 'passive_held_identifier_binding.json'
    record = json.loads(path.read_bytes())
    record['schema_version'] = True
    path.write_text(json.dumps(record))
    stack, call = case.execution(unit, case.allocation())
    with stack, patch.object(fixtures.wrapper, 'owned_query') as query:
        try:
            fixtures.wrapper.run(unit)
        except ValueError:
            rejected = True
        else:
            rejected = False
    assert rejected and query.call_count == 0 and call.call_count == 0
    assert json.loads((unit / 'failure.json').read_bytes())['completed'] is False
    checks.append({'name': 'boolean_held_binding_schema_refuses_before_query', 'passed': True})
finally:
    case.doCleanups()

case = temporary_case()
try:
    unit = case.prepare()
    case.bind(unit)
    stack, call = case.execution(unit, case.allocation())
    topology = {'capture_status': 'completed', 'returncode': 0, 'stderr_byte_count': 0,
        'filter_status': 'parsed', 'fields': [{'name': 'NodeName', 'value': 'jaguar02'},
            {'name': 'ThreadsPerCore', 'value': '2'}]}
    with stack, patch.object(fixtures.wrapper.passive, 'controller_query', return_value=topology):
        try:
            fixtures.wrapper.run(unit)
        except ValueError:
            rejected = True
        else:
            rejected = False
    assert rejected and call.call_count == 0
    assert json.loads((unit / 'observation.json').read_bytes())['completed'] is False
    checks.append({'name': 'foreign_node_topology_cannot_widen_cpu_allocation', 'passed': True})
finally:
    case.doCleanups()

after = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in FILES}
record = {'schema_version': 1, 'kind': 'passive_wrapper_independent_metadata_tests',
    'at_utc': datetime.now(timezone.utc).isoformat(), 'job_id': job, 'step_id': step,
    'hostname': os.uname().nodename, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
    'pid': os.getpid(), 'process_start_time': process_birth, 'cgroup': process_cgroup,
    'extra_operational_checks': checks, 'source_approved': True,
    'limitations': ['synthetic metadata only; no actual collector or controller query',
        'source approval does not approve an unreviewed packet or launch',
        'no physical mapping, enforcement, qualification or admission'],
    'worktree': str(ROOT), 'base': 'f5b353822d3383cac3082c2442e7e9e5a0dd7279',
    'source_hashes_before': before, 'source_hashes_after': after,
    'tests': result.testsRun, 'expected_tests': expected, 'failures': len(result.failures),
    'errors': len(result.errors), 'seconds': time.monotonic() - started,
    'stdout': stream.getvalue(), 'blocked': blocked, 'local_git_commands': len(git_calls),
    'passed': result.wasSuccessful() and result.testsRun == expected == 18 and before == after and not blocked and len(checks) == 3,
    'real_scheduler_queries': 0, 'slurmd_executed': False, 'GPU_or_device_access': False,
    'environment_transitions': 0, 'case_credit': 0, 'idea_credit': 0, 'admission_granted': False}
path = ROOT / '.review_metadata' / ('passive_wrapper_independent_' + job + '_' + step + '.json')
with path.open('x') as output:
    json.dump(record, output, indent=2, allow_nan=False)
    output.write('\n')
    output.flush()
    os.fsync(output.fileno())
print(json.dumps({'passed': record['passed'], 'tests': result.testsRun, 'receipt': str(path),
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'seconds': record['seconds'], 'extra_checks': len(checks)}, indent=2))
if not record['passed']:
    print(stream.getvalue())
    raise SystemExit(1)
