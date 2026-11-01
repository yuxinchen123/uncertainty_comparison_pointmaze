"""Run exact-source standard-library tests on the already owned CPU allocation only."""

import argparse
import builtins
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / '10_jax_exploration_platform/scripts/atari_pixel_campaign'


def main():
    """Pin source bytes before and after tests and reject live commands or numerical imports."""
    parser = argparse.ArgumentParser()
    parser.add_argument('modules', nargs='+')
    arguments = parser.parse_args()
    if os.environ.get('SLURM_JOB_ID') != '6781087':
        raise ValueError('these metadata tests require the exact currently owned CPU job 6781087')
    sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / 'tests')]
    paths = list(SCRIPTS.glob('*.py')) + list((SCRIPTS / 'tests').glob('*.py'))
    before = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    original_process, original_import = subprocess.Popen, builtins.__import__
    calls = []

    def local_git(command, *args, **kwargs):
        """Allow bounded local fixture Git operations and reject every other actual process."""
        if not isinstance(command, list) or command[0] != 'git' or any('://' in item for item in command if isinstance(item, str)):
            raise AssertionError('metadata tests may run only bounded local Git commands')
        allowed = {'init', 'add', 'commit', 'checkout', 'clone', 'rev-parse', 'symbolic-ref', 'status', 'ls-tree',
                   'show', 'diff', 'ls-files', 'hash-object', 'cat-file', 'rev-list'}
        operation = next((item for item in command[1:] if item in allowed), None)
        if operation is None or any(item in {'push', 'pull', 'fetch', 'ls-remote'} for item in command):
            raise AssertionError('no remote Git or other process is permitted')
        if operation == 'clone' and (not Path(command[-2]).is_absolute() or not Path(command[-1]).is_absolute()):
            raise AssertionError('fixture cloning requires two explicit local absolute paths')
        if operation in {'add', 'commit', 'checkout'}:
            target = Path(command[command.index('-C') + 1])
            if not target.is_relative_to(Path(tempfile.gettempdir())):
                raise AssertionError('tests cannot mutate a real project Git checkout')
        calls.append(operation)
        return original_process(command, *args, **kwargs)

    def standard_library(name, *args, **kwargs):
        """Stop a numerical import before executing any numerical package initializer."""
        if name.split('.')[0] in {'jax', 'jaxlib', 'numpy', 'torch', 'scipy'}:
            raise AssertionError('numerical packages are forbidden in these metadata tests: ' + name)
        return original_import(name, *args, **kwargs)

    def no_network(*args, **kwargs):
        """Reject external network or shell commands while allowing local socket-pair metadata tests."""
        raise AssertionError('network and shell commands are forbidden in these metadata tests')

    stream, started = io.StringIO(), time.monotonic()
    with patch.object(subprocess, 'Popen', side_effect=local_git), patch.object(builtins, '__import__', side_effect=standard_library), \
            patch.object(socket, 'create_connection', side_effect=no_network), patch.object(socket.socket, 'connect', side_effect=no_network), \
            patch.object(os, 'system', side_effect=no_network):
        suite = unittest.defaultTestLoader.loadTestsFromNames(arguments.modules)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    after = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if before != after:
        raise AssertionError('source changed while an exact-source test batch was running')
    record = {'schema_version': 1, 'kind': 'final_phase_metadata_author_tests',
        'at_utc': datetime.now(timezone.utc).isoformat(), 'passed': result.wasSuccessful(),
        'owned_job_id': os.environ['SLURM_JOB_ID'], 'owned_step_id': os.environ.get('SLURM_STEP_ID'),
        'process': {'hostname': socket.gethostname(), 'pid': os.getpid(),
                    'process_start_time': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                    'cgroup': Path('/proc/self/cgroup').read_text()},
        'modules': arguments.modules, 'tests': result.testsRun, 'seconds': time.monotonic() - started,
        'test_output': stream.getvalue(), 'source_sha256': before, 'source_unchanged': before == after,
        'actual_processes': {'local_git': len(calls), 'other': 0}, 'network_calls': 0,
        'numerical_imports': [name for name in ('jax', 'numpy', 'torch') if name in sys.modules],
        'gpu_jobs': 0, 'submissions': 0, 'release_calls': 0}
    raw = (json.dumps(record, indent=2) + '\n').encode()
    output = Path(__file__).parent / ('final_phase_metadata_author_' + datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S') + '.json')
    with output.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({'passed': record['passed'], 'tests': result.testsRun, 'seconds': record['seconds'],
        'step': record['owned_step_id'], 'receipt': str(output), 'sha256': hashlib.sha256(raw).hexdigest()}))
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit(1)


if __name__ == '__main__':
    main()
