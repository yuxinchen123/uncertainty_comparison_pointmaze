"""Five metadata-only author checks; source must match the held whole union."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FREEZE = ROOT / '.review_metadata/configured_native_process/corrected_union_source_freeze.json'
EXPECTED = '36524765c7fba07fad274b0ee170d03df05ead3ea2a1217ffbbfb22ecd84dd39'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inspect(frozen):
    actual = {}
    for relative, expected in frozen['files'].items():
        path = ROOT / relative
        entry = {'sha256': sha(path), 'mode': oct(stat.S_IMODE(path.stat().st_mode))}
        if entry != expected:
            raise ValueError('held source mismatch: ' + relative)
        actual[relative] = entry
    return actual

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

def main():
    started = time.time()
    if os.environ.get('SLURM_JOB_ID') != '6821437' or not os.environ.get('SLURM_STEP_ID', '').isdigit():
        raise ValueError('only exact owned numeric CPU step admitted')
    if sha(FREEZE) != EXPECTED:
        raise ValueError('union freeze changed')
    frozen = json.loads(FREEZE.read_bytes())
    before = inspect(frozen)
    sys.path.insert(0, str(ROOT / '10_jax_exploration_platform/scripts/atari_pixel_campaign/tests'))
    suite = unittest.defaultTestLoader.loadTestsFromName('test_configured_native_profile_plumbing.NativeProfilePlumbingTests')
    if suite.countTestCases() != 5:
        raise ValueError('five-method scope changed')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = inspect(frozen)
    step = os.environ['SLURM_STEP_ID']
    record = {
        'kind': 'configured_native_profile_plumbing_five_author_checks',
        'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
        'failures': [{'test': str(test), 'traceback': trace} for test, trace in result.failures],
        'errors': [{'test': str(test), 'traceback': trace} for test, trace in result.errors],
        'freeze': {'path': str(FREEZE), 'sha256': EXPECTED},
        'source_before': before, 'source_after': after,
        'source_unchanged': before == after,
        'job_id': os.environ['SLURM_JOB_ID'], 'step_id': step,
        'pid': os.getpid(), 'host': os.uname().nodename,
        'started_at_seconds': started, 'finished_at_seconds': time.time(),
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'synthetic_upstream_and_tool_premises': True,
        'actual_target_qualification': False, 'formal_profile_admission': False,
        'scientific_credit': 0,
    }
    path = HERE / ('five_author_checks_6821437_' + step + '.json')
    write(path, record)
    print(json.dumps({'receipt': str(path), 'sha256': sha(path), 'passed': record['passed'],
        'tests_run': result.testsRun, 'step_id': step, 'elapsed_seconds': record['finished_at_seconds'] - started}, sort_keys=True))
    if not result.wasSuccessful():
        raise SystemExit(1)

if __name__ == '__main__':
    main()
