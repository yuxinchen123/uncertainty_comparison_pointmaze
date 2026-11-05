"""Run only temporary input-snapshot fixtures, never the actual guard observation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

PACKET = Path(__file__).resolve(strict=True).parent
NUMERICAL = {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'jaxtari', 'nvidia', 'cupy', 'exploration_platform'}


def audit(event, values):
    """Reject every real process, numerical import, device or network operation during these synthetic tests."""
    if event == 'subprocess.Popen' or event in {'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'ctypes.dlopen', 'ctypes.dlsym'}:
        raise RuntimeError('live process, library or network operation forbidden in snapshot fixture')
    if event == 'import' and values[0].split('.')[0] in NUMERICAL:
        raise RuntimeError('numerical import forbidden in snapshot fixture')
    if event == 'open' and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(('/dev/nvidia', '/dev/dri')):
        raise RuntimeError('device operation forbidden in snapshot fixture')


def main():
    """Record real owned CPU identity and exact source around the finite temporary metadata tests."""
    if os.environ.get('SLURM_JOB_ID') != '6783003' or not os.environ.get('SLURM_STEP_ID', '').isdigit():
        raise RuntimeError('snapshot fixtures require this exact owned numeric CPU step')
    paths = [PACKET / name for name in ('run_guard_observation.py', 'observe_guard_phases.py',
        'test_run_guard_observation.py', 'run_wrapper_metadata_tests.py')]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    specification = importlib.util.spec_from_file_location('snapshot_fixture_tests', PACKET / 'test_run_guard_observation.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    sys.addaudithook(audit)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if before != {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} or NUMERICAL.intersection(sys.modules):
        raise RuntimeError('snapshot fixture source changed or a numerical package was imported')
    print(json.dumps({'kind': 'temporary_snapshot_wrapper_metadata_fixture_receipt', 'passed': result.wasSuccessful(),
        'tests_run': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures),
        'actual_job_id': os.environ['SLURM_JOB_ID'], 'actual_step_id': os.environ['SLURM_STEP_ID'],
        'actual_process': module.observer.process(os.getpid()), 'source_sha256': before,
        'actual_guard_observation_invoked': False, 'existing_scientific_inputs_consumed': False,
        'numerical_imports': [], 'gpu_operations': 0, 'scientific_credit': 0,
        'production_guard_qualified': False, 'admission_granted': False}, indent=2, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
