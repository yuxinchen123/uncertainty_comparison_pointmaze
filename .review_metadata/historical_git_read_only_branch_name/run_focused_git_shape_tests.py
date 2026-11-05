"""Exercise only source-level historical Git call shapes on the exact owned CPU allocation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
sys.path.insert(0, str(SCRIPTS))
import direct_common_confirmation as guard


def audit(event, values):
    """Forbid real processes, numerical imports, devices, later library loads and network operations."""
    if event in {'subprocess.Popen', 'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'ctypes.dlopen', 'ctypes.dlsym'}:
        raise RuntimeError('live operation forbidden in historical Git call-shape fixture')
    if event == 'import' and values[0].split('.')[0] in {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'jaxtari', 'nvidia', 'cupy'}:
        raise RuntimeError('numerical import forbidden in historical Git call-shape fixture')
    if event == 'open' and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(('/dev/nvidia', '/dev/dri')):
        raise RuntimeError('device operation forbidden in historical Git call-shape fixture')


def main():
    """Record real step/process and exact source around the finite synthetic command-shape controls."""
    if os.environ.get('SLURM_JOB_ID') != '6783003' or not os.environ.get('SLURM_STEP_ID', '').isdigit():
        raise RuntimeError('use the exact owned numeric CPU step for these source fixtures')
    files = [SCRIPTS / 'direct_common_confirmation.py', SCRIPTS / 'tests/test_direct_historical_git_inspection.py',
        Path(__file__), Path(__file__).with_name('old_twenty_six_git_call_inventory.json')]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    specification = importlib.util.spec_from_file_location('historical_git_shape_tests', files[1])
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    bootstrap = guard.stdlib_ctypes_bootstrap()
    guard.validate_bootstrap_record(bootstrap)
    sys.addaudithook(audit)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if before != {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}:
        raise RuntimeError('historical Git call-shape source changed during checks')
    import direct_historical_consumer as historical
    print(json.dumps({'kind': 'historical_git_read_only_call_shape_author_metadata_receipt',
        'passed': result.wasSuccessful(), 'tests_run': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures),
        'actual_job_id': os.environ['SLURM_JOB_ID'], 'actual_step_id': os.environ['SLURM_STEP_ID'],
        'actual_process': historical.process(os.getpid()), 'source_sha256': before,
        'stdlib_ctypes_bootstrap': bootstrap, 'actual_git_commands_executed': 0,
        'old_source_function_shapes_used_only_with_synthetic_replies': True,
        'saved_scientific_results_replayed': False, 'numerical_imports': [], 'gpu_operations': 0,
        'scientific_credit': 0, 'production_guard_qualified': False, 'admission_granted': False}, indent=2, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
