"""Load only the focused stdlib CPU fixture module by its exact local path."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
W = HERE.parents[1]
path = W / '10_jax_exploration_platform/scripts/atari_pixel_campaign/tests/test_profile_metadata_runtime.py'
spec = importlib.util.spec_from_file_location('focused_profile_metadata_runtime_checks', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
started = time.time()
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
receipt = {'step_id': os.environ.get('SLURM_STEP_ID'), 'job_id': os.environ.get('SLURM_JOB_ID'),
    'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
    'passed': result.wasSuccessful(), 'started_at_seconds': started, 'finished_at_seconds': time.time(),
    'numerical_or_GPU_work': False, 'operational_admission': False}
target = HERE / ('runtime_checks_' + str(receipt['job_id']) + '_' + str(receipt['step_id']) + '.json')
with target.open('x') as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True);handle.write('\n');handle.flush();os.fsync(handle.fileno())
raise SystemExit(not result.wasSuccessful())
