import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import unittest

w = Path("/p/rlprojects/RND_worktrees/2026-11-05-07-34_a4500-profile-preparation_common-gpu-types-2_copies-1-2_reviewed-native-process-contract_admission-closed_gpu-operations-0")
d = w / '.review_metadata/configured_profile_prerequisites'
freeze = json.loads((d / 'source_freeze_2.json').read_text())

def check():
    for name, value in freeze['paths'].items():
        path = w / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == value['sha256']
        assert path.stat().st_mode & 511 == value['mode']

check()
started = time.monotonic()
path = w / '10_jax_exploration_platform/scripts/atari_pixel_campaign/tests/test_configured_profile_prerequisites.py'
spec = importlib.util.spec_from_file_location('author_configured_profile_tests', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
original = subprocess.check_output(['git', '-C', str(w), 'show',
    '90ea7a898d69ee23c51fff7cfbcfd0499a768e09:10_jax_exploration_platform/scripts/atari_pixel_campaign/profile_selected.py'])
current = (w / '10_jax_exploration_platform/scripts/atari_pixel_campaign/profile_selected.py').read_bytes()

def functions(raw):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

a, b = functions(original), functions(current)
assert a['_capture_core'] == b['_capture_core']
assert a['require_reviewed_profile_process_admission'] == b['require_reviewed_profile_process_admission']
check()
value = {'passed': result.wasSuccessful(), 'tests': result.testsRun, 'failures': len(result.failures),
    'errors': len(result.errors), 'seconds': time.monotonic() - started,
    'job_id': os.environ['SLURM_JOB_ID'], 'step_id': os.environ['SLURM_STEP_ID'],
    'source_freeze_sha256': hashlib.sha256((d / 'source_freeze_2.json').read_bytes()).hexdigest(),
    'changed_existing_functions': [name for name in a if a[name] != b[name]],
    'capture_core_unchanged': True, 'execution_gate_unchanged': True,
    'synthetic_reviews_not_actual_qualification': True, 'GPU_or_target_operations': False}
(d / 'focused_author_2.json').write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
print(json.dumps(value))
raise SystemExit(0 if value['passed'] else 1)
