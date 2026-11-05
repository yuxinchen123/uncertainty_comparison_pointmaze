"""Independently replay exact399 and final-custody budget classification controls."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

W = Path(__file__).resolve().parents[2]
M = W / '.review_metadata/direct_diagnostic_time_budget'
S = W / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
STEP = os.environ.get('SLURM_STEP_ID', '')
OUT = M / ('late_completion_independent_' + STEP + '.json')
ARTIFACTS = M / ('late_completion_independent_metadata_' + STEP)
assert os.environ.get('SLURM_JOB_ID') == '6783003' and STEP.isdecimal()
assert os.environ.get('SLURM_CPUS_PER_TASK') == '1' and not os.environ.get('SLURM_STEP_GPUS')
assert not os.environ.get('CUDA_VISIBLE_DEVICES') and not OUT.exists() and not ARTIFACTS.exists()
ARTIFACTS.mkdir()
hashes, findings = {}, []


def sha(path):
    path = Path(path)
    value = hashlib.sha256(path.read_bytes()).hexdigest();hashes[str(path)] = value
    return value


def save(name, value):
    path = ARTIFACTS / (name + '.json')
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


freeze_path, handoff_path = M / 'late_completion_source_freeze.json', M / 'late_completion_author_handoff.json'
assert sha(freeze_path) == 'ca727b26b0394425922de07674470be635dfa6d8c3a80186e673e574f089e6f1'
assert sha(handoff_path) == '3ee8fccc5c83052c786da7a6d607756f99fff627313f1005c0311ccce4123fa5'
freeze, handoff = json.loads(freeze_path.read_bytes()), json.loads(handoff_path.read_bytes())
assert len(freeze['protected_files']) == 34
for path, expected in freeze['files'].items():assert sha(W / path) == expected
for path, expected in freeze['protected_files'].items():assert sha(path) == expected
for key in ('author_execution', 'preserved_rejected_source', 'preserved_rejected_review'):
    ref = handoff[key];assert sha(ref['path']) == ref['sha256']
preserved = json.loads(Path(handoff['preserved_rejected_source']['path']).read_bytes())
assert preserved['source_files'] == 9
for record in preserved['files'].values():assert sha(record['path']) == record['sha256']
old_freeze = json.loads(Path(preserved['files']['author_source_freeze_6.json']['path']).read_bytes())
changed = [name for name, expected in old_freeze['files'].items() if freeze['files'][name] != expected]
assert changed == ['10_jax_exploration_platform/scripts/atari_pixel_campaign/direct_tool_descendants.py']
scope_text = (S / 'direct_tool_descendants.py').read_text()
for name, expected in freeze['unchanged_tool_scope_function_hashes'].items():
    nodes = [node for node in ast.walk(ast.parse(scope_text)) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(nodes) == 1 and hashlib.sha256(ast.get_source_segment(scope_text, nodes[0]).encode()).hexdigest() == expected
before = dict(hashes)
sys.path.insert(0, str(S));sys.path.insert(0, str(S / 'tests'))
import test_direct_diagnostic_time_budget as tested
import test_direct_diagnostic_late_completion as corrected
direct, scope_source = tested.direct, tested.scope_source
os.environ['LATE_DIAGNOSTIC_COMPLETION_REVIEW_OUTPUT'] = str(ARTIFACTS / 'final_classification_cases')
# Load only the exact rejected independent class, never the old full test runner.
old_program = Path(preserved['files']['run_independent_review.py']['path'])
old_class = next(node for node in ast.parse(old_program.read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name == 'IndependentSlowClosureBoundary')
exec(compile(ast.Module(body=[old_class], type_ignores=[]), str(old_program), 'exec'), globals())
started = time.time()
suite = unittest.TestSuite([
    unittest.defaultTestLoader.loadTestsFromTestCase(IndependentSlowClosureBoundary),
    unittest.defaultTestLoader.loadTestsFromTestCase(corrected.LateDiagnosticCompletionTests)])
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert all(sha(path) == expected for path, expected in before.items())
assert time.time() - started < 240
assert not any(name.split('.')[0] in {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'cupy', 'jaxtari', 'exploration_platform'} for name in sys.modules)
raw_artifacts = {str(path): sha(path) for path in ARTIFACTS.rglob('*.json')}
payload = {'kind': 'independent_corrected_late_closure_and_custody_budget_review',
    'passed': result.wasSuccessful(), 'source_approved': result.wasSuccessful(),
    'job_id': os.environ['SLURM_JOB_ID'], 'step_id': STEP, 'pid': os.getpid(),
    'sys_executable': sys.executable, 'requested_memory_mib': 512, 'hard_slurm_time_limit_seconds': 300,
    'started_at_seconds': started, 'finished_at_seconds': time.time(), 'tests_run': result.testsRun,
    'failures': [{'test': str(test), 'traceback': text} for test, text in result.failures],
    'errors': [{'test': str(test), 'traceback': text} for test, text in result.errors],
    'findings': findings, 'files_sha256': hashes, 'metadata_fixture_sha256': raw_artifacts,
    'source_freeze': {'path': str(freeze_path), 'sha256': sha(freeze_path)},
    'review_program': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).resolve())},
    'other_eight_prior_frozen_paths_unchanged': True, 'protected_files_unchanged': 34,
    'unchanged_scope_functions': 18, 'exact_original399_class_replayed': True,
    'actual_process_control_or_lock_or_target_or_Nsight_GPU_operations': 0,
    'scientific_credit': 0, 'admission_granted': False}
raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
with OUT.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
print(json.dumps({'passed': payload['passed'], 'tests_run': result.testsRun, 'failures': len(result.failures),
                  'errors': len(result.errors), 'receipt': str(OUT), 'sha256': hashlib.sha256(raw).hexdigest()}))
raise SystemExit(0 if result.wasSuccessful() else 1)
