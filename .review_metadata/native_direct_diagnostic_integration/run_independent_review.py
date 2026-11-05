"""Independently check diagnostic-only native wiring on owned CPU fixtures."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

W = Path(__file__).resolve().parents[2]
M = W / '.review_metadata/native_direct_diagnostic_integration'
S = W / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
STEP = os.environ.get('SLURM_STEP_ID', '')
OUT = M / ('independent_' + STEP + '.json')
ARTIFACTS = M / ('independent_actual_cpu_' + STEP)
assert os.environ.get('SLURM_JOB_ID') == '6783003' and STEP.isdecimal()
assert os.environ.get('SLURM_CPUS_PER_TASK') == '1' and not os.environ.get('SLURM_STEP_GPUS')
assert not os.environ.get('CUDA_VISIBLE_DEVICES')
assert sys.executable == '/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python'
assert not OUT.exists() and not ARTIFACTS.exists()
ARTIFACTS.mkdir()
hashes, findings = {}, []


def sha(path):
    path = Path(path)
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes[str(path)] = value
    return value


def save(name, value):
    path = ARTIFACTS / (name + '.json')
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


freeze_path = M / 'author_source_freeze_5.json'
assert sha(freeze_path) == '4d389d484b091e3b3dfc125529b01e0633862d3101c4467c0d6bbd5b50b5548c'
assert sha(M / 'author_handoff.json') == '5b5e985908e60540a70387733d983def3b48cff4bacbec9d8b866dedf954c674'
freeze = json.loads(freeze_path.read_bytes())
assert len(freeze['protected_files']) == 34
for path, expected in freeze['files'].items():assert sha(W / path) == expected
for path, expected in freeze['protected_files'].items():assert sha(path) == expected
handoff = json.loads((M / 'author_handoff.json').read_bytes())
for value in handoff['actual_CPU_artifacts'].values():assert sha(value['path']) == value['sha256']
assert sha(handoff['author_execution']['path']) == handoff['author_execution']['sha256']
before = dict(hashes)
sys.path.insert(0, str(S));sys.path.insert(0, str(S / 'tests'))
import test_native_process_evidence as calls
import test_native_process_role_evidence as roles
import test_native_direct_process_cpu as actual
import test_native_diagnostic_order as order
evidence = calls.evidence
os.environ['NATIVE_DIRECT_CPU_REVIEW_OUTPUT'] = str(ARTIFACTS / 'native_scope')


class IndependentSavedFdinfoBoundaries(unittest.TestCase):
    """Mutate only explicitly synthetic saved metadata; perform no live native work."""
    def test_raw_fdinfo_inode_must_join_the_asserted_descriptor_inode(self):
        value, owner, allowed, contract = calls.synthetic_calls()
        # Positive fixture first, with native-loading explicitly forbidden.
        with patch.object(evidence, 'load_provider', side_effect=AssertionError('saved reader loaded native provider')):
            evidence.validate_calls(value, owner, allowed, contract)
            changed = copy.deepcopy(value)
            for event in changed['descriptor_events']:
                event['identity']['inode'] += 1000
            accepted = False
            error = None
            try:evidence.validate_calls(changed, owner, allowed, contract);accepted = True
            except (ValueError, KeyError, AssertionError) as failure:error = str(failure)
        finding = {'synthetic_saved_metadata_only': True, 'actual_native_calls': 0,
            'mutation': 'all per-generation summary inodes changed; retained raw fdinfo ino remains original',
            'accepted': accepted, 'error': error, 'original': value, 'changed': changed,
            'source_sha256': sha(S / 'native_process_evidence.py')}
        findings.append({'name': 'raw_fdinfo_inode_summary_mismatch', 'accepted_invalid_input': accepted,
                         'receipt': save('raw_fdinfo_inode_boundary', finding)})
        self.assertFalse(accepted, 'saved native reader accepted inode summaries contradicting retained raw fdinfo')

    def test_raw_fdinfo_must_retain_close_on_exec(self):
        value, owner, allowed, contract = calls.synthetic_calls()
        with patch.object(evidence, 'load_provider', side_effect=AssertionError('saved reader loaded native provider')):
            changed = copy.deepcopy(value)
            for event in changed['descriptor_events']:
                raw = event['identity']['raw_fdinfo'].replace('flags:\t02000002\n', 'flags:\t02\n')
                event['identity']['raw_fdinfo'] = raw
                event['identity']['fdinfo_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
            accepted = False
            error = None
            try:evidence.validate_calls(changed, owner, allowed, contract);accepted = True
            except (ValueError, KeyError, AssertionError) as failure:error = str(failure)
        finding = {'synthetic_saved_metadata_only': True, 'actual_native_calls': 0,
            'mutation': 'retained raw fdinfo flags no longer contain O_CLOEXEC; hashes recomputed',
            'accepted': accepted, 'error': error, 'original': value, 'changed': changed,
            'source_sha256': sha(S / 'native_process_evidence.py')}
        findings.append({'name': 'raw_fdinfo_close_on_exec_missing', 'accepted_invalid_input': accepted,
                         'receipt': save('raw_fdinfo_close_on_exec_boundary', finding)})
        self.assertFalse(accepted, 'saved native reader accepted raw handle flags contradicting noninheritance')


started = time.time()
suite = unittest.TestSuite([
    unittest.defaultTestLoader.loadTestsFromTestCase(calls.NativeDiagnosticPolicyTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(calls.NativeSavedCallTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(roles.NativeSavedRoleTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(actual.ActualNativeCpuTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(order.NativeDiagnosticOrderTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(IndependentSavedFdinfoBoundaries)])
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert all(sha(path) == expected for path, expected in before.items())
assert not any(name.split('.')[0] in {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'cupy', 'jaxtari', 'exploration_platform'} for name in sys.modules)
cpu_files = {str(path): sha(path) for path in ARTIFACTS.rglob('*.json')}
payload = {'kind': 'independent_explicit_native_direct_diagnostic_source_review',
    'passed': result.wasSuccessful(), 'source_approved': result.wasSuccessful(),
    'job_id': os.environ['SLURM_JOB_ID'], 'step_id': STEP, 'pid': os.getpid(),
    'sys_executable': sys.executable, 'sys_version': sys.version, 'requested_memory_mib': 512,
    'started_at_seconds': started, 'finished_at_seconds': time.time(), 'tests_run': result.testsRun,
    'failures': [{'test': str(test), 'traceback': text} for test, text in result.failures],
    'errors': [{'test': str(test), 'traceback': text} for test, text in result.errors],
    'findings': findings, 'source_and_author_records_sha256': hashes, 'actual_cpu_raw_sha256': cpu_files,
    'source_freeze': {'path': str(freeze_path), 'sha256': sha(freeze_path)},
    'review_program': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).resolve())},
    'protected_files_unchanged': 34, 'all_frozen_source_bytes_unchanged': True,
    'actual_cpu_startup_and_orphan_fixtures_only': True, 'actual_Nsight_GPU_standing_lock_work': False,
    'actual_target_qualification': False, 'scientific_credit': 0, 'formal_profile_admission': False}
raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
with OUT.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
print(json.dumps({'passed': payload['passed'], 'tests_run': result.testsRun, 'failures': len(result.failures),
                  'errors': len(result.errors), 'receipt': str(OUT), 'sha256': hashlib.sha256(raw).hexdigest()}))
raise SystemExit(0 if result.wasSuccessful() else 1)
