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
OUT = M / ('independent_fdinfo_' + STEP + '.json')
ARTIFACTS = M / ('independent_fdinfo_actual_cpu_' + STEP)
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


freeze_path = M / 'fdinfo_correction_source_freeze.json'
assert sha(freeze_path) == '08d01eea45353d77fbd4982e242763e38a42aa67cf9bb9fa86237fe085f7bbcf'
assert sha(M / 'fdinfo_correction_author_handoff.json') == 'd84c4433ac3c163f1904247a776e817e0400eada75a3d56d1cebcce8488e593d'
freeze = json.loads(freeze_path.read_bytes())
assert len(freeze['protected_files']) == 34
for path, expected in freeze['files'].items():assert sha(W / path) == expected
for path, expected in freeze['protected_files'].items():assert sha(path) == expected
handoff = json.loads((M / 'fdinfo_correction_author_handoff.json').read_bytes())
for key in ('focused_author_execution', 'exact_rejected_mutations_author_execution', 'preserved_rejected_source', 'preserved_rejected_review', 'previous_full_author_execution'):
    ref = handoff[key]
    assert sha(ref['path']) == ref['sha256']
preserved = json.loads(Path(handoff['preserved_rejected_source']['path']).read_bytes())
assert preserved['source_files'] == 14
for record in preserved['files'].values():assert sha(record['path']) == record['sha256']
assert freeze['files']['10_jax_exploration_platform/scripts/atari_pixel_campaign/direct_profile_process.py'] == '2c251b8c58bd6138bbf2455a7bf27d1e63873862e61b7261529339048f512088'
assert freeze['files']['10_jax_exploration_platform/scripts/atari_pixel_campaign/diagnose_direct_nsight_process.py'] == '84169667299d9bace11ac29ec268b265947ebb999e6d6cf7784080cd158b7c17'
before = dict(hashes)
sys.path.insert(0, str(S));sys.path.insert(0, str(S / 'tests'))
import test_native_process_evidence as calls
import test_native_process_role_evidence as roles
import test_native_direct_process_cpu as actual
import test_native_diagnostic_order as order
import test_native_pidfd_fdinfo as fdinfo
import test_linux_pidfd_process_control as provider_tests
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
    unittest.defaultTestLoader.loadTestsFromTestCase(fdinfo.PureFdinfoTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(fdinfo.ActualOwnFdinfoTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(calls.NativeSavedCallTests),
    roles.NativeSavedRoleTests('test_three_roles_keep_exact_owned_opens_wait_references_and_chronology'),
    roles.NativeSavedRoleTests('test_missing_terminal_never_becomes_a_completed_role'),
    actual.ActualNativeCpuTests('test_actual_startup_cleanup_keeps_temporary_lock_until_child_exit'),
    actual.ActualNativeCpuTests('test_actual_orphan_cleanup_keeps_temporary_lock_until_kernel_terminal'),
    provider_tests.NativePidfdContractTests('test_last_ordinary_call_allows_only_reserved_close'),
    provider_tests.NativePidfdContractTests('test_reserved_term_kill_close_remain_available_after_ordinary_limit'),
    unittest.defaultTestLoader.loadTestsFromTestCase(IndependentSavedFdinfoBoundaries)])
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert all(sha(path) == expected for path, expected in before.items())
assert not any(name.split('.')[0] in {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'cupy', 'jaxtari', 'exploration_platform'} for name in sys.modules)
cpu_files = {str(path): sha(path) for path in ARTIFACTS.rglob('*.json')}
payload = {'kind': 'independent_corrected_raw_fdinfo_native_direct_diagnostic_review',
    'passed': result.wasSuccessful(), 'source_approved': result.wasSuccessful(),
    'job_id': os.environ['SLURM_JOB_ID'], 'step_id': STEP, 'pid': os.getpid(),
    'sys_executable': sys.executable, 'sys_version': sys.version, 'requested_memory_mib': 512,
    'started_at_seconds': started, 'finished_at_seconds': time.time(), 'hard_slurm_time_limit_seconds': 300, 'tests_run': result.testsRun,
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
