"""Independently review the opt-in provider on own CPU processes only."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import traceback
import unittest
from unittest.mock import patch

W = Path(__file__).resolve().parents[1]
S = W / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
M = W / '.review_metadata'
STEP = os.environ.get('SLURM_STEP_ID', '')
OUT = M / ('linux_pidfd_native_corrected_independent_' + STEP + '.json')
ARTIFACTS = M / ('linux_pidfd_native_corrected_independent_cpu_' + STEP)
checks, snapshots, findings = [], {}, []


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def check(value, message):
    if not value:
        raise AssertionError(message)
    checks.append(message)


def sha(path):
    path = Path(path)
    value = digest(path.read_bytes())
    snapshots[str(path)] = value
    return value


def save(name, value):
    path = ARTIFACTS / (name + '.json')
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return {'path': str(path), 'sha256': digest(raw)}


def node_source(text, name):
    node = next(item for item in ast.parse(text).body if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name == name)
    return ast.get_source_segment(text, node)


assert os.environ.get('SLURM_JOB_ID') == '6783003' and STEP.isdecimal()
assert os.environ.get('SLURM_CPUS_PER_TASK') == '1' and not os.environ.get('SLURM_STEP_GPUS')
assert not os.environ.get('CUDA_VISIBLE_DEVICES')
assert sys.executable == '/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python'
assert not OUT.exists() and not ARTIFACTS.exists()
ARTIFACTS.mkdir()
started = time.time()
frozen_path = M / 'linux_pidfd_native_corrected_source_freeze_author_2.json'
check(sha(frozen_path) == '16abb944415412b692a26a43601b6c3577d4e83a14f5f45e09611335f0a0aff4', 'exact independent source freeze')
handoff_path = M / 'linux_pidfd_native_corrected_author_handoff.json'
check(sha(handoff_path) == 'c7a87be923ec64b6567eee165088e69c418e8581a2e70a8e50ec1f7f75283295', 'exact author handoff')
freeze = json.loads(frozen_path.read_bytes())
handoff = json.loads(handoff_path.read_bytes())
check(len(freeze['files']) == 3 and len(freeze['protected_files']) == 26, 'three changed paths and 26 protected paths')
for name, expected in freeze['files'].items():check(sha(W / name) == expected, 'exact frozen source: ' + name)
for name, expected in freeze['protected_files'].items():check(sha(name) == expected, 'protected source unchanged: ' + name)
core = (S / 'profile_selected.py').read_text()
check(digest(node_source(core, '_capture_core').encode()) == freeze['capture_core_source_sha256'], 'capture core source bytes unchanged')
identity_text = (S / 'nsight_process_identity.py').read_text()
for name, expected in freeze['default_process_control_source_hashes'].items():
    check(digest(node_source(identity_text, name).encode()) == expected, 'unchanged default/identity source: ' + name)
for key in ('author_execution', 'independent_rejection', 'rejected_source_preservation'):
    ref = handoff[key]
    check(sha(ref['path']) == ref['sha256'], 'actual author metadata: ' + key)
for key, ref in handoff['actual_cpu_proofs'].items():
    check(sha(ref['path']) == ref['sha256'], 'actual author CPU proof: ' + key)
preservation = json.loads(Path(handoff['rejected_source_preservation']['path']).read_bytes())
for name, record in preservation['files'].items():
    check(sha(record['path']) == record['sha256'], 'rejected source/receipt preserved: ' + name)
for name in ('call_budget_boundary.json', 'selected_executable_image_boundary.json'):
    prior = M / 'linux_pidfd_native_independent_cpu_354' / name
    check(prior.is_file(), 'actual 354 failed boundary record retained: ' + name)
    sha(prior)
check(sha(M / 'run_linux_pidfd_native_independent_review.py') == '35b08ea9c4f8bf1b30dd2512a5d0c4c5abf42497f1175595d2259e23909caeef', 'original failed 354 program retained')
before_files = dict(snapshots)
sys.path.insert(0, str(S)); sys.path.insert(0, str(S / 'tests'))
import linux_pidfd_process_control as native
import nsight_process_identity as metadata
import test_linux_pidfd_process_control as author_tests
exports_before = (getattr(os, 'pidfd_open', None), getattr(signal, 'pidfd_send_signal', None))
check(exports_before == (None, None), 'selected Python exports are actually absent')
os.environ['NATIVE_PIDFD_REVIEW_OUTPUT'] = str(ARTIFACTS / 'independent_focused_author_cases')


class IndependentNativeBoundaries(unittest.TestCase):
    """Use synthetic boundary metadata and actual own-self descriptors only."""
    def test_last_open_evidence_slot_keeps_a_close_path(self):
        """A bounded evidence list must not make an actual retained handle impossible to close."""
        contract, unused = native.current_contract()
        provider = native.NativePidfdProvider(contract)
        descriptor = None
        error = None
        try:
            with patch.object(native, 'MAX_CALLS', provider.normal_call_count + 1):
                descriptor = provider.open_process_handle(os.getpid())
                try:
                    provider.close_process_handle(descriptor)
                except BaseException as failure:
                    error = repr(failure)
                finding = {'synthetic_evidence_limit': True, 'actual_target': 'own reviewer process only',
                    'target_pid': os.getpid(), 'descriptor': descriptor,
                    'descriptor_target': provider.descriptor_identity(descriptor, os.getpid()) if descriptor in provider.handles else None,
                    'close_error': error, 'still_retained_before_fixture_cleanup': descriptor in provider.handles,
                    'native_evidence': copy.deepcopy(provider.evidence()),
                    'source_sha256': sha(S / 'linux_pidfd_process_control.py')}
            if descriptor in provider.handles:
                # Test-only cleanup of this exact own descriptor; do not leave a known leak in the reviewer.
                os.close(descriptor); provider.handles.pop(descriptor)
                finding['explicit_fixture_only_close_completed'] = True
            findings.append({'issue': 'evidence_budget_refuses_required_close', 'reproduced': error is not None,
                             'receipt': save('call_budget_boundary', finding)})
            self.assertIsNone(error, 'last allowed open left an actual own pidfd that provider refuses to close')
        finally:
            if descriptor is not None and descriptor in provider.handles:
                os.close(descriptor); provider.handles.pop(descriptor)

    def test_selected_python_contract_is_the_actual_running_executable(self):
        """Hashing an arbitrary pathname cannot establish the selected running executable image."""
        actual_link = os.readlink('/proc/self/exe')
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / 'plain_metadata_not_an_executable'
            fixture.write_bytes(b'synthetic metadata file; never executed\n')
            rejected = False
            error = None
            actual = None
            with patch.object(native.sys, 'executable', str(fixture)):
                try:
                    actual, unused = native.current_contract()
                except BaseException as failure:
                    rejected = True; error = repr(failure)
            value = {'synthetic_sys_executable_path_fixture': True, 'no_fixture_executed': True,
                'no_pidfd_syscall_in_this_test': True, 'actual_proc_self_exe': actual_link,
                'fixture_path': str(fixture), 'fixture_sha256': digest(fixture.read_bytes()),
                'fixture_mode': oct(fixture.stat().st_mode & 0o777), 'rejected': rejected,
                'error': error, 'returned_selected_python': None if actual is None else actual['selected_python'],
                'source_sha256': sha(S / 'linux_pidfd_process_control.py')}
            findings.append({'issue': 'selected_executable_path_not_joined_to_actual_image', 'reproduced': not rejected,
                             'receipt': save('selected_executable_image_boundary', value)})
            self.assertTrue(rejected, 'current_contract accepted a non-executable metadata file as its selected Python image')


    def test_actual_executable_descriptor_joins_the_contract(self):
        """Independently hash the actual kernel-held executable descriptor and compare its inode."""
        contract, bootstrap = native.current_contract()
        descriptor = os.open('/proc/self/exe', os.O_RDONLY)
        try:
            first = os.fstat(descriptor)
            hasher = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:break
                hasher.update(block)
            last = os.fstat(descriptor)
            image = contract['selected_python']
            self.assertEqual((first.st_dev, first.st_ino), (last.st_dev, last.st_ino))
            self.assertEqual((image['device'], image['inode']), (first.st_dev, first.st_ino))
            self.assertEqual(hasher.hexdigest(), image['sha256'])
            self.assertEqual(os.readlink('/proc/self/exe'), image['actual_running_image_path'])
            self.assertTrue(image['mode'] & 0o111)
            save('actual_kernel_held_executable_descriptor_join', {'selected_python_contract': image,
                'descriptor_device': first.st_dev, 'descriptor_inode': first.st_ino,
                'descriptor_sha256': hasher.hexdigest(), 'kernel_image_link': os.readlink('/proc/self/exe'),
                'actual_sys_executable': sys.executable, 'no_native_pidfd_calls_in_this_check': True})
        finally:
            os.close(descriptor)

    def test_actual_own_pidfd_reuse_cannot_redirect_a_signal(self):
        """Reject a child descriptor number replaced by another actual own-self pidfd."""
        contract, unused = native.current_contract()
        provider = native.NativePidfdProvider(contract)
        parent = metadata.ProcReader().snapshot(os.getpid(), os.getuid())
        child = author_tests.subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c',
            "import sys;print('ready',flush=True);sys.stdin.readline()"],
            stdin=author_tests.subprocess.PIPE, stdout=author_tests.subprocess.PIPE,
            stderr=author_tests.subprocess.PIPE, close_fds=True, pass_fds=())
        target = own = None
        rejected = []
        try:
            self.assertEqual(child.stdout.readline(), b'ready\n')
            actual = metadata.ProcReader().snapshot(child.pid, os.getuid(), parent['cgroup'])
            self.assertEqual(actual['parent_pid'], os.getpid())
            target = provider.open_process_handle(child.pid)
            own = provider.open_process_handle(os.getpid())
            os.dup2(own, target, inheritable=False)
            before = len(provider.calls)
            for label, action in [('signal_zero', lambda: provider.signal_handle(target, 0)),
                                  ('close', lambda: provider.close_process_handle(target))]:
                with self.assertRaises(ValueError) as refused:action()
                rejected.append({'operation': label, 'error': str(refused.exception)})
            self.assertEqual(len(provider.calls), before)
            save('actual_own_pidfd_substitution_refused', {'parent': parent, 'fresh_own_child': actual,
                'substitution': 'actual child pidfd number replaced with actual own-self pidfd, close-on-exec retained',
                'rejections_before_any_new_native_call': rejected, 'native_call_count_unchanged': True,
                'provider_evidence': provider.evidence(), 'no_signal_sent_to_child_or_self_after_substitution': True})
        finally:
            if target is not None:
                os.close(target); provider.handles.pop(target)
            if own is not None:provider.close_process_handle(own)
            if child.poll() is None:
                child.stdin.write(b'\n'); child.stdin.flush()
            child.wait(timeout=5)
            child.stdin.close(); child.stdout.close(); child.stderr.close()

suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(author_tests.NativePidfdContractTests),
                           unittest.defaultTestLoader.loadTestsFromTestCase(IndependentNativeBoundaries)])
result = unittest.TextTestRunner(verbosity=2).run(suite)
check((getattr(os, 'pidfd_open', None), getattr(signal, 'pidfd_send_signal', None)) == exports_before,
      'no global Python pidfd exports changed')
check(all(sha(path) == expected for path, expected in before_files.items()), 'frozen source/protected/author records remain unchanged')
check(not any(name.split('.')[0] in {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'jaxtari', 'cupy', 'exploration_platform'} for name in sys.modules), 'no numerical runtime imported')
source = (S / 'linux_pidfd_process_control.py').read_text()
check('os.kill(' not in source and 'os.pidfd_open =' not in source and 'signal.pidfd_send_signal =' not in source,
      'provider source has no numeric-PID signal fallback or global API replacement')
payload = {'kind': 'independent_corrected_unwired_native_pidfd_provider_source_and_cpu_review',
    'passed': result.wasSuccessful(), 'source_approved': result.wasSuccessful(),
    'job_id': os.environ['SLURM_JOB_ID'], 'step_id': STEP, 'pid': os.getpid(),
    'selected_python': sys.executable, 'sys_version': sys.version, 'requested_memory_mib': 512,
    'started_at_seconds': started, 'finished_at_seconds': time.time(), 'tests_run': result.testsRun,
    'failures': [{'test': str(test), 'traceback': text} for test, text in result.failures],
    'errors': [{'test': str(test), 'traceback': text} for test, text in result.errors],
    'findings': findings, 'checks': checks, 'files_sha256': snapshots,
    'source_freeze': {'path': str(frozen_path), 'sha256': sha(frozen_path)},
    'review_program': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).resolve())},
    'actual_own_self_and_fresh_own_cpu_child_only': True, 'numerical_imports': [],
    'tool_gpu_or_standing_lock_actions': 0, 'future_plan_helper_union_saved_reader_pending': True,
    'actual_target_qualification_pending': True, 'scientific_credit': 0, 'admission_granted': False}
raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
with OUT.open('xb') as stream:stream.write(raw); stream.flush(); os.fsync(stream.fileno())
print(json.dumps({'passed': payload['passed'], 'tests_run': result.testsRun, 'failures': len(result.failures),
                  'errors': len(result.errors), 'receipt': str(OUT), 'sha256': digest(raw)}))
raise SystemExit(0 if result.wasSuccessful() else 1)
