"""Review frozen direct process source using owned CPU fixtures and one private temporary lock."""
import ast
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import stat
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

W = Path(__file__).resolve().parents[1]
S = W / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
R = W / '.review_metadata'
F = R / 'direct_nsight_producer_source_freeze_author_2.json'
frozen = json.loads(F.read_text())
assert hashlib.sha256(F.read_bytes()).hexdigest() == 'f508925a1bde5623a347952ad93b7c3158ddfc7841bf95ace56e91247b42166a'
assert os.environ['SLURM_JOB_ID'] == '6783003' and os.environ['SLURM_STEP_ID'].isdecimal()
assert all(hashlib.sha256((W / p).read_bytes()).hexdigest() == h for p, h in frozen['files'].items())
sys.path.insert(0, str(S / 'tests')); sys.path.insert(0, str(S))
import test_direct_profile_process as fixtures
import direct_profile_process as direct
from test_nsight_process_identity import M

output = io.StringIO()
suite = unittest.TestSuite()
for name in ('test_direct_profile_process.ActualTemporaryDescriptorTests',
             'test_direct_profile_process.DirectSavedProcessTests',
             'test_direct_profile_process.DirectDiagnosticSourceTests'):
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
started = time.time()
result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)

# Before: only a new private ordinary file exists. After: the exact source producer
# fails before full ancestry observation while a positively identified own payload survives.
reproduction = {'kind': 'actual_cpu_temporary_lock_startup_failure', 'nsight_executed': False,
    'standing_gpu_lock_opened': False, 'gpu_operations': 0, 'scientific_credit': 0}
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    marker = root / 'actual_owned_payload.json'
    lock = root / 'private_temporary_cpu_lock'
    lock.write_text('independent CPU fixture only\n')
    info = lock.stat()
    lock_identity = {'device': info.st_dev, 'inode': info.st_ino, 'uid': os.getuid(),
        'mode': stat.S_IMODE(info.st_mode), 'regular_file': True}
    unit = root / 'unit'; unit.mkdir()
    parent = M.ProcReader().snapshot(os.getpid(), os.getuid())
    payload_code = 'import time; time.sleep(6)'
    payload_command = ['/usr/bin/python3', '-I', '-S', '-B', '-c', payload_code]
    launcher_code = ('import json,os,pathlib,subprocess,sys,time;'
        "child=subprocess.Popen(['/usr/bin/python3','-I','-S','-B','-c',sys.argv[2]],close_fds=True,pass_fds=());"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'root_pid':os.getpid(),'payload_pid':child.pid}));"
        'time.sleep(30)')
    command = ['/usr/bin/python3', '-I', '-S', '-B', '-c', launcher_code, str(marker), payload_code]
    intent = {'producer': parent, 'nsight_command': command}
    plan = {'kind': direct.DIAGNOSTIC, 'maximum_active_seconds': 2.0}
    known = {}; failures = []
    def observe_exact_new_child():
        """Pin only the child of the exact Python root created by this source invocation."""
        deadline = time.monotonic() + 4
        try:
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert marker.exists(), 'the fixture root did not publish its own Popen payload'
            value = json.loads(marker.read_text())
            root_record = M.ProcReader().snapshot(value['root_pid'], os.getuid(), parent['cgroup'])
            assert root_record['parent_pid'] == os.getpid() and root_record['command'] == command
            children, lists = M.ProcReader().children(value['root_pid'])
            assert value['payload_pid'] in children
            handle = M.open_process_handle(value['payload_pid'])
            child = M.ProcReader().snapshot(value['payload_pid'], os.getuid(), parent['cgroup'])
            assert child['parent_pid'] == value['root_pid'] and child['command'] == payload_command
            assert not M.handle_exited(handle)
            known.update(root=root_record, child=child, handle=handle, positive_child_lists=lists)
        except BaseException as error:
            failures.append(repr(error))
    watcher = threading.Thread(target=observe_exact_new_child)
    descriptor = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    watcher.start()
    caught = None
    try:
        with patch.object(direct, 'generic', return_value=M), patch.object(direct, 'make_intent', return_value=intent), \
                patch.object(direct, 'LOCK', str(lock)), patch.object(direct, 'LOCK_IDENTITY', lock_identity), \
                patch.object(direct, 'OWNER', os.getuid()):
            try:
                direct.run_tool(unit, plan, descriptor)
            except BaseException as error:
                caught = {'type': type(error).__name__, 'message': str(error)}
        watcher.join(timeout=5)
        assert not watcher.is_alive() and not failures and known, (failures, known)
        assert caught is not None
        source_failure = M.read_json(unit / 'direct_process_failure.json')
        alive_before_close = not M.handle_exited(known['handle'])
        # This duplicates the caller's finally-close, using only the private CPU fixture.
        os.close(descriptor); descriptor = None
        with lock.open('r+') as contender:
            fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            reacquired_while_child_alive = not M.handle_exited(known['handle'])
        reproduction.update(error=caught, exact_root=known['root'], exact_payload=known['child'],
            positive_owned_child_lists=known['positive_child_lists'], source_failure=source_failure,
            payload_alive_after_run_tool_cleanup=alive_before_close,
            temporary_lock_reacquired_while_exact_payload_alive=reacquired_while_child_alive,
            cleanup_stopped_only_observed_root=source_failure.get('cleanup', {}).get('unobserved_descendant_exit_not_claimed') is True)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        watcher.join(timeout=5)
        if known:
            # The only child handle was pinned before its owned UID/source/parent reads.
            # The payload has a six-second natural lifetime; do not signal any unrelated PID.
            exited = M.handle_exited(known['handle'], 10000)
            reproduction['exact_payload_natural_exit_verified'] = exited
            if not exited:
                M.signal_handle(known['handle'], signal.SIGTERM)
                reproduction['own_fixture_cleanup_signal_required'] = True
                assert M.handle_exited(known['handle'], 10000)
            M.close_process_handle(known['handle'])

f = fixtures.DirectSavedProcessTests(); f.setUp()
try:
    unit, plan = f.saved()
    missing = not (unit / 'capture_release.json').exists()
    accepted = direct.validate_saved(unit, plan)
    release_control = {'capture_release_file_missing': missing,
        'saved_reader_accepted_missing_release': accepted.get('passed') is True,
        'fixture_scope': 'synthetic saved metadata only; no actual process or qualification credit'}
finally:
    f.doCleanups()

unchanged = all(hashlib.sha256((W / p).read_bytes()).hexdigest() == h for p, h in frozen['files'].items())
label = 'direct_nsight_independent_' + os.environ['SLURM_JOB_ID'] + '_' + os.environ['SLURM_STEP_ID']
raw = output.getvalue().encode(); (R / (label + '_unittest.raw')).write_bytes(raw)
(R / (label + '_cpu_reproduction.json')).write_text(json.dumps(reproduction, indent=2) + '\n')
receipt = {'kind': 'independent_direct_nsight_source_failure_review', 'source_freeze': {'path': str(F),
    'sha256': hashlib.sha256(F.read_bytes()).hexdigest()}, 'base': frozen['base'],
    'job_id': os.environ['SLURM_JOB_ID'], 'step_id': os.environ['SLURM_STEP_ID'], 'hostname': socket.gethostname(),
    'pid': os.getpid(), 'started_at_seconds': started, 'finished_at_seconds': time.time(),
    'focused_tests': {'count': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'output': {'path': str(R / (label + '_unittest.raw')), 'sha256': hashlib.sha256(raw).hexdigest()}},
    'source_unchanged_before_and_after': unchanged, 'cpu_reproduction': {'path': str(R / (label + '_cpu_reproduction.json')),
        'sha256': hashlib.sha256((R / (label + '_cpu_reproduction.json')).read_bytes()).hexdigest()},
    'release_record_control': release_control, 'approved': False, 'gpu_operations': 0,
    'actual_nsight_or_standing_lock_actions': 0, 'job_submissions_releases_or_remote_actions': 0,
    'scientific_credit': 0, 'admission_granted': False}
(R / (label + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'receipt': str(R / (label + '.json')), 'tests': result.testsRun,
    'failures': len(result.failures), 'errors': len(result.errors), 'source_unchanged': unchanged,
    'payload_survived_cleanup': reproduction.get('payload_alive_after_run_tool_cleanup'),
    'lock_reacquired_with_payload_alive': reproduction.get('temporary_lock_reacquired_while_exact_payload_alive'),
    'missing_release_accepted': release_control['saved_reader_accepted_missing_release'], 'approved': False}))
assert result.wasSuccessful() and unchanged
assert reproduction['temporary_lock_reacquired_while_exact_payload_alive'] is True
assert reproduction['exact_payload_natural_exit_verified'] is True
