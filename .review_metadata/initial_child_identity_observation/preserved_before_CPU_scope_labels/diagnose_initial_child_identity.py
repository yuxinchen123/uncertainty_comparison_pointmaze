"""Observe one selected-Python CPU child; never run a learner or touch a GPU."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time
import uuid

SELECTED_PYTHON = Path('/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python')
CONTROLLER = Path(__file__).with_name('direct_common_timing_controller.py')
NUMERICAL = {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'scipy', 'cupy', 'nvidia',
    'tensorflow', 'jaxtari', 'exploration_platform', 'jax_cuda12_plugin', 'jax_cuda13_plugin'}


def digest_file(path):
    """Hash only this source, its metadata controller or the selected Python image."""
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    """Keep each actual observation without replacing a previous attempt."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def load_controller(expected):
    """Load only the exact reviewed standard-library process reader."""
    if digest_file(CONTROLLER) != expected:
        raise ValueError('the diagnostic metadata controller changed')
    spec = importlib.util.spec_from_file_location('cpu_child_identity_metadata_controller', CONTROLLER)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.no_numerical_modules()
    return module


def audit(event, values):
    """Refuse numerical imports, device access, network access and new native loads."""
    if event == 'import' and values[0].split('.')[0] in NUMERICAL:
        raise RuntimeError('numerical imports are forbidden in this CPU diagnostic')
    if event in {'ctypes.dlopen', 'ctypes.dlsym', 'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'os.system'}:
        raise RuntimeError('native library, network and shell calls are forbidden')
    if event == 'open' and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(
            ('/dev/nvidia', '/dev/dri', '/tmp/sl5nw_serval02_card1.lock')):
        raise RuntimeError('GPU devices and the standing lock are forbidden')


def child_acknowledgment(args):
    """Keep this exact child alive only for its parent's bounded metadata reads."""
    core = load_controller(args.expected_controller_sha256)
    sys.addaudithook(audit)
    value = {'nonce': args.ack_child, 'identity': core.read_process(os.getpid(), os.getuid()),
        'source_sha256': digest_file(__file__), 'actual_image_sha256': digest_file('/proc/self/exe')}
    print(json.dumps(value, sort_keys=True), flush=True)
    remaining = max(0., args.work_end - time.monotonic())
    readable, unused, errors = select.select([sys.stdin.buffer], [], [], remaining)
    return 0 if readable and sys.stdin.buffer.read(1) == b'R' else 2


def run_diagnostic(directory, source_sha, controller_sha, python_sha, *, cpu_count=8):
    """Record immediate and acknowledged identities separately under one 300-second ceiling."""
    started = time.monotonic();work_end, total_end = started + 240, started + 300
    core = load_controller(controller_sha);directory = Path(directory)
    if not directory.is_dir() or directory.resolve(strict=True) != directory:
        raise ValueError('the root-prepared diagnostic output directory is required')
    if digest_file(__file__) != source_sha or digest_file(SELECTED_PYTHON) != python_sha:
        raise ValueError('the diagnostic source or actual selected executable changed')
    if os.getuid() != os.geteuid() or os.getuid() != 2618919 or os.getresuid() != (2618919,) * 3:
        raise ValueError('the exact current owner is required')
    parent = core.read_process(os.getpid(), 2618919)
    original = parent['affinity'];chosen = core.chosen_affinity(original, cpu_count)
    nonce = uuid.uuid4().hex
    command = [str(SELECTED_PYTHON), '-I', '-S', '-B', str(Path(__file__).resolve(strict=True)),
        '--ack-child', nonce, '--expected-controller-sha256', controller_sha, '--work-end', str(work_end)]
    origin = {'kind': 'cpu_child_identity_diagnostic_origin', 'parent': parent, 'command': command,
        'selected_executable': str(SELECTED_PYTHON), 'resolved_selected_executable': str(SELECTED_PYTHON.resolve(strict=True)),
        'source_sha256': source_sha, 'controller_sha256': controller_sha, 'selected_executable_sha256': python_sha,
        'requested_cpu_count': cpu_count, 'selected_affinity': chosen, 'started_monotonic': started,
        'work_end_monotonic': work_end, 'total_end_monotonic': total_end,
        'numerical_modules_or_devices_requested': False, 'environment_transitions': 0,
        'interpreter_startup_native_work_quantified': False, 'isolated_python_startup_flags': ['-I', '-S', '-B'],
        'matches_original_scientific_startup_flags': False, 'full_original_launch_reproduced': False}
    write(directory / 'diagnostic_origin.json', origin)
    child = None;first = acknowledged = ack = None;returned = None;error = None;released = False
    ack_raw = tail = errors = b''
    previous = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
        raise ValueError('the CPU diagnostic cannot replace an existing timer')
    def expired(number, frame):
        raise TimeoutError('the single CPU diagnostic work allowance expired')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, work_end - time.monotonic()))
    try:
        os.sched_setaffinity(0, chosen)
        if sorted(os.sched_getaffinity(0)) != chosen:
            raise ValueError('the selected parent CPU affinity did not take effect')
        environment = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8', 'TZ': 'UTC',
            'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=environment, close_fds=True, bufsize=0)
        initial = core.read_process(child.pid, 2618919)
        first = core.child_identity_observation(initial, command, chosen, os.getpid(), child.pid)
        write(directory / 'immediate_child_observation.json', first)
        ready, unused, exceptional = select.select([child.stdout], [], [], max(0., work_end-time.monotonic()))
        if not ready:
            raise TimeoutError('the exact child did not acknowledge within the original work allowance')
        ack_raw = child.stdout.readline(65537)
        if not ack_raw.endswith(b'\n') or len(ack_raw) > 65536:
            raise ValueError('the exact child acknowledgment was incomplete or oversized')
        ack = json.loads(ack_raw)
        current = core.read_process(child.pid, 2618919)
        acknowledged = core.child_identity_observation(current, command, chosen, os.getpid(), child.pid,
            initial_birth=initial['process_start_time'])
        acknowledged['actual_executable_path'] = os.readlink('/proc/' + str(child.pid) + '/exe')
        acknowledged['actual_executable_sha256'] = digest_file('/proc/' + str(child.pid) + '/exe')
        write(directory / 'acknowledged_child_observation.json', acknowledged)
        if ack['nonce'] != nonce or ack['source_sha256'] != source_sha or ack['actual_image_sha256'] != python_sha:
            raise ValueError('the child acknowledgment changed its challenge, source or actual image')
        if any(ack['identity'][key] != current[key] for key in ('pid','parent_pid','process_start_time','uid','command','affinity')):
            raise ValueError('the acknowledged child identity differs from the same observed child')
        if not first['all_predicates_match'] or not acknowledged['all_predicates_match'] or acknowledged['actual_executable_sha256'] != python_sha:
            raise ValueError('the diagnostic preserves an immediate or acknowledged identity mismatch')
        child.stdin.write(b'R');child.stdin.flush();released = True
    except BaseException as problem:
        error = {'type': type(problem).__name__, 'error': str(problem)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.);signal.signal(signal.SIGALRM, previous)
        if child is not None:
            child.stdin.close();child.stdin = None
            try:
                tail, errors = child.communicate(timeout=max(.001, total_end-time.monotonic()))
                returned = child.wait(timeout=0)
            except BaseException as problem:
                error = error or {'type': type(problem).__name__, 'error': str(problem)}
        os.sched_setaffinity(0, original)
    finished = time.monotonic()
    affinity_restored = sorted(os.sched_getaffinity(0)) == original
    result = {'kind': 'cpu_selected_interpreter_child_identity_diagnostic', 'passed': error is None
        and type(returned) is int and returned == 0 and released and finished <= total_end and affinity_restored,
        'first_observation': first, 'acknowledged_observation': acknowledged, 'actual_child_acknowledgment': ack,
        'acknowledgment_raw_hex': ack_raw.hex(), 'acknowledgment_sha256': hashlib.sha256(ack_raw).hexdigest(),
        'stdout_tail_hex': tail.hex(), 'stderr_hex': errors.hex(), 'actual_popen_pid': None if child is None else child.pid,
        'actual_wait_returncode': returned, 'actual_wait_verified': type(returned) is int,
        'release_sent_after_both_positive_observations': released, 'failure': error,
        'elapsed_seconds': finished-started, 'within_total_300_seconds': finished <= total_end,
        'historical_failure_cause_established': False, 'ordinary_success_if_only_later_observation_matches': False,
        'CPU_affinity_restored': affinity_restored,
        'source_sha256': source_sha, 'controller_sha256': controller_sha, 'selected_executable_sha256': python_sha,
        'requested_cpu_count': cpu_count, 'scientific_credit': 0, 'worker_admission': False,
        'environment_transitions': 0, 'interpreter_startup_native_work_quantified': False, 'isolated_python_startup_flags': ['-I', '-S', '-B'],
        'matches_original_scientific_startup_flags': False, 'full_original_launch_reproduced': False}
    write(directory / 'diagnostic_result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true');parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--expected-source-sha256');parser.add_argument('--expected-controller-sha256', required=True)
    parser.add_argument('--expected-python-sha256');parser.add_argument('--ack-child');parser.add_argument('--work-end', type=float)
    args = parser.parse_args()
    if args.ack_child:
        return child_acknowledgment(args)
    if not args.execute or args.run_dir is None or args.expected_source_sha256 is None or args.expected_python_sha256 is None:
        parser.error('explicit execution requires the root-prepared directory and all actual source/image hashes')
    result = run_diagnostic(args.run_dir, args.expected_source_sha256, args.expected_controller_sha256,
        args.expected_python_sha256, cpu_count=8)
    print(json.dumps(result, sort_keys=True));return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
