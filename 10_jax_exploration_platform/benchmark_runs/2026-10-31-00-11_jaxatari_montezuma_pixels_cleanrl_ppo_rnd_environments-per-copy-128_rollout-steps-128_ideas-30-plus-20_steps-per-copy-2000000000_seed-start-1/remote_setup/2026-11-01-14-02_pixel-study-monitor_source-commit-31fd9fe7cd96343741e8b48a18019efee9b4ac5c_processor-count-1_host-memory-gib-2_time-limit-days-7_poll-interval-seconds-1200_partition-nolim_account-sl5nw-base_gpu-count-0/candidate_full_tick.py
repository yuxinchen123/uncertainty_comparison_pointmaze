"""Run one actual frozen monitor tick on the old owned CPU allocation without job actions or protected edits."""

import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import traceback


PACKET = Path(__file__).resolve().parent
RUN = PACKET.parent.parent
COMMIT = '31fd9fe7cd96343741e8b48a18019efee9b4ac5c'
SOURCE = PACKET / 'source_snapshot' / COMMIT / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
BLOCKED = []
PROTECTED = {RUN / name for name in ('next_actions.json', 'campaign_provenance.json', 'SOURCE_AND_GATES_READY.json')}
IDS = set()


def save(name, value):
    """Retain candidate evidence as new fsynced files without replacing any prepared record."""
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2) + '\n').encode()
    with (PACKET / name).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(PACKET, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)
    return {'path': str(PACKET / name), 'sha256': hashlib.sha256(raw).hexdigest()}


def protected(path):
    """Recognize only the exact canonical protected files and their ordinary temporary-write names."""
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    path = Path(os.fsdecode(path)).absolute()
    return any(path.parent == item.parent and (path.name == item.name or path.name.startswith(item.name + '.'))
               for item in PROTECTED)


def restrict(event, arguments):
    """Allow metadata reads only; refuse numerical imports, remote connections, job actions and protected writes."""
    denied = False
    if event == 'import':
        denied = str(arguments[0]).split('.')[0] in {'jax', 'jaxlib', 'numpy', 'torch', 'jaxtari', 'flax', 'optax', 'cupy'}
    elif event == 'ctypes.dlopen':
        denied = arguments[0] is not None
    elif event.startswith('socket.'):
        denied = event not in {'socket.__new__', 'socket.gethostname'}
    elif event == 'open' and protected(arguments[0]):
        mode, flags = arguments[1], arguments[2]
        denied = (isinstance(mode, str) and any(token in mode for token in ('w', 'a', '+', 'x'))) or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    elif event in {'os.rename', 'os.remove', 'os.rmdir'}:
        denied = any(protected(value) for value in arguments[:2])
    elif event == 'subprocess.Popen':
        argv = arguments[1]
        denied = True
        if isinstance(argv, (list, tuple)) and argv:
            program = Path(argv[0]).name
            if program == 'git':
                denied = not any(token in argv for token in ('show', 'rev-parse', 'ls-tree', 'status', 'diff'))
                if list(argv[-3:]) == ['symbolic-ref', '--quiet', 'HEAD']:
                    denied = False
            elif program in {'squeue', 'sacct'}:
                selectors = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value in {'-j', '--jobs'}]
                denied = len(selectors) != 1 or not set(selectors[0].split(',')) <= IDS
            elif program == 'scontrol' and list(argv[1:3]) == ['show', 'job']:
                selected = [value for value in argv[3:] if not value.startswith('-')]
                denied = len(selected) != 1 or selected[0] not in IDS
            elif program == 'scontrol' and list(argv[1:3]) == ['show', 'step']:
                selected = [value for value in argv[3:] if not value.startswith('-')]
                denied = len(selected) != 1 or selected[0].split('.')[0] not in IDS
            elif program == 'scontrol' and list(argv[1:3]) == ['show', 'node']:
                denied = len([value for value in argv[3:] if not value.startswith('-')]) != 1
    if denied:
        BLOCKED.append({'event': event, 'arguments': repr(arguments)})
        raise RuntimeError('Candidate refused a forbidden action: ' + event)


def module(name, path):
    """Import only a named immutable helper path without executing any of its CLI actions."""
    specification = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loaded)
    return loaded


def main():
    """Verify source/ownership and execute the complete monitor once, retaining actual report and raw output."""
    assert os.environ.get('SLURM_JOB_ID') == '6781231'
    assert os.getuid() == os.geteuid() == 2618919 and os.environ.get('SLURM_STEP_ID', '').isdecimal()
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', GIT_OPTIONAL_LOCKS='0')
    old = RUN / 'remote_setup/2026-11-01-08-35_pixel-study-monitor_source-commit-fb3071f3772ed836144f5876dbdda1113a5339d5_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0'
    ledger = (RUN / 'slurm/submitted_jobids_uva.txt').read_text().splitlines()
    assert len(ledger) == len(set(ledger)) and all(re.fullmatch(r'[1-9][0-9]*', value) for value in ledger)
    IDS.update(ledger)
    assert '6781231' in IDS
    rows = [line.split('\t') for line in (RUN / 'slurm/submissions.tsv').read_text().splitlines() if line.split('\t')[0] == '6781231']
    assert rows == [['6781231', 'monitor', 'fb3071f3772ed836144f5876dbdda1113a5339d5', str(old / 'monitor.sbatch')]]
    verification = subprocess.run(['/usr/bin/python3', '-B', str(PACKET / 'verify_source.py')], capture_output=True)
    save('owned_compute_source_visibility_stdout.raw', verification.stdout)
    save('owned_compute_source_visibility_stderr.raw', verification.stderr)
    assert verification.returncode == 0
    guard = module('candidate_monitor_allocation_guard', PACKET / 'allocation_guard.py')
    sys.addaudithook(restrict)
    job_command = ['scontrol', 'show', 'job', '6781231', '-o']
    job = subprocess.run(job_command, capture_output=True, text=True, timeout=20, check=True)
    host, cgroup = socket.gethostname(), Path('/proc/self/cgroup').read_text()
    assert cgroup.splitlines() == ['0::/system.slice/slurmstepd.scope/job_6781231/step_' + os.environ['SLURM_STEP_ID'] + '/user/task_0']
    node_command = ['scontrol', 'show', 'node', host.split('.')[0], '-o']
    node = subprocess.run(node_command, capture_output=True, text=True, timeout=20, check=True)
    fields = dict(token.split('=', 1) for token in job.stdout.split() if '=' in token)
    topology = dict(token.split('=', 1) for token in node.stdout.split() if '=' in token)
    guard.validate(fields, '6781231', host, cgroup, topology)
    save('candidate_allocation.json', {'job_id': '6781231', 'step_id': os.environ['SLURM_STEP_ID'], 'uid': os.getuid(),
        'hostname': host, 'pid': os.getpid(), 'birth': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        'cgroup': cgroup, 'job': {'command': job_command, 'returncode': job.returncode, 'stdout': job.stdout, 'stderr': job.stderr},
        'node': {'command': node_command, 'returncode': node.returncode, 'stdout': node.stdout, 'stderr': node.stderr},
        'new_monitor_job_id': None, 'old_monitor_guard_passed': True})
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in PROTECTED}
    before_counts = json.loads((RUN / 'campaign_state.json').read_bytes()).get('queue_counts', {})
    sys.path.insert(0, str(SOURCE))
    monitor = module('frozen_candidate_monitor', SOURCE / 'monitor.py')
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        complete = monitor.tick(RUN)
    raw_stdout, raw_stderr = stdout.getvalue().encode(), stderr.getvalue().encode()
    save('candidate_full_tick_stdout.raw', raw_stdout)
    save('candidate_full_tick_stderr.raw', raw_stderr)
    report = (RUN / '20_mins_monitoring/latest_report.md').read_bytes()
    status_raw = (RUN / '20_mins_monitoring/latest_status.json').read_bytes()
    status = json.loads(status_raw)
    save('candidate_full_tick_report.md', report)
    save('candidate_full_tick_status.json', status_raw)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in PROTECTED}
    assert before == after and not BLOCKED and complete is False
    assert all(heading.encode() in raw_stdout for heading in ('TRAINING\n', 'WAITING\n', 'HELD\n', 'FINISHED\n', 'PLANNED\n'))
    assert raw_stdout in report and status['campaign']['queue_counts'].get('completed', 0) >= before_counts.get('completed', 0)
    assert not (RUN / '20_mins_monitoring/controller_error.json').exists()
    assert not any(name.split('.')[0] in {'jax', 'jaxlib', 'numpy', 'torch', 'jaxtari', 'flax', 'optax'} for name in sys.modules)
    result = {'schema_version': 1, 'kind': 'frozen_published_monitor_candidate_full_tick_verification', 'passed': True,
        'at_utc': datetime.now(timezone.utc).isoformat(), 'source_commit': COMMIT,
        'source_visibility': json.loads(verification.stdout), 'actual_job_id': '6781231', 'actual_step_id': os.environ['SLURM_STEP_ID'],
        'new_monitor_job_id': None, 'candidate_full_tick_ran': True, 'complete_study': complete,
        'queue_counts_before': before_counts, 'queue_counts_after': status['campaign']['queue_counts'],
        'protected_before': before, 'protected_after': after, 'blocked_action_attempts': BLOCKED,
        'numerical_imports': False, 'network_connections': False, 'new_jobs_submitted': 0,
        'jobs_released_or_cancelled': 0, 'old_monitor_preserved': '6781231', 'all_five_status_sections_present': True,
        'scientific_case_credit': 0, 'scientific_hypothesis_credit': 0, 'final_training_credit': 0, 'worker_admission': False}
    proof = save('candidate_full_tick_verification.json', result)
    print(json.dumps({'passed': True, 'verification': proof, 'actual_step_id': os.environ['SLURM_STEP_ID'],
                      'queue_counts': result['queue_counts_after']}))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        save('candidate_full_tick_failure.json', {'type': type(error).__name__, 'error': repr(error),
            'traceback': traceback.format_exc(), 'blocked_action_attempts': BLOCKED,
            'job_id': os.environ.get('SLURM_JOB_ID'), 'step_id': os.environ.get('SLURM_STEP_ID'),
            'job_actions': 0, 'worker_admission': False})
        raise
