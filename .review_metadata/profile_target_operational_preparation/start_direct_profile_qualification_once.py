"""Check or start one published full-original H100 profile qualification controller.

The default checks metadata only. Root alone invokes --execute on serval02.
The sole producer owns the standing GPU descriptor; this transport never opens it.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import pwd
import re
import shlex
import signal
import socket
import stat
import subprocess
import time

P = None
R = None
UID = 2618919
RUNTIME = 'd1469fd063a44167ec56760729eb0beea7ecf7d3'
SOURCE = 'd28e6735e79788f618acb0ebfde8609db1f776af'
PREFIX = '10_jax_exploration_platform/scripts/atari_pixel_campaign/'
ATTEMPTS = ('root_actual_controller_spawn.json', 'root_transport_popen_attempt.json',
    'root_transport_failure.json', 'root_transport_custody_inconclusive.json')

def write(path,x):
    """Publish a new launch record without replacing an earlier attempt."""
    raw=(json.dumps(x,indent=2,sort_keys=True)+'\n').encode()
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)

def retain_custody(child,error):
    """Retain the barrier through ordinary observer errors until this exact Popen child is waited."""
    # Ignore transport timeout/disconnect signals before any fallible report can run.
    previous={number:signal.getsignal(number) for number in (signal.SIGHUP,signal.SIGINT,signal.SIGTERM)}
    for number in previous:signal.signal(number,signal.SIG_IGN)
    ordinary_errors=[]
    def note(phase,problem):
        """Keep bounded error metadata; a broken report never releases descriptor custody."""
        try:
            if len(ordinary_errors)<32:
                item={'phase':phase,'error_type':type(problem).__name__,'completed':False,
                    'metadata_barrier_retained':True,'signal_sent':False}
                ordinary_errors.append(item)
                try:print(json.dumps(item),flush=True)
                except Exception:pass
        except Exception:pass
    try:
        write(P/'root_transport_custody_inconclusive.json',{
            'kind':'root_transport_cleanup_inconclusive_barrier_retained','completed':False,
            'actual_popen_pid':child.pid,'identity_or_cleanup_error_type':type(error).__name__,
            'metadata_barrier_retained':True,'signal_sent_by_custody_loop':False,'scientific_credit':0})
    except Exception as record_error:note('initial_custody_report',record_error)
    # A direct bounded wait avoids poll as a separate fallible exit claim; only a typed actual wait permits return.
    while True:
        try:
            returned=child.wait(timeout=0.1)
            if type(returned) is int:break
            note('wait_return_type',ValueError('the exact Popen wait did not return an integer status'))
        except subprocess.TimeoutExpired:
            pass
        except Exception as wait_error:
            note('exact_child_wait',wait_error)
        try:time.sleep(0.1)
        except Exception as sleep_error:note('custody_wait_sleep',sleep_error)
    # The exact original child is now positively waited; restoring handlers cannot expose a live unknown child.
    for number,handler in previous.items():signal.signal(number,handler)
    return {'kind':'exact_popen_wait_after_inconclusive_identity','actual_popen_pid':child.pid,
        'returncode':returned,'signal_sent':False,'kernel_child_exit_observed':True,
        'ordinary_observer_errors':ordinary_errors}

def cleanup_under_barrier(control,child,identity,error):
    """Reap only the exact positively verified child, or retain the barrier without signaling it."""
    try:
        polled=child.poll()
    except Exception as poll_error:
        return retain_custody(child,poll_error)
    if polled is not None:
        try:
            returned=child.wait(timeout=0)
            if type(returned) is not int:raise ValueError('the exact Popen wait returned no integer status')
        except Exception as wait_error:
            return retain_custody(child,wait_error)
        return {'kind':'exact_popen_wait_before_barrier_release','actual_popen_pid':child.pid,
            'returncode':returned,'signal_sent':False,'kernel_child_exit_observed':True}
    if identity is not None:
        try:
            returned,actions=control.wait_exact(child,identity,2618919,0)
            if type(returned) is not int:raise ValueError('verified child cleanup returned no integer wait status')
            return {'kind':'verified_owned_child_cleanup_under_metadata_barrier','actual_popen_pid':child.pid,
                'returncode':returned,'interventions':actions,'kernel_child_exit_observed':True}
        except Exception as cleanup_error:
            return retain_custody(child,cleanup_error)
    return retain_custody(child,error)

def require(value, message):
    """Refuse an incomplete actual publication or a reused launch."""
    if not value:
        raise ValueError(message)


def capture(path):
    """Read bounded stable owned regular metadata without following a substituted path."""
    path = pathlib.Path(path).absolute()
    before = path.lstat()
    require(path.resolve(strict=True) == path and stat.S_ISREG(before.st_mode)
        and before.st_uid == os.getuid() and before.st_size <= 2500000, 'metadata must be a bounded exact owned file')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode, value.st_uid, value.st_gid, value.st_nlink)
        require(identity(os.fstat(descriptor)) == identity(before), 'metadata changed before its pinned read')
        raw = os.read(descriptor, 2500001)
        require(len(raw) == before.st_size and identity(os.fstat(descriptor)) == identity(before)
            and identity(path.lstat()) == identity(before), 'metadata changed during its pinned read')
        return raw, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    finally:
        os.close(descriptor)


def document(reference):
    """Match a complete actual reference before interpreting its JSON."""
    require(type(reference) is dict and set(reference) == {'path', 'sha256'}
        and re.fullmatch('[0-9a-f]{64}', reference['sha256']) is not None, 'an exact metadata reference is required')
    raw, actual = capture(reference['path'])
    require(actual == reference, 'the declared metadata bytes changed')
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate metadata JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def git(root, *arguments):
    """Use only explicit read-only Git commands with optional index writes disabled."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    environment.update(GIT_OPTIONAL_LOCKS='0', GIT_TERMINAL_PROMPT='0')
    result = subprocess.run(['git', '-C', str(root), *arguments], capture_output=True,
        check=True, timeout=15, env=environment)
    require(result.stderr == b'', 'the read-only publication query failed')
    return result.stdout


def published_tracking(root, publication):
    """Require the exact shared origin and its already recorded pushed branch, without fetching."""
    require(git(root, 'config', '--get-all', 'remote.origin.url')
        == b'git@github.com:yuxinchen123/uncertainty_comparison_pointmaze.git\n',
        'the shared repository origin URL changed or is absent')
    require(git(root, 'config', '--get-all', 'remote.origin.fetch')
        == b'+refs/heads/*:refs/remotes/origin/*\n', 'the exact origin fetch mapping changed or is absent')
    git(root, 'merge-base', '--is-ancestor', publication, 'refs/remotes/origin/merge-atari-cleanrl')


def published(root, publication, reference):
    """Require the actual referenced bytes in the actual prior publication."""
    path = pathlib.Path(reference['path'])
    require(path.is_relative_to(root), 'a launch input is outside the published integration checkout')
    raw, actual = capture(path)
    require(actual == reference and git(root, 'show', publication + ':' + str(path.relative_to(root))) == raw,
        'a launch input was not published with these exact bytes')
    return raw


def command(controller):
    """Preserve the isolated system-Python metadata controller and exact card selector."""
    return ['/usr/bin/env', '-i', 'PATH=/usr/local/bin:/usr/bin:/bin', 'LANG=C.UTF-8', 'TZ=UTC',
        'CUDA_VISIBLE_DEVICES=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3', 'PYTHONDONTWRITEBYTECODE=1',
        'PYTHONNOUSERSITE=1', '/usr/bin/python3', '-I', '-S', '-B',
        str(controller / 'code/profile_target_qualification.py'), '--prepared-unit', str(controller)]


def verify(configuration, publication):
    """Check actual plans, source, reviews, standard records and unused launch paths."""
    global P, R
    raw, configuration_ref = capture(configuration)
    record = document(configuration_ref)
    require(record.get('schema_version') == 1 and type(record['schema_version']) is int
        and record.get('kind') == 'direct_profile_target_qualification_root_transport'
        and record.get('automatic_retry_permitted') is False, 'the explicit no-retry transport record is required')
    P, R = pathlib.Path(record['packet']), pathlib.Path(record['campaign_run_dir'])
    require(P.parent == R / 'remote_setup' and P.resolve(strict=True) == P
        and R.resolve(strict=True) == R and pathlib.Path(configuration).absolute() == P / 'root_transport_configuration.json',
        'the actual transport configuration must belong to its exact campaign packet')
    root = pathlib.Path(git(R, 'rev-parse', '--show-toplevel').decode().strip())
    require(re.fullmatch('[0-9a-f]{40}', publication) is not None, 'supply an actual full publication commit')
    published_tracking(root, publication)
    git(root, 'merge-base', '--is-ancestor', publication, 'HEAD')
    published(root, publication, configuration_ref)
    source_ref = capture(__file__)[1]
    require(record['transport_source'] == source_ref, 'the transport source changed')
    published(root, publication, source_ref)
    for key in ('controller_plan', 'child_plan', 'prepared_child_binding', 'launch_review',
            'controller_source', 'process_source', 'metadata_permission_procedure', 'log_directory_evidence'):
        published(root, publication, record[key])
    parent, child, binding, review = (document(record[key]) for key in
        ('controller_plan', 'child_plan', 'prepared_child_binding', 'launch_review'))
    controller = pathlib.Path(record['controller_unit'])
    require(controller == R / 'data' / parent['unit_id'] and controller.resolve(strict=True) == controller
        and record['controller_plan']['path'] == str(controller / 'measurement_plan.json')
        and parent['kind'] == 'profile_target_qualification_controller'
        and parent['profile_unit'] == str(R / 'data' / child['unit_id']) and parent['campaign_run_dir'] == str(R)
        and parent['source_commit'] == child['source_commit'] == SOURCE
        and child['kind'] == 'full_original_profile_source_target_qualification'
        and child['hardware_class'] == 'h100' and child['copy_seeds'] == [1]
        and type(child['copy_seeds'][0]) is int and child['configuration']['learner']['n_copies'] == 1,
        'the reviewed one-copy H100 controller/capture plans changed')
    require(child['helpers']['source_commit'] == RUNTIME and len(child['helpers']['files']) == 44,
        'the exact reviewed H100 runtime helper closure is required')
    child_directory = pathlib.Path(parent['profile_unit'])
    for relative, sha256 in child['helpers']['files'].items():
        require(relative == PREFIX + pathlib.Path(relative).name and relative.endswith('.py'), 'another helper namespace was supplied')
        staged = capture(child_directory / 'code' / pathlib.Path(relative).name)
        require(staged[1]['sha256'] == sha256 and git(root, 'show', RUNTIME + ':' + relative) == staged[0],
            'a staged runtime helper differs from its exact reviewed committed bytes')
    require(record['controller_source']['path'] == str(controller / 'code/profile_target_qualification.py')
        and record['process_source']['path'] == str(controller / 'code/direct_common_timing_controller.py')
        and capture(record['controller_source']['path'])[0] == git(root, 'show', RUNTIME + ':' + PREFIX + 'profile_target_qualification.py')
        and capture(record['process_source']['path'])[0] == git(root, 'show', RUNTIME + ':' + PREFIX + 'direct_common_timing_controller.py'),
        'the staged controller/process source differs from the exact runtime publication')
    require(binding['controller_plan'] == record['controller_plan'] and binding['child_plan'] == record['child_plan']
        and binding['controller_source'] == record['controller_source'] and binding['helpers'] == child['helpers']
        and child['profile_target_qualification']['parent'] == record['controller_plan']
        and review.get('kind') == 'independent_profile_target_qualification_controller_launch_review'
        and review.get('approved') is review.get('passed') is True and review.get('independent_of_producer') is True
        and review.get('controller_plan') == record['controller_plan'] and review.get('child_plan') == record['child_plan'],
        'the genuine bound controller launch review is required')
    require(shlex.split((controller / 'command.txt').read_text()) == command(controller)[8:]
        and binding['command'] == shlex.split((child_directory / 'command.txt').read_text()), 'a prepared command changed')
    require(parent['controller_maximum_seconds'] == 10800 and type(parent['controller_maximum_seconds']) is int
        and parent['qualification_time_budget'] == {'schema_version': 1, 'work_seconds': 10200, 'cleanup_seconds': 600, 'total_seconds': 10800},
        'the separate three-hour qualification budget changed')
    require(type(record['standard_records']) is list and len(record['standard_records']) == 8
        and {value['path'] for value in record['standard_records']} == {str(folder / name)
            for folder in (P, controller) for name in ('manifest.yaml', 'config_resolved.yaml', 'command.txt', 'experiment_background.md')},
        'both actual packet and controller need four published standard records')
    for item in record['standard_records']:
        published(root, publication, item)
    transport_review_path = P / 'root_transport_independent_review.json'
    transport_review_ref = capture(transport_review_path)[1]
    published(root, publication, transport_review_ref)
    transport_review = document(transport_review_ref)
    require(transport_review.get('kind') == 'independent_direct_profile_target_qualification_root_transport_review'
        and transport_review.get('approved') is transport_review.get('passed') is True
        and transport_review.get('independent_of_producer') is True
        and transport_review.get('configuration') == configuration_ref and transport_review.get('source') == source_ref
        and transport_review.get('unresolved_anomalies') == [], 'the independent actual transport review is required')
    procedure = document(record['metadata_permission_procedure'])
    require(procedure.get('kind') == 'root_only_owned_metadata_permission_protection_and_same_file_restore_procedure'
        and procedure.get('protected_mode_before_ssh') == 0o640
        and procedure.get('restore_mode_after_actual_registration_or_verified_failed_startup_exit') == 0o660
        and {row['path'] for row in procedure['current_metadata_lock_snapshots']}
            == {str(R / 'queue/metadata.lock'), str(R / 'processes.lock')},
        'the two exact owner metadata files need the reviewed protection and restoration procedure')
    log_directory = R / 'slurm/direct_jobs' / P.name
    require(log_directory.resolve(strict=True) == log_directory and log_directory.is_dir()
        and log_directory.stat().st_uid == os.getuid(), 'the exact owned controller log directory must already exist')
    evidence = document(record['log_directory_evidence'])
    require(evidence.get('path') == str(log_directory) and evidence.get('target_visibility_verified') is True
        and evidence.get('hostname') == 'serval02' and evidence.get('owner_uid') == UID,
        'actual direct-host visibility of the owned log directory must be recorded')
    require(record['stdout'] == str(log_directory / 'controller_stdout.raw')
        and record['stderr'] == str(log_directory / 'controller_stderr.raw')
        and not any(os.path.lexists(record[key]) for key in ('stdout', 'stderr'))
        and not any(os.path.lexists(P / name) for name in ATTEMPTS)
        and not any(os.path.lexists(controller / name) for name in ('controller_intent.json', 'controller_spawn.json',
            'controller_wait.json', 'controller_failure.json', 'controller_inconclusive.json')),
        'the exact launch and output paths must never have been attempted')
    return record, controller, command(controller)


def metadata_module(record):
    """Load the exact frozen pure owned-process functions, not a numerical runtime."""
    spec = importlib.util.spec_from_file_location('profile_target_root_transport_metadata', record['process_source']['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.no_numerical_modules()
    return module


def execute(record, controller, argv, publication):
    """Fsync the real controller child identity while the same metadata barrier remains held."""
    require(socket.gethostname().split('.')[0] == 'serval02'
        and os.getuid() == os.geteuid() == UID and os.getresuid() == (UID, UID, UID)
        and pwd.getpwuid(UID).pw_name == 'sl5nw' and not any(key.startswith('SLURM_') for key in os.environ),
        'only root session direct-host execution under the exact owner is supported')
    intent = document(capture(P / 'root_remote_launch_intent.json')[1])
    require(intent.get('kind') == 'one_root_direct_profile_qualification_ssh_intent'
        and intent.get('configuration') == capture(P / 'root_transport_configuration.json')[1]
        and intent.get('publication') == publication and intent.get('automatic_retry_permitted') is False,
        'the actual one-shot SSH intent must be fsynced with the exact configuration and publication first')
    for snapshot in document(record['metadata_permission_procedure'])['current_metadata_lock_snapshots']:
        path = pathlib.Path(snapshot['path']);raw, actual = capture(path);status = path.lstat()
        require(status.st_ino == snapshot['st_ino'] and status.st_uid == UID == snapshot['st_uid']
            and status.st_nlink == snapshot['st_nlink'] == 1 and stat.S_IMODE(status.st_mode) == 0o640
            and len(raw) == snapshot['size'] and actual['sha256'] == snapshot['sha256'],
            'the same exact owned metadata files must be protected before the one actual SSH')
    control = metadata_module(record)
    parent = control.read_process(os.getpid(), UID)
    lock = R / 'queue/metadata.lock'
    require(lock.is_file() and not lock.is_symlink(), 'the owner metadata barrier is missing')
    child, identity = None, None
    with control.metadata_lock(lock, UID):
        write(P / 'root_transport_popen_attempt.json', {'kind': 'one_root_profile_qualification_controller_popen_attempt',
            'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'parent': parent, 'command': argv,
            'actual_child_pid': None, 'metadata_barrier_path': str(lock), 'automatic_retry_permitted': False})
        try:
            with pathlib.Path(record['stdout']).open('xb', buffering=0) as stdout, pathlib.Path(record['stderr']).open('xb', buffering=0) as stderr:
                child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    start_new_session=True, close_fds=True, cwd=P)
                candidate = control.read_process(child.pid, UID)
                require(candidate['parent_pid'] == os.getpid() and candidate['hostname'] == 'serval02'
                    and candidate['command'] in (argv, argv[8:]), 'the real controller identity or command differs')
                identity = candidate
                after = control.read_process(child.pid, UID)
                require(after['process_start_time'] == identity['process_start_time'] and after['uid'] == UID
                    and child.poll() is None, 'the real controller generation changed before its spawn record')
                result = {'kind': 'actual_controller_transport_popen_return',
                    'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'actual_pid': child.pid,
                    'actual_birth_counter': identity['process_start_time'], 'host': 'serval02', 'uid': UID,
                    'intended_command': argv, 'actual_controller_identity': identity, 'actual_numerical_child_pid': None,
                    'automatic_retry_permitted': False, 'actual_qualification_verified': False, 'scientific_credit': 0,
                    'metadata_barrier_path': str(lock), 'metadata_barrier_held_through_record_fsync': True}
                write(P / 'root_actual_controller_spawn.json', result)
        except Exception as error:
            cleanup = None if child is None else cleanup_under_barrier(control, child, identity, error)
            write(P / 'root_transport_failure.json', {'kind': 'failed_root_controller_transport_under_metadata_barrier',
                'error_type': type(error).__name__, 'error': str(error),
                'actual_popen_pid': None if child is None else child.pid, 'cleanup': cleanup,
                'ordinary_launch_success': False, 'automatic_retry_permitted': False, 'scientific_credit': 0})
            raise
    return result


def main(argv=None):
    """Require explicit root execution; the default performs read-only publication checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', type=pathlib.Path, required=True)
    parser.add_argument('--publication', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    record, controller, command_line = verify(args.configuration, args.publication)
    if not args.execute:
        print(json.dumps({'action_taken': False, 'command': command_line, 'scientific_credit': 0}))
        return
    print(json.dumps(execute(record, controller, command_line, args.publication)), flush=True)


if __name__ == '__main__':
    main()
