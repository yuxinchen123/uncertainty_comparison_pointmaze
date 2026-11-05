"""Observe selected-Python native process control on one owned A4500 with no scientific payload."""
import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import nsight_process_identity as identity
import configured_native_process as native

NAME = 'diagnose_configured_native_process.py'
HELPERS = (NAME, 'nsight_process_identity.py', *native.HELPERS)
NUMERICAL = {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'cupy', 'scipy', 'jaxtari', 'exploration_platform'}
ZERO = {'scientific_credit': 0, 'profile_credit': 0, 'case_credit': 0, 'idea_credit': 0,
    'environment_transitions': 0, 'training_updates': 0, 'admission_granted': False,
    'worker_admitted': False, 'physical_gres_mapping_verified': False,
    'kernel_device_enforcement_verified': False, 'standing_gpu_descriptor_held': False,
    'whole_tool_library_or_device_activity_quantified': False}


@contextmanager
def producer_interruptions():
    """Enter owned descendant cleanup on ordinary termination before restoring prior signal handlers."""
    def interrupted(number, frame):
        """Preserve one exact producer interruption as an unsuccessful tool-scope outcome."""
        raise InterruptedError('the configured native producer received signal ' + str(number))
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    for number in previous:signal.signal(number, interrupted)
    try:yield
    finally:
        for number, handler in previous.items():signal.signal(number, handler)


def metadata_guard():
    """Forbid numerical imports and late library, network or Python device operations after bootstrap."""
    identity.require(not any(name.split('.')[0] in NUMERICAL for name in sys.modules), 'numerical startup is forbidden')
    class NoNumerical(importlib.abc.MetaPathFinder):
        """Keep every standard-library diagnostic role free of a scientific runtime."""
        def find_spec(self, name, path=None, target=None):
            """Refuse numerical imports before they execute any source."""
            if name.split('.')[0] in NUMERICAL:raise RuntimeError('numerical imports are forbidden')
    sys.meta_path.insert(0, NoNumerical())
    def audit(event, values):
        """Refuse late library, network, shell and direct Python device work."""
        if event in {'ctypes.dlopen', 'ctypes.dlsym', 'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'os.system'}:
            raise RuntimeError('late library, network and shell work is forbidden')
        if event == 'open' and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(('/dev/nvidia', '/dev/dri')):
            raise RuntimeError('Python device opening is forbidden')
    sys.addaudithook(audit)


def read_plan(directory):
    """Require the exact staged seven-helper diagnostic without accepting execution overrides."""
    directory = Path(directory).resolve(strict=True)
    plan = identity.read_json(directory / 'measurement_plan.json');native.policy(plan)
    identity.require(plan['kind'] == native.DIAGNOSTIC and directory.name == plan['unit_id']
        and directory == Path(plan['campaign_run_dir']) / 'data' / plan['unit_id']
        and Path(__file__).resolve(strict=True) == directory / 'code' / NAME
        and str(Path(sys.executable).absolute()) == plan['python']
        and plan['diagnostic_boundary'] == ZERO
        and identity.read_json(directory / 'measurement_prepared.json') == {'plan_sha256': identity.receipt(directory / 'measurement_plan.json')['sha256']}
        and set(plan['helpers']['files']) == {native.PREFIX + name for name in HELPERS}, 'execute only the exact prepared native allocation diagnostic')
    for name in HELPERS:
        path = directory / 'code' / name
        identity.require(path.resolve(strict=True) == path and identity.receipt(path)['sha256'] == plan['helpers']['files'][native.PREFIX + name],
            'a configured native diagnostic helper changed')
    return directory, plan


def owned_launch(directory, plan, deadline):
    """Require actual fsynced owned submission and release records before any tool is executed."""
    path = Path(directory) / 'owned_launch_inputs.json'
    while not os.path.lexists(path):
        time.sleep(min(.05, deadline.remaining('work')))
    value = identity.read_json(path);job = os.environ.get('SLURM_JOB_ID')
    identity.require(value['job_id'] == job and value['plan'] == identity.receipt(Path(directory) / 'measurement_plan.json')
        and os.environ.get('SLURM_CPUS_PER_TASK') == '1' and os.environ.get('SLURM_MEM_PER_NODE') == '2048',
        'this diagnostic requires its own exact one-CPU two-GiB one-GPU numeric step')
    row = [job, native.DIAGNOSTIC, plan['helpers']['source_commit'], str(Path(plan['packet']) / 'job.sbatch')]
    identity.require(value['submission_row'] == row and (Path(plan['campaign_run_dir']) / 'slurm/submissions.tsv').read_text().splitlines().count('\t'.join(row)) == 1,
        'the exact native diagnostic row must already be durably campaign-owned')
    for field, kind in (('submission', 'owned_configured_native_process_diagnostic_submission'), ('release', 'owned_configured_native_process_diagnostic_release')):
        actual = native.reference_value(value[field], identity)
        identity.require(actual.get('kind') == kind and actual.get('job_id') == job
            and type(actual.get('returncode')) is int and actual['returncode'] == 0
            and actual.get('plan_sha256') == value['plan']['sha256'] and actual.get('data_unit') == str(directory),
            'the native diagnostic needs actual successful owned submission and release receipts')
    return value


def read_release(directory, plan, claim, observer, parent, deadline):
    """Wait on the pinned original producer until its exact observation authorizes this payload."""
    path = directory / 'capture_release.json'
    while not os.path.lexists(path):
        identity.require(not observer.handle_exited(parent), 'the pinned producer disappeared before release')
        time.sleep(min(.05, deadline.remaining('work')))
    release = observer.read_json(path)
    owner = observer.read_json(directory / 'profiler_process.json')
    identity.require(release == {**owner, 'capture_identity': claim,
        'process_observation': observer.receipt(directory / 'capture_process_observation.json')}, 'the actual payload release changed')
    identity.require(not observer.handle_exited(parent), 'the pinned producer exited before release validation')
    observer.validate_live_observation(directory, plan, claim)
    identity.require(not observer.handle_exited(parent), 'the pinned producer exited during release validation')
    observer.publish(directory / 'capture_release_observed.json', {'release': observer.receipt(path),
        'capture_identity': observer.receipt(directory / 'capture_identity.json'), 'observed_at_seconds': time.time(), **ZERO})


def probe(directory, plan, runtime, deadline):
    """Execute no learner or environment work after a positively observed exact native release."""
    observer = runtime.observer;claim = observer.claim_self(directory, plan)
    intent = observer.read_json(directory / 'profile_process_intent.json');parent = observer.parent_handle(intent)
    try:
        read_release(directory, plan, claim, observer, parent, deadline)
        started = time.time();finished = time.time();deadline.remaining('work')
        observer.publish(directory / 'native_capture_payload_timing.json', {'payload_started_at_seconds': started,
            'payload_finished_at_seconds': finished, 'diagnostic_deadline': deadline.snapshot(), **ZERO})
        observer.publish(directory / 'probe_result.json', {'completed': True, 'pid': claim['pid'],
            'process_start_time': claim['process_start_time'], 'numerical_modules': [], 'cuda_profiler_api_called': False, **ZERO})
    finally:
        observer.close_process_handle(parent)
    runtime.complete(native.completion_references(plan, 'capture'))


def waiter(directory, plan, runtime, deadline):
    """Popen only the exact probe command and retain its real parent wait and pinned producer watch."""
    observer = runtime.observer
    intent = observer.validate_intent(directory, plan, observer.read_json(directory / 'profile_process_intent.json'))
    current = native.context(directory, plan, observer)
    identity.require(current['command'] == intent['waiter_command'], 'the actual native waiter command changed')
    parent = observer.parent_handle(intent);child = None;child_handle = None
    try:
        child = subprocess.Popen(intent['capture_command'], close_fds=True, pass_fds=())
        child_handle = observer.open_process_handle(child.pid)
        while not (directory / 'capture_identity.json').exists():
            identity.require(child.poll() is None and not observer.handle_exited(parent), 'the actual probe or original producer exited before its claim')
            time.sleep(min(.05, deadline.remaining('work')))
        claim = observer.validate_claim(directory, plan, intent, observer.read_json(directory / 'capture_identity.json'))
        actual = observer.ProcReader().snapshot(child.pid, os.getuid(), current['cgroup'])
        identity.require(actual['parent_pid'] == current['pid'] and observer.stable(actual) == observer.stable(claim['process']),
            'the actual fresh probe Popen child differs from its source-bound claim')
        identity.require(not observer.handle_exited(child_handle), 'the positively observed native probe handle exited before release')
        observer.publish(directory / 'capture_actual_launch.json', {'schema_version': 1, 'challenge': intent['challenge'],
            'waiter': current, 'child': actual, 'command': intent['capture_command'], 'claim': observer.receipt(directory / 'capture_identity.json')})
        while child.poll() is None:
            identity.require(not observer.handle_exited(parent), 'the pinned producer disappeared before the actual child wait')
            time.sleep(min(.05, deadline.remaining('work')))
        returned = child.wait(timeout=deadline.timeout())
        waited_at = time.time()
        observer.publish(directory / 'capture_actual_wait.json', {'schema_version': 1, 'challenge': intent['challenge'],
            'command': intent['capture_command'], 'waited_pid': child.pid, 'returncode': returned,
            'exited_normally': returned >= 0, 'interventions': [], 'actual_wait_finished_at_seconds': waited_at,
            'launch': observer.receipt(directory / 'capture_actual_launch.json')})
        identity.require(returned == 0, 'the actual zero-transition native probe failed')
    finally:
        # Unknown or incomplete cleanup remains failed; the producer owns the complete descendant scope.
        if child_handle is not None:observer.close_process_handle(child_handle)
        observer.close_process_handle(parent)
    runtime.complete(native.completion_references(plan, 'waiter'))
    return returned


def producer(directory, plan, runtime, deadline):
    """Reuse one provider and closed child scope from before the first tool through all final waits."""
    observer = runtime.observer;owner = runtime.owner
    scope = native.tool_scope(directory, plan, observer, owner, deadline)
    metadata_guard()
    processes = None
    with producer_interruptions(), scope:
        owned_launch(directory, plan, deadline)
        tool = plan['profile_tool']
        identity.require(identity.receipt(tool['path'])['sha256'] == tool['sha256'], 'the actual Nsight tool bytes changed')
        version = scope.run([tool['path'], '--version'], capture_output=True, timeout=deadline.timeout(15))
        observer.publish(directory / 'tool_version_return.json', {'command': [tool['path'], '--version'],
            'returncode': version.returncode, 'stdout': version.stdout.decode(), 'stderr': version.stderr.decode()})
        identity.require(version.returncode == 0 and version.stderr == b'' and version.stdout.decode().strip() == tool['version'], 'the actual Nsight version changed')
        (directory / 'trace').mkdir()
        command = observer.nsight_command(directory, plan);observer.publish(directory / 'executed_command.json', command)
        intent = observer.make_intent(directory, plan, command)
        with (directory / 'capture.log').open('x') as log:
            child = scope.spawn(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            root = scope.nodes[child.pid]['initial']
            launched = {'pid': child.pid, 'process_start_time': root['process_start_time'], 'parent_pid': owner['pid'],
                'parent_start_time': owner['process_start_time'], 'command': command}
            observer.publish(directory / 'profiler_process.json', launched)
            try:
                while not all((directory / name).exists() for name in ('capture_identity.json', 'capture_actual_launch.json')):
                    identity.require(child.poll() is None, 'the actual Nsight tool exited before the blocked probe claim')
                    scope.scan();time.sleep(min(.05, deadline.remaining('work')))
                claim = observer.validate_claim(directory, plan, intent, observer.read_json(directory / 'capture_identity.json'))
                processes = observer.OwnedProcesses(child.pid, root['process_start_time'], intent['parent'])
                observation = processes.observe(claim['pid'], claim['process_start_time'], intent['capture_command'])
                observer.publish(directory / 'capture_process_observation.json', observation)
                scope.scan()
                observer.publish(directory / 'capture_release.json', {**launched, 'capture_identity': claim,
                    'process_observation': observer.receipt(directory / 'capture_process_observation.json')})
                while child.poll() is None:
                    scope.scan();time.sleep(min(.05, deadline.remaining('work')))
                returned = child.wait(timeout=deadline.timeout());scope.record_wait(child)
                observer.record_root_wait(directory, intent, launched, returned)
                identity.require(returned == 0 and observer.read_json(directory / 'capture_actual_wait.json')['returncode'] == 0,
                    'the actual source-bound tool and probe must both return zero')
            finally:
                if processes is not None:processes.close()
    runtime.complete(native.completion_references(plan, 'producer'))
    intent = observer.validate_saved_intent(directory, plan, observer.read_json(directory / 'profile_process_intent.json'))
    observation = observer.read_json(directory / 'capture_process_observation.json')
    claim = observer.validate_claim(directory, plan, intent, observer.read_json(directory / 'capture_identity.json'))
    observer.validate_observation(observation, claim, intent['capture_command'], intent['parent'])
    observer.validate_root_wait(directory, intent, observer.read_json(directory / 'profiler_process.json'))
    native_saved = native.validate_saved(directory, plan, observer, intent, observation,
        observer.read_json(directory / 'capture_actual_launch.json'), observer.read_json(directory / 'capture_actual_wait.json'), deadline)
    deadline.remaining('total')
    observer.publish(directory / 'result.json', {'schema_version': 1, 'kind': native.DIAGNOSTIC, 'completed': True,
        'plan': observer.receipt(directory / 'measurement_plan.json'), 'target_contract': observer.receipt(directory / 'native_process_target_contract.json'),
        'source': plan['helpers'], 'native_process_evidence': native_saved, 'tool': tool,
        'diagnostic_deadline': deadline.snapshot(), **ZERO})


def prepare(args):
    """Write immutable standard records only; never submit a job or execute the target tool."""
    root = args.source_root.resolve(strict=True);run = args.run_dir.resolve(strict=True)
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    identity.require(not subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all'], text=True)
        and root / native.PREFIX == SCRIPTS, 'prepare only from the exact clean published helper source')
    identity.require(Path(args.unit_id).name == args.unit_id and args.unit_id not in ('', '.', '..'), 'one fresh literal unit identity is required')
    tool = identity.read_json(args.profile_tool_record)
    identity.require(set(tool) == {'path', 'version', 'sha256'} and Path(tool['path']).is_absolute()
        and type(tool['sha256']) is str and len(tool['sha256']) == 64 and type(tool['version']) is str and tool['version'],
        'use the exact previously observed Nsight tool metadata without executing it during preparation')
    directory = run / 'data' / args.unit_id
    plan = {'schema_version': 1, 'kind': native.DIAGNOSTIC, 'unit_id': args.unit_id, 'campaign_run_dir': str(run),
        'packet': str(run / 'remote_setup' / args.unit_id), 'python': str(args.python.absolute()), 'hardware_class': 'jaguar03_rtx_a4500',
        'gpu_uuid': None, 'maximum_active_seconds': 240, 'diagnostic_time_budget': dict(native.BUDGET), 'diagnostic_boundary': ZERO,
        'profile_tool': tool, 'helpers': {'source_commit': commit,
            'files': {native.PREFIX + name: identity.receipt(SCRIPTS / name)['sha256'] for name in HELPERS}}}
    plan.update(native.declaration(plan));native.policy(plan)
    directory.mkdir();(directory / 'code').mkdir()
    for name in HELPERS:
        with (directory / 'code' / name).open('xb') as stream:
            stream.write((SCRIPTS / name).read_bytes());stream.flush();os.fsync(stream.fileno())
    for name, value in (('measurement_plan.json', plan), ('manifest.yaml', plan), ('config_resolved.yaml', {**ZERO, 'budget': native.BUDGET})):
        identity.publish(directory / name, value)
    identity.publish(directory / 'measurement_prepared.json', {'plan_sha256': identity.receipt(directory / 'measurement_plan.json')['sha256']})
    with (directory / 'command.txt').open('x') as stream:
        stream.write(shlex.join(native.producer_command(directory, plan)) + '\n');stream.flush();os.fsync(stream.fileno())
    with (directory / 'experiment_background.md').open('x') as stream:
        stream.write('## Purpose\n\nObserve native process handles on one owned jaguar03 A4500 with selected Python. '
            'The standard-library probe performs zero environment transitions and zero learner updates.\n\n'
            '## Key hyperparameters\n\nOne GPU, one CPU, two GiB of host memory and five minutes. '
            'One 300-second total limit reserves 240 seconds for all work and 60 seconds for cleanup. '
            'The exact selected Python and tool metadata are in the immutable plan.\n\n'
            '## Code and config changes\n\nExplicit native process handles replace missing Python exports for this diagnostic only. '
            'One provider per actual process and one producer child scope retain real owned process evidence. '
            'No standing GPU descriptor is held. Actual tool library and device work is unquantified. '
            'No scientific, profile, capacity or admission credit is granted.\n\n'
            '## Git state\n\nCommit: ' + commit + '. Working tree clean at preparation.\n')
        stream.flush();os.fsync(stream.fileno())
    return directory


def main(argv=None):
    """Separate immutable metadata preparation from explicit actual producer, waiter and probe roles."""
    started = time.monotonic();parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--prepare-only', action='store_true')
    for name in ('prepared-unit', 'waiter-unit', 'probe-unit'):operation.add_argument('--' + name, type=Path)
    for name in ('source-root', 'run-dir', 'profile-tool-record', 'python'):parser.add_argument('--' + name, type=Path)
    parser.add_argument('--unit-id');args = parser.parse_args(argv)
    if args.prepare_only:
        identity.require(all(getattr(args, name) is not None for name in ('source_root', 'run_dir', 'profile_tool_record', 'python', 'unit_id')), 'preparation inputs must be explicit')
        print(json.dumps({'prepared_unit': str(prepare(args)), 'jobs_submitted': 0, 'gpu_operations': 0}));return
    identity.require(not any(getattr(args, name) is not None for name in ('source_root', 'run_dir', 'profile_tool_record', 'python', 'unit_id')), 'execution cannot override a frozen plan')
    directory, plan = read_plan(args.prepared_unit or args.waiter_unit or args.probe_unit)
    owner = native.context(directory, plan, identity)
    role = 'producer' if args.prepared_unit else 'waiter' if args.waiter_unit else 'capture'
    expected = native.producer_command(directory, plan) if role == 'producer' else identity.commands(directory, plan)[0 if role == 'waiter' else 1]
    identity.require(owner['command'] == expected, 'native bootstrap requires the exact actual staged role command')
    clock = native.deadline(directory, plan, identity, owner if role == 'producer' else None, started if role == 'producer' else None)
    runtime = native.bootstrap(directory, plan, role, identity, owner, clock)
    try:
        if role == 'producer':producer(directory, plan, runtime, clock)
        else:
            metadata_guard()
            if role == 'waiter':return waiter(directory, plan, runtime, clock)
            probe(directory, plan, runtime, clock)
    except BaseException as error:
        runtime.failure(error)
        identity.publish(directory / ('failure.json' if role == 'producer' else role + '_failure.json'),
            {'error': repr(error), 'traceback': traceback.format_exc(), 'completed': False, **ZERO})
        raise


if __name__ == '__main__':raise SystemExit(main())
