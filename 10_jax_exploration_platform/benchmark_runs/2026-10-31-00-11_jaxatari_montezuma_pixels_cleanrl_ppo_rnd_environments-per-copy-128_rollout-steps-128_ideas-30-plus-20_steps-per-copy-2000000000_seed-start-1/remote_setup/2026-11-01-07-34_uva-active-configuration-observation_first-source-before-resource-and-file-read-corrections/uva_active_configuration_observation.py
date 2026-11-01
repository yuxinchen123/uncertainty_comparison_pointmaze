"""Observe owned CPU allocation and cached Slurm configuration without GPU access.

This component records evidence, including missing and untrusted files. It never
qualifies a device mapping, changes a source profile or admits a training worker.
"""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess


KIND = 'uva_active_configuration_metadata_observation'
FILENAME = 'uva_active_configuration_observation.py'
PREFIX = '10_jax_exploration_platform/scripts/atari_pixel_campaign/'
OWNER = {'username': 'sl5nw', 'uid': 2618919}
SHAPE = {'hostname': 'jaguar03', 'nodes': 1, 'ntasks': 1, 'requested_cpus_per_task': 1,
         'host_memory_gib': 2, 'time_limit_minutes': 5, 'gpu_count': 0,
         'partition': 'gpu', 'account': 'sl5nw_base', 'qos': 'csresnolim'}
SETTINGS = ('SLURM_CONF', 'SlurmdSpoolDir', 'TaskPlugin', 'ProctrackType', 'GresTypes')
ZERO_CREDIT = {'environment_transitions': 0, 'learner_updates': 0, 'case_credit': 0,
               'idea_credit': 0, 'packing_point_credit': 0, 'throughput_credit': 0,
               'site_qualification_credit': 0, 'admission_granted': False,
               'physical_assignment_verified': False, 'kernel_device_enforcement_verified': False}


def require(condition, message):
    """Reject unsupported ownership or preparation before querying another allocation."""
    if not condition:
        raise ValueError(message)


def encoded(value):
    """Serialize finite metadata with stable readable formatting."""
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def write_once(path, contents):
    """Keep each original artifact immutable, including unsuccessful observations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as output:
        output.write(contents)
        output.flush()
        os.fsync(output.fileno())


def sha256(contents):
    """Hash exact captured or staged bytes."""
    return hashlib.sha256(contents).hexdigest()


def progress(unit, value):
    """Replace only this unit's mutable status after its complete bytes reach disk."""
    temporary = unit / 'progress.json.tmp'
    write_once(temporary, encoded(value))
    temporary.replace(unit / 'progress.json')


def ordinary(path):
    """Read an exact ordinary metadata artifact without following a symlink."""
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            'observation artifacts require exact ordinary absolute paths')
    return path.read_bytes()


def source_identity(root, commit):
    """Bind the observer to a clean detached published source and its exact committed bytes."""
    require(root.is_absolute() and root.resolve(strict=True) == root and re.fullmatch(r'[0-9a-f]{40}', commit),
            'an exact ordinary source root and full commit are required')
    command = ['git', '-C', str(root)]
    require(subprocess.check_output(command + ['rev-parse', '--show-toplevel'], text=True).strip() == str(root)
        and subprocess.check_output(command + ['rev-parse', 'HEAD'], text=True).strip() == commit
        and subprocess.run(command + ['symbolic-ref', '--quiet', 'HEAD'], capture_output=True).returncode == 1
        and not subprocess.check_output(command + ['status', '--porcelain=v1', '-uall'], text=True).strip(),
        'the observer source must remain clean and detached at the recorded commit')
    relative = PREFIX + FILENAME
    raw = subprocess.check_output(command + ['show', commit + ':' + relative])
    require(ordinary(root / relative) == raw, 'observer source differs from its committed bytes')
    return {'source_root': str(root), 'source_commit': commit, 'files': {relative: sha256(raw)}}


def prepare(run, unit_id, source_root, source_commit, python):
    """Create standard records and staged code before any Slurm identifier exists."""
    require(run.is_absolute() and run.resolve(strict=True) == run and run.is_dir(), 'exact existing campaign required')
    require(unit_id and Path(unit_id).name == unit_id and unit_id not in {'.', '..'}, 'plain observation unit name required')
    source = source_identity(source_root, source_commit)
    unit = run / 'data' / unit_id
    plan = {'schema_version': 1, 'kind': KIND, 'campaign_run_dir': str(run), 'unit_id': unit_id,
        'source': source, 'owner': OWNER, 'shape': SHAPE, 'python': str(python.absolute()),
        'job_id': None, 'reservation': None, 'gpu_uuid': None,
        'owned_job_ledger': str(run / 'slurm/submitted_jobids_uva.txt'),
        'settings_allowlist': list(SETTINGS), **ZERO_CREDIT}
    command = [plan['python'], str(unit / 'code' / FILENAME), 'run', str(unit)]
    background = ('## Purpose\n\nObserve the actual owned CPU allocation and Slurm configuration paths. '
        'No GPU library, device or learner is used. Missing or untrusted configuration remains evidence, '
        'and this observation grants no physical assignment, qualification or training admission.\n\n'
        '## Key hyperparameters\n\nOne requested processor, two GiB host memory, five minutes and zero GPUs '
        'on jaguar03. Record actual allocated processors and affinity, including simultaneous multithreading.\n\n'
        '## Code and config changes\n\nRead only /etc/slurm and the conf-cache directory under the actual '
        'SlurmdSpoolDir reported by Slurm. Filter slurm.conf to the explicit non-secret setting list; '
        'retain ordinary gres.conf and cgroup.conf bytes and path ownership metadata.\n\n'
        '## Git state\n\nObserver source: ' + source_commit + '.\n').encode()
    for name, value in {'observation_plan.json': encoded(plan), 'manifest.yaml': encoded(plan),
            'config_resolved.yaml': encoded({'shape': SHAPE, 'settings_allowlist': list(SETTINGS)}),
            'experiment_background.md': background,
            'command.txt': (__import__('shlex').join(command) + '\n').encode(),
            'code/' + FILENAME: ordinary(source_root / PREFIX / FILENAME)}.items():
        write_once(unit / name, value)
    write_once(unit / 'prepared.json', encoded({'plan_sha256': sha256(ordinary(unit / 'observation_plan.json'))}))
    progress(unit, {'kind': KIND, 'unit_id': unit_id, 'state': 'planned', 'phase': 'prepared_unsubmitted',
        'time': datetime.now(timezone.utc).isoformat(), **ZERO_CREDIT})
    return unit


def verify_static(unit, execution=False):
    """Verify the unresolved plan and exact staged source before reading allocation metadata."""
    plan = json.loads(ordinary(unit / 'observation_plan.json'))
    require(type(plan.get('schema_version')) is int and plan['schema_version'] == 1 and plan['kind'] == KIND
        and plan['owner'] == OWNER and plan['shape'] == SHAPE and plan['settings_allowlist'] == list(SETTINGS)
        and unit == Path(plan['campaign_run_dir']) / 'data' / plan['unit_id']
        and plan['owned_job_ledger'] == str(Path(plan['campaign_run_dir']) / 'slurm/submitted_jobids_uva.txt')
        and all(plan[name] is None for name in ('job_id', 'gpu_uuid', 'reservation'))
        and all(type(plan[name]) is type(value) and plan[name] == value for name, value in ZERO_CREDIT.items()),
        'unsupported observation preparation or nonzero credit')
    require(json.loads(ordinary(unit / 'prepared.json')) == {'plan_sha256': sha256(ordinary(unit / 'observation_plan.json'))},
            'the original observation plan changed')
    require(source_identity(Path(plan['source']['source_root']), plan['source']['source_commit']) == plan['source']
        and sha256(ordinary(unit / 'code' / FILENAME)) == plan['source']['files'][PREFIX + FILENAME],
        'the committed or staged observer changed')
    if execution:
        require(Path(__file__).absolute() == unit / 'code' / FILENAME, 'execute only the staged observer')
    return plan


def filtered_settings(raw):
    """Keep allowed non-secret assignments in order, including duplicates for later review."""
    allowed = {name.lower() for name in SETTINGS}
    rows = []
    for line in raw.decode().splitlines():
        match = re.fullmatch(r'\s*([A-Za-z_]+)\s*=\s*([^#]*?)(?:\s*#.*)?', line)
        if match and match[1].lower() in allowed:
            rows.append(match[1] + ' = ' + match[2].strip() + '\n')
    return ''.join(rows)


def bind_held_submission(unit, job_id, reservation):
    """Bind the actual returned held identifier after its fsynced canonical ledger entry, without editing the plan."""
    plan = verify_static(unit)
    require(re.fullmatch(r'[1-9][0-9]*', job_id) and re.fullmatch(r'sl5nw_[1-9][0-9]*', reservation),
            'actual returned job identifier and owned reservation required')
    ledger = ordinary(Path(plan['owned_job_ledger']))
    require(ledger.decode().splitlines().count(job_id) == 1, 'returned held identifier must already occur once in the owned ledger')
    record = {'schema_version': 1, 'kind': KIND + '_returned_held_identifier_binding', 'cluster': 'uva',
        'job_id': job_id, 'reservation': reservation, 'owner': OWNER,
        'plan_sha256': sha256(ordinary(unit / 'observation_plan.json')),
        'owned_ledger_origin': plan['owned_job_ledger'], 'owned_ledger_sha256': sha256(ledger),
        'held_state_verified_by_this_function': False, 'job_released_by_producer': False, **ZERO_CREDIT}
    write_once(unit / 'held_owned_job_ledger_snapshot.txt', ledger)
    write_once(unit / 'held_identifier_binding.json', encoded(record))
    return record


def configuration_query():
    """Capture only allowlisted running settings, never the complete controller configuration."""
    command = ['scontrol', 'show', 'config']
    result = subprocess.run(command, capture_output=True, timeout=20, check=False)
    filtered = filtered_settings(result.stdout)
    return {'command': command, 'returncode': result.returncode, 'filtered_bytes': filtered,
        'filtered_sha256': sha256(filtered.encode()), 'full_output_sha256': sha256(result.stdout),
        'stderr_sha256': sha256(result.stderr), 'stderr_byte_count': len(result.stderr)}


def path_metadata(path):
    """Record ordinary-file or directory metadata and preserve every inaccessible path explicitly."""
    row = {'path': str(path)}
    try:
        value = path.lstat()
        row.update(status='observed', owner_uid=value.st_uid, mode=stat.S_IMODE(value.st_mode),
            inode=value.st_ino, device=value.st_dev, directory=stat.S_ISDIR(value.st_mode),
            regular_file=stat.S_ISREG(value.st_mode), symlink=stat.S_ISLNK(value.st_mode),
            exact_path=path.resolve(strict=True) == path)
    except (OSError, RuntimeError) as error:
        row.update(status='unavailable', error_type=type(error).__name__, error=str(error))
    return row


def configuration_file(path, kind):
    """Read only exact ordinary allowlisted files and retain failures without accepting their ownership."""
    row = {'kind': kind, **path_metadata(path), 'directory_chain':
        [path_metadata(parent) for parent in reversed((path.parent, *path.parent.parents))]}
    if (row.get('regular_file') is not True or row.get('exact_path') is not True
            or any(part.get('symlink') is not False or part.get('directory') is not True
                   or part.get('exact_path') is not True for part in row['directory_chain'])):
        row['read_status'] = 'not_read_ordinary_path_unavailable'
        return row
    try:
        require(path.stat().st_size <= 8 * 1024 * 1024, 'configuration file exceeds the observation size limit')
        raw = path.read_bytes()
        row.update(read_status='read', byte_count=len(raw), sha256=sha256(raw))
        if kind == 'slurm':
            filtered = filtered_settings(raw)
            row.update(filtered_bytes=filtered, filtered_sha256=sha256(filtered.encode()),
                include_directive_present=bool(re.search(rb'(?im)^\s*include\b', raw)))
        else:
            row['raw_base64'] = base64.b64encode(raw).decode('ascii')
            row['raw'] = raw.decode()
    except Exception as error:
        row.update(read_status='failed', error_type=type(error).__name__, error=str(error))
    return row


def configuration_candidates(query):
    """Read /etc/slurm and only a unique actually reported SlurmdSpoolDir/conf-cache directory."""
    paths = [Path('/etc/slurm')]
    matches = re.findall(r'(?im)^SlurmdSpoolDir = (.+)\n', query['filtered_bytes'])
    discovery = {'raw_slurmd_spool_values': matches, 'cache_path_derived': False}
    if query['returncode'] == 0 and len(matches) == 1:
        value = Path(matches[0])
        if value.is_absolute() and '..' not in value.parts and not any(ord(char) < 32 for char in matches[0]):
            paths.append(value / 'conf-cache')
            discovery.update(cache_path_derived=True, observed_slurmd_spool_dir=str(value))
    # For example, an observed /var/spool/slurmd adds only /var/spool/slurmd/conf-cache/{slurm,gres,cgroup}.conf.
    unique = list(dict.fromkeys(paths))
    return discovery, [configuration_file(directory / (kind + '.conf'), kind)
        for directory in unique for kind in ('slurm', 'gres', 'cgroup')]


def own_process():
    """Capture this process's UID fields, birth, affinity and own kernel hierarchy without inspecting another process."""
    status, raw_stat, cgroup = [Path('/proc/self/' + name).read_text() for name in ('status', 'stat', 'cgroup')]
    match = re.search(r'^Uid:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$', status, re.M)
    require(match is not None, 'own process UID fields unavailable')
    fields = raw_stat[raw_stat.rfind(')') + 2:].split()
    return {'hostname': socket.gethostname().split('.')[0], 'pid': os.getpid(), 'uid': os.getuid(),
        'effective_uid': os.geteuid(), 'uid_fields': [int(value) for value in match.groups()],
        'parent_pid': int(fields[1]), 'process_start_time': fields[19], 'raw_proc_stat': raw_stat,
        'cgroup': cgroup, 'cpu_affinity': sorted(os.sched_getaffinity(0))}


def scheduler_field(raw, name):
    """Keep each required owned scheduler field exactly once."""
    values = re.findall(r'(?:^|\s)' + re.escape(name) + r'=([^\s]+)', raw)
    require(len(values) == 1, 'missing or repeated owned scheduler field: ' + name)
    return values[0]


def resources(raw, name):
    """Read exact resource keys without silently collapsing duplicate values."""
    pairs = [part.split('=', 1) for part in scheduler_field(raw, name).split(',')]
    require(all(len(pair) == 2 for pair in pairs) and len({pair[0] for pair in pairs}) == len(pairs),
            'invalid or duplicate observed allocation resource')
    return dict(pairs)


def validate_allocation(record):
    """Bind only this actual zero-GPU CPU allocation, allowing its recorded one or two simultaneous threads."""
    job, step, process = record['job_id'], record['step_id'], record['process']
    raw_stat = process['raw_proc_stat']
    fields = raw_stat[raw_stat.rfind(')') + 2:].split()
    require(re.fullmatch(r'[1-9][0-9]*', job) and re.fullmatch(r'[0-9]+', step)
        and record['environment'].get('SLURM_JOB_ID') == job and record['environment'].get('SLURM_STEP_ID') == step
        and record['environment'].get('SLURM_CPUS_PER_TASK') == '1'
        and raw_stat.split(' ', 1)[0] == str(process['pid']) and fields[1] == str(process['parent_pid'])
        and fields[19] == process['process_start_time'], 'own process birth or scheduler environment changed')
    require(process['hostname'] == 'jaguar03' and process['uid'] == process['effective_uid'] == OWNER['uid']
        and process['uid_fields'] == [OWNER['uid']] * 4 and type(process['pid']) is int and process['pid'] > 0
        and type(process['parent_pid']) is int and process['parent_pid'] > 0 and process['parent_pid'] != process['pid']
        and re.fullmatch(r'[1-9][0-9]*', process['process_start_time'])
        and process['cgroup'] == '0::/system.slice/slurmstepd.scope/job_' + job + '/step_' + step + '/user/task_0\n'
        and len(process['cpu_affinity']) in {1, 2} and len(set(process['cpu_affinity'])) == len(process['cpu_affinity']),
        'actual observation process ownership or numeric-step hierarchy differs')
    row, task = record['job_query']['stdout'], record['step_query']['stdout']
    for name, command in [('job_query', ['scontrol', 'show', 'job', '-d', job]),
            ('step_query', ['scontrol', 'show', 'step', '-d', job + '.' + step])]:
        query = record[name]
        require(query['command'] == command and type(query['returncode']) is int and query['returncode'] == 0
            and query['stderr'] == '', 'owned scheduler observation changed')
    expected = {'JobId': job, 'UserId': 'sl5nw(2618919)', 'JobState': 'RUNNING', 'NodeList': 'jaguar03',
        'Account': 'sl5nw_base', 'Partition': 'gpu', 'QOS': 'csresnolim', 'NumNodes': '1', 'NumTasks': '1',
        'CPUs/Task': '1', 'TimeLimit': '00:05:00', 'Requeue': '0'}
    require(all(scheduler_field(row, name) == value for name, value in expected.items()), 'owned CPU job shape changed')
    require(scheduler_field(row, 'Reservation') == record['held_identifier_binding']['reservation'],
            'actual owned reservation differs from the returned held job binding')
    require(scheduler_field(task, 'StepId') == job + '.' + step and scheduler_field(task, 'State') == 'RUNNING'
        and scheduler_field(task, 'NodeList') == 'jaguar03'
        and scheduler_field(task, 'UserId') in {'2618919', 'sl5nw(2618919)'}, 'owned numeric CPU step changed')
    job_resources, step_resources = resources(row, 'AllocTRES'), resources(task, 'TRES')
    require(scheduler_field(row, 'NumCPUs') == job_resources.get('cpu') in {'1', '2'}
        and job_resources.get('mem') == '2G' and job_resources.get('node') == '1'
        and step_resources.get('cpu') in {'1', '2'} and step_resources.get('node') == '1'
        and all(value == '0' for values in (job_resources, step_resources) for key, value in values.items()
                if key == 'gres/gpu' or key.startswith('gres/gpu:')),
        'actual CPU observation allocation exceeds its approved processors, memory or zero-GPU shape')


def run(unit):
    """Preserve actual owned allocation and file observations, including every incomplete failed attempt."""
    plan = verify_static(unit, execution=True)
    require(not any((unit / name).exists() for name in ('observation.json', 'owned_job_ledger_snapshot.txt', 'hardware.json')),
            'preserve the prior observation attempt and prepare a new unit')
    observation = {'schema_version': 1, 'kind': KIND, 'plan_sha256': sha256(ordinary(unit / 'observation_plan.json')),
        'started_utc': datetime.now(timezone.utc).isoformat(), 'source': plan['source'], **ZERO_CREDIT}
    try:
        observation['process'] = own_process()
        job, step = os.environ.get('SLURM_JOB_ID', ''), os.environ.get('SLURM_STEP_ID', '')
        observation.update(job_id=job, step_id=step, environment={name: os.environ[name] for name in
            ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_CPUS_PER_TASK', 'SLURM_CONF', 'SLURM_GPUS_ON_NODE',
             'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES') if name in os.environ})
        require(re.fullmatch(r'[1-9][0-9]*', job) and re.fullmatch(r'[0-9]+', step), 'actual owned numeric job/step required')
        require(observation['process']['uid_fields'] == [OWNER['uid']] * 4 and observation['process']['uid'] == OWNER['uid'],
                'refuse another owner before any scheduler query')
        binding = json.loads(ordinary(unit / 'held_identifier_binding.json'))
        frozen_ledger = ordinary(unit / 'held_owned_job_ledger_snapshot.txt')
        require(binding['kind'] == KIND + '_returned_held_identifier_binding' and type(binding['schema_version']) is int
            and binding['schema_version'] == 1 and binding['cluster'] == 'uva' and binding['owner'] == OWNER
            and binding['job_id'] == job and binding['plan_sha256'] == observation['plan_sha256']
            and binding['owned_ledger_origin'] == plan['owned_job_ledger']
            and binding['owned_ledger_sha256'] == sha256(frozen_ledger)
            and frozen_ledger.decode().splitlines().count(job) == 1
            and re.fullmatch(r'sl5nw_[1-9][0-9]*', binding['reservation'])
            and binding['job_released_by_producer'] is False
            and all(type(binding[name]) is type(value) and binding[name] == value for name, value in ZERO_CREDIT.items()),
            'actual numeric step differs from its immutable returned held identifier')
        observation['held_identifier_binding'] = binding
        ledger = ordinary(Path(plan['owned_job_ledger']))
        require(ledger.decode().splitlines().count(job) == 1, 'returned job ID must occur once in the canonical owned ledger')
        write_once(unit / 'owned_job_ledger_snapshot.txt', ledger)
        progress(unit, {'kind': KIND, 'unit_id': plan['unit_id'], 'state': 'running', 'phase': 'owned_configuration_observation',
            'slurm_job_id': job, 'pid': observation['process']['pid'],
            'time': datetime.now(timezone.utc).isoformat(), **ZERO_CREDIT})
        observation['owned_ledger'] = {'origin': plan['owned_job_ledger'], 'sha256': sha256(ledger)}
        for name, command in [('job_query', ['scontrol', 'show', 'job', '-d', job]),
                ('step_query', ['scontrol', 'show', 'step', '-d', job + '.' + step])]:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
            observation[name] = {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
            require(result.returncode == 0 and not result.stderr, 'owned allocation metadata query failed')
        validate_allocation(observation)
        observation['allocation_binding_passed'] = True
        write_once(unit / 'hardware.json', encoded({'kind': KIND, 'hostname': observation['process']['hostname'],
            'pid': observation['process']['pid'], 'slurm_job_id': job, 'gpu_query': '',
            'actual_cpu_affinity': observation['process']['cpu_affinity'], 'allocated_gpu_count': 0, **ZERO_CREDIT}))
        observation['running_configuration'] = configuration_query()
        discovery, files = configuration_candidates(observation['running_configuration'])
        observation.update(cache_discovery=discovery, configuration_files=files)
        observation['completed'] = True
    except Exception as error:
        observation.update(completed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        observation['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_once(unit / 'observation.json', encoded(observation))
        progress(unit, {'kind': KIND, 'unit_id': plan['unit_id'], 'state': 'completed' if observation.get('completed') else 'failed',
            'phase': 'owned_configuration_observation', 'slurm_job_id': observation.get('job_id'),
            'time': observation['finished_utc'], 'completed': observation.get('completed', False), **ZERO_CREDIT})
    return observation


def main():
    """Expose preparation and one admitted metadata collection, with no submission command."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    preparation = actions.add_parser('prepare')
    for name in ('run-dir', 'source-root', 'python'):
        preparation.add_argument('--' + name, type=Path, required=True)
    preparation.add_argument('--source-commit', required=True)
    preparation.add_argument('--unit-id', required=True)
    execution = actions.add_parser('run')
    execution.add_argument('unit', type=Path)
    binding = actions.add_parser('bind-held-identifier')
    binding.add_argument('unit', type=Path)
    binding.add_argument('--job-id', required=True)
    binding.add_argument('--reservation', required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps({'unit': str(prepare(args.run_dir, args.unit_id, args.source_root, args.source_commit, args.python)),
                          'submitted': False}))
    elif args.action == 'bind-held-identifier':
        print(json.dumps(bind_held_submission(args.unit, args.job_id, args.reservation)))
    else:
        print(json.dumps(run(args.unit)))


if __name__ == '__main__':
    main()
