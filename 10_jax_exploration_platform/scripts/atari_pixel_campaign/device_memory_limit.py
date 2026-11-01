"""Record a failed next doubling separately from completed packing measurements.

This optional protocol never converts a failure into a packing point. Native
allocator evidence and a same-context diagnostic allocation are collected only
after the configured learner has failed. Unsupported evidence remains a failure.
"""

import ctypes
import hashlib
import importlib.util
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import threading
import time
import traceback
import uuid


PROTOCOL = {"schema_version": 1, "kind": "instrumented_device_memory_limit",
            "producer_capability": "native_gpu_bfc_and_same_context_cuda_allocation_v1"}
ZERO_CREDIT = {"completed": False, "measurement_valid": False, "packing_point_credit": 0,
               "case_credit": 0, "idea_credit": 0, "throughput_credit": 0, "admission_granted": False}
OWNERS = {'uva': {'username': 'sl5nw', 'uid': 2618919}, 'aces': {'username': 'u.sl359075', 'uid': 56626}}
CONFIGURATION_PROFILES = {
    'uva': {'schema_version': 1, 'cluster': 'uva', 'node_cache': '/etc/slurm', 'controller_configuration': '/etc/slurm/slurm.conf'},
    'aces': {'schema_version': 1, 'cluster': 'aces', 'node_cache': '/var/spool/slurmd/conf-cache', 'controller_configuration': '/etc/slurm/slurm.conf'}}
FILENAMES = ("measurement_plan.json", "measurement_prepared.json", "hardware.json", "installed_backend.json",
             "asset_receipt.json", "failure.json", "capacity_observation.json", "capacity_child_exit.json",
             "capacity_failure_stderr.bin", "capacity_native_memory_map.pb", "capacity_parent.json", 'capacity_owned_job_ledger.txt')


def require(condition, message):
    """Refuse incomplete evidence instead of guessing its missing fields."""
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    """Keep Boolean values and negative counters out of actual byte and process records."""
    require(type(value) is int and value >= minimum, name + " must be an exact integer")
    return value


def json_document(contents):
    """Reject duplicate object keys in the new protocol's original JSON artifacts."""
    def unique(pairs):
        """Keep each original field exactly once rather than selecting one duplicate."""
        result = {}
        for name, value in pairs:
            require(name not in result, 'duplicate instrumented capacity JSON key: ' + name)
            result[name] = value
        return result
    return json.loads(contents, object_pairs_hook=unique)


def file_receipt(path):
    """Hash one exact ordinary path without accepting a symlink to different evidence."""
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path and path.is_file(), "capacity evidence must use exact ordinary paths")
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def captured_driver(identity, directory):
    """Reuse the reviewed policy-independent shared-driver capture while its exact mapping is alive."""
    path = Path(__file__).resolve().with_name('h100_nvl_slurm_site_validation.py')
    specification = importlib.util.spec_from_file_location('capacity_shared_driver_capture', path)
    helper = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helper)
    directory.mkdir()
    return helper.capture_driver_image({'pid': os.getpid(), 'driver_library': identity['path'],
        'driver_library_sha256': identity['sha256']}, directory)


def capture_backend_image(original, directory):
    """Stream the actually mapped CUDA backend into an ordinary shared file for later CPU readers."""
    directory.mkdir()
    target = directory / 'xla_cuda_plugin.so'
    digest, count = hashlib.sha256(), 0
    with Path(original['path']).open('rb') as source, target.open('xb') as output:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            output.write(block)
            digest.update(block)
            count += len(block)
        require(digest.hexdigest() == original['sha256'] and count > 0, 'mapped CUDA backend changed before shared capture')
        output.flush()
        os.fsync(output.fileno())
    for path in (directory, directory.parent):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {'schema_version': 1, 'kind': 'captured_mapped_cuda_backend_image', 'pid': os.getpid(),
            'original': original, 'captured': {'path': str(target), 'sha256': original['sha256']}, 'bytes': count}


def validate_captured_image(value, original, expected, pid, evidence, kind):
    """Read only shared captured bytes, preserving the unavailable node-local path as original identity."""
    require(type(value.get('schema_version')) is int and value['schema_version'] == 1 and value['kind'] == kind
            and type(value['pid']) is int and value['pid'] == pid and value['original'] == original
            and value['captured'] == {'path': str(expected), 'sha256': original['sha256']}
            and type(value['bytes']) is int and value['bytes'] > 0, 'captured mapped image changed identity, process or shared path')
    if evidence.resolver is None:
        require(expected.resolve(strict=True) == expected and expected.is_file(), 'captured images cannot use symlinks')
    evidence.image(expected, original['sha256'])
    require(evidence.physical_file(expected).stat().st_size == value['bytes'], 'captured mapped image byte count changed')
    return expected


def process_identity():
    """Record this process and its actual birth counter, owner and CPU placement."""
    raw = Path('/proc/self/stat').read_text()
    return {"hostname": socket.gethostname().split('.')[0], "pid": os.getpid(), "uid": os.getuid(),
            "process_start_time": raw[raw.rfind(')') + 2:].split()[19],
            "parent_pid": os.getppid(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cgroup": Path('/proc/self/cgroup').read_text()}


def contains_marker(path, marker):
    """Check a mapped backend capability in bounded chunks without loading its image."""
    previous = b''
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            if marker in previous + block:
                return True
            previous = block[-len(marker):]
    return False


def scheduler_field(text, name):
    """Require a unique exact Slurm field in a captured owned query."""
    values = re.findall(r'(?:^|\s)' + re.escape(name) + r'=([^\s]+)', text)
    require(len(values) == 1, "missing or repeated owned scheduler field: " + name)
    return values[0]


def ownership(plan):
    """Capture only this process and its recorded direct lock or exact owned Slurm allocation."""
    identity = process_identity()
    specification = plan['device_memory_limit_protocol']
    environment = {name: os.environ[name] for name in ('CUDA_VISIBLE_DEVICES', 'SLURM_CONF', 'SLURM_JOB_ID', 'SLURM_STEP_ID',
        'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'SLURM_GPUS_ON_NODE', 'SLURM_CPUS_PER_TASK') if name in os.environ}
    result = {"process": identity, "environment": environment, "kind": specification['allocation_kind']}
    if specification['allocation_kind'] == 'direct':
        # The standing admission context owns the actual lock for this entire observation.
        import fcntl
        require(identity['hostname'] == 'serval02' and plan['gpu_uuid'] == 'GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3'
                and 'SLURM_JOB_ID' not in environment, 'direct capacity evidence requires the authorized card-one host')
        lock = Path('/tmp/sl5nw_serval02_card1.lock')
        with lock.open('a+') as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held = True
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                held = False
        info = lock.stat()
        require(held and info.st_uid == identity['uid'], 'the standing card lock must remain held by this owner')
        result['lock'] = {"path": str(lock), "inode": info.st_ino, "uid": info.st_uid, "held": held}
    else:
        # The ledger is an explicit cluster-specific input; IDs never cross cluster namespaces.
        identifier, step = environment.get('SLURM_JOB_ID', ''), environment.get('SLURM_STEP_ID', '')
        require(re.fullmatch(r'[1-9][0-9]*', identifier) and re.fullmatch(r'(?:[0-9]+|batch)', step),
                'capacity evidence requires an exact job and step')
        ledger = Path(specification['owned_job_ledger']['path'])
        origin = plan['device_memory_limit_ledger_source']
        require(origin['path'] == str(Path(plan['campaign_run_dir']) / 'slurm' / ('submitted_jobids_' + specification['cluster'] + '.txt'))
                and origin['sha256'] == specification['owned_job_ledger']['sha256']
                and file_receipt(ledger) == specification['owned_job_ledger'] and ledger.read_text().splitlines().count(identifier) == 1,
                'the exact job is absent from its frozen cluster ledger')
        result.update(cluster=specification['cluster'], ledger=specification['owned_job_ledger'], job_id=identifier, step_id=step)
        for key, command in (('job', ['scontrol', 'show', 'job', '-d', identifier]),
                             ('step', ['scontrol', 'show', 'step', '-d', identifier + '.' + step]),
                             ('node', ['scontrol', 'show', 'node', '-d', identity['hostname']])):
            query = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
            result[key] = {"command": command, "returncode": query.returncode, "stdout": query.stdout, "stderr": query.stderr}
        result['physical_mapping'] = collect_mapping(environment, plan['gpu_uuid'], specification['cluster'])
    validate_ownership(result, plan)
    return result


def validate_ownership(value, plan):
    """Rebuild owner/job/step/lock checks from raw observations without querying another server."""
    protocol = plan['device_memory_limit_protocol']
    process, environment = value['process'], value['environment']
    owner = OWNERS[protocol['cluster']]
    for key in ('pid', 'uid', 'parent_pid'):
        integer(process[key], key, 1 if key != 'uid' else 0)
    require(isinstance(process['process_start_time'], str) and re.fullmatch(r'[1-9][0-9]*', process['process_start_time'])
            and isinstance(process['cpu_affinity'], list) and process['cpu_affinity']
            and len(set(process['cpu_affinity'])) == len(process['cpu_affinity'])
            and all(type(cpu) is int and cpu >= 0 for cpu in process['cpu_affinity']), 'invalid owned process identity')
    require(environment.get('CUDA_VISIBLE_DEVICES') == plan['gpu_uuid'] and value['kind'] == protocol['allocation_kind'],
            'capacity attempt changed the physical CUDA selector or allocation kind')
    require(process['uid'] == owner['uid'], 'the capacity process is not this campaign cluster owner')
    if value['kind'] == 'direct':
        require(protocol['cluster'] == 'uva' and protocol['owned_job_ledger'] is None
                and process['hostname'] == 'serval02' and plan['gpu_uuid'] == 'GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3'
                and 'SLURM_JOB_ID' not in environment and value['lock']['held'] is True
                and value['lock']['uid'] == process['uid']
                and value['lock']['path'] == '/tmp/sl5nw_serval02_card1.lock', 'direct capacity ownership is not exact')
        integer(value['lock']['inode'], 'standing lock inode', 1)
        return
    require(value['kind'] == 'slurm' and value['cluster'] == protocol['cluster'] in {'uva', 'aces'}
            and value['ledger'] == protocol['owned_job_ledger'], 'capacity allocation cluster or ledger changed')
    job, step = value['job_id'], value['step_id']
    require(re.fullmatch(r'[1-9][0-9]*', job) and re.fullmatch(r'(?:[0-9]+|batch)', step)
            and environment.get('SLURM_JOB_ID') == job and environment.get('SLURM_STEP_ID') == step
            and environment.get('SLURM_GPUS_ON_NODE') == '1'
            and re.search(r'(?:^|/)uid_' + str(owner['uid']) + r'(?:/|$)', process['cgroup'], re.M)
            and re.search(r'(?:^|/)job_' + job + r'(?:/|$)', process['cgroup'], re.M)
            and re.search(r'(?:^|/)step_' + step + r'(?:/|$)', process['cgroup'], re.M), 'capacity job/step/cgroup mismatch')
    for key, arguments in (('job', ['scontrol', 'show', 'job', '-d', job]),
                           ('step', ['scontrol', 'show', 'step', '-d', job + '.' + step]),
                           ('node', ['scontrol', 'show', 'node', '-d', process['hostname']])):
        query = value[key]
        require(query['command'] == arguments and type(query['returncode']) is int and query['returncode'] == 0
                and query['stderr'] == '', 'owned scheduler query failed or changed')
    row, task = value['job']['stdout'], value['step']['stdout']
    require(scheduler_field(row, 'JobId') == job and scheduler_field(row, 'JobState') == 'RUNNING'
            and scheduler_field(row, 'NodeList') == process['hostname'] and scheduler_field(row, 'NumNodes') == '1'
            and scheduler_field(row, 'UserId') == owner['username'] + '(' + str(owner['uid']) + ')',
            'capacity job does not belong to this running single-node owner')
    pairs = [item.split('=', 1) for item in scheduler_field(row, 'AllocTRES').split(',')]
    require(all(len(item) == 2 for item in pairs) and len({item[0] for item in pairs}) == len(pairs), 'duplicate allocation resource')
    resources = dict(pairs)
    require(resources.get('gres/gpu') == '1' and all(number == '1' for key, number in resources.items() if key.startswith('gres/gpu:'))
            and scheduler_field(task, 'StepId') == job + '.' + step and scheduler_field(task, 'State') == 'RUNNING'
            and scheduler_field(task, 'NodeList') == process['hostname']
            and scheduler_field(task, 'UserId') in {str(owner['uid']), owner['username'] + '(' + str(owner['uid']) + ')'},
            'capacity step does not contain exactly one owned GPU')
    pairs = [item.split('=', 1) for item in scheduler_field(task, 'TRES').split(',')]
    require(all(len(pair) == 2 for pair in pairs) and len({pair[0] for pair in pairs}) == len(pairs), 'duplicate step resources')
    task_resources = dict(pairs)
    requested_cpu = environment.get('SLURM_CPUS_PER_TASK', '')
    require(task_resources.get('gres/gpu') == task_resources.get('node') == '1'
            and all(number == '1' for key, number in task_resources.items() if key.startswith('gres/gpu:'))
            and {key for key in task_resources if key.startswith('gres/gpu:')} == {key for key in resources if key.startswith('gres/gpu:')}
            and requested_cpu.isdecimal() and int(requested_cpu) > 0
            and task_resources.get('cpu') == requested_cpu and len(process['cpu_affinity']) == int(requested_cpu),
            'the actual step GPU, node or CPU allocation differs from this owned task')
    validate_mapping(value['physical_mapping'], value, plan['gpu_uuid'])


def collect_mapping(environment, gpu_uuid, cluster):
    """Read explicit root-owned node-cache files and the assigned GPU's hardware association only."""
    query = subprocess.run(['scontrol', 'show', 'config'], capture_output=True, text=True, timeout=20, check=False)
    require(query.returncode == 0, 'running Slurm configuration query failed')
    filtered = ''.join(line for line in query.stdout.splitlines(keepends=True)
        if re.match(r'\s*(?:SLURM_CONF|TaskPlugin|ProctrackType)\s*=', line))
    selected = environment.get('SLURM_CONF', '')
    if not selected:
        matches = re.findall(r'^\s*SLURM_CONF\s*=\s*(\S+)\s*$', filtered, re.M)
        require(len(matches) == 1, 'the active node configuration path is unknown')
        selected = matches[0]
    require(Path(selected).is_absolute() and Path(selected).name == 'slurm.conf', 'an explicit active node-cache path is required')
    directory = Path(selected).parent
    result = {'slurm_conf': selected, 'configuration': [], 'running_configuration': {
        'command': ['scontrol', 'show', 'config'], 'returncode': query.returncode,
        'filtered_bytes': filtered, 'filtered_sha256': hashlib.sha256(filtered.encode()).hexdigest()}}
    result['profile'] = CONFIGURATION_PROFILES[cluster]
    result['directory_chain'] = []
    for path in reversed((directory, *directory.parents)):
        information = path.lstat()
        require(path.resolve(strict=True) == path, 'node cache directory chains cannot contain symlinks')
        result['directory_chain'].append({'path': str(path), 'owner_uid': information.st_uid,
            'mode': stat.S_IMODE(information.st_mode), 'directory': stat.S_ISDIR(information.st_mode),
            'symlink': stat.S_ISLNK(information.st_mode)})
    for name in ('slurm', 'gres', 'cgroup'):
        path = directory / (name + '.conf')
        information = path.lstat()
        require(stat.S_ISREG(information.st_mode) and information.st_uid == 0 and information.st_mode & 0o022 == 0
                and path.resolve(strict=True) == path, 'capacity configuration must be exact root-owned ordinary files')
        raw = path.read_text()
        rows = [line for line in raw.splitlines(keepends=True) if line.strip() and not line.lstrip().startswith('#')]
        if name == 'slurm':
            rows = [line for line in rows if re.match(r'\s*(?:GresTypes|TaskPlugin|ProctrackType|Include)\b', line, re.I)]
        elif name == 'cgroup':
            rows = [line for line in rows if re.match(r'\s*(?:ConstrainDevices|ConstrainCores|Include)\b', line, re.I)]
        filtered = ''.join(rows)
        result['configuration'].append({'path': str(path), 'kind': name, 'owner_uid': information.st_uid,
            'mode': stat.S_IMODE(information.st_mode), 'regular_file': True, 'symlink': False,
            'sha256': hashlib.sha256(raw.encode()).hexdigest(), 'filtered_bytes': filtered,
            'filtered_sha256': hashlib.sha256(filtered.encode()).hexdigest()})
    command = ['nvidia-smi', '-i', gpu_uuid, '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader,nounits']
    query = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    result['hardware_query'] = {'command': command, 'returncode': query.returncode, 'stdout': query.stdout, 'stderr': query.stderr}
    import csv
    rows = list(csv.reader(query.stdout.splitlines()))
    require(query.returncode == 0 and len(rows) == 1 and len(rows[0]) == 2, 'selected hardware query is unavailable')
    selected_uuid, bus = [part.strip() for part in rows[0]]
    require(selected_uuid == gpu_uuid, 'selected hardware query is ambiguous')
    pci = normalize_pci(bus)
    path = Path('/proc/driver/nvidia/gpus') / pci / 'information'
    result['proc_information'] = {'path': str(path), 'raw': path.read_text()}
    minors = re.findall(r'^Device Minor:\s*([0-9]+)\s*$', result['proc_information']['raw'], re.M)
    require(len(minors) == 1, 'the selected proc GPU lacks an exact physical device minor')
    minor = minors[0]
    device = Path('/dev/nvidia' + minor)
    information = device.lstat()
    result['device_file'] = {'path': str(device), 'owner_uid': information.st_uid,
        'character_device': stat.S_ISCHR(information.st_mode), 'symlink': stat.S_ISLNK(information.st_mode),
        'major': os.major(information.st_rdev), 'minor': os.minor(information.st_rdev)}
    # Another present configuration directory is ambiguous until a separate profile resolves it.
    result['other_configuration_paths'] = [{'path': str(Path('/etc/slurm') / (name + '.conf')),
        'exists': (Path('/etc/slurm') / (name + '.conf')).exists()} for name in ('slurm', 'gres', 'cgroup')
        if directory != Path('/etc/slurm')]
    return result


def normalize_pci(value):
    """Normalize an explicitly observed PCI domain without guessing a device ordinal."""
    match = re.fullmatch(r'(?:0000)?([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])', value)
    require(match is not None, 'invalid physical PCI address')
    return match[1].lower()


def expand_devices(value):
    """Expand only bounded explicit decimal device-file ranges in their recorded order."""
    match = re.fullmatch(r'/dev/nvidia\[([0-9]+)-([0-9]+)\]', value)
    if match:
        first, last = map(int, match.groups())
        require(match[1] == str(first) and match[2] == str(last) and first <= last < first + 256, 'invalid explicit GPU file range')
        return ['/dev/nvidia' + str(number) for number in range(first, last + 1)]
    require(re.fullmatch(r'/dev/nvidia(?:0|[1-9][0-9]*)', value), 'unsupported or noncanonical explicit GPU file roster')
    return [value]


def validate_mapping(mapping, owned, gpu_uuid):
    """Join the recorded GRES index through explicit root-owned files to proc, PCI and physical UUID."""
    import csv
    directory = Path(mapping['slurm_conf']).parent
    profile = CONFIGURATION_PROFILES[owned['cluster']]
    require(mapping['profile'] == profile and type(mapping['profile']['schema_version']) is int, 'unreviewed configuration path profile')
    expected = Path(profile['node_cache'])
    chain = mapping['directory_chain']
    require([row['path'] for row in chain] == [str(path) for path in reversed((expected, *expected.parents))]
            and all(type(row['owner_uid']) is int and row['owner_uid'] == 0 and type(row['mode']) is int
                    and row['mode'] & 0o022 == 0 and row['directory'] is True and row['symlink'] is False for row in chain),
            'node cache directory chain must be exact, root-owned and unwritable by other owners')
    others = mapping['other_configuration_paths']
    expected_others = {'/etc/slurm/' + name + '.conf' for name in ('slurm', 'gres', 'cgroup')} if owned['cluster'] == 'aces' else set()
    require(directory == expected and Path(mapping['slurm_conf']).name == 'slurm.conf'
            and len(others) == len(expected_others) and {row['path'] for row in others} == expected_others
            and all(row['exists'] is False for row in others), 'ambiguous active node configuration')
    records = mapping['configuration']
    require(len(records) == 3 and {row['kind'] for row in records} == {'slurm', 'gres', 'cgroup'}
            and len({row['path'] for row in records}) == 3, 'missing or duplicate cached node configuration')
    texts = {}
    for row in records:
        require(row['path'] == str(directory / (row['kind'] + '.conf')) and type(row['owner_uid']) is int and row['owner_uid'] == 0
                and row['regular_file'] is True and row['symlink'] is False and type(row['mode']) is int
                and row['mode'] & 0o022 == 0 and re.fullmatch(r'[0-9a-f]{64}', row['sha256'])
                and hashlib.sha256(row['filtered_bytes'].encode()).hexdigest() == row['filtered_sha256']
                and not re.search(r'(?im)^\s*Include\b', row['filtered_bytes']), 'cached node configuration is untrusted or incomplete')
        texts[row['kind']] = row['filtered_bytes']
    def assignments(text):
        """Reject duplicate keys case-insensitively in selected root-cache settings."""
        result = {}
        for line in text.splitlines():
            match = re.fullmatch(r'\s*([A-Za-z_]+)\s*=\s*([^#]+?)\s*(?:#.*)?', line)
            require(match is not None and match[1].lower() not in result, 'ambiguous root-cache assignment')
            result[match[1].lower()] = match[2]
        return result
    slurm, cgroup = assignments(texts['slurm']), assignments(texts['cgroup'])
    running = mapping['running_configuration']
    require(running['command'] == ['scontrol', 'show', 'config'] and type(running['returncode']) is int and running['returncode'] == 0
            and hashlib.sha256(running['filtered_bytes'].encode()).hexdigest() == running['filtered_sha256'], 'running node configuration changed')
    actual = assignments(running['filtered_bytes'])
    require(set(actual) == {'slurm_conf', 'taskplugin', 'proctracktype'}
            and actual['slurm_conf'] == profile['controller_configuration']
            and all(actual[key] == slurm[key] for key in ('taskplugin', 'proctracktype'))
            and mapping['slurm_conf'] == owned['environment'].get('SLURM_CONF', actual['slurm_conf']), 'node cache and running configuration disagree')
    require('gpu' in slurm.get('grestypes', '').split(',') and slurm.get('taskplugin') == 'task/cgroup,task/affinity'
            and slurm.get('proctracktype') == 'proctrack/cgroup'
            and all(cgroup.get(name, '').lower() == 'yes' for name in ('constraindevices', 'constraincores')),
            'node configuration does not bind GPU and CPU allocation')
    files, types = [], set()
    for line in texts['gres'].splitlines():
        pairs = [part.split('=', 1) for part in line.split('#', 1)[0].split()]
        require(all(len(pair) == 2 for pair in pairs) and len({pair[0].lower() for pair in pairs}) == len(pairs), 'ambiguous cached GRES line')
        row = {key.lower(): value for key, value in pairs}
        require(set(row) <= {'name', 'type', 'file', 'nodename', 'autodetect', 'cpus', 'cores', 'links', 'count'}, 'unreviewed GRES field')
        if 'nodename' in row:
            require(row['nodename'] == owned['process']['hostname'], 'unresolved GRES host selector')
        if 'name' not in row:
            require(set(row) <= {'autodetect', 'nodename'}, 'missing GRES name')
            continue
        require(row['name'] == 'gpu' and row.get('type') and 'file' in row, 'a GPU GRES needs explicit typed File entries')
        expanded = expand_devices(row['file'])
        require(row.get('count', str(len(expanded))) == str(len(expanded)), 'GRES Count differs from its explicit files')
        files.extend(expanded)
        types.add(row['type'])
    require(files and len(set(files)) == len(files) and len(types) == 1
            and files == sorted(files, key=lambda name: int(name.removeprefix('/dev/nvidia'))), 'ambiguous ordered GPU GRES file roster')
    gpu_type = next(iter(types))
    matches = re.findall(r'\bGRES=gpu:' + re.escape(gpu_type) + r':1\(IDX:([0-9]+)\)', owned['job']['stdout'])
    require(len(matches) == 1 and owned['environment'].get('SLURM_JOB_GPUS') == owned['environment'].get('SLURM_STEP_GPUS') == matches[0]
            and int(matches[0]) < len(files), 'owned GRES index differs from its explicit file mapping')
    node = owned['node']['stdout']
    require(scheduler_field(node, 'NodeName') == owned['process']['hostname']
            and re.fullmatch(r'gpu:' + re.escape(gpu_type) + ':' + str(len(files)) + r'(?:\(S:[0-9,-]+\))?', scheduler_field(node, 'Gres')),
            'node GPU roster differs from its explicit cached configuration')
    query = mapping['hardware_query']
    require(query['command'] == ['nvidia-smi', '-i', gpu_uuid, '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader,nounits']
            and type(query['returncode']) is int and query['returncode'] == 0 and query['stderr'] == '', 'assigned hardware query changed')
    rows = list(csv.reader(query['stdout'].splitlines()))
    require(len(rows) == 1 and len(rows[0]) == 2, 'ambiguous assigned hardware query')
    actual, pci = [part.strip() for part in rows[0]]
    proc = mapping['proc_information']
    minors = re.findall(r'^Device Minor:\s*([0-9]+)\s*$', proc['raw'], re.M)
    require(len(minors) == 1, 'the selected proc GPU lacks one exact minor')
    minor = minors[0]
    device = mapping['device_file']
    require(actual == gpu_uuid and minor.isdecimal() and device['path'] == files[int(matches[0])] == '/dev/nvidia' + minor
            and device['owner_uid'] == 0 and device['character_device'] is True and device['symlink'] is False
            and type(device['major']) is int and device['major'] == 195 and type(device['minor']) is int and device['minor'] == int(minor),
            'owned GRES file does not join the assigned physical NVIDIA character device')
    require(proc['path'] == '/proc/driver/nvidia/gpus/' + normalize_pci(pci) + '/information', 'selected proc GPU path changed')
    for key, expected in (('GPU UUID', gpu_uuid), ('Bus Location', normalize_pci(pci)), ('Device Minor', minor)):
        values = re.findall(r'^' + key + r':\s*(\S+)\s*$', proc['raw'], re.M)
        require(values == [expected], 'proc UUID/PCI/minor does not join the owned configured device')
    return normalize_pci(pci)


def host_memory_counters():
    """Capture own memory-controller failure counters so a host OOM cannot become a device boundary."""
    paths = host_counter_paths(Path('/proc/self/cgroup').read_text())
    result = []
    for path in paths:
        require(path.is_file() and path.stat().st_uid == 0, 'own root-owned host-memory counters are unavailable')
        raw = path.read_text()
        if path.name == 'memory.events':
            pairs = [line.split() for line in raw.splitlines()]
            require(all(len(pair) == 2 for pair in pairs) and len({pair[0] for pair in pairs}) == len(pairs), 'invalid host-memory counters')
            counters = {key: int(value) for key, value in pairs if key in {'oom', 'oom_kill', 'max'}}
            require({'oom', 'oom_kill'} <= set(counters), 'host OOM counters are missing')
        else:
            counters = {'failcnt': int(raw.strip())}
        result.append({"path": str(path), "raw": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest(), "counters": counters})
    require(result, 'a supported own host-memory controller is required')
    return result


def host_counter_paths(cgroup):
    """Include the exact task, step and job memory limits without inspecting a foreign job."""
    paths = []
    rows = cgroup.splitlines()
    for row in rows:
        _, controllers, name = row.split(':', 2)
        if controllers == '':
            directory, filename = Path('/sys/fs/cgroup') / name.lstrip('/'), 'memory.events'
        elif 'memory' in controllers.split(','):
            directory, filename = Path('/sys/fs/cgroup/memory') / name.lstrip('/'), 'memory.failcnt'
        else:
            continue
        require('..' not in Path(name).parts, 'invalid own memory-controller path')
        paths.append(directory / filename)
        if any(re.fullmatch(r'job_[1-9][0-9]*', part) for part in directory.parts):
            while not re.fullmatch(r'job_[1-9][0-9]*', directory.name):
                directory = directory.parent
                paths.append(directory / filename)
    require(len(paths) == len(set(paths)), 'duplicate own memory-controller paths')
    return paths


def protobuf_fields(contents):
    """Read bounded protobuf wire fields without importing protobuf or any numerical package."""
    cursor, fields = 0, []
    def varint():
        """Reject truncated or oversized native integers instead of padding missing bytes."""
        nonlocal cursor
        value = 0
        for shift in range(0, 70, 7):
            require(cursor < len(contents), 'truncated native memory-map integer')
            byte = contents[cursor]
            cursor += 1
            value |= (byte & 127) << shift
            if byte < 128:
                require(value < 2 ** 64, 'oversized native memory-map integer')
                return value
        raise ValueError('oversized native memory-map varint')
    while cursor < len(contents):
        tag = varint()
        number, wire = tag >> 3, tag & 7
        require(number > 0 and wire in {0, 1, 2, 5}, 'unsupported native memory-map wire field')
        if wire == 0:
            value = varint()
        else:
            length = varint() if wire == 2 else (8 if wire == 1 else 4)
            require(cursor + length <= len(contents), 'truncated native memory-map field')
            value, cursor = contents[cursor:cursor + length], cursor + length
        fields.append((number, wire, value))
    return fields


def memory_map(contents, actual_allocator):
    """Reconstruct exact failure-time bytes from OpenXLA's native BFC MemoryDump protobuf."""
    fields = protobuf_fields(contents)
    require(all(number in {1, 2, 3, 4, 5} for number, _, _ in fields), 'unsupported native memory-map schema')
    names = [value for number, wire, value in fields if number == 1 and wire == 2]
    stats = [value for number, wire, value in fields if number == 5 and wire == 2]
    require(names == [b'GPU_0_bfc'] and len(stats) == 1, 'native map does not identify exactly the failed GPU allocator')
    values = protobuf_fields(stats[0])
    require(all(number in {1, 2, 3, 4, 5} for number, _, _ in values)
            and len({number for number, _, _ in values}) == len(values), 'duplicate or unsupported native allocator statistics')
    numbers = {number: value for number, wire, value in values if wire == 0}
    in_use, peak = numbers.get(2, 0), numbers.get(3, 0)
    limit = integer(actual_allocator.get('bytes_limit'), 'actual allocator byte limit', 1)
    require(integer(actual_allocator.get('peak_bytes_in_use'), 'actual allocator peak') == peak
            and 0 <= in_use <= peak <= limit, 'failure-time allocator counters do not bind the actual process allocator')
    require(integer(actual_allocator.get('bytes_in_use'), 'current allocator bytes') <= peak,
            'current allocator usage exceeds its lifetime peak')
    chunks = []
    for number, wire, value in fields:
        if number != 3:
            continue
        require(wire == 2, 'native chunk has an unsupported wire type')
        raw = protobuf_fields(value)
        require(all(key in set(range(1, 10)) for key, _, _ in raw)
                and len({key for key, _, _ in raw}) == len(raw), 'duplicate or unsupported native chunk field')
        chunk = {key: item for key, kind, item in raw if kind == 0}
        address, size, used = chunk.get(1, 0), chunk.get(2, 0), chunk.get(8, 0)
        require(0 < address < 2 ** 64 and 0 < size <= limit and used in {0, 1}
                and 0 <= chunk.get(3, 0) <= size, 'impossible native GPU chunk')
        chunks.append((address, size, used))
    chunks.sort()
    require(len({address for address, _, _ in chunks}) == len(chunks)
            and all(address + size <= later[0] for (address, size, _), later in zip(chunks, chunks[1:]))
            and sum(size for _, size, used in chunks if used) == in_use
            and sum(size for _, size, _ in chunks) <= limit, 'native chunk map disagrees with failure-time allocator statistics')
    return {'Limit': limit, 'InUse': in_use, 'MaxInUse': peak,
            'LargestFreeBlock': max((size for _, size, used in chunks if not used), default=0),
            'AllocatedRegionBytes': sum(size for _, size, _ in chunks)}


def native_request(exception, native, native_map, actual_allocator):
    """Bind one original GPU BFC request to its exact native failure-time memory map."""
    text = exception['error']
    values = re.findall(r'Out of memory while trying to allocate ([0-9]+(?:\.[0-9]+)?(?:[KMGT]iB| bytes))', text)
    require(len(values) == 1 and 'RESOURCE_EXHAUSTED' in text, 'no unambiguous original device allocation failure')
    raw = native.decode('utf-8', errors='strict')
    # The integer rounded size is the allocator's actual attempted size; the displayed client request may be rounded text.
    events = list(re.finditer(r'Allocator \(GPU_0_bfc\) ran out of memory trying to allocate ([^\n ]+) \(rounded to ([1-9][0-9]*)\)', raw))
    require(len(events) == 1 and events[0][1] == values[0], 'native GPU allocator event is absent, ambiguous or disagrees')
    counters = memory_map(native_map, actual_allocator)
    requested = int(events[0][2])
    require(0 < requested < 2 ** 64 and requested > counters['Limit'] - counters['InUse'],
            'native map shows fragmentation or an unsupported request, not measured capacity exhaustion')
    return {"requested_bytes": requested, "request_semantics": 'rounded_gpu_bfc_allocation_request',
            "displayed_client_request": values[0], "allocator": 'GPU_0_bfc', "native_counters": counters}


class Capture:
    """Observe the actual failed process while its original GPU state and admission lock still exist."""

    def __init__(self, directory, plan, hardware, installed, sampler_factory, write_artifact, encode_json):
        """Keep only metadata until the unchanged learner exposes its actual initialized JAX device."""
        self.directory, self.plan, self.hardware, self.installed = Path(directory), plan, hardware, installed
        self.sampler_factory, self.device = sampler_factory, None
        self.write_artifact, self.encode_json = write_artifact, encode_json
        self.samples, self.errors, self.stop, self.thread = [], [], threading.Event(), None
        # Actual owner, allocation, configured GRES mapping and host counters precede any JAX import.
        self.owned_before, self.host_before = ownership(self.plan), host_memory_counters()

    def bind(self, device):
        """Bind the initialized device before learner initialization without creating another CUDA context."""
        self.device, self.sampler = device, self.sampler_factory(device, self.plan['gpu_uuid'])
        mapped = ''.join(line + '\n' for line in Path('/proc/self/maps').read_text().splitlines()
                         if line.split() and Path(line.split()[-1]).name == 'xla_cuda_plugin.so')
        paths = sorted({line.split()[-1] for line in mapped.splitlines()})
        require(len(paths) == 1 and contains_marker(paths[0], b'TF_BFC_MEMORY_DUMP'),
                'the actually mapped CUDA backend does not expose native failure-time memory maps')
        package = metadata.distribution('jax-cuda12-pjrt')
        require(Path(package.locate_file('jax_plugins/xla_cuda12/xla_cuda_plugin.so')).resolve() == Path(paths[0]).resolve()
                and package.version == self.installed['packages']['jax-cuda12-pjrt'], 'mapped CUDA backend differs from its installed package')
        original = file_receipt(Path(paths[0]).resolve())
        self.native_backend = {'pid': os.getpid(), 'library': original, 'required_marker': 'TF_BFC_MEMORY_DUMP',
            'package': 'jax-cuda12-pjrt', 'version': package.version, 'mapped_image_rows': mapped,
            'mapped_image_rows_sha256': hashlib.sha256(mapped.encode()).hexdigest(),
            'capture': capture_backend_image(original, self.directory / 'capacity_backend')}
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        """Keep actual device and allocator observations through initialization, warmup and failure."""
        while not self.stop.is_set():
            try:
                self.samples.append({"device": self.sampler.sample(), "allocator": self.device.memory_stats() or {}})
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                self.errors.append(repr(error))
                return
            self.stop.wait(.2)

    def close(self):
        """Finish the lifetime sampler without touching learner state or allocator settings."""
        self.stop.set()
        if self.thread is not None:
            self.thread.join()

    def failure(self, error):
        """Save original evidence before the diagnostic allocation and retain unknown outcomes explicitly."""
        import train_optimized as training
        require(self.device is not None, 'the failed attempt did not bind an initialized GPU')
        original = {"error": repr(error), "exception_type": type(error).__name__, "traceback": traceback.format_exc()}
        # Copy only bytes already emitted by the failed learner, before any diagnostic driver operation.
        native = (self.directory / 'capacity_child_stderr.bin').read_bytes()
        self.write_artifact(self.directory / 'capacity_failure_stderr.bin', native)
        maps = sorted(self.directory.glob('capacity_native_map_GPU_0_bfc.*'))
        require(len(maps) == 1 and re.fullmatch(r'capacity_native_map_GPU_0_bfc\.[0-9]+', maps[0].name),
                'one new native failed-allocation memory map is required')
        native_map = maps[0].read_bytes()
        self.write_artifact(self.directory / 'capacity_native_memory_map.pb', native_map)
        allocator = self.device.memory_stats()
        parsed = native_request(original, native, native_map, allocator)
        self.samples.append({'device': self.sampler.sample(), 'allocator': allocator})
        captured_at = time.time()
        diagnostic = cuda_allocation_check(self.device, parsed['requested_bytes'])
        driver_capture = captured_driver(diagnostic['driver_library'], self.directory / 'capacity_driver')
        self.samples.append({'device': self.sampler.sample(), 'allocator': self.device.memory_stats()})
        self.close()
        require(not self.errors and self.samples, 'capacity memory sampling was incomplete')
        observation = {"schema_version": 1, "kind": 'instrumented_device_memory_limit_observation', **ZERO_CREDIT,
            "protocol": self.plan['device_memory_limit_protocol'], "source_commit": self.plan['source_commit'],
            "original_exception": original, "native_allocation": parsed, "cuda_allocation_check": diagnostic,
            "native_map_original_name": maps[0].name, "failure_allocator_stats": allocator,
            "pre_probe_capture_time_seconds": captured_at,
            "native_backend": self.native_backend,
            'captured_driver_image': driver_capture,
            "ownership_before": self.owned_before, "ownership_after": ownership(self.plan),
            "host_memory_before": self.host_before, "host_memory_after": host_memory_counters(),
            "memory_samples": sorted(self.samples, key=lambda row: row['device']['timestamp_seconds']), "sampler_errors": self.errors,
            "loaded_gpu_libraries": training.actual_gpu_library_receipt(self.device.client, self.installed['gpu_libraries']),
            "failure_progress": json.loads((self.directory / 'progress.json').read_text())}
        self.write_artifact(self.directory / 'capacity_observation.json', self.encode_json(observation))


def cuda_allocation_check(device, requested):
    """Check the exact failed request in the existing CUDA context without clearing or resetting it."""
    integer(requested, 'diagnostic allocation bytes', 1)
    require(requested < 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)), 'the actual allocator request exceeds the CUDA size type')
    paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if line.split() and Path(line.split()[-1]).name.startswith('libcuda.so')})
    require(len(paths) == 1, 'one actually mapped CUDA driver is required')
    driver = ctypes.CDLL(paths[0], mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
    signatures = {'cuCtxGetCurrent': [ctypes.POINTER(ctypes.c_void_p)], 'cuCtxGetDevice': [ctypes.POINTER(ctypes.c_int)],
        'cuDeviceGetUuid_v2': [ctypes.c_void_p, ctypes.c_int], 'cuDeviceGetPCIBusId': [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
        'cuMemGetInfo_v2': [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)],
        'cuMemAlloc_v2': [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t], 'cuMemFree_v2': [ctypes.c_uint64],
        'cuCtxSynchronize': []}
    for name, arguments in signatures.items():
        function = getattr(driver, name)
        function.argtypes, function.restype = arguments, ctypes.c_int
    context, ordinal = ctypes.c_void_p(), ctypes.c_int()
    codes = {'cuCtxGetCurrent': driver.cuCtxGetCurrent(ctypes.byref(context)),
             'cuCtxGetDevice': driver.cuCtxGetDevice(ctypes.byref(ordinal)), 'cuCtxSynchronize': driver.cuCtxSynchronize()}
    require(context.value and all(value == 0 for value in codes.values()) and ordinal.value == device.local_hardware_id,
            'the original JAX CUDA context is absent or unhealthy')
    raw_uuid, pci = (ctypes.c_ubyte * 16)(), ctypes.create_string_buffer(32)
    codes['cuDeviceGetUuid_v2'] = driver.cuDeviceGetUuid_v2(ctypes.byref(raw_uuid), ordinal.value)
    codes['cuDeviceGetPCIBusId'] = driver.cuDeviceGetPCIBusId(pci, len(pci), ordinal.value)
    require(all(value == 0 for value in codes.values()), 'same-context CUDA identity failed')
    def memory():
        """Retain each actual current-context memory query and its numeric result code."""
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        code = driver.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
        require(code == 0, 'current-context CUDA memory query failed')
        return {"returncode": code, "free_bytes": free.value, "total_bytes": total.value, "time_seconds": time.time()}
    before, pointer = memory(), ctypes.c_uint64()
    # The native configured request can itself exceed device capacity; it is never replaced by an invented larger probe.
    started = time.time()
    status = driver.cuMemAlloc_v2(ctypes.byref(pointer), requested)
    free_status = driver.cuMemFree_v2(pointer) if status == 0 else None
    require(status in (0, 2) and (free_status is None or free_status == 0), 'diagnostic allocation returned an unsupported device error')
    require(status != 0 or requested <= before['total_bytes'], 'diagnostic success exceeds reported context capacity')
    current, selected_device = ctypes.c_void_p(), ctypes.c_int()
    post_codes = {'cuCtxGetCurrent': driver.cuCtxGetCurrent(ctypes.byref(current)),
        'cuCtxGetDevice': driver.cuCtxGetDevice(ctypes.byref(selected_device)), 'cuCtxSynchronize': driver.cuCtxSynchronize()}
    require(all(code == 0 for code in post_codes.values()) and current.value == context.value
            and selected_device.value == ordinal.value, 'the original CUDA context changed or became unhealthy')
    after = memory()
    return {"pid": os.getpid(), "context": context.value, "jax_local_hardware_id": device.local_hardware_id,
        "cuda_device_handle": ordinal.value, "gpu_uuid": 'GPU-' + str(uuid.UUID(bytes=bytes(raw_uuid))),
        "pci_address": pci.value.decode().lower(), "driver_library": file_receipt(Path(paths[0]).resolve()),
        "api_return_codes": codes, "requested_bytes": requested, "cuMemAlloc_v2": status, "cuMemFree_v2": free_status,
        'post_api_return_codes': post_codes, 'context_after': current.value, 'cuda_device_handle_after': selected_device.value,
        "started_at_seconds": started, "finished_at_seconds": after['time_seconds'], "before": before, "after": after,
        "context_created_or_reset": False, "allocator_settings_changed": False, "training_retried": False}


def run_child(directory, python):
    """Wait for one exact fresh child and seal separate evidence only after an ordinary failed exit."""
    import measure_selected as selected
    directory = Path(directory).resolve(strict=True)
    plan = json.loads((directory / 'measurement_plan.json').read_text())
    require(str(Path(python).absolute()) == plan['python'], 'capacity parent must use the exact recorded Python')
    selected.verify_prepared(directory, plan)
    require(plan.get('device_memory_limit_protocol', {}).get('producer_capability') == PROTOCOL['producer_capability'],
            'the prepared producer does not admit this instrumented protocol')
    parent = {"schema_version": 1, "kind": 'device_memory_limit_parent', "process": process_identity(),
              "plan": file_receipt(directory / 'measurement_plan.json'), 'started_at_seconds': time.time()}
    selected.write_once(directory / 'capacity_parent.json', selected.json_bytes(parent))
    command = [str(python), str(directory / 'code/measure_selected.py'), '--prepared-unit', str(directory), '--instrumented-child']
    with (directory / 'capacity_child_stderr.bin').open('xb', buffering=0) as stderr:
        require(not list(directory.glob('capacity_native_map*')), 'an instrumented attempt must start without earlier native maps')
        environment = {**os.environ, 'TF_BFC_MEMORY_DUMP': str(directory / 'capacity_native_map')}
        require('TF_BFC_MEMORY_DUMP' not in os.environ, 'do not replace an existing allocator diagnostic setting')
        child = subprocess.Popen(command, stderr=stderr, env=environment)
        # The child records its own birth before CUDA initialization; waitpid proves this exact child exited.
        status = child.wait()
    exit_record = {"schema_version": 1, "kind": 'device_memory_limit_child_exit', "parent": parent['process'],
        "child_pid": child.pid, "command": command, "returncode": status, "wait_completed": True,
        "finished_at_seconds": time.time(), "parent_requested_termination": False, "timeout_used": False}
    selected.write_once(directory / 'capacity_child_exit.json', selected.json_bytes(exit_record))
    if status > 0 and (directory / 'capacity_observation.json').is_file():
        # The raw failed attempt remains failed; the seal makes no training or throughput claim.
        seal = {"schema_version": 1, "kind": 'device_memory_limit_capacity_exclusion', **ZERO_CREDIT,
                "protocol": plan['device_memory_limit_protocol'], "artifacts": {name: file_receipt(directory / name) for name in FILENAMES}}
        selected.write_once(directory / 'device_memory_limit.json', selected.json_bytes(seal))
    return status


def protocol(value):
    """Require an explicit new capability; historical generic failures have no conversion path."""
    require(isinstance(value, dict) and set(value) == set(PROTOCOL) | {'allocation_kind', 'cluster', 'owned_job_ledger'}
            and type(value['schema_version']) is int and all(value[key] == expected for key, expected in PROTOCOL.items()),
            'unsupported instrumented device-memory-limit producer capability')
    require(value['allocation_kind'] in {'direct', 'slurm'} and value['cluster'] in {'uva', 'aces'},
            'the instrumented protocol needs an exact allocation kind and cluster')
    require((value['owned_job_ledger'] is None) == (value['allocation_kind'] == 'direct'), 'Slurm requires a frozen owned-job ledger')
    return value


def raw_host_counters(value, cgroup):
    """Reparse actual host-controller bytes instead of accepting supplied success summaries."""
    require(isinstance(value, list) and value and len({row['path'] for row in value}) == len(value), 'missing or duplicate host counters')
    require({row['path'] for row in value} == {str(path) for path in host_counter_paths(cgroup)},
            'host counters must cover this exact task, step and job')
    result = {}
    for row in value:
        require(hashlib.sha256(row['raw'].encode()).hexdigest() == row['sha256'], 'host counter bytes changed')
        path = Path(row['path'])
        require(path.is_absolute() and '..' not in path.parts and path.name in {'memory.events', 'memory.failcnt'}, 'unknown host-memory counter')
        if path.name == 'memory.events':
            pairs = [line.split() for line in row['raw'].splitlines()]
            require(all(len(pair) == 2 for pair in pairs) and len({pair[0] for pair in pairs}) == len(pairs), 'duplicate host event')
            parsed = {key: int(value) for key, value in pairs if key in {'oom', 'oom_kill', 'max'}}
            require({'oom', 'oom_kill'} <= set(parsed), 'missing host OOM events')
        else:
            parsed = {'failcnt': int(row['raw'].strip())}
        require(parsed == row['counters'] and all(type(number) is int and number >= 0 for number in parsed.values()), 'host counters disagree')
        result[row['path']] = parsed
    return result


def source_capability(plan, evidence):
    """Verify the exact committed capability declaration and every staged producer dependency."""
    import ast
    import validate_training_deployment as deployment
    helpers = plan['helpers']
    deployment.committed_files(helpers, deployment.REQUIRED_HELPERS, evidence)
    relative = deployment.PREFIX + 'device_memory_limit.py'
    require(relative in helpers['files'], 'the optional producer omitted its required instrumented helper')
    root = evidence.git_root(helpers['source_root'], helpers['source_commit'])
    raw = subprocess.check_output(['git', '-C', str(root), 'show', helpers['source_commit'] + ':' + relative])
    declarations = [ast.literal_eval(node.value) for node in ast.parse(raw).body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'PROTOCOL' for target in node.targets)]
    require(declarations == [PROTOCOL], 'the committed producer does not declare this exact capability')
    for relative, expected in helpers['files'].items():
        staged = Path(plan['campaign_run_dir']) / 'data' / plan['unit_id'] / 'code' / Path(relative).name
        require(hashlib.sha256(evidence.file(staged)).hexdigest() == expected, 'a staged instrumented helper changed')


def validate_exclusion(path, specification, evidence):
    """Independently validate a failed capacity exclusion, including raw backend and terminal-child evidence."""
    import post_comparison_validation as validation
    import validate_training_deployment as deployment
    # Use the existing exact portable resolver and backend-image reader without changing original identities.
    proof = deployment.DeploymentEvidence(evidence.resolver)
    path = Path(path)
    require(path.name == 'device_memory_limit.json', 'a capacity exclusion has its own exact filename')
    directory, record = path.parent, json_document(proof.file(path))
    if proof.resolver is None:
        require(path.is_absolute() and path.resolve(strict=True) == path and directory.resolve(strict=True) == directory,
                'capacity unit paths cannot contain symlinks')
    require(set(record) == {'schema_version', 'kind', 'protocol', 'artifacts'} | set(ZERO_CREDIT)
            and type(record['schema_version']) is int and record['schema_version'] == 1
            and record['kind'] == 'device_memory_limit_capacity_exclusion'
            and all(type(record[key]) is type(value) and record[key] == value for key, value in ZERO_CREDIT.items()),
            'a failed attempt must have zero throughput, point, case, idea and admission credit')
    require(set(record['artifacts']) == set(FILENAMES), 'capacity exclusion omitted an original artifact')
    originals = {}
    for name, receipt in record['artifacts'].items():
        original = directory / name
        require(set(receipt) == {'path', 'sha256'}, 'capacity artifact receipts need exact path and hash fields')
        if proof.resolver is None:
            require(original.resolve(strict=True) == original and original.is_file(), 'capacity artifacts cannot contain symlinks')
        require(receipt['path'] == str(original)
                and hashlib.sha256(proof.file(original)).hexdigest() == receipt['sha256'], 'capacity artifact path or bytes changed')
        if name.endswith('.json'):
            originals[name] = json_document(proof.file(original))
    plan, observation, terminal, parent = (originals[name] for name in
        ('measurement_plan.json', 'capacity_observation.json', 'capacity_child_exit.json', 'capacity_parent.json'))
    require(record['protocol'] == protocol(plan.get('device_memory_limit_protocol')) == observation.get('protocol'), 'capacity protocol changed')
    require(directory == Path(plan['campaign_run_dir']) / 'data' / plan['unit_id']
            and originals['measurement_prepared.json'] == {'plan_sha256': hashlib.sha256(proof.file(directory / 'measurement_plan.json')).hexdigest()}
            and plan['mode'] == 'packing-point' and plan['kind'] == 'selected_configuration_measurement'
            and plan['source_commit'] == specification['source_commit'] and plan['hardware_class'] == specification['hardware_class'],
            'capacity exclusion is not this exact fresh prepared packing attempt')
    has_result = ((directory / 'result.json').exists() if proof.resolver is None
                  else str(directory / 'result.json') in proof.resolver.files)
    require(not has_result, 'a capacity exclusion cannot accompany a completed result')
    configuration = validation.configuration(plan['configuration'])
    options = configuration['execution']
    multiple = options['iterations_per_call']
    if options['report_readout'] != 'none':
        multiple = math.lcm(multiple, options['report_every_iterations'])
    updates = ((10 + multiple - 1) // multiple) * multiple
    require(type(plan.get('rounds')) is int and plan['rounds'] == 7
            and type(plan.get('minimum_updates_per_round')) is int and plan['minimum_updates_per_round'] == 10
            and type(plan.get('iterations')) is int and plan['iterations'] == updates
            and plan.get('window_protocol') == 'complete groups/report windows; requested minimum and actual count recorded',
            'the failed next doubling changed its full seven-round complete-group/report protocol')
    require(plan['reference_configuration'] is None and plan['compilation_cache_policy'] == 'disabled'
            and type(plan.get('hypothesis_count_contribution')) is int and plan['hypothesis_count_contribution'] == 0
            and plan.get('selected_by_producer') is False and plan.get('jobs_submitted_by_producer') is False
            and validation.runtime_environment(plan['runtime_environment']) == validation.runtime_environment(specification['runtime_environment']),
            'capacity changed implementation mode, precision or compiler controls')
    source = deployment.git_root(plan['source_root'], plan['source_commit'], proof)
    require(plan['core'] == {**deployment.diagnostic.core_identity(source, plan['source_commit']), 'source_root': plan['source_root']},
            'capacity numerical source changed')
    defaults = deployment.diagnostic.default_configuration(source)
    require(all(set(configuration[key]) == set(defaults[key]) for key in ('learner', 'environment')), 'capacity configuration is incomplete')
    source_capability(plan, proof)
    paired = plan['paired_source']
    require((paired['source_root'], paired['source_commit']) == (plan['source_root'], plan['source_commit']), 'capacity Executor source changed')
    deployment.committed_files(paired, {deployment.PREFIX + name for name in ('paired_benchmark.py', 'ideas.py')}, proof)
    hardware, installed = originals['hardware.json'], originals['installed_backend.json']
    import csv
    rows = list(csv.reader(hardware['gpu_query'].splitlines()))
    require(len(rows) == 1 and len(rows[0]) == 4, 'capacity hardware needs its exact UUID/model/memory/driver query')
    physical_uuid, model, memory, driver = [part.strip() for part in rows[0]]
    size = re.fullmatch(r'([1-9][0-9]*) MiB', memory)
    require(physical_uuid == plan['gpu_uuid'] and size is not None and driver
            and specification['hardware_class'] in validation.CONTEXTS
            and ('H100' in model if specification['hardware_class'] == 'h100'
                 else 'A4500' in model and hardware['hostname'] == 'jaguar03'), 'capacity physical hardware differs from its context')
    nvml_total = int(size[1]) * 2 ** 20
    actual = deployment.actual_runtime(hardware['runtime_environment'], specification['runtime_environment'])
    backend = deployment.backend_identity(installed, observation['loaded_gpu_libraries'], proof)
    assets = deployment.asset_identity(directory, plan, proof)
    seeds = plan['copy_seeds']
    require(len(seeds) == configuration['learner']['n_copies'] and seeds == list(range(seeds[0], seeds[0] + len(seeds)))
            and all(type(seed) is int and 0 <= seed < 2 ** 32 for seed in seeds), 'capacity independent seed prefix changed')
    before, after = observation['ownership_before'], observation['ownership_after']
    require(type(observation.get('schema_version')) is int and observation['schema_version'] == 1
            and observation.get('kind') == 'instrumented_device_memory_limit_observation'
            and observation.get('source_commit') == plan['source_commit']
            and all(type(observation.get(key)) is type(value) and observation[key] == value for key, value in ZERO_CREDIT.items()),
            'the original instrumented observation has an unsupported schema or nonzero credit')
    for value in (before, after):
        validate_ownership(value, plan)
    process = before['process']
    require(process == after['process'] and before['environment'] == after['environment']
            and before.get('lock') == after.get('lock') and before.get('physical_mapping') == after.get('physical_mapping'),
            'capacity ownership changed during failure confirmation')
    for key in ('hostname', 'pid', 'process_start_time'):
        require(hardware[key] == process[key], 'capacity hardware belongs to another process')
    require(observation['loaded_gpu_libraries']['pid'] == process['pid'], 'capacity backend receipt belongs to another process')
    if before['kind'] == 'slurm':
        ledger = plan['device_memory_limit_protocol']['owned_job_ledger']
        origin = plan['device_memory_limit_ledger_source']
        require(ledger['path'] == str(directory / 'capacity_owned_job_ledger.txt')
                and origin['path'] == str(Path(plan['campaign_run_dir']) / 'slurm' / ('submitted_jobids_' + before['cluster'] + '.txt'))
                and origin['sha256'] == ledger['sha256']
                and hashlib.sha256(proof.file(ledger['path'])).hexdigest() == ledger['sha256']
                and proof.file(ledger['path']).decode().splitlines().count(before['job_id']) == 1, 'capacity owned ledger changed')
    else:
        require(plan['device_memory_limit_ledger_source'] is None and proof.file(directory / 'capacity_owned_job_ledger.txt') == b'',
                'a direct capacity attempt cannot claim a Slurm ledger')
    require(raw_host_counters(observation['host_memory_before'], process['cgroup']) == raw_host_counters(observation['host_memory_after'], process['cgroup']),
            'host-memory failures cannot establish a device-memory limit')
    require(terminal['kind'] == 'device_memory_limit_child_exit' and type(terminal['schema_version']) is int and terminal['schema_version'] == 1
            and terminal['child_pid'] == process['pid'] and type(terminal['returncode']) is int and terminal['returncode'] > 0
            and terminal['wait_completed'] is True and terminal['parent_requested_termination'] is False and terminal['timeout_used'] is False,
            'signal, timeout, requested termination or missing wait proof is not device exhaustion')
    require(parent['kind'] == 'device_memory_limit_parent' and type(parent['schema_version']) is int and parent['schema_version'] == 1
            and parent['process'] == terminal['parent'] and parent['process']['pid'] == process['parent_pid']
            and all(parent['process'][key] == process[key] for key in ('hostname', 'uid', 'cgroup', 'cpu_affinity'))
            and parent['plan'] == record['artifacts']['measurement_plan.json']
            and terminal['command'] == [plan['python'], str(directory / 'code/measure_selected.py'), '--prepared-unit', str(directory), '--instrumented-child'],
            'capacity child was not launched and waited by its recorded parent')
    integer(parent['process']['pid'], 'parent PID', 1)
    integer(parent['process']['parent_pid'], 'parent ancestor PID', 1)
    require(parent['process']['pid'] != process['pid'] and parent['process']['parent_pid'] != parent['process']['pid']
            and isinstance(parent['process']['process_start_time'], str)
            and re.fullmatch(r'[1-9][0-9]*', parent['process']['process_start_time']),
            'the parent birth or distinct parent/child chain is invalid')
    failure = originals['failure.json']
    require(all(failure.get(key) == observation['original_exception'][key] for key in ('error', 'exception_type', 'traceback'))
            and failure['source_commit'] == plan['source_commit'] and failure['mode'] == 'packing-point'
            and failure['failure_stage'] == 'producer_exception', 'original failed attempt was replaced or reclassified')
    require(observation['original_exception']['exception_type'] in {'JaxRuntimeError', 'XlaRuntimeError'},
            'an unknown exception is not a supported backend allocation failure')
    parsed = native_request(observation['original_exception'], proof.file(directory / 'capacity_failure_stderr.bin'),
                            proof.file(directory / 'capacity_native_memory_map.pb'), observation['failure_allocator_stats'])
    require(parsed == observation['native_allocation'], 'native allocator summary disagrees with its original bytes')
    native_backend = observation['native_backend']
    require(native_backend['pid'] == process['pid'] and native_backend['required_marker'] == 'TF_BFC_MEMORY_DUMP'
            and Path(native_backend['library']['path']).name == 'xla_cuda_plugin.so'
            and native_backend['package'] == 'jax-cuda12-pjrt' and native_backend['version'] == installed['packages']['jax-cuda12-pjrt']
            and hashlib.sha256(native_backend['mapped_image_rows'].encode()).hexdigest() == native_backend['mapped_image_rows_sha256'],
            'native allocator dump belongs to another mapped backend/package')
    mapped = {line.split()[-1] for line in native_backend['mapped_image_rows'].splitlines() if len(line.split()) >= 6}
    require(mapped == {native_backend['library']['path']}, 'native backend path is absent from its actual mapped-image rows')
    shared_backend = validate_captured_image(native_backend['capture'], native_backend['library'],
        directory / 'capacity_backend/xla_cuda_plugin.so', process['pid'], proof, 'captured_mapped_cuda_backend_image')
    require(contains_marker(proof.physical_file(shared_backend), b'TF_BFC_MEMORY_DUMP'), 'native backend lacks the required capability')
    check = observation['cuda_allocation_check']
    require(check['pid'] == process['pid'] and check['gpu_uuid'] == hardware['gpu_uuid'] == plan['gpu_uuid']
            and check['requested_bytes'] == parsed['requested_bytes'] and check['context_created_or_reset'] is False
            and check['allocator_settings_changed'] is False and check['training_retried'] is False,
            'device confirmation changed process, physical GPU, context, requested bytes or allocator')
    if before['kind'] == 'slurm':
        require(check['pci_address'] == validate_mapping(before['physical_mapping'], before, plan['gpu_uuid']),
                'same-process CUDA PCI does not match the configured owned physical GPU')
    integer(check['context'], 'existing CUDA context', 1)
    require(type(check['jax_local_hardware_id']) is int and check['jax_local_hardware_id'] == check['cuda_device_handle']
            and re.fullmatch(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', check['pci_address'])
            and set(check['api_return_codes']) == {'cuCtxGetCurrent', 'cuCtxGetDevice', 'cuCtxSynchronize', 'cuDeviceGetUuid_v2', 'cuDeviceGetPCIBusId'}
            and all(type(code) is int and code == 0 for code in check['api_return_codes'].values()), 'original CUDA context is unhealthy or unmapped')
    require(check['context_after'] == check['context'] and type(check['context_after']) is int
            and check['cuda_device_handle_after'] == check['cuda_device_handle'] and type(check['cuda_device_handle_after']) is int
            and set(check['post_api_return_codes']) == {'cuCtxGetCurrent', 'cuCtxGetDevice', 'cuCtxSynchronize'}
            and all(type(code) is int and code == 0 for code in check['post_api_return_codes'].values()),
            'the original CUDA context changed or became unhealthy after its diagnostic allocation')
    require(Path(check['driver_library']['path']).name.startswith('libcuda.so'), 'capacity identity needs the actual CUDA driver image')
    require(check['driver_library']['path'] in observation['loaded_gpu_libraries']['loaded_shared_libraries'],
            'captured driver is absent from this process mapped-library record')
    validate_captured_image(observation['captured_driver_image'], check['driver_library'],
        directory / 'capacity_driver/captured_driver' / Path(check['driver_library']['path']).name,
        process['pid'], proof, 'captured_mapped_cuda_driver_image')
    physical_total = None
    for value in (check['before'], check['after']):
        require(type(value['returncode']) is int and value['returncode'] == 0, 'CUDA memory query failed')
        total, free = integer(value['total_bytes'], 'physical memory', 1), integer(value['free_bytes'], 'free physical memory')
        require(free <= total <= nvml_total and (physical_total is None or total == physical_total), 'CUDA context memory changed or exceeds the physical device')
        physical_total = total
    require(type(check['cuMemAlloc_v2']) is int and check['cuMemAlloc_v2'] in (0, 2)
            and ((check['cuMemAlloc_v2'] == 0 and type(check['cuMemFree_v2']) is int and check['cuMemFree_v2'] == 0)
                 or (check['cuMemAlloc_v2'] == 2 and check['cuMemFree_v2'] is None)), 'unsupported diagnostic CUDA allocation result')
    require(check['cuMemAlloc_v2'] != 0 or parsed['requested_bytes'] <= physical_total,
            'diagnostic success exceeds reported context capacity')
    require(parsed['native_counters']['Limit'] <= physical_total, 'native allocator limit exceeds current-context capacity')
    samples = observation['memory_samples']
    require(observation['sampler_errors'] == [] and isinstance(samples, list) and len(samples) >= 2, 'lost capacity sampler evidence')
    for value in samples:
        device, allocator = value['device'], value['allocator']
        require(device['gpu_uuid'] == plan['gpu_uuid'] and device['total_bytes'] == nvml_total
                and all(row['pid'] == process['pid'] and type(row['bytes']) is int and 0 <= row['bytes'] <= nvml_total for row in device['processes']),
                'another process, unknown allocation or different device contaminated capacity')
        require(type(device['used_bytes']) is int and type(device['free_bytes']) is int
                and min(device['used_bytes'], device['free_bytes']) >= 0
                and device['used_bytes'] + device['free_bytes'] <= nvml_total, 'impossible actual device memory')
        require(isinstance(allocator, dict) and all(type(number) is int and number >= 0 for number in allocator.values()), 'invalid actual allocator counters')
        if 'bytes_limit' in allocator:
            require(allocator['bytes_limit'] == parsed['native_counters']['Limit'], 'actual allocator limit changed during the attempt')
        if any(key in allocator for key in ('bytes_in_use', 'peak_bytes_in_use', 'bytes_limit')):
            require(all(key in allocator for key in ('bytes_in_use', 'peak_bytes_in_use', 'bytes_limit'))
                    and 0 <= allocator['bytes_in_use'] <= allocator['peak_bytes_in_use'] <= allocator['bytes_limit'],
                    'actual lifetime allocator samples have impossible usage or peak counters')
    timestamps = [validation.positive(value['device']['timestamp_seconds'], 'actual memory sample time') for value in samples]
    started, finished = (validation.positive(check[key], key) for key in ('started_at_seconds', 'finished_at_seconds'))
    captured = validation.positive(observation['pre_probe_capture_time_seconds'], 'original failure capture time')
    parent_start = validation.positive(parent['started_at_seconds'], 'parent start time')
    terminal_time = validation.positive(terminal['finished_at_seconds'], 'wait completion time')
    require(timestamps == sorted(timestamps) and parent_start <= timestamps[0] <= captured <= check['before']['time_seconds'] <= started
            <= finished == check['after']['time_seconds'] <= timestamps[-1] <= terminal_time,
            'capacity evidence does not cover the original failure and diagnostic interval in order')
    name = re.fullmatch(r'capacity_native_map_GPU_0_bfc\.([1-9][0-9]*)', observation['native_map_original_name'])
    require(name is not None and parent_start <= int(name[1]) / 1e6 <= captured,
            'native failure-time map is absent, stale or from another attempt')
    require(any(row['allocator'].get('peak_bytes_in_use') == parsed['native_counters']['MaxInUse'] for row in samples)
            and any(any(process_row['pid'] == process['pid'] and process_row['bytes'] > 0 for process_row in row['device']['processes'])
                    for row in samples), 'native failure peak has no actual process-memory observation')
    counters = parsed['native_counters']
    allocator_limited = parsed['requested_bytes'] > counters['Limit'] - counters['InUse']
    physically_limited = check['cuMemAlloc_v2'] == 2 and parsed['requested_bytes'] > check['before']['free_bytes']
    require(allocator_limited, 'the exact failed allocation has no supported measured capacity limit')
    # A successful catch-time diagnostic allocation can follow buffer release; it never proves a complete fit.
    outcome = 'physical_device_allocation_failure' if physically_limited else 'configured_gpu_allocator_limit'
    require(observation['failure_progress']['state'] in {'initializing', 'normalization_warmup', 'compile_warmup', 'running'},
            'failure did not occur in the original configured packing work')
    proof.recheck_images()
    evidence.receipts.update(proof.receipts)
    return {"path": str(path), "configuration": configuration, "copies": len(seeds), "copy_seeds": seeds,
        "process": [process[key] for key in ('hostname', 'pid', 'process_start_time')], "hardware": hardware,
        "gpu_uuid": plan['gpu_uuid'], "runtime": actual, "backend": backend, "assets": assets,
        "source_root": plan['source_root'], "campaign_run_dir": plan['campaign_run_dir'], "outcome": outcome,
        'iterations': updates,
        "diagnostic_allocation_proves_full_fit": False, **ZERO_CREDIT}
