"""Qualify native memory capture with array fixtures, never formal packing work.

Preparation freezes no guessed job ID. The admitted owned compute job makes a
separate immutable ledger/allocation record, then starts two fresh children.
No experiment is submitted, released, cancelled or admitted by this module.
"""

import argparse
import ast
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
import traceback
import uuid


SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import device_memory_limit as capacity
import diagnose_gpu_execution as diagnostic
import measure_selected as measurement
import post_comparison_validation as validation
import train_optimized as training
import validate_training_deployment as deployment

KIND = 'device_memory_capture_component_regression'
CAPTURE_HELPER_COMMIT = '7f5bf87a8400a73afff08098c1422642ecad69f9'
SHAPE = {'nodes': 1, 'ntasks': 1, 'gres': 'gpu:nvidia_rtx_a4500:1', 'cpus_per_task': 8,
         'host_memory_gib': 64, 'time_limit_minutes': 20, 'hostname': 'jaguar03',
         'account': 'sl5nw_base', 'partition': 'gpu', 'qos': 'csresnolim'}
FIXTURE = {'allocation_dtype': 'uint8', 'allocation_alignment_bytes': 256,
           'failure_request': 'smallest 256-byte multiple strictly greater than measured current-context capacity',
           'repeat_elements': 4096, 'repeat_rounds': 7, 'synchronized_calls_per_round': 10,
           'environment_transitions': 0, 'learner_updates': 0, 'formal_packing_evidence': False}
HELPERS = measurement.HELPERS


def write(path, value):
    """Save an immutable original record; retries need a new unit."""
    measurement.write_once(Path(path), measurement.json_bytes(value))


def read(path):
    """Read metadata with duplicate-key rejection and no numerical import."""
    return capacity.json_document(binary(path))


def binary(path):
    """Read an ordinary artifact at its exact path, refusing links to another unit's bytes."""
    path = Path(path)
    capacity.require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
                     'diagnostic artifacts must be ordinary files at their exact absolute paths')
    return path.read_bytes()


def next_request(total):
    """Derive a diagnostic request from an actual context measurement, never from an assumed card size."""
    capacity.integer(total, 'measured current-context capacity', 1)
    result = (total // FIXTURE['allocation_alignment_bytes'] + 1) * FIXTURE['allocation_alignment_bytes']
    capacity.require(result < 2 ** 64, 'diagnostic request exceeds CUDA size range')
    return result


def require_capture_version(root, profile=None):
    """Refuse published helpers that predate the requested-stop sampler contract."""
    path = Path(root) / '10_jax_exploration_platform/scripts/atari_pixel_campaign/device_memory_limit.py'
    values = [ast.literal_eval(node.value) for node in ast.parse(path.read_bytes()).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'PROTOCOL' for target in node.targets)]
    capacity.require(len(values) == 1 and type(values[0].get('schema_version')) is int and values[0]['schema_version'] == 2
        and values[0]['producer_capability'] == 'native_gpu_bfc_same_context_cuda_and_requested_sampler_stop_v2',
        'the diagnostic requires the separately reviewed version-two capture helper publication')
    if profile is not None:
        capacity.selected_ownership_profile({'ownership_profile': profile,
            'device_memory_limit_protocol': {'allocation_kind': 'slurm', 'cluster': 'uva'}})
        expected = {'UVA_CGROUP_V2_OWNERSHIP_PROFILE': profile,
            'UVA_CGROUP_V2_OWNERSHIP_CAPABILITY': capacity.UVA_CGROUP_V2_OWNERSHIP_CAPABILITY}
        for name, value in expected.items():
            found = [ast.literal_eval(node.value) for node in ast.parse(path.read_bytes()).body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
            capacity.require(found == [value], 'the capture source lacks the explicit version-two ownership capability: ' + name)


def prepare(args):
    """Freeze standard records and source bytes before any job ID or GPU UUID is known."""
    run = args.run_dir.resolve(strict=True)
    capacity.require(args.unit_id and Path(args.unit_id).name == args.unit_id and args.unit_id not in {'.', '..'}, 'plain diagnostic unit name required')
    source = args.source_root.resolve(strict=True)
    core = diagnostic.core_identity(source, args.source_commit)
    runtime = validation.runtime_environment(read(args.runtime_environment))
    assets = read(args.asset_record)
    capacity.require(set(assets) == {'root', 'manifest', 'sha256'}, 'diagnostic needs the original asset manifest identity')
    helper_root = SCRIPTS.parents[2]
    helper_commit = subprocess.check_output(['git', '-C', str(helper_root), 'rev-parse', 'HEAD'], text=True).strip()
    capture_root = args.capture_helper_root.resolve(strict=True)
    deployment.git_root(str(capture_root), CAPTURE_HELPER_COMMIT, validation.Evidence())
    deployment.git_root(str(helper_root), helper_commit, validation.Evidence())
    helpers = training.source_receipt(capture_root, CAPTURE_HELPER_COMMIT,
        ['10_jax_exploration_platform/scripts/atari_pixel_campaign/' + name for name in HELPERS])
    selected_profile = getattr(args, 'ownership_profile', None)
    if selected_profile is not None:
        capacity.require(selected_profile == capacity.UVA_CGROUP_V2_OWNERSHIP_PROFILE['kind'], 'unknown explicit diagnostic ownership profile')
    profile = dict(capacity.UVA_CGROUP_V2_OWNERSHIP_PROFILE) if selected_profile is not None else None
    require_capture_version(capture_root, profile)
    driver_source = training.source_receipt(helper_root, helper_commit,
        ['10_jax_exploration_platform/scripts/atari_pixel_campaign/device_memory_capture_regression.py'])
    unit = run / 'data' / args.unit_id
    plan = {'schema_version': 1, 'kind': KIND, 'source_root': str(source), 'source_commit': args.source_commit,
        'core': core, 'helpers': helpers, 'driver_source': driver_source, 'runtime_environment': runtime, 'asset_record': assets,
        'jaxtari_source_root': str(args.jaxtari_source_root.resolve(strict=True)), 'python': str(args.python.absolute()),
        'campaign_run_dir': str(run), 'unit_id': args.unit_id, 'shape': SHAPE, 'fixture': FIXTURE,
        'job_id': None, 'gpu_uuid': None, 'reservation': None,
        'owned_job_ledger': str(run / 'slurm/submitted_jobids_uva.txt'),
        'allocation_binding': 'created once on owned compute after the exact returned held ID is in the live ledger',
        'environment_transitions': 0, 'learner_updates': 0, 'packing_point_credit': 0, 'case_credit': 0,
        'idea_credit': 0, 'admission_granted': False, 'jobs_submitted_by_producer': False}
    if profile is not None:
        plan['ownership_profile'] = profile
    command = [plan['python'], str(unit / 'code/device_memory_capture_regression.py'), 'run', str(unit)]
    artifacts = {'regression_plan.json': measurement.json_bytes(plan), 'manifest.yaml': measurement.json_bytes(plan),
        'config_resolved.yaml': measurement.json_bytes({'runtime_environment': runtime, 'shape': SHAPE, 'fixture': FIXTURE}),
        'command.txt': (__import__('shlex').join(command) + '\n').encode(),
        'experiment_background.md': ('## Purpose\n\nTest native allocation-failure capture and a fresh small array fixture. '
            'No pixel learner is executed and this produces zero environment transitions, learner updates, packing points, cases or ideas.\n\n'
            '## Key hyperparameters\n\nOne RTX A4500 on jaguar03, eight processors, 64 GiB host memory and twenty minutes. '
            'The first fresh child requests one unsigned-byte device buffer just larger than measured current-context capacity. '
            'The second fresh child runs seven groups of ten synchronized small-array calls.\n\n'
            '## Code and config changes\n\nThe native-capture helper remains unchanged. Diagnostic records have a separate kind '
            'that formal capacity and packing readers must reject. Allocation identity is written separately after the owned job starts; '
            'the frozen plan is never rewritten to insert a guessed identifier.\n\n'
            '## Git state\n\nNumerical source provenance: ' + args.source_commit + '. Helper source: ' + helper_commit + '.\n').encode()}
    for name, contents in artifacts.items():
        measurement.write_once(unit / name, contents)
    for name in HELPERS:
        measurement.write_once(unit / 'code' / name,
            (capture_root / '10_jax_exploration_platform/scripts/atari_pixel_campaign' / name).read_bytes())
    measurement.write_once(unit / 'code/device_memory_capture_regression.py', Path(__file__).read_bytes())
    write(unit / 'regression_prepared.json', {'plan': capacity.file_receipt(unit / 'regression_plan.json')})
    return unit


def verify_static(unit, *, execution=False):
    """Recheck the unchanged static plan and complete source closure without any GPU or scheduler call."""
    unit = Path(unit).absolute()
    capacity.require(unit.resolve(strict=True) == unit, 'diagnostic unit cannot contain symlink parents')
    plan = read(unit / 'regression_plan.json')
    capacity.require(type(plan.get('schema_version')) is int and plan['schema_version'] == 1 and plan['kind'] == KIND
        and plan['shape'] == SHAPE and plan['fixture'] == FIXTURE and plan['job_id'] is None
        and plan['gpu_uuid'] is None and plan['reservation'] is None
        and unit == Path(plan['campaign_run_dir']) / 'data' / plan['unit_id'], 'unsupported diagnostic-only plan')
    capacity.require(all(type(plan[key]) is int and plan[key] == 0 for key in
        ('environment_transitions', 'learner_updates', 'packing_point_credit', 'case_credit', 'idea_credit'))
        and plan['admission_granted'] is False and plan['jobs_submitted_by_producer'] is False,
        'the array diagnostic must retain zero training, packing and admission credit')
    capacity.require(read(unit / 'regression_prepared.json') == {'plan': capacity.file_receipt(unit / 'regression_plan.json')}, 'static diagnostic plan changed')
    helpers = plan['helpers']
    capacity.require(set(helpers['files']) == {'10_jax_exploration_platform/scripts/atari_pixel_campaign/' + name for name in HELPERS},
                     'diagnostic preparation needs its exact complete published helper roster')
    capacity.require(training.source_receipt(Path(helpers['source_root']), helpers['source_commit'], list(helpers['files'])) == helpers,
                     'diagnostic source receipt changed')
    capacity.require(helpers['source_commit'] == CAPTURE_HELPER_COMMIT, 'the reviewed published capture helper changed')
    if 'ownership_profile' in plan:
        capacity.selected_ownership_profile({'ownership_profile': plan['ownership_profile'],
            'device_memory_limit_protocol': {'allocation_kind': 'slurm', 'cluster': 'uva'}})
    require_capture_version(Path(helpers['source_root']), plan.get('ownership_profile'))
    for relative, expected in helpers['files'].items():
        capacity.require(training.file_sha256(unit / 'code' / Path(relative).name) == expected, 'a staged diagnostic helper changed')
    driver = plan['driver_source']
    capacity.require(training.source_receipt(Path(driver['source_root']), driver['source_commit'], list(driver['files'])) == driver
        and set(driver['files']) == {'10_jax_exploration_platform/scripts/atari_pixel_campaign/device_memory_capture_regression.py'}
        and training.file_sha256(unit / 'code/device_memory_capture_regression.py') == next(iter(driver['files'].values())),
        'the separately published diagnostic driver changed')
    if execution:
        verify_executing_modules(unit)
    capacity.require(diagnostic.core_identity(Path(plan['source_root']), plan['source_commit']) == plan['core'], 'numerical source provenance changed')
    return plan


def verify_executing_modules(unit):
    """Check the actual imported metadata modules before either child can import numerical packages."""
    for name, module in (('device_memory_limit.py', capacity), ('measure_selected.py', measurement),
            ('diagnose_gpu_execution.py', diagnostic), ('post_comparison_validation.py', validation),
            ('train_optimized.py', training), ('validate_training_deployment.py', deployment)):
        capacity.require(Path(module.__file__).resolve(strict=True) == unit / 'code' / name,
                         'execute the staged diagnostic helper closure: ' + name)


def validate_resolution(value):
    """Rebuild the original GRES-index to explicit File to proc-UUID association using only captured raw metadata."""
    job, environment = value['job'], value['environment']
    selector = re.findall(r'\bGRES=gpu:nvidia_rtx_a4500:1\(IDX:([0-9]+)\)', job)
    capacity.require(len(selector) == 1 and environment.get('SLURM_JOB_GPUS') == environment.get('SLURM_STEP_GPUS') == selector[0], 'one actual owned A4500 GRES index required')
    configuration = value['gres_file']
    capacity.require(configuration['path'] == '/etc/slurm/gres.conf' and configuration['owner_uid'] == 0
        and type(configuration['mode']) is int and configuration['mode'] & 0o022 == 0
        and configuration['regular_file'] is True and configuration['symlink'] is False
        and hashlib.sha256(configuration['raw'].encode()).hexdigest() == configuration['sha256'],
        'explicit root-owned UVA GRES file required')
    files = []
    for line in configuration['raw'].splitlines():
        pairs = [part.split('=', 1) for part in line.split('#', 1)[0].split()]
        if not pairs:
            continue
        capacity.require(all(len(pair) == 2 for pair in pairs) and len({key.lower() for key, _ in pairs}) == len(pairs), 'ambiguous GRES row')
        values = {key.lower(): value for key, value in pairs}
        if 'name' not in values:
            capacity.require(set(values) <= {'autodetect', 'nodename'}, 'unsupported untyped GRES row')
            continue
        capacity.require(values.get('name') == 'gpu' and values.get('type') == 'nvidia_rtx_a4500'
                         and values.get('nodename', 'jaguar03') == 'jaguar03' and 'file' in values, 'explicit A4500 file roster required')
        files.extend(capacity.expand_devices(values['file']))
    capacity.require(files and len(set(files)) == len(files) and files == sorted(files, key=lambda value: int(value.removeprefix('/dev/nvidia')))
                     and int(selector[0]) < len(files), 'ambiguous GRES file order')
    minor = files[int(selector[0])].removeprefix('/dev/nvidia')
    matches = []
    capacity.require(value['proc_information'] and len({row['path'] for row in value['proc_information']}) == len(value['proc_information']),
                     'missing or duplicate captured proc mapping')
    for row in value['proc_information']:
        path, raw = Path(row['path']), row['raw']
        capacity.require(path.parent.parent == Path('/proc/driver/nvidia/gpus') and path.name == 'information'
            and hashlib.sha256(raw.encode()).hexdigest() == row['sha256'], 'invalid captured proc information')
        if re.findall(r'^Device Minor:\s*([0-9]+)\s*$', raw, re.M) == [minor]:
            values = re.findall(r'^GPU UUID:\s*(GPU-[0-9a-fA-F-]+)\s*$', raw, re.M)
            capacity.require(len(values) == 1, 'selected proc GPU has no unique UUID')
            matches.extend(values)
    capacity.require(len(matches) == 1, 'configured device minor has no unique proc UUID')
    return matches[0]


def resolve_uuid(job, environment, evidence):
    """Capture the original metadata before resolving an owned physical card, with no ordinal or minor assumption."""
    evidence.update(job=job, environment=dict(environment), proc_information=[])
    path = Path('/etc/slurm/gres.conf')
    information = path.lstat()
    capacity.require(path.resolve(strict=True) == path, 'GRES configuration cannot contain symlinks')
    raw = path.read_text()
    evidence['gres_file'] = {'path': str(path), 'owner_uid': information.st_uid, 'mode': stat.S_IMODE(information.st_mode),
        'regular_file': stat.S_ISREG(information.st_mode), 'symlink': stat.S_ISLNK(information.st_mode),
        'raw': raw, 'sha256': hashlib.sha256(raw.encode()).hexdigest()}
    for path in sorted(Path('/proc/driver/nvidia/gpus').glob('*/information')):
        raw = path.read_text()
        evidence['proc_information'].append({'path': str(path), 'raw': raw, 'sha256': hashlib.sha256(raw.encode()).hexdigest()})
    result = validate_resolution(evidence)
    evidence['resolved_gpu_uuid'] = result
    return result


def validate_shape(owned, reservation=None):
    """Keep the diagnostic's exact approved job shape separate from the shared one-GPU ownership contract."""
    raw = owned['job']['stdout']
    expected = {'Account': SHAPE['account'], 'Partition': SHAPE['partition'], 'QOS': SHAPE['qos'],
        'NumNodes': '1', 'NumCPUs': '8', 'NumTasks': '1', 'CPUs/Task': '8', 'TimeLimit': '00:20:00'}
    capacity.require(all(capacity.scheduler_field(raw, key) == value for key, value in expected.items()),
                     'diagnostic job changed its approved account, processors, node, task or time limit')
    pairs = [value.split('=', 1) for value in capacity.scheduler_field(raw, 'AllocTRES').split(',')]
    capacity.require(all(len(pair) == 2 for pair in pairs) and len({pair[0] for pair in pairs}) == len(pairs),
                     'duplicate diagnostic job resource')
    resources = dict(pairs)
    typed = {key for key in resources if key.startswith('gres/gpu:')}
    capacity.require(resources.get('cpu') == '8' and resources.get('mem') == '64G' and resources.get('node') == '1'
        and resources.get('gres/gpu') == '1'
        and typed <= {'gres/gpu:nvidia_rtx_a4500'} and all(resources[key] == '1' for key in typed)
        and capacity.scheduler_field(raw, 'TresPerNode') == 'gres/gpu:nvidia_rtx_a4500:1',
        'diagnostic needs exactly eight CPUs, 64 GiB host memory, one node and one allocated A4500')
    actual = capacity.scheduler_field(raw, 'Reservation')
    capacity.require(re.fullmatch(r'sl5nw_[1-9][0-9]*', actual) and (reservation is None or actual == reservation),
                     'diagnostic reservation must be the actual recorded owned allocation reservation')
    return actual


def admit(unit, plan):
    """Bind the returned actual owned allocation to a new sidecar, preserving the frozen static plan bytes."""
    process = capacity.process_identity()
    identifier = os.environ.get('SLURM_JOB_ID', '')
    capacity.require(process['hostname'] == 'jaguar03' and process['uid'] == capacity.OWNERS['uva']['uid']
        and re.fullmatch(r'[1-9][0-9]*', identifier), 'diagnostic admission requires this owned jaguar03 compute job')
    ledger = Path(plan['owned_job_ledger'])
    capacity.require(ledger == Path(plan['campaign_run_dir']) / 'slurm/submitted_jobids_uva.txt', 'diagnostic ledger namespace changed')
    contents = ledger.read_bytes()
    capacity.require(contents.decode().splitlines().count(identifier) == 1, 'exact returned job ID must be recorded once before admission')
    measurement.write_once(unit / 'diagnostic_owned_job_ledger.txt', contents)
    command = ['scontrol', 'show', 'job', '-d', identifier]
    query = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    environment = {name: os.environ[name] for name in ('CUDA_VISIBLE_DEVICES', 'SLURM_CONF', 'SLURM_JOB_ID',
        'SLURM_STEP_ID', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'SLURM_GPUS_ON_NODE', 'SLURM_CPUS_PER_TASK') if name in os.environ}
    bootstrap = {'job_id': identifier, 'process': process, 'environment': environment,
        'query': {'command': command, 'returncode': query.returncode,
        'stdout': query.stdout, 'stderr': query.stderr}, 'ledger_origin': {'path': str(ledger), 'sha256': hashlib.sha256(contents).hexdigest()}}
    write(unit / 'allocation_bootstrap.json', bootstrap)
    if 'ownership_profile' in plan:
        bootstrap_plan = {**plan, 'device_memory_limit_protocol': {**capacity.PROTOCOL,
            'allocation_kind': 'slurm', 'cluster': 'uva',
            'owned_job_ledger': capacity.file_receipt(unit / 'diagnostic_owned_job_ledger.txt')}}
        capacity.capture_v2_ownership(bootstrap_plan, process, identifier, environment.get('SLURM_STEP_ID', ''),
            unit / 'allocation_bootstrap_cgroup_v2.json')
    capacity.require(query.returncode == 0 and capacity.scheduler_field(query.stdout, 'JobId') == identifier
        and capacity.scheduler_field(query.stdout, 'UserId') == 'sl5nw(2618919)'
        and capacity.scheduler_field(query.stdout, 'JobState') == 'RUNNING'
        and capacity.scheduler_field(query.stdout, 'NodeList') == 'jaguar03', 'owned running job identity failed')
    resolution = {}
    try:
        gpu = resolve_uuid(query.stdout, environment, resolution)
    finally:
        # A missing or ambiguous mapping remains a preserved unsupported attempt, never a guessed card.
        write(unit / 'allocation_resolution.json', resolution)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    reservation = capacity.scheduler_field(query.stdout, 'Reservation')
    runtime_plan = {**plan, 'job_id': identifier, 'gpu_uuid': gpu, 'reservation': reservation,
        'hardware_class': 'jaguar03_rtx_a4500',
        'device_memory_limit_protocol': {**capacity.PROTOCOL, 'allocation_kind': 'slurm', 'cluster': 'uva',
            'owned_job_ledger': capacity.file_receipt(unit / 'diagnostic_owned_job_ledger.txt')},
        'device_memory_limit_ledger_source': bootstrap['ledger_origin']}
    owned = capacity.recorded_ownership(runtime_plan, unit, 'allocation_owned_observation.json')
    validate_shape(owned, reservation)
    admission = {'schema_version': 1, 'kind': KIND + '_allocation', 'static_plan': capacity.file_receipt(unit / 'regression_plan.json'),
        'bootstrap': capacity.file_receipt(unit / 'allocation_bootstrap.json'), 'job_id': identifier, 'gpu_uuid': gpu,
        'resolution': capacity.file_receipt(unit / 'allocation_resolution.json'), 'reservation': reservation,
        'runtime_plan': runtime_plan, 'ownership': owned}
    if 'ownership_profile' in plan:
        admission['bootstrap_ownership_profile'] = capacity.file_receipt(unit / 'allocation_bootstrap_cgroup_v2.json')
    write(unit / 'allocation_admission.json', admission)
    return admission


def validate_admission(unit, plan):
    """Reconstruct the second-stage runtime inputs from immutable plan and actual allocation records."""
    value = read(unit / 'allocation_admission.json')
    bootstrap = read(unit / 'allocation_bootstrap.json')
    resolution = read(unit / 'allocation_resolution.json')
    ledger = capacity.file_receipt(unit / 'diagnostic_owned_job_ledger.txt')
    query = bootstrap['query']
    capacity.require(query['command'] == ['scontrol', 'show', 'job', '-d', value['job_id']]
        and type(query['returncode']) is int and query['returncode'] == 0 and query['stderr'] == ''
        and capacity.scheduler_field(query['stdout'], 'JobId') == value['job_id']
        and capacity.scheduler_field(query['stdout'], 'UserId') == 'sl5nw(2618919)'
        and capacity.scheduler_field(query['stdout'], 'JobState') == 'RUNNING'
        and capacity.scheduler_field(query['stdout'], 'NodeList') == 'jaguar03', 'original owned bootstrap query changed')
    capacity.require(type(value.get('schema_version')) is int and value['schema_version'] == 1
        and value['kind'] == KIND + '_allocation' and value['static_plan'] == capacity.file_receipt(unit / 'regression_plan.json')
        and value['bootstrap'] == capacity.file_receipt(unit / 'allocation_bootstrap.json')
        and value['resolution'] == capacity.file_receipt(unit / 'allocation_resolution.json')
        and resolution['environment'] == bootstrap['environment'] and resolution['job'] == bootstrap['query']['stdout']
        and validate_resolution(resolution) == resolution['resolved_gpu_uuid'] == value['gpu_uuid']
        and value['job_id'] == bootstrap['job_id']
        and value['reservation'] == capacity.scheduler_field(bootstrap['query']['stdout'], 'Reservation')
        and bootstrap['ledger_origin'] == {'path': plan['owned_job_ledger'], 'sha256': ledger['sha256']}
        and (unit / 'diagnostic_owned_job_ledger.txt').read_text().splitlines().count(value['job_id']) == 1,
        'diagnostic allocation or frozen owned ledger changed')
    expected = {**plan, 'job_id': value['job_id'], 'gpu_uuid': value['gpu_uuid'], 'reservation': value['reservation'],
        'hardware_class': 'jaguar03_rtx_a4500',
        'device_memory_limit_protocol': {**capacity.PROTOCOL, 'allocation_kind': 'slurm', 'cluster': 'uva', 'owned_job_ledger': ledger},
        'device_memory_limit_ledger_source': bootstrap['ledger_origin']}
    capacity.require(value['runtime_plan'] == expected, 'runtime inputs may add only measured allocation identity to the frozen plan')
    capacity.validate_ownership(value['ownership'], expected)
    if 'ownership_profile' in plan:
        capacity.require(value.get('bootstrap_ownership_profile') == capacity.file_receipt(unit / 'allocation_bootstrap_cgroup_v2.json'),
                         'the original pre-GRES ownership observation changed')
        capacity.validate_v2_ownership(read(unit / 'allocation_bootstrap_cgroup_v2.json'), bootstrap['process'],
            value['job_id'], bootstrap['environment']['SLURM_STEP_ID'])
    validate_shape(value['ownership'], value['reservation'])
    capacity.require(value['ownership']['process'] == bootstrap['process'], 'admission process differs from its original bootstrap')
    capacity.require(all(value['ownership']['environment'].get(name) == original for name, original in
        bootstrap['environment'].items() if name != 'CUDA_VISIBLE_DEVICES'), 'Slurm identity changed while resolving the physical selector')
    return value


def child_plan(unit, phase, admission):
    """Describe one actual allocation-bound fixture without editing the generic pre-submission plan."""
    capacity.require(phase in {'failure', 'repeat'}, 'unknown diagnostic child phase')
    return {'schema_version': 1, 'kind': KIND + '_resolved_child_plan', 'phase': phase,
        'job_id': admission['job_id'], 'gpu_uuid': admission['gpu_uuid'], 'reservation': admission['reservation'],
        'static_plan': admission['static_plan'], 'allocation_admission': capacity.file_receipt(unit / 'allocation_admission.json'),
        'runtime_plan': admission['runtime_plan'], 'expected_parent': admission['ownership']['process'],
        'fixture': FIXTURE, 'environment_transitions': 0, 'learner_updates': 0, 'packing_point_credit': 0,
        'case_credit': 0, 'idea_credit': 0, 'formal_packing_evidence': False}


def context_capacity(device):
    """Read only the already active current CUDA context; missing context has no fallback."""
    paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if line.split() and Path(line.split()[-1]).name.startswith('libcuda.so')})
    capacity.require(len(paths) == 1, 'one actually mapped CUDA driver required')
    driver = ctypes.CDLL(paths[0], mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
    context, ordinal, free, total = ctypes.c_void_p(), ctypes.c_int(), ctypes.c_size_t(), ctypes.c_size_t()
    for name, signature in {'cuCtxGetCurrent': [ctypes.POINTER(ctypes.c_void_p)], 'cuCtxGetDevice': [ctypes.POINTER(ctypes.c_int)],
        'cuCtxSynchronize': [], 'cuMemGetInfo_v2': [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]}.items():
        function = getattr(driver, name)
        function.argtypes, function.restype = signature, ctypes.c_int
    codes = {'cuCtxGetCurrent': driver.cuCtxGetCurrent(ctypes.byref(context)),
        'cuCtxGetDevice': driver.cuCtxGetDevice(ctypes.byref(ordinal)), 'cuCtxSynchronize': driver.cuCtxSynchronize(),
        'cuMemGetInfo_v2': driver.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))}
    capacity.require(all(code == 0 for code in codes.values()) and context.value and ordinal.value == device.local_hardware_id
                     and 0 <= free.value <= total.value and total.value > 0, 'no healthy already active current CUDA context')
    raw_uuid, pci = (ctypes.c_ubyte * 16)(), ctypes.create_string_buffer(32)
    driver.cuDeviceGetUuid_v2.argtypes, driver.cuDeviceGetUuid_v2.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_int
    driver.cuDeviceGetPCIBusId.argtypes, driver.cuDeviceGetPCIBusId.restype = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int], ctypes.c_int
    codes['cuDeviceGetUuid_v2'] = driver.cuDeviceGetUuid_v2(ctypes.byref(raw_uuid), ordinal.value)
    codes['cuDeviceGetPCIBusId'] = driver.cuDeviceGetPCIBusId(pci, len(pci), ordinal.value)
    capacity.require(all(code == 0 for code in codes.values()), 'same-context diagnostic UUID/PCI query failed')
    return {'pid': os.getpid(), 'context': context.value, 'device': ordinal.value, 'return_codes': codes,
            'free_bytes': free.value, 'total_bytes': total.value, 'observed_at_seconds': time.time(),
            'gpu_uuid': 'GPU-' + str(uuid.UUID(bytes=bytes(raw_uuid))), 'pci_address': capacity.normalize_pci(pci.value.decode()),
            'driver_library': capacity.file_receipt(Path(paths[0]).resolve())}


def child(unit, phase):
    """Run one clearly labeled array fixture after source, assets and actual allocation checks."""
    plan = verify_static(unit, execution=True)
    admission = validate_admission(unit, plan)
    capacity.require(admission['static_plan'] == capacity.file_receipt(unit / 'regression_plan.json')
        and admission['job_id'] == os.environ.get('SLURM_JOB_ID')
        and os.getppid() == admission['ownership']['process']['pid'], 'child allocation admission or actual parent changed')
    parent_stat = Path('/proc/' + str(os.getppid()) + '/stat').read_text()
    capacity.require(parent_stat[parent_stat.rfind(')') + 2:].split()[19] == admission['ownership']['process']['process_start_time'],
                     'the recorded diagnostic parent process has been replaced')
    runtime_plan = admission['runtime_plan']
    capacity.require(read(unit / phase / 'child_plan.json') == child_plan(unit, phase, admission),
                     'the allocation-bound immutable child plan changed')
    os.environ['CUDA_VISIBLE_DEVICES'] = admission['gpu_uuid']
    runtime = training.enforce_runtime(plan['runtime_environment'], os.environ)
    directory = unit / phase
    capacity.require(directory.is_dir() and not (directory / 'fixture_identity.json').exists(), 'the parent must create one fresh child directory')
    owned = capacity.recorded_ownership(runtime_plan, directory, 'fixture_owned_observation.json')
    assets = training.verify_assets(plan['asset_record'], os.environ)
    installed = measurement.installed_backend_receipt(training.jaxtari_python_receipt(plan['jaxtari_source_root']), training.installed_gpu_library_receipts())
    hardware = {**owned['process'], 'gpu_uuid': admission['gpu_uuid'], 'runtime_environment': runtime}
    rows = measurement.selected_query(admission['gpu_uuid'], 'uuid,name,memory.total,driver_version,mig.mode.current,memory.used')
    capacity.require(len(rows) == 1 and len(rows[0]) == 6, 'one selected hardware row required')
    gpu, model, total, driver, mig, used = [part.strip() for part in rows[0]]
    capacity.require(gpu == admission['gpu_uuid'] and 'A4500' in model and mig in {'Disabled', '[N/A]'}
        and int(used) < 1024 and not measurement.selected_query(gpu, 'pid', processes=True), 'selected diagnostic card is not free')
    hardware['gpu_query'] = f'{gpu}, {model}, {total} MiB, {driver}'
    write(directory / 'fixture_identity.json', {'kind': KIND + '_child', 'phase': phase, 'ownership': owned,
        'assets': assets, 'installed_backend': installed, 'runtime_environment': runtime, 'hardware': hardware,
        'static_plan': admission['static_plan'], 'environment_transitions': 0})
    capture = capacity.Capture(directory, runtime_plan, hardware, installed, measurement.MemoryObservation,
                               measurement.write_once, measurement.json_bytes)
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        jax.config.update('jax_default_matmul_precision', 'highest')
        devices = jax.devices()
        capacity.require(len(devices) == 1 and devices[0].platform == 'gpu' and 'A4500' in devices[0].device_kind, 'one assigned A4500 required')
        capture.bind(devices[0])
        # A small dynamic convolution loads the actual CUDA/cuDNN path; it is never called a learner update.
        convolution = jax.jit(lambda image, kernel: jax.lax.conv_general_dilated(image, kernel, (1, 1), 'SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')))
        convolution(jnp.ones((1, 16, 16, 8), dtype=jnp.float32), jnp.ones((3, 3, 8, 8), dtype=jnp.float32)).block_until_ready()
        context = context_capacity(devices[0])
        capacity.require(context['gpu_uuid'] == admission['gpu_uuid']
            and context['pci_address'] == capacity.validate_mapping(owned['physical_mapping'], owned, admission['gpu_uuid']),
            'the actual JAX context differs from the configured owned physical card')
        write(directory / 'current_context.json', context)
        if phase == 'failure':
            requested = next_request(context['total_bytes'])
            write(directory / 'requested_fixture.json', {'kind': KIND + '_intentional_failure', 'requested_bytes': requested,
                'context': context, 'dtype': 'uint8', 'environment_transitions': 0, 'formal_packing_evidence': False})
            diagnostic.record_progress(directory, 'diagnostic_array_allocation', requested_bytes=requested)
            try:
                jax.jit(lambda: jnp.zeros((requested,), dtype=jnp.uint8))().block_until_ready()
            except BaseException as error:
                write(directory / 'fixture_exception.json', {'error': repr(error), 'exception_type': type(error).__name__,
                    'traceback': traceback.format_exc(), 'kind': KIND + '_expected_failure', 'environment_transitions': 0})
                try:
                    capture.failure(error)
                except BaseException as capture_error:
                    write(directory / 'capture_unsupported.json', {'error': repr(capture_error), 'traceback': traceback.format_exc(), 'packing_point_credit': 0})
                return 1
            raise RuntimeError('the intentional larger-than-context array unexpectedly completed')
        update = jax.jit(lambda value: value + jnp.uint8(1))
        value = jnp.zeros((FIXTURE['repeat_elements'],), dtype=jnp.uint8)
        for _ in range(2):
            value = update(value).block_until_ready()
        rounds = []
        for index in range(FIXTURE['repeat_rounds']):
            memory = measurement.MemoryObservation(devices[0], admission['gpu_uuid'])
            memory.begin()
            started = time.monotonic()
            for _ in range(FIXTURE['synchronized_calls_per_round']):
                value = update(value).block_until_ready()
            seconds = time.monotonic() - started
            rounds.append({'round': index + 1, 'fixture_calls': 10, 'seconds': seconds, 'memory': memory.finish(), 'environment_transitions': 0})
        expected = (2 + FIXTURE['repeat_rounds'] * FIXTURE['synchronized_calls_per_round']) % 256
        correct = bool(jnp.all(value == jnp.uint8(expected)).block_until_ready().item())
        capacity.require(correct, 'the small array repeat returned an incorrect fixture value')
        shutdown = capture.close()
        validate_repeat_shutdown(shutdown, capture.errors)
        write(directory / 'fixture_result.json', {'schema_version': 1, 'kind': KIND + '_small_repeat',
            'rounds': rounds, 'process': capacity.process_identity(), 'gpu_uuid': admission['gpu_uuid'],
            'context': context, 'captured_driver_image': capacity.captured_driver(context['driver_library'], directory / 'context_driver'),
            'native_backend': capture.native_backend,
            'loaded_gpu_libraries': training.actual_gpu_library_receipt(devices[0].client, installed['gpu_libraries']),
            'fixture_correct': correct, 'expected_array_value': expected,
            'sampler_shutdown': shutdown, 'sampler_errors': list(capture.errors),
            'environment_transitions': 0, 'learner_updates': 0, 'packing_point_credit': 0, 'case_credit': 0, 'idea_credit': 0})
        return 0
    finally:
        capture.close()


def formal_refusals(unit, plan, *, create=False):
    """Demonstrate that even correctly named diagnostic bytes are rejected by formal readers."""
    proof = unit / 'formal_reader_rejection'
    if create:
        proof.mkdir()
    capacity.require(proof.resolve(strict=True) == proof and proof.is_dir(), 'formal refusal evidence must use the exact diagnostic directory')
    candidate = proof / 'device_memory_limit.json'
    expected = {'schema_version': 1, 'kind': KIND, 'environment_transitions': 0, 'formal_packing_evidence': False}
    if create:
        write(candidate, expected)
    capacity.require(read(candidate) == expected, 'the formal rejection fixture changed')
    errors = {}
    try:
        capacity.validate_exclusion(candidate, {'source_commit': plan['source_commit'], 'hardware_class': 'jaguar03_rtx_a4500',
            'runtime_environment': plan['runtime_environment']}, validation.Evidence())
    except (KeyError, ValueError) as error:
        errors['capacity_reader'] = type(error).__name__ + ': ' + str(error)
    else:
        raise ValueError('formal capacity reader accepted unrelated diagnostic fixture bytes')
    try:
        validation.validate_packing({'source_commit': plan['source_commit'], 'hardware_class': 'jaguar03_rtx_a4500',
            'runtime_environment': plan['runtime_environment'], 'packing_records': [str(unit / 'repeat/fixture_result.json')]}, validation.Evidence())
    except (KeyError, ValueError) as error:
        errors['packing_reader'] = type(error).__name__ + ': ' + str(error)
    else:
        raise ValueError('formal packing reader accepted a small array fixture')
    if create:
        write(unit / 'formal_reader_refusals.json', errors)
    else:
        capacity.require(read(unit / 'formal_reader_refusals.json') == errors, 'formal reader refusal results changed')
    return errors


def validate_context(value, process, gpu_uuid, pci):
    """Require the already active context's actual process, UUID, PCI address and exact successful API roster."""
    expected = {'cuCtxGetCurrent', 'cuCtxGetDevice', 'cuCtxSynchronize', 'cuMemGetInfo_v2',
                'cuDeviceGetUuid_v2', 'cuDeviceGetPCIBusId'}
    capacity.require(value['pid'] == process['pid'] and type(value['pid']) is int
        and value['gpu_uuid'] == gpu_uuid and value['pci_address'] == pci
        and set(value['return_codes']) == expected
        and all(type(code) is int and code == 0 for code in value['return_codes'].values()),
        'diagnostic current context identity or health changed')
    capacity.integer(value['context'], 'existing context', 1)
    capacity.integer(value['device'], 'actual CUDA device handle')
    capacity.integer(value['total_bytes'], 'current context capacity', 1)
    capacity.integer(value['free_bytes'], 'current context free bytes')
    capacity.require(value['free_bytes'] <= value['total_bytes'], 'impossible current-context memory')
    validation.positive(value['observed_at_seconds'], 'current-context observation time')


def validate_failure_detail(unit, plan, admission, identity, process, terminal, observation, parsed):
    """Check diagnostic native bytes, original exception, lifetime samples and the unchanged current-context probe."""
    directory = unit / 'failure'
    requested = read(directory / 'requested_fixture.json')
    context = read(directory / 'current_context.json')
    pci = capacity.validate_mapping(identity['ownership']['physical_mapping'], identity['ownership'], admission['gpu_uuid'])
    validate_context(context, process, admission['gpu_uuid'], pci)
    capacity.require(requested['kind'] == KIND + '_intentional_failure' and requested['dtype'] == 'uint8'
        and requested['context'] == context and requested['environment_transitions'] == 0
        and requested['formal_packing_evidence'] is False
        and parsed == observation['native_allocation'] and parsed['requested_bytes'] == requested['requested_bytes']
        == next_request(context['total_bytes']), 'native diagnostic request differs from its measured context capacity')
    capacity.require(type(observation['schema_version']) is int and observation['schema_version'] == 1
        and observation['kind'] == 'instrumented_device_memory_limit_observation'
        and observation['protocol'] == admission['runtime_plan']['device_memory_limit_protocol']
        and observation['source_commit'] == plan['source_commit']
        and all(type(observation[key]) is type(value) and observation[key] == value for key, value in capacity.ZERO_CREDIT.items())
        and observation['original_exception']['exception_type'] in {'JaxRuntimeError', 'XlaRuntimeError'}
        and read(directory / 'fixture_exception.json') == {**observation['original_exception'],
            'kind': KIND + '_expected_failure', 'environment_transitions': 0}, 'original diagnostic exception or zero-credit scope changed')
    check = observation['cuda_allocation_check']
    capacity.require(check['pid'] == process['pid'] and check['gpu_uuid'] == admission['gpu_uuid']
        and check['pci_address'] == pci and check['requested_bytes'] == parsed['requested_bytes']
        and type(check['cuMemAlloc_v2']) is int and check['cuMemAlloc_v2'] == 2 and check['cuMemFree_v2'] is None
        and check['context_created_or_reset'] is False and check['allocator_settings_changed'] is False
        and check['training_retried'] is False and check['driver_library'] == context['driver_library']
        and check['jax_local_hardware_id'] == check['cuda_device_handle'] == check['cuda_device_handle_after'] == context['device'],
        'diagnostic probe changed request, context, driver, physical GPU or allocator')
    capacity.require(set(check['api_return_codes']) == {'cuCtxGetCurrent', 'cuCtxGetDevice', 'cuCtxSynchronize',
            'cuDeviceGetUuid_v2', 'cuDeviceGetPCIBusId'} and set(check['post_api_return_codes'])
        == {'cuCtxGetCurrent', 'cuCtxGetDevice', 'cuCtxSynchronize'}
        and all(type(code) is int and code == 0 for code in [*check['api_return_codes'].values(), *check['post_api_return_codes'].values()])
        and check['context'] == check['context_after'] == context['context'], 'diagnostic probe lost the original healthy context')
    native = observation['native_backend']
    capacity.require(native['pid'] == process['pid'] and native['required_marker'] == 'TF_BFC_MEMORY_DUMP'
        and Path(native['library']['path']).name == 'xla_cuda_plugin.so'
        and hashlib.sha256(native['mapped_image_rows'].encode()).hexdigest() == native['mapped_image_rows_sha256']
        and {line.split()[-1] for line in native['mapped_image_rows'].splitlines() if len(line.split()) >= 6}
        == {native['library']['path']} and observation['loaded_gpu_libraries']['pid'] == process['pid']
        and check['driver_library']['path'] in observation['loaded_gpu_libraries']['loaded_shared_libraries'],
        'diagnostic native backend or driver does not belong to the actual failed process')
    total = capacity.integer(observation['memory_samples'][0]['device']['total_bytes'], 'NVML physical capacity', 1)
    capacity.require(parsed['native_counters']['Limit'] <= context['total_bytes'] <= total
        and observation['sampler_errors'] == [] and len(observation['memory_samples']) >= 2, 'incomplete native or device memory evidence')
    for row in observation['memory_samples']:
        device, allocator = row['device'], row['allocator']
        capacity.require(device['gpu_uuid'] == admission['gpu_uuid'] and device['total_bytes'] == total
            and all(item['pid'] == process['pid'] and type(item['bytes']) is int and 0 <= item['bytes'] <= total
                for item in device['processes'])
            and all(type(device[key]) is int and device[key] >= 0 for key in ('used_bytes', 'free_bytes'))
            and device['used_bytes'] + device['free_bytes'] <= total, 'diagnostic sample is contaminated or has impossible memory')
        capacity.require(all(type(number) is int and number >= 0 for number in allocator.values())
            and allocator['bytes_limit'] == parsed['native_counters']['Limit']
            and 0 <= allocator['bytes_in_use'] <= allocator['peak_bytes_in_use'] <= allocator['bytes_limit'],
            'diagnostic lifetime allocator counters disagree')
    samples = [validation.positive(row['device']['timestamp_seconds'], 'device sample time') for row in observation['memory_samples']]
    captured = validation.positive(observation['pre_probe_capture_time_seconds'], 'original capture time')
    before, after = check['before'], check['after']
    capacity.require(all(type(value['returncode']) is int and value['returncode'] == 0
        and value['total_bytes'] == context['total_bytes'] and type(value['free_bytes']) is int
        and 0 <= value['free_bytes'] <= value['total_bytes'] for value in (before, after))
        and samples == sorted(samples) and context['observed_at_seconds'] <= captured
        and samples[0] <= captured <= before['time_seconds']
        <= check['started_at_seconds'] <= check['finished_at_seconds'] == after['time_seconds'] <= samples[-1]
        <= terminal['finished_at_seconds'], 'diagnostic samples do not cover the original failure and current-context check')
    native_time = re.fullmatch(r'capacity_native_map_GPU_0_bfc\.([1-9][0-9]*)', observation['native_map_original_name'])
    capacity.require(native_time is not None and context['observed_at_seconds'] <= int(native_time[1]) / 1e6 <= captured
        and any(row['allocator']['peak_bytes_in_use'] == parsed['native_counters']['MaxInUse'] for row in observation['memory_samples'])
        and observation['failure_progress']['state'] == 'diagnostic_array_allocation', 'native diagnostic failure is stale or from another operation')


def validate_terminal(terminal, unit, plan, parent, processes):
    """Bind each parent's actual wait and original proc-stat birth to the two distinct fresh child processes."""
    capacity.require([row['phase'] for row in terminal] == ['failure', 'repeat']
        and [row['returncode'] for row in terminal] == [1, 0], 'diagnostic children did not exit in the required order')
    for row, process in zip(terminal, processes):
        raw = row['child_proc_stat']
        birth = raw[raw.rfind(')') + 2:].split()[19]
        capacity.require(type(row['returncode']) is int and row['child_pid'] == process['pid']
            and raw.split(' ', 1)[0] == str(process['pid']) and birth == process['process_start_time']
            and row['child_process_start_time'] == birth and re.fullmatch(r'[1-9][0-9]*', birth)
            and row['wait_completed'] is True and row['timeout_used'] is False and row['parent'] == parent
            and process['parent_pid'] == parent['pid'] and process['pid'] != parent['pid']
            and int(birth) >= int(parent['process_start_time'])
            and row['command'] == [plan['python'], str(unit / 'code/device_memory_capture_regression.py'), 'child', str(unit), row['phase']],
            'diagnostic wait, command, parent birth or exact child birth changed')
        validation.positive(row['finished_at_seconds'], 'diagnostic wait completion time')


def validate_repeat_shutdown(shutdown, errors):
    """Require the fresh successful child's lifetime sampler to finish normally without ordinary sampler errors."""
    capacity.validate_sampler_shutdown(shutdown)
    capacity.require(errors == [], 'the fresh successful fixture lost its lifetime memory sampler')


def verify(unit, *, create_refusals=False):
    """Validate diagnostic scope, raw native failure and distinct successful fixture process without packing credit."""
    plan = verify_static(unit)
    admission = validate_admission(unit, plan)
    failure, repeat = (read(unit / phase / 'fixture_identity.json') for phase in ('failure', 'repeat'))
    for identity in (failure, repeat):
        capacity.validate_ownership(identity['ownership'], admission['runtime_plan'])
        validate_shape(identity['ownership'], admission['reservation'])
        capacity.require(identity['static_plan'] == admission['static_plan'] and identity['environment_transitions'] == 0,
                         'fixture identity or scope changed')
    first, second = (identity['ownership']['process'] for identity in (failure, repeat))
    parent = admission['ownership']['process']
    capacity.require(first['pid'] != second['pid'] and all(process['parent_pid'] == parent['pid']
        and process['pid'] != parent['pid'] and process['uid'] == parent['uid']
        and process['cgroup'] == parent['cgroup'] for process in (first, second)), 'fixtures need fresh children of the owned parent')
    capacity.require(failure['installed_backend'] == repeat['installed_backend'] and failure['assets'] == repeat['assets']
        and deployment.actual_runtime(failure['runtime_environment'], plan['runtime_environment'])
        == deployment.actual_runtime(repeat['runtime_environment'], plan['runtime_environment']), 'fixture source/assets/backend/runtime group changed')
    terminal = read(unit / 'child_exit_records.json')
    validate_terminal(terminal, unit, plan, parent, (first, second))
    observation = read(unit / 'failure/capacity_observation.json')
    parsed = capacity.native_request(observation['original_exception'], binary(unit / 'failure/capacity_failure_stderr.bin'),
        binary(unit / 'failure/capacity_native_memory_map.pb'), observation['failure_allocator_stats'])
    validate_failure_detail(unit, plan, admission, failure, first, terminal[0], observation, parsed)
    requested = read(unit / 'failure/requested_fixture.json')
    capacity.require(parsed['requested_bytes'] == requested['requested_bytes'] == next_request(requested['context']['total_bytes']),
                     'native failure is not the exact measured-capacity diagnostic request')
    for owned in (observation['ownership_before'], observation['ownership_after']):
        capacity.validate_ownership(owned, admission['runtime_plan'])
        validate_shape(owned, admission['reservation'])
        capacity.require(owned['process'] == first and owned['environment'] == failure['ownership']['environment']
            and owned['physical_mapping'] == failure['ownership']['physical_mapping'],
            'native capture belongs to another fixture process or physical assignment')
    capacity.require(capacity.raw_host_counters(observation['host_memory_before'], first['cgroup'])
        == capacity.raw_host_counters(observation['host_memory_after'], first['cgroup'])
        and observation['sampler_errors'] == [], 'host-memory failures or lost native sampling cannot pass the diagnostic')
    capacity.validate_sampler_shutdown(observation['sampler_shutdown'])
    evidence = deployment.DeploymentEvidence()
    capacity.source_capability(admission['runtime_plan'], evidence)
    failed_backend = deployment.backend_identity(failure['installed_backend'], observation['loaded_gpu_libraries'], evidence)
    check = observation['cuda_allocation_check']
    capacity.require(check['pid'] == first['pid'] and check['gpu_uuid'] == admission['gpu_uuid'] and check['cuMemAlloc_v2'] == 2,
                     'the diagnostic did not retain the same-process actual device allocation error')
    capacity.require(check['context'] == check['context_after'] == requested['context']['context']
        and check['before']['total_bytes'] == check['after']['total_bytes'] == requested['context']['total_bytes']
        and all(type(code) is int and code == 0 for code in check['api_return_codes'].values())
        and all(type(code) is int and code == 0 for code in check['post_api_return_codes'].values()), 'diagnostic context or health changed')
    capacity.validate_captured_image(observation['captured_driver_image'], check['driver_library'],
        unit / 'failure/capacity_driver/captured_driver' / Path(check['driver_library']['path']).name,
        first['pid'], evidence, 'captured_mapped_cuda_driver_image')
    native = observation['native_backend']
    shared = capacity.validate_captured_image(native['capture'], native['library'], unit / 'failure/capacity_backend/xla_cuda_plugin.so',
        first['pid'], evidence, 'captured_mapped_cuda_backend_image')
    capacity.require(native['pid'] == first['pid'] and native['package'] == 'jax-cuda12-pjrt'
        and native['version'] == failure['installed_backend']['packages']['jax-cuda12-pjrt']
        and capacity.contains_marker(evidence.physical_file(shared), b'TF_BFC_MEMORY_DUMP'), 'native backend capture changed')
    result = read(unit / 'repeat/fixture_result.json')
    validate_repeat_shutdown(result['sampler_shutdown'], result['sampler_errors'])
    capacity.require(result['kind'] == KIND + '_small_repeat' and result['environment_transitions'] == result['learner_updates'] == 0
        and result['fixture_correct'] is True and result['expected_array_value'] == 72
        and len(result['rounds']) == 7 and [row['round'] for row in result['rounds']] == list(range(1, 8))
        and all(row['fixture_calls'] == 10 and row['environment_transitions'] == 0
            and row['memory']['other_gpu_processes'] == [] for row in result['rounds']), 'fresh fixture is incomplete or contaminated')
    for row in result['rounds']:
        validation.positive(row['seconds'], 'synchronized array fixture time')
    capacity.require(result['process'] == second and result['context']['pid'] == second['pid']
        and result['context']['gpu_uuid'] == result['gpu_uuid'] == admission['gpu_uuid']
        and result['context']['pci_address'] == capacity.validate_mapping(repeat['ownership']['physical_mapping'], repeat['ownership'], admission['gpu_uuid'])
        and all(type(code) is int and code == 0 for code in result['context']['return_codes'].values()), 'fresh repeat context belongs to another process or card')
    validate_context(result['context'], second, admission['gpu_uuid'], check['pci_address'])
    capacity.require(result['context'] == read(unit / 'repeat/current_context.json')
        and result['context']['driver_library'] == check['driver_library'], 'the fresh process changed its captured CUDA driver')
    capacity.require(deployment.backend_identity(repeat['installed_backend'], result['loaded_gpu_libraries'], evidence) == failed_backend,
                     'the fresh process changed its actual backend')
    capacity.validate_captured_image(result['captured_driver_image'], result['context']['driver_library'],
        unit / 'repeat/context_driver/captured_driver' / Path(result['context']['driver_library']['path']).name,
        second['pid'], evidence, 'captured_mapped_cuda_driver_image')
    evidence.recheck_images()
    return {'schema_version': 1, 'kind': KIND + '_verification', 'passed': True, 'environment_transitions': 0,
        'learner_updates': 0, 'packing_point_credit': 0, 'case_credit': 0, 'idea_credit': 0, 'admission_granted': False,
        'formal_packing_evidence': False, 'formal_reader_refusals': formal_refusals(unit, plan, create=create_refusals),
        'limitation': 'array capture/identity/terminal regression only; actual full-size failed doubling and fresh full packing repeat remain mandatory'}


def run(unit):
    """Run the two child fixtures inside an already owned allocation; never submit or release work."""
    plan = verify_static(unit, execution=True)
    capacity.require(str(Path(sys.executable).absolute()) == plan['python'], 'use the recorded diagnostic Python')
    capacity.require(not (unit / 'allocation_admission.json').exists(), 'preserve the prior attempt and choose a new diagnostic unit')
    capacity.require('TF_BFC_MEMORY_DUMP' not in os.environ, 'do not replace an existing native allocator diagnostic setting')
    admission = admit(unit, plan)
    terminal = []
    for phase in ('failure', 'repeat'):
        # Each child has its own fresh interpreter, stderr and native-map prefix.
        command = [plan['python'], str(unit / 'code/device_memory_capture_regression.py'), 'child', str(unit), phase]
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': admission['gpu_uuid']}
        if phase == 'failure':
            environment['TF_BFC_MEMORY_DUMP'] = str(unit / 'failure/capacity_native_map')
        (unit / phase).mkdir()
        write(unit / phase / 'child_plan.json', child_plan(unit, phase, admission))
        stderr_path = unit / phase / 'capacity_child_stderr.bin'
        with stderr_path.open('xb', buffering=0) as stderr:
            process = subprocess.Popen(command, stderr=stderr, env=environment)
            try:
                raw_stat = Path('/proc/' + str(process.pid) + '/stat').read_text()
                birth = raw_stat[raw_stat.rfind(')') + 2:].split()[19]
            finally:
                # Even an unsupported metadata read waits for the exact child and cannot leave it running alone.
                status = process.wait()
        terminal.append({'phase': phase, 'child_pid': process.pid, 'returncode': status,
            'child_proc_stat': raw_stat, 'child_process_start_time': birth,
            'parent': capacity.process_identity(), 'command': command, 'wait_completed': True, 'timeout_used': False,
            'finished_at_seconds': time.time()})
        write(unit / ('child_exit_' + phase + '.json'), terminal[-1])
    write(unit / 'child_exit_records.json', terminal)
    outcome = verify(unit, create_refusals=True)
    write(unit / 'regression_verification.json', outcome)
    return outcome


def main(argv=None):
    """Separate metadata preparation, owned execution and the private admitted child entry."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    preparation = commands.add_parser('prepare')
    for name in ('run-dir', 'source-root', 'capture-helper-root', 'asset-record', 'runtime-environment', 'jaxtari-source-root', 'python'):
        preparation.add_argument('--' + name, type=Path, required=True)
    preparation.add_argument('--source-commit', required=True)
    preparation.add_argument('--unit-id', required=True)
    preparation.add_argument('--ownership-profile', choices=('uva_slurm_cgroup_v2_process_uid_ownership',))
    for action in ('run', 'verify', 'child'):
        command = commands.add_parser(action)
        command.add_argument('unit', type=Path)
        if action == 'child':
            command.add_argument('phase', choices=('failure', 'repeat'))
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        print(json.dumps({'unit': str(prepare(args)), 'submitted': False}))
    elif args.action == 'run':
        print(json.dumps(run(args.unit)))
    elif args.action == 'verify':
        print(json.dumps(verify(args.unit)))
    else:
        raise SystemExit(child(args.unit, args.phase))


if __name__ == '__main__':
    main()
