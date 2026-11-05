"""Bind explicit native process control to one owned A4500 allocation without admitting profiles."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

PREFIX = '10_jax_exploration_platform/scripts/atari_pixel_campaign/'
DIAGNOSTIC = 'configured_native_nsight_process_diagnostic'
PROFILE = 'selected_configuration_profile'
PROTOCOL = 'configured-allocation-native-process-control-version-1'
HELPERS = ('configured_native_process.py', 'linux_pidfd_process_control.py', 'native_process_evidence.py',
    'direct_tool_descendants.py', 'direct_profile_process.py')
BUDGET = {'schema_version': 1, 'work_seconds': 240, 'cleanup_seconds': 60, 'total_seconds': 300}
MODES = {'observation': 'observe_configured_target_for_zero_credit_diagnostic',
    'profile': 'use_independently_reviewed_configured_target_contract'}


def require(condition, message):
    """Refuse incomplete policy or ownership instead of selecting a weaker process mechanism."""
    if not condition:raise ValueError(message)


def same(left, right):
    """Compare exact JSON types so a Boolean cannot replace a resource or process count."""
    if type(left) is not type(right):return False
    if type(right) is dict:return left.keys() == right.keys() and all(same(left[key], right[key]) for key in right)
    if type(right) is list:return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def adjacent(plan, name):
    """Load only a declared, hash-pinned helper next to this actual policy source."""
    path = Path(__file__).with_name(name + '.py')
    require(path.resolve(strict=True) == path and path.is_file()
        and hashlib.sha256(path.read_bytes()).hexdigest() == plan['helpers']['files'].get(PREFIX + path.name),
        'the configured native helper changed: ' + name)
    key = 'configured_native_' + hashlib.sha256(str(path).encode()).hexdigest()
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules[key] = module
    module = sys.modules[key]
    require(Path(module.__file__).resolve(strict=True) == path, 'a configured native module namespace changed')
    return module


def declaration(plan, qualification=None):
    """Declare observation separately from future use of an actual independently reviewed target."""
    mode = MODES['observation'] if plan.get('kind') == DIAGNOSTIC else MODES['profile']
    return {'native_process_control': {'schema_version': 1,
        'backend': 'linux-x86-64-libc-syscall-pidfd-version-1', 'mode': mode,
        'expected_target_contract': qualification, 'qualification': False, 'python': plan['python'],
        'helper_hashes': {name: plan['helpers']['files'][PREFIX + name]
            for name in ('linux_pidfd_process_control.py', 'native_process_evidence.py')}},
        'configured_native_process_control': {'schema_version': 1, 'protocol': PROTOCOL,
            'hardware_class': 'jaguar03_rtx_a4500', 'standing_gpu_descriptor_held': False,
            'physical_gres_mapping_verified': False, 'kernel_device_enforcement_verified': False,
            'configured_isolation_trust': True,
            'helper_hashes': {name: plan['helpers']['files'][PREFIX + name] for name in HELPERS}}}


def policy(plan):
    """Permit exact configured diagnostic or contract-bound future profile declarations only."""
    require(plan.get('kind') in (DIAGNOSTIC, PROFILE) and plan.get('hardware_class') == 'jaguar03_rtx_a4500'
        and 'direct_binding' not in plan and plan.get('gpu_uuid') is None
        and type(plan.get('maximum_active_seconds')) is int and plan['maximum_active_seconds'] == 240
        and same(plan.get('diagnostic_time_budget'), BUDGET), 'new native allocation work requires its exact 300-second shared budget')
    qualification = plan.get('native_process_control', {}).get('expected_target_contract')
    expected = declaration(plan, qualification)
    require(all(same(plan.get(key), value) for key, value in expected.items()), 'native plan or helper policy differs')
    require(all(type(value) is str and re.fullmatch('[0-9a-f]{64}', value)
        for value in expected['configured_native_process_control']['helper_hashes'].values()), 'exact native helper hashes are required')
    if plan['kind'] == DIAGNOSTIC:
        require(qualification is None and 'configured_isolation' not in plan, 'diagnostic observation cannot inherit scientific admission')
    else:
        require('configured_isolation' in plan and type(qualification) is dict
            and set(qualification) == {'diagnostic_plan', 'diagnostic_result', 'independent_review', 'target_contract'},
            'formal native profiles need an actual independently reviewed allocation target contract')
    return expected['native_process_control']


def optional_policy(plan):
    """Leave old/default plans unchanged and require both declarations for every opt-in plan."""
    selected = 'configured_native_process_control' in plan or 'native_process_control' in plan
    return policy(plan) if selected else None


def configure_plan(plan, qualification_path, review_path, metadata):
    """Bind actual target-result and independent-review bytes during source-only profile preparation."""
    result_ref, review_ref = metadata.receipt(qualification_path), metadata.receipt(review_path)
    result, review = reference_value(result_ref, metadata), reference_value(review_ref, metadata)
    require(plan.get('kind') == PROFILE and result.get('kind') == DIAGNOSTIC
        and review.get('result') == result_ref, 'a formal native declaration needs its distinct actual configured diagnostic')
    references = {'diagnostic_plan': result['plan'], 'diagnostic_result': result_ref,
        'independent_review': review_ref, 'target_contract': result['target_contract']}
    plan.update(declaration(plan, references))
    plan['diagnostic_time_budget'], plan['maximum_active_seconds'] = dict(BUDGET), 240
    observed = reference_value(references['target_contract'], metadata)
    check_contract(plan, observed['contract'], metadata)
    return plan


def reference_value(record, metadata):
    """Read one exact immutable JSON reference without inspecting a saved process identifier."""
    require(type(record) is dict and set(record) == {'path', 'sha256'}
        and type(record['path']) is str and Path(record['path']).is_absolute()
        and type(record['sha256']) is str and re.fullmatch('[0-9a-f]{64}', record['sha256'])
        and metadata.receipt(record['path']) == record, 'configured native evidence changed')
    return metadata.read_json(record['path'])


def check_contract(plan, contract, metadata):
    """Compare future profile startup with genuine same-target diagnostic bytes, never an H100 proof."""
    selected = policy(plan)
    if plan['kind'] == DIAGNOSTIC:return contract
    refs = selected['expected_target_contract']
    diagnostic = reference_value(refs['diagnostic_plan'], metadata)
    result = reference_value(refs['diagnostic_result'], metadata)
    review = reference_value(refs['independent_review'], metadata)
    observed = reference_value(refs['target_contract'], metadata)
    require(diagnostic.get('kind') == DIAGNOSTIC and diagnostic.get('hardware_class') == 'jaguar03_rtx_a4500'
        and diagnostic.get('python') == plan['python'] and result.get('kind') == DIAGNOSTIC
        and result.get('completed') is True and result.get('plan') == refs['diagnostic_plan']
        and result.get('target_contract') == refs['target_contract'] and result.get('profile_credit') == 0
        and result.get('admission_granted') is False and review.get('kind') == 'independent_configured_native_process_review'
        and review.get('approved') is True and review.get('passed') is True
        and review.get('independent_of_producer') is True and review.get('plan') == refs['diagnostic_plan']
        and review.get('result') == refs['diagnostic_result'] and review.get('target_contract') == refs['target_contract']
        and review.get('actual_owned_numeric_step_verified') is True
        and review.get('all_tool_descendants_exited_and_waited') is True
        and review.get('scientific_credit') == 0 and review.get('profile_admission_granted') is False
        and same(observed.get('contract'), contract)
        and all(diagnostic['helpers']['files'].get(PREFIX + name) == plan['helpers']['files'][PREFIX + name]
            for name in HELPERS), 'the actual configured native target is not independently qualified for these exact bytes')
    evidence = adjacent(plan, 'native_process_evidence')
    evidence.validate_contract(observed, diagnostic, metadata)
    return contract


def context(directory, plan, metadata):
    """Require this exact source command, all user identifiers and the owned numeric allocation."""
    policy(plan)
    current = metadata.ProcReader().snapshot(os.getpid(), os.getuid())
    job, step = os.environ.get('SLURM_JOB_ID'), os.environ.get('SLURM_STEP_ID')
    require(type(job) is str and re.fullmatch('[1-9][0-9]*', job) and type(step) is str and step.isdecimal()
        and os.uname().nodename.split('.')[0] == 'jaguar03' and current['uids'] == [os.getuid()] * 4
        and re.search(r'/job_' + job + r'/step_' + step + r'(?:/|$)', current['cgroup'])
        and current['command'][0] == plan['python']
        and len(os.environ.get('SLURM_STEP_GPUS', '').split(',')) == 1 and os.environ.get('SLURM_STEP_GPUS')
        and len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) == 1 and os.environ.get('CUDA_VISIBLE_DEVICES'),
        'native process work requires its actual owned one-GPU numeric step')
    run = Path(plan['campaign_run_dir'])
    for name in ('submitted_jobids.txt', 'submitted_jobids_uva.txt'):
        require((run / 'slurm' / name).read_text().splitlines().count(job) == 1, 'the native allocation is not exactly campaign-owned')
    require(sum(line.split('\t')[0] == job for line in (run / 'slurm/submissions.tsv').read_text().splitlines()) == 1,
        'the native allocation needs one exact owned submission row')
    return current


def producer_command(directory, plan):
    """Return the exact source-bound producer command used by saved role reading."""
    entry = 'diagnose_configured_native_process.py' if plan['kind'] == DIAGNOSTIC else 'profile_selected.py'
    return [plan['python'], '-I', '-S', '-B', str(Path(directory) / 'code' / entry), '--prepared-unit', str(directory)]


def completion_references(plan, role):
    """Declare exact real completion files for each diagnostic or future scientific role."""
    if role == 'producer':
        return ('tool_descendant_scope_terminal.json', 'tool_descendant_scope_restored.json', 'profiler_actual_wait.json')
    if role == 'waiter':return ('capture_actual_launch.json', 'capture_actual_wait.json')
    require(role == 'capture', 'one exact configured native role is required')
    return ('capture_release_observed.json', 'native_capture_payload_timing.json',
        'probe_result.json' if plan['kind'] == DIAGNOSTIC else 'capture_result.json')


def deadline(directory, plan, metadata, owner=None, started=None):
    """Share one actual same-host monotonic origin before any role bootstrap or tool."""
    policy(plan);directory = Path(directory)
    helper = adjacent(plan, 'direct_profile_process')
    path = directory / 'configured_native_time_origin.json'
    if owner is not None:
        require(type(started) in (int, float) and started <= time.monotonic() < started + 240
            and owner['command'] == producer_command(directory, plan), 'the native producer needs its exact original startup clock')
        metadata.publish(path, {'kind': 'configured_native_process_time_origin', 'schema_version': 1,
            'plan': metadata.receipt(directory / 'measurement_plan.json'), 'producer': owner,
            'source': metadata.receipt(directory / 'code' / Path(owner['command'][-3]).name),
            'budget': BUDGET, 'clock_domain': helper.monotonic_clock_domain(), 'started_monotonic': started})
    result = saved_deadline(directory, plan, metadata)
    require(result.origin['clock_domain'] == helper.monotonic_clock_domain(), 'the native role changed its actual host or boot clock')
    result.remaining('work')
    return result


def saved_deadline(directory, plan, metadata):
    """Validate saved origin bytes on owned CPU without querying the target host or a historical PID."""
    policy(plan);directory = Path(directory);path = directory / 'configured_native_time_origin.json'
    helper = adjacent(plan, 'direct_profile_process');record = metadata.read_json(path)
    metadata.validate_snapshot(record['producer'], os.getuid(), record['producer']['cgroup'])
    require(set(record) == {'kind', 'schema_version', 'plan', 'producer', 'source', 'budget', 'clock_domain', 'started_monotonic'}
        and record['kind'] == 'configured_native_process_time_origin' and type(record['schema_version']) is int and record['schema_version'] == 1
        and record['plan'] == metadata.receipt(directory / 'measurement_plan.json') and same(record['budget'], BUDGET)
        and set(record['clock_domain']) == {'hostname', 'boot_id', 'clock', 'implementation'}
        and record['clock_domain']['hostname'] == 'jaguar03' and record['clock_domain']['clock'] == 'monotonic'
        and type(record['clock_domain']['implementation']) is str and record['clock_domain']['implementation']
        and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', record['clock_domain']['boot_id'])
        and record['producer']['command'] == producer_command(directory, plan)
        and record['source'] == metadata.receipt(directory / 'code' / Path(record['producer']['command'][-3]).name)
        and record['source']['sha256'] == plan['helpers']['files'][PREFIX + Path(record['source']['path']).name]
        and type(record['started_monotonic']) in (int, float) and record['started_monotonic'] >= 0,
        'the actual configured native origin changed source, owner, clock or plan')
    return helper.DiagnosticDeadline(record, metadata.receipt(path))


def bootstrap(directory, plan, role, metadata, owner, deadline=None):
    """Start one fresh explicit provider in each actual process, never inherited runtime state."""
    policy(plan)
    require(deadline is not None, 'each native role requires the original absolute time budget')
    return adjacent(plan, 'native_process_evidence').bootstrap(directory, plan, role, metadata, owner, deadline=deadline)


def runtime_observer(plan, observer):
    """Refuse a missing or inherited provider on the explicit native execution path."""
    policy(plan)
    runtime = getattr(observer, 'native_process_runtime', None)
    require(runtime is not None and runtime.observer is observer, 'the native role requires its own explicit observer')
    runtime.require_owner(plan)
    return observer


def custody_record(directory, plan, metadata, owner):
    """Recheck only the exact producer and allocation; claim no standing GPU descriptor or hardware mapping."""
    actual = context(directory, plan, metadata)
    require(metadata.stable(actual) == metadata.stable(owner), 'the configured producer changed while retaining tool custody')
    return {'kind': 'owned_configured_allocation_process_custody', 'owner': actual,
        'plan': metadata.receipt(Path(directory) / 'measurement_plan.json'), 'standing_gpu_descriptor_held': False,
        'physical_gres_mapping_verified': False, 'kernel_device_enforcement_verified': False}


def check_custody(record, directory, plan, metadata, owner):
    """Read saved allocation custody without querying a historical process or asserting physical mapping."""
    metadata.validate_snapshot(record['owner'], owner['uids'][0], owner['cgroup'])
    require(set(record) == {'kind', 'owner', 'plan', 'standing_gpu_descriptor_held', 'physical_gres_mapping_verified', 'kernel_device_enforcement_verified'}
        and record['kind'] == 'owned_configured_allocation_process_custody'
        and metadata.stable(record['owner']) == metadata.stable(owner)
        and record['plan'] == metadata.receipt(Path(directory) / 'measurement_plan.json')
        and record['standing_gpu_descriptor_held'] is False and record['physical_gres_mapping_verified'] is False
        and record['kernel_device_enforcement_verified'] is False, 'saved configured allocation custody is incomplete')


def tool_scope(directory, plan, observer, owner, deadline=None):
    """Own the whole tool lifetime before the first version command through final export and ECHILD."""
    runtime_observer(plan, observer)
    require(deadline is not None, 'configured native tools require one shared deadline')
    module = adjacent(plan, 'direct_tool_descendants')
    return module.OwnedToolScope(directory, plan, observer, owner,
        lambda: custody_record(directory, plan, observer, owner), grace_seconds=60, deadline=deadline,
        custody_kind='configured_allocation_process')


def validate_saved(directory, plan, metadata, intent, observation, launch, waited, deadline):
    """Read source-bound native handles and terminal tool closure without importing native libraries."""
    policy(plan)
    scope = adjacent(plan, 'direct_tool_descendants')
    owner = intent['parent'];root = metadata.read_json(Path(directory) / 'profiler_process.json')
    closed = scope.validate_saved_scope(directory, plan, metadata, owner, root['pid'], root['process_start_time'],
        lambda record: check_custody(record, directory, plan, metadata, owner), diagnostic_deadline=deadline)
    roles = adjacent(plan, 'native_process_evidence').validate_saved(directory, plan, metadata, intent, observation, launch, waited,
        diagnostic_deadline=deadline)
    return {'closed_tool_scope': closed, 'native_roles': roles, 'profile_admission_granted': False}
