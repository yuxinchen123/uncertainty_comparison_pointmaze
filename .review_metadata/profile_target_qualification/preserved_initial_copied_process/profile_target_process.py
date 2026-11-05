"""Explicit native process roles for a separately admitted profile-source target.

Old diagnostic plans, providers, raw call validators and default APIs are not
changed. A prior component is an ABI baseline only. Every actual role records
this new adapter's source, its own fresh provider and real target contract.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

NAME = 'profile_target_process.py'
PREFIX = '10_jax_exploration_platform/scripts/atari_pixel_campaign/'
KIND = 'linux-x86-64-libc-syscall-pidfd-version-1'
QUALIFICATION = 'full_original_profile_source_target_qualification'
PROFILE = 'selected_configuration_profile'
PROTOCOL = 'profile-target-native-process-control-version-1'
HELPERS = ('linux_pidfd_process_control.py', NAME, 'native_process_evidence.py',
    'nsight_process_identity.py', 'direct_tool_descendants.py', 'direct_profile_process.py',
    'configured_native_process.py', 'profile_target_qualification.py', 'profile_target_direct_process.py')
BUDGET = {'schema_version': 1, 'work_seconds': 240, 'cleanup_seconds': 60, 'total_seconds': 300}
ROLES = ('producer', 'waiter', 'capture')
PRIOR_REVIEWS = {'a4500': '4264288c209606dccc15c883724ffeccaf5a480c943c5e29dd9b1c409c814da5',
    'h100': '3f4173e5b8d52624e10f90e66095df0c2c0587ecc4beb7b07c8c2874709ae788'}


def require(value, message):
    if not value:raise ValueError(message)


def same(left, right):
    """Preserve exact JSON types rather than equating booleans and integers."""
    if type(left) is not type(right):return False
    if type(right) is dict:return left.keys() == right.keys() and all(same(left[k], right[k]) for k in right)
    if type(right) is list:return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def helpers():
    """Return the exact explicit process policy, provider and read-only validator closure."""
    return HELPERS


def adjacent(plan, name):
    """Load only the staged adjacent bytes, never ambient or old-kind runtime aliases."""
    path = Path(__file__).with_name(name + '.py')
    raw = path.read_bytes()
    expected = plan['helpers']['files'].get(PREFIX + path.name)
    require(path.resolve(strict=True) == path and hashlib.sha256(raw).hexdigest() == expected,
        'the exact profile-target helper is missing or changed: ' + name)
    key = 'profile_target_' + hashlib.sha256((str(path) + expected).encode()).hexdigest()
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        module._profile_target_executed_sha256 = expected
        sys.modules[key] = module
    module = sys.modules[key]
    require(Path(module.__file__) == path and module._profile_target_executed_sha256 == expected,
        'an ambient helper cannot replace the actual staged process source')
    return module


def metadata(plan):
    return adjacent(plan, 'nsight_process_identity')


def reference_value(value, observer):
    require(type(value) is dict and set(value) == {'path', 'sha256'}
        and type(value['path']) is str and Path(value['path']).is_absolute()
        and type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256'])
        and same(observer.receipt(value['path']), value), 'the exact profile-target reference changed')
    return observer.read_json(value['path'])


def declaration(plan, prior):
    return {'schema_version': 1, 'protocol': PROTOCOL, 'backend': KIND, 'python': plan['python'],
        'hardware_class': plan['hardware_class'], 'prior_abi_baseline': prior,
        'helper_hashes': {name: plan['helpers']['files'][PREFIX + name] for name in helpers()},
        'prior_component_qualifies_current_adapter': False, 'scientific_credit': 0, 'admission_granted': False}


def check_baseline(plan, refs, observer, actual=None):
    """Reuse exact reviewed ABI facts without relabeling an old diagnostic as this new adapter."""
    require(type(refs) is dict and set(refs) == {'diagnostic_plan', 'diagnostic_result', 'independent_review', 'target_contract'},
        'one actual reviewed same-target ABI component is required')
    hardware = plan['hardware_class']
    require(hardware in PRIOR_REVIEWS and refs['independent_review']['sha256'] == PRIOR_REVIEWS[hardware],
        'only the exact independently accepted same-target ABI baseline may be reused')
    diagnostic = reference_value(refs['diagnostic_plan'], observer)
    result = reference_value(refs['diagnostic_result'], observer)
    review = reference_value(refs['independent_review'], observer)
    observed = reference_value(refs['target_contract'], observer)
    aa = hardware == 'a4500'
    require(diagnostic.get('kind') == ('configured_native_nsight_process_diagnostic' if aa else 'direct_nsight_process_descriptor_diagnostic')
        and diagnostic.get('python') == plan['python']
        and diagnostic.get('hardware_class') == ('jaguar03_rtx_a4500' if aa else 'h100')
        and result.get('kind') == diagnostic['kind'] and result.get('completed') is True
        and same(result.get('plan'), refs['diagnostic_plan'])
        and type(result.get('scientific_credit')) is int and result['scientific_credit'] == 0
        and type(result.get('profile_credit')) is int and result['profile_credit'] == 0
        and result.get('admission_granted') is False
        and review.get('approved') is True and review.get('passed') is True
        and same(review.get('plan'), refs['diagnostic_plan']) and same(review.get('result'), refs['diagnostic_result'])
        and diagnostic['helpers']['files'][PREFIX + 'linux_pidfd_process_control.py'] ==
            plan['helpers']['files'][PREFIX + 'linux_pidfd_process_control.py'],
        'the actual reviewed result, source, Python or unchanged native provider baseline differs')
    if aa:
        require(review.get('kind') == 'independent_configured_native_process_review'
            and review.get('independent_of_producer') is True and review.get('actual_owned_numeric_step_verified') is True
            and review.get('all_tool_descendants_exited_and_waited') is True
            and same(result.get('target_contract'), refs['target_contract'])
            and same(review.get('target_contract'), refs['target_contract']) and review.get('profile_admission_granted') is False,
            'the exact actual AA component review is required')
    else:
        require(review.get('kind') == 'independent_actual_direct_native_process_component_review'
            and review.get('all_tool_children_ECHILD_and_empty_lists') is True
            and review.get('normal_tool_and_payload_waits') is True and review.get('formal_profile_admission') is False
            and same(result['process_proof']['native_process_evidence']['target_contract'], refs['target_contract']),
            'the exact actual H100 component review is required')
    require(observed.get('qualification') is False and observed.get('admission_granted') is False,
        'old observed ABI facts cannot grant current profile qualification')
    if actual is not None:
        require(same(observed['contract'], actual), 'the actual selected Python/libc/header/symbol target differs from its ABI baseline')
    return observed['contract']


def configure_plan(plan, component_result_path, component_review_path):
    """Bind actual reviewed metadata; this does not run a diagnostic or admit a worker."""
    qualification = adjacent(plan, 'profile_target_qualification')
    (qualification.contract if plan.get('kind') == QUALIFICATION else qualification.authorize)(plan)
    observer = metadata(plan)
    result_ref, review_ref = observer.receipt(component_result_path), observer.receipt(component_review_path)
    result = reference_value(result_ref, observer)
    contract_ref = result['target_contract'] if plan['hardware_class'] == 'a4500' else result['process_proof']['native_process_evidence']['target_contract']
    refs = {'diagnostic_plan': result['plan'], 'diagnostic_result': result_ref,
        'independent_review': review_ref, 'target_contract': contract_ref}
    check_baseline(plan, refs, observer)
    if plan['hardware_class'] == 'a4500':
        plan.update(adjacent(plan, 'configured_native_process').declaration(plan, refs))
    else:
        require('native_process_control' not in plan and 'configured_native_process_control' not in plan,
            'H100 cannot borrow the AA or old direct diagnostic policy')
    plan['profile_target_process_control'] = declaration(plan, refs)
    plan['diagnostic_time_budget'], plan['maximum_active_seconds'] = dict(BUDGET), 240
    policy(plan)
    return plan


def policy(plan):
    """Require the new exact route, real hardware policy and separate admission authority."""
    require(plan.get('kind') in (QUALIFICATION, PROFILE) and plan.get('hardware_class') in PRIOR_REVIEWS,
        'only the explicit qualification or admitted selected profile may use this adapter')
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['helpers']['files'].get(PREFIX + NAME),
        'the actual new profile-target adapter differs from its staged source')
    qualifier = adjacent(plan, 'profile_target_qualification')
    (qualifier.contract if plan['kind'] == QUALIFICATION else qualifier.authorize)(plan)
    declared = plan.get('profile_target_process_control')
    require(type(declared) is dict and same(declared, declaration(plan, declared.get('prior_abi_baseline')))
        and all(type(sha) is str and re.fullmatch('[0-9a-f]{64}', sha) for sha in declared['helper_hashes'].values())
        and same(plan.get('diagnostic_time_budget'), BUDGET)
        and type(plan.get('maximum_active_seconds')) is int and plan['maximum_active_seconds'] == 240,
        'the exact new process-source declaration and shared five-minute budget are required')
    if plan['hardware_class'] == 'a4500':
        require('configured_isolation' in plan and 'direct_binding' not in plan and plan.get('gpu_uuid') is None,
            'AA must retain its real unresolved configured-isolation policy')
        expected = adjacent(plan, 'configured_native_process').declaration(plan, declared['prior_abi_baseline'])
        require(all(same(plan.get(key), value) for key, value in expected.items()),
            'AA prior policy metadata must remain exact without invoking its old runtime')
    else:
        require('direct_binding' in plan and 'configured_isolation' not in plan
            and plan.get('gpu_uuid') == 'GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3'
            and 'native_process_control' not in plan and 'configured_native_process_control' not in plan,
            'H100 must retain only its real card-one direct policy')
    return declared


def optional_policy(plan):
    return policy(plan) if 'profile_target_process_control' in plan else None


def producer_command(directory, plan):
    return [plan['python'], str(Path(directory) / 'code/profile_selected.py'), '--prepared-unit', str(directory)]


def direct_facade(plan):
    """Select the new explicit H lifetime facade without changing the old AA helper namespace."""
    require(plan.get('hardware_class') == 'h100', 'the direct facade is only for the real card-one target')
    return adjacent(plan, 'profile_target_direct_process')


def role_commands(directory, plan):
    helper = direct_facade(plan) if plan['hardware_class'] == 'h100' else adjacent(plan, 'nsight_process_identity')
    return helper.commands(directory, plan)


def context(directory, plan, observer):
    """Observe only this actual own process; AA additionally proves the exact owned numeric step."""
    policy(plan)
    owner = observer.ProcReader().snapshot(os.getpid(), 2618919)
    require(os.getuid() == os.geteuid() == 2618919 and owner['uids'] == [2618919] * 4
        and owner['command'][0] == plan['python']
        and os.uname().nodename.split('.')[0] == ('jaguar03' if plan['hardware_class'] == 'a4500' else 'serval02'),
        'a new profile-target role requires its actual owning Python/host/UID')
    if plan['hardware_class'] == 'a4500':
        job, step = os.environ.get('SLURM_JOB_ID'), os.environ.get('SLURM_STEP_ID')
        require(type(job) is str and re.fullmatch('[1-9][0-9]*', job)
            and type(step) is str and re.fullmatch('[0-9]+', step)
            and re.search(r'/job_' + job + r'/step_' + step + r'(?:/|$)', owner['cgroup'])
            and all(type(os.environ.get(key)) is str and os.environ[key]
                and len(os.environ[key].split(',')) == 1 for key in ('SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES')),
            'AA process control requires its actual one-GPU numeric step')
        run = Path(plan['campaign_run_dir'])
        for name in ('submitted_jobids.txt', 'submitted_jobids_uva.txt'):
            require((run / 'slurm' / name).read_text().splitlines().count(job) == 1,
                'the allocation must be exactly campaign-owned before work')
        require(sum(line.split('\t')[0] == job for line in (run / 'slurm/submissions.tsv').read_text().splitlines()) == 1,
            'the allocation requires its one unique owned submission row')
    else:
        selectors = {key: value for key, value in os.environ.items()
            if key == 'CUDA_VISIBLE_DEVICES' or key.startswith('SLURM_')}
        require(selectors == {'CUDA_VISIBLE_DEVICES': 'GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3'},
            'the actual H100 role cannot rewrite the card-one selector or borrow a scheduler')
    return owner


def completion_references(plan, role):
    if role == 'producer':return ('tool_descendant_scope_terminal.json', 'tool_descendant_scope_restored.json', 'profiler_actual_wait.json')
    if role == 'waiter':return ('capture_actual_launch.json', 'capture_actual_wait.json')
    require(role == 'capture', 'one exact profile-target role is required')
    return ('capture_release_observed.json', 'native_capture_payload_timing.json', 'capture_result.json')


def role_files(directory, role):
    require(role in ROLES, 'one exact new profile-target role is required')
    return {name: Path(directory) / ('profile_target_process_' + role + '_' + name + '.json')
        for name in ('begin', 'terminal', 'failure')}


def saved_deadline(directory, plan, observer):
    """Validate saved same-origin arithmetic without querying historical PIDs or live clocks."""
    policy(plan);directory = Path(directory)
    helper = adjacent(plan, 'direct_profile_process')
    path = directory / 'profile_target_process_time_origin.json'
    record = observer.read_json(path)
    observer.validate_snapshot(record['producer'], 2618919, record['producer']['cgroup'])
    require(set(record) == {'kind', 'schema_version', 'plan', 'producer', 'source', 'budget', 'clock_domain', 'started_monotonic'}
        and record['kind'] == 'profile_target_process_time_origin' and type(record['schema_version']) is int and record['schema_version'] == 1
        and same(record['plan'], observer.receipt(directory / 'measurement_plan.json')) and same(record['budget'], BUDGET)
        and set(record['clock_domain']) == {'hostname', 'boot_id', 'clock', 'implementation'}
        and record['clock_domain']['hostname'] == ('jaguar03' if plan['hardware_class'] == 'a4500' else 'serval02')
        and record['clock_domain']['clock'] == 'monotonic'
        and type(record['clock_domain']['implementation']) is str and record['clock_domain']['implementation']
        and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', record['clock_domain']['boot_id'])
        and same(record['producer']['command'], producer_command(directory, plan))
        and same(record['source'], observer.receipt(directory / 'code/profile_selected.py'))
        and record['source']['sha256'] == plan['helpers']['files'][PREFIX + 'profile_selected.py']
        and type(record['started_monotonic']) in (int, float) and math.isfinite(record['started_monotonic'])
        and record['started_monotonic'] >= 0, 'the new role origin changed actual source, plan, owner or clock')
    return helper.DiagnosticDeadline(record, observer.receipt(path))


def deadline(directory, plan, observer, owner=None, started=None):
    policy(plan);directory = Path(directory);helper = adjacent(plan, 'direct_profile_process')
    if owner is not None:
        require(type(started) in (int, float) and math.isfinite(started)
            and started <= time.monotonic() < started + 240
            and same(owner['command'], producer_command(directory, plan)), 'retain one actual producer origin before native bootstrap')
        observer.publish(directory / 'profile_target_process_time_origin.json', {'kind': 'profile_target_process_time_origin',
            'schema_version': 1, 'plan': observer.receipt(directory / 'measurement_plan.json'), 'producer': owner,
            'source': observer.receipt(directory / 'code/profile_selected.py'), 'budget': BUDGET,
            'clock_domain': helper.monotonic_clock_domain(), 'started_monotonic': started})
    result = saved_deadline(directory, plan, observer)
    require(same(result.origin['clock_domain'], helper.monotonic_clock_domain()), 'the actual host/boot clock changed between native roles')
    result.remaining('work')
    return result


def load_provider(plan):
    """Native imports occur only after exact new policy and owned live context checks."""
    policy(plan)
    require(not {'jax', 'jaxlib', 'torch', 'numpy', 'cupy', 'scipy', 'flax', 'optax', 'jaxtari',
        'exploration_platform'}.intersection(sys.modules), 'each native role must bootstrap before numerical imports')
    return adjacent(plan, 'linux_pidfd_process_control')


def check_contract(plan, contract, observer):
    return check_baseline(plan, policy(plan)['prior_abi_baseline'], observer, actual=contract)


def runtime_observer(plan, observer):
    policy(plan)
    runtime = getattr(observer, 'native_process_runtime', None)
    require(runtime is not None and type(runtime).__module__ == __name__ and runtime.observer is observer,
        'each new role requires its own current explicit native provider')
    runtime.require_owner(plan)
    return observer


def custody_record(directory, plan, observer, owner):
    actual = context(directory, plan, observer)
    require(same(observer.stable(actual), observer.stable(owner)), 'the configured producer changed while retaining allocation custody')
    return {'kind': 'profile_target_configured_allocation_custody', 'owner': actual,
        'plan': observer.receipt(Path(directory) / 'measurement_plan.json'), 'standing_gpu_descriptor_held': False,
        'physical_gres_mapping_verified': False, 'kernel_device_enforcement_verified': False}


def check_custody(record, directory, plan, observer, owner):
    observer.validate_snapshot(record['owner'], 2618919, owner['cgroup'])
    require(set(record) == {'kind', 'owner', 'plan', 'standing_gpu_descriptor_held', 'physical_gres_mapping_verified', 'kernel_device_enforcement_verified'}
        and record['kind'] == 'profile_target_configured_allocation_custody'
        and same(observer.stable(record['owner']), observer.stable(owner))
        and same(record['plan'], observer.receipt(Path(directory) / 'measurement_plan.json'))
        and record['standing_gpu_descriptor_held'] is False and record['physical_gres_mapping_verified'] is False
        and record['kernel_device_enforcement_verified'] is False, 'saved actual allocation custody is incomplete')


def tool_scope(directory, plan, observer, owner, deadline=None, descriptor=None, check_descriptor=None):
    runtime_observer(plan, observer)
    require(deadline is not None, 'all native tools require the same absolute role deadline')
    if plan['hardware_class'] == 'h100':
        require(type(descriptor) is int and descriptor >= 0 and callable(check_descriptor),
            'H100 tools require the actual standing descriptor and explicit validated custody callback')
        callback, custody = check_descriptor, 'standing_gpu_descriptor'
    else:
        require(descriptor is None and check_descriptor is None, 'AA cannot inherit or fabricate a standing H100 descriptor')
        callback, custody = lambda: custody_record(directory, plan, observer, owner), 'configured_allocation_process'
    return adjacent(plan, 'direct_tool_descendants').OwnedToolScope(directory, plan, observer, owner,
        callback, grace_seconds=60, deadline=deadline, custody_kind=custody)


def capture_contract(directory, contract, observer):
    """Preserve actual public image and header bytes beside the immutable observed contract."""
    import os
    root = Path(directory) / 'profile_target_process_contract_bytes'
    root.mkdir()
    records = {record['path']: record for record in [*contract['headers'].values(),
        contract['selected_python'], *(item['image'] for item in contract['symbols'].values())]}
    captured = {}
    for path, record in records.items():
        raw = Path(path).read_bytes()
        require(len(raw) == record['size_bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'],
            'an actual public native image or header changed before its immutable capture')
        output = root / record['sha256']
        if not output.exists():
            with output.open('xb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        captured[path] = observer.receipt(output)
    value = {'kind': 'profile_target_process_observed_contract', 'contract': contract, 'captured_files': captured,
        'source': {name: plan_hash for name, plan_hash in observer.native_source_hashes.items()},
        'actual_target_observation_only': True, 'qualification': False, 'admission_granted': False}
    observer.publish(Path(directory) / 'profile_target_process_contract.json', value)
    return observer.receipt(Path(directory) / 'profile_target_process_contract.json')

class Runtime:
    """Retain one explicit provider for one actual process and publish immutable role boundaries."""
    def __init__(self, directory, plan, role, metadata, owner, *, deadline=None):
        """Join actual context before native startup and finish self preflight before tools or parent watches."""
        import os
        policy(plan)
        actual = context(directory, plan, metadata)
        expected_commands = dict(zip(('waiter', 'capture'), role_commands(directory, plan)))
        expected_commands['producer'] = producer_command(directory, plan)
        require(role in ROLES and same(owner['command'], expected_commands[role])
            and same(metadata.stable(actual), metadata.stable(owner)),
            'the new role must be this actual owned source command before native bootstrap')
        self.directory, self.plan, self.role, self.metadata = Path(directory), plan, role, metadata
        self.owner, self.pid, self.completed = owner, os.getpid(), False
        self.deadline = deadline
        require(('diagnostic_time_budget' in plan) == (deadline is not None),
            'native role startup must receive exactly its declared diagnostic deadline')
        if deadline is not None:
            require(same(deadline.origin['budget'], plan['diagnostic_time_budget']), 'native deadline budget differs from the plan')
            deadline.remaining('work')
        self.files = role_files(directory, role)
        require(owner['pid'] == self.pid and owner['uids'] == [os.getuid()] * 4
            and not any(path.exists() for path in self.files.values()), 'native role startup must be fresh and actually owned')
        metadata.validate_snapshot(owner, os.getuid(), owner['cgroup'])
        require(Path(__file__).resolve(strict=True) == Path(__file__)
            and hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            == plan['helpers']['files'][PREFIX + HELPERS[1]], 'the actual native evidence helper changed')
        module = load_provider(plan)
        started = time.time()
        contract, unused = module.current_contract()
        check_contract(plan, contract, metadata)
        self.provider = module.NativePidfdProvider(contract)
        self.observer = module.NativeProcessObserver(metadata, self.provider)
        self.observer.native_process_runtime = self
        self.observer.native_source_hashes = policy(plan)['helper_hashes']
        if role == 'producer':
            contract_ref = capture_contract(directory, contract, self.observer)
        else:
            contract_ref = metadata.receipt(Path(directory) / 'profile_target_process_contract.json')
            saved = metadata.read_json(contract_ref['path'])
            require(same(saved['contract'], contract), 'this child process does not match its actual producer target contract')
        self.begin = {'kind': 'actual_profile_target_process_role_begin', 'schema_version': 1, 'role': role,
            'plan': metadata.receipt(Path(directory) / 'measurement_plan.json'), 'policy': policy(plan),
            'owner': owner, 'contract': contract_ref, 'provider': self.provider.evidence(),
            'started_at_seconds': started, 'published_at_seconds': time.time(),
            'scientific_credit': 0, 'admission_granted': False}
        if deadline is not None:
            deadline.remaining('work')
            self.begin['diagnostic_deadline'] = deadline.snapshot()
        metadata.publish(self.files['begin'], self.begin)
        if deadline is not None:deadline.remaining('work')

    def require_owner(self, plan):
        """Refuse inherited, completed or unrelated runtime observer state."""
        import os
        require(os.getpid() == self.pid and not self.completed and same(plan, self.plan),
            'native observer state belongs only to its one unfinished actual process')

    def complete(self, references):
        """Publish success only after the caller's real waits and all retained handle closes."""
        self.require_owner(self.plan)
        if self.deadline is not None:self.deadline.remaining('total')
        evidence = self.provider.evidence()
        require(evidence['open_handle_count'] == 0 and not evidence['native_call_failure'],
            'native completion requires every retained handle closed and no failed native calls')
        value = {'kind': 'actual_profile_target_process_role_terminal', 'schema_version': 1, 'role': self.role,
            'begin': self.metadata.receipt(self.files['begin']), 'owner': self.owner,
            'references': {name: self.metadata.receipt(self.directory / name) for name in references},
            'provider': evidence, 'completed_at_seconds': time.time(), 'completed': True,
            'scientific_credit': 0, 'admission_granted': False}
        if self.deadline is not None:
            value['diagnostic_deadline'] = self.deadline.snapshot()
            self.deadline.verify_snapshot(value['diagnostic_deadline'], phase='total')
        self.metadata.publish(self.files['terminal'], value)
        if self.deadline is not None:self.deadline.remaining('total')
        self.completed = True
        return self.metadata.receipt(self.files['terminal'])

    def failure(self, error):
        """Preserve partial native state without converting an unknown cleanup into success."""
        self.metadata.publish(self.files['failure'], {'kind': 'actual_profile_target_process_role_failure',
            'role': self.role, 'error_type': type(error).__name__, 'error': str(error),
            'provider': self.provider.evidence(), 'completed': False, 'scientific_credit': 0,
            'admission_granted': False, 'terminal_previously_published': self.completed,
            **({} if self.deadline is None else {'diagnostic_deadline': self.deadline.snapshot()})})
        return self.metadata.receipt(self.files['failure'])

def bootstrap(directory, plan, role, metadata, owner, *, deadline=None):
    """Preserve failed startup evidence even when an immutable role begin cannot be published."""
    runtime = Runtime.__new__(Runtime)
    try:
        Runtime.__init__(runtime, directory, plan, role, metadata, owner, deadline=deadline)
        return runtime
    except BaseException as error:
        provider = getattr(runtime, 'provider', None)
        evidence = provider.evidence() if provider is not None else getattr(error, 'native_process_control_evidence', None)
        metadata.publish(role_files(directory, role)['failure'], {'kind': 'actual_profile_target_process_role_bootstrap_failure',
            'role': role, 'owner': owner, 'plan': metadata.receipt(Path(directory) / 'measurement_plan.json'),
            'error_type': type(error).__name__, 'error': str(error), 'provider': evidence,
            'partial_native_state_verified': evidence is not None, 'completed': False,
            'scientific_credit': 0, 'admission_granted': False})
        raise

def validate_contract(saved, plan, observer, mapping=None):
    """Check captured public bytes and fixed ABI declarations without importing any native library."""
    require(set(saved) == {'kind', 'contract', 'captured_files', 'source', 'actual_target_observation_only',
        'qualification', 'admission_granted'} and saved['kind'] == 'profile_target_process_observed_contract',
        'only the exact observed native diagnostic contract schema is supported')
    contract = saved['contract']
    require(set(contract) == {'kind', 'abi', 'syscall_numbers', 'headers', 'symbols', 'selected_python'}
        and contract['kind'] == KIND and same(contract['abi'], {'platform': 'linux', 'machine': 'x86_64',
        'byteorder': 'little', 'long_bytes': 8, 'int_bytes': 4, 'pointer_bytes': 8})
        and same(contract['syscall_numbers'], {'pidfd_open': 434, 'pidfd_send_signal': 424}), 'saved native ABI differs')
    require(set(contract['headers']) == {'/usr/include/unistd.h', '/usr/include/x86_64-linux-gnu/asm/unistd_64.h'}
        and set(contract['symbols']) == {'syscall', 'close'}
        and same(contract['symbols']['syscall']['image'], contract['symbols']['close']['image']), 'saved native images differ')
    records = [*contract['headers'].values(), contract['selected_python'], contract['symbols']['syscall']['image']]
    require(set(saved['captured_files']) == {row['path'] for row in records}
        and same(saved['source'], policy(plan)['helper_hashes']) and saved['actual_target_observation_only'] is True
        and saved['qualification'] is False and saved['admission_granted'] is False, 'target observation cannot grant qualification')
    for row in records:
        fields = {'path', 'sha256', 'size_bytes', 'uid', 'mode', 'device', 'inode'}
        require(set(row) == (fields | {'actual_running_image_path'} if row is contract['selected_python'] else fields)
            and type(row['path']) is str and Path(row['path']).is_absolute()
            and type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256'])
            and all(type(row[key]) is int and row[key] >= 0 for key in ('uid', 'mode', 'device', 'inode', 'size_bytes'))
            and row['inode'] > 0 and 0 < row['size_bytes'] <= 128 * 1024 * 1024, 'saved native file identity is incomplete')
        ref = saved['captured_files'][row['path']]
        actual, mapped = reference(ref['path'], observer, mapping)
        require(same(actual, ref) and ref['sha256'] == row['sha256'] and mapped.stat().st_size == row['size_bytes'],
            'actual captured native header or image bytes changed')
        if row is not contract['selected_python']:
            require(row['uid'] == 0 and row['mode'] & 0o022 == 0, 'public native headers and libc must retain root ownership')
    for row in contract['symbols'].values():
        require(set(row) == {'image', 'symbol_file_offset'} and type(row['symbol_file_offset']) is int
            and 0 <= row['symbol_file_offset'] < row['image']['size_bytes'], 'saved native symbol offset is invalid')
    for name, row in contract['headers'].items():
        raw = Path((mapping or {}).get(saved['captured_files'][row['path']]['path'],
            saved['captured_files'][row['path']]['path'])).read_bytes()
        if name.endswith('unistd_64.h'):
            for key, number in contract['syscall_numbers'].items():
                require(re.findall(rb'^#define\s+__NR_' + key.encode() + rb'\s+(\d+)\s*$', raw, re.MULTILINE)
                    == [str(number).encode()], 'captured syscall number declaration differs')
        else:
            require(re.search(rb'extern\s+long\s+int\s+syscall\s*\(long\s+int\s+__sysno,\s*\.\.\.\)', raw),
                'captured variadic syscall declaration is missing')
    python = contract['selected_python']
    require(python['mode'] & 0o111 != 0 and python['actual_running_image_path'] == python['path'],
        'saved actual running Python image identity is incomplete')
    return contract

def finite_time(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0

def validate_calls(evidence, owner, allowed, contract, plan):
    return adjacent(plan, 'native_process_evidence').validate_calls(evidence, owner, allowed, contract)

def validate_scope_snapshot(record, uid, cgroup, observer, plan):
    return adjacent(plan, 'native_process_evidence').validate_scope_snapshot(record, uid, cgroup, observer)

def validate_roles(directory, plan, observer, intent, observation, launch, waited, mapping=None, diagnostic_deadline=None):
    """Read all three actual role records without loading a library or observing any historical live process."""
    policy(plan);directory = Path(directory)
    budgeted = 'diagnostic_time_budget' in plan
    require(budgeted == (diagnostic_deadline is not None), 'saved native roles need their exact declared deadline proof')
    configured = plan['hardware_class'] == 'a4500'
    owners = {'producer': intent['parent'] if configured else intent['producer'],
        'waiter': launch['waiter'], 'capture': launch['child']}
    contract_ref, unused = reference(directory / 'profile_target_process_contract.json', observer, mapping)
    contract = validate_contract(read_reference(contract_ref, observer, mapping), plan, observer, mapping)
    scope_ref, unused = reference(directory / 'tool_descendant_scope_terminal.json', observer, mapping)
    terminal_scope = read_reference(scope_ref, observer, mapping)
    allowed = {owner['pid']: owner for owner in owners.values()}
    for node in terminal_scope['positive_nodes'].values():
        record = node['initial']
        validate_scope_snapshot(record, owners['producer']['uids'][0], owners['producer']['cgroup'], observer, plan)
        require(record['pid'] not in allowed or all(record[key] == allowed[record['pid']][key] for key in ('pid', 'process_start_time', 'uids', 'cgroup')),
            'native opened process has conflicting positive identity')
        allowed[record['pid']] = record
    for record in observation['nodes'].values():
        observer.validate_snapshot(record, owners['producer']['uids'][0], owners['producer']['cgroup'])
        require(record['pid'] not in allowed or all(record[key] == allowed[record['pid']][key]
            for key in ('pid', 'process_start_time', 'uids', 'cgroup')), 'native ancestry and scope identities conflict')
        allowed[record['pid']] = record
    result = {}
    for role in ROLES:
        paths = role_files(directory, role)
        require(not Path((mapping or {}).get(str(paths['failure']), str(paths['failure']))).exists(),
            'a failed native role cannot become successful saved evidence')
        begin_ref, unused = reference(paths['begin'], observer, mapping)
        begin = read_reference(begin_ref, observer, mapping)
        terminal_ref, unused = reference(paths['terminal'], observer, mapping)
        terminal = read_reference(terminal_ref, observer, mapping)
        owner = owners[role];observer.validate_snapshot(owner, owner['uids'][0], owner['cgroup'])
        expected_command = (intent['capture_command'] if role == 'capture' else intent['waiter_command']
            if role == 'waiter' else producer_command(directory, plan))
        observer.validate_snapshot(begin['owner'], owner['uids'][0], owner['cgroup'])
        require(set(begin) == {'kind', 'schema_version', 'role', 'plan', 'policy', 'owner', 'contract', 'provider',
            'started_at_seconds', 'published_at_seconds', 'scientific_credit', 'admission_granted'} | ({'diagnostic_deadline'} if budgeted else set())
            and begin['kind'] == 'actual_profile_target_process_role_begin'
            and owner['command'] == expected_command and observer.stable(begin['owner']) == observer.stable(owner)
            and begin['role'] == role and same(begin['policy'], policy(plan)) and same(begin['contract'], contract_ref)
            and same(begin['plan'], observer.receipt(directory / 'measurement_plan.json'))
            and finite_time(begin['started_at_seconds']) and finite_time(begin['published_at_seconds'])
            and begin['started_at_seconds'] <= begin['published_at_seconds'], 'native begin has different source, plan, owner or chronology')
        require(set(terminal) == {'kind', 'schema_version', 'role', 'begin', 'owner', 'references', 'provider',
            'completed_at_seconds', 'completed', 'scientific_credit', 'admission_granted'} | ({'diagnostic_deadline'} if budgeted else set())
            and terminal['kind'] == 'actual_profile_target_process_role_terminal'
            and terminal['completed'] is True and terminal['role'] == role and same(terminal['owner'], begin['owner'])
            and same(terminal['begin'], begin_ref) and finite_time(terminal['completed_at_seconds'])
            and begin['published_at_seconds'] <= terminal['completed_at_seconds'], 'native terminal is incomplete or belongs to another role')
        for record in (begin, terminal):
            require(type(record['schema_version']) is int and record['schema_version'] == 1
                and type(record['scientific_credit']) is int and record['scientific_credit'] == 0
                and record['admission_granted'] is False, 'native records cannot grant scientific or admission credit')
        if budgeted:
            diagnostic_deadline.verify_snapshot(begin['diagnostic_deadline'], phase='work')
            diagnostic_deadline.verify_snapshot(terminal['diagnostic_deadline'], phase='total')
        validate_calls(begin['provider'], owner, {owner['pid']: owner}, contract, plan)
        require(len(begin['provider']['calls']) == 3, 'native begin must contain only its actual self preflight')
        require(begin['started_at_seconds'] <= begin['provider']['calls'][0]['started_at_seconds']
            and begin['provider']['calls'][-1]['finished_at_seconds'] <= begin['published_at_seconds'],
            'native self preflight must occur inside the recorded actual startup window')
        role_allowed = allowed if role == 'producer' else {owner['pid']: owner, owners['producer']['pid']: owners['producer']}
        if configured and role == 'waiter':role_allowed[owners['capture']['pid']] = owners['capture']
        counts = validate_calls(terminal['provider'], owner, role_allowed, contract, plan)
        opened = [call['arguments']['pid'] for call in terminal['provider']['calls'] if call['name'] == 'pidfd_open']
        if role == 'producer':
            require(opened.count(owner['pid']) >= 2 and all(opened.count(pid) >= 1 for pid in role_allowed),
                'native producer evidence omits a known scope or ancestry process handle')
            require(all(opened.count(record['pid']) >= 2 for record in observation['nodes'].values()
                if str(record['pid']) in terminal_scope['positive_nodes']),
                'native evidence omits the separate scope and ancestry handles')
        else:
            expected_opened = [owner['pid'], owners['producer']['pid']]
            if configured and role == 'waiter':expected_opened.append(owners['capture']['pid'])
            require(opened == expected_opened,
                'native child evidence must contain its self preflight and exact known-producer watcher')
            for event in terminal['provider']['descriptor_events']:
                exited_capture_close = configured and role == 'waiter' and event['opened_pid'] == owners['capture']['pid']
                if exited_capture_close:
                    require(event['operation'] in ('pidfd_open', 'close')
                        and (event['identity']['pid'] == event['opened_pid'] or event['operation'] == 'close' and event['identity']['pid'] == -1)
                        and (event['operation'] != 'close' or type(waited.get('returncode')) is int and waited['returncode'] == 0
                            and waited.get('exited_normally') is True and waited.get('interventions') == []
                            and finite_time(waited.get('actual_wait_finished_at_seconds'))
                            and waited['actual_wait_finished_at_seconds'] <= event['observed_at_seconds']),
                        'an exited configured capture handle requires its actual successful prior Popen wait')
                else:
                    require(event['identity']['pid'] == event['opened_pid'],
                        'a successful native child cannot retain an exited known producer')
        require(not any(call['name'] == 'pidfd_send_signal' and call['arguments']['signal'] in (15, 9)
            for call in terminal['provider']['calls']), 'successful native diagnostic evidence cannot hide cleanup intervention')
        require(terminal['provider']['calls'][-1]['finished_at_seconds'] <= terminal['completed_at_seconds'],
            'native terminal predates its last actual call')
        require(same(terminal['provider']['calls'][:3], begin['provider']['calls'])
            and same(terminal['provider']['descriptor_events'][:3], begin['provider']['descriptor_events']),
            'native terminal changed its frozen self preflight')
        required = completion_references(plan, role)
        require(set(terminal['references']) == set(required), 'native role did not retain all required actual completion records')
        for name, ref in terminal['references'].items():
            actual, unused = reference(directory / name, observer, mapping)
            require(same(ref, actual), 'native completion reference changed')
        result[role] = {'begin': begin_ref, 'terminal': terminal_ref, **counts}
    producer_begin = read_reference(result['producer']['begin'], observer, mapping)
    scope_ref, unused = reference(directory / 'tool_descendant_scope_begin.json', observer, mapping)
    scope_begin = read_reference(scope_ref, observer, mapping)
    require(producer_begin['published_at_seconds'] <= scope_begin['established_at_seconds'],
        'the producer native provider must precede its first tool scope')
    begin = {role: read_reference(result[role]['begin'], observer, mapping) for role in ROLES}
    terminal = {role: read_reference(result[role]['terminal'], observer, mapping) for role in ROLES}
    payload_ref, unused = reference(directory / 'native_capture_payload_timing.json', observer, mapping)
    payload = read_reference(payload_ref, observer, mapping)
    if budgeted:diagnostic_deadline.verify_snapshot(payload['diagnostic_deadline'], phase='work')
    require(begin['waiter']['published_at_seconds'] <= begin['capture']['started_at_seconds']
        and begin['capture']['published_at_seconds'] <= payload['payload_started_at_seconds']
        <= payload['payload_finished_at_seconds'] <= terminal['capture']['completed_at_seconds']
        <= terminal['waiter']['completed_at_seconds'] <= terminal['producer']['completed_at_seconds'],
        'native role startup, payload and terminal chronology is incomplete')
    return {'kind': 'saved_profile_target_three_role_process_evidence', 'passed': True, 'roles': result,
        'target_contract': contract_ref, 'qualification': False, 'admission_granted': False, 'scientific_credit': 0}


def reference(path, observer, mapping=None):
    """Read immutable bytes through an explicit optional portable path mapping, never live process state."""
    original = str(path)
    mapped = Path((mapping or {}).get(original, original))
    actual = observer.receipt(mapped)
    return {'path': original, 'sha256': actual['sha256']}, mapped

def read_reference(expected, observer, mapping=None):
    """Verify an exact recorded reference before decoding its saved JSON."""
    require(type(expected) is dict and set(expected) == {'path', 'sha256'}
        and type(expected['path']) is str and Path(expected['path']).is_absolute()
        and type(expected['sha256']) is str and re.fullmatch('[0-9a-f]{64}', expected['sha256']),
        'an exact native file reference is required')
    actual, mapped = reference(expected['path'], observer, mapping)
    require(same(expected, actual), 'saved native evidence bytes changed')
    return observer.read_json(mapped)


def validate_saved(directory, plan, observer, intent, observation, launch, waited, deadline=None, *, check_descriptor=None):
    """Verify actual source/ancestry/waits, full tool closure and all three new role receipts."""
    policy(plan);directory = Path(directory)
    require(deadline is not None, 'saved new roles require their actual same-origin five-minute budget')
    direct = plan['hardware_class'] == 'h100'
    parent = intent['producer'] if direct else intent['parent']
    if direct:
        require(callable(check_descriptor), 'H100 saved tool closure requires its actual descriptor validator')
        adapter = direct_facade(plan)
        adapter.validate_intent(directory, plan, intent)
        claim = adapter.validate_claim(directory, plan, intent, observer.read_json(directory / 'capture_identity.json'))
        custody = check_descriptor
    else:
        require(check_descriptor is None, 'AA saved evidence cannot borrow H100 descriptor custody')
        observer.validate_saved_intent(directory, plan, intent)
        claim = observer.read_json(directory / 'capture_identity.json')
        observer.validate_claim(directory, plan, intent, claim)
        custody = lambda value: check_custody(value, directory, plan, observer, parent)
    observer.validate_observation(observation, claim, intent['capture_command'], parent)
    require(same(observation, observer.read_json(directory / 'capture_process_observation.json'))
        and same(launch, observer.read_json(directory / 'capture_actual_launch.json'))
        and same(waited, observer.read_json(directory / 'capture_actual_wait.json'))
        and same(launch['claim'], observer.receipt(directory / 'capture_identity.json'))
        and same(launch['command'], intent['capture_command'])
        and same(launch['challenge'], intent['challenge']) and same(waited['challenge'], intent['challenge'])
        and same(waited['launch'], observer.receipt(directory / 'capture_actual_launch.json'))
        and same(waited['command'], launch['command']) and type(waited['returncode']) is int
        and waited['returncode'] == 0 and waited['exited_normally'] is True and waited['interventions'] == []
        and type(waited['waited_pid']) is int and waited['waited_pid'] == launch['child']['pid']
        and launch['child']['parent_pid'] == launch['waiter']['pid']
        and same(observer.stable(launch['child']), observer.stable(claim['process'])),
        'new native success requires exact actual launch, positive claim and normal captured-child wait')
    contract = observer.read_json(directory / 'profile_target_process_contract.json')
    check_contract(plan, contract['contract'], observer)
    root = observer.read_json(directory / 'profiler_process.json')
    scope = adjacent(plan, 'direct_tool_descendants').validate_saved_scope(directory, plan, observer,
        parent, root['pid'], root['process_start_time'], custody, diagnostic_deadline=deadline)
    roles = validate_roles(directory, plan, observer, intent, observation, launch, waited, diagnostic_deadline=deadline)
    return {'kind': 'saved_profile_target_process_proof', 'closed_tool_scope': scope,
        'native_roles': roles, 'prior_component_is_abi_baseline_only': True,
        'scientific_credit': 0, 'profile_admission_granted': False, 'admission_granted': False}
