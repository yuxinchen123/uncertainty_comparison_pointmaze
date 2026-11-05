"""Declare explicit diagnostic native control and read its saved evidence without native imports."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

KIND = 'linux-x86-64-libc-syscall-pidfd-version-1'
MODE = 'observe_target_contract_for_zero_credit_diagnostic'
DIAGNOSTIC = 'direct_nsight_process_descriptor_diagnostic'
PREFIX = '10_jax_exploration_platform/scripts/atari_pixel_campaign/'
HELPERS = ('linux_pidfd_process_control.py', 'native_process_evidence.py')
ROLES = ('producer', 'waiter', 'capture')
LIMIT = 4096


def require(value, message):
    """Refuse incomplete native declarations and saved evidence."""
    if not value:
        raise ValueError(message)


def same(left, right):
    """Compare values and exact JSON types, including Boolean versus integer fields."""
    if type(left) is not type(right):return False
    if type(right) is dict:return left.keys() == right.keys() and all(same(left[k], right[k]) for k in right)
    if type(right) is list:return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def declaration(plan):
    """Describe an observed-only target contract without inventing target-specific bytes."""
    return {'schema_version': 1, 'backend': KIND, 'mode': MODE, 'expected_target_contract': None,
        'qualification': False, 'python': plan['python'],
        'helper_hashes': {name: plan['helpers']['files'][PREFIX + name] for name in HELPERS}}


def policy(plan):
    """Permit this first native integration only in the explicit zero-credit direct diagnostic."""
    require(plan.get('kind') == DIAGNOSTIC and 'native_process_control' in plan
        and same(plan['native_process_control'], declaration(plan)),
        'native target observation is diagnostic only; formal work needs a separately reviewed contract and admission')
    require(all(type(value) is str and re.fullmatch('[0-9a-f]{64}', value)
        for value in plan['native_process_control']['helper_hashes'].values()), 'native helper hashes must be exact')
    return plan['native_process_control']


def parse_pidfd_fdinfo(raw, expected_inode, expected_pid=None):
    """Join duplicate-safe kernel fdinfo inode, process and octal CLOEXEC fields to one descriptor."""
    require(type(raw) is str and raw.isascii() and raw.endswith('\n') and '\r' not in raw and '\0' not in raw
        and len(raw.encode()) <= 16384, 'pidfd fdinfo must be one complete bounded ASCII record')
    require(type(expected_inode) is int and 0 < expected_inode <= 18446744073709551615
        and (expected_pid is None or type(expected_pid) is int and 0 < expected_pid <= 2147483647),
        'pidfd expected inode and process must use exact positive integer types')
    # Before: raw key/value lines; after: one value per kernel field, with no ignored duplicate spelling.
    fields = {}
    for line in raw.splitlines():
        key, separator, value = line.partition(':')
        require(separator == ':' and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) and key not in fields,
            'pidfd fdinfo contains a malformed or duplicate kernel field')
        fields[key] = value.strip(' \t')
    require({'ino', 'Pid', 'flags'} <= set(fields)
        and re.fullmatch(r'[1-9][0-9]*', fields['ino'])
        and re.fullmatch(r'(?:[1-9][0-9]*|-1)', fields['Pid'])
        and re.fullmatch(r'[0-7]{8,11}', fields['flags']), 'pidfd raw inode, process or octal flags are invalid')
    inode, pid, flags = int(fields['ino']), int(fields['Pid']), int(fields['flags'], 8)
    require(inode == expected_inode and (expected_pid is None or pid in (expected_pid, -1))
        and -1 <= pid <= 2147483647 and flags <= 4294967295 and flags & os.O_CLOEXEC == os.O_CLOEXEC,
        'pidfd raw inode, process or CLOEXEC flags differ from its retained descriptor')
    return {'inode': inode, 'pid': pid, 'flags': flags, 'close_on_exec': True}


def role_files(directory, role):
    """Return fixed unit-local role paths so one role cannot overwrite another."""
    require(role in ROLES, 'an exact native diagnostic role is required')
    return {name: Path(directory) / ('native_process_' + role + '_' + name + '.json')
        for name in ('begin', 'terminal', 'failure')}


def load_provider(plan):
    """Import native operations only during explicit owned-runtime startup, never during saved reads."""
    policy(plan)
    path = Path(__file__).with_name(HELPERS[0])
    require(path.resolve(strict=True) == path and hashlib.sha256(path.read_bytes()).hexdigest()
        == plan['helpers']['files'][PREFIX + path.name], 'the staged native provider changed')
    name = 'explicit_direct_diagnostic_native_pidfd'
    if name not in sys.modules:
        specification = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        sys.modules[name] = module
    module = sys.modules[name]
    require(Path(module.__file__).resolve(strict=True) == path, 'an ambient native provider cannot replace staged source')
    return module


def capture_contract(directory, contract, observer):
    """Preserve actual public image and header bytes beside the immutable observed contract."""
    import os
    root = Path(directory) / 'native_process_contract_bytes'
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
    value = {'kind': 'observed_native_process_target_contract', 'contract': contract, 'captured_files': captured,
        'source': {name: plan_hash for name, plan_hash in observer.native_source_hashes.items()},
        'actual_target_observation_only': True, 'qualification': False, 'admission_granted': False}
    observer.publish(Path(directory) / 'native_process_target_contract.json', value)
    return observer.receipt(Path(directory) / 'native_process_target_contract.json')


class Runtime:
    """Retain one explicit provider for one actual process and publish immutable role boundaries."""
    def __init__(self, directory, plan, role, metadata, owner, *, deadline=None):
        """Join actual context before native startup and finish self preflight before tools or parent watches."""
        import os
        policy(plan)
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
        self.provider = module.NativePidfdProvider(contract)
        self.observer = module.NativeProcessObserver(metadata, self.provider)
        self.observer.native_process_runtime = self
        self.observer.native_source_hashes = policy(plan)['helper_hashes']
        if role == 'producer':
            contract_ref = capture_contract(directory, contract, self.observer)
        else:
            contract_ref = metadata.receipt(Path(directory) / 'native_process_target_contract.json')
            saved = metadata.read_json(contract_ref['path'])
            require(same(saved['contract'], contract), 'this child process does not match its actual producer target contract')
        self.begin = {'kind': 'actual_native_process_role_begin', 'schema_version': 1, 'role': role,
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
        value = {'kind': 'actual_native_process_role_terminal', 'schema_version': 1, 'role': self.role,
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
        self.metadata.publish(self.files['failure'], {'kind': 'actual_native_process_role_failure',
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
        metadata.publish(role_files(directory, role)['failure'], {'kind': 'actual_native_process_role_bootstrap_failure',
            'role': role, 'owner': owner, 'plan': metadata.receipt(Path(directory) / 'measurement_plan.json'),
            'error_type': type(error).__name__, 'error': str(error), 'provider': evidence,
            'partial_native_state_verified': evidence is not None, 'completed': False,
            'scientific_credit': 0, 'admission_granted': False})
        raise


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


def validate_contract(saved, plan, observer, mapping=None):
    """Check captured public bytes and fixed ABI declarations without importing any native library."""
    require(set(saved) == {'kind', 'contract', 'captured_files', 'source', 'actual_target_observation_only',
        'qualification', 'admission_granted'} and saved['kind'] == 'observed_native_process_target_contract',
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
    """Accept only finite nonnegative real timestamps, never Boolean values."""
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_calls(evidence, owner, allowed, contract):
    """Reconstruct bounded calls and descriptor generations from retained raw fdinfo observations."""
    require(set(evidence) == {'kind', 'contract', 'bootstrap', 'preflight_passed', 'preflight_call_count',
        'calls', 'open_handle_count', 'native_call_failure', 'descriptor_events', 'ordinary_call_count',
        'ordinary_call_limit', 'reserved_cleanup_call_count', 'maximum_cleanup_records_per_open',
        'numeric_pid_signal_fallback', 'python_exports_changed', 'gpu_or_cuda_qualification', 'scientific_credit'}
        and evidence['kind'] == KIND and same(evidence['contract'], contract)
        and evidence['preflight_passed'] is True and type(evidence['preflight_call_count']) is int
        and evidence['preflight_call_count'] == 3 and type(evidence['open_handle_count']) is int
        and evidence['open_handle_count'] == 0 and evidence['native_call_failure'] is False
        and evidence['numeric_pid_signal_fallback'] is False and evidence['python_exports_changed'] is False
        and evidence['gpu_or_cuda_qualification'] is False and type(evidence['scientific_credit']) is int
        and evidence['scientific_credit'] == 0 and type(evidence['ordinary_call_limit']) is int
        and evidence['ordinary_call_limit'] == LIMIT and type(evidence['maximum_cleanup_records_per_open']) is int
        and evidence['maximum_cleanup_records_per_open'] == 3, 'saved native provider boundary differs')
    calls, events = evidence['calls'], evidence['descriptor_events']
    require(type(calls) is list and 3 <= len(calls) <= 4 * LIMIT and type(events) is list
        and len(events) == len(calls), 'every successful saved native call needs one bounded descriptor identity')
    active, by_call, ordinary, cleanup, previous = {}, {}, 0, 0, 0
    for index, event in enumerate(events):
        require(set(event) == {'event_sequence', 'operation', 'descriptor', 'call_sequence', 'open_generation',
            'opened_pid', 'identity', 'observed_at_seconds'} and type(event['event_sequence']) is int
            and event['event_sequence'] == index and type(event['call_sequence']) is int
            and event['call_sequence'] not in by_call and finite_time(event['observed_at_seconds']), 'native identity event is duplicated or malformed')
        by_call[event['call_sequence']] = event
    for index, call in enumerate(calls):
        require(set(call) == {'sequence', 'name', 'phase', 'arguments', 'result', 'errno', 'started_at_seconds',
            'finished_at_seconds', 'producer_pid', 'call_budget'} and type(call['sequence']) is int
            and call['sequence'] == index and type(call['producer_pid']) is int and call['producer_pid'] == owner['pid']
            and type(call['result']) is int and type(call['errno']) is int and call['errno'] == 0
            and call['phase'] == ('self_preflight' if index < 3 else 'owned_process')
            and finite_time(call['started_at_seconds']) and finite_time(call['finished_at_seconds'])
            and previous <= call['started_at_seconds'] <= call['finished_at_seconds'], 'saved native return, time or sequence differs')
        previous_finished = previous
        previous = call['finished_at_seconds'];event = by_call[index];args = call['arguments']
        require(type(args) is dict and event['operation'] == call['name'], 'native call and descriptor observation disagree')
        if call['name'] == 'pidfd_open':
            descriptor = call['result']
            require(type(args.get('pid')) is int and args['pid'] in allowed and descriptor >= 0 and descriptor not in active
                and same(args, {'syscall': 434, 'pid': args['pid'], 'flags': 0}) and call['call_budget'] == 'ordinary',
                'native open is not an exact positively owned process generation')
            active[descriptor] = {'generation': index, 'pid': args['pid'], 'identity': event['identity'], 'cleanup': {15, 9, 'close'}}
            require(event['open_generation'] == index and call['finished_at_seconds'] <= event['observed_at_seconds'],
                'native opened identity must follow its actual open return')
        else:
            descriptor = args.get('fd')
            require(type(descriptor) is int and descriptor in active and call['result'] == 0
                and event['observed_at_seconds'] <= call['started_at_seconds'], 'native operation has no active retained descriptor')
            require(previous_finished <= event['observed_at_seconds'], 'retained identity observation predates the preceding call')
            handle = active[descriptor]
            if call['name'] == 'pidfd_send_signal':
                number = args.get('signal')
                require(type(number) is int and number in (0, 15, 9) and same(args,
                    {'syscall': 424, 'fd': descriptor, 'signal': number, 'siginfo': None, 'flags': 0}), 'native signal arguments differ')
                operation = number
            else:
                require(call['name'] == 'close' and same(args, {'fd': descriptor}), 'unknown native operation')
                operation = 'close'
            require(same({key: event['identity'][key] for key in ('device', 'inode')},
                {key: handle['identity'][key] for key in ('device', 'inode')}), 'retained pidfd image changed within a generation')
            if operation != 0:
                require(call['call_budget'] == 'reserved_descriptor_cleanup' and operation in handle['cleanup'],
                    'native cleanup reservation was absent or reused')
                handle['cleanup'].remove(operation)
                cleanup += 1
            else:require(call['call_budget'] == 'ordinary', 'signal zero cannot consume reserved cleanup')
        handle = active[descriptor];identity = event['identity']
        require(type(event['descriptor']) is int and event['descriptor'] == descriptor
            and type(event['open_generation']) is int and event['open_generation'] == handle['generation']
            and type(event['opened_pid']) is int and event['opened_pid'] == handle['pid']
            and all(type(identity[key]) is int for key in ('device', 'inode', 'pid'))
            and identity['device'] >= 0 and identity['inode'] > 0 and type(identity['raw_fdinfo']) is str
            and len(identity['raw_fdinfo'].encode()) <= 16384
            and hashlib.sha256(identity['raw_fdinfo'].encode()).hexdigest() == identity['fdinfo_sha256']
            and type(identity['flags']) is int and identity['inheritable'] is False
            and identity['pid'] in (handle['pid'], -1), 'actual saved pidfd identity is incomplete or substituted')
        parsed = parse_pidfd_fdinfo(identity['raw_fdinfo'], identity['inode'], handle['pid'])
        require(parsed['pid'] == identity['pid'] and parsed['flags'] == identity['flags'],
            'saved raw pidfd process or flags contradict their retained summaries')
        if call['name'] == 'pidfd_open':require(identity['pid'] == handle['pid'], 'open identity cannot already be exited')
        if handle['pid'] == owner['pid']:
            require(identity['pid'] == owner['pid'], 'the actual running owner cannot have an exited self pidfd')
        if index + 1 < len(calls):
            require(event['observed_at_seconds'] <= calls[index + 1]['started_at_seconds'],
                'a descriptor identity observation crosses the next actual call')
        if call['call_budget'] == 'ordinary':ordinary += 1
        if call['name'] == 'close':del active[descriptor]
    require(not active and ordinary <= LIMIT and type(evidence['ordinary_call_count']) is int
        and evidence['ordinary_call_count'] == ordinary and type(evidence['reserved_cleanup_call_count']) is int
        and evidence['reserved_cleanup_call_count'] == cleanup, 'saved native budgets or retained handle terminal count differ')
    first = calls[:3]
    require([row['name'] for row in first] == ['pidfd_open', 'pidfd_send_signal', 'close']
        and first[0]['arguments']['pid'] == owner['pid'] and first[1]['arguments']['signal'] == 0
        and all(row['phase'] == 'self_preflight' for row in first), 'native self preflight must precede all other operations')
    bootstrap = evidence['bootstrap']
    require(set(bootstrap) == {'symbols', 'sys_executable', 'sys_version', 'pid', 'gpu_or_cuda_qualification'}
        and set(bootstrap['symbols']) == {'syscall', 'close'}
        and type(bootstrap['pid']) is int and bootstrap['pid'] == owner['pid']
        and bootstrap['sys_executable'] == owner['command'][0]
        and owner['executable'] == contract['selected_python']['actual_running_image_path']
        and bootstrap['gpu_or_cuda_qualification'] is False, 'native bootstrap belongs to a different actual process')
    for name in ('syscall', 'close'):
        actual, expected = bootstrap['symbols'][name], contract['symbols'][name]
        fields = actual['actual_mapping'].split(maxsplit=5)
        require(len(fields) == 6 and 'x' in fields[1] and fields[5] == expected['image']['path'],
            'the saved native pointer lacks its actual executable libc mapping')
        first_address, last_address = (int(value, 16) for value in fields[0].split('-'))
        require(type(actual['actual_pointer']) is int and first_address <= actual['actual_pointer'] < last_address
            and same(actual['image'], expected['image']) and type(actual['symbol_file_offset']) is int
            and actual['symbol_file_offset'] == expected['symbol_file_offset']
            == actual['actual_pointer'] - first_address + int(fields[2], 16)
            and int(fields[4]) == expected['image']['inode']
            and tuple(int(value, 16) for value in fields[3].split(':'))
            == (os.major(expected['image']['device']), os.minor(expected['image']['device'])),
            'saved native pointer, inode, device or symbol offset does not join its captured image')
    return {'calls': len(calls), 'ordinary_calls': ordinary, 'cleanup_calls': cleanup}


def validate_scope_snapshot(record, uid, cgroup, observer):
    """Check the scope producer's saved schema without inventing absent raw command fields."""
    fields = {'pid', 'parent_pid', 'process_group', 'session', 'process_start_time', 'state', 'uids', 'cgroup',
        'stat_before', 'stat_after', 'status', 'status_after', 'command', 'command_raw_sha256', 'observed_at_seconds'}
    require(type(record) is dict and set(record) == fields and type(uid) is int and uid >= 0
        and type(cgroup) is str and bool(cgroup)
        and all(type(record[key]) is str and 0 < len(record[key].encode()) <= observer.MAX_BYTES
            for key in ('stat_before', 'stat_after', 'status', 'status_after')),
        'saved scoped process requires its exact bounded stat/status schema')
    # Scope snapshots retain the final state; a scheduling state may change between the two reads.
    before, after = observer.parsed_stat(record['stat_before']), observer.parsed_stat(record['stat_after'])
    require(all(before[key] == after[key] for key in ('pid', 'parent_pid', 'process_start_time'))
        and all(type(record[key]) is type(value) and record[key] == value for key, value in after.items())
        and type(record['pid']) is int and record['pid'] > 0
        and type(record['state']) is str and len(record['state']) == 1
        and record['state'] == record['stat_after'].rsplit(') ', 1)[1].split()[0]
        and type(record['uids']) is list and len(record['uids']) == 4
        and all(type(value) is int for value in record['uids'])
        and observer.parsed_uids(record['status']) == observer.parsed_uids(record['status_after'])
        == record['uids'] == [uid] * 4 and record['cgroup'] == cgroup
        and finite_time(record['observed_at_seconds']),
        'saved scoped process differs from its raw PID, birth, user identifiers, state or cgroup')
    # Before: ['tool', '--version']; after: hash of Linux argument bytes, never a fabricated cmdline field.
    command = record['command']
    require(command is None or type(command) is list and bool(command)
        and all(type(part) is str and '\0' not in part for part in command) and bool(command[0]),
        'saved scoped command must be a typed argument list or an actual empty-command observation')
    encoded = b'' if command is None else b'\0'.join(part.encode() for part in command) + b'\0'
    require(type(record['command_raw_sha256']) is str
        and re.fullmatch('[0-9a-f]{64}', record['command_raw_sha256'])
        and hashlib.sha256(encoded).hexdigest() == record['command_raw_sha256'],
        'saved scoped command arguments differ from their retained actual command hash')
    return {key: record[key] for key in ('pid', 'parent_pid', 'process_start_time', 'uids', 'cgroup')}


def validate_saved(directory, plan, observer, intent, observation, launch, waited, mapping=None, diagnostic_deadline=None):
    """Read all three actual role records without loading a library or observing any historical live process."""
    policy(plan);directory = Path(directory)
    budgeted = 'diagnostic_time_budget' in plan
    require(budgeted == (diagnostic_deadline is not None), 'saved native roles need their exact declared deadline proof')
    owners = {'producer': intent['producer'], 'waiter': launch['waiter'], 'capture': launch['child']}
    contract_ref, unused = reference(directory / 'native_process_target_contract.json', observer, mapping)
    contract = validate_contract(read_reference(contract_ref, observer, mapping), plan, observer, mapping)
    scope_ref, unused = reference(directory / 'tool_descendant_scope_terminal.json', observer, mapping)
    terminal_scope = read_reference(scope_ref, observer, mapping)
    allowed = {owner['pid']: owner for owner in owners.values()}
    for node in terminal_scope['positive_nodes'].values():
        record = node['initial']
        validate_scope_snapshot(record, owners['producer']['uids'][0], owners['producer']['cgroup'], observer)
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
        expected_command = intent['capture_command'] if role == 'capture' else intent['waiter_command'] if role == 'waiter' else [
            plan['python'], '-I', '-S', '-B', str(directory / 'code/diagnose_direct_nsight_process.py'), '--producer-unit', str(directory)]
        observer.validate_snapshot(begin['owner'], owner['uids'][0], owner['cgroup'])
        require(set(begin) == {'kind', 'schema_version', 'role', 'plan', 'policy', 'owner', 'contract', 'provider',
            'started_at_seconds', 'published_at_seconds', 'scientific_credit', 'admission_granted'} | ({'diagnostic_deadline'} if budgeted else set())
            and begin['kind'] == 'actual_native_process_role_begin'
            and owner['command'] == expected_command and observer.stable(begin['owner']) == observer.stable(owner)
            and begin['role'] == role and same(begin['policy'], policy(plan)) and same(begin['contract'], contract_ref)
            and same(begin['plan'], observer.receipt(directory / 'measurement_plan.json'))
            and finite_time(begin['started_at_seconds']) and finite_time(begin['published_at_seconds'])
            and begin['started_at_seconds'] <= begin['published_at_seconds'], 'native begin has different source, plan, owner or chronology')
        require(set(terminal) == {'kind', 'schema_version', 'role', 'begin', 'owner', 'references', 'provider',
            'completed_at_seconds', 'completed', 'scientific_credit', 'admission_granted'} | ({'diagnostic_deadline'} if budgeted else set())
            and terminal['kind'] == 'actual_native_process_role_terminal'
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
        validate_calls(begin['provider'], owner, {owner['pid']: owner}, contract)
        require(len(begin['provider']['calls']) == 3, 'native begin must contain only its actual self preflight')
        require(begin['started_at_seconds'] <= begin['provider']['calls'][0]['started_at_seconds']
            and begin['provider']['calls'][-1]['finished_at_seconds'] <= begin['published_at_seconds'],
            'native self preflight must occur inside the recorded actual startup window')
        role_allowed = allowed if role == 'producer' else {owner['pid']: owner, owners['producer']['pid']: owners['producer']}
        counts = validate_calls(terminal['provider'], owner, role_allowed, contract)
        opened = [call['arguments']['pid'] for call in terminal['provider']['calls'] if call['name'] == 'pidfd_open']
        if role == 'producer':
            require(opened.count(owner['pid']) >= 2 and all(opened.count(pid) >= 1 for pid in role_allowed),
                'native producer evidence omits a known scope or ancestry process handle')
            require(all(opened.count(record['pid']) >= 2 for record in observation['nodes'].values()
                if str(record['pid']) in terminal_scope['positive_nodes']),
                'native evidence omits the separate scope and ancestry handles')
        else:
            require(opened == [owner['pid'], owners['producer']['pid']],
                'native child evidence must contain its self preflight and exact known-producer watcher')
            require(all(event['identity']['pid'] == event['opened_pid']
                for event in terminal['provider']['descriptor_events']),
                'a successful native child cannot retain an exited known producer')
        require(not any(call['name'] == 'pidfd_send_signal' and call['arguments']['signal'] in (15, 9)
            for call in terminal['provider']['calls']), 'successful native diagnostic evidence cannot hide cleanup intervention')
        require(terminal['provider']['calls'][-1]['finished_at_seconds'] <= terminal['completed_at_seconds'],
            'native terminal predates its last actual call')
        require(same(terminal['provider']['calls'][:3], begin['provider']['calls'])
            and same(terminal['provider']['descriptor_events'][:3], begin['provider']['descriptor_events']),
            'native terminal changed its frozen self preflight')
        required = ('tool_descendant_scope_terminal.json', 'tool_descendant_scope_restored.json', 'profiler_actual_wait.json') if role == 'producer' else (
            ('capture_actual_launch.json', 'capture_actual_wait.json') if role == 'waiter' else ('capture_release_observed.json', 'probe_result.json'))
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
    payload_ref, unused = reference(directory / 'probe_result.json', observer, mapping)
    payload = read_reference(payload_ref, observer, mapping)
    if budgeted:diagnostic_deadline.verify_snapshot(payload['diagnostic_deadline'], phase='work')
    require(begin['waiter']['published_at_seconds'] <= begin['capture']['started_at_seconds']
        and begin['capture']['published_at_seconds'] <= payload['payload_started_at_seconds']
        <= payload['payload_finished_at_seconds'] <= terminal['capture']['completed_at_seconds']
        <= terminal['waiter']['completed_at_seconds'] <= terminal['producer']['completed_at_seconds'],
        'native role startup, payload and terminal chronology is incomplete')
    return {'kind': 'saved_native_diagnostic_three_role_process_evidence', 'passed': True, 'roles': result,
        'target_contract': contract_ref, 'qualification': False, 'admission_granted': False, 'scientific_credit': 0}
