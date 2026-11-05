"""Test synthetic three-role saved evidence with the separate target-contract gate explicitly patched."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_native_process_evidence import declared_plan, evidence, synthetic_calls
from test_nsight_process_identity import M, stat_text


def saved_provider(owner, contract, started, additional_targets=()):
    """Create synthetic successful calls and descriptor observations without touching a real process."""
    value, unused_owner, unused_allowed, unused_contract = synthetic_calls()
    value['contract'] = copy.deepcopy(contract)
    value['bootstrap']['pid'] = owner['pid']
    value['bootstrap']['sys_executable'] = owner['command'][0]
    for call, event in zip(value['calls'], value['descriptor_events']):
        call['producer_pid'] = owner['pid']
        if call['name'] == 'pidfd_open':call['arguments']['pid'] = owner['pid']
        shift = started - 10.0
        call['started_at_seconds'] += shift;call['finished_at_seconds'] += shift
        event['observed_at_seconds'] += shift
        event['opened_pid'] = owner['pid']
        raw = event['identity']['raw_fdinfo'].replace('101', str(owner['pid']))
        event['identity'].update(pid=owner['pid'], raw_fdinfo=raw, fdinfo_sha256=hashlib.sha256(raw.encode()).hexdigest())
    # Before: one completed synthetic self preflight; after: separately closed synthetic process generations.
    for cycle, pid in enumerate(additional_targets):
        generation = len(value['calls'])
        raw = 'pos:\t0\nflags:\t02000002\nmnt_id:\t5\nino:\t' + str(1000 + generation) + '\nPid:\t' + str(pid) + '\n'
        identity = {'device': 6, 'inode': 1000 + generation, 'pid': pid, 'flags': 0o2000002, 'inheritable': False, 'raw_fdinfo': raw,
            'fdinfo_sha256': hashlib.sha256(raw.encode()).hexdigest()}
        for offset, (name, arguments, returned, budget) in enumerate([
                ('pidfd_open', {'syscall': 434, 'pid': pid, 'flags': 0}, 7, 'ordinary'),
                ('close', {'fd': 7}, 0, 'reserved_descriptor_cleanup')]):
            sequence = len(value['calls']);at = 60.0 + cycle * 2 + offset
            value['calls'].append({'sequence': sequence, 'name': name, 'phase': 'owned_process',
                'arguments': arguments, 'result': returned, 'errno': 0, 'started_at_seconds': at,
                'finished_at_seconds': at + .1, 'producer_pid': owner['pid'], 'call_budget': budget})
            value['descriptor_events'].append({'event_sequence': sequence, 'operation': name,
                'descriptor': 7, 'call_sequence': sequence, 'open_generation': generation, 'opened_pid': pid,
                'identity': copy.deepcopy(identity), 'observed_at_seconds': at + .15 if offset == 0 else at - .1})
        value['ordinary_call_count'] += 1;value['reserved_cleanup_call_count'] += 1
    return value


class NativeSavedRoleTests(unittest.TestCase):
    """Exercise whole-role joins while retaining raw snapshot, call, reference and chronology checks."""
    def setUp(self):
        """Write explicitly synthetic unit files and raw process snapshots in a private temporary folder."""
        self.temporary = tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / 'synthetic_unit';self.directory.mkdir()
        (self.directory / 'code').mkdir()
        self.proc = self.root / 'synthetic_proc';self.proc.mkdir()
        self.plan = declared_plan()
        entry = str(self.directory / 'code/diagnose_direct_nsight_process.py')
        python = self.plan['python']
        commands = {
            'producer': [python, '-I', '-S', '-B', entry, '--producer-unit', str(self.directory)],
            'waiter': [python, '-I', '-S', '-B', entry, '--direct-owned-waiter-unit', str(self.directory)],
            'capture': [python, '-I', '-S', '-B', entry, '--direct-owned-capture-unit', str(self.directory)],
            'tool': ['/synthetic/nsys', 'profile', 'explicit-synthetic-fixture']}
        self.cgroup = '0::/synthetic/native-role-reader/no-live-process\n'
        self.owners = {}
        for role, pid, parent, birth in [('producer', 101, 100, 10), ('tool', 104, 101, 15),
                ('waiter', 102, 104, 20), ('capture', 103, 102, 30)]:
            node = self.proc / str(pid);node.mkdir()
            (node / 'stat').write_text(stat_text(pid, parent, pid, pid, birth))
            (node / 'status').write_text('Name:\tsynthetic\nUid:\t888\t888\t888\t888\n')
            (node / 'cgroup').write_text(self.cgroup)
            (node / 'cmdline').write_bytes(b'\0'.join(word.encode() for word in commands[role]) + b'\0')
            (node / 'exe').symlink_to(python if role != 'tool' else '/synthetic/nsys')
            self.owners[role] = M.ProcReader(self.proc).snapshot(pid, 888)
        self.intent = {'producer': self.owners['producer'], 'waiter_command': commands['waiter'],
            'capture_command': commands['capture']}
        self.observation = {'nodes': {str(self.owners[role]['pid']): self.owners[role]
            for role in ('tool', 'waiter', 'capture')}}
        self.launch = {'waiter': self.owners['waiter'], 'child': self.owners['capture']}
        self.waited = {'explicit_synthetic_supporting_wait_gate_prevalidated': True}
        unused, unused_owner, unused_allowed, self.contract = synthetic_calls()
        self.save('measurement_plan.json', self.plan)
        self.save('native_process_target_contract.json', {'explicit_synthetic_contract_byte_gate_patched': True})
        self.save('tool_descendant_scope_terminal.json', {'positive_nodes': {
            str(self.owners[role]['pid']): {'initial': self.owners[role]}
            for role in ('tool', 'waiter', 'capture')}})
        self.save('tool_descendant_scope_begin.json', {'established_at_seconds': 15.0})
        self.save('tool_descendant_scope_restored.json', {'explicit_synthetic_supporting_scope_gate_prevalidated': True})
        self.save('profiler_actual_wait.json', self.waited)
        self.save('capture_actual_launch.json', self.launch)
        self.save('capture_actual_wait.json', self.waited)
        self.save('capture_release_observed.json', {'explicit_synthetic_supporting_release_gate_prevalidated': True})
        self.save('probe_result.json', {'payload_started_at_seconds': 50.0, 'payload_finished_at_seconds': 51.0})
        self.begins, self.terminals = {}, {}
        for role, start, published, completed in [('producer', 10.0, 13.0, 100.0),
                ('waiter', 30.0, 33.0, 90.0), ('capture', 40.0, 43.0, 80.0)]:
            owner = self.owners[role]
            begin_provider = saved_provider(owner, self.contract, start)
            targets = (101, 104, 102, 103, 104, 102, 103) if role == 'producer' else (101,)
            terminal_provider = saved_provider(owner, self.contract, start, targets)
            paths = evidence.role_files(self.directory, role)
            begin = {'kind': 'actual_native_process_role_begin', 'schema_version': 1, 'role': role,
                'plan': M.receipt(self.directory / 'measurement_plan.json'), 'policy': evidence.policy(self.plan),
                'owner': owner, 'contract': M.receipt(self.directory / 'native_process_target_contract.json'),
                'provider': begin_provider, 'started_at_seconds': start - 1,
                'published_at_seconds': published, 'scientific_credit': 0, 'admission_granted': False}
            self.save(paths['begin'].name, begin)
            names = ('tool_descendant_scope_terminal.json', 'tool_descendant_scope_restored.json', 'profiler_actual_wait.json') if role == 'producer' else (
                ('capture_actual_launch.json', 'capture_actual_wait.json') if role == 'waiter' else ('capture_release_observed.json', 'probe_result.json'))
            terminal = {'kind': 'actual_native_process_role_terminal', 'schema_version': 1, 'role': role,
                'begin': M.receipt(paths['begin']), 'owner': owner,
                'references': {name: M.receipt(self.directory / name) for name in names},
                'provider': terminal_provider, 'completed_at_seconds': completed, 'completed': True,
                'scientific_credit': 0, 'admission_granted': False}
            self.save(paths['terminal'].name, terminal)
            self.begins[role], self.terminals[role] = begin, terminal

    def save(self, name, value):
        """Write or change one explicitly synthetic fixture file, never a campaign artifact."""
        path = self.directory / name
        path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

    def read_saved(self):
        """Patch only target contract-byte validation; keep every role and native-call predicate active."""
        with patch.object(evidence, 'validate_contract', return_value=self.contract) as supporting_gate:
            result = evidence.validate_saved(self.directory, self.plan, M, self.intent, self.observation,
                self.launch, self.waited)
        supporting_gate.assert_called_once()
        return result

    def test_three_roles_keep_exact_owned_opens_wait_references_and_chronology(self):
        """A complete synthetic graph passes the actual role, raw snapshot and native-call reader."""
        result = self.read_saved()
        self.assertIs(result['passed'], True)
        self.assertEqual(set(result['roles']), {'producer', 'waiter', 'capture'})
        self.assertIs(result['qualification'], False)
        self.assertIs(result['admission_granted'], False)

    def test_missing_producer_watcher_calls_refuse_with_other_fields_valid(self):
        """A child's valid self preflight cannot replace its required known-producer watcher handle."""
        terminal = self.terminals['capture']
        terminal['provider'] = copy.deepcopy(self.begins['capture']['provider'])
        self.save(evidence.role_files(self.directory, 'capture')['terminal'].name, terminal)
        with self.assertRaisesRegex(ValueError, 'known-producer watcher'):self.read_saved()

    def test_changed_role_begin_refuses_even_when_terminal_references_new_bytes(self):
        """A fresh hash does not permit a Boolean schema version in an otherwise complete role."""
        self.begins['waiter']['schema_version'] = True
        paths = evidence.role_files(self.directory, 'waiter')
        self.save(paths['begin'].name, self.begins['waiter'])
        self.terminals['waiter']['begin'] = M.receipt(paths['begin'])
        self.save(paths['terminal'].name, self.terminals['waiter'])
        with self.assertRaises(ValueError):self.read_saved()

    def test_missing_terminal_never_becomes_a_completed_role(self):
        """A complete begin and other roles cannot replace a missing actual terminal file."""
        evidence.role_files(self.directory, 'capture')['terminal'].unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):self.read_saved()

    def test_payload_before_capture_bootstrap_refuses_after_reference_update(self):
        """Exact new payload bytes still fail if work started before the capture provider was ready."""
        self.save('probe_result.json', {'payload_started_at_seconds': 42.0, 'payload_finished_at_seconds': 51.0})
        self.terminals['capture']['references']['probe_result.json'] = M.receipt(self.directory / 'probe_result.json')
        self.save(evidence.role_files(self.directory, 'capture')['terminal'].name, self.terminals['capture'])
        with self.assertRaisesRegex(ValueError, 'chronology'):self.read_saved()

    def test_partial_failure_record_refuses_a_previously_complete_terminal(self):
        """A saved failure cannot be ignored because a terminal was published earlier."""
        self.save(evidence.role_files(self.directory, 'capture')['failure'].name,
            {'explicit_synthetic_partial_failure_fixture': True})
        with self.assertRaisesRegex(ValueError, 'failed native role'):self.read_saved()

    def test_unknown_terminal_fields_refuse(self):
        """Extra terminal fields cannot silently introduce another completion interpretation."""
        self.terminals['producer']['unknown_completion_override'] = True
        self.save(evidence.role_files(self.directory, 'producer')['terminal'].name, self.terminals['producer'])
        with self.assertRaises(ValueError):self.read_saved()


if __name__ == '__main__':
    unittest.main()
