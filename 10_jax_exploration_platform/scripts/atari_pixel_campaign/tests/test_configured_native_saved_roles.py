"""Exercise configured native saved-role dispatch using explicitly synthetic metadata only."""
import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from test_configured_native_process import declared_plan, native, identity, evidence
from test_native_process_role_evidence import NativeSavedRoleTests, saved_provider
import direct_profile_process as timing


class ConfiguredNativeSavedRoleTests(unittest.TestCase):
    """Keep raw identity, syscall and role-reference gates active while target bytes stay synthetic."""
    def setUp(self):
        """Convert the older synthetic fixture to actual scope-schema shapes without changing production readers."""
        fixture = NativeSavedRoleTests(methodName='runTest');fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture = fixture;fixture.plan = declared_plan()
        python = fixture.plan['python'];directory = fixture.directory
        commands = dict(zip(('waiter', 'capture'), identity.commands(directory, fixture.plan)))
        commands['producer'] = native.producer_command(directory, fixture.plan)
        for role in ('producer', 'waiter', 'capture'):
            row = fixture.owners[role];path = fixture.proc / str(row['pid']) / 'cmdline'
            path.write_bytes(b'\0'.join(value.encode() for value in commands[role]) + b'\0')
            fixture.owners[role] = identity.ProcReader(fixture.proc).snapshot(row['pid'], 888)
        fixture.intent = {'parent': fixture.owners['producer'], 'waiter_command': commands['waiter'], 'capture_command': commands['capture']}
        fixture.observation = {'nodes': {str(fixture.owners[role]['pid']): fixture.owners[role] for role in ('tool', 'waiter', 'capture')}}
        fixture.launch = {'waiter': fixture.owners['waiter'], 'child': fixture.owners['capture']}
        fixture.waited = {'returncode': 0, 'exited_normally': True, 'interventions': [], 'actual_wait_finished_at_seconds': 62.5}
        fixture.save('capture_actual_wait.json', fixture.waited)
        fixture.save('measurement_plan.json', fixture.plan);fixture.save('capture_actual_launch.json', fixture.launch)
        # Before: raw ancestry snapshots with cmdline/executable; after: the distinct bounded scope schema.
        nodes = {}
        for role in ('tool', 'waiter', 'capture'):
            row = fixture.owners[role]
            scoped = {key: row[key] for key in ('pid', 'parent_pid', 'process_group', 'session', 'process_start_time',
                'uids', 'cgroup', 'stat_before', 'stat_after', 'status', 'status_after', 'command', 'observed_at_seconds')}
            scoped.update(state=row['stat_after'].rsplit(') ', 1)[1].split()[0], command_raw_sha256=row['cmdline_sha256'])
            nodes[str(row['pid'])] = {'initial': scoped}
        fixture.save('tool_descendant_scope_terminal.json', {'positive_nodes': nodes})
        self.clock = timing.DiagnosticDeadline({'started_monotonic': 0., 'clock_domain': {'fixture_only': True}},
            {'path': '/synthetic/origin', 'sha256': '0' * 64})
        self.snapshot = {'origin': self.clock.reference, 'clock_domain': self.clock.origin['clock_domain'],
            'started_monotonic': 0., 'work_end_monotonic': 240., 'total_end_monotonic': 300.,
            'observed_monotonic': 50., 'cleanup_started_monotonic': None, 'cleanup_end_monotonic': None}
        fixture.save('native_capture_payload_timing.json', {'payload_started_at_seconds': 50., 'payload_finished_at_seconds': 51.,
            'diagnostic_deadline': self.snapshot})
        for role, start, published, completed in [('producer', 10., 13., 100.), ('waiter', 30., 33., 90.), ('capture', 40., 43., 80.)]:
            owner = fixture.owners[role]
            begin, terminal = fixture.begins[role], fixture.terminals[role]
            begin.update(plan=identity.receipt(directory / 'measurement_plan.json'), policy=native.policy(fixture.plan), owner=owner,
                provider=saved_provider(owner, fixture.contract, start), diagnostic_deadline={**self.snapshot, 'observed_monotonic': published})
            paths = evidence.role_files(directory, role);fixture.save(paths['begin'].name, begin)
            targets = (101, 104, 102, 103, 104, 102, 103) if role == 'producer' else (101, 103) if role == 'waiter' else (101,)
            terminal.update(begin=identity.receipt(paths['begin']), owner=owner, provider=saved_provider(owner, fixture.contract, start, targets),
                references={name: identity.receipt(directory / name) for name in native.completion_references(fixture.plan, role)},
                diagnostic_deadline={**self.snapshot, 'observed_monotonic': completed})
            fixture.save(paths['terminal'].name, terminal)

    def read_saved(self):
        """Patch only synthetic image-byte validation, retaining all configured role and native descriptor checks."""
        f = self.fixture
        with patch.object(evidence, 'validate_contract', return_value=f.contract):
            return evidence.validate_saved(f.directory, f.plan, identity, f.intent, f.observation, f.launch, f.waited,
                diagnostic_deadline=self.clock)

    def test_configured_three_roles_use_exact_commands_and_extra_owned_child_handle(self):
        """Configured waiter retains its exact capture pidfd and all three roles keep complete closes."""
        actual = self.read_saved()
        self.assertIs(actual['passed'], True);self.assertIs(actual['admission_granted'], False)
        self.assertEqual(set(actual['roles']), {'producer', 'waiter', 'capture'})

    def test_waiter_omitting_its_actual_capture_handle_refuses(self):
        """A successful self/producer watcher cannot replace the configured waiter's actual child handle."""
        f = self.fixture;terminal = f.terminals['waiter']
        terminal['provider'] = saved_provider(f.owners['waiter'], f.contract, 30., (101,))
        f.save(evidence.role_files(f.directory, 'waiter')['terminal'].name, terminal)
        with self.assertRaisesRegex(ValueError, 'known-producer watcher'):self.read_saved()

    def test_direct_h100_producer_command_cannot_replace_configured_producer(self):
        """An otherwise valid native role is refused when its actual producer source command belongs to H100."""
        f = self.fixture;f.intent['parent'] = copy.deepcopy(f.owners['producer'])
        f.intent['parent']['command'][-3] = str(f.directory / 'code/diagnose_direct_nsight_process.py')
        with self.assertRaises(ValueError):self.read_saved()

    def test_missing_payload_timing_reference_refuses(self):
        """A capture result alone cannot replace the declared pre/post-payload timing boundary."""
        f = self.fixture;del f.terminals['capture']['references']['native_capture_payload_timing.json']
        f.save(evidence.role_files(f.directory, 'capture')['terminal'].name, f.terminals['capture'])
        with self.assertRaisesRegex(ValueError, 'completion records'):self.read_saved()

    def test_exited_capture_close_needs_exact_successful_prior_wait(self):
        """Only the actual waiter's own exited capture descriptor may report minus one after its real wait."""
        f = self.fixture;terminal = f.terminals['waiter']
        event = terminal['provider']['descriptor_events'][-1]
        self.assertEqual((event['operation'], event['opened_pid']), ('close', f.owners['capture']['pid']))
        raw = event['identity']['raw_fdinfo'].replace('Pid:\t103\n', 'Pid:\t-1\n')
        event['identity'].update(pid=-1, raw_fdinfo=raw, fdinfo_sha256=hashlib.sha256(raw.encode()).hexdigest())
        f.save(evidence.role_files(f.directory, 'waiter')['terminal'].name, terminal)
        self.assertTrue(self.read_saved()['passed'])
        f.waited['actual_wait_finished_at_seconds'] = event['observed_at_seconds'] + 1
        with self.assertRaisesRegex(ValueError, 'prior Popen wait'):self.read_saved()
        del f.waited['actual_wait_finished_at_seconds']
        with self.assertRaisesRegex(ValueError, 'prior Popen wait'):self.read_saved()


if __name__ == '__main__':unittest.main()
