"""Read actual failed diagnostic schemas without a live process, native call or tool operation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import native_process_evidence as evidence
import nsight_process_identity as observer

REVIEW = SCRIPTS.parents[2] / '.review_metadata/native_saved_scope_records'
FIXTURES = REVIEW / 'failed_attempt_preserved'
HASHES = {
    'tool_descendant_scope_terminal.json': 'de5d2bdd5cf831005547436e8f6a1a9bbb201b0ecc107f32501e08badb798deb',
    'direct_profile_process_intent.json': '0e38a38edd36be45bdf5f6acdbc98afbab766323529539ea51ef81a2f2afa772',
    'capture_process_observation.json': '9ad159a5b9cf14b2fc29a29b57eaff1856ed8d4aa543bac2a3133f2915b14682',
    'failure.json': 'e60545743f1f7520c2de917590e4d22afdeb498c6dfb76321287826ebfe2d406',
}


def actual(name):
    """Pin exact retained actual bytes before using them as read-only regression inputs."""
    raw = (FIXTURES / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != HASHES[name]:
        raise ValueError('actual schema regression input changed: ' + name)
    return json.loads(raw)


class ActualSavedScopeRecordsTests(unittest.TestCase):
    """Keep distinct scope and ancestry schemas strict while preserving the failed attempt."""
    @classmethod
    def setUpClass(cls):
        """Load only published actual JSON fields, with no current process query."""
        cls.scope = actual('tool_descendant_scope_terminal.json')
        cls.intent = actual('direct_profile_process_intent.json')
        cls.producer = cls.intent['producer']
        cls.uid, cls.cgroup = cls.producer['uids'][0], cls.producer['cgroup']
        cls.record = next(iter(cls.scope['positive_nodes'].values()))['initial']

    def check_record(self, record):
        """Validate one exact scope record through the corrected pure reader."""
        return evidence.validate_scope_snapshot(record, self.uid, self.cgroup, observer)

    def test_actual_initial_and_final_scope_records(self):
        """All retained scope identities pass their actual schema without fabricated cmdline bytes."""
        for node in self.scope['positive_nodes'].values():
            for stage in ('initial', 'last'):
                with self.subTest(pid=node[stage]['pid'], stage=stage):
                    record = copy.deepcopy(node[stage]);before = copy.deepcopy(record)
                    checked = self.check_record(record)
                    self.assertEqual(checked['pid'], record['pid'])
                    self.assertEqual(record, before)
                    self.assertNotIn('cmdline', record)

    def test_original_reader_refuses_scope_but_accepts_actual_full_records(self):
        """The original raw-command ancestry and owner proof stays unchanged and required."""
        with self.assertRaises(KeyError):observer.validate_snapshot(self.record, self.uid, self.cgroup)
        observer.validate_snapshot(self.producer, self.uid, self.cgroup)
        observed = actual('capture_process_observation.json')
        for record in observed['nodes'].values():
            observer.validate_snapshot(record, self.uid, self.cgroup)
            with self.assertRaises(ValueError):self.check_record(record)

    def test_raw_uid_and_expected_owner_changes_refused(self):
        """Summary or retained status changes cannot authorize a foreign process."""
        for field in ('uids', 'status', 'status_after'):
            record = copy.deepcopy(self.record)
            if field == 'uids':record[field] = [self.uid + 1] * 4
            else:record[field] = record[field].replace('Uid:\t' + str(self.uid), 'Uid:\t' + str(self.uid + 1), 1)
            with self.subTest(field=field), self.assertRaises(ValueError):self.check_record(record)
        with self.assertRaises(ValueError):
            evidence.validate_scope_snapshot(self.record, self.uid + 1, self.cgroup, observer)

    def test_raw_stat_birth_and_parent_changes_refused(self):
        """Raw birth/parent and every retained typed identity must agree."""
        for field in ('pid', 'parent_pid', 'process_group', 'session', 'process_start_time', 'state'):
            record = copy.deepcopy(self.record)
            record[field] = '0' if type(record[field]) is str else record[field] + 1
            with self.subTest(field=field), self.assertRaises(ValueError):self.check_record(record)
        for field in ('stat_before', 'stat_after'):
            record = copy.deepcopy(self.record)
            first, remaining = record[field].rsplit(') ', 1);parts = remaining.split()
            parts[19] = str(int(parts[19]) + 1);record[field] = first + ') ' + ' '.join(parts)
            with self.subTest(field=field), self.assertRaises(ValueError):self.check_record(record)

    def test_command_hash_cgroup_and_types_refused(self):
        """Altered commands, cgroups, Boolean identities and missing fields fail directly."""
        changes = [('command_raw_sha256', '0' * 64), ('command', ['changed-tool']), ('command', [1]),
            ('command', []), ('command', ['tool\0other']), ('cgroup', '/foreign/cgroup'), ('pid', True),
            ('uids', [True] * 4), ('observed_at_seconds', True)]
        for field, value in changes:
            record = copy.deepcopy(self.record);record[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):self.check_record(record)
        record = copy.deepcopy(self.record);del record['stat_before']
        with self.assertRaises(ValueError):self.check_record(record)

    def test_actual_empty_command_observation_remains_exact(self):
        """An actual empty scope command requires its actual empty-byte hash, with no claimed argv."""
        empty = [node['last'] for node in self.scope['positive_nodes'].values() if node['last']['command'] is None]
        self.assertTrue(empty)
        for record in empty:
            self.check_record(record)
            self.assertEqual(record['command_raw_sha256'], hashlib.sha256(b'').hexdigest())

    def test_changed_actual_input_and_reference_refused(self):
        """A changed fixture or scope reference cannot use the accepted actual input identity."""
        changed = copy.deepcopy(self.scope);changed['positive_nodes'] = {}
        self.assertNotEqual(hashlib.sha256(json.dumps(changed).encode()).hexdigest(),
            HASHES['tool_descendant_scope_terminal.json'])
        reference = {'path': str(FIXTURES / 'tool_descendant_scope_terminal.json'), 'sha256': '0' * 64}
        with self.assertRaises(ValueError):evidence.read_reference(reference, observer)

    def test_actual_attempt_is_still_failed_and_zero_credit(self):
        """A reader regression never reclassifies the failed producer or grants any study credit."""
        failure = actual('failure.json')
        self.assertEqual(failure['error_type'], 'KeyError')
        self.assertEqual(failure['error'], "'cmdline'")
        self.assertIs(failure['measurement_valid'], False)
        self.assertIs(failure['admission_granted'], False)
        self.assertEqual(failure['environment_transitions'], 0)


if __name__ == '__main__':
    unittest.main()
