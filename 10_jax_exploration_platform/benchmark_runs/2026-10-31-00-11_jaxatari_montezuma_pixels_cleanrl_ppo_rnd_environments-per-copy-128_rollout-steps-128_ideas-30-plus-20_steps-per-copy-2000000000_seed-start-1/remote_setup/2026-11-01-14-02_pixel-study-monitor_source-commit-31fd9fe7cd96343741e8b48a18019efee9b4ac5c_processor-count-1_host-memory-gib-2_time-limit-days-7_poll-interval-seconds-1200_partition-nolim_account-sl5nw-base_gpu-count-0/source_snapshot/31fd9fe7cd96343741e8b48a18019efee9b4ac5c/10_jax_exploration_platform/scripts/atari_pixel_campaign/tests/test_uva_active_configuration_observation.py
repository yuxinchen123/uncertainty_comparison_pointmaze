"""Check observer ownership, exact preparation and read-only path limits with synthetic metadata."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import uva_active_configuration_observation as observation


class ConfigurationObservationTests(unittest.TestCase):
    """Keep every source repository, ledger and observed file inside a temporary metadata fixture."""

    def setUp(self):
        """Build a real local detached source fixture without publishing or querying any cluster."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        path = self.source / observation.PREFIX / observation.FILENAME
        path.parent.mkdir(parents=True)
        path.write_bytes(Path(observation.__file__).read_bytes())
        subprocess.run(['git', 'init', '-q', str(self.source)], check=True)
        command = ['git', '-C', str(self.source)]
        subprocess.run(command + ['add', '.'], check=True)
        subprocess.run(command + ['-c', 'user.name=Metadata Fixture', '-c', 'user.email=metadata@example.invalid',
            'commit', '-qm', 'Record synthetic observer source'], check=True)
        self.commit = subprocess.check_output(command + ['rev-parse', 'HEAD'], text=True).strip()
        subprocess.run(command + ['checkout', '--detach', '-q', 'HEAD'], check=True)
        self.run_dir = self.root / 'campaign'
        self.run_dir.mkdir()

    def prepare(self):
        """Prepare actual standard records and source checks against the detached synthetic repository."""
        return observation.prepare(self.run_dir, 'synthetic_configuration_observation', self.source,
            self.commit, Path(sys.executable))

    def process(self, cpus=2):
        """Describe a synthetic own numeric process, with two actual simultaneous threads permitted."""
        return {'hostname': 'jaguar03', 'pid': 500, 'uid': 2618919, 'effective_uid': 2618919,
            'uid_fields': [2618919] * 4, 'parent_pid': 499, 'process_start_time': '10000',
            'raw_proc_stat': '500 (python) ' + ' '.join(['R', '499', *['0'] * 17, '10000']) + '\n',
            'cgroup': '0::/system.slice/slurmstepd.scope/job_123/step_0/user/task_0\n',
            'cpu_affinity': list(range(cpus))}

    def record(self, cpus=2):
        """Give the reader explicitly synthetic owned one-requested-CPU, zero-GPU scheduler rows."""
        job = ('JobId=123 UserId=sl5nw(2618919) JobState=RUNNING NodeList=jaguar03 Account=sl5nw_base '
            'Partition=gpu QOS=csresnolim NumNodes=1 NumTasks=1 CPUs/Task=1 TimeLimit=00:05:00 Requeue=0 '
            'Reservation=sl5nw_168 NumCPUs=' + str(cpus) + ' ReqTRES=cpu=1,mem=2G,node=1 '
            'AllocTRES=cpu=' + str(cpus) + ',mem=2G,node=1,billing=1')
        step = 'StepId=123.0 State=RUNNING NodeList=jaguar03 UserId=2618919 TRES=cpu=' + str(cpus) + ',mem=2G,node=1'
        return {'job_id': '123', 'step_id': '0', 'process': self.process(cpus),
            'environment': {'SLURM_JOB_ID': '123', 'SLURM_STEP_ID': '0', 'SLURM_CPUS_PER_TASK': '1'},
            'held_identifier_binding': {'reservation': 'sl5nw_168'},
            'job_query': {'command': ['scontrol', 'show', 'job', '-d', '123'], 'returncode': 0, 'stdout': job, 'stderr': ''},
            'step_query': {'command': ['scontrol', 'show', 'step', '-d', '123.0'], 'returncode': 0, 'stdout': step, 'stderr': ''}}

    def bind(self, unit):
        """Model the already returned held ID being fsynced by the root before metadata binding."""
        path = self.run_dir / 'slurm/submitted_jobids_uva.txt'
        path.parent.mkdir()
        with path.open('xb') as output:
            output.write(b'123\n')
            output.flush()
            os.fsync(output.fileno())
        return observation.bind_held_submission(unit, '123', 'sl5nw_168')

    def run_fixture(self, unit, *, changes=None):
        """Execute actual source/ledger/metadata writing while mocking only the owned process and scheduler/file reads."""
        row = self.record()
        plan = observation.verify_static(unit)
        if changes:
            changes(row)
        def query(command, **kwargs):
            """Serve only this fixture's two exact owned ID queries."""
            if command == row['job_query']['command']:
                return SimpleNamespace(**{key: row['job_query'][key] for key in ('returncode', 'stdout', 'stderr')})
            if command == row['step_query']['command']:
                return SimpleNamespace(**{key: row['step_query'][key] for key in ('returncode', 'stdout', 'stderr')})
            raise AssertionError('unexpected live command: ' + repr(command))
        # For example, absent cache rows remain observation data; they never become a qualification result.
        candidates = ({'cache_path_derived': False}, [{'path': '/fixture/missing', 'status': 'unavailable'}])
        with patch.object(observation, '__file__', str(unit / 'code' / observation.FILENAME)), \
                patch.object(observation, 'own_process', return_value=row['process']), \
                patch.dict(os.environ, row['environment'], clear=True), \
                patch.object(observation.subprocess, 'run', side_effect=query) as queries, \
                patch.object(observation, 'configuration_query', return_value={'returncode': 0, 'filtered_bytes': ''}), \
                patch.object(observation, 'configuration_candidates', return_value=candidates):
            # Source checks use real Git subprocess calls, so verify them immediately before the query mock.
            with patch.object(observation, 'verify_static', return_value=plan):
                result = observation.run(unit)
            self.assertEqual(queries.call_count, 2)
        return result

    def test_exact_detached_source_preparation_has_null_id_and_zero_credit(self):
        """Actual local Git bytes are staged once, and no identifier or mapping is invented."""
        unit = self.prepare()
        plan = observation.verify_static(unit)
        self.assertIsNone(plan['job_id'])
        self.assertIsNone(plan['reservation'])
        self.assertIsNone(plan['gpu_uuid'])
        self.assertEqual(plan['source']['source_commit'], self.commit)
        self.assertEqual((unit / 'code' / observation.FILENAME).read_bytes(), Path(observation.__file__).read_bytes())
        self.assertEqual(plan['shape']['gpu_count'], 0)
        self.assertFalse(plan['physical_assignment_verified'])
        self.assertFalse(plan['admission_granted'])
        self.assertTrue(all((unit / name).is_file() for name in ['experiment_background.md', 'manifest.yaml',
            'config_resolved.yaml', 'command.txt', 'progress.json']))
        with patch.object(observation, '__file__', str(unit / 'code' / observation.FILENAME)):
            self.assertEqual(observation.verify_static(unit, execution=True), plan)
        with self.assertRaisesRegex(ValueError, 'execute only the staged'):
            observation.verify_static(unit, execution=True)

    def test_changed_source_and_staged_module_are_rejected_before_collection(self):
        """Uncommitted source or a changed staged copy cannot collect metadata under the original source identity."""
        unit = self.prepare()
        path = unit / 'code' / observation.FILENAME
        path.write_bytes(path.read_bytes() + b'\n# changed fixture\n')
        with self.assertRaisesRegex(ValueError, 'committed or staged'):
            observation.verify_static(unit)
        source = self.source / observation.PREFIX / observation.FILENAME
        source.write_bytes(source.read_bytes() + b'\n# changed source fixture\n')
        with self.assertRaisesRegex(ValueError, 'clean and detached'):
            observation.source_identity(self.source, self.commit)

    def test_credit_or_reservation_cannot_be_added_to_the_static_plan(self):
        """A coherently rehashed plan still cannot change the diagnostic's unresolved identity or zero-credit scope."""
        unit = self.prepare()
        original = (unit / 'observation_plan.json').read_bytes()
        for change in ({'case_credit': 1}, {'case_credit': False}, {'gpu_uuid': 'GPU-guessed'}, {'reservation': 'sl5nw_168'}):
            value = json.loads(original)
            value.update(change)
            (unit / 'observation_plan.json').write_bytes(observation.encoded(value))
            (unit / 'prepared.json').write_bytes(observation.encoded({'plan_sha256': observation.sha256((unit / 'observation_plan.json').read_bytes())}))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'unsupported observation'):
                observation.verify_static(unit)

    def test_returned_held_id_binding_requires_the_canonical_fsynced_ledger(self):
        """An unresolved plan stays unchanged while the actual returned ID is bound in a separate immutable artifact."""
        unit = self.prepare()
        original = (unit / 'observation_plan.json').read_bytes()
        with self.assertRaises(FileNotFoundError):
            observation.bind_held_submission(unit, '123', 'sl5nw_168')
        result = self.bind(unit)
        self.assertEqual(result['job_id'], '123')
        self.assertFalse(result['held_state_verified_by_this_function'])
        self.assertEqual((unit / 'observation_plan.json').read_bytes(), original)
        self.assertEqual((unit / 'held_owned_job_ledger_snapshot.txt').read_bytes(), b'123\n')
        with self.assertRaises(FileExistsError):
            observation.bind_held_submission(unit, '123', 'sl5nw_168')

    def test_duplicate_or_unrecorded_returned_ids_are_rejected(self):
        """An arbitrary number or duplicate ledger entry cannot become the held observation identity."""
        unit = self.prepare()
        path = self.run_dir / 'slurm/submitted_jobids_uva.txt'
        path.parent.mkdir()
        for contents in (b'124\n', b'123\n123\n'):
            path.write_bytes(contents)
            with self.subTest(contents=contents), self.assertRaisesRegex(ValueError, 'already occur once'):
                observation.bind_held_submission(unit, '123', 'sl5nw_168')

    def test_one_or_two_actual_threads_are_recorded_without_claiming_one_affinity_entry(self):
        """One requested CPU may receive one or two actual threads; a GPU or larger CPU/memory shape is rejected."""
        for cpus in (1, 2):
            observation.validate_allocation(self.record(cpus))
        # Actual CPU6780635.50 separately records requested1, allocated2 and step1; affinity here is synthetic1.
        actual_format = self.record(2)
        actual_format['process']['cpu_affinity'] = [0]
        actual_format['step_query']['stdout'] = actual_format['step_query']['stdout'].replace('TRES=cpu=2', 'TRES=cpu=1')
        observation.validate_allocation(actual_format)
        changes = [lambda value: value['job_query'].update(stdout=value['job_query']['stdout'] + ',gres/gpu=1'),
            lambda value: value['job_query'].update(stdout=value['job_query']['stdout'].replace('mem=2G', 'mem=64G')),
            lambda value: value['step_query'].update(stdout=value['step_query']['stdout'] + ',gres/gpu:nvidia_rtx_a4500=1'),
            lambda value: value['job_query'].update(stdout=value['job_query']['stdout'].replace('Requeue=0', 'Requeue=1'))]
        for change in changes:
            value = self.record()
            change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                observation.validate_allocation(value)
        with self.assertRaises(ValueError):
            observation.validate_allocation(self.record(3))

    def test_requested_step_and_affinity_resources_cannot_exceed_actual_job(self):
        """Contradictory requested memory/GPU or step resources cannot hide behind a smaller allocated row."""
        changes = [lambda value: value['job_query'].update(stdout=value['job_query']['stdout'].replace(
                'ReqTRES=cpu=1,mem=2G,node=1', 'ReqTRES=cpu=2,mem=64G,node=1,gres/gpu=1')),
            lambda value: value['job_query'].update(stdout=value['job_query']['stdout'].replace('ReqTRES=cpu=1', 'ReqTRES=cpu=2')),
            lambda value: value['step_query'].update(stdout=value['step_query']['stdout'].replace('mem=2G', 'mem=64G')),
            lambda value: value['process'].update(cpu_affinity=[0, 1]),
            lambda value: value['step_query'].update(stdout=value['step_query']['stdout'].replace('TRES=cpu=1', 'TRES=cpu=2'))]
        for change in changes:
            value = self.record(1)
            change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                observation.validate_allocation(value)

    def test_process_birth_owner_and_query_identity_are_bound(self):
        """Another owner, replaced birth or changed owned job query cannot satisfy the observation reader."""
        changes = [lambda value: value['process'].update(uid_fields=[999] * 4),
            lambda value: value['process'].update(process_start_time='20000'),
            lambda value: value['process'].update(parent_pid=500),
            lambda value: value['environment'].update(SLURM_STEP_ID='batch'),
            lambda value: value['job_query'].update(command=['scontrol', 'show', 'job', '-d', '999']),
            lambda value: value['step_query'].update(returncode=True),
            lambda value: value['held_identifier_binding'].update(reservation='sl5nw_999')]
        for change in changes:
            value = self.record()
            change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                observation.validate_allocation(value)

    def test_allowed_configuration_filter_preserves_duplicates_but_omits_secret_settings(self):
        """Only the five approved names survive, with case-insensitive matching and comment contents removed."""
        raw = b'TaskPlugin = task/cgroup # omitted comment\nSecretSetting = SECRET\nTASKPLUGIN=task/affinity\nSlurmdSpoolDir=/var/spool/slurmd\n'
        actual = observation.filtered_settings(raw)
        self.assertEqual(actual, 'TaskPlugin = task/cgroup\nTASKPLUGIN = task/affinity\nSlurmdSpoolDir = /var/spool/slurmd\n')
        self.assertNotIn('SECRET', actual)

    def test_running_configuration_does_not_store_full_output_or_stderr(self):
        """Controller output outside the allowlist is represented only by its hash and stderr byte count."""
        result = SimpleNamespace(returncode=0, stdout=b'SlurmdSpoolDir=/var/spool/slurmd\nSecret=SECRET\n', stderr=b'SECRET')
        with patch.object(observation.subprocess, 'run', return_value=result) as query:
            value = observation.configuration_query()
        self.assertEqual(query.call_args.args[0], ['scontrol', 'show', 'config'])
        self.assertNotIn('SECRET', json.dumps(value))
        self.assertEqual(value['stderr_byte_count'], 6)

    def test_only_unique_actual_spool_cache_candidates_are_read(self):
        """Absent, duplicate or relative spool settings never cause guessed cache paths to be read."""
        def file(path, kind):
            """Return the requested path so the test can inspect the exact bounded read roster."""
            return {'path': str(path), 'kind': kind}
        values = ['', 'SlurmdSpoolDir = relative\n', 'SlurmdSpoolDir = /one\nSlurmdSpoolDir = /two\n']
        with patch.object(observation, 'configuration_file', side_effect=file):
            for value in values:
                discovery, rows = observation.configuration_candidates({'returncode': 0, 'filtered_bytes': value})
                self.assertFalse(discovery['cache_path_derived'])
                self.assertEqual(len(rows), 3)
            discovery, rows = observation.configuration_candidates({'returncode': 0,
                'filtered_bytes': 'SlurmdSpoolDir = /var/spool/slurmd\n'})
        self.assertTrue(discovery['cache_path_derived'])
        self.assertEqual([row['path'] for row in rows], [str(Path(directory) / (kind + '.conf'))
            for directory in ['/etc/slurm', '/var/spool/slurmd/conf-cache'] for kind in ['slurm', 'gres', 'cgroup']])

    def test_untrusted_ordinary_gres_bytes_are_observed_without_qualification(self):
        """A service-owned AutoDetect file remains original evidence, never an accepted physical File mapping."""
        path = self.root / 'gres.conf'
        raw = b'NodeName=jaguar[01-03,05-06] AutoDetect=nvml\n'
        path.write_bytes(raw)
        value = observation.configuration_file(path, 'gres')
        self.assertEqual(value['read_status'], 'read')
        self.assertEqual(value['sha256'], observation.sha256(raw))
        self.assertEqual(value['raw'], raw.decode())
        self.assertNotIn('physical_assignment_verified', value)

    def test_missing_symlink_and_read_failure_remain_explicit(self):
        """An unavailable or replaced candidate is not followed or silently omitted."""
        missing = observation.configuration_file(self.root / 'missing/gres.conf', 'gres')
        self.assertEqual(missing['status'], 'unavailable')
        target = self.root / 'target'
        target.write_text('original')
        link = self.root / 'gres.conf'
        link.symlink_to(target)
        value = observation.configuration_file(link, 'gres')
        self.assertEqual(value['read_status'], 'not_read_ordinary_path_unavailable')
        self.assertNotIn('raw', value)
        actual_open = os.open
        def unavailable(path, *args, **kwargs):
            """Refuse the pinned regular-inode reader while leaving metadata handles observable."""
            if str(path).startswith('/proc/self/fd/'):
                raise PermissionError('synthetic unreadable file')
            return actual_open(path, *args, **kwargs)
        with patch.object(observation.os, 'open', side_effect=unavailable):
            value = observation.configuration_file(target, 'gres')
        self.assertEqual(value['read_status'], 'failed')
        self.assertEqual(value['error_type'], 'PermissionError')

    def test_atomic_path_replacement_never_pairs_old_metadata_with_new_bytes(self):
        """A renamed replacement leaves the captured bytes tied to the original pinned regular inode."""
        path = self.root / 'gres.conf'
        path.write_bytes(b'original bytes\n')
        original_inode = path.stat().st_ino
        actual_read = os.read
        replaced = []
        def replace_then_read(descriptor, length):
            """Replace only the pathname after the regular read descriptor has already been bound."""
            if not replaced:
                replacement = self.root / 'replacement'
                replacement.write_bytes(b'new bytes\n')
                replacement.replace(path)
                replaced.append(True)
            return actual_read(descriptor, length)
        with patch.object(observation.os, 'read', side_effect=replace_then_read):
            value = observation.configuration_file(path, 'gres')
        self.assertEqual(value['raw'], 'original bytes\n')
        self.assertEqual(value['inode'], original_inode)
        self.assertNotEqual(value['inode'], path.stat().st_ino)
        self.assertTrue(value['read_trace']['path_replaced_during_read'])
        self.assertNotEqual(value['read_status'], 'read')

    def test_late_nonregular_replacement_never_opens_a_read_descriptor(self):
        """Only a metadata handle reaches a replaced nonregular inode; no ordinary read-open is attempted."""
        path = self.root / 'gres.conf'
        path.write_bytes(b'original\n')
        actual_open = os.open
        opens = []
        def replace_before_pin(name, flags, *args, **kwargs):
            """Replace the final file with a FIFO immediately before its non-following metadata handle."""
            opens.append(str(name))
            if name == path.name:
                path.unlink()
                os.mkfifo(path)
            return actual_open(name, flags, *args, **kwargs)
        with patch.object(observation.os, 'open', side_effect=replace_before_pin):
            value = observation.configuration_file(path, 'gres')
        self.assertEqual(value['read_status'], 'failed')
        self.assertFalse(value['read_trace']['descriptor_before']['regular_file'])
        self.assertFalse(any(name.startswith('/proc/self/fd/') for name in opens))

    def test_file_growth_is_bounded_and_never_claimed_as_a_complete_file(self):
        """Growth after descriptor metadata cannot bypass the bounded stream or produce a full-file hash claim."""
        path = self.root / 'gres.conf'
        path.write_bytes(b'first\n')
        actual_read = os.read
        grown = []
        def grow_then_read(descriptor, length):
            """Grow the same inode after the before-size observation, with a deliberately tiny test limit."""
            if not grown:
                with path.open('ab') as output:
                    output.write(b'x' * 100)
                grown.append(True)
            return actual_read(descriptor, length)
        trace = {}
        with patch.object(observation.os, 'read', side_effect=grow_then_read):
            raw = observation.pinned_configuration_bytes(path, trace, limit=16)
        self.assertEqual(len(raw), 16)
        self.assertEqual(trace['bytes_read'], 17)
        self.assertFalse(trace['complete_file'])
        self.assertTrue(trace['inode_changed_during_read'])
        with path.open('wb') as output:
            output.truncate(8 * 1024 * 1024 + 1)
        value = observation.configuration_file(path, 'gres')
        self.assertEqual(value['read_status'], 'size_limit_exceeded')
        self.assertNotIn('sha256', value)

    def test_slurm_file_retains_only_filtered_settings_and_full_hash(self):
        """A raw slurm.conf secret cannot appear in the observation payload."""
        path = self.root / 'slurm.conf'
        raw = b'TaskPlugin=task/cgroup\nSecretSetting=SECRET\nInclude=/fixture/other.conf\n'
        path.write_bytes(raw)
        value = observation.configuration_file(path, 'slurm')
        self.assertEqual(value['sha256'], observation.sha256(raw))
        self.assertEqual(value['filtered_bytes'], 'TaskPlugin = task/cgroup\n')
        self.assertTrue(value['include_directive_present'])
        self.assertNotIn('SECRET', json.dumps(value))
        self.assertNotIn('raw', value)

    def test_successful_metadata_collection_preserves_null_plan_and_ledger_growth(self):
        """A separate returned-ID snapshot survives later ledger growth and records missing cache evidence with zero credit."""
        unit = self.prepare()
        before = (unit / 'observation_plan.json').read_bytes()
        self.bind(unit)
        (self.run_dir / 'slurm/submitted_jobids_uva.txt').write_bytes(b'123\n124\n')
        result = self.run_fixture(unit)
        self.assertTrue(result['completed'])
        self.assertEqual(result['process']['cpu_affinity'], [0, 1])
        self.assertEqual(result['configuration_files'][0]['status'], 'unavailable')
        self.assertFalse(result['physical_assignment_verified'])
        self.assertFalse(result['admission_granted'])
        self.assertEqual((unit / 'observation_plan.json').read_bytes(), before)
        self.assertEqual(json.loads((unit / 'observation.json').read_bytes()), result)
        self.assertEqual(json.loads((unit / 'progress.json').read_bytes())['state'], 'completed')

    def test_foreign_or_nonbound_id_stops_before_any_query_and_preserves_failure(self):
        """Actual own-process failure metadata survives, while another owner or another recorded ID is never queried."""
        for foreign in (True, False):
            unit = self.prepare()
            self.bind(unit)
            process = self.process()
            if foreign:
                process['uid_fields'] = [999] * 4
            environment = {'SLURM_JOB_ID': '123' if foreign else '124', 'SLURM_STEP_ID': '0', 'SLURM_CPUS_PER_TASK': '1'}
            with patch.object(observation, 'verify_static', return_value=json.loads((unit / 'observation_plan.json').read_bytes())), \
                    patch.object(observation, 'own_process', return_value=process), \
                    patch.dict(os.environ, environment, clear=True), patch.object(observation.subprocess, 'run') as query:
                with self.assertRaises(ValueError):
                    observation.run(unit)
            query.assert_not_called()
            result = json.loads((unit / 'observation.json').read_bytes())
            self.assertFalse(result['completed'])
            self.assertIn('error', result)
            # Start a distinct synthetic campaign for the second immutable attempt.
            if foreign:
                self.run_dir = self.root / 'second_campaign'
                self.run_dir.mkdir()


if __name__ == '__main__':
    unittest.main()
