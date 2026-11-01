"""Check an explicit UVA ownership profile using synthetic metadata, never a GPU qualification."""

import ast
import copy
import hashlib
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
import device_memory_limit as capacity
import measure_selected as measurement
import train_optimized as training


class ExplicitUvaOwnershipTests(unittest.TestCase):
    """Keep actual CPU metadata, derived synthetic structural evidence and GPU admission distinct."""

    def setUp(self):
        """Read the unchanged CPU observation and construct explicitly synthetic missing directory/file metadata."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        path = SCRIPTS / 'tests/fixtures/uva_owned_cpu_numeric_step_6780635_50.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
            '041551ef9c6c3e9cc495be561e44b4d84c628baf10f019615f2453249aa78eb3')
        self.actual = json.loads(path.read_text())
        self.job, self.step = self.actual['job_id'], self.actual['step_id']
        self.plan = {'ownership_profile': copy.deepcopy(capacity.UVA_CGROUP_V2_OWNERSHIP_PROFILE),
            'device_memory_limit_protocol': {**capacity.PROTOCOL, 'allocation_kind': 'slurm', 'cluster': 'uva',
                'owned_job_ledger': {'path': str(self.directory / 'owned.txt'), 'sha256': 'a' * 64}}}
        raw_stat = self.actual['process_stat']
        self.process = {'hostname': self.actual['hostname'], 'pid': self.actual['pid'], 'uid': self.actual['uid'],
            'parent_pid': self.actual['parent_pid'], 'process_start_time': raw_stat[raw_stat.rfind(')') + 2:].split()[19],
            'cpu_affinity': [0], 'cgroup': self.actual['cgroup']}
        self.value = self.synthetic_proof()

    def process_metadata(self):
        """Adapt the actual selected UID/stat/membership bytes into the new format without claiming original profile production."""
        uid = ''.join(row + '\n' for row in self.actual['status_uid_and_gid_lines'] if row.startswith('Uid:'))
        raw_stat, group = self.actual['process_stat'], self.actual['cgroup']
        return {'pid': self.actual['pid'], 'uid': self.actual['uid'], 'effective_uid': self.actual['effective_uid'],
            'status': {'path': '/proc/self/status', 'uid_bytes': uid, 'uid_sha256': hashlib.sha256(uid.encode()).hexdigest(),
                'full_sha256': hashlib.sha256(uid.encode()).hexdigest()},
            'stat': {'path': '/proc/self/stat', 'raw': raw_stat, 'sha256': hashlib.sha256(raw_stat.encode()).hexdigest()},
            'cgroup': {'path': '/proc/self/cgroup', 'raw': group, 'sha256': hashlib.sha256(group.encode()).hexdigest()},
            'namespace': {'path': '/proc/self/ns/cgroup', 'inode': self.actual['namespace_inode'],
                'target': 'cgroup:[' + str(self.actual['namespace_inode']) + ']'}}

    def synthetic_proof(self):
        """Add synthetic ancestor/inode metadata absent from the actual CPU observation; this is not an actual GPU proof."""
        _, chain = capacity.v2_paths(self.job, self.step)
        observed = {row['path']: row for row in self.actual['directories']}
        directories, members = [], []
        for index, path in enumerate(chain):
            actual = observed.get(str(path), {})
            directories.append({'path': str(path), 'status': 'readable', 'uid': actual.get('uid', 0),
                'mode': int(actual.get('mode', '0o755'), 8), 'inode': actual.get('inode', 100 + index),
                'device': actual.get('device', 29), 'regular_file': False, 'directory': True,
                'symlink': False, 'exact_path': True})
            if str(path) in observed:
                row = observed[str(path)]['cgroup.procs']
                members.append({'path': str(path / 'cgroup.procs'), 'status': 'readable', 'uid': row['uid'],
                    'mode': int(row['mode'], 8), 'inode': 1000 + index, 'device': 29,
                    'regular_file': True, 'directory': False, 'symlink': False, 'exact_path': True,
                    'raw': row['raw'], 'sha256': row['sha256']})
        mount = '\n'.join(self.actual['cgroup2_mountinfo']) + '\n'
        return {'schema_version': 1, 'kind': 'uva_cgroup_v2_ownership_observation',
            'ownership_profile': copy.deepcopy(capacity.UVA_CGROUP_V2_OWNERSHIP_PROFILE),
            'ownership_capability': capacity.UVA_CGROUP_V2_OWNERSHIP_CAPABILITY,
            'before': self.process_metadata(), 'after': self.process_metadata(),
            'mountinfo': {'path': '/proc/self/mountinfo', 'raw': mount, 'sha256': hashlib.sha256(mount.encode()).hexdigest(),
                'full_sha256': hashlib.sha256(mount.encode()).hexdigest()},
            'directories_before': directories, 'directories_after': copy.deepcopy(directories), 'memberships': members}

    def validate(self, value=None):
        """Invoke the actual portable structural reader without any live process or cgroup read."""
        return capacity.validate_v2_ownership(self.value if value is None else value, self.process, self.job, self.step)

    def rehash(self, value):
        """Keep mutated raw-byte hashes coherent so rejection must inspect actual metadata meaning."""
        for stage in ('before', 'after'):
            current = value[stage]
            for key in ('stat', 'cgroup'):
                current[key]['sha256'] = hashlib.sha256(current[key]['raw'].encode()).hexdigest()
            current['status']['uid_sha256'] = hashlib.sha256(current['status']['uid_bytes'].encode()).hexdigest()
        value['mountinfo']['sha256'] = hashlib.sha256(value['mountinfo']['raw'].encode()).hexdigest()
        for row in value['memberships']:
            row['sha256'] = hashlib.sha256(row['raw'].encode()).hexdigest()

    def owned_cpu(self, with_profile=True):
        """Combine the real CPU job/step queries with a synthetic claimed GPU selector to prove the full GPU gate still refuses it."""
        queries = {row['command'][2]: copy.deepcopy(row) for row in self.actual['owned_scheduler_queries']}
        value = {'kind': 'slurm', 'cluster': 'uva', 'ledger': self.plan['device_memory_limit_protocol']['owned_job_ledger'],
            'job_id': self.job, 'step_id': self.step, 'process': self.process,
            'environment': {'CUDA_VISIBLE_DEVICES': 'GPU-synthetic', 'SLURM_JOB_ID': self.job,
                'SLURM_STEP_ID': self.step, 'SLURM_GPUS_ON_NODE': '1', 'SLURM_CPUS_PER_TASK': '1'},
            'job': queries['job'], 'step': queries['step'],
            'node': {'command': ['scontrol', 'show', 'node', '-d', self.process['hostname']],
                'returncode': 0, 'stdout': 'NodeName=' + self.process['hostname'], 'stderr': ''}, 'physical_mapping': {}}
        if with_profile:
            value.update(ownership_profile=copy.deepcopy(self.plan['ownership_profile']),
                ownership_capability=capacity.UVA_CGROUP_V2_OWNERSHIP_CAPABILITY, cgroup_v2_ownership=self.value)
        return value

    def test_structural_fixture_passes_but_real_cpu_job_cannot_qualify_gpu_ownership(self):
        """The actual CPU record supports UID/path design only; its zero-GPU resources cannot pass a GPU reader."""
        self.assertEqual(self.validate(), self.value)
        self.assertTrue(all(row['raw'] == '' for row in self.value['memberships'][:-1]))
        plan = {**self.plan, 'gpu_uuid': 'GPU-synthetic'}
        with self.assertRaisesRegex(ValueError, 'exactly one owned GPU'):
            capacity.validate_ownership(self.owned_cpu(), plan)

    def test_old_missing_profile_keeps_uid_in_path_rule(self):
        """Absence of a profile cannot reinterpret an old unified-path observation as new evidence."""
        plan = {key: value for key, value in self.plan.items() if key != 'ownership_profile'}
        plan['gpu_uuid'] = 'GPU-synthetic'
        with self.assertRaisesRegex(ValueError, 'job/step/cgroup mismatch'):
            capacity.validate_ownership(self.owned_cpu(with_profile=False), plan)

    def test_profile_is_exact_and_only_for_uva_slurm(self):
        """Null, Boolean, changed flags and another allocation kind never select the new profile."""
        for value in (None, False, {**self.plan['ownership_profile'], 'schema_version': True},
                {**self.plan['ownership_profile'], 'uid_encoded_in_cgroup_path': True},
                {**self.plan['ownership_profile'], 'extra': 'unknown'}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                capacity.selected_ownership_profile({**self.plan, 'ownership_profile': value})
        for changes in ({'cluster': 'aces'}, {'allocation_kind': 'direct'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                capacity.selected_ownership_profile({**self.plan,
                    'device_memory_limit_protocol': {**self.plan['device_memory_limit_protocol'], **changes}})

    def test_uid_pid_parent_and_birth_changes_are_rejected(self):
        """Rehashed proc summaries cannot substitute another owner or another lifetime of the same process."""
        changes = [lambda row: row['before']['status'].update(uid_bytes=''),
            lambda row: row['before']['status'].update(uid_bytes=row['before']['status']['uid_bytes'] * 2),
            lambda row: row['before']['status'].update(uid_bytes='Uid:\t2618919\t1\t2618919\t2618919\n'),
            lambda row: row['after'].update(effective_uid=1), lambda row: row['before'].update(pid=1),
            lambda row: row['before']['stat'].update(raw=row['before']['stat']['raw'].replace(') R ', ')xR ', 1)),
            lambda row: row['before']['stat'].update(raw=row['before']['stat']['raw'].replace(self.process['process_start_time'], '99999', 1)),
            lambda row: row['before']['stat'].update(raw=row['before']['stat']['raw'].replace(' ' + str(self.process['parent_pid']) + ' ', ' 99999 ', 1))]
        for change in changes:
            value = copy.deepcopy(self.value); change(value); self.rehash(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(value)

    def test_exact_numeric_hierarchy_mount_and_namespace_are_required(self):
        """Foreign or duplicate memberships, batch steps and another mount cannot become this numeric-step proof."""
        for step in ('batch', 'extern', 'unknown'):
            with self.subTest(step=step), self.assertRaises(ValueError):
                capacity.validate_v2_ownership(self.value, self.process, self.job, step)
        changes = [lambda row: row['before']['cgroup'].update(raw=row['before']['cgroup']['raw'] * 2),
            lambda row: row['after']['cgroup'].update(raw=row['after']['cgroup']['raw'].replace('step_50', 'step_51')),
            lambda row: row['mountinfo'].update(raw=row['mountinfo']['raw'].replace(' / /sys/fs/cgroup ', ' /other /sys/fs/cgroup ')),
            lambda row: row['mountinfo'].update(raw=row['mountinfo']['raw'] * 2),
            lambda row: row['after']['namespace'].update(inode=1, target='cgroup:[1]')]
        for change in changes:
            value = copy.deepcopy(self.value); change(value); self.rehash(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(value)

    def test_root_directory_file_and_task_pid_evidence_is_required(self):
        """Every owned path needs trusted metadata, while only the task list needs the current PID."""
        changes = [lambda row: row['directories_before'][0].update(uid=1),
            lambda row: row['directories_before'][0].update(mode=0o775),
            lambda row: row['directories_after'][-1].update(inode=99999),
            lambda row: row['directories_before'][-1].update(status='absent'),
            lambda row: row['memberships'][-1].update(symlink=True),
            lambda row: row['memberships'][-1].update(mode=0o666),
            lambda row: row['memberships'][-1].update(raw=''),
            lambda row: row['memberships'][-1].update(raw=row['memberships'][-1]['raw'] * 2)]
        for change in changes:
            value = copy.deepcopy(self.value); change(value); self.rehash(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(value)

    def test_batch_gpu_record_is_not_numeric_step_profile_evidence(self):
        """The preserved actual GPU batch guard cannot replace a fresh numeric-step observation."""
        path = SCRIPTS / 'tests/fixtures/uva_owned_checkpoint_allocation_6780569.json'
        guard = json.loads(path.read_text())
        self.assertIn('step_batch/user/task_0', guard['cgroup'])
        with self.assertRaises(ValueError):
            capacity.v2_paths(guard['job_id'], 'batch')

    def test_old_published_helper_cannot_declare_the_new_profile(self):
        """The actual old7f publication has the native sampler but no explicit version-two ownership capability."""
        root = Path('/p/rlprojects/RND_worktrees/2026-11-01-05-05_memory-capture-helper_source-commit-7f5bf87a8400a73afff08098c1422642ecad69f9')
        raw = subprocess.check_output(['git', '-C', str(root), 'show',
            '7f5bf87a8400a73afff08098c1422642ecad69f9:10_jax_exploration_platform/scripts/atari_pixel_campaign/measure_selected.py'])
        declarations = [ast.literal_eval(node.value) for node in ast.parse(raw).body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'HELPERS' for target in node.targets)]
        self.assertEqual(len(declarations), 1)
        self.assertNotIn('short_learning.py', declarations[0])
        helpers = training.source_receipt(root, '7f5bf87a8400a73afff08098c1422642ecad69f9',
            ['10_jax_exploration_platform/scripts/atari_pixel_campaign/' + name for name in declarations[0]])
        with self.assertRaisesRegex(ValueError, 'exact versioned ownership capability'):
            capacity.require_ownership_source({**self.plan, 'helpers': helpers})

    def test_capture_preserves_raw_failed_validation_and_partial_read(self):
        """An invalid owner or interrupted proc read remains an immutable raw record before the exception escapes."""
        bad = self.process_metadata(); bad['status']['uid_bytes'] = 'Uid:\t1\t1\t1\t1\n'
        bad['status']['uid_sha256'] = hashlib.sha256(bad['status']['uid_bytes'].encode()).hexdigest()
        rows = {row['path']: row for row in [*self.value['directories_before'], *self.value['memberships']]}
        with patch.object(capacity, 'v2_process_metadata', side_effect=[bad, bad]), \
                patch.object(capacity.Path, 'read_text', return_value=self.value['mountinfo']['raw']), \
                patch.object(capacity, 'v2_file', side_effect=lambda path: copy.deepcopy(rows[str(path)])):
            path = self.directory / 'bad_uid.json'
            with self.assertRaisesRegex(ValueError, 'four actual process UIDs'):
                capacity.capture_v2_ownership(self.plan, self.process, self.job, self.step, path)
        self.assertEqual(json.loads(path.read_text())['before']['status'], bad['status'])
        def interrupted(observation):
            """Model an ordinary read failure after UID bytes were collected."""
            observation['status'] = {'uid_bytes': 'Uid:\t2618919\t2618919\t2618919\t2618919\n'}
            raise PermissionError('injected later proc read failure')
        path = self.directory / 'partial.json'
        with patch.object(capacity, 'v2_process_metadata', side_effect=interrupted), self.assertRaises(PermissionError):
            capacity.capture_v2_ownership(self.plan, self.process, self.job, self.step, path)
        saved = json.loads(path.read_text())
        self.assertIn('Uid:', saved['before']['status']['uid_bytes'])
        self.assertEqual(saved['capture_error']['exception_type'], 'PermissionError')

    def test_foreign_placement_stops_before_reading_job_directories(self):
        """A changed environment cannot make the collector inspect another job's cgroup paths."""
        bad = self.process_metadata(); bad['cgroup']['raw'] = bad['cgroup']['raw'].replace(self.job, '99999')
        with patch.object(capacity, 'v2_process_metadata', return_value=bad), patch.object(capacity, 'v2_file') as read, self.assertRaises(ValueError):
            capacity.capture_v2_ownership(self.plan, self.process, self.job, self.step, self.directory / 'foreign.json')
        read.assert_not_called()

    def test_saved_structural_reader_never_reopens_node_local_paths(self):
        """The raw nested object remains verifiable after its job and node-local cgroups have disappeared."""
        with patch.object(Path, 'read_text', side_effect=AssertionError('no historical node file reads')), \
                patch.object(Path, 'lstat', side_effect=AssertionError('no historical node metadata')), \
                patch.object(subprocess, 'run', side_effect=AssertionError('no live commands')):
            self.assertEqual(self.validate(), self.value)

    def test_new_committed_helper_declares_the_exact_profile_capability(self):
        """A detached synthetic published helper tree contains every exact dependency and both literal declarations."""
        root = self.directory / 'synthetic_helper_source'
        relatives = ['10_jax_exploration_platform/scripts/atari_pixel_campaign/' + name for name in measurement.HELPERS]
        for relative in relatives:
            target = root / relative; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((SCRIPTS.parents[2] / relative).read_bytes())
        subprocess.run(['git', 'init', '-q', str(root)], check=True)
        subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(root), '-c', 'user.name=Synthetic metadata test', '-c',
            'user.email=synthetic@example.invalid', '-c', 'core.hooksPath=/dev/null', 'commit', '-qm',
            'Create synthetic explicit ownership source'], check=True)
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        subprocess.run(['git', '-C', str(root), 'checkout', '-q', '--detach', commit], check=True)
        helpers = training.source_receipt(root, commit, relatives)
        capacity.require_ownership_source({**self.plan, 'helpers': helpers})

    def test_full_collector_preserves_queries_and_partial_mapping_on_error(self):
        """A mapping failure retains the already collected own process and exact owned scheduler queries."""
        run = self.directory / 'campaign'; (run / 'slurm').mkdir(parents=True)
        origin = run / 'slurm/submitted_jobids_uva.txt'; origin.write_text(self.job + '\n')
        frozen = self.directory / 'owned.txt'; frozen.write_bytes(origin.read_bytes())
        plan = {**self.plan, 'campaign_run_dir': str(run), 'gpu_uuid': 'GPU-synthetic',
            'device_memory_limit_protocol': {**self.plan['device_memory_limit_protocol'], 'owned_job_ledger': capacity.file_receipt(frozen)},
            'device_memory_limit_ledger_source': capacity.file_receipt(origin)}
        environment = {'CUDA_VISIBLE_DEVICES': 'GPU-synthetic', 'SLURM_JOB_ID': self.job, 'SLURM_STEP_ID': self.step,
            'SLURM_GPUS_ON_NODE': '1', 'SLURM_CPUS_PER_TASK': '1'}
        def mapping(environment, gpu, cluster, observation=None):
            """Model a later node-cache failure after its exact query record was collected."""
            observation['running_configuration'] = {'command': ['scontrol', 'show', 'config'], 'returncode': 1}
            raise PermissionError('injected node-cache error')
        path = self.directory / 'owned_capture.json'
        with patch.object(capacity, 'require_ownership_source'), patch.object(capacity, 'process_identity', return_value=self.process), \
                patch.dict(os.environ, environment, clear=True), \
                patch.object(capacity, 'capture_v2_ownership', return_value=self.value), \
                patch.object(capacity.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='owned query fixture', stderr='')) as query, \
                patch.object(capacity, 'collect_mapping', side_effect=mapping), self.assertRaises(PermissionError):
            capacity.ownership(plan, path)
        saved = json.loads(path.read_text())
        self.assertEqual(saved['cgroup_v2_ownership'], self.value)
        self.assertEqual(saved['capture_error']['exception_type'], 'PermissionError')
        self.assertEqual(saved['physical_mapping']['running_configuration']['returncode'], 1)
        self.assertEqual([call.args[0] for call in query.call_args_list], [
            ['scontrol', 'show', 'job', '-d', self.job], ['scontrol', 'show', 'step', '-d', self.job + '.' + self.step],
            ['scontrol', 'show', 'node', '-d', self.process['hostname']]])

    def test_metadata_read_failure_is_retained_without_a_live_fallback(self):
        """Absent and unreadable cgroup paths become explicit rejected rows instead of another path guess."""
        path = Path('/sys/fs/cgroup/synthetic_missing_ownership')
        for error, expected in ((FileNotFoundError('absent'), 'absent'), (PermissionError('denied'), 'unreadable')):
            with patch.object(Path, 'lstat', side_effect=error):
                row = capacity.v2_file(path)
            self.assertEqual(row['status'], expected)
            self.assertEqual(row['path'], str(path))


if __name__ == '__main__':
    unittest.main()
