"""Metadata-only checks for the new explicit route; no tool, native call or GPU operation."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tested_profile_target_process', SCRIPTS / 'profile_target_process.py')
process = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process)
CAMPAIGN = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/'
    '2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')


class Observer:
    """Small fixture persistence; target runtime uses the unchanged strict raw process observer."""
    def receipt(self, path):
        path = Path(path)
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def read_json(self, path):
        return json.loads(Path(path).read_bytes())

    def publish(self, path, value):
        with Path(path).open('x') as stream:json.dump(value, stream, sort_keys=True)

    def validate_snapshot(self, owner, uid, cgroup):
        process.require(owner['uids'] == [uid] * 4 and owner['cgroup'] == cgroup, 'synthetic owner mismatch')

    def stable(self, owner):
        return {key: owner[key] for key in ('pid', 'process_start_time', 'uids', 'cgroup', 'command')}


class ProfileTargetProcessTests(unittest.TestCase):
    """Test policy/ordering without claiming actual target qualification from synthetic records."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        (self.directory / 'code').mkdir()
        (self.directory / 'code/profile_selected.py').write_text('synthetic source, never executed\n')
        self.observer = Observer()
        self.plan = {'schema_version': 1, 'kind': process.QUALIFICATION, 'hardware_class': 'a4500',
            'python': '/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python', 'configured_isolation': {},
            'gpu_uuid': None, 'campaign_run_dir': str(CAMPAIGN),
            'helpers': {'files': {process.PREFIX + name: hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest()
                if (SCRIPTS / name).is_file() else '0' * 64 for name in process.helpers()}}}
        self.plan['helpers']['files'][process.PREFIX + 'profile_selected.py'] = self.observer.receipt(self.directory / 'code/profile_selected.py')['sha256']
        self.reads = []
        self.qualifier = SimpleNamespace(contract=lambda p:self.reads.append('metadata_contract'),
            authorize=lambda p:(_ for _ in ()).throw(ValueError('formal admission closed')))
        self.original_adjacent = process.adjacent
        self.patcher = patch.object(process, 'adjacent', side_effect=self.adjacent)
        self.patcher.start();self.addCleanup(self.patcher.stop)

    def adjacent(self, plan, name):
        if name == 'profile_target_qualification':return self.qualifier
        if name == 'profile_target_direct_process':
            return SimpleNamespace(commands=self.original_adjacent(plan, 'direct_profile_process').commands)
        return self.original_adjacent(plan, name)

    def configure(self, hardware='a4500'):
        self.plan['hardware_class'] = hardware
        if hardware == 'h100':
            self.plan.pop('configured_isolation', None)
            self.plan.update(direct_binding={}, gpu_uuid='GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3')
        if hardware == 'a4500':
            result = next(CAMPAIGN.glob('data/2026-11-05-06-39*/result.json'))
            review = next(CAMPAIGN.glob('remote_setup/2026-11-05-06-39*/independent_actual_native_component_review.json'))
        else:
            result = next(CAMPAIGN.glob('data/2026-11-05-01-01*/result.json'))
            review = next(CAMPAIGN.glob('remote_setup/2026-11-05-01-01*/independent_actual_native_success_review_434.json'))
        process.configure_plan(self.plan, result, review)
        (self.directory / 'measurement_plan.json').write_text(json.dumps(self.plan))

    def owner(self, role='producer'):
        command = process.producer_command(self.directory, self.plan) if role == 'producer' else process.role_commands(self.directory, self.plan)[0 if role == 'waiter' else 1]
        return {'pid': os.getpid(), 'process_start_time': '123', 'uids': [2618919] * 4,
            'cgroup': 'synthetic fixture cgroup; no target claim', 'command': command}

    def test_absent_selection_preserves_default_without_loading_any_adapter(self):
        with patch.object(process, 'adjacent', side_effect=AssertionError('must remain unused')):
            self.assertIsNone(process.optional_policy({}))

    def test_actual_prior_aa_component_is_metadata_baseline_only_and_new_fields_are_explicit(self):
        self.configure()
        declared = process.policy(self.plan)
        self.assertFalse(declared['prior_component_qualifies_current_adapter'])
        self.assertEqual(declared['prior_abi_baseline']['independent_review']['sha256'], process.PRIOR_REVIEWS['a4500'])
        self.assertIn(process.NAME, declared['helper_hashes'])
        self.assertIn('configured_native_process_control', self.plan)
        self.assertEqual(self.plan['diagnostic_time_budget'], process.BUDGET)
        self.assertFalse(declared['admission_granted'])

    def test_actual_prior_h_component_has_no_borrowed_aa_or_old_direct_runtime_declaration(self):
        self.configure('h100')
        declared = process.policy(self.plan)
        self.assertEqual(declared['prior_abi_baseline']['independent_review']['sha256'], process.PRIOR_REVIEWS['h100'])
        self.assertNotIn('configured_native_process_control', self.plan)
        self.assertNotIn('native_process_control', self.plan)

    def test_formal_profile_remains_closed_before_native_provider_or_any_tool(self):
        self.plan['kind'] = process.PROFILE
        with patch.object(process, 'load_provider', side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError, 'formal admission closed'):
                process.configure_plan(self.plan, '/not-read/result', '/not-read/review')

    def test_exact_source_types_wrong_baseline_and_cross_target_contract_refuse(self):
        self.configure()
        original = copy.deepcopy(self.plan)
        for key, value in [('maximum_active_seconds', True), ('maximum_active_seconds', 241)]:
            self.plan[key] = value
            with self.assertRaises(ValueError):process.policy(self.plan)
            self.plan = copy.deepcopy(original)
        self.plan['profile_target_process_control']['helper_hashes'][process.NAME] = '1' * 64
        with self.assertRaises(ValueError):process.policy(self.plan)
        self.plan = copy.deepcopy(original)
        refs = copy.deepcopy(self.plan['profile_target_process_control']['prior_abi_baseline'])
        refs['independent_review']['sha256'] = process.PRIOR_REVIEWS['h100']
        with self.assertRaisesRegex(ValueError, 'exact independently accepted'):process.check_baseline(self.plan, refs, self.observer)
        refs = self.plan['profile_target_process_control']['prior_abi_baseline']
        actual = copy.deepcopy(process.check_baseline(self.plan, refs, self.observer))
        actual['abi']['pointer_bytes'] = 4
        with self.assertRaisesRegex(ValueError, 'actual selected Python'):process.check_baseline(self.plan, refs, self.observer, actual)

    def test_roles_and_commands_have_their_real_new_profile_shape(self):
        self.configure()
        self.assertEqual(process.producer_command(self.directory, self.plan)[1], str(self.directory / 'code/profile_selected.py'))
        self.assertIn('--owned-waiter-unit', process.role_commands(self.directory, self.plan)[0])
        self.assertIn('--capture-unit', process.role_commands(self.directory, self.plan)[1])
        for role in process.ROLES:
            self.assertTrue(process.role_files(self.directory, role)['begin'].name.startswith('profile_target_process_'))
        with self.assertRaises(ValueError):process.role_files(self.directory, '../producer')
        self.assertIn('native_capture_payload_timing.json', process.completion_references(self.plan, 'capture'))
        self.assertEqual(process.completion_references(self.plan, 'producer')[0], 'tool_descendant_scope_terminal.json')

    def test_shared_deadline_saved_proof_is_typed_and_no_live_clock_is_needed(self):
        self.configure()
        owner = self.owner()
        record = {'kind': 'profile_target_process_time_origin', 'schema_version': 1,
            'plan': self.observer.receipt(self.directory / 'measurement_plan.json'), 'producer': owner,
            'source': self.observer.receipt(self.directory / 'code/profile_selected.py'), 'budget': process.BUDGET,
            'clock_domain': {'hostname': 'jaguar03', 'boot_id': '00000000-0000-0000-0000-000000000001',
                'clock': 'monotonic', 'implementation': 'synthetic-saved-fixture'}, 'started_monotonic': 10.}
        path = self.directory / 'profile_target_process_time_origin.json';path.write_text(json.dumps(record))
        with patch.object(process.time, 'monotonic', side_effect=AssertionError('saved read cannot sample target clock')):
            saved = process.saved_deadline(self.directory, self.plan, self.observer)
            self.assertEqual(saved.work_end, 250);self.assertEqual(saved.total_end, 310)
        record['started_monotonic'] = True;path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):process.saved_deadline(self.directory, self.plan, self.observer)

    def test_whole_tool_scope_requires_this_runtime_and_real_h_descriptor_callback(self):
        self.configure('h100')
        runtime = process.Runtime.__new__(process.Runtime)
        runtime.pid, runtime.completed, runtime.plan = os.getpid(), False, self.plan
        observer = SimpleNamespace(native_process_runtime=runtime);runtime.observer = observer
        with self.assertRaisesRegex(ValueError, 'actual standing descriptor'):
            process.tool_scope(self.directory, self.plan, observer, self.owner(), deadline=object())
        captured = {}
        class Scope:
            def __init__(self, *args, **kwargs):captured.update(kwargs)
        prior = process.adjacent
        with patch.object(process, 'adjacent', side_effect=lambda plan, name:SimpleNamespace(OwnedToolScope=Scope)
                if name == 'direct_tool_descendants' else prior(plan, name)):
            process.tool_scope(self.directory, self.plan, observer, self.owner(), deadline=object(),
                descriptor=7, check_descriptor=lambda:{'synthetic': True})
        self.assertEqual(captured['custody_kind'], 'standing_gpu_descriptor')
        runtime.completed = True
        with self.assertRaises(ValueError):process.runtime_observer(self.plan, observer)

    def test_saved_reader_cannot_accept_missing_deadline_or_h_custody(self):
        self.configure('h100')
        for deadline in [None, object()]:
            with self.assertRaises(ValueError):
                process.validate_saved(self.directory, self.plan, self.observer, {'producer':self.owner()}, {}, {}, {}, deadline)

    def test_wrong_source_role_refuses_before_provider_startup(self):
        self.configure()
        owner = self.owner();owner['command'] = ['/some/foreign/command']
        with patch.object(process, 'context', return_value=owner), \
                patch.object(process, 'load_provider', side_effect=AssertionError('provider must not start')):
            with self.assertRaisesRegex(ValueError, 'actual owned source command'):
                process.Runtime(self.directory, self.plan, 'producer', self.observer, owner, deadline=object())

    def test_completed_role_cannot_hide_retained_native_handles(self):
        self.configure()
        runtime = process.Runtime.__new__(process.Runtime)
        runtime.pid, runtime.completed, runtime.plan = os.getpid(), False, self.plan
        runtime.deadline = SimpleNamespace(remaining=lambda phase:1.)
        runtime.provider = SimpleNamespace(evidence=lambda:{'open_handle_count':1,'native_call_failure':False})
        with self.assertRaisesRegex(ValueError, 'every retained handle closed'):runtime.complete([])


if __name__ == '__main__':unittest.main()
