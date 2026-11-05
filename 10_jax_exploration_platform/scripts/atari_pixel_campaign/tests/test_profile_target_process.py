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
        self.plan = {'schema_version': 1, 'kind': process.QUALIFICATION, 'hardware_class': 'jaguar03_rtx_a4500',
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

    def configure(self, hardware='jaguar03_rtx_a4500'):
        self.plan['hardware_class'] = hardware
        if hardware == 'h100':
            self.plan.pop('configured_isolation', None)
            self.plan.update(direct_binding={}, gpu_uuid='GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3')
        if hardware == 'jaguar03_rtx_a4500':
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
        self.assertEqual(declared['prior_abi_baseline']['independent_review']['sha256'], process.PRIOR_REVIEWS['jaguar03_rtx_a4500'])
        self.assertIn(process.NAME, declared['helper_hashes'])
        self.assertIn('configured_native_process_control', self.plan)
        self.assertEqual(self.plan['diagnostic_time_budget'], process.QUALIFICATION_BUDGET)
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
            'source': self.observer.receipt(self.directory / 'code/profile_selected.py'), 'budget': process.QUALIFICATION_BUDGET,
            'clock_domain': {'hostname': 'jaguar03', 'boot_id': '00000000-0000-0000-0000-000000000001',
                'clock': 'monotonic', 'implementation': 'synthetic-saved-fixture'}, 'started_monotonic': 10.}
        path = self.directory / 'profile_target_process_time_origin.json';path.write_text(json.dumps(record))
        with patch.object(process.time, 'monotonic', side_effect=AssertionError('saved read cannot sample target clock')):
            saved = process.saved_deadline(self.directory, self.plan, self.observer)
            self.assertEqual(saved.work_end, 10210);self.assertEqual(saved.total_end, 10810)
        record['started_monotonic'] = True;path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):process.saved_deadline(self.directory, self.plan, self.observer)

    def test_whole_tool_scope_requires_this_runtime_and_real_h_descriptor_callback(self):
        self.configure('h100')
        runtime = process.Runtime.__new__(process.Runtime)
        runtime.pid, runtime.completed, runtime.plan, runtime.role = os.getpid(), False, self.plan, 'producer'
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
        runtime.pid, runtime.completed, runtime.plan, runtime.role = os.getpid(), False, self.plan, 'producer'
        runtime.deadline = SimpleNamespace(remaining=lambda phase:1.)
        runtime.provider = SimpleNamespace(evidence=lambda:{'open_handle_count':1,'native_call_failure':False})
        with self.assertRaisesRegex(ValueError, 'every retained handle closed'):runtime.complete([])

    def test_only_producer_runtime_can_construct_the_whole_tool_scope(self):
        self.configure('h100')
        runtime = process.Runtime.__new__(process.Runtime)
        runtime.pid, runtime.completed, runtime.plan, runtime.role = os.getpid(), False, self.plan, 'capture'
        observer = SimpleNamespace(native_process_runtime=runtime);runtime.observer = observer
        with self.assertRaisesRegex(ValueError, 'only the actual producer'):
            process.tool_scope(self.directory, self.plan, observer, self.owner(), deadline=object(),
                descriptor=7, check_descriptor=lambda:{})

    def test_qualification_budget_is_separate_typed_and_original_profile_budget_is_unchanged(self):
        self.configure()
        self.assertEqual(process.budget(self.plan), process.QUALIFICATION_BUDGET)
        self.assertEqual(self.plan['profile_target_process_control']['usage'], 'qualification')
        self.assertEqual(self.plan['maximum_active_seconds'], 10200)
        ordinary = copy.deepcopy(self.plan);ordinary['kind'] = process.PROFILE
        ordinary.pop('diagnostic_time_budget')
        self.assertEqual(process.budget(ordinary), process.BUDGET)
        ordinary['diagnostic_time_budget'] = dict(process.BUDGET)
        self.assertEqual(process.budget(ordinary), {'schema_version':1,'work_seconds':240,'cleanup_seconds':60,'total_seconds':300})
        for field, value in [('schema_version',True), ('work_seconds',10200.), ('cleanup_seconds',False), ('total_seconds',300)]:
            changed = copy.deepcopy(self.plan);changed['diagnostic_time_budget'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):process.budget(changed)

    def test_own_deadline_uses_actual_qualification_limits_without_restarting_cleanup(self):
        origin = {'started_monotonic':10.,'budget':dict(process.QUALIFICATION_BUDGET),'clock_domain':{'fixture':True}}
        deadline = process.ProfileTargetDeadline(origin, {'path':'/fixture/origin','sha256':'0'*64})
        self.assertEqual((deadline.work_end,deadline.total_end),(10210,10810))
        with patch.object(process.time,'monotonic',return_value=10209.):self.assertEqual(deadline.remaining('work'),1.)
        with patch.object(process.time,'monotonic',return_value=10210.), self.assertRaises(TimeoutError):deadline.remaining('work')
        with patch.object(process.time,'monotonic',return_value=10250.):self.assertEqual(deadline.begin_cleanup(),10810.)
        with patch.object(process.time,'monotonic',return_value=10300.):self.assertEqual(deadline.begin_cleanup(),10810.)
        with patch.object(process.time,'monotonic',return_value=10809.):
            self.assertEqual(deadline.remaining('cleanup'),1.)
            deadline.verify_snapshot(deadline.snapshot(),phase='total',cleanup_required=True)
        with patch.object(process.time,'monotonic',return_value=10810.), self.assertRaises(TimeoutError):deadline.remaining('total')
        old = process.ProfileTargetDeadline({**origin,'budget':dict(process.BUDGET)}, {})
        self.assertEqual((old.work_end,old.total_end),(250,310))
        with patch.object(process.time,'monotonic',return_value=260.):self.assertEqual(old.begin_cleanup(),310.)

    def test_public_capture_refuses_oversized_sparse_file_before_any_payload_read(self):
        source=self.directory/'oversized-public-image'
        with source.open('wb') as stream:stream.truncate(process.MAX_PUBLIC_BYTES+1)
        info=source.stat()
        row={'path':str(source),'sha256':'0'*64,'size_bytes':1,'uid':info.st_uid,'mode':0o644,
            'device':info.st_dev,'inode':info.st_ino}
        contract={'headers':{'fixture':row},'selected_python':row,'symbols':{'syscall':{'image':row},'close':{'image':row}}}
        observer=SimpleNamespace(native_source_hashes={},publish=lambda *args:None,
            receipt=lambda *args:(_ for _ in ()).throw(AssertionError('unbounded receipt is forbidden')))
        with patch.object(process.os,'read',side_effect=AssertionError('oversized sparse payload must not be read')):
            with self.assertRaisesRegex(ValueError,'before its first bounded'):process.capture_contract(self.directory,contract,observer)

    def test_public_reader_is_pinned_bounded_and_refuses_aliases_or_replaced_names(self):
        source=self.directory/'small-public-image';raw=b'small synthetic CPU image; never loaded';source.write_bytes(raw)
        info=source.stat();record={'path':str(source),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw),
            'uid':info.st_uid,'mode':0o644,'device':info.st_dev,'inode':info.st_ino}
        target=self.directory/'captured-image'
        self.assertEqual(process.bounded_public_file(source,record,original=True,target=target)['size_bytes'],len(raw))
        self.assertEqual(target.read_bytes(),raw)
        link=self.directory/'source-alias';link.symlink_to(source)
        with self.assertRaises(ValueError):process.bounded_public_file(link,record)
        original_read=process.os.read;replaced=[False]
        replacement=self.directory/'replacement';replacement.write_bytes(raw)
        def replace_after_read(fd,size):
            block=original_read(fd,size)
            if not replaced[0]:
                replaced[0]=True;replacement.replace(source)
            return block
        with patch.object(process.os,'read',side_effect=replace_after_read), self.assertRaisesRegex(ValueError,'pinned descriptor changed'):
            process.bounded_public_file(source,record)

    def test_portable_contract_checks_captured_image_bound_before_observer_receipt(self):
        self.configure();self.plan['unit_id']='synthetic-unit'
        row=lambda path,code,mode=0o444:{'path':path,'sha256':code*64,'size_bytes':1,'uid':0,'mode':mode,'device':1,'inode':1}
        h1=row('/fixture/unistd.h','1');h2=row('/fixture/unistd_64.h','2');libc=row('/fixture/libc.so','3')
        python=row('/fixture/python','4',0o555);python['actual_running_image_path']=python['path']
        contract={'kind':process.KIND,'abi':{'platform':'linux','machine':'x86_64','byteorder':'little','long_bytes':8,'int_bytes':4,'pointer_bytes':8},
            'syscall_numbers':{'pidfd_open':434,'pidfd_send_signal':424},'headers':{'/usr/include/unistd.h':h1,
                '/usr/include/x86_64-linux-gnu/asm/unistd_64.h':h2},'selected_python':python,
            'symbols':{'syscall':{'image':libc,'symbol_file_offset':0},'close':{'image':libc,'symbol_file_offset':0}}}
        saved={'kind':'profile_target_process_observed_contract','contract':contract,'source':process.policy(self.plan)['helper_hashes'],
            'captured_files':{},'actual_target_observation_only':True,'qualification':False,'admission_granted':False}
        sparse=self.directory/'oversized-captured-image'
        with sparse.open('wb') as stream:stream.truncate(process.MAX_PUBLIC_BYTES+1)
        mapping={}
        for record in [h1,h2,python,libc]:
            path=str(CAMPAIGN/'data/synthetic-unit/profile_target_process_contract_bytes'/record['sha256'])
            saved['captured_files'][record['path']]={'path':path,'sha256':record['sha256']};mapping[path]=str(sparse)
        with patch.object(self.observer,'receipt',side_effect=AssertionError('unbounded image receipt forbidden')), \
                patch.object(process.os,'read',side_effect=AssertionError('oversized payload must not be read')):
            with self.assertRaisesRegex(ValueError,'before its first bounded'):process.validate_contract(saved,self.plan,self.observer,mapping)

    def test_aa_and_h_waiter_exited_capture_close_requires_the_actual_prior_wait(self):
        event={'opened_pid':123,'operation':'close','identity':{'pid':-1},'observed_at_seconds':20.}
        waited={'returncode':0,'exited_normally':True,'interventions':[],'actual_wait_finished_at_seconds':19.}
        for target in ['jaguar03_rtx_a4500','h100']:
            with self.subTest(target=target):self.assertEqual(process.validate_waiter_capture_event(event,123,waited),event)
        for key,value in [('returncode',True),('returncode',1),('actual_wait_finished_at_seconds',21.),
                ('actual_wait_finished_at_seconds',True),('exited_normally',False),('interventions',['cleanup'])]:
            changed=copy.deepcopy(waited);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):process.validate_waiter_capture_event(event,123,changed)
        changed=copy.deepcopy(event);changed['operation']='pidfd_open'
        with self.assertRaises(ValueError):process.validate_waiter_capture_event(changed,123,waited)

    def test_actual_saved_gpu_claim_is_validated_on_different_owned_cpu_without_live_environment(self):
        """Read only accepted zero-environment metadata; never replay the old diagnostic."""
        unit=next(CAMPAIGN.glob('data/2026-11-05-06-39*'))
        plan=json.loads((unit/'measurement_plan.json').read_bytes())
        intent=json.loads((unit/'profile_process_intent.json').read_bytes())
        claim=json.loads((unit/'capture_identity.json').read_bytes())
        observer=self.original_adjacent(plan,'nsight_process_identity')
        self.assertEqual(intent['job_id'],'6825842')
        self.assertEqual(os.environ['SLURM_JOB_ID'],'6825983')
        self.assertNotEqual(intent['job_id'],os.environ['SLURM_JOB_ID'])
        with patch.object(observer,'validate_intent',side_effect=AssertionError('saved review must never call a live allocation gate')):
            self.assertEqual(process.validate_saved_claim(unit,plan,intent,claim,observer),claim)
            for key,value in [('challenge','0'*64),('pid',True),('process_start_time','1')]:
                changed=copy.deepcopy(claim);changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):process.validate_saved_claim(unit,plan,intent,changed,observer)
            for field in ['source','plan','intent']:
                changed=copy.deepcopy(claim);changed[field]['sha256']='0'*64
                with self.subTest(field=field),self.assertRaises(ValueError):process.validate_saved_claim(unit,plan,intent,changed,observer)
            changed=copy.deepcopy(claim);changed['process']['uids']=[9000]*4
            with self.assertRaises(ValueError):process.validate_saved_claim(unit,plan,intent,changed,observer)

    def test_saved_owned_job_requires_both_exact_ledgers_and_unique_matching_source(self):
        run=self.directory/'owned_fixture';(run/'slurm').mkdir(parents=True)
        source='a'*40;job='123'
        plan={'campaign_run_dir':str(run),'helpers':{'source_commit':source}}
        def reset():
            (run/'slurm/submitted_jobids.txt').write_text(job+'\n')
            (run/'slurm/submitted_jobids_uva.txt').write_text(job+'\n')
            (run/'slurm/submissions.tsv').write_text(job+'\tqualification\t'+source+'\t'+str(run/'packet/job.sbatch')+'\n')
        reset();self.assertEqual(process.saved_owned_job(plan,job)[2],source)
        for name in ['submitted_jobids.txt','submitted_jobids_uva.txt']:
            reset();(run/'slurm'/name).write_text(job+'\n'+job+'\n')
            with self.subTest(ledger=name),self.assertRaises(ValueError):process.saved_owned_job(plan,job)
        reset();(run/'slurm/submissions.tsv').write_text(job+'\tqualification\t'+'b'*40+'\t'+str(run/'packet/job.sbatch')+'\n')
        with self.assertRaises(ValueError):process.saved_owned_job(plan,job)
        reset();row=(run/'slurm/submissions.tsv').read_text();(run/'slurm/submissions.tsv').write_text(row+row)
        with self.assertRaises(ValueError):process.saved_owned_job(plan,job)
        reset()
        with self.assertRaises(ValueError):process.saved_owned_job(plan,'999')


if __name__ == '__main__':unittest.main()
