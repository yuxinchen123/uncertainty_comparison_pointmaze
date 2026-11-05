"""Pure metadata fixtures for the root-only full qualification launch utility.

Every source, plan, review and scheduler result below is explicitly synthetic.
No actual campaign unit, process, device, tool or scheduler is invoked.
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


class LaunchTests(unittest.TestCase):
    """Exercise durable private filesystem operations with a synthetic scheduler only."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'synthetic_integration'
        self.run = self.root/'platform/benchmark_runs/synthetic_campaign'
        self.packet = self.run/'remote_setup/synthetic-aa-capture';self.child = self.run/'data'/self.packet.name
        self.parent = self.run/'data/synthetic-metadata-parent'
        for path in (self.packet,self.child/'code',self.parent/'code',self.run/'queue',self.run/'slurm/jobs'):
            path.mkdir(parents=True,exist_ok=True)
        for name in ('submit_profile_qualification_held_once.py','test_submit_profile_qualification_held_once.py'):
            (self.packet/name).write_bytes((HERE/name).read_bytes())
        spec = importlib.util.spec_from_file_location('synthetic_qualification_held',self.packet/'submit_profile_qualification_held_once.py')
        self.m = importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
        self.m.OWNER_UID = os.getuid()
        self.git('init','-q');self.git('config','user.name','Synthetic fixture');self.git('config','user.email','synthetic@invalid')
        source_dir = self.root/self.m.PREFIX;source_dir.mkdir(parents=True)
        names = ('profile_target_qualification.py','direct_common_timing_controller.py','synthetic_capture.py')
        for name in names:
            raw = ('# Synthetic metadata fixture only: '+name+'\nVALUE = 1\n').encode()
            (source_dir/name).write_bytes(raw);(self.child/'code'/name).write_bytes(raw)
            if name != 'synthetic_capture.py':(self.parent/'code'/name).write_bytes(raw)
        source = self.commit()
        helpers = {'source_commit':source,'source_root':str(self.root),'files':{
            self.m.PREFIX+name:self.sha(source_dir/name) for name in names}}
        p = {'schema_version':1,'kind':self.m.PARENT_KIND,'protocol':self.m.PROTOCOL,'unit_id':self.parent.name,
            'campaign_run_dir':str(self.run),'profile_unit':str(self.child),'controller_maximum_seconds':10800,
            'qualification_time_budget':dict(self.m.BUDGET),'synthetic_fixture':True}
        self.write(self.parent/'measurement_plan.json',p)
        self.plan = {'schema_version':1,'kind':self.m.KIND,'unit_id':self.child.name,'campaign_run_dir':str(self.run),
            'hardware_class':'jaguar03_rtx_a4500','gpu_uuid':None,'diagnostic_time_budget':dict(self.m.BUDGET),
            'profile_target_qualification':{'schema_version':1,'protocol':self.m.PROTOCOL,'parent':self.m.reference(self.parent/'measurement_plan.json')},
            'configured_isolation':{'owned_inputs_path':str(self.child/'owned_allocation_inputs.json')},'helpers':helpers,
            'configuration':{'learner':{'n_copies':1,'num_envs':128,'num_steps':128}},'synthetic_fixture':True}
        self.write(self.child/'measurement_plan.json',self.plan)
        for path in (self.packet,self.parent,self.child):
            for name in self.m.STANDARD:
                (path/name).write_text('Synthetic distinct '+path.name+' '+name+'\n')
        self.parent_command = ['/usr/bin/python3','-I','-S','-B',str(self.parent/'code/profile_target_qualification.py'),'--prepared-unit',str(self.parent)]
        (self.parent/'command.txt').write_text(shlex.join(self.parent_command)+'\n')
        (self.child/'command.txt').write_text(shlex.join(['/synthetic/never-executed-python',str(self.child/'code/synthetic_capture.py')])+'\n')
        self.write(self.parent/'prepared_child_binding.json',{'controller_plan':self.m.reference(self.parent/'measurement_plan.json'),
            'child_plan':self.m.reference(self.child/'measurement_plan.json'),'command':shlex.split((self.child/'command.txt').read_text()),
            'controller_source':self.m.reference(self.parent/'code/profile_target_qualification.py'),'helpers':helpers})
        self.write(self.parent/'controller_launch_independent_review.json',{
            'kind':'independent_profile_target_qualification_controller_launch_review','approved':True,'passed':True,
            'independent_of_producer':True,'synthetic_fixture':True,'controller_plan':self.m.reference(self.parent/'measurement_plan.json'),
            'child_plan':self.m.reference(self.child/'measurement_plan.json')})
        self.write(self.child/'measurement_prepared.json',{'plan_sha256':self.m.reference(self.child/'measurement_plan.json')['sha256']})
        for name in ('submitted_jobids.txt','submitted_jobids_uva.txt','submissions.tsv'):(self.run/'slurm'/name).write_text('')
        (self.run/'queue/metadata.lock').write_text('')
        self.m.prepare_request(self.parent,'reservation');self.m.prepare_binding_once()
        self.write(self.packet/'packet_static_verification.json',{'passed':True,'synthetic_fixture':True,
            'prepared_binding':self.m.reference(self.packet/'prepared_launch_binding.json')})
        paths = set(self.m.document(self.packet/'immutable_artifacts.json')['files'])|{str(self.packet/name) for name in (
            'immutable_artifacts.json','prepared_launch_binding.json','packet_static_verification.json',
            'submit_profile_qualification_held_once.py','test_submit_profile_qualification_held_once.py')}
        self.write(self.packet/'launcher_source_freeze.json',{'files':{name:self.sha(Path(name)) for name in paths}})
        self.write(self.packet/'launcher_independent_review.json',{
            'kind':'independent_profile_target_qualification_prepared_launcher_review','approved':True,'passed':True,
            'independent_of_producer':True,'synthetic_fixture':True,'source_freeze':self.m.reference(self.packet/'launcher_source_freeze.json'),
            'prepared_binding':self.m.reference(self.packet/'prepared_launch_binding.json')})
        self.publication = self.commit();self.calls = [];self.events = [];self.job = '991234'

    def sha(self,path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def write(self,path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    def git(self,*args):
        return subprocess.check_output(['git','-C',str(self.root),*args],stderr=subprocess.PIPE,
            env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
    def commit(self):
        self.git('add','-A');self.git('commit','-qm','Explicitly synthetic private fixture')
        commit = self.git('rev-parse','HEAD').decode().strip()
        self.git('config','remote.origin.url',self.m.ORIGIN_URL);self.git('config','remote.origin.fetch',self.m.ORIGIN_FETCH)
        self.git('update-ref',self.m.PUBLICATION_REF,commit)
        return commit

    def scheduler(self,command,stem):
        """Save synthetic command outputs without starting any native command."""
        self.calls.append(command)
        if command[:3] == ['scontrol','show','reservation']:
            raw = b'ReservationName=sl5nw_168 StartTime=2020-01-01T00:00:00 EndTime=2099-01-01T00:00:00 Nodes=jaguar03,puma01 State=ACTIVE\n'
        elif command[0] == 'sbatch':raw = (self.job+'\n').encode()
        elif command[:3] == ['scontrol','show','job']:
            raw = ('JobId='+self.job+' UserId=sl5nw('+str(self.m.OWNER_UID)+') JobState=PENDING Reason=JobHeldUser '
                'Requeue=0 Restarts=0 TimeLimit=03:00:00 Partition=gpu Account=sl5nw_base QOS=csresnolim '
                'ReqNodeList=jaguar03 NumNodes=1 CPUs/Task=8 NumTasks=1 NumCPUs=16 Reservation=sl5nw_168 '
                'Command='+str(self.packet/'job.sbatch')+' ReqTRES=cpu=16,mem=64G,node=1,gres/gpu=1 '
                'TresPerNode=gres/gpu:nvidia_rtx_a4500:1\n').encode()
        else:
            self.assertEqual(command,['scontrol','release',self.job]);raw = b''
        for suffix,value in (('_stdout.raw',raw),('_stderr.raw',b''),('_return.json',{
                'command':command,'returncode':0,'synthetic_fixture':True,'started_at_utc':self.m.now(),'finished_at_utc':self.m.now()})):
            self.m.publish(self.packet/(stem+suffix),value)
        return SimpleNamespace(stdout=raw,stderr=b'',returncode=0)

    def submit(self):
        with mock.patch.object(self.m,'scheduler',side_effect=self.scheduler):return self.m.submit_held_once(self.publication)

    def release_binding(self):
        self.submit()
        with mock.patch.object(self.m,'scheduler',side_effect=self.scheduler):self.m.prepare_release(self.publication)
        self.write(self.packet/'release_independent_review.json',{'kind':'independent_profile_target_qualification_release_review',
            'approved':True,'passed':True,'independent_of_producer':True,'synthetic_fixture':True,
            'binding':self.m.reference(self.packet/'release_binding.json')})
        self.publication = self.commit()

    def test_default_has_no_scheduler_calls(self):
        with mock.patch.object(self.m,'scheduler',side_effect=AssertionError('default scheduler call')),contextlib.redirect_stdout(io.StringIO()) as output:
            self.m.main(['--publication-commit',self.publication])
        self.assertFalse(json.loads(output.getvalue())['action_performed'])

    def test_packet_parent_child_standards_separate_and_complete(self):
        inventory = self.m.document(self.packet/'immutable_artifacts.json')['files']
        for unit in (self.packet,self.parent,self.child):
            for name in self.m.STANDARD:self.assertIn(str(unit/name),inventory)
        self.assertNotEqual((self.packet/'manifest.yaml').read_bytes(),(self.parent/'manifest.yaml').read_bytes())
        script = (self.packet/'job.sbatch').read_text()
        self.assertIn('--exact --nodes=1 --ntasks=1 --cpus-per-task=8',script)
        self.assertIn(shlex.join(self.parent_command),script)
        self.assertNotIn('environment_transitions',self.m.document(self.packet/'qualification_launch_request.json'))

    def test_mutated_child_or_bound_review_refuses_before_query(self):
        path = self.parent/'controller_launch_independent_review.json';review = self.m.document(path);review['child_plan']['sha256'] = '0'*64;self.write(path,review)
        with mock.patch.object(self.m,'scheduler',side_effect=AssertionError('query before canonical validation')):
            with self.assertRaises(ValueError):self.m.submit_held_once(self.publication)

    def test_missing_unpublished_and_stale_origin_refuse(self):
        self.git('config','--remove-section','remote.origin')
        with self.assertRaises(ValueError):self.m.check_publication(self.publication)
        self.git('config','remote.origin.url',self.m.ORIGIN_URL);self.git('config','remote.origin.fetch',self.m.ORIGIN_FETCH)
        self.git('update-ref','-d',self.m.PUBLICATION_REF)
        with self.assertRaises(ValueError):self.m.check_publication(self.publication)
        self.git('update-ref',self.m.PUBLICATION_REF,self.plan['helpers']['source_commit'])
        with self.assertRaises(ValueError):self.m.check_publication(self.publication)

    def test_all_three_fsyncs_precede_submission_receipt(self):
        standards = {path/name:(path/name).read_bytes() for path in (self.packet,self.parent,self.child) for name in self.m.STANDARD}
        original_append,original_publish = self.m.append_member,self.m.publish
        def append(path,member):
            result = original_append(path,member);self.events.append('fsync:'+path.name);return result
        def publish(path,value):
            if path.name == 'submission_receipt.json':self.events.append('receipt')
            return original_publish(path,value)
        with mock.patch.object(self.m,'append_member',side_effect=append),mock.patch.object(self.m,'publish',side_effect=publish):self.submit()
        self.assertEqual(self.events[:4],['fsync:submitted_jobids.txt','fsync:submitted_jobids_uva.txt','fsync:submissions.tsv','receipt'])
        row = self.m.submission_row(self.job,self.plan)
        self.assertEqual(row,[self.job,self.m.KIND,self.plan['helpers']['source_commit'],str(self.packet/'job.sbatch')])
        self.assertEqual(self.m.owned_submission(self.plan)['held_command'],self.m.held_command(self.m.resources('reservation')))
        self.assertEqual(shlex.split((self.run/'slurm/jobs'/self.job/'command.txt').read_text()),
            self.m.document(self.packet/'submission_receipt.json')['held_command'])
        for path,before in standards.items():self.assertEqual(path.read_bytes(),before)

    def test_submission_attempt_and_duplicate_id_never_resubmit(self):
        self.submit();count = len(self.calls)
        with mock.patch.object(self.m,'scheduler',side_effect=self.scheduler):
            with self.assertRaises(ValueError):self.m.submit_held_once(self.publication)
        self.assertEqual(len(self.calls),count)

    def test_unowned_identifier_refuses_before_job_query(self):
        with mock.patch.object(self.m,'scheduler',side_effect=AssertionError('foreign query')):
            with self.assertRaises((ValueError,FileNotFoundError)):self.m.query_held('998877','never',self.plan)

    def test_exact_held_resource_and_owner_refusals(self):
        self.submit()
        original = self.scheduler
        def wrong(command,stem):
            result = original(command,stem)
            if command[:3] == ['scontrol','show','job']:result.stdout = result.stdout.replace(b'CPUs/Task=8',b'CPUs/Task=1')
            return result
        with mock.patch.object(self.m,'scheduler',side_effect=wrong):
            with self.assertRaises(ValueError):self.m.query_held(self.job,'bad_shape',self.plan)

    def test_release_receipt_before_atomic_unchanged_envelope(self):
        self.release_binding();original = self.m.publish
        def publish(path,value):
            if path.name == 'release_receipt.json':self.events.append('release_receipt')
            if path.name == 'owned_allocation_inputs.json':
                self.assertTrue((self.packet/'release_receipt.json').is_file());self.events.append('envelope')
            return original(path,value)
        with mock.patch.object(self.m,'publish',side_effect=publish),mock.patch.object(self.m,'scheduler',side_effect=self.scheduler):
            result = self.m.release_once(self.publication)
        self.assertEqual(self.events,['release_receipt','envelope'])
        value = self.m.document(self.child/'owned_allocation_inputs.json')
        self.assertEqual(set(value),{'schema_version','kind','job_id','plan','ledger','submissions','submission','release','submission_row'})
        self.assertEqual(value['kind'],'uva_configured_isolation_owned_inputs')
        self.assertEqual(result['command'],['scontrol','release',self.job])
        self.assertEqual(result['submission_receipt_sha256'],value['submission']['sha256'])
        self.assertEqual(self.m.document(self.packet/'submission_receipt.json')['kind'],'owned_uva_configured_isolation_submission')
        self.assertEqual(result['kind'],'owned_uva_configured_isolation_release')

    def test_failed_release_has_no_final_envelope_and_no_retry(self):
        self.release_binding();original = self.scheduler
        def failed(command,stem):
            if command[0:2] == ['scontrol','release']:raise RuntimeError('synthetic native failure')
            return original(command,stem)
        with mock.patch.object(self.m,'scheduler',side_effect=failed):
            with self.assertRaises(RuntimeError):self.m.release_once(self.publication)
        self.assertFalse((self.child/'owned_allocation_inputs.json').exists())
        with mock.patch.object(self.m,'scheduler',side_effect=AssertionError('repeat action')):
            with self.assertRaises(ValueError):self.m.release_once(self.publication)

    def test_fixed_release_job_and_snapshot_substitution_refuse(self):
        self.release_binding();binding = self.m.document(self.packet/'release_binding.json');binding['command'][-1] = '998877'
        self.write(self.packet/'release_binding.json',binding)
        with mock.patch.object(self.m,'scheduler',side_effect=AssertionError('substituted release')):
            with self.assertRaises(ValueError):self.m.release_once(self.publication)

    def test_bounded_regular_duplicate_and_boolean_metadata(self):
        target = self.packet/'small.json';target.write_text('{"x":1,"x":2}')
        with self.assertRaises(ValueError):self.m.document(target)
        target.write_text('{"x":NaN}')
        with self.assertRaises(ValueError):self.m.document(target)
        fifo = self.packet/'fifo';os.mkfifo(fifo)
        with mock.patch.object(self.m.os,'open',side_effect=AssertionError('FIFO opened')):
            with self.assertRaises(ValueError):self.m.ordinary_bytes(fifo)
        target.write_bytes(b'x'*(self.m.LIMIT+1))
        with self.assertRaises(ValueError):self.m.ordinary_bytes(target)
        self.assertFalse(self.m.same(True,1));self.assertFalse(self.m.same([True],[1]))

    def test_fresh_reservation_wrong_window_and_node_refuse(self):
        for index,replacement in enumerate((b'2000-01-01T00:00:00',b'2099-01-01T00:00:00')):
            def wrong(command,stem):
                result = self.scheduler(command,stem)
                result.stdout = result.stdout.replace(b'2099-01-01T00:00:00',replacement)
                if index:result.stdout = result.stdout.replace(b'jaguar03,puma01',b'puma01')
                return result
            with mock.patch.object(self.m,'scheduler',side_effect=wrong):
                with self.assertRaises(ValueError):self.m.reservation_observation(self.m.resources('reservation'),'wrong_reservation_'+str(index))
        with mock.patch.object(self.m,'scheduler',side_effect=AssertionError('normal must not query reservation')):
            self.assertIsNone(self.m.reservation_observation(self.m.resources('normal'),'none'))

    def test_one_shot_metadata_request_and_binding_refuse(self):
        with self.assertRaises(ValueError):self.m.prepare_request(self.parent,'reservation')
        with self.assertRaises(ValueError):self.m.prepare_binding_once()


if __name__ == '__main__':unittest.main()
