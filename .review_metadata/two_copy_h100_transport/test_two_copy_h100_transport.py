"""Check a separate two-copy transport without running a controller or contacting a host."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

D=Path(__file__).resolve().parent
W=D.parent.parent
OPERATIONS=W/'.review_metadata/profile_target_operational_preparation'

def load(name):
    spec=importlib.util.spec_from_file_location(name,OPERATIONS/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

TRANSPORT=load('start_direct_two_copy_profile_qualification_once')

class TwoCopyTransport(unittest.TestCase):
    def test_complete_synthetic_transport_checks_and_changed_plan_refuse(self):
        """Exercise every actual metadata check with explicit synthetic review/Git premises."""
        root = W
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            run = Path(temporary) / 'campaign';packet = run/'remote_setup/packet';packet.mkdir(parents=True)
            controller = run/'data/controller';(controller/'code').mkdir(parents=True)
            child = run/'data/capture';(child/'code').mkdir(parents=True)
            log = run/'slurm/direct_jobs/packet';log.mkdir(parents=True)
            def put(path, value):
                path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
                return TRANSPORT.capture(path)[1]
            names = ['profile_target_qualification.py', 'direct_common_timing_controller.py'] + ['synthetic_helper_'+str(i)+'.py' for i in range(43)]
            helper_raw = {name: ('# explicit synthetic source premise '+name+'\n').encode() for name in names}
            helper = {'source_commit': TRANSPORT.RUNTIME, 'files': {TRANSPORT.PREFIX+name: hashlib.sha256(raw).hexdigest() for name, raw in helper_raw.items()}}
            for name, raw in helper_raw.items():put(child/'code'/name, raw)
            parent_source = put(controller/'code/profile_target_qualification.py', helper_raw['profile_target_qualification.py'])
            process_source = put(controller/'code/direct_common_timing_controller.py', helper_raw['direct_common_timing_controller.py'])
            original_ref = put(run/'original_reference.json',{'configuration':{'learner':{'n_copies':1}},'copy_seeds':[1]})
            parent = {'original_qualification_plan':original_ref,'unit_id':'controller','kind':'profile_target_qualification_controller','profile_unit':str(child),
                'campaign_run_dir':str(run),'source_commit':TRANSPORT.SOURCE,'controller_maximum_seconds':10800,
                'qualification_time_budget':{'schema_version':1,'work_seconds':10200,'cleanup_seconds':600,'total_seconds':10800}}
            parent_ref = put(controller/'measurement_plan.json',parent)
            child_plan = {'unit_id':'capture','source_commit':TRANSPORT.SOURCE,'kind':'full_original_profile_source_target_qualification',
                'hardware_class':'h100','copy_seeds':[1,2],'configuration':{'learner':{'n_copies':2}},'helpers':helper,
                'profile_target_qualification':{'parent':parent_ref}}
            child_ref = put(child/'measurement_plan.json',child_plan)
            child_command = ['/selected/bin/python',str(child/'code/profile_selected.py'),'--prepared-unit',str(child)]
            put(child/'command.txt',(' '.join(child_command)+'\n').encode())
            binding_ref = put(controller/'prepared_child_binding.json',{'controller_plan':parent_ref,'child_plan':child_ref,
                'controller_source':parent_source,'helpers':helper,'command':child_command})
            launch_review = put(controller/'controller_launch_independent_review.json',{'kind':'independent_profile_target_qualification_controller_launch_review',
                'approved':True,'passed':True,'independent_of_producer':True,'controller_plan':parent_ref,'child_plan':child_ref})
            procedure_ref = put(packet/'permission_procedure.json',{'kind':'root_only_owned_metadata_permission_protection_and_same_file_restore_procedure',
                'protected_mode_before_ssh':0o640,'restore_mode_after_actual_registration_or_verified_failed_startup_exit':0o660,
                'current_metadata_lock_snapshots':[{'path':str(run/'queue/metadata.lock')},{'path':str(run/'processes.lock')}]})
            log_ref = put(packet/'log_directory_evidence.json',{'path':str(log),'target_visibility_verified':True,
                'hostname':'serval02','owner_uid':TRANSPORT.UID})
            standards=[]
            for folder in (packet,controller):
                for name in ('manifest.yaml','config_resolved.yaml','command.txt','experiment_background.md'):
                    raw=(' '.join(TRANSPORT.command(controller)[8:])+'\n').encode() if folder==controller and name=='command.txt' else b'fixture metadata\n'
                    standards.append(put(folder/name,raw))
            configuration_preparer = load('prepare_direct_two_copy_transport_configuration')
            generated = configuration_preparer.prepare(packet,controller,Path(procedure_ref['path']),Path(log_ref['path']))
            self.assertFalse(generated['actual_transport_attempted'])
            with self.assertRaisesRegex(ValueError,'regenerate'):
                configuration_preparer.prepare(packet,controller,Path(procedure_ref['path']),Path(log_ref['path']))
            source_ref = generated['source']
            record={'schema_version':1,'kind':'direct_profile_target_qualification_root_transport','automatic_retry_permitted':False,
                'packet':str(packet),'campaign_run_dir':str(run),'transport_source':source_ref,'controller_unit':str(controller),
                'controller_plan':parent_ref,'child_plan':child_ref,'prepared_child_binding':binding_ref,
                'launch_review':launch_review,'controller_source':parent_source,'process_source':process_source,
                'metadata_permission_procedure':procedure_ref,'log_directory_evidence':log_ref,'standard_records':standards,
                'stdout':str(log/'controller_stdout.raw'),'stderr':str(log/'controller_stderr.raw')}
            configuration_ref = generated['configuration']
            self.assertEqual(TRANSPORT.document(configuration_ref),record)
            put(packet/'root_transport_independent_review.json',{'kind':'independent_direct_profile_target_qualification_root_transport_review',
                'approved':True,'passed':True,'independent_of_producer':True,'configuration':configuration_ref,'source':source_ref,'unresolved_anomalies':[]})
            publication='a'*40
            def git(root_arg,*argv):
                if argv==('rev-parse','--show-toplevel'):return (str(root)+'\n').encode()
                if argv==('config','--get-all','remote.origin.url'):return b'git@github.com:yuxinchen123/uncertainty_comparison_pointmaze.git\n'
                if argv==('config','--get-all','remote.origin.fetch'):return b'+refs/heads/*:refs/remotes/origin/*\n'
                if argv in (('merge-base','--is-ancestor',publication,'HEAD'),('merge-base','--is-ancestor',publication,'refs/remotes/origin/merge-atari-cleanrl')):return b''
                self.assertEqual(argv[0],'show')
                commit,relative=argv[1].split(':',1)
                if commit==TRANSPORT.RUNTIME:return helper_raw[Path(relative).name]
                self.assertEqual(commit,publication)
                return (root/relative).read_bytes()
            with patch.object(TRANSPORT,'git',side_effect=git), patch.object(TRANSPORT,'__file__',source_ref['path']):
                checked, actual_controller, argv = TRANSPORT.verify(Path(configuration_ref['path']),publication)
                self.assertEqual(actual_controller,controller)
                self.assertEqual(argv,TRANSPORT.command(controller))
                put(child/'measurement_plan.json',{**child_plan,'copy_seeds':[True]})
                with self.assertRaisesRegex(ValueError,'not published with these exact bytes'):
                    TRANSPORT.verify(Path(configuration_ref['path']),publication)
            self.assertFalse((packet/'root_transport_popen_attempt.json').exists())
            self.assertFalse((packet/'root_actual_controller_spawn.json').exists())
    def test_process_cleanup_and_metadata_protection_functions_are_exact(self):
        old=(D/'preserved_one_copy_runtime_4bb/start_direct_profile_qualification_once.py').read_text()
        new=(OPERATIONS/'start_direct_two_copy_profile_qualification_once.py').read_text()
        def functions(source):return {node.name:ast.get_source_segment(source,node) for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
        before,after=functions(old),functions(new)
        self.assertEqual(set(before),set(after))
        for name in before:
            if name!='verify':self.assertEqual(before[name],after[name],name)
        old=(D/'preserved_one_copy_runtime_4bb/prepare_direct_transport_configuration.py').read_text()
        new=(OPERATIONS/'prepare_direct_two_copy_transport_configuration.py').read_text()
        before,after=functions(old),functions(new)
        self.assertEqual(set(before),set(after))
        for name in before:
            if name!='prepare':self.assertEqual(before[name],after[name],name)
        self.assertEqual(TRANSPORT.RUNTIME,'4bb3a61cb92fe48ab5ba68fb52f70b47736a1174')

    def test_preparer_refuses_wrong_count_seeds_and_original_reference(self):
        module=load('prepare_direct_two_copy_transport_configuration')
        for copies,seeds,reference_copies in [(1,[1],1),(2.0,[1,2],1),(2,[True,2],1),(2,[1,1],1),(2,[1,2],2)]:
            with self.subTest(copies=copies,seeds=seeds,reference_copies=reference_copies),tempfile.TemporaryDirectory() as temporary:
                run=Path(temporary)/'campaign';packet=run/'remote_setup/capture';packet.mkdir(parents=True)
                parent=run/'data/controller';(parent/'code').mkdir(parents=True)
                child=run/'data/capture';(child/'code').mkdir(parents=True)
                def put(path,value):path.write_text(json.dumps(value));return TRANSPORT.capture(path)[1]
                original=put(run/'original.json',{'configuration':{'learner':{'n_copies':reference_copies}},'copy_seeds':[1]})
                parent_ref=put(parent/'measurement_plan.json',{'kind':'profile_target_qualification_controller',
                    'profile_unit':str(child),'original_qualification_plan':original})
                child_ref=put(child/'measurement_plan.json',{'kind':'full_original_profile_source_target_qualification',
                    'hardware_class':'h100','copy_seeds':seeds,'configuration':{'learner':{'n_copies':copies}}})
                put(parent/'prepared_child_binding.json',{'controller_plan':parent_ref,'child_plan':child_ref})
                put(parent/'controller_launch_independent_review.json',{'kind':'independent_profile_target_qualification_controller_launch_review',
                    'approved':True,'passed':True,'independent_of_producer':True,'controller_plan':parent_ref,'child_plan':child_ref})
                with self.assertRaises(ValueError):module.prepare(packet,parent,run/'absent_procedure',run/'absent_visibility')
                self.assertFalse((packet/'root_transport_configuration.json').exists())
                self.assertFalse((packet/'start_direct_two_copy_profile_qualification_once.py').exists())

    def test_default_preparer_does_not_write_or_start(self):
        module=load('prepare_direct_two_copy_transport_configuration')
        with patch.object(module,'prepare',side_effect=AssertionError('no action')):
            module.main(['--packet','/absent','--controller-unit','/absent','--permission-procedure','/absent',
                '--log-directory-evidence','/absent','--owned-job-id','6825983'])


if __name__=='__main__':unittest.main(verbosity=2)
