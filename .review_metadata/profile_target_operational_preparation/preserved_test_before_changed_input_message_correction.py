"""Check operational metadata boundaries without preparing or launching a target unit."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

D = Path(__file__).resolve().parent


def load(name):
    specification = importlib.util.spec_from_file_location(name, D / (name + '.py'))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


DERIVER = load('derive_one_copy_qualification_request')
TRANSPORT = load('start_direct_profile_qualification_once')


class OperationalChecks(unittest.TestCase):
    """Use explicit synthetic upstream premises; these fixtures grant no qualification."""
    def test_proven_custody_functions_are_exact(self):
        old = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-05-05-49_common-timing_cache-room-backgrounds-true_render-final-two-only-true_h100-nvl_serval02-card-1_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_paired-rounds-7_updates-per-arm-10_seed-start-1/root_start_controller_once.py')
        sources = [old.read_text(), Path(TRANSPORT.__file__).read_text()]
        functions = [{node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)} for source in sources]
        for name in ('write', 'retain_custody', 'cleanup_under_barrier'):
            self.assertEqual(functions[0][name], functions[1][name])

    def test_actual_controller_command_preserves_source_and_selector(self):
        controller = Path('/exact/controller')
        command = TRANSPORT.command(controller)
        self.assertEqual(command[8:], ['/usr/bin/python3', '-I', '-S', '-B',
            '/exact/controller/code/profile_target_qualification.py', '--prepared-unit', '/exact/controller'])
        self.assertIn('CUDA_VISIBLE_DEVICES=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3', command)
        self.assertNotIn('--launch', command)

    def test_changed_transport_source_and_duplicate_json_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            path.write_text('{"key":1,"key":2}')
            ref = TRANSPORT.capture(path)[1]
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                TRANSPORT.document(ref)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                TRANSPORT.document(ref)
            link = Path(directory) / 'link.json'
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, 'bounded exact'):
                TRANSPORT.capture(link)

    def test_default_request_deriver_never_prepares_or_writes(self):
        argv = ['request', '--helper-root', '/absent', '--run-dir', '/absent', '--packet', '/absent/new',
            '--common-proof', '/absent', '--common-review', '/absent', '--component-result', '/absent',
            '--component-review', '/absent', '--backend-record', '/absent', '--hardware-class', 'h100',
            '--controller-unit-id', 'controller', '--profile-unit-id', 'profile', '--owned-job-id', '6825983']
        with patch('sys.argv', argv), patch.object(DERIVER, 'derive', side_effect=AssertionError('no action')):
            DERIVER.main()

    def test_complete_synthetic_transport_checks_and_changed_plan_refuse(self):
        """Exercise every actual metadata check with explicit synthetic review/Git premises."""
        root = D.parent.parent
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            run = Path(temporary) / 'campaign';packet = run/'remote_setup/packet';packet.mkdir(parents=True)
            controller = run/'data/controller';(controller/'code').mkdir(parents=True)
            child = run/'data/capture';(child/'code').mkdir(parents=True)
            log = run/'slurm/direct_jobs/packet';log.mkdir(parents=True)
            def put(path, value):
                path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
                return TRANSPORT.capture(path)[1]
            names = ['profile_target_qualification.py', 'direct_common_timing_controller.py'] + ['synthetic_helper_'+str(i)+'.py' for i in range(42)]
            helper_raw = {name: ('# explicit synthetic source premise '+name+'\n').encode() for name in names}
            helper = {'source_commit': TRANSPORT.RUNTIME, 'files': {TRANSPORT.PREFIX+name: hashlib.sha256(raw).hexdigest() for name, raw in helper_raw.items()}}
            for name, raw in helper_raw.items():put(child/'code'/name, raw)
            parent_source = put(controller/'code/profile_target_qualification.py', helper_raw['profile_target_qualification.py'])
            process_source = put(controller/'code/direct_common_timing_controller.py', helper_raw['direct_common_timing_controller.py'])
            parent = {'unit_id':'controller','kind':'profile_target_qualification_controller','profile_unit':str(child),
                'campaign_run_dir':str(run),'source_commit':TRANSPORT.SOURCE,'controller_maximum_seconds':10800,
                'qualification_time_budget':{'schema_version':1,'work_seconds':10200,'cleanup_seconds':600,'total_seconds':10800}}
            parent_ref = put(controller/'measurement_plan.json',parent)
            child_plan = {'unit_id':'capture','source_commit':TRANSPORT.SOURCE,'kind':'full_original_profile_source_target_qualification',
                'hardware_class':'h100','copy_seeds':[1],'configuration':{'learner':{'n_copies':1}},'helpers':helper,
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
            source_ref = TRANSPORT.capture(TRANSPORT.__file__)[1]
            record={'schema_version':1,'kind':'direct_profile_target_qualification_root_transport','automatic_retry_permitted':False,
                'packet':str(packet),'campaign_run_dir':str(run),'transport_source':source_ref,'controller_unit':str(controller),
                'controller_plan':parent_ref,'child_plan':child_ref,'prepared_child_binding':binding_ref,
                'launch_review':launch_review,'controller_source':parent_source,'process_source':process_source,
                'metadata_permission_procedure':procedure_ref,'log_directory_evidence':log_ref,'standard_records':standards,
                'stdout':str(log/'controller_stdout.raw'),'stderr':str(log/'controller_stderr.raw')}
            configuration_ref = put(packet/'root_transport_configuration.json',record)
            put(packet/'root_transport_independent_review.json',{'kind':'independent_direct_profile_target_qualification_root_transport_review',
                'approved':True,'passed':True,'independent_of_producer':True,'configuration':configuration_ref,'source':source_ref,'unresolved_anomalies':[]})
            publication='a'*40
            def git(root_arg,*argv):
                if argv==('rev-parse','--show-toplevel'):return (str(root)+'\n').encode()
                if argv==('merge-base','--is-ancestor',publication,'HEAD'):return b''
                self.assertEqual(argv[0],'show')
                commit,relative=argv[1].split(':',1)
                if commit==TRANSPORT.RUNTIME:return helper_raw[Path(relative).name]
                self.assertEqual(commit,publication)
                return (root/relative).read_bytes()
            with patch.object(TRANSPORT,'git',side_effect=git):
                checked, actual_controller, argv = TRANSPORT.verify(Path(configuration_ref['path']),publication)
                self.assertEqual(actual_controller,controller)
                self.assertEqual(argv,TRANSPORT.command(controller))
                put(child/'measurement_plan.json',{**child_plan,'copy_seeds':[True]})
                with self.assertRaisesRegex(ValueError,'bytes changed'):
                    TRANSPORT.verify(Path(configuration_ref['path']),publication)
            self.assertFalse((packet/'root_transport_popen_attempt.json').exists())
            self.assertFalse((packet/'root_actual_controller_spawn.json').exists())

    def test_request_uses_actual_component_plan_not_result_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);run = root / 'campaign';(run / 'remote_setup').mkdir(parents=True);(run / 'data').mkdir()
            helper = root / 'helper';(helper / DERIVER.PREFIX).mkdir(parents=True)
            (helper / DERIVER.PREFIX / 'profile_selected.py').write_text('# synthetic helper premise\n')
            def put(name, value):
                path = root / name;path.write_text(json.dumps(value));return path
            def reference(path):
                path = Path(path);return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            def checked(ref):return json.loads(Path(ref['path']).read_bytes())
            selected_python = '/selected/bin/python'
            configuration = {'learner': {'n_copies': 1}}
            original = put('original.json', {'source_commit': DERIVER.SOURCE, 'python': selected_python})
            qualification_dir = root / 'qualification';qualification_dir.mkdir()
            (qualification_dir / 'measurement_plan.json').write_bytes(original.read_bytes())
            qualification_result = put('qualification/result.json', {})
            profile_input = put('profile.json', {'schema_version': 1})
            common = put('common.json', {'hardware_class': 'jaguar03_rtx_a4500', 'copy_seeds': [1],
                'configuration': configuration, 'python': selected_python, 'source_commit': DERIVER.SOURCE,
                'source_root': '/source', 'runtime_environment': {}, 'asset_record': {},
                'inputs': {'configured_isolation_profile': reference(profile_input)},
                'configured_isolation': {'profile': checked(reference(profile_input))},
                'target_qualification_inputs': {'result': reference(qualification_result), 'review': {'path': '/old/review', 'sha256': '0'*64}}})
            accepted = put('accepted.json', {'plan': reference(common), 'result': {'path': '/old/result', 'sha256': '1'*64}})
            proof = put('proof.json', {'contexts': [{'hardware_class': 'jaguar03_rtx_a4500', 'copies': 1, 'review': reference(accepted)}],
                'initial_snapshot_inputs': {}})
            review = put('review.json', {})
            component_plan = put('component_plan.json', {'python': selected_python, 'profile_tool': {'path': '/tool', 'sha256': '2'*64}})
            component = put('component.json', {'plan': reference(component_plan)})
            component_review = put('component_review.json', {'approved': True, 'passed': True, 'result': reference(component)})
            backend = put('backend.json', {})
            q = types.SimpleNamespace(reference=reference, checked=checked,
                PROTOCOL='full-original-profile-source-target-qualification-version-1', ZERO={},
                QUALIFICATION_BUDGET={'total_seconds': 10800},
                require_four_contexts=lambda *args: {'contexts': [{'hardware_class': 'h100', 'copies': 1, 'historical_plan': {}}]})
            metadata = types.SimpleNamespace(MetadataFiles=lambda: None,
                initial_snapshot=lambda *args: {'historical_registry': {'path': '/registry', 'sha256': '3'*64}})
            profile = types.SimpleNamespace(helper_receipt=lambda **kwargs: {'files': {}})
            backend_api = types.SimpleNamespace(validate_expected=lambda *args: None)
            q.adjacent = lambda plan, name: {'profile_selected': profile, 'configured_profile_prerequisites': metadata,
                'profile_qualification_backend': backend_api}[name]
            q.publish = lambda path, value: path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
            args = types.SimpleNamespace(helper_root=helper, run_dir=run, packet=run/'remote_setup/actual-packet',
                common_proof=proof, common_review=review, component_result=component, component_review=component_review,
                backend_record=backend, hardware_class='jaguar03_rtx_a4500', controller_unit_id='controller', profile_unit_id='capture')
            real_reference = q.reference
            def synthetic_review_reference(path):
                value = real_reference(path)
                if Path(path) == component_review:
                    value['sha256'] = '4264288c209606dccc15c883724ffeccaf5a480c943c5e29dd9b1c409c814da5'
                return value
            q.reference = synthetic_review_reference
            with patch.object(DERIVER, 'runtime', return_value=q):
                result = DERIVER.derive(args)
            self.assertFalse(result['actual_runtime_prepared'])
            self.assertFalse((run/'data/controller').exists())
            self.assertEqual(json.loads((args.packet/'profile_tool_record.json').read_bytes()), {'path': '/tool', 'sha256': '2'*64})
            request = json.loads((args.packet/'preparation_request.json').read_bytes())
            self.assertEqual(request['controller_maximum_seconds'], 10800)
            self.assertNotIn('--direct-backend-image-record', request['profile_prepare_arguments'])


if __name__ == '__main__':unittest.main(verbosity=2)
