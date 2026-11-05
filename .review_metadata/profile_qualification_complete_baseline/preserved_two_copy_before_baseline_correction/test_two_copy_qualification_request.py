"""Check count-specific metadata with real plans and explicitly synthetic upstream premises."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

D=Path(__file__).resolve().parent
W=D.parent.parent
SOURCE=W/'.review_metadata/profile_target_operational_preparation/derive_two_copy_qualification_request.py'
HELPER=Path('/p/rlprojects/RND_worktrees/2026-11-05-11-56_profile-qualification-helper_source-d1469fd063a4_h100-helpers-44_a4500-helpers-35_gpu-count-0')
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
FACT_PACKET=R/'remote_setup/2026-11-05-11-58_common-performance-verification_hardware-types-2_training-copies-1-2_paired-rounds-7_updates-per-arm-10_cpu-count-1_gpu-count-0_time-limit-seconds-300'
spec=importlib.util.spec_from_file_location('two_copy_operational_fixture',SOURCE)
DERIVER=importlib.util.module_from_spec(spec);spec.loader.exec_module(DERIVER)


def load_json(path):return json.loads(Path(path).read_bytes())


class TwoCopyRequest(unittest.TestCase):
    """No real request, unit, prepin, job, child, tool or numerical operation is created."""
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.run=self.root/'campaign';(self.run/'remote_setup').mkdir(parents=True);(self.run/'data').mkdir()
        self.fact=load_json(FACT_PACKET/'four_context_prerequisite_fact.json')
        self.q=DERIVER.runtime(HELPER)
        self.backend=next((R/'remote_setup').glob('2026-11-05-12-06_profile-backend-preparation*/profile_qualification_backend_expected.json'))

    def tearDown(self):self.temporary.cleanup()

    def put(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return path

    def fixture(self,hardware):
        entry=next(row for row in self.fact['contexts'] if row['hardware_class']==hardware and row['copies']==2)
        accepted=load_json(entry['review']['path']);common=load_json(accepted['plan']['path'])
        # Only the campaign path and accepted plan reference are replaced for a private fixture.
        common['campaign_run_dir']=str(self.run)
        plan=self.put('private_common_plan.json',common);accepted['plan']=self.q.reference(plan)
        accepted_review=self.put('private_review.json',accepted)
        synthetic_fact=copy.deepcopy(self.fact)
        next(row for row in synthetic_fact['contexts'] if row['hardware_class']==hardware and row['copies']==2)['review']=self.q.reference(accepted_review)
        proof=self.put('private_fact.json',synthetic_fact);review=self.put('private_fact_review.json',{'explicit_fixture':True})
        if hardware=='jaguar03_rtx_a4500':
            component_ref=self.fact['native_process_qualification']['diagnostic_result']
            component_review=self.fact['native_process_qualification']['independent_review']
        else:
            n=load_json(R/'next_actions.json')['native_direct_process_diagnostic_source']
            component_ref=n['actual_target_result'];component_review=n['independent_actual_review']
        args=types.SimpleNamespace(helper_root=HELPER,run_dir=self.run,packet=self.run/'remote_setup/two-copy-capture',
            common_proof=proof,common_review=review,component_result=Path(component_ref['path']),
            component_review=Path(component_review['path']),backend_record=self.backend,hardware_class=hardware,
            controller_unit_id='two-copy-controller',profile_unit_id='two-copy-capture')
        return args,common,synthetic_fact,plan,proof

    def invoke(self,args,common):
        profile=self.q.adjacent({'helpers':{'files':{DERIVER.PREFIX+'profile_selected.py':hashlib.sha256((HELPER/DERIVER.PREFIX/'profile_selected.py').read_bytes()).hexdigest()}}},'profile_selected')
        helper=profile.helper_receipt(configured=args.hardware_class=='jaguar03_rtx_a4500',direct=args.hardware_class=='h100',
            owned_processes=args.hardware_class=='jaguar03_rtx_a4500',direct_profile=args.hardware_class=='h100',
            configured_native=args.hardware_class=='jaguar03_rtx_a4500',configured_prerequisites=args.hardware_class=='jaguar03_rtx_a4500',profile_target=True)
        metadata=self.q.adjacent({'helpers':helper},'configured_profile_prerequisites')
        historical=next(row for row in self.fact['contexts'] if row['hardware_class']=='h100' and row['copies']==1)
        history=load_json(load_json(historical['review']['path'])['plan']['path'])
        # The fixture does not consume/reapprove the real snapshot; the exact runtime gate is a declared premise.
        registry={'path':'/explicit/private-fixture-registry','sha256':'1'*64}
        def accepted_four_contexts(proposed,proof,review):
            self.assertEqual(proposed['copy_seeds'],[1,2]);self.assertEqual(proposed['configuration']['learner']['n_copies'],2)
            self.assertEqual(proposed['inputs']['accepted_confirmation'],load_json(next(row['review']['path'] for row in load_json(args.common_proof)['contexts'] if row['hardware_class']==args.hardware_class and row['copies']==2))['result'])
            return {'contexts':[{'hardware_class':'h100','copies':1,'historical_plan':history}]}
        with patch.object(DERIVER,'runtime',return_value=self.q),patch.object(self.q,'require_four_contexts',side_effect=accepted_four_contexts),\
                patch.object(metadata,'initial_snapshot',return_value={'historical_registry':registry}):
            return DERIVER.derive(args)

    def test_both_real_hardware_layouts_emit_only_two_copy_requests(self):
        for hardware in ('h100','jaguar03_rtx_a4500'):
            with self.subTest(hardware=hardware):
                args,common,fact,plan,proof=self.fixture(hardware)
                if args.packet.exists():
                    # Separate private fixture campaign for the second independent hardware case.
                    self.tearDown();self.setUp();args,common,fact,plan,proof=self.fixture(hardware)
                result=self.invoke(args,common)
                request=load_json(args.packet/'preparation_request.json');argv=request['profile_prepare_arguments']
                self.assertFalse(result['actual_runtime_prepared']);self.assertEqual(result['gpu_operations'],0)
                self.assertEqual(load_json(args.packet/'configuration.json')['learner']['n_copies'],2)
                self.assertEqual(load_json(args.packet/'config_resolved.yaml')['copy_seeds'],[1,2])
                self.assertEqual(argv[argv.index('--seed-start')+1],'1')
                original=load_json(request['original_qualification_plan']['path'])
                self.assertEqual(original['configuration']['learner']['n_copies'],1)
                self.assertEqual(original['copy_seeds'],[1])
                self.assertEqual(request['controller_maximum_seconds'],10800)
                self.assertFalse((self.run/'data/two-copy-controller').exists())
                self.assertFalse((self.run/'data/two-copy-capture').exists())
                if hardware=='h100':
                    self.assertEqual(argv[argv.index('--direct-confirmation-review')+1],next(row['review']['path'] for row in fact['contexts'] if row['hardware_class']=='h100' and row['copies']==2))
                    self.assertEqual(argv[argv.index('--direct-backend-image-record')+1],request['backend_image_record']['path'])
                self.assertFalse(load_json(args.packet/'manifest.yaml')['admission_granted'])

    def test_missing_count_two_never_uses_count_one(self):
        args,common,fact,plan,proof=self.fixture('h100')
        fact['contexts']=[row for row in fact['contexts'] if not(row['hardware_class']=='h100' and row['copies']==2)]
        proof.write_text(json.dumps(fact))
        with patch.object(DERIVER,'runtime',return_value=self.q),self.assertRaisesRegex(ValueError,'two-copy context'):
            DERIVER.derive(args)
        self.assertFalse(args.packet.exists())

    def test_count_one_or_boolean_seeds_refuse_before_output(self):
        for update in ({'copy_seeds':[1]}, {'copy_seeds':[True,2]}):
            args,common,fact,plan,proof=self.fixture('h100');common.update(update);plan.write_text(json.dumps(common))
            selected=next(row for row in fact['contexts'] if row['hardware_class']=='h100' and row['copies']==2)
            accepted=load_json(selected['review']['path']);accepted['plan']=self.q.reference(plan)
            Path(selected['review']['path']).write_text(json.dumps(accepted));selected['review']=self.q.reference(selected['review']['path']);proof.write_text(json.dumps(fact))
            with patch.object(DERIVER,'runtime',return_value=self.q),self.assertRaisesRegex(ValueError,'common context differs'):
                DERIVER.derive(args)
            self.assertFalse(args.packet.exists())

    def test_original_reference_must_remain_one_copy(self):
        args,common,fact,plan,proof=self.fixture('h100')
        key='direct_qualification_inputs'
        old=Path(common[key]['result']['path']).parent/'measurement_plan.json'
        original=load_json(old)
        original['configuration']['learner']['n_copies']=2
        original['reference_configuration']['learner']['n_copies']=2
        original['copy_seeds']=[1,2]
        copied=self.root/'explicit_invalid_original_reference';copied.mkdir()
        (copied/'measurement_plan.json').write_text(json.dumps(original))
        (copied/'result.json').write_text('{}')
        common[key]['result']=self.q.reference(copied/'result.json')
        plan.write_text(json.dumps(common))
        selected=next(row for row in fact['contexts'] if row['hardware_class']=='h100' and row['copies']==2)
        accepted=load_json(selected['review']['path']);accepted['plan']=self.q.reference(plan)
        Path(selected['review']['path']).write_text(json.dumps(accepted))
        selected['review']=self.q.reference(selected['review']['path']);proof.write_text(json.dumps(fact))
        with self.assertRaisesRegex(ValueError,'one reference copy'):
            self.invoke(args,common)
        self.assertFalse(args.packet.exists())

    def test_existing_one_copy_and_runtime_files_are_unchanged(self):
        import subprocess
        relative='.review_metadata/profile_target_operational_preparation/derive_one_copy_qualification_request.py'
        before=subprocess.check_output(['git','-C',str(W),'show','cf036bcee1459165366553ffbc5eea57a1843c38:'+relative])
        self.assertEqual((W/relative).read_bytes(),before)
        self.assertEqual(subprocess.check_output(['git','-C',str(W),'diff','--name-only','cf036bcee1459165366553ffbc5eea57a1843c38','--','10_jax_exploration_platform/scripts/atari_pixel_campaign']),b'')


if __name__=='__main__':unittest.main(verbosity=2)
