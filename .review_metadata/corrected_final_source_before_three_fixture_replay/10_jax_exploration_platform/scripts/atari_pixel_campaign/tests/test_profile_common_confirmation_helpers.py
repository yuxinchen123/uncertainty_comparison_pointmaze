"""Test the profile-only reader closure with explicit synthetic metadata and no numerical work."""
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import profile_selected as profile
import configured_measurement as support
import test_profile_selected as profile_fixtures

PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'


class ProfileCommonConfirmationHelpersTests(unittest.TestCase):
    """Keep an accepted common confirmation's reader in the staged profile without changing its protocol."""

    def setUp(self):
        """Reuse full configuration fixtures while marking every prior scientific gate substitution."""
        self.fixture=profile_fixtures.SelectedProfileTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.profile_path=self.fixture.run/'configured_profile.json'
        self.profile_path.write_text(json.dumps(support.core().PROFILE))
        self.fixture.confirmation.update(hardware_class='jaguar03_rtx_a4500',configured_isolation={},
            common_correctness_inputs={'result':{'path':'/synthetic_metadata/correctness.json','sha256':'a'*64},
                'review':{'path':'/synthetic_metadata/review.json','sha256':'b'*64}})
        self.fixture.files['confirmation'].write_text(json.dumps(self.fixture.confirmation))
        self.fixture.documents['baseline']['configured_isolation_profile']=support.core().PROFILE
        self.fixture.files['baseline'].write_text(json.dumps(self.fixture.documents['baseline']))
        self.legacy_names=profile.required_helpers(True,False,True)
        self.names=profile.required_helpers(True,False,True,True)
        self.helpers={'source_root':str(profile.SCRIPTS.parents[2]),'source_commit':'b'*40,
            'files':{PREFIX+name:hashlib.sha256((profile.SCRIPTS/name).read_bytes()).hexdigest() for name in self.names}}

    def prepared(self):
        """Stage an explicitly synthetic profile; all prior qualification and timing checks remain mocked."""
        args=self.fixture.arguments();args.gpu_uuid=None;args.hardware_class='jaguar03_rtx_a4500'
        args.configured_isolation_profile=self.profile_path;args.python=Path(sys.executable)
        args.target_qualification_record=self.fixture.run/'synthetic_qualification.json'
        args.target_qualification_review=self.fixture.run/'synthetic_qualification_review.json'
        def previous(plan,*unused):
            """Record only a named metadata substitution, never a claimed measured qualification."""
            plan['target_qualification_inputs']={'synthetic_metadata_fixture':True}
        def source_receipt(root,commit,paths):
            """Retain the real requested path roster while substituting scientific source checks."""
            return {'source_root':str(root),'source_commit':commit,'files':{name:'c'*64 for name in paths},
                'synthetic_metadata_fixture':True}
        with patch.object(profile.diagnostic,'core_identity',return_value={'synthetic_metadata_fixture':True}), \
                patch.object(profile,'require_reviewed_profile_process_admission'), \
                patch.object(profile.training,'source_receipt',side_effect=source_receipt), \
                patch.object(profile,'helper_receipt',return_value=self.helpers), \
                patch.object(support,'prepare_qualification_inputs',side_effect=previous), \
                patch.object(support,'verify_saved_measurement') as saved:
            directory=profile.prepare(args)
        saved.assert_called_once()
        return directory,json.loads((directory/'measurement_plan.json').read_bytes())

    def verify(self,directory,plan):
        """Exercise actual staged-byte checks while explicitly substituting unrelated stable source gates."""
        with ExitStack() as stack:
            stack.enter_context(patch.object(profile,'require_reviewed_profile_process_admission'))
            stack.enter_context(patch.object(profile,'__file__',str(directory/'code/profile_selected.py')))
            for name,module in (('measure_selected.py',profile.measurement),('diagnose_gpu_execution.py',profile.diagnostic),
                    ('post_comparison_validation.py',profile.validation),('train_optimized.py',profile.training)):
                stack.enter_context(patch.object(module,'__file__',str(directory/'code'/name)))
            stack.enter_context(patch.object(profile.measurement,'producer_support'))
            stack.enter_context(patch.object(profile.diagnostic,'core_identity',return_value=plan['core']))
            stack.enter_context(patch.object(profile.training,'source_receipt',return_value=plan['paired_source']))
            profile.verify_prepared(directory,plan)

    def test_conditional_union_adds_only_the_common_reader(self):
        """The new configured closure differs by exactly one file; historical default/direct rosters stay fixed."""
        for configured,direct in ((False,False),(True,False),(False,True)):
            old=tuple(dict.fromkeys(('profile_selected.py',*profile.measurement.required_helpers(configured,direct))))
            self.assertEqual(profile.required_helpers(configured,direct),old)
        self.assertEqual(set(self.legacy_names)-set(profile.required_helpers(True)),{'common_confirmation.py'})
        self.assertEqual(len(self.legacy_names),len(profile.required_helpers(True))+1)
        self.assertEqual(set(self.names)-set(self.legacy_names),{'nsight_process_identity.py'})

    def test_default_and_direct_cannot_request_the_common_reader(self):
        """The extra A4500 reader never opts another consumer into common-confirmation support."""
        for configured,direct in ((False,False),(False,True),(True,True)):
            with self.subTest(configured=configured,direct=direct),self.assertRaises(ValueError):
                profile.required_helpers(configured,direct,True)

    def test_preparation_records_the_reader_without_claiming_own_correctness_inputs(self):
        """The declaration belongs only to the accepted input, not the profile's numerical protocol."""
        directory,plan=self.prepared()
        self.assertIs(plan['accepted_confirmation_requires_common_reader'],True)
        self.assertNotIn('common_correctness_inputs',plan)
        self.assertEqual(set(plan['helpers']['files']),{PREFIX+name for name in self.names})
        self.assertEqual((directory/'code/common_confirmation.py').read_bytes(),(profile.SCRIPTS/'common_confirmation.py').read_bytes())
        self.verify(directory,plan)

    def test_missing_or_changed_staged_reader_refuses_before_numerical_work(self):
        """A declared file cannot be dropped or changed after the profile is staged."""
        directory,plan=self.prepared();path=directory/'code/common_confirmation.py';raw=path.read_bytes();path.unlink()
        with self.assertRaises(OSError):self.verify(directory,plan)
        path.write_bytes(raw+b'\n# changed synthetic helper\n')
        with self.assertRaisesRegex(ValueError,'staged profile helper'):self.verify(directory,plan)

    def test_helper_receipt_requests_the_same_conditional_roster(self):
        """The committed receipt and staged bytes use one complete helper list."""
        with patch.object(profile.subprocess,'check_output',return_value='c'*40+'\n'), \
                patch.object(profile.training,'source_receipt',return_value={'synthetic_receipt':True}) as receipt:
            value=profile.helper_receipt(True,False,True,True)
        self.assertEqual(value,{'synthetic_receipt':True})
        self.assertEqual(receipt.call_args.args[2],[PREFIX+name for name in self.names])

    def test_changed_or_coherently_rehashed_accepted_field_cannot_change_the_declaration(self):
        """Input bytes and the declared helper requirement both remain bound."""
        directory,plan=self.prepared();path=Path(plan['inputs']['accepted_confirmation']['path'])
        changed=json.loads(path.read_bytes());changed.pop('common_correctness_inputs');path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'confirmation changed'):profile.accepted_confirmation_common_reader(plan)
        plan['inputs']['accepted_confirmation']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'helper declaration'):profile.accepted_confirmation_common_reader(plan)

    def test_missing_false_or_non_boolean_declaration_refuses(self):
        """A common record cannot silently use an old or ambiguous helper declaration."""
        _,plan=self.prepared()
        for value in (None,False,1,'true'):
            changed=copy.deepcopy(plan)
            if value is None:changed.pop('accepted_confirmation_requires_common_reader')
            else:changed['accepted_confirmation_requires_common_reader']=value
            with self.subTest(value=value),self.assertRaises(ValueError):
                profile.accepted_confirmation_common_reader(changed)

    def test_removed_helper_map_entry_refuses_even_if_the_file_remains(self):
        """The plan must declare the file instead of relying on a nearby source checkout."""
        _,plan=self.prepared();plan['helpers']['files'].pop(PREFIX+'common_confirmation.py')
        with self.assertRaisesRegex(ValueError,'optional reader'):profile.accepted_confirmation_common_reader(plan)

    def test_direct_or_default_readback_cannot_borrow_the_common_declaration(self):
        """Changing only a consumer's metadata cannot widen its accepted evidence support."""
        _,plan=self.prepared()
        for direct in (False,True):
            changed=copy.deepcopy(plan);changed.pop('configured_isolation')
            if direct:changed['direct_binding']={'synthetic_metadata_fixture':True}
            with self.subTest(direct=direct),self.assertRaises(ValueError):
                profile.accepted_confirmation_common_reader(changed)

    def test_baseline_replay_checks_the_declaration_before_cohort_readers(self):
        """A dropped declaration fails before any original-case or combination validator is invoked."""
        directory,plan=self.prepared();plan.pop('accepted_confirmation_requires_common_reader')
        with patch.object(profile.validation,'validate_cohort_case') as cases,patch.object(profile.validation,'validate_combination') as combination:
            with self.assertRaisesRegex(ValueError,'helper declaration'):profile.validate_measured_baseline(directory,plan)
        cases.assert_not_called();combination.assert_not_called()

    def test_unreviewed_backend_refusal_is_not_changed_by_the_extra_file(self):
        """A valid closure still propagates a scientific backend refusal without a baseline-validation record."""
        directory,plan=self.prepared()
        with patch.object(profile.validation,'validate_cohort_case',return_value={'synthetic_cohort_check':True}), \
                patch.object(profile.validation,'validate_combination',side_effect=ValueError('unreviewed backend proof remains refused')):
            with self.assertRaisesRegex(ValueError,'unreviewed backend'):profile.validate_measured_baseline(directory,plan)
        self.assertFalse((directory/'baseline_validation.json').exists())


if __name__=='__main__':
    unittest.main()

