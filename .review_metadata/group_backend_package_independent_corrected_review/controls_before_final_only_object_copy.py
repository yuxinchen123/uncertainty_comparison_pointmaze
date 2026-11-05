"""Independently test the three group-package corrections using private, unexecuted file fixtures."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
import test_group_backend_package as original
B = original.backend


class IndependentCorrectionControls(unittest.TestCase):
    """Keep fixture work separate from any actual qualification or group execution."""
    def setUp(self):
        """Create only a small private synthetic package and reserved worker."""
        self.f = original.GroupBackendPackageTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def test_one_two_and_eight_copy_totals_have_explicit_per_copy_counters(self):
        """The same logical update retains per-copy counters and scales actual declared transition totals."""
        f = self.f; f.plan['mode'] = 'qualification'
        for copies in (1, 2, 8):
            f.plan['configuration']['learner']['n_copies'] = copies
            f.plan['copy_seeds'] = [list(range(1, copies + 1))]
            f.save_plan_and_binding(); result = f.result()
            result['executed_work'] = {'normalization_warmup_transitions': 50 * 128 * 128 * copies,
                'training_transitions': 128 * 128 * copies, 'complete_updates': 1, 'minibatch_updates': 16}
            self.assertEqual(B.completed_work(f.worker, f.plan, result), 'qualification')

    def test_old_one_copy_totals_are_explicitly_rejected_for_multiple_copies(self):
        """Do not rely on the corrected fixture's result generator to construct the old invalid input."""
        f = self.f; f.plan['mode'] = 'qualification'
        for copies in (2, 8):
            f.plan['configuration']['learner']['n_copies'] = copies
            f.plan['copy_seeds'] = [list(range(1, copies + 1))]
            f.save_plan_and_binding(); result = f.result()
            result['executed_work'] = {'normalization_warmup_transitions': 819200,
                'training_transitions': 16384, 'complete_updates': 1, 'minibatch_updates': 16}
            with self.assertRaisesRegex(ValueError, 'controlled reference update is incomplete'):
                B.completed_work(f.worker, f.plan, result)

    def test_each_qualification_total_and_counter_rejects_boolean_or_float_substitution(self):
        """Strict serialized work proof must not equate Boolean or floating fields to integer counters."""
        f = self.f; f.plan['mode'] = 'qualification'; result = f.result()
        for field, changed in (('normalization_warmup_transitions', 819200.0), ('training_transitions', 16384.0),
                ('complete_updates', True), ('minibatch_updates', 16.0)):
            value = copy.deepcopy(result); value['executed_work'][field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                B.completed_work(f.worker, f.plan, value)

    def test_qualification_copies_must_be_integer_and_match_the_seed_count(self):
        """The scaled count cannot be guessed from an inconsistent policy roster."""
        f = self.f; f.plan['mode'] = 'qualification'; result = f.result()
        for copies in (True, 2.0, 0, 2):
            f.plan['configuration']['learner']['n_copies'] = copies
            with self.subTest(copies=copies), self.assertRaisesRegex(ValueError, 'copy count'):
                B.completed_work(f.worker, f.plan, result)

    def test_coherent_float_boolean_or_negative_result_pids_are_rejected(self):
        """A coherent final-record rewrite cannot change the exact typed bound worker PID."""
        f = self.f; f.begin(); final, result = f.complete()
        for pid in (float(f.process['pid']), True, -1):
            changed_result = copy.deepcopy(result); changed_result['execution_identity']['pid'] = pid
            changed_final = copy.deepcopy(final); changed_final['numerical_execution'] = changed_result['execution_identity']
            f.write(f.worker / 'group_backend_capture.json', changed_final)
            with self.subTest(pid=pid), self.assertRaisesRegex(ValueError, 'actual completed worker execution'):
                B.validate_capture(f.worker, f.plan, f.metadata, f.binding, changed_result['loaded_gpu_libraries'], changed_result)

    def test_final_only_pid_type_change_is_not_hidden_by_dictionary_equality(self):
        """The typed result and final proof must agree even when ordinary Python dictionary equality would pass."""
        f = self.f; f.begin(); final, result = f.complete()
        final['numerical_execution']['pid'] = float(f.process['pid'])
        f.write(f.worker / 'group_backend_capture.json', final)
        with self.assertRaisesRegex(ValueError, 'actual completed worker execution'):
            B.validate_capture(f.worker, f.plan, f.metadata, f.binding, result['loaded_gpu_libraries'], result)

    def test_coherent_other_environment_refuses_before_installed_image_reads(self):
        """The exact selected interpreter directory must own the same site-packages root."""
        f = self.f; other = f.root / 'other-environment/bin/python'
        other.parent.mkdir(parents=True); other.write_bytes(b'private synthetic Python; never executed')
        changed = copy.deepcopy(f.expected); changed['selected_python'] = str(other)
        f.plan['python'] = str(other)
        f.write(f.prepared, changed); f.write(f.unit / 'backend_image_expected.json', changed)
        f.plan['inputs']['backend_image_record'] = B.files.reference(f.prepared)
        f.plan['group_backend_image']['expected'] = B.files.reference(f.unit / 'backend_image_expected.json')
        f.save_plan_and_binding()
        with patch.object(B.files, 'describe_image', side_effect=AssertionError('image read happened')) as reader:
            with self.assertRaisesRegex(ValueError, 'selected package environment'):
                B.validate_prepared(f.plan)
            reader.assert_not_called()

    def test_python_and_site_paths_reject_lexical_environment_aliases(self):
        """Neither selected Python nor site-packages may escape or disguise the declared environment."""
        f = self.f
        for field, value in (('selected_python', str(f.root) + '/environment/bin/../bin/python'),
                ('selected_python', 'relative/environment/bin/python'),
                ('site_packages', str(f.site) + '/../site-packages'),
                ('site_packages', str(f.root) + '/another-environment/lib/python3.12/site-packages')):
            changed = copy.deepcopy(f.expected); changed[field] = value
            python = changed['selected_python']
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                B.validate_expected(changed, python)
