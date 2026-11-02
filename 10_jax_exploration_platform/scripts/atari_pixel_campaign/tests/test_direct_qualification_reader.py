"""Isolate real direct reader joins with explicit unrelated proof stubs; grant no qualification."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "tests")]
import direct_measurement as direct
import test_configured_measurement as fixtures


class DirectQualificationReaderTests(unittest.TestCase):
    """Keep source/native/lock/window/terminal stubs explicit while rebuilding saved semantic gates."""

    def setUp(self):
        self.f = fixtures.QualificationReaderTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        f = self.f; self.core = direct.core()
        f.plan.pop("configured_isolation"); f.plan.pop("target_qualification")
        f.plan.update(direct_binding={"profile": copy.deepcopy(self.core.PROFILE), "consumer": "selected_measurement",
            "current_launch_inputs_path": "explicit synthetic saved launch"}, direct_qualification=direct.QUALIFICATION,
            hardware_class="h100", copy_seeds=[1])
        self.process = {"hostname": "serval02", "pid": fixtures.PID, "process_start_time": "100"}
        self.package = {"selected_python": f.plan["python"], "site_packages": "/synthetic/site-packages",
            "image": {"explicit_unrelated_backend_stub": True}, "distribution": {"explicit_unrelated_owner_stub": True}}
        f.result.pop("configured_isolation")
        f.result.update(kind="direct_serval02_card1_full_original_qualification", qualification_protocol=direct.QUALIFICATION,
            hardware_class="h100", direct_binding=copy.deepcopy(f.plan["direct_binding"]), copy_seeds=[1],
            execution_identity={**self.process, "started_at_seconds": 1., "finished_at_seconds": 2.}, **direct.BOUNDARY)

    def validate(self, mutation=None, outer_mutation=None, review_mutation=None):
        f = self.f; f.publish()
        for name, value in (("direct_process_metadata.json", {"after": {"identity": self.process}}),
                ("direct_device_binding.json", f.binding), ("direct_terminal_process.json", {"explicit_unrelated_terminal_stub": True})):
            (f.unit / name).write_text(json.dumps(value))
        result = copy.deepcopy(f.result)
        if mutation: mutation(result)
        path = f.unit / "direct_qualification_result.json"; path.write_text(json.dumps(result))
        review = {"schema_version": 1, "kind": "independent_full_original_direct_qualification_review",
            "result": self.core.receipt(path), "approved": True, "admission_granted": False, "scientific_credit": 0,
            "independent_of_producer": True, "terminal_process": self.core.receipt(f.unit / "direct_terminal_process.json"),
            "actual_returncode": 0, **self.process}
        if review_mutation: review_mutation(review)
        ref = f.unit / "direct_review.json"; ref.write_text(json.dumps(review))
        outer = copy.deepcopy(f.plan)
        outer["direct_qualification_inputs"] = {"result": self.core.receipt(path), "review": self.core.receipt(ref)}
        if outer_mutation: outer_mutation(outer)
        # These unrelated physical/source/backend/terminal stubs never establish real qualification.
        with patch.object(self.core, "validate_plan"), patch.object(self.core, "validate_binding"),\
                patch.object(direct, "validate_original_qualification"), patch.object(direct, "verify_saved_sources"),\
                patch.object(direct, "verify_qualification_source_receipts"),\
                patch.object(direct, "validate_window", return_value=f.result["qualification_memory"]),\
                patch.object(direct, "validate_startup_checks"), patch.object(direct, "validate_backend_capture"),\
                patch.object(direct, "backend_support", return_value=SimpleNamespace(expected_for_plan=lambda plan: self.package)),\
                patch.object(direct, "terminal_process"):
            return direct.verify_qualification_inputs(outer)

    def test_complete_saved_direct_semantic_reader_passes_without_numerical_work(self):
        value = self.validate()
        self.assertTrue(value["completed"])
        self.assertEqual(value["renderer"]["rooms"], 33)
        self.assertEqual(value["renderer"]["cases"], 132)
        self.assertEqual(value["executed_work"]["full_reference_training_transitions"], 16384)
        self.assertEqual(value["scientific_credit"], 0)

    def test_outer_optimized_configuration_seeds_consumer_and_history_may_differ(self):
        def change(plan):
            plan.update(configuration={"explicit_synthetic_optimized_configuration": True}, copy_seeds=[8, 9])
            plan["direct_binding"].update(consumer="component_profile", current_launch_inputs_path="different actual profile launch")
        self.assertTrue(self.validate(outer_mutation=change)["completed"])

    def test_every_saved_result_identity_field_is_joined_to_qualified_plan(self):
        for key, changed in (("source_commit", "a" * 40), ("hardware_class", "different GPU class"),
                ("runtime_environment", {"changed": "yes"}), ("asset_record", {"changed": True}),
                ("direct_binding", {"profile": self.core.PROFILE}), ("copy_seeds", [999]), ("copy_seeds", [True])):
            with self.subTest(key=key, changed=changed), self.assertRaisesRegex(ValueError, "changed its saved"):
                self.validate(lambda value: value.update({key: changed}))

    def test_coherent_changed_result_and_outer_runtime_cannot_hide_saved_plan(self):
        changed = {"explicit_synthetic_changed_runtime": "yes"}
        with self.assertRaisesRegex(ValueError, "changed its saved runtime_environment"):
            self.validate(lambda value: value.update(runtime_environment=changed), lambda plan: plan.update(runtime_environment=changed))

    def test_outer_source_root_python_runtime_assets_or_profile_must_match_qualification(self):
        for key, changed in (("source_commit", "a" * 40), ("source_root", "/different/source"), ("python", "/different/python"),
                ("hardware_class", "other GPU"), ("runtime_environment", {"changed": "yes"}), ("asset_record", {"changed": True})):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "outer direct consumer"):
                self.validate(outer_mutation=lambda plan: plan.update({key: changed}))
        with self.assertRaisesRegex(ValueError, "qualified profile"):
            self.validate(outer_mutation=lambda plan: plan["direct_binding"]["profile"].update(driver="different driver"))

    def test_execution_identity_matches_saved_process_and_has_real_ordered_time(self):
        for key, changed in (("pid", fixtures.PID + 1), ("hostname", "other-host"), ("process_start_time", "101"),
                ("started_at_seconds", 0), ("finished_at_seconds", 1.), ("finished_at_seconds", float("inf")),
                ("started_at_seconds", float("nan")), ("pid", True)):
            with self.subTest(key=key, changed=changed), self.assertRaises(ValueError):
                self.validate(lambda value: value["execution_identity"].update({key: changed}))

    def test_raw_full_renderer_dtype_and_original_test_gates_remain_required(self):
        self.f.renderer.update(rooms=2, cases=8)
        with self.assertRaises(ValueError): self.validate()
        self.f.renderer.update(rooms=33, cases=132)
        with self.assertRaises(ValueError): self.validate(lambda value: value["dtypes"].update(parameters=["float64"]))
        self.f.tests["reports"].pop()
        with self.assertRaises(ValueError): self.validate()

    def test_full_helpers_resolved_binding_and_independent_terminal_review_stay_required(self):
        for mutation in (lambda value: value.update(helpers={"changed": True}),
                lambda value: value["resolved_device_binding"].update(gpu_uuid="GPU-other")):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.validate(mutation)
        for mutation in (lambda value: value.update(independent_of_producer=False),
                lambda value: value.update(actual_returncode=1), lambda value: value.update(actual_returncode=True),
                lambda value: value.update(pid=fixtures.PID + 1)):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.validate(review_mutation=mutation)


if __name__ == "__main__":
    unittest.main()
