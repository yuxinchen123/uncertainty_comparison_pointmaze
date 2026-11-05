"""Check reporting joins with private metadata fixtures and no numerical or job execution."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import controller
import ideas
import test_queue as queue_fixtures


class PostInitialReportingTests(unittest.TestCase):
    """Use synthetic approved metadata to check display coverage without scientific recomputation."""

    def setUp(self):
        """Reuse the existing private queue setup without exposing its class to duplicate discovery."""
        self.fixture = queue_fixtures.QueueProtocolTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.run, self.source = self.fixture.run, self.fixture.commit
        self.configuration = {"learner": {**controller.REFERENCE_ALGORITHM, "n_copies": 1},
                              "environment": {"cache_room_backgrounds": True, "render_final_two_only": True},
                              "execution": {"iterations_per_call": 1, "report_readout": "none"}}
        self.index = {"schema_version": 1, "kind": "post_initial_reporting_evidence",
                      "source_commit": self.source,
                      "configuration_record": self.write(self.run / "configuration.json", self.configuration)}
        self.sequence = 0

    def write(self, path, value):
        """Write an exact private metadata fixture and return its byte receipt."""
        controller.atomic_json(path, value)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def save_index(self):
        """Publish only the optional display index in this temporary campaign."""
        self.write(self.run / "next_actions.json", {"post_initial_reporting_evidence": self.index})

    def evidence(self, requirement, context, copies):
        """Build a labelled synthetic plan/result/review chain using the existing producer shapes."""
        self.sequence += 1
        directory = self.run / "data" / ("metadata_fixture_" + str(self.sequence))
        configuration = copy.deepcopy(self.configuration)
        configuration["learner"]["n_copies"] = copies
        mode = {"common_correctness": "correctness", "common_confirmation": "confirmation",
                "kernel_profiles": None, "packing": "packing-point"}[requirement]
        plan = {"kind": "selected_configuration_profile" if mode is None else "selected_configuration_measurement",
                "mode": mode, "source_commit": self.source, "hardware_class": context,
                "campaign_run_dir": str(self.run), "unit_id": directory.name,
                "configuration": configuration, "copy_seeds": list(range(1, copies + 1)),
                "rounds": 7, "minimum_updates_per_round": 10, "iterations": 10}
        result = {"source_commit": self.source, "hardware_class": context, "copies": copies,
                  "copy_seeds": list(range(1, copies + 1)), "measurement_mode": mode,
                  "fixture_scope": "Private metadata fixture; no actual numerical work or approval."}
        if requirement in {"common_correctness", "common_confirmation"}:
            result.update(candidate=configuration, correctness_attempted=True, correctness_passed=True,
                          implementation_distinct=True)
        else:
            result.update(configuration=configuration, completed=True)
        if requirement == "common_confirmation":
            # The real common-confirmation producer has no required top-level protocol field.
            result.update(measurement_valid=True, decision="accepted", paired_rounds=7,
                          rounds=[{"reference": {"iterations": 10}, "candidate": {"iterations": 10}}
                                  for unused in range(7)])
        if requirement == "kernel_profiles":
            result["profile_updates"] = 1
        result_ref = self.write(directory / "result.json", result)
        review = {"kind": "independent_" + requirement + "_review", "approved": True,
                  "passed": True, "source_commit": self.source, "result": result_ref,
                  "fixture_scope": "Synthetic receipt only; never a real scientific review."}
        if requirement == "packing":
            review["packing"] = {"complete": True, "hardware_class": context, "source_commit": self.source,
                "configuration": configuration, "selected_copies": copies,
                "points": [{"copies": copies, "free_memory_fraction": .20}],
                "confirmation": [{"copies": copies, "free_memory_fraction": .19}],
                "stop_reason": "confirmed_doubling_gain_below_ten_percent"}
        entry = {"plan": self.write(directory / "measurement_plan.json", plan),
                 "result": result_ref, "review": self.write(directory / "review.json", review)}
        self.index.setdefault(requirement, []).append(entry)
        return entry

    def approved_both(self):
        """Provide every requested display pair and both profile/packing contexts."""
        for context in ideas.REQUIRED_HARDWARE:
            for requirement in ("common_correctness", "common_confirmation"):
                for copies in (1, 2):
                    self.evidence(requirement, context, copies)
            for requirement in ("kernel_profiles", "packing"):
                self.evidence(requirement, context, 1)
        self.save_index()

    def complete_initial_registry(self):
        """Record 120 private fixture summaries without creating or launching measured cases."""
        registry = ideas.new_registry(self.source)
        registry["initial_matched_copy_counts"] = [1, 2]
        for idea in registry["initial_ideas"]:
            for context in ideas.REQUIRED_HARDWARE:
                idea["results"][context] = [{"copies": copies, "implementation_distinct": True,
                                             "measurement_valid": True, "correctness_passed": True}
                                            for copies in (1, 2)]
        self.write(self.run / "study_registry.json", registry)
        self.fixture.ready["matched_copy_counts"] = [1, 2]
        self.write(self.fixture.sentinel, self.fixture.ready)
        return registry

    def pending(self):
        """Read the reporting result without invoking queue, measurement or admission helpers."""
        return controller.post_initial_reporting_requirements(self.run, self.source, [1, 2])

    def test_initial_120_without_index_keep_four_real_prerequisites_due(self):
        """An empty queue does not claim that adaptive research is ready."""
        self.complete_initial_registry()
        state = controller.tick(self.run)
        self.assertEqual(state["phase"], "post_initial_validation_pending")
        self.assertEqual(state["queue_counts"], {"pending": 0, "running": 0, "completed": 0, "failed": 0})
        self.assertEqual(len(state["remaining_work"]), 4)
        self.assertIn("common correctness", state["next_action"])

    def test_a4500_only_approvals_leave_h100_due(self):
        """Matching records for one GPU never stand in for the other GPU."""
        for requirement in ("common_correctness", "common_confirmation"):
            for copies in (1, 2):
                self.evidence(requirement, "jaguar03_rtx_a4500", copies)
        for requirement in ("kernel_profiles", "packing"):
            self.evidence(requirement, "jaguar03_rtx_a4500", 1)
        self.save_index()
        rows = self.pending()
        self.assertTrue(all(row["state"] == "awaiting_reviewed_evidence" for row in rows))
        self.assertTrue(all({item["hardware_class"] for item in row["missing"]} == {"h100"} for row in rows))

    def test_matching_reviewed_both_allow_only_the_adaptive_reporting_phase(self):
        """All four display prerequisites can be satisfied without granting scientific authority."""
        self.approved_both()
        self.complete_initial_registry()
        before_registry = controller.read_json(self.run / "study_registry.json")
        state = controller.tick(self.run)
        self.assertEqual(state["phase"], "adaptive_research_pending")
        self.assertEqual(state["remaining_work"], [])
        self.assertEqual(state["queued"], 0)
        self.assertTrue(all(row["scientific_credit"] == 0 and row["admission_authority"] is False
                            for row in state["post_initial_reporting_requirements"]))
        saved = controller.read_json(self.run / "study_registry.json")
        self.assertEqual(saved["adaptive_batches"], before_registry["adaptive_batches"])

    def test_prepared_producer_and_failed_review_labels_cannot_complete_a_requirement(self):
        """A plan and successful producer result are insufficient without actual reviewed bytes."""
        entry = self.evidence("common_correctness", "h100", 1)
        review_path = Path(entry["review"]["path"])
        review = controller.read_json(review_path)
        review["approved"] = False
        entry["review"] = self.write(review_path, review)
        self.save_index()
        row = self.pending()[0]
        self.assertEqual(row["state"], "awaiting_reviewed_evidence")
        self.assertTrue(row["errors"])
        self.assertIn({"hardware_class": "h100", "copies": 1}, row["missing"])

    def test_changed_source_configuration_hardware_copies_and_seeds_remain_due(self):
        """Coherently rehashed metadata substitutions still do not match the declared common work."""
        for field in ("source", "configuration", "hardware", "copies", "seeds"):
            with self.subTest(field=field):
                self.index.pop("common_correctness", None)
                entry = self.evidence("common_correctness", "h100", 1)
                result_path = Path(entry["result"]["path"])
                result = controller.read_json(result_path)
                if field == "source":
                    result["source_commit"] = "b" * 40
                elif field == "configuration":
                    result["candidate"]["learner"]["learning_rate"] = .5
                    plan_path = Path(entry["plan"]["path"])
                    plan = controller.read_json(plan_path)
                    plan["configuration"] = copy.deepcopy(result["candidate"])
                    entry["plan"] = self.write(plan_path, plan)
                elif field == "hardware":
                    result["hardware_class"] = "jaguar03_rtx_a4500"
                elif field == "copies":
                    result["copies"] = True
                else:
                    result["copy_seeds"] = [2]
                entry["result"] = self.write(result_path, result)
                review_path = Path(entry["review"]["path"])
                review = controller.read_json(review_path)
                review["result"] = entry["result"]
                entry["review"] = self.write(review_path, review)
                self.save_index()
                row = self.pending()[0]
                self.assertEqual(row["state"], "awaiting_reviewed_evidence")
                self.assertTrue(row["errors"])

    def test_independent_approval_of_another_result_cannot_fill_coverage(self):
        """Matching approval flags and source do not replace the exact measured-result receipt."""
        entry = self.evidence("common_correctness", "h100", 1)
        path = Path(entry["review"]["path"])
        review = controller.read_json(path)
        review["result"] = {"path": str(self.run / "another_result.json"), "sha256": "0" * 64}
        entry["review"] = self.write(path, review)
        self.save_index()
        self.assertTrue(self.pending()[0]["errors"])

    def test_matching_values_in_another_units_plan_cannot_replace_the_actual_plan(self):
        """The actual plan and result must share their exact campaign data unit."""
        first = self.evidence("common_correctness", "h100", 1)
        second = self.evidence("common_correctness", "h100", 1)
        first["plan"] = second["plan"]
        self.index["common_correctness"] = [first]
        self.save_index()
        self.assertIn("same direct campaign data unit", self.pending()[0]["errors"][0])

    def test_changed_or_missing_receipt_stays_due_with_visible_error(self):
        """Missing or changed exact result bytes remain an explicit reporting problem."""
        entry = self.evidence("common_correctness", "h100", 1)
        Path(entry["result"]["path"]).write_text('{}\n')
        self.save_index()
        row = self.pending()[0]
        self.assertEqual(row["state"], "awaiting_reviewed_evidence")
        self.assertIn("changed its recorded bytes", row["errors"][0])

    def test_exact_metadata_between_sixteen_and_thirty_two_mib_is_supported(self):
        """The accepted real two-copy timing metadata exceeds sixteen MiB without being a binary image."""
        document = {"private_fixture_text": "x" * (17 * 2 ** 20)}
        reference = self.write(self.run / "large_metadata.json", document)
        self.assertEqual(controller.reporting_evidence_document(self.run, reference), document)
        with mock.patch.object(controller, "REPORTING_METADATA_LIMIT_BYTES", 1024):
            with self.assertRaisesRegex(ValueError, "size bound"):
                controller.reporting_evidence_document(self.run, reference)

    def test_binary_reference_is_refused_before_opening_any_file(self):
        """A malformed index cannot make this metadata report read a captured library image."""
        reference = {"path": str(self.run / "captured_image.so"), "sha256": "0" * 64}
        with mock.patch.object(controller.os, "open") as opened:
            with self.assertRaisesRegex(ValueError, "ordinary JSON path"):
                controller.reporting_evidence_document(self.run, reference)
        opened.assert_not_called()

    def test_common_confirmation_requires_the_complete_seven_by_ten_shape(self):
        """A short or incomplete timing fixture cannot satisfy the common confirmation display."""
        entry = self.evidence("common_confirmation", "h100", 1)
        plan_path = Path(entry["plan"]["path"])
        plan = controller.read_json(plan_path)
        plan["iterations"] = 1
        entry["plan"] = self.write(plan_path, plan)
        self.save_index()
        self.assertTrue(self.pending()[1]["errors"])

    def test_packing_point_without_repeat_or_headroom_remains_due(self):
        """Only an approved complete repeated boundary reports packing complete."""
        entry = self.evidence("packing", "h100", 1)
        review_path = Path(entry["review"]["path"])
        review = controller.read_json(review_path)
        review["packing"]["confirmation"][0]["free_memory_fraction"] = .14
        entry["review"] = self.write(review_path, review)
        self.save_index()
        self.assertTrue(self.pending()[3]["errors"])

    def test_active_adaptive_queue_keeps_its_existing_phase(self):
        """Reporting never blocks or changes an already registered adaptive queue."""
        registry, specifications = self.fixture.synthetic_initial_registry_and_additional_specs()
        ideas.register_adaptive_batch(registry, specifications, "Synthetic metadata fixture", self.source)
        self.write(self.run / "study_registry.json", registry)
        state = controller.tick(self.run)
        self.assertEqual(state["phase"], "adaptive_benchmarks")
        self.assertEqual(state["queued"], 10)

    def test_final_phase_and_existing_final_gate_call_are_unchanged(self):
        """The reporting index does not replace or bypass the existing final validation function."""
        self.complete_initial_registry()
        flags = {"fixture_scope": "No scientific work; existing final validator is explicitly mocked."}
        self.write(self.run / "final_verification.json", flags)
        with mock.patch.object(controller, "investigated_hypotheses", return_value=set(range(50))), \
                mock.patch.object(controller, "final_requirements_met", return_value=True) as gate:
            state = controller.tick(self.run)
        self.assertEqual(state["phase"], "ready_for_final")
        self.assertEqual(state["post_initial_reporting_requirements"], [])
        gate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
