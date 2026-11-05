"""Check thin fact production with explicit synthetic accepted-context premises."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location("common_profile_fact_producer_test", HERE / "produce_common_profile_prerequisite_fact.py")
PRODUCER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRODUCER)
HELPER_SPEC = importlib.util.spec_from_file_location("common_profile_fact_exact_reader_test",
    ROOT / (PRODUCER.PREFIX + PRODUCER.NAME))
HELPER = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(HELPER)


class FactProducerTests(unittest.TestCase):
    """Test request routing and refusal without inventing an actual H100 approval."""

    def setUp(self):
        """Create bounded real metadata files for explicitly synthetic context premises."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.proposed = {"source_commit": HELPER.SOURCE, "hardware_class": "jaguar03_rtx_a4500",
            "campaign_run_dir": str(self.root), "configuration": {"learner": {"n_copies": 1}}}
        self.proposed_ref = self.write("accepted_A4500_plan.json", self.proposed)
        result = self.write("synthetic_qualification_result.json", {"synthetic_premise": True})
        review = self.write("synthetic_qualification_review.json", {
            "kind": "independent_full_original_target_qualification_review", "approved": True,
            "result": result, "actual_terminal_state": "COMPLETED", "actual_exit_code": "0:0", "admission_granted": False})
        self.target = {"review": review, "result": result}
        native_plan = self.write("synthetic_native_plan.json", {"synthetic_premise": True})
        native_result = self.write("synthetic_native_result.json", {"synthetic_premise": True})
        native_contract = self.write("synthetic_native_contract.json", {"synthetic_premise": True})
        native_review = self.write("synthetic_native_review.json", {
            "kind": "independent_configured_native_process_review", "approved": True, "passed": True,
            "independent_of_producer": True, "plan": native_plan, "result": native_result,
            "target_contract": native_contract, "profile_admission_granted": False})
        self.native = {"diagnostic_plan": native_plan, "diagnostic_result": native_result,
            "target_contract": native_contract, "independent_review": native_review}
        self.entries = [{"hardware_class": hardware, "copies": copies,
            "review": self.write(hardware + "_" + str(copies) + ".json", {"synthetic_premise": True})}
            for hardware, copies in sorted(HELPER.CONTEXTS)]
        self.request = {"proposed_plan": self.proposed_ref, "contexts": self.entries,
            "initial_snapshot_inputs": {"synthetic_accepted_snapshot_premise": True}, "native_process_qualification": self.native}

    def write(self, name, value):
        """Write one real temporary JSON file and return its actual exact reference."""
        raw = HELPER.encoded(value)
        path = self.root / name
        path.write_bytes(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    def accepted_context_premise(self, files, entry, proposed):
        """Supply an explicitly synthetic accepted gate result, not a genuine scientific review."""
        files.document(entry["review"])
        return {**entry, "plan": self.proposed_ref, "target_qualification_inputs": self.target,
            "historical_plan": {"synthetic_premise": True}}

    def production(self, request):
        """Patch only accepted premises and constant receipt hashes for this synthetic test."""
        with patch.object(HELPER, "reviewed_context", side_effect=self.accepted_context_premise) as contexts, \
                patch.object(HELPER, "initial_snapshot", return_value={"synthetic_premise": True}) as snapshot, \
                patch.object(HELPER, "NATIVE_REVIEW", self.native["independent_review"]["sha256"]), \
                patch.object(HELPER, "QUALIFICATION_REVIEW", self.target["review"]["sha256"]), \
                patch.object(HELPER, "QUALIFICATION_RESULT", self.target["result"]["sha256"]):
            fact, summary = PRODUCER.produce_fact(HELPER, request, HELPER.MetadataFiles())
        self.assertEqual(contexts.call_count, 4)
        snapshot.assert_called_once()
        return fact, summary

    def test_complete_synthetic_contexts_produce_existing_fact_without_review(self):
        """The complete synthetic premise routes all four gates and grants no admission."""
        fact, _ = self.production(self.request)
        self.assertEqual(fact["kind"], HELPER.PROTOCOL["kind"])
        self.assertEqual(fact["contexts"], self.entries)
        self.assertEqual(fact["native_process_qualification"], self.native)
        self.assertEqual(fact["target_qualification_inputs"], self.target)
        self.assertIs(fact["operational_admission"], False)
        self.assertIs(fact["admission_granted"], False)
        self.assertNotIn("approved", fact)
        self.assertNotIn("passed", fact)

    def test_missing_duplicate_boolean_and_unaccepted_contexts_refuse(self):
        """Missing H100 two-copy, duplicate, Boolean and rejected reviews never produce a fact."""
        missing = [row for row in self.entries if (row["hardware_class"], row["copies"]) != ("h100", 2)]
        for entries in (missing, self.entries[:-1] + [self.entries[0]],
                [{**row, "copies": True} if index == 0 else row for index, row in enumerate(self.entries)]):
            request = {**self.request, "contexts": entries}
            with patch.object(HELPER, "reviewed_context", side_effect=AssertionError("must refuse first")), \
                    self.assertRaisesRegex(ValueError, "four distinct"):
                PRODUCER.produce_fact(HELPER, request, HELPER.MetadataFiles())
        # The real reviewed-context implementation refuses an actual unaccepted file.
        entry = {"hardware_class": "h100", "copies": 2,
            "review": self.write("unaccepted_H100_two_copy_review.json", {"approved": False})}
        with self.assertRaisesRegex(ValueError, "complete independently accepted"):
            HELPER.reviewed_context(HELPER.MetadataFiles(), entry, self.proposed)

    def test_proposed_plan_and_both_target_qualification_joins_are_exact(self):
        """A changed accepted-plan reference or divergent second A4500 qualification refuses."""
        with patch.object(HELPER, "reviewed_context", side_effect=lambda files, entry, proposed:
                {**self.accepted_context_premise(files, entry, proposed), "plan": {**self.proposed_ref, "sha256": "0" * 64}}), \
                self.assertRaisesRegex(ValueError, "exact accepted A4500"):
            PRODUCER.produce_fact(HELPER, self.request, HELPER.MetadataFiles())
        def divergent(files, entry, proposed):
            """Make only the second synthetic A4500 qualification disagree."""
            row = self.accepted_context_premise(files, entry, proposed)
            if entry["hardware_class"] == "jaguar03_rtx_a4500" and entry["copies"] == 2:
                row["target_qualification_inputs"] = {"different_synthetic_premise": True}
            return row
        with patch.object(HELPER, "reviewed_context", side_effect=divergent), \
                patch.object(HELPER, "NATIVE_REVIEW", self.native["independent_review"]["sha256"]), \
                self.assertRaisesRegex(ValueError, "both accepted A4500"):
            PRODUCER.produce_fact(HELPER, self.request, HELPER.MetadataFiles())

    def test_actual_three_accepted_metadata_refs_do_not_supply_fourth_context(self):
        """Actual three accepted review references still refuse before any missing fourth output."""
        run = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/"
            "2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
        records = json.loads((run / "next_actions.json").read_text())
        entries = [{"hardware_class": "jaguar03_rtx_a4500", "copies": copies,
            "review": records["common_a4500_confirmation_preparation"]["independent_reviews"][str(copies)]} for copies in (1, 2)]
        entries.append({"hardware_class": "h100", "copies": 1,
            "review": records["h100_common_timing_controller_preparation"]["copy_one_independent_timing_review"]})
        files = HELPER.MetadataFiles()
        for row in entries:
            self.assertIs(files.document(row["review"])["approved"], True)
        with patch.object(HELPER, "reviewed_context", side_effect=AssertionError("no fourth accepted review")), \
                self.assertRaisesRegex(ValueError, "four distinct"):
            PRODUCER.produce_fact(HELPER, {**self.request, "contexts": entries}, files)

    def test_default_is_no_action_and_output_is_exclusive(self):
        """No-action touches no supplied path, and a saved output cannot be replaced."""
        with patch.object(PRODUCER, "load_helper", side_effect=AssertionError("no action")):
            self.assertEqual(PRODUCER.main(["--helper-root", "/absent", "--request", "/absent",
                "--proof", "/absent/proof", "--audit", "/absent/audit", "--owned-job-id", "6825983"]), 0)
        path = self.root / "once.json"
        PRODUCER.write_once(path, b"{}\n")
        with self.assertRaises(FileExistsError):
            PRODUCER.write_once(path, b"changed")
        self.assertEqual(path.read_bytes(), b"{}\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
