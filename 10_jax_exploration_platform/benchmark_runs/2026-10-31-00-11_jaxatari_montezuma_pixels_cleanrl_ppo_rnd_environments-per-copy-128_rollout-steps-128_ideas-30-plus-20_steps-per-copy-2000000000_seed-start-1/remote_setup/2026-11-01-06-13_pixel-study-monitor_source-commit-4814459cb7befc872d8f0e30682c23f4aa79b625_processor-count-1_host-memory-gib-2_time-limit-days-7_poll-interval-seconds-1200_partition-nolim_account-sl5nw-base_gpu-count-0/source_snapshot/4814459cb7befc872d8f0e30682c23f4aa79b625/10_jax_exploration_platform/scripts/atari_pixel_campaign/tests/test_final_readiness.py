"""Check final controller readiness against synthetic raw evidence without numerical imports."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import controller
import ideas
import test_post_comparison_validation as raw_fixtures
import test_queue as queue_fixtures


class RawFinalReadinessTests(unittest.TestCase):
    """Use the real validator on fifty mechanisms with full synthetic raw records."""

    def setUp(self):
        """Reuse complete metadata fixtures and add truthful registry summaries for the controller."""
        self.fixture = raw_fixtures.PostComparisonTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.run, self.commit = self.fixture.directory, self.fixture.commit
        self.registry = json.loads(Path(self.fixture.specification["study_registry"]).read_text())
        entries = list(self.registry["initial_ideas"])
        entries += [row for batch in self.registry["adaptive_batches"] for row in batch["hypotheses"]]
        for entry in entries:
            for rows in entry["results"].values():
                for row in rows:
                    row.update(implementation_distinct=True, measurement_valid=True, correctness_passed=True)
        controller.atomic_json(self.run / "study_registry.json", self.registry)
        self.fixture.specification["study_registry"] = str(self.run / "study_registry.json")
        self.receipt = raw_fixtures.VALIDATION.validate_final_verification(self.fixture.specification)
        self.allocation = mock.patch.object(controller, "final_validation_allocation").start()
        self.addCleanup(mock.patch.stopall)

    def check(self, receipt=None):
        """Exercise the actual final boundary while mocking only compute placement for synthetic data."""
        return controller.final_requirements_met(self.registry, self.receipt if receipt is None else receipt,
                                                 run_dir=self.run, source_commit=self.commit)

    def test_complete_raw_receipt_passes_and_imports_no_numerical_library(self):
        """Read all original cases, confirmations, packing and synchronized scaling before readiness."""
        self.assertTrue(self.check())
        self.allocation.assert_called_once_with(self.run)
        self.assertFalse(any(name in sys.modules for name in ("jax", "numpy", "torch")))

    def test_five_flags_are_rejected_before_any_real_data_calculation(self):
        """A supplied success marker cannot substitute for the validator's complete evidence receipt."""
        flags = {name: True for name in raw_fixtures.VALIDATION.FINAL_FLAGS}
        flags["source_commit"] = self.commit
        with self.assertRaisesRegex(ValueError, "complete post-comparison validation receipt"):
            self.check(flags)
        self.allocation.assert_not_called()

    def test_current_source_registry_and_specification_digest_are_bound(self):
        """A genuine receipt from another source or campaign does not authorize current work."""
        changes = []
        wrong = copy.deepcopy(self.receipt)
        wrong["source_commit"] = "b" * 40
        changes.append((wrong, "currently admitted source"))
        wrong = copy.deepcopy(self.receipt)
        wrong["specification_digest"] = "0" * 64
        changes.append((wrong, "complete post-comparison"))
        wrong = copy.deepcopy(self.receipt)
        wrong["specification"]["study_registry"] = str(self.run / "other_registry.json")
        wrong["specification_digest"] = controller.content_digest(wrong["specification"])
        changes.append((wrong, "this campaign"))
        for receipt, reason in changes:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                self.check(receipt)
        self.allocation.assert_not_called()

    def test_changed_raw_evidence_requires_a_fresh_receipt_even_if_values_still_pass(self):
        """Hash changes in a still-valid packing record invalidate its previous validation receipt."""
        path = Path(self.fixture.specification["contexts"]["h100"]["packing_records"][0])
        contents = json.loads(path.read_text())
        contents["new_observation"] = "synthetic extra metadata"
        path.write_text(json.dumps(contents) + "\n")
        with self.assertRaisesRegex(ValueError, "stale or differs"):
            self.check()

    def test_missing_matched_count_cannot_use_fifty_one_count_summaries(self):
        """Full coverage is rechecked at every count saved for each initial/adaptive cohort."""
        self.registry["initial_ideas"][0]["results"]["h100"].pop()
        controller.atomic_json(self.run / "study_registry.json", self.registry)
        self.assertEqual(len(ideas.investigated_hypotheses(self.registry)), 50)
        with self.assertRaisesRegex(ValueError, "unfinished mechanism/copy measurement"):
            self.check()

    def test_real_raw_failure_and_registry_change_are_rejected(self):
        """Reject actual missing measurements and an in-memory registry differing from disk."""
        current = copy.deepcopy(self.registry)
        current["new_field"] = "unpersisted controller change"
        with self.assertRaisesRegex(ValueError, "registry being considered"):
            controller.final_requirements_met(current, self.receipt, run_dir=self.run, source_commit=self.commit)
        path = Path(self.fixture.specification["contexts"]["jaguar03_rtx_a4500"]["confirmation_record"])
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()


class FinalReadinessStateTests(unittest.TestCase):
    """Keep initial/adaptive scheduling unchanged and expose a clear final failure reason."""

    def setUp(self):
        """Make a temporary completed study; never register ideas in the actual campaign."""
        self.fixture = queue_fixtures.QueueProtocolTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.run, self.commit = self.fixture.run, self.fixture.commit
        registry, proposals = self.fixture.synthetic_initial_registry_and_additional_specs()
        for number, batch in enumerate(registry["adaptive_batches"]):
            batch.update(reference_commit=self.commit, matched_copy_counts=[1])
            for offset, slot in enumerate(batch["hypotheses"]):
                proposal = copy.deepcopy(proposals[offset])
                proposal["name"] += "_batch_" + str(number + 1)
                slot.update(proposal, state="specified", results={context: [{"copies": 1,
                    "implementation_distinct": True, "measurement_valid": True, "correctness_passed": True}]
                    for context in ideas.REQUIRED_HARDWARE})
        controller.atomic_json(self.run / "study_registry.json", registry)
        self.flags = self.run / "final_verification.json"

    def test_missing_forged_and_malformed_receipts_stay_pending_with_reason(self):
        """A missing or invalid file cannot become ready or crash an otherwise complete controller tick."""
        for contents, expected in ((None, "has not been produced"), ("{}\n", "complete post-comparison"),
                                   ("{malformed\n", "JSONDecodeError")):
            if contents is None:
                self.flags.unlink(missing_ok=True)
            else:
                self.flags.write_text(contents)
            state = controller.tick(self.run)
            with self.subTest(contents=contents):
                self.assertEqual(state["phase"], "final_verification_pending")
                self.assertIn(expected, state["final_verification_error"])
                self.assertIn("revalidate final evidence:", state["next_action"])
                self.assertEqual(state["queued"], 0)

    def test_revalidation_success_and_failure_set_exact_readiness(self):
        """Only a successful raw boundary can produce ready_for_final; failures remain visible."""
        controller.atomic_json(self.flags, {"synthetic": "receipt placeholder"})
        with mock.patch.object(controller, "final_requirements_met", return_value=True) as validate:
            state = controller.tick(self.run)
        self.assertEqual(state["phase"], "ready_for_final")
        self.assertIsNone(state["final_verification_error"])
        self.assertEqual(validate.call_args.kwargs, {"run_dir": self.run, "source_commit": self.commit})
        with mock.patch.object(controller, "final_requirements_met", side_effect=ValueError("synthetic stale raw evidence")):
            state = controller.tick(self.run)
        self.assertEqual(state["phase"], "final_verification_pending")
        self.assertEqual(state["final_verification_error"], "ValueError: synthetic stale raw evidence")

    def test_pending_adaptive_work_prevents_raw_validation(self):
        """Do not recalculate final statistics while any matching case is still waiting or running."""
        registry = controller.read_json(self.run / "study_registry.json")
        name = registry["adaptive_batches"][-1]["hypotheses"][-1]["name"]
        case = controller.immutable_case(ideas.registry_idea(name, registry), "h100", 1,
                                         controller.ready_specification(self.run))
        controller.enqueue_case(self.run, case)
        with mock.patch.object(controller, "final_requirements_met") as validate:
            state = controller.tick(self.run)
        validate.assert_not_called()
        self.assertEqual(state["phase"], "adaptive_benchmarks")
        self.assertIsNone(state["final_verification_error"])


class FinalValidationAllocationTests(unittest.TestCase):
    """Require an actual owned compute cgroup for real-data statistics."""

    def test_owned_cgroup_is_required_and_login_hosts_reject(self):
        """An inherited SLURM_JOB_ID without the matching physical allocation cannot pass."""
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            ledger = run / "slurm/submitted_jobids_uva.txt"
            ledger.parent.mkdir()
            ledger.write_text("333\n")
            original = Path.read_text
            def read(path, *args, **kwargs):
                """Supply an explicit synthetic owned cgroup while preserving the real ledger read."""
                if path == Path("/proc/self/cgroup"):
                    return "0::/system.slice/slurmstepd.scope/job_333/step_batch/task_0\n"
                return original(path, *args, **kwargs)
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "333"}), mock.patch.object(Path, "read_text", read):
                with mock.patch.object(controller.socket, "gethostname", return_value="heartpiece"):
                    controller.final_validation_allocation(run)
                with mock.patch.object(controller.socket, "gethostname", return_value="portal05"):
                    with self.assertRaisesRegex(ValueError, "owned Slurm compute"):
                        controller.final_validation_allocation(run)
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "999"}):
                with self.assertRaisesRegex(ValueError, "owned Slurm compute"):
                    controller.final_validation_allocation(run)

    def test_foreign_cluster_job_id_cannot_prove_uva_ownership(self):
        """A real UVA cgroup needs a UVA ledger entry even if its number occurs at ACES."""
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            ledger = run / "slurm/submitted_jobids_aces.txt"
            ledger.parent.mkdir()
            ledger.write_text("333\n")
            original = Path.read_text

            def read(path, *args, **kwargs):
                """Supply a matching synthetic cgroup while retaining cluster-specific ledger reads."""
                if path == Path("/proc/self/cgroup"):
                    return "0::/system.slice/slurmstepd.scope/job_333/step_batch/task_0\n"
                return original(path, *args, **kwargs)

            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "333"}), \
                    mock.patch.object(controller.socket, "gethostname", return_value="heartpiece"), \
                    mock.patch.object(Path, "read_text", read):
                with self.assertRaisesRegex(ValueError, "owned Slurm compute"):
                    controller.final_validation_allocation(run)


if __name__ == "__main__":
    unittest.main()
