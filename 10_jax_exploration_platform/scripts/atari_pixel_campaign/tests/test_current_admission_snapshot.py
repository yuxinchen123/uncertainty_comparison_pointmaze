"""Exercise only the new small-metadata claim branch; no target operation occurs."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "serval02_card1_binding.py"
SPEC = importlib.util.spec_from_file_location("claim_branch_fixture", PATH)
core = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(core)


class CurrentAdmissionSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"accepted_evidence_snapshot_protocol": {}, "campaign_run_dir": "/campaign",
            "unit_id": "new-timing", "source_root": "/original/source",
            "direct_binding": {"frozen_inputs": {"path": "/pointer", "sha256": "0" * 64}}}
        self.frozen = {"path": "/campaign/data/accepted-qualification/direct_history/admission.json",
            "sha256": core.ORIGINAL_ADMISSION_SHA256}
        self.admission = {"authorized": True, "source_root": "/original/source", "source_commit": core.SOURCE_COMMIT}
        self.historical = {"admission": self.frozen}
        self.selected = SimpleNamespace(qual={"historical_inputs": "history",
            "plan": {"path": "/campaign/data/accepted-qualification/measurement_plan.json"}},
            proof={"declared_files": [copy.deepcopy(self.frozen)]})
        self.selected.files = SimpleNamespace(json=lambda ref: self.historical if ref == "history" else self.admission)
        self.consumer = SimpleNamespace(consume=lambda plan: self.selected)
        self.metadata = SimpleNamespace(adjacent=lambda name: self.consumer)

    def claim(self):
        with patch.object(core, "read_receipt", return_value={"no_old_admission_key": True}), \
                patch.object(core, "adjacent", return_value=self.metadata):
            return core.current_admission_claim(self.plan)

    def test_exact_original_claim_schema(self):
        self.assertEqual(self.claim(), {"schema_version": 1,
            "kind": "current_direct_admission_matches_frozen_original",
            "canonical_path": "/campaign/SOURCE_AND_GATES_READY.json", "frozen_admission": self.frozen,
            "source_root": "/original/source", "source_commit": core.SOURCE_COMMIT})

    def test_missing_proof_coverage_refused(self):
        self.selected.proof["declared_files"] = []
        with self.assertRaises(ValueError): self.claim()

    def test_substituted_original_path_refused(self):
        self.frozen["path"] = "/campaign/data/elsewhere/direct_history/admission.json"
        self.selected.proof["declared_files"] = [copy.deepcopy(self.frozen)]
        with self.assertRaises(ValueError): self.claim()

    def test_changed_hash_refused(self):
        self.frozen["sha256"] = "1" * 64
        with self.assertRaises(ValueError): self.claim()

    def test_changed_authority_and_source_refused(self):
        for key, value in (("authorized", False), ("authorized", 1), ("source_root", "/other"),
                ("source_commit", "0" * 40)):
            with self.subTest(key=key, value=value):
                old = self.admission[key]; self.admission[key] = value
                with self.assertRaises(ValueError): self.claim()
                self.admission[key] = old

    def test_legacy_branch_keeps_own_unit_reference(self):
        del self.plan["accepted_evidence_snapshot_protocol"]
        frozen = {"path": "/campaign/data/new-timing/direct_history/admission.json",
            "sha256": core.ORIGINAL_ADMISSION_SHA256}
        with patch.object(core, "read_receipt", return_value={"admission": frozen}), \
                patch.object(core, "adjacent", side_effect=AssertionError("legacy must not use new snapshot")):
            self.assertEqual(core.current_admission_claim(self.plan)["frozen_admission"], frozen)


if __name__ == "__main__":
    unittest.main()
