"""Check the new metadata-only four-context route using explicit synthetic approvals."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("configured_profile_prerequisite_tests", SCRIPTS / "configured_profile_prerequisites.py")
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)
PROFILE_SPEC = importlib.util.spec_from_file_location("configured_profile_route_tests", SCRIPTS / "profile_selected.py")
PROFILE = importlib.util.module_from_spec(PROFILE_SPEC)
PROFILE_SPEC.loader.exec_module(PROFILE)


class MetadataPrerequisiteTests(unittest.TestCase):
    """Exercise strict joins without inventing real target or raw-review success."""

    def setUp(self):
        """Create four explicitly synthetic independently reviewed metadata contexts."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.configuration = {"learner": {"n_copies": 1, "num_envs": 128},
            "environment": {"cache_room_backgrounds": True, "render_final_two_only": True},
            "execution": {"iterations_per_call": 1}}
        self.runtime = {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest"}
        self.assets = {"root": "/synthetic/assets", "manifest": "/synthetic/assets.json", "sha256": "a" * 64}
        self.numerical = {"source_root": "/synthetic/numerical", "core": {"synthetic_premise": True},
            "paired_source": {"synthetic_premise": True}}
        self.proposed = {"kind": "selected_configuration_profile", "hardware_class": "jaguar03_rtx_a4500",
            "source_commit": HELPER.SOURCE, "campaign_run_dir": str(self.run), "gpu_uuid": None,
            "configured_isolation": {"synthetic_premise": True}, "configuration": self.configuration,
            "configured_native_process_control": {"synthetic_premise": True},
            "native_process_control": {"expected_target_contract": {"synthetic_premise": True}},
            "runtime_environment": self.runtime, "asset_record": self.assets, "copy_seeds": [1],
            "configured_profile_prerequisite_protocol": HELPER.PROTOCOL,
            "inputs": {"study_registry": {"path": str(self.run / "registry.json"), "sha256": "b" * 64}}}
        self.proposed.update(self.numerical)

    def write(self, name, value):
        """Write an actual temporary metadata reference for one synthetic input."""
        path = self.run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = HELPER.encoded(value);path.write_bytes(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    def context(self, hardware, copies):
        """Build the exact public review schemas while marking the source premise synthetic."""
        name = hardware + "_copies_" + str(copies)
        configuration = copy.deepcopy(self.configuration);configuration["learner"]["n_copies"] = copies
        original = copy.deepcopy(configuration)
        original["environment"] = {"cache_room_backgrounds": False, "render_final_two_only": False}
        result = {"path": str(self.run / "data" / name / "result.json"), "sha256": "c" * 64}
        plan = {"source_commit": HELPER.SOURCE, "campaign_run_dir": str(self.run), "hardware_class": hardware,
            "unit_id": name, "mode": "confirmation", "rounds": 7, "iterations": 10, "minimum_updates_per_round": 10,
            "configuration": configuration, "reference_configuration": original, "copy_seeds": list(range(1, copies + 1)),
            "runtime_environment": self.runtime, "asset_record": self.assets, "helpers": {"source_commit": "d" * 40},
            "common_correctness_inputs": {"synthetic_premise": True}, "target_qualification_inputs": {"synthetic_premise": True},
            "configured_isolation": {"synthetic_premise": True}}
        plan.update(copy.deepcopy(self.numerical))
        plan_reference = self.write("data/" + name + "/measurement_plan.json", plan)
        review = {"schema_version": 1, "source_commit": HELPER.SOURCE, "helper_commit": "d" * 40,
            "copies": copies, "copy_seeds": list(range(1, copies + 1)), "approved": True, "passed": True,
            "admission_granted": False, "worker_admitted": False, "plan": plan_reference, "result": result}
        if hardware == "jaguar03_rtx_a4500":
            review.update(kind="independent_actual_common_configuration_confirmation_review", actual_terminal_state="COMPLETED",
                actual_exit_code="0:0", exact_staged_saved_consumer_passed=True,
                current_intermediate_combination_reader_passed=True, seven_alternating_pairs_ten_full_updates_per_arm=True,
                actual_assets_verified={"file_count": 4531, "root": self.assets["root"],
                    "manifest_sha256": self.assets["sha256"], "tree_digest": "a" * 64},
                loaded_CUDA_cuDNN_image_hashes_verified={"/synthetic/libcudart.so.12": "a" * 64,
                    "/synthetic/libcudnn.so.9": "b" * 64},
                same_supplied_reference_state_origins=list(range(3, 64, 10)), packing_or_headroom_credit_granted=False,
                accepted_combination={"configuration": configuration, "copies": copies, "hardware_class": hardware,
                    "source_commit": HELPER.SOURCE, "runtime_environment": self.runtime, "paired_rounds": 7,
                    "median_speedup": 1.5, "one_sided_sign_probability": .0078125})
            inventory = {value["path"]: {"sha256": value["sha256"]} for value in (plan_reference, result)}
            review.update(all_files_before=inventory, all_files_after=copy.deepcopy(inventory))
        else:
            first = self.write(name + "_first.json", {"synthetic_premise": True})
            review.update(kind="independent_actual_direct_common_configuration_confirmation_review", independent_of_producer=True,
                decision="accepted", paired_rounds=7, updates_per_arm_per_round=10, raw_window_recomputation_passed=True,
                normal_actual_child_returncode=0, actual_controller_and_numerical_child_exits_verified=True,
                unresolved_anomalies=[], configuration=configuration, reference_configuration=original,
                raw_audit=self.write(name + "_raw.json", {"synthetic_premise": True}), raw_first_approval=first,
                exit_observation=self.write(name + "_exit.json", {"synthetic_premise": True}),
                unchanged_staged_consumer=self.write(name + "_consumer.json", {
                    "kind": "exact_staged_direct_common_timing_saved_consumer_result", "passed": True,
                    "plan": plan_reference, "result": result, "source_helper_commit": "d" * 40,
                    "source_helper_count": 29, "raw_first_approval": first}))
        return {"hardware_class": hardware, "copies": copies, "review": self.write(name + "_review.json", review)}, plan, review

    def test_actual_owned_metadata_duplicate_changed_and_symlink_refusals(self):
        """An ordinary matching file passes; ambiguous, changed and indirect bytes fail."""
        files = HELPER.MetadataFiles()
        value = self.write("owned.json", {"one": 1})
        self.assertEqual(files.document(value), {"one": 1})
        Path(value["path"]).write_text('{"one":1,"one":2}')
        duplicate = {"path": value["path"], "sha256": hashlib.sha256(Path(value["path"]).read_bytes()).hexdigest()}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            files.document(duplicate)
        with self.assertRaisesRegex(ValueError, "changed"):
            files.document(value)
        link = self.run / "link.json";link.symlink_to(value["path"])
        with self.assertRaises(ValueError):
            files.document({"path": str(link), "sha256": duplicate["sha256"]})

    def test_each_context_keeps_its_original_review_schema_without_result_reads(self):
        """All four synthetic reviewed schemas join without opening any result payload."""
        for hardware, copies in sorted(HELPER.CONTEXTS):
            entry, plan, unused = self.context(hardware, copies)
            with patch.object(HELPER, "source_closure", return_value=plan["helpers"]):
                result = HELPER.reviewed_context(HELPER.MetadataFiles(), entry, self.proposed)
            self.assertEqual(result["copies"], copies)
            self.assertFalse(Path(result["result"]["path"]).exists())

    def test_wrong_copy_type_source_counts_candidate_and_consumer_are_refused(self):
        """Distinct context, exact complete updates, candidate and original consumer remain mandatory."""
        for mutation in ("Boolean count", "source", "updates", "candidate", "consumer"):
            entry, plan, review = self.context("h100", 1)
            if mutation == "Boolean count":entry["copies"] = True
            elif mutation == "source":review["source_commit"] = "f" * 40
            elif mutation == "updates":plan["iterations"] = 9
            elif mutation == "candidate":plan["configuration"]["execution"]["iterations_per_call"] = 2
            else:
                consumer = json.loads(Path(review["unchanged_staged_consumer"]["path"]).read_text())
                consumer["passed"] = False
                review["unchanged_staged_consumer"] = self.write("bad_consumer.json", consumer)
            review["plan"] = self.write("data/" + plan["unit_id"] + "/measurement_plan.json", plan)
            entry["review"] = self.write("bad_review.json", review)
            with patch.object(HELPER, "source_closure", return_value=plan["helpers"]), self.assertRaises(ValueError, msg=mutation):
                HELPER.reviewed_context(HELPER.MetadataFiles(), entry, self.proposed)

    def test_only_reviewed_metadata_can_supply_confirmation_summary(self):
        """Preparation reads an approved plan, not a large raw result or an unapproved declaration."""
        entry, plan, review = self.context("jaguar03_rtx_a4500", 1)
        proof = self.write("proof.json", {"contexts": [entry]})
        approval = self.write("approval.json", {"proof": proof, "approved": True, "passed": True})
        summary, reference = HELPER.confirmation_metadata(proof["path"], approval["path"], "jaguar03_rtx_a4500", 1)
        self.assertEqual(summary["candidate"], plan["configuration"])
        self.assertEqual(reference, review["result"])
        self.assertFalse(Path(reference["path"]).exists())
        bad = self.write("bad_approval.json", {"proof": proof, "approved": False, "passed": True})
        with self.assertRaises(ValueError):HELPER.confirmation_metadata(proof["path"], bad["path"], "jaguar03_rtx_a4500", 1)

    def test_a4500_typed_asset_library_and_origin_records_cannot_be_replaced_by_booleans(self):
        """The genuine approval records are structured facts, not invented success flags."""
        for field in ("actual_assets_verified", "loaded_CUDA_cuDNN_image_hashes_verified", "same_supplied_reference_state_origins"):
            entry, plan, review = self.context("jaguar03_rtx_a4500", 1)
            review[field] = True
            entry["review"] = self.write("bad_typed_review.json", review)
            with patch.object(HELPER, "source_closure", return_value=plan["helpers"]), self.assertRaisesRegex(ValueError, "asset, loaded-image"):
                HELPER.reviewed_context(HELPER.MetadataFiles(), entry, self.proposed)

    def test_numerical_receipts_and_seed_integer_types_remain_exact(self):
        """A commit label cannot substitute changed numerical receipts or Boolean random seeds."""
        for field in ("source_root", "core", "paired_source", "copy_seeds"):
            entry, plan, review = self.context("h100", 1)
            if field == "copy_seeds":plan[field] = [True]
            elif field == "source_root":plan[field] = "/synthetic/other_numerical"
            else:plan[field] = {"changed_synthetic_premise": True}
            review["plan"] = self.write("data/" + plan["unit_id"] + "/measurement_plan.json", plan)
            entry["review"] = self.write("changed_receipt_review.json", review)
            with patch.object(HELPER, "source_closure", return_value=plan["helpers"]), self.assertRaises(ValueError):
                HELPER.reviewed_context(HELPER.MetadataFiles(), entry, self.proposed)
        entry, plan, review = self.context("h100", 1)
        review["copy_seeds"] = [True]
        entry["review"] = self.write("Boolean_seed_review.json", review)
        with patch.object(HELPER, "source_closure", return_value=plan["helpers"]), self.assertRaises(ValueError):
            HELPER.reviewed_context(HELPER.MetadataFiles(), entry, self.proposed)

    def test_incomplete_or_repeated_four_context_proof_is_rejected_before_snapshot(self):
        """Three contexts and a duplicate fourth cannot supply the missing accepted hardware/count."""
        for duplicate in (False, True):
            entries = [self.context(hardware, copies)[0] for hardware, copies in sorted(HELPER.CONTEXTS)]
            entries = entries[:3] + ([entries[0]] if duplicate else [])
            proof = self.write("full_proof.json", {"schema_version": 1, "kind": HELPER.PROTOCOL["kind"],
                "source_commit": HELPER.SOURCE, "campaign_run_dir": str(self.run), "operational_admission": False,
                "contexts": entries})
            approval = self.write("full_approval.json", {"kind": "independent_reviewed_four_context_configured_profile_prerequisites",
                "proof": proof, "approved": True, "passed": True, "independent_of_producer": True,
                "all_four_raw_reviews_verified": True, "exact_historical_consumer_namespaces_verified": True,
                "admission_granted": False, "worker_admitted": False, "unresolved_anomalies": []})
            self.proposed["configured_profile_prerequisite_inputs"] = {"proof": proof, "review": approval}
            with patch.object(HELPER, "initial_snapshot") as snapshot, self.assertRaisesRegex(ValueError, "all four"):
                HELPER.validate(self.proposed)
            snapshot.assert_not_called()

    def test_snapshot_reuse_refuses_replaced_existing_proof_before_any_import(self):
        """Only the already genuine complete snapshot/review can reach its isolated old consumer."""
        inputs = {"proof": {"path": "/missing/proof.json", "sha256": "0" * 64},
            "review": {"path": "/missing/review.json", "sha256": HELPER.SNAPSHOT_REVIEW}}
        with self.assertRaisesRegex(ValueError, "existing complete"):
            HELPER.initial_snapshot(HELPER.MetadataFiles(), inputs, {"accepted_evidence_snapshot_inputs": inputs})

    def test_full_four_context_metadata_fact_still_grants_no_operation_or_numerical_admission(self):
        """Complete synthetic reviews pass only with explicit unchanged snapshot/source premises."""
        entries = [self.context(hardware, copies)[0] for hardware, copies in sorted(HELPER.CONTEXTS)]
        qualification_result = {"path": str(self.run / "original_qualification_result.json"), "sha256": "1" * 64}
        qualification_review = self.write("qualification_review.json", {
            "kind": "independent_full_original_target_qualification_review", "approved": True,
            "result": qualification_result, "actual_terminal_state": "COMPLETED", "actual_exit_code": "0:0", "admission_granted": False})
        qualification = {"result": qualification_result, "review": qualification_review}
        native = {"diagnostic_plan": {"path": "/synthetic/native_plan.json", "sha256": "2" * 64},
            "diagnostic_result": {"path": "/synthetic/native_result.json", "sha256": "3" * 64},
            "target_contract": {"path": "/synthetic/native_contract.json", "sha256": "4" * 64}}
        native["independent_review"] = self.write("native_review.json", {
            "kind": "independent_configured_native_process_review", "approved": True, "passed": True,
            "independent_of_producer": True, "result": native["diagnostic_result"], "plan": native["diagnostic_plan"],
            "target_contract": native["target_contract"], "profile_admission_granted": False})
        self.proposed["native_process_control"]["expected_target_contract"] = native
        for entry in entries:
            review = json.loads(Path(entry["review"]["path"]).read_text())
            plan = json.loads(Path(review["plan"]["path"]).read_text())
            plan["target_qualification_inputs"] = qualification
            review["plan"] = self.write("data/" + plan["unit_id"] + "/measurement_plan.json", plan)
            if entry["hardware_class"] == "jaguar03_rtx_a4500":
                inventory = {value["path"]: {"sha256": value["sha256"]} for value in (review["plan"], review["result"])}
                review.update(all_files_before=inventory, all_files_after=copy.deepcopy(inventory))
            else:
                consumer = json.loads(Path(review["unchanged_staged_consumer"]["path"]).read_text())
                consumer["plan"] = review["plan"]
                review["unchanged_staged_consumer"] = self.write(plan["unit_id"] + "_consumer.json", consumer)
            entry["review"] = self.write(plan["unit_id"] + "_review.json", review)
            if entry["hardware_class"] == "jaguar03_rtx_a4500" and entry["copies"] == 1:
                self.proposed["inputs"]["accepted_confirmation"] = review["result"]
        proof = self.write("complete_proof.json", {"schema_version": 1, "kind": HELPER.PROTOCOL["kind"],
            "source_commit": HELPER.SOURCE, "campaign_run_dir": str(self.run), "operational_admission": False,
            "contexts": entries, "initial_snapshot_inputs": {"synthetic_premise": True},
            "target_qualification_inputs": qualification, "native_process_qualification": native})
        approval = self.write("complete_approval.json", {
            "kind": "independent_reviewed_four_context_configured_profile_prerequisites", "proof": proof,
            "approved": True, "passed": True, "independent_of_producer": True, "all_four_raw_reviews_verified": True,
            "exact_historical_consumer_namespaces_verified": True, "admission_granted": False,
            "worker_admitted": False, "unresolved_anomalies": []})
        self.proposed["configured_profile_prerequisite_inputs"] = {"proof": proof, "review": approval}
        synthetic_snapshot = {"historical_registry": self.proposed["inputs"]["study_registry"], "synthetic_premise": True}
        with patch.object(HELPER, "source_closure", side_effect=lambda files, plan, unit: plan["helpers"]), \
                patch.object(HELPER, "initial_snapshot", return_value=synthetic_snapshot), \
                patch.object(HELPER, "QUALIFICATION_REVIEW", qualification_review["sha256"]), \
                patch.object(HELPER, "QUALIFICATION_RESULT", qualification_result["sha256"]), \
                patch.object(HELPER, "NATIVE_REVIEW", native["independent_review"]["sha256"]):
            result = HELPER.validate(self.proposed)
        self.assertEqual(len(result["contexts"]), 4)
        self.assertFalse(result["admission_granted"])
        self.assertFalse(result["GPU_numerical_qualification"])
        self.assertFalse(result["current_physical_identity_verified"])

    def test_new_helper_union_is_explicit_and_all_execution_admission_remains_closed(self):
        """Preparation gains one pure helper while default/direct paths and execution stay unchanged."""
        baseline = PROFILE.required_helpers(True, False, False, True, configured_native=True)
        selected = PROFILE.required_helpers(True, False, False, True, configured_native=True, configured_prerequisites=True)
        self.assertEqual(set(selected) - set(baseline), {HELPER.NAME})
        self.assertNotIn("common_confirmation.py", selected)
        self.assertNotIn(HELPER.NAME, PROFILE.required_helpers())
        for arguments in ((False, False, False, False), (True, False, True, True), (False, True, False, False)):
            with self.assertRaises(ValueError):PROFILE.required_helpers(*arguments, configured_native=True, configured_prerequisites=True)
        with self.assertRaisesRegex(ValueError, "disabled"):
            PROFILE.require_reviewed_profile_process_admission()


if __name__ == "__main__":
    unittest.main()
