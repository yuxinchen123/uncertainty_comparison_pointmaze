"""Check one owned allocation-observation status row without scientific credit or scheduler actions."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitor

BOUNDARY = {"mapping_accepted": False, "compatibility_verified": False, "workers_permitted": False,
            "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0}


def write(path, value):
    """Write a synthetic metadata file and bind its actual bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class AllocationObservationMonitorTests(unittest.TestCase):
    """Keep collection visible while qualification and admission remain separate."""

    def setUp(self):
        """Create one complete prepared observer packet and prevent live scheduler queries."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.packet = self.root / "remote_setup/2026-10-31-17-37_h100-nvl-allocation-observation"
        self.unit = self.root / "data" / self.packet.name
        self.unit.mkdir(parents=True)
        self.plan_path = self.packet / "allocation_observation_plan.json"
        self.plan = {"schema_version": 1, "kind": "h100_nvl_allocation_observation", "campaign_run_dir": str(self.root),
            "data_unit": {"path": str(self.unit), "name": self.unit.name}, "job_shape": {
                "account": "sl5nw_base", "partition": "gpu", "gres": "gpu:nvidia_h100_nvl:1"}, **BOUNDARY}
        plan = write(self.plan_path, self.plan)
        write(self.packet / "config_resolved.yaml", self.plan)
        (self.packet / "job.sbatch").write_text("#!/bin/bash\nexit 0\n")
        self.command = ["sbatch", "--hold", "--account=sl5nw_base", "--partition=gpu", "--gres=gpu:nvidia_h100_nvl:1", str(self.packet / "job.sbatch")]
        (self.packet / "command.txt").write_text("prepared metadata command\n")
        (self.packet / "experiment_background.md").write_text("Metadata only; no mapping acceptance.\n")
        self.manifest = {"schema_version": 1, "kind": "prepared_h100_nvl_allocation_observation", "submission_command": self.command,
            "artifacts": {name: {"path": str(self.packet / name), "sha256": hashlib.sha256((self.packet / name).read_bytes()).hexdigest()}
                          for name in ("allocation_observation_plan.json", "config_resolved.yaml", "job.sbatch", "command.txt", "experiment_background.md")}, **BOUNDARY}
        manifest = write(self.packet / "manifest.yaml", self.manifest)
        self.receipt = {"kind": "owned_h100_nvl_allocation_observation_submission", "cluster": "uva", "job_id": "123",
            "data_unit": self.plan["data_unit"], "held_command": self.command,
            "plan_sha256": plan["sha256"], "manifest_sha256": manifest["sha256"],
            "job_script_sha256": self.manifest["artifacts"]["job.sbatch"]["sha256"], **BOUNDARY}
        write(self.packet / "submission_receipt.json", self.receipt)
        for filename in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md"):
            (self.unit / filename).write_bytes((self.packet / filename).read_bytes())
        write(self.unit / "progress.json", {"state": "waiting_for_allocation_observation", **BOUNDARY})
        ledger = self.root / "slurm/submitted_jobids_uva.txt"
        ledger.parent.mkdir()
        ledger.write_text("123\n")
        write(self.root / "campaign_state.json", {"phase": "initial_benchmarks", "queue_counts": {
            "completed": 87, "pending": 32, "running": 1, "failed": 0}})
        self.query = patch.object(monitor.subprocess, "run", side_effect=AssertionError("unexpected scheduler query")).start()
        self.addCleanup(patch.stopall)

    def rows(self, jobs=None):
        """Read the owned observer with an explicit current scheduler observation."""
        return monitor.site_validation_rows(self.root, [["123", "PENDING", "", "gpu:1", "Resources"]] if jobs is None else jobs)

    def hardware(self, identifier="123"):
        """Record the model and host actually observed by this owned metadata process."""
        write(self.unit / "hardware.json", {"hostname": "serval08", "slurm_job_id": identifier,
            "cuda_visible_devices": "0", "gpu_query": "0, NVIDIA H100 NVL, 95830 MiB, 580.159.04, GPU-observed", **BOUNDARY})

    def collected(self):
        """Bind raw completed collection bytes while explicitly leaving qualification false."""
        self.hardware()
        observations = write(self.unit / "hardware_observations.json", {"job_id": "123", "hostname": "serval08",
            "plan": {"path": str(self.plan_path), "sha256": self.receipt["plan_sha256"]}, **BOUNDARY})
        write(self.unit / "result.json", {"schema_version": 1, "kind": "h100_nvl_allocation_observation_result", "passed": True,
            "observations_completed": True, "observations": observations, **BOUNDARY})
        write(self.unit / "progress.json", {"state": "completed", "passed": True, **BOUNDARY})

    def report(self, jobs):
        """Run a synthetic metadata tick and return its report and compact status."""
        with patch.object(monitor, "own_jobs", return_value=jobs), contextlib.redirect_stdout(io.StringIO()):
            monitor.tick(self.root)
        return ((self.root / "20_mins_monitoring/latest_report.md").read_text(),
                monitor.read_json(self.root / "20_mins_monitoring/latest_status.json"))

    def test_pending_has_one_literal_row_owned_job_and_requested_pool(self):
        """The prepared data shard and pending job become one readable operational row."""
        report, status = self.report([["123", "PENDING", "serval09", "gpu:1", "Resources"]])
        self.assertEqual(report.count("H100 NVL allocation observation"), 1)
        self.assertNotIn("scheduled study job", report)
        self.assertNotIn(self.unit.name, report)
        self.assertEqual(status["units"], [])
        row = status["allocation_observations"][0]["row"]
        self.assertEqual(status["allocation_observations"][0]["submission_receipt"]["path"], str(self.packet / "submission_receipt.json"))
        self.assertEqual(row["pool"], "UVA gpu/sl5nw")
        self.assertEqual(row["gpu"], ["H100 NVL requested"])
        self.assertEqual(row["slurm_job_id"], "123")
        self.assertNotIn("123", row["progress"])
        self.assertIsNone(row["observed_host"])
        self.assertFalse(row["qualification_verified"])
        self.assertEqual(status["site_validation"]["sites"], [])
        self.assertEqual(status["campaign"]["queue_counts"]["completed"], 87)
        self.query.assert_not_called()

    def test_held_and_running_show_one_row_with_actual_host_only_when_allocated(self):
        """Held placement stays unassigned; running hardware shows the actual model and host."""
        held = self.rows([["123", "PENDING", "serval09", "gpu:1", "JobHeldUser"]])[0][0]
        self.assertEqual(held["state"], "held")
        self.assertIsNone(held["observed_host"])
        self.hardware()
        report, status = self.report([["123", "RUNNING", "serval08", "gpu:1", "None"]])
        row = status["allocation_observations"][0]["row"]
        self.assertEqual(row["state"], "running")
        self.assertEqual(row["observed_host"], "serval08")
        self.assertEqual(row["gpu"], ["NVIDIA H100 NVL"])
        self.assertEqual(row["gpu_uuid"], "GPU-observed")
        self.assertIn("qualification required", row["progress"])
        self.assertEqual(report.count("H100 NVL allocation observation"), 1)
        self.assertNotIn("study job", report)

    def test_completed_collection_is_not_qualified_or_admitted(self):
        """A raw result is shown as COLLECTED even if an unrelated file claims qualification."""
        self.collected()
        write(self.unit / "qualification.json", {"qualified": True, "workers_permitted": True})
        report, status = self.report([])
        row = status["allocation_observations"][0]["row"]
        self.assertIn("COLLECTED", report)
        self.assertEqual(row["display_state"], "COLLECTED")
        self.assertTrue(row["metadata_collected"])
        self.assertFalse(row["qualification_verified"])
        self.assertFalse(row["mapping_accepted"])
        self.assertFalse(row["workers_permitted"])
        self.assertFalse(row["campaign_claims_permitted"])
        self.assertEqual(row["scientific_hypothesis_credit"], 0)
        self.assertFalse((self.root / "SOURCE_AND_GATES_READY.json").exists())
        self.assertFalse((self.root / "queue").exists())
        self.query.assert_not_called()

    def test_any_failure_file_overrides_collection_and_keeps_one_row(self):
        """Empty, null and malformed failures remain failures after the job leaves the scheduler."""
        self.collected()
        for contents in ('{"error":"hardware observation failed"}', '{}', 'null', '{"error":'):
            with self.subTest(contents=contents):
                (self.unit / "failure.json").write_text(contents)
                report, status = self.report([])
                self.assertEqual(report.count("H100 NVL allocation observation"), 1)
                row = status["allocation_observations"][0]["row"]
                self.assertEqual(row["state"], "failed")
                self.assertFalse(row["metadata_collected"])
                self.assertFalse(row["qualification_verified"])
                self.assertNotIn("COLLECTED", report)
        self.query.assert_not_called()

    def test_foreign_cluster_ledger_never_allows_a_query(self):
        """The same numeric ID in a foreign ledger does not establish UVA ownership."""
        (self.root / "slurm/submitted_jobids_uva.txt").write_text("999\n")
        (self.root / "slurm/submitted_jobids_aces.txt").write_text("123\n")
        rows, owned, _ = self.rows([])
        self.assertEqual(owned, set())
        self.assertEqual(rows[0]["state"], "failed")
        self.assertEqual(rows[0]["slurm_job_id"], "")
        self.assertIn("UVA ledger", rows[0]["progress"])
        self.query.assert_not_called()

    def test_changed_artifacts_are_failed_without_duplicate_data_rows(self):
        """A changed prepared script cannot silently become a different observation job."""
        (self.packet / "job.sbatch").write_text("changed script\n")
        report, status = self.report([["123", "PENDING", "", "gpu:1", "Resources"]])
        self.assertEqual(report.count("H100 NVL allocation observation"), 1)
        self.assertNotIn("scheduled study job", report)
        self.assertEqual(status["units"], [])
        self.assertEqual(status["allocation_observations"][0]["row"]["state"], "failed")

    def test_raw_mapping_permission_or_credit_claim_is_rejected(self):
        """Neither a raw result nor altered observation bytes can turn collection into admission."""
        self.collected()
        result = monitor.read_json(self.unit / "result.json")
        result["mapping_accepted"] = True
        write(self.unit / "result.json", result)
        row = self.rows([])[0][0]
        self.assertEqual(row["state"], "failed")
        self.assertFalse(row["qualification_verified"])
        self.collected()
        (self.unit / "hardware_observations.json").write_text("changed bytes\n")
        self.assertEqual(self.rows([])[0][0]["state"], "failed")

    def test_hardware_job_or_allocated_host_mismatch_is_not_displayed_as_valid(self):
        """A different job or host cannot supply the observer's displayed physical identity."""
        self.hardware("999")
        self.assertEqual(self.rows()[0][0]["state"], "failed")
        self.hardware()
        row = self.rows([["123", "RUNNING", "serval09", "gpu:1", "None"]])[0][0]
        self.assertEqual(row["state"], "failed")
        self.assertIn("host differs", row["progress"])

    def test_completed_collection_requires_its_matching_recorded_host(self):
        """A changed hardware sidecar cannot display a host different from the bound raw observations."""
        self.collected()
        hardware = monitor.read_json(self.unit / "hardware.json")
        hardware["hostname"] = "serval09"
        write(self.unit / "hardware.json", hardware)
        row = self.rows([])[0][0]
        self.assertEqual(row["state"], "failed")
        self.assertIn("matching recorded host", row["progress"])

    def test_ended_without_outcome_uses_only_owned_cached_accounting(self):
        """A completed scheduler allocation alone does not prove metadata collection."""
        self.query.side_effect = None
        self.query.return_value = subprocess.CompletedProcess([], 0, "123|COMPLETED|0:0|serval08\n", "")
        row = self.rows([])[0][0]
        self.assertEqual(row["state"], "failed")
        self.assertIn("ended without result or failure", row["progress"])
        self.assertFalse(row["metadata_collected"])
        self.assertEqual(self.query.call_args.args[0][0:5], ["sacct", "-n", "-P", "-j", "123"])
        self.query.side_effect = AssertionError("cached terminal accounting must be reused")
        self.assertEqual(self.rows([])[0][0]["progress"], row["progress"])

    def test_replaced_data_shard_and_other_command_pool_are_rejected(self):
        """The recorded data path and held resource command remain bound to the prepared packet."""
        other = self.root / "data/unrelated unit"
        other.mkdir()
        self.receipt["data_unit"] = {"path": str(other), "name": other.name}
        write(self.packet / "submission_receipt.json", self.receipt)
        self.assertEqual(self.rows()[0][0]["state"], "failed")
        self.receipt["data_unit"] = self.plan["data_unit"]
        self.command[3] = "--partition=another_pool"
        self.manifest["submission_command"] = self.command
        proof = write(self.packet / "manifest.yaml", self.manifest)
        self.receipt.update(held_command=self.command, manifest_sha256=proof["sha256"])
        write(self.packet / "submission_receipt.json", self.receipt)
        row = self.rows()[0][0]
        self.assertEqual(row["state"], "failed")
        self.assertIn("pool or GPU", row["progress"])

    def test_malformed_mutable_progress_is_skipped_for_a_bound_observer(self):
        """The verified operational reader replaces its shard before generic progress parsing."""
        self.collected()
        (self.unit / "progress.json").write_text('{"state":')
        report, status = self.report([])
        self.assertIn("COLLECTED", report)
        self.assertEqual(status["units"], [])

    def profiled_site(self, schema=3, include_profile=True, profile_changes=None):
        """Prepare the explicit new site-display metadata without importing its qualification implementation."""
        packet = self.root / "remote_setup/explicit driver site validation"
        boundary = {"workers_permitted": False, "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0}
        profile = {"schema_version": 1, "name": "slurm-configured-device-isolation-driver-580.159.04",
            "expected_driver": "580.159.04", "numeric_gpu_mapping_reconstructed": False,
            "device_bpf_rules_independently_inspected": False, **boundary, **(profile_changes or {})}
        plan = {"schema_version": schema, "kind": "h100_nvl_slurm_site_validation", "campaign_run_dir": str(self.root),
                "job_shape": {"gres": "gpu:nvidia_h100_nvl:1"}, "reference": {"driver": "590.48.01"}, **boundary}
        if include_profile:
            plan["qualification_profile"] = profile
        proof = write(packet / "site_validation_plan.json", plan)
        (packet / "job.sbatch").write_text("#!/bin/bash\nexit 0\n")
        script = {"path": str(packet / "job.sbatch"), "sha256": hashlib.sha256((packet / "job.sbatch").read_bytes()).hexdigest()}
        command = ["sbatch", "--hold", str(packet / "job.sbatch")]
        manifest = write(packet / "manifest.yaml", {"plan": proof, "job_script": script, "submission_command": command})
        receipt = {"kind": "owned_h100_nvl_site_validation_submission", "cluster": "uva", "job_id": "456",
            "held_command": command, "plan_sha256": proof["sha256"], "manifest_sha256": manifest["sha256"],
            "job_script_sha256": script["sha256"], **boundary}
        write(packet / "submission_receipt.json", receipt)
        (self.root / "slurm/submitted_jobids_uva.txt").write_text("123\n456\n")
        return packet, plan, receipt

    def test_explicit_schema_three_profile_is_displayed_as_raw_checks_requiring_review(self):
        """An approved display profile never replaces the separately pinned full site verifier."""
        packet, plan, receipt = self.profiled_site()
        identifier, verified = monitor.site_packet(self.root, packet, receipt, {"123", "456"})
        self.assertEqual(identifier, "456")
        self.assertEqual(verified["schema_version"], 3)
        write(packet / "result.json", {"schema_version": 1, "kind": "h100_nvl_slurm_site_evidence", "passed": True,
            "plan": {"path": str(packet / "site_validation_plan.json"), "sha256": receipt["plan_sha256"]},
            "qualification_profile": plan["qualification_profile"], "workers_permitted": False,
            "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0})
        rows, _, _ = self.rows()
        site = next(row for row in rows if row["kind"] == "site_validation")
        self.assertEqual(site["state"], "completed")
        self.assertIn("independent review required", site["progress"])
        self.assertEqual(site["expected_driver"], "580.159.04")
        self.assertEqual(site["reference_driver"], "590.48.01")
        self.assertFalse(site["workers_permitted"])
        self.assertEqual(site["scientific_hypothesis_credit"], 0)
        self.query.assert_not_called()

    def test_site_profile_requires_schema_three_and_explicit_approved_fields(self):
        """A missing profile, wrong marker, other driver or permission claim cannot display as the new context."""
        for options in ({"include_profile": False}, {"schema": 2}, {"profile_changes": {"name": "unapproved profile"}},
                        {"profile_changes": {"expected_driver": "590.48.01"}}, {"profile_changes": {"workers_permitted": True}}):
            with self.subTest(options=options):
                packet, _, receipt = self.profiled_site(**options)
                with self.assertRaisesRegex(ValueError, "schema 3|approved qualification"):
                    monitor.site_packet(self.root, packet, receipt, {"123", "456"})

    def test_schema_three_raw_result_must_retain_its_exact_profile(self):
        """The completed raw result may not drop the profile bound into its prepared plan."""
        packet, _, receipt = self.profiled_site()
        write(packet / "result.json", {"schema_version": 1, "kind": "h100_nvl_slurm_site_evidence", "passed": True,
            "plan": {"path": str(packet / "site_validation_plan.json"), "sha256": receipt["plan_sha256"]},
            "workers_permitted": False, "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0})
        rows, _, _ = self.rows()
        site = next(row for row in rows if row["kind"] == "site_validation")
        self.assertEqual(site["state"], "failed")
        self.assertIn("qualification profile", site["progress"])


if __name__ == "__main__":
    unittest.main()
