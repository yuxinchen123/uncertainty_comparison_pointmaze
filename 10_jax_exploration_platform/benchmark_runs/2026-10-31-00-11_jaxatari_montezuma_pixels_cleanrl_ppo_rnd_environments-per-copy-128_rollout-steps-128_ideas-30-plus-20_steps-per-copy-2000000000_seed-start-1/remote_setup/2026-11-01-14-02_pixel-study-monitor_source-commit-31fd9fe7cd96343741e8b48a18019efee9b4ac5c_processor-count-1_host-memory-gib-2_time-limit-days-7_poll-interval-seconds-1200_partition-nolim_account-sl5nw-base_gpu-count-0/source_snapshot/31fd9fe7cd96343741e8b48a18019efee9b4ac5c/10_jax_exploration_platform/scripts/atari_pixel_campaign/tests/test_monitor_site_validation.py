"""Check owned site monitoring with small records and no numerical imports or live jobs."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitor


def write(path, value):
    """Write one synthetic JSON record and return its exact path and byte hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepared_site(root, identifier="123"):
    """Create a complete owned submission without source admission, claims or live resources."""
    packet = root / "remote_setup" / "H100 NVL site validation"
    boundary = {"workers_permitted": False, "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0}
    plan = write(packet / "site_validation_plan.json", {"schema_version": 2,
        "kind": "h100_nvl_slurm_site_validation", "campaign_run_dir": str(root),
        "job_shape": {"gres": "gpu:nvidia_h100_nvl:1"}, **boundary})
    script = packet / "job.sbatch"
    script.write_text("#!/bin/bash\nexit 0\n")
    script_record = {"path": str(script), "sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
    command = ["sbatch", "--parsable", "--hold", str(script)]
    manifest = write(packet / "manifest.yaml", {"plan": plan, "job_script": script_record,
                                                "submission_command": command})
    write(packet / "submission_receipt.json", {"kind": "owned_h100_nvl_site_validation_submission",
        "cluster": "uva", "job_id": identifier, "held_command": command,
        "plan_sha256": plan["sha256"], "manifest_sha256": manifest["sha256"],
        "job_script_sha256": script_record["sha256"], **boundary})
    ledger = root / "slurm/submitted_jobids_uva.txt"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(identifier + "\n")
    return packet


class SiteValidationMonitorTests(unittest.TestCase):
    """Keep operational site outcomes visible without granting study or queue credit."""

    def setUp(self):
        """Use an isolated campaign and prevent accidental real scheduler queries."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.packet = prepared_site(self.root)
        self.scheduler = patch.object(monitor.subprocess, "run", side_effect=AssertionError("unexpected scheduler query"))
        self.run = self.scheduler.start()
        self.addCleanup(self.scheduler.stop)

    def rows(self, state="PENDING", reason="Resources", node=""):
        """Inspect the synthetic job in its supplied active state."""
        return monitor.site_validation_rows(self.root, [["123", state, node, "gpu:1", reason]])

    def assignment(self):
        """Record a physical GPU identity for this exact owned Slurm job."""
        return write(self.packet / "site_assignment.json", {"owned_job": {"job_id": "123"},
            "allocation": {"job_id": "123", "hostname": "serval07", "model": "NVIDIA H100 NVL",
                           "gpu_uuid": "GPU-owned", "driver": "580.159.04"}})

    def passed(self):
        """Write raw checks passed, retaining every explicit no-admission boundary."""
        receipt = monitor.read_json(self.packet / "submission_receipt.json")
        return write(self.packet / "result.json", {"schema_version": 1,
            "kind": "h100_nvl_slurm_site_evidence", "passed": True,
            "plan": {"path": str(self.packet / "site_validation_plan.json"), "sha256": receipt["plan_sha256"]},
            "assignment": self.assignment(), "workers_permitted": False,
            "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0})

    def accounting(self, state="FAILED", exit_code="1:0"):
        """Supply one exact top-level accounting row plus an ignored batch row."""
        output = f"123|{state}|{exit_code}|serval07\n123.batch|FAILED|1:0|serval07\n"
        self.run.side_effect = None
        self.run.return_value = subprocess.CompletedProcess([], 0, output, "")

    def test_pending_and_held_keep_requested_gpu_without_planned_host(self):
        """Pending scheduling hints never become a claimed physical allocation."""
        rows, owned, _ = self.rows(node="serval08")
        self.assertEqual(owned, {"123"})
        self.assertEqual(rows[0]["state"], "waiting")
        self.assertEqual(rows[0]["pool"], "UVA")
        self.assertEqual(rows[0]["requested_gpu"], "NVIDIA H100 NVL")
        self.assertIsNone(rows[0]["observed_gpu"])
        self.assertIsNone(rows[0]["gpu_uuid"])
        self.assertEqual(self.rows(reason="JobHeldUser")[0][0]["state"], "held")
        self.run.assert_not_called()

    def test_running_uses_owned_recorded_host_model_and_uuid(self):
        """Only actual allocation metadata supplies the observed model and identity."""
        self.assignment()
        rows, _, _ = self.rows(state="RUNNING", reason="None", node="serval07")
        self.assertEqual(rows[0]["state"], "running")
        self.assertEqual(rows[0]["pool"], "serval07")
        self.assertEqual(rows[0]["observed_gpu"], "NVIDIA H100 NVL")
        self.assertEqual(rows[0]["gpu_uuid"], "GPU-owned")
        self.assertEqual(rows[0]["driver"], "580.159.04")
        conflict = self.rows(state="RUNNING", reason="None", node="serval08")[0][0]
        self.assertEqual(conflict["state"], "failed")
        self.assertIn("recorded site host differs", conflict["progress"])

    def test_failure_before_assignment_remains_visible_after_scheduler_removal(self):
        """The original mapping error is reported without guessing which identifiers were absent."""
        error = "every recorded Slurm job/step allocation must name this exact physical GPU"
        write(self.packet / "failure.json", {"passed": False, "error": error})
        rows, owned, status = monitor.site_validation_rows(self.root, [])
        self.assertEqual(owned, {"123"})
        self.assertEqual(rows[0]["state"], "failed")
        self.assertIn(error, rows[0]["progress"])
        self.assertEqual(rows[0]["pool"], "UVA")
        self.assertIsNone(rows[0]["gpu_uuid"])
        self.assertIn(str(self.packet), status["anomalies"])
        self.run.assert_not_called()

    def test_raw_passed_checks_require_review_and_never_change_admission(self):
        """A site result is an operational outcome rather than a comparison or source authorization."""
        self.passed()
        guard = self.root / "SOURCE_AND_GATES_READY.json"
        guard.write_text('{"untouched": true}\n')
        before = guard.read_bytes()
        rows, _, status = monitor.site_validation_rows(self.root, [])
        row = rows[0]
        self.assertEqual(row["state"], "completed")
        self.assertIn("independent review required", row["progress"])
        self.assertIn("zero case and idea credit", row["progress"])
        self.assertFalse(row["workers_permitted"])
        self.assertFalse(row["campaign_claims_permitted"])
        self.assertEqual(row["scientific_hypothesis_credit"], 0)
        self.assertEqual(guard.read_bytes(), before)
        self.assertFalse((self.root / "queue").exists())
        self.assertEqual(status["anomalies"], {})
        self.run.assert_not_called()

    def test_raw_permission_or_credit_claim_is_rejected(self):
        """Raw success cannot introduce worker permission or research credit."""
        for key, value in (("workers_permitted", True), ("campaign_claims_permitted", True),
                           ("scientific_hypothesis_credit", 1)):
            with self.subTest(key=key):
                self.passed()
                result = monitor.read_json(self.packet / "result.json")
                result[key] = value
                write(self.packet / "result.json", result)
                row = self.rows()[0][0]
                self.assertEqual(row["state"], "failed")
                self.assertIn("zero credit boundary", row["progress"])

    def test_any_failure_file_takes_precedence_over_raw_passed_checks(self):
        """Empty, null or malformed failure records must never permit a passed-site row."""
        self.passed()
        for failure in ("{}", "null", '{"error": "actual site error"}', '{"error":'):
            with self.subTest(failure=failure):
                (self.packet / "failure.json").write_text(failure)
                row = monitor.site_validation_rows(self.root, [])[0][0]
                self.assertEqual(row["state"], "failed")
                self.assertNotIn("checks complete", row["progress"])
        self.run.assert_not_called()

    def test_foreign_cluster_ledger_id_is_never_queried(self):
        """A numeric ID present only in ACES records is not owned on UVA."""
        (self.root / "slurm/submitted_jobids_uva.txt").write_text("999\n")
        (self.root / "slurm/submitted_jobids_aces.txt").write_text("123\n")
        rows, owned, status = monitor.site_validation_rows(self.root, [])
        self.assertEqual(owned, set())
        self.assertEqual(rows[0]["state"], "failed")
        self.assertEqual(rows[0]["slurm_job_id"], "")
        self.assertIn("UVA ledger", rows[0]["progress"])
        self.assertTrue(status["anomalies"])
        self.run.assert_not_called()

    def test_changed_script_and_other_campaign_are_rejected(self):
        """Exact byte binding precedes scheduler fallback and keeps unrelated packets out."""
        (self.packet / "job.sbatch").write_text("changed\n")
        rows, owned, _ = self.rows()
        self.assertEqual(owned, {"123"})
        self.assertIn("job.sbatch", rows[0]["progress"])
        prepared_site(self.root)
        plan = monitor.read_json(self.packet / "site_validation_plan.json")
        plan["campaign_run_dir"] = str(self.root / "other campaign")
        plan_record = write(self.packet / "site_validation_plan.json", plan)
        manifest = monitor.read_json(self.packet / "manifest.yaml")
        manifest["plan"] = plan_record
        manifest_record = write(self.packet / "manifest.yaml", manifest)
        receipt = monitor.read_json(self.packet / "submission_receipt.json")
        receipt.update(plan_sha256=plan_record["sha256"], manifest_sha256=manifest_record["sha256"])
        write(self.packet / "submission_receipt.json", receipt)
        self.assertIn("this campaign", self.rows()[0][0]["progress"])
        self.run.assert_not_called()

    def test_absent_active_job_caches_exact_terminal_accounting(self):
        """A completed allocation without a raw outcome never becomes successful site evidence."""
        self.accounting("COMPLETED", "0:0")
        with patch.dict(os.environ, {"TZ": "America/Los_Angeles"}):
            rows, _, status = monitor.site_validation_rows(self.root, [])
        self.assertEqual(rows[0]["state"], "failed")
        self.assertIn("ended without result or failure: COMPLETED", rows[0]["progress"])
        self.assertIn("123", status["terminal_accounting"])
        self.assertEqual(self.run.call_args.args[0], ["sacct", "-n", "-P", "-j", "123",
                                                    "--format=JobIDRaw,State%64,ExitCode,NodeList"])
        self.assertEqual(self.run.call_args.kwargs["env"]["TZ"], "UTC")
        self.run.side_effect = AssertionError("terminal cache should avoid another query")
        again, _, _ = monitor.site_validation_rows(self.root, [])
        self.assertEqual(again[0]["progress"], rows[0]["progress"])

    def test_complete_scheduler_failure_state_is_preserved(self):
        """The accounting format reserves enough characters for OUT_OF_MEMORY."""
        self.accounting("OUT_OF_MEMORY", "0:9")
        rows, _, status = monitor.site_validation_rows(self.root, [])
        self.assertIn("OUT_OF_MEMORY", rows[0]["progress"])
        self.assertEqual(status["terminal_accounting"]["123"]["state"], "OUT_OF_MEMORY")

    def test_unknown_or_unavailable_accounting_is_an_explicit_anomaly(self):
        """Missing accounting and query errors stay visible while the monitor continues."""
        self.run.side_effect = subprocess.CalledProcessError(1, ["sacct"])
        rows, _, status = monitor.site_validation_rows(self.root, [])
        self.assertEqual(rows[0]["state"], "failed")
        self.assertTrue(status["anomalies"])
        self.accounting("RUNNING", "0:0")
        row = monitor.site_validation_rows(self.root, [])[0][0]
        self.assertEqual(row["state"], "held")
        self.assertIn("accounting state RUNNING", row["progress"])

    def test_ambiguous_top_level_accounting_does_not_infer_an_end(self):
        """A batch substep or duplicated top-level row cannot replace exact job evidence."""
        self.accounting()
        for output in ("123.batch|FAILED|1:0|serval07\n", "123|FAILED|1:0|serval07\n123|COMPLETED|0:0|serval07\n"):
            with self.subTest(output=output):
                self.run.return_value.stdout = output
                row = monitor.site_validation_rows(self.root, [])[0][0]
                self.assertEqual(row["state"], "failed")
                self.assertIn("unambiguous accounting row", row["progress"])

    def test_incomplete_json_does_not_drop_existing_science_rows(self):
        """An interrupted site write is a failed operational row, not a failed whole tick."""
        (self.packet / "submission_receipt.json").write_text('{"kind":')
        science = self.root / "data" / "scientific comparison"
        write(science / "progress.json", {"state": "completed", "passed": True})
        with patch.object(monitor, "own_jobs", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
            monitor.tick(self.root)
        status = monitor.read_json(self.root / "20_mins_monitoring/latest_status.json")
        self.assertEqual(len(status["units"]), 1)
        self.assertEqual(status["units"][0]["run"], "scientific comparison")
        self.assertEqual(status["site_validation"]["sites"][0]["row"]["state"], "failed")
        self.assertEqual(status["campaign"], {"phase": "implementation"})
        self.run.assert_not_called()

    def test_site_ids_have_one_row_and_other_generic_jobs_stay_visible(self):
        """Pending, held and running site jobs must not also get generic study-job rows."""
        for state, reason in (("PENDING", "Resources"), ("PENDING", "JobHeldUser"), ("RUNNING", "None")):
            with self.subTest(state=state, reason=reason):
                jobs = [["123", state, "serval07", "gpu:1", reason], ["999", "PENDING", "", "gpu:1", "OtherResources"]]
                with patch.object(monitor, "own_jobs", return_value=jobs), contextlib.redirect_stdout(io.StringIO()):
                    monitor.tick(self.root)
                report = (self.root / "20_mins_monitoring/latest_report.md").read_text()
                self.assertEqual(report.count("H100 NVL site validation"), 1)
                self.assertEqual(report.count("scheduled study job"), 1)
                self.assertIn("OtherResources", report)
                self.assertNotIn("held study job", report)
                status = monitor.read_json(self.root / "20_mins_monitoring/latest_status.json")
                self.assertEqual(status["units"], [])

    def test_only_direct_child_submission_kind_is_discovered(self):
        """Nested and unrelated receipts do not become site work or scheduler queries."""
        (self.packet / "submission_receipt.json").unlink()
        prepared_site(self.root / "remote_setup/nested campaign")
        write(self.root / "remote_setup/other packet/submission_receipt.json", {"kind": "another submission"})
        rows, owned, _ = monitor.site_validation_rows(self.root, [])
        self.assertEqual((rows, owned), ([], set()))
        self.run.assert_not_called()

    def test_linked_packet_or_receipt_cannot_substitute_outside_records(self):
        """A direct-child name alone does not permit linked submissions outside the packet."""
        original = self.packet / "submission_receipt.json"
        outside = self.root / "outside submission.json"
        outside.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(outside)
        row = self.rows()[0][0]
        self.assertEqual(row["state"], "failed")
        self.assertIn("inside its own packet", row["progress"])
        original.unlink()
        original.write_bytes(outside.read_bytes())
        self.packet.rename(self.root / "outside packet")
        self.packet.symlink_to(self.root / "outside packet", target_is_directory=True)
        row = self.rows()[0][0]
        self.assertIn("direct child", row["progress"])
        self.run.assert_not_called()

    def test_damaged_monitoring_cache_and_incomplete_result_stay_visible(self):
        """A broken operational cache or partial result does not end the standing monitor."""
        write(self.root / "20_mins_monitoring/site_validation_status.json", {"anomalies": [], "terminal_accounting": []})
        (self.packet / "result.json").write_text('{"passed":')
        rows, owned, status = self.rows()
        self.assertEqual(owned, {"123"})
        self.assertEqual(rows[0]["state"], "failed")
        self.assertIn("monitoring record", status["anomalies"])
        self.assertIn(str(self.packet), status["anomalies"])
        self.assertEqual(status["terminal_accounting"], {})
        self.run.assert_not_called()

    def test_recorded_mapping_observation_is_preserved_without_gpu_compatibility_claim(self):
        """Raw mapping records expose the actual host/card while retaining the actual error."""
        observation = write(self.packet / "allocation_observation.json", {"hostname": "serval08",
            "owned_job": {"job_id": "123"}, "gpu_csv": "0, GPU-observed, NVIDIA H100 NVL, 95830, 580.159.04\n",
            "absent_allocation_fields": ["SLURM_STEP_GPUS"]})
        write(self.packet / "allocation_details.json", {"recorded": "metadata only"})
        write(self.packet / "failure.json", {"error": "allocation identifiers disagree"})
        row = monitor.site_validation_rows(self.root, [])[0][0]
        self.assertEqual(row["pool"], "serval08")
        self.assertEqual(row["gpu_uuid"], "GPU-observed")
        self.assertIn(observation, row["allocation_records"])
        self.assertIn("allocation identifiers disagree", row["progress"])
        self.assertNotIn("compatibility", row["progress"])


if __name__ == "__main__":
    unittest.main()
