"""Check bounded ACES site mirrors without scheduler, network or numerical execution."""

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aces_site_monitor as adapter


def write(path, value):
    """Write one exact synthetic native record and return its byte checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value) + "\n")
    return adapter.digest(path.read_bytes())


class SiteMonitorTests(unittest.TestCase):
    """Keep native ACES ownership, UTC freshness and pending independent verification explicit."""

    def setUp(self):
        """Create a native-shaped packet whose local and remote logical paths remain distinct."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.packet = self.run / "remote_setup/2026-11-01-01-00_ACES_full-site-qualification_gpu-count-1"
        self.unit = self.run / "data" / self.packet.name
        self.unit.mkdir(parents=True)
        self.remote = adapter.REMOTE_ROOT / self.run.name
        self.remote_packet = self.remote / "remote_setup" / self.packet.name
        self.identifier = "2274999"
        self.plan = {"schema_version": 1, "kind": "aces_full_site_qualification", "cluster": "aces",
            "campaign_run_dir": str(self.remote), "source_commit": adapter.SOURCE_COMMIT,
            "owner": {"username": adapter.OWNER, "uid": 56626}, "helpers": {"source_commit": "a" * 40},
            "data_unit": {"name": self.packet.name, "path": str(self.remote / "data" / self.packet.name)},
            "job_shape": {"account": "158590754315", "partition": "gpu", "qos": "normal", "nodes": 1,
                "ntasks": 1, "gres": "gpu:h100:1", "cpus_per_task": 8, "memory": "32G", "time": "03:00:00"},
            **adapter.BOUNDARY}
        software = write(self.packet / "software_at_prepare.json", {"fixture": "native observed software"})
        self.plan["software_at_prepare"] = {"path": str(self.remote_packet / "software_at_prepare.json"), "sha256": software}
        write(self.packet / "site_validation_plan.json", self.plan)
        for name in ("job.sbatch", "command.txt", "config_resolved.yaml", "experiment_background.md"):
            write(self.packet / name, "synthetic native prepared bytes\n")
        self.command = ["sbatch", "--parsable", "--hold", "--account=158590754315", "--partition=gpu", "--qos=normal",
            "--nodes=1", "--ntasks=1", "--gres=gpu:h100:1", "--cpus-per-task=8", "--mem=32G", "--time=03:00:00",
            str(self.remote_packet / "job.sbatch")]
        self.manifest = {"schema_version": 1, "kind": "aces_full_site_qualification_preparation", "cluster": "aces",
            "submission_command": self.command, "artifacts": {name: {"path": str(self.remote_packet / name),
                "sha256": adapter.digest((self.packet / name).read_bytes())} for name in adapter.ARTIFACTS}, **adapter.BOUNDARY}
        self.receipt = {"schema_version": 1, "kind": "owned_aces_full_site_qualification_submission", "cluster": "aces",
            "job_id": self.identifier, "data_unit": self.plan["data_unit"], "held_command": self.command,
            "submission_row": [self.identifier, self.packet.name, "a" * 40, str(self.remote)],
            "ledger_path": str(self.remote / "slurm/submitted_jobids_aces.txt"), **adapter.BOUNDARY}
        write(self.run / "slurm/submitted_jobids_aces.txt", self.identifier + "\n")
        write(self.run / "slurm/submitted_jobids_uva.txt", self.identifier + "\n")
        write(self.run / "slurm/submissions_aces.tsv", "\t".join(self.receipt["submission_row"]) + "\n")
        self.release = {"schema_version": 1, "kind": "owned_aces_full_site_qualification_release", "cluster": "aces",
            "job_id": self.identifier, "command": ["scontrol", "release", self.identifier], "returncode": 0}
        self.publish_packet()
        self.snapshot()

    def publish_packet(self):
        """Rebind intentional fixture changes without pretending they are unchanged native bytes."""
        write(self.packet / "manifest.yaml", self.manifest)
        for name, key in (("site_validation_plan.json", "plan_sha256"), ("manifest.yaml", "manifest_sha256"),
                          ("job.sbatch", "job_script_sha256")):
            self.receipt[key] = adapter.digest((self.packet / name).read_bytes())
        self.release["submission_receipt_sha256"] = write(self.packet / "submission_receipt.json", self.receipt)
        write(self.packet / "release_receipt.json", self.release)

    def snapshot(self, queue="PENDING", state="PENDING", exit_code="0:0", reason="Priority"):
        """Turn exact synthetic native file bytes and raw scheduler rows into one local UTC snapshot."""
        commands = adapter.query_commands(self.identifier)
        self.document = {"schema_version": 1, "kind": "aces_full_site_monitor_snapshot", "cluster": "aces", "user": adapter.OWNER,
            "job_id": self.identifier, "origin_host": "portal05", "query_host": "alogin1.cluster",
            "local_campaign_run_dir": str(self.run), "remote_campaign_run_dir": str(self.remote),
            **{key: self.receipt[key] for key in ("plan_sha256", "manifest_sha256", "job_script_sha256")},
            "submission_receipt_sha256": adapter.digest((self.packet / "submission_receipt.json").read_bytes()),
            "release_receipt_sha256": adapter.digest((self.packet / "release_receipt.json").read_bytes())
                if (self.packet / "release_receipt.json").exists() else None,
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "squeue": {"command": commands[0], "returncode": 0, "stdout": self.identifier + "|" + adapter.OWNER + "|" + queue +
                "|ac041|" + reason + "\n" if queue else "", "stderr": ""},
            "sacct": {"command": commands[1], "returncode": 0, "stdout": self.identifier + "|" + adapter.OWNER + "|" + state +
                "|" + exit_code + "|ac041\n", "stderr": ""},
            "mirrored_files": {str(path.relative_to(self.run)): {"source_path": str(self.remote / path.relative_to(self.run)),
                "sha256": adapter.digest(path.read_bytes())} for path in self.unit.glob("*.json")}}
        self.save_snapshot()

    def save_snapshot(self):
        """Publish only the fixture's deliberately changed snapshot record."""
        write(self.packet / "remote_monitor_snapshot.json", self.document)

    def row(self):
        """Read the optional mirror through the same contained entry point as the persistent monitor."""
        rows, _ = adapter.rows(self.run)
        self.assertEqual(len(rows), 1)
        return rows[0]

    def result(self):
        """Create a zero-credit reported result bound to this native plan and exact owned allocation."""
        parent = {"owned_allocation": {"job_id": self.identifier}}
        assignment = {"job_id": self.identifier, "hostname": "ac041"}
        value = {"schema_version": 1, "kind": "aces_full_site_evidence", "passed": True,
            "plan": {"path": str(self.remote_packet / "site_validation_plan.json"), "sha256": self.receipt["plan_sha256"]},
            **adapter.BOUNDARY}
        for key, name, record in (("parent_observation", "parent_allocation_observation.json", parent),
                                  ("assignment", "site_assignment.json", assignment)):
            checksum = write(self.unit / name, record)
            value[key] = {"path": str(self.remote / "data" / self.unit.name / name), "sha256": checksum}
        write(self.unit / "result.json", value)
        return value

    def test_held_and_running_stage_progress_never_qualify_or_query_uva(self):
        """An equal UVA number does not change the ACES job identity or enable any scheduler call."""
        self.assertEqual(self.row()["state"], "waiting")
        write(self.unit / "identity_terminal.json", {"job_id": self.identifier, "kind": "identity", "exit_code": 0})
        self.snapshot(queue="RUNNING", state="PREEMPTED", reason="")
        row = self.row()
        self.assertEqual(row["state"], "running")
        self.assertIn("identity passed", row["progress"])
        self.assertIs(row["qualification_verified"], False)
        self.assertEqual(row["scientific_case_credit"], 0)

    def test_completed_result_requires_terminal_zero_exit_and_independent_review(self):
        """A saved result is operational evidence and cannot become a worker or idea admission."""
        self.result()
        self.snapshot(queue=None, state="COMPLETED")
        row = self.row()
        self.assertEqual(row["state"], "completed")
        self.assertIn("independent verification required", row["progress"])
        self.assertIs(row["workers_permitted"], False)
        self.assertEqual(row["idea_credit"], 0)
        self.snapshot(queue="RUNNING", state="COMPLETED", reason="")
        self.assertEqual(self.row()["state"], "running")
        self.snapshot(queue=None, state="COMPLETED", exit_code="1:0")
        self.assertEqual(self.row()["state"], "failed")

    def test_missing_foreign_or_changed_result_never_looks_finished(self):
        """A terminal row cannot substitute for the exact plan, allocation and preserved result bytes."""
        self.snapshot(queue=None, state="COMPLETED")
        self.assertEqual(self.row()["state"], "failed")
        value = self.result()
        value["plan"]["sha256"] = "f" * 64
        write(self.unit / "result.json", value)
        self.snapshot(queue=None, state="COMPLETED")
        self.assertEqual(self.row()["state"], "unknown")
        self.result()
        self.snapshot(queue=None, state="COMPLETED")
        write(self.unit / "site_assignment.json", {"job_id": "other", "hostname": "ac041"})
        self.assertEqual(self.row()["state"], "unknown")

    def test_failure_bytes_remain_available_and_never_become_a_success(self):
        """Native failure details are displayed directly while the exact raw JSON remains unchanged."""
        path = self.unit / "failure.json"
        write(path, {"kind": "aces_full_site_failure", "passed": False, "error": "missing root-owned GRES File mapping", **adapter.BOUNDARY})
        before = path.read_bytes()
        self.snapshot(queue=None, state="FAILED", exit_code="1:0")
        self.assertEqual(self.row()["state"], "failed")
        self.assertIn("missing root-owned", self.row()["progress"])
        self.assertEqual(path.read_bytes(), before)

    def test_stale_future_naive_or_malformed_utc_is_unknown_including_dst_hour(self):
        """Freshness uses actual UTC, never ambiguous Pacific report text during the repeated hour."""
        for value in ((datetime.now(timezone.utc) - timedelta(seconds=2401)).isoformat(),
                      (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(), "2026-11-01T01:05:00", "bad"):
            self.document["observed_at_utc"] = value
            self.save_snapshot()
            self.assertEqual(self.row()["state"], "unknown")

    def test_foreign_duplicate_unknown_and_empty_scheduler_records_are_unknown(self):
        """Malformed optional remote metadata cannot end the complete study report."""
        for text in ("", "other|u.sl359075|RUNNING|ac041|\n", self.document["squeue"]["stdout"] * 2,
                     self.identifier + "|u.sl359075|UNRECOGNIZED|ac041|\n"):
            self.document["squeue"]["stdout"] = text
            self.document["sacct"]["stdout"] = ""
            self.save_snapshot()
            self.assertEqual(self.row()["state"], "unknown")

    def test_missing_or_malformed_optional_packet_is_contained(self):
        """Missing snapshots, empty commands and malformed receipts become UNKNOWN without reads elsewhere."""
        (self.packet / "remote_monitor_snapshot.json").unlink()
        self.assertEqual(self.row()["state"], "unknown")
        self.manifest["submission_command"] = []
        self.receipt["held_command"] = []
        self.publish_packet()
        self.assertEqual(self.row()["state"], "unknown")
        write(self.packet / "submission_receipt.json", [])
        self.assertEqual(self.row()["state"], "unknown")

    def test_exact_aces_ledger_and_source_row_precede_any_status(self):
        """UVA ownership, a duplicate ACES ID or another helper source never authorizes this mirror."""
        write(self.run / "slurm/submitted_jobids_aces.txt", "")
        self.assertEqual(self.row()["state"], "unknown")
        write(self.run / "slurm/submitted_jobids_aces.txt", self.identifier + "\n" + self.identifier + "\n")
        self.assertEqual(self.row()["state"], "unknown")
        write(self.run / "slurm/submitted_jobids_aces.txt", self.identifier + "\n")
        write(self.run / "slurm/submissions_aces.tsv", "\t".join([self.identifier, self.packet.name, "b" * 40, str(self.remote)]) + "\n")
        self.assertEqual(self.row()["state"], "unknown")

    def test_symlink_and_outside_mirror_paths_are_rejected(self):
        """Identical copied bytes cannot redirect a packet artifact or its original path identity."""
        outside = self.run / "outside.json"
        path = self.packet / "software_at_prepare.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        self.assertEqual(self.row()["state"], "unknown")

    def test_optional_index_or_date_overflow_cannot_end_the_combined_report(self):
        """Malformed optional records are contained even when a parser raises an index or overflow error."""
        for error in (IndexError("empty optional list"), OverflowError("invalid optional timestamp")):
            with patch.object(adapter, "site_row", side_effect=error):
                self.assertEqual(self.row()["state"], "unknown")

    def test_combined_tick_keeps_gpu_label_in_every_remote_state_and_zero_case_counts(self):
        """WAITING, HELD, TRAINING, FINISHED and UNKNOWN all preserve the requested card and study counts."""
        import monitor
        import final_training_monitor
        counts = {"completed": 103, "running": 1, "pending": 16, "failed": 0}
        write(self.run / "campaign_state.json", {"phase": "initial_benchmarks", "queue_counts": counts})
        for state, section in (("waiting", "WAITING"), ("held", "HELD"), ("running", "TRAINING"),
                               ("completed", "FINISHED"), ("unknown", "UNKNOWN")):
            self.publish_packet()
            if state == "held":
                (self.packet / "release_receipt.json").unlink()
                self.snapshot(reason="(JobHeldUser)")
            elif state == "completed":
                self.result()
                self.snapshot(queue=None, state="COMPLETED")
            else:
                self.snapshot(queue="RUNNING" if state == "running" else "PENDING", reason="" if state == "running" else "Priority")
            if state == "unknown":
                self.document["observed_at_utc"] = "invalid UTC time"
                self.save_snapshot()
            with self.subTest(state=state), patch.object(monitor, "own_jobs", return_value=[]), \
                    patch.object(final_training_monitor, "tick", return_value={"state": "planned", "reason": "not admitted", "units": []}), \
                    patch.object(subprocess, "run", side_effect=AssertionError("live scheduler forbidden")), \
                    patch.object(subprocess, "check_output", side_effect=AssertionError("live command forbidden")), redirect_stdout(io.StringIO()):
                monitor.tick(self.run)
            report = (self.run / "20_mins_monitoring/latest_report.md").read_text()
            status = json.loads((self.run / "20_mins_monitoring/latest_status.json").read_text())
            self.assertIn(section, report)
            self.assertIn("H100 PCIe", report)
            self.assertEqual(status["campaign"]["queue_counts"], counts)
            self.assertEqual(status["aces_full_site_checks"][0]["row"]["state"], state)
            self.assertEqual(status["aces_full_site_checks"][0]["row"]["scientific_case_credit"], 0)


if __name__ == "__main__":
    unittest.main()
