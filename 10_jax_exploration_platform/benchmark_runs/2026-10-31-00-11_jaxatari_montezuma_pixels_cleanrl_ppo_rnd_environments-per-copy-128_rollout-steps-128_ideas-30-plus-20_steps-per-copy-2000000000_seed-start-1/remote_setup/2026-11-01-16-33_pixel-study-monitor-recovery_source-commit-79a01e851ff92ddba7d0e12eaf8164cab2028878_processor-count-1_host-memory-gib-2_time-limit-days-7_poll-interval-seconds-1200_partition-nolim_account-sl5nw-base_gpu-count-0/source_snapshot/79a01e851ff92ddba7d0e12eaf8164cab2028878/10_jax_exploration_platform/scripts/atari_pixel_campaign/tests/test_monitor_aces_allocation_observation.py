"""Exercise ACES observation mirrors without SSH, GPU work or live scheduler queries."""

import contextlib
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitor

BOUNDARY = {"mapping_accepted": False, "compatibility_verified": False, "workers_permitted": False,
            "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0}


def write(path, value):
    """Store synthetic metadata as complete JSON or exact supplied text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ACESObservationMonitorTests(unittest.TestCase):
    """Keep foreign job numbers, actual states and metadata credit separate from UVA training."""

    def setUp(self):
        """Prepare one complete local mirror with original remote path identities."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.packet = self.root / "remote_setup/2026-11-01-00-00_aces_allocation-observation_gpu-count-1"
        self.unit = self.root / "data" / self.packet.name
        self.remote = Path("/scratch/user/u.sl359075/rnd_atari_pixels/runs") / self.root.name
        self.remote_packet = self.remote / "remote_setup" / self.packet.name
        self.commit, self.identifier = "a" * 40, "2274380"
        self.plan = {"schema_version": 1, "kind": "aces_allocation_observation", "cluster": "aces",
            "campaign_run_dir": str(self.remote), "data_unit": {"path": str(self.remote / "data" / self.unit.name), "name": self.unit.name},
            "helpers": {"source_commit": self.commit}, "owner": {"username": "u.sl359075", "uid": 1001},
            "job_shape": {"account": "158590754315", "partition": "gpu", "qos": "normal", "nodes": 1,
                "ntasks": 1, "gres": "gpu:h100:1", "cpus_per_task": 8, "memory": "32G", "time": "00:05:00"}, **BOUNDARY}
        plan_sha = write(self.packet / "allocation_observation_plan.json", self.plan)
        write(self.packet / "config_resolved.yaml", self.plan)
        script_sha = write(self.packet / "job.sbatch", "#!/bin/bash\nexit 0\n")
        self.command = ["sbatch", "--hold", "--account=158590754315", "--partition=gpu", "--qos=normal",
            "--gres=gpu:h100:1", "--cpus-per-task=8", "--mem=32G", "--time=00:05:00", str(self.remote_packet / "job.sbatch")]
        write(self.packet / "command.txt", "recorded held command\n")
        write(self.packet / "experiment_background.md", "Only metadata is collected.\n")
        write(self.packet / "resource_basis/aces_submitted_jobids.txt", "122\n")
        self.manifest = {"schema_version": 1, "kind": "aces_allocation_observation_preparation", "cluster": "aces",
            "submission_command": self.command, "data_unit": self.plan["data_unit"], "artifacts": {
                name: {"path": str(self.remote_packet / name), "sha256": hashlib.sha256((self.packet / name).read_bytes()).hexdigest()}
                for name in ("allocation_observation_plan.json", "config_resolved.yaml", "job.sbatch", "command.txt",
                             "experiment_background.md", "resource_basis/aces_submitted_jobids.txt")}, **BOUNDARY}
        manifest_sha = write(self.packet / "manifest.yaml", self.manifest)
        self.row = [self.identifier, self.packet.name, self.commit, str(self.remote)]
        write(self.root / "slurm/submitted_jobids_aces.txt", self.identifier + "\n")
        write(self.root / "slurm/submissions_aces.tsv", "\t".join(self.row) + "\n")
        write(self.root / "slurm/submitted_jobids_uva.txt", self.identifier + "\n")
        self.receipt = {"schema_version": 1, "kind": "owned_aces_allocation_observation_submission", "cluster": "aces",
            "job_id": self.identifier, "held_command": self.command, "plan_sha256": plan_sha, "manifest_sha256": manifest_sha,
            "job_script_sha256": script_sha, "data_unit": self.plan["data_unit"], "submission_row": self.row,
            "ledger_path": str(self.remote / "slurm/submitted_jobids_aces.txt"), **BOUNDARY}
        submission_sha = write(self.packet / "submission_receipt.json", self.receipt)
        self.release = {"schema_version": 1, "kind": "owned_aces_allocation_observation_release", "cluster": "aces",
            "job_id": self.identifier, "command": ["scontrol", "release", self.identifier], "returncode": 0,
            "submission_receipt_sha256": submission_sha}
        write(self.packet / "release_receipt.json", self.release)
        write(self.unit / "progress.json", {"state": "waiting_for_allocation_observation", **BOUNDARY})
        write(self.root / "campaign_state.json", {"phase": "initial_benchmarks", "queue_counts": {"completed": 98, "pending": 21, "running": 1, "failed": 0}})
        self.make_snapshot()
        self.query = patch.object(monitor.subprocess, "run", side_effect=AssertionError("unexpected scheduler query")).start()
        self.addCleanup(patch.stopall)

    def make_snapshot(self, queue_state="PENDING", reason="Priority", accounting_state="PENDING", exit_code="0:0"):
        """Bind exact raw scheduler commands and current transported bytes to their original paths."""
        self.snapshot = {"schema_version": 1, "kind": "aces_allocation_monitor_snapshot", "cluster": "aces",
            "user": "u.sl359075", "job_id": self.identifier, "origin_host": "portal05", "query_host": "alogin1.cluster",
            "local_campaign_run_dir": str(self.root), "remote_campaign_run_dir": str(self.remote),
            "plan_sha256": self.receipt["plan_sha256"], "job_script_sha256": self.receipt["job_script_sha256"],
            "submission_receipt_sha256": hashlib.sha256((self.packet / "submission_receipt.json").read_bytes()).hexdigest(),
            "release_receipt_sha256": hashlib.sha256((self.packet / "release_receipt.json").read_bytes()).hexdigest()
                if (self.packet / "release_receipt.json").exists() else None,
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "squeue": {"command": ["squeue", "-h", "-j", self.identifier, "-o", "%i|%u|%T|%N|%R"],
                "returncode": 0, "stdout": "" if queue_state is None else self.identifier + "|u.sl359075|" + queue_state + "|ac045|" + reason + "\n", "stderr": ""},
            "sacct": {"command": ["sacct", "-X", "-n", "-P", "-j", self.identifier, "--format=JobIDRaw,User%64,State%64,ExitCode,NodeList"],
                "returncode": 0, "stdout": "" if accounting_state is None else self.identifier + "|u.sl359075|" + accounting_state + "|" + exit_code + "|ac045\n", "stderr": ""},
            "mirrored_files": {str(path.relative_to(self.root)): {"source_path": str(self.remote / path.relative_to(self.root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in self.unit.glob("*.json")}}
        self.save_snapshot()

    def save_snapshot(self):
        """Publish the fixture's current transported status without changing any immutable packet."""
        write(self.packet / "remote_monitor_snapshot.json", self.snapshot)

    def rows(self):
        """Read the actual adapter without making any live job or host query."""
        return monitor.aces_allocation_observation_rows(self.root)

    def report(self, jobs=()):
        """Render the actual full monitor using supplied UVA-only active rows."""
        with patch.object(monitor, "own_jobs", return_value=list(jobs)), contextlib.redirect_stdout(io.StringIO()):
            monitor.tick(self.root)
        return (self.root / "20_mins_monitoring/latest_report.md").read_text(), monitor.read_json(self.root / "20_mins_monitoring/latest_status.json")

    def collected(self):
        """Create unchanged remote collection bytes with a complete linked ownership chain."""
        plan = {"path": str(self.remote_packet / "allocation_observation_plan.json"), "sha256": self.receipt["plan_sha256"]}
        owned = {"schema_version": 1, "kind": "aces_owned_allocation", "cluster": "aces", "job_id": self.identifier,
                 "step_id": "0", "uid": 1001, "hostname": "ac045", **BOUNDARY}
        owned_sha = write(self.unit / "owned_allocation.json", owned)
        observations = {"schema_version": 1, "kind": "aces_allocation_hardware_observations", "cluster": "aces", "job_id": self.identifier,
            "step_id": "0", "plan": plan, "owned_allocation": {"path": str(self.remote / "data" / self.unit.name / "owned_allocation.json"),
                                                                    "sha256": owned_sha}, **BOUNDARY}
        observed_sha = write(self.unit / "hardware_observations.json", observations)
        write(self.unit / "result.json", {"schema_version": 1, "kind": "aces_allocation_observation_result", "cluster": "aces", "job_id": self.identifier,
            "step_id": "0", "data_unit": self.plan["data_unit"], "plan": plan, "passed": True, "observations_completed": True,
            "observations": {"path": str(self.remote / "data" / self.unit.name / "hardware_observations.json"), "sha256": observed_sha}, **BOUNDARY})
        self.make_snapshot(queue_state=None, accounting_state="COMPLETED")

    def test_pending_has_one_requested_h100_row_without_scientific_credit(self):
        """A local pending shard becomes one ACES operational row and preserves scientific counts."""
        report, status = self.report()
        self.assertEqual(report.count("ACES allocation observation"), 1)
        self.assertEqual(status["units"], [])
        row = status["aces_allocation_observations"][0]["row"]
        self.assertEqual((row["state"], row["pool"], row["gpu"]), ("waiting", "ACES gpu/u.sl359075", ["H100"]))
        self.assertFalse(row["qualification_verified"])
        self.assertNotIn("observed_host", row)
        self.assertEqual(status["campaign"]["queue_counts"]["completed"], 98)
        self.query.assert_not_called()

    def test_actual_held_row_before_release_and_missing_release_other_states(self):
        """Only a current owned JobHeldUser row permits a null release hash and HELD display."""
        (self.packet / "release_receipt.json").unlink()
        self.make_snapshot(reason="JobHeldUser")
        self.assertEqual(self.rows()[0][0]["state"], "held")
        self.make_snapshot(reason="Priority")
        self.assertEqual(self.rows()[0][0]["state"], "unknown")
        self.make_snapshot(queue_state="RUNNING")
        self.assertEqual(self.rows()[0][0]["state"], "unknown")

    def test_actual_parenthesized_held_reason_retains_raw_bytes_and_rejects_other_wrapping(self):
        """Accept the actual ACES JobHeldUser row without stripping arbitrary parentheses or changing raw output."""
        (self.packet / "release_receipt.json").unlink()
        self.make_snapshot(reason="(JobHeldUser)")
        actual = "2274380|u.sl359075|PENDING||(JobHeldUser)\n"
        self.snapshot["squeue"]["stdout"] = actual
        self.save_snapshot()
        row = self.rows()[0][0]
        self.assertEqual(row["state"], "held")
        self.assertEqual(json.loads((self.packet / "remote_monitor_snapshot.json").read_text())["squeue"]["stdout"], actual)
        for reason in ("((JobHeldUser))", "(JobHeldUser", "JobHeldUser)", "(Priority)", "(another reason)"):
            self.snapshot["squeue"]["stdout"] = "2274380|u.sl359075|PENDING||" + reason + "\n"
            self.save_snapshot()
            with self.subTest(reason=reason):
                self.assertEqual(self.rows()[0][0]["state"], "unknown")

    def test_running_all_hardware_rows_do_not_infer_two_allocated_gpus_or_pcie(self):
        """Multiple observed UUID rows remain hardware metadata, never an allocated-card count."""
        write(self.unit / "hardware.json", {"cluster": "aces", "slurm_job_id": self.identifier, "gpu_query": "", "gpu_uuid_selected": False,
            "observed_hardware_rows": [{"name": "NVIDIA H100 PCIe", "uuid": "GPU-one"}, {"name": "NVIDIA H100 PCIe", "uuid": "GPU-two"}]})
        self.make_snapshot(queue_state="RUNNING", reason="None", accounting_state="RUNNING")
        row = self.rows()[0][0]
        self.assertEqual((row["state"], row["gpu"]), ("running", ["H100"]))
        self.assertNotIn("PCIe", str(row))

    def test_same_job_number_cannot_suppress_a_running_uva_job(self):
        """A raw mirrored ACES hardware identifier never enters UVA suppression or queries."""
        write(self.unit / "hardware.json", {"slurm_job_id": self.identifier, "gpu_query": ""})
        self.make_snapshot()
        report, _ = self.report([[self.identifier, "RUNNING", "slurm5", "none", "None"]])
        self.assertIn("study job", report)
        self.assertEqual(report.count("ACES allocation observation"), 1)
        self.query.assert_not_called()

    def test_collected_is_only_metadata_and_uses_complete_unchanged_chain(self):
        """Successful raw collection is visible without compatibility, mapping or training admission."""
        self.collected()
        row = self.rows()[0][0]
        self.assertEqual(row["display_state"], "COLLECTED")
        self.assertTrue(row["metadata_collected"])
        self.assertFalse(row["workers_permitted"])
        self.assertEqual(row["scientific_hypothesis_credit"], 0)
        (self.unit / "owned_allocation.json").write_text("{}\n")
        self.assertEqual(self.rows()[0][0]["state"], "unknown")

    def test_missing_stale_future_failed_or_duplicate_snapshots_are_visible_unknown(self):
        """Unavailable status remains UNKNOWN instead of inventing a current running or held state."""
        original = copy.deepcopy(self.snapshot)
        for kind in ("stale", "future", "naive", "non_utc", "cluster", "owner", "query_host", "query_failed", "duplicate", "empty", "schema"):
            self.snapshot = copy.deepcopy(original)
            if kind == "stale": self.snapshot["observed_at_utc"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            elif kind == "future": self.snapshot["observed_at_utc"] = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
            elif kind == "naive": self.snapshot["observed_at_utc"] = "2026-11-01T06:00:00"
            elif kind == "non_utc": self.snapshot["observed_at_utc"] = "2026-11-01T01:00:00-05:00"
            elif kind == "cluster": self.snapshot["cluster"] = "uva"
            elif kind == "owner": self.snapshot["squeue"]["stdout"] = self.snapshot["squeue"]["stdout"].replace("u.sl359075", "another_user")
            elif kind == "query_host": self.snapshot["query_host"] = "another_login"
            elif kind == "query_failed": self.snapshot["squeue"]["returncode"] = 2
            elif kind == "duplicate": self.snapshot["sacct"]["stdout"] *= 2
            elif kind == "empty":
                self.snapshot["squeue"]["stdout"] = ""
                self.snapshot["sacct"]["stdout"] = ""
            else: self.snapshot["schema_version"] = True
            self.save_snapshot()
            with self.subTest(kind=kind):
                report, status = self.report()
                self.assertEqual(status["aces_allocation_observations"][0]["row"]["state"], "unknown")
                self.assertIn("UNKNOWN", report)
                self.assertNotIn("COLLECTED", report)
        (self.packet / "remote_monitor_snapshot.json").unlink()
        self.assertIn("UNKNOWN", self.report()[0])

    def test_foreign_ledger_and_changed_receipt_cannot_hide_under_uva_id_collision(self):
        """UVA ownership cannot authorize ACES, and changed release bytes invalidate an old snapshot."""
        write(self.root / "slurm/submitted_jobids_aces.txt", "999\n")
        self.assertEqual(self.rows()[0][0]["state"], "unknown")
        write(self.unit / "hardware.json", {"slurm_job_id": self.identifier, "gpu_query": ""})
        report, status = self.report([[self.identifier, "RUNNING", "slurm5", "none", "None"]])
        self.assertIn("study job", report)
        self.assertIn("UNKNOWN", report)
        self.assertEqual(status["units"], [])
        write(self.root / "slurm/submitted_jobids_aces.txt", self.identifier + "\n")
        self.release["returncode"] = 1
        write(self.packet / "release_receipt.json", self.release)
        self.assertEqual(self.rows()[0][0]["state"], "unknown")

    def test_malformed_null_list_and_wrong_kind_receipts_stay_unknown_without_uva_suppression(self):
        """Corrupt recognized metadata packets never become generic UVA jobs or suppress an owned job."""
        write(self.unit / "hardware.json", {"slurm_job_id": self.identifier, "gpu_query": ""})
        for value in ("{malformed\n", "null\n", "[]\n", json.dumps({**self.receipt, "kind": "another_submission_kind"})):
            write(self.packet / "submission_receipt.json", value)
            with self.subTest(receipt=value[:35]):
                report, status = self.report([[self.identifier, "RUNNING", "slurm5", "none", "None"]])
                self.assertIn("UNKNOWN", report)
                self.assertIn("study job", report)
                self.assertEqual(status["units"], [])
                self.assertEqual(len(status["aces_allocation_observations"]), 1)
                self.assertEqual(status["aces_allocation_observations"][0]["row"]["state"], "unknown")
                self.assertFalse(status["aces_allocation_observations"][0]["row"]["workers_permitted"])
                self.assertEqual(status["campaign"]["queue_counts"]["completed"], 98)

    def test_wrong_namespace_does_not_claim_a_generic_data_leaf(self):
        """An arbitrary packet name cannot use an ACES receipt to exclude another data directory."""
        original = self.packet
        self.packet = self.root / "remote_setup/unrelated_observation"
        original.rename(self.packet)
        self.assertEqual(self.rows(), ([], []))

    def test_requeued_active_row_overrides_terminal_accounting_and_saved_result(self):
        """An actual pending allocation remains waiting even when old accounting or artifacts are terminal."""
        self.collected()
        self.make_snapshot(queue_state="PENDING", accounting_state="PREEMPTED")
        self.assertEqual(self.rows()[0][0]["state"], "waiting")

    def test_finished_job_missing_result_is_failed_and_invalid_job_queue_code_is_allowed(self):
        """An absent queue row needs exact terminal accounting and actual collection files."""
        self.make_snapshot(queue_state=None, accounting_state="COMPLETED")
        self.snapshot["squeue"].update(returncode=1, stderr="slurm_load_jobs error: Invalid job id specified\n")
        self.save_snapshot()
        self.assertEqual(self.rows()[0][0]["state"], "failed")

    def test_data_directory_symlink_and_changed_zero_credit_type_reject(self):
        """Mirrors cannot escape through a parent symlink or use Boolean research-credit values."""
        self.receipt["scientific_hypothesis_credit"] = False
        write(self.packet / "submission_receipt.json", self.receipt)
        self.assertEqual(self.rows()[0][0]["state"], "unknown")
        self.receipt["scientific_hypothesis_credit"] = 0
        write(self.packet / "submission_receipt.json", self.receipt)
        moved = self.root / "outside_data"
        self.unit.rename(moved)
        self.unit.symlink_to(moved, target_is_directory=True)
        self.assertEqual(self.rows()[0][0]["state"], "unknown")


if __name__ == "__main__":
    unittest.main()
