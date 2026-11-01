"""Check read-only process observations using synthetic standard-library records."""

import copy
import getpass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import advance_adaptive_research as advance
import collect_worker_exit_receipts as collection


class WorkerExitCollectionTests(unittest.TestCase):
    """Cover pending launches, retry jobs, live processes, ownership and immutable inputs."""

    def setUp(self):
        """Prepare one ordinary launch, one retry worker and two direct process identities."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.write("slurm/submitted_jobids_uva.txt", "123\n456\n")
        self.rows = {"123": "123\tqueue-worker\t" + "a" * 40 + "\t/ordinary.sh",
                     "456": "456\tinfrastructure-retry\t" + "a" * 40 + "\t/retry.sh"}
        self.write("slurm/submissions.tsv", "\n".join(self.rows.values()) + "\n")
        self.launch = {"kind": "queue_worker", "hostname": "serval02", "pid": 77, "start_time": "100",
                       "time": "2026-10-31T10:00:00+00:00"}
        self.write("processes.json", [self.launch])
        self.worker = {"worker_id": "retry_worker", "hostname": "jaguar03", "worker_pid": 88,
                       "worker_start_time": "110", "slurm_job_id": "456", "time": "2026-10-31T10:00:00+00:00"}
        self.write("queue/workers/retry.json", self.worker)
        self.runner = {"unit_id": "original_case", "attempt": 4, "hostname": "serval02", "runner_pid": 79,
                       "runner_start_time": "101", "slurm_job_id": "", "finished_at": "2026-10-31T10:01:00+00:00"}
        self.write("queue/leases/original.json", self.runner)
        self.prepared = collection.prepare(self.run, self.run / "remote_setup/exit_observations")
        self.plan = json.loads(self.prepared.read_text())

    def write(self, name, value):
        """Write fixture text or complete JSON below this temporary campaign."""
        path = self.run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value) + "\n")
        return path

    def observations(self, *, live=False):
        """Represent actual terminal jobs and missing or reused direct process identifiers."""
        jobs = {identifier: {"job_id": identifier, "submission_row": row, "state": "COMPLETED",
                            "observed_at": "2026-11-01T05:00:00+00:00"} for identifier, row in self.rows.items()}
        direct = {("serval02", pid, birth): {"hostname": "serval02", "pid": pid, "expected_birth": birth,
                  "observed_birth": birth if live else None, "original_process_alive": live,
                  "observed_at": "2026-11-01T05:00:00+00:00"} for pid, birth in ((77, "100"), (79, "101"))}
        return jobs, direct

    def test_complete_collection_passes_actual_guard_and_preserves_inputs(self):
        """The complete producer receipt covers pending launches, retry workers and direct children."""
        jobs, direct = self.observations()
        before = {name: row["sha256"] for name, row in self.plan["inputs"]["files"].items()}
        with mock.patch.object(collection, "observe_jobs", return_value=(jobs, {"stdout": "raw accounting"})), \
             mock.patch.object(collection, "observe_direct", return_value=(direct, [{"stdout": "raw process births"}])):
            status = collection.collect(self.prepared, self.prepared.parent / "first_observation")
        result = self.prepared.parent / "first_observation/worker_stop.json"
        self.assertEqual(status["state"], "complete")
        self.assertEqual(advance.stopped_workers(self.run, result, advance.validation.Evidence()), [self.worker])
        self.assertEqual(before, {name: row["sha256"] for name, row in collection.snapshot(self.run)["files"].items()})

    def test_live_or_unreported_items_write_waiting_without_stop_receipt(self):
        """A still-live direct process or absent accounting record is never a terminal proof."""
        for reason in ("live_process", "missing_job", "running_job"):
            jobs, direct = self.observations(live=reason == "live_process")
            if reason == "missing_job":
                del jobs["123"]
            if reason == "running_job":
                jobs["123"]["state"] = "RUNNING"
            output = self.prepared.parent / reason
            with self.subTest(reason=reason), mock.patch.object(collection, "observe_jobs", return_value=(jobs, {})), \
                 mock.patch.object(collection, "observe_direct", return_value=(direct, [])):
                self.assertEqual(collection.collect(self.prepared, output)["state"], "waiting")
                self.assertFalse((output / "worker_stop.json").exists())

    def test_ownership_and_exact_rows_reject_missing_ledger_or_duplicate_submission(self):
        """Neither a foreign-cluster ledger nor duplicate rows establishes exact UVA ownership."""
        args = ("\n".join(self.rows.values()), [self.worker], [self.runner])
        with self.assertRaisesRegex(ValueError, "exact UVA ledger"):
            collection.required_jobs(*args, {"123"})
        with self.assertRaisesRegex(ValueError, "one exact submission"):
            collection.required_jobs(args[0] + "\n" + self.rows["456"], *args[1:], {"123", "456"})

    def test_retry_guard_requires_exact_row_owned_ledger_and_complete_coverage(self):
        """The admission guard independently rejects an unowned or omitted retry allocation."""
        jobs, direct = self.observations()
        receipt, _ = collection.make_receipt(self.plan, jobs, direct)
        path = self.write("retry_worker_stop.json", receipt)
        self.write("slurm/submitted_jobids_uva.txt", "123\n")
        with self.assertRaisesRegex(ValueError, "exact UVA job ledger ownership"):
            advance.stopped_workers(self.run, path, advance.validation.Evidence())
        self.write("slurm/submitted_jobids_uva.txt", "123\n456\n")
        self.write("slurm/submissions.tsv", self.rows["123"] + "\n")
        with self.assertRaisesRegex(ValueError, "one exact submission row"):
            advance.stopped_workers(self.run, path, advance.validation.Evidence())
        self.write("slurm/submissions.tsv", "\n".join(self.rows.values()) + "\n")
        receipt["jobs"] = [jobs["123"]]
        self.write("retry_worker_stop.json", receipt)
        with self.assertRaisesRegex(ValueError, "recorded retry jobs"):
            advance.stopped_workers(self.run, path, advance.validation.Evidence())

    def test_guard_rejects_active_scheduler_flag_despite_terminal_accounting(self):
        """The admission guard independently rejects an active requeued allocation."""
        jobs, direct = self.observations()
        receipt, _ = collection.make_receipt(self.plan, jobs, direct)
        receipt["jobs"][0].update(state="PREEMPTED", active_in_scheduler=True, queue_state="PENDING")
        path = self.write("active_worker_stop.json", receipt)
        with self.assertRaisesRegex(ValueError, "not terminal"):
            advance.stopped_workers(self.run, path, advance.validation.Evidence())

    def test_changed_metadata_or_derived_job_list_rejects_before_query(self):
        """A new worker or manipulated prepared identifier cannot evade frozen coverage."""
        manipulated = copy.deepcopy(self.plan)
        manipulated["inputs"]["jobs"]["999"] = "999\tqueue-worker\tignored\t/ignored.sh"
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            collection.verify_inputs(manipulated)
        manipulated = copy.deepcopy(self.plan)
        manipulated["inputs"]["workers"][0]["worker_pid"] = 999
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            collection.verify_inputs(manipulated)
        self.write("queue/workers/new.json", {**self.worker, "worker_id": "new_worker"})
        with mock.patch.object(collection, "observe_jobs") as query, self.assertRaisesRegex(ValueError, "metadata changed"):
            collection.collect(self.prepared, self.prepared.parent / "changed")
        query.assert_not_called()

    def test_changes_during_observation_reject_before_publication(self):
        """Input changes after the read-only commands cannot produce an obsolete stop proof."""
        jobs, direct = self.observations()
        def observe(_):
            """Change one fixture only after the scheduler observation has begun."""
            self.write("processes.json", [self.launch, {**self.launch, "pid": 90}])
            return direct, []
        output = self.prepared.parent / "changed_during_observation"
        with mock.patch.object(collection, "observe_jobs", return_value=(jobs, {})), \
             mock.patch.object(collection, "observe_direct", side_effect=observe), self.assertRaisesRegex(ValueError, "metadata changed"):
            collection.collect(self.prepared, output)
        self.assertFalse((output / "worker_stop.json").exists())

    def test_accounting_queries_only_exact_owned_ids_and_checks_owner(self):
        """An exact allocation query preserves raw cancelled-state details and rejects foreign owners."""
        raw = "123|" + getpass.getuser() + "|CANCELLED by 10|0:0|jaguar03|start|end\n"
        with mock.patch.object(collection, "command_output", side_effect=[raw, ""]) as command:
            jobs, receipt = collection.observe_jobs({"123": self.rows["123"]}, getpass.getuser())
        self.assertEqual(jobs["123"]["state"], "CANCELLED")
        self.assertEqual(receipt["stdout"], raw)
        self.assertIn("--jobs=123", command.call_args_list[0].args[0])
        self.assertIn("-X", command.call_args_list[0].args[0])
        self.assertIn("--jobs=123", command.call_args_list[1].args[0])
        with mock.patch.object(collection, "command_output", side_effect=[raw, ""]), self.assertRaisesRegex(ValueError, "foreign"):
            collection.observe_jobs({"123": self.rows["123"]}, "another_user")

    def test_requeued_job_in_queue_blocks_terminal_accounting_receipt(self):
        """A pending requeued job is live even when accounting still reports PREEMPTED."""
        raw = "123|" + getpass.getuser() + "|PREEMPTED|0:0|jaguar03|start|end\n"
        queue = "123|" + getpass.getuser() + "|PENDING\n"
        with mock.patch.object(collection, "command_output", side_effect=[raw, queue]):
            jobs, _ = collection.observe_jobs({"123": self.rows["123"]}, getpass.getuser())
        all_jobs, direct = self.observations()
        all_jobs.update(jobs)
        receipt, waiting = collection.make_receipt(self.plan, all_jobs, direct)
        self.assertIsNone(receipt)
        self.assertEqual(waiting, [{"job_id": "123", "state": "PENDING"}])

    def test_matching_process_birth_requires_same_user(self):
        """A same-birth process owned by someone else cannot become a study observation."""
        raw = json.dumps({"hostname": "serval02", "observed_at": "2026-11-01T05:00:00+00:00", "observations": [
            {"pid": 77, "expected_birth": "100", "observed_birth": "100", "observed_uid": -1,
             "original_process_alive": True}]})
        with mock.patch.object(collection, "command_output", return_value=raw), self.assertRaisesRegex(ValueError, "different user"):
            collection.observe_direct([("serval02", 77, "100")])

    def test_remote_identity_observation_is_single_call_and_never_signals(self):
        """A reused identifier is observed dead; one failed SSH call is not retried."""
        raw = json.dumps({"hostname": "serval02", "observed_at": "2026-11-01T05:00:00+00:00", "observations": [
            {"pid": 77, "expected_birth": "100", "observed_birth": "200", "observed_uid": 999,
             "original_process_alive": False}]})
        with mock.patch.object(collection, "command_output", return_value=raw) as command:
            records, _ = collection.observe_direct([("serval02", 77, "100")])
        self.assertFalse(records["serval02", 77, "100"]["original_process_alive"])
        self.assertEqual(command.call_count, 1)
        self.assertIn("BatchMode=yes", command.call_args.args[0])
        self.assertNotIn("kill", command.call_args.args[0][-1])
        with mock.patch.object(collection, "command_output", side_effect=RuntimeError("SSH failed")) as command:
            with self.assertRaisesRegex(RuntimeError, "SSH failed"):
                collection.observe_direct([("serval02", 77, "100")])
            self.assertEqual(command.call_count, 1)

    def test_direct_hosts_and_positive_identity_are_restricted(self):
        """The collector cannot open a jaguar03 SSH connection or query an unspecified process."""
        value = copy.deepcopy(self.plan["inputs"])
        value["launches"][0]["hostname"] = "jaguar03"
        with self.assertRaisesRegex(ValueError, "authorized serval02"):
            collection.direct_identities(value)
        value["launches"][0].update(hostname="serval02", pid=0)
        with self.assertRaisesRegex(ValueError, "positive process"):
            collection.direct_identities(value)

    def test_remote_program_reads_exact_requested_birth_counter(self):
        """The generated standard-library remote program reports actual process birth counters."""
        import os
        pid = os.getpid()
        birth = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        def execute(command, **_):
            """Run the generated observation program locally while retaining the requested host."""
            script = command[-1].split("python3 -c ", 1)[1]
            import shlex
            result = subprocess.run([sys.executable, "-c", shlex.split(script)[0]], capture_output=True, text=True, check=True)
            value = json.loads(result.stdout)
            value["hostname"] = "serval02"
            return json.dumps(value)
        with mock.patch.object(collection, "command_output", side_effect=execute):
            records, _ = collection.observe_direct([("serval02", pid, birth)])
        self.assertTrue(records["serval02", pid, birth]["original_process_alive"])


if __name__ == "__main__":
    unittest.main()
