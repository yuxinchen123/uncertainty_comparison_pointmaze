"""Use temporary ledgers and synthetic sbatch returns; never submit or query a job."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("reviewed_monitor_held_submission", HERE / "submit_monitor_held.py")
submit = importlib.util.module_from_spec(spec); spec.loader.exec_module(submit)


class HeldMonitorSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); self.packet = self.root / "packet"; self.packet.mkdir()
        self.campaign = self.root / "campaign"; (self.campaign / "slurm").mkdir(parents=True)
        self.paths = [self.campaign / ("slurm/" + name) for name in ("submitted_jobids.txt", "submitted_jobids_uva.txt", "submissions.tsv")]
        for path, text in zip(self.paths, ("10\n", "10\n", "10\tmonitor\told\t/old/script\n")): path.write_text(text)
        (self.packet / "monitor.sbatch").write_text("#!/bin/bash\n")
        (self.packet / "manifest.yaml").write_text("synthetic metadata only\n")
        (self.packet / "held_submission_inputs.json").write_text("{}\n")
        self.inputs = {"campaign_run_dir": str(self.campaign), "plan": {"path": str(self.packet / "plan.json"), "sha256": "a" * 64}}
        self.plan = {"submission_command": ["sbatch", "--parsable", "--hold", "--no-requeue", str(self.packet / "monitor.sbatch")],
            "data_unit": {"name": "synthetic", "path": str(self.campaign / "data/synthetic")},
            "job_script": submit.receipt(self.packet / "monitor.sbatch")}
        self.addCleanup(patch.stopall)
        patch.object(submit, "PACKET", self.packet).start()
        def preflight():
            submit.unattempted(self.campaign); return self.inputs, self.plan
        patch.object(submit, "validate_inputs", side_effect=preflight).start()
        self.publication = patch.object(submit, "validate_publication", return_value="a" * 40).start()
        patch.object(submit.socket, "gethostname", return_value="portal05").start()
        patch.object(submit.os, "getuid", return_value=2618919).start()
        patch.object(submit.os, "geteuid", return_value=2618919).start()

    def result(self, stdout=b"9910001\n", stderr=b"", code=0):
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=code)

    def test_one_exact_held_call_records_both_ledgers_and_row_before_receipt(self):
        original = submit.publish; observed = []
        def publish(path, value):
            if path.name == "submission_receipt.json":
                self.assertEqual(self.paths[0].read_text().splitlines()[-1], "9910001")
                self.assertEqual(self.paths[1].read_text().splitlines()[-1], "9910001")
                self.assertEqual(self.paths[2].read_text().splitlines()[-1],
                    "\t".join(["9910001", "monitor", submit.SOURCE, str(self.packet / "monitor.sbatch")]))
                observed.append("all three durable records exist")
            return original(path, value)
        with patch.object(submit.subprocess, "run", return_value=self.result()) as scheduler, patch.object(submit, "publish", side_effect=publish):
            record = submit.submit_once("a" * 40)
        scheduler.assert_called_once_with(self.plan["submission_command"], capture_output=True, timeout=90)
        self.assertEqual(observed, ["all three durable records exist"])
        self.assertEqual(record["job_id"], "9910001"); self.assertTrue(record["held"]); self.assertFalse(record["released"])
        self.assertTrue((self.packet / "held_binding.json").is_file())
        with patch.object(submit.subprocess, "run") as scheduler:
            with self.assertRaises(ValueError): submit.submit_once("a" * 40)
            scheduler.assert_not_called()

    def test_publication_failure_precedes_intent_and_scheduler(self):
        self.publication.side_effect = ValueError("not published")
        with patch.object(submit.subprocess, "run") as scheduler:
            with self.assertRaises(ValueError): submit.submit_once("a" * 40)
            scheduler.assert_not_called()
        self.assertFalse((self.packet / "submission_intent.json").exists())

    def test_existing_intent_or_receipt_cannot_be_retried(self):
        for name in ("submission_intent.json", "submission_receipt.json", "held_binding.json"):
            path = self.packet / name; path.write_bytes(b"preserved old bytes")
            with patch.object(submit.subprocess, "run") as scheduler:
                with self.assertRaises(ValueError): submit.submit_once("a" * 40)
                scheduler.assert_not_called()
            self.assertEqual(path.read_bytes(), b"preserved old bytes"); path.unlink()

    def test_existing_packet_row_refuses_even_without_receipt(self):
        self.paths[2].write_text("9910000\tmonitor\t" + submit.SOURCE + "\t" + str(self.packet / "monitor.sbatch") + "\n")
        with patch.object(submit.subprocess, "run") as scheduler:
            with self.assertRaises(ValueError): submit.submit_once("a" * 40)
            scheduler.assert_not_called()

    def test_failed_return_keeps_exact_raw_bytes_and_never_writes_ledgers(self):
        before = [p.read_bytes() for p in self.paths]
        with patch.object(submit.subprocess, "run", return_value=self.result(b"ambiguous\xff\n", b"native error\x00", 1)) as scheduler:
            with self.assertRaises(ValueError): submit.submit_once("a" * 40)
        self.assertEqual((self.packet / "submission_stdout.raw").read_bytes(), b"ambiguous\xff\n")
        self.assertEqual((self.packet / "submission_stderr.raw").read_bytes(), b"native error\x00")
        self.assertEqual(before, [p.read_bytes() for p in self.paths])
        self.assertTrue((self.packet / "submission_return.json").is_file()); self.assertFalse((self.packet / "submission_receipt.json").exists())
        self.assertEqual(scheduler.call_count, 1)

    def test_timeout_is_preserved_and_never_retried(self):
        with patch.object(submit.subprocess, "run", side_effect=subprocess.TimeoutExpired(self.plan["submission_command"], 90, output=b"partial")) as scheduler:
            with self.assertRaises(ValueError): submit.submit_once("a" * 40)
        self.assertEqual((self.packet / "submission_stdout.raw").read_bytes(), b"partial")
        self.assertFalse(json.loads((self.packet / "submission_return.json").read_bytes())["completed"])
        self.assertEqual(scheduler.call_count, 1)

    def test_duplicate_returned_id_refuses_without_partial_ledger_appends(self):
        before = [p.read_bytes() for p in self.paths]
        with patch.object(submit.subprocess, "run", return_value=self.result(b"10\n")):
            with self.assertRaises(ValueError): submit.submit_once("a" * 40)
        self.assertEqual(before, [p.read_bytes() for p in self.paths])
        self.assertFalse((self.packet / "submission_receipt.json").exists())

    def test_ambiguous_success_cannot_select_one_identifier(self):
        with patch.object(submit.subprocess, "run", return_value=self.result(b"9910001\n9910002\n")):
            with self.assertRaises(ValueError): submit.submit_once("a" * 40)
        self.assertEqual(self.paths[0].read_text(), "10\n")

    def test_default_cli_only_validates_publication_and_does_not_submit(self):
        with patch.object(submit.subprocess, "run") as scheduler:
            submit.main(["--publication-commit", "a" * 40])
            scheduler.assert_not_called()
        self.assertFalse((self.packet / "submission_intent.json").exists())

    def test_wrong_execution_owner_or_host_refuses_before_intent(self):
        with patch.object(submit.socket, "gethostname", return_value="different"), patch.object(submit.subprocess, "run") as scheduler:
            with self.assertRaises(ValueError): submit.submit_once("a" * 40)
            scheduler.assert_not_called()
        self.assertFalse((self.packet / "submission_intent.json").exists())


class RecoveryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.inputs = json.loads((HERE / "held_submission_inputs.json").read_text())
        self.basis = json.loads((HERE / "recovery_basis.json").read_text())
        self.original_document = submit.document

    def changed_document(self, changed_path, changed):
        return patch.object(submit, "document", side_effect=lambda path:
            changed if str(path) == str(changed_path) else self.original_document(path))

    def test_actual_static_recovery_evidence_passes_without_candidate_or_old_running(self):
        value = submit.validate_recovery(self.inputs)
        self.assertIs(value["prelaunch_full_candidate_available"], False)
        self.assertIs(value["old_monitor_running_required"], False)
        self.assertIs(value["actual_first_full_report_required_after_start"], True)

    def test_no_prelaunch_full_candidate_or_old_running_claim_may_be_added(self):
        for key in ("prelaunch_full_candidate_available", "prelaunch_full_candidate_executed", "old_monitor_running_required"):
            value = json.loads(json.dumps(self.basis)); value[key] = True
            with self.changed_document(HERE / "recovery_basis.json", value), self.assertRaises(ValueError):
                submit.validate_recovery(self.inputs)

    def test_first_actual_full_report_gate_cannot_be_removed(self):
        value = json.loads(json.dumps(self.basis))
        value["future_work_blocked_until_first_full_report_independent_review"] = False
        with self.changed_document(HERE / "recovery_basis.json", value), self.assertRaises(ValueError):
            submit.validate_recovery(self.inputs)

    def test_coherently_changed_tested_source_does_not_match_archive(self):
        path = next(iter(self.basis["source_validation"].values()))["path"]
        value = self.original_document(path)
        name = next(iter(value["source_after"]))
        value["source_after"][name] = value["source_before"][name] = "a" * 64
        with self.changed_document(path, value), self.assertRaisesRegex(ValueError, "published archive"):
            submit.validate_recovery(self.inputs)

    def test_current_recurring_monitor_must_still_match_actual_recovery_basis(self):
        path = Path(self.inputs["campaign_run_dir"]) / "next_actions.json"
        value = self.original_document(path)
        value["monitoring_session_cron"]["id"] = "private-synthetic-different-task"
        with self.changed_document(path, value), self.assertRaises(ValueError):
            submit.validate_recovery(self.inputs)

    def test_missing_real_root_approval_refuses_publication_before_any_git_or_intent(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(submit, "PACKET", Path(temporary)), \
             patch.object(submit.subprocess, "run", side_effect=AssertionError("no unapproved action")):
            with self.assertRaises(FileNotFoundError):
                submit.validate_publication(self.inputs, "a" * 40)
        self.assertFalse((HERE / "submission_intent.json").exists())


if __name__ == "__main__":
    unittest.main()

