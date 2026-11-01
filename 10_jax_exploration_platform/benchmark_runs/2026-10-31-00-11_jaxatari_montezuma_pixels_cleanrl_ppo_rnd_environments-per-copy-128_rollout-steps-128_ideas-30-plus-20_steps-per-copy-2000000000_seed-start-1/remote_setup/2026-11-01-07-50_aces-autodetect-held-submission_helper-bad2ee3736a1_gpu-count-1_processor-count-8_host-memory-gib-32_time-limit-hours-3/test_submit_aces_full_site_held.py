"""Test one-shot held submission metadata with all real cluster commands forbidden."""

import base64
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("tested_held_submission", ROOT / "submit_aces_full_site_held.py")
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)
PLAN = ROOT / "aces_full_site_held_submission_plan.json"


class HeldSubmissionTests(unittest.TestCase):
    """Use private ordinary temporary records and synthetic scheduler returns only."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run = self.root / "run"
        (self.run / "slurm").mkdir(parents=True)
        self.ledger = self.run / "slurm/submitted_jobids_aces.txt"
        self.table = self.run / "slurm/submissions_aces.tsv"
        self.ledger.write_text("101\n")
        self.table.write_text("101\told packet\told source\told run\n")
        self.packet = self.run / "remote_setup/new-packet"
        self.packet.mkdir(parents=True)
        self.plan = {"native_run": str(self.run), "held_command": ["sbatch", "--parsable", "--hold", "synthetic"],
            "helpers": {"source_commit": "a" * 40}, "data_unit": {"name": "new-packet", "path": str(self.run / "data/new-packet")},
            "plan_sha256": "1" * 64, "manifest_sha256": "2" * 64, "job_script_sha256": "3" * 64}

    def fake_submit(self, stdout=b"202\n", returncode=0):
        def invoke(command, **arguments):
            self.assertEqual(command, self.plan["held_command"])
            self.assertTrue((self.packet / "held_submission_intent.json").is_file())
            self.assertEqual(self.ledger.read_text(), "101\n")
            arguments["stdout"].write(stdout)
            arguments["stderr"].write(b"synthetic account message\n")
            return SimpleNamespace(returncode=returncode)
        return invoke

    def test_real_local_frozen_inputs_validate_without_commands(self):
        with mock.patch.object(driver.subprocess, "run", side_effect=AssertionError("commands forbidden")):
            payload, plan = driver.local_payload(PLAN)
        self.assertEqual(len(plan["local_files"]), 523)
        self.assertEqual(plan["private_python"], json.loads(Path(plan["local_packet"]).joinpath("site_validation_plan.json").read_bytes())["python"])
        self.assertNotIn("torch", sys.modules)

    def test_success_is_held_once_and_both_remote_ledgers_precede_receipt(self):
        with mock.patch.object(driver.subprocess, "run", side_effect=self.fake_submit()) as submit:
            receipt, paths = driver.submit_once(self.plan, self.packet, "b" * 40, "c" * 64)
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(self.ledger.read_text(), "101\n202\n")
        self.assertEqual(self.table.read_text().splitlines()[-1], "\t".join(receipt["submission_row"]))
        self.assertEqual(receipt["job_id"], "202")
        self.assertTrue((self.packet / "held_submission_return.json").is_file())
        self.assertEqual(len(paths), 7)
        self.assertFalse((self.packet / "release_receipt.json").exists())
        with mock.patch.object(driver.subprocess, "run") as retry:
            with self.assertRaises(FileExistsError):
                driver.submit_once(self.plan, self.packet, "b" * 40, "c" * 64)
            retry.assert_not_called()

    def test_failed_or_ambiguous_return_preserves_raw_and_does_not_append(self):
        for output, code in ((b"202\n203\n", 0), (b"", 1)):
            packet = self.packet / ("case" + str(code))
            packet.mkdir()
            with mock.patch.object(driver.subprocess, "run", side_effect=lambda command, **kw: (kw["stdout"].write(output), SimpleNamespace(returncode=code))[1]):
                with self.assertRaisesRegex(ValueError, "ambiguous ID"):
                    driver.submit_once(self.plan, packet, "b" * 40, "c" * 64)
            self.assertEqual((packet / "held_submission_stdout.raw").read_bytes(), output)
            self.assertTrue((packet / "held_submission_return.json").is_file())
            self.assertEqual(self.ledger.read_text(), "101\n")
            self.assertFalse((packet / "submission_receipt.json").exists())

    def test_timeout_keeps_intent_and_raw_without_guessing_a_job_id(self):
        def timeout(command, **arguments):
            arguments["stdout"].write(b"202\n")
            raise subprocess.TimeoutExpired(command, 90)
        with mock.patch.object(driver.subprocess, "run", side_effect=timeout) as submit:
            with self.assertRaises(subprocess.TimeoutExpired):
                driver.submit_once(self.plan, self.packet, "b" * 40, "c" * 64)
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(self.ledger.read_text(), "101\n")
        self.assertEqual((self.packet / "held_submission_stdout.raw").read_bytes(), b"202\n")
        self.assertIsNone(json.loads((self.packet / "held_submission_return.json").read_bytes())["returncode"])

    def test_incomplete_or_duplicate_existing_ledger_stops_before_submission(self):
        for value in ("101", "101\n101\n", "foreign\n"):
            self.ledger.write_text(value)
            with mock.patch.object(driver.subprocess, "run") as submit:
                with self.assertRaises(ValueError):
                    driver.submit_once(self.plan, self.packet, "b" * 40, "c" * 64)
                submit.assert_not_called()
            self.assertFalse((self.packet / "held_submission_intent.json").exists())

    def test_returned_existing_id_never_duplicates_the_ledger(self):
        with mock.patch.object(driver.subprocess, "run", side_effect=self.fake_submit(b"101\n")):
            with self.assertRaisesRegex(ValueError, "conflicts"):
                driver.submit_once(self.plan, self.packet, "b" * 40, "c" * 64)
        self.assertEqual(self.ledger.read_text(), "101\n")
        self.assertFalse((self.packet / "submission_receipt.json").exists())

    def test_linked_ledger_or_existing_immutable_intent_is_never_replaced(self):
        self.ledger.unlink()
        target = self.root / "existing.txt"
        target.write_text("101\n")
        self.ledger.symlink_to(target)
        with mock.patch.object(driver.subprocess, "run") as submit:
            with self.assertRaisesRegex(ValueError, "ordinary"):
                driver.submit_once(self.plan, self.packet, "b" * 40, "c" * 64)
            submit.assert_not_called()
        self.assertEqual(target.read_text(), "101\n")

    def test_receipt_matches_existing_full_site_monitor_contract(self):
        actual = json.loads(PLAN.read_bytes())
        run = self.root / Path(actual["local_campaign_run_dir"]).name
        packet = run / "remote_setup" / Path(actual["local_packet"]).name
        shutil.copytree(actual["local_packet"], packet)
        unit = run / "data" / packet.name
        shutil.copytree(Path(actual["local_campaign_run_dir"]) / "data" / packet.name, unit)
        (run / "slurm").mkdir()
        receipt = driver.submission_receipt(actual, actual["native_packet"], "202", "b" * 40, "c" * 64, b"202\n", b"")
        (packet / "submission_receipt.json").write_text(json.dumps(receipt))
        (run / "slurm/submitted_jobids_aces.txt").write_text("202\n")
        (run / "slurm/submissions_aces.tsv").write_text("\t".join(receipt["submission_row"]) + "\n")
        monitor_path = Path(actual["local_repository"]) / "10_jax_exploration_platform/scripts/atari_pixel_campaign/aces_site_monitor.py"
        spec = importlib.util.spec_from_file_location("actual_frozen_aces_monitor_contract", monitor_path)
        monitor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monitor)
        with mock.patch.object(driver.subprocess, "run", side_effect=AssertionError("commands forbidden")):
            result = monitor.packet_inputs(run, packet)
        self.assertEqual(result["identifier"], "202")
        self.assertIsNone(result["release"])

    def test_foreign_publication_is_rejected_before_any_local_intent(self):
        with mock.patch.object(driver.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout=b"", stderr=b"")) as git:
            with self.assertRaisesRegex(ValueError, "published"):
                driver.local_payload(PLAN, "b" * 40)
        self.assertTrue(all(call.args[0][0] == "git" for call in git.call_args_list))

    def test_exact_response_keeps_remote_bytes_separate_without_local_ledger_changes(self):
        self.plan["native_packet"] = str(self.packet)
        with mock.patch.object(driver.subprocess, "run", side_effect=self.fake_submit()):
            receipt, paths = driver.submit_once(self.plan, self.packet, "b" * 40, "c" * 64)
        files = [{**driver.record(path), "name": path.name, "contents_base64": base64.b64encode(path.read_bytes()).decode()} for path in paths]
        response = {"receipt": receipt, "files": files}
        destination = self.root / "local-received"
        destination.mkdir()
        result = driver.accept_response(json.dumps(response).encode(), destination, self.plan, "c" * 64, "b" * 40)
        self.assertEqual(result, receipt)
        for path in paths:
            self.assertEqual((destination / "received" / path.name).read_bytes(), path.read_bytes())
        self.assertEqual(self.ledger.read_text(), "101\n202\n")
        self.assertFalse((destination / "submission_receipt.json").exists())
        for name, change in (("foreign-path", lambda r: r["files"][0].update(path="/foreign/submission_receipt.json")),
                ("boolean-credit", lambda r: r["receipt"].update(idea_credit=False))):
            changed = copy.deepcopy(response)
            change(changed)
            output = self.root / name
            output.mkdir()
            with self.assertRaises(ValueError):
                driver.accept_response(json.dumps(changed).encode(), output, self.plan, "c" * 64, "b" * 40)

    def test_busy_ledger_lock_blocks_before_any_submission_or_intent(self):
        with mock.patch.object(driver.fcntl, "flock", side_effect=BlockingIOError("busy own ACES ledger")), \
                mock.patch.object(driver.subprocess, "run") as submit:
            with self.assertRaises(BlockingIOError):
                driver.submit_once(self.plan, self.packet, "b" * 40, "c" * 64)
            submit.assert_not_called()
        self.assertFalse((self.packet / "held_submission_intent.json").exists())

    def test_bootstrap_checks_source_before_invoking_a_metadata_function(self):
        source = 'def remote_submit(payload):\n print("synthetic held metadata only")\n'
        payload = (json.dumps({"driver_source": source}) + "\n").encode()
        for source_hash, expected in ((driver.digest(source.encode()), 0), ("f" * 64, 1)):
            result = subprocess.run([sys.executable, "-I", "-B", "-c", driver.BOOTSTRAP, source_hash],
                input=payload, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, expected)
            self.assertEqual(result.stdout, b"synthetic held metadata only\n" if expected == 0 else b"")


if __name__ == "__main__":
    unittest.main()
