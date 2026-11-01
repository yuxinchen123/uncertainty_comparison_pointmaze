"""Check portal05 refresh ownership, transport and immutable publication without live SSH."""

import base64
import copy
import hashlib
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import refresh_aces_allocation_observation as refresh
import test_monitor_aces_allocation_observation as fixtures


class ACESObservationRefreshTests(unittest.TestCase):
    """Use complete synthetic ownership records and the actual monitor as the publication gate."""

    def setUp(self):
        """Reuse complete metadata fixtures while forbidding any unmocked external command."""
        self.fixture = fixtures.ACESObservationMonitorTests("test_pending_has_one_requested_h100_row_without_scientific_credit")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        patch.object(refresh.socket, "gethostname", return_value="portal05").start()
        patch.object(refresh.pwd, "getpwuid", return_value=SimpleNamespace(pw_name="sl5nw")).start()
        self.addCleanup(patch.stopall)
        self.run, self.packet = self.fixture.root, self.fixture.packet

    def payload(self):
        """Represent unchanged original remote files and exact raw scheduler output."""
        inputs = refresh.local_inputs(self.run, self.packet)
        def encoded(path, logical):
            """Bind one actual fixture file to its original logical remote path."""
            contents = path.read_bytes()
            return {"path": str(logical), "sha256": hashlib.sha256(contents).hexdigest(),
                    "base64": base64.b64encode(contents).decode()}
        packet_files = {name: encoded(self.packet / name, self.fixture.remote_packet / name)
                        for name in inputs["packet_hashes"]}
        if (self.packet / "release_receipt.json").exists():
            packet_files["release_receipt.json"] = encoded(self.packet / "release_receipt.json",
                                                          self.fixture.remote_packet / "release_receipt.json")
        return {"schema_version": 1, "kind": "aces_allocation_refresh_payload", "query_host": "alogin1.cluster",
            "user": "u.sl359075", "job_id": self.fixture.identifier, "remote_campaign_run_dir": str(self.fixture.remote),
            "packet_name": self.packet.name, "packet_files": packet_files,
            "data_files": {path.name: encoded(path, self.fixture.remote / "data" / self.fixture.unit.name / path.name)
                           for path in self.fixture.unit.glob("*.json") if path.name in refresh.DATA_FILES},
            "squeue": copy.deepcopy(self.fixture.snapshot["squeue"]), "sacct": copy.deepcopy(self.fixture.snapshot["sacct"]),
            "observed_at_utc": self.fixture.snapshot["observed_at_utc"]}

    def response(self, payload):
        """Return one remote result without checking a master or using a network."""
        return [{"returncode": 0, "stdout": json.dumps(payload), "stderr": ""}]

    def test_existing_master_only_exact_owned_id_and_unchanged_preparation(self):
        """Refresh uses one owned ACES identifier, even when UVA issued the same number."""
        payload = self.payload()
        before = {name: (self.packet / name).read_bytes() for name in payload["packet_files"]}
        with patch.object(refresh, "command_result", side_effect=self.response(payload)) as command:
            result = refresh.refresh(self.run, self.packet)
        self.assertEqual((result["state"], result["observed_state"]), ("refreshed", "waiting"))
        self.assertEqual(command.call_count, 1)
        argv = command.call_args_list[0].args[0]
        self.assertNotIn("-O", argv)
        self.assertFalse(result["transport"]["master_check"]["performed"])
        self.assertNotIn("master", result["transport"])
        for option in ("ControlMaster=no", "ProxyCommand=false", "BatchMode=yes"):
            self.assertIn(option, argv)
        remote = shlex.split(argv[-1])
        self.assertEqual(remote[:2], [refresh.REMOTE_PYTHON, "-I"])
        request = json.loads(remote[-1])
        self.assertEqual(request["job_id"], self.fixture.identifier)
        self.assertEqual(request["remote_campaign_run_dir"], str(self.fixture.remote))
        self.assertNotIn("scancel", remote[-2])
        self.assertEqual(before, {name: (self.packet / name).read_bytes() for name in before})
        self.assertEqual(refresh.MONITOR.aces_allocation_observation_rows(self.run)[0][0]["state"], "waiting")

    def test_repeated_refreshes_keep_master_checks_in_the_separate_hourly_review(self):
        """Three status refreshes make three forced socket-reuse commands and zero explicit master checks."""
        payload = self.payload()
        with patch.object(refresh, "command_result", side_effect=self.response(payload) * 3) as command:
            results = [refresh.refresh(self.run, self.packet) for _ in range(3)]
        self.assertEqual(command.call_count, 3)
        for call in command.call_args_list:
            self.assertNotIn("-O", call.args[0])
            for option in ("ControlMaster=no", "ProxyCommand=false", "BatchMode=yes"):
                self.assertIn(option, call.args[0])
        self.assertTrue(all(result["transport"]["master_check"]["performed"] is False for result in results))

    def test_foreign_ids_wrong_portal_and_path_fail_before_any_command(self):
        """Local ownership, host and packet bounds fail before any transport command."""
        for kind in ("foreign_id", "wrong_portal", "outside_packet"):
            previous = (self.packet / "remote_monitor_snapshot.json").read_bytes()
            with self.subTest(kind=kind), patch.object(refresh, "command_result") as command:
                if kind == "foreign_id":
                    (self.run / "slurm/submitted_jobids_aces.txt").write_text("999\n")
                    target = self.packet
                elif kind == "wrong_portal":
                    (self.run / "slurm/submitted_jobids_aces.txt").write_text(self.fixture.identifier + "\n")
                    target = self.packet
                else:
                    target = self.run / "outside_packet"
                with patch.object(refresh.socket, "gethostname", return_value="portal06" if kind == "wrong_portal" else "portal05"):
                    with self.assertRaisesRegex(RuntimeError, "error record"):
                        refresh.refresh(self.run, target)
                command.assert_not_called()
                self.assertEqual((self.packet / "remote_monitor_snapshot.json").read_bytes(), previous)

    def test_closed_master_never_attempts_another_command_or_retries(self):
        """An absent control master creates an error record and preserves the prior snapshot."""
        previous = (self.packet / "remote_monitor_snapshot.json").read_bytes()
        with patch.object(refresh, "command_result", return_value={"returncode": 1, "stdout": "", "stderr": "no socket"}) as command:
            with self.assertRaisesRegex(RuntimeError, "prior snapshot retained=True"):
                refresh.refresh(self.run, self.packet)
        self.assertEqual(command.call_count, 1)
        self.assertNotIn("-O", command.call_args.args[0])
        for option in ("ControlMaster=no", "ProxyCommand=false", "BatchMode=yes"):
            self.assertIn(option, command.call_args.args[0])
        self.assertEqual((self.packet / "remote_monitor_snapshot.json").read_bytes(), previous)
        errors = list((self.run / "20_mins_monitoring/aces_refresh_errors").glob("*.json"))
        self.assertEqual(len(errors), 1)
        self.assertTrue(json.loads(errors[0].read_text())["prior_snapshot_retained"])

    def test_actual_parenthesized_held_row_without_release_is_preserved(self):
        """The actual ACES held-row format remains unchanged and permits no release inference."""
        (self.packet / "release_receipt.json").unlink()
        self.fixture.make_snapshot(reason="(JobHeldUser)")
        raw = "2274380|u.sl359075|PENDING||(JobHeldUser)\n"
        self.fixture.snapshot["squeue"]["stdout"] = raw
        self.fixture.save_snapshot()
        payload = self.payload()
        with patch.object(refresh, "command_result", side_effect=self.response(payload)):
            result = refresh.refresh(self.run, self.packet)
        self.assertEqual(result["observed_state"], "held")
        saved = json.loads((self.packet / "remote_monitor_snapshot.json").read_text())
        self.assertEqual(saved["squeue"]["stdout"], raw)
        self.assertIsNone(saved["release_receipt_sha256"])

    def test_completed_payload_transfers_zero_credit_raw_file_chain(self):
        """A remote completion adds actual immutable raw files and never grants admission or research credit."""
        previous = (self.packet / "remote_monitor_snapshot.json").read_bytes()
        self.fixture.collected()
        payload = self.payload()
        for name in ("owned_allocation.json", "hardware_observations.json", "result.json"):
            (self.fixture.unit / name).unlink()
        (self.packet / "remote_monitor_snapshot.json").write_bytes(previous)
        with patch.object(refresh, "command_result", side_effect=self.response(payload)):
            result = refresh.refresh(self.run, self.packet)
        self.assertEqual(result["observed_state"], "completed")
        row = refresh.MONITOR.aces_allocation_observation_rows(self.run)[0][0]
        self.assertEqual(row["display_state"], "COLLECTED")
        self.assertEqual(row["scientific_hypothesis_credit"], 0)
        self.assertFalse(row["workers_permitted"])
        self.assertEqual(json.loads((self.fixture.unit / "result.json").read_text())["observations"],
                         {"path": str(self.fixture.remote / "data" / self.fixture.unit.name / "hardware_observations.json"),
                          "sha256": hashlib.sha256((self.fixture.unit / "hardware_observations.json").read_bytes()).hexdigest()})

    def test_changed_paths_hashes_or_preparation_reject_without_snapshot_replacement(self):
        """A plausible remote response cannot substitute another file, job, path or immutable preparation."""
        original = self.payload()
        previous = (self.packet / "remote_monitor_snapshot.json").read_bytes()
        for kind in ("path", "bytes", "preparation", "job", "extra_file", "query"):
            payload = copy.deepcopy(original)
            if kind == "path": payload["data_files"]["progress.json"]["path"] = "/outside/progress.json"
            elif kind == "bytes": payload["data_files"]["progress.json"]["base64"] = base64.b64encode(b"changed").decode()
            elif kind == "preparation":
                record = payload["packet_files"]["job.sbatch"]
                record.update(base64=base64.b64encode(b"changed").decode(), sha256=hashlib.sha256(b"changed").hexdigest())
            elif kind == "job": payload["job_id"] = "999"
            elif kind == "extra_file": payload["data_files"]["../outside.json"] = payload["data_files"]["progress.json"]
            else: payload["squeue"]["returncode"] = 2
            with self.subTest(kind=kind), patch.object(refresh, "command_result", side_effect=self.response(payload)):
                with self.assertRaises(RuntimeError):
                    refresh.refresh(self.run, self.packet)
                self.assertEqual((self.packet / "remote_monitor_snapshot.json").read_bytes(), previous)

    def test_local_data_symlink_and_concurrent_immutable_change_reject(self):
        """A changed destination or source cannot redirect or replace the validated mirror."""
        payload = self.payload()
        inputs = refresh.local_inputs(self.run, self.packet)
        snapshot, writes = refresh.proposed_snapshot(inputs, payload)
        (self.packet / "job.sbatch").write_text("changed after query\n")
        with self.assertRaisesRegex(ValueError, "changed during refresh"):
            refresh.publish(inputs, snapshot, writes)
        moved = self.run / "other_data"
        self.fixture.unit.rename(moved)
        self.fixture.unit.symlink_to(moved, target_is_directory=True)
        with patch.object(refresh, "command_result") as command, self.assertRaises(RuntimeError):
            refresh.refresh(self.run, self.packet)
        command.assert_not_called()

    def test_publication_failure_restores_prior_progress_and_snapshot(self):
        """A late write failure cannot make the old snapshot point at changed mutable progress."""
        inputs = refresh.local_inputs(self.run, self.packet)
        snapshot, writes = refresh.proposed_snapshot(inputs, self.payload())
        progress = self.fixture.unit / "progress.json"
        previous_progress = progress.read_bytes()
        previous_snapshot = (self.packet / "remote_monitor_snapshot.json").read_bytes()
        writes[progress] = b'{"state":"running"}\n'
        atomic = refresh.atomic_bytes
        failed = False
        def fail_snapshot_once(path, contents):
            """Fail after the new progress has been saved, then permit the required restoration."""
            nonlocal failed
            if Path(path).name == "remote_monitor_snapshot.json" and not failed:
                failed = True
                raise OSError("synthetic snapshot publication failure")
            return atomic(path, contents)
        with patch.object(refresh, "atomic_bytes", side_effect=fail_snapshot_once), self.assertRaises(OSError):
            refresh.publish(inputs, snapshot, writes)
        self.assertEqual(progress.read_bytes(), previous_progress)
        self.assertEqual((self.packet / "remote_monitor_snapshot.json").read_bytes(), previous_snapshot)

    def test_new_immutable_collision_is_never_replaced(self):
        """An immutable file appearing after validation keeps its bytes and the prior snapshot."""
        inputs = refresh.local_inputs(self.run, self.packet)
        snapshot, writes = refresh.proposed_snapshot(inputs, self.payload())
        target = self.fixture.unit / "hardware_observations.json"
        writes[target] = b'{"received":"new"}\n'
        previous = (self.packet / "remote_monitor_snapshot.json").read_bytes()
        original_link = refresh.os.link
        def create_conflict(source, destination):
            """Create conflicting bytes at the exclusive publication boundary."""
            if Path(destination) == target:
                target.write_bytes(b"another writer\n")
            return original_link(source, destination)
        with patch.object(refresh.os, "link", side_effect=create_conflict), self.assertRaisesRegex(ValueError, "different bytes"):
            refresh.publish(inputs, snapshot, writes)
        self.assertEqual(target.read_bytes(), b"another writer\n")
        self.assertEqual((self.packet / "remote_monitor_snapshot.json").read_bytes(), previous)

    def test_wrong_submission_kind_and_lock_symlink_never_use_network(self):
        """Invalid ownership labels and a redirected lock fail before any transport."""
        original = (self.packet / "submission_receipt.json").read_bytes()
        self.fixture.receipt["kind"] = "another_submission"
        (self.packet / "submission_receipt.json").write_text(json.dumps(self.fixture.receipt))
        with patch.object(refresh, "command_result") as command, self.assertRaises(RuntimeError):
            refresh.refresh(self.run, self.packet)
        command.assert_not_called()
        (self.packet / "submission_receipt.json").write_bytes(original)
        target = self.run / "unrelated_lock"
        target.write_text("unchanged\n")
        (self.packet / ".refresh.lock").symlink_to(target)
        with patch.object(refresh, "command_result") as command, self.assertRaises(RuntimeError):
            refresh.refresh(self.run, self.packet)
        command.assert_not_called()
        self.assertEqual(target.read_text(), "unchanged\n")

    def test_new_remote_release_receipt_is_validated_before_immutable_copy(self):
        """A new actual release can be mirrored without changing the older held preparation."""
        self.fixture.make_snapshot(queue_state="RUNNING", accounting_state="RUNNING")
        payload = self.payload()
        release = (self.packet / "release_receipt.json").read_bytes()
        (self.packet / "release_receipt.json").unlink()
        self.fixture.make_snapshot(reason="(JobHeldUser)")
        with patch.object(refresh, "command_result", side_effect=self.response(payload)):
            result = refresh.refresh(self.run, self.packet)
        self.assertEqual(result["observed_state"], "running")
        self.assertEqual((self.packet / "release_receipt.json").read_bytes(), release)


if __name__ == "__main__":
    unittest.main()
