"""Check existing-master-only ACES site refresh using mocked transport and exact metadata bytes."""

import base64
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refresh_aces_site as refresh
import test_aces_site_monitor as fixtures


class SiteRefreshTests(unittest.TestCase):
    """Protect own-cluster IDs, immutable artifacts and the last successful status on every failure."""

    def setUp(self):
        """Reuse native-shaped local metadata while forbidding all actual transport commands."""
        self.fixture = fixtures.SiteMonitorTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.host = patch.object(refresh.socket, "gethostname", return_value="portal05").start()
        patch.object(refresh.pwd, "getpwuid", return_value=SimpleNamespace(pw_name="sl5nw")).start()
        patch.object(refresh.transport, "command_result", side_effect=AssertionError("live transport forbidden")).start()
        self.addCleanup(patch.stopall)
        self.inputs = refresh.local_inputs(self.fixture.run, self.fixture.packet)

    def item(self, value, path):
        """Bind synthetic bytes to their literal native path and complete checksum."""
        contents = value if isinstance(value, bytes) else (json.dumps(value) + "\n").encode()
        return {"path": str(path), "sha256": refresh.adapter.digest(contents), "base64": base64.b64encode(contents).decode()}

    def payload(self):
        """Build the bounded remote response from the fixture's original exact path identities."""
        f = self.fixture
        return {"schema_version": 1, "kind": "aces_full_site_refresh_payload", "query_host": "alogin1.cluster",
            "user": "u.sl359075", "job_id": f.identifier, "remote_campaign_run_dir": str(f.remote), "packet_name": f.packet.name,
            "packet_files": {name: self.item((f.packet / name).read_bytes(), f.remote_packet / name) for name in self.inputs["packet_hashes"]},
            "data_files": {path.name: self.item(path.read_bytes(), f.remote / "data" / f.unit.name / path.name) for path in f.unit.glob("*.json")},
            "observed_at_utc": f.document["observed_at_utc"], "squeue": f.document["squeue"], "sacct": f.document["sacct"]}

    def test_transport_uses_only_existing_master_and_never_separately_checks_it(self):
        """Each refresh uses one command with a false new-connection proxy and no authentication retry."""
        payload = self.payload()
        result = {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""}
        with patch.object(refresh.transport, "command_result", return_value=result) as call:
            received, details = refresh.receive(self.inputs)
        command = call.call_args.args[0]
        self.assertEqual(received, payload)
        self.assertEqual(call.call_count, 1)
        self.assertNotIn("-O", command)
        for option in ("ControlMaster=no", "ProxyCommand=false", "BatchMode=yes", "ConnectTimeout=10"):
            self.assertIn(option, command)
        self.assertIs(details["master_check"]["performed"], False)
        self.assertIn("hourly resource review", details["master_check"]["reason"])

    def test_missing_master_and_oversized_response_do_not_create_an_authentication_retry(self):
        """An absent socket or an excessive response stops after one attempted existing-master command."""
        for result in ({"returncode": 255, "stdout": "", "stderr": "socket absent"},
                       {"returncode": 0, "stdout": "x" * 65, "stderr": ""}):
            with patch.object(refresh.transport, "MAX_RESPONSE_BYTES", 64), \
                    patch.object(refresh.transport, "command_result", return_value=result) as call:
                with self.assertRaisesRegex(ValueError, "existing-master"):
                    refresh.receive(self.inputs)
            self.assertEqual(call.call_count, 1)

    def test_foreign_local_id_is_rejected_before_any_transport(self):
        """An equal numeric UVA entry cannot supply missing ACES ownership."""
        f = self.fixture
        (f.run / "slurm/submitted_jobids_aces.txt").write_text("")
        with patch.object(refresh, "receive") as call:
            with self.assertRaisesRegex(RuntimeError, "refresh failed"):
                refresh.refresh(f.run, f.packet)
        call.assert_not_called()

    def test_complete_response_is_validated_before_atomic_snapshot_publication(self):
        """Original raw bytes are copied unchanged and the visible snapshot is published last."""
        f, payload = self.fixture, self.payload()
        progress = {"state": "running", "stage": "learner"}
        payload["data_files"]["progress.json"] = self.item(progress, f.remote / "data" / f.unit.name / "progress.json")
        with patch.object(refresh, "receive", return_value=(payload, {"mocked": True})):
            result = refresh.refresh(f.run, f.packet)
        self.assertEqual(result["observed_state"], "waiting")
        self.assertEqual(json.loads((f.unit / "progress.json").read_text()), progress)
        self.assertEqual(refresh.adapter.rows(f.run)[0][0]["state"], "waiting")

    def test_bad_partial_foreign_or_driver_payload_preserves_previous_snapshot(self):
        """Missing immutable files, wrong ownership and captured binaries never enter the small status mirror."""
        f = self.fixture
        before = (f.packet / "remote_monitor_snapshot.json").read_bytes()
        for change in ("partial", "foreign", "driver", "stale"):
            payload = self.payload()
            if change == "partial":
                payload["packet_files"].pop("site_validation_plan.json")
            elif change == "foreign":
                payload["job_id"] = "999"
            elif change == "driver":
                payload["data_files"]["probe_driver/captured_driver/libcuda.so.580.178.04"] = self.item(b"binary", "/outside")
            else:
                payload["observed_at_utc"] = "2020-01-01T00:00:00+00:00"
            with self.subTest(change=change), patch.object(refresh, "receive", return_value=(payload, {})):
                with self.assertRaises(RuntimeError):
                    refresh.refresh(f.run, f.packet)
            self.assertEqual((f.packet / "remote_monitor_snapshot.json").read_bytes(), before)

    def test_saved_failure_is_mirrored_exactly_and_remains_a_failure(self):
        """A valid failure response refreshes diagnostics without granting a success state."""
        f = self.fixture
        failure = {"kind": "aces_full_site_failure", "passed": False, "error": "original stage failed", **refresh.adapter.BOUNDARY}
        payload = self.payload()
        payload["data_files"]["failure.json"] = self.item(failure, f.remote / "data" / f.unit.name / "failure.json")
        payload["squeue"]["stdout"] = ""
        payload["sacct"]["stdout"] = f.identifier + "|u.sl359075|FAILED|1:0|ac041\n"
        with patch.object(refresh, "receive", return_value=(payload, {})):
            result = refresh.refresh(f.run, f.packet)
        self.assertEqual(result["observed_state"], "failed")
        self.assertEqual((f.unit / "failure.json").read_bytes(), base64.b64decode(payload["data_files"]["failure.json"]["base64"]))

    def test_concurrent_immutable_collision_preserves_foreign_bytes_and_old_snapshot(self):
        """A file appearing at publication is never overwritten by the refresh."""
        f, payload = self.fixture, self.payload()
        target = f.unit / "parent_owner.json"
        payload["data_files"][target.name] = self.item({"pid": 123}, f.remote / "data" / f.unit.name / target.name)
        before = (f.packet / "remote_monitor_snapshot.json").read_bytes()
        original = refresh.transport.os.link
        def collide(source, destination):
            """Represent another writer creating conflicting immutable bytes before exclusive publication."""
            if Path(destination) == target and not target.exists():
                target.write_bytes(b"other writer bytes")
            return original(source, destination)
        with patch.object(refresh, "receive", return_value=(payload, {})), patch.object(refresh.transport.os, "link", side_effect=collide):
            with self.assertRaises(RuntimeError):
                refresh.refresh(f.run, f.packet)
        self.assertEqual(target.read_bytes(), b"other writer bytes")
        self.assertEqual((f.packet / "remote_monitor_snapshot.json").read_bytes(), before)

    def test_remote_metadata_precondition_rejects_a_numerical_import_before_any_request(self):
        """The fresh remote metadata program stops before path, ledger or scheduler work if a model was imported."""
        prefix = refresh.REMOTE_PROGRAM.split("request=json.loads", 1)[0]
        compile(refresh.REMOTE_PROGRAM, "remote_site_metadata", "exec")
        with patch.dict(sys.modules, {"jax": SimpleNamespace()}):
            with self.assertRaisesRegex(RuntimeError, "numerical runtime"):
                exec(prefix, {})


if __name__ == "__main__":
    unittest.main()
