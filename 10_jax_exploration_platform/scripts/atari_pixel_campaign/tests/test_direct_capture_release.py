"""Use synthetic process metadata to test exact direct release and pre-work joins."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import direct_capture_release as release
from test_nsight_process_identity import M, ProcessObservationTests


class DirectCaptureReleaseTests(unittest.TestCase):
    """Retain and tamper only labelled temporary metadata; perform no process or device work."""
    def setUp(self):
        """Build one private positive-ancestry fixture and its exact immutable unit graph."""
        self.fixture = ProcessObservationTests(); self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.unit = self.fixture.root / "data/unit"; (self.unit / "code").mkdir(parents=True)
        source = self.unit / "code/diagnose_direct_nsight_process.py"
        source.write_text("# synthetic metadata source; never executed\n")
        self.plan = {"kind": release.DIAGNOSTIC, "hardware_class": "h100", "gpu_uuid": release.GPU,
            "capture_process_protocol": release.PROTOCOL, "campaign_run_dir": str(self.fixture.root), "unit_id": "unit",
            "helpers": {"files": {release.PREFIX + source.name: M.receipt(source)["sha256"]}}}
        M.publish(self.unit / "measurement_plan.json", self.plan)
        command = ["/synthetic/python", str(source), "--direct-owned-capture-unit", str(self.unit)]
        (self.fixture.root / "103/cmdline").write_bytes(b"\0".join(part.encode() for part in command) + b"\0")
        self.producer = self.fixture.parent
        self.capture = self.fixture.reader.snapshot(103, 888)
        observer = M.OwnedProcesses(101, "20", self.producer, self.fixture.reader)
        self.addCleanup(observer.close)
        self.observation = json.loads(json.dumps(observer.observe(103, "40", command)))
        self.observation["observed_at_seconds"] = 3.
        self.lock = {"path": "/tmp/synthetic-only-not-opened", "fd": 7, "fd_stat": {"synthetic": True},
            "path_stat": {"synthetic": True}, "path_symlink": False, "fd_target": "/tmp/synthetic-only-not-opened",
            "fdinfo": "synthetic fixture bytes; never a GPU descriptor", "fdinfo_sha256": "a" * 64,
            "local_to_producer": True, "read_by_pid": 100, "producer": self.producer}
        self.intent = {"plan": M.receipt(self.unit / "measurement_plan.json"), "source": M.receipt(source),
            "challenge": "b" * 64, "producer": self.producer, "capture_command": command,
            "producer_lock_before": self.lock, "created_at_seconds": 1.}
        M.publish(self.unit / "direct_profile_process_intent.json", self.intent)
        self.claim = {"plan": self.intent["plan"], "source": self.intent["source"], "challenge": self.intent["challenge"],
            "intent": M.receipt(self.unit / "direct_profile_process_intent.json"), "pid": 103, "process_start_time": "40",
            "process": self.capture, "claimed_at_seconds": 2.}
        M.publish(self.unit / "capture_identity.json", self.claim)
        self.observation_ref = M.publish(self.unit / "capture_process_observation.json", self.observation)
        self.lock_ref = M.publish(self.unit / "producer_lock_before_release.json", {**self.lock, "observed_at_seconds": 4.})
        self.work = {"pid": 103, "process_start_time": "40", "started_at_seconds": 7., "finished_at_seconds": 8.}

    def publish(self):
        """Exercise the real release writer under explicitly synthetic producer identity and clock."""
        with patch.object(release.os, "getpid", return_value=100), patch.object(release.time, "time", return_value=5.):
            return release.publish_release(self.unit, self.plan, self.intent, self.claim, self.observation_ref, self.lock_ref, M)

    def acknowledge(self):
        """Exercise the actual byte-retaining acknowledgment with a synthetic current child snapshot."""
        with patch.object(release.os, "getpid", return_value=103), patch.object(release.time, "time", return_value=6.):
            return release.acknowledge_release(self.unit, self.plan, self.intent, self.claim, self.capture, M)

    def completed(self):
        """Create the full pre-work graph before a saved-reader refusal test."""
        self.publish(); self.acknowledge()
        return release.validate_saved_release(self.unit, self.plan, self.intent, self.claim, self.observation, M, self.work)

    def change(self, name, key, value):
        """Replace only a deliberate synthetic artifact, preserving production publication semantics separately."""
        path = self.unit / name
        document = json.loads(path.read_bytes()); document[key] = value
        path.write_bytes(release.encoded(document))

    def test_complete_release_and_ack_bind_exact_actual_bytes_and_prework_time(self):
        """The accepted record identifies all three retained artifacts and the actual child time."""
        proof = self.completed()
        self.assertTrue(proof["prework_chronology_verified"])
        self.assertFalse(proof["admission_granted"])
        self.assertEqual(proof["capture_release_observed_at_seconds"], 6.)
        self.assertEqual((self.unit / "capture_release.json").read_bytes(), (self.unit / "capture_release_observed.raw").read_bytes())
        self.assertEqual(proof["capture_release"], M.receipt(self.unit / "capture_release.json"))

    def test_ack_reads_the_actual_release_file_once_and_keeps_the_same_raw_bytes(self):
        """Exact observed bytes cannot be replaced by a second read or JSON serialization."""
        self.publish()
        calls = []
        original = release.read_bytes
        def observed(path):
            """Count the specific release read without changing its bytes."""
            calls.append(Path(path))
            return original(path)
        with patch.object(release, "read_bytes", side_effect=observed):
            self.acknowledge()
        self.assertEqual(calls.count(self.unit / "capture_release.json"), 1)
        self.assertEqual((self.unit / "capture_release_observed.raw").read_bytes(), original(self.unit / "capture_release.json"))

    def test_saved_reader_requires_release_ack_and_raw_files_independently(self):
        """Missing pre-work artifacts cannot be inferred from successful later waits."""
        self.completed()
        for name in ("capture_release.json", "capture_release_observed.json", "capture_release_observed.raw"):
            path = self.unit / name; raw = path.read_bytes(); path.unlink()
            with self.subTest(name=name), self.assertRaises((FileNotFoundError, ValueError)):
                release.validate_saved_release(self.unit, self.plan, self.intent, self.claim, self.observation, M, self.work)
            path.write_bytes(raw)

    def test_changed_release_bytes_refuse_even_when_json_values_are_unchanged(self):
        """Whitespace changes after child observation change the actual retained release hash."""
        self.completed()
        path = self.unit / "capture_release.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "actual release"):
            release.validate_saved_release(self.unit, self.plan, self.intent, self.claim, self.observation, M, self.work)

    def test_changed_retained_raw_bytes_are_rejected(self):
        """The raw acknowledgment file must match the actual release byte for byte."""
        self.completed()
        path = self.unit / "capture_release_observed.raw"; path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "retained release bytes"):
            release.validate_saved_release(self.unit, self.plan, self.intent, self.claim, self.observation, M, self.work)

    def test_cross_attempt_release_and_substituted_source_refs_are_rejected(self):
        """A different attempt or copied source cannot stand in for the actual release."""
        self.completed()
        for key, value in (("challenge", "f" * 64), ("source", {**self.intent["source"], "sha256": "f" * 64}),
            ("capture_process", {**M.stable(self.capture), "process_start_time": "999"})):
            path = self.unit / "capture_release.json"; raw = path.read_bytes(); self.change(path.name, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                release.validate_saved_release(self.unit, self.plan, self.intent, self.claim, self.observation, M, self.work)
            path.write_bytes(raw)

    def test_ack_wrong_child_source_plan_and_prework_flags_are_rejected(self):
        """Every child acknowledgment preserves exact actor and source joins."""
        self.completed()
        changes = (("plan", {**self.intent["plan"], "sha256": "f" * 64}), ("producer", M.stable(self.capture)),
            ("capture_process", M.stable(self.producer)), ("numerical_modules", ["jax"]),
            ("environment_work_started", True), ("release_size_bytes", True), ("schema_version", True))
        for key, value in changes:
            path = self.unit / "capture_release_observed.json"; raw = path.read_bytes(); self.change(path.name, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                release.validate_saved_release(self.unit, self.plan, self.intent, self.claim, self.observation, M, self.work)
            path.write_bytes(raw)

    def test_finite_typed_chronology_covers_release_ack_and_actual_payload_window(self):
        """Boolean times, backdated events and work before acknowledgment never pass."""
        self.completed()
        for name, key, value in (("capture_release.json", "released_at_seconds", 2.5),
            ("capture_release_observed.json", "observed_at_seconds", 4.5),
            ("capture_release_observed.json", "observed_at_seconds", True)):
            path = self.unit / name; raw = path.read_bytes(); self.change(name, key, value)
            with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                release.validate_saved_release(self.unit, self.plan, self.intent, self.claim, self.observation, M, self.work)
            path.write_bytes(raw)
        for changed in ({**self.work, "started_at_seconds": 5.5}, {**self.work, "finished_at_seconds": 6.5},
            {**self.work, "pid": True}, {**self.work, "process_start_time": "999"}, {**self.work, "started_at_seconds": float("nan")}):
            with self.subTest(work=changed), self.assertRaises(ValueError):
                release.validate_saved_release(self.unit, self.plan, self.intent, self.claim, self.observation, M, changed)

    def test_release_requires_preceding_full_process_and_local_producer_custody(self):
        """No stale observation or readonly child's descriptor may authorize the release."""
        for key, value in (("local_to_producer", False), ("read_by_pid", 103), ("observed_at_seconds", 2.5),
            ("fdinfo", "changed synthetic descriptor")):
            path = self.unit / "producer_lock_before_release.json"; raw = path.read_bytes(); self.change(path.name, key, value)
            changed_ref = M.receipt(path)
            with patch.object(release.os, "getpid", return_value=100), patch.object(release.time, "time", return_value=5.):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    release.publish_release(self.unit, self.plan, self.intent, self.claim, self.observation_ref, changed_ref, M)
            path.write_bytes(raw)

    def test_producer_and_current_child_only_publish_their_respective_records(self):
        """Metadata controllers and unrelated child snapshots cannot create a valid acknowledgment."""
        with patch.object(release.os, "getpid", return_value=103), patch.object(release.time, "time", return_value=5.):
            with self.assertRaisesRegex(ValueError, "sole producer"):
                release.publish_release(self.unit, self.plan, self.intent, self.claim, self.observation_ref, self.lock_ref, M)
        self.publish()
        with patch.object(release.os, "getpid", return_value=100), self.assertRaisesRegex(ValueError, "actual child"):
            release.acknowledge_release(self.unit, self.plan, self.intent, self.claim, self.capture, M)
        with patch.object(release.os, "getpid", return_value=103), patch.object(release.time, "time", return_value=6.):
            with self.assertRaises(ValueError):release.acknowledge_release(self.unit, self.plan, self.intent, self.claim, self.producer, M)

    def test_reserved_refs_are_checked_before_attempting_any_other_result_read(self):
        """A mistaken release reference to result.json is rejected before it is opened."""
        self.publish()
        forbidden = self.unit / "result.json"
        original = release.read_bytes
        calls = []
        def observed(path):
            """Reject any attempt to open the deliberately absent scientific result."""
            calls.append(Path(path)); self.assertNotEqual(Path(path), forbidden)
            return original(path)
        self.change("capture_release.json", "observation", {"path": str(forbidden), "sha256": "f" * 64})
        with patch.object(release.os, "getpid", return_value=103), patch.object(release, "read_bytes", side_effect=observed):
            with self.assertRaises(ValueError):release.acknowledge_release(self.unit, self.plan, self.intent, self.claim, self.capture, M)
        self.assertNotIn(forbidden, calls)

    def test_prior_complete_or_partial_publication_is_preserved(self):
        """New evidence cannot overwrite a prior complete or partial attempt."""
        self.publish()
        raw = (self.unit / "capture_release.json").read_bytes()
        with self.assertRaises(ValueError):self.publish()
        self.assertEqual((self.unit / "capture_release.json").read_bytes(), raw)
        pending = self.unit / ".capture_release_observed.raw.pending"; pending.write_bytes(b"partial synthetic fixture")
        with self.assertRaises(ValueError):self.acknowledge()
        self.assertEqual(pending.read_bytes(), b"partial synthetic fixture")

    def test_duplicate_nonfinite_alias_and_oversized_metadata_are_rejected(self):
        """No ambiguous JSON or substituted file kind can become release evidence."""
        for raw in (b'{"value":1,"value":2}', b'{"value":NaN}', b'{"value":1e309}'):
            with self.assertRaises(ValueError):release.document(raw)
        alias = self.unit / "alias"; alias.symlink_to(self.unit / "measurement_plan.json")
        with self.assertRaises(ValueError):release.read_bytes(alias)
        oversized = self.unit / "oversized_metadata"
        with oversized.open("wb") as stream:
            stream.truncate(release.MAX_METADATA_BYTES + 1)
        with self.assertRaises(ValueError):release.read_bytes(oversized)


if __name__ == "__main__":
    unittest.main()
