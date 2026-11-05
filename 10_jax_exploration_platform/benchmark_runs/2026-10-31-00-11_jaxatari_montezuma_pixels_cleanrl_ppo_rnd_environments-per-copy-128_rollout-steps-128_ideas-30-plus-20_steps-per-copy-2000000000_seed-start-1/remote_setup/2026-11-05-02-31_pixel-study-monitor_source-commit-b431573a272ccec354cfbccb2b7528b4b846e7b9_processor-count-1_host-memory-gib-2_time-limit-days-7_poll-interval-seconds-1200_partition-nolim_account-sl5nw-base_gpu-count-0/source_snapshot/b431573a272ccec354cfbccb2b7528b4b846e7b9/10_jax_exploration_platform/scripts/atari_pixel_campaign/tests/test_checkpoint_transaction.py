"""Check full checkpoint states after interrupted publication using stdlib fixtures."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_optimized as training
from tests import test_train_optimized as fixtures


class InterruptedCheckpoint(RuntimeError):
    """Represent a stopped process after a real filesystem operation."""


class CheckpointTransactionTests(unittest.TestCase):
    """Recover exact old or new full state and refuse ambiguous or changed records."""

    def setUp(self):
        """Start with one real complete checkpoint using a full admitted identity fixture."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "checkpoints/latest.pkl"
        self.identity = fixtures.example_identity()
        self.report = {"pending_sums": None, "pending_count": 0, "last_report_iteration": 0}
        self.save(3)

    def save(self, iteration):
        """Serialize every field in a distinct complete state, including the true iteration."""
        training.save_checkpoint(self.path, fixtures.example_state(iteration), self.report,
                                 self.identity, iteration, float(iteration))

    def pair_bytes(self):
        """Capture both canonical files for exact no-mutation assertions."""
        return tuple(p.read_bytes() for p in (self.path, self.path.with_name("latest.metadata.json")))

    def interrupt(self, boundary):
        """Raise after the requested real directory fsync, keeping its actual filesystem effects."""
        original, count = training.checkpoint_sync_directory, 0
        def sync(paths):
            """Inject an interruption only after the real durable directory operation."""
            nonlocal count
            original(paths)
            count += 1
            if count == boundary:
                raise InterruptedCheckpoint(str(boundary))
        with mock.patch.object(training, "checkpoint_sync_directory", side_effect=sync):
            with self.assertRaises(InterruptedCheckpoint):
                self.save(4)
        return count

    def reset(self):
        """Rebuild only this isolated fixture after a deliberately invalid transaction."""
        for path in self.path.parent.iterdir():
            path.unlink()
        self.save(3)

    def test_every_publication_and_cleanup_boundary_retains_complete_state(self):
        """Both files and all state fields recover after each durable second-save boundary."""
        # Before: iteration 3 is complete. After interruption: exactly iteration 3 or 4
        # is complete, with its matching metadata, seeds, runtime and unfinished report.
        for boundary in range(1, 10):
            with self.subTest(boundary=boundary):
                self.interrupt(boundary)
                with mock.patch.object(training.pickle, "load", side_effect=AssertionError("metadata recovery unpickled")):
                    recovery = training.recover_checkpoint(self.path, self.identity)
                loaded = training.load_checkpoint(self.path, self.identity)
                expected = 3 if boundary <= 3 else 4
                self.assertEqual(loaded["iteration"], expected)
                self.assertEqual(loaded["state"], fixtures.example_state(expected))
                self.assertEqual(loaded["elapsed_seconds"], float(expected))
                self.assertEqual(loaded["report_state"], self.report)
                self.assertEqual({p.name for p in self.path.parent.iterdir()},
                                 {"latest.pkl", "latest.metadata.json", "checkpoint.lock"})
                self.assertEqual(recovery["recovered"], boundary < 9)
                if expected != 3:
                    self.save(3)

    def test_first_checkpoint_interruption_never_invents_a_committed_state(self):
        """An unprepared first save stays absent; a fully prepared first save finishes exactly."""
        for boundary in range(1, 6):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as temporary:
                original_path, self.path = self.path, Path(temporary) / "latest.pkl"
                try:
                    self.interrupt(boundary)
                    training.recover_checkpoint(self.path, self.identity)
                    if boundary < 3:
                        self.assertFalse(self.path.exists())
                        with self.assertRaisesRegex(ValueError, "missing"):
                            training.load_checkpoint(self.path, self.identity)
                    else:
                        self.assertEqual(training.load_checkpoint(self.path, self.identity)["iteration"], 4)
                finally:
                    self.path = original_path

    def test_partial_payload_sidecar_or_replacement_marker_keeps_previous_pair(self):
        """A failed stage write while preparing discards only uncommitted reserved files."""
        for suffix in ("latest.next.pkl", "latest.next.metadata.json", "second_marker"):
            before, original, marker_count = self.pair_bytes(), training.checkpoint_write, 0
            def write(path, writer):
                """Leave the partial bytes of an interrupted writer at the exact reserved stage."""
                nonlocal marker_count
                if path.name == "latest.transaction.next.json":
                    marker_count += 1
                if path.name == suffix or (suffix == "second_marker" and marker_count == 2):
                    path.write_bytes(b"partial new bytes")
                    raise InterruptedCheckpoint(suffix)
                return original(path, writer)
            with self.subTest(suffix=suffix), mock.patch.object(training, "checkpoint_write", side_effect=write):
                with self.assertRaises(InterruptedCheckpoint):
                    self.save(4)
            training.recover_checkpoint(self.path, self.identity)
            self.assertEqual(self.pair_bytes(), before)

    def test_interrupted_first_marker_recovers_intact_previous_pair_without_trusting_partial_json(self):
        """Only the known temporary marker can be discarded when the prior pair still proves complete."""
        before = self.pair_bytes()
        (self.path.parent / "latest.transaction.next.json").write_bytes(b'{"incomplete":')
        self.assertEqual(training.recover_checkpoint(self.path, self.identity)["action"], "discarded_uncommitted_marker")
        self.assertEqual(self.pair_bytes(), before)
        self.assertEqual(training.load_checkpoint(self.path, self.identity)["iteration"], 3)

    def test_legacy_valid_pair_is_unchanged_and_unmarked_mismatch_is_held(self):
        """Schema-one pairs need no rewrite; an unmarked mismatch is never guessed or repaired."""
        before = self.pair_bytes()
        self.assertEqual(training.recover_checkpoint(self.path, self.identity), {"recovered": False})
        self.assertEqual(self.pair_bytes(), before)
        self.path.write_bytes(b"unrecorded replacement")
        changed = self.pair_bytes()
        with self.assertRaisesRegex(ValueError, "hash differs"):
            training.recover_checkpoint(self.path, self.identity)
        self.assertEqual(self.pair_bytes(), changed)

    def test_changed_source_config_runtime_seeds_helpers_and_backend_hold_without_mutation(self):
        """A durable marker cannot authorize changing any part of the current admitted identity."""
        self.interrupt(5)
        snapshot = {p.name: p.read_bytes() for p in self.path.parent.iterdir()}
        changes = [("source_commit", "c" * 40), ("copy_seeds", [3, 4]),
                   ("configuration", {"different": "identity"}), ("runtime_environment", {"JAX_ENABLE_X64": "false"}),
                   ("helpers", {"commit": "d" * 40}), ("backend", {"jax": "changed"})]
        for field, value in changes:
            changed = copy.deepcopy(self.identity)
            changed[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity"):
                training.recover_checkpoint(self.path, changed)
            self.assertEqual({p.name: p.read_bytes() for p in self.path.parent.iterdir()}, snapshot)
        training.recover_checkpoint(self.path, self.identity)
        self.assertEqual(training.load_checkpoint(self.path, self.identity)["iteration"], 4)

    def test_missing_or_unknown_prepared_files_hold_before_changing_either_canonical_file(self):
        """Both next roles must validate before either file is moved into its canonical path."""
        for missing in ("latest.next.pkl", "latest.next.metadata.json"):
            with self.subTest(missing=missing):
                self.interrupt(4)
                (self.path.parent / missing).unlink()
                before = self.pair_bytes()
                with self.assertRaisesRegex(ValueError, "no complete recorded next"):
                    training.recover_checkpoint(self.path, self.identity)
                self.assertEqual(self.pair_bytes(), before)
                self.reset()
        self.interrupt(4)
        (self.path.parent / "latest.next.pkl").write_bytes(b"unknown generation")
        before = self.pair_bytes()
        with self.assertRaisesRegex(ValueError, "differs from its recorded generation"):
            training.recover_checkpoint(self.path, self.identity)
        self.assertEqual(self.pair_bytes(), before)

    def test_forged_sidecar_description_does_not_authorize_file_replacement(self):
        """Real file hashes alone do not excuse a marker describing different metadata values."""
        self.interrupt(4)
        marker_path = self.path.parent / "latest.transaction.json"
        marker = json.loads(marker_path.read_text())
        marker["next"]["metadata"]["elapsed_seconds"] = 123
        marker_path.write_text(json.dumps(marker))
        before = self.pair_bytes()
        with self.assertRaisesRegex(ValueError, "recorded metadata"):
            training.recover_checkpoint(self.path, self.identity)
        self.assertEqual(self.pair_bytes(), before)

    def test_symlink_ancestor_reserved_symlink_and_nonregular_file_are_rejected(self):
        """No checkpoint path can redirect recovery outside its real reserved directory."""
        outside = Path(self.temporary.name) / "outside"
        outside.write_text("unchanged")
        reserved = self.path.parent / "latest.transaction.json"
        reserved.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            training.recover_checkpoint(self.path, self.identity)
        reserved.unlink()
        reserved.mkdir()
        with self.assertRaisesRegex(ValueError, "regular"):
            training.recover_checkpoint(self.path, self.identity)
        reserved.rmdir()
        alias = Path(self.temporary.name) / "alias"
        alias.symlink_to(self.path.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink ancestors"):
            training.recover_checkpoint(alias / "latest.pkl", self.identity)
        self.assertEqual(outside.read_text(), "unchanged")

    def test_save_load_and_metadata_recovery_share_one_nonblocking_lock(self):
        """All three operations reject an active lock without changing the previous checkpoint."""
        before = self.pair_bytes()
        with training.checkpoint_lock(self.path):
            for action in (lambda: self.save(4), lambda: training.load_checkpoint(self.path, self.identity),
                           lambda: training.recover_checkpoint(self.path, self.identity)):
                with self.assertRaises(BlockingIOError):
                    action()
        self.assertEqual(self.pair_bytes(), before)

    def test_unknown_unmarked_stage_or_malformed_marker_is_held(self):
        """Unknown records never count as a valid legacy pair or permission to repair one."""
        before = self.pair_bytes()
        reserved = self.path.parent / "latest.next.pkl"
        reserved.write_bytes(b"unknown")
        with self.assertRaisesRegex(ValueError, "without a durable marker"):
            training.recover_checkpoint(self.path, self.identity)
        reserved.unlink()
        marker_path = self.path.parent / "latest.transaction.json"
        for value in (None, [], {}, {"schema_version": True}):
            marker_path.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                training.recover_checkpoint(self.path, self.identity)
            self.assertEqual(self.pair_bytes(), before)


if __name__ == "__main__":
    unittest.main()
