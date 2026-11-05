"""Exercise only the four independently rejected boundaries and genuine small-record joins."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("corrected_snapshot_under_test", SCRIPTS / "accepted_evidence_snapshot.py")
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)
ROOT = SCRIPTS.parents[2]


class CorrectedBoundaryTests(unittest.TestCase):
    """Use small ordinary fixtures and refuse restoration after any failed metadata-cache check."""

    def test_metadata_reuse_failure_remains_failed_after_path_restoration(self):
        for method in ("begin_metadata_reuse", "check_entry"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); parent = root / "original"; parent.mkdir()
                path = parent / "small.json"; path.write_bytes(b"{}\n")
                with snapshot.ValidationFiles(metadata_only=True) as files:
                    entry = files.file(path, snapshot.digest(b"{}\n"), keep_bytes=True)
                    files.finish([str(path)])
                    moved = root / "moved"; parent.rename(moved); parent.symlink_to(moved, target_is_directory=True)
                    with self.assertRaises(ValueError):
                        files.begin_metadata_reuse() if method == "begin_metadata_reuse" else files.check_entry(entry)
                    self.assertTrue(files.failed)
                    parent.unlink(); moved.rename(parent)
                    with self.assertRaises(ValueError):
                        files.begin_metadata_reuse()
                    with self.assertRaises(ValueError):
                        files.finish([str(path)])

    def test_fifo_is_refused_before_open_and_post_open_kind_is_checked_before_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fifo = root / "fifo"; os.mkfifo(fifo)
            with snapshot.ValidationFiles() as files, patch.object(snapshot.os, "open", side_effect=AssertionError("FIFO open forbidden")) as opened:
                with self.assertRaises(ValueError):
                    files.file(fifo, "0" * 64)
                opened.assert_not_called()
            ordinary = root / "ordinary"; ordinary.write_bytes(b"x")
            with snapshot.ValidationFiles() as files, patch.object(snapshot.os, "open", return_value=999) as opened, \
                    patch.object(snapshot.os, "fstat", return_value=fifo.lstat()), \
                    patch.object(snapshot.os, "close") as closed, \
                    patch.object(snapshot.os, "read", side_effect=AssertionError("special-file read forbidden")) as read:
                with self.assertRaises(ValueError):
                    files.file(ordinary, snapshot.digest(b"x"))
                self.assertTrue(opened.call_args.args[1] & os.O_NOFOLLOW)
                self.assertTrue(opened.call_args.args[1] & os.O_NONBLOCK)
                read.assert_not_called(); closed.assert_called_once_with(999)

    def test_unmapped_historical_driver_has_no_original_path_fallback(self):
        with snapshot.ValidationFiles() as files:
            inventory = snapshot.AcceptedInventory(files)
            inventory.add({"path": "/usr/lib/x86_64-linux-gnu/libcuda.so.590.48.01", "sha256": "0" * 64}, metadata=False)
            with patch.object(snapshot, "ordinary", side_effect=AssertionError("no original node-local access")) as ordinary:
                with self.assertRaises(ValueError):
                    inventory.resolve_blobs()
                ordinary.assert_not_called()
            with self.assertRaises(ValueError):
                inventory.add({"path": "/usr/lib/x86_64-linux-gnu/libcuda.so.590.48.01", "sha256": "0" * 64},
                    metadata=False, shared_metadata_role="source_receipt_python")
        with tempfile.TemporaryDirectory() as temporary, snapshot.ValidationFiles() as files:
            path = Path(temporary) / "driver.raw"; path.write_bytes(b"previously accepted captured driver")
            inventory = snapshot.AcceptedInventory(files)
            inventory.references({"original": {"path": "/unavailable/node/local/driver.so", "sha256": snapshot.digest(path.read_bytes())},
                "captured": {"path": str(path), "sha256": snapshot.digest(path.read_bytes())}})
            inventory.resolve_blobs()
            self.assertEqual(set(inventory.declared), {str(path)})

    def test_actual_correctness_asset_schema_and_negative_asset_receipt(self):
        summary = json.loads((ROOT / ".review_metadata/accepted_evidence_snapshot_independent_review/independent_primary_review_summary.json").read_bytes())
        ordered = {row["role"]: row for row in summary["real_small_schema_inspection"]}
        with snapshot.ValidationFiles(150) as files:
            inventory = snapshot.AcceptedInventory(files)
            qualification = snapshot.accepted_role(files, inventory, ordered["qualification"]["review"], "qualification")
            qualification_review = files.json(ordered["qualification"]["review"])
            counts = []
            for role in ("1", "2"):
                result = files.json(ordered[role]["result"])
                self.assertNotIn("asset_record", result)
                accepted = snapshot.accepted_role(files, inventory, ordered[role]["review"], "correctness",
                    qualification_review, qualification["qualified_asset_receipt"])
                self.assertEqual(accepted["asset_record"], files.json(accepted["asset_input"]))
                counts.append(accepted["configuration"]["learner"]["n_copies"])
            self.assertEqual(counts, [1, 2])
            original_json = files.json
            def changed_asset(reference):
                value = original_json(reference)
                if Path(reference["path"]).name == "asset_receipt.json":
                    value = deepcopy(value); value["root"] = "/wrong/assets"
                return value
            with patch.object(files, "json", side_effect=changed_asset), self.assertRaises(ValueError):
                snapshot.accepted_role(files, inventory, ordered["1"]["review"], "correctness",
                    qualification_review, qualification["qualified_asset_receipt"])


if __name__ == "__main__":
    unittest.main()
