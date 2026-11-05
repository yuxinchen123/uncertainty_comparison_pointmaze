"""Check the declared controller size against new, cached, empty and aliased pinned files."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("controller_size_snapshot", SCRIPTS / "accepted_evidence_snapshot.py")
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


class ControllerSizeTests(unittest.TestCase):
    """A failed declared size must permanently invalidate the same pinned-file cache."""

    def test_declared_size_contract(self):
        for mode in ("new_file", "cached_metadata", "physical_alias", "empty_file", "two_controller_fields"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); path = root / "direct_controller_stdout.raw"
                raw = b"" if mode == "empty_file" else b"abcdef"; path.write_bytes(raw)
                expected = snapshot.digest(raw)
                with snapshot.ValidationFiles(metadata_only=mode == "cached_metadata") as files:
                    if mode == "empty_file":
                        entry = files.file(path, expected, keep_bytes=True, expected_size_bytes=0)
                        self.assertEqual(entry["bytes"], 0); files.finish([str(path)])
                        continue
                    if mode in {"cached_metadata", "physical_alias"}:
                        files.file(path, expected, keep_bytes=mode == "cached_metadata", expected_size_bytes=6)
                    if mode == "cached_metadata":
                        files.finish([str(path)]); files.begin_metadata_reuse()
                    if mode == "physical_alias":
                        alias = root / "direct_controller_stderr.raw"; os.link(path, alias)
                        # Link creation changes nlink/ctime, so pin both aliases only after creation.
                        files.close()
                        files = snapshot.ValidationFiles()
                        self.addCleanup(files.close)
                        files.file(path, expected, expected_size_bytes=6)
                        path = alias
                    if mode == "two_controller_fields":
                        alias = root / "direct_controller_stderr.raw"; os.link(path, alias)
                        outputs = {"schema_version": 1, "kind": "actual_direct_child_output_hashes_after_wait",
                            "stdout": {"path": str(path), "sha256": expected, "size_bytes": 6},
                            "stderr": {"path": str(alias), "sha256": expected, "size_bytes": 7}}
                        authority = root / "direct_controller_output_hashes.json"; authority.write_bytes(snapshot.encoded(outputs))
                        inventory = snapshot.AcceptedInventory(files); inventory.accepted_units[root] = {"plan": "fixture"}
                        inventory.add({"path": str(authority), "sha256": snapshot.digest(authority.read_bytes())})
                        with self.assertRaises(ValueError):
                            inventory.hash_complete()
                    else:
                        with self.assertRaises(ValueError):
                            files.file(path, expected, keep_bytes=mode == "cached_metadata", expected_size_bytes=7)
                    self.assertTrue(files.failed)
                    if mode == "cached_metadata":
                        with self.assertRaises(ValueError):
                            files.begin_metadata_reuse()
                    with self.assertRaises(ValueError):
                        files.file(path, expected, keep_bytes=mode == "cached_metadata", expected_size_bytes=6)
                    with self.assertRaises(ValueError):
                        files.finish(files.paths)


if __name__ == "__main__":
    unittest.main()
