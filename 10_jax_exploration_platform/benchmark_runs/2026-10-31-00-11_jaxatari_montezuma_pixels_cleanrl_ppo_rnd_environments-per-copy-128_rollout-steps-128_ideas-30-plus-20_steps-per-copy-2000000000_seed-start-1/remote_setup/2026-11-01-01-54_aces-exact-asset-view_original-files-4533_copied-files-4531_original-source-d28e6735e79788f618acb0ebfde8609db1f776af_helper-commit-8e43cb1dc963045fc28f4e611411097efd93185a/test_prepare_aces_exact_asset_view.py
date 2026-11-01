"""Check exact asset copies and metadata-only repair without SSH or numerical imports."""

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("tested_exact_asset_repair", ROOT / "prepare_aces_exact_asset_view.py")
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)
BASE_SOURCE = (ROOT / "stage_aces_full_site_inputs.py").read_text()
PLAN = ROOT / "aces_exact_asset_view_repair_plan.json"


class AssetViewTests(unittest.TestCase):
    """Keep original files, failures and existing checkouts as read-only fixture inputs."""

    def setUp(self):
        """Load the frozen utility source and create a private synthetic asset tree."""
        self.base = repair.base_module(BASE_SOURCE)
        self.plan = json.loads(PLAN.read_bytes())
        self.original = json.loads((ROOT / "aces_full_site_native_staging_plan.json").read_bytes())
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.assets = self.directory / "assets"
        self.assets.mkdir()
        (self.assets / "jaxtari").mkdir()
        (self.assets / "jaxtari/one.bin").write_bytes(b"synthetic original pixels")
        (self.assets / "jaxtari/two.bin").write_bytes(b"another synthetic original asset")
        (self.assets / "operational.log").write_bytes(b"existing operational record")
        self.expected = {str(path.relative_to(self.assets)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((self.assets / "jaxtari").iterdir())}
        self.observed = repair.asset_roster(self.base, self.assets)

    def test_real_local_frozen_inputs_and_observation_have_exact_known_names(self):
        """The real4533-file observation contains all4531 original hashes plus the two known logs."""
        payload, base, plan, original, observation = repair.local_payload(PLAN)
        self.assertEqual(len(observation["files"]), 4533)
        self.assertEqual(plan["excluded_operational_files"], ["asset_verification.log", "assets_tree.sha256"])
        self.assertEqual(len(original["exact_copy_allowlist"]), 13)
        self.assertEqual(base.PLAN_SHA256, repair.ORIGINAL_SHA)
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_changed_plan_or_source_or_observation_never_starts_network(self):
        """A foreign repair plan or edited observed input fails before any SSH process is possible."""
        altered = self.directory / "altered.json"
        altered.write_bytes(PLAN.read_bytes() + b" ")
        with mock.patch.object(repair.subprocess, "Popen") as network:
            with self.assertRaisesRegex(ValueError, "reviewed hash"):
                repair.local_payload(altered)
            network.assert_not_called()
        payload, *_ = repair.local_payload(PLAN)
        for field in ("base_source", "observation", "failed_execution"):
            changed = dict(payload)
            changed[field] += " "
            with self.subTest(field=field), self.assertRaises(ValueError):
                repair.checked_payload(changed)

    def test_fresh_view_copies_only_expected_bytes_into_distinct_physical_files(self):
        """Exact new files share neither inodes nor links with any original, and logs remain outside the view."""
        before = repair.asset_roster(self.base, self.assets)
        view = self.directory / "view"
        with mock.patch.object(os, "link", side_effect=AssertionError("hardlinks forbidden")), \
                mock.patch.object(os, "symlink", side_effect=AssertionError("symlinks forbidden")):
            records = repair.copy_exact_assets(self.base, self.assets, view, self.expected, self.observed)
        self.assertEqual(len(records), 2)
        self.assertEqual(repair.asset_roster(self.base, self.assets), before)
        self.assertEqual(set(repair.asset_roster(self.base, view)), set(self.expected))
        self.assertFalse((view / "operational.log").exists())
        for row in records:
            self.assertNotEqual(row["original_inode"], row["copied_inode"])
            self.assertEqual(row["copied_link_count"], 1)

    def test_changed_missing_or_linked_source_asset_cannot_prepare_a_view(self):
        """A source mismatch, missing file or symlink stops the copy without changing original bytes."""
        path = self.assets / "jaxtari/one.bin"
        original = path.read_bytes()
        path.write_bytes(b"changed bytes")
        with self.assertRaisesRegex(ValueError, "changed bytes"):
            repair.copy_exact_assets(self.base, self.assets, self.directory / "changed_view", self.expected, self.observed)
        self.assertEqual(path.read_bytes(), b"changed bytes")
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            repair.copy_exact_assets(self.base, self.assets, self.directory / "missing_view", self.expected, self.observed)
        outside = self.directory / "outside.bin"
        outside.write_bytes(original)
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "ordinary absolute"):
            repair.copy_exact_assets(self.base, self.assets, self.directory / "linked_view", self.expected, self.observed)
        self.assertEqual(outside.read_bytes(), original)

    def test_existing_view_is_preserved_and_wrong_hash_cannot_be_relaxed(self):
        """Existing destinations remain untouched and a revised expected hash is never inferred."""
        view = self.directory / "existing"
        view.mkdir()
        (view / "keep.bin").write_bytes(b"existing evidence")
        with self.assertRaises(FileExistsError):
            repair.copy_exact_assets(self.base, self.assets, view, self.expected, self.observed)
        self.assertEqual((view / "keep.bin").read_bytes(), b"existing evidence")
        wrong = {**self.expected, "jaxtari/one.bin": "f" * 64}
        with self.assertRaisesRegex(ValueError, "changed bytes"):
            repair.copy_exact_assets(self.base, self.assets, self.directory / "wrong_hash", wrong, self.observed)

    def test_roster_rejects_extra_symlink_even_when_outside_manifest(self):
        """The old root observation cannot silently omit an unexpected linked file."""
        (self.assets / "linked.bin").symlink_to(self.assets / "jaxtari/one.bin")
        with self.assertRaisesRegex(ValueError, "symlink"):
            repair.asset_roster(self.base, self.assets)

    def test_effective_command_changes_only_asset_record_and_new_packet_path(self):
        """Every original source/config/runtime input remains identical in the new prepare command."""
        actual = repair.effective_plan(self.base, self.plan, self.original)
        before = self.original["native_prepare_parameters"]
        after = actual["native_prepare_parameters"]
        self.assertEqual([key for key in before if before[key] != after[key]], ["output-dir", "asset-record"])
        self.assertEqual(actual["native_prepare_command"][3], "prepare")
        self.assertIn("ControlMaster=no", self.base.SSH_OPTIONS)
        self.assertIn("ProxyCommand=false", self.base.SSH_OPTIONS)
        self.assertNotIn("-O", self.base.SSH_OPTIONS)

    def test_bootstrap_hash_is_checked_before_remote_function_can_execute(self):
        """The small metadata wrapper accepts only the exact supplied reviewed source hash."""
        source = 'REMOTE_COMMANDS=[]\ndef remote_prepare(payload):\n print("synthetic metadata only")\n'
        payload = json.dumps({"driver_source": source}).encode() + b"\n"
        for source_hash, expected in ((repair.digest(source.encode()), 0), ("0" * 64, 1)):
            result = subprocess.run([sys.executable, "-I", "-c", repair.BOOTSTRAP, source_hash], input=payload,
                capture_output=True, timeout=10)
            self.assertEqual(result.returncode, expected)
            if expected == 0:
                self.assertEqual(result.stdout, b"synthetic metadata only\n")
            else:
                self.assertNotIn(b"synthetic metadata only", result.stdout)

    def response(self, change=None):
        """Build synthetic exact prepared-file bytes without claiming any native ACES execution."""
        rows, contents = [], []
        for relative, path in repair.returned_files(self.base, self.plan, self.original):
            data = ("synthetic repair record " + relative).encode()
            rows.append({"relative_path": relative, "path": str(path), "bytes": len(data), "sha256": repair.digest(data)})
            contents.append(data)
        receipt = {"schema_version": 1, "kind": "aces_actual_exact_asset_view_metadata_preparation", "passed": True,
            "repair_plan_sha256": repair.REPAIR_SHA, "driver_sha256": "a" * 64, "base_driver_sha256": repair.BASE_SHA,
            "expected_manifest_sha256": repair.MANIFEST_SHA, "old_assets_inputs_roots_unchanged": True,
            "copied_file_count": 4531, "asset_view": self.plan["asset_view"], "numerical_imports": [],
            "workers_permitted": False, "jobs_submitted": 0, "case_credit": 0, "idea_credit": 0, "admission_credit": 0}
        data = (json.dumps(receipt) + "\n").encode()
        rows.append({"relative_path": "repair/native_asset_view_preparation_receipt.json", "path": self.plan["repair_directory"] + "/native_asset_view_preparation_receipt.json",
            "bytes": len(data), "sha256": repair.digest(data)})
        contents.append(data)
        header = {"receipt": receipt, "files": rows}
        if change:
            change(header)
        path = self.directory / "response.raw"
        path.write_bytes((json.dumps(header) + "\n").encode() + b"".join(contents))
        return path, receipt

    def test_exact_returned_metadata_binds_all_packet_and_repair_bytes(self):
        """A complete zero-job repair stream is saved under only the exact expected local mirror paths."""
        path, receipt = self.response()
        actual = repair.accept_response(self.base, path, self.directory, self.plan, self.original, "a" * 64)
        self.assertEqual(actual, receipt)
        self.assertEqual(len(list((self.directory / "received").rglob("*.*"))), 24)

    def test_returned_paths_identity_and_credit_are_strict(self):
        """Foreign paths, changed repair identity and Boolean credit substitutes cannot pass."""
        for name, change in (("path", lambda h: h["files"][0].update(path="/outside")),
                ("identity", lambda h: h["receipt"].update(repair_plan_sha256="f" * 64)),
                ("credit", lambda h: h["receipt"].update(case_credit=False))):
            path, _ = self.response(change)
            with self.subTest(name=name), self.assertRaises(ValueError):
                repair.accept_response(self.base, path, self.directory, self.plan, self.original, "a" * 64)


if __name__ == "__main__":
    unittest.main()
