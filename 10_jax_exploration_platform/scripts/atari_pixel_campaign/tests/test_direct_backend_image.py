"""Check separate backend package-image metadata with small synthetic ordinary files.

These tests perform no numerical work, GPU operation or qualification. The fake
package name varies to prove that ownership is discovered from actual RECORDs.
"""
import base64
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import direct_backend_image as backend
import direct_measurement as direct
import measure_selected as selected


class DirectBackendImageTests(unittest.TestCase):
    """Exercise immutable preparation, same-process capture and saved-byte replay."""

    def setUp(self):
        """Build one synthetic installed distribution and a direct metadata plan."""
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.python = self.root / "environment/bin/python"; self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b"synthetic Python file; never executed")
        self.site = self.root / "environment/lib/python3.12/site-packages"; self.site.mkdir(parents=True)
        self.image = self.site / "jax_plugins/xla_cuda12/xla_cuda_plugin.so"; self.image.parent.mkdir(parents=True)
        self.image.write_bytes(b"small synthetic backend bytes; never loaded")
        self.owner = self.site / "different_backend_owner-1.0.dist-info"; self.owner.mkdir()
        self.metadata = self.owner / "METADATA"; self.metadata.write_bytes(b"Name: Different-Backend-Owner\nVersion: 1.0\n\n")
        self.record = self.owner / "RECORD"; self.write_record()
        self.unit = self.root / "campaign/data/unit"; self.unit.mkdir(parents=True)
        self.expected = backend.prepare_image(self.python, self.site, self.image)
        self.prepared = self.root / "prepared.json"; self.write(self.prepared, self.expected)
        self.plan = {"kind": "selected_configuration_measurement", "mode": "correctness",
            "campaign_run_dir": str(self.root / "campaign"), "unit_id": "unit", "python": str(self.python),
            "direct_binding": {}, "hardware_class": "h100", "gpu_uuid": backend.CARD_ONE,
            "helpers": {"files": {backend.PREFIX + backend.NAME: backend.digest(Path(backend.__file__).read_bytes())}},
            "inputs": {}, "qualification_software": {"packages": {"different_backend_owner": "1.0"}}}
        backend.configure_plan(self.plan, self.prepared)
        self.write(self.unit / "measurement_plan.json", self.plan)
        self.process = {**backend.own_process(), "hostname": "serval02"}
        self.binding = {"process": self.process, "gpu_uuid": backend.CARD_ONE, "native": {"pid": os.getpid()}}
        self.write(self.unit / "direct_device_binding.json", self.binding)
        self.start = time.time()

    def write(self, path, value):
        """Write synthetic inputs; production publication separately forbids replacement."""
        path.write_bytes(backend.encoded(value))

    def write_record(self, *, value=None, size=None, duplicate=False):
        """Rebuild the synthetic wheel roster without importing its image."""
        raw = self.image.read_bytes()
        value = value or "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        size = len(raw) if size is None else size
        relative = self.image.relative_to(self.site).as_posix()
        row = f"{relative},{value},{size}\n"
        self.record.write_text(row + (row if duplicate else "")
            + f"{self.metadata.relative_to(self.site).as_posix()},,\n{self.record.relative_to(self.site).as_posix()},,\n")

    def maps(self, path):
        """Represent an own-process mapping using the real fixture file's filesystem identity."""
        identity = backend.file_identity(self.image.stat())
        raw = f"1000-2000 r-xp 00000000 {identity['device_major']:02x}:{identity['device_minor']:02x} {identity['inode']} {path}\n"
        return {"path": "/proc/self/maps", "raw": raw, "sha256": backend.digest(raw.encode()),
            "file_identity": identity, "observed_at_seconds": time.time()}

    def begin(self):
        """Run only the metadata producer against fake current-process mappings."""
        with patch.object(backend, "own_process", return_value=self.process), patch.object(backend, "mapped_snapshot", side_effect=self.maps):
            return backend.capture_begin(self.unit, self.plan, self.binding)

    def result(self):
        """Keep CUDA receipts separate; the synthetic backend is deliberately absent there."""
        return {"correctness_passed": True, "correctness_attempted": True,
            "loaded_gpu_libraries": {"pid": os.getpid(), "loaded_shared_libraries": ["/synthetic/libcudnn.so"],
                "verified_package_libraries": {"/synthetic/libcudnn.so": {"sha256": "c" * 64}}},
            "execution_identity": {key: self.process[key] for key in ("hostname", "pid", "process_start_time")} |
                {"started_at_seconds": self.start, "finished_at_seconds": time.time()}}

    def complete(self):
        """Create a synthetic complete receipt without performing learner work."""
        self.begin(); result = self.result()
        with patch.object(backend, "own_process", return_value=self.process), patch.object(backend, "mapped_snapshot", side_effect=self.maps):
            value = backend.capture_complete(self.unit, self.plan, result)
        return value, result

    def test_actual_unique_record_owner_is_discovered_without_assuming_package_name(self):
        """The distinct fake owner is read from actual file ownership and headers."""
        self.assertEqual(self.expected["distribution"]["name"], "Different-Backend-Owner")
        self.assertEqual(self.expected["distribution"]["version"], "1.0")
        self.assertEqual(backend.validate_prepared(self.plan), self.expected)
        self.assertEqual(self.expected["gpu_operations"], 0)
        self.assertEqual(self.expected["numerical_imports"], [])

    def test_duplicate_owner_wrong_record_hash_size_and_repeated_image_row_are_rejected(self):
        """Ambiguous ownership or any image/RECORD disagreement stops preparation."""
        for arguments in ({"value": "sha256=" + "A" * 43}, {"size": 999}, {"duplicate": True}):
            with self.subTest(arguments=arguments):
                self.write_record(**arguments)
                with self.assertRaises(ValueError):backend.describe_image(self.site, self.image)
        self.write_record()
        other = self.site / "second_owner-2.dist-info"; other.mkdir()
        (other / "METADATA").write_bytes(b"Name: Second-Owner\nVersion: 2\n\n")
        (other / "RECORD").write_text(self.record.read_text().splitlines()[0] + "\nsecond_owner-2.dist-info/METADATA,,\nsecond_owner-2.dist-info/RECORD,,\n")
        with self.assertRaises(ValueError):backend.describe_image(self.site, self.image)

    def test_duplicate_metadata_headers_are_rejected(self):
        """One distribution must name one package and one version."""
        for raw in (b"Name: First\nName: Second\nVersion: 1\n", b"Name: First\nVersion: 1\nVersion: 2\n"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):backend.package_headers(raw)

    def test_exact_prepared_bytes_are_copied_and_a_different_prior_record_is_preserved(self):
        """The immutable sidecar keeps the caller's actual JSON bytes."""
        self.assertEqual(self.prepared.read_bytes(), (self.unit / "direct_backend_image_expected.json").read_bytes())
        backend.configure_plan(self.plan, self.prepared)
        wrong = copy.deepcopy(self.expected); wrong["created_at_seconds"] += 1; self.write(self.prepared, wrong)
        with self.assertRaises(ValueError):backend.configure_plan(self.plan, self.prepared)
        self.assertEqual(backend.document(self.unit / "direct_backend_image_expected.json"), self.expected)

    def test_image_metadata_record_and_software_version_changes_are_rejected_before_work(self):
        """Current files and the unchanged qualification package versions must match the pin."""
        for path in (self.image, self.metadata, self.record):
            original = path.read_bytes(); path.write_bytes(original + b"changed\n")
            with self.subTest(path=path), self.assertRaises(ValueError):backend.validate_prepared(self.plan)
            path.write_bytes(original)
        changed = copy.deepcopy(self.plan); changed["qualification_software"]["packages"]["different_backend_owner"] = "2"
        with self.assertRaises(ValueError):backend.validate_prepared(changed)

    def test_legacy_missing_helper_and_changed_prepared_source_are_rejected(self):
        """A historical helper closure cannot start or admit a new direct run."""
        for mutate in (lambda p:p.pop("direct_backend_image"),
                lambda p:p["direct_backend_image"].update(schema_version=1),
                lambda p:p["helpers"]["files"].clear()):
            changed = copy.deepcopy(self.plan); mutate(changed)
            with self.assertRaises(ValueError):backend.expected_for_plan(changed)
        wrong = {**self.expected, "producer_source_sha256": "0" * 64}
        with self.assertRaises(ValueError):backend.validate_expected(wrong)
        with self.assertRaises(ValueError):direct.backend_support({"helpers": {"files": {}}})

    def test_same_process_full_capture_replays_using_shared_bytes_and_separate_cuda_receipt(self):
        """Replay uses the new package proof while the CUDA receipt omits the image."""
        final, result = self.complete()
        with patch.object(backend, "describe_image", side_effect=AssertionError("saved replay must use captured bytes")), \
                patch.object(backend, "mapped_snapshot", side_effect=AssertionError("no live mapping reads during replay")):
            self.assertEqual(backend.validate_capture(self.unit, self.plan, self.binding, result["loaded_gpu_libraries"], result), final)
        self.assertNotIn(str(self.image), result["loaded_gpu_libraries"]["loaded_shared_libraries"])
        self.assertEqual((self.unit / "direct_backend/xla_cuda_plugin.so").read_bytes(), self.image.read_bytes())

    def test_changed_shared_binary_metadata_and_record_are_rejected(self):
        """Portable replay hashes all actual captured bytes and parses their ownership."""
        unused, result = self.complete()
        for name in ("xla_cuda_plugin.so", "METADATA", "RECORD"):
            path = self.unit / "direct_backend" / name; original = path.read_bytes(); path.write_bytes(original + b"changed\n")
            with self.subTest(name=name), self.assertRaises(ValueError):backend.validate_capture(self.unit, self.plan, self.binding, result["loaded_gpu_libraries"], result)
            path.write_bytes(original)

    def test_mapping_wrong_inode_deleted_image_multiple_images_and_missing_mapping_are_rejected(self):
        """Filesystem mappings stay separate from GPU selectors and reject aliases."""
        snapshot = self.maps(str(self.image)); raw = snapshot["raw"]
        for wrong in (raw.replace(str(self.image), str(self.image) + " (deleted)"),
                raw + raw.replace(str(self.image), "/other/xla_cuda_plugin.so"), ""):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):backend.mapping_rows(wrong, str(self.image))
        with patch.object(Path, "open", return_value=io.BytesIO((raw + raw.replace(str(self.image), "/other/xla_cuda_plugin.so")).encode())):
            with self.assertRaises(ValueError):backend.mapped_snapshot(str(self.image))
        self.begin(); result = self.result(); wrong = self.maps(str(self.image)); wrong["file_identity"]["inode"] += 1
        with patch.object(backend, "own_process", return_value=self.process), patch.object(backend, "mapped_snapshot", return_value=wrong):
            with self.assertRaises(ValueError):backend.capture_complete(self.unit, self.plan, result)

    def test_different_process_binding_and_missing_completed_work_are_rejected(self):
        """A post-work proof must remain this bound process and include actual completion."""
        self.begin(); result = self.result()
        for wrong in ({**self.process, "pid": self.process["pid"] + 1}, {**self.process, "process_start_time": "1"}):
            with patch.object(backend, "own_process", return_value=wrong), patch.object(backend, "mapped_snapshot", side_effect=self.maps):
                with self.assertRaises(ValueError):backend.capture_complete(self.unit, self.plan, result)
        result["correctness_passed"] = False
        with patch.object(backend, "own_process", return_value=self.process), patch.object(backend, "mapped_snapshot", side_effect=self.maps):
            with self.assertRaises(ValueError):backend.capture_complete(self.unit, self.plan, result)

    def test_full_original_reference_completion_and_times_are_required(self):
        """One full original update and its finish time precede the post-work mapping."""
        plan = {**self.plan, "direct_qualification": {}}
        result = self.result()
        for fields in ({"all_original_gates_passed": True, "reference_iteration": True, "reference_optimizer_step": 16},
                {"all_original_gates_passed": True, "reference_iteration": 1, "reference_optimizer_step": 15},
                {"all_original_gates_passed": True, "reference_iteration": 1, "reference_optimizer_step": 16.0}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):backend.completed_work(plan, {**result, **fields})
        self.assertEqual(backend.completed_work(plan, {**result, "all_original_gates_passed": True,
            "reference_iteration": 1, "reference_optimizer_step": 16}), "full_original_qualification")
        unused, result = self.complete()
        result["execution_identity"]["finished_at_seconds"] = self.start - 10
        with self.assertRaises(ValueError):backend.validate_capture(self.unit, self.plan, self.binding, result["loaded_gpu_libraries"], result)

    def test_failed_capture_is_preserved_and_cannot_be_retried(self):
        """A missing map leaves an explicit failed attempt and never replaces it."""
        with patch.object(backend, "own_process", return_value=self.process), patch.object(backend, "mapped_snapshot", side_effect=ValueError("synthetic missing current map")):
            with self.assertRaises(ValueError):backend.capture_begin(self.unit, self.plan, self.binding)
        self.assertTrue((self.unit / "direct_backend_capture_failure.json").exists())
        with self.assertRaises(ValueError):self.begin()

    def test_helper_union_keeps_default_and_a4500_unchanged(self):
        """Only the explicit new direct protocol stages the separate backend module."""
        self.assertEqual(len(selected.required_helpers()), 16)
        self.assertEqual(len(selected.required_helpers(True)), 20)
        self.assertEqual(len(selected.required_helpers(True, False, True)), 21)
        self.assertEqual(len(selected.required_helpers(False, True)), 25)
        self.assertEqual(len(selected.required_helpers(False, True, False, True)), 26)
        self.assertEqual(len(selected.required_helpers(False, True, True, True)), 27)
        self.assertIn("direct_common_confirmation.py", selected.required_helpers(False, True, True, True))
        with self.assertRaises(ValueError):selected.required_helpers(True, False, False, True)

    def test_own_mapping_read_is_bounded_before_utf8_decode(self):
        """A large own mapping file is rejected after at most the declared byte limit plus one."""
        for raw in (b"x" * (backend.SMALL_FILE_LIMIT + 100), b"\xff"):
            stream = io.BytesIO(raw)
            with patch.object(Path, "open", return_value=stream), patch.object(stream, "read", wraps=stream.read) as read:
                with self.assertRaises(ValueError):backend.mapped_snapshot(str(self.image))
                read.assert_called_once_with(backend.SMALL_FILE_LIMIT + 1)

    def test_post_work_installed_package_hashes_descriptor_and_times_are_retained_and_checked(self):
        """The existing post-work hash pass supplies independently readable actual file evidence."""
        with patch.object(backend, "describe_image", wraps=backend.describe_image) as describe:
            final, result = self.complete()
        self.assertEqual(describe.call_count, 2)  # One initial check and the already-required post-work pass.
        post = final["post_installed_package"]; description = post["description"]
        self.assertEqual(description["image"], self.expected["image"])
        self.assertEqual(description["distribution"], self.expected["distribution"])
        self.assertEqual(description["observed_descriptor_before"], description["observed_descriptor_after"])
        self.assertGreaterEqual(post["started_at_seconds"], result["execution_identity"]["finished_at_seconds"])
        changes = [lambda p:p["post_installed_package"]["description"]["image"].update(sha256="0" * 64),
            lambda p:p["post_installed_package"]["description"]["distribution"]["record"].update(sha256="0" * 64),
            lambda p:p["post_installed_package"]["description"]["distribution"]["metadata"].update(sha256="0" * 64),
            lambda p:p["post_installed_package"]["description"]["observed_file_identity"].update(inode=1),
            lambda p:p["post_installed_package"]["description"]["observed_descriptor_after"].update(size_bytes=1),
            lambda p:p["post_installed_package"].update(started_at_seconds=self.start - 1)]
        for number, change in enumerate(changes):
            wrong = copy.deepcopy(final); change(wrong); self.write(self.unit / "direct_backend_capture.json", wrong)
            with self.subTest(number=number), self.assertRaises(ValueError):backend.validate_capture(
                self.unit, self.plan, self.binding, result["loaded_gpu_libraries"], result)
        self.write(self.unit / "direct_backend_capture.json", final)

    def test_saved_receipt_rejects_changed_cuda_digest_process_or_work_kind(self):
        """Backend proof joins the separate unchanged CUDA receipt and actual completed work."""
        unused, result = self.complete()
        original = backend.document(self.unit / "direct_backend_capture.json")
        for key, value in (("loaded_gpu_libraries_sha256", "0" * 64), ("completed_work_kind", "not-completed"),
                ("process", {**original["process"], "process_start_time": "1"})):
            changed = {**original, key: value}; self.write(self.unit / "direct_backend_capture.json", changed)
            with self.subTest(key=key), self.assertRaises(ValueError):backend.validate_capture(
                self.unit, self.plan, self.binding, result["loaded_gpu_libraries"], result)
        self.write(self.unit / "direct_backend_capture.json", original)

    def test_source_preserves_numerical_compute_original_gates_and_cuda_collectors(self):
        """Compare exact existing functions and files against the declared published base."""
        root = SCRIPTS.parents[2]; prefix = backend.PREFIX
        def baseline(name):
            return subprocess.check_output(["git", "-C", str(root), "show", "9fcc43a478b93d731df780805fd7c3c514cdf67f:" + prefix + name], text=True)
        def function(text, name):
            return next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == name)
        self.assertEqual(ast.dump(function(baseline("measure_selected.py"), "_compute")),
            ast.dump(function((SCRIPTS / "measure_selected.py").read_text(), "_compute")))
        self.assertEqual(ast.dump(function(baseline("direct_measurement.py"), "run_qualification")),
            ast.dump(function((SCRIPTS / "direct_measurement.py").read_text(), "run_qualification")))
        old = function(baseline("direct_measurement.py"), "verify_qualification_inputs").body
        current = function((SCRIPTS / "direct_measurement.py").read_text(), "verify_qualification_inputs").body
        retained = iter(ast.dump(node) for node in current)
        self.assertTrue(all(any(value == ast.dump(node) for value in retained) for node in old))
        self.assertEqual(len(current), len(old) + 3)
        for name in ("train_optimized.py", "diagnose_gpu_execution.py", "configured_measurement.py", "common_confirmation.py"):
            self.assertEqual(baseline(name), (SCRIPTS / name).read_text())


if __name__ == "__main__":
    unittest.main()
