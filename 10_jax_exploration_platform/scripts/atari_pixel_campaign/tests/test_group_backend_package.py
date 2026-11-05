"""Exercise group package-image proofs using small files that are never loaded."""
import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
specification = importlib.util.spec_from_file_location("tested_group_backend_package", SCRIPTS / "group_backend_package.py")
backend = importlib.util.module_from_spec(specification)
specification.loader.exec_module(backend)


class GroupBackendPackageTests(unittest.TestCase):
    """Test immutable preparation and capture without a library load or GPU call."""

    def setUp(self):
        """Build one fake installed package and one reserved group worker directory."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.python = self.root / "environment/bin/python"
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b"synthetic selected Python; never executed")
        self.site = self.root / "environment/lib/python3.12/site-packages"
        self.image = self.site / "jax_plugins/xla_cuda12/xla_cuda_plugin.so"
        self.image.parent.mkdir(parents=True)
        self.image.write_bytes(b"small synthetic group backend image; never loaded")
        self.owner = self.site / "unassumed_group_owner-7.3.dist-info"
        self.owner.mkdir()
        self.metadata_file = self.owner / "METADATA"
        self.metadata_file.write_bytes(b"Name: Unassumed-Group-Owner\nVersion: 7.3\n\n")
        self.record_file = self.owner / "RECORD"
        self.write_record()
        self.unit = self.root / "campaign/data/group"
        self.worker = self.unit / "workers/worker_1"
        self.worker.mkdir(parents=True)
        self.plan = {"schema_version": 2, "kind": "configured_a4500_scaling_group",
            "hardware_class": "jaguar03_rtx_a4500", "mode": "measurement", "gpu_count": 1,
            "campaign_run_dir": str(self.root / "campaign"), "unit_id": "group", "python": str(self.python),
            "configuration": {"learner": {"n_copies": 1}}, "copy_seeds": [[1]], "rounds": 7, "iterations": 10,
            "helpers": {"files": {backend.PREFIX + name: value for name, value in backend.source_hashes().items()}},
            "inputs": {}}
        self.expected = backend.prepare_image(self.python, self.site, self.image)
        self.prepared = self.root / "prepared_group_image.json"
        self.write(self.prepared, self.expected)
        backend.configure_plan(self.plan, self.prepared)
        self.process = {**backend.files.own_process(), "hostname": "jaguar03"}
        self.gpu_uuid = "GPU-00000000-0000-0000-0000-000000000001"
        self.save_plan_and_binding()
        self.execution_start = time.time()

    def write(self, path, value):
        """Write only synthetic fixture inputs; production publication never replaces files."""
        path.write_bytes(backend.files.encoded(value))

    def write_record(self):
        """Give the synthetic owner its real file hash and size instead of an assumed name."""
        raw = self.image.read_bytes()
        value = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        relative = self.image.relative_to(self.site).as_posix()
        self.record_file.write_text(f"{relative},sha256={value},{len(raw)}\n"
            + f"{self.metadata_file.relative_to(self.site).as_posix()},,\n"
            + f"{self.record_file.relative_to(self.site).as_posix()},,\n")

    def save_plan_and_binding(self):
        """Join the current private plan and fake native record to this actual test PID."""
        self.write(self.unit / "scaling_plan.json", self.plan)
        self.metadata = {"role": "worker", "directory": str(self.worker), "process": self.process}
        self.write(self.worker / "group_step_metadata.json", self.metadata)
        self.binding = {"process": self.process, "gpu_uuid": self.gpu_uuid,
            "native": {"pid": os.getpid(), "gpu_uuid": self.gpu_uuid},
            "plan_sha256": backend.files.reference(self.unit / "scaling_plan.json")["sha256"],
            "metadata_sha256": backend.files.digest(backend.files.encoded(self.metadata))}
        self.write(self.worker / "resolved_device_binding.json", self.binding)

    def maps(self, image):
        """Represent one own-process mapping with the real fixture file's filesystem identity."""
        identity = backend.files.file_identity(self.image.stat())
        raw = (f"1000-2000 r-xp 00000000 {identity['device_major']:02x}:{identity['device_minor']:02x} "
            + f"{identity['inode']} {image}\n")
        return {"path": "/proc/self/maps", "raw": raw, "sha256": backend.files.digest(raw.encode()),
            "file_identity": identity, "observed_at_seconds": time.time()}

    def begin(self):
        """Run the metadata capture against fake mappings and the actual fixture process."""
        with patch.object(backend.files, "own_process", return_value=self.process), \
                patch.object(backend.files, "mapped_snapshot", side_effect=self.maps):
            return backend.capture_begin(self.worker, self.plan, self.metadata, self.binding)

    def result(self):
        """Describe synthetic completed work while deliberately omitting XLA from CUDA receipts."""
        copies = self.plan["configuration"]["learner"]["n_copies"]
        rows = [{"round": number, "iterations": 10, "copies": copies, "environments_per_copy": 128,
            "rollout_steps": 128, "training_transitions": 163840 * copies,
            "report_origin_iteration": (number - 1) * 10, "actual_end_iteration": number * 10,
            "actual_optimizer_step": number * 160} for number in range(1, 8)]
        value = {"schema_version": 2, "kind": backend.MODES[self.plan["mode"]], "completed": True,
            "copy_seeds": self.plan["copy_seeds"][int(self.worker.name.split("_")[1]) - 1],
            "scientific_credit": 0, "admission_granted": False,
            "execution_identity": {key: self.process[key] for key in ("hostname", "pid", "process_start_time")},
            "loaded_gpu_libraries": {"pid": os.getpid(), "loaded_shared_libraries": ["synthetic-libcudnn.so"],
                "verified_package_libraries": {}}, "rounds": rows}
        value["execution_identity"].update(started_at_seconds=self.execution_start, finished_at_seconds=time.time())
        if self.plan["mode"] == "qualification":
            value.update(all_original_gates_passed=True, reference_iteration=1, reference_optimizer_step=16,
                executed_work={"normalization_warmup_transitions": 819200 * copies, "training_transitions": 16384 * copies,
                    "complete_updates": 1, "minibatch_updates": 16})
        if self.plan["mode"] == "correctness":
            value.update(correctness_attempted=True, correctness_passed=True,
                implementation_distinct=True, measurement_valid=False)
        return value

    def complete(self, result=None):
        """Finish only the fake controlled work with fresh actual metadata-file reads."""
        result = self.result() if result is None else result
        with patch.object(backend.files, "own_process", return_value=self.process), \
                patch.object(backend.files, "mapped_snapshot", side_effect=self.maps):
            return backend.capture_complete(self.worker, self.plan, result), result

    def test_actual_owner_discovery_and_external_sidecar_source_are_exact(self):
        """The package owner is discovered and both real prepin references remain distinct."""
        value = backend.validate_prepared(self.plan)
        self.assertEqual(value["distribution"]["name"], "Unassumed-Group-Owner")
        self.assertEqual(value["distribution"]["version"], "7.3")
        self.assertNotEqual(self.plan["inputs"]["backend_image_record"]["path"],
            self.plan["group_backend_image"]["expected"]["path"])
        self.assertEqual(set(backend.helper_names()), {"group_backend_package.py", "direct_backend_image.py"})

    def test_selected_python_and_package_environment_cannot_be_relabelled(self):
        """A coherently rewritten plan must not attach another environment's image to its selected Python."""
        other = self.root / "another-environment/bin/python"
        other.parent.mkdir(parents=True); other.write_bytes(b"another synthetic Python; never executed")
        changed = copy.deepcopy(self.expected); changed["selected_python"] = str(other)
        with self.assertRaisesRegex(ValueError, "selected package environment"):
            backend.validate_expected(changed, str(other))
        self.plan["python"] = str(other)
        self.write(self.prepared, changed); self.write(self.unit / "backend_image_expected.json", changed)
        self.plan["inputs"]["backend_image_record"] = backend.files.reference(self.prepared)
        self.plan["group_backend_image"]["expected"] = backend.files.reference(self.unit / "backend_image_expected.json")
        self.save_plan_and_binding()
        with self.assertRaisesRegex(ValueError, "selected package environment"):
            backend.validate_prepared(self.plan)

    def test_selected_python_paths_must_be_exact_absolute_bin_paths(self):
        """Relative paths and lexical aliases cannot change the selected environment proof."""
        for python in ("environment/bin/python", str(self.root / "environment/bin/../bin/python"),
                       str(self.root / "environment/not-bin/python")):
            changed = copy.deepcopy(self.expected); changed["selected_python"] = python
            with self.subTest(python=python), self.assertRaisesRegex(ValueError, "exact environment bin path"):
                backend.validate_expected(changed, python)

    def test_measurement_round_trip_keeps_xla_separate_from_cuda_receipts(self):
        """Shared actual package bytes do not need a false entry in the CUDA receipt list."""
        self.begin()
        final, result = self.complete()
        self.assertEqual(backend.validate_capture(self.worker, self.plan, self.metadata, self.binding,
            result["loaded_gpu_libraries"], result), final)
        self.assertNotIn(str(self.image), result["loaded_gpu_libraries"]["loaded_shared_libraries"])
        self.assertEqual((self.worker / "group_backend/xla_cuda_plugin.so").read_bytes(), self.image.read_bytes())

    def test_each_explicit_nonmeasurement_kind_requires_its_completed_fields(self):
        """Qualification and correctness have explicit modes and no timing or scientific credit."""
        for mode in ("qualification", "correctness"):
            with self.subTest(mode=mode):
                self.plan["mode"] = mode
                self.save_plan_and_binding()
                value = self.result()
                self.assertEqual(backend.completed_work(self.worker, self.plan, value), mode)
                value["completed"] = False
                with self.assertRaises(ValueError): backend.completed_work(self.worker, self.plan, value)

    def test_eighth_worker_is_distinct_and_worker_nine_is_refused(self):
        """An eight-card group uses its eighth reserved worker without guessing a physical card."""
        self.plan["gpu_count"] = 8
        self.plan["copy_seeds"] = [[number] for number in range(1, 9)]
        self.worker = self.unit / "workers/worker_8"
        self.worker.mkdir()
        self.save_plan_and_binding()
        self.begin()
        final, result = self.complete()
        self.assertEqual(final["worker_index"], 7)
        ninth = self.unit / "workers/worker_9"
        ninth.mkdir()
        with self.assertRaises(ValueError): backend.worker_index(ninth, self.plan)

    def test_exact_eight_mode_requires_all_eight_copies_on_one_worker(self):
        """A smaller measurement cannot be relabeled as exact-eight memory evidence."""
        self.plan["mode"] = "exact_eight_eligibility"
        with self.assertRaises(ValueError): backend.completed_work(self.worker, self.plan, self.result())
        self.plan["configuration"]["learner"]["n_copies"] = 8
        self.plan["copy_seeds"] = [list(range(1, 9))]
        self.save_plan_and_binding()
        self.assertEqual(backend.completed_work(self.worker, self.plan, self.result()), "exact_eight_eligibility")

    def test_legacy_direct_or_changed_producer_preparation_is_refused(self):
        """A direct record or changed helper hash cannot be relabeled as the group protocol."""
        for field, value in (("schema_version", 1), ("kind", "prepared_direct_backend_package_image"),
                ("capability", "old"), ("gpu_operations", True)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.expected)
                changed[field] = value
                with self.assertRaises(ValueError): backend.validate_expected(changed, self.plan["python"])
        changed = copy.deepcopy(self.expected)
        changed["producer_sources"]["group_backend_package.py"] = "0" * 64
        with self.assertRaises(ValueError): backend.validate_expected(changed, self.plan["python"])

    def test_missing_or_changed_helper_and_external_record_are_refused(self):
        """Both staged helper declarations and the original prepin bytes remain necessary."""
        wanted = self.plan["helpers"]["files"].pop(backend.PREFIX + "direct_backend_image.py")
        with self.assertRaises(ValueError): backend.expected_for_plan(self.plan)
        self.plan["helpers"]["files"][backend.PREFIX + "direct_backend_image.py"] = wanted
        self.prepared.write_bytes(self.prepared.read_bytes() + b" ")
        with self.assertRaises(ValueError): backend.expected_for_plan(self.plan)

    def test_changed_installed_image_or_owner_record_is_refused_before_capture(self):
        """A coherent new wheel RECORD still cannot change an already pinned package image."""
        self.image.write_bytes(b"different installed bytes")
        self.write_record()
        with self.assertRaises(ValueError): backend.validate_prepared(self.plan)
        self.assertFalse((self.worker / "group_backend_capture_attempt.json").exists())

    def test_duplicate_distribution_owner_is_refused(self):
        """Two real RECORD owners are ambiguous even when both claim the same image bytes."""
        other = self.site / "second_owner-7.3.dist-info"
        other.mkdir()
        (other / "METADATA").write_bytes(b"Name: Another-Owner\nVersion: 7.3\n\n")
        (other / "RECORD").write_bytes(self.record_file.read_bytes().replace(
            self.owner.name.encode(), other.name.encode()))
        with self.assertRaises(ValueError): backend.prepare_image(self.python, self.site, self.image)

    def test_wrong_actual_process_native_pid_or_gpu_is_refused(self):
        """Saved worker membership cannot substitute for a different current process."""
        with patch.object(backend.files, "own_process", return_value={**self.process, "process_start_time": "1"}):
            with self.assertRaises(ValueError):
                backend.process_join(self.worker, self.plan, self.metadata, self.binding, current=True)
        for change in ({"pid": os.getpid() + 1}, {"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000002"}):
            with self.subTest(change=change):
                value = copy.deepcopy(self.binding)
                value["native"].update(change)
                self.write(self.worker / "resolved_device_binding.json", value)
                with self.assertRaises(ValueError):
                    backend.process_join(self.worker, self.plan, self.metadata, value, current=False)

    def test_changed_supplied_plan_and_coordinator_directory_are_refused(self):
        """The reader uses the actual frozen plan and never treats the coordinator as a worker."""
        value = copy.deepcopy(self.plan)
        value["iterations"] = 20
        with self.assertRaises(ValueError):
            backend.process_join(self.worker, value, self.metadata, self.binding, current=False)
        with self.assertRaises(ValueError): backend.worker_index(self.unit, self.plan)

    def test_prior_attempt_or_dangling_marker_cannot_be_reused(self):
        """A recorded or dangling attempt permanently prevents another capture in that worker."""
        marker = self.worker / "group_backend_capture_attempt.json"
        marker.symlink_to(self.root / "absent")
        with self.assertRaises(ValueError): self.begin()
        marker.unlink()
        self.begin()
        with self.assertRaises(ValueError): self.begin()

    def test_copied_image_or_owning_record_changes_are_refused(self):
        """A later reader rehashes actual shared copied bytes rather than trusting their labels."""
        self.begin()
        _, result = self.complete()
        for filename in ("xla_cuda_plugin.so", "METADATA", "RECORD"):
            path = self.worker / "group_backend" / filename
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.assertRaises(ValueError):
                backend.validate_capture(self.worker, self.plan, self.metadata, self.binding,
                    result["loaded_gpu_libraries"], result)
            path.write_bytes(original)

    def test_changed_post_package_or_mapping_bytes_are_refused(self):
        """A coherent final JSON rewrite cannot replace actual mapped-file evidence."""
        self.begin()
        final, result = self.complete()
        path = self.worker / "group_backend_capture.json"
        for section in ("post_installed_package", "maps_after_numerical_work"):
            changed = copy.deepcopy(final)
            if section == "post_installed_package":
                changed[section]["description"]["image"]["sha256"] = "0" * 64
            else:
                changed[section]["raw"] = changed[section]["raw"].replace("r-xp", "r--p")
            self.write(path, changed)
            with self.assertRaises(ValueError):
                backend.validate_capture(self.worker, self.plan, self.metadata, self.binding,
                    result["loaded_gpu_libraries"], result)
        self.write(path, final)

    def test_partial_wrong_kind_boolean_or_unfinished_work_is_refused(self):
        """All seven windows and typed original counters must be present before completion."""
        for mutate in (lambda value: value["rounds"].pop(), lambda value: value.update(kind="legacy"),
                lambda value: value["rounds"][0].update(iterations=True),
                lambda value: value["rounds"][6].update(actual_end_iteration=69),
                lambda value: value.update(copy_seeds=[True])):
            with self.subTest(mutate=mutate):
                value = self.result()
                mutate(value)
                with self.assertRaises(ValueError): backend.completed_work(self.worker, self.plan, value)

    def test_qualification_counts_cannot_use_boolean_one(self):
        """A Boolean does not prove one complete optimizer update."""
        self.plan["mode"] = "qualification"
        value = self.result()
        value["executed_work"]["complete_updates"] = True
        with self.assertRaises(ValueError): backend.completed_work(self.worker, self.plan, value)

    def test_controlled_qualification_transition_totals_scale_with_copies(self):
        """Keep one complete update per copy, with total environment transitions across every declared copy."""
        self.plan["mode"] = "qualification"
        for copies in (1, 2, 8):
            self.plan["configuration"]["learner"]["n_copies"] = copies
            self.plan["copy_seeds"] = [list(range(1, copies + 1))]
            self.save_plan_and_binding(); value = self.result()
            self.assertEqual(backend.completed_work(self.worker, self.plan, value), "qualification")
            self.assertEqual(value["executed_work"]["normalization_warmup_transitions"], 819200 * copies)
            self.assertEqual(value["executed_work"]["training_transitions"], 16384 * copies)
            for name in ("normalization_warmup_transitions", "training_transitions"):
                changed = copy.deepcopy(value); changed["executed_work"][name] -= 1
                with self.assertRaisesRegex(ValueError, "controlled reference update is incomplete"):
                    backend.completed_work(self.worker, self.plan, changed)
                if copies > 1:
                    changed["executed_work"][name] = 819200 if name.startswith("normalization") else 16384
                    with self.assertRaisesRegex(ValueError, "controlled reference update is incomplete"):
                        backend.completed_work(self.worker, self.plan, changed)
        self.plan["configuration"]["learner"]["n_copies"] = True
        with self.assertRaisesRegex(ValueError, "copy count"):
            backend.completed_work(self.worker, self.plan, self.result())

    def test_float_or_boolean_numerical_pid_cannot_pass_a_coherent_saved_rewrite(self):
        """Typed actual worker identity survives coherent final/result JSON edits."""
        self.begin(); final, result = self.complete()
        for pid in (float(self.process["pid"]), True):
            changed_result = copy.deepcopy(result); changed_result["execution_identity"]["pid"] = pid
            changed_final = copy.deepcopy(final); changed_final["numerical_execution"] = changed_result["execution_identity"]
            self.write(self.worker / "group_backend_capture.json", changed_final)
            with self.subTest(pid=pid), self.assertRaisesRegex(ValueError, "actual completed worker execution"):
                backend.validate_capture(self.worker, self.plan, self.metadata, self.binding,
                    changed_result["loaded_gpu_libraries"], changed_result)
        self.write(self.worker / "group_backend_capture.json", final)

    def test_wrong_loaded_receipt_or_execution_process_is_refused(self):
        """Separate CUDA receipts and numerical execution must belong to the same real worker."""
        self.begin()
        _, result = self.complete()
        wrong = {**result["loaded_gpu_libraries"], "pid": os.getpid() + 1}
        with self.assertRaises(ValueError):
            backend.validate_capture(self.worker, self.plan, self.metadata, self.binding, wrong, result)
        changed = copy.deepcopy(result)
        changed["execution_identity"]["process_start_time"] = "1"
        with self.assertRaises(ValueError):
            backend.validate_capture(self.worker, self.plan, self.metadata, self.binding,
                result["loaded_gpu_libraries"], changed)

    def test_completion_cannot_repeat_or_precede_complete_work(self):
        """Final publication is exclusive and incomplete work cannot receive a final capture."""
        self.begin()
        wrong = self.result()
        wrong["completed"] = False
        with self.assertRaises(ValueError): self.complete(wrong)
        final, result = self.complete()
        with self.assertRaises(ValueError): self.complete(result)
        self.assertEqual(final["completed_work_kind"], "measurement")


if __name__ == "__main__":
    unittest.main()

