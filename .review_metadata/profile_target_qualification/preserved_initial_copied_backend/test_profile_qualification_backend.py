"""Small synthetic package/record fixtures; no file here is executed or loaded."""
import base64
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
specification = importlib.util.spec_from_file_location("tested_profile_qualification_backend", SCRIPTS / "profile_qualification_backend.py")
backend = importlib.util.module_from_spec(specification)
specification.loader.exec_module(backend)


class ProfileQualificationBackendTests(unittest.TestCase):
    """Exercise both separate binding policies using synthetic raw metadata only."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.python = self.root / "environment/bin/python"
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b"synthetic selected Python image; never executed")
        self.site = self.root / "environment/lib/python3.12/site-packages"
        self.image = self.site / "jax_plugins/xla_cuda12/xla_cuda_plugin.so"
        self.image.parent.mkdir(parents=True)
        self.image.write_bytes(b"synthetic XLA image; never loaded")
        self.owner = self.site / "unassumed_profile_owner-7.3.dist-info"
        self.owner.mkdir()
        self.metadata = self.owner / "METADATA"
        self.metadata.write_bytes(b"Name: Unassumed-Profile-Owner\nVersion: 7.3\n\n")
        self.record = self.owner / "RECORD"
        self.write_record()
        self.unit = self.root / "campaign/data/qualification"
        self.unit.mkdir(parents=True)
        parent = self.root / "parent_plan.json"
        self.write(parent, {"schema_version": 1, "kind": "profile_target_qualification_controller"})
        self.plan = {"schema_version": 1, "kind": backend.KIND, "hardware_class": "a4500",
            "configured_isolation": {"synthetic_fixture_not_target_evidence": True}, "gpu_uuid": None,
            "profile_target_qualification": {"schema_version": 1, "protocol": backend.PROTOCOL,
                "parent": backend.files.reference(parent)},
            "campaign_run_dir": str(self.root / "campaign"), "unit_id": "qualification", "python": str(self.python),
            "profile_updates": 1, "configuration": {"learner": {"n_copies": 1}}, "copy_seeds": [1],
            "runtime_environment": {"synthetic_fixture": True}, "qualification_software": {
                "packages": {"Unassumed-Profile-Owner": "7.3"}},
            "helpers": {"files": {backend.PREFIX + name: sha for name, sha in backend.source_hashes().items()}}, "inputs": {}}
        self.process = {**backend.files.own_process(), "hostname": "jaguar03"}
        self.original_resolve = Path.resolve
        with self.live():
            self.expected = backend.prepare_image(self.python, self.site, self.image)
        self.external = self.root / "prepared.json"
        self.write(self.external, self.expected)
        backend.configure_plan(self.plan, self.external)
        self.save_plan_and_binding()

    def write(self, path, value):
        path.write_bytes(backend.files.encoded(value))

    def write_record(self):
        raw = self.image.read_bytes()
        sha = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        self.record.write_text(f"{self.image.relative_to(self.site).as_posix()},sha256={sha},{len(raw)}\n"
            + f"{self.metadata.relative_to(self.site).as_posix()},,\n"
            + f"{self.record.relative_to(self.site).as_posix()},,\n")

    @contextmanager
    def live(self):
        """Mock the target executable/mapping context; production never uses this fixture."""
        python, resolve = self.python, self.original_resolve
        def proc_executable(path, *args, **kwargs):
            return python if str(path) == "/proc/self/exe" else resolve(path, *args, **kwargs)
        with patch.object(backend.sys, "executable", str(self.python)), \
                patch.object(backend.Path, "resolve", new=proc_executable), \
                patch.object(backend.files, "own_process", return_value=self.process), \
                patch.object(backend.files, "mapped_snapshot", side_effect=self.maps):
            yield

    def save_plan_and_binding(self):
        self.write(self.unit / "measurement_plan.json", self.plan)
        h100 = self.plan["hardware_class"] == "h100"
        gpu = backend.CARD_ONE if h100 else "GPU-00000000-0000-0000-0000-000000000001"
        self.binding = {"schema_version": 1,
            "kind": "serval02_card1_current_process_resolved_device_binding" if h100 else "uva_current_process_resolved_device_binding",
            "process": self.process, "gpu_uuid": gpu, "pci_address": "0000:01:00.0", "physical_minor": 1,
            "native": {"pid": os.getpid(), "cuda_device_count": 1, "jax_local_hardware_id": 0,
                "cuda_device_handle": 0, "current_context": 123, "current_context_device": 0,
                "jax_device_kind": "NVIDIA H100 NVL" if h100 else "NVIDIA RTX A4500",
                "api_return_codes": {key: 0 for key in ("cuDeviceGetCount", "cuDeviceGet", "cuCtxGetCurrent",
                    "cuCtxGetDevice", "cuDeviceGetUuid_v2", "cuDeviceGetPCIBusId")},
                "jax_identity_basis": "sole local JAX GPU, sole CUDA-visible ordinal, and that device's current CUDA context",
                "gpu_uuid": gpu, "pci_address": "0000:01:00.0"},
            "device": {"path": "/dev/nvidia1", "character_device": True, "symlink": False, "major": 195, "minor": 1},
            "physical_gres_mapping_verified": False, "kernel_device_enforcement_verified": False}
        plan_ref = backend.files.reference(self.unit / "measurement_plan.json")
        self.binding.update({"plan": plan_ref} if h100 else {"plan_sha256": plan_ref["sha256"]})
        self.write(self.unit / backend.binding_file(self.plan), self.binding)

    def maps(self, path):
        identity = backend.files.file_identity(self.image.stat())
        raw = (f"1000-2000 r-xp 00000000 {identity['device_major']:02x}:{identity['device_minor']:02x} "
            + f"{identity['inode']} {path}\n")
        return {"path": "/proc/self/maps", "raw": raw, "sha256": backend.files.digest(raw.encode()),
            "file_identity": identity, "observed_at_seconds": time.time()}

    def begin(self):
        with self.live():
            return backend.begin(self.unit, self.plan, self.binding)

    def work(self):
        capture_start, capture_finish, gate_start, gate_finish = [time.time() for _ in range(4)]
        def execution(start, finish):
            return {**{key: self.process[key] for key in ("hostname", "pid", "process_start_time")},
                "started_at_seconds": start, "finished_at_seconds": finish}
        return {"kind": "full_original_profile_source_target_qualification_work", "all_original_gates_passed": True,
            "configuration": copy.deepcopy(self.plan["configuration"]), "copy_seeds": self.plan["copy_seeds"],
            "actual_runtime_environment": self.plan["runtime_environment"], "test_fixture_work": "unknown",
            "scientific_credit": 0, "admission_granted": False,
            "original_reference": {"copies": 1, "normalization_warmup_iterations": 50, "reference_iteration": 1,
                "reference_optimizer_step": 16, "executed_work": {"normalization_warmup_transitions": 819200,
                    "training_transitions": 16384, "complete_updates": 1, "minibatch_updates": 16}},
            "profile_capture": {"copies": self.plan["configuration"]["learner"]["n_copies"], "capture_completed": True,
                "profile_updates": 1, "normalization_warmup_iterations": 50, "compile_warmup_updates": 4,
                "profile_start_iteration": 4, "profile_end_iteration": 5,
                "compiled_signatures_before": [1, 1], "compiled_signatures_after": [1, 1]},
            "execution_identity": execution(capture_start, capture_finish),
            "original_gate_execution": execution(gate_start, gate_finish)}

    def complete(self, work=None):
        work = self.work() if work is None else work
        loaded = {"pid": os.getpid(), "synthetic_separate_cuda_receipt": True}
        with self.live():
            result = backend.complete(self.unit, self.plan, loaded, work)
        return result, loaded, work

    def saved(self, loaded, work):
        return backend.validate_saved(self.unit, self.plan, self.binding, loaded, work)

    def test_separate_preparation_discovers_owner_and_pins_selected_executable(self):
        self.assertEqual(self.expected["kind"], backend.PREPARED_KIND)
        self.assertEqual(self.expected["distribution"]["name"], "Unassumed-Profile-Owner")
        self.assertEqual(self.expected["python_image"]["sha256"], hashlib.sha256(self.python.read_bytes()).hexdigest())
        self.assertEqual(set(backend.helpers()), {"profile_qualification_backend.py", "direct_backend_image.py"})
        self.assertEqual(backend.expected_for_plan(self.plan), self.expected)
        self.assertNotEqual(self.plan["inputs"]["profile_qualification_backend_record"]["path"],
            self.plan["profile_qualification_backend"]["expected"]["path"])

    def test_full_round_trip_on_aa_preserves_zero_credit_and_both_phase_scope(self):
        self.begin(); final, loaded, work = self.complete()
        self.assertEqual(self.saved(loaded, work), final)
        self.assertEqual(final["completed_work_kind"], "full_original_reference_and_profile_capture")
        self.assertFalse(final["admission_granted"])
        self.assertEqual(final["scientific_credit"], 0)

    def test_h100_has_real_distinct_binding_and_two_copy_profile_reference_stays_one(self):
        self.plan.pop("configured_isolation")
        self.plan.update(hardware_class="h100", direct_binding={"synthetic_fixture": True}, gpu_uuid=backend.CARD_ONE)
        self.process["hostname"] = "serval02"
        self.plan["configuration"]["learner"]["n_copies"] = 2
        self.plan["copy_seeds"] = [1, 2]
        self.save_plan_and_binding()
        self.begin(); final, loaded, work = self.complete()
        self.assertEqual(self.saved(loaded, work), final)
        self.assertEqual(work["original_reference"]["copies"], 1)
        self.assertEqual(work["profile_capture"]["copies"], 2)
        self.assertEqual(final["gpu_uuid"], backend.CARD_ONE)

    def test_old_direct_group_or_undeclared_protocols_are_not_qualification(self):
        for field, value in (("kind", "selected_configuration_profile"), ("schema_version", True),
                             ("gpu_uuid", backend.CARD_ONE)):
            changed = copy.deepcopy(self.plan); changed[field] = value
            if field == "gpu_uuid": changed["direct_binding"] = {}
            with self.subTest(field=field), self.assertRaises(ValueError): backend.unit_directory(changed)
        for field in ("direct_backend_image", "group_backend_image", "device_memory_limit_protocol"):
            changed = copy.deepcopy(self.plan); changed[field] = {}
            with self.subTest(field=field), self.assertRaises(ValueError): backend.unit_directory(changed)

    def test_prepin_types_owner_source_and_python_environment_are_strict(self):
        changes = [("schema_version", True), ("created_at_seconds", True), ("selected_python", "relative/bin/python"),
            ("observing_python", "/other/bin/python"), ("producer_sources", {}), ("gpu_operations", False),
            ("scientific_credit", False), ("admission_granted", True)]
        for key, value in changes:
            changed = copy.deepcopy(self.expected); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): backend.validate_expected(changed, str(self.python))
        changed = copy.deepcopy(self.expected)
        changed["distribution"]["record"]["path"] = str(self.root / "foreign/RECORD")
        with self.assertRaises(ValueError): backend.validate_expected(changed, str(self.python))
        with self.assertRaises(ValueError): backend.validate_expected(copy.deepcopy(self.expected), "/other/bin/python")

    def test_changed_installed_package_owner_or_python_fails_before_begin(self):
        self.python.write_bytes(b"changed synthetic executable")
        with self.live(), self.assertRaisesRegex(ValueError, "Python bytes changed"):
            backend.validate_prepared(self.plan)
        self.python.write_bytes(b"synthetic selected Python image; never executed")
        self.plan["qualification_software"]["packages"] = {"Other": "7.3"}
        with self.live(), self.assertRaisesRegex(ValueError, "original software"):
            backend.validate_prepared(self.plan)

    def test_ambiguous_distribution_ownership_is_refused(self):
        other = self.site / "another-7.3.dist-info"
        other.mkdir()
        (other / "METADATA").write_bytes(self.metadata.read_bytes())
        (other / "RECORD").write_bytes(self.record.read_bytes())
        with self.assertRaisesRegex(ValueError, "exactly one distribution owner"):
            backend.files.describe_image(self.site, self.image)

    def test_saved_validation_uses_only_actual_captured_bytes_not_installed_fallback(self):
        self.begin(); final, loaded, work = self.complete()
        self.site.rename(self.site.with_name("moved-original-site-packages"))
        self.assertEqual(self.saved(loaded, work), final)
        (self.unit / backend.CAPTURE / "xla_cuda_plugin.so").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)): self.saved(loaded, work)

    def test_missing_context_wrong_uuid_birth_minor_or_typed_binding_is_refused(self):
        for category, key, value in (("native", "current_context", 0), ("native", "current_context", True),
                ("native", "gpu_uuid", backend.CARD_ONE), ("process", "process_start_time", "1"),
                ("device", "minor", 2), ("device", "character_device", 1)):
            changed = copy.deepcopy(self.binding); changed[category][key] = value
            self.write(self.unit / backend.binding_file(self.plan), changed)
            with self.subTest(category=category, key=key), self.live(), self.assertRaises(ValueError):
                backend.process_join(self.unit, self.plan, changed, current=True)
        self.write(self.unit / backend.binding_file(self.plan), self.binding)

    def test_completed_work_refuses_partial_reference_profile_or_invented_fixture_zero(self):
        work = self.work()
        for target, key, value in (("original_reference", "copies", 2), ("original_reference", "reference_iteration", True),
                ("profile_capture", "capture_completed", False), ("profile_capture", "profile_end_iteration", 4),
                ("profile_capture", "compile_warmup_updates", 0), ("profile_capture", "copies", True)):
            changed = copy.deepcopy(work); changed[target][key] = value
            with self.subTest(target=target, key=key), self.assertRaises(ValueError): backend.completed_work(self.plan, changed)
        for key, value in (("test_fixture_work", 0), ("all_original_gates_passed", False), ("copy_seeds", [2])):
            changed = copy.deepcopy(work); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): backend.completed_work(self.plan, changed)

    def test_work_plan_joins_preserve_boolean_integer_and_container_types(self):
        """The three exact reproduced Boolean/integer substitutions never earn completion."""
        work = self.work()
        self.assertEqual(backend.completed_work(self.plan, work), "full_original_reference_and_profile_capture")
        mutations = []
        changed = copy.deepcopy(work); changed["copy_seeds"] = [True]; mutations.append(changed)
        changed = copy.deepcopy(work); changed["configuration"]["learner"]["n_copies"] = True; mutations.append(changed)
        changed = copy.deepcopy(work); changed["actual_runtime_environment"] = {"synthetic_fixture": 1}; mutations.append(changed)
        for index, changed in enumerate(mutations):
            with self.subTest(reproduced=index), self.assertRaisesRegex(ValueError, "actual separate full-original"):
                backend.completed_work(self.plan, changed)

    def test_tampered_shared_image_metadata_record_and_symlink_never_pass_saved_reader(self):
        self.begin(); _, loaded, work = self.complete()
        for filename in ("xla_cuda_plugin.so", "METADATA", "RECORD"):
            path = self.unit / backend.CAPTURE / filename
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(filename=filename), self.assertRaises(ValueError): self.saved(loaded, work)
            path.write_bytes(original)
        path = self.unit / backend.CAPTURE / "METADATA"
        path.unlink(); path.symlink_to(self.metadata)
        with self.assertRaises(ValueError): self.saved(loaded, work)

    def test_saved_phase_process_chronology_and_maps_are_not_replaceable(self):
        self.begin(); _, loaded, work = self.complete()
        changed = copy.deepcopy(work); changed["original_gate_execution"]["process_start_time"] = "1"
        final = backend.files.document(self.unit / backend.FINAL)
        original = copy.deepcopy(final)
        final["work_record_sha256"] = backend.files.digest(backend.files.encoded(changed))
        self.write(self.unit / backend.FINAL, final)
        with self.assertRaisesRegex(ValueError, "same owning Python"): self.saved(loaded, changed)
        changed = copy.deepcopy(work)
        changed["original_gate_execution"]["started_at_seconds"] = changed["execution_identity"]["started_at_seconds"] - 1
        final["work_record_sha256"] = backend.files.digest(backend.files.encoded(changed))
        self.write(self.unit / backend.FINAL, final)
        with self.assertRaisesRegex(ValueError, "surround both"): self.saved(loaded, changed)
        final = copy.deepcopy(original); final["maps_after_numerical_work"]["raw"] += "malformed\n"
        self.write(self.unit / backend.FINAL, final)
        with self.assertRaises(ValueError): self.saved(loaded, work)

    def test_failure_presence_retry_and_changed_helper_prevent_success(self):
        self.begin(); _, loaded, work = self.complete()
        with self.live(), self.assertRaisesRegex(ValueError, "no retry"): backend.begin(self.unit, self.plan, self.binding)
        self.write(self.unit / backend.FAILURE, {})
        with self.assertRaisesRegex(ValueError, "failed backend"): self.saved(loaded, work)
        (self.unit / backend.FAILURE).unlink()
        self.plan["helpers"]["files"][backend.PREFIX + backend.NAME] = "0" * 64
        with self.assertRaisesRegex(ValueError, "exact qualification backend helpers"): backend.expected_for_plan(self.plan)


if __name__ == "__main__":
    unittest.main()
