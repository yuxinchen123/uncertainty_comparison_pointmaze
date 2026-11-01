"""Validate schema3 execution from exact captured files after original machines and paths are unavailable."""

import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import h100_nvl_slurm_evidence as history
import h100_nvl_slurm_qualification as qualification
import h100_nvl_slurm_queue as queue
import h100_nvl_slurm_site_validation as site
import post_comparison_validation as post
from portable_evidence import ExactPathResolver
import test_h100_nvl_slurm_historical_evidence as original_tests
import test_h100_nvl_slurm_qualification as profile_tests


class Schema3EvidenceTests(unittest.TestCase):
    """Reuse original raw/Git fixtures while qualifying a separate target driver and GPU identity."""

    @classmethod
    def setUpClass(cls):
        """Read the frozen numerical source only as committed bytes for the temporary fixture."""
        original_tests.HistoricalEvidenceTests.setUpClass()

    def setUp(self):
        """Create a complete synthetic schema3 policy, both site packets and one completed ordinary case."""
        helper = original_tests.HistoricalEvidenceTests(methodName="test_complete_history_uses_captured_admission_and_never_reader_packages")
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        self.helper = helper
        self.gpu = "GPU-f3f58504-1234-5678-abcd-123456789abc"
        helper.gpu = self.gpu
        self.driver = helper.base / "installed_driver/libcuda.so.580.159.04"
        helper.write(self.driver, b"synthetic actual qualified CUDA driver image")
        helper.loaded["loaded_shared_libraries"][-1] = str(self.driver)
        # Before: committed base12 helper files. After: the same files plus two explicit schema3 metadata files.
        added = (queue.H100_NVL_SLURM_QUALIFICATION_MODULE, "h100_nvl_allocation_observation.py")
        for name in added:
            helper.write(helper.helper_root / history.PREFIX / name, ("# synthetic pinned schema3 helper " + name + "\n").encode())
        subprocess.run(["git", "-C", str(helper.helper_root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(helper.helper_root), "-c", "user.name=Synthetic metadata test", "-c", "user.email=test@example.invalid",
            "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Add explicit schema3 metadata fixture"], check=True)
        commit = subprocess.check_output(["git", "-C", str(helper.helper_root), "rev-parse", "HEAD"], text=True).strip()
        helper.git_roots[str(helper.helper_root)] = commit
        helper.helper_receipt = helper.git_receipt(helper.helper_root, commit, [history.PREFIX + name for name in queue.required_worker_helpers(qualification.PROFILE)])
        helper.site_helpers = helper.git_receipt(helper.helper_root, commit, [history.PREFIX + name for name in site.required_helpers(qualification.PROFILE)])
        first = helper.make_site(helper.run / "remote_setup/passed_site", "111")
        current = helper.make_site(helper.run / "remote_setup/current_allocation", "333")
        self.qualify_site(first, "111")
        self.qualify_site(current, "333")
        helper.policy.update(worker_helpers=helper.helper_receipt, qualification_profile=qualification.PROFILE,
            site_result=helper.receipt(first / "result.json"), approved_targets=[{"hostname": "serval06", "gpu_uuid": self.gpu}])
        helper.write(helper.policy_path, helper.policy)
        helper.make_case(current)
        hardware_path = helper.attempt / "hardware.json"
        hardware = json.loads(hardware_path.read_bytes())
        hardware["gpu_query"] = hardware["gpu_query"].replace(site.DRIVER, qualification.PROFILE["expected_driver"])
        helper.write(hardware_path, hardware)
        helper.execution.update(hardware=helper.receipt(hardware_path), qualification_profile=qualification.PROFILE,
            qualification_observation=self.observation("333", 303, "403"), qualification_cuda_identity=self.cuda_identity(303),
            qualification_driver_image=self.driver_capture(helper.attempt, 303))
        helper.write(helper.attempt / "slurm_execution.json", helper.execution)
        helper.write_intent()

    def observation(self, job, pid, birth):
        """Construct fresh process-bound raw observations with scheduler ID1 and physical minor0 kept separate."""
        raw = profile_tests.fixture()
        text = json.dumps(raw).replace("serval08", "serval06").replace("333", job)
        raw = json.loads(text)
        raw["qualification_profile"] = qualification.PROFILE
        raw.update(pid=pid, process_start_time=birth)
        raw["device_isolation_observations"]["own_placement"].update(pid=pid, process_start_time=birth)
        membership = raw["device_isolation_observations"]["own_placement"]["own_cgroup_file"]
        membership["sha256"] = hashlib.sha256(membership["text"].encode()).hexdigest()
        raw["allocation_environment"]["CUDA_VISIBLE_DEVICES"] = self.gpu
        return raw

    def cuda_identity(self, pid):
        """Record synthetic actual CUDA UUID/PCI and the real temporary image bytes for this process."""
        return {"schema_version": 1, "kind": "same_process_jax_cuda_device_identity", "pid": pid,
            "jax_local_hardware_id": 0, "cuda_device_count": 1, "cuda_device_handle": 0,
            "api_return_codes": {name: 0 for name in ("cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid_v2", "cuDeviceGetPCIBusId")},
            "gpu_uuid": self.gpu, "pci_address": "0000:c1:00.0", "driver_library": str(self.driver),
            "driver_library_sha256": self.helper.receipt(self.driver)["sha256"]}

    def driver_capture(self, directory, pid):
        """Create independent fixture bytes at the exact shared artifact path before its raw proof is hashed."""
        target = Path(directory) / "captured_driver" / self.driver.name
        self.helper.write(target, self.driver.read_bytes())
        return {"schema_version": 1, "kind": "captured_mapped_cuda_driver_image", "pid": pid,
            "original": self.helper.receipt(self.driver), "captured": self.helper.receipt(target), "bytes": target.stat().st_size}

    def qualify_site(self, directory, job):
        """Convert only the temporary site's target observations; retain its original590 canonical reference."""
        helper = self.helper
        plan = json.loads((directory / "site_validation_plan.json").read_bytes())
        plan.update(schema_version=3, qualification_profile=qualification.PROFILE, helpers=helper.site_helpers)
        helper.write(directory / "site_validation_plan.json", plan)
        manifest = json.loads((directory / "manifest.yaml").read_bytes())
        manifest.update(plan=helper.receipt(directory / "site_validation_plan.json"), helpers=helper.site_helpers)
        helper.write(directory / "manifest.yaml", manifest)
        raw = self.observation(job, 90, "99")
        proof = qualification.validate_allocation(raw)
        assignment = json.loads((directory / "site_assignment.json").read_bytes())
        assignment.update(allocation=proof, qualification_observation=helper.write(directory / "qualification_observation.json", raw))
        assignment["owned_job"]["cgroup"] = raw["device_isolation_observations"]["own_placement"]["own_cgroup_file"]["text"]
        helper.write(directory / "site_assignment.json", assignment)
        result = json.loads((directory / "result.json").read_bytes())
        result.update(plan=helper.receipt(directory / "site_validation_plan.json"), assignment=helper.receipt(directory / "site_assignment.json"),
            qualification_profile=qualification.PROFILE)
        for kind in ("learner", "renderer", "probe"):
            path = directory / (kind + "_process.json")
            process = json.loads(path.read_bytes())
            process["qualification_observation"] = self.observation(job, process["pid"], process["process_start_time"])
            helper.write(path, process)
            result["children"][kind]["process"] = helper.receipt(path)
        path = directory / "probe_result.json"
        probe = json.loads(path.read_bytes())
        probe["qualification_cuda_identity"] = self.cuda_identity(probe["pid"])
        probe["qualification_driver_image"] = self.driver_capture(directory, probe["pid"])
        helper.write(path, probe)
        result["probe"] = helper.receipt(path)
        helper.write(directory / "result.json", result)

    def save_execution(self, record):
        """Rehash the test execution and publication intent so rejection checks its actual semantic change."""
        self.helper.write(self.helper.attempt / "slurm_execution.json", record)
        self.helper.write_intent()

    def validate(self, evidence=None):
        """Run the actual historical proof code without current packages, driver queries or live admission."""
        with patch.object(importlib.metadata, "distribution", side_effect=AssertionError("reader packages are forbidden")), \
                patch.object(site, "prepared", side_effect=AssertionError("live site reads are forbidden")), \
                patch.object(site, "sha256", side_effect=AssertionError("reader driver paths are forbidden")), \
                patch.object(self.helper.constants, "prepared", side_effect=AssertionError("live site reads are forbidden")), \
                patch.object(self.helper.constants, "sha256", side_effect=AssertionError("reader driver paths are forbidden")):
            return self.helper.validate(evidence)

    def test_complete_profile_keeps_original_reference_and_records_actual_target(self):
        """Canonical reference590 remains strict while qualified target580 images and process observations pass."""
        evidence = post.Evidence()
        record, result = self.validate(evidence)
        self.assertEqual(record["qualification_profile"], qualification.PROFILE)
        self.assertEqual(result, self.helper.raw_result)
        self.assertEqual(self.helper.reference["driver"], "590.48.01")
        self.assertIn(str(self.helper.attempt / "captured_driver" / self.driver.name), evidence.receipts)
        self.assertNotIn(str(self.driver), evidence.receipts)
        self.assertEqual(len(queue.HELPERS), 12)
        self.assertEqual(len(queue.required_worker_helpers(qualification.PROFILE)), 14)

    def test_strict_mapping_survives_removed_sources_and_rejects_missing_or_changed_driver(self):
        """No reader fallback is allowed even when an original image exists or every old root is deleted."""
        helper = self.helper
        original = post.Evidence()
        self.validate(original)
        manifest = helper.portable_package(original.records())
        contents = json.loads(manifest.read_bytes())
        captured = str(helper.attempt / "captured_driver" / self.driver.name)
        missing = {**contents, "files": [row for row in contents["files"] if row["original_path"] != captured]}
        missing_path = manifest.with_name("missing_driver_mapping.json")
        helper.write(missing_path, missing)
        self.assertTrue(self.driver.exists())
        with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
            self.validate(post.Evidence(ExactPathResolver(missing_path)))
        mapped_capture = ExactPathResolver(manifest).file_path(captured)
        original_bytes = mapped_capture.read_bytes()
        mapped_capture.write_bytes(original_bytes + b"changed while original driver still exists")
        with self.assertRaises((ValueError, KeyError)):
            self.validate(post.Evidence(ExactPathResolver(manifest)))
        mapped_capture.write_bytes(original_bytes)
        for path in list(helper.base.iterdir()):
            if path != manifest.parent:
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        relocated = post.Evidence(ExactPathResolver(manifest))
        record, result = self.validate(relocated)
        self.assertEqual(result, helper.raw_result)
        self.assertEqual(original.records(), relocated.records())
        driver_copy = relocated.physical_file(captured)
        driver_copy.write_bytes(driver_copy.read_bytes() + b"changed")
        with self.assertRaises((ValueError, KeyError)):
            self.validate(post.Evidence(ExactPathResolver(manifest)))

    def test_missing_profile_observation_or_actual_cuda_proof_cannot_fall_back(self):
        """Stripping a new optional-looking marker never admits target580 through the legacy path."""
        for name in ("qualification_profile", "qualification_observation", "qualification_cuda_identity", "qualification_driver_image"):
            changed = copy.deepcopy(self.helper.execution)
            changed.pop(name)
            self.save_execution(changed)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate()

    def test_rehashed_cuda_or_own_process_changes_reject(self):
        """Matching raw hashes cannot replace the exact completed process, device, PCI or scheduler records."""
        for change in (lambda record: record["qualification_cuda_identity"].update(pid=999),
            lambda record: record["qualification_cuda_identity"].update(pci_address="0000:61:00.0"),
            lambda record: record["qualification_observation"]["allocation_environment"].update(SLURM_STEP_GPUS="0"),
            lambda record: record["qualification_observation"].update(pid=999),
            lambda record: record["qualification_observation"]["device_isolation_observations"]["own_placement"].update(cpu_affinity=list(range(8, 16)))):
            changed = copy.deepcopy(self.helper.execution)
            change(changed)
            self.save_execution(changed)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate()

    def test_nonzero_child_rejects_even_with_complete_original_results(self):
        """Target qualification does not change the existing zero-exit requirement before publication."""
        path = self.helper.attempt / "slurm_child_terminal.json"
        terminal = json.loads(path.read_bytes())
        terminal["exit_code"] = 1
        self.helper.write(path, terminal)
        self.helper.write_intent()
        with self.assertRaisesRegex(ValueError, "successful child"):
            self.validate()

    def test_default_cpu_reader_uses_only_shared_captures_after_original_driver_removed(self):
        """Ordinary CPU indexing must not depend on an NVL node's local driver image remaining readable."""
        self.driver.unlink()
        evidence = post.Evidence()
        record, result = self.validate(evidence)
        self.assertEqual(result, self.helper.raw_result)
        self.assertNotIn(str(self.driver), evidence.receipts)
        self.assertIn(record["qualification_driver_image"]["captured"]["path"], evidence.receipts)

    def test_new_reader_closure_is_source_sensitive_and_legacy_worker_tuple_is_unchanged(self):
        """Future readers pin both transitive files, while older verified callers retain their original minimum."""
        prefix = queue.PREFIX
        required = {prefix + name for name in queue.required_worker_helpers(qualification.PROFILE)}
        callers = {prefix + "post_comparison_validation.py": b"H100_NVL_SLURM_QUEUE_MODULE = 'h100_nvl_slurm_queue.py'\n",
            prefix + "h100_nvl_slurm_queue.py": b"H100_NVL_SLURM_QUALIFICATION_MODULE = 'h100_nvl_slurm_qualification.py'\n"}
        self.assertTrue({prefix + name for name in site.QUALIFICATION_HELPERS} <= post.helper_requirements(required, callers))
        callers[prefix + "h100_nvl_slurm_queue.py"] = b"# older worker without schema3 support\n"
        self.assertTrue({prefix + name for name in site.QUALIFICATION_HELPERS}.isdisjoint(post.helper_requirements(required, callers)))


if __name__ == "__main__":
    unittest.main()
