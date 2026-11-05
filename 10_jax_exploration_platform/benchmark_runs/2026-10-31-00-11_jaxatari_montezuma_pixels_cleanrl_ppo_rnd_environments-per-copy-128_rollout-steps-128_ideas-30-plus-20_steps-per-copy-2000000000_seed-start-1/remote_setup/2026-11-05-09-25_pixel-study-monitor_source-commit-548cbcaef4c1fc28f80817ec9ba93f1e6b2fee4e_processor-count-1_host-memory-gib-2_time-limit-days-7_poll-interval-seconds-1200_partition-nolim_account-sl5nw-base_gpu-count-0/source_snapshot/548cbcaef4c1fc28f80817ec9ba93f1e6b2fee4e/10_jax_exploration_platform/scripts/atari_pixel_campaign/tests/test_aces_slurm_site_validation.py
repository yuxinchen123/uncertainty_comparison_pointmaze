"""Verify ACES orchestration boundaries with metadata-only fixtures."""

import hashlib
import importlib.util
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1]
specification = importlib.util.spec_from_file_location("tested_aces_site", SCRIPT / "aces_slurm_site_validation.py")
site = importlib.util.module_from_spec(specification)
specification.loader.exec_module(site)
from test_aces_slurm_qualification import valid_fixture


def full_result_fixture(unit):
    """Create a synthetic complete result chain with real raw observation structure and temporary saved images."""
    old, policy = site.common(), site.qualifier()
    directory = unit / "packet"
    directory.mkdir()
    def save(path, value):
        """Write one synthetic fixture and return its exact current receipt."""
        path.write_text(json.dumps(value))
        return old.receipt(path)
    software = {"installed_backend": {"gpu_libraries": {}}, "asset_receipt": {"fixture": True}}
    for name in ("libcudnn.so.9", "libcudart.so.12"):
        path = unit / name
        path.write_bytes(b"synthetic package image")
        software["installed_backend"]["gpu_libraries"][str(path)] = {"package": "fixture", "version": "fixture", "sha256": old.sha256(path)}
    plan = {"python": "/python", "helpers": {"source_root": "/helpers"}, "numerical_source": {"source_root": "/frozen"},
        "configuration": {"fixture": True}, "cleanrl_reference": {"fixture": True},
        "software_at_prepare": save(directory / "software.json", software)}
    plan_receipt = save(directory / "site_validation_plan.json", plan)
    parent_observation = valid_fixture()
    parent = {"pid": parent_observation["owned_allocation"]["pid"], "process_start_time": parent_observation["owned_allocation"]["process_start_time"]}
    assignment = policy.validate_allocation(parent_observation)
    result = {"schema_version": 1, "kind": "aces_full_site_evidence", "passed": True, "plan": plan_receipt,
        "qualification_profile": policy.PROFILE, "parent_observation": save(unit / "parent_allocation_observation.json", parent_observation),
        "parent_owner": save(unit / "parent_owner.json", parent), "assignment": save(unit / "site_assignment.json", assignment),
        "children": {}, **policy.BOUNDARY}
    for number, kind in enumerate(site.KINDS):
        owner = {"pid": 4000000 + number, "process_start_time": str(100 + number), "parent_pid": parent["pid"],
            "parent_start_time": parent["process_start_time"], "job_id": assignment["job_id"], "step_id": assignment["step_id"], "kind": kind}
        observation = copy.deepcopy(parent_observation)
        selector = assignment["cuda_selector_original"] if kind == "identity" else assignment["gpu_uuid"]
        observation["owned_allocation"]["allocation_environment"]["CUDA_VISIBLE_DEVICES"] = selector
        for record in (observation["owned_allocation"], observation["device_isolation_observations"]["own_placement"]):
            record.update(pid=owner["pid"], process_start_time=owner["process_start_time"])
        allocation = policy.validate_allocation(observation)
        process = {**owner, "source_commit": site.SOURCE_COMMIT, "configuration": plan["configuration"],
            "cuda_selector": selector,
            "runtime_environment": old.RUNTIME, "software": software, "allocation": allocation,
            "allocation_observation": save(unit / (kind + "_allocation_observation.json"), observation),
            "original_command": old.original_command(plan, kind) if kind in ("learner", "renderer") else None,
            "cleanrl_reference": plan["cleanrl_reference"], "gate_environment_policy": old.GATE_ENVIRONMENT_POLICY}
        (unit / (kind + ".log")).write_text("synthetic successful stage\n")
        terminal = {**owner, "exit_code": 0, "log": old.receipt(unit / (kind + ".log")),
            "command": ["/python", "-I", "/helpers/" + site.PREFIX + "aces_slurm_site_validation.py", "child", "--directory", str(directory), "--kind", kind]}
        result["children"][kind] = {name: save(unit / (kind + suffix), value) for name, suffix, value in
            (("owner", "_owner.json", owner), ("process", "_process.json", process), ("terminal", "_terminal.json", terminal))}
        if kind in ("identity", "probe"):
            image = unit / (kind + "_driver/captured_driver/libcuda.so.580.178.04")
            image.parent.mkdir(parents=True)
            image.write_bytes(b"synthetic driver image")
            digest = old.sha256(image)
            identity = {"schema_version": 1, "kind": "same_process_aces_jax_cuda_identity", "pid": owner["pid"],
                "cuda_device_count": 1, "jax_local_hardware_id": 0, "cuda_device_handle": 0,
                "api_return_codes": {name: 0 for name in ("cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid_v2", "cuDeviceGetPCIBusId")},
                "gpu_uuid": allocation["gpu_uuid"], "pci_address": allocation["pci_address"],
                "driver_library": "/missing-node-local/libcuda.so.580.178.04", "driver_library_sha256": digest}
            capture = {"schema_version": 1, "kind": "captured_mapped_cuda_driver_image", "pid": owner["pid"],
                "original": {"path": identity["driver_library"], "sha256": digest}, "captured": old.receipt(image), "bytes": image.stat().st_size}
            value = {**process, "completed": True, "identity": identity, "driver_capture": capture, **policy.BOUNDARY}
            if kind == "probe":
                libraries = software["installed_backend"]["gpu_libraries"]
                paths = {"exploration_platform.atari_pixels." + name: "/frozen/10_jax_exploration_platform/src/exploration_platform/atari_pixels/" + name + ".py"
                    for name in ("config", "environment", "training", "networks", "state")}
                value.update(kind="original_aces_reference_site_probe", iteration=1, dtypes=old.EXPECTED_DTYPES,
                    formal_measurement=False, capacity_evidence=False,
                    finite_state={"passed": True, "floating_leaves_checked": 12, "scope": "complete updated PixelTrainState"},
                    numerical_module_paths=paths, loaded_libraries={"pid": owner["pid"], "platform": "gpu", "platform_version": "fixture",
                        "cudaRuntimeGetVersion": 12060, "cudnnGetVersion": 90000, "verified_package_libraries": libraries,
                        "loaded_shared_libraries": [*libraries, identity["driver_library"]],
                        "cudnn_library": next(path for path in libraries if "libcudnn" in path),
                        "cuda_runtime_library": next(path for path in libraries if "libcudart" in path)})
            result[kind] = save(unit / (kind + "_result.json"), value)
    return directory, plan, result


class SiteTests(unittest.TestCase):
    """Check saved preparation, selector ordering and portable captured-driver validation."""

    def test_held_aces_command_has_no_uva_placement(self):
        """Preparation names the exact ACES account and typed one-GPU shape without submitting."""
        command = site.submission_command("/run", "/run/remote_setup/qualification")
        self.assertIn("--hold", command)
        self.assertIn("--account=158590754315", command)
        self.assertIn("--gres=gpu:h100:1", command)
        self.assertIn("--mem=32G", command)
        self.assertFalse(any("constraint" in value or "nodelist" in value or "sl5nw_base" in value for value in command))

    def test_selector_changes_only_after_identity_stage(self):
        """The first child keeps Slurm's selector and following children receive the proved UUID."""
        plan = {"asset_record": {"root": "/assets"}, "cleanrl_reference": {"file": {"path": "/cleanrl/model.py"}},
            "numerical_source": {"source_root": "/source"}}
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True):
            self.assertEqual(site.child_environment(plan, {"gpu_uuid": "GPU-owned"}, "identity")["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(site.child_environment(plan, {"gpu_uuid": "GPU-owned"}, "learner")["CUDA_VISIBLE_DEVICES"], "GPU-owned")
        self.assertEqual(site.KINDS, ("identity", "learner", "renderer", "probe"))

    def test_credit_types_and_missing_boundary_fields_fail(self):
        """Site evidence cannot turn booleans into numeric zero or omit permissions."""
        boundary = site.qualifier().BOUNDARY
        site.require_boundary(boundary)
        for change in ({"workers_permitted": 0}, {"scientific_hypothesis_credit": False}, {"idea_credit": 1}):
            with self.assertRaises(ValueError):
                site.require_boundary({**boundary, **change})
        with self.assertRaises(ValueError):
            site.require_boundary({})

    def test_portable_driver_verification_never_reads_node_original(self):
        """A saved driver copy is sufficient after the original node-local path disappears."""
        with tempfile.TemporaryDirectory() as folder:
            unit = Path(folder)
            capture_dir = unit / "identity_driver/captured_driver"
            capture_dir.mkdir(parents=True)
            target = capture_dir / "libcuda.so.580.178.04"
            target.write_bytes(b"synthetic captured driver fixture")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            allocation = {"gpu_uuid": "GPU-owned", "pci_address": "0000:14:00.0"}
            identity = {"schema_version": 1, "kind": "same_process_aces_jax_cuda_identity", "pid": 123,
                "jax_local_hardware_id": 0, "cuda_device_count": 1, "cuda_device_handle": 0,
                "api_return_codes": {name: 0 for name in ("cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid_v2", "cuDeviceGetPCIBusId")},
                "gpu_uuid": "GPU-owned", "pci_address": "0000:14:00.0",
                "driver_library": "/nonexistent-node-local/libcuda.so.580.178.04", "driver_library_sha256": digest}
            capture = {"schema_version": 1, "kind": "captured_mapped_cuda_driver_image", "pid": 123,
                "original": {"path": identity["driver_library"], "sha256": digest},
                "captured": {"path": str(target), "sha256": digest}, "bytes": target.stat().st_size}
            process = {"pid": 123, "process_start_time": "42", "allocation": allocation, "kind": "identity"}
            value = {**process, "completed": True, "identity": identity, "driver_capture": capture, **site.qualifier().BOUNDARY}
            self.assertEqual(site.verify_identity_result(value, process, unit, "identity"), identity)
            target.write_bytes(b"changed capture")
            with self.assertRaises(ValueError):
                site.verify_identity_result(value, process, unit, "identity")

    def test_original_gate_commands_and_precision_are_reused(self):
        """Original complete test paths and numerical controls retain their established values."""
        old = site.common()
        plan = {"python": "/python", "numerical_source": {"source_root": "/frozen"}}
        self.assertEqual(old.original_command(plan, "learner"), ["/python", "-m", "pytest", "-q",
            "/frozen/10_jax_exploration_platform/tests/atari_pixels", "/frozen/10_jax_exploration_platform/tests/test_atari_pixel_environment.py"])
        self.assertEqual(old.RUNTIME["JAX_ENABLE_X64"], "true")
        self.assertEqual(old.RUNTIME["JAX_DEFAULT_MATMUL_PRECISION"], "highest")
        self.assertEqual(old.RUNTIME["XLA_FLAGS"], "--xla_gpu_exclude_nondeterministic_ops=true")
        self.assertEqual(old.EXPECTED_DTYPES["parameters"], ["float32"])
        self.assertEqual(old.EXPECTED_DTYPES["observation_statistics"], ["float64"])

    def test_complete_saved_chain_and_negative_result_paths(self):
        """A complete portable synthetic chain passes while substituted paths and credit changes fail."""
        with tempfile.TemporaryDirectory() as folder:
            unit = Path(folder)
            directory, plan, result = full_result_fixture(unit)
            proof = site.verify_result(directory, unit, plan, result)
            self.assertIs(proof["passed"], True)
            self.assertIs(proof["workers_permitted"], False)
            for change in ("credit", "parent", "probe", "schema", "profile_schema"):
                modified = copy.deepcopy(result)
                if change == "credit":
                    modified["idea_credit"] = 1
                elif change == "schema":
                    modified["schema_version"] = True
                elif change == "profile_schema":
                    modified["qualification_profile"]["schema_version"] = True
                else:
                    modified["parent_owner" if change == "parent" else "probe"] = modified["assignment"]
                with self.assertRaises(ValueError):
                    site.verify_result(directory, unit, plan, modified)

    def test_full_chain_rejects_observer_process_substitution(self):
        """Matching child receipts still cannot replace its fresh observer with another process."""
        with tempfile.TemporaryDirectory() as folder:
            unit = Path(folder)
            directory, plan, result = full_result_fixture(unit)
            old = site.common()
            path = unit / "learner_process.json"
            process = json.loads(path.read_text())
            observation_path = unit / "learner_allocation_observation.json"
            observation = json.loads(observation_path.read_text())
            observation["owned_allocation"]["pid"] = 99999
            observation_path.write_text(json.dumps(observation))
            process["allocation_observation"] = old.receipt(observation_path)
            path.write_text(json.dumps(process))
            result["children"]["learner"]["process"] = old.receipt(path)
            with self.assertRaisesRegex(ValueError, "observer process"):
                site.verify_result(directory, unit, plan, result)

    def test_actual_selector_is_required_in_process_and_raw_observation(self):
        """Missing or substituted selectors cannot qualify even with otherwise matching hash receipts."""
        for change in ("process", "raw"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                unit = Path(folder)
                directory, plan, result = full_result_fixture(unit)
                path = unit / "learner_process.json"
                process = json.loads(path.read_text())
                if change == "process":
                    process.pop("cuda_selector")
                else:
                    raw_path = unit / "learner_allocation_observation.json"
                    raw = json.loads(raw_path.read_text())
                    raw["owned_allocation"]["allocation_environment"]["CUDA_VISIBLE_DEVICES"] = "0"
                    raw_path.write_text(json.dumps(raw))
                    process["allocation_observation"] = site.receipt(raw_path)
                path.write_text(json.dumps(process))
                result["children"]["learner"]["process"] = site.receipt(path)
                with self.assertRaisesRegex(ValueError, "selector"):
                    site.verify_result(directory, unit, plan, result)

    def test_saved_numerical_result_retains_every_process_field(self):
        """A probe cannot retain PID while substituting source, runtime, software or original metadata."""
        with tempfile.TemporaryDirectory() as folder:
            unit = Path(folder)
            directory, plan, result = full_result_fixture(unit)
            process = json.loads((unit / "probe_process.json").read_text())
            value = json.loads((unit / "probe_result.json").read_text())
            for key in ("source_commit", "configuration", "runtime_environment", "software", "cleanrl_reference"):
                changed = {**value, key: "foreign"}
                with self.subTest(key=key), self.assertRaises(ValueError):
                    site.verify_identity_result(changed, process, unit, "probe")

    def test_library_pid_and_cross_stage_driver_bytes_are_bound(self):
        """Foreign library processes and a later different driver image both invalidate the saved chain."""
        with tempfile.TemporaryDirectory() as folder:
            unit = Path(folder)
            directory, plan, result = full_result_fixture(unit)
            probe = json.loads((unit / "probe_result.json").read_text())
            changed = {**probe["loaded_libraries"], "pid": 999}
            with self.assertRaises(ValueError):
                site.validate_loaded(changed, probe["software"]["installed_backend"], probe["identity"])
            image = unit / "probe_driver/captured_driver/libcuda.so.580.178.04"
            image.write_bytes(b"a different driver image with the same version filename")
            digest = site.common().sha256(image)
            probe["identity"]["driver_library_sha256"] = digest
            probe["driver_capture"]["original"]["sha256"] = digest
            probe["driver_capture"]["captured"]["sha256"] = digest
            probe["driver_capture"]["bytes"] = image.stat().st_size
            (unit / "probe_result.json").write_text(json.dumps(probe))
            result["probe"] = site.receipt(unit / "probe_result.json")
            with self.assertRaisesRegex(ValueError, "different actual CUDA driver"):
                site.verify_result(directory, unit, plan, result)

    def test_symlink_artifact_and_parent_are_rejected_even_with_identical_bytes(self):
        """An exact known child artifact cannot resolve to an external file or through a linked directory."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            unit = root / "unit"
            unit.mkdir()
            directory, plan, result = full_result_fixture(unit)
            path = unit / "learner_process.json"
            external = root / "external.json"
            external.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(external)
            result["children"]["learner"]["process"] = site.common().receipt(path)
            with self.assertRaisesRegex(ValueError, "ordinary paths"):
                site.verify_result(directory, unit, plan, result)
            linked = root / "linked-unit"
            linked.symlink_to(unit, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "ordinary paths"):
                site.ordinary_path(linked, directory=True)

    def test_manifest_kind_cluster_and_schema_fail_before_inputs_are_read(self):
        """A foreign preparation header cannot reach the helper or numerical stage."""
        for key, value in (("schema_version", True), ("kind", "foreign"), ("cluster", "uva")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder:
                path = Path(folder)
                manifest = {"schema_version": 1, "kind": "aces_full_site_qualification_preparation", "cluster": "aces"}
                manifest[key] = value
                (path / "manifest.yaml").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "manifest"):
                    site.prepared(path)


if __name__ == "__main__":
    unittest.main()
