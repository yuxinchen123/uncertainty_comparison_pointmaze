"""Validate deployment evidence with synthetic Git, binary, hardware and timing records only."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import validate_training_deployment as deployment
import portable_evidence


class TrainingDeploymentTests(unittest.TestCase):
    """Keep NVL study proof separate from fresh PCIe deployment and independent final seeds."""

    def setUp(self):
        """Build a tiny detached source and a complete synthetic global verification fixture."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "numerical_source"
        repository = SCRIPTS.parents[2]
        relatives = ["10_jax_exploration_platform/" + name for name in (
            "src/exploration_platform/atari_pixels/config.py", "src/exploration_platform/atari_pixels/environment.py")]
        relatives += [deployment.PREFIX + name for name in ("paired_benchmark.py", "ideas.py", "worker_environment.sh",
            "measure_selected.py", "diagnose_gpu_execution.py", "post_comparison_validation.py", "train_optimized.py")]
        for relative in relatives:
            target = self.source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((repository / relative).read_bytes())
        self.commit = self.commit_source(self.source)
        self.public = self.directory / "jaxtari_source"
        python = self.public / "src/jaxtari/__init__.py"
        python.parent.mkdir(parents=True)
        python.write_text('"""Synthetic public package fixture, never imported."""\n')
        public_commit = self.commit_source(self.public)
        patcher = patch.object(deployment, "JAXTARI_COMMIT", public_commit)
        patcher.start()
        self.addCleanup(patcher.stop)
        package_versions = {name: "synthetic-version" for name in deployment.training.PACKAGE_NAMES}
        contents = python.read_bytes()
        files = {"jaxtari/__init__.py": {"git_blob": hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest(),
                                             "sha256": hashlib.sha256(contents).hexdigest()}}
        self.installed = {"packages": package_versions, "jaxtari_source": {"source_root": str(self.public),
            "source_commit": public_commit, "version": package_versions["jaxtari"], "python_files": files,
            "tree_digest": deployment.validation.canonical_digest(files)}, "gpu_libraries": {}}
        for name, package in (("libcudnn.so.9", "nvidia-cudnn-cu12"), ("libcudart.so.12", "nvidia-cuda-runtime-cu12")):
            path = self.directory / "libraries" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(("synthetic shared-library bytes " + name).encode())
            self.installed["gpu_libraries"][str(path)] = {"package": package, "version": package_versions[package],
                                                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        paths = list(self.installed["gpu_libraries"])
        self.loaded = {"platform": "gpu", "platform_version": "synthetic CUDA platform", "loaded_shared_libraries": paths,
            "cudnn_library": paths[0], "cuda_runtime_library": paths[1], "cudnnGetVersion": 92400,
            "cudaRuntimeGetVersion": 12090, "verified_package_libraries": self.installed["gpu_libraries"]}
        spec = importlib.util.spec_from_file_location("deployment_global_fixture", SCRIPTS / "tests/test_post_comparison_validation.py")
        fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        self.fixture = fixtures.PostComparisonTests("test_all_five_gates_are_derived_from_complete_evidence")
        self.addCleanup(self.fixture.doCleanups)
        with patch.object(fixtures, "SCRIPTS", self.source / "10_jax_exploration_platform/scripts/atari_pixel_campaign"):
            self.fixture.setUp()
        self.runtime = self.fixture.runtime
        self.actual_runtime = {**self.runtime, "JAX_ENABLE_COMPILATION_CACHE": "false", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "",
                               "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
                               "NVIDIA_VISIBLE_DEVICES": "GPU-target"}
        self.core = deployment.diagnostic.core_identity(self.source, self.commit)
        self.helpers = {"source_root": str(self.source), "source_commit": self.commit,
            "files": {relative: hashlib.sha256((self.source / relative).read_bytes()).hexdigest() for relative in deployment.REQUIRED_HELPERS}}
        self.paired = {"source_root": str(self.source), "source_commit": self.commit,
            "files": {deployment.PREFIX + name: hashlib.sha256((self.source / deployment.PREFIX / name).read_bytes()).hexdigest()
                      for name in ("paired_benchmark.py", "ideas.py")}}
        self.asset_files = {"jaxtari/sprites/player.npy": "a" * 64, "jaxtari/fonts/font.npy": "b" * 64}
        self.asset_manifest = {"schema_version": 1, "source_commit": public_commit, "files": self.asset_files}
        self.global_asset_record = self.assets_record("global")
        self.target_asset_record = self.assets_record("target")
        confirmation = Path(self.fixture.specification["contexts"]["h100"]["confirmation_record"])
        result = json.loads(confirmation.read_text())
        result.update(helpers=self.helpers, measurement_mode="confirmation")
        self.write(confirmation, result)
        global_plan = {"schema_version": 1, "kind": "selected_configuration_measurement", "mode": "confirmation",
            "source_commit": self.commit, "hardware_class": "h100", "runtime_environment": self.runtime,
            "configuration": result["candidate"], "reference_configuration": result["reference"],
            "helpers": self.helpers, "asset_record": self.global_asset_record}
        self.write_prepared_plan(confirmation.parent, global_plan)
        self.write_asset_receipt(confirmation.parent, self.global_asset_record)
        self.write_helpers(confirmation.parent)
        final = deployment.validation.validate_final_verification(self.fixture.specification)
        self.final_path = self.write(self.directory / "global_final_verification.json", final)
        self.run = self.directory / "target_campaign"
        self.target = {"server": "aces", "hostnames": ["ac049", "ac055"], "gpu_model": "NVIDIA H100 PCIe",
                       "total_memory_bytes": 1000 * 2 ** 20, "driver_version": "synthetic-driver"}
        self.process_counter = 5000
        first = self.target_record("correctness", 2, "target_correctness")
        second = self.target_record("confirmation", 2, "target_confirmation")
        points = [self.target_record("packing-point", count, "target_packing_" + str(count), seconds)
                  for count, seconds in ((1, 1.), (2, 1.), (4, 1.9))]
        repeats = [self.target_record("packing-point", count, "target_repeat_" + str(count), seconds)
                   for count, seconds in ((2, 1.), (4, 1.9))]
        self.selected = json.loads(json.dumps(self.fixture.configuration))
        self.selected["learner"]["n_copies"] = 2
        self.seeds = [1001, 1002]
        seed_row = {"unit_id": "final_pcie_policies", "hardware_class": "h100", "copy_seeds": self.seeds,
                    "configuration_digest": deployment.validation.canonical_digest(self.selected)}
        self.seed_path = self.write(self.directory / "seed_manifest.json", {"schema_version": 1,
            "kind": "final_policy_seed_blocks", "units": {seed_row["unit_id"]: seed_row,
                "final_a4500_policies": {"unit_id": "final_a4500_policies", "hardware_class": "jaguar03_rtx_a4500",
                    "copy_seeds": [1, 2], "configuration_digest": "a" * 64}}})
        self.specification = {"schema_version": 1, "final_verification": str(self.final_path), "source_root": str(self.source),
            "source_commit": self.commit, "hardware_class": "h100", "runtime_environment": self.runtime,
            "campaign_run_dir": str(self.run), "target": self.target, "correctness_record": str(first), "confirmation_record": str(second),
            "packing_records": list(map(str, points)), "packing_confirmation_records": list(map(str, repeats)), "selected_copies": 2,
            "unit_id": "final_pcie_policies", "copy_seeds": self.seeds, "seed_manifest": str(self.seed_path)}

    def commit_source(self, root):
        """Create one real detached synthetic Git tree for source and public-package checks."""
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(root), "-c", "user.name=Synthetic deployment test",
            "-c", "user.email=synthetic@example.invalid", "-c", "core.hooksPath=/dev/null",
            "commit", "-qm", "Create synthetic deployment source"], check=True)
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        subprocess.run(["git", "-C", str(root), "checkout", "-q", "--detach", commit], check=True)
        return commit

    def write(self, path, value):
        """Publish one independently hashable synthetic JSON artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def assets_record(self, name):
        """Create identical manifest bytes for different roots without creating any actual asset files."""
        manifest = self.write(self.directory / (name + "_asset_metadata/manifest.json"), self.asset_manifest)
        return {"root": str(self.directory / (name + "_assets")), "manifest": str(manifest),
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}

    def write_asset_receipt(self, directory, record):
        """Retain the actual producer receipt's exact root/manifest/tree/source/count schema."""
        manifest = json.loads(Path(record["manifest"]).read_text())
        return self.write(directory / "asset_receipt.json", {"root": record["root"], "manifest_sha256": record["sha256"],
            "source_commit": manifest["source_commit"], "file_count": len(manifest["files"]),
            "tree_digest": deployment.validation.canonical_digest(manifest["files"])})

    def write_prepared_plan(self, directory, plan):
        """Bind each fixture plan to its actual serialized bytes."""
        plan_path = self.write(directory / "measurement_plan.json", plan)
        self.write(directory / "measurement_prepared.json", {"plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest()})

    def write_helpers(self, directory):
        """Copy the exact committed producer helpers beside each fixture plan."""
        for relative in self.helpers["files"]:
            path = directory / "code" / Path(relative).name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((self.source / relative).read_bytes())

    def target_record(self, mode, copies, name, seconds=1.):
        """Create actual-shape immutable selected-measurement plans and raw-bound backend records."""
        self.process_counter += 1
        pid = self.process_counter
        configuration = copy.deepcopy(self.fixture.configuration)
        configuration["learner"]["n_copies"] = copies
        reference = copy.deepcopy(configuration)
        reference["learner"]["rollout_unroll"] = 1
        hardware = {"hostname": "ac049", "pid": pid, "process_start_time": str(pid), "gpu_uuid": "GPU-target",
            "gpu_query": "GPU-target, NVIDIA H100 PCIe, 1000 MiB, synthetic-driver", "slurm_job_id": "9999",
            "runtime_environment": self.actual_runtime}
        loaded = {**self.loaded, "pid": pid}
        directory = self.run / "data" / name
        plan = {"schema_version": 1, "kind": "selected_configuration_measurement", "mode": mode,
            "source_root": str(self.source), "source_commit": self.commit, "hardware_class": "h100",
            "campaign_run_dir": str(self.run), "unit_id": name, "compilation_cache_policy": "disabled",
            "runtime_environment": self.runtime, "core": self.core, "paired_source": self.paired, "helpers": self.helpers,
            "configuration": configuration, "reference_configuration": None if mode == "packing-point" else reference,
            "copy_seeds": list(range(1, copies + 1)), "gpu_uuid": "GPU-target", "rounds": 7, "iterations": 10,
            "minimum_updates_per_round": 10, "asset_record": self.target_asset_record}
        self.write_prepared_plan(directory, plan)
        self.write_asset_receipt(directory, self.target_asset_record)
        self.write_helpers(directory)
        if mode == "packing-point":
            result = self.fixture.packing("h100", copies, seconds, pid)
            result["configuration"] = configuration
        else:
            result = self.fixture.comparison("h100")
            result.update(candidate=configuration, reference=reference, copies=copies)
            for pair in result["rounds"]:
                pair["reference"] = self.fixture.row(copies, 2., gpu="GPU-target")
                pair["candidate"] = self.fixture.row(copies, 1., gpu="GPU-target")
            if mode == "correctness":
                result.pop("rounds")
                result.update(measurement_valid=False, decision="correctness_passed")
        for row in result.get("rounds", []) if mode == "packing-point" else []:
            row["memory"]["gpu_uuid"] = "GPU-target"
            for sample in row["memory"]["device_memory_samples"]:
                sample["gpu_uuid"] = "GPU-target"
        result.update(hardware=hardware, installed_backend=self.installed, loaded_gpu_libraries=loaded,
            runtime_environment=self.runtime, source_commit=self.commit, helpers=self.helpers, measurement_mode=mode,
            copy_seeds=plan["copy_seeds"], protocol={"rounds": 7, "requested_minimum_updates_per_round": 10, "actual_updates_per_round": 10},
            execution_identity={"hostname": "ac049", "pid": pid, "process_start_time": str(pid),
                "started_at_seconds": 10000. + pid, "finished_at_seconds": 10002. + pid})
        self.write(directory / "hardware.json", hardware)
        self.write(directory / "installed_backend.json", self.installed)
        return self.write(directory / "result.json", result)

    def change_result(self, key, change, update_hardware=False, update_backend=False):
        """Apply one explicit fixture mutation while optionally preserving raw/sidecar equality."""
        path = Path(self.specification[key])
        result = json.loads(path.read_text())
        change(result)
        self.write(path, result)
        if update_hardware:
            self.write(path.parent / "hardware.json", result["hardware"])
        if update_backend:
            self.write(path.parent / "installed_backend.json", result["installed_backend"])

    def test_actual_pcie_evidence_selects_its_own_copy_count_and_backend(self):
        """The NVL-selected single policy does not override two-policy PCIe packing evidence."""
        with patch.object(Path, "rglob", side_effect=AssertionError("deployment must not rehash individual assets")):
            receipt = deployment.validate_deployment(self.specification)
        self.assertEqual(receipt["configuration"], self.selected)
        self.assertEqual(receipt["copy_seeds"], self.seeds)
        self.assertEqual(receipt["packing"]["selected_copies"], 2)
        self.assertEqual(receipt["target"]["gpu_model"], "NVIDIA H100 PCIe")
        self.assertEqual(set(receipt["backend"]["packages"]), set(deployment.training.PACKAGE_NAMES))
        self.assertEqual(len(receipt["executions"]), 7)
        self.assertEqual(receipt["assets"]["tree_digest"], deployment.validation.canonical_digest(self.asset_files))
        self.assertNotEqual(receipt["global_asset_proof"]["root"], receipt["executions"][0]["asset_proof"]["root"])
        recorded = {entry["path"] for entry in receipt["evidence"]}
        for proof in [receipt["global_asset_proof"], *[row["asset_proof"] for row in receipt["executions"]]]:
            self.assertTrue({proof["measurement_plan"], proof["asset_receipt"], proof["manifest"]} <= recorded)
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_global_raw_gate_cannot_be_replaced_with_old_success_flags(self):
        """Deployment always rebuilds the original fifty-idea and A4500-scaling proof."""
        registry_path = Path(self.fixture.specification["study_registry"])
        registry = json.loads(registry_path.read_text())
        registry["adaptive_batches"][-1]["hypotheses"][-1]["results"]["h100"] = []
        self.write(registry_path, registry)
        with self.assertRaisesRegex(ValueError, "unfinished mechanism"):
            deployment.validate_deployment(self.specification)

    def test_missing_global_asset_proof_or_manifest_inside_assets_is_rejected(self):
        """A global success flag cannot replace its actual producer receipt and external manifest."""
        confirmation = Path(self.fixture.specification["contexts"]["h100"]["confirmation_record"])
        receipt = confirmation.parent / "asset_receipt.json"
        original = receipt.read_bytes()
        receipt.unlink()
        with self.assertRaises(FileNotFoundError):
            deployment.validate_deployment(self.specification)
        receipt.write_bytes(original)
        plan = json.loads((confirmation.parent / "measurement_plan.json").read_text())
        inside = Path(plan["asset_record"]["root"]) / "manifest.json"
        self.write(inside, self.asset_manifest)
        plan["asset_record"]["manifest"] = str(inside)
        self.write_prepared_plan(confirmation.parent, plan)
        with self.assertRaisesRegex(ValueError, "outside its asset root"):
            deployment.validate_deployment(self.specification)

    def test_target_asset_manifest_bytes_are_pinned_by_the_prepared_plan(self):
        """Even changed JSON whitespace invalidates the manifest SHA named by the target plan."""
        path = Path(self.target_asset_record["manifest"])
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "manifest differs from its prepared SHA256"):
            deployment.validate_deployment(self.specification)

    def test_target_asset_receipt_fields_must_match_its_actual_manifest(self):
        """Tree/source/file-count claims are derived from the manifest rather than trusted separately."""
        path = Path(self.specification["correctness_record"]).parent / "asset_receipt.json"
        original = json.loads(path.read_text())
        for name, value in (("tree_digest", "c" * 64), ("source_commit", "d" * 40), ("file_count", True)):
            self.write(path, {**original, name: value})
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "game-asset receipt disagrees"):
                deployment.validate_deployment(self.specification)

    def test_a_consistent_target_asset_tree_still_must_match_the_global_proof(self):
        """Another internally valid asset manifest cannot replace the globally selected game pixels."""
        directory = Path(self.specification["correctness_record"]).parent
        manifest = copy.deepcopy(self.asset_manifest)
        manifest["files"]["jaxtari/sprites/player.npy"] = "c" * 64
        path = self.write(self.directory / "alternate_asset_metadata/manifest.json", manifest)
        record = {**self.target_asset_record, "manifest": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        plan = json.loads((directory / "measurement_plan.json").read_text())
        plan["asset_record"] = record
        self.write_prepared_plan(directory, plan)
        self.write_asset_receipt(directory, record)
        with self.assertRaisesRegex(ValueError, "target game assets differ from the global"):
            deployment.validate_deployment(self.specification)

    def test_model_driver_memory_and_bound_sidecars_cannot_change(self):
        """A copied NVL result or changed hardware sidecar cannot become PCIe deployment evidence."""
        self.change_result("confirmation_record", lambda result: result["hardware"].update(gpu_query=
            "GPU-target, NVIDIA H100 NVL, 1000 MiB, synthetic-driver"), update_hardware=True)
        with self.assertRaisesRegex(ValueError, "node/model/memory/driver"):
            deployment.validate_deployment(self.specification)

    def test_changed_sidecar_and_missing_package_map_are_rejected(self):
        """Backend evidence must come from the exact raw result, with every installed package present."""
        path = Path(self.specification["confirmation_record"])
        sidecar = json.loads((path.parent / "hardware.json").read_text())
        sidecar["pid"] += 1
        self.write(path.parent / "hardware.json", sidecar)
        with self.assertRaisesRegex(ValueError, "exact hardware"):
            deployment.validate_deployment(self.specification)
        result = json.loads(path.read_text())
        self.write(path.parent / "hardware.json", result["hardware"])
        self.change_result("confirmation_record", lambda result: result["installed_backend"]["packages"].pop("numpy"), update_backend=True)
        with self.assertRaisesRegex(ValueError, "every required installed package"):
            deployment.validate_deployment(self.specification)

    def test_target_cannot_reuse_an_original_nvl_process_identity(self):
        """Changing paths/model labels cannot make an original execution a fresh target run."""
        original = json.loads(self.final_path.read_text())
        record = json.loads(Path(original["specification"]["contexts"]["h100"]["confirmation_record"]).read_text())
        identity = record["execution_identity"]
        self.target["hostnames"].append(identity["hostname"])
        def reuse(result):
            """Keep all sidecars bound while copying the original process tuple."""
            result["execution_identity"] = identity
            result["hardware"].update({name: identity[name] for name in ("hostname", "pid", "process_start_time")})
            result["loaded_gpu_libraries"]["pid"] = identity["pid"]
        self.change_result("correctness_record", reuse, update_hardware=True)
        with self.assertRaisesRegex(ValueError, "original/global"):
            deployment.validate_deployment(self.specification)

    def test_confirmation_window_and_packing_selection_stay_strict(self):
        """One-update confirmation and a caller-selected larger packing count cannot pass."""
        path = Path(self.specification["confirmation_record"])
        original = path.read_bytes()
        self.change_result("confirmation_record", lambda result: result["rounds"][0]["candidate"].update(iterations=1))
        with self.assertRaises(ValueError):
            deployment.validate_deployment(self.specification)
        path.write_bytes(original)
        self.specification["selected_copies"] = 4
        with self.assertRaisesRegex(ValueError, "smaller copy count"):
            deployment.validate_deployment(self.specification)

    def test_global_seed_overlap_is_rejected_and_disjoint_additions_preserve_receipt(self):
        """Reserved H100 seeds remain independent from every final A4500 unit."""
        before = deployment.validate_deployment(self.specification)
        manifest = json.loads(self.seed_path.read_text())
        manifest["units"]["another_final_unit"] = {"unit_id": "another_final_unit", "hardware_class": "h100",
            "copy_seeds": [2001], "configuration_digest": "b" * 64}
        self.write(self.seed_path, manifest)
        self.assertEqual(deployment.validate_deployment(self.specification), before)
        manifest["units"]["final_a4500_policies"]["copy_seeds"] = [1001, 1002]
        self.write(self.seed_path, manifest)
        with self.assertRaisesRegex(ValueError, "globally disjoint"):
            deployment.validate_deployment(self.specification)

    def test_changed_binary_or_jax_tari_tree_cannot_reuse_backend_receipt(self):
        """Package labels cannot replace rehashed GPU images and pinned public Python blobs."""
        image = Path(next(iter(self.installed["gpu_libraries"])))
        original = image.read_bytes()
        image.write_bytes(b"changed CUDA library bytes")
        with self.assertRaisesRegex(ValueError, "recorded bytes"):
            deployment.validate_deployment(self.specification)
        image.write_bytes(original)
        self.change_result("confirmation_record", lambda result: result["installed_backend"]["jaxtari_source"].update(tree_digest="a" * 64), update_backend=True)
        with self.assertRaisesRegex(ValueError, "JAXtari source/version/tree digest"):
            deployment.validate_deployment(self.specification)

    def test_actual_backend_match_allows_new_uuid_in_the_same_validated_group(self):
        """A continuation may use another physical card while preserving model/driver/software/capacity."""
        receipt = deployment.validate_deployment(self.specification)
        hardware = json.loads(Path(self.specification["confirmation_record"]).read_text())["hardware"]
        hardware.update(hostname="ac055", pid=9000, process_start_time="9000", gpu_uuid="GPU-new",
                        gpu_query="GPU-new, NVIDIA H100 PCIe, 1000 MiB, synthetic-driver")
        hardware["runtime_environment"]["NVIDIA_VISIBLE_DEVICES"] = "GPU-new"
        matched = deployment.match_actual_backend(receipt, "aces", hardware, self.installed, {**self.loaded, "pid": 9000})
        self.assertEqual(matched["gpu_uuid"], "GPU-new")
        for field, value in (("gpu_query", "GPU-new, NVIDIA H100 PCIe, 1000 MiB, another-driver"), ("hostname", "serval02")):
            changed = {**hardware, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                deployment.match_actual_backend(receipt, "aces", changed, self.installed, {**self.loaded, "pid": 9000})
        installed = copy.deepcopy(self.installed)
        installed["packages"]["numpy"] = "another-version"
        with self.assertRaisesRegex(ValueError, "backend/runtime differs"):
            deployment.match_actual_backend(receipt, "aces", hardware, installed, {**self.loaded, "pid": 9000})

    def test_target_backend_or_actual_runtime_cannot_change_between_measurements(self):
        """All target checks belong to one exact runtime and installed-software group."""
        self.change_result("confirmation_record", lambda result: result["hardware"]["runtime_environment"].update(
            XLA_PYTHON_CLIENT_MEM_FRACTION="0.86"), update_hardware=True)
        with self.assertRaisesRegex(ValueError, "different software/runtime groups"):
            deployment.validate_deployment(self.specification)

    def test_staged_helper_tampering_is_rejected(self):
        """An immutable plan does not authorize a different executed source file."""
        path = Path(self.specification["confirmation_record"]).parent / "code/measure_selected.py"
        path.write_text("changed helper bytes")
        with self.assertRaisesRegex(ValueError, "staged helper differs"):
            deployment.validate_deployment(self.specification)

    def test_api_and_saved_spec_receipts_keep_the_original_specification_path_explicit(self):
        """API receipts name no source file; CLI-style receipts retain and hash their explicit original file."""
        direct = deployment.validate_deployment(self.specification)
        self.assertIsNone(direct["specification_path"])
        path = self.write(self.directory / "deployment_specification.json", self.specification)
        saved = deployment.validate_deployment(self.specification, specification_path=path)
        self.assertEqual(saved["specification_path"], str(path))
        self.assertIn({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, saved["evidence"])
        self.assertEqual({key: value for key, value in direct.items() if key not in {"specification_path", "evidence"}},
                         {key: value for key, value in saved.items() if key not in {"specification_path", "evidence"}})

    def test_saved_spec_retains_original_flag_whitespace_and_rebuilds_exactly(self):
        """Runtime normalization cannot rewrite the original saved specification identity."""
        original = copy.deepcopy(self.specification)
        original["runtime_environment"]["XLA_FLAGS"] = "  " + self.runtime["XLA_FLAGS"] + "  "
        path = self.write(self.directory / "deployment_specification.json", original)
        saved = deployment.validate_deployment(original, specification_path=path)
        self.assertEqual(saved["specification"], original)
        self.assertEqual(saved["specification_digest"], deployment.validation.canonical_digest(original))
        self.assertEqual(saved["runtime_environment"], self.runtime)
        self.assertEqual(deployment.validate_deployment(saved["specification"], specification_path=saved["specification_path"]), saved)

    def test_strict_resolver_preserves_deployment_receipt_after_original_paths_are_removed(self):
        """Target binaries, public Git source, global gates and prepared helpers all resolve explicitly."""
        specification_path = self.write(self.directory / "deployment_specification.json", self.specification)
        original = deployment.validate_deployment(self.specification, specification_path=specification_path)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        staging = Path(temporary.name)
        manifest = {"schema_version": 1, "kind": "portable_raw_evidence", "files": [], "git_roots": []}
        for root, name, commit in ((self.source, "numerical", self.commit), (self.public, "jaxtari", deployment.JAXTARI_COMMIT)):
            target = staging / "repositories" / name
            subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(root), str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "-q", "--detach", commit], check=True)
            manifest["git_roots"].append({"original_root": str(root), "relative_path": "repositories/" + name, "commit": commit})
        for root in (self.directory, self.fixture.directory):
            for path in sorted(root.rglob("*")):
                if not path.is_file() or ".git" in path.relative_to(root).parts:
                    continue
                contents = path.read_bytes()
                digest = hashlib.sha256(contents).hexdigest()
                target = staging / "files" / digest
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(contents)
                manifest["files"].append({"original_path": str(path), "relative_path": "files/" + digest, "sha256": digest})
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        resolver = portable_evidence.ExactPathResolver(manifest_path)
        shutil.rmtree(self.directory)
        shutil.rmtree(self.fixture.directory)
        self.assertEqual(deployment.validate_deployment(self.specification, resolver=resolver, specification_path=specification_path), original)
        self.assertEqual(original["specification_path"], str(specification_path))
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)


if __name__ == "__main__":
    unittest.main()
