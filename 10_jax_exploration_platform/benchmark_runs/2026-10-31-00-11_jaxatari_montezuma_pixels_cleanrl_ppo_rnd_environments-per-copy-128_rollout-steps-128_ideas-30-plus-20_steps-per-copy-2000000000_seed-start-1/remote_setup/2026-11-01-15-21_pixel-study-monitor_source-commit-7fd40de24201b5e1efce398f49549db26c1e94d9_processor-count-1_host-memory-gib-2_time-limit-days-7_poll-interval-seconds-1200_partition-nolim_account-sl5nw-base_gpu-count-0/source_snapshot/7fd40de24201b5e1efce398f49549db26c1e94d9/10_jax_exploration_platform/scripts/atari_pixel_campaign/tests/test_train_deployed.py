"""Check deployed orchestration and actual trainer metadata callbacks without arrays or device work."""

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_deployed as deployed


def fixture(directory):
    """Separate original identities from actual operational paths in a complete metadata fixture."""
    required = {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}
    runtime = deployed.runtime_environment(required, {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85"})
    target = {"server": "aces", "hostnames": ["ac049", "ac055"], "gpu_model": "NVIDIA H100 PCIe",
              "total_memory_bytes": 1000 * 2 ** 20, "driver_version": "recorded-driver"}
    hardware = {"hostname": "ac049", "gpu_uuid": "GPU-first", "gpu_query": "GPU-first, NVIDIA H100 PCIe, 1000 MiB, recorded-driver",
                "pid": 100, "process_start_time": "12345", "runtime_environment": runtime}
    public = {"source_root": str(directory / "physical/public"), "source_commit": "b" * 40, "version": "0.2.0",
              "python_files": {"jaxtari/__init__.py": {"git_blob": "c" * 40, "sha256": "d" * 64}}, "tree_digest": "e" * 64}
    installed = {"packages": {name: "fixture-version" for name in deployed.training.PACKAGE_NAMES},
                 "jaxtari_source": public, "gpu_libraries": {"/installed/libcudnn.so.9": {"sha256": "f" * 64}}}
    asset_identity = {"manifest_sha256": "a" * 64, "source_commit": "b" * 40, "tree_digest": "c" * 64, "file_count": 2}
    receipt = {"schema_version": 1, "kind": "training_deployment_verification", "target": target,
        "runtime_environment": required, "actual_runtime_environment": runtime, "assets": asset_identity,
        "backend": {"packages": installed["packages"], "jaxtari_source": {key: value for key, value in public.items() if key != "source_root"},
                    "gpu_libraries": installed["gpu_libraries"], "mapped_library_api": {"cudnnGetVersion": 92400}}}
    receipt_path = directory / "deployment.json"
    receipt_path.write_text(json.dumps(receipt))
    specification = {"schema_version": 1, "unit_id": "aces_final_training", "source_root": "/original/source", "source_commit": "a" * 40,
        "hardware_class": "h100", "copy_seeds": [1001, 1002], "final_verification": "/original/global_final.json",
        "jaxtari_source_root": "/original/public", "runtime_environment": required,
        "asset_record": {"root": "/original/assets", "manifest": "/original/assets.json", "sha256": "a" * 64},
        "deployment_verification": "/original/deployment.json", "deployment_verification_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "configuration": {"learner": {"n_copies": 2, "n_envs": 128, "num_steps": 128, "update_epochs": 4, "num_minibatches": 4,
                                     "total_timesteps": 2_000_000_000}, "environment": {"num_envs": 128},
                          "execution": {"compilation_mode": "whole_iteration", "iterations_per_call": 5,
                                        "report_readout": "none", "report_every_iterations": 10}},
        "operational_paths": {"source_root": str(directory / "physical/source"), "jaxtari_source_root": public["source_root"],
            "asset_root": str(directory / "physical/assets"), "asset_manifest": str(directory / "physical/assets.json"),
            "deployment_verification": str(receipt_path)}, "evidence_resolution": {"manifest_sha256": "f" * 64},
        "validation": {"kind": "deployed_training_admission", "deployment_digest": deployed.training.canonical_digest(receipt),
                       "specification_digest": "d" * 64}}
    assets = {"root": specification["operational_paths"]["asset_root"], **asset_identity}
    backend = {"device_kind": "NVIDIA H100 PCIe", "driver_version": target["driver_version"], "packages": installed["packages"],
               "jaxtari_source": public, "installed_gpu_libraries": installed["gpu_libraries"]}
    return SimpleNamespace(spec=specification, receipt=receipt, hardware=hardware, assets=assets, installed=installed,
                           runtime=runtime, backend=backend)


class DeployedTrainingTests(unittest.TestCase):
    """Preserve scientific identity and reject unproved actual placement/backend before training."""

    def setUp(self):
        """Use isolated temporary metadata only; no scientific evidence or GPU libraries are read."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.data = fixture(self.directory)
        self.attempt = self.directory / "attempt"
        self.attempt.mkdir()

    def hooks(self, data=None):
        """Construct the actual metadata gate against the supplied explicit hardware/asset proof."""
        data = data or self.data
        return deployed.DeploymentMetadata(data.spec, data.receipt, "aces", data.hardware, data.assets, self.directory, self.attempt)

    def test_placement_requires_actual_compute_cgroup_or_exact_direct_card(self):
        """An environment variable alone does not establish a Slurm compute allocation."""
        environment = {"SLURM_JOB_ID": "123"}
        self.assertEqual(deployed.compute_placement(environment, "ac049", "0::/slurm/uid_1/job_123/step_batch\n")["kind"], "slurm")
        for cgroup in ("0::/user.slice", "0::/slurm/job_1234/step_batch", "0::/slurm/job_12/step_batch"):
            with self.subTest(cgroup=cgroup), self.assertRaisesRegex(ValueError, "outside"):
                deployed.compute_placement(environment, "ac049", cgroup)
        for host in ("serval05", "portal05", "alogin1", "gh-login02"):
            with self.subTest(host=host), self.assertRaisesRegex(RuntimeError, "login host"):
                deployed.compute_placement(environment, host, "0::/slurm/job_123")
        with self.assertRaisesRegex(RuntimeError, "Slurm"):
            deployed.compute_placement({}, "jaguar03", "")
        self.assertEqual(deployed.compute_placement({"CUDA_VISIBLE_DEVICES": deployed.training.SERVAL02_CARD1}, "serval02")["kind"], "serval02")
        with self.assertRaisesRegex(ValueError, "card 1"):
            deployed.compute_placement({"CUDA_VISIBLE_DEVICES": "0"}, "serval02")

    def test_actual_runtime_includes_original_game_defaults_and_rejects_conflicts(self):
        """Preallocation/sprite flags are applied before the actual runtime proof is captured."""
        runtime = deployed.runtime_environment(self.data.spec["runtime_environment"], {})
        self.assertEqual(runtime["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")
        self.assertEqual(runtime["JAXTARI_SKIP_SPRITE_UPDATE"], "1")
        self.assertEqual(runtime["JAX_ENABLE_COMPILATION_CACHE"], "false")
        for name, value in (("XLA_PYTHON_CLIENT_PREALLOCATE", "true"), ("JAXTARI_SKIP_SPRITE_UPDATE", "0"),
                            ("JAX_COMPILATION_CACHE_DIR", "/unvalidated/cache")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                deployed.runtime_environment(self.data.spec["runtime_environment"], {name: value})

    def test_receipt_bytes_and_immutable_unit_identity_are_separate_from_relocation(self):
        """Reformatting receipt JSON fails its raw hash while physical transport does not change unit identity."""
        self.assertEqual(deployed.deployment_receipt(self.data.spec), self.data.receipt)
        original = deployed.immutable_specification(self.data.spec)
        relocated = copy.deepcopy(self.data.spec)
        relocated["operational_paths"]["source_root"] = "/another/physical/root"
        relocated["evidence_resolution"]["manifest_sha256"] = "0" * 64
        self.assertEqual(deployed.immutable_specification(relocated), original)
        self.assertEqual(original["source_root"], "/original/source")
        self.assertNotIn("validation", original)
        path = Path(self.data.spec["operational_paths"]["deployment_verification"])
        path.write_text(json.dumps(self.data.receipt, indent=2))
        with self.assertRaisesRegex(ValueError, "changed"):
            deployed.deployment_receipt(self.data.spec)

    def test_target_hardware_assets_and_installed_backend_fail_before_numerical_imports(self):
        """A wrong model, driver, game tree, package or installed Python tree cannot reach the learner."""
        for field, value in (("gpu_query", "GPU-first, NVIDIA H100 NVL, 1000 MiB, recorded-driver"), ("hostname", "unvalidated-node")):
            data = copy.deepcopy(self.data)
            data.hardware[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "node/model"):
                self.hooks(data)
        data = copy.deepcopy(self.data)
        data.assets["tree_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "assets"):
            self.hooks(data)
        for field in ("packages", "jaxtari_source", "gpu_libraries"):
            installed = copy.deepcopy(self.data.installed)
            installed[field]["tree_digest" if field == "jaxtari_source" else next(iter(installed[field]))] = "changed"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "installed training"):
                self.hooks().before_numerical_imports(installed)
        self.assertFalse(any(name in sys.modules for name in ("jax", "numpy", "torch")))

    def test_checkpoint_identity_uses_logical_paths_and_allows_another_validated_uuid(self):
        """Node/UUID relocation retains scientific identity; changed source/configuration/seeds still change it."""
        hooks = self.hooks()
        hooks.before_numerical_imports(self.data.installed)
        metadata = hooks.identity_metadata(self.data.runtime, self.data.backend, self.data.assets)
        self.assertEqual(metadata["backend"]["jaxtari_source"]["source_root"], "/original/public")
        self.assertEqual(metadata["assets"]["root"], "/original/assets")
        identity = deployed.training.checkpoint_identity(self.data.spec, {"source_commit": "f" * 40},
            metadata["runtime_environment"], metadata["backend"], metadata["assets"])
        moved = copy.deepcopy(self.data)
        moved.hardware.update(hostname="ac055", gpu_uuid="GPU-second", gpu_query="GPU-second, NVIDIA H100 PCIe, 1000 MiB, recorded-driver")
        moved.hardware["runtime_environment"] = {**moved.hardware["runtime_environment"], "CUDA_VISIBLE_DEVICES": "GPU-second"}
        moved.installed["jaxtari_source"]["source_root"] = "/new/physical/public"
        moved.backend["jaxtari_source"] = moved.installed["jaxtari_source"]
        moved.assets["root"] = "/new/physical/assets"
        other = self.hooks(moved)
        other.before_numerical_imports(moved.installed)
        self.assertEqual(other.identity_metadata(moved.hardware["runtime_environment"], moved.backend, moved.assets), metadata)
        for field in ("source_commit", "configuration", "copy_seeds"):
            specification = copy.deepcopy(self.data.spec)
            if field == "configuration":
                specification[field]["execution"]["iterations_per_call"] = 1
            else:
                specification[field] = "0" * 40 if field == "source_commit" else [1003, 1004]
            altered = deployed.training.checkpoint_identity(specification, {"source_commit": "f" * 40},
                metadata["runtime_environment"], metadata["backend"], metadata["assets"])
            self.assertNotEqual(altered, identity)
        changed_backend = copy.deepcopy(self.data.backend)
        changed_backend["driver_version"] = "another-driver"
        with self.assertRaisesRegex(ValueError, "changed after"):
            hooks.identity_metadata(self.data.runtime, changed_backend, self.data.assets)

    def test_mapped_library_gate_publishes_only_after_actual_match(self):
        """The first-update callback sends the actual process/library proof to the deployment matcher."""
        hooks = self.hooks()
        hooks.before_numerical_imports(self.data.installed)
        loaded = {"pid": self.data.hardware["pid"], "cudnnGetVersion": 92400}
        with mock.patch.object(deployed.deployment, "match_actual_backend", side_effect=ValueError("wrong mapped image")):
            with self.assertRaisesRegex(ValueError, "wrong mapped"):
                hooks.after_first_update(loaded)
        self.assertFalse((self.directory / "deployment_backend_match.json").exists())
        with mock.patch.object(deployed.deployment, "match_actual_backend", return_value={"kind": "matched"}) as match:
            hooks.after_first_update(loaded)
            match.assert_called_once_with(self.data.receipt, "aces", self.data.hardware, self.data.installed, loaded)
        self.assertEqual(json.loads((self.attempt / "deployment_backend_match.json").read_text()), {"kind": "matched"})

    def test_actual_shared_trainer_calls_installed_gate_before_importing_arrays(self):
        """An installed-backend failure in the real shared trainer journals the attempt without importing JAX."""
        hooks = mock.Mock()
        hooks.before_numerical_imports.side_effect = ValueError("installed bytes changed")
        handlers = {value: signal.getsignal(value) for value in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
        self.addCleanup(lambda: [signal.signal(value, handler) for value, handler in handlers.items()])
        before = set(sys.modules)
        with mock.patch.object(deployed.training, "jaxtari_python_receipt", return_value=self.data.installed["jaxtari_source"]) as public, \
                mock.patch.object(deployed.training, "installed_gpu_library_receipts", return_value=self.data.installed["gpu_libraries"]), \
                mock.patch.object(deployed.training.metadata, "version", return_value="fixture-version"):
            with self.assertRaisesRegex(ValueError, "installed bytes"):
                deployed.training._train(SimpleNamespace(), self.data.spec, Path("/physical/source"), {}, self.data.runtime,
                    self.data.assets, self.directory, 7, self.attempt,
                    operational_paths=self.data.spec["operational_paths"], metadata_hooks=hooks)
            public.assert_called_once_with(self.data.spec["operational_paths"]["jaxtari_source_root"])
        self.assertFalse({"jax", "numpy", "torch"} & (set(sys.modules) - before))
        failure = json.loads((self.attempt / "failure.json").read_text())
        self.assertEqual(failure["attempt"], 7)
        self.assertIn("installed bytes", failure["error"])
        self.assertEqual(json.loads((self.directory / "progress.json").read_text())["state"], "failed")

    def test_run_delegates_once_with_physical_paths_while_retaining_logical_specification(self):
        """The orchestration retains placement lock and attempt metadata around the existing training loop."""
        campaign = self.directory / "campaign"
        campaign.mkdir()
        for name in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt"):
            (campaign / name).write_text("fixture launch record\n")
        arguments = SimpleNamespace(final_spec=Path("/original/training_specification.json"), run_dir=campaign,
            unit_id=self.data.spec["unit_id"], server="aces", evidence_manifest=None, evidence_manifest_sha256=None)
        events = []

        @contextmanager
        def placement_guard(placement):
            """Require all preflight/assets/training work to remain inside the placement lock."""
            events.append("locked")
            try:
                yield
            finally:
                events.append("released")

        def train(*args, **kwargs):
            """Inspect actual delegation without implementing or running an update."""
            self.assertEqual(events, ["locked"])
            self.assertEqual(args[1]["source_root"], "/original/source")
            self.assertEqual(args[2], Path(self.data.spec["operational_paths"]["source_root"]))
            self.assertEqual(kwargs["operational_paths"], self.data.spec["operational_paths"])
            self.assertIsInstance(kwargs["metadata_hooks"], deployed.DeploymentMetadata)

        with mock.patch.dict(deployed.os.environ, {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85"}, clear=True), \
                mock.patch.object(deployed, "compute_placement", return_value={"kind": "slurm", "slurm_job_id": "123"}), \
                mock.patch.object(deployed.admission, "validate_deployed_training_admission", return_value=self.data.spec) as admitted, \
                mock.patch.object(deployed.admission.validation, "validate_training_admission", side_effect=AssertionError("NVL admission must not run")), \
                mock.patch.object(deployed, "helper_receipt", return_value={"source_commit": "f" * 40}), \
                mock.patch.object(deployed.training, "source_receipt") as source, \
                mock.patch.object(deployed.measurement, "allocated_gpu_uuid", return_value="GPU-first"), \
                mock.patch.object(deployed.measurement, "preflight", return_value=self.data.hardware), \
                mock.patch.object(deployed.training, "placement_guard", side_effect=placement_guard), \
                mock.patch.object(deployed.training, "verify_assets", return_value=self.data.assets) as assets, \
                mock.patch.object(deployed.training, "_train", side_effect=train) as numerical:
            deployed.run(arguments)
        admitted.assert_called_once_with(arguments.final_spec, None)
        source.assert_called_once_with(Path(self.data.spec["operational_paths"]["source_root"]), self.data.spec["source_commit"])
        self.assertEqual(assets.call_args.args[0]["root"], self.data.spec["operational_paths"]["asset_root"])
        numerical.assert_called_once()
        self.assertEqual(events, ["locked", "released"])
        unit = campaign / "data" / self.data.spec["unit_id"]
        marker = json.loads((unit / "training_unit.json").read_text())
        self.assertEqual(marker["specification_digest"], deployed.training.canonical_digest(deployed.immutable_specification(self.data.spec)))
        self.assertEqual(json.loads((unit / "progress.json").read_text())["attempt"], 1)

    def test_orchestration_failure_keeps_current_attempt_failure_under_the_placement_lock(self):
        """Failure before trainer entry is durable and cannot be mistaken for a previous attempt's failure."""
        campaign = self.directory / "campaign"
        campaign.mkdir()
        for name in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt"):
            (campaign / name).write_text("fixture launch record\n")
        arguments = SimpleNamespace(final_spec=Path("/original/spec.json"), run_dir=campaign, unit_id=self.data.spec["unit_id"],
                                    server="aces", evidence_manifest=None, evidence_manifest_sha256=None)
        events = []

        @contextmanager
        def guard(placement):
            """Observe the journal before the failed process relinquishes its placement lock."""
            events.append("locked")
            try:
                yield
            finally:
                events.append("released")

        with mock.patch.dict(deployed.os.environ, {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85"}, clear=True), \
                mock.patch.object(deployed, "compute_placement", return_value={"kind": "slurm", "slurm_job_id": "123"}), \
                mock.patch.object(deployed.admission, "validate_deployed_training_admission", return_value=self.data.spec), \
                mock.patch.object(deployed, "helper_receipt", return_value={}), \
                mock.patch.object(deployed.training, "source_receipt"), \
                mock.patch.object(deployed.measurement, "allocated_gpu_uuid", return_value="GPU-first"), \
                mock.patch.object(deployed.measurement, "preflight", side_effect=ValueError("actual GPU is busy")), \
                mock.patch.object(deployed.training, "placement_guard", side_effect=guard), \
                mock.patch.object(deployed.training, "_train") as numerical:
            with self.assertRaisesRegex(ValueError, "GPU is busy"):
                deployed.run(arguments)
            numerical.assert_not_called()
        unit = campaign / "data" / self.data.spec["unit_id"]
        failure = json.loads((unit / "attempts/attempt-0001/failure.json").read_text())
        self.assertEqual(failure["attempt"], 1)
        self.assertIn("GPU is busy", failure["error"])
        self.assertEqual(json.loads((unit / "progress.json").read_text())["state"], "failed")
        self.assertEqual(events, ["locked", "released"])

    def test_login_host_and_unit_mismatch_fail_before_helpers_or_numerical_work(self):
        """Neither raw evidence admission nor a valid receipt can override actual placement or the reserved unit."""
        arguments = SimpleNamespace(final_spec=Path("/original/spec.json"), unit_id="another_unit",
                                    evidence_manifest=None, evidence_manifest_sha256=None)
        with mock.patch.object(deployed.socket, "gethostname", return_value="portal05"), \
                mock.patch.object(deployed.admission, "validate_deployed_training_admission") as admit, \
                mock.patch.object(deployed, "helper_receipt") as helpers:
            with self.assertRaisesRegex(RuntimeError, "login host"):
                deployed.run(arguments)
            admit.assert_not_called()
            helpers.assert_not_called()
        with mock.patch.object(deployed, "compute_placement", return_value={"kind": "slurm", "slurm_job_id": "123"}), \
                mock.patch.object(deployed.admission, "validate_deployed_training_admission", return_value=self.data.spec), \
                mock.patch.object(deployed, "helper_receipt") as helpers:
            with self.assertRaisesRegex(ValueError, "reserved"):
                deployed.run(arguments)
            helpers.assert_not_called()

    def test_parser_keeps_original_path_and_rejects_invalid_stop_or_manifest_inputs(self):
        """CLI validation cannot silently replace an original specification path or accept an unsafe stop value."""
        base = ["--final-spec", "/original/spec.json", "--run-dir", str(self.directory), "--unit-id", "final_training", "--server", "aces"]
        parsed = deployed.parse_arguments(base)
        self.assertEqual(parsed.final_spec, Path("/original/spec.json"))
        for options in (["--maximum-seconds", "nan"], ["--checkpoint-seconds", "1201"], ["--evidence-manifest-sha256", "a" * 64],
                        ["--unit-id", "../other"]):
            with self.subTest(options=options), self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                deployed.parse_arguments(base + options)


if __name__ == "__main__":
    unittest.main()
