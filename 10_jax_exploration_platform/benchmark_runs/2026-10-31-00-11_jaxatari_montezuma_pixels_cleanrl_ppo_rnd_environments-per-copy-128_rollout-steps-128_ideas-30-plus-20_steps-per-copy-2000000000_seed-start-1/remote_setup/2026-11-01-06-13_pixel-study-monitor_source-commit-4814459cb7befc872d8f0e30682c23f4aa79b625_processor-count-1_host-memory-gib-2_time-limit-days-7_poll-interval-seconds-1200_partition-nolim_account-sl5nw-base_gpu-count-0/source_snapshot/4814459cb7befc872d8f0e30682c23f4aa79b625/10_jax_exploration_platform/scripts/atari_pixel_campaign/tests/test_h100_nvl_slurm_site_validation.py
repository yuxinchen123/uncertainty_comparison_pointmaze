"""Verify the new H100 site wrapper with temporary metadata and no numerical execution."""

import copy
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import h100_nvl_slurm_site_validation as site_validation


class H100SiteTests(unittest.TestCase):
    """Keep all fixtures local and replace every GPU, process and numerical operation."""

    def setUp(self):
        """Construct an owned single-card allocation with a physical index different from CUDA zero."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name).resolve()
        (self.run / "slurm").mkdir()
        (self.run / "slurm/submitted_jobids_uva.txt").write_text("333\n")
        self.gpu = "GPU-owned"
        self.environment = {**site_validation.RUNTIME, "SLURM_JOB_ID": "333", "SLURM_JOB_GPUS": "7", "SLURM_STEP_GPUS": "7",
            "SLURM_CPUS_PER_TASK": "8", "SLURM_CPU_BIND_TYPE": "threads", "CUDA_VISIBLE_DEVICES": "0"}
        self.csv = "1, " + self.gpu + ", NVIDIA H100 NVL, 95830, 590.48.01\n"
        self.xml = "<nvidia_smi_log><gpu><uuid>" + self.gpu + "</uuid><product_name>NVIDIA H100 NVL</product_name><minor_number>7</minor_number></gpu></nvidia_smi_log>"
        self.plan = {"campaign_run_dir": str(self.run), "reference": {"driver": site_validation.DRIVER},
            "python": sys.executable, "helpers": {"source_root": str(self.run / "helpers")},
            "numerical_source": {"source_root": str(self.run / "original")},
            "cleanrl_reference": {"source_root": str(self.run / "cleanrl"),
                "file": {"path": str(self.run / "cleanrl/cleanrl/ppo_rnd_envpool.py")}}}

    def test_step_zero_resolves_only_after_complete_physical_minor_proof(self):
        """CUDA zero remains valid after the XML, visible row and every job/step field identify the same card."""
        result = site_validation.resolve_allocation(self.environment, "serval06", self.csv, self.xml)
        self.assertEqual(result["gpu_uuid"], self.gpu)
        self.assertEqual(result["physical_minor"], "7")
        self.assertEqual(result["cuda_visible_devices_original"], "0")
        for field in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "exact physical GPU"):
                site_validation.resolve_allocation({**self.environment, field: "6"}, "serval06", self.csv, self.xml)

    def test_other_hosts_models_memory_and_extra_visible_cards_reject(self):
        """No direct server, different H100 package, Grace card or larger visible roster can substitute."""
        for host in ("serval02", "serval03", "serval05", "serval10", "portal05"):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "serval06-through"):
                site_validation.resolve_allocation(self.environment, host, self.csv, self.xml)
        for model in ("NVIDIA H100 PCIe", "NVIDIA H100 SXM5", "NVIDIA GH200", "NVIDIA H200 NVL"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                site_validation.resolve_allocation(self.environment, "serval07", self.csv.replace(site_validation.MODEL, model), self.xml.replace(site_validation.MODEL, model))
        with self.assertRaises(ValueError):
            site_validation.resolve_allocation(self.environment, "serval08", self.csv.replace("95830", "81559"), self.xml)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            site_validation.resolve_allocation(self.environment, "serval09", self.csv + self.csv, self.xml)

    def test_foreign_ledger_collision_or_wrong_cgroup_cannot_authorize_children(self):
        """Only the exact UVA ledger and its actual current cgroup establish this job's ownership."""
        valid = "0::/slurm/uid_1/job_333/step_0/task_0\n"
        self.assertEqual(site_validation.require_owned_job(self.run, self.environment, valid)["job_id"], "333")
        (self.run / "slurm/submitted_jobids_aces.txt").write_text("444\n")
        with self.assertRaisesRegex(ValueError, "campaign-owned UVA"):
            site_validation.require_owned_job(self.run, {**self.environment, "SLURM_JOB_ID": "444"}, valid.replace("333", "444"))
        with self.assertRaisesRegex(ValueError, "campaign-owned UVA"):
            site_validation.require_owned_job(self.run, self.environment, valid.replace("333", "334"))

    def test_runtime_rejects_extra_controls_and_records_no_secret_values(self):
        """An additional allocator or JAX mode cannot disappear when metadata is projected."""
        self.assertEqual(site_validation.require_runtime(self.environment), site_validation.RUNTIME)
        with self.assertRaisesRegex(ValueError, "JAX_DISABLE_JIT") as error:
            site_validation.require_runtime({**self.environment, "JAX_DISABLE_JIT": "private-value"})
        self.assertNotIn("private-value", str(error.exception))
        for name, value in (("TF_GPU_ALLOCATOR", "cuda_malloc_async"), ("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async"),
                            ("JAX_ENABLE_X64", "false"), ("XLA_FLAGS", "--xla_gpu_exclude_nondeterministic_ops=false")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                site_validation.require_runtime({**self.environment, name: value})

    def test_selected_card_queries_never_inspect_another_card_or_process_owner(self):
        """All mode and process queries use the already proved selected UUID."""
        with patch.object(site_validation, "query", side_effect=["Disabled, 0\n", "", "CC status: OFF\n"]) as query:
            result = site_validation.require_selected_card(self.gpu)
        self.assertEqual(result["compute_processes"], [])
        for call in query.call_args_list:
            self.assertEqual(call.args[0][call.args[0].index("-i") + 1], self.gpu)
        with patch.object(site_validation, "query", side_effect=["Disabled, 0\n", "999, GPU-owned\n", "CC status: OFF\n"]):
            with self.assertRaisesRegex(ValueError, "idle"):
                site_validation.require_selected_card(self.gpu)

    def test_driver_mismatch_stops_before_any_jax_or_child_context(self):
        """A Slurm label does not waive the reference driver's exact version requirement."""
        original_read = Path.read_text
        def read(path, *args, **kwargs):
            return "0::/slurm/job_333/step_0/task_0\n" if path == Path("/proc/self/cgroup") else original_read(path, *args, **kwargs)
        with patch.dict(os.environ, self.environment, clear=True), patch.object(site_validation.socket, "gethostname", return_value="serval06"), \
                patch.object(Path, "read_text", read), patch.object(site_validation, "query", side_effect=[self.csv.replace(site_validation.DRIVER, "580.159.04"), self.xml]), \
                patch.object(site_validation, "capture_allocation_details", return_value={"test": "metadata capture"}), \
                patch.object(site_validation, "run_child") as child:
            with self.assertRaisesRegex(ValueError, "stop before JAX"):
                site_validation.placement(self.run, self.plan)
        child.assert_not_called()
        self.assertFalse((self.run / "site_assignment.json").exists())

    def test_rejected_mapping_preserves_original_allowlisted_observations_before_validation(self):
        """A physical-allocation mismatch retains actual fields and GPU records without an inherited secret."""
        original_read = Path.read_text
        def read(path, *args, **kwargs):
            """Supply only this fixture's owned cgroup while leaving saved observations real."""
            return "0::/slurm/job_333/step_0/task_0\n" if path == Path("/proc/self/cgroup") else original_read(path, *args, **kwargs)
        environment = {**self.environment, "SLURM_JOB_GPUS": "6", "SLURM_STEP_GPUS": "6",
            "SLURM_STEP_ID": "0", "SLURM_GPUS_ON_NODE": "1", "PRIVATE_TOKEN": "must not be recorded"}
        with patch.dict(os.environ, environment, clear=True), patch.object(site_validation.socket, "gethostname", return_value="serval09"), \
                patch.object(Path, "read_text", read), patch.object(site_validation, "query", side_effect=[self.csv, self.xml]), \
                patch.object(site_validation, "capture_allocation_details", return_value={"test": "metadata capture"}), \
                patch.object(site_validation, "run_child") as child:
            with self.assertRaisesRegex(ValueError, "exact physical GPU"):
                site_validation.placement(self.run, self.plan)
        observation = site_validation.read(self.run / "allocation_observation.json")
        self.assertEqual(observation["allocation_environment"]["SLURM_JOB_GPUS"], "6")
        self.assertEqual(observation["allocation_environment"]["SLURM_STEP_GPUS"], "6")
        self.assertEqual(observation["allocation_environment"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(observation["gpu_csv"], self.csv)
        self.assertEqual(observation["gpu_xml"], self.xml)
        self.assertNotIn("must not be recorded", (self.run / "allocation_observation.json").read_text())
        self.assertNotIn("PRIVATE_TOKEN", observation["allocation_environment"])
        self.assertFalse((self.run / "site_assignment.json").exists())
        child.assert_not_called()

    def test_missing_gpu_fields_are_recorded_explicitly_instead_of_inferred_as_remapped(self):
        """An absent job/step GPU identity remains distinguishable from an observed conflicting value."""
        original_read = Path.read_text
        def read(path, *args, **kwargs):
            """Provide the owned cgroup only; preserve ordinary filesystem behavior for captured receipts."""
            return "0::/slurm/job_333/step_0/task_0\n" if path == Path("/proc/self/cgroup") else original_read(path, *args, **kwargs)
        environment = {name: value for name, value in self.environment.items() if name not in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")}
        with patch.dict(os.environ, environment, clear=True), patch.object(site_validation.socket, "gethostname", return_value="serval09"), \
                patch.object(Path, "read_text", read), patch.object(site_validation, "query", side_effect=[self.csv, self.xml]), \
                patch.object(site_validation, "capture_allocation_details", return_value={"test": "metadata capture"}):
            with self.assertRaisesRegex(ValueError, "exact physical GPU"):
                site_validation.placement(self.run, self.plan)
        observation = site_validation.read(self.run / "allocation_observation.json")
        for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
            self.assertIsNone(observation["allocation_environment"][name])
            self.assertIn(name, observation["absent_allocation_fields"])

    def test_extra_allocation_capture_queries_only_owned_job_and_gpu_device_metadata(self):
        """Additional diagnosis never queries another job, user, GPU process or unrelated configuration file."""
        config = Path("/etc/slurm/gres.conf")
        original_read = Path.read_text
        def read(path, *args, **kwargs):
            """Supply explicit GPU mappings and prove unrelated configuration lines are not recorded."""
            if path == config:
                return "# comment\nNodeName=serval[06-09] AutoDetect=nvml\nName=gpu File=/dev/nvidia7\nUnrelated=private-value\n"
            return original_read(path, *args, **kwargs)
        information = types.SimpleNamespace(st_mode=stat.S_IFCHR | 0o666, st_rdev=os.makedev(195, 7))
        with patch.object(site_validation.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout="JobId=333 GRES_IDX=0", stderr="")) as query, \
                patch.object(Path, "glob", return_value=[Path("/dev/nvidia7")]), patch.object(Path, "lstat", return_value=information), \
                patch.object(Path, "exists", return_value=True), patch.object(Path, "read_text", read):
            captured = site_validation.capture_allocation_details({"job_id": "333", "ledger_prefix": ["333"]})
        query.assert_called_once_with(["scontrol", "show", "job", "-d", "333"], capture_output=True, text=True, check=False, timeout=20)
        self.assertEqual(captured["device_files"], [{"path": "/dev/nvidia7", "character_device": True, "major": 195, "minor": 7}])
        self.assertEqual(len(captured["gpu_gres_mapping"]["gpu_mapping_lines"]), 2)
        self.assertNotIn("private-value", json.dumps(captured))
        with patch.object(site_validation.subprocess, "run") as query:
            with self.assertRaisesRegex(ValueError, "exact owned job"):
                site_validation.capture_allocation_details({"job_id": "334", "ledger_prefix": ["333"]})
        query.assert_not_called()

    def admission_fixture(self):
        """Construct four exact raw admission documents and a separate new preparation directory."""
        original = self.run / "original_admission"
        directory = self.run / "new_prepared_site"
        directory.mkdir()
        files = []
        for name in ("SOURCE_AND_GATES_READY.json", "reference_configuration.json", "h100_gate.json", "a4500_gate.json"):
            path = self.write(original / name, {"original_document": name})
            files.append(site_validation.receipt(path))
        return directory, {"files": files}

    def test_admission_archive_keeps_raw_bytes_and_original_identity_after_originals_removed(self):
        """Historical copied evidence remains valid after live admission changes or is removed."""
        directory, admission = self.admission_fixture()
        archive = site_validation.archive_admission(directory, admission)
        original_records = copy.deepcopy(admission["files"])
        for original, item in zip(admission["files"], archive["files"]):
            self.assertEqual(item["original"], original)
            self.assertEqual(Path(item["captured"]["path"]).read_bytes(), Path(original["path"]).read_bytes())
            Path(original["path"]).unlink()
        self.assertEqual(site_validation.validate_admission_archive(directory, admission, archive), archive)
        self.assertEqual([item["original"] for item in archive["files"]], original_records)

    def test_admission_archive_rejects_changed_copy_missing_record_or_substituted_path(self):
        """Recomputed transport metadata cannot replace the exact captured original documents."""
        directory, admission = self.admission_fixture()
        archive = site_validation.archive_admission(directory, admission)
        for field in (None, {**archive, "files": archive["files"][:-1]},
                {**archive, "files": list(reversed(archive["files"]))}):
            with self.subTest(field=field), self.assertRaises(ValueError):
                site_validation.validate_admission_archive(directory, admission, field)
        substituted = copy.deepcopy(archive)
        substituted["files"][0]["captured"]["path"] = admission["files"][0]["path"]
        with self.assertRaisesRegex(ValueError, "exact captured path"):
            site_validation.validate_admission_archive(directory, admission, substituted)
        target = Path(archive["files"][0]["captured"]["path"])
        target.write_bytes(target.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "copy changed"):
            site_validation.validate_admission_archive(directory, admission, archive)

    def test_admission_capture_rejects_changed_original_and_duplicate_records(self):
        """Preparation never records newly changed bytes under a previously admitted SHA."""
        directory, admission = self.admission_fixture()
        duplicate = {"files": [admission["files"][0]] * 4}
        with self.assertRaisesRegex(ValueError, "distinct"):
            site_validation.archive_admission(directory, duplicate)
        Path(admission["files"][0]["path"]).write_bytes(b'{"changed":true}\n')
        with self.assertRaisesRegex(ValueError, "changed before"):
            site_validation.archive_admission(directory, admission)
        self.assertFalse((directory / "admission_files").exists())

    def test_original_gate_commands_are_complete_and_selector_free(self):
        """A site validation cannot replace the original gates with a subset or a worker launch."""
        learner = site_validation.original_command(self.plan, "learner")
        renderer = site_validation.original_command(self.plan, "renderer")
        self.assertEqual(learner[1:4], ["-m", "pytest", "-q"])
        self.assertEqual(len(learner), 6)
        self.assertTrue(learner[-2].endswith("/tests/atari_pixels"))
        self.assertTrue(learner[-1].endswith("/tests/test_atari_pixel_environment.py"))
        self.assertEqual(renderer[-2:], ["--candidate", "all"])
        self.assertEqual(site_validation.JOB_SHAPE["gres"], "gpu:nvidia_h100_nvl:1")
        with self.assertRaises(ValueError):
            site_validation.original_command(self.plan, "worker")

    def test_detached_clean_git_source_identity_rejects_branch_or_changed_bytes(self):
        """Use a real isolated Git repository rather than touching the integration index or a frozen source."""
        source = self.run / "temporary_source"
        source.mkdir()
        path = source / "metadata.py"
        path.write_text("# isolated source bytes\n")
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "add", "metadata.py"], check=True)
        subprocess.run(["git", "-C", str(source), "-c", "user.name=Metadata test", "-c", "user.email=metadata@example.invalid",
            "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Create isolated metadata fixture"], check=True)
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        with self.assertRaisesRegex(ValueError, "detached"):
            site_validation.source_identity(source, commit, ["metadata.py"])
        subprocess.run(["git", "-C", str(source), "checkout", "-q", "--detach"], check=True)
        record = site_validation.source_identity(source, commit, ["metadata.py"])
        site_validation.verify_source_identity(record)
        path.write_text("# changed bytes\n")
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            site_validation.verify_source_identity(record)

    def test_child_cleanup_covers_first_birth_read_and_owner_receipt_failures(self):
        """A started unreaped child remains owned even if its first metadata operation raises."""
        assignment = {"allocation": {"gpu_uuid": self.gpu, "job_id": "333"}}
        self.plan["asset_record"] = {"root": str(self.run / "assets")}
        for stage in ("birth", "owner"):
            directory = self.run / stage
            directory.mkdir()
            process = Mock(pid=777)
            process.poll.return_value = None
            process.wait.return_value = 0
            birth = OSError("initial metadata read failed") if stage == "birth" else "100"
            with self.subTest(stage=stage), patch.object(site_validation.subprocess, "Popen", return_value=process), \
                    patch.dict(os.environ, {}, clear=True), \
                    patch.object(site_validation, "process_start_time", side_effect=birth if isinstance(birth, BaseException) else None,
                        return_value=birth if isinstance(birth, str) else None), \
                    patch.object(site_validation, "write_once", side_effect=OSError("owner metadata write failed")), \
                    patch.object(site_validation.os, "killpg") as kill:
                with self.assertRaises(OSError):
                    site_validation.run_child(self.plan, directory, "learner", assignment)
                kill.assert_called_once_with(777, signal.SIGTERM)
            self.assertFalse((directory / "learner.released").exists())

    def test_stale_source_admission_cannot_be_reinterpreted_as_current_site_permission(self):
        """The wrapper neither edits admission nor launches workers to recover a stale site_validation."""
        with patch.object(site_validation, "snapshot_admission", return_value={"changed": True}):
            with self.assertRaisesRegex(ValueError, "no workers"):
                site_validation.check_admission(self.run, {"changed": False})
        self.assertFalse((self.run / "SOURCE_AND_GATES_READY.json").exists())

    def write(self, path, value):
        """Store a synthetic proof document only in this test's temporary directory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")
        return path

    def test_reference_accepts_real_old_schema_but_not_a_different_driver_or_runtime(self):
        """The old diagnostic's actual hardware fields are authoritative; no absent new field is invented."""
        directory = self.run / "old_reference"
        hardware = {"hostname": "serval02", "gpu_query": site_validation.REFERENCE_GPU + ", " + site_validation.MODEL + ", 95830 MiB, " + site_validation.DRIVER,
            "runtime_environment": {**site_validation.RUNTIME, "XLA_FLAGS": " --xla_gpu_exclude_nondeterministic_ops=true"}}
        self.write(directory / "hardware.json", hardware)
        self.write(directory / "loaded_gpu_libraries.json", {"platform": "gpu", "platform_version": "PJRT C API\ncuda 12090",
            "cudnnGetVersion": 92400, "cudaRuntimeGetVersion": 12090})
        self.write(directory / "diagnostic_plan.json", {"core": {"source_commit": site_validation.SOURCE_COMMIT},
            "configuration": {"learner": {}, "environment": {}, "execution": {}}})
        self.write(directory / "dtypes.json", site_validation.EXPECTED_DTYPES)
        installed = self.write(directory / "installed_backend.json", {})
        assets = self.write(directory / "asset_receipt.json", {})
        result = self.write(directory / "result.json", {})
        with patch.object(site_validation, "software_reference", return_value={"packages": {}}):
            reference = site_validation.reference_identity(directory, installed, assets, result)
            self.assertEqual(reference["driver"], site_validation.DRIVER)
            self.assertEqual(reference["hardware"], site_validation.receipt(directory / "hardware.json"))
            hardware["gpu_query"] = hardware["gpu_query"].replace(site_validation.DRIVER, "580.159.04")
            self.write(directory / "hardware.json", hardware)
            with self.assertRaisesRegex(ValueError, "original deterministic serval02"):
                site_validation.reference_identity(directory, installed, assets, result)
            hardware["gpu_query"] = hardware["gpu_query"].replace("580.159.04", site_validation.DRIVER)
            hardware["runtime_environment"]["JAX_ENABLE_X64"] = "false"
            self.write(directory / "hardware.json", hardware)
            with self.assertRaisesRegex(ValueError, "full-precision"):
                site_validation.reference_identity(directory, installed, assets, result)

    def test_original_configuration_binds_all_fields_to_frozen_defaults_and_admission(self):
        """A changed discount or learning rate cannot be hidden by matching only rollout dimensions."""
        defaults = {"learner": {"n_copies": 1, "n_envs": 128, "num_steps": 128, "learning_rate": .0001,
            "gamma_int": .99, "per_copy_learning_rates": ()}, "environment": {"num_envs": 128, "frame_skip": 4}}
        reference = self.write(self.run / "admitted_reference.json", {**defaults["learner"], "n_copies": 2})
        admission = {"reference_configuration": site_validation.receipt(reference)}
        fake = types.SimpleNamespace(default_configuration=lambda root: copy.deepcopy(defaults))
        with patch.object(site_validation, "load", return_value=fake):
            result = site_validation.original_configuration(self.run / "source", admission)
            self.assertEqual(result["learner"]["per_copy_learning_rates"], [])
            self.assertEqual(result["environment"], defaults["environment"])
            for field, value in (("learning_rate", .001), ("gamma_int", .95)):
                altered = {**defaults["learner"], field: value}
                self.write(reference, altered)
                admission["reference_configuration"] = site_validation.receipt(reference)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "admitted original"):
                    site_validation.original_configuration(self.run / "source", admission)

    def test_software_package_map_requires_a_completed_production_validated_packet(self):
        """Fresh preparation metadata cannot become a substitute for the actual captured shared-software map."""
        directory = self.run / "completed_formal/attempt-0004"
        path = self.write(directory / "result.json", {"attempt": 4, "case_completed": True, "correctness_passed": True, "measurement_valid": True})
        self.write(directory / "case.json", {"hypothesis": "original_mechanism", "hardware_class": "jaguar03_rtx_a4500",
            "configuration": {"n_copies": 2}})
        installed = self.write(directory / "installed_backend.json", {"packages": {name: "recorded-version" for name in site_validation.PACKAGE_NAMES}})
        assets = self.write(directory / "asset_receipt.json", {"original": "assets"})
        verifier = Mock(return_value={"scope": "actual_original_packet"})
        fake = types.SimpleNamespace(validate_cohort_case=verifier, Evidence=lambda: object())
        with patch.object(site_validation, "load", return_value=fake):
            reference = site_validation.software_reference(path, installed, assets)
            self.assertEqual(set(reference["packages"]), set(site_validation.PACKAGE_NAMES))
            verifier.assert_called_once()
            self.write(installed, {"packages": {"jax": "one-version-only"}})
            with self.assertRaisesRegex(ValueError, "complete package map"):
                site_validation.software_reference(path, installed, assets)

    def test_probe_must_report_every_actual_original_numerical_module_path(self):
        """A matching source-commit string alone cannot hide a mutable imported model module."""
        source = self.run / "source"
        paths = {"exploration_platform.atari_pixels." + name:
            str(source / "10_jax_exploration_platform/src/exploration_platform/atari_pixels" / (name + ".py"))
            for name in ("config", "environment", "training", "networks", "state")}
        site_validation.validate_numerical_paths(paths, source)
        paths["exploration_platform.atari_pixels.networks"] = "/mutable/networks.py"
        with self.assertRaisesRegex(ValueError, "different numerical"):
            site_validation.validate_numerical_paths(paths, source)
        with self.assertRaisesRegex(ValueError, "omitted"):
            site_validation.validate_numerical_paths({}, source)

    def test_test_selectors_and_a_different_cleanrl_reference_are_rejected(self):
        """An unchanged pytest command cannot conceal selectors or a substituted external class definition."""
        reference = self.plan["cleanrl_reference"]
        site_validation.require_gate_controls({}, reference)
        site_validation.require_gate_controls({"CLEANRL_REFERENCE_FILE": reference["file"]["path"]}, reference)
        for environment in ({"PYTEST_ADDOPTS": "-k one_small_test"}, {"PYTEST_PLUGINS": "unapproved_plugin"},
                {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}, {"PYTHONOPTIMIZE": "1"},
                {"CLEANRL_REFERENCE_FILE": "/unapproved/models.py"}):
            with self.subTest(environment=list(environment)), self.assertRaises(ValueError):
                site_validation.require_gate_controls(environment, reference)
        with self.assertRaisesRegex(ValueError, "PYTEST_PLUGINS") as error:
            site_validation.require_gate_controls({"PYTEST_PLUGINS": "private-plugin-value"}, reference)
        self.assertNotIn("private-plugin-value", str(error.exception))
        site_validation.require_gate_controls({"PYTEST_ADDOPTS": "", "PYTHONOPTIMIZE": ""}, reference)

    def test_parent_package_initializer_is_part_of_the_checked_numerical_source(self):
        """Changing executable package initialization must fail the same original-source byte check."""
        source = self.run / "parent-package-source"
        parent = source / "10_jax_exploration_platform/src/exploration_platform/__init__.py"
        network = parent.parent / "atari_pixels/networks.py"
        parent.parent.mkdir(parents=True)
        network.parent.mkdir()
        parent.write_text("# synthetic original package initialization\n")
        network.write_text("# synthetic original model\n")
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(source), "-c", "user.name=Metadata test", "-c", "user.email=metadata@example.invalid",
            "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Create isolated package source"], check=True)
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        subprocess.run(["git", "-C", str(source), "checkout", "-q", "--detach"], check=True)
        with patch.object(site_validation, "SOURCE_COMMIT", commit):
            paths = site_validation.numerical_paths(source)
        self.assertIn(str(parent.relative_to(source)), paths)
        identity = site_validation.source_identity(source, commit, paths)
        parent.write_text("# changed executable initialization\n")
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            site_validation.verify_source_identity(identity)

    def test_cleanrl_reference_file_is_checked_against_its_original_git_blob(self):
        """The external equations-test input is pinned even when its upstream checkout is on a branch."""
        source = self.run / "upstream-reference"
        path = source / "cleanrl/ppo_rnd_envpool.py"
        path.parent.mkdir(parents=True)
        path.write_text("# synthetic original class definitions, never executed\n")
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(source), "-c", "user.name=Metadata test", "-c", "user.email=metadata@example.invalid",
            "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Create isolated original reference"], check=True)
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        with patch.object(site_validation, "CLEANRL_COMMIT", commit):
            reference = site_validation.cleanrl_identity(source)
            self.assertEqual(reference["file"], site_validation.receipt(path))
            path.write_text("# different external class definitions\n")
            with self.assertRaisesRegex(ValueError, "unchanged original CleanRL"):
                site_validation.cleanrl_identity(source)

    def completed_site(self):
        """Create result-bound actual-style child records while performing no numerical calculation."""
        directory = self.run / "completed_site"
        directory.mkdir()
        installed = {"packages": {name: "captured-version" for name in site_validation.PACKAGE_NAMES},
            "jaxtari_source": {"source_commit": "a" * 40}, "gpu_libraries": {}}
        for name in ("libcudnn.so.9", "libcudart.so.12"):
            path = self.run / name
            path.write_bytes(b"synthetic installed image " + name.encode())
            installed["gpu_libraries"][str(path)] = {"sha256": site_validation.sha256(path), "version": "captured-version"}
        installed_path = self.write(self.run / "actual_shared_software.json", installed)
        assets = {"manifest_sha256": "a" * 64, "file_count": 1}
        assets_path = self.write(self.run / "actual_shared_assets.json", assets)
        plan = copy.deepcopy(self.plan)
        plan["reference"].update(shared_installed_backend=site_validation.receipt(installed_path), shared_asset_receipt=site_validation.receipt(assets_path),
            backend={"platform": "gpu", "platform_version": "PJRT C API\ncuda 12090", "cudnnGetVersion": 92400, "cudaRuntimeGetVersion": 12090})
        plan["configuration"] = {"learner": {"n_copies": 1, "n_envs": 128}, "environment": {"num_envs": 128}}
        self.write(directory / "site_validation_plan.json", plan)
        proof = site_validation.resolve_allocation(self.environment, "serval06", self.csv, self.xml)
        cpu = site_validation.load("infrastructure_retry").cpu_allocation(self.environment, [self.gpu], range(8))
        assignment = {"allocation": proof, "owned_job": {"job_id": "333", "ledger_path": str(self.run / "slurm/submitted_jobids_uva.txt"),
            "ledger_prefix": ["333"], "cgroup": "0::/slurm/job_333/step_0/task_0\n"}, "cpu_allocation": cpu,
            "cpu_affinity": list(range(8)), "runtime_environment": site_validation.RUNTIME, "parent_pid": 90, "parent_start_time": "99",
            "gpu_modes": {"gpu_uuid": self.gpu, "mig_mode": "Disabled", "memory_used_mib": 0, "compute_processes": [], "confidential_compute": "OFF"}}
        self.write(directory / "site_assignment.json", assignment)
        children = {}
        for index, kind in enumerate(("learner", "renderer", "probe")):
            owner = {"pid": 301 + index, "process_start_time": str(401 + index), "parent_pid": 90,
                "parent_start_time": "99", "kind": kind, "job_id": "333"}
            process = {**owner, "gpu_uuid": self.gpu, "source_commit": site_validation.SOURCE_COMMIT, "runtime_environment": site_validation.RUNTIME,
                "cpu_affinity": list(range(8)), "configuration": plan["configuration"], "installed_backend": installed,
                "asset_receipt": assets, "cleanrl_reference": plan["cleanrl_reference"],
                "gate_environment_policy": site_validation.GATE_ENVIRONMENT_POLICY,
                "original_command": site_validation.original_command(plan, kind) if kind != "probe" else None}
            self.write(directory / (kind + "_owner.json"), owner)
            self.write(directory / (kind + "_process.json"), process)
            (directory / (kind + ".log")).write_text("synthetic completed original child\n")
            self.write(directory / (kind + "_terminal.json"), {**owner, "exit_code": 0, "log": site_validation.receipt(directory / (kind + ".log"))})
            children[kind] = {key: site_validation.receipt(directory / (kind + suffix)) for key, suffix in
                (("owner", "_owner.json"), ("process", "_process.json"), ("terminal", "_terminal.json"))}
        libraries = installed["gpu_libraries"]
        loaded = {**plan["reference"]["backend"], "pid": 303, "verified_package_libraries": libraries,
            "loaded_shared_libraries": [*libraries, "/usr/lib/libcuda.so." + site_validation.DRIVER],
            "cudnn_library": str(self.run / "libcudnn.so.9"), "cuda_runtime_library": str(self.run / "libcudart.so.12")}
        probe = {"kind": "original_reference_site_probe", "completed": True, "iteration": 1, "pid": 303,
            "process_start_time": "403", "source_commit": site_validation.SOURCE_COMMIT, "gpu_uuid": self.gpu,
            "formal_measurement": False, "capacity_evidence": False, "scientific_hypothesis_credit": 0,
            "finite_state": {"passed": True, "floating_leaves_checked": 50, "scope": "complete updated PixelTrainState"},
            "dtypes": {name: ["float64"] if "statistics" in name else ["float32"] for name in
                ("parameters", "target", "optimizer_first", "optimizer_second", "observation_statistics", "reward_statistics")},
            "loaded_libraries": loaded, "numerical_module_paths": {"exploration_platform.atari_pixels." + name:
                str(Path(plan["numerical_source"]["source_root"]) / "10_jax_exploration_platform/src/exploration_platform/atari_pixels" / (name + ".py"))
                for name in ("config", "environment", "training", "networks", "state")}}
        self.write(directory / "probe_result.json", probe)
        result = {"schema_version": 1, "kind": "h100_nvl_slurm_site_evidence", "passed": True,
            "plan": site_validation.receipt(directory / "site_validation_plan.json"), "assignment": site_validation.receipt(directory / "site_assignment.json"),
            "children": children, "probe": site_validation.receipt(directory / "probe_result.json"), "workers_permitted": False,
            "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0}
        return directory, plan, result, probe

    def test_complete_site_receipt_verifies_original_children_without_admitting_workers(self):
        """A complete site validation is limited to its exact allocated site and remains outside scientific results."""
        directory, plan, result, probe = self.completed_site()
        checked = site_validation.verify_result(directory, plan, result)
        self.assertEqual(checked["gpu_uuid"], self.gpu)
        self.assertFalse(checked["workers_permitted"])
        result["campaign_claims_permitted"] = True
        with self.assertRaisesRegex(ValueError, "cannot grant workers"):
            site_validation.verify_result(directory, plan, result)

    def test_rehashed_probe_precision_change_or_gate_failure_cannot_pass(self):
        """Valid JSON hashes cannot replace real expected dtypes or successful original full gates."""
        directory, plan, result, probe = self.completed_site()
        probe["dtypes"]["observation_statistics"] = ["float32"]
        self.write(directory / "probe_result.json", probe)
        result["probe"] = site_validation.receipt(directory / "probe_result.json")
        with self.assertRaisesRegex(ValueError, "float32/float64"):
            site_validation.verify_result(directory, plan, result)
        probe["dtypes"]["observation_statistics"] = ["float64"]
        self.write(directory / "probe_result.json", probe)
        result["probe"] = site_validation.receipt(directory / "probe_result.json")
        path = directory / "learner_terminal.json"
        terminal = site_validation.read(path)
        terminal["exit_code"] = 1
        self.write(path, terminal)
        result["children"]["learner"]["terminal"] = site_validation.receipt(path)
        with self.assertRaisesRegex(ValueError, "successful children"):
            site_validation.verify_result(directory, plan, result)

    def test_site_probe_requires_successful_finite_evidence_for_updated_state(self):
        """Otherwise valid dtype/source/gate evidence cannot conceal NaN/Inf or an omitted state check."""
        directory, plan, result, probe = self.completed_site()
        for evidence in (None, {"passed": False, "floating_leaves_checked": 50, "scope": "complete updated PixelTrainState"},
                {"passed": True, "floating_leaves_checked": 0, "scope": "complete updated PixelTrainState"},
                {"passed": True, "floating_leaves_checked": True, "scope": "complete updated PixelTrainState"}):
            altered = copy.deepcopy(probe)
            if evidence is None:
                altered.pop("finite_state")
            else:
                altered["finite_state"] = evidence
            self.write(directory / "probe_result.json", altered)
            result["probe"] = site_validation.receipt(directory / "probe_result.json")
            with self.subTest(evidence=evidence), self.assertRaisesRegex(ValueError, "finite check"):
                site_validation.verify_result(directory, plan, result)

    def test_prepared_plan_rebuilds_complete_configuration_and_reference_semantics(self):
        """Rehashed transport metadata cannot change a learning knob, environment knob or observed backend."""
        directory, plan, result, probe = self.completed_site()
        reference = plan["reference"]
        for name in ("hardware", "loaded_libraries", "diagnostic_plan", "dtype_record"):
            path = self.write(self.run / "reference" / (name + ".json"), {"original": name})
            reference[name] = site_validation.receipt(path)
        reference.update(driver=site_validation.DRIVER, model=site_validation.MODEL, memory_mib=site_validation.MEMORY_MIB,
            software={"result": site_validation.receipt(self.write(self.run / "software_result.json", {})),
                "installed_backend": reference["shared_installed_backend"], "asset_receipt": reference["shared_asset_receipt"],
                "packages": {name: "captured-version" for name in site_validation.PACKAGE_NAMES}})
        plan.update(schema_version=2, kind="h100_nvl_slurm_site_validation", source_commit=site_validation.SOURCE_COMMIT,
            runtime_environment=site_validation.RUNTIME, gate_environment_policy=site_validation.GATE_ENVIRONMENT_POLICY,
            job_shape=site_validation.JOB_SHAPE, submission_control=site_validation.SUBMISSION_CONTROL,
            reference_updates=1, workers_permitted=False, campaign_claims_permitted=False,
            scientific_hypothesis_credit=0, packages=reference["software"]["packages"], fresh_packages_at_prepare=reference["software"]["packages"])
        plan["helpers"] = {"source_root": str(SCRIPTS.parents[2]), "source_commit": "a" * 40,
            "files": {site_validation.PREFIX + name: "b" * 64 for name in site_validation.HELPERS}}
        paths = ["10_jax_exploration_platform/src/exploration_platform/__init__.py"]
        plan["numerical_source"].update(source_commit=site_validation.SOURCE_COMMIT, files={name: "c" * 64 for name in paths})
        admission_files = [site_validation.receipt(self.write(self.run / "admission" / (name + ".json"), {"record": name}))
            for name in ("source_and_gates", "reference_configuration", "h100_gate", "a4500_gate")]
        plan["admission"] = {"source_root": plan["numerical_source"]["source_root"], "files": admission_files}
        plan["admission_archive"] = site_validation.archive_admission(directory, plan["admission"])
        plan["configuration"]["learner"]["learning_rate"] = .0001
        expected_configuration = copy.deepcopy(plan["configuration"])
        expected_reference = copy.deepcopy(reference)
        (directory / "job.sbatch").write_text("#!/bin/bash\n# synthetic metadata-only script, never executed\n")
        def publish(value):
            self.write(directory / "site_validation_plan.json", value)
            self.write(directory / "manifest.yaml", {"plan": site_validation.receipt(directory / "site_validation_plan.json"),
                "job_script": site_validation.receipt(directory / "job.sbatch")})
        with patch.object(site_validation, "verify_source_identity"), patch.object(site_validation, "numerical_paths", return_value=paths), \
                patch.object(site_validation, "cleanrl_identity", return_value=plan["cleanrl_reference"]), \
                patch.object(site_validation, "original_configuration", return_value=expected_configuration), \
                patch.object(site_validation, "reference_identity", return_value=expected_reference) as rebuild:
            publish(plan)
            site_validation.prepared(directory)
            rebuild.assert_called_once()
            missing_archive = copy.deepcopy(plan)
            missing_archive.pop("admission_archive")
            publish(missing_archive)
            with self.assertRaisesRegex(ValueError, "admission archive"):
                site_validation.prepared(directory)
            for scope, field, value in (("learner", "learning_rate", .01), ("environment", "num_envs", 64)):
                altered = copy.deepcopy(plan)
                altered["configuration"][scope][field] = value
                publish(altered)
                with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "complete original fixed"):
                    site_validation.prepared(directory)
            altered = copy.deepcopy(plan)
            altered["reference"]["backend"]["cudnnGetVersion"] = 123
            publish(altered)
            with self.assertRaisesRegex(ValueError, "canonical software proof"):
                site_validation.prepared(directory)

    def test_generated_submission_stays_held_until_parent_fsyncs_and_releases_exact_id(self):
        """Building the command cannot race job startup or perform submission/release itself."""
        with patch.object(site_validation.subprocess, "Popen") as launch, patch.object(site_validation, "query") as query:
            command = site_validation.submission_command(self.run, self.run / "new_site_validation")
        self.assertEqual(command[:3], ["sbatch", "--parsable", "--hold"])
        self.assertIn("--gres=gpu:nvidia_h100_nvl:1", command)
        self.assertIn("--cpus-per-task=8", command)
        self.assertFalse(site_validation.SUBMISSION_CONTROL["helper_submits_or_releases"])
        self.assertTrue(site_validation.SUBMISSION_CONTROL["release_exact_saved_id_only"])
        procedure = site_validation.SUBMISSION_CONTROL["parent_procedure"]
        self.assertIn("fsync", procedure[1])
        self.assertIn("submitted_jobids_uva.txt", procedure[1])
        self.assertIn("submission receipt", procedure[1])
        self.assertIn("Only after both records are durable", procedure[2])
        self.assertEqual(site_validation.SUBMISSION_CONTROL["release_command_template"], ["scontrol", "release", "<exact_saved_uva_job_id>"])
        launch.assert_not_called()
        query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
