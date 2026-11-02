"""Check selected-configuration preparation and memory records without numerical imports."""

import copy
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
SPECIFICATION = importlib.util.spec_from_file_location("selected_measurement_producer", SCRIPTS / "measure_selected.py")
PRODUCER = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(PRODUCER)


class SelectedMeasurementTests(unittest.TestCase):
    """Keep preparation metadata-only and distinguish physical-device memory from process memory."""

    def setUp(self):
        """Create complete source-default configuration fixtures and explicit runtime inputs."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = Path(self.temporary.name)
        self.source = SCRIPTS.parents[2]
        defaults = PRODUCER.diagnostic.default_configuration(self.source)
        self.reference = {name: defaults[name] for name in ("learner", "environment")}
        self.reference["execution"] = {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
                                       "report_readout": "none", "report_every_iterations": 10}
        self.selected = copy.deepcopy(self.reference)
        self.selected["learner"]["rollout_unroll"] = 2
        self.runtime = {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                        "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}
        self.files = {}
        for name, document in (("selected", self.selected), ("reference", self.reference), ("runtime", self.runtime),
                               ("assets", {"root": "/staged/assets", "manifest": "/staged/manifest.json", "sha256": "a" * 64})):
            path = self.run_dir / (name + ".json")
            path.write_text(json.dumps(document))
            self.files[name] = path
        self.helpers = {"source_root": str(self.source), "source_commit": "b" * 40,
            "files": {"10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name:
                      hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest() for name in PRODUCER.HELPERS}}
        self.frozen_source_requests = []

    def test_backend_versions_are_actual_complete_and_missing_packages_reject(self):
        """Retain every installed package identity without guessing a missing distribution version."""
        source, libraries = {"source_commit": "a" * 40}, {"images": ["verified-image"]}
        versions = {name: "observed-" + name for name in PRODUCER.training.PACKAGE_NAMES}
        with patch.object(PRODUCER.metadata, "version", side_effect=versions.__getitem__):
            self.assertEqual(PRODUCER.installed_backend_receipt(source, libraries),
                             {"jaxtari_source": source, "gpu_libraries": libraries, "packages": versions})
        with patch.object(PRODUCER.metadata, "version", side_effect=PRODUCER.metadata.PackageNotFoundError("jax")):
            with self.assertRaises(PRODUCER.metadata.PackageNotFoundError):
                PRODUCER.installed_backend_receipt(source, libraries)
        self.assertNotIn("jax", sys.modules)

    def test_compute_receives_the_same_full_observed_hardware_as_its_sidecar(self):
        """Capture applied defaults and cache flags once, rather than replacing them with planned flags."""
        arguments = self.arguments()
        arguments.python = Path(sys.executable)
        directory = self.prepare(arguments)
        hardware = {"hostname": "jaguar03", "pid": os.getpid(), "gpu_uuid": "GPU-fixture"}

        def compute(output, plan, observed, backend):
            """Inspect the producer's real metadata boundary while performing no numerical work."""
            self.assertEqual(observed, json.loads((output / "hardware.json").read_text()))
            runtime = observed["runtime_environment"]
            self.assertEqual(runtime["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")
            self.assertEqual(runtime["JAXTARI_SKIP_SPRITE_UPDATE"], "1")
            self.assertEqual(runtime["JAX_ENABLE_COMPILATION_CACHE"], "false")
            self.assertEqual(runtime["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"], "")
            self.assertGreater(len(runtime), len(plan["runtime_environment"]))

        with patch.object(PRODUCER, "__file__", str(directory / "code/measure_selected.py")), \
                patch.object(PRODUCER, "verify_prepared"), \
                patch.object(PRODUCER, "card_admission", return_value=nullcontext()), \
                patch.object(PRODUCER, "preflight", return_value=hardware), \
                patch.object(PRODUCER.training, "verify_assets", return_value={}), \
                patch.object(PRODUCER.training, "jaxtari_python_receipt", return_value={}), \
                patch.object(PRODUCER.training, "installed_gpu_library_receipts", return_value={}), \
                patch.object(PRODUCER, "installed_backend_receipt", return_value={}), \
                patch.object(PRODUCER, "_compute", side_effect=compute) as numeric, \
                patch.dict(os.environ, {}, clear=True):
            PRODUCER.run_prepared(directory)
        numeric.assert_called_once()
        self.assertNotIn("jax", sys.modules)

    def arguments(self, *extra):
        """Parse the real CLI with explicit input paths and no GPU execution flag."""
        return PRODUCER.parse_arguments(["--prepare-only", "--mode", "confirmation", "--source-root", str(self.source),
            "--source-commit", "c" * 40, "--configuration", str(self.files["selected"]),
            "--reference-configuration", str(self.files["reference"]), "--runtime-environment", str(self.files["runtime"]),
            "--asset-record", str(self.files["assets"]), "--run-dir", str(self.run_dir),
            "--unit-id", "selected_configuration_complete_updates_seed-1", "--gpu-uuid", "GPU-fixture",
            "--hardware-class", "h100", *extra])

    def prepare(self, arguments):
        """Mock only committed-source identity, preserving real metadata validation and staging."""
        def receipt(root, commit, paths):
            """Retain the exact source dependency list rather than bypassing its coverage check."""
            self.frozen_source_requests.append(list(paths))
            return {"source_commit": commit, "files": {name: "c" * 64 for name in paths}}
        with patch.object(PRODUCER.diagnostic, "core_identity", return_value={"source_commit": "c" * 40}), \
                patch.object(PRODUCER.training, "source_receipt", side_effect=receipt), \
                patch.object(PRODUCER, "helper_receipt", return_value=self.helpers), \
                patch.object(PRODUCER, "selected_query") as gpu, patch.object(PRODUCER, "_compute") as numeric:
            directory = PRODUCER.prepare(arguments)
            gpu.assert_not_called()
            numeric.assert_not_called()
        return directory

    def test_full_configuration_and_helpers_are_staged_before_any_gpu_access(self):
        """A group-five selected variant preserves the requested seven-by-ten protocol."""
        selected = copy.deepcopy(self.selected)
        selected["execution"]["iterations_per_call"] = 5
        self.files["selected"].write_text(json.dumps(selected))
        directory = self.prepare(self.arguments())
        plan = json.loads((directory / "measurement_plan.json").read_text())
        self.assertEqual((plan["rounds"], plan["iterations"]), (7, 10))
        self.assertEqual(plan["configuration"]["execution"]["iterations_per_call"], 5)
        self.assertEqual(plan["reference_configuration"]["execution"]["iterations_per_call"], 1)
        self.assertEqual(plan["hypothesis_count_contribution"], 0)
        self.assertEqual(set(self.frozen_source_requests[0]), {
            "10_jax_exploration_platform/scripts/atari_pixel_campaign/paired_benchmark.py",
            "10_jax_exploration_platform/scripts/atari_pixel_campaign/ideas.py"})
        for name in ("manifest.yaml", "config_resolved.yaml", "experiment_background.md", "command.txt", "measurement_prepared.json"):
            self.assertTrue((directory / name).is_file())
        for name in PRODUCER.HELPERS:
            self.assertEqual((directory / "code" / name).read_bytes(), (SCRIPTS / name).read_bytes())
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_packing_changes_only_copy_count_and_retains_no_reference(self):
        """One packing invocation has one resident configuration and explicit independent seeds."""
        arguments = self.arguments()
        arguments.mode, arguments.reference_configuration, arguments.copies = "packing-point", None, 2
        directory = self.prepare(arguments)
        plan = json.loads((directory / "measurement_plan.json").read_text())
        self.assertIsNone(plan["reference_configuration"])
        self.assertEqual(plan["configuration"]["learner"]["n_copies"], 2)
        self.assertEqual(plan["copy_seeds"], [1, 2])
        expected = copy.deepcopy(plan["configuration"])
        expected["learner"]["n_copies"] = 1
        self.assertEqual(expected, json.loads(json.dumps(self.selected)))
        self.assertNotIn('device_memory_limit_protocol', plan)
        self.assertFalse((directory / 'capacity_owned_job_ledger.txt').exists())

    def test_instrumented_packing_opt_in_snapshots_ledger_without_gpu_access(self):
        """The optional capability copies ownership bytes once while leaving ordinary preparation unchanged."""
        arguments = self.arguments()
        ledger = self.run_dir / 'slurm/submitted_jobids_uva.txt'
        ledger.parent.mkdir()
        ledger.write_text('123\n')
        arguments.mode, arguments.reference_configuration, arguments.copies = 'packing-point', None, 2
        arguments.record_device_memory_limit = True
        arguments.memory_limit_allocation_kind, arguments.memory_limit_cluster = 'slurm', 'uva'
        arguments.memory_limit_owned_job_ledger = ledger
        directory = self.prepare(arguments)
        plan = json.loads((directory / 'measurement_plan.json').read_text())
        snapshot = directory / 'capacity_owned_job_ledger.txt'
        self.assertEqual(snapshot.read_bytes(), b'123\n')
        self.assertEqual(plan['device_memory_limit_protocol']['owned_job_ledger']['path'], str(snapshot))
        self.assertEqual(plan['device_memory_limit_ledger_source']['path'], str(ledger))
        ledger.write_text('123\n124\n')
        self.assertEqual(snapshot.read_bytes(), b'123\n')
        self.assertNotIn('jax', sys.modules)

    def test_memory_limit_opt_in_refuses_paired_comparison_modes(self):
        """Diagnostic memory failures cannot be attached to ordinary correctness or confirmation work."""
        arguments = self.arguments()
        arguments.record_device_memory_limit = True
        arguments.memory_limit_allocation_kind, arguments.memory_limit_cluster = 'direct', 'uva'
        arguments.memory_limit_owned_job_ledger = None
        with self.assertRaisesRegex(ValueError, 'isolated packing'):
            self.prepare(arguments)

    def test_prepared_execution_rejects_all_memory_preparation_overrides(self):
        """Execution cannot silently ignore a memory-protocol preparation flag."""
        for extra in (['--record-device-memory-limit'], ['--memory-limit-allocation-kind', 'slurm'],
                      ['--memory-limit-cluster', 'uva'], ['--memory-limit-owned-job-ledger', '/unused/ledger.txt']):
            with self.subTest(extra=extra), patch.object(sys, 'stderr', io.StringIO()), self.assertRaises(SystemExit):
                PRODUCER.parse_arguments(['--prepared-unit', '/unused/prepared-unit', *extra])

    def test_memory_preparation_options_require_explicit_opt_in(self):
        """A misplaced memory option fails at parsing before any launch artifact is created."""
        before = sorted(path.relative_to(self.run_dir) for path in self.run_dir.rglob('*'))
        for extra in (['--memory-limit-allocation-kind', 'direct'], ['--memory-limit-cluster', 'uva'],
                      ['--memory-limit-owned-job-ledger', '/unused/ledger.txt']):
            with self.subTest(extra=extra), patch.object(sys, 'stderr', io.StringIO()), self.assertRaises(SystemExit):
                self.arguments(*extra)
        self.assertEqual(before, sorted(path.relative_to(self.run_dir) for path in self.run_dir.rglob('*')))

    def test_private_child_flag_requires_a_declared_prepared_packing_protocol(self):
        """The private entry cannot be attached to preparation or an ordinary prepared measurement."""
        with patch.object(sys, 'stderr', io.StringIO()), self.assertRaises(SystemExit):
            self.arguments('--instrumented-child')
        ordinary = self.prepare(self.arguments())
        with patch.object(sys, 'stderr', io.StringIO()), self.assertRaises(SystemExit):
            PRODUCER.parse_arguments(['--prepared-unit', str(ordinary), '--instrumented-child'])
        arguments = self.arguments()
        arguments.unit_id += '_instrumented'
        arguments.mode, arguments.reference_configuration = 'packing-point', None
        arguments.record_device_memory_limit = True
        arguments.memory_limit_allocation_kind, arguments.memory_limit_cluster = 'direct', 'uva'
        arguments.memory_limit_owned_job_ledger = None
        instrumented = self.prepare(arguments)
        parsed = PRODUCER.parse_arguments(['--prepared-unit', str(instrumented), '--instrumented-child'])
        self.assertTrue(parsed.instrumented_child)
        self.assertNotIn('jax', sys.modules)

    def test_explicit_ownership_profile_requires_instrumentation_and_rejects_execution_override(self):
        """A profile cannot be silently ignored during ordinary preparation or prepared execution."""
        name = 'uva_slurm_cgroup_v2_process_uid_ownership'
        with patch.object(sys, 'stderr', io.StringIO()), self.assertRaises(SystemExit):
            self.arguments('--memory-limit-ownership-profile', name)
        ordinary = self.prepare(self.arguments())
        with patch.object(sys, 'stderr', io.StringIO()), self.assertRaises(SystemExit):
            PRODUCER.parse_arguments(['--prepared-unit', str(ordinary), '--memory-limit-ownership-profile', name])

    def test_explicit_uva_profile_is_staged_only_for_matching_instrumented_allocation(self):
        """Mock publication only; real profile selection rejects direct or ACES use before producing artifacts."""
        import device_memory_limit as capacity
        args = self.arguments(); args.unit_id += '_explicit_ownership'
        args.mode, args.reference_configuration = 'packing-point', None
        args.record_device_memory_limit = True
        args.memory_limit_ownership_profile = capacity.UVA_CGROUP_V2_OWNERSHIP_PROFILE['kind']
        args.memory_limit_allocation_kind, args.memory_limit_cluster = 'direct', 'uva'
        args.memory_limit_owned_job_ledger = None
        with self.assertRaisesRegex(ValueError, 'only for UVA Slurm'):
            self.prepare(args)
        ledger = self.run_dir / 'slurm/submitted_jobids_uva.txt'
        ledger.parent.mkdir(); ledger.write_text('123\n')
        args.memory_limit_allocation_kind, args.memory_limit_owned_job_ledger = 'slurm', ledger
        with patch.object(capacity, 'require_ownership_source') as source:
            directory = self.prepare(args)
        plan = json.loads((directory / 'measurement_plan.json').read_text())
        self.assertEqual(plan['ownership_profile'], capacity.UVA_CGROUP_V2_OWNERSHIP_PROFILE)
        self.assertEqual(source.call_args.args[0]['ownership_profile'], plan['ownership_profile'])
        self.assertEqual((directory / 'capacity_owned_job_ledger.txt').read_bytes(), b'123\n')
        self.assertNotIn('jax', sys.modules)

    def test_completed_instrumented_attempt_retains_raw_data_without_ordinary_result_credit(self):
        """A successful extra-sampler attempt needs a fresh ordinary measurement before throughput can be used."""
        directory = self.run_dir / 'completed_instrumented'
        directory.mkdir()
        plan = {'mode': 'packing-point', 'device_memory_limit_protocol': {}}
        raw = {'completed': True, 'rounds': [{'seconds': 2., 'training_transitions': 163840, 'memory': {'actual': True}}]}
        PRODUCER.save_measurement_result(directory, plan, raw)
        self.assertFalse((directory / 'result.json').exists())
        result = json.loads((directory / 'instrumented_completion.json').read_text())
        self.assertEqual(result['rounds'], raw['rounds'])
        self.assertFalse(result['measurement_valid'])
        self.assertFalse(result['formal_packing_evidence'])
        for key in ('packing_point_credit', 'throughput_credit', 'case_credit', 'idea_credit'):
            self.assertEqual(result[key], 0)
        ordinary = self.run_dir / 'ordinary_completed'
        ordinary.mkdir()
        PRODUCER.save_measurement_result(ordinary, {'mode': 'packing-point'}, raw)
        self.assertEqual(json.loads((ordinary / 'result.json').read_text()), raw)

    def test_algorithm_changes_and_output_reuse_are_rejected(self):
        """No source preparation or partial artifact can authorize a changed CleanRL setting."""
        arguments = self.arguments()
        directory = self.prepare(arguments)
        before = (directory / "measurement_plan.json").read_bytes()
        changed = copy.deepcopy(self.selected)
        changed["learner"]["learning_rate"] = .01
        self.files["selected"].write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "CleanRL"):
            self.prepare(arguments)
        self.assertEqual((directory / "measurement_plan.json").read_bytes(), before)

    def test_expanded_complete_update_window_is_explicit(self):
        """A future group-three configuration needs twelve real updates rather than padded work."""
        selected = copy.deepcopy(self.selected)
        selected["execution"]["iterations_per_call"] = 3
        self.assertEqual(PRODUCER.measurement_window([self.reference, selected]), 12)
        reported = copy.deepcopy(self.reference)
        reported["execution"]["report_readout"] = "scheduled"
        self.assertEqual(PRODUCER.measurement_window([reported, selected]), 30)
        with self.assertRaises(ValueError):
            PRODUCER.measurement_window([self.reference], 1)

    def test_unsupported_frozen_correctness_group_pair_fails_during_preparation(self):
        """The old gate cannot silently replace its five-versus-three update check with a new one."""
        reference, selected = copy.deepcopy(self.reference), copy.deepcopy(self.selected)
        reference["execution"]["iterations_per_call"] = 5
        selected["execution"]["iterations_per_call"] = 3
        self.files["reference"].write_text(json.dumps(reference))
        self.files["selected"].write_text(json.dumps(selected))
        with self.assertRaisesRegex(ValueError, "frozen correctness protocol"):
            self.prepare(self.arguments())
        self.assertFalse((self.run_dir / "data").exists())

    def test_memory_sampling_records_driver_usage_and_contamination(self):
        """Device used/free samples include memory outside the measured Python process."""
        device = types.SimpleNamespace(memory_stats=lambda: {"peak_bytes_in_use": 1234})
        observation = PRODUCER.MemoryObservation(device, "GPU-fixture")
        def query(gpu, fields, **options):
            """Supply one selected physical device and actual-style process-memory rows."""
            self.assertEqual(gpu, "GPU-fixture")
            if options.get("processes"):
                return [[str(os.getpid()), "GPU-fixture", "95"], ["999999", "GPU-fixture", "1"]]
            return [["GPU-fixture", "1000", "100", "900"]]
        with patch.object(PRODUCER, "selected_query", side_effect=query):
            observation.begin()
            measured = observation.finish()
        self.assertEqual(measured["sampled_process_peak_bytes"], 95 * 2 ** 20)
        self.assertEqual(measured["other_gpu_processes"], [999999])
        self.assertGreaterEqual(len(measured["device_memory_samples"]), 2)
        self.assertEqual(measured["device_memory_samples"][0]["used_bytes"], 100 * 2 ** 20)
        self.assertEqual(measured["device_memory_samples"][0]["free_bytes"], 900 * 2 ** 20)

    def test_login_and_serval02_other_card_fail_before_gpu_access(self):
        """A mistaken location or UUID cannot initialize arrays or query an unrelated card."""
        plan = {"gpu_uuid": "GPU-fixture", "hardware_class": "h100"}
        for host in ("portal05", "serval05"):
            with self.subTest(host=host), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-fixture"}, clear=True), \
                    patch.object(PRODUCER.socket, "gethostname", return_value=host), \
                    patch.object(PRODUCER, "selected_query") as query:
                with self.assertRaisesRegex(ValueError, "compute allocation"):
                    PRODUCER.preflight(plan)
                query.assert_not_called()
        with patch.object(PRODUCER.socket, "gethostname", return_value="serval02"), \
                patch.object(PRODUCER, "selected_query") as query:
            with self.assertRaisesRegex(ValueError, "card-1"):
                with PRODUCER.card_admission(self.run_dir, plan):
                    self.fail("wrong-card admission entered")
            query.assert_not_called()

    def test_slurm_mapping_uses_allocated_physical_gpu(self):
        """A selected UUID is verified against allocation index five rather than CUDA ordinal zero."""
        calls = []
        def query(gpu, fields, **options):
            """Return the exact allocated A4500 for the selected UUID and physical Slurm index."""
            calls.append((gpu, fields))
            if options.get("processes"):
                return []
            return [["GPU-fifth", "NVIDIA RTX A4500", "20470", "driver", "[N/A]", "0"]]
        xml = ("<nvidia_smi_log><gpu><uuid>GPU-fifth</uuid><minor_number>5</minor_number>"
               "<product_name>NVIDIA RTX A4500</product_name></gpu></nvidia_smi_log>")
        def remapped_query(command, **options):
            """Visible index zero has physical minor five; '-i 5' is never a valid query here."""
            output = {("nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"):
                      "0, GPU-fifth, NVIDIA RTX A4500\n", ("nvidia-smi", "-q", "-x"): xml}
            return subprocess.CompletedProcess(command, 0, output[tuple(command)], "")
        environment = {"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_ID": "1234", "SLURM_JOB_GPUS": "5"}
        with patch.dict(os.environ, environment, clear=True), \
                patch.object(PRODUCER.socket, "gethostname", return_value="jaguar03"), \
                patch.object(PRODUCER.subprocess, "run", side_effect=remapped_query), \
                patch.object(PRODUCER, "selected_query", side_effect=query):
            record = PRODUCER.preflight({"gpu_uuid": "GPU-fifth", "hardware_class": "jaguar03_rtx_a4500"})
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-fifth")
        self.assertEqual(record["gpu_uuid"], "GPU-fifth")
        self.assertEqual(record["cuda_visible_devices_original"], "0")
        self.assertTrue(all(gpu == "GPU-fifth" for gpu, fields in calls))

    def test_remapped_visible_card_must_match_every_slurm_physical_allocation(self):
        """A visible ordinal zero cannot select another physical card or ignore a step conflict."""
        xml = ("<nvidia_smi_log><gpu><uuid>GPU-fifth</uuid><minor_number>5</minor_number>"
               "<product_name>NVIDIA RTX A4500</product_name></gpu></nvidia_smi_log>")
        def query(command, **options):
            """Expose one remapped physical device through the same two metadata queries."""
            value = xml if command == ["nvidia-smi", "-q", "-x"] else "0, GPU-fifth, NVIDIA RTX A4500\n"
            return subprocess.CompletedProcess(command, 0, value, "")
        for changes in ({"SLURM_JOB_GPUS": "0"}, {"SLURM_JOB_GPUS": "GPU-other"},
                        {"SLURM_STEP_GPUS": "4"}, {"CUDA_VISIBLE_DEVICES": "GPU-other"}):
            environment = {"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "5", **changes}
            with self.subTest(changes=changes), patch.object(PRODUCER.subprocess, "run", side_effect=query):
                with self.assertRaises(ValueError):
                    PRODUCER.allocated_gpu_uuid("jaguar03_rtx_a4500", environment)

    def test_imported_source_and_installed_jaxtari_paths_are_bound(self):
        """A valid installed-file receipt cannot admit a different shadowing Python package."""
        installed = self.run_dir / "installed/jaxtari/__init__.py"
        installed.parent.mkdir(parents=True)
        installed.write_text("")
        root = self.source / "10_jax_exploration_platform"
        paired = types.SimpleNamespace(__file__=str(SCRIPTS / "paired_benchmark.py"))
        ideas = types.SimpleNamespace(__file__=str(SCRIPTS / "ideas.py"))
        jaxtari = types.SimpleNamespace(__file__=str(installed))
        distribution = types.SimpleNamespace(locate_file=lambda name: installed)
        with patch.dict(sys.modules, {"ideas": ideas, "jaxtari": jaxtari}), \
                patch.object(PRODUCER.metadata, "distribution", return_value=distribution):
            PRODUCER.verify_imported_sources(root, paired)
            ideas.__file__ = str(self.run_dir / "shadow/ideas.py")
            with self.assertRaisesRegex(ValueError, "ideas"):
                PRODUCER.verify_imported_sources(root, paired)
            ideas.__file__ = str(SCRIPTS / "ideas.py")
            jaxtari.__file__ = str(self.run_dir / "shadow/jaxtari/__init__.py")
            with self.assertRaisesRegex(ValueError, "imported JAXtari"):
                PRODUCER.verify_imported_sources(root, paired)

    def test_completed_result_created_before_lock_acquisition_is_preserved(self):
        """The locked duplicate check closes the gap after the initial result-presence check."""
        arguments = self.arguments()
        arguments.python = Path(sys.executable)
        directory = self.prepare(arguments)
        result = b'{"completed":true,"original":true}\n'
        def locked(descriptor, operation):
            """Represent the first process completing just before the second takes the lock."""
            (directory / "result.json").write_bytes(result)
        with patch.object(PRODUCER, "__file__", str(directory / "code/measure_selected.py")), \
                patch.object(PRODUCER.fcntl, "flock", side_effect=locked), \
                patch.object(PRODUCER, "verify_prepared") as verify, patch.object(PRODUCER, "_compute") as numeric:
            with self.assertRaisesRegex(ValueError, "retain the previous"):
                PRODUCER.run_prepared(directory)
            verify.assert_not_called()
            numeric.assert_not_called()
        self.assertEqual((directory / "result.json").read_bytes(), result)
        self.assertFalse((directory / "failure.json").exists())

    def test_mutable_entrypoint_cannot_execute_a_prepared_unit(self):
        """Actual numerical invocation must use the staged producer command rather than mutable working code."""
        directory = self.prepare(self.arguments())
        with patch.object(PRODUCER, "_compute") as numeric:
            with self.assertRaisesRegex(ValueError, "immutable staged"):
                PRODUCER.run_prepared(directory)
            numeric.assert_not_called()


if __name__ == "__main__":
    unittest.main()
