"""Verify diagnostic metadata and unchanged original calls without importing numerical libraries."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import diagnose_pregather_allocator as diagnosis


class PregatherAllocatorTests(unittest.TestCase):
    """Use explicit synthetic original-case/reference/parser records and no compute allocation."""

    def setUp(self):
        """Build complete source-derived configuration and a separate held-failure input."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source, self.run = self.root / "source", self.root / "campaign"
        self.source.mkdir()
        self.run.mkdir()
        defaults = diagnosis.diagnostic.default_configuration(SCRIPTS.parents[2])
        self.configuration = {name: defaults[name] for name in ("learner", "environment")}
        self.configuration["learner"]["n_copies"] = 2
        self.configuration["execution"] = {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
            "report_readout": "none", "report_every_iterations": 10}
        self.case_path = self.root / "original_case/attempt-0004/case.json"
        self.case = {"source_root": str(self.source), "source_commit": diagnosis.SOURCE_COMMIT,
            "campaign_run_dir": str(self.run), "hypothesis_position": 23, "hypothesis": "gather_epoch_minibatches_in_advance",
            "candidate": {"learner": {"pre_gather_minibatches": True}, "environment": {}, "execution": {}},
            "configuration": self.configuration["learner"], "rounds": 7, "iterations": 1, "seed_start": 1,
            "reference_environment": None, "reference_execution": None}
        self.write(self.case_path, self.case)
        self.write(self.case_path.with_name("failure.json"), {"attempt": 4, "case_completed": False,
            "source_commit": diagnosis.SOURCE_COMMIT, "error": "RESOURCE_EXHAUSTED: Out of memory in jit_update"})
        self.reference = self.write(self.root / "completed_reference.json", {"source_commit": diagnosis.SOURCE_COMMIT,
            "case_completed": True, "hardware_class": "jaguar03_rtx_a4500", "copies": 2, "reference": self.configuration})
        self.runtime = self.write(self.root / "runtime.json", {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
            "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"})
        self.assets = self.write(self.root / "assets.json", {"root": "/original/assets", "manifest": "/original/manifest.json", "sha256": "a" * 64})
        self.public = self.root / "public_source"
        self.public.mkdir()
        self.parser = self.root / "xla_client.py"
        self.parser.write_text("raise AssertionError('never import parser source on metadata host')\n"
            "def generate_pjrt_gpu_plugin_options():\n"
            "    allowed_allocators = ('default', 'cuda_async')\n"
            "    allocator = os.getenv('XLA_PYTHON_CLIENT_ALLOCATOR', 'default')\n"
            "    fraction = os.getenv('XLA_PYTHON_CLIENT_MEM_FRACTION', '')\n"
            "    preallocate = os.getenv('XLA_PYTHON_CLIENT_PREALLOCATE', '')\n")
        self.case_patch = patch.object(diagnosis, "CASE_SHA256", hashlib.sha256(self.case_path.read_bytes()).hexdigest())
        self.case_patch.start()
        self.addCleanup(self.case_patch.stop)

    def write(self, path, value):
        """Write one complete synthetic metadata fixture."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")
        return path

    def arguments(self):
        """Keep every diagnostic input explicit and independently inspectable."""
        return types.SimpleNamespace(source_root=self.source, run_dir=self.run, case=self.case_path,
            reference_record=self.reference, runtime_environment=self.runtime, asset_record=self.assets,
            jaxtari_source_root=self.public, allocator_parser_source=self.parser, unit_id="original_pregather_async_allocator_diagnosis",
            gpu_uuid="GPU-diagnostic", python=Path(sys.executable))

    def prepare(self):
        """Mock only source/Git plumbing that requires a committed fixture repository."""
        helpers = {"source_root": str(SCRIPTS.parents[2]), "source_commit": "b" * 40,
            "files": {"10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name:
                hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest() for name in diagnosis.HELPERS}}
        with patch.object(diagnosis, "helper_receipt", return_value=helpers), \
                patch.object(diagnosis.measurement, "validate_complete_fields"), \
                patch.object(diagnosis.diagnostic, "core_identity", return_value={}), \
                patch.object(diagnosis.training, "source_receipt", return_value={"files": {}}), \
                patch.object(diagnosis.measurement, "preflight") as preflight, patch.object(diagnosis, "compute") as compute:
            directory = diagnosis.prepare(self.arguments())
            preflight.assert_not_called()
            compute.assert_not_called()
        return directory

    def test_metadata_preparation_preserves_original_case_and_zero_credit(self):
        """No original record, numerical setting, timing protocol or held case is rewritten."""
        original = self.case_path.read_bytes()
        directory = self.prepare()
        plan = json.loads((directory / "measurement_plan.json").read_text())
        self.assertEqual((plan["rounds"], plan["iterations"], plan["latency_calls"], plan["copy_seeds"]), (7, 1, 3, [1, 2]))
        self.assertEqual(plan["allocator"], {"XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
        self.assertIsNone(plan["memory_preparation"])
        self.assertEqual(plan["hypothesis_count_contribution"], 0)
        self.assertFalse(plan["eligible_for_formal_campaign_admission"])
        self.assertEqual((directory / "inputs/original_case.json").read_bytes(), original)
        self.assertEqual(self.case_path.read_bytes(), original)
        self.assertIn("failure.json", plan["inputs"]["original_failure"]["path"])
        self.assertNotIn("jax", sys.modules)

    def test_installed_allocator_parser_is_read_as_source_not_imported(self):
        """The real supported option must be literal in the installed parser before any compute work."""
        record = diagnosis.allocator_source_receipt(self.parser)
        self.assertIn("cuda_async", record["allowed_allocators"])
        self.assertEqual(record["sha256"], hashlib.sha256(self.parser.read_bytes()).hexdigest())
        self.parser.write_text(self.parser.read_text().replace("'cuda_async'", "'unsupported'"))
        with self.assertRaisesRegex(ValueError, "does not explicitly support"):
            diagnosis.allocator_source_receipt(self.parser)

    def test_changed_case_or_failure_stage_is_rejected(self):
        """A different case, minibatch count or unrelated failure cannot be diagnosed under this identity."""
        self.case_path.write_bytes(self.case_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "exact original pregather"):
            diagnosis.original_inputs(self.case_path, self.reference)
        self.write(self.case_path, self.case)
        self.write(self.case_path.with_name("failure.json"), {"attempt": 4, "source_commit": diagnosis.SOURCE_COMMIT,
            "case_completed": False, "error": "unrelated failure"})
        with self.assertRaisesRegex(ValueError, "allocation failure"):
            diagnosis.original_inputs(self.case_path, self.reference)

    def test_original_correctness_functions_receive_the_same_inputs_and_return_objects(self):
        """Only metadata callbacks are added around original fixed-input and whole-iteration calls."""
        events, report = [], {"original_checks": True}
        def fixed(*arguments, **keywords):
            """Retain exact arguments for the original fixed-input comparison."""
            events.append(("original fixed", arguments, keywords))
            return report
        paired = types.SimpleNamespace(check_fixed_inputs=fixed)
        def correctness(*arguments, **keywords):
            """Use the wrapped fixed-input function before the unchanged original whole-iteration checks."""
            self.assertIs(paired.check_fixed_inputs(*arguments, **keywords), report)
            events.append("original whole iteration")
            return report
        paired.correctness_checks = correctness
        snapshots = []
        with diagnosis.observe_checks(paired, snapshots.append):
            returned = paired.correctness_checks("reference", "candidate", state="supplied")
        self.assertIs(returned, report)
        self.assertIs(paired.check_fixed_inputs, fixed)
        self.assertIs(paired.correctness_checks, correctness)
        self.assertEqual(events, [("original fixed", ("reference", "candidate"), {"state": "supplied"}), "original whole iteration"])
        self.assertEqual(snapshots, ["correctness_checks_entered", "check_fixed_inputs_entered",
            "check_fixed_inputs_completed", "correctness_checks_completed"])

    def test_failure_observation_preserves_original_exception_and_restores_bindings(self):
        """OOM or numerical mismatch propagates unchanged even if recording failure memory also fails."""
        error = ValueError("original strict check failed")
        fixed = Mock(side_effect=error)
        paired = types.SimpleNamespace(check_fixed_inputs=fixed, correctness_checks=Mock())
        def observe(phase):
            """Simulate an independent metadata-write failure at the same exception boundary."""
            if phase.endswith("failed"):
                raise OSError("snapshot write failed")
        with self.assertRaises(ValueError) as caught:
            with diagnosis.observe_checks(paired, observe):
                paired.check_fixed_inputs("same state")
        self.assertIs(caught.exception, error)
        self.assertIs(paired.check_fixed_inputs, fixed)
        self.assertIn("snapshot write failed", str(error.__cause__))

    def test_unavailable_async_counters_are_not_fabricated(self):
        """Actual NVML bytes remain useful when this allocator returns no BFC-style statistics."""
        device = types.SimpleNamespace(memory_stats=lambda: None)
        observation = types.SimpleNamespace(sample=lambda: {"free_bytes": 123, "gpu_uuid": "GPU-diagnostic"})
        record = diagnosis.memory_snapshot(device, observation)
        self.assertIsNone(record["allocator"])
        self.assertEqual(record["device_memory"]["free_bytes"], 123)
        self.assertNotIn("peak_bytes_in_use", record)
        device.memory_stats = Mock(side_effect=RuntimeError("not available"))
        record = diagnosis.memory_snapshot(device, observation)
        self.assertIsNone(record["allocator"])
        self.assertIn("not available", record["allocator_statistics_unavailable"])

    def test_current_controller_shadow_is_rejected_in_the_fresh_compute_path(self):
        """Current metadata-controller imports must not replace the pinned original source."""
        with patch.dict(sys.modules, {"controller": types.SimpleNamespace(__file__="/mutable/controller.py")}):
            with self.assertRaisesRegex(ValueError, "fresh process"):
                diagnosis.load_original_modules({"source_root": str(self.source)})

    def test_fresh_standard_library_process_loads_original_controller_ideas_and_paired(self):
        """The actual loader selects every original module before importing any array library."""
        original = self.source / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
        original.mkdir(parents=True)
        for name in ("controller", "ideas", "paired_benchmark"):
            (original / (name + ".py")).write_text("identity='original " + name + "'\n")
        code = "import sys,json;sys.path.insert(0,sys.argv[1]);import diagnose_pregather_allocator as d;p=d.load_original_modules(json.loads(sys.argv[2]));assert 'jax' not in sys.modules;print(sys.modules['controller'].identity,p.identity)"
        result = subprocess.run([sys.executable, "-I", "-c", code, str(SCRIPTS), json.dumps({"source_root": str(self.source)})],
            capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "original controller original paired_benchmark")

    def test_mutable_entrypoint_cannot_execute_prepared_diagnostic(self):
        """The recorded staged script must run in a new process, rather than mutable integration code."""
        directory = self.prepare()
        with patch.object(diagnosis, "compute") as compute:
            with self.assertRaisesRegex(ValueError, "immutable staged diagnostic"):
                diagnosis.run_prepared(directory)
            compute.assert_not_called()

    def test_original_benchmark_protocol_and_raw_result_remain_unmodified(self):
        """A diagnostic success has zero credit and stores the original result separately from memory records."""
        directory = self.prepare()
        plan = json.loads((directory / "measurement_plan.json").read_text())
        runtime = types.ModuleType("jax")
        runtime.config = types.SimpleNamespace(update=Mock())
        runtime.devices = lambda: [types.SimpleNamespace(platform="gpu", device_kind="NVIDIA RTX A4500", memory_stats=lambda: None, client=None)]
        runtime.clear_caches = Mock()
        parser = types.ModuleType("jaxlib.xla_client")
        parser.__file__ = str(self.parser)
        parser.generate_pjrt_gpu_plugin_options = lambda: {"allocator": "cuda_async", "memory_fraction": .85, "preallocate": False}
        package = types.ModuleType("jaxlib")
        package.xla_client = parser
        config = types.ModuleType("exploration_platform.atari_pixels.config")
        config.PixelPPOConfig = lambda **values: values
        original_result = {"original paired result": True}
        fixed, complete = Mock(return_value={"original": True}), Mock(return_value={"original": True})
        paired = types.SimpleNamespace(check_fixed_inputs=fixed, correctness_checks=complete)
        def run(*arguments, **keywords):
            """Perform both original check callbacks before the original timing boundary."""
            paired.check_fixed_inputs("reference", "candidate", "host")
            paired.correctness_checks("reference", "candidate", "host")
            return original_result
        paired.run_paired_benchmark = Mock(side_effect=run)
        modules = {"jax": runtime, "jaxlib": package, "jaxlib.xla_client": parser,
            "jaxtari": types.ModuleType("jaxtari"), "exploration_platform.atari_pixels.config": config}
        with patch.dict(sys.modules, modules), patch.object(diagnosis, "load_original_modules", return_value=paired), \
                patch.object(diagnosis.measurement, "verify_imported_sources"), \
                patch.object(diagnosis.measurement, "MemoryObservation", return_value=types.SimpleNamespace(sample=lambda: {"free_bytes": 1})), \
                patch.object(diagnosis.training, "actual_gpu_library_receipt", return_value={}), \
                patch.dict(diagnosis.os.environ, diagnosis.ALLOCATOR):
            diagnosis.compute(directory, plan, {"gpu_uuid": "GPU-diagnostic"}, {})
        call = paired.run_paired_benchmark.call_args
        self.assertEqual((call.kwargs["rounds"], call.kwargs["iterations"], call.kwargs["seed_start"], call.kwargs["latency_calls"]), (7, 1, 1, 3))
        self.assertEqual(call.args[1], {"pre_gather_minibatches": True})
        self.assertEqual(json.loads((directory / "original_paired_result.json").read_text()), original_result)
        saved = json.loads((directory / "result.json").read_text())
        self.assertFalse(saved["eligible_for_formal_campaign_admission"])
        self.assertEqual(saved["hypothesis_count_contribution"], 0)
        self.assertIs(paired.check_fixed_inputs, fixed)
        self.assertIs(paired.correctness_checks, complete)
        runtime.clear_caches.assert_not_called()


if __name__ == "__main__":
    unittest.main()
