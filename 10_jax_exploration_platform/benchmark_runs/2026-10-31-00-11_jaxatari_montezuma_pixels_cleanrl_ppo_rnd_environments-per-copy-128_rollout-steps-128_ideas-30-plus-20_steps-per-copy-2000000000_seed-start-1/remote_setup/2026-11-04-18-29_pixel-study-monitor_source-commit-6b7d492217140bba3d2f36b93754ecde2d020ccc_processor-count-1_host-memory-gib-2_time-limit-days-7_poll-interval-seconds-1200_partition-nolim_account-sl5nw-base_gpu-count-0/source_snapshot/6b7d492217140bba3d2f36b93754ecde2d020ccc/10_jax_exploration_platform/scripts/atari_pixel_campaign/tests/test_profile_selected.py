"""Check selected-baseline profile metadata and trace publication without an array runtime."""

import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
SPECIFICATION = importlib.util.spec_from_file_location("selected_profile_producer", SCRIPTS / "profile_selected.py")
PRODUCER = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(PRODUCER)


class SelectedProfileTests(unittest.TestCase):
    """Keep source preparation separate from actual compute-only profile evidence."""

    def setUp(self):
        """Create full literal configuration and initial-cohort metadata fixtures."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.source = SCRIPTS.parents[2]
        defaults = PRODUCER.diagnostic.default_configuration(self.source)
        self.configuration = {name: defaults[name] for name in ("learner", "environment")}
        self.configuration["execution"] = {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
            "report_readout": "none", "report_every_iterations": 10}
        self.runtime = {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                        "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}
        self.tool = {"path": "/target/nsys", "version": "Nsight fixture version", "sha256": "a" * 64}
        self.registry = {"reference_commit": "c" * 40, "initial_matched_copy_counts": [1, 2], "initial_ideas": [
            {"name": "initial mechanism " + str(index), "results": {context: [
                {"copies": count, "implementation_distinct": True, "measurement_valid": True,
                 "correctness_passed": True, "result_path": "/evidence/result.json"} for count in (1, 2)]
                for context in PRODUCER.validation.CONTEXTS}} for index in range(30)]}
        self.confirmation = {"source_commit": "c" * 40, "hardware_class": "h100", "runtime_environment": self.runtime,
            "measurement_mode": "confirmation", "decision": "accepted", "candidate": self.configuration}
        self.documents = {"configuration": self.configuration, "runtime": self.runtime, "tool": self.tool,
            "registry": self.registry, "confirmation": self.confirmation,
            "assets": {"root": "/assets", "manifest": "/asset_manifest.json", "sha256": "d" * 64},
            "baseline": {"correctness_record": str(self.run / "correctness.json"),
                         "confirmation_record": str(self.run / "confirmation.json")}}
        self.files = {}
        for name, value in self.documents.items():
            self.files[name] = self.run / (name + ".json")
            self.files[name].write_text(json.dumps(value))
        self.helpers = {"source_root": str(self.source), "source_commit": "b" * 40,
            "files": {"10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name:
                      hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest() for name in PRODUCER.HELPERS}}

    def arguments(self, *extra):
        """Use the real preparation CLI with explicit metadata inputs."""
        return PRODUCER.parse_arguments(["--prepare-only", "--source-root", str(self.source),
            "--source-commit", "c" * 40, "--configuration", str(self.files["configuration"]),
            "--runtime-environment", str(self.files["runtime"]), "--asset-record", str(self.files["assets"]),
            "--study-registry", str(self.files["registry"]), "--accepted-baseline-specification", str(self.files["baseline"]),
            "--profile-tool-record", str(self.files["tool"]), "--run-dir", str(self.run),
            "--unit-id", "selected_baseline_profile_seed-1", "--gpu-uuid", "GPU-fixture",
            "--hardware-class", "h100", *extra])

    def prepare(self, arguments=None):
        """Mock committed bytes only; GPU access and scientific evidence validation must remain absent."""
        def source_receipt(root, commit, paths):
            """Preserve the complete frozen helper dependency list."""
            return {"source_root": str(root), "source_commit": commit, "files": {name: "c" * 64 for name in paths}}
        with patch.object(PRODUCER.diagnostic, "core_identity", return_value={"source_commit": "c" * 40}), \
                patch.object(PRODUCER.training, "source_receipt", side_effect=source_receipt), \
                patch.object(PRODUCER, "helper_receipt", return_value=self.helpers), \
                patch.object(PRODUCER.measurement, "preflight") as gpu, \
                patch.object(PRODUCER.validation, "validate_cohort_case") as original_cases, \
                patch.object(PRODUCER.validation, "validate_combination") as numerical, \
                patch.object(PRODUCER, "tool_record") as actual_tool:
            directory = PRODUCER.prepare(arguments or self.arguments())
            for action in (gpu, original_cases, numerical, actual_tool):
                action.assert_not_called()
        return directory

    def test_preparation_records_metadata_and_complete_helper_closure(self):
        """The prepared unit contains standard records and no claimed measured evidence."""
        directory = self.prepare()
        plan = json.loads((directory / "measurement_plan.json").read_text())
        self.assertEqual(plan["profile_updates"], 1)
        self.assertEqual(plan["profile_tool"], self.tool)
        self.assertEqual(plan["hypothesis_count_contribution"], 0)
        self.assertFalse(plan["jobs_submitted_by_producer"])
        self.assertEqual(set(plan["paired_source"]["files"]), {
            "10_jax_exploration_platform/scripts/atari_pixel_campaign/paired_benchmark.py",
            "10_jax_exploration_platform/scripts/atari_pixel_campaign/ideas.py"})
        for name in PRODUCER.HELPERS:
            self.assertEqual((directory / "code" / name).read_bytes(), (SCRIPTS / name).read_bytes())
        for name in ("manifest.yaml", "config_resolved.yaml", "experiment_background.md", "command.txt", "measurement_prepared.json"):
            self.assertTrue((directory / name).is_file())
        self.assertFalse((directory / "result.json").exists())
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_smallest_capture_covers_execution_group_or_report_window(self):
        """Grouping and scheduled reporting retain every complete original update."""
        selected = copy.deepcopy(self.configuration)
        selected["execution"]["iterations_per_call"] = 5
        self.assertEqual(PRODUCER.profile_updates(selected), 5)
        self.assertEqual(PRODUCER.profile_updates(selected, 6), 10)
        selected["execution"].update(iterations_per_call=1, report_readout="scheduled")
        self.assertEqual(PRODUCER.profile_updates(selected), 10)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                PRODUCER.profile_updates(selected, invalid)

    def test_initial_cohort_requires_every_hardware_and_copy_count(self):
        """One missing hardware result blocks profiling even when other counts completed."""
        PRODUCER.initial_summaries_complete(self.registry)
        missing = copy.deepcopy(self.registry)
        missing["initial_ideas"][0]["results"]["h100"].pop()
        with self.assertRaisesRegex(ValueError, "matched"):
            PRODUCER.initial_summaries_complete(missing)
        duplicate = copy.deepcopy(self.registry)
        duplicate["initial_ideas"][0]["name"] = duplicate["initial_ideas"][1]["name"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            PRODUCER.initial_summaries_complete(duplicate)

    def test_unaccepted_or_different_baseline_is_rejected_before_staging(self):
        """A valid profile cannot relabel another configuration or a rejected comparison."""
        for change in ("different", "rejected"):
            confirmation = copy.deepcopy(self.confirmation)
            if change == "different":
                confirmation["candidate"]["learner"]["rollout_unroll"] = 2
            else:
                confirmation["decision"] = "rejected"
            self.files["confirmation"].write_text(json.dumps(confirmation))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "accepted combined baseline"):
                self.prepare()
        self.assertFalse((self.run / "data").exists())

    def test_tool_receipt_and_immutable_unit_cannot_be_changed(self):
        """The actual target tool receipt is bound before an immutable unit is published."""
        directory = self.prepare()
        before = (directory / "measurement_plan.json").read_bytes()
        changed = {**self.tool, "sha256": "e" * 64}
        self.files["tool"].write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.prepare()
        self.assertEqual((directory / "measurement_plan.json").read_bytes(), before)
        changed["path"] = "relative/nsys"
        self.files["tool"].write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "target host"):
            self.prepare()

    def test_staged_initial_registry_is_rehashed_before_compute_validation(self):
        """A changed registry copy cannot claim completion of the original thirty mechanisms."""
        directory = self.prepare()
        plan = json.loads((directory / "measurement_plan.json").read_text())
        (directory / "inputs/study_registry.json").write_text("{}")
        with patch.object(PRODUCER.validation, "validate_cohort_case") as cases:
            with self.assertRaisesRegex(ValueError, "prepared bytes"):
                PRODUCER.validate_measured_baseline(directory, plan)
            cases.assert_not_called()

    def test_accepted_confirmation_cannot_change_after_preparation(self):
        """The accepted timing record is bound to its prepared bytes before raw cohort checks."""
        directory = self.prepare()
        plan = json.loads((directory / "measurement_plan.json").read_text())
        self.files["confirmation"].write_text("{}")
        with patch.object(PRODUCER.validation, "validate_cohort_case") as cases:
            with self.assertRaisesRegex(ValueError, "confirmation changed"):
                PRODUCER.validate_measured_baseline(directory, plan)
            cases.assert_not_called()

    def test_capture_child_passes_actual_runtime_and_cache_settings_to_core(self):
        """The capture child's runtime evidence comes from enforced process settings before JAX."""
        directory = self.prepare()
        identity = {"hostname": PRODUCER.socket.gethostname().split(".")[0], "pid": os.getpid(),
                    "process_start_time": PRODUCER.process_birth(os.getpid())}
        owner = {"pid": 1234, "process_start_time": "profiler birth", "parent_pid": os.getpid(),
                 "parent_start_time": identity["process_start_time"]}
        PRODUCER.diagnostic.atomic_json(directory / "profiler_process.json", owner)
        PRODUCER.diagnostic.atomic_json(directory / "capture_release.json", {**owner, "capture_identity": identity})
        PRODUCER.diagnostic.atomic_json(directory / "installed_backend.json", {"gpu_libraries": {}})
        with patch.object(PRODUCER, "verify_prepared"), \
                patch.object(PRODUCER, "process_birth", side_effect=lambda pid: "profiler birth" if pid == 1234 else identity["process_start_time"]), \
                patch.object(PRODUCER, "process_parent", return_value=os.getpid()), \
                patch.object(PRODUCER.os, "getpgrp", return_value=1234), \
                patch.object(PRODUCER.measurement, "preflight", return_value=dict(identity)), \
                patch.object(PRODUCER, "_capture_core") as core, patch.dict(os.environ, {}, clear=True):
            PRODUCER.capture(directory)
        actual = core.call_args.args[2]["runtime_environment"]
        self.assertEqual({name: actual[name] for name in self.runtime}, self.runtime)
        self.assertEqual(actual["JAX_ENABLE_COMPILATION_CACHE"], "false")
        self.assertEqual(actual["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"], "")
        self.assertEqual(actual.get("JAX_COMPILATION_CACHE_DIR", ""), "")
        self.assertNotIn("jax", sys.modules)

    def test_prepared_plan_and_staged_helper_changes_fail_closed(self):
        """Changing the manifest or a transitive staged dependency prevents execution."""
        directory = self.prepare()
        plan = json.loads((directory / "measurement_plan.json").read_text())
        with patch.object(PRODUCER, "__file__", str(directory / "code/profile_selected.py")), \
                patch.object(PRODUCER.sys, "executable", plan["python"]):
            (directory / "code/measure_selected.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "staged profile helper"):
                PRODUCER.verify_prepared(directory, plan)
        (directory / "measurement_plan.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "immutable profile plan"):
            PRODUCER.verify_prepared(directory, plan)

    def test_partial_capture_requires_a_new_unit_before_card_admission(self):
        """An incomplete prior capture cannot be reused or mixed with a later execution."""
        directory = self.prepare()
        (directory / "capture_result.json").write_text("{}")
        with patch.object(PRODUCER.measurement, "card_admission") as card:
            with self.assertRaisesRegex(ValueError, "previous profile attempt"):
                PRODUCER.run_prepared(directory)
            card.assert_not_called()

    def test_private_execution_mode_cannot_override_configuration(self):
        """A staged command cannot select a new runtime, GPU, or update window."""
        with self.assertRaises(SystemExit):
            PRODUCER.parse_arguments(["--capture-unit", str(self.run), "--minimum-updates", "10"])
        command = PRODUCER.child_command(self.run, {"profile_tool": self.tool, "python": "/python"})
        self.assertIn("--capture-range=cudaProfilerApi", command)
        self.assertIn("--capture-range-end=stop", command)
        self.assertIn("--force-overwrite=false", command)

    def test_termination_uses_exception_cleanup_and_restores_handlers(self):
        """SIGTERM reaches the owned-child cleanup before the standing card lock is released."""
        before = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(InterruptedError):
            with PRODUCER.interruption_cleanup():
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)

    def test_profiler_receipt_failure_stops_child_before_card_lock_release(self):
        """A failed PID receipt cannot leave the newly started profiler outside owned-child cleanup."""
        directory = self.prepare()
        events = []
        child = types.SimpleNamespace(pid=1234, poll=lambda: None, wait=lambda **options: 0)
        original_json = PRODUCER.diagnostic.atomic_json
        @contextmanager
        def card_lock(*args):
            """Expose exact lock ownership ordering without accessing a GPU or standing lock."""
            events.append("card lock acquired")
            try:
                yield
            finally:
                events.append("card lock released")
        def write_record(path, value):
            """Fail only the required child receipt after the synthetic Popen has succeeded."""
            if Path(path).name == "profiler_process.json":
                raise OSError("synthetic child receipt write failure")
            original_json(path, value)
        with patch.object(PRODUCER, "verify_prepared"), \
                patch.object(PRODUCER.measurement, "card_admission", side_effect=card_lock), \
                patch.object(PRODUCER.measurement, "preflight", return_value={}), \
                patch.object(PRODUCER, "validate_measured_baseline"), \
                patch.object(PRODUCER.training, "verify_assets", return_value={}), \
                patch.object(PRODUCER.training, "jaxtari_python_receipt", return_value={}), \
                patch.object(PRODUCER.training, "installed_gpu_library_receipts", return_value={}), \
                patch.object(PRODUCER, "tool_record", return_value=self.tool), \
                patch.object(PRODUCER.subprocess, "Popen", return_value=child) as started, \
                patch.object(PRODUCER, "process_birth", return_value="birth"), \
                patch.object(PRODUCER.diagnostic, "atomic_json", side_effect=write_record), \
                patch.object(PRODUCER.os, "killpg", side_effect=lambda pid, number: events.append("owned child stopped")) as stop, \
                patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(OSError, "child receipt write failure"):
                PRODUCER.run_prepared(directory)
        started.assert_called_once()
        stop.assert_called_once_with(1234, signal.SIGTERM)
        self.assertEqual(events, ["card lock acquired", "owned child stopped", "card lock released"])
        self.assertFalse((directory / "capture_release.json").exists())
        self.assertFalse((directory / "result.json").exists())
        self.assertTrue((directory / "failure.json").is_file())

    def capture_fixture(self, empty_summary=False, foreign_group=False):
        """Emulate only process metadata and precomputed CSVs, never an actual profiling process."""
        directory = self.run / ("empty_capture" if empty_summary else "positive_capture")
        directory.mkdir()
        plan = {"profile_tool": self.tool, "python": "/python", "maximum_active_seconds": 10}
        identity = {"hostname": "fixture", "pid": 1235, "process_start_time": "birth"}
        child = types.SimpleNamespace(pid=1234, poll=lambda: 0, wait=lambda **options: 0)
        def create_child(*args, **options):
            """Create the files that an already recorded, mocked capture would publish."""
            PRODUCER.diagnostic.atomic_json(directory / "capture_identity.json", identity)
            PRODUCER.diagnostic.atomic_json(directory / "capture_result.json", {"profile_updates": 1})
            (directory / "trace/capture.nsys-rep").write_bytes(b"raw fixture capture")
            return child
        def summaries(*args):
            """Return actual CSV receipts, with one optionally invalid header-only summary."""
            result = {}
            for name, count in (("cuda_gpu_kern_sum", "Instances"), ("cuda_api_sum", "Num Calls")):
                path = directory / "trace" / (name + ".csv")
                path.write_text("Name," + count + ",Total Time (ns)\n" + ("" if empty_summary else "measured kernel,1,10\n"))
                # The actual publication path must reread and validate these rows itself.
                result[name] = {"path": str(path), "rows": 1, "columns": ["Name", count, "Total Time (ns)"],
                                "sha256": PRODUCER.training.file_sha256(path)}
            return {"summaries": result}
        with patch.object(PRODUCER.subprocess, "Popen", side_effect=create_child), \
                patch.object(PRODUCER, "process_birth", return_value="birth"), \
                patch.object(PRODUCER.socket, "gethostname", return_value="fixture"), \
                patch.object(PRODUCER.os, "getpgid", return_value=9999 if foreign_group else 1234), \
                patch.object(PRODUCER, "tool_record", return_value=self.tool), \
                patch.object(PRODUCER.diagnostic, "summarize_trace", side_effect=summaries):
            PRODUCER.capture_and_summarize(directory, plan)
        return directory

    def test_positive_trace_rows_are_required_before_publication(self):
        """Header-only Nsight output cannot produce a completed selected-baseline result."""
        directory = self.capture_fixture()
        result = json.loads((directory / "result.json").read_text())
        self.assertTrue(result["completed"])
        self.assertEqual(result["profile_tool"], self.tool)
        self.assertTrue((directory / "profiler_process.json").is_file())
        self.assertTrue((directory / "capture_release.json").is_file())
        with self.assertRaisesRegex(ValueError, "no data rows"):
            self.capture_fixture(empty_summary=True)
        self.assertFalse((self.run / "empty_capture/result.json").exists())

    def test_foreign_capture_process_is_not_released(self):
        """A PID with the right birth counter must still belong to this newly created profiler."""
        with self.assertRaisesRegex(ValueError, "this live Nsight child"):
            self.capture_fixture(foreign_group=True)
        self.assertFalse((self.run / "positive_capture/capture_release.json").exists())


if __name__ == "__main__":
    unittest.main()
