"""Check trace-export failures and cache isolation using standard-library fixtures only."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
SPECIFICATION = importlib.util.spec_from_file_location("pixel_trace_helper", SCRIPTS / "diagnose_gpu_execution.py")
HELPER = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(HELPER)
SPECIFICATION = importlib.util.spec_from_file_location("pixel_trace_launcher", SCRIPTS / "launch_diagnostic.py")
LAUNCHER = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(LAUNCHER)


class TraceSummaryTests(unittest.TestCase):
    """Require usable measurements even when the profiling tool reports success."""

    def setUp(self):
        """Create a separate raw capture fixture without running or importing numerical code."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.unit = Path(self.temporary.name)
        self.trace = self.unit / "trace"
        self.trace.mkdir()
        self.raw = self.trace / "capture.nsys-rep"
        self.raw.write_bytes(b"unchanged raw capture")

    def fake_stats(self, command, *, cwd, capture_output, text, timeout):
        """Create representative local SQLite and CSV files in the provided short export directory."""
        self.assertTrue(str(cwd).startswith("/tmp/rnd-nsys-"))
        self.assertEqual(command[-1], "capture.nsys-rep")
        self.assertIn("--output=summary", command)
        self.assertEqual((cwd / "capture.nsys-rep").read_bytes(), self.raw.read_bytes())
        (cwd / "capture.sqlite").write_bytes(b"derived SQLite fixture")
        (cwd / "summary_cuda_gpu_kern_sum.csv").write_text("Name,Instances,Total Time (ns)\nconvolution,2,1000\n")
        (cwd / "summary_cuda_api_sum.csv").write_text("Name,Num Calls,Total Time (ns)\ncudaLaunchKernel,2,400\n")
        return subprocess.CompletedProcess(command, 0, "export completed\n", "")

    def test_success_requires_data_and_preserves_derived_records(self):
        """Copy valid local outputs back with hashes and leave the original capture unchanged."""
        with patch.object(HELPER.subprocess, "run", side_effect=self.fake_stats):
            result = HELPER.summarize_trace(self.unit, "/mock/nsys")
        self.assertTrue(result["completed"])
        self.assertEqual(result["summaries"]["cuda_gpu_kern_sum"]["rows"], 1)
        self.assertEqual((self.trace / "capture.sqlite").read_bytes(), b"derived SQLite fixture")
        self.assertEqual((self.trace / "stats.stdout.log").read_text(), "export completed\n")
        self.assertEqual(self.raw.read_bytes(), b"unchanged raw capture")
        self.assertFalse(Path(result["working_directory"]).exists())

    def test_zero_exit_with_empty_csv_is_a_failure(self):
        """Reproduce Nsight's zero exit status with empty CSV outputs after a failed SQLite export."""
        def empty_stats(command, **kwargs):
            result = self.fake_stats(command, **kwargs)
            for path in kwargs["cwd"].glob("*.csv"):
                path.write_bytes(b"")
            return subprocess.CompletedProcess(command, 0, result.stdout, "unable to open database file\n")
        with patch.object(HELPER.subprocess, "run", side_effect=empty_stats):
            with self.assertRaisesRegex(RuntimeError, "validated kernel and API"):
                HELPER.summarize_trace(self.unit, "/mock/nsys")
        analysis = json.loads((self.trace / "analysis.json").read_text())
        self.assertFalse(analysis["completed"])
        self.assertEqual(analysis["exit_code"], 0)
        self.assertEqual((self.trace / "summary_cuda_gpu_kern_sum.csv").stat().st_size, 0)
        self.assertIn("unable to open database", (self.trace / "stats.stderr.log").read_text())
        self.assertEqual(self.raw.read_bytes(), b"unchanged raw capture")

    def test_header_only_output_is_a_failure(self):
        """A valid header without any measured kernel still fails the trace check."""
        def headers_only(command, **kwargs):
            result = self.fake_stats(command, **kwargs)
            (kwargs["cwd"] / "summary_cuda_gpu_kern_sum.csv").write_text("Name,Instances,Total Time (ns)\n")
            return result
        with patch.object(HELPER.subprocess, "run", side_effect=headers_only):
            with self.assertRaises(RuntimeError):
                HELPER.summarize_trace(self.unit, "/mock/nsys")
        self.assertFalse(json.loads((self.trace / "analysis.json").read_text())["completed"])

    def test_nonzero_exit_cannot_pass_with_valid_rows(self):
        """Usable-looking files do not override a failed export process."""
        def failed_stats(command, **kwargs):
            result = self.fake_stats(command, **kwargs)
            return subprocess.CompletedProcess(command, 2, result.stdout, "failed export\n")
        with patch.object(HELPER.subprocess, "run", side_effect=failed_stats):
            with self.assertRaises(RuntimeError):
                HELPER.summarize_trace(self.unit, "/mock/nsys")
        self.assertEqual(json.loads((self.trace / "analysis.json").read_text())["exit_code"], 2)

    def test_existing_analysis_is_never_replaced(self):
        """An earlier record remains unchanged and requires a separate recovery directory."""
        analysis = self.trace / "analysis.json"
        analysis.write_text('{"completed": true, "old": true}\n')
        original = analysis.read_bytes()
        with patch.object(HELPER.subprocess, "run") as process:
            with self.assertRaisesRegex(ValueError, "immutable"):
                HELPER.summarize_trace(self.unit, "/mock/nsys")
            process.assert_not_called()
        self.assertEqual(analysis.read_bytes(), original)

    def test_timeout_retains_failure_and_process_logs(self):
        """A timed-out export cannot be reported as complete and its captured logs remain available."""
        failure = subprocess.TimeoutExpired(["/mock/nsys"], 180, output=b"partial output\n", stderr=b"partial error\n")
        with patch.object(HELPER.subprocess, "run", side_effect=failure):
            with self.assertRaises(RuntimeError):
                HELPER.summarize_trace(self.unit, "/mock/nsys")
        self.assertFalse(json.loads((self.trace / "analysis.json").read_text())["completed"])
        self.assertEqual((self.trace / "stats.stdout.log").read_text(), "partial output\n")


class CacheIsolationTests(unittest.TestCase):
    """Keep every new diagnostic independent of persistent or imported autotune results."""

    def test_disabled_control_changes_one_flag_and_disables_file_caches(self):
        """Preserve precision and other flags while recording an explicit disabled-cache policy."""
        flags = " --other-option=7  --xla_gpu_exclude_nondeterministic_ops=true"
        environment = {"XLA_FLAGS": flags, "JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest"}
        with patch.dict(os.environ, environment, clear=True):
            actual = LAUNCHER.configure_runtime(False)
            self.assertEqual(os.environ["XLA_FLAGS"], flags.replace("ops=true", "ops=false"))
            self.assertEqual(os.environ["JAX_ENABLE_COMPILATION_CACHE"], "false")
            self.assertEqual(os.environ["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"], "")
        self.assertEqual(actual["compilation_cache_policy"], "disabled")
        self.assertIsNone(actual["compilation_cache_environment_before"]["JAX_COMPILATION_CACHE_DIR"])

    def test_inherited_cache_or_autotune_results_are_rejected(self):
        """Refuse imported OFF-mode artifacts rather than deleting an unknown runtime option."""
        baseline = {"XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true", "JAX_ENABLE_X64": "true",
                    "JAX_DEFAULT_MATMUL_PRECISION": "highest"}
        cases = [{"JAX_COMPILATION_CACHE_DIR": "/other/mode"},
                 {"JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "xla_gpu_per_fusion_autotune_cache_dir"},
                 {"XLA_FLAGS": baseline["XLA_FLAGS"] + " --xla_gpu_load_autotune_results_from=/other/mode"},
                 {"XLA_FLAGS": baseline["XLA_FLAGS"] + " --xla_gpu_per_fusion_autotune_cache_dir=/other/mode"}]
        for changes in cases:
            with self.subTest(changes=changes), patch.dict(os.environ, {**baseline, **changes}, clear=True):
                with self.assertRaises(ValueError):
                    LAUNCHER.configure_runtime(False)


class LayoutControlTests(unittest.TestCase):
    """Keep compiler layout experiments separate from the disabled-determinism control."""

    def environment(self):
        """Supply the same enabled flag and full precision used by the frozen worker environment."""
        return {"XLA_FLAGS": " --xla_gpu_exclude_nondeterministic_ops=true",
                "JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest"}

    def test_channels_last_adds_only_the_planned_layout_flag(self):
        """Keep determinism enabled while requesting the compiler's physical NHWC path."""
        before = self.environment()
        with patch.dict(os.environ, before, clear=True):
            actual = LAUNCHER.configure_runtime(True, "channels-last")
            self.assertEqual(os.environ["XLA_FLAGS"], before["XLA_FLAGS"] + " --xla_gpu_force_conv_nhwc=true")
            self.assertEqual(actual["convolution_backend_layout"], "channels-last")
            self.assertTrue(actual["expected_deterministic_operations"])
            self.assertEqual(os.environ["JAX_ENABLE_COMPILATION_CACHE"], "false")

    def test_layout_control_rejects_disabled_or_inherited_settings(self):
        """Refuse a combined flag experiment or an unrecorded forced physical layout."""
        with patch.dict(os.environ, self.environment(), clear=True):
            with self.assertRaises(ValueError):
                LAUNCHER.configure_runtime(False, "channels-last")
        environment = self.environment()
        environment["XLA_FLAGS"] += " --xla_gpu_force_conv_nchw=true"
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(ValueError, "inherit"):
                LAUNCHER.configure_runtime(True, "channels-last")

    def test_actual_layout_must_match_the_plan(self):
        """The numerical helper rejects missing, opposite or unrequested physical-layout flags."""
        environment = {**self.environment(), "CUDA_VISIBLE_DEVICES": HELPER.GPU_UUID,
            "JAX_ENABLE_COMPILATION_CACHE": "false", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": ""}
        environment["XLA_FLAGS"] += " --xla_gpu_force_conv_nhwc=true"
        query = HELPER.GPU_UUID + ", NVIDIA H100 NVL, 95830 MiB, 590.48.01\n"
        with patch.dict(os.environ, environment, clear=True), \
                patch.object(HELPER.subprocess, "check_output", return_value=query):
            actual = HELPER.hardware_record(convolution_backend_layout="channels-last")
            self.assertEqual(actual["convolution_backend_layout"], "channels-last")
            for expected in ("default", "channels-first"):
                with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, "layout differs"):
                    HELPER.hardware_record(convolution_backend_layout=expected)


if __name__ == "__main__":
    unittest.main()
