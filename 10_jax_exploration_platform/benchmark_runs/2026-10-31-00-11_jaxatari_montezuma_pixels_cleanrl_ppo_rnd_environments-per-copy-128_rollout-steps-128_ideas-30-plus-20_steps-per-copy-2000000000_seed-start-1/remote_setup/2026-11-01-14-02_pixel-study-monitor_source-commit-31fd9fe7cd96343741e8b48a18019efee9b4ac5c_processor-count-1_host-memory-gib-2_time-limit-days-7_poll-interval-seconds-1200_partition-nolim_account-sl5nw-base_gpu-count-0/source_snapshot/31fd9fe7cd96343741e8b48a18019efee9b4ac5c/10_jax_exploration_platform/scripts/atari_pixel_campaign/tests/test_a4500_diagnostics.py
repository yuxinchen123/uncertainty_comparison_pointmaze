"""Exercise A4500 diagnostic metadata with every scheduler and numerical command mocked."""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

SPECIFICATION = importlib.util.spec_from_file_location(
    "a4500_diagnostic_launcher", Path(__file__).resolve().parents[1] / "launch_a4500_diagnostics.py")
LAUNCHER = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(LAUNCHER)


def write_once(path, data):
    """Create ordinary temporary test metadata, refusing a changed existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("changed test artifact")
    else:
        path.write_bytes(data)


def encoded_json(value):
    """Use the production artifact encoding for synthetic metadata records."""
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


class A4500DiagnosticTests(unittest.TestCase):
    """Check cold-process controls, capture evidence, and exact job ownership without compute."""

    def test_cache_policy_is_explicit_and_rejects_external_autotune_inputs(self):
        """Only the intended flag remains, and both processes receive disabled file caches."""
        expected = " --xla_gpu_exclude_nondeterministic_ops=true"
        with mock.patch.dict(os.environ, {"XLA_FLAGS": expected}, clear=True):
            record = LAUNCHER.disabled_cache_environment()
            self.assertEqual(os.environ["XLA_FLAGS"], expected)
            self.assertEqual(record["runtime"]["JAX_ENABLE_COMPILATION_CACHE"], "false")
            self.assertEqual(record["runtime"]["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"], "")
        invalid = ({"XLA_FLAGS": expected + " --xla_gpu_load_autotune_results_from=/other/mode"},
                   {"XLA_FLAGS": expected, "JAX_COMPILATION_CACHE_DIR": "/other/mode"},
                   {"XLA_FLAGS": expected, "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all"})
        for environment in invalid:
            with self.subTest(environment=environment), mock.patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(ValueError):
                    LAUNCHER.disabled_cache_environment()
                self.assertEqual(dict(os.environ), environment)

    def test_only_the_allocated_physical_card_can_bind_a_template(self):
        """The final plan keeps the explicit control and records the exact one-card allocation."""
        template = {"gpu_uuid": LAUNCHER.GPU_PLACEHOLDER, "hardware_context": LAUNCHER.GPU_CONTEXT,
                    "expected_deterministic_operations": False}
        plan = LAUNCHER.resolved_plan(template, "GPU-allocated-five", "6779999")
        self.assertEqual(plan["gpu_uuid"], "GPU-allocated-five")
        self.assertFalse(plan["expected_deterministic_operations"])
        self.assertEqual(template["gpu_uuid"], LAUNCHER.GPU_PLACEHOLDER)
        with self.assertRaises(ValueError):
            LAUNCHER.resolved_plan(template, "0", "6779999")
        with self.assertRaises(ValueError):
            LAUNCHER.resolved_plan({**template, "gpu_uuid": "GPU-other"}, "GPU-allocated-five", "6779999")

    def test_submit_records_exact_id_and_refuses_a_second_submission(self):
        """One mocked job requests the complete shape and enters both exact-ID ledgers immediately."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "slurm/diagnostic_jobs/synthetic"
            directory.mkdir(parents=True)
            queue = types.SimpleNamespace(current_reservation=lambda: "sl5nw_current")
            common = types.SimpleNamespace(write_once=write_once, encoded_json=encoded_json)
            batch = {"run_dir": str(root), "core": {"source_commit": "d" * 40}}
            completed = subprocess.CompletedProcess([], 0, "6779999;uva\n", "")
            with mock.patch.object(LAUNCHER, "load_helper", side_effect=[queue, common, queue, common]), \
                    mock.patch.object(LAUNCHER.subprocess, "run", return_value=completed) as submission:
                result = LAUNCHER.submit(directory, batch)
                with self.assertRaisesRegex(ValueError, "already has a submission attempt"):
                    LAUNCHER.submit(directory, batch)
            self.assertEqual(submission.call_count, 1)
            arguments = submission.call_args.args[0]
            for setting in ("--account=sl5nw_base", "--gres=gpu:nvidia_rtx_a4500:1",
                            "--cpus-per-task=8", "--mem=24G", "--time=01:00:00",
                            "--reservation=sl5nw_current", "--qos=csresnolim"):
                self.assertIn(setting, arguments)
            self.assertEqual(result["job_id"], "6779999")
            self.assertEqual((root / "slurm/submitted_jobids_uva.txt").read_text(), "6779999\n")
            self.assertIn("6779999\ta4500-diagnostics\t", (root / "slurm/submissions.tsv").read_text())

    def test_a_missing_kernel_capture_preserves_timing_and_runs_the_other_control(self):
        """Two fresh profile commands execute with different flags, and failed capture stays visible."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "slurm/diagnostic_jobs/synthetic"
            code = directory / "code"
            code.mkdir(parents=True)
            helpers = {}
            for name in LAUNCHER.HELPER_FILES:
                (code / name).write_bytes(b"synthetic metadata fixture\n")
                helpers[name] = {"sha256": hashlib.sha256((code / name).read_bytes()).hexdigest()}
            nsight = {"path": "/synthetic/nsys", "version": "synthetic", "sha256": "a" * 64}
            units = []
            for expected, name in ((True, "enabled"), (False, "disabled")):
                unit = root / "data" / name
                (unit / "trace").mkdir(parents=True)
                (unit / "trace/capture.nsys-rep").write_bytes(b"synthetic report")
                (unit / "timing_result.json").write_text('{"completed": true, "diagnostic_only": true}\n')
                template = {"gpu_uuid": LAUNCHER.GPU_PLACEHOLDER, "hardware_context": LAUNCHER.GPU_CONTEXT,
                            "expected_deterministic_operations": expected, "nsight": nsight,
                            "python": "/synthetic/python", "code_directory": str(code),
                            "maximum_active_seconds": 1440}
                (unit / "diagnostic_plan_template.json").write_bytes(encoded_json(template))
                units.append({"unit_id": name, "directory": str(unit), "expected_deterministic_operations": expected})
            batch = {"helpers": helpers, "nsight": nsight, "units": units, "run_dir": str(root)}
            (directory / "batch_plan.json").write_bytes(encoded_json(batch))
            profile_flags = []

            def configure(expected):
                """Model only the common helper's explicit one-token control change."""
                os.environ["XLA_FLAGS"] = "--xla_gpu_exclude_nondeterministic_ops=" + str(expected).lower()
                return {"expected_deterministic_operations": expected}

            def bounded(command, seconds):
                """Capture a new numerical subprocess request without executing it."""
                if "profile" in command:
                    profile_flags.append(os.environ["XLA_FLAGS"])
                return 0

            def atomic(path, value):
                """Write synthetic node-side metadata for inspection after the mocked job fails."""
                Path(path).write_bytes(encoded_json(value))

            def summarize(unit, nsys):
                """Model a zero-status empty export, then a valid export in the other control."""
                if unit.name == "enabled":
                    atomic(unit / "trace/analysis.json", {
                        "completed": False, "stage": "shared_summary_validation",
                        "stats_exit_code": 0, "error": "no CUDA events captured"})
                    raise ValueError("no CUDA events captured")
                atomic(unit / "trace/analysis.json", {"completed": True, "kernel_rows": 1, "api_rows": 1})

            common = types.SimpleNamespace(write_once=write_once, encoded_json=encoded_json,
                configure_runtime=configure, bounded_command=bounded)
            summary = mock.Mock(side_effect=summarize)
            helper = types.SimpleNamespace(atomic_json=atomic, record_progress=mock.Mock(), summarize_trace=summary)
            queue = types.SimpleNamespace(allocated_gpu_uuid=lambda: "GPU-allocated-five")
            environment = {"XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true",
                           "CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "5", "SLURM_JOB_ID": "6779999"}
            with mock.patch.object(LAUNCHER, "load_helper", side_effect=[common, helper, queue]), \
                    mock.patch.object(LAUNCHER, "nsight_identity", return_value=nsight), \
                    mock.patch.object(LAUNCHER.socket, "gethostname", return_value="jaguar03"), \
                    mock.patch.object(LAUNCHER.getpass, "getuser", return_value="sl5nw"), \
                    mock.patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(RuntimeError, "A4500 diagnostic failed"):
                    LAUNCHER.run_allocated(directory)
            self.assertEqual(profile_flags, ["--xla_gpu_exclude_nondeterministic_ops=true",
                                             "--xla_gpu_exclude_nondeterministic_ops=false"])
            self.assertTrue(json.loads((root / "data/enabled/timing_result.json").read_text())["completed"])
            self.assertFalse(json.loads((root / "data/enabled/trace/analysis.json").read_text())["completed"])
            self.assertEqual(json.loads((root / "data/enabled/trace/analysis.json").read_text())["stage"],
                             "shared_summary_validation")
            self.assertTrue(json.loads((root / "data/disabled/trace/analysis.json").read_text())["completed"])
            self.assertFalse(json.loads((directory / "result.json").read_text())["completed"])
            self.assertEqual(summary.call_args_list, [mock.call(root / "data/enabled", "/synthetic/nsys"),
                                                      mock.call(root / "data/disabled", "/synthetic/nsys")])


if __name__ == "__main__":
    unittest.main()
