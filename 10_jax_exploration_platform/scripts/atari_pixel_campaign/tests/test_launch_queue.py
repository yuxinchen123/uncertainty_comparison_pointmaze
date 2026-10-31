"""Mocked metadata checks: no GPU initialization, SSH connection, or Slurm submission."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SPECIFICATION = importlib.util.spec_from_file_location(
    "pixel_queue_launcher", Path(__file__).resolve().parents[1] / "launch_queue.py")
LAUNCHER = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(LAUNCHER)


class QueueLauncherTests(unittest.TestCase):
    """Keep allocation identity and recording behavior testable without touching hardware."""

    def gpu_queries(self, rows="0, GPU-allocated-five, NVIDIA RTX A4500\n", *, minor="5",
                    gpu_uuid="GPU-allocated-five", extra_device=False):
        """Represent the driver's supported CSV fields and its separate physical-device XML."""
        device = (f"<gpu><uuid>{gpu_uuid}</uuid><minor_number>{minor}</minor_number>"
                  "<product_name>NVIDIA RTX A4500</product_name></gpu>")
        xml = "<nvidia_smi_log>" + device * (2 if extra_device else 1) + "</nvidia_smi_log>"

        def query(command, **options):
            """Return only the two supported unfiltered metadata commands."""
            outputs = {("nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"): rows,
                       ("nvidia-smi", "-q", "-x"): xml}
            return subprocess.CompletedProcess(command, 0, outputs[tuple(command)], "")

        return query

    def test_remapped_cuda_zero_uses_slurm_physical_gpu(self):
        """NVML index zero and CUDA zero may identify physical device minor five."""
        with patch.object(LAUNCHER.subprocess, "run", side_effect=self.gpu_queries()) as query:
            actual = LAUNCHER.allocated_gpu_uuid({"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "5"})
        self.assertEqual(actual, "GPU-allocated-five")
        self.assertEqual(query.call_args_list[0].args[0], ["nvidia-smi",
            "--query-gpu=index,uuid,name", "--format=csv,noheader"])
        self.assertEqual(query.call_args_list[1].args[0], ["nvidia-smi", "-q", "-x"])

    def test_ambiguous_or_conflicting_visibility_is_rejected(self):
        """A worker cannot choose a card when the allocation or visibility is ambiguous."""
        with patch.object(LAUNCHER.subprocess, "run") as query:
            with self.assertRaisesRegex(ValueError, "one CUDA-visible"):
                LAUNCHER.allocated_gpu_uuid({"CUDA_VISIBLE_DEVICES": "0,1", "SLURM_JOB_GPUS": "5"})
            query.assert_not_called()
        with patch.object(LAUNCHER.subprocess, "run", side_effect=self.gpu_queries()):
            with self.assertRaisesRegex(ValueError, "conflicts"):
                LAUNCHER.allocated_gpu_uuid({"CUDA_VISIBLE_DEVICES": "2", "SLURM_JOB_GPUS": "5"})

    def test_physical_minor_and_allocation_uuids_must_agree(self):
        """Reject a wrong physical card even when it is the only visible ordinal zero."""
        with patch.object(LAUNCHER.subprocess, "run", side_effect=self.gpu_queries()):
            for visible in ("0", "5", "GPU-allocated-five"):
                with self.subTest(visible=visible):
                    self.assertEqual(LAUNCHER.allocated_gpu_uuid({"CUDA_VISIBLE_DEVICES": visible,
                        "SLURM_JOB_GPUS": "5", "SLURM_STEP_GPUS": "GPU-allocated-five"}),
                        "GPU-allocated-five")
            for allocated in ("0", "4", "GPU-other-card"):
                with self.subTest(allocated=allocated), self.assertRaises(ValueError):
                    LAUNCHER.allocated_gpu_uuid({"CUDA_VISIBLE_DEVICES": "0",
                                                "SLURM_JOB_GPUS": allocated})
            with self.assertRaisesRegex(ValueError, "conflicts"):
                LAUNCHER.allocated_gpu_uuid({"CUDA_VISIBLE_DEVICES": "0",
                    "SLURM_JOB_GPUS": "5", "SLURM_STEP_GPUS": "0"})
            with self.assertRaisesRegex(ValueError, "CUDA visibility conflicts"):
                LAUNCHER.allocated_gpu_uuid({"CUDA_VISIBLE_DEVICES": "GPU-other-card",
                                            "SLURM_JOB_GPUS": "5"})

    def test_visible_query_must_identify_one_physical_a4500(self):
        """Reject multiple rows, missing minor numbers, malformed rows and another GPU type."""
        row = "0, GPU-allocated-five, NVIDIA RTX A4500\n"
        for output in ("", row + "1, GPU-other, NVIDIA RTX A4500\n",
                       "0, NVIDIA RTX A4500\n", "0, GPU-allocated-five, NVIDIA H100\n"):
            with self.subTest(output=output), patch.object(
                    LAUNCHER.subprocess, "run", side_effect=self.gpu_queries(output)):
                with self.assertRaises(ValueError):
                    LAUNCHER.allocated_gpu_uuid({"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "5"})
        for options in ({"minor": "N/A"}, {"minor": ""}, {"gpu_uuid": "GPU-other-card"},
                        {"extra_device": True}):
            with self.subTest(options=options), patch.object(
                    LAUNCHER.subprocess, "run", side_effect=self.gpu_queries(**options)):
                with self.assertRaises(ValueError):
                    LAUNCHER.allocated_gpu_uuid({"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "5"})

    def test_direct_supervisor_has_per_case_lock_and_durable_pid(self):
        """Record the remote process before releasing a worker, without an outer standing lock."""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            directory = run_dir / "slurm/queue_workers"
            directory.mkdir(parents=True)
            result = subprocess.CompletedProcess([], 0, json.dumps({
                "pid": 12345, "start_time": "9000", "hostname": "serval02"}) + "\n", "")
            with patch.object(LAUNCHER.subprocess, "run", return_value=result) as launch:
                record = LAUNCHER.launch_direct(Path("/frozen/source"), run_dir, "a" * 40,
                    Path("/shared/python"), directory, "queue_worker_h100")
            script = Path(record["script"]).read_text()
            self.assertIn("--standing-lock /tmp/sl5nw_serval02_card1.lock", script)
            self.assertIn(LAUNCHER.SERVAL02_CARD1, script)
            self.assertNotIn("flock", script)
            self.assertIn("--idle-exit", script)
            self.assertEqual(json.loads((run_dir / "processes.json").read_text())[0]["pid"], 12345)
            self.assertEqual((directory / "queue_worker_h100.pid").read_text(), "12345\n")
            self.assertTrue((directory / "queue_worker_h100.released").exists())
            self.assertEqual(launch.call_args.args[0][:2], ["ssh", "-n"])

    def test_slurm_shape_and_exact_identifier_recording(self):
        """Request one card and record the exact returned ID in both run ledgers."""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            directory = run_dir / "slurm/queue_workers"
            directory.mkdir(parents=True)
            result = subprocess.CompletedProcess([], 0, "6779000;uva\n", "")
            with patch.object(LAUNCHER, "current_reservation", return_value="sl5nw_reserved"), \
                    patch.object(LAUNCHER.subprocess, "run", return_value=result) as launch:
                record = LAUNCHER.launch_slurm(Path("/frozen/source"), run_dir, "a" * 40,
                    Path("/shared/python"), directory, "queue_worker_jaguar03", Path("/metadata/launcher.py"))
            arguments = launch.call_args.args[0]
            for value in ("--gres=gpu:nvidia_rtx_a4500:1", "--cpus-per-task=8", "--mem=64G",
                          "--time=12:00:00", "--reservation=sl5nw_reserved", "--qos=csresnolim"):
                self.assertIn(value, arguments)
            self.assertEqual((run_dir / "slurm/submitted_jobids_uva.txt").read_text(), "6779000\n")
            self.assertIn("6779000\tqueue-worker\t", (run_dir / "slurm/submissions.tsv").read_text())
            self.assertEqual(record["job_id"], "6779000")
            self.assertIn("--resolve-slurm-gpu", Path(record["script"]).read_text())


if __name__ == "__main__":
    unittest.main()
