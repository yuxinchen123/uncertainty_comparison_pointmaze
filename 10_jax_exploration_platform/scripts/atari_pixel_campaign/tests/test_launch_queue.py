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

    def test_remapped_cuda_zero_uses_slurm_physical_gpu(self):
        """CUDA ordinal zero must resolve to the allocated physical index, not card zero."""
        result = subprocess.CompletedProcess([], 0, "5, GPU-allocated-five, NVIDIA RTX A4500\n", "")
        with patch.object(LAUNCHER.subprocess, "run", return_value=result) as query:
            actual = LAUNCHER.allocated_gpu_uuid({"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "5"})
        self.assertEqual(actual, "GPU-allocated-five")
        self.assertEqual(query.call_args.args[0][:3], ["nvidia-smi", "-i", "5"])

    def test_ambiguous_or_conflicting_visibility_is_rejected(self):
        """A worker cannot choose a card when the allocation or visibility is ambiguous."""
        with patch.object(LAUNCHER.subprocess, "run") as query:
            with self.assertRaisesRegex(ValueError, "one CUDA-visible"):
                LAUNCHER.allocated_gpu_uuid({"CUDA_VISIBLE_DEVICES": "0,1", "SLURM_JOB_GPUS": "5"})
            query.assert_not_called()
        result = subprocess.CompletedProcess([], 0, "5, GPU-allocated-five, NVIDIA RTX A4500\n", "")
        with patch.object(LAUNCHER.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(ValueError, "conflicts"):
                LAUNCHER.allocated_gpu_uuid({"CUDA_VISIBLE_DEVICES": "2", "SLURM_JOB_GPUS": "5"})

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
            for value in ("--gres=gpu:nvidia_rtx_a4500:1", "--cpus-per-task=8", "--mem=24G",
                          "--time=12:00:00", "--reservation=sl5nw_reserved", "--qos=csresnolim"):
                self.assertIn(value, arguments)
            self.assertEqual((run_dir / "slurm/submitted_jobids_uva.txt").read_text(), "6779000\n")
            self.assertIn("6779000\tqueue-worker\t", (run_dir / "slurm/submissions.tsv").read_text())
            self.assertEqual(record["job_id"], "6779000")
            self.assertIn("--resolve-slurm-gpu", Path(record["script"]).read_text())


if __name__ == "__main__":
    unittest.main()
