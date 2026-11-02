"""Prepare private diagnostic packets from real local Git fixtures without invoking Nsight or Slurm."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

W = Path(__file__).resolve().parents[4]
SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synthetic_diagnostic_preparation", W / ".review_metadata/prepare_nsight_process_diagnostic.py")
P = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(P)


class PreparationTests(unittest.TestCase):
    """Use ordinary private files and local Git only; never run the tool, learner or scheduler."""
    def setUp(self):
        """Commit the two real draft modules in a clearly synthetic detached local repository."""
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name); self.root = self.base / "source"; self.root.mkdir()
        for name in P.FILES:
            path = self.root / (P.RELATIVE + name); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((SCRIPTS / name).read_bytes())
        def git(*arguments):
            """Run only exact local fixture Git operations, with no hooks or remote."""
            return subprocess.check_output(["git", "-C", str(self.root), *arguments], stderr=subprocess.PIPE)
        git("init", "-q"); git("-c", "user.name=Synthetic metadata fixture", "-c", "user.email=synthetic@example.invalid",
            "-c", "core.hooksPath=/dev/null", "add", "--", *[P.RELATIVE + name for name in P.FILES])
        git("-c", "user.name=Synthetic metadata fixture", "-c", "user.email=synthetic@example.invalid",
            "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", "Synthetic process diagnostic source fixture")
        self.commit = git("rev-parse", "HEAD").decode().strip(); git("checkout", "-q", "--detach", self.commit)
        self.run = self.base / "campaign"
        for relative in ("remote_setup", "data", "slurm/jobs"): (self.run / relative).mkdir(parents=True, exist_ok=True)
        self.tool = self.base / "synthetic_tool_record.json"
        self.tool.write_text(json.dumps({"path": "/synthetic/unexecuted/nsys",
            "version": "NVIDIA Nsight Systems version 2024.7.1.84-247135125610v0",
            "sha256": "48e4c308f538c8971b8f62c389834372d986f00f200743f1eae90918f2f04e1d"}))
        self.name = "2026-11-01-23-45_nsight-process-observation_gpu-count-1_processor-count-1_memory-gib-2_time-minutes-5"

    def test_real_preparation_has_all_standards_null_ids_and_one_bounded_held_command(self):
        """A fresh private packet binds exact Git bytes and contains no future launch identity."""
        result = P.prepare(self.root, self.commit, self.run, self.name, self.tool)
        packet, unit = Path(result["packet"]), Path(result["unit"])
        for directory in (packet, unit):
            for name in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md", "progress.json"):
                self.assertTrue((directory / name).is_file())
        plan = json.loads((unit / "measurement_plan.json").read_text())
        self.assertIsNone(plan["job_id"]); self.assertIsNone(plan["gpu_uuid"])
        self.assertEqual(plan["work"]["learner_updates"], 0)
        command = json.loads((packet / "held_submission_command.json").read_text())
        for argument in ("--hold", "--no-requeue", "--cpus-per-task=1", "--mem=2G", "--time=00:05:00",
                         "--gres=gpu:nvidia_rtx_a4500:1", "--nodelist=jaguar03"):
            self.assertIn(argument, command)
        script = (packet / "job.sbatch").read_text()
        self.assertEqual(script.count("exec srun "), 1); self.assertIn(" -I -S -B ", script)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=", script)
        self.assertFalse((unit / "owned_launch_inputs.json").exists())
        self.assertFalse((unit / "profile_process_intent.json").exists())

    def test_partial_or_previously_prepared_unit_is_never_reused(self):
        """A second preparation refuses rather than replacing any old artifact."""
        P.prepare(self.root, self.commit, self.run, self.name, self.tool)
        with self.assertRaisesRegex(ValueError, "never reuse"):
            P.prepare(self.root, self.commit, self.run, self.name, self.tool)

    def test_dirty_source_refuses_before_any_packet_creation(self):
        """An uncommitted source change cannot enter the executable staging closure."""
        (self.root / (P.RELATIVE + P.FILES[0])).write_text("# changed private source\n")
        with self.assertRaisesRegex(ValueError, "detached and clean"):
            P.prepare(self.root, self.commit, self.run, self.name, self.tool)
        self.assertEqual(list((self.run / "data").iterdir()), [])


if __name__ == "__main__": unittest.main()
