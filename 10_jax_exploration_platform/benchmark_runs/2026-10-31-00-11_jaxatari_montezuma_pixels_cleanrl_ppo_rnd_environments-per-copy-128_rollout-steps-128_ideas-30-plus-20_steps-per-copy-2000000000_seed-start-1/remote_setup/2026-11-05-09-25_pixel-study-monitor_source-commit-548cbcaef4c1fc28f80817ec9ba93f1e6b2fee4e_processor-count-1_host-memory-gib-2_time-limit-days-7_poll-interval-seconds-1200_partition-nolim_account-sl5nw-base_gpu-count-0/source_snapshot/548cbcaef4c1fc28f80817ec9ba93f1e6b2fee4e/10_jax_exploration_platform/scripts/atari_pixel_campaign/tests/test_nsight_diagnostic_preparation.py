"""Prepare private diagnostic packets from real local Git fixtures without invoking Nsight or Slurm."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

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
        result = P.prepare(self.root, self.commit, self.run, self.name, self.tool, unreserved=True)
        packet, unit = Path(result["packet"]), Path(result["unit"])
        for directory in (packet, unit):
            for name in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md", "progress.json"):
                self.assertTrue((directory / name).is_file())
        plan = json.loads((unit / "measurement_plan.json").read_text())
        self.assertEqual(plan["python"], "/usr/bin/python3")
        self.assertEqual(json.loads((packet / "manifest.yaml").read_text())["python"], "/usr/bin/python3")
        self.assertIsNone(plan["job_id"]); self.assertIsNone(plan["gpu_uuid"])
        self.assertEqual(plan["work"]["learner_updates"], 0)
        self.assertIsNone(plan["resources"]["reservation"])
        self.assertEqual(plan["resources"]["qos"], "normal")
        command = json.loads((packet / "held_submission_command.json").read_text())
        self.assertIn("--qos=normal", command)
        self.assertNotIn("--qos=csresnolim", command)
        for argument in ("--hold", "--no-requeue", "--cpus-per-task=1", "--mem=2G", "--time=00:05:00",
                         "--gres=gpu:nvidia_rtx_a4500:1", "--nodelist=jaguar03"):
            self.assertIn(argument, command)
        script = (packet / "job.sbatch").read_text()
        self.assertEqual(script.count("exec srun "), 1); self.assertIn(" -I -S -B ", script)
        self.assertFalse(any(argument.startswith("--reservation") for argument in command))
        self.assertNotIn("#SBATCH --reservation", script)
        self.assertIn("An unreserved job was explicitly selected.", script)
        self.assertNotIn("sl5nw_168", script)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=", script)
        self.assertFalse((unit / "owned_launch_inputs.json").exists())
        self.assertFalse((unit / "profile_process_intent.json").exists())

    def test_partial_or_previously_prepared_unit_is_never_reused(self):
        """A second preparation refuses rather than replacing any old artifact."""
        P.prepare(self.root, self.commit, self.run, self.name, self.tool, unreserved=True)
        with self.assertRaisesRegex(ValueError, "never reuse"):
            P.prepare(self.root, self.commit, self.run, self.name, self.tool, unreserved=True)

    def test_dirty_source_refuses_before_any_packet_creation(self):
        """An uncommitted source change cannot enter the executable staging closure."""
        (self.root / (P.RELATIVE + P.FILES[0])).write_text("# changed private source\n")
        with self.assertRaisesRegex(ValueError, "detached and clean"):
            P.prepare(self.root, self.commit, self.run, self.name, self.tool, unreserved=True)
        self.assertEqual(list((self.run / "data").iterdir()), [])

    def test_explicit_reservation_matches_plan_script_and_held_command(self):
        """An explicitly supplied name is retained everywhere without using an old default."""
        reservation = "synthetic_selected_reservation"
        result = P.prepare(self.root, self.commit, self.run, self.name, self.tool, reservation=reservation)
        packet, unit = Path(result["packet"]), Path(result["unit"])
        plan = json.loads((unit / "measurement_plan.json").read_text())
        self.assertEqual(plan["resources"]["reservation"], reservation)
        self.assertEqual(plan["resources"]["qos"], "csresnolim")
        for directory in (packet, unit):
            for name in ("manifest.yaml", "config_resolved.yaml"):
                self.assertEqual(json.loads((directory / name).read_text()), plan)
            self.assertIn("explicitly selected reservation " + reservation,
                          (directory / "experiment_background.md").read_text())
        command = json.loads((packet / "held_submission_command.json").read_text())
        self.assertIn("--qos=csresnolim", command)
        self.assertNotIn("--qos=normal", command)
        self.assertEqual([argument for argument in command if argument.startswith("--reservation")],
                         ["--reservation=" + reservation])
        script = (packet / "job.sbatch").read_text()
        self.assertEqual(script.count("#SBATCH --reservation=" + reservation + "\n"), 1)
        self.assertNotIn("sl5nw_168", json.dumps(plan) + script + json.dumps(command))

    def test_missing_selection_refuses_before_source_reads_or_output(self):
        """No API default can silently restore an expired reservation."""
        with mock.patch.object(P, "git", side_effect=AssertionError("source must not be read")):
            with self.assertRaisesRegex(ValueError, "explicit reservation"):
                P.prepare(self.root, self.commit, self.run, self.name, self.tool)
        self.assertEqual(list((self.run / "data").iterdir()), [])

    def test_conflicting_selection_refuses_before_source_reads_or_output(self):
        """A caller cannot ask for both a reservation and an unreserved job."""
        with mock.patch.object(P, "git", side_effect=AssertionError("source must not be read")):
            with self.assertRaisesRegex(ValueError, "never both"):
                P.prepare(self.root, self.commit, self.run, self.name, self.tool,
                          reservation="synthetic_selected_reservation", unreserved=True)
        self.assertEqual(list((self.run / "data").iterdir()), [])

    def test_invalid_reservation_names_refuse_before_source_reads_or_output(self):
        """Reservation text cannot inject another argument or a batch directive."""
        for reservation in ("", " spaced ", "../name", "name\n#SBATCH --nodes=2", "--unreserved", True):
            with self.subTest(reservation=reservation):
                with mock.patch.object(P, "git", side_effect=AssertionError("source must not be read")):
                    with self.assertRaisesRegex(ValueError, "explicit reservation"):
                        P.prepare(self.root, self.commit, self.run, self.name, self.tool, reservation=reservation)
        self.assertEqual(list((self.run / "data").iterdir()), [])

    def test_unreserved_selection_requires_an_exact_boolean(self):
        """Integer or string values cannot stand in for an explicit Boolean selection."""
        for value in (1, 0, "true"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be Boolean"):
                    P.prepare(self.root, self.commit, self.run, self.name, self.tool, unreserved=value)
        self.assertEqual(list((self.run / "data").iterdir()), [])


class CommandPlacementTests(unittest.TestCase):
    """Check the required CLI choice without reading a source tree or preparing a packet."""
    def arguments(self):
        """Return labelled synthetic values; the preparation call is replaced in these CLI fixtures."""
        return ["--helper-root", "/synthetic/source", "--helper-commit", "1" * 40,
                "--campaign-run", "/synthetic/campaign", "--unit-id", "2026-synthetic-diagnostic",
                "--tool-record", "/synthetic/tool.json"]

    def test_cli_requires_one_explicit_choice(self):
        """No-choice CLI calls stop before preparing output."""
        with mock.patch.object(P, "prepare") as prepare, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                P.main(self.arguments())
        self.assertEqual(error.exception.code, 2); prepare.assert_not_called()

    def test_cli_rejects_both_choices(self):
        """Mutually exclusive CLI flags cannot produce an ambiguous packet."""
        with mock.patch.object(P, "prepare") as prepare, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                P.main([*self.arguments(), "--reservation", "synthetic_reservation", "--unreserved"])
        self.assertEqual(error.exception.code, 2); prepare.assert_not_called()

    def test_cli_passes_an_explicit_unreserved_choice(self):
        """The unreserved CLI branch passes None plus the explicit true selection."""
        with mock.patch.object(P, "prepare", return_value={}) as prepare, contextlib.redirect_stdout(io.StringIO()):
            P.main([*self.arguments(), "--unreserved"])
        self.assertEqual(prepare.call_args.kwargs, {"reservation": None, "unreserved": True})

    def test_cli_passes_the_explicit_reservation_name(self):
        """The reserved CLI branch retains the exact caller-supplied name."""
        with mock.patch.object(P, "prepare", return_value={}) as prepare, contextlib.redirect_stdout(io.StringIO()):
            P.main([*self.arguments(), "--reservation", "synthetic_reservation"])
        self.assertEqual(prepare.call_args.kwargs,
                         {"reservation": "synthetic_reservation", "unreserved": False})


if __name__ == "__main__": unittest.main()
