"""The remote wrappers' local parts (no server is contacted) and the frozen code copy (on a temporary repository)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from maze_test_support import write_shard
from grid_campaign import common, load, own
import freeze_code

remote = own("remote_fleet")
fleet = load("remote_fleet")


class RemoteWrappers(unittest.TestCase):
    """What this study replaces in the grid campaign's remote fleet."""

    def test_the_worker_job_script_starts_this_studys_worker(self):
        """The grid campaign's script writer fills this study's template; bash accepts the result."""
        remote.adapt()
        entry = {"remote_run": "/scratch/maze/run", "python": "/scratch/env/bin/python",
                 "code": "/scratch/maze/code/current/10_jax_exploration_platform", "unit_seconds": 1800}
        pool = {"gpu_class": "h100_mit", "time_limit": "12:00:00"}
        text = fleet.worker_script("mit", entry, "preemptable", pool)
        self.assertIn('exec "$PY" "$CODE/scripts/maze_campaign/worker.py" --run-dir "$RUN" --remote-records '
                      "--minimum-seconds 1800", text)
        self.assertNotIn("@", text.replace("@SERVER@", ""))
        self.assertNotIn("train_unit.py", text)
        self.assertEqual(subprocess.run(["bash", "-n"], input=text, text=True, capture_output=True).returncode, 0)
        self.assertEqual((fleet.JOB_PREFIX, fleet.prepare, fleet.verify_unit),
                         ("mazefleet", remote.prepare, remote.verify_unit))

    def test_a_pulled_unit_is_checked_by_its_shard(self):
        """A complete shard replaces UVA's partial one, the old checkpoint goes, and the observed line is added."""
        with tempfile.TemporaryDirectory() as folder:
            run_dir, name = Path(folder), "unit-a"
            pulled = run_dir / "data/research" / name
            pulled.mkdir(parents=True)
            self.assertEqual(remote.verify_unit(run_dir, name), "summary.json is missing")
            observed = {"workload": "pointmaze-test", "device_name": "NVIDIA H100 80GB HBM3", "copies": 2048,
                        "seconds_per_iteration": 0.02}
            common.write_json(pulled / "summary.json", {"completed": True, "observed": observed})
            # UVA still holds a partial shard and a checkpoint of an earlier attempt
            write_shard(run_dir, name, windows=1)
            (run_dir / "checkpoints").mkdir()
            (run_dir / "checkpoints" / f"{name}.npz").write_text("old")
            # the server's shard is not complete yet
            write_shard(run_dir, "from-server", windows=2).rename(pulled / f"{name}.jsonl")
            with mock.patch.object(remote, "OBSERVED", run_dir / "observed"):
                self.assertEqual(remote.verify_unit(run_dir, name), "the shard does not end with a unit_complete record")
                self.assertFalse((run_dir / "observed").exists())
                # the complete shard arrives
                write_shard(run_dir, "from-server", windows=2, complete=True).rename(pulled / f"{name}.jsonl")
                (pulled / f"{name}.log").write_text("log\n")
                self.assertIsNone(remote.verify_unit(run_dir, name))
            lines = (run_dir / "data" / f"{name}.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(lines[-1])["record"], "unit_complete")
            self.assertFalse((run_dir / "checkpoints" / f"{name}.npz").exists())
            self.assertTrue((run_dir / "logs" / f"{name}.log").exists())
            self.assertEqual(json.loads((run_dir / "observed/pointmaze-test.jsonl").read_text()), observed)

    def test_no_cancel_command_is_offered(self):
        """The grid campaign's cancel-pending is refused here."""
        with mock.patch.object(sys, "argv", ["remote_fleet.py", "--run-dir", "/nowhere", "cancel-pending",
                                             "--server", "mit"]), self.assertRaises(SystemExit) as refusal:
            remote.main()
        self.assertIn("cancel_ledger.py", str(refusal.exception))

    def test_the_order_of_servers(self):
        """MIT, then Purdue Gautschi, then TAMU ACES, then NCSA DeltaAI."""
        self.assertEqual(remote.SERVER_ORDER, ("mit", "gautschi", "aces", "deltaai"))


class FrozenCode(unittest.TestCase):
    """The code copy a run folder's jobs run."""

    def setUp(self):
        """Make the temporary folder the tests of this class work in."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.repository, self.run_dir = Path(folder.name) / "repository", Path(folder.name) / "run"
        for folder in freeze_code.FOLDERS:
            (self.repository / "10_jax_exploration_platform" / folder).mkdir(parents=True)
            (self.repository / "10_jax_exploration_platform" / folder / "tool.py").write_text("print('committed')\n")
        (self.repository / "notes.md").write_text("outside the three code folders\n")
        self.git("init", "--quiet")
        self.commit("first")

    def git(self, *arguments):
        """Run git in the temporary repository."""
        return subprocess.check_output(["git", "-C", str(self.repository), "-c", "user.name=test",
                                        "-c", "user.email=test@example.com", *arguments], text=True)

    def commit(self, message):
        """Commit everything in the temporary repository."""
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", message)

    def test_the_copy_is_the_commit_and_is_recorded(self):
        """source.json names the commit and the copy, which holds the committed files and its commit's name."""
        commit = freeze_code.freeze(self.run_dir, self.repository)
        source = common.read_json(self.run_dir / "source.json")
        platform = Path(source["snapshot"]) / "10_jax_exploration_platform"
        self.assertEqual((source["commit"], source["campaign"], source["previous_commits"]), (commit, "maze", []))
        self.assertEqual((platform / "scripts/tool.py").read_text(), "print('committed')\n")
        self.assertEqual((platform / "SOURCE_COMMIT").read_text().strip(), commit)
        self.assertFalse((Path(source["snapshot"]) / "notes.md").exists())
        # a later commit gives a second copy; the first one stays and is named as the earlier one
        (self.repository / "10_jax_exploration_platform/scripts/tool.py").write_text("print('second')\n")
        self.commit("second")
        second = freeze_code.freeze(self.run_dir, self.repository)
        self.assertEqual(common.read_json(self.run_dir / "source.json")["previous_commits"], [commit])
        self.assertTrue((self.run_dir / "code" / f"source_{commit}").exists())
        self.assertTrue((self.run_dir / "code" / f"source_{second}").exists())

    def test_code_that_is_not_committed_is_refused(self):
        """A changed or new file in the code folders stops the freeze; a change elsewhere does not."""
        (self.repository / "notes.md").write_text("changed outside the code folders\n")
        freeze_code.freeze(self.run_dir, self.repository)
        (self.repository / "10_jax_exploration_platform/scripts/new_tool.py").write_text("x = 1\n")
        with self.assertRaises(SystemExit) as refusal:
            freeze_code.freeze(self.run_dir, self.repository)
        self.assertIn("new_tool.py", str(refusal.exception))

    def test_uncommitted_code_can_be_left_out_of_the_copy(self):
        """With leave_out_uncommitted the copy is the commit alone: the uncommitted file is not in it."""
        (self.repository / "10_jax_exploration_platform/scripts/tool.py").write_text("print('not committed')\n")
        (self.repository / "10_jax_exploration_platform/scripts/new_tool.py").write_text("x = 1\n")
        commit = freeze_code.freeze(self.run_dir, self.repository, leave_out_uncommitted=True)
        platform = self.run_dir / "code" / f"source_{commit}" / "10_jax_exploration_platform"
        self.assertEqual((platform / "scripts/tool.py").read_text(), "print('committed')\n")
        self.assertFalse((platform / "scripts/new_tool.py").exists())


if __name__ == "__main__":
    unittest.main()
