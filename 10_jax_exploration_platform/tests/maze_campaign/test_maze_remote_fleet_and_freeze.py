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
        remote.adapt("/runs/folder-one")
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
                         (remote.job_prefix("/runs/folder-one"), remote.prepare, remote.verify_unit))

    def test_two_run_folders_never_share_a_job_name_prefix(self):
        """The prefix comes from the folder's name: the same for one folder, different for another, and a job
        named for one folder is not among the other's jobs unless its id is in that folder's ledger."""
        one, two = remote.job_prefix("/runs/folder-one"), remote.job_prefix("/elsewhere/runs/folder-two")
        self.assertEqual(one, remote.job_prefix("/other/place/folder-one"))
        self.assertNotEqual(one, two)
        self.assertRegex(one, r"^mazefleet-[0-9a-f]{8}$")
        remote.adapt("/runs/folder-one")
        status = {"jobs": {"11": {"name": f"{one}-gilbreth-standby-20261112-150000-1", "state": "RUNNING"},
                           "12": {"name": f"{two}-gilbreth-standby-20261112-150000-1", "state": "RUNNING"},
                           "13": {"name": "mazefleet-gilbreth-standby-20261112-134302-1", "state": "RUNNING"}}}
        self.assertEqual(sorted(fleet.fleet_jobs("gilbreth", status, set())), ["11"])
        self.assertEqual(sorted(fleet.fleet_jobs("gilbreth", status, {"13"})), ["11", "13"])

    def test_a_pulled_unit_is_checked_by_its_shard(self):
        """A complete shard replaces UVA's partial one, the old checkpoint goes, and the observed line is added."""
        with tempfile.TemporaryDirectory() as folder:
            run_dir, name = Path(folder), "unit-a"
            pulled = run_dir / "data/research" / name
            pulled.mkdir(parents=True)
            common.write_json(run_dir / "specifications" / f"{name}.json", {"windows_expected": 2})
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
                # a complete shard with fewer windows than the unit has is refused, and UVA's shard stays
                write_shard(run_dir, "from-server", windows=1, complete=True).rename(pulled / f"{name}.jsonl")
                self.assertIn("2 are expected", remote.verify_unit(run_dir, name))
                self.assertEqual(len((run_dir / "data" / f"{name}.jsonl").read_text().splitlines()), 2)
                # the complete shard arrives
                write_shard(run_dir, "from-server", windows=2, complete=True).rename(pulled / f"{name}.jsonl")
                (pulled / f"{name}.log").write_text("log\n")
                self.assertIsNone(remote.verify_unit(run_dir, name))
            lines = (run_dir / "data" / f"{name}.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(lines[-1])["record"], "unit_complete")
            self.assertFalse((run_dir / "checkpoints" / f"{name}.npz").exists())
            self.assertTrue((run_dir / "logs" / f"{name}.log").exists())
            self.assertEqual(json.loads((run_dir / "observed/pointmaze-test.jsonl").read_text()), observed)

    def test_the_status_call_reads_each_units_progress(self):
        """The status script stays valid bash, and its progress section is read beside the grid campaign's."""
        remote.adapt("/runs/folder-one")
        entry = {"remote_run": "/scratch/maze/run", "pools": {"l40s": {}}}
        script = fleet.status_script("gautschi", entry, ["unit-a", "unit-b"])
        self.assertEqual(subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode, 0)
        self.assertTrue(script.endswith(f'echo; echo "{fleet.MARK}end"\nexit 0\n'))
        answer = "\n".join([
            "", "@@fleet:queue", "running unit-a", "pending unit-b", "", "@@fleet:claims",
            "running unit-a job=12 host=g003 gpu=NVIDIA_L40S started=2026-11-12T16:30:00-05:00", "", "@@fleet:jobs",
            f"12|RUNNING|None|{remote.job_prefix('/runs/folder-one')}-gautschi-l40s-20261112-133000-1|5:02|smallgpu",
            "", "@@fleet:squeue_exit 0",
            "", "@@fleet:progress", "unit-a window=41 complete=no checkpoint=yes",
            "unit-b window= complete=no checkpoint=no", "", "@@fleet:estimates", "13 207000", "14 unknown",
            "", "@@fleet:end"])
        status = fleet.parse_status(answer)
        self.assertEqual(status["queue"]["running"], ["unit-a"])
        self.assertEqual(status["progress"], {
            "unit-a": {"windows_written": 42, "complete": False, "checkpoint": True},
            "unit-b": {"windows_written": 0, "complete": False, "checkpoint": False}})
        self.assertEqual(status["start_in_seconds"], {"13": 207000, "14": None})
        # unit-b waits, and a unit whose job the scheduler no longer lists has no job either
        self.assertEqual(remote.units_without_job("gautschi", status, ["unit-a", "unit-b"], {"12"}), 1)
        del status["jobs"]["12"]
        self.assertEqual(remote.units_without_job("gautschi", status, ["unit-a", "unit-b"], {"12"}), 2)
        lines = remote.unit_lines(Path("/nowhere"), "gautschi", status, [], [])
        self.assertEqual(lines, [])

    def test_units_wait_on_a_server_only_while_a_job_is_expected_to_start(self):
        """With a limit, a server whose only jobs wait and are estimated far away holds no unit."""
        entry = {"hand_over": {"start_within_seconds": 7200}}
        waiting, running = {"state": "PENDING"}, {"state": "RUNNING"}
        self.assertFalse(remote.start_expected(entry, {"12": waiting}, {"12": 207000}))
        self.assertTrue(remote.start_expected(entry, {"12": waiting}, {"12": 3600}))
        # a job that runs, a job with no estimate yet, no job at all, or a server without the rule
        self.assertTrue(remote.start_expected(entry, {"12": waiting, "13": running}, {"12": 207000}))
        self.assertTrue(remote.start_expected(entry, {"12": waiting}, {"12": None}))
        self.assertTrue(remote.start_expected(entry, {}, {}))
        self.assertTrue(remote.start_expected({"hand_over": {}}, {"12": waiting}, {"12": 207000}))

    def test_the_format_of_a_remote_shard_is_compared_with_a_uva_shard(self):
        """Equal record kinds, keys and kinds of values give no difference; a missing key and a changed kind do."""
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder)
            lines = [{"record": "job", "host": "node"}, {"record": "unit_start", "unit_id": "u", "copies": 2},
                     {"record": "episode_window", "window_index": 0, "loss": 0.5, "coverage_per_copy": [0.1, 0.2],
                      "clock": None},
                     {"record": "unit_complete", "windows": 1, "copies": 2}]
            (run_dir / "uva.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
            # the same unit with 3 copies elsewhere: other values, the same format
            other = json.loads(json.dumps(lines))
            other[1]["copies"], other[2]["coverage_per_copy"], other[3]["copies"] = 3, [0.3, 0.4, 0.5], 3
            (run_dir / "same.jsonl").write_text("".join(json.dumps(line) + "\n" for line in other))
            uva = remote.shard_format(run_dir / "uva.jsonl")
            self.assertEqual(uva["episode_window"], {"count": 1, "key_sets": 1, "keys": {
                "record": "text", "window_index": "whole number", "loss": "number",
                "coverage_per_copy": "list, one entry per copy", "clock": "null"}})
            self.assertEqual(remote.format_differences(uva, remote.shard_format(run_dir / "same.jsonl")), [])
            # a key that is missing, and a value of another kind
            del other[2]["loss"]
            other[3]["windows"] = "1"
            (run_dir / "different.jsonl").write_text("".join(json.dumps(line) + "\n" for line in other))
            differences = remote.format_differences(uva, remote.shard_format(run_dir / "different.jsonl"))
            self.assertEqual(differences, [
                "episode_window.loss: UVA has number, the remote shard has no such key",
                "unit_complete.windows: UVA has whole number, the remote shard has text"])

    def test_the_first_unit_of_a_server_is_compared_with_a_uva_unit_before_it_counts(self):
        """Without a UVA unit to compare with the unit waits; with one, the result is written in the server table."""
        with tempfile.TemporaryDirectory() as folder:
            run_dir, name = Path(folder), "unit-a"
            pulled = run_dir / "data/research" / name
            pulled.mkdir(parents=True)
            (run_dir / "queue/done").mkdir(parents=True)
            common.write_json(pulled / "summary.json", {"completed": True})
            common.write_json(run_dir / "specifications" / f"{name}.json", {
                "windows_expected": 2, "workload": "w", "reward_setting": "r", "method": "m"})
            common.write_json(run_dir / "slurm/remote_assignment.json", {name: {"server": "gautschi"}})
            common.write_json(run_dir / "infrastructure/remote_fleet_servers.json", {"gautschi": {}})
            write_shard(run_dir, "from-server", windows=2, complete=True).rename(pulled / f"{name}.jsonl")
            with mock.patch.object(remote, "OBSERVED", run_dir / "observed"), \
                    mock.patch.dict(remote.CURRENT, {"table": None}):
                self.assertIn("no unit trained on UVA", remote.verify_unit(run_dir, name))
                self.assertFalse((run_dir / "data" / f"{name}.jsonl").exists())
                # a UVA unit of the same workload is done: the formats agree, and the shard moves
                write_shard(run_dir, "uva-unit", windows=2, complete=True)
                common.write_json(run_dir / "queue/done/uva-unit.json", {
                    "name": "uva-unit", "workload": "w", "reward_setting": "r", "method": "m", "result": {}})
                # the stand-in shards have no job record; the comparison needs the four record kinds
                for shard in (run_dir / "data/uva-unit.jsonl", pulled / f"{name}.jsonl"):
                    shard.write_text(json.dumps({"record": "job"}) + "\n" + shard.read_text().replace(
                        '"record": "unit_start", ', '"record": "unit_start", "copies": 16, '))
                self.assertIsNone(remote.verify_unit(run_dir, name))
            first = common.read_json(run_dir / "infrastructure/remote_fleet_servers.json")["gautschi"]["first_unit"]
            self.assertEqual((first["state"], first["unit"], first["compared_with_uva_unit"], first["differences"]),
                             ("passed", name, "uva-unit", []))
            self.assertTrue((run_dir / "data" / f"{name}.jsonl").exists())

    def test_the_uva_unit_a_format_is_compared_with(self):
        """Only a unit trained on UVA with the same workload and reward setting counts; the same method first."""
        unit = {"workload": "w", "reward_setting": "r", "method": "m"}
        done = [dict(unit, name="other-method", method="x", finished_at="3", result={}),
                dict(unit, name="same-method", finished_at="1", result={}),
                dict(unit, name="from-a-server", finished_at="4", result={"server": "gautschi"}),
                dict(unit, name="other-workload", workload="v", finished_at="5", result={})]
        self.assertEqual(remote.uva_reference(done, unit), "same-method")
        self.assertEqual(remote.uva_reference(done[:1], unit), "other-method")
        self.assertIsNone(remote.uva_reference(done[2:], unit))

    def test_a_hand_over_takes_one_comparable_unit_from_the_end_until_the_first_unit_is_verified(self):
        """Before the first unit is verified a server gets one unit that has a UVA unit to compare with."""
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder)
            (run_dir / "slurm").mkdir()
            for state in ("pending", "done"):
                (run_dir / "queue" / state).mkdir(parents=True)
            record = {"workload": "w", "reward_setting": "r", "method": "m"}
            for order, name in enumerate(("pointmaze-a", "pointmaze-b", "pointmaze-other-workload", "antmaze-c")):
                common.write_json(run_dir / "queue/pending" / f"{name}.json", dict(
                    record, name=name, order=order, workload="v" if "other" in name else "w"))
            common.write_json(run_dir / "queue/done/pointmaze-done.json", dict(record, name="pointmaze-done", result={}))
            table = {"gautschi": {"hand_over": {"units_at_once": 4, "match": "pointmaze", "pool": "l40s"}}}
            # not checked yet: one unit, the last of the order that can be compared
            self.assertEqual(remote.hand_over(run_dir, table, "gautschi", dry_run=True), ["pointmaze-b"])
            # verified: up to four, any workload, from the end of the order; the ant unit does not match
            table["gautschi"]["first_unit"] = {"state": "passed"}
            self.assertEqual(remote.hand_over(run_dir, table, "gautschi", dry_run=True),
                             ["pointmaze-other-workload", "pointmaze-b", "pointmaze-a"])
            # a failed check: none
            table["gautschi"]["first_unit"] = {"state": "failed"}
            self.assertEqual(remote.hand_over(run_dir, table, "gautschi", dry_run=True), [])
            # a dry run moves nothing
            self.assertEqual(len(list((run_dir / "queue/pending").glob("*.json"))), 4)

    def test_a_returned_unit_brings_its_checkpoint_and_shard(self):
        """Checkpoint and shard replace UVA's together, and UVA's record takes over the server's attempts."""
        with tempfile.TemporaryDirectory() as folder:
            run_dir, name = Path(folder), "unit-a"
            for sub in ("queue/remote_gautschi", "queue/pending", "slurm", "data"):
                (run_dir / sub).mkdir(parents=True)
            common.write_json(run_dir / "queue/remote_gautschi" / f"{name}.json", {"name": name, "attempts": 0})
            common.write_json(run_dir / "slurm/remote_assignment.json", {name: {"server": "gautschi"}})
            (run_dir / "data" / f"{name}.jsonl").write_text("what UVA held\n")
            table = {"gautschi": {"ssh": ["ssh", "gautschi"], "remote_run": "/scratch/run"}}

            def copy(run, servers, server, arguments):
                """Stand in for rsync: the server has the unit's record, checkpoint and shard."""
                landing = Path(arguments[-1])
                (landing / f"{name}.json").write_text(json.dumps({"name": name, "attempts": 1}))
                (landing / f"{name}.npz").write_text("state")
                (landing / f"{name}.jsonl").write_text("the server's shard\n")

            with mock.patch.object(fleet, "rsync", copy):
                remote.bring_back(run_dir, table, "gautschi", [(name, False)], "tick")
            self.assertEqual(common.read_json(run_dir / "queue/pending" / f"{name}.json")["attempts"], 1)
            self.assertEqual((run_dir / "data" / f"{name}.jsonl").read_text(), "the server's shard\n")
            self.assertEqual((run_dir / "checkpoints" / f"{name}.npz").read_text(), "state")
            self.assertEqual(common.read_json(run_dir / "slurm/remote_assignment.json")[name]["state"], "returned")

    def test_the_code_copy_becomes_a_git_repository_on_the_server(self):
        """The script that gives the code copy a commit is valid bash and names the UVA commit."""
        script = remote.repository_script("/scratch/maze/code/source_abc", "abc")
        self.assertEqual(subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode, 0)
        self.assertIn("copy-of-uva-commit-abc", script)

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
