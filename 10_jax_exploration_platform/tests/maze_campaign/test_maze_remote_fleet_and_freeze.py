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
import unit_queue

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

    def test_a_server_may_ask_for_a_login_shell_and_its_own_job_lines(self):
        """Without the two table fields the script is the one every server had; with them (TAMU ACES) it starts
        as a login shell and holds the server's lines before the library path is cleared, and nothing else differs."""
        remote.adapt("/runs/folder-one")
        entry = {"remote_run": "/scratch/maze/run", "python": "/scratch/env/bin/python",
                 "code": "/scratch/maze/code/current/10_jax_exploration_platform", "unit_seconds": 1800}
        pool = {"gpu_class": "h100_pcie_aces", "time_limit": "1-00:00:00"}
        plain = fleet.worker_script("aces", entry, "gpu-h100", pool)
        self.assertTrue(plain.startswith("#!/bin/bash\n# Worker job of the maze study on aces, pool gpu-h100."))
        self.assertIn("MKL_NUM_THREADS=1\n# Module-provided CUDA libraries", plain)
        # fields that are present and empty give the same script as fields that are absent
        self.assertEqual(fleet.worker_script("aces", dict(entry, job_login_shell=False, job_setup_lines=[]),
                                             "gpu-h100", pool), plain)
        lines = ["module load WebProxy/0000", 'export WARP_CACHE_PATH="${TMPDIR:-/tmp}/$(id -un)/warp_kernel_cache"']
        changed = fleet.worker_script("aces", dict(entry, job_login_shell=True, job_setup_lines=lines),
                                      "gpu-h100", pool)
        self.assertTrue(changed.startswith("#!/bin/bash -l\n# Worker job of the maze study on aces, pool gpu-h100."))
        self.assertIn(f"MKL_NUM_THREADS=1\n{lines[0]}\n{lines[1]}\n# Module-provided CUDA libraries", changed)
        self.assertLess(changed.index(lines[0]), changed.index("unset LD_LIBRARY_PATH"))
        self.assertEqual(changed.replace("#!/bin/bash -l\n", "#!/bin/bash\n", 1)
                         .replace(f"{lines[0]}\n{lines[1]}\n", "", 1), plain)
        for text in (plain, changed):
            self.assertNotIn("@", text)
            self.assertEqual(subprocess.run(["bash", "-n"], input=text, text=True, capture_output=True).returncode, 0)

    def test_a_unit_handed_over_again_is_not_taken_back_for_its_old_withdrawn_marker(self):
        """Only a unit the server holds in withdrawn ALONE is stranded; a unit handed over again stays."""
        queue = {"incoming": ["d"], "pending": ["b"], "running": ["c"], "done": [], "failed": [],
                 "withdrawn": ["a", "b", "c", "d", "x"]}
        self.assertEqual(remote.stranded_units(["a", "b", "c", "d"], queue), ["a"])
        self.assertEqual(remote.stranded_units(["b", "c"], queue), [])
        self.assertEqual(remote.stranded_units([], queue), [])

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
            # a server set up with `units_at_once` 0 gets none, before its first unit is verified and after
            table["gautschi"]["hand_over"]["units_at_once"] = 0
            self.assertEqual(remote.hand_over(run_dir, table, "gautschi", dry_run=True), [])
            table["gautschi"]["first_unit"] = {"state": "passed"}
            self.assertEqual(remote.hand_over(run_dir, table, "gautschi", dry_run=True), [])
            # verified: up to four, any workload, from the end of the order; the ant unit does not match
            table["gautschi"]["hand_over"]["units_at_once"] = 4
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


class CodeCopiesOnServers(unittest.TestCase):
    """A unit's lowest code copy number goes to a server with its record and comes back with it."""

    def run_folder(self, folder, record):
        """A run folder with one unit in a given UVA queue folder; returns the run folder."""
        run_dir = Path(folder)
        for sub in ("queue/pending", "queue/remote_gilbreth", "slurm", "specifications"):
            (run_dir / sub).mkdir(parents=True)
        common.write_json(run_dir / "queue" / record.pop("in") / f"{record['name']}.json", record)
        common.write_json(run_dir / "specifications" / f"{record['name']}.json", {"unit_id": record["name"]})
        return run_dir

    def test_a_handed_over_record_carries_the_number_to_the_server(self):
        """The record copied to the server's incoming folder is UVA's record, lowest copy number included."""
        with tempfile.TemporaryDirectory() as folder:
            name = "pointmaze-a"
            run_dir = self.run_folder(folder, {"in": "pending", "name": name, "order": 1, "workload": "w",
                                               "reward_setting": "r", "method": "m", "attempts": 2,
                                               "minimum_copy_number": 3})
            table = {"gilbreth": {"ssh": ["ssh", "gilbreth"], "remote_run": "/scratch/run",
                                  "first_unit": {"state": "passed"},
                                  "hand_over": {"units_at_once": 4, "match": "pointmaze", "pool": "standby"}}}
            sent = {}

            def copy(run, servers, server, arguments):
                """Stand in for rsync: keep what the hand-over tree holds for the server's incoming folder."""
                tree = Path(arguments[0])
                sent.update(common.read_json(tree / "queue/incoming" / f"{name}.json"))

            answer = f"{fleet.MARK}moved place {name} no\n{fleet.MARK}end\n"
            with mock.patch.object(fleet, "rsync", copy), mock.patch.object(fleet, "remote", return_value=answer):
                self.assertEqual(remote.hand_over(run_dir, table, "gilbreth", dry_run=False), [name])
            self.assertEqual((sent["minimum_copy_number"], sent["attempts"]), (3, 2))
            self.assertEqual(common.read_json(run_dir / "queue/remote_gilbreth" / f"{name}.json")["minimum_copy_number"], 3)

    def bring_back(self, uva_record, server_record):
        """Bring one unit back from a server that holds the given record of it; returns UVA's pending record."""
        with tempfile.TemporaryDirectory() as folder:
            name = "unit-a"
            run_dir = self.run_folder(folder, dict(uva_record, name=name, **{"in": "remote_gilbreth"}))
            common.write_json(run_dir / "slurm/remote_assignment.json", {name: {"server": "gilbreth"}})
            table = {"gilbreth": {"ssh": ["ssh", "gilbreth"], "remote_run": "/scratch/run"}}

            def copy(run, servers, server, arguments):
                """Stand in for rsync: the server's record of the unit lands, when the server has one."""
                if server_record is not None:
                    common.write_json(Path(arguments[-1]) / f"{name}.json", dict(server_record, name=name))

            with mock.patch.object(fleet, "rsync", copy):
                remote.bring_back(run_dir, table, "gilbreth", [(name, False)], "tick")
            return common.read_json(run_dir / "queue/pending" / f"{name}.json")

    def test_a_returned_unit_keeps_the_larger_number(self):
        """A worker on the server raised the number: UVA takes it over. The number never falls on the way back."""
        # the server's worker (copy 3) claimed a unit built for copy 2
        record = self.bring_back({"attempts": 0, "minimum_copy_number": 2}, {"attempts": 1, "minimum_copy_number": 3})
        self.assertEqual((record["minimum_copy_number"], record["attempts"]), (3, 1))
        # the server's record is older than UVA's knowledge of the unit: UVA's number stays
        record = self.bring_back({"attempts": 0, "minimum_copy_number": 3}, {"attempts": 1, "minimum_copy_number": 1})
        self.assertEqual((record["minimum_copy_number"], record["attempts"]), (3, 1))
        record = self.bring_back({"attempts": 0, "minimum_copy_number": 3}, {"attempts": 1})
        self.assertEqual(record["minimum_copy_number"], 3)
        # no record came back from the server: UVA's record returns as it was
        self.assertEqual(self.bring_back({"attempts": 0, "minimum_copy_number": 2}, None),
                         {"name": "unit-a", "attempts": 0, "minimum_copy_number": 2})
        # records from before the numbers stay without the field
        self.assertNotIn("minimum_copy_number", self.bring_back({"attempts": 0}, {"attempts": 1}))

    def test_a_move_on_the_server_keeps_the_record_as_the_worker_wrote_it(self):
        """Recovering a unit whose job ended, and withdrawing a waiting one, rename the record: the number stays."""
        with tempfile.TemporaryDirectory() as folder:
            server = Path(folder)
            for state in ("running", "pending", "withdrawn"):
                (server / "queue" / state).mkdir(parents=True)
            (server / "logs").mkdir()
            common.write_json(server / "queue/running/unit-a.json", {"name": "unit-a", "minimum_copy_number": 3, "job_id": "12"})
            common.write_json(server / "queue/pending/unit-b.json", {"name": "unit-b", "minimum_copy_number": 2})
            script = fleet.move_script({"remote_run": str(server)}, {
                "place": [], "recover": ["unit-a"], "pull_back": [], "withdraw": ["unit-b"]}, 1)
            answer = subprocess.run(["bash"], input=script, text=True, capture_output=True, check=True).stdout
            self.assertEqual([row[:2] for row in fleet.parse_moves(answer)], [("recover", "unit-a"), ("withdraw", "unit-b")])
            self.assertEqual(common.read_json(server / "queue/pending/unit-a.json")["minimum_copy_number"], 3)
            self.assertEqual(common.read_json(server / "queue/withdrawn/unit-b.json")["minimum_copy_number"], 2)

    def test_prepare_sends_the_copy_with_its_number(self):
        """The folder prepare copies to a server is the frozen copy's platform folder, which holds the number file."""
        with tempfile.TemporaryDirectory() as folder:
            run_dir, snapshot = Path(folder) / "run", Path(folder) / "run/code/source_abc"
            platform = snapshot / "10_jax_exploration_platform"
            platform.mkdir(parents=True)
            (platform / unit_queue.COPY_NUMBER_FILE).write_text("4\n")
            (run_dir / "slurm").mkdir()
            common.write_json(run_dir / "source.json", {"commit": "abc", "snapshot": str(snapshot), "copy_number": 4,
                                                        "highest_copy_number": 4})
            common.write_json(run_dir / "infrastructure/remote_fleet_servers.json", {})
            table = {"gilbreth": {"ssh": ["ssh", "gilbreth"], "remote_run": "/scratch/run", "code_root": "/scratch/code",
                                  "python": "/scratch/env/bin/python", "unit_seconds": 1800,
                                  "pools": {"standby": {"gpu_class": "a100", "time_limit": "04:00:00"}}}}
            copied = []

            def copy(run, servers, server, arguments):
                """Stand in for rsync: note every source folder that is sent and whether it holds the number file."""
                sources = [Path(argument.rstrip("/")) for argument in arguments if argument.startswith(str(run_dir))]
                copied.extend((str(source), (source / unit_queue.COPY_NUMBER_FILE).exists()) for source in sources)

            answer = f"{fleet.MARK}repository 0123abc 0\n{fleet.MARK}end\n"
            remote.adapt(run_dir)
            with mock.patch.object(fleet, "rsync", copy), mock.patch.object(fleet, "remote", return_value=answer):
                remote.prepare(run_dir, table, "gilbreth")
            self.assertIn((str(platform), True), copied)
            entry = common.read_json(run_dir / "infrastructure/remote_fleet_servers.json")["gilbreth"]
            self.assertEqual((entry["prepared_commit"], entry["prepared_copy_number"], entry["code"]),
                             ("abc", 4, "/scratch/code/source_abc/10_jax_exploration_platform"))
            # the worker on the server reads the number from the folder its job script names as CODE
            self.assertIn("CODE=/scratch/code/source_abc/10_jax_exploration_platform",
                          (run_dir / "infrastructure/remote_fleet/gilbreth/worker_job_standby.sh").read_text())


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

    def test_every_new_copy_gets_the_next_copy_number(self):
        """The first numbered copy is 1 and each new commit's copy is one higher; the number is in the copy and in source.json."""
        first = freeze_code.freeze(self.run_dir, self.repository)
        source = common.read_json(self.run_dir / "source.json")
        platform = Path(source["snapshot"]) / "10_jax_exploration_platform"
        self.assertEqual((source["copy_number"], source["highest_copy_number"]), (1, 1))
        # the number file is in the platform folder, the folder a worker runs from and a server receives
        self.assertEqual(((platform / unit_queue.COPY_NUMBER_FILE).read_text(), unit_queue.copy_number(platform)), ("1\n", 1))
        # freezing the same commit again changes nothing
        freeze_code.freeze(self.run_dir, self.repository)
        self.assertEqual(common.read_json(self.run_dir / "source.json")["copy_number"], 1)
        (self.repository / "10_jax_exploration_platform/scripts/tool.py").write_text("print('second')\n")
        self.commit("second")
        second = freeze_code.freeze(self.run_dir, self.repository)
        source = common.read_json(self.run_dir / "source.json")
        self.assertEqual((source["copy_number"], source["highest_copy_number"]), (2, 2))
        self.assertEqual(unit_queue.copy_number(self.run_dir / "code" / f"source_{second}" / "10_jax_exploration_platform"), 2)
        self.assertEqual(unit_queue.copy_number(self.run_dir / "code" / f"source_{first}" / "10_jax_exploration_platform"), 1)
        # back at the first commit the existing copy keeps its number, and the next new copy is still one above the highest
        self.git("checkout", "--quiet", first)
        freeze_code.freeze(self.run_dir, self.repository)
        source = common.read_json(self.run_dir / "source.json")
        self.assertEqual((source["commit"], source["copy_number"], source["highest_copy_number"]), (first, 1, 2))
        (self.repository / "10_jax_exploration_platform/scripts/tool.py").write_text("print('third')\n")
        self.commit("third")
        freeze_code.freeze(self.run_dir, self.repository)
        self.assertEqual(common.read_json(self.run_dir / "source.json")["copy_number"], 3)

    def test_a_folder_frozen_before_the_numbers_counts_as_zero(self):
        """An earlier source.json and copy without a number count as 0, so the next new copy is 1."""
        commit = freeze_code.freeze(self.run_dir, self.repository)
        # make the folder look as one frozen before copy numbers existed
        platform = self.run_dir / "code" / f"source_{commit}" / "10_jax_exploration_platform"
        (platform / unit_queue.COPY_NUMBER_FILE).unlink()
        source = common.read_json(self.run_dir / "source.json")
        common.write_json(self.run_dir / "source.json", {key: value for key, value in source.items()
                                                         if key not in ("copy_number", "highest_copy_number")})
        # the same commit again: the old copy is kept and counts as 0
        freeze_code.freeze(self.run_dir, self.repository)
        source = common.read_json(self.run_dir / "source.json")
        self.assertEqual((source["copy_number"], source["highest_copy_number"], unit_queue.copy_number(platform)), (0, 0, 0))
        # a new commit: copy number 1
        (self.repository / "10_jax_exploration_platform/scripts/tool.py").write_text("print('with numbers')\n")
        self.commit("with numbers")
        freeze_code.freeze(self.run_dir, self.repository)
        self.assertEqual(common.read_json(self.run_dir / "source.json")["copy_number"], 1)


if __name__ == "__main__":
    unittest.main()


class TableReadUnderTheLock(unittest.TestCase):
    """The server table is read only after the fleet lock is taken, so a waiting step cannot write back an old one."""

    def test_a_step_sees_the_change_made_while_it_waited_on_the_lock(self):
        """A writer that holds the lock adds an entry; the step that waited for the lock works on the new table."""
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder)
            (run_dir / "infrastructure").mkdir()
            (run_dir / "slurm").mkdir()
            path = fleet.table_path(run_dir)
            common.write_json(path, {"gilbreth": {"ready": True}})
            real_lock, seen = fleet.fleet_lock, []

            def lock_held_by_another_writer(where):
                """Before the lock is handed over, the other writer adds a server (as a waiting step would see it)."""
                common.write_json(path, {"gilbreth": {"ready": True}, "mit": {"ready": False}})
                return real_lock(where)

            with mock.patch.object(fleet, "fleet_lock", lock_held_by_another_writer), \
                    mock.patch.object(remote, "hand_over_tick", lambda run, table, dry: seen.append(sorted(table))), \
                    mock.patch.object(sys, "argv", ["remote_fleet.py", "--run-dir", str(run_dir), "hand-over-tick"]):
                remote.main()
            self.assertEqual(seen, [["gilbreth", "mit"]])
