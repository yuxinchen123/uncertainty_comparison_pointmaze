"""The maze campaign's monitor on a temporary folder: counts, progress, recovery, warnings, the status files, the cadence."""
import contextlib
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from maze_test_support import small_run, write_shard
from grid_campaign import common, own
import unit_queue

monitor = own("monitor")
CLAIMANT = {"job_id": "local", "restart_count": 0, "account": "sl5nw", "hostname": "jaguar03",
            "card": "NVIDIA RTX A4500", "card_memory_mb": 20470, "source_commit": "abc"}


def quiet_pass(run_dir, **options):
    """Run one monitoring pass without its printed table."""
    with contextlib.redirect_stdout(io.StringIO()):
        return monitor.one_pass(run_dir, **options)


class MonitorCase(unittest.TestCase):
    """Four units: one done, one failed after three failures, one running with one window, one pending with one."""

    def setUp(self):
        """Make the temporary folder the tests of this class work in."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.run_dir = Path(folder.name)
        self.done, self.failed, self.running, self.pending = small_run(self.run_dir, units=4)
        unit_queue.finish(self.run_dir, unit_queue.claim(self.run_dir, 20470, CLAIMANT), {"wall_seconds": 600.0})
        for _ in range(3):
            unit_queue.give_back(self.run_dir, unit_queue.claim(self.run_dir, 20470, dict(CLAIMANT, hostname="lynx01")),
                                 "failed", counts_as_failure=True, log="/logs/failed.log")
        unit_queue.claim(self.run_dir, 20470, CLAIMANT)
        write_shard(self.run_dir, self.running, windows=1)
        write_shard(self.run_dir, self.pending, windows=1)

    def rows(self, previous=None):
        """The monitor's per-unit rows now."""
        return monitor.unit_rows(self.run_dir, previous or {}, time.time())


class Counts(MonitorCase):
    """Counts by queue folder, and progress from the shards."""

    def test_rows_and_sweep_counts(self):
        """Each unit is in its folder; windows are read from the shards, and a done unit has all of its windows."""
        rows = {row["name"]: row for row in self.rows()}
        self.assertEqual({name: row["state"] for name, row in rows.items()},
                         {self.done: "done", self.failed: "failed", self.running: "running", self.pending: "pending"})
        self.assertEqual([rows[name]["windows_written"] for name in (self.done, self.failed, self.running, self.pending)],
                         [2, 0, 1, 1])
        sweeps = monitor.sweep_rows(list(rows.values()))
        self.assertEqual(list(sweeps.values()), [{"pending": 1, "running": 1, "done": 1, "failed": 1, "elsewhere": 0,
                                                  "windows_written": 4, "windows_expected": 8}])
        self.assertEqual(list(sweeps), [("development-first-pass", "pointmaze-large",
                                         "arrival_plus_one_and_step_cost", "cleanrl-ppo-rnd")])

    def test_units_held_by_a_remote_server_are_counted_elsewhere(self):
        """A unit in queue/remote_<server> belongs to its sweep and to no UVA state."""
        (self.run_dir / "queue/remote_mit").mkdir()
        unit_queue.unit_path(self.run_dir, "pending", self.pending).rename(
            self.run_dir / "queue/remote_mit" / f"{self.pending}.json")
        sweep = list(monitor.sweep_rows(self.rows()).values())[0]
        self.assertEqual((sweep["pending"], sweep["elsewhere"]), (0, 1))

    def test_node_rows(self):
        """Per node: the jobs, the units running now, the done units with their minutes, and the failed attempts."""
        nodes = monitor.node_rows(self.rows())
        self.assertEqual((nodes["jaguar03"]["running"], nodes["jaguar03"]["done"], nodes["jaguar03"]["seconds"],
                          nodes["jaguar03"]["submitters"]), (1, 1, [600.0], {"sl5nw"}))
        self.assertEqual((nodes["lynx01"]["failed"], nodes["lynx01"]["done"]), (3, 0))


class Warnings(MonitorCase):
    """What a pass tells the reader to look at."""

    def test_failed_units_and_units_without_a_worker(self):
        """The failed unit is named with its log; the pending unit has no worker job."""
        warnings = monitor.warnings_of(self.run_dir, self.rows(), {}, [])
        self.assertTrue(any(self.failed in text and "/logs/failed.log" in text for text in warnings))
        self.assertTrue(any("1 pending units have no worker job" in text for text in warnings))
        # one waiting worker card covers the one pending unit; the running unit holds a card of job 5
        submissions = [{"job_id": "5", "kind": "worker", "cards": 1, "node": "jaguar03", "submitted_at": common.now()},
                       {"job_id": "6", "kind": "worker", "cards": 1, "node": None, "submitted_at": common.now()}]
        live = {"5": ["5", "sl5nw", "gpu", "RUNNING", "n", "jaguar03", "gpu:1"],
                "6": ["6", "sl5nw", "gpu", "PENDING", "n", "(Priority)", "gpu:1"]}
        warnings = monitor.warnings_of(self.run_dir, self.rows(), live, submissions)
        self.assertFalse(any("no worker job" in text or "pinned" in text for text in warnings))

    def test_a_pinned_job_that_still_waits_is_named(self):
        """A job pinned to a node and still waiting after more than one pass is named with its id."""
        submissions = [{"job_id": "7", "kind": "worker", "cards": 1, "node": "nekomata01",
                        "submitted_at": "2026-11-11T20:00:00-08:00"}]
        live = {"7": ["7", "sl5nw", "gpu", "PENDING", "n", "(Priority)", "gpu:1"]}
        warnings = monitor.warnings_of(self.run_dir, self.rows(), live, submissions)
        self.assertTrue(any("job 7 is pinned to nekomata01 and still waits" in text and "--job-id 7" in text
                            for text in warnings))

    def test_a_running_unit_whose_shard_does_not_grow_is_named(self):
        """Same size and same job as 50 minutes ago: no progress; a new job starts the count again."""
        size = (self.run_dir / "data" / f"{self.running}.jsonl").stat().st_size
        before = {self.running: {"bytes": size, "unchanged_since": time.time() - 3000, "job_id": "local"}}
        rows = {row["name"]: row for row in self.rows(before)}
        self.assertTrue(rows[self.running]["stale"])
        self.assertTrue(any("no progress for 40 minutes" in text and self.running in text
                            for text in monitor.warnings_of(self.run_dir, list(rows.values()), {}, [])))
        before[self.running]["job_id"] = "an earlier job"
        self.assertFalse({row["name"]: row for row in self.rows(before)}[self.running]["stale"])

    def test_collaborator_problem_reports_are_listed(self):
        """An open problem report of a collaborator appears in the warnings."""
        folder = self.run_dir / "for_collaborator/problems/open"
        folder.mkdir(parents=True)
        (folder / "2026-11-12_worker-cannot-read-queue.md").write_text("x")
        self.assertTrue(any("worker-cannot-read-queue" in text
                            for text in monitor.warnings_of(self.run_dir, self.rows(), {}, [])))


class Passes(MonitorCase):
    """Whole passes: recovery, the files written, the end of the sweep, the cadence."""

    def test_one_pass_writes_the_status_files(self):
        """status.json, a table of at most 100 characters per line, and one Markdown report."""
        complete, cadence = quiet_pass(self.run_dir)
        self.assertEqual((complete, cadence), (False, 1200))
        status = common.read_json(self.run_dir / "status.json")
        self.assertEqual(status["counts"], {"done": 1, "failed": 1, "pending": 1, "running": 1})
        self.assertEqual({unit["name"]: (unit["windows_written"], unit["windows_expected"]) for unit in status["units"]},
                         {self.done: (2, 2), self.failed: (0, 2), self.running: (1, 2), self.pending: (1, 2)})
        lines = (self.run_dir / "status.txt").read_text().splitlines()
        self.assertLessEqual(max(len(line) for line in lines), 100)
        for heading in ("TRAINING", "WAITING", "FINISHED", "PLANNED"):
            self.assertIn(heading, lines)
        self.assertTrue(any("window 1 of 2, 16 policies" in line for line in lines))
        self.assertTrue(lines[-1].startswith("Updated ") and " PT; the monitor passes every 20 minutes" in lines[-1])
        report = list((self.run_dir / "20_mins_monitoring/outputs").glob("*_monitoring.md"))[0].read_text()
        self.assertIn("| 1 | 1 | 1 | 1 | 0 | 4 of 8 |", report)
        self.assertIn("| jaguar03 | 1 | 1 | 1 | 0 | 10.0 | sl5nw |", report)

    def test_a_unit_whose_job_is_gone_returns_to_pending(self):
        """A claim under a Slurm job id that the scheduler no longer lists is recovered by the pass."""
        path = unit_queue.unit_path(self.run_dir, "running", self.running)
        common.write_json(path, dict(common.read_json(path), job_id="6873000"))
        quiet_pass(self.run_dir)
        status = common.read_json(self.run_dir / "status.json")
        self.assertEqual(status["recovered_units"], {self.running: "pending"})
        self.assertEqual(status["counts"], {"done": 1, "failed": 1, "pending": 2, "running": 0})

    def test_the_sweep_is_complete_when_the_sentinel_exists_or_the_queue_is_drained(self):
        """Without the option a drained queue is not complete; with it, or with the file, it is."""
        running = common.read_json(unit_queue.unit_path(self.run_dir, "running", self.running))
        unit_queue.finish(self.run_dir, running, {"wall_seconds": 60.0})
        unit_queue.finish(self.run_dir, unit_queue.claim(self.run_dir, 20470, CLAIMANT), {"wall_seconds": 60.0})
        self.assertFalse(quiet_pass(self.run_dir)[0])
        self.assertTrue(quiet_pass(self.run_dir, complete_when_drained=True)[0])
        self.assertTrue((self.run_dir / "SWEEP_COMPLETE").exists())
        self.assertTrue(quiet_pass(self.run_dir)[0])

    def test_the_cadence_is_hourly_after_three_unchanged_passes_over_an_hour(self):
        """Nothing to warn about and nothing changed for over an hour: 60 minutes; a change: 20 minutes again."""
        run_dir = Path(tempfile.mkdtemp(dir=self.run_dir))
        name, = small_run(run_dir, units=1)
        unit_queue.finish(run_dir, unit_queue.claim(run_dir, 20470, CLAIMANT), {"wall_seconds": 60.0})
        cadences = []
        for moment in (1000.0, 2800.0, 4700.0):
            with mock.patch("time.time", return_value=moment):
                cadences.append(quiet_pass(run_dir)[1])
        self.assertEqual(cadences, [1200, 1200, 3600])
        # a new pending unit changes the counts and has no worker: back to 20 minutes
        unit_queue.unit_path(run_dir, "done", name).rename(unit_queue.unit_path(run_dir, "pending", name))
        with mock.patch("time.time", return_value=8300.0):
            self.assertEqual(quiet_pass(run_dir)[1], 1200)

    def test_a_folder_without_a_queue_is_refused(self):
        """A pass needs units."""
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(SystemExit):
            monitor.one_pass(folder)


if __name__ == "__main__":
    unittest.main()
