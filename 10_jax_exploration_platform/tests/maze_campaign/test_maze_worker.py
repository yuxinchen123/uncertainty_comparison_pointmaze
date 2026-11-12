"""The maze campaign's worker on a temporary folder, with a stand-in for run_training.py (no trainer is started)."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from maze_test_support import CARD, FAKE_TRAINER, small_run
from grid_campaign import common, load, own
import unit_queue

worker = own("worker")


class WorkerCase(unittest.TestCase):
    """A temporary run folder with two small units, a stand-in trainer, and no Slurm job around the worker."""

    def setUp(self):
        """Make the temporary folder the tests of this class work in."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.run_dir = Path(folder.name) / "run"
        self.names = small_run(self.run_dir, units=2)
        self.trainer = Path(folder.name) / "fake_trainer.py"
        self.trainer.write_text(FAKE_TRAINER)
        self.observed = Path(folder.name) / "observed"

    def work(self, mode, *options, card=CARD, restart_count=None):
        """Run one worker to its end with the stand-in trainer in the given mode; returns its exit code."""
        environment = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
        environment["FAKE_TRAINER_MODE"] = mode
        if restart_count is not None:
            environment["SLURM_RESTART_COUNT"] = str(restart_count)
        with mock.patch.dict(os.environ, environment, clear=True), contextlib.redirect_stdout(io.StringIO()):
            return worker.main(["--run-dir", str(self.run_dir), "--max-seconds", "7200",
                                "--observed-dir", str(self.observed), *options],
                               card=card, trainer=self.trainer, sample_memory=None)

    def record(self, state, name):
        """Read a unit's queue record from one queue folder."""
        return common.read_json(unit_queue.unit_path(self.run_dir, state, name))


class Outcomes(WorkerCase):
    """Where a unit goes after its process ends."""

    def test_complete_units_move_to_done_and_the_worker_takes_the_next(self):
        """Both units are done by one worker; each records its job and card and adds an observed line."""
        self.assertEqual(self.work("complete"), 0)
        self.assertEqual(unit_queue.counts(self.run_dir), {"done": 2, "failed": 0, "pending": 0, "running": 0})
        record = self.record("done", self.names[0])
        attempt = record["history"][0]
        self.assertEqual((attempt["job_id"], attempt["hostname"], attempt["card"], attempt["outcome"],
                          attempt["source_commit"]), ("local", "testnode", "NVIDIA RTX A4500", "complete", "working tree"))
        self.assertTrue(Path(record["result"]["log"]).exists())
        workload = common.read_json(self.run_dir / record["specification"])["workload"]
        lines = [json.loads(line) for line in (self.observed / f"{workload}.jsonl").read_text().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual({key: lines[0][key] for key in ("workload", "device_name", "copies", "seconds_per_iteration",
                                                         "host", "unit_id")},
                         {"workload": workload, "device_name": "NVIDIA RTX A4500", "copies": 32,
                          "seconds_per_iteration": 0.05, "host": "testnode", "unit_id": self.names[0]})
        self.assertTrue((self.run_dir / "slurm/jobs/local_testnode_card-0/hardware.json").exists())

    def test_the_command_has_checkpoints_a_time_limit_and_the_units_configuration(self):
        """run_training.py gets the unit's own file, a checkpoint every 20 minutes and what is left of the job."""
        self.work("complete")
        command = json.loads((self.run_dir / f"command_{self.names[0]}.json").read_text())
        specification = common.read_json(self.run_dir / "specifications" / f"{self.names[0]}.json")
        value = lambda option: command[command.index(option) + 1]
        self.assertEqual((value("--run-dir"), value("--unit-id"), value("--checkpoint-seconds")),
                         (str(self.run_dir), self.names[0], "1200"))
        self.assertEqual(Path(value("--configuration-file")).read_text(), specification["configuration_toml"])
        self.assertEqual(value("--reward-setting"), "arrival_plus_one_and_step_cost")
        self.assertIn("--no-aggregate", command)
        # 7,200 s for the worker, less the 600 s kept for saving
        self.assertTrue(6500 < int(value("--time-limit-seconds")) <= 6600)

    def test_a_stopped_unit_returns_to_pending_and_a_later_worker_completes_it(self):
        """A stop with the state saved is no failure; the worker ends, and the next one continues the unit."""
        self.assertEqual(self.work("stop"), 0)
        self.assertEqual(unit_queue.counts(self.run_dir), {"done": 0, "failed": 0, "pending": 2, "running": 0})
        record = self.record("pending", self.names[0])
        self.assertEqual((record["attempts"], record["failures"], record["history"][0]["outcome"],
                          record["history"][0]["exit_code"]), (1, 0, "stopped_with_state_saved", 1))
        self.assertEqual(self.work("complete"), 0)
        record = self.record("done", self.names[0])
        self.assertEqual((record["attempts"], [attempt["outcome"] for attempt in record["history"]]),
                         (2, ["stopped_with_state_saved", "complete"]))

    def test_a_unit_that_fails_three_times_goes_to_failed_with_its_log(self):
        """Each failing worker ends after its one failure; the third failure moves the unit to failed."""
        for expected in ("pending", "pending", "failed"):
            self.assertEqual(self.work("fail"), 1)
            self.assertTrue(unit_queue.unit_path(self.run_dir, expected, self.names[0]).exists())
        record = self.record("failed", self.names[0])
        self.assertEqual(record["failures"], 3)
        self.assertIn("the stand-in trainer was told to fail", Path(record["failure_log"]).read_text())
        # the other unit was never touched by the failing workers
        self.assertEqual(self.record("pending", self.names[1])["attempts"], 0)

    def test_out_of_card_memory_leaves_the_unit_to_larger_cards(self):
        """The unit returns to pending for cards with more memory; the same card then takes the other unit only."""
        self.assertEqual(self.work("out_of_memory"), 1)
        record = self.record("pending", self.names[0])
        self.assertEqual((record["minimum_device_memory_mb"], record["failures"]), (CARD["memory_mb"] + 1, 0))
        self.assertEqual(self.work("complete"), 0)
        self.assertEqual(unit_queue.counts(self.run_dir), {"done": 1, "failed": 0, "pending": 1, "running": 0})
        self.assertEqual(self.work("complete", card=dict(CARD, gpu="NVIDIA H100 NVL", memory_mb=95830)), 0)
        self.assertEqual(unit_queue.counts(self.run_dir)["done"], 2)


class Ends(WorkerCase):
    """When a worker takes no unit."""

    def test_stop_file_too_little_time_and_too_small_a_card(self):
        """RUN/STOP, a job with too little left, and a card that holds no pending unit all leave the queue as it is."""
        small = dict(CARD, memory_mb=4000)
        self.assertEqual(self.work("complete", card=small), 0)
        (self.run_dir / "STOP").write_text("stop\n")
        self.assertEqual(self.work("complete"), 0)
        (self.run_dir / "STOP").unlink()
        self.assertEqual(self.work("complete", "--minimum-seconds", "7000"), 0)
        self.assertEqual(unit_queue.counts(self.run_dir), {"done": 0, "failed": 0, "pending": 2, "running": 0})

    def test_a_restarted_job_releases_its_earlier_claims(self):
        """A unit claimed by an earlier start of the same job goes back to pending and is then run."""
        claimed = unit_queue.claim(self.run_dir, 20470, {"job_id": "local", "restart_count": 0})
        self.assertEqual(self.work("complete", restart_count=1), 0)
        record = self.record("done", claimed["name"])
        self.assertEqual([attempt["outcome"] for attempt in record["history"]], ["job_started_again", "complete"])


class Classify(unittest.TestCase):
    """How the end of a unit's process is named."""

    def test_classify(self):
        """The shard decides completion; a stop needs the stop sentence or a stop signal; memory has its own name."""
        stop = "unit u stopped at iteration 5 of 10 with its state saved; run the same command again"
        self.assertEqual(worker.classify(True, 1, "Traceback", False), "complete")
        self.assertEqual(worker.classify(False, 1, stop, False), "stopped")
        self.assertEqual(worker.classify(False, -15, "", True), "stopped")
        self.assertEqual(worker.classify(False, 1, "RESOURCE_EXHAUSTED: Out of memory", False), "out_of_card_memory")
        self.assertEqual(worker.classify(False, 1, "Traceback", False), "failed")
        self.assertEqual(worker.classify(False, 0, "", False), "failed")


class RemoteRecords(WorkerCase):
    """The extra records a worker writes on a remote server, in the form the grid campaign's remote fleet reads."""

    def test_claim_record_and_completed_unit_folder(self):
        """A done unit has its claim record beside it and its shard, log and summary in data/research/<unit>."""
        self.assertEqual(self.work("complete", "--remote-records"), 0)
        name = self.names[0]
        claim_text = (self.run_dir / "queue/done" / f"{name}.job").read_text()
        # the status call of the grid campaign prints each claim record on one line after its folder and unit
        fleet = load("remote_fleet")
        status = fleet.parse_status("\n".join([
            "@@fleet:queue", f"done {name}", "@@fleet:claims", f"done {name} " + claim_text.replace("\n", " "),
            "@@fleet:jobs", "@@fleet:squeue_exit 0", "@@fleet:starts",
            (self.run_dir / "slurm/job_starts.txt").read_text().strip(), "@@fleet:end"]))
        claim = status["claims"][("done", name)]
        self.assertEqual((claim["job"], claim["host"], claim["gpu"], claim["exit"]),
                         ("local", "testnode", "NVIDIA_RTX_A4500", "0"))
        self.assertEqual(fleet.result_block("mit", claim)["exit_code"], 0)
        self.assertEqual(status["starts"][0][0], "local")
        folder = self.run_dir / "data/research" / name
        summary = common.read_json(folder / "summary.json")
        self.assertEqual((summary["completed"], summary["observed"]["copies"]), (True, 32))
        self.assertTrue((folder / f"{name}.jsonl").samefile(self.run_dir / "data" / f"{name}.jsonl"))
        self.assertTrue((folder / f"{name}.log").exists())

    def test_a_unit_that_returns_keeps_no_claim_record_in_the_queue(self):
        """After a stop the claim record is in the logs, not in queue/running or queue/pending."""
        self.assertEqual(self.work("stop", "--remote-records"), 0)
        self.assertEqual(list((self.run_dir / "queue").rglob("*.job")), [])
        self.assertEqual(len(list((self.run_dir / "logs").glob(f"{self.names[0]}_local_released_*.job"))), 1)


if __name__ == "__main__":
    unittest.main()
