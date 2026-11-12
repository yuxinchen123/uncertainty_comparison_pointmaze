"""The queue of the maze campaign on a temporary folder: claims, returns, failures, recovery, shard progress."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from maze_test_support import small_run, write_shard
from grid_campaign import common
import unit_queue

CLAIMANT = {"job_id": "71", "restart_count": 0, "account": "sl5nw", "hostname": "jaguar03",
            "card": "NVIDIA RTX A4500", "card_memory_mb": 20470, "source_commit": "abc"}


class QueueCase(unittest.TestCase):
    """A temporary run folder with three small units."""

    def setUp(self):
        """Make the temporary folder the tests of this class work in."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.run_dir = Path(folder.name)
        self.names = small_run(self.run_dir, units=3)

    def state_of(self, name):
        """The queue folder a unit is in; a unit must be in exactly one."""
        found = [state for state in unit_queue.counts(self.run_dir)
                 if unit_queue.unit_path(self.run_dir, state, name).exists()]
        self.assertEqual(len(found), 1)
        return found[0]


class Claims(QueueCase):
    """Claiming by rename."""

    def test_claim_takes_units_in_order_and_once(self):
        """The first claim gets the first unit, the next claim the next; the record moves to running."""
        first = unit_queue.claim(self.run_dir, 20470, CLAIMANT)
        second = unit_queue.claim(self.run_dir, 20470, dict(CLAIMANT, job_id="72"))
        self.assertEqual([first["name"], second["name"]], self.names[:2])
        self.assertEqual((self.state_of(first["name"]), first["attempts"], first["job_id"]), ("running", 1, "71"))
        self.assertEqual(common.read_json(unit_queue.unit_path(self.run_dir, "running", second["name"]))["job_id"], "72")
        self.assertEqual(unit_queue.counts(self.run_dir), {"done": 0, "failed": 0, "pending": 1, "running": 2})

    def test_a_card_takes_only_units_it_can_hold(self):
        """A unit that needs more card memory is passed over; nothing fitting gives None."""
        path = unit_queue.unit_path(self.run_dir, "pending", self.names[0])
        common.write_json(path, dict(common.read_json(path), minimum_device_memory_mb=40000))
        self.assertEqual(unit_queue.claim(self.run_dir, 20470, CLAIMANT)["name"], self.names[1])
        self.assertIsNone(unit_queue.claim(self.run_dir, 6000, CLAIMANT))
        self.assertEqual(unit_queue.claim(self.run_dir, 95830, CLAIMANT)["name"], self.names[0])

    def test_many_claimants_never_share_a_unit(self):
        """Eight claimants at once claim three units: each unit once."""
        claimed = []

        def take(index):
            """Claim until nothing is left."""
            while (unit := unit_queue.claim(self.run_dir, 20470, dict(CLAIMANT, job_id=str(index)))) is not None:
                claimed.append(unit["name"])

        threads = [threading.Thread(target=take, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(claimed), sorted(self.names))

    def test_enqueue_knows_a_unit_in_any_folder(self):
        """A unit held by a remote server is not queued a second time."""
        name = self.names[0]
        (self.run_dir / "queue/remote_mit").mkdir()
        unit_queue.unit_path(self.run_dir, "pending", name).rename(self.run_dir / "queue/remote_mit" / f"{name}.json")
        specification = common.read_json(self.run_dir / "specifications" / f"{name}.json")
        self.assertFalse(unit_queue.enqueue(self.run_dir, specification, {"name": name, "specification": "x"}))
        self.assertEqual(unit_queue.counts(self.run_dir)["remote_mit"], 1)


class Returns(QueueCase):
    """What happens to a claimed unit that did not complete."""

    def test_a_stopped_unit_returns_to_pending_without_a_failure(self):
        """Its attempt goes into the history, and the claim's fields leave the record."""
        unit = unit_queue.claim(self.run_dir, 20470, CLAIMANT)
        self.assertEqual(unit_queue.give_back(self.run_dir, unit, "stopped_with_state_saved", exit_code=1), "pending")
        record = common.read_json(unit_queue.unit_path(self.run_dir, "pending", unit["name"]))
        self.assertEqual((record["failures"], record["attempts"]), (0, 1))
        self.assertNotIn("job_id", record)
        self.assertEqual((record["history"][0]["job_id"], record["history"][0]["hostname"],
                          record["history"][0]["card"], record["history"][0]["outcome"]),
                         ("71", "jaguar03", "NVIDIA RTX A4500", "stopped_with_state_saved"))

    def test_three_failures_move_a_unit_to_failed_with_its_log(self):
        """Two failures return it to pending; the third moves it to failed."""
        name = self.names[0]
        for attempt, expected in ((1, "pending"), (2, "pending"), (3, "failed")):
            unit = unit_queue.claim(self.run_dir, 20470, CLAIMANT)
            self.assertEqual(unit["name"], name)
            self.assertEqual(unit_queue.next_folder(unit, True), expected)
            self.assertEqual(unit_queue.give_back(self.run_dir, unit, "failed", counts_as_failure=True,
                                                  log=f"/logs/attempt_{attempt}.log"), expected)
        record = common.read_json(unit_queue.unit_path(self.run_dir, "failed", name))
        self.assertEqual((record["failures"], record["failure_log"], len(record["history"])),
                         (3, "/logs/attempt_3.log", 3))

    def test_out_of_card_memory_leaves_the_unit_to_larger_cards(self):
        """The unit's least card memory rises above the card it failed on; that is no failure of the three."""
        unit = unit_queue.claim(self.run_dir, 20470, CLAIMANT)
        unit_queue.give_back(self.run_dir, unit, "out_of_card_memory", minimum_device_memory_mb=20471)
        record = common.read_json(unit_queue.unit_path(self.run_dir, "pending", unit["name"]))
        self.assertEqual((record["minimum_device_memory_mb"], record["failures"], record["memory_failures"]),
                         (20471, 0, 1))
        self.assertNotEqual(unit_queue.claim(self.run_dir, 20470, CLAIMANT)["name"], unit["name"])

    def test_finish_moves_to_done_with_the_result(self):
        """A complete unit keeps its result and its attempt."""
        unit = unit_queue.claim(self.run_dir, 20470, CLAIMANT)
        unit_queue.finish(self.run_dir, unit, {"wall_seconds": 12.5, "log": "/logs/x.log"})
        record = common.read_json(unit_queue.unit_path(self.run_dir, "done", unit["name"]))
        self.assertEqual((record["result"]["wall_seconds"], record["history"][0]["outcome"],
                          record["history"][0]["wall_seconds"]), (12.5, "complete", 12.5))


class Recovery(QueueCase):
    """Units of jobs that ended, and of a job that started again."""

    def test_only_units_of_jobs_that_no_longer_run_are_recovered(self):
        """A running job keeps its unit; an absent or waiting job, or a claim without a job id, loses it."""
        running = unit_queue.claim(self.run_dir, 20470, dict(CLAIMANT, job_id="71"))
        ended = unit_queue.claim(self.run_dir, 20470, dict(CLAIMANT, job_id="72"))
        requeued = unit_queue.claim(self.run_dir, 20470, dict(CLAIMANT, job_id="73"))
        live = {"71": ["71", "sl5nw", "gpu", "RUNNING", "name", "jaguar03", "gpu:1"],
                "73": ["73", "sl5nw", "gpu", "PENDING", "name", "(BeginTime)", "gpu:1"]}
        moved = unit_queue.recover_claims(self.run_dir, live)
        self.assertEqual(moved, {ended["name"]: "pending", requeued["name"]: "pending"})
        self.assertEqual(self.state_of(running["name"]), "running")
        record = common.read_json(unit_queue.unit_path(self.run_dir, "pending", ended["name"]))
        self.assertEqual((record["recoveries"], record["failures"], record["history"][0]["outcome"]),
                         (1, 0, "job_ended_without_a_result"))
        # a claim cut off before its record was written has no job id
        path = unit_queue.unit_path(self.run_dir, "running", running["name"])
        common.write_json(path, {key: value for key, value in common.read_json(path).items() if key != "job_id"})
        self.assertEqual(unit_queue.recover_claims(self.run_dir, live), {running["name"]: "pending"})

    def test_a_claim_made_outside_slurm_is_left_alone(self):
        """A job id that is not a number is not judged against the scheduler."""
        unit = unit_queue.claim(self.run_dir, 20470, dict(CLAIMANT, job_id="local"))
        self.assertEqual(unit_queue.recover_claims(self.run_dir, {}), {})
        self.assertEqual(self.state_of(unit["name"]), "running")

    def test_a_unit_recovered_too_often_goes_to_failed(self):
        """More than RECOVERY_LIMIT recoveries end in failed, with the reason."""
        name = self.names[0]
        for _ in range(unit_queue.RECOVERY_LIMIT + 1):
            self.assertEqual(unit_queue.claim(self.run_dir, 20470, CLAIMANT)["name"], name)
            moved = unit_queue.recover_claims(self.run_dir, {})
        self.assertEqual(moved, {name: "failed"})
        self.assertIn("ended without a result", common.read_json(
            unit_queue.unit_path(self.run_dir, "failed", name))["failure_reason"])

    def test_a_job_that_starts_again_releases_what_its_earlier_start_claimed(self):
        """Claims of the same job and an earlier start go back; claims of this start and of other jobs stay."""
        earlier = unit_queue.claim(self.run_dir, 20470, dict(CLAIMANT, job_id="71", restart_count=0))
        sibling = unit_queue.claim(self.run_dir, 20470, dict(CLAIMANT, job_id="71", restart_count=1))
        other = unit_queue.claim(self.run_dir, 20470, dict(CLAIMANT, job_id="99", restart_count=0))
        self.assertEqual(unit_queue.release_claims_of_job(self.run_dir, "71", 1), [earlier["name"]])
        self.assertEqual([self.state_of(unit["name"]) for unit in (earlier, sibling, other)],
                         ["pending", "running", "running"])


class ShardProgress(unittest.TestCase):
    """Completion and windows read from the end of a shard."""

    def setUp(self):
        """Make the temporary folder the tests of this class work in."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.run_dir = Path(folder.name)

    def test_no_shard_partial_and_complete(self):
        """Windows are counted from the last window record; only a unit_complete record means complete."""
        self.assertEqual(unit_queue.shard_progress(self.run_dir / "data/none.jsonl")["windows_written"], 0)
        partial = unit_queue.shard_progress(write_shard(self.run_dir, "partial", windows=3))
        self.assertEqual((partial["complete"], partial["windows_written"], partial["last_iteration"]), (False, 3, 15))
        complete = unit_queue.shard_progress(write_shard(self.run_dir, "complete", windows=4, complete=True))
        self.assertEqual((complete["complete"], complete["windows_written"], complete["completion"]["windows"]),
                         (True, 4, 4))

    def test_a_line_still_being_written_does_not_count(self):
        """Text after the last newline is ignored."""
        path = write_shard(self.run_dir, "growing", windows=2)
        with path.open("a") as handle:
            handle.write('{"record": "unit_complete", "unit_id": "grow')
        progress = unit_queue.shard_progress(path)
        self.assertEqual((progress["complete"], progress["windows_written"]), (False, 2))

    def test_only_the_end_of_a_large_shard_is_read(self):
        """With a small tail, the last window is still found; with a tail shorter than one line, it is unknown."""
        path = write_shard(self.run_dir, "large", windows=50, complete=True)
        with mock.patch.object(unit_queue, "TAIL_BYTES", 1000):
            progress = unit_queue.shard_progress(path)
            self.assertEqual((progress["complete"], progress["windows_written"]), (True, 50))
        with mock.patch.object(unit_queue, "TAIL_BYTES", 150):
            self.assertIsNone(unit_queue.shard_progress(write_shard(self.run_dir, "cut", windows=50))["windows_written"])


if __name__ == "__main__":
    unittest.main()
