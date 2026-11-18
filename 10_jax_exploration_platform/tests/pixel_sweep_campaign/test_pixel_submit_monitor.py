"""The submission plan of the pixel sweep (no job is submitted), its job scripts, and one monitoring pass."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from pixel_test_support import (CARD_ONE_COPY, build_units, claimant, common, metric_lines, own, pixel_queue,
                                queue_units, small_description, unit_queue, write_run_records)
from campaign_imports import grid, maze

submit = own("submit")
monitor = own("monitor")
maze_submit = maze("submit")
PROFILES = pixel_queue.read_profiles()

CLASSES = [
    {"gres_type": "nvidia_h100_nvl", "partition": "gpu", "nodes": ["serval06"], "gpus_per_node": 2, "gpu_mem_mb": 95830,
     "capability_rank": 1, "display_name": "H100 NVL"},
    {"gres_type": "a100", "partition": "gpu", "nodes": ["cheetah04"], "gpus_per_node": 4, "gpu_mem_mb": 81101,
     "capability_rank": 2, "display_name": "A100"},
    {"gres_type": "nvidia_geforce_rtx_5080", "partition": "gpu", "nodes": ["nekomata01", "nekomata02"], "gpus_per_node": 2,
     "gpu_mem_mb": 16303, "capability_rank": 3, "display_name": "RTX 5080"},
    {"gres_type": "nvidia_rtx_a4500", "partition": "gpu", "nodes": ["jaguar03"], "gpus_per_node": 8, "gpu_mem_mb": 20470,
     "capability_rank": 6, "display_name": "RTX A4500"},
    {"gres_type": "nvidia_a16", "partition": "gpu", "nodes": ["jaguar02"], "gpus_per_node": 8, "gpu_mem_mb": 15356,
     "capability_rank": 8, "display_name": "A16"},
    {"gres_type": "nvidia_geforce_gtx_1080", "partition": "gnolim", "nodes": ["ai05"], "gpus_per_node": 4,
     "gpu_mem_mb": 8192, "capability_rank": 15, "display_name": "GTX 1080"},
]


# what the maze campaign's planner asks per card and under which accounts, as it was before these tests: the
# tests below set the pixel sweep's request in that shared module, and the maze campaign's own tests, which may
# run in the same process afterwards, expect its own
MAZE_SETTINGS = {}


def setUpModule():
    """Keep the maze planner's request, accounts and time limit as they are before this file's tests."""
    MAZE_SETTINGS.update(cpus=dict(maze_submit.CPUS_PER_CARD), memory=maze_submit.MEMORY_GIB_PER_CARD,
                         hours=maze_submit.WORKER_HOURS, identities=maze_submit.IDENTITIES)


def tearDownModule():
    """Put the maze planner's request, accounts and time limit back."""
    maze_submit.CPUS_PER_CARD.update(MAZE_SETTINGS["cpus"])
    maze_submit.MEMORY_GIB_PER_CARD = MAZE_SETTINGS["memory"]
    maze_submit.WORKER_HOURS = MAZE_SETTINGS["hours"]
    maze_submit.IDENTITIES = MAZE_SETTINGS["identities"]


def state():
    """The planner's inputs: free cards on five nodes, both accounts' room, and the owner's reservation on jaguar03."""
    return {
        "classes": deepcopy(CLASSES),
        "nodes": {"serval06": {"cards": 1, "cpus": 64, "memory_gib": 500, "reservation": None},
                  "cheetah04": {"cards": 2, "cpus": 64, "memory_gib": 500, "reservation": None},
                  "nekomata01": {"cards": 2, "cpus": 22, "memory_gib": 100, "reservation": None},
                  "nekomata02": {"cards": 1, "cpus": 22, "memory_gib": 100, "reservation": "(null)"},
                  "jaguar03": {"cards": 5, "cpus": 200, "memory_gib": 900, "reservation": "sl5nw_168"},
                  "jaguar02": {"cards": 8, "cpus": 30, "memory_gib": 900, "reservation": None},
                  "ai05": {"cards": 4, "cpus": 16, "memory_gib": 64, "reservation": None}},
        "budgets": {(identity, partition): {"cards": cards, "cpus": cpus, "memory_gib": 1000}
                    for identity in maze_submit.ALL_IDENTITIES for partition, cards, cpus in (("gpu", 40, 384), ("gnolim", 20, 80))},
        "reservation_budgets": {"jaguar03": {"cards": 5, "cpus": 184, "memory_gib": 900, "name": "sl5nw_168"}},
        "held_nodes": set(),
        "type_room": {"nvidia_h100_nvl": 2, "a100": 4, "nvidia_geforce_rtx_5080": 4, "nvidia_rtx_a4500": 0,
                      "nvidia_a16": 8, "nvidia_geforce_gtx_1080": 4},
        "waiting": {},
    }


def units(singles, packs):
    """Pending queue records: some singles and some four-copy units that never started."""
    return ([{"name": f"single-{index}", "copies": 1, "execution_profile": None, "minimum_device_memory_mb": 0}
             for index in range(singles)]
            + [{"name": f"pack-{index}", "copies": 4, "execution_profile": None, "minimum_device_memory_mb": 0}
               for index in range(packs)])


def plan(pending, **changes):
    """Plan with the fixture above, the default card types, and 4 processors and 16 GiB per card."""
    maze_submit.set_request_per_card(submit.CPUS_PER_CARD, submit.MEMORY_GIB_PER_CARD)
    inputs = {"waiting_cards": {}, "state": state(), "allowed_types": submit.DEFAULT_CARD_TYPES, "max_new_cards": None,
              "max_waiting": 0}
    inputs.update(changes)
    return submit.plan_profiles(pending, PROFILES, inputs["waiting_cards"], inputs["state"], inputs["allowed_types"],
                                inputs["max_new_cards"], inputs["max_waiting"])


def cards(jobs, **wanted):
    """The cards of the planned jobs that have the given values."""
    return sum(job["cards"] for job in jobs if all(job[key] == value for key, value in wanted.items()))


class Plan(unittest.TestCase):
    """Which worker jobs one submission sends."""

    def test_singles_go_to_small_cards_and_packs_to_large_ones(self):
        """Cards free now: 3 RTX 5080 and 5 reserved RTX A4500 for singles, 1 H100 and 2 A100 for packs."""
        jobs = plan(units(40, 10))
        self.assertEqual(cards(jobs, execution_profile="one copy"), 8)
        self.assertEqual(cards(jobs, execution_profile="four copies"), 3)
        self.assertEqual({job["gpu_type"] for job in jobs if job["execution_profile"] == "one copy"},
                         {"nvidia_geforce_rtx_5080", "nvidia_rtx_a4500"})
        self.assertEqual({job["gpu_type"] for job in jobs if job["execution_profile"] == "four copies"},
                         {"nvidia_h100_nvl", "a100"})
        # every job is pinned to a node with free cards, asks 4 processors and 16 GiB per card, and the
        # reservation's node comes last among the one-copy jobs
        self.assertTrue(all(job["node"] and job["cpus"] == 4 * job["cards"] and job["memory_gib"] == 16 * job["cards"]
                            for job in jobs))
        one_copy = [job for job in jobs if job["execution_profile"] == "one copy"]
        self.assertEqual((one_copy[-1]["node"], one_copy[-1]["reservation"]), ("jaguar03", "sl5nw_168"))

    def test_card_types_outside_the_allow_list_get_no_job(self):
        """The A16 has 8 free cards and enough memory for one copy, and the GTX 1080 has 4: neither is asked for."""
        jobs = plan(units(40, 0))
        self.assertEqual(cards(jobs, gpu_type="nvidia_a16") + cards(jobs, gpu_type="nvidia_geforce_gtx_1080"), 0)
        self.assertTrue(set(submit.DEFAULT_CARD_TYPES) >= {job["gpu_type"] for job in jobs})
        self.assertEqual(len(submit.DEFAULT_CARD_TYPES), 10)
        only = plan(units(40, 0), allowed_types=("nvidia_geforce_rtx_5080",))
        self.assertEqual({job["gpu_type"] for job in only}, {"nvidia_geforce_rtx_5080"})

    def test_never_more_worker_cards_than_units_without_a_worker(self):
        """Two singles and one pack get two small cards and one large card; waiting cards of this run count as workers."""
        jobs = plan(units(2, 1))
        self.assertEqual((cards(jobs, execution_profile="one copy"), cards(jobs, execution_profile="four copies")), (2, 1))
        self.assertEqual(plan(units(2, 1), waiting_cards={"one copy": 2, "four copies": 1}), [])
        self.assertEqual(cards(plan(units(40, 10), max_new_cards=4)), 4)
        # with no pack pending the large cards are left alone, however many singles wait
        self.assertEqual(cards(plan(units(40, 0)), execution_profile="four copies"), 0)

    def test_a_unit_fixed_to_one_profile_counts_for_that_profile_only(self):
        """A single that started under "one copy" needs a small card; a record of another profile needs none here."""
        pending = units(1, 0) + [{"name": "other", "copies": 1, "execution_profile": "another", "minimum_device_memory_mb": 0}]
        self.assertEqual(cards(plan(pending)), 1)

    def test_jobs_that_wait_in_the_queue_are_planned_only_when_allowed(self):
        """With --only-cards-free-now no job waits; without it the units left get one-card jobs by card type."""
        self.assertTrue(all(job["node"] for job in plan(units(40, 0), max_waiting=0)))
        waiting = [job for job in plan(units(40, 0), max_waiting=5) if job["node"] is None]
        self.assertTrue(waiting)
        self.assertTrue(all(job["cards"] == 1 and job["gpu_type"] in submit.DEFAULT_CARD_TYPES for job in waiting))

    def test_the_two_profiles_share_one_accounts_room(self):
        """With room for two cards in all under each account, both profiles together get four cards, not more."""
        tight = state()
        for budget in tight["budgets"].values():
            budget["cards"] = 2
        tight["reservation_budgets"] = {}
        jobs = plan(units(40, 10), state=tight)
        self.assertEqual(cards(jobs), 4)
        for identity in maze_submit.ALL_IDENTITIES:
            self.assertLessEqual(cards(jobs, identity=identity), 2)


class LeaveCards(unittest.TestCase):
    """Cards another study's waiting units need stay out of the plan."""

    def test_the_most_capable_free_cards_are_left_first(self):
        """Three cards to leave: the free H100 and the two free A100; the plan then has no large card."""
        inputs = state()
        left = submit.leave_free_cards(inputs, inputs["classes"], 3, submit.DEFAULT_CARD_TYPES)
        self.assertEqual(left, [("serval06", "nvidia_h100_nvl", 1), ("cheetah04", "a100", 2)])
        self.assertEqual((inputs["nodes"]["serval06"]["cards"], inputs["nodes"]["cheetah04"]["cards"]), (0, 0))
        self.assertEqual((inputs["type_room"]["nvidia_h100_nvl"], inputs["type_room"]["a100"]), (1, 2))
        jobs = plan(units(40, 10), state=inputs)
        self.assertEqual(cards(jobs, execution_profile="four copies"), 0)
        self.assertEqual(cards(jobs, execution_profile="one copy"), 8)

    def test_named_types_and_more_cards_than_are_free(self):
        """Left from the named types only, the reservation's node included; asking for more than are free leaves all."""
        inputs = state()
        left = submit.leave_free_cards(inputs, inputs["classes"], 99, ("nvidia_geforce_rtx_5080", "nvidia_rtx_a4500"))
        self.assertEqual(left, [("nekomata01", "nvidia_geforce_rtx_5080", 2), ("nekomata02", "nvidia_geforce_rtx_5080", 1),
                                ("jaguar03", "nvidia_rtx_a4500", 5)])
        self.assertEqual(inputs["reservation_budgets"]["jaguar03"]["cards"], 0)
        self.assertEqual(cards(plan(units(40, 0), state=inputs)), 0)
        untouched = state()
        self.assertEqual(submit.leave_free_cards(untouched, untouched["classes"], 0), [])
        self.assertEqual(untouched["nodes"], state()["nodes"])


class JobScripts(unittest.TestCase):
    """What a worker job runs and asks for."""

    def test_the_worker_job_starts_one_worker_per_card_and_passes_the_stop_signal_on(self):
        """bash accepts the script; it sources the job environment and starts this folder's worker."""
        text = submit.worker_job_text(Path("/runs/pixel"), Path("/runs/pixel/code/source_abc/10_jax_exploration_platform"),
                                      submit.PIXEL_PYTHON, 2)
        self.assertEqual(subprocess.run(["bash", "-n"], input=text, text=True, capture_output=True).returncode, 0)
        self.assertIn("source /runs/pixel/slurm/worker_env.sh", text)
        self.assertIn("/.venvs/atari_pixels_jax/bin/python /runs/pixel/code/source_abc/10_jax_exploration_platform/"
                      "scripts/pixel_sweep_campaign/worker.py --run-dir /runs/pixel &", text)
        self.assertIn("trap 'kill -TERM", text)
        self.assertIn('-eq 2 ]', text)

    def test_the_job_environment_is_the_designs(self):
        """Determinism flag, asset pack, no preallocation, 85% of the card, one thread, no user packages."""
        for line in ("PYTHONNOUSERSITE=1", "OMP_NUM_THREADS=1", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                     "XLA_PYTHON_CLIENT_MEM_FRACTION=0.85", "XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true",
                     "JAXTARI_SKIP_SPRITE_UPDATE=1",
                     "XDG_DATA_HOME=/p/rlprojects/RND/reference_repos/jaxtari_pixel_study_assets_montezuma_room_geometry_from_png"):
            self.assertIn(line, submit.WORKER_ENVIRONMENT)
        self.assertEqual(subprocess.run(["bash", "-n"], input=submit.WORKER_ENVIRONMENT, text=True,
                                        capture_output=True).returncode, 0)

    def test_a_submission_names_its_account_ledger_and_is_requeued(self):
        """The command goes through the account's wrapper with the ledger of the phase and the account."""
        job = {"identity": "uva-yuxinchen", "partition": "gpu", "node": "nekomata01", "gpu_type": "nvidia_geforce_rtx_5080",
               "cards": 2, "cpus": 8, "memory_gib": 32, "reservation": None, "kind": "worker",
               "execution_profile": "one copy"}
        output = io.StringIO()
        with mock.patch.object(grid("submit"), "allowed_seconds", return_value=4 * 86400), contextlib.redirect_stdout(output):
            self.assertIsNone(submit.submit_job(Path("/runs/pixel"), "grid", job, "text", dry_run=True))
        command = output.getvalue()
        for words in ("submit_as.sh uva-yuxinchen --idfile /runs/pixel/slurm/submitted_jobids_grid_yuxinchen.txt",
                      "--job-name pixel-worker_grid_", "--cpus-per-task 8 --mem 32G --time 96:00:00", "--requeue",
                      "--gres gpu:nvidia_geforce_rtx_5080:2 --nodelist nekomata01"):
            self.assertIn(words, command)

    def test_the_command_line_takes_the_plan_options(self):
        """--only-cards-free-now, --leave-cards and the card types reach the plan; the accounts are narrowed and restored."""
        calls = []
        with mock.patch.object(submit, "top_up", side_effect=lambda *a, **k: calls.append((a, k)) or []), \
                contextlib.redirect_stdout(io.StringIO()):
            submit.main(["--run-dir", "/runs/pixel", "--phase", "grid", "--only-cards-free-now", "--leave-cards", "6",
                         "--gpu-types", "nvidia_rtx_a4500,a40", "--accounts", "uva-sl5nw", "--dry-run"])
        self.addCleanup(maze_submit.set_identities, list(maze_submit.ALL_IDENTITIES))
        arguments, keywords = calls[0]
        self.assertEqual(arguments[1], "grid")
        self.assertEqual((keywords["only_free_now"], keywords["leave_cards"], keywords["gpu_types"], keywords["dry_run"]),
                         (True, 6, ["nvidia_rtx_a4500", "a40"], True))
        self.assertEqual(maze_submit.IDENTITIES, ("uva-sl5nw",))


class TopUp(unittest.TestCase):
    """A whole submission with the scheduler's state given and nothing sent."""

    def test_a_dry_run_plans_within_the_pending_units_and_leaves_cards(self):
        """Six singles pending, two cards to leave: jobs for six cards at most, on the cards that stay."""
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / "run"
            queue_units(run_dir, build_units.study_units(small_description())[:6])
            sent = []
            with mock.patch.object(maze_submit, "scheduler_state", return_value=state()), \
                    mock.patch.object(maze_submit, "frozen_code", return_value=(Path("/code/platform"), "abc")), \
                    mock.patch.object(common, "own_live_jobs", return_value={}), \
                    mock.patch.object(submit, "submit_job", side_effect=lambda run, phase, job, text, dry: sent.append(job)), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                submit.top_up(run_dir, "grid", dry_run=True, only_free_now=True, leave_cards=2,
                              leave_card_types=["nvidia_geforce_rtx_5080"])
            self.assertEqual(sum(job["cards"] for job in sent), 6)
            self.assertEqual(sum(job["cards"] for job in sent if job["gpu_type"] == "nvidia_geforce_rtx_5080"), 1)
            self.assertIn("left free for another study (2 of 2 cards asked): 2 x nvidia_geforce_rtx_5080 on nekomata01",
                          output.getvalue())
            self.assertIn("profile 'one copy': 6 pending units", output.getvalue())
            self.assertFalse((run_dir / "slurm/worker_env.sh").exists())

    def test_nothing_is_submitted_for_a_stopped_run(self):
        """RUN/STOP ends a top-up before the scheduler is read."""
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "STOP").write_text("stop\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(submit.top_up(folder, "grid"), [])


class MonitorPass(unittest.TestCase):
    """One monitoring pass on a small run folder; no scheduler is asked and no job is sent."""

    def setUp(self):
        """Queue the small study; one unit done, one training in a job that is not Slurm's, six pending."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.run_dir = Path(folder.name) / "run"
        self.names = queue_units(self.run_dir, build_units.study_units(small_description()))
        done = pixel_queue.claim(self.run_dir, 20470, claimant(CARD_ONE_COPY), "one copy", PROFILES["one copy"])
        write_run_records(pixel_queue.run_folder(self.run_dir, done["name"]), metric_lines(244), 244, completed=True)
        unit_queue.finish(self.run_dir, done, {"wall_seconds": 300.0, "seconds_per_update": 1.4})
        self.running = pixel_queue.claim(self.run_dir, 20470, claimant(CARD_ONE_COPY), "one copy", PROFILES["one copy"])
        write_run_records(pixel_queue.run_folder(self.run_dir, self.running["name"]), metric_lines(130), 244)
        resources = Path(folder.name) / "resources.json"
        resources.write_text(json.dumps({"classes": CLASSES}))
        patch = mock.patch.object(grid("submit"), "RESOURCE_FILE", resources)
        patch.start()
        self.addCleanup(patch.stop)

    def one_pass(self, **options):
        """Run one pass with no live Slurm job; returns (complete, cadence, printed text)."""
        output = io.StringIO()
        with mock.patch.object(common, "own_live_jobs", return_value={}), contextlib.redirect_stdout(output):
            complete, cadence = monitor.one_pass(self.run_dir, **options)
        return complete, cadence, output.getvalue()

    def test_a_pass_writes_the_status_the_report_and_the_early_signals(self):
        """Counts, the training unit's progress, the pending units' warning, and early_signals.md."""
        complete, cadence, printed = self.one_pass()
        self.assertEqual((complete, cadence), (False, 1200))
        status = common.read_json(self.run_dir / "status.json")
        self.assertEqual(status["counts"], {"done": 1, "failed": 0, "pending": 6, "running": 1})
        self.assertIn("update 130 of 244 (53%), 1 copy, 1.40 s per update", " ".join(printed.split()))
        self.assertIn("grid, fast last layer", printed)
        self.assertIn("6 pending units of the profile 'one copy' have no worker job", "\n".join(status["warnings"]))
        self.assertTrue(max(len(line) for line in (self.run_dir / "status.txt").read_text().splitlines()) <= 100)
        page = (self.run_dir / "early_signals.md").read_text()
        self.assertIn("## Standing checks", page)
        report = next((self.run_dir / "20_mins_monitoring/outputs").glob("*_monitoring.md")).read_text()
        self.assertIn("| grid<br>RND | 2 | 1 | 1 | 0 | 0 | 4 | 374 of<br>976 |", report)
        self.assertIn("Times are Pacific", report)

    def test_a_unit_without_progress_is_named_after_forty_minutes(self):
        """The same update count under the same job for 40 minutes makes a running unit stale; a new job starts the count again."""
        name = self.running["name"]
        rows = monitor.unit_rows(self.run_dir, {}, 1000.0)
        row = next(row for row in rows if row["name"] == name)
        self.assertEqual((row["update"], row["stale"], row["unchanged_since"]), (130, False, 1000.0))
        previous = {name: {"update": 130, "unchanged_since": 1000.0, "job_id": "local"}}
        later = next(row for row in monitor.unit_rows(self.run_dir, previous, 1000.0 + 2400) if row["name"] == name)
        self.assertTrue(later["stale"])
        other_job = {name: {"update": 130, "unchanged_since": 1000.0, "job_id": "77"}}
        again = next(row for row in monitor.unit_rows(self.run_dir, other_job, 1000.0 + 2400) if row["name"] == name)
        self.assertFalse(again["stale"])

    def test_a_report_that_cannot_be_built_is_a_warning_and_the_pass_goes_on(self):
        """A record file the report cannot read does not end the monitoring; the pass names the problem."""
        (pixel_queue.run_folder(self.run_dir, self.running["name"]) / "game_first_times.json").write_text("{not json")
        _, _, printed = self.one_pass()
        self.assertIn("early_signals.md could not be rebuilt at this pass", printed)
        self.assertTrue((self.run_dir / "status.json").exists())

    def test_the_sweep_is_complete_when_every_unit_is_done_or_failed(self):
        """With --complete-when-drained the sentinel is written only when nothing is pending, running or held."""
        self.assertFalse(self.one_pass(complete_when_drained=True, report=False)[0])
        for state_name in ("running", "pending"):
            for unit in unit_queue.read_units(self.run_dir, state_name):
                unit_queue.move(self.run_dir, unit, state_name, "done")
        self.assertTrue(self.one_pass(complete_when_drained=True, report=False)[0])
        self.assertTrue((self.run_dir / "SWEEP_COMPLETE").exists())


if __name__ == "__main__":
    unittest.main()
