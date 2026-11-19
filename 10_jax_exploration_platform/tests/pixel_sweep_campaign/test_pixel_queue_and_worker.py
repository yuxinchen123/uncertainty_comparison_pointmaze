"""The claim filter by execution profile, and the worker on a temporary folder with a stand-in trainer."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pixel_test_support import (ARCHIVE_DATA, CARD_FOUR_COPIES, CARD_ONE_COPY, CARD_SECOND_TYPE, FAKE_TRAINER, PLATFORM,
                                build_units, claimant, common, own, pixel_queue, queue_units, small_description,
                                unit_queue)

worker = own("worker")
PROFILES = pixel_queue.read_profiles()


def quiet(function, *arguments, **keywords):
    """Call a function without its printed lines."""
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*arguments, **keywords)


class Profiles(unittest.TestCase):
    """The three execution profiles and which card runs under which."""

    def test_a_card_under_60_gb_runs_one_copy_a_larger_one_four_and_an_h200_eight(self):
        """RTX 2080 Ti to A40: "one copy"; A100 80 GB, H100, GH200: "four copies"; H200: "eight copies"; GTX 1080: none."""
        named = {10854: "one copy", 16303: "one copy", 20470: "one copy", 40960: "one copy", 46068: "one copy",
                 81101: "four copies", 95830: "four copies", 97871: "four copies", 139999: "four copies",
                 140000: "eight copies", 143771: "eight copies", 8192: None}
        for memory, name in named.items():
            self.assertEqual(pixel_queue.profile_for_card(PROFILES, memory), name)
        self.assertEqual([profile["copies"] for profile in PROFILES.values()], [1, 4, 8])
        self.assertEqual({name: profile["continues"] for name, profile in PROFILES.items()},
                         {"one copy": [], "four copies": [], "eight copies": ["four copies"]})

    def test_eight_copies_are_the_four_copy_settings_with_the_reset_lane_budget_doubled(self):
        """The settings job 29603853 measured on an MIT H200: nothing else differs from the four-copy profile."""
        four, eight = PROFILES["four copies"], PROFILES["eight copies"]
        self.assertEqual(eight["learner"], four["learner"])
        self.assertEqual(eight["environment"], dict(four["environment"], reset_lane_budget=32))
        self.assertEqual(four["environment"]["reset_lane_budget"], 16)
        self.assertEqual(eight["archived_runs"][0], four["archived_runs"][0])

    @unittest.skipIf(ARCHIVE_DATA is None, "the archived baseline is not on this machine")
    def test_each_profiles_archived_runs_hold_its_copies(self):
        """The archived runs of a profile hold, together, as many copies as the profile runs."""
        for name, profile in PROFILES.items():
            seeds = [seed for run in profile["archived_runs"]
                     for seed in json.loads((ARCHIVE_DATA / run / "run/run.json").read_text())["copy_seeds"]]
            self.assertEqual(len(seeds), profile["copies"], name)
        self.assertEqual(seeds, list(range(1, 9)))

    @unittest.skipIf(ARCHIVE_DATA is None, "the archived baseline is not on this machine")
    def test_the_profiles_are_the_archived_baselines_settings(self):
        """The one-copy and four-copy profiles give the archived run.json of their card class, key by key."""
        PROFILES_OF_RUNS = {name: PROFILES[name] for name in ("one copy", "four copies")}
        for name, profile in PROFILES_OF_RUNS.items():
            archived = json.loads((ARCHIVE_DATA / profile["archived_runs"][0] / "run/run.json").read_text())
            self.assertEqual(archived["learner"]["n_copies"], profile["copies"])
            for key, value in profile["learner"].items():
                self.assertEqual(archived["learner"][key], value, f"{name}: learner {key}")
            # the trainer adds its two accepted environment settings to the command line's
            self.assertEqual(archived["environment"], {"cache_room_backgrounds": True, "render_final_two_only": True,
                                                       **profile["environment"]}, name)
        # nothing else differs between the two archived card classes than what the profiles hold
        learners = [json.loads((ARCHIVE_DATA / profile["archived_runs"][0] / "run/run.json").read_text())["learner"]
                    for profile in PROFILES_OF_RUNS.values()]
        differing = {key for key in learners[0] if learners[0][key] != learners[1][key]}
        named = set(PROFILES["one copy"]["learner"]) | set(PROFILES["four copies"]["learner"]) | {"n_copies"}
        self.assertTrue(differing <= named, differing - named)


class Claims(unittest.TestCase):
    """A worker claims only units of its card's profile; a unit keeps the profile of its first start."""

    def setUp(self):
        """Queue the eight single units of the small study, and pack RND's two... none: packs need four."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.run_dir = Path(folder.name) / "run"
        # four step sizes, so one pack can be made per seed
        self.description = small_description(methods=[
            {"method": "rnd", "pack_over": "intrinsic_weight", "grid": {"intrinsic_weight": [1, 2]}},
            {"method": "fast-last-layer", "pack_over": "step_size",
             "grid": {"sharing_group": [16], "intrinsic_weight": [1], "step_size": [0.01, 0.03, 0.1, 0.3]}}])
        self.names = queue_units(self.run_dir, build_units.study_units(self.description))

    def claim(self, card, job_id="local"):
        """Claim the next unit a worker on this card may take."""
        name = pixel_queue.profile_for_card(PROFILES, card["memory_mb"])
        return pixel_queue.claim(self.run_dir, card["memory_mb"], claimant(card, job_id), name, PROFILES[name])

    def test_a_large_card_takes_only_packs_and_a_small_card_only_singles(self):
        """With singles alone in the queue a large card finds nothing; after packing it takes the pack, in claim order."""
        self.assertIsNone(self.claim(CARD_FOUR_COPIES))
        first = self.claim(CARD_ONE_COPY)
        self.assertEqual((first["name"], first["execution_profile"]), ("grid_rnd_intrinsic-weight-1_seed-13", "one copy"))
        made = quiet(build_units.pack_pending, self.run_dir, 2)
        pack = self.claim(CARD_FOUR_COPIES)
        self.assertEqual((pack["name"], pack["copies"], pack["execution_profile"]), (made[0], 4, "four copies"))
        self.assertEqual(pack["execution_profile_settings"]["learner"], PROFILES["four copies"]["learner"])
        # the small card goes on with singles and never takes the second pack
        taken = [self.claim(CARD_ONE_COPY)["name"] for _ in range(3)]
        self.assertEqual(taken, ["grid_rnd_intrinsic-weight-2_seed-13", "grid_rnd_intrinsic-weight-1_seed-14",
                                 "grid_rnd_intrinsic-weight-2_seed-14"])
        self.assertIsNone(self.claim(CARD_ONE_COPY))
        self.assertEqual(self.claim(CARD_FOUR_COPIES)["name"], made[1])

    def test_a_unit_continues_on_another_card_type_of_the_same_profile_only(self):
        """Started on an RTX A4500, a single is continued by an RTX 5080; its recorded settings stay the first start's."""
        first = self.claim(CARD_ONE_COPY, "71")
        unit_queue.give_back(self.run_dir, first, "stopped_with_state_saved")
        again = self.claim(CARD_SECOND_TYPE, "72")
        self.assertEqual((again["name"], again["attempts"], again["card"]), (first["name"], 2, "NVIDIA GeForce RTX 5080"))
        self.assertEqual(again["execution_profile_settings"], first["execution_profile_settings"])
        self.assertEqual(again["history"][0]["card"], "NVIDIA RTX A4500")
        # a record whose profile is another one is never taken, whatever its number of copies says
        unit_queue.give_back(self.run_dir, again, "stopped_with_state_saved")
        path = unit_queue.unit_path(self.run_dir, "pending", first["name"])
        common.write_json(path, dict(common.read_json(path), execution_profile="another profile"))
        self.assertNotEqual(self.claim(CARD_ONE_COPY)["name"], first["name"])

    def test_a_unit_that_ran_out_of_memory_waits_for_a_card_with_more(self):
        """Raised above an RTX 5080's memory, the unit is left by that card type and taken by an RTX A4500."""
        first = self.claim(CARD_SECOND_TYPE)
        unit_queue.give_back(self.run_dir, first, "out_of_card_memory", minimum_device_memory_mb=16304)
        self.assertNotEqual(self.claim(CARD_SECOND_TYPE)["name"], first["name"])
        self.assertEqual(self.claim(CARD_ONE_COPY)["name"], first["name"])

    def test_an_h200_starts_eight_copy_packs_and_continues_started_four_copy_packs_only(self):
        """A never-started four-copy pack is left to four-copy cards; a started one an H200 continues with its settings."""
        card_eight = dict(CARD_FOUR_COPIES, gpu="NVIDIA H200", memory_mb=143771, hostname="h200node")
        description = small_description(seeds={"first": 13, "count": 3}, methods=[
            {"method": "fast-last-layer", "pack_over": "step_size",
             "grid": {"sharing_group": [16], "intrinsic_weight": [1, 2], "step_size": [0.01, 0.03, 0.1, 0.3]}}])
        run_dir = self.run_dir.parent / "eight"
        queue_units(run_dir, build_units.study_units(description))
        four = quiet(build_units.pack_pending, run_dir, 2, match="seed-13")
        eight = quiet(build_units.pack_pending, run_dir, 1, match="seed-14", size=8)

        def claim(card, own_copy=0):
            """Claim the next unit a worker of this card and code copy may take in the folder of this test."""
            name = pixel_queue.profile_for_card(PROFILES, card["memory_mb"])
            return pixel_queue.claim(run_dir, card["memory_mb"], dict(claimant(card), copy_number=own_copy), name,
                                     PROFILES[name])

        # the H200 takes the eight-copy pack, not the four-copy packs that stand before it in the claim order
        taken = claim(card_eight)
        self.assertEqual((taken["name"], taken["execution_profile"]), (eight[0], "eight copies"))
        self.assertEqual(taken["execution_profile_settings"]["environment"]["reset_lane_budget"], 32)
        self.assertIsNone(claim(card_eight))
        # an H100 takes a four-copy pack; given back, an H200 continues it under its four-copy settings
        started = claim(CARD_FOUR_COPIES)
        self.assertEqual(started["name"], four[0])
        unit_queue.give_back(run_dir, started, "stopped_with_state_saved")
        again = claim(card_eight)
        self.assertEqual((again["name"], again["execution_profile"], again["attempts"]), (four[0], "four copies", 2))
        self.assertEqual(again["execution_profile_settings"]["environment"]["reset_lane_budget"], 16)
        # an H100 never takes an eight-copy pack, and the left units are counted by reason
        unit_queue.give_back(run_dir, taken, "stopped_with_state_saved")
        self.assertEqual(claim(CARD_FOUR_COPIES)["name"], four[1])
        self.assertIsNone(claim(CARD_FOUR_COPIES))
        self.assertEqual(pixel_queue.pending_for_others(run_dir, "four copies", PROFILES["four copies"], 95830, 0),
                         {"another profile": 1 + 8, "more card memory": 0, "a newer code copy": 0})

    def test_a_worker_of_an_older_code_copy_leaves_an_eight_copy_pack_alone(self):
        """A pack built against copy 3 is not claimed by a worker of copy 2 on the same card."""
        description = small_description(seeds={"first": 13, "count": 1}, methods=[
            {"method": "fast-last-layer", "pack_over": "step_size",
             "grid": {"sharing_group": [16], "intrinsic_weight": [1, 2], "step_size": [0.01, 0.03, 0.1, 0.3]}}])
        run_dir = self.run_dir.parent / "copies"
        queue_units(run_dir, build_units.study_units(description))
        made = quiet(build_units.pack_pending, run_dir, 1, size=8, minimum_copy=3)
        card = dict(CARD_FOUR_COPIES, gpu="NVIDIA H200", memory_mb=143771)
        self.assertIsNone(pixel_queue.claim(run_dir, 143771, dict(claimant(card), copy_number=2), "eight copies",
                                            PROFILES["eight copies"]))
        self.assertEqual(pixel_queue.claim(run_dir, 143771, dict(claimant(card), copy_number=3), "eight copies",
                                           PROFILES["eight copies"])["name"], made[0])

    def test_the_trainer_settings_join_the_profile_and_the_unit(self):
        """A pack's learner JSON holds the H100 settings, the shared settings and the per-copy step sizes."""
        quiet(build_units.pack_pending, self.run_dir, 1)
        pack = self.claim(CARD_FOUR_COPIES)
        learner, environment = pixel_queue.trainer_settings(pack, common.read_json(self.run_dir / pack["specification"]))
        self.assertEqual(learner["convolution_algorithm"], "space_to_depth")
        self.assertEqual(learner["per_copy_scalar_settings"], {"fast_last_layer_step": [0.01, 0.03, 0.1, 0.3]})
        self.assertEqual((learner["fast_last_layer_sharing_group"], learner["record_game_events"]), (16, True))
        self.assertEqual((environment["reset_lane_budget"], environment["report_game_events"]), (16, True))


class WorkerCase(unittest.TestCase):
    """A temporary run folder with the small study, a stand-in trainer, and no Slurm job around the worker."""

    def setUp(self):
        """Make the temporary folder the tests of this class work in."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.run_dir = self.folder / "run"
        self.names = queue_units(self.run_dir, build_units.study_units(small_description()))
        self.trainer = self.folder / "fake_trainer.py"
        self.trainer.write_text(FAKE_TRAINER)

    def work(self, mode, *options, card=CARD_ONE_COPY):
        """Run one worker to its end with the stand-in trainer in the given mode; returns its exit code."""
        environment = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
        environment.update(FAKE_TRAINER_MODE=mode, JAX_COMPILATION_CACHE_DIR=str(self.folder / "cache"))
        with mock.patch.dict(os.environ, environment, clear=True), contextlib.redirect_stdout(io.StringIO()):
            return worker.main(["--run-dir", str(self.run_dir), "--max-seconds", "7200", *options],
                               card=card, trainer=self.trainer, sample_memory=None)

    def record(self, state, name):
        """Read a unit's queue record from one queue folder."""
        return common.read_json(unit_queue.unit_path(self.run_dir, state, name))

    def command(self, name):
        """The options the stand-in trainer was started with for one unit, as {option: value}."""
        words = json.loads((pixel_queue.run_folder(self.run_dir, name) / "command.json").read_text())
        return dict(zip(words[0::2], words[1::2]))


class Outcomes(WorkerCase):
    """Where a unit goes after its trainer ends."""

    def test_completed_units_move_to_done_and_the_worker_takes_the_next(self):
        """All eight singles are done by one worker; each records its job, its card and its profile."""
        self.assertEqual(self.work("complete"), 0)
        self.assertEqual(unit_queue.counts(self.run_dir), {"done": 8, "failed": 0, "pending": 0, "running": 0})
        record = self.record("done", self.names[0])
        attempt = record["history"][0]
        self.assertEqual((attempt["job_id"], attempt["hostname"], attempt["card"], attempt["outcome"]),
                         ("local", "testnode", "NVIDIA RTX A4500", "complete"))
        self.assertEqual((record["execution_profile"], record["result"]["update_at_end"]), ("one copy", 244))
        self.assertTrue(Path(record["result"]["log"]).exists())
        self.assertTrue((self.run_dir / "slurm/jobs/local_testnode_card-0/hardware.json").exists())

    def test_the_command_is_the_trainers_with_the_units_settings(self):
        """Run folder, seeds, target, checkpoints, a time limit under the job's, and both JSON documents."""
        self.work("complete")
        name = "grid_fast-last-layer_step-size-0.03_sharing-group-16_intrinsic-weight-1_seed-14"
        command = self.command(name)
        self.assertEqual(command["--run-dir"], str(self.run_dir / "data" / name / "run"))
        self.assertEqual((command["--copies"], command["--copy-seeds"], command["--target-updates"],
                          command["--observation-warmup-iterations"], command["--checkpoint-seconds"]),
                         ("1", "14", "244", "50", "1200"))
        # 7,200 s for the worker, less the 900 s kept for the trainer's last update and save
        self.assertTrue(6200 < int(command["--time-limit-seconds"]) <= 6300)
        learner, environment = json.loads(command["--learner"]), json.loads(command["--environment"])
        self.assertEqual(learner, {**PROFILES["one copy"]["learner"], "total_timesteps": 2000000000,
                                   "fast_last_layer_step": 0.03, "fast_last_layer_sharing_group": 16,
                                   "intrinsic_coefficient": 1.0, "record_game_events": True,
                                   "record_bonus_profile": True})
        self.assertEqual(environment, {**PROFILES["one copy"]["environment"], "report_game_events": True})
        # RND's command leaves the fast layer at its default, so its compiled program is the archived one
        rnd = json.loads(self.command("grid_rnd_intrinsic-weight-2_seed-13")["--learner"])
        self.assertEqual(rnd["intrinsic_coefficient"], 2.0)
        self.assertNotIn("fast_last_layer_step", rnd)
        cache = json.loads((pixel_queue.run_folder(self.run_dir, name) / "environment.json").read_text())
        self.assertEqual(cache["JAX_COMPILATION_CACHE_DIR"], str(self.folder / "cache"))

    def test_a_stopped_unit_returns_to_pending_and_is_no_failure(self):
        """Exit code 0 without a completed record: the state is saved, the unit waits for a later job."""
        self.assertEqual(self.work("stop"), 0)
        self.assertEqual(unit_queue.counts(self.run_dir), {"done": 0, "failed": 0, "pending": 8, "running": 0})
        record = self.record("pending", self.names[0])
        attempt = record["history"][0]
        self.assertEqual((record["attempts"], record["failures"], attempt["outcome"], attempt["exit_code"],
                          attempt["update_at_end"]), (1, 0, "stopped_with_state_saved", 0, 5))
        self.assertEqual(record["execution_profile"], "one copy")
        # a later job continues it and completes it; the same run folder is used again
        self.assertEqual(self.work("complete", card=CARD_SECOND_TYPE), 0)
        record = self.record("done", self.names[0])
        self.assertEqual([attempt["card"] for attempt in record["history"]],
                         ["NVIDIA RTX A4500", "NVIDIA GeForce RTX 5080"])
        self.assertEqual(record["result"]["update_at_start"], 5)

    def test_every_start_records_its_card_in_the_run_folders_events(self):
        """events.jsonl holds one start and one end line of the worker per attempt, with the device and the copies."""
        self.work("stop")
        self.work("complete", card=CARD_SECOND_TYPE)
        events = [json.loads(line) for line in
                  (pixel_queue.run_folder(self.run_dir, self.names[0]) / "events.jsonl").read_text().splitlines()]
        starts = [event for event in events if event["event"] == "worker_start"]
        self.assertEqual([(event["device"], event["execution_profile"], event["copies"], event["attempt"], event["update"])
                          for event in starts],
                         [("NVIDIA RTX A4500", "one copy", 1, 1, 0), ("NVIDIA GeForce RTX 5080", "one copy", 1, 2, 5)])
        self.assertEqual([event["outcome"] for event in events if event["event"] == "worker_end"], ["stopped", "complete"])

    def test_a_short_check_job_keeps_a_smaller_margin(self):
        """--margin-seconds 240 and --minimum-seconds 240: the trainer's limit is the job's time less 240 s."""
        self.work("complete", "--margin-seconds", "240", "--minimum-seconds", "240")
        command = self.command(self.names[0])
        self.assertTrue(6860 < int(command["--time-limit-seconds"]) <= 6960)

    def test_a_failed_unit_returns_to_pending_and_fails_at_its_third_failure(self):
        """One failure ends the worker; three workers move the first unit to failed with its log."""
        self.assertEqual(self.work("fail"), 1)
        counts = unit_queue.counts(self.run_dir)
        self.assertEqual((counts["pending"], counts["failed"]), (8, 0))
        self.assertEqual(self.record("pending", self.names[0])["failures"], 1)
        self.work("fail")
        self.work("fail")
        record = self.record("failed", self.names[0])
        self.assertEqual((record["failures"], len(record["history"])), (3, 3))
        self.assertIn("the stand-in trainer was told to fail", Path(record["failure_log"]).read_text())

    def test_exit_code_0_without_a_saved_state_is_a_failure(self):
        """A trainer that ends without a completed record and without a checkpoint did not stop with its state saved."""
        self.assertEqual(self.work("exit_without_state"), 1)
        record = self.record("pending", self.names[0])
        self.assertEqual((record["failures"], record["history"][0]["outcome"]), (1, "failed"))

    def test_out_of_card_memory_leaves_the_unit_to_larger_cards(self):
        """The unit returns to pending without a failure and needs more memory than this card has."""
        self.assertEqual(self.work("out_of_memory", card=CARD_SECOND_TYPE), 1)
        record = self.record("pending", self.names[0])
        self.assertEqual((record["failures"], record["minimum_device_memory_mb"], record["history"][0]["outcome"]),
                         (0, 16304, "out_of_card_memory"))

    def test_a_unit_whose_run_folder_is_at_the_target_needs_no_trainer(self):
        """A completed progress record found at the claim finishes the unit; the trainer is not started."""
        folder = pixel_queue.run_folder(self.run_dir, self.names[0])
        folder.mkdir(parents=True)
        (folder / "progress.json").write_text(json.dumps({"update": 244, "target_updates": 244, "completed": True}))
        self.assertEqual(self.work("fail"), 1)
        self.assertEqual(self.record("done", self.names[0])["result"]["wall_seconds"], 0.0)
        self.assertFalse((folder / "command.json").exists())

    def test_a_completed_record_of_a_lower_target_is_not_complete(self):
        """After its target was raised, a unit with the earlier target's completed record is trained on."""
        self.work("complete")
        description = small_description(target_updates=488)
        quiet(build_units.continue_units, self.run_dir, build_units.study_units(description))
        self.assertEqual(unit_queue.counts(self.run_dir)["pending"], 8)
        self.assertEqual(self.work("complete"), 0)
        record = self.record("done", self.names[0])
        self.assertEqual((record["result"]["update_at_start"], record["result"]["update_at_end"]), (244, 488))
        self.assertEqual(self.command(self.names[0])["--target-updates"], "488")


class WorkerEnds(WorkerCase):
    """When a worker takes no unit."""

    def test_a_large_card_leaves_singles_and_a_small_card_leaves_packs(self):
        """A worker of the other profile ends at once with exit code 0 and claims nothing."""
        self.assertEqual(self.work("complete", card=CARD_FOUR_COPIES), 0)
        self.assertEqual(unit_queue.counts(self.run_dir)["pending"], 8)

    def test_a_card_below_every_profile_takes_nothing(self):
        """A card with 8 GB has no profile; its worker ends without a claim."""
        self.assertEqual(self.work("complete", card=dict(CARD_ONE_COPY, memory_mb=8192)), 0)
        self.assertEqual(unit_queue.counts(self.run_dir)["pending"], 8)

    def test_the_stop_file_and_a_short_job_keep_the_worker_from_claiming(self):
        """RUN/STOP, or less training time than the minimum, ends the worker before any claim."""
        environment = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
        with mock.patch.dict(os.environ, environment, clear=True), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(worker.main(["--run-dir", str(self.run_dir), "--max-seconds", "1000"],
                                         card=CARD_ONE_COPY, trainer=self.trainer, sample_memory=None), 0)
        (self.run_dir / "STOP").write_text("stop\n")
        self.assertEqual(self.work("complete"), 0)
        self.assertEqual(unit_queue.counts(self.run_dir)["pending"], 8)

    def test_a_remote_worker_leaves_the_files_a_tick_copies(self):
        """With --remote-records a complete unit has its claim record in done/ and its files in data/research/."""
        self.assertEqual(self.work("complete", "--remote-records", "--server", "gautschi"), 0)
        name = self.names[0]
        claim = (self.run_dir / "queue/done" / f"{name}.job").read_text()
        self.assertIn("job=local host=testnode gpu=NVIDIA_RTX_A4500", claim)
        folder = self.run_dir / "data/research" / name
        summary = common.read_json(folder / "summary.json")
        self.assertEqual((summary["completed"], summary["target_updates"], summary["execution_profile"], summary["card"]),
                         (True, 244, "one copy", "NVIDIA RTX A4500"))
        self.assertTrue((folder / "checkpoint.pkl").samefile(pixel_queue.run_folder(self.run_dir, name) / "checkpoint.pkl"))
        self.assertTrue((folder / "progress.json").exists())
        events = (pixel_queue.run_folder(self.run_dir, name) / "events.jsonl").read_text()
        self.assertIn('"server": "gautschi"', events)


class MemoryFraction(unittest.TestCase):
    """The share of card memory the trainer's JAX may take, by card."""

    def test_the_rtx_2080_ti_gets_0_95_and_other_cards_keep_the_jobs_setting(self):
        """An RTX 2080 Ti's trainer gets 0.95 (gate 6); an RTX A4500's and an H100's keep the job's 0.85."""
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(
                os.environ, {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85", "JAX_COMPILATION_CACHE_DIR": folder}):
            fractions = {card["gpu"]: worker.trainer_environment(card)["XLA_PYTHON_CLIENT_MEM_FRACTION"]
                         for card in (dict(CARD_ONE_COPY, gpu="NVIDIA GeForce RTX 2080 Ti", memory_mb=11264),
                                      CARD_ONE_COPY, CARD_FOUR_COPIES)}
            # the worker's own environment is not changed
            self.assertEqual(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.85")
        self.assertEqual(fractions, {"NVIDIA GeForce RTX 2080 Ti": "0.95", "NVIDIA RTX A4500": "0.85",
                                     "NVIDIA H100 NVL": "0.85"})


class NoNumericalLibrary(unittest.TestCase):
    """The files of this folder that run on a login node import no numerical library."""

    def test_no_file_imports_jax_or_numpy(self):
        """Only the two compute-node check files of the smoke job may import them."""
        compute_node_only = {"smoke_trainer_checks.py"}
        for path in sorted((PLATFORM / "scripts/pixel_sweep_campaign").glob("*.py")):
            if path.name in compute_node_only:
                continue
            for line in path.read_text().splitlines():
                words = line.split()
                if len(words) >= 2 and words[0] in ("import", "from"):
                    self.assertNotIn(words[1].split(".")[0], ("jax", "numpy", "torch", "jaxtari"), f"{path.name}: {line}")


if __name__ == "__main__":
    unittest.main()
