"""The unit builder of the pixel sweep: the stage-1 grid, names, claim order, packs, and continued units."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from pixel_test_support import PIXEL, STAGE_ONE, build_units, common, pixel_queue, queue_units, small_description, unit_queue


def quiet(function, *arguments, **keywords):
    """Call a function of the builder without its printed lines."""
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*arguments, **keywords)


class StageOne(unittest.TestCase):
    """The description of stage 1 that is kept in the scripts folder."""

    @classmethod
    def setUpClass(cls):
        """Build the stage-1 units once."""
        cls.description = json.loads(STAGE_ONE.read_text())
        cls.units = build_units.study_units(cls.description)

    def test_the_grid_has_624_single_units(self):
        """72 method configurations and 6 RND weights on 8 seeds, each one copy with its own name."""
        names = [unit["unit_id"] for unit in self.units]
        self.assertEqual((len(names), len(set(names))), (624, 624))
        self.assertEqual(sum(unit["method"] == "rnd" for unit in self.units), 48)
        self.assertEqual(sum(unit["method"] == "fast-last-layer" for unit in self.units), 576)
        self.assertTrue(all(unit["copies"] == 1 and unit["copy_seeds"] == [unit["runs"][0]["seed"]] for unit in self.units))
        configurations = {unit["configuration"] for unit in self.units}
        self.assertEqual(len(configurations), 78)
        self.assertEqual({unit["runs"][0]["seed"] for unit in self.units}, set(range(13, 21)))

    def test_names_are_in_words(self):
        """The plan's two example names are built, and no name carries a short code."""
        names = {unit["unit_id"] for unit in self.units}
        self.assertIn("grid_fast-last-layer_step-size-0.03_sharing-group-16_intrinsic-weight-1_seed-13", names)
        self.assertIn("grid_rnd_intrinsic-weight-1_seed-13", names)
        self.assertIn("grid_fast-last-layer_step-size-0.3_sharing-group-128_intrinsic-weight-0.25_seed-20", names)
        self.assertTrue(all(word not in name for word in build_units.PACK_WORDS.values() for name in names))

    def test_claims_are_interleaved_by_seed(self):
        """Every configuration at seed 13 stands before any at seed 14, and so on; RND comes first in a seed."""
        seeds = [unit["runs"][0]["seed"] for unit in self.units]
        self.assertEqual(seeds, sorted(seeds))
        for start in range(0, 624, 78):
            block = self.units[start:start + 78]
            self.assertEqual(len({unit["runs"][0]["seed"] for unit in block}), 1)
            self.assertEqual(len({unit["configuration"] for unit in block}), 78)
            self.assertEqual([unit["method"] for unit in block[:6]], ["rnd"] * 6)
            # the four step sizes of one sharing group and one weight are neighbours
            self.assertEqual([unit["runs"][0]["step_size"] for unit in block[6:10]], [0.01, 0.03, 0.1, 0.3])
            self.assertEqual({(unit["runs"][0]["sharing_group"], unit["runs"][0]["intrinsic_weight"])
                              for unit in block[6:10]}, {(128, 0.25)})

    def test_every_run_has_the_plans_length_and_records(self):
        """12,207 updates, the learning-rate schedule of 2 billion steps, and both record flags, RND included."""
        for unit in self.units:
            self.assertEqual((unit["target_updates"], unit["steps_per_run"]), (12207, 12207 * 16384))
            learner = unit["learner"]
            self.assertEqual((learner["total_timesteps"], learner["record_game_events"], learner["record_bonus_profile"]),
                             (2000000000, True, True))
            self.assertEqual(unit["environment"], {"report_game_events": True})
        method = next(unit for unit in self.units if unit["unit_id"] ==
                      "grid_fast-last-layer_step-size-0.03_sharing-group-16_intrinsic-weight-2_seed-14")
        self.assertEqual((method["learner"]["fast_last_layer_step"], method["learner"]["fast_last_layer_sharing_group"],
                          method["learner"]["intrinsic_coefficient"]), (0.03, 16, 2.0))
        # the trainer wants a whole number for the sharing group
        self.assertIs(type(method["learner"]["fast_last_layer_sharing_group"]), int)
        rnd = next(unit for unit in self.units if unit["unit_id"] == "grid_rnd_intrinsic-weight-0.5_seed-13")
        self.assertEqual(rnd["learner"]["intrinsic_coefficient"], 0.5)
        self.assertNotIn("fast_last_layer_step", rnd["learner"])
        self.assertNotIn("fast_last_layer_sharing_group", rnd["learner"])

    def test_the_command_line_prints_the_plan_and_writes_nothing(self):
        """--dry-run on the stage-1 description names 624 units and leaves the run folder as it was."""
        with tempfile.TemporaryDirectory() as folder:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                build_units.main(["--run-dir", folder, "--study", str(STAGE_ONE), "--dry-run"])
            self.assertIn("624 single units", output.getvalue())
            self.assertIn("dry run: 624 units, nothing written", output.getvalue())
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_the_json_files_of_the_folder_are_not_left_out_by_git(self):
        """The repository ignores *.json; this folder's own rule keeps its JSON files in a commit."""
        inside = subprocess.run(["git", "-C", str(PIXEL), "rev-parse", "--is-inside-work-tree"],
                                capture_output=True, text=True)
        if inside.returncode != 0:
            self.skipTest("not inside a git work tree (a frozen code copy)")
        for path in (PIXEL / "card_profiles.json", PIXEL / "server_table.json", STAGE_ONE):
            ignored = subprocess.run(["git", "-C", str(PIXEL), "check-ignore", "-q", str(path)])
            self.assertEqual(ignored.returncode, 1, f"git would leave {path.name} out of a commit")


class Descriptions(unittest.TestCase):
    """What a description may and may not say."""

    def test_later_stages_use_the_same_format(self):
        """Stage 2: listed configurations, other seeds, a higher target; the names keep the same form."""
        stage_two = small_description(
            phase="long", seeds={"first": 1, "count": 12}, target_updates=61035,
            methods=[{"method": "fast-last-layer", "pack_over": "seed",
                      "configurations": [{"step_size": 0.1, "sharing_group": 16, "intrinsic_weight": 1}]},
                     {"method": "rnd", "pack_over": "seed", "configurations": [{"intrinsic_weight": 2}]}])
        units = build_units.study_units(stage_two)
        self.assertEqual(len(units), 24)
        self.assertEqual(units[0]["unit_id"], "long_fast-last-layer_step-size-0.1_sharing-group-16_intrinsic-weight-1_seed-1")
        self.assertEqual(units[1]["unit_id"], "long_rnd_intrinsic-weight-2_seed-1")
        self.assertEqual({unit["target_updates"] for unit in units}, {61035})

    def test_a_wrong_description_is_refused(self):
        """A step size without a sharing group, a method listed twice, a phase with an underscore, an unknown setting."""
        wrong = [
            small_description(methods=[{"method": "fast-last-layer", "configurations": [
                {"step_size": 0.1, "intrinsic_weight": 1}]}]),
            small_description(methods=[{"method": "rnd", "grid": {"intrinsic_weight": [1]}}] * 2),
            small_description(phase="first_grid"),
            small_description(methods=[{"method": "rnd", "grid": {"intrinsic_weight": [1], "learning_rate": [1e-4]}}]),
            small_description(methods=[{"method": "rnd", "pack_over": "sharing_group", "grid": {"intrinsic_weight": [1]}}]),
            small_description(seeds={"list": [3, 3]}),
        ]
        for description in wrong:
            with self.assertRaises(ValueError):
                build_units.study_units(description)


class Packs(unittest.TestCase):
    """Four never-started singles become one four-copy unit, under the queue lock."""

    def setUp(self):
        """Queue the 78 units of seed 13 and the 78 of seed 14 of the stage-1 grid in a temporary run folder."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.run_dir = Path(folder.name) / "run"
        description = dict(json.loads(STAGE_ONE.read_text()), seeds={"first": 13, "count": 2})
        self.names = queue_units(self.run_dir, build_units.study_units(description))

    def test_a_dry_run_names_the_packs_and_writes_nothing(self):
        """The plan is printed; every single stays pending."""
        made = quiet(build_units.pack_pending, self.run_dir, 3, dry_run=True)
        self.assertEqual(len(made), 3)
        self.assertEqual(unit_queue.counts(self.run_dir)["pending"], 156)
        self.assertFalse((self.run_dir / "queue/packed").exists())

    def test_rnd_packs_four_weights_and_leaves_two_singles(self):
        """The first pack of the claim order is RND's weights 0.25 to 2 at seed 13; the weights 4 and 8 stay singles."""
        made = quiet(build_units.pack_pending, self.run_dir, 1)
        self.assertEqual(made, ["grid_rnd_four-copies_intrinsic-weights-0.25-0.5-1-2_seed-13"])
        pack = common.read_json(self.run_dir / "specifications" / f"{made[0]}.json")
        self.assertEqual((pack["copies"], pack["copy_seeds"]), (4, [13, 13, 13, 13]))
        self.assertEqual(pack["per_copy_settings"], {"intrinsic_coefficient": [0.25, 0.5, 1.0, 2.0]})
        self.assertEqual([run["intrinsic_weight"] for run in pack["runs"]], [0.25, 0.5, 1.0, 2.0])
        self.assertNotIn("fast_last_layer_step", pack["learner"])
        pending = {unit["name"] for unit in unit_queue.read_units(self.run_dir, "pending")}
        self.assertTrue({"grid_rnd_intrinsic-weight-4_seed-13", "grid_rnd_intrinsic-weight-8_seed-13"} <= pending)
        self.assertNotIn("grid_rnd_intrinsic-weight-1_seed-13", pending)
        packed = common.read_json(unit_queue.unit_path(self.run_dir, "packed", "grid_rnd_intrinsic-weight-1_seed-13"))
        self.assertEqual(packed["packed_into"], made[0])
        # 156 singles: 4 moved to packed, 1 pack added
        self.assertEqual((unit_queue.counts(self.run_dir)["pending"], unit_queue.counts(self.run_dir)["packed"]), (153, 4))

    def test_a_method_pack_holds_the_four_step_sizes_of_one_group_weight_and_seed(self):
        """One sharing group, one weight, one seed; the step sizes go in per copy, and the pack keeps the first single's place."""
        made = quiet(build_units.pack_pending, self.run_dir, 5, match="sharing-group-16_intrinsic-weight-1_seed-14")
        name = "grid_fast-last-layer_four-copies_step-sizes-0.01-0.03-0.1-0.3_sharing-group-16_intrinsic-weight-1_seed-14"
        self.assertEqual(made, [name])
        pack = common.read_json(self.run_dir / "specifications" / f"{name}.json")
        self.assertEqual(pack["per_copy_settings"], {"fast_last_layer_step": [0.01, 0.03, 0.1, 0.3]})
        self.assertEqual((pack["learner"]["fast_last_layer_sharing_group"], pack["learner"]["intrinsic_coefficient"],
                          pack["learner"]["fast_last_layer_step"]), (16, 1.0, 0.01))
        self.assertEqual(pack["members"], [
            f"grid_fast-last-layer_step-size-{step}_sharing-group-16_intrinsic-weight-1_seed-14"
            for step in ("0.01", "0.03", "0.1", "0.3")])
        record = common.read_json(unit_queue.unit_path(self.run_dir, "pending", name))
        first = common.read_json(unit_queue.unit_path(self.run_dir, "packed", pack["members"][0]))
        self.assertEqual((record["copies"], record["order"], record["attempts"]), (4, first["order"], 0))

    def test_a_started_single_is_never_packed(self):
        """A single a worker has claimed once stays a single; a pack is made only of singles that never started."""
        profiles = pixel_queue.read_profiles()
        claimed = pixel_queue.claim(self.run_dir, 20470, {"job_id": "local", "copy_number": 0}, "one copy",
                                    profiles["one copy"])
        self.assertEqual(claimed["name"], "grid_rnd_intrinsic-weight-0.25_seed-13")
        unit_queue.give_back(self.run_dir, claimed, "stopped_with_state_saved")
        # RND: the five weights that never started still give one pack, without the started one
        made = quiet(build_units.pack_pending, self.run_dir, 1)
        self.assertEqual(made, ["grid_rnd_four-copies_intrinsic-weights-0.5-1-2-4_seed-13"])
        self.assertIn("grid_rnd_intrinsic-weight-0.25_seed-13",
                      {unit["name"] for unit in unit_queue.read_units(self.run_dir, "pending")})
        # the method: with one of the four step sizes started, the other three stay singles
        started = "grid_fast-last-layer_step-size-0.1_sharing-group-128_intrinsic-weight-0.25_seed-13"
        path = unit_queue.unit_path(self.run_dir, "pending", started)
        common.write_json(path, dict(common.read_json(path), attempts=1, execution_profile="one copy"))
        self.assertEqual(quiet(build_units.pack_pending, self.run_dir, 9,
                               match="sharing-group-128_intrinsic-weight-0.25_seed-13"), [])

    def test_packs_from_the_end_and_all_possible_packs(self):
        """--from-end takes seed 14's last group first; asking for more packs than exist makes all 38."""
        made = quiet(build_units.pack_pending, self.run_dir, 1, from_end=True)
        self.assertEqual(made, ["grid_fast-last-layer_four-copies_step-sizes-0.01-0.03-0.1-0.3_sharing-group-4_"
                                "intrinsic-weight-8_seed-14"])
        rest = quiet(build_units.pack_pending, self.run_dir, 1000)
        # per seed: 18 packs of the method and 1 of RND
        self.assertEqual(len(rest) + 1, 38)
        counts = unit_queue.counts(self.run_dir)
        self.assertEqual((counts["pending"], counts["packed"]), (38 + 4, 152))

    def test_a_pack_that_never_started_is_taken_apart_again(self):
        """The four singles are pending again with their old places; the pack leaves no record."""
        before = {unit["name"]: unit["order"] for unit in unit_queue.read_units(self.run_dir, "pending")}
        made = quiet(build_units.pack_pending, self.run_dir, 2)
        undone = quiet(build_units.unpack_pending, self.run_dir, match="rnd")
        self.assertEqual(undone, [made[0]])
        after = {unit["name"]: unit for unit in unit_queue.read_units(self.run_dir, "pending")}
        self.assertNotIn(made[0], after)
        self.assertFalse((self.run_dir / "specifications" / f"{made[0]}.json").exists())
        self.assertEqual(after["grid_rnd_intrinsic-weight-1_seed-13"]["order"], before["grid_rnd_intrinsic-weight-1_seed-13"])
        self.assertNotIn("packed_into", after["grid_rnd_intrinsic-weight-1_seed-13"])
        self.assertIn(made[1], after)

    def test_seeds_pack_four_to_a_unit_when_a_method_says_so(self):
        """Stage 2 and 3: four seeds of one configuration in one process, as the archived baseline's seeds 1 to 4."""
        run_dir = self.run_dir.parent / "long"
        stage_two = small_description(phase="long", seeds={"first": 1, "count": 6}, target_updates=61035, methods=[
            {"method": "rnd", "pack_over": "seed", "configurations": [{"intrinsic_weight": 2}]}])
        queue_units(run_dir, build_units.study_units(stage_two))
        made = quiet(build_units.pack_pending, run_dir, 9)
        self.assertEqual(made, ["long_rnd_four-copies_intrinsic-weight-2_seeds-1-2-3-4"])
        pack = common.read_json(run_dir / "specifications" / f"{made[0]}.json")
        self.assertEqual((pack["copy_seeds"], pack["per_copy_settings"]), ([1, 2, 3, 4], {}))
        self.assertEqual(unit_queue.counts(run_dir)["pending"], 3)


class EightCopyPacks(unittest.TestCase):
    """Eight never-started singles become one eight-copy unit for the H200: a full cross product of two settings."""

    def setUp(self):
        """Queue the 78 units of seed 13 and the 78 of seed 14 of the stage-1 grid in a temporary run folder."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.run_dir = Path(folder.name) / "run"
        description = dict(json.loads(STAGE_ONE.read_text()), seeds={"first": 13, "count": 2})
        self.names = queue_units(self.run_dir, build_units.study_units(description))

    def test_a_method_pack_of_eight_holds_the_four_step_sizes_at_two_weights(self):
        """One sharing group and seed; step size and weight go in per copy, weight by weight; the first single's place."""
        made = quiet(build_units.pack_pending, self.run_dir, 9, match="sharing-group-16_intrinsic-weight-1_seed-14",
                     size=8)
        name = ("grid_fast-last-layer_eight-copies_step-sizes-0.01-0.03-0.1-0.3_sharing-group-16_"
                "intrinsic-weights-1-2_seed-14")
        self.assertEqual(made, [name])
        pack = common.read_json(self.run_dir / "specifications" / f"{name}.json")
        self.assertEqual((pack["copies"], pack["copy_seeds"]), (8, [14] * 8))
        self.assertEqual(pack["per_copy_settings"], {"fast_last_layer_step": [0.01, 0.03, 0.1, 0.3] * 2,
                                                     "intrinsic_coefficient": [1.0] * 4 + [2.0] * 4})
        self.assertEqual((pack["learner"]["fast_last_layer_sharing_group"], pack["learner"]["intrinsic_coefficient"],
                          pack["learner"]["fast_last_layer_step"]), (16, 1.0, 0.01))
        self.assertEqual(pack["members"], [
            f"grid_fast-last-layer_step-size-{step}_sharing-group-16_intrinsic-weight-{weight}_seed-14"
            for weight in ("1", "2") for step in ("0.01", "0.03", "0.1", "0.3")])
        record = common.read_json(unit_queue.unit_path(self.run_dir, "pending", name))
        first = common.read_json(unit_queue.unit_path(self.run_dir, "packed", pack["members"][0]))
        self.assertEqual((record["copies"], record["order"], record["attempts"]), (8, first["order"], 0))
        # the other weights of the group and seed pair up the same way: 0.25 with 0.5, 4 with 8
        rest = quiet(build_units.pack_pending, self.run_dir, 9, match="sharing-group-16_intrinsic-weight-4_seed-14",
                     size=8)
        self.assertEqual(rest, [name.replace("weights-1-2", "weights-4-8")])

    def test_rnd_packs_four_weights_at_two_seeds_and_the_first_pack_stands_first_in_the_claim_order(self):
        """RND's weights 0.25 to 2 at seeds 13 and 14 make the first pack; the weights 4 and 8 stay singles."""
        made = quiet(build_units.pack_pending, self.run_dir, 1, size=8)
        self.assertEqual(made, ["grid_rnd_eight-copies_intrinsic-weights-0.25-0.5-1-2_seeds-13-14"])
        pack = common.read_json(self.run_dir / "specifications" / f"{made[0]}.json")
        self.assertEqual(pack["copy_seeds"], [13] * 4 + [14] * 4)
        self.assertEqual(pack["per_copy_settings"], {"intrinsic_coefficient": [0.25, 0.5, 1.0, 2.0] * 2})
        self.assertNotIn("fast_last_layer_step", pack["learner"])
        pending = {unit["name"] for unit in unit_queue.read_units(self.run_dir, "pending")}
        self.assertTrue({f"grid_rnd_intrinsic-weight-{weight}_seed-{seed}" for weight in (4, 8) for seed in (13, 14)}
                        <= pending)

    def test_every_pack_is_a_full_cross_product_when_a_single_has_started(self):
        """With one step size of weight 0.5 started, weight 0.5 joins no pack and weight 0.25 pairs with weight 1."""
        started = "grid_fast-last-layer_step-size-0.1_sharing-group-16_intrinsic-weight-0.5_seed-14"
        path = unit_queue.unit_path(self.run_dir, "pending", started)
        common.write_json(path, dict(common.read_json(path), attempts=1, execution_profile="one copy"))
        made = quiet(build_units.pack_pending, self.run_dir, 9, match="sharing-group-16_intrinsic-weight-0.25_seed-14",
                     size=8)
        self.assertEqual(made, ["grid_fast-last-layer_eight-copies_step-sizes-0.01-0.03-0.1-0.3_sharing-group-16_"
                                "intrinsic-weights-0.25-1_seed-14"])
        # all packs of both seeds: per seed 9 of the method (8 without the started weight at seed 14) and 1 of RND
        rest = quiet(build_units.pack_pending, self.run_dir, 1000, size=8)
        self.assertEqual(len(made) + len(rest), 9 + 8 + 1)
        for name in made + rest:
            pack = common.read_json(self.run_dir / "specifications" / f"{name}.json")
            settings = [tuple(run.get(word) for word in ("step_size", "intrinsic_weight", "seed")) for run in pack["runs"]]
            values = [sorted({setting[index] for setting in settings}, key=str) for index in range(3)]
            self.assertEqual(len(set(settings)), 8)
            self.assertEqual(len(values[0]) * len(values[1]) * len(values[2]), 8, name)

    def test_an_eight_copy_pack_that_never_started_is_taken_apart_again(self):
        """Its eight singles are pending again with their old places."""
        before = {unit["name"]: unit["order"] for unit in unit_queue.read_units(self.run_dir, "pending")}
        made = quiet(build_units.pack_pending, self.run_dir, 2, size=8)
        undone = quiet(build_units.unpack_pending, self.run_dir, match=made[1])
        self.assertEqual(undone, [made[1]])
        after = {unit["name"]: unit["order"] for unit in unit_queue.read_units(self.run_dir, "pending")}
        members = common.read_json(self.run_dir / "specifications" / f"{made[0]}.json")["members"]
        self.assertEqual({name: order for name, order in after.items() if name in before},
                         {name: order for name, order in before.items() if name not in members})

    def test_an_eight_copy_pack_needs_the_runs_frozen_copy_and_a_four_copy_pack_does_not(self):
        """Through the command line: the frozen copy's number 3 goes into the eight-copy pack, not into the four-copy one."""
        snapshot = self.run_dir.parent / "code/source_abc"
        (snapshot / "10_jax_exploration_platform").mkdir(parents=True)
        (snapshot / "10_jax_exploration_platform/SOURCE_COPY_NUMBER").write_text("3\n")
        common.write_json(self.run_dir / "source.json", {"commit": "abc", "snapshot": str(snapshot), "copy_number": 3})
        quiet(build_units.main, ["--run-dir", str(self.run_dir), "--pack-pending", "--packs", "1", "--pack-size", "8"])
        quiet(build_units.main, ["--run-dir", str(self.run_dir), "--pack-pending", "--packs", "1"])
        records = {unit["name"]: unit for unit in unit_queue.read_units(self.run_dir, "pending") if unit["copies"] > 1}
        self.assertEqual({(unit["copies"], unit["minimum_copy_number"]) for unit in records.values()}, {(8, 3), (4, 0)})
        with self.assertRaises(ValueError):
            build_units.plan_packs([], 1, size=6)


class ContinuedUnits(unittest.TestCase):
    """Stage 3 raises the target of stage 2's units and keeps their run folders."""

    def test_a_finished_unit_goes_back_to_pending_with_the_higher_target(self):
        """Its profile, its history and its earlier result stay in the record; a pack is raised once."""
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / "run"
            stage_two = small_description(phase="long", seeds={"first": 1, "count": 5}, target_updates=61035, methods=[
                {"method": "rnd", "pack_over": "seed", "configurations": [{"intrinsic_weight": 2}]}])
            queue_units(run_dir, build_units.study_units(stage_two))
            pack_name = quiet(build_units.pack_pending, run_dir, 1)[0]
            profiles = pixel_queue.read_profiles()
            # the pack finishes on a large card, the fifth seed on a small one
            pack = pixel_queue.claim(run_dir, 95830, {"job_id": "local", "copy_number": 0}, "four copies",
                                     profiles["four copies"])
            unit_queue.finish(run_dir, pack, {"wall_seconds": 10.0})
            single = pixel_queue.claim(run_dir, 20470, {"job_id": "local", "copy_number": 0}, "one copy",
                                       profiles["one copy"])
            unit_queue.finish(run_dir, single, {"wall_seconds": 10.0})
            stage_three = build_units.study_units(dict(stage_two, target_updates=122070))
            self.assertEqual(quiet(build_units.continue_units, run_dir, stage_three, dry_run=True),
                             {"raised": 2, "left": 0, "missing": 0})
            self.assertEqual(unit_queue.counts(run_dir)["done"], 2)
            self.assertEqual(quiet(build_units.continue_units, run_dir, stage_three),
                             {"raised": 2, "left": 0, "missing": 0})
            record = common.read_json(unit_queue.unit_path(run_dir, "pending", pack_name))
            self.assertEqual((record["target_updates"], record["execution_profile"], len(record["history"])),
                             (122070, "four copies", 1))
            self.assertEqual(record["earlier_targets"][0]["target_updates"], 61035)
            self.assertEqual(record["earlier_targets"][0]["result"], {"wall_seconds": 10.0})
            self.assertNotIn("result", record)
            # a second call finds nothing left to raise
            self.assertEqual(quiet(build_units.continue_units, run_dir, stage_three),
                             {"raised": 0, "left": 2, "missing": 0})


if __name__ == "__main__":
    unittest.main()
