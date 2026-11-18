"""The parts of train_final.py and compare_metric_prefix.py that need no numerical library.

Standard library only: the run identity by content, the source commit of a frozen code copy, the
seeds of the copies, and the comparison of metric lines with an archive. May run on a login node:
`python3 -m unittest tests/atari_pixels/test_train_final_records.py` from the platform folder.
"""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

PLATFORM = Path(__file__).resolve().parents[2]
SCRIPTS = PLATFORM / "scripts/atari_pixel_campaign"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_metric_prefix  # noqa: E402


def load_train_final():
    """Import train_final.py as a module; its top level imports only the standard library."""
    specification = importlib.util.spec_from_file_location("train_final_under_test", SCRIPTS / "train_final.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


train_final = load_train_final()
GAME = {"installed_version": "0.1.0", "asset_record_sha256": "a" * 64, "game_source_sha256": "b" * 64,
        "path": "/first/server/jaxtari", "asset_root": "/first/server/assets"}


def identity(**changes):
    """A run identity as train_final.py builds it, with some fields replaced."""
    return {"learner": {"n_copies": 1, "fast_last_layer_step": 0.1}, "environment": {"reset_lane_budget": 8},
            "copy_seeds": [13], "game_source": dict(GAME), **changes}


class RunIdentityTest(unittest.TestCase):
    """A run folder continues only with the same settings, seeds and game content."""

    def test_the_same_game_stored_elsewhere_may_resume(self):
        """Paths are information: another server's folders do not block a resume."""
        moved = identity(game_source={**GAME, "path": "/other/server/jaxtari", "asset_root": "/other/assets"})
        self.assertEqual(train_final.mismatched_identity(identity(), moved), [])

    def test_each_content_field_of_the_game_blocks_a_resume(self):
        """A different version, asset record or source hash is a different game."""
        for field in train_final.GAME_CONTENT_FIELDS:
            changed = identity(game_source={**GAME, field: "different"})
            self.assertEqual(train_final.mismatched_identity(identity(), changed), ["game_source"], field)

    def test_settings_and_seeds_are_compared_exactly(self):
        """A changed learner setting, environment setting or seed list is named."""
        self.assertEqual(train_final.mismatched_identity(
            identity(), identity(learner={"n_copies": 1, "fast_last_layer_step": 0.3})), ["learner"])
        self.assertEqual(train_final.mismatched_identity(identity(), identity(copy_seeds=[14])), ["copy_seeds"])
        self.assertEqual(train_final.mismatched_identity(
            identity(), identity(environment={"reset_lane_budget": 16})), ["environment"])

    def test_a_record_without_the_source_hash_does_not_match(self):
        """Run folders written before the game was compared by content cannot be continued."""
        recorded = identity(game_source={"path": GAME["path"], "commit": "c" * 40, "uncommitted_changes": False,
                                         "asset_root": GAME["asset_root"],
                                         "asset_record_sha256": GAME["asset_record_sha256"]})
        self.assertEqual(train_final.mismatched_identity(recorded, identity()), ["game_source"])

    def test_source_hash_follows_content_and_relative_paths_only(self):
        """Two copies of a package hash the same wherever they are; any Python file change shows."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("first", "second"):
                (root / name / "games").mkdir(parents=True)
                (root / name / "__init__.py").write_text("VALUE = 1\n")
                (root / name / "games" / "montezuma.py").write_text("ROOMS = 33\n")
            # Files that are not Python source (compiled files, notes) do not enter the hash.
            (root / "second" / "notes.txt").write_text("not source\n")
            first = train_final.source_files_sha256(root / "first")
            self.assertEqual(first, train_final.source_files_sha256(root / "second"))
            (root / "second" / "games" / "montezuma.py").write_text("ROOMS = 34\n")
            self.assertNotEqual(first, train_final.source_files_sha256(root / "second"))
            (root / "second" / "games" / "montezuma.py").rename(root / "second" / "montezuma.py")
            (root / "second" / "montezuma.py").write_text("ROOMS = 33\n")
            self.assertNotEqual(first, train_final.source_files_sha256(root / "second"))

    def test_source_commit_comes_from_a_frozen_copy_when_there_is_one(self):
        """Inside a frozen code copy the commit is the copy's own record, not the enclosing repository's."""
        original = train_final.PLATFORM_ROOT
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "SOURCE_COMMIT").write_text("0123456789abcdef0123456789abcdef01234567\n")
            train_final.PLATFORM_ROOT = Path(temporary)
            try:
                self.assertEqual(train_final.source_commit(), "0123456789abcdef0123456789abcdef01234567")
            finally:
                train_final.PLATFORM_ROOT = original

    def test_copy_seeds_are_listed_or_consecutive(self):
        """--copy-seeds gives one seed per copy and may repeat a seed; otherwise seeds count up."""
        self.assertEqual(train_final.copy_seeds(4, 5, None), (5, 6, 7, 8))
        self.assertEqual(train_final.copy_seeds(4, 1, "13,13,13,13"), (13, 13, 13, 13))
        self.assertEqual(train_final.copy_seeds(2, 1, "20, 14"), (20, 14))
        with self.assertRaises(SystemExit):
            train_final.copy_seeds(4, 1, "13,14")

    def test_copies_with_a_non_finite_loss_are_named_by_seed(self):
        """NaN and both infinities count; a finite loss of any size does not."""
        self.assertEqual(train_final.seeds_with_non_finite_loss([0.31, 1e30, -4.0, 0.0], (13, 14, 15, 16)), [])
        self.assertEqual(train_final.seeds_with_non_finite_loss(
            [0.31, float("nan"), float("-inf"), float("inf")], (13, 14, 15, 16)), [14, 15, 16])
        self.assertEqual(train_final.seeds_with_non_finite_loss([float("nan")], (13,)), [13])

    def test_per_copy_settings_survive_the_learner_json(self):
        """The learner JSON of a packed unit keeps its per-copy mapping for the configuration."""
        settings = train_final.learner_settings(json.dumps({
            "fast_last_layer_sharing_group": 16,
            "per_copy_scalar_settings": {"fast_last_layer_step": [0.01, 0.03, 0.1, 0.3]},
            "iteration_compiler_options": [["xla_gpu_enable_while_loop_double_buffering", "true"]]}))
        self.assertEqual(settings["per_copy_scalar_settings"], {"fast_last_layer_step": [0.01, 0.03, 0.1, 0.3]})
        self.assertEqual(settings["iteration_compiler_options"],
                         (("xla_gpu_enable_while_loop_double_buffering", "true"),))


def lines(count, **changes):
    """Metric lines of one copy; `changes` maps a 1-based line number to fields replaced in it."""
    result = []
    for update in range(1, count + 1):
        line = {"update": update, "unix": 50.0 + update, "loss": [0.5 / update], "episode_count": [update % 2],
                "rooms_visited": [[False, True]], "steps_per_policy": 16384 * update}
        line.update(changes.get(f"line_{update}", {}))
        result.append(line)
    return result


class MetricPrefixTest(unittest.TestCase):
    """Every archived field must be equal, line by line; the write time and new fields do not count."""

    def test_equal_lines_pass_and_new_fields_are_listed(self):
        """A run with the game events on has more fields; the archived ones still decide."""
        new = [{**line, "unix": 9000.0, "event_counts": [[0, 1]]} for line in lines(5)]
        record = compare_metric_prefix.compare(lines(5), new, wanted=5)
        self.assertTrue(record["passed"])
        self.assertEqual(record["lines_compared"], 5)
        self.assertEqual(record["fields_compared"], 5 * 5)
        self.assertEqual(record["fields_only_in_new_run"], ["event_counts"])

    def test_one_differing_number_fails_and_is_reported(self):
        """The record names the line, the field and both values."""
        new = lines(5, line_3={"loss": [0.16666666666666669]})
        record = compare_metric_prefix.compare(lines(5), new, wanted=5)
        self.assertFalse(record["passed"])
        self.assertEqual(record["differing_fields"], 1)
        self.assertEqual(record["first_differences"][0],
                         {"line": 3, "field": "loss", "archived": [0.5 / 3], "new": [0.16666666666666669]})

    def test_a_missing_field_a_short_file_and_a_changed_flag_fail(self):
        """A dropped archived field, too few lines, and true against 1 are all differences."""
        missing = [{key: value for key, value in line.items() if key != "episode_count"} for line in lines(3)]
        self.assertFalse(compare_metric_prefix.compare(lines(3), missing, wanted=3)["passed"])
        self.assertFalse(compare_metric_prefix.compare(lines(3), lines(2), wanted=3)["passed"])
        self.assertFalse(compare_metric_prefix.compare(lines(2), lines(3), wanted=3)["passed"])
        flags = lines(2, line_1={"rooms_visited": [[0, 1]]})
        self.assertFalse(compare_metric_prefix.compare(lines(2), flags, wanted=2)["passed"])
        self.assertFalse(compare_metric_prefix.compare([], [], wanted=0)["passed"])

    def test_not_a_number_equals_not_a_number(self):
        """Two NaN values in the same place are the same recorded value."""
        archived = lines(1, line_1={"loss": [float("nan")]})
        self.assertTrue(compare_metric_prefix.compare(archived, lines(1, line_1={"loss": [float("nan")]}), 1)["passed"])
        self.assertFalse(compare_metric_prefix.compare(archived, lines(1), 1)["passed"])

    def test_command_reads_only_the_wanted_lines_and_sets_the_exit_status(self):
        """The command compares the first N lines and returns 1 on a difference."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "archived.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines(50)))
            (root / "new.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines(10)))
            (root / "bad.jsonl").write_text("".join(
                json.dumps(line) + "\n" for line in lines(10, line_10={"steps_per_policy": 1})))
            arguments = [str(root / "archived.jsonl"), str(root / "new.jsonl"), str(root / "record.json")]
            self.assertEqual(compare_metric_prefix.main(arguments), 0)
            record = json.loads((root / "record.json").read_text())
            self.assertEqual((record["lines_compared"], record["archived_lines_read"]), (10, 10))
            self.assertEqual(compare_metric_prefix.main(arguments + ["--lines", "4"]), 0)
            self.assertEqual(json.loads((root / "record.json").read_text())["lines_compared"], 4)
            self.assertEqual(compare_metric_prefix.main(
                [str(root / "archived.jsonl"), str(root / "bad.jsonl"), str(root / "record.json")]), 1)
            self.assertEqual(compare_metric_prefix.main(arguments + ["--lines", "11"]), 1)


if __name__ == "__main__":
    unittest.main()
