"""Tests of the table generator's own functions on small made-up runs (standard library only).

    python3 -m unittest test_make_all.py      (from this folder)
"""
import math
import unittest

import make_all


def run(first_steps, update=100, method="rnd", archived=False, seed=13, events=True):
    """A run in the shape of report.collect's runs, with the given first-time steps (None: not had)."""
    first_event = {name: ({"step": step} if step is not None else None) for name, step in first_steps.items()}
    return {"method": method, "archived": archived, "seed": seed, "started": True, "update": update,
            "state": "running", "has_event_fields": events, "first": {"first_event": first_event}, "windows": []}


def configuration(runs, settings, archived=False, key=(0,)):
    """A configuration in the shape of report.configurations' groups."""
    return {"runs": runs, "settings": settings, "archived": archived, "method": runs[0]["method"], "key": key}


class MarksTest(unittest.TestCase):
    """Bold the smallest, underline the second smallest, ties share, infinite and missing values unmarked."""

    def test_ties_and_missing(self):
        self.assertEqual(make_all.marks([5, math.inf, 7, 5, None, 9]),
                         ["\\textbf{%s}", "%s", "\\underline{%s}", "\\textbf{%s}", "%s", "%s"])

    def test_one_value(self):
        self.assertEqual(make_all.marks([math.inf, 3]), ["%s", "\\textbf{%s}"])


class MedianFirstTest(unittest.TestCase):
    """A run without the event counts as later than every run with it."""

    def test_missing_counts_last(self):
        runs = [run({"left_start_room": 100}), run({"left_start_room": None}), run({"left_start_room": 300})]
        self.assertEqual(make_all.median_first(runs, "left_start_room"), (300, 2))

    def test_none_had_it(self):
        self.assertEqual(make_all.median_first([run({"left_start_room": None})], "left_start_room"), (math.inf, 0))


class EarlyTabularTest(unittest.TestCase):
    """Rows sorted by first exit, the baseline below a rule, unrecorded events named, an empty block said."""

    def setUp(self):
        steps = {"key_pickup": 1000, "door_opened": 2000, "left_start_room": 3000}
        self.baseline = make_all.early_row(configuration(
            [run(dict(steps, door_opened=None), archived=True, seed=seed, events=False, update=122070)
             for seed in (1, 2)], {"intrinsic_weight": 1.0}, archived=True))

    def test_sorted_by_exit_with_baseline_last(self):
        late = make_all.early_row(configuration(
            [run({"key_pickup": 10, "door_opened": 20, "left_start_room": 9_000_000})], {"intrinsic_weight": 2.0}, key=(1,)))
        never = make_all.early_row(configuration(
            [run({"key_pickup": None, "door_opened": None, "left_start_room": None})], {"intrinsic_weight": 4.0}, key=(2,)))
        early = make_all.early_row(configuration(
            [run({"key_pickup": 50, "door_opened": 60, "left_start_room": 70})], {"intrinsic_weight": 0.5}, key=(3,)))
        lines = make_all.early_tabular([never, late, early], self.baseline).split("\n")
        labels = [line.split(" & ")[0] for line in lines if " of " in line]
        self.assertEqual(labels, ["weight 0.5", "weight 2", "weight 4",
                                  "RND, archived baseline:\\newline weight 1, seeds 1 to 2"])
        self.assertIn("not yet (0/1)", lines[labels.index("weight 4") + 4])
        self.assertIn("not recorded", lines[-3])
        self.assertEqual(lines[-4], "\\midrule")

    def test_empty_block(self):
        text = make_all.early_tabular([], self.baseline)
        self.assertIn("no run of this block had started", text)
        self.assertIn("archived baseline", text)


class TrainedTest(unittest.TestCase):
    """A unit claimed by a worker but still before its first update does not count as started."""

    def test_claimed_unit_not_counted(self):
        compiling = configuration([run({"left_start_room": None}, update=0)], {"intrinsic_weight": 1.0})
        training = configuration([run({"left_start_room": None}, update=5)], {"intrinsic_weight": 2.0})
        self.assertEqual([row["label"] for row in make_all.started_rows([compiling, training])], ["weight 2"])


class SpliceTest(unittest.TestCase):
    """Only the text between one table's markers changes; a missing or repeated marker stops the generator."""

    TEXT = "a\n% >>> AUTO-GENERATED TABLE START: x\nold\n% <<< AUTO-GENERATED TABLE END: x\nb"

    def test_replaces_between_markers(self):
        self.assertEqual(make_all.splice(self.TEXT, "x", "new"),
                         "a\n% >>> AUTO-GENERATED TABLE START: x\nnew\n% <<< AUTO-GENERATED TABLE END: x\nb")

    def test_missing_marker(self):
        with self.assertRaises(SystemExit):
            make_all.splice(self.TEXT, "y", "new")

    def test_repeated_marker(self):
        with self.assertRaises(SystemExit):
            make_all.splice(self.TEXT + self.TEXT, "x", "new")


class MillionTest(unittest.TestCase):
    """Two decimals below a billion steps, whole millions above."""

    def test_formats(self):
        self.assertEqual(make_all.million(162176), "0.16")
        self.assertEqual(make_all.million(2_000_015_360), "2,000")


if __name__ == "__main__":
    unittest.main()
