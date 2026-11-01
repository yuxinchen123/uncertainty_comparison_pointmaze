"""Check infrastructure choices and reporting arithmetic without importing numerical runtimes."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import atomic_json, timing_summary
from ideas import choose_adaptive_ideas, initial_ideas, resolve_idea


def records_for_choices():
    """Supply complete four-case measurements with deliberately different improvements."""
    # Before: registry choices; after: explicit rates for both policies and both geometries.
    documents = []
    for index, entry in enumerate(initial_ideas()):
        cases = [{"policy_kind": policy, "geometry": geometry, "completed": True,
                  "steps_per_second_total": 100 * (1 + index / 100),
                  "correctness": {"passed": True}}
                 for policy in ("tabular", "neural") for geometry in ("straight", "snake")]
        documents.append({"idea": entry["name"], "completed": True, "cases": cases})
    return documents


class RegistryTests(unittest.TestCase):
    """Keep every measured choice distinct and require evidence for adaptive combinations."""

    def test_twenty_independent_choices_and_copies(self):
        """Require twenty named candidates, baseline, and one change per initial candidate."""
        entries = initial_ideas()
        self.assertEqual(len(entries), 21)
        self.assertEqual(len({entry["name"] for entry in entries}), 21)
        self.assertTrue(all(len(entry["changes"]) == 1 for entry in entries[1:]))
        entries[0]["changes"]["unexpected"] = True
        self.assertEqual(resolve_idea("baseline")["changes"], {})

    def test_missing_or_repeated_choice_is_rejected(self):
        """Prevent misspelled or multiply defined choices from silently selecting a setting."""
        with self.assertRaises(ValueError):
            resolve_idea("unknown")
        with self.assertRaises(ValueError):
            resolve_idea("baseline", [resolve_idea("baseline")])

    def test_adaptive_choices_follow_measured_results(self):
        """Prefer measured improvements and record concrete reasons for ten unique combinations."""
        documents = records_for_choices()
        entries = choose_adaptive_ideas(documents)
        self.assertEqual(len(entries), 10)
        self.assertEqual(len({entry["name"] for entry in entries}), 10)
        self.assertTrue(all(len(entry["changes"]) == 2 for entry in entries))
        self.assertTrue(all("unmeasured" in entry["rationale"] for entry in entries))
        first = entries[0]["constituents"]
        self.assertIn("flat_state_prediction_gather", first)
        # A failed case removes its candidate from every adaptive combination.
        target = next(row for row in documents if row["idea"] == "flat_state_prediction_gather")
        target["cases"][0]["correctness"]["passed"] = False
        revised = choose_adaptive_ideas(documents)
        self.assertTrue(all("flat_state_prediction_gather" not in row["constituents"] for row in revised))

    def test_adaptive_choice_requires_complete_evidence(self):
        """Reject absent baselines and candidates without enough checked cases."""
        with self.assertRaises(ValueError):
            choose_adaptive_ideas([])
        with self.assertRaises(ValueError):
            choose_adaptive_ideas(records_for_choices()[:2])
        with self.assertRaises(ValueError):
            choose_adaptive_ideas(records_for_choices(), count=11)


class ReportingTests(unittest.TestCase):
    """Check rates and atomic JSON publication independently of the GPU program."""

    def test_three_blocks_report_total_and_per_copy(self):
        """Use the median iteration duration to retain the required throughput units."""
        blocks = [{"seconds": 2, "iterations": 4}, {"seconds": 3, "iterations": 6},
                  {"seconds": 8, "iterations": 8}]
        summary = timing_summary(blocks, copies=8, rollout_steps=16)
        self.assertEqual(summary["seconds_per_iteration"], 0.5)
        self.assertEqual(summary["steps_per_second_total"], 256)
        self.assertEqual(summary["steps_per_second_per_copy"], 32)
        self.assertAlmostEqual(summary["hours_per_million_steps_per_copy"], 1_000_000 / 115_200)

    def test_empty_or_invalid_timing_blocks_are_rejected(self):
        """Never turn incomplete or zero-duration measurements into a valid throughput."""
        with self.assertRaises(ValueError):
            timing_summary([], 8, 16)
        with self.assertRaises(ValueError):
            timing_summary([{"seconds": 0, "iterations": 1}] * 3, 8, 16)

    def test_atomic_json_replaces_prior_record(self):
        """Persist actual booleans and newlines and reject non-finite JSON values."""
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "nested" / "record.json"
            atomic_json(target, {"completed": False})
            atomic_json(target, {"completed": True})
            self.assertEqual(json.loads(target.read_text()), {"completed": True})
            self.assertFalse(target.with_name("record.json.temporary").exists())
            with self.assertRaises(ValueError):
                atomic_json(target, {"rate": float("inf")})


if __name__ == "__main__":
    unittest.main()
