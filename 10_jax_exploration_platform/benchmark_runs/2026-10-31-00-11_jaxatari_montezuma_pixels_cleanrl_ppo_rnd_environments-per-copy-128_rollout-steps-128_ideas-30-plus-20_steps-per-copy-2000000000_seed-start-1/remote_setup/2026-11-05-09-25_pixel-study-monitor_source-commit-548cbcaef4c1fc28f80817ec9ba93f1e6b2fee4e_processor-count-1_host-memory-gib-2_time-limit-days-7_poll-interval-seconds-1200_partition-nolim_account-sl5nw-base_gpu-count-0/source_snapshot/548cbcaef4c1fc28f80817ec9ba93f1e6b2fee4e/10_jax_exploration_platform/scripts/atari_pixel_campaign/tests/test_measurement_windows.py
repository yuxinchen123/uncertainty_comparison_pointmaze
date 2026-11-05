"""Check paired measurement metadata using complete updates and no numerical packages."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import admit_measurements
import controller
import ideas
import test_admit_measurements as admission_fixtures


class MeasurementWindowTests(unittest.TestCase):
    """Keep all thirty hypotheses while varying only the length of their paired rounds."""

    def setUp(self):
        """Represent the frozen scientific source and the unchanged original learner settings."""
        self.ready = {
            "campaign_run_dir": "/synthetic_campaign", "source_root": "/synthetic_source",
            "source_commit": "d28e6735e79788f618acb0ebfde8609db1f776af",
            "reference_configuration": {**controller.REFERENCE_ALGORITHM, "n_copies": 1},
        }

    def test_all_thirty_default_windows_preserve_seven_rounds_and_science(self):
        """Only grouping and reporting require more than one full training update."""
        # Before: thirty catalog hypotheses; after: thirty cases with windows 1, 5, or 10.
        cases = [controller.immutable_case(entry, "h100", 1, self.ready)
                 for entry in ideas.INITIAL_IDEAS]
        self.assertEqual(len(cases), 30)
        for entry, case in zip(ideas.INITIAL_IDEAS, cases):
            with self.subTest(position=entry.position):
                self.assertEqual(case["iterations"], {27: 5, 29: 10}.get(entry.position, 1))
                self.assertEqual(case["rounds"], 7)
                self.assertEqual(case["seed_start"], 1)
                self.assertEqual(case["source_commit"], self.ready["source_commit"])
                self.assertEqual(case["configuration"], self.ready["reference_configuration"])
                self.assertEqual(case["candidate"], {"learner": entry.learner,
                    "environment": entry.environment, "execution": entry.execution})
                self.assertEqual(case["reference_execution"], entry.reference_execution)
                self.assertNotIn("reference_environment", case)
                self.assertLessEqual(len(case["unit_id"] + ".json"), 255)

    def test_windows_cover_both_arms_without_padding(self):
        """Round up to a common multiple rather than dropping or adding hidden updates."""
        grouping = ideas.INITIAL_IDEAS[26]
        self.assertEqual(controller.compatible_updates_per_round(grouping, 6), 10)
        both = replace(grouping, execution={"iterations_per_call": 3},
                       reference_execution={"iterations_per_call": 2})
        self.assertEqual(controller.compatible_updates_per_round(both), 6)
        reporting = replace(grouping, execution={"report_readout": "scheduled",
            "report_every_iterations": 6}, reference_execution={"report_readout": "per_update",
            "report_every_iterations": 10})
        self.assertEqual(controller.compatible_updates_per_round(reporting), 30)
        inactive = replace(grouping, execution={"report_every_iterations": 10})
        self.assertEqual(controller.compatible_updates_per_round(inactive), 1)

    def test_case_identity_and_digest_include_effective_update_count(self):
        """Different measured windows cannot reuse an immutable case path or digest."""
        entry = ideas.INITIAL_IDEAS[0]
        short = controller.immutable_case(entry, "h100", 1, self.ready)
        long = controller.immutable_case(entry, "h100", 1, {**self.ready, "iterations": 10})
        self.assertIn("__updates-per-round-1__", short["unit_id"])
        self.assertIn("__updates-per-round-10__", long["unit_id"])
        self.assertNotEqual(short["unit_id"], long["unit_id"])
        self.assertNotEqual(short["specification_digest"], long["specification_digest"])
        self.assertEqual(short["configuration"], long["configuration"])
        self.assertEqual(short["seed_start"], long["seed_start"])

    def test_case_filenames_sort_by_original_catalog_position(self):
        """The frozen worker's alphabetical order follows the catalog and retains full names."""
        # Before: filenames sorted by mechanism name; after: original positions 01 through 30.
        cases = [controller.immutable_case(entry, context, 1, self.ready)
                 for entry in ideas.INITIAL_IDEAS for context in ideas.REQUIRED_HARDWARE]
        ordered = sorted(cases, key=lambda case: case["unit_id"])
        expected = [entry.position for entry in ideas.INITIAL_IDEAS for _ in ideas.REQUIRED_HARDWARE]
        self.assertEqual([case["hypothesis_position"] for case in ordered], expected)
        for case in ordered:
            self.assertTrue(case["unit_id"].startswith(
                f"idea-position-{case['hypothesis_position']:02d}__{case['hypothesis']}__"))
            self.assertLess(len((case["unit_id"] + ".json").encode()), 255)

    def test_invalid_counts_and_too_few_rounds_fail(self):
        """Require positive exact integers and retain at least seven paired rounds."""
        entry = ideas.INITIAL_IDEAS[0]
        for count in (0, -1, True, 1.5, "1"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                controller.immutable_case(entry, "h100", 1, {**self.ready, "iterations": count})
        for field in ("iterations_per_call", "report_every_iterations"):
            invalid = replace(entry, execution={field: 0, "report_readout": "scheduled"})
            with self.subTest(field=field), self.assertRaises(ValueError):
                controller.compatible_updates_per_round(invalid)
        with self.assertRaises(ValueError):
            controller.immutable_case(entry, "h100", 1, {**self.ready, "rounds": 6})

    def test_admission_records_default_and_explicit_minimum(self):
        """Publish seven-round metadata with default one or an explicit longer confirmation."""
        for minimum in (None, 10):
            fixture = admission_fixtures.AdmissionTests("runTest")
            fixture.setUp()
            self.addCleanup(fixture.doCleanups)
            if minimum is not None:
                fixture.arguments.iterations_per_round = minimum
            fixture.run_admission()
            sentinel = controller.read_json(fixture.run_dir / controller.SENTINEL_NAME)
            self.assertEqual(sentinel["rounds"], 7)
            self.assertEqual(sentinel["iterations"], 1 if minimum is None else minimum)
            self.assertEqual(sentinel["source_commit"], fixture.commit)
            self.assertEqual(sentinel["seed_start"], 1)

    def test_admission_rejects_invalid_minimum_before_publication(self):
        """An invalid update window cannot create the source-admission sentinel."""
        fixture = admission_fixtures.AdmissionTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.arguments.iterations_per_round = 0
        with self.assertRaisesRegex(ValueError, "positive integer"):
            fixture.run_admission()
        self.assertFalse((fixture.run_dir / controller.SENTINEL_NAME).exists())
        self.assertFalse((fixture.run_dir / "data/source_checks").exists())

    def test_admission_cli_defaults_to_one_update(self):
        """Expose the shorter default and retain an explicit ten-update confirmation option."""
        arguments = [part for name in ("run-dir", "source-root", "source-commit",
            "reference-configuration-source", "h100-learner-record", "h100-renderer-record",
            "a4500-learner-record", "a4500-renderer-record") for part in ("--" + name, "/fixture")]
        arguments.extend(("--matched-copy-counts", "1"))
        parser = admit_measurements.argument_parser()
        self.assertEqual(parser.parse_args(arguments).iterations_per_round, 1)
        self.assertEqual(parser.parse_args(arguments + ["--iterations-per-round", "10"])
                         .iterations_per_round, 10)


if __name__ == "__main__":
    unittest.main()
