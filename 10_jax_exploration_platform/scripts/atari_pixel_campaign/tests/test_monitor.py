"""Verify readable status reports using metadata records and no numerical imports."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitor


class MonitorTests(unittest.TestCase):
    """Keep finished outcomes, current round counts, and held jobs unambiguous."""

    def test_paired_round_count_is_one_based(self):
        """The running round already starts at one and must not be incremented."""
        value = {"state": "running", "variant": "candidate", "round": 7, "total_rounds": 7}
        self.assertEqual(monitor.measurement_progress(value, None), "candidate 7/7 rounds")
        self.assertIsNone(monitor.measurement_progress({"state": "initializing"}, None))

    def test_completed_case_shows_outcome_and_no_fictitious_rounds(self):
        """A measured case has a result while a correctness rejection has no timed pairs."""
        result = {"case_completed": True, "decision": "accepted", "paired_rounds": 7,
                  "median_speedup": 1.110732001139316}
        self.assertEqual(monitor.measurement_progress({"state": "completed"}, result),
                         "7/7 pairs; accepted; +11.1%")
        result = {"case_completed": True, "decision": "rejected", "failure_stage": "correctness"}
        self.assertEqual(monitor.measurement_progress({}, result), "rejected")

    def test_table_wraps_without_discarding_words_or_gpu_type(self):
        """All lines fit the terminal while the full mechanism remains readable."""
        name = "reconstruct stacks from individual frames (2 copies)"
        row = monitor.table_row(name, "jaguar03", monitor.gpu_label("NVIDIA RTX A4500"),
                                "RUNNING", "candidate 7/7 rounds")
        self.assertTrue(all(len(line) <= 100 for line in row.splitlines()))
        self.assertIn("1 RTX A4500", row)
        self.assertIn("individual", row)
        self.assertIn("frames (2 copies)", row)
        self.assertEqual(monitor.gpu_label("unknown"), "unknown")

    def test_unit_rows_use_specification_and_completed_result(self):
        """For example, an opaque case directory becomes cache room backgrounds (1 copy)."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unit = root / "data" / "immutable_case"
            unit.mkdir(parents=True)
            specifications = root / "queue" / "specifications"
            specifications.mkdir(parents=True)
            (specifications / "immutable_case.json").write_text(json.dumps({
                "hypothesis": "cache_room_backgrounds", "configuration": {"n_copies": 1}}))
            (unit / "progress.json").write_text(json.dumps({"state": "completed"}))
            (unit / "result.json").write_text(json.dumps({"case_completed": True,
                "decision": "accepted", "paired_rounds": 7, "median_speedup": 1.11}))
            rows = monitor.unit_rows(root)
        self.assertEqual(rows[0]["run"], "cache room backgrounds (1 copy)")
        self.assertEqual(rows[0]["progress"], "7/7 pairs; accepted; +11.0%")

    def test_held_job_is_not_printed_in_waiting_group(self):
        """A held job requires release and is listed only in the held section."""
        jobs = [["123", "PENDING", "", "gpu:1", "JobHeldUser"]]
        with tempfile.TemporaryDirectory() as temporary, patch.object(monitor, "own_jobs", return_value=jobs):
            root = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()):
                monitor.tick(root)
            report = (root / "20_mins_monitoring" / "latest_report.md").read_text()
        waiting = report.split("WAITING\n", 1)[1].split("HELD\n", 1)[0]
        held = report.split("HELD\n", 1)[1].split("FINISHED\n", 1)[0]
        self.assertNotIn("JobHeldUser", waiting)
        self.assertIn("JobHeldUser", held)

    def test_runner_exit_and_later_attempt_remain_distinct(self):
        """An interrupted first attempt must not mark its live retry as failed."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unit = root / "data" / "case"
            first = unit / "attempts" / "attempt-0001"
            first.mkdir(parents=True)
            failure = json.dumps({"failure_stage": "runner_exit", "exit_code": -9})
            (first / "failure.json").write_text(failure)
            (unit / "failure.json").write_text(failure)
            (unit / "progress.json").write_text(json.dumps({"state": "failed", "attempt": 1}))
            rows = monitor.unit_rows(root)
            self.assertEqual(rows[0]["progress"], "runner exit; exit -9")
            (unit / "progress.json").write_text(json.dumps({"state": "initializing", "attempt": 2,
                                                            "variant": "reference"}))
            rows = monitor.unit_rows(root)
            self.assertEqual(rows[0]["state"], "initializing")
            self.assertEqual(rows[0]["progress"], "initializing; reference")


if __name__ == "__main__":
    unittest.main()
