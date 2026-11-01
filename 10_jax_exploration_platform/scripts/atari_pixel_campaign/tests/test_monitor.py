"""Verify readable status reports using metadata records and no numerical imports."""

import contextlib
import hashlib
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

    def test_long_auxiliary_name_stays_separate_from_registered_case(self):
        """A completed 252-byte auxiliary name adds no paired-case credit and survives both metadata lookups."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            auxiliary_name = "checkpoint_restart_validation_" + "a" * (252 - len("checkpoint_restart_validation_"))
            self.assertEqual(len(auxiliary_name.encode()), 252)
            auxiliary = root / "data" / auxiliary_name
            formal = root / "data/registered_case"
            specifications = root / "queue/specifications"
            final_statuses = root / "final_training_queue/status"
            for path in (auxiliary, formal, specifications, final_statuses):
                path.mkdir(parents=True)
            (auxiliary / "progress.json").write_text(json.dumps({"state": "completed", "passed": True}))
            (auxiliary / "result.json").write_text(json.dumps({"passed": True, "scientific_hypothesis_credit": 0,
                "initial_case_credit": 0, "final_training_credit": 0}))
            (formal / "progress.json").write_text(json.dumps({"state": "completed"}))
            (formal / "result.json").write_text(json.dumps({"case_completed": True,
                "decision": "accepted", "paired_rounds": 7, "median_speedup": 1.11}))
            (specifications / "registered_case.json").write_text(json.dumps({
                "hypothesis": "cache_room_backgrounds", "configuration": {"n_copies": 1}}))
            (final_statuses / "registered_case.json").write_text(json.dumps({
                "unit_id": "registered_case", "state": "completed"}))
            rows = {row["run"]: row for row in monitor.unit_rows(root)}
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[auxiliary_name]["state"], "completed")
        self.assertEqual(rows[auxiliary_name]["progress"], "checks passed")
        self.assertEqual(rows["cache room backgrounds (1 copy)"]["progress"], "7/7 pairs; accepted; +11.0%")
        self.assertEqual(sum("pairs" in row["progress"] for row in rows.values()), 1)

    def test_malformed_registered_specification_still_raises(self):
        """Existing malformed input is still an error rather than being treated as an auxiliary unit."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unit = root / "data/registered_case"
            unit.mkdir(parents=True)
            specifications = root / "queue/specifications"
            specifications.mkdir(parents=True)
            (unit / "progress.json").write_text(json.dumps({"state": "completed"}))
            (specifications / "registered_case.json").write_text('{"hypothesis":')
            with self.assertRaises(json.JSONDecodeError):
                monitor.unit_rows(root)

    def test_training_gpu_uuid_selects_its_card_from_a_full_node_query(self):
        """Eight allocated cards remain distinguishable when a child records its UUID last."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unit = root / "data" / "final_training"
            unit.mkdir(parents=True)
            (unit / "progress.json").write_text(json.dumps({
                "state": "running", "update": 1, "target_updates": 122070}))
            hardware = {"hostname": "jaguar03", "cuda_visible_devices": "GPU-5",
                "gpu_query": "\n".join(
                    f"{index}, NVIDIA RTX A4500, 20470 MiB, 580.159.04, GPU-{index}"
                    for index in range(8))}
            (unit / "hardware.json").write_text(json.dumps(hardware))
            self.assertEqual(monitor.unit_rows(root)[0]["gpu"], ["NVIDIA RTX A4500"])
            # A different UUID must stay unknown instead of borrowing another card's identity.
            hardware["cuda_visible_devices"] = "GPU-99"
            (unit / "hardware.json").write_text(json.dumps(hardware))
            self.assertEqual(monitor.unit_rows(root)[0]["gpu"], [])

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

    def test_queued_initialization_reports_zero_measured_rounds(self):
        """Initialization is active work with zero of the seven paired rounds measured."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unit = root / "data" / "case"
            unit.mkdir(parents=True)
            specifications = root / "queue" / "specifications"
            specifications.mkdir(parents=True)
            (specifications / "case.json").write_text(json.dumps({"hypothesis": "cache_room_backgrounds",
                "configuration": {"n_copies": 2}, "rounds": 7}))
            (unit / "progress.json").write_text(json.dumps({"state": "initializing", "variant": "candidate"}))
            rows = monitor.unit_rows(root)
            (unit / "progress.json").write_text(json.dumps({"state": "correctness_passed"}))
            checked_rows = monitor.unit_rows(root)
        self.assertEqual(rows[0]["progress"], "0/7 rounds; initializing; candidate")
        self.assertEqual(checked_rows[0]["progress"], "0/7 rounds; correctness passed")


    def test_scaling_groups_show_the_proved_one_or_eight_physical_cards(self):
        """Recorded allocation cards and the immutable plan replace the unknown-GPU label."""
        for count in (1, 8):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                unit = root / "data/original_reference_scaling_component"
                unit.mkdir(parents=True)
                roster = ["GPU-physical-" + str(index) for index in range(count)]
                plan = unit / "scaling_plan.json"
                plan.write_text(json.dumps({"gpu_uuids": roster}))
                assignment = {"plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                    "hostname": "jaguar03", "gpu_uuids": roster, "allocation": {
                        "hostname": "jaguar03", "slurm_job_id": "9001", "cards": [
                            {"gpu_uuid": gpu, "physical_minor": str(index), "name": "NVIDIA RTX A4500"}
                            for index, gpu in enumerate(roster)]}}
                (unit / "assignment.json").write_text(json.dumps(assignment))
                (unit / "progress.json").write_text(json.dumps({"state": "completed", "rounds": 7}))
                row = monitor.unit_rows(root)[0]
                self.assertEqual(row["gpu"], [str(count) + "× RTX A4500"])
                self.assertEqual(monitor.gpu_label(row["gpu"][0]), row["gpu"][0])
                self.assertEqual(row["pool"], "jaguar03")
                self.assertEqual(row["slurm_job_id"], "9001")
                assignment["allocation"]["cards"][0]["gpu_uuid"] = "GPU-not-in-plan"
                (unit / "assignment.json").write_text(json.dumps(assignment))
                self.assertIsNone(monitor.recorded_scaling_group(unit))
                self.assertEqual(monitor.unit_rows(root)[0]["gpu"], [])

    def test_failed_final_unit_is_held_with_its_diagnosis_reason(self):
        """The final queue's explicit hold prevents a failed application looking eligible for retry."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unit = root / "data/explicit_final_policy"
            unit.mkdir(parents=True)
            (unit / "progress.json").write_text(json.dumps({"state": "failed", "attempt": 2}))
            status = root / "final_training_queue/status"
            status.mkdir(parents=True)
            (status / "explicit_final_policy.json").write_text(json.dumps({"unit_id": "explicit_final_policy", "state": "held",
                                                                          "reason": "current attempt requires diagnosis"}))
            row = monitor.unit_rows(root)[0]
            self.assertEqual(row["state"], "held")
            self.assertIn("requires diagnosis", row["progress"])

    def test_final_completion_and_waiting_summary_use_their_actual_report_sections(self):
        """Completed UVA units stay in FINISHED while a pending continuation remains in WAITING."""
        import final_training_monitor
        for state, section in (("finished", "FINISHED"), ("waiting", "WAITING")):
            value = {"state": state, "reason": "explicit continuation state", "counts": {"completed": 2}, "units": []}
            with self.subTest(state=state), tempfile.TemporaryDirectory() as temporary, \
                    patch.object(monitor, "own_jobs", return_value=[]), patch.object(final_training_monitor, "tick", return_value=value):
                root = Path(temporary)
                with contextlib.redirect_stdout(io.StringIO()):
                    monitor.tick(root)
                report = (root / "20_mins_monitoring/latest_report.md").read_text()
                chosen = report.split(section + "\n", 1)[1].split("\nPLANNED\n", 1)[0]
                self.assertIn("final training continuation", chosen)
                self.assertNotIn("final training continuation", report.split("PLANNED\n", 1)[1])

    def test_final_hook_receives_fresh_controller_state_and_reports_a_hold(self):
        """The final continuation decision follows this tick's controller before owned-job reporting."""
        import controller
        import final_training_monitor
        state = {"phase": "final_verification_pending"}
        held = {"state": "held", "reason": "raw scaling proof is stale", "units": []}
        with tempfile.TemporaryDirectory() as temporary, patch.object(monitor, "own_jobs", return_value=[]), \
                patch.object(controller, "tick", return_value=state), patch.object(final_training_monitor, "tick", return_value=held) as final_tick:
            root = Path(temporary)
            (root / "SOURCE_AND_GATES_READY.json").write_text("{}")
            with contextlib.redirect_stdout(io.StringIO()):
                monitor.tick(root)
            final_tick.assert_called_once_with(root, controller_state=state, controller_error=None)
            report = (root / "20_mins_monitoring/latest_report.md").read_text()
            held_section = report.split("HELD\n", 1)[1].split("FINISHED\n", 1)[0]
            self.assertIn("scaling proof is stale", " ".join(held_section.split()))
            result = json.loads((root / "20_mins_monitoring/latest_status.json").read_text())
            self.assertEqual(result["final_training"], held)


if __name__ == "__main__":
    unittest.main()
