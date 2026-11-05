"""Check direct preparation/registration display without starting a process or qualifying a result."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("monitor_direct_display_under_test", SCRIPTS / "monitor.py")
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)
GPU = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"


class DirectDisplayTests(unittest.TestCase):
    """Keep absent/prepared/unknown metadata distinct from recorded owned execution."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name); self.unit = self.run / "data/direct_common_copy_one"
        self.unit.mkdir(parents=True)
        self.write(self.run / "processes.json", []); self.write(self.run / "next_actions.json", {})
        self.plan = {"schema_version": 1, "kind": "selected_configuration_measurement", "mode": "confirmation",
            "campaign_run_dir": str(self.run), "unit_id": self.unit.name, "hardware_class": "h100", "gpu_uuid": GPU,
            "source_root": "/original/numerical_source", "source_commit": "d" * 40, "helpers": {"source_commit": "f" * 40},
            "direct_binding": {"schema_version": 1, "consumer": "selected_measurement",
                "profile": {"hostname": "serval02", "gpu_uuid": GPU, "uid": 2618919, "model": "NVIDIA H100 NVL"},
                "current_launch_inputs_path": str(self.unit / "direct_current_launch.json")}}
        self.plan_reference = self.write(self.unit / "measurement_plan.json", self.plan)
        self.write(self.unit / "measurement_prepared.json", {"plan_sha256": self.plan_reference["sha256"]})
        self.row = {"hostname": "serval02", "gpu_uuid": GPU, "unit": self.unit.name, "kind": self.plan["kind"],
            "source_root": self.plan["source_root"], "source_commit": self.plan["source_commit"],
            "helper_commit": self.plan["helpers"]["source_commit"], "pid": 12345, "start_time": "67890"}

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(); path.write_bytes(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    def register(self, duplicate=False):
        rows = [self.row, self.row] if duplicate else [self.row]
        ledger = self.write(self.unit / "direct_current_process_ledger_snapshot.json", rows)
        launch = {"schema_version": 1, "kind": "serval02_card1_current_launch", "plan": self.plan_reference,
            "process_row": self.row, "process_ledger": {"source_path": str(self.run / "processes.json"), "snapshot": ledger}}
        self.write(self.unit / "direct_current_launch.json", launch)
        self.write(self.run / "processes.json", [self.row])

    def hardware(self, **changes):
        value = {"hostname": "serval02", "pid": self.row["pid"], "process_start_time": self.row["start_time"],
            "gpu_uuid": GPU, "gpu_query": GPU + ", NVIDIA H100 NVL, 95830 MiB, 590.48.01",
            "cuda_visible_devices": GPU, "slurm_job_id": ""}
        value.update(changes); self.write(self.unit / "hardware.json", value)

    def displayed(self):
        rows = monitor.unit_rows(self.run)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["slurm_job_id"], "")
        self.assertEqual(row.get("scientific_credit"), 0)
        self.assertIs(row.get("admission_authority"), False)
        return row

    def test_prepared_without_progress_is_visible_once_and_not_running(self):
        self.assertFalse((self.unit / "progress.json").exists())
        row = self.displayed(); self.assertEqual(row["state"], "planned")
        self.assertIn("registration and launch not recorded", row["progress"])

    def test_prepared_phase_names_stay_planned(self):
        for phase in ("prepared", "prepared_not_submitted", "planned"):
            self.write(self.unit / "progress.json", {"state": phase})
            with self.subTest(phase=phase):
                self.assertEqual(self.displayed()["state"], "planned")

    def test_bound_registration_without_numerical_progress_is_planned(self):
        self.register()
        row = self.displayed(); self.assertEqual(row["state"], "planned")
        self.assertIn("numerical progress not observed", row["progress"])

    def test_controller_metadata_binding_does_not_claim_registration_or_running(self):
        packet = self.run / "remote_setup" / self.unit.name
        request = self.write(packet / "preparation_request.json", {
            "kind": "direct_common_timing_preparation_request", "campaign_run_dir": str(self.run),
            "packet": str(packet), "data_unit": str(self.unit), "unit_id": self.unit.name})
        bound = {"schema_version": 1, "kind": "actual_prepared_direct_common_timing_controller_inputs",
            "plan": self.plan_reference, "request": request}
        self.write(packet / "controller_bound_metadata.json", bound)
        row = self.displayed(); self.assertEqual(row["state"], "planned")
        self.assertIn("controller metadata bound", row["progress"])
        self.assertFalse((self.unit / "direct_current_launch.json").exists())
        bound["plan"] = {**self.plan_reference, "sha256": "0" * 64}
        self.write(packet / "controller_bound_metadata.json", bound)
        self.assertEqual(self.displayed()["state"], "unknown")

    def test_running_needs_registration_progress_and_matching_actual_hardware(self):
        self.write(self.unit / "progress.json", {"state": "running", "round": 1, "total_rounds": 7, "variant": "reference"})
        self.assertEqual(self.displayed()["state"], "unknown")
        self.register(); self.assertEqual(self.displayed()["state"], "unknown")
        self.hardware(); row = self.displayed()
        self.assertEqual(row["state"], "running"); self.assertTrue(row["direct_metadata_display_only"])

    def test_unknown_phase_and_duplicate_owner_are_not_running(self):
        self.register(); self.hardware()
        self.write(self.unit / "progress.json", {"state": "unknown_future_phase"})
        self.assertEqual(self.displayed()["state"], "unknown")
        self.write(self.unit / "progress.json", {"state": "running"}); self.register(duplicate=True)
        self.assertEqual(self.displayed()["state"], "unknown")

    def test_direct_metadata_cannot_claim_a_slurm_identifier_or_other_birth(self):
        self.register(); self.write(self.unit / "progress.json", {"state": "running"})
        for changes in ({"slurm_job_id": "999999"}, {"process_start_time": "wrong"}, {"pid": 54321}):
            self.hardware(**changes)
            with self.subTest(changes=changes):
                self.assertEqual(self.displayed()["state"], "unknown")

    def test_initial_direct_worker_without_selected_plan_keeps_existing_display(self):
        (self.unit / "measurement_plan.json").unlink(); (self.unit / "measurement_prepared.json").unlink()
        self.write(self.unit / "progress.json", {"state": "running", "update": 3, "target_updates": 10})
        self.hardware(); self.write(self.run / "processes.json", [self.row])
        rows = monitor.unit_rows(self.run)
        self.assertEqual(len(rows), 1); self.assertEqual(rows[0]["state"], "running")
        self.assertNotIn("direct_metadata_display_only", rows[0])


if __name__ == "__main__":
    unittest.main()
