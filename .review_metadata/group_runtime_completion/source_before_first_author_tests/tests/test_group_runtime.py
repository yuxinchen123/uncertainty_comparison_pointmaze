"""Exercise group runtime metadata without a numerical package, scheduler or GPU."""
import copy
import importlib
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "tests"))
import group_runtime as runtime
import group_runtime_evidence as evidence
import test_configured_a4500_group as existing


class RuntimePlanTests(existing.GroupMetadataTests):
    """Extend the independently structured group fixture with explicit current runtime fields."""
    def setUp(self):
        """Create synthetic plans only, preserving the actual shared source tree."""
        super().setUp()
        self.plan.update(job_id=None, metadata_python="/usr/bin/python3", python="/synthetic/environment/bin/python",
            wall_seconds=3600, reference_configuration=copy.deepcopy(self.plan["configuration"]), barrier_timeout_seconds=1200)
        self.write(self.unit / "scaling_plan.json", self.plan)
        wanted = runtime.group.receipt(self.unit / "scaling_plan.json")
        submission = json.loads((self.unit / "group_submission.json").read_text())
        submission["plan_sha256"] = wanted["sha256"]
        self.write(self.unit / "group_submission.json", submission)
        release = json.loads((self.unit / "group_release.json").read_text())
        release.update(plan_sha256=wanted["sha256"],
            submission_receipt_sha256=runtime.group.receipt(self.unit / "group_submission.json")["sha256"])
        self.write(self.unit / "group_release.json", release)
        owned = json.loads((self.unit / "owned_group_inputs.json").read_text())
        owned.update(plan=wanted, submission=runtime.group.receipt(self.unit / "group_submission.json"),
            release=runtime.group.receipt(self.unit / "group_release.json"))
        self.write(self.unit / "owned_group_inputs.json", owned)
        self.metadata.update(inputs=owned, inputs_receipt=runtime.group.receipt(self.unit / "owned_group_inputs.json"))
        self.binding.update(plan_sha256=wanted["sha256"], metadata_sha256=runtime.group.digest(runtime.group.encoded(self.metadata)))

    def test_runtime_valid_full_measurement_plan(self):
        """A complete explicit plan passes metadata validation with unresolved physical identities."""
        self.assertEqual(runtime.validate_plan(self.plan), self.plan["configuration"])

    def test_runtime_schema_one_and_guessed_identity_refuse(self):
        """Historical schemas or guessed future job/device identities never enable current execution."""
        for key, value in (("schema_version", 1), ("job_id", "999999"), ("gpu_uuids", ["GPU-guessed"] * 8)):
            plan = copy.deepcopy(self.plan); plan[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runtime.validate_plan(plan)

    def test_literal_held_resources_and_duration(self):
        """Seconds become HH:MM:SS and an explicit reservation remains in the held request."""
        command = runtime.held_submission_command(self.plan, Path("/synthetic/job.sbatch"))
        self.assertIn("--hold", command)
        self.assertIn("--no-requeue", command)
        self.assertIn("--gres=gpu:nvidia_rtx_a4500:8", command)
        self.assertIn("--time=1:00:00", command)
        self.assertIn("--reservation=synthetic_group_reservation", command)
        self.assertIn("--qos=csresnolim", command)

    def test_unreserved_quality_of_service_and_invalid_pair(self):
        """The unreserved shape uses normal; reservation-only quality of service cannot be implicit."""
        self.plan["placement"].update(reservation=None, qos="normal")
        command = runtime.held_submission_command(self.plan, Path("/synthetic/job.sbatch"))
        self.assertIn("--qos=normal", command)
        self.assertFalse(any(x.startswith("--reservation=") for x in command))
        self.plan["placement"]["qos"] = "csresnolim"
        with self.assertRaises(ValueError):
            runtime.validate_plan(self.plan)

    def test_distinct_python_and_zero_one_step_requests(self):
        """Only the standard-library coordinator uses -S; the worker retains scientific Python."""
        coordinator = runtime.numeric_command(self.plan, "coordinator")
        worker = runtime.numeric_command(self.plan, "worker", 3)
        self.assertEqual(coordinator[:4], ["/usr/bin/python3", "-I", "-S", "-B"])
        self.assertEqual(worker[:3], [self.plan["python"], "-I", "-B"])
        self.assertNotIn("-S", worker)
        self.assertEqual(worker[-2:], ["--worker-index", "3"])
        one = runtime.producer.step_command(self.plan, self.old.job, "worker", worker)
        zero = runtime.producer.step_command(self.plan, self.old.job, "coordinator", coordinator)
        self.assertIn("--gres=gpu:nvidia_rtx_a4500:1", one)
        self.assertIn("--gres=none", zero)

    def test_wrong_uid_and_reservation_metadata_refuse(self):
        """All four UIDs and actual native reservation must match, rather than only real UID."""
        for field, value in (("uids", [2618919, 2618919, 2618919, 0]),):
            metadata = copy.deepcopy(self.metadata); metadata[field] = value
            with self.assertRaises(ValueError):
                runtime.group.validate_metadata(self.plan, metadata)
        metadata = copy.deepcopy(self.metadata)
        metadata["job"]["stdout"] = metadata["job"]["stdout"].replace("Reservation=synthetic_group_reservation", "Reservation=another")
        self.old.rehash_query(metadata["job"])
        with self.assertRaises(ValueError):
            runtime.group.validate_metadata(self.plan, metadata)

    def test_missing_launch_review_refuses_without_process(self):
        """No source-only or old registered plan can launch wrappers without current review."""
        with patch.object(runtime.subprocess, "Popen") as create:
            with self.assertRaises(ValueError):
                runtime.require_launch_review(self.plan)
        create.assert_not_called()

    def test_reused_preparation_refuses_before_writes(self):
        """An existing immutable unit cannot be reused even while it has no completed result."""
        with self.assertRaisesRegex(ValueError, "preserve"):
            runtime.prepare(self.plan, "/synthetic/backend.json")


class CounterTests(unittest.TestCase):
    """Check actual compilation warmup and all seven update windows without numerical arrays."""
    def setUp(self):
        """Use two ten-update warmups and subsequent full ten-update windows."""
        self.warmup = [{"call": x, "updates": 10, "seconds": .25, "compiled_signatures": [1]}
            for x in (1, 2)]
        self.result = {"warmup": self.warmup, "training_start_iteration": 20,
            "rounds": [{"training_start_iteration": 20, "report_origin_iteration": 20 + index * 10,
                "actual_end_iteration": 30 + index * 10, "actual_optimizer_step": (30 + index * 10) * 16}
                for index in range(7)]}

    def test_real_warmup_origin_and_seven_windows(self):
        """Two full compilation calls imply origin20 and measured end90, never a reset to zero."""
        self.assertEqual(evidence.validate_round_counters({"iterations": 10}, self.result), 20)

    def test_reset_zero_omitted_or_changed_warmup_refuse(self):
        """A padded, missing, type-substituted or unstable compilation call cannot be hidden."""
        for mutate in (lambda x: x.update(training_start_iteration=0),
                lambda x: x["warmup"].pop(),
                lambda x: x["warmup"][1].update(updates=True),
                lambda x: x["warmup"][1].update(compiled_signatures=[2]),
                lambda x: x["warmup"][0].update(seconds=float("nan"))):
            result = copy.deepcopy(self.result); mutate(result)
            with self.assertRaises(ValueError):
                evidence.validate_round_counters({"iterations": 10}, result)

    def test_round_counter_change_and_float_substitution_refuse(self):
        """Actual group rows must continue from the same integer origin and complete optimizer work."""
        for key, value in (("report_origin_iteration", 0), ("actual_end_iteration", 30.),
                ("actual_optimizer_step", 479), ("training_start_iteration", True)):
            result = copy.deepcopy(self.result); result["rounds"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                evidence.validate_round_counters({"iterations": 10}, result)


class ProcessControlTests(unittest.TestCase):
    """Exercise exact process records and startup failure custody with synthetic native handles."""
    def test_missing_controller_api_refuses_before_wrapper_creation(self):
        """Missing process-descriptor support must fail before Popen, not after launching a child."""
        with patch.object(runtime.os, "pidfd_open", None), patch.object(runtime.subprocess, "Popen") as create:
            with self.assertRaises(ValueError):
                runtime.start_wrapper({}, "coordinator", registry=[])
        create.assert_not_called()

    def test_actual_process_record_strict_types_and_all_uids(self):
        """The numerical process cannot be substituted by its wrapper PID or a floating PID."""
        process = {"pid": 100, "parent_pid": 99, "process_start_time": "123", "cgroup": "own"}
        metadata = {"process": process}
        value = {**process, "uids": [2618919] * 4, "argv": ["/selected/python", "worker.py"]}
        self.assertEqual(runtime.validate_process(value, metadata, value["argv"]), value)
        for key, changed in (("pid", 100.), ("parent_pid", 99.), ("uids", [2618919] * 3 + [0]),
                ("argv", ["/unapproved/python", "worker.py"]), ("cgroup", "foreign")):
            altered = copy.deepcopy(value); altered[key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                runtime.validate_process(altered, metadata, value["argv"])

    def test_startup_open_failure_keeps_child_in_registry(self):
        """A child exists in controller custody before a later handle-open failure can raise."""
        with tempfile.TemporaryDirectory() as directory:
            unit = Path(directory); (unit / "wrapper_records").mkdir()
            fake = SimpleNamespace(pid=12345)
            registry = []
            with patch.object(runtime, "process_control_apis"), patch.object(runtime.group, "owned_inputs", return_value={"job_id": "777"}), \
                    patch.object(runtime.producer, "step_command", return_value=["synthetic-srun"]), \
                    patch.object(runtime, "numeric_command", return_value=["synthetic-worker"]), \
                    patch.object(runtime.group, "unit_directory", return_value=unit), \
                    patch.object(runtime, "process_identity", return_value={"pid": 100}), \
                    patch.object(runtime.subprocess, "Popen", return_value=fake), \
                    patch.object(runtime.os, "pidfd_open", side_effect=OSError("synthetic handle failure")):
                with self.assertRaises(OSError):
                    runtime.start_wrapper({}, "coordinator", registry=registry)
            self.assertEqual(len(registry), 1)
            self.assertIs(registry[0]["process"], fake)
            self.assertIsNone(registry[0]["descriptor"])
            self.assertFalse(registry[0]["waited"])
            registry[0]["stdout"].close(); registry[0]["stderr"].close()

    def test_exact_wrapper_wait_uses_only_pinned_handle(self):
        """A cleanup signal targets the saved descriptor and still waits for its exact child."""
        with tempfile.TemporaryDirectory() as directory:
            unit = Path(directory); (unit / "wrapper_records").mkdir()
            out, err = (unit / "out").open("wb"), (unit / "err").open("wb")
            process = Mock(); process.poll.return_value = None; process.wait.return_value = -15
            wrapper = {"name": "worker_1", "process": process, "descriptor": 71,
                "identity": {"pid": 123}, "stdout": out, "stderr": err, "waited": False}
            with patch.object(runtime.group, "unit_directory", return_value=unit), \
                    patch.object(runtime.signal, "pidfd_send_signal") as send, patch.object(runtime.os, "close") as close:
                self.assertEqual(runtime.finish_wrapper({"barrier_timeout_seconds": 1}, wrapper, terminate=True), -15)
            send.assert_called_once_with(71, signal.SIGTERM, None, 0)
            close.assert_called_once_with(71)
            self.assertTrue(wrapper["waited"])
            saved = json.loads((unit / "wrapper_records/worker_1_wait.json").read_text())
            self.assertTrue(saved["terminated_for_failure"])
            self.assertFalse(saved["is_numerical_wait"])

    def test_default_runtime_entrypoint_is_inactive(self):
        """No command-line default can launch a controller, coordinator or numerical worker."""
        with patch.object(runtime.subprocess, "Popen") as create:
            with self.assertRaises(ValueError):
                runtime.main([])
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
