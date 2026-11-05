"""Exercise group runtime metadata without a numerical package, scheduler or GPU."""
import copy
import importlib
import json
from pathlib import Path
import signal
import shutil
import subprocess
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
import group_runtime_launch as actor
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

    def test_actual_preparation_writes_all_standards_without_submission(self):
        """Committed private helper bytes stage into a complete unresolved packet with no scheduler operation."""
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "helper_source"
            scripts = source / runtime.group.PREFIX
            scripts.mkdir(parents=True)
            names = runtime.helper_names()
            for name in names:
                shutil.copyfile(SCRIPTS / name, scripts / name)
            for command in (["git", "init", "-q", str(source)], ["git", "-C", str(source), "add", "."],
                    ["git", "-C", str(source), "-c", "user.name=group fixture", "-c", "user.email=group@invalid", "commit", "-qm", "synthetic metadata fixture"]):
                subprocess.run(command, capture_output=True, check=True)
            commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            plan = copy.deepcopy(self.plan)
            plan.update(unit_id="new_synthetic_group", source_commit="0" * 40, inputs={})
            plan["helpers"] = {"source_root": str(source), "source_commit": commit,
                "files": {runtime.group.PREFIX + name: runtime.group.receipt(scripts / name)["sha256"] for name in names}}
            plan["configured_group"]["owned_inputs_path"] = str(runtime.group.unit_directory(plan) / "owned_group_inputs.json")
            def pin(group_plan, record):
                """Use a synthetic small package record; no installed image is read by this staging fixture."""
                runtime.group.publish(runtime.group.unit_directory(group_plan) / "backend_image_expected.json", {"synthetic": True})
            with patch.object(runtime.package, "configure_plan", side_effect=pin), \
                    patch.object(runtime.aggregation, "verify_sources"), \
                    patch.object(runtime.producer.configured, "verify_qualification_source_receipts"):
                prepared = runtime.prepare(plan, "/synthetic/external_backend.json")
            self.assertFalse(prepared["submitted"])
            unit = runtime.group.unit_directory(plan)
            packet = Path(plan["campaign_run_dir"]) / "remote_setup" / plan["unit_id"]
            for directory in (unit, packet):
                for name in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt", "progress.json"):
                    self.assertTrue((directory / name).is_file(), (directory, name))
            frozen = json.loads((unit / "scaling_plan.json").read_text())
            self.assertIsNone(frozen["job_id"])
            self.assertEqual(frozen["gpu_uuids"], [None] * 8)
            self.assertEqual({p.name for p in (unit / "code").iterdir()}, set(names))
            self.assertIn("--execute-owned-plan", (packet / "job.sbatch").read_text())


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


class MemberPrerequisiteTests(unittest.TestCase):
    """Keep physical-device matching separate from scheduler worker order and historical credit."""
    def test_prerequisites_match_uuid_after_permuted_allocation(self):
        """The same actual physical cards can appear under different later Slurm worker indices."""
        rows = [{"result": {"path": str(x)}, "review": {"path": "review" + str(x)}} for x in range(8)]
        plan = {"mode": "correctness", "gpu_count": 8, "member_qualification": rows}
        resolved = {"members": [{"gpu_uuid": "GPU-" + str(x)} for x in reversed(range(8))]}
        def read(item):
            """Return only synthetic physical identity metadata."""
            return {"resolved_device_binding": {"gpu_uuid": "GPU-" + item["path"]}}
        def reviewed(item, mode, supplied_plan, member):
            """Record the exact synthetic prerequisite chosen for each physical member."""
            return item["result"]["path"]
        with patch.object(evidence.group, "read_receipt", side_effect=read), patch.object(evidence, "reviewed_member", side_effect=reviewed):
            result = evidence.verify_prerequisites(plan, resolved)
        self.assertEqual(result["qualification"], [str(x) for x in reversed(range(8))])

    def test_duplicate_physical_prerequisite_refuses(self):
        """One reviewed physical card cannot qualify several group members."""
        plan = {"mode": "correctness", "gpu_count": 8, "member_qualification": [
            {"result": {"path": str(x)}, "review": {}} for x in range(8)]}
        with patch.object(evidence.group, "read_receipt", return_value={"resolved_device_binding": {"gpu_uuid": "GPU-one"}}):
            with self.assertRaises(ValueError):
                evidence.verify_prerequisites(plan, {"members": [{"gpu_uuid": "GPU-one"}] * 8})

    def test_group_qualification_has_no_circular_prior_qualification(self):
        """The separate zero-credit original qualification can establish, rather than assume, its actual members."""
        self.assertEqual(evidence.verify_prerequisites({"mode": "qualification"}, {"members": []}), {})

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
            self.assertEqual(sum(item.args == (71,) for item in close.call_args_list), 1)
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


class LaunchRecordsTests(unittest.TestCase):
    """Exercise one-shot transport records using private files and synthetic native replies only."""
    @classmethod
    def setUpClass(cls):
        """Create the same private source fixture without inheriting unrelated test methods."""
        RuntimePlanTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        """Remove only that private source fixture."""
        RuntimePlanTests.tearDownClass()

    def setUp(self):
        """Create a fresh private unsubmitted unit and exact standard files for the transport fixture."""
        fixture = RuntimePlanTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.old, self.plan, self.write = fixture.old, copy.deepcopy(fixture.plan), fixture.write
        self.plan["unit_id"] = "synthetic_new_runtime_group"
        self.unit = actor.group.unit_directory(self.plan)
        self.unit.mkdir()
        self.plan["configured_group"]["owned_inputs_path"] = str(self.unit / "owned_group_inputs.json")
        self.packet = actor.packet_for(self.plan)
        self.packet.mkdir(parents=True)
        self.write(self.unit / "scaling_plan.json", self.plan)
        for name in ("job.sbatch", "manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md", "progress.json"):
            (self.packet / name).write_text("synthetic metadata only\n")
        self.command = runtime.held_submission_command(self.plan, self.packet / "job.sbatch")
        self.write(self.packet / "held_submission_command.json", self.command)
        (self.old.run / "slurm/submitted_jobids.txt").touch(exist_ok=True)
        (self.old.run / "slurm/jobs").mkdir(exist_ok=True)
        (self.old.run / "queue").mkdir(exist_ok=True)
        (self.old.run / "queue/metadata.lock").touch()
        self.checked = patch.object(actor, "checked_plan", return_value=self.plan)
        self.published = patch.object(actor, "published_packet", return_value=[])
        self.checked.start(); self.published.start()
        self.addCleanup(self.checked.stop); self.addCleanup(self.published.stop)

    def held_text(self, job):
        """Return an explicitly synthetic native held record matching the frozen request."""
        return (f"JobId={job} UserId=sl5nw(2618919) JobState=PENDING Reason=JobHeldUser RunTime=00:00:00 NodeList= "
            f"Command={self.packet / 'job.sbatch'} Partition=gpu Account=sl5nw_base QOS=csresnolim "
            "Reservation=synthetic_group_reservation ReqNodeList=jaguar03 NumNodes=1-1 NumTasks=1 CPUs/Task=65 "
            "Requeue=0 Restarts=0 TresPerNode=gres/gpu:nvidia_rtx_a4500:8 ReqTRES=cpu=65,mem=514G,node=1,gres/gpu=8 TimeLimit=01:00:00\n").encode()

    def native(self, command, **unused):
        """Simulate only this private fixture's submission, held query and exact release."""
        if command[0] == "sbatch":
            stdout = b"777777\n"
        elif command[:3] == ["scontrol", "show", "job"]:
            stdout = self.held_text("777777")
        elif command == ["scontrol", "release", "777777"]:
            stdout = b""
        else:
            raise AssertionError(command)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    def test_submission_fsyncs_three_ledgers_before_receipt_and_refuses_reuse(self):
        """The actual private identifier/row files exist before successful held submission becomes visible."""
        original_publish = actor.group.publish
        def publish(path, value):
            """Check the ordering at the real receipt publication boundary."""
            if Path(path).name == "group_submission.json":
                for name in ("submitted_jobids.txt", "submitted_jobids_uva.txt"):
                    self.assertIn("777777", (self.old.run / "slurm" / name).read_text().splitlines())
                self.assertIn("\t".join(actor.row_for(self.plan, "777777")),
                    (self.old.run / "slurm/submissions.tsv").read_text().splitlines())
                self.assertTrue((self.packet / "submission_ledgers_fsynced.json").is_file())
            return original_publish(path, value)
        with patch.object(actor.subprocess, "run", side_effect=self.native) as native, patch.object(actor.group, "publish", side_effect=publish):
            actual = actor.submit_held_once(self.unit, "0" * 40)
            self.assertEqual(actual["job_id"], "777777")
            with self.assertRaises(ValueError):
                actor.submit_held_once(self.unit, "0" * 40)
        self.assertEqual(native.call_count, 1)

    def test_actual_release_receipt_precedes_valid_atomic_owned_inputs(self):
        """Successful synthetic release produces the exact unmodified runtime ownership schema only afterward."""
        with patch.object(actor.subprocess, "run", side_effect=self.native):
            actor.submit_held_once(self.unit, "0" * 40)
            actor.prepare_release(self.unit, "0" * 40)
            actor.release_once(self.unit, "0" * 40)
        owned = actor.group.owned_inputs(self.plan)
        self.assertEqual(owned["job_id"], "777777")
        self.assertEqual(owned["release"], actor.group.receipt(self.packet / "group_release.json"))
        self.assertEqual(owned["submission_row"], actor.row_for(self.plan, "777777"))
        with patch.object(actor.subprocess, "run") as native:
            with self.assertRaises(ValueError):
                actor.release_once(self.unit, "0" * 40)
        native.assert_not_called()

    def test_failed_release_retains_attempt_without_success_sidecar(self):
        """A failed native release cannot look like successful ownership or become retryable."""
        with patch.object(actor.subprocess, "run", side_effect=self.native):
            actor.submit_held_once(self.unit, "0" * 40)
            actor.prepare_release(self.unit, "0" * 40)
        def fail(command, **kwargs):
            """Retain a distinct explicit synthetic release failure."""
            if command == ["scontrol", "release", "777777"]:
                return SimpleNamespace(returncode=1, stdout=b"", stderr=b"synthetic failure")
            return self.native(command, **kwargs)
        with patch.object(actor.subprocess, "run", side_effect=fail):
            with self.assertRaises(ValueError):
                actor.release_once(self.unit, "0" * 40)
        self.assertTrue((self.packet / "release_intent.json").is_file())
        self.assertFalse((self.unit / "owned_group_inputs.json").exists())
        self.assertFalse((self.packet / "group_release.json").exists())

    def test_missing_live_owned_member_prevents_any_query(self):
        """Cached receipts cannot authorize querying an identifier removed from the live campaign ledger."""
        with patch.object(actor.subprocess, "run", side_effect=self.native):
            actor.submit_held_once(self.unit, "0" * 40)
        (self.old.run / "slurm/submitted_jobids_uva.txt").write_text("")
        with patch.object(actor.subprocess, "run") as native:
            with self.assertRaises(ValueError):
                actor.held_query(self.plan, "unreachable_query")
        native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
