"""Check bounded final submission decisions with synthetic metadata and no numerical imports."""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_training_monitor as monitor


class FinalTrainingMonitorTests(unittest.TestCase):
    """Keep prepared policy, full compute admission and exact owned job state separate."""

    def setUp(self):
        """Create two explicit already-prepared descriptors without making training outputs."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.queue = self.run / "final_training_queue"
        self.source = {"root": "/synthetic/detached/numerical-source", "commit": "a" * 40}
        self.ready = {"source_root": self.source["root"], "source_commit": self.source["commit"]}
        self.shape = {"gpus": 2, "walltime_seconds": 43200, "host_memory_gib_per_gpu": 64,
                      "cpus_per_gpu": 8, "checkpoint_reserve_seconds": 1800, "allow_partial_group": True}
        self.write("SOURCE_AND_GATES_READY.json", self.ready)
        self.write("study_registry.json", {"synthetic": "fifty raw-validated mechanisms"})
        self.write("final_verification.json", {"kind": "final_verification", "source_commit": self.source["commit"]})
        self.names = ["explicit_policy_seed_101", "explicit_policy_seed_102"]
        for index, name in enumerate(self.names):
            spec = self.write("prepared/" + name + ".json", {"final_verification": str(self.run / "final_verification.json")})
            unit = {"unit_id": name, "hardware_class": "jaguar03_rtx_a4500", "campaign_run_dir": str(self.run),
                    "numerical_source": self.source, "copy_seeds": [101 + index],
                    "final_specification": {"path": str(spec), "sha256": monitor.launcher.file_digest(spec)}}
            self.write("final_training_queue/units/" + name + ".json", unit)
        self.proposal = self.write("explicit_policy_proposal.json", {"schema_version": 1,
                                    "unit_ids": self.names, "resources": self.shape})
        self.verify = patch.object(monitor.launcher, "verify_unit")
        self.verify.start()
        self.addCleanup(self.verify.stop)
        self.source_ready = patch.object(monitor.controller, "ready_specification", return_value=self.ready)
        self.source_ready.start()
        self.addCleanup(self.source_ready.stop)
        monitor.prepare_policy(self.run, self.proposal)
        self.phase = {"phase": "ready_for_final"}

    def write(self, relative, value):
        """Write one complete small synthetic metadata record."""
        path = self.run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n" if not isinstance(value, str) else value)
        return path

    def snapshot(self, *, ready=2, held=0, jobs=None, completed=0):
        """Describe states for the same explicit unit list without deriving new identities."""
        states = ["held"] * held + ["completed"] * completed + ["ready"] * ready
        states += ["running"] * (len(self.names) - len(states))
        return {"cluster": "uva", "jobs": jobs or {},
                "units": [{"unit_id": name, "state": state} for name, state in zip(self.names, states)],
                "counts": {"ready": ready, "held": held, "completed": completed, "running": states.count("running"), "checkpointed": 0}}

    @contextmanager
    def allocated(self, snapshot=None):
        """Replace compute-side statistics with a synthetic successful raw-gate boundary."""
        with patch.object(monitor.controller, "final_validation_allocation") as allocation, \
                patch.object(monitor.controller, "final_requirements_met", return_value=True) as gates, \
                patch.object(monitor.launcher, "status", return_value=snapshot or self.snapshot()) as status, \
                patch.object(monitor.launcher, "submit", return_value={"job_id": "9001"}) as submit:
            yield allocation, gates, status, submit

    def test_absent_policy_does_not_query_jobs_or_create_queue(self):
        """An earlier study phase does not acquire new final-unit identities or jobs."""
        with tempfile.TemporaryDirectory() as temporary, patch.object(monitor.launcher, "status") as status:
            path = Path(temporary)
            result = monitor.tick(path)
            self.assertEqual(result["state"], "disabled")
            status.assert_not_called()
            self.assertFalse((path / "final_training_queue").exists())

    def test_policy_is_idempotent_and_explicit_without_seed_or_unit_creation(self):
        """Repeated preparation keeps original descriptor bytes and explicit independent seeds."""
        before = {p.name: p.read_bytes() for p in (self.queue / "units").iterdir()}
        first = (self.queue / monitor.POLICY_NAME).read_bytes()
        policy = monitor.prepare_policy(self.run, self.proposal)
        self.assertEqual(set(policy["units"]), set(self.names))
        self.assertEqual((self.queue / monitor.POLICY_NAME).read_bytes(), first)
        self.assertEqual({p.name: p.read_bytes() for p in (self.queue / "units").iterdir()}, before)
        self.assertFalse((self.run / "data").exists())
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_policy_rejects_implicit_names_or_changed_resource_shape(self):
        """A subset cannot leave generic workers free to claim an unlisted unit."""
        self.write(self.proposal, {"schema_version": 1, "unit_ids": self.names[:1], "resources": self.shape})
        with self.assertRaisesRegex(ValueError, "every registered"):
            monitor.prepare_policy(self.run, self.proposal)
        self.write(self.proposal, {"schema_version": 1, "unit_ids": self.names,
                                  "resources": {**self.shape, "gpus": 1}})
        with self.assertRaisesRegex(ValueError, "immutable"):
            monitor.prepare_policy(self.run, self.proposal)

    def test_resources_reject_boolean_counts_missing_fields_and_node_overcommit(self):
        """Explicit policy cannot exceed one eight-card node or reduce the known memory allowance."""
        for field, value in (("gpus", True), ("gpus", 9), ("host_memory_gib_per_gpu", 32),
                             ("host_memory_gib_per_gpu", 600), ("walltime_seconds", 345601),
                             ("cpus_per_gpu", 1), ("checkpoint_reserve_seconds", 100), ("allow_partial_group", 1)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                monitor.resources({**self.shape, field: value})
        with self.assertRaises(ValueError):
            monitor.resources({key: value for key, value in self.shape.items() if key != "gpus"})

    def test_phase_waits_and_changed_raw_proof_or_descriptor_holds_before_submission(self):
        """A phase label cannot bypass changed proof bytes or changed registered identities."""
        with self.allocated() as (_, gates, _, submit):
            self.assertEqual(monitor.tick(self.run, controller_state={"phase": "initial_benchmarks"})["state"], "waiting")
            gates.assert_not_called()
            submit.assert_not_called()
            path = self.run / "final_verification.json"
            path.write_bytes(path.read_bytes() + b"\n")
            held = monitor.tick(self.run, controller_state=self.phase)
            self.assertEqual(held["state"], "held")
            self.assertIn("changed", held["reason"])
            submit.assert_not_called()

    def test_registered_descriptor_addition_and_source_change_are_held(self):
        """New units and source revisions require a separately reviewed prepared policy."""
        with self.allocated() as (_, _, _, submit):
            self.write("final_training_queue/units/another_unit.json", {
                "unit_id": "another_unit", "hardware_class": "jaguar03_rtx_a4500", "campaign_run_dir": str(self.run)})
            self.assertIn("registered final units differ", monitor.tick(self.run, controller_state=self.phase)["reason"])
            submit.assert_not_called()

    def test_full_raw_revalidation_and_cpu_guard_precede_one_locked_submission(self):
        """A CPU monitor validates raw evidence while GPU admission remains inside the new node job."""
        with self.allocated() as (allocation, gates, status, submit), patch.object(monitor.launcher, "owned_compute", side_effect=AssertionError("not a node child")):
            def check_submission(*args, **kwargs):
                """Verify both metadata decision locks still cover the final submission API call."""
                allocation.assert_called_once_with(self.run)
                gates.assert_called_once()
                for lock_path in (self.queue / "monitor.lock", self.run / "queue/metadata.lock"):
                    with lock_path.open("a+") as stream, self.assertRaises(BlockingIOError):
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return {"job_id": "9001"}
            submit.side_effect = check_submission
            result = monitor.tick(self.run, controller_state=self.phase)
            self.assertEqual(result["state"], "submitted")
            submit.assert_called_once_with(self.run, **{key: value for key, value in self.shape.items() if key != "allow_partial_group"})
            self.assertEqual(monitor.launcher.read(self.queue / "monitor_latest.json")["submission"]["job_id"], "9001")
        with (self.queue / "monitor.lock").open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_proof_bytes_changed_before_source_lock_cannot_be_submitted(self):
        """The source and final-proof hashes are rechecked inside the admission revision lock."""
        for name in ("SOURCE_AND_GATES_READY.json", "final_verification.json"):
            path = self.run / name
            original = path.read_bytes()
            with self.subTest(name=name), self.allocated() as (allocation, gates, _, submit):
                allocation.side_effect = lambda run: path.write_bytes(original + b"\n")
                result = monitor.tick(self.run, controller_state=self.phase)
                self.assertEqual(result["state"], "held")
                self.assertIn("changed", result["reason"])
                gates.assert_not_called()
                submit.assert_not_called()
            path.write_bytes(original)

    def test_false_raw_gate_or_controller_error_never_submits(self):
        """All fifty/combo/packing/scaling admission is recomputed, not inferred from five flags."""
        with self.allocated() as (_, gates, _, submit):
            gates.return_value = False
            self.assertEqual(monitor.tick(self.run, controller_state=self.phase)["state"], "held")
            submit.assert_not_called()
            gates.reset_mock()
            result = monitor.tick(self.run, controller_state=self.phase, controller_error="stale comparison receipt")
            self.assertIn("controller failed", result["reason"])
            gates.assert_not_called()

    def test_active_or_unknown_owned_job_cannot_create_another_allocation(self):
        """Pending work and scheduler-query failures block repeated sbatch calls."""
        with self.allocated(self.snapshot(ready=1, jobs={"9001": "PENDING"})) as (_, _, status, submit):
            for _ in range(2):
                self.assertEqual(monitor.tick(self.run, controller_state=self.phase)["state"], "waiting")
            submit.assert_not_called()
            status.side_effect = ValueError("previous exact final-job state is unknown")
            self.assertIn("unknown", monitor.tick(self.run, controller_state=self.phase)["reason"])
            submit.assert_not_called()

    def test_held_application_is_not_automatically_retried(self):
        """A failed application stays held even when another registered seed is ready."""
        with self.allocated(self.snapshot(ready=1, held=1)) as (_, _, _, submit):
            for _ in range(2):
                result = monitor.tick(self.run, controller_state=self.phase)
                self.assertEqual(result["state"], "held")
                self.assertEqual(result["units"][0]["state"], "held")
            submit.assert_not_called()

    def test_terminal_checkpoint_resume_and_explicit_partial_group_are_bounded(self):
        """The existing launcher admits a checkpoint only after its exact prior job is terminal."""
        with self.allocated(self.snapshot(ready=1, completed=1, jobs={"9001": "COMPLETED"})) as (_, _, _, submit):
            result = monitor.tick(self.run, controller_state=self.phase)
            self.assertEqual(result["state"], "submitted")
            self.assertEqual(submit.call_args.kwargs["gpus"], 1)
            submit.assert_called_once()
        policy = monitor.launcher.read(self.queue / monitor.POLICY_NAME)
        policy["resources"]["allow_partial_group"] = False
        self.write(self.queue / monitor.POLICY_NAME, policy)
        with self.allocated(self.snapshot(ready=1, completed=1)) as (_, _, _, submit):
            self.assertEqual(monitor.tick(self.run, controller_state=self.phase)["state"], "waiting")
            submit.assert_not_called()

    def test_invalid_policy_is_held_and_existing_hold_retains_the_campaign_phase(self):
        """Malformed metadata cannot terminate the standing monitor or hide a diagnosed application hold."""
        path = self.queue / monitor.POLICY_NAME
        original = path.read_bytes()
        path.write_text("{invalid")
        with self.allocated() as (_, _, _, submit):
            self.assertEqual(monitor.tick(self.run, controller_state=self.phase)["state"], "held")
            submit.assert_not_called()
            path.write_bytes(original)
            self.write("final_training_queue/status/" + self.names[0] + ".json",
                       {"unit_id": self.names[0], "state": "held", "reason": "current application failure"})
            result = monitor.tick(self.run, controller_state={"phase": "final_verification_pending"})
            self.assertEqual(result["state"], "held")
            self.assertIn("final_verification_pending", result["reason"])
            self.assertEqual(result["units"][0]["state"], "held")
            submit.assert_not_called()

    def test_completion_does_not_end_other_hardware_and_foreign_ids_are_not_queried(self):
        """UVA completion is separate from campaign completion, and numeric IDs are cluster-local."""
        with self.allocated(self.snapshot(ready=0, completed=2)) as (_, _, _, submit):
            self.assertEqual(monitor.tick(self.run, controller_state=self.phase)["state"], "finished")
            submit.assert_not_called()
            self.assertFalse((self.run / "STUDY_COMPLETE").exists())
        self.write("slurm/submitted_jobids_uva.txt", "123\n")
        self.write("slurm/submitted_jobids_aces.txt", "999\n")
        self.write("final_training_queue/jobs/uva_job-999.json", {"job_id": "999"})
        with patch.object(monitor.controller, "final_validation_allocation"), \
                patch.object(monitor.controller, "final_requirements_met", return_value=True), \
                patch.object(monitor.launcher.subprocess, "check_output", side_effect=AssertionError("foreign scheduler query")), \
                patch.object(monitor.launcher, "submit") as submit:
            result = monitor.tick(self.run, controller_state=self.phase)
            self.assertIn("exact UVA job ledger", result["reason"])
            submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
