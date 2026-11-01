"""Exercise queue admission, ownership, recovery and completion with synthetic metadata only.

No JAX, game, model, GPU process or scheduler is started by these tests.
"""

from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))
import controller
import ideas
import queue_worker
import run_case


class QueueProtocolTests(unittest.TestCase):
    """Use temporary filesystem state and real atomic renames without numerical dependencies."""

    def setUp(self):
        """Create a synthetic authorized study whose source inspection is separately mocked."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.run = self.directory / "campaign"
        self.source = self.directory / "frozen_source"
        self.source.mkdir()
        self.run.mkdir()
        self.commit = "a" * 40
        self.configuration = dict(controller.REFERENCE_ALGORITHM)
        configuration_path = self.run / "reference.json"
        controller.atomic_json(configuration_path, self.configuration)
        gates = {}
        for hardware in ideas.REQUIRED_HARDWARE:
            path = self.run / f"gate_{hardware}.json"
            controller.atomic_json(path, {"passed": True, "source_commit": self.commit})
            gates[hardware] = str(path)
        self.ready = {"authorized": True, "source_root": str(self.source), "source_commit": self.commit,
                      "contexts": list(ideas.REQUIRED_HARDWARE), "matched_copy_counts": [1],
                      "gate_records": gates, "reference_configuration_path": str(configuration_path),
                      "maximum_attempts": 2}
        self.sentinel = self.run / controller.SENTINEL_NAME
        controller.atomic_json(self.sentinel, self.ready)
        capabilities = ideas.source_capabilities()
        patches = (
            mock.patch.object(controller, "verify_source", return_value={"source_root": str(self.source),
                                                                         "source_commit": self.commit}),
            mock.patch.object(queue_worker, "verify_source", return_value={"source_root": str(self.source),
                                                                           "source_commit": self.commit}),
            mock.patch.object(controller, "source_capabilities", return_value=capabilities),
            mock.patch.object(controller, "new_registry", side_effect=lambda commit, root=None: ideas.new_registry(commit)),
            mock.patch.object(controller, "refresh_implementations", side_effect=lambda registry, root=None:
                              ideas.refresh_implementations(registry)),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        controller.layout(self.run)
        self.identity = {"worker_id": "synthetic-worker", "hostname": socket.gethostname(),
                         "worker_pid": os.getpid(), "worker_start_time": controller.process_start_time(os.getpid()),
                         "slurm_job_id": "", "hardware_class": "h100",
                         "gpu_uuid": "GPU-synthetic-test", "gpu_name": "synthetic H100"}

    def single_case(self, context="h100"):
        """Enqueue one concrete specification for ownership/recovery tests."""
        ready = controller.ready_specification(self.run)
        case = controller.immutable_case(ideas.INITIAL_IDEAS[0], context, 1, ready)
        with controller.queue_lock(self.run):
            controller.enqueue_case(self.run, case)
        return case

    def synthetic_initial_registry_and_additional_specs(self):
        """Prepare test-only completed initial rows and five distinct hypothetical combinations."""
        registry = ideas.new_registry(self.commit)
        registry["initial_matched_copy_counts"] = [1]
        for entry in registry["initial_ideas"]:
            for context in ideas.REQUIRED_HARDWARE:
                entry["results"][context] = [{"copies": 1, "implementation_distinct": True,
                                               "measurement_valid": True, "correctness_passed": True}]
        flags = ("cache_target_features", "hoist_critic", "flat_parameters",
                 "pre_gather_minibatches", "associative_gae")
        specifications = [{"name": f"synthetic_additional_combination_{index}",
                           "hypothesis": "Synthetic metadata fixture, not an actual research choice",
                           "mechanism": "Exercise distinct queue specifications in a temporary directory",
                           "learner": {"donate_state": True, flag: True}, "environment": {}, "execution": {},
                           "correctness_checks": ["synthetic fixture"],
                           "measurement_settings": {"rounds": 7, "iterations": 10},
                           "acceptance_criteria": "synthetic fixture"}
                          for index, flag in enumerate(flags)]
        return registry, specifications

    def finish_synthetic_case(self, case, lease, rejected=False):
        """Provide explicit test-only results to exercise indexing, never report real measurements."""
        unit, attempt = controller.attempt_paths(self.run, case["unit_id"], lease["attempt"])
        before = {"learner": case["configuration"], "environment": {},
                  "execution": case["reference_execution"] or {}}
        after = {"learner": {**case["configuration"], **case["candidate"]["learner"]},
                 "environment": case["candidate"]["environment"],
                 "execution": {**before["execution"], **case["candidate"]["execution"]}}
        result = {"case_completed": True, "specification_digest": case["specification_digest"],
                  "idea": case["hypothesis"], "hardware_class": case["hardware_class"],
                  "copies": case["configuration"]["n_copies"], "source_commit": self.commit,
                  "decision": "rejected" if rejected else "accepted", "correctness_attempted": True,
                  "correctness_passed": not rejected, "measurement_valid": not rejected,
                  "implementation_distinct": True, "reference": before, "candidate": after,
                  "median_speedup": 1.1, "paired_rounds": 7,
                  "median_candidate_transitions_per_second_total": 100.0}
        if rejected:
            result["failure_stage"] = "correctness"
        controller.atomic_json(attempt / "result.json", result)
        queue_worker.finish_claim(self.run, case, lease, 0)
        return result

    def test_no_admission_without_parent_and_both_gates(self):
        """A missing sentinel, absent authorization or failed context gate queues no work."""
        self.sentinel.unlink()
        self.assertEqual(controller.tick(self.run)["queued"], 0)
        controller.atomic_json(self.sentinel, {**self.ready, "authorized": False})
        self.assertEqual(controller.tick(self.run)["queued"], 0)
        controller.atomic_json(self.sentinel, self.ready)
        controller.atomic_json(Path(self.ready["gate_records"]["h100"]),
                               {"passed": False, "source_commit": self.commit})
        self.assertEqual(controller.tick(self.run)["queued"], 0)
        self.assertFalse(list((self.run / "queue/pending").glob("*.json")))

    def test_admission_is_complete_and_idempotent(self):
        """All thirty ideas appear on both contexts exactly once with immutable output/source paths."""
        first = controller.tick(self.run)
        second = controller.tick(self.run)
        self.assertEqual(first["queued"], 60)
        self.assertEqual(second["queued"], 0)
        self.assertEqual(second["queue_counts"]["pending"], 60)
        for path in (self.run / "queue/pending").glob("*.json"):
            case = controller.read_json(path)
            self.assertEqual(case["source_commit"], self.commit)
            self.assertEqual(case["campaign_run_dir"], str(self.run.resolve()))

    def test_smoke_configuration_cannot_become_the_study_reference(self):
        """A short warmup used for a smoke check cannot replace the agreed fifty-rollout setup."""
        controller.atomic_json(Path(self.ready["reference_configuration_path"]),
                               {**self.configuration, "observation_warmup_iterations": 1})
        state = controller.tick(self.run)
        self.assertEqual(state["queued"], 0)
        self.assertIn("CleanRL settings", state["authorization_error"])

    def test_two_workers_cannot_claim_the_same_case(self):
        """Concurrent atomic claims produce one owner and one empty result."""
        case = self.single_case()
        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(lambda unused: queue_worker.claim_next(self.run, "h100", self.identity), range(2)))
        self.assertEqual(sum(claim is not None for claim in claims), 1)
        self.assertEqual(controller.case_states(self.run, case["unit_id"]), ["running"])
        self.assertEqual(controller.read_json(self.run / "queue/leases" / f"{case['unit_id']}.json")["attempt"], 1)

    def test_context_and_revocation_are_respected(self):
        """An H100 worker cannot take A4500 work or claim anything after authorization is revoked."""
        self.single_case("jaguar03_rtx_a4500")
        self.assertIsNone(queue_worker.claim_next(self.run, "h100", self.identity))
        self.single_case("h100")
        controller.atomic_json(self.sentinel, {**self.ready, "authorized": False})
        self.assertIsNone(queue_worker.claim_next(self.run, "h100", self.identity))

    def test_standing_lock_is_rechecked_and_released_between_cases(self):
        """Each numerical case owns admission independently; the interval between cases stays free."""
        self.single_case()
        standing = self.directory / "standing.lock"

        def assert_held():
            """A separate open-file description must not acquire an already held standing lock."""
            with standing.open("a+") as stream:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def probe(gpu, context):
            """Check that readiness is queried after admission, not before it."""
            assert_held()
            return {"ready": True}

        def execute(run_dir, case, lease, python, private_fd, standing_fd):
            """Verify the admission FD is passed to the child-launch path while still held."""
            assert_held()
            self.assertIsNotNone(standing_fd)
            self.assertEqual(os.readlink(f"/proc/self/fd/{standing_fd}"), str(standing))
            self.finish_synthetic_case(case, lease)
            return "completed"

        with (self.directory / "private.lock").open("a+") as private:
            with mock.patch.object(queue_worker, "gpu_readiness", side_effect=probe), mock.patch.object(
                    queue_worker, "run_one", side_effect=execute) as child:
                for index in range(2):
                    if index:
                        ready = controller.ready_specification(self.run)
                        case = controller.immutable_case(ideas.INITIAL_IDEAS[1], "h100", 1, ready)
                        with controller.queue_lock(self.run):
                            controller.enqueue_case(self.run, case)
                    outcome = queue_worker.perform_next_case(self.run, "h100", self.identity,
                                                              sys.executable, private.fileno(), standing)
                    self.assertTrue(outcome["ran_case"])
                    with standing.open("a+") as free:
                        fcntl.flock(free.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(free.fileno(), fcntl.LOCK_UN)
                self.assertEqual(child.call_count, 2)

    def test_busy_gpu_releases_admission_without_claiming(self):
        """A busy GPU leaves work pending and releases its standing lock before polling again."""
        case = self.single_case()
        standing = self.directory / "standing.lock"
        with (self.directory / "private.lock").open("a+") as private:
            with mock.patch.object(queue_worker, "gpu_readiness", return_value={"ready": False,
                                   "reason": "gpu_has_compute_processes", "processes": [101]}), mock.patch.object(
                                       queue_worker, "run_one") as child:
                result = queue_worker.perform_next_case(self.run, "h100", self.identity,
                                                         sys.executable, private.fileno(), standing)
                self.assertEqual(result["state"], "waiting_for_gpu")
                child.assert_not_called()
        self.assertEqual(controller.case_states(self.run, case["unit_id"]), ["pending"])
        with standing.open("a+") as free:
            fcntl.flock(free.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(free.fileno(), fcntl.LOCK_UN)

    def test_empty_slurm_worker_exits_without_sleeping(self):
        """An empty Slurm allocation returns immediately instead of waiting through code preparation."""
        gpu_uuid = f"GPU-synthetic-{self.directory.name}"
        lock = Path("/tmp") / f"sl5nw_atari_pixel_benchmark_{gpu_uuid}.lock"
        self.addCleanup(lambda: lock.unlink(missing_ok=True))
        arguments = ["queue_worker.py", "--run-dir", str(self.run), "--context", "h100",
                     "--gpu-uuid", gpu_uuid]
        outcomes = [{"state": "idle", "ran_case": False}, AssertionError("empty Slurm allocation was retained")]
        with mock.patch.object(sys, "argv", arguments), mock.patch.dict(os.environ, {"SLURM_JOB_ID": "synthetic-job"}), \
                mock.patch.object(queue_worker.socket, "gethostname", return_value="metadata-test-host"), \
                mock.patch.object(queue_worker, "gpu_identity", return_value={"gpu_uuid": gpu_uuid}), \
                mock.patch.object(queue_worker, "worker_identity", return_value=self.identity), \
                mock.patch.object(queue_worker, "perform_next_case", side_effect=outcomes), \
                mock.patch.object(queue_worker.time, "sleep") as sleep:
            queue_worker.main()
            sleep.assert_not_called()
        self.assertTrue(queue_worker.exit_when_idle(slurm_job_id="123"))
        self.assertFalse(queue_worker.exit_when_idle(slurm_job_id=""))

    def test_no_pending_work_takes_no_standing_lock_or_gpu_probe(self):
        """A waiting direct supervisor does not occupy admission or initialize GPU work."""
        standing = self.directory / "not_opened.lock"
        with mock.patch.object(queue_worker, "gpu_readiness") as probe:
            result = queue_worker.perform_next_case(self.run, "h100", self.identity, sys.executable, -1, standing)
            self.assertEqual(result["state"], "idle")
            probe.assert_not_called()
        self.assertFalse(standing.exists())

    def test_serval02_other_card_is_refused_before_gpu_access(self):
        """A launcher mistake cannot redirect the study from its permitted card 1 to card 0."""
        arguments = ["queue_worker.py", "--run-dir", str(self.run), "--context", "h100",
                     "--gpu-uuid", "GPU-wrong-card", "--standing-lock", "/tmp/sl5nw_serval02_card1.lock"]
        with mock.patch.object(sys, "argv", arguments), mock.patch.object(
                queue_worker.socket, "gethostname", return_value="serval02"), mock.patch.object(
                    queue_worker, "gpu_identity") as gpu:
            with self.assertRaisesRegex(ValueError, "never card 0"):
                queue_worker.main()
            gpu.assert_not_called()

    def test_incomplete_metadata_writes_can_be_repaired_without_reclaiming_work(self):
        """Repair a specification-only write or an unstarted claim with no lease/data."""
        ready = controller.ready_specification(self.run)
        case = controller.immutable_case(ideas.INITIAL_IDEAS[0], "h100", 1, ready)
        name = f"{case['unit_id']}.json"
        controller.atomic_json(self.run / "queue/specifications" / name, case)
        with controller.queue_lock(self.run):
            self.assertTrue(controller.enqueue_case(self.run, case))
            (self.run / "queue/pending" / name).rename(self.run / "queue/running" / name)
            self.assertEqual(controller.recover_unstarted_claims(self.run), 1)
        self.assertEqual(controller.case_states(self.run, case["unit_id"]), ["pending"])

    def test_attempts_are_preserved_and_retry_limit_is_bounded(self):
        """Retryable failures get new attempt directories and stop at the explicit limit."""
        case = self.single_case()
        first_case, first_lease = queue_worker.claim_next(self.run, "h100", self.identity)
        queue_worker.finish_claim(self.run, first_case, first_lease, 1)
        with controller.queue_lock(self.run):
            self.assertTrue(controller.retry_failed(self.run, case["unit_id"]))
        second_case, second_lease = queue_worker.claim_next(self.run, "h100", self.identity)
        self.assertEqual(second_lease["attempt"], 2)
        queue_worker.finish_claim(self.run, second_case, second_lease, 1)
        with controller.queue_lock(self.run):
            self.assertFalse(controller.retry_failed(self.run, case["unit_id"]))
        _, first = controller.attempt_paths(self.run, case["unit_id"], 1)
        _, second = controller.attempt_paths(self.run, case["unit_id"], 2)
        self.assertTrue((first / "failure.json").exists())
        self.assertTrue((second / "failure.json").exists())

    def test_age_does_not_reclaim_a_live_claim(self):
        """Even an arbitrarily old lease remains owned when its recorded process still exists."""
        case = self.single_case()
        _, lease = queue_worker.claim_next(self.run, "h100", self.identity)
        path = self.run / "queue/leases" / f"{case['unit_id']}.json"
        controller.atomic_json(path, {**lease, "claimed_at": "2000-01-01T00:00:00+00:00"})
        with self.assertRaisesRegex(ValueError, "still alive"):
            controller.recover_local_claim(self.run, case["unit_id"], lease["token"])
        self.assertEqual(controller.case_states(self.run, case["unit_id"]), ["running"])

    def test_local_dead_claim_recovery_requires_exact_token(self):
        """Recovery refuses another owner and retains the interrupted attempt before retry."""
        case = self.single_case()
        _, lease = queue_worker.claim_next(self.run, "h100", self.identity)
        with self.assertRaisesRegex(ValueError, "exact lease token"):
            controller.recover_local_claim(self.run, case["unit_id"], "wrong-token")
        with mock.patch.object(controller, "process_matches", return_value=False):
            controller.recover_local_claim(self.run, case["unit_id"], lease["token"])
        self.assertEqual(controller.case_states(self.run, case["unit_id"]), ["failed"])
        _, attempt = controller.attempt_paths(self.run, case["unit_id"], 1)
        self.assertEqual(controller.read_json(attempt / "failure.json")["failure_stage"], "worker_interruption")

    def test_release_handshake_checks_this_child_pid(self):
        """A child proceeds only once its own PID/birth counter appears in its exact lease."""
        case = self.single_case()
        _, lease = queue_worker.claim_next(self.run, "h100", self.identity)
        released = {**lease, "runner_pid": os.getpid(),
                    "runner_start_time": controller.process_start_time(os.getpid()), "state": "running"}
        with mock.patch.object(run_case, "read_json", side_effect=[lease, released]), mock.patch.object(run_case.time, "sleep"):
            self.assertEqual(run_case.wait_for_recorded_release(self.run, case, 1, lease["token"]), released)

    def test_recovered_contaminated_measurement_is_retried(self):
        """A dead worker's invalid timing result never becomes completed scientific work."""
        case = self.single_case()
        _, lease = queue_worker.claim_next(self.run, "h100", self.identity)
        _, attempt = controller.attempt_paths(self.run, case["unit_id"], 1)
        controller.atomic_json(attempt / "result.json", {
            "case_completed": True, "measurement_valid": False, "correctness_passed": True,
            "contaminated": True, "decision": "rejected"})
        with mock.patch.object(controller, "process_matches", return_value=False):
            controller.recover_local_claim(self.run, case["unit_id"], lease["token"])
        self.assertEqual(controller.case_states(self.run, case["unit_id"]), ["failed"])
        with controller.queue_lock(self.run):
            self.assertTrue(controller.retry_failed(self.run, case["unit_id"]))
        self.assertTrue((attempt / "result.json").exists())

    def test_completed_results_are_indexed_once_including_rejections(self):
        """Restarting a controller cannot duplicate a valid or numerically rejected result."""
        registry = ideas.new_registry(self.commit)
        for context, rejected in (("h100", False), ("jaguar03_rtx_a4500", True)):
            self.single_case(context)
            case, lease = queue_worker.claim_next(self.run, context, self.identity)
            self.finish_synthetic_case(case, lease, rejected)
        controller.atomic_json(self.run / "study_registry.json", registry)
        self.assertEqual(controller.index_completed(self.run, registry), 2)
        self.assertEqual(controller.index_completed(self.run, registry), 0)
        self.assertIn(ideas.INITIAL_IDEAS[0].name, ideas.investigated_hypotheses(registry))
        self.assertNotIn(ideas.INITIAL_IDEAS[0].name, ideas.measured_hypotheses(registry))

    def test_completed_attempt_is_a_read_only_noop(self):
        """Re-entering a finished attempt returns its result without another numerical import or write."""
        self.single_case()
        case, lease = queue_worker.claim_next(self.run, "h100", self.identity)
        expected = self.finish_synthetic_case(case, lease)
        _, attempt = controller.attempt_paths(self.run, case["unit_id"], 1)
        before = (attempt / "result.json").stat().st_mtime_ns
        actual = run_case.run_case(self.run, case, 1, lease["token"])
        self.assertEqual(actual, expected)
        self.assertEqual((attempt / "result.json").stat().st_mtime_ns, before)
        self.assertNotIn("jax", sys.modules)

    def test_wrong_output_root_fails_before_creating_outputs(self):
        """A frozen-source runner cannot redirect its data into another campaign directory."""
        case = self.single_case()
        path = self.run / "queue/specifications" / f"{case['unit_id']}.json"
        wrong = self.directory / "wrong_output_root"
        result = subprocess.run([sys.executable, str(SCRIPT_ROOT / "run_case.py"),
                                 "--run-dir", str(wrong), "--case", str(path),
                                 "--attempt", "1", "--lease-token", "not-released"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("original absolute campaign", result.stderr)
        self.assertFalse(wrong.exists())

    def test_initial_completion_waits_for_every_matched_copy_count(self):
        """One-copy results cannot stand in for the required larger matched comparison."""
        registry = ideas.new_registry(self.commit)
        for entry in registry["initial_ideas"]:
            for context in ideas.REQUIRED_HARDWARE:
                entry["results"][context] = [{"copies": 1, "implementation_distinct": True,
                                               "measurement_valid": True, "correctness_passed": True}]
        self.assertTrue(controller.investigated_at_counts(registry, ideas.IDEAS_BY_NAME, [1]))
        self.assertFalse(controller.investigated_at_counts(registry, ideas.IDEAS_BY_NAME, [1, 2]))
        flags = {name: True for name in ("final_combination_correctness_passed", "final_confirmation_passed",
                                         "h100_packing_complete", "a4500_packing_complete", "eight_gpu_scaling_complete")}
        self.assertFalse(controller.final_requirements_met(registry, flags))

    def test_thirty_completed_ideas_leave_twenty_unchosen(self):
        """The controller stops at adaptive research instead of fabricating additional ideas or training."""
        controller.tick(self.run)
        for context in ideas.REQUIRED_HARDWARE:
            while True:
                claimed = queue_worker.claim_next(self.run, context, self.identity)
                if claimed is None:
                    break
                case, lease = claimed
                self.finish_synthetic_case(case, lease, rejected=case["hypothesis_position"] % 2 == 0)
        state = controller.tick(self.run)
        self.assertEqual(state["phase"], "adaptive_research_pending")
        self.assertEqual(state["investigated_hypotheses"], 30)
        self.assertEqual(state["queue_counts"]["pending"], 0)
        registry = controller.read_json(self.run / "study_registry.json")
        self.assertTrue(all(entry["name"] is None for batch in registry["adaptive_batches"] for entry in batch["hypotheses"]))

    def test_additional_cohort_can_have_another_source_and_copy_grid(self):
        """Later matched counts and a new frozen checkout do not reopen the historical initial cohort."""
        registry, specifications = self.synthetic_initial_registry_and_additional_specs()
        next_commit = "b" * 40
        ideas.register_adaptive_batch(registry, specifications, "Synthetic profile fixture", next_commit)
        controller.atomic_json(self.run / "study_registry.json", registry)
        controller.atomic_json(self.sentinel, {**self.ready, "source_commit": next_commit,
                                               "matched_copy_counts": [1, 2]})
        for path in self.ready["gate_records"].values():
            controller.atomic_json(Path(path), {"passed": True, "source_commit": next_commit})
        state = controller.tick(self.run)
        self.assertEqual(state["phase"], "adaptive_benchmarks")
        self.assertEqual(state["queued"], 20)
        self.assertEqual(state["initial_remaining"], [])
        saved = controller.read_json(self.run / "study_registry.json")
        self.assertEqual(saved["reference_commit"], self.commit)
        self.assertEqual(saved["initial_matched_copy_counts"], [1])
        self.assertEqual(saved["adaptive_batches"][0]["matched_copy_counts"], [1, 2])
        self.assertTrue(all(controller.read_json(path)["hypothesis_position"] >= 31
                            for path in (self.run / "queue/pending").glob("*.json")))

    def test_additional_names_cannot_repeat_settings_or_parameter_trials(self):
        """Reject renamed trials and duplicate configurations before modifying any adaptive slots."""
        registry, specifications = self.synthetic_initial_registry_and_additional_specs()
        specifications[0]["learner"] = {"rollout_unroll": 2}
        with self.assertRaisesRegex(ValueError, "earlier setting"):
            ideas.register_adaptive_batch(registry, specifications, "Synthetic profile fixture", self.commit)
        self.assertTrue(all(entry["name"] is None for entry in registry["adaptive_batches"][0]["hypotheses"]))
        registry, specifications = self.synthetic_initial_registry_and_additional_specs()
        specifications[1]["learner"] = dict(specifications[0]["learner"])
        with self.assertRaisesRegex(ValueError, "identical configuration"):
            ideas.register_adaptive_batch(registry, specifications, "Synthetic profile fixture", self.commit)


class SourceAndImportTests(unittest.TestCase):
    """Check real source immutability and metadata-only imports without mocking their substance."""

    def test_source_changes_are_refused(self):
        """A real committed checkout is accepted, then rejected after its study source changes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            source = root / controller.SOURCE_SCOPES[0] / "configuration.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Queue protocol tests",
                            "-c", "user.email=queue-tests@example.invalid", "commit", "-qm", "Test source"], check=True)
            commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            controller.verify_source(root, commit)
            source.write_text("VALUE = 2\n")
            with self.assertRaisesRegex(ValueError, "uncommitted"):
                controller.verify_source(root, commit)

    def test_metadata_imports_do_not_load_jax(self):
        """Queue status and standard-library tests remain safe for a login node."""
        code = (f"import sys;sys.path.insert(0,{str(SCRIPT_ROOT)!r});"
                "import controller,queue_worker,run_case;assert 'jax' not in sys.modules")
        subprocess.run([sys.executable, "-I", "-c", code], check=True)

    def test_gpu_admission_observes_processes_without_stopping_them(self):
        """Filter compute processes by the exact leased UUID and perform only a read query."""
        response = mock.Mock(stdout="101, GPU-one\n202, GPU-two\n303, GPU-one\n")
        with mock.patch.object(queue_worker.subprocess, "run", return_value=response) as command:
            self.assertEqual(queue_worker.gpu_processes("GPU-one"), [101, 303])
        self.assertEqual(command.call_args.args[0][0], "nvidia-smi")

    def test_h100_mode_probe_targets_the_exact_uuid(self):
        """Confidential-computing and MIG checks query the selected card before it can be used."""
        responses = [mock.Mock(stdout="Disabled, 0\n"), mock.Mock(stdout="CC status: OFF\n")]
        with mock.patch.object(queue_worker, "gpu_processes", return_value=[]), mock.patch.object(
                queue_worker.subprocess, "run", side_effect=responses) as command:
            self.assertTrue(queue_worker.gpu_readiness("GPU-one", "h100")["ready"])
        self.assertEqual(command.call_args_list[1].args[0], ["nvidia-smi", "conf-compute", "-i", "GPU-one", "-f"])


if __name__ == "__main__":
    unittest.main()
