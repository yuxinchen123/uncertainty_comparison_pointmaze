"""Exercise inactive H100 placement, queue ownership and terminal publication with standard-library fixtures."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import h100_nvl_slurm_queue as queue
import h100_nvl_slurm_case as case_module
import h100_nvl_slurm_site_validation as site


class SlurmQueueTests(unittest.TestCase):
    """Keep every source, GPU and child fixture temporary; never import array packages."""

    def setUp(self):
        """Create a real metadata queue and a small immutable ordinary-case fixture."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name).resolve()
        self.controller = queue.load("controller")
        self.controller.layout(self.run)
        self.policy_path = self.run / "placement_policy.json"
        self.policy = {"campaign_run_dir": str(self.run), "case_time_budget_seconds": 3600,
            "shutdown_reserve_seconds": 120, "admission": {"files": []},
            "approved_targets": [{"hostname": "serval06", "gpu_uuid": "GPU-owned"}],
            "worker_helpers": {"source_root": str(self.run)}}
        self.policy_path.write_text(json.dumps(self.policy))
        self.source = str(self.run / "original")
        self.plan = {"numerical_source": {"source_root": self.source}, "reference": {}}
        self.case = {"unit_id": "original_case", "hardware_class": "h100", "source_commit": site.SOURCE_COMMIT,
            "source_root": self.source, "campaign_run_dir": str(self.run), "maximum_attempts": 3,
            "configuration": {"n_copies": 1}, "hypothesis": "original hypothesis", "candidate": {},
            "rounds": 7, "iterations": 10, "seed_start": 1}
        self.case["specification_digest"] = self.controller.content_digest(self.case)
        self.identifier = self.case["unit_id"]
        self.specification = self.run / "queue/specifications" / (self.identifier + ".json")
        self.controller.atomic_json(self.specification, self.case)
        self.pending = self.run / "queue/pending" / self.specification.name
        self.pending.write_bytes(self.specification.read_bytes())
        self.identity = {"worker_id": "worker", "worker_pid": 123, "worker_start_time": "456",
            "hostname": "serval06", "gpu_uuid": "GPU-owned", "slurm_job_id": "333", "cpu_affinity": list(range(8)),
            "placement_policy": site.receipt(self.policy_path)}
        self.allocation = {"path": str(self.run / "allocation/result.json"), "sha256": "allocation"}

    def claim_context(self):
        """Use the real fcntl queue lock while substituting only unavailable scientific admission inputs."""
        return (patch.object(queue, "validate_policy", return_value=(self.policy, site, self.plan)),
            patch.object(queue, "require_dispatch"),
            patch.object(self.controller, "ready_specification", return_value={"source_root": self.source, "source_commit": site.SOURCE_COMMIT}),
            patch.object(self.controller, "verify_source"),
            patch.object(queue, "load", side_effect=lambda name: self.controller if name == "controller" else site))

    def claim(self):
        """Claim through the production function with its real lock and original byte checks."""
        a, b, c, d, e = self.claim_context()
        with a, b, c, d, e:
            return queue.claim_next(self.policy_path, self.identity, self.allocation, 5000)

    def test_missing_policy_and_unpublished_dispatch_prevent_any_claim(self):
        """Neither a missing approval nor current unintegrated consumers can authorize a pending case."""
        with self.assertRaises(FileNotFoundError):
            queue.run_worker(self.run / "absent.json", self.run / "allocation", self.run / "output")
        unpublished = self.run / "unpublished_helpers"
        for name in ("controller.py", "post_comparison_validation.py"):
            path = unpublished / queue.PREFIX / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic helper without Slurm dispatch.\n")
        with self.assertRaisesRegex(ValueError, "not yet published"):
            queue.require_dispatch({"worker_helpers": {"source_root": str(unpublished)}})
        self.assertTrue(self.pending.exists())
        self.assertEqual(list((self.run / "queue/leases").iterdir()), [])

    def test_unpassed_site_rejects_even_otherwise_explicit_policy(self):
        """A reviewed-looking document cannot replace a genuinely completed site result."""
        policy = {**self.policy, "schema_version": 1, "kind": "h100_nvl_slurm_placement_policy", "scope": "exact_passed_site",
            "decision": "approved", "review_rationale": "reviewed exact site", "reviewed_at_utc": "2026-10-31T00:00:00+00:00",
            "runtime_environment": site.RUNTIME, "source_commit": site.SOURCE_COMMIT, "ordinary_attempt_limit": 3,
            "scientific_hypothesis_credit": 0, "site_directory": str(self.run / "site"),
            "site_result": {"path": str(self.run / "site/result.json"), "sha256": "absent"}}
        self.policy_path.write_text(json.dumps(policy))
        verifier = Mock()
        verifier.prepared.return_value = (self.run / "site", self.plan)
        verifier.receipt.side_effect = FileNotFoundError("actual site result absent")
        with patch.object(queue, "site_module", return_value=verifier):
            with self.assertRaises(FileNotFoundError):
                queue.validate_policy(self.policy_path)
        self.assertTrue(self.pending.exists())
        verifier.verify_result.assert_not_called()

    def test_site_verifier_loads_exact_detached_historical_root_not_adjacent_copy(self):
        """Use a real temporary Git source and ensure the historical self-path check receives its original root."""
        root = self.run / "historical_site_helpers"
        script = root / queue.PREFIX / "h100_nvl_slurm_site_validation.py"
        script.parent.mkdir(parents=True)
        script.write_bytes((SCRIPTS / script.name).read_bytes())
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(root), "-c", "user.name=Metadata test", "-c", "user.email=metadata@example.invalid",
            "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Create isolated site verifier fixture"], check=True)
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", "-q", commit], check=True)
        directory = self.run / "historical_site_validation"
        directory.mkdir()
        plan_path = directory / "site_validation_plan.json"
        self.controller.atomic_json(plan_path, {"helpers": site.source_identity(root, commit, [queue.PREFIX + script.name])})
        self.controller.atomic_json(directory / "manifest.yaml", {"plan": site.receipt(plan_path)})
        loaded = queue.site_module(directory)
        self.assertEqual(Path(loaded.__file__).resolve(), script)
        self.assertNotEqual(Path(loaded.__file__).resolve(), SCRIPTS / script.name)

    def test_two_workers_can_claim_one_case_only_and_keep_case_bytes(self):
        """The original fcntl lock serializes two workers and creates exactly one original attempt."""
        original = self.pending.read_bytes()
        a, b, c, d, e = self.claim_context()
        with a, b, c, d, e, ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(queue.claim_next, self.policy_path, self.identity, self.allocation, 5000) for _ in range(2)]
            claims = [future.result() for future in futures]
        self.assertEqual(sum(value is not None for value in claims), 1)
        case, lease = next(value for value in claims if value is not None)
        self.assertEqual((self.run / "queue/running" / self.specification.name).read_bytes(), original)
        self.assertEqual(case, self.case)
        self.assertEqual(lease["attempt"], 1)
        self.assertEqual(lease["execution_kind"], queue.EXECUTION_KIND)
        self.assertEqual(self.case["maximum_attempts"], 3)

    def test_exhausted_original_limit_cannot_create_fourth_attempt(self):
        """The new placement path provides no permission for a fourth or fifth ordinary attempt."""
        self.controller.atomic_json(self.run / "queue/leases" / self.specification.name, {"attempt": 3})
        with self.assertRaisesRegex(ValueError, "exhausted"):
            self.claim()
        self.assertTrue(self.pending.exists())

    def test_insufficient_walltime_never_takes_pending_case(self):
        """A queue worker exits rather than retaining work beyond its explicit reviewed budget."""
        a, b, c, d, e = self.claim_context()
        with a, b, c, d, e:
            self.assertIsNone(queue.claim_next(self.policy_path, self.identity, self.allocation, 3719))
            with self.assertRaisesRegex(ValueError, "finite"):
                queue.claim_next(self.policy_path, self.identity, self.allocation, float("nan"))
        self.assertTrue(self.pending.exists())
        now = datetime(2026, 10, 31, tzinfo=timezone.utc)
        self.assertEqual(queue.remaining_time("333", "JobId=333 JobState=RUNNING EndTime=2026-10-31T01:00:00", now), 3600)
        for text in ("JobId=334 JobState=RUNNING EndTime=2026-10-31T01:00:00", "JobId=333 JobState=PENDING EndTime=Unknown"):
            with self.assertRaises(ValueError):
                queue.remaining_time("333", text, now)

    def test_removed_optional_marker_still_requires_slurm_proof(self):
        """Every H100 Slurm result remains a strict new execution even without the lease marker."""
        self.assertTrue(queue.requires_execution(self.case, {}, {"hostname": "serval06", "slurm_job_id": "333"}))
        self.assertTrue(queue.requires_execution(self.case, {}, {"hostname": "substituted-host", "slurm_job_id": "333"}))
        self.assertFalse(queue.requires_execution(self.case, {}, {"hostname": "serval02", "slurm_job_id": ""}))

    def test_wrong_allocation_driver_or_other_board_fails_before_claim(self):
        """Resolving inside the step never permits broadening the reviewed exact-card scope."""
        proof = {"hostname": "serval06", "gpu_uuid": "GPU-owned", "job_id": "333", "physical_minor": "7",
            "driver": site.DRIVER, "slurm_allocations": {"SLURM_JOB_GPUS": "7", "SLURM_STEP_GPUS": "7"}}
        assignment = {"allocation": proof, "cpu_affinity": list(range(8))}
        fake = types.SimpleNamespace(require_owned_job=Mock(return_value={}), query=Mock(return_value="raw"),
            resolve_allocation=Mock(return_value=proof), DRIVER=site.DRIVER, require_runtime=Mock())
        with patch.object(queue.socket, "gethostname", return_value="serval06"), patch.object(os, "sched_getaffinity", return_value=set(range(8))), \
                patch.dict(os.environ, {"SLURM_STEP_ID": "0"}, clear=True), \
                patch.object(Path, "read_text", return_value="0::/slurm/job_333/step_0/task_0\n"):
            queue.current_allocation(self.policy, fake, assignment)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-owned")
            for values in ({**proof, "driver": "other"}, {**proof, "gpu_uuid": "GPU-other"}):
                fake.resolve_allocation.return_value = values
                with self.assertRaisesRegex(ValueError, "differs"):
                    queue.current_allocation(self.policy, fake, assignment)
        self.assertTrue(self.pending.exists())

    def test_full_allocation_configuration_and_software_cannot_change(self):
        """A matching card alone cannot substitute a different learning rate, interpreter or image packet."""
        names = ("campaign_run_dir", "numerical_source", "reference", "packages", "asset_record", "jaxtari_source_root",
            "python", "cleanrl_reference", "configuration", "runtime_environment", "admission")
        reference = {name: {"literal": name} for name in names}
        queue.compare_site_plans(reference, copy.deepcopy(reference))
        for name in names:
            changed = copy.deepcopy(reference)
            changed[name] = {"literal": "different"}
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "complete reviewed"):
                queue.compare_site_plans(reference, changed)

    def test_resolve_must_happen_inside_explicit_srun_step(self):
        """A batch shell cannot stand in for the step where CUDA ordinals and CPU binding are resolved."""
        with patch.dict(os.environ, {"SLURM_STEP_ID": "batch"}, clear=True):
            with self.assertRaisesRegex(ValueError, "inside.*srun"):
                queue.current_allocation(self.policy, site, {})

    def test_shutdown_during_case_defers_interruption_and_restores_handlers(self):
        """A graceful signal records the next-claim stop without interrupting current case work."""
        previous = signal.getsignal(signal.SIGTERM)
        completed = []
        with queue.stop_after_case() as requested:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            completed.append("current case reached its normal completion")
            self.assertEqual(requested, [signal.SIGTERM])
        self.assertEqual(completed, ["current case reached its normal completion"])
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_step_command_checks_policy_and_fresh_gates_inside_srun_before_worker(self):
        """The described wrapper does not resolve a UUID in the pre-srun batch environment."""
        verifier = Mock()
        verifier.prepared.return_value = (self.run / "allocation", {"python": sys.executable, "helpers": {"source_root": "/frozen/site"}})
        with patch.object(queue, "validate_policy", return_value=(self.policy, verifier, self.plan)), patch.object(queue, "site_module", return_value=verifier):
            command = queue.step_command(self.policy_path, self.run / "allocation", self.run / "worker")
        self.assertEqual(command[:2], ["srun", "--cpu-bind=threads"])
        script = command[-1]
        self.assertLess(script.index("validate-policy"), script.index("h100_nvl_slurm_site_validation.py run"))
        self.assertLess(script.index("h100_nvl_slurm_site_validation.py run"), script.index("h100_nvl_slurm_queue.py run"))

    def prepare_running(self):
        """Create one released ordinary lease without spawning a process."""
        case, lease = self.claim()
        lease.update(state="running", runner_pid=987, runner_start_time="654")
        unit, attempt = self.controller.attempt_paths(self.run, self.identifier, 1)
        self.controller.atomic_json(attempt / "lease.json", lease)
        self.controller.atomic_json(self.run / "queue/leases" / self.specification.name, lease)
        self.controller.atomic_json(attempt / "slurm_execution.json", {"test": "mocked scientific proof"})
        self.controller.atomic_json(attempt / "slurm_child_terminal.json", {"exit_code": 0})
        return case, lease, unit, attempt

    def test_nonzero_child_with_full_staged_result_never_publishes_completion(self):
        """An apparent complete raw result cannot override the actual child exit status."""
        case, lease, unit, attempt = self.prepare_running()
        staging = attempt / "slurm_terminal_staging"
        staging.mkdir()
        self.controller.atomic_json(staging / "attempt_result.json", {"case_completed": True})
        with patch.object(queue, "validate_execution") as validation:
            self.assertEqual(queue.finish_claim(self.run, case, lease, 1), "failed")
        validation.assert_not_called()
        self.assertFalse((unit / "result.json").exists())
        self.assertFalse((attempt / "result.json").exists())
        self.assertTrue((self.run / "queue/failed" / self.specification.name).exists())
        self.assertEqual(site.read(attempt / "lease.json")["state"], "failed")

    def test_missing_execution_record_with_exit_zero_is_failed(self):
        """Zero exit alone cannot publish any original terminal result."""
        case, lease, unit, attempt = self.prepare_running()
        with patch.object(queue, "validate_execution", side_effect=FileNotFoundError("missing execution record")):
            self.assertEqual(queue.finish_claim(self.run, case, lease, 0), "failed")
        self.assertFalse((attempt / "result.json").exists())

    def test_success_publishes_exact_raw_bytes_and_both_leases_before_queue_rename(self):
        """Only verified successful children expose unchanged terminal payloads and durable terminal leases."""
        case, lease, unit, attempt = self.prepare_running()
        staging = attempt / "slurm_terminal_staging"
        staging.mkdir()
        result = {"case_completed": True, "attempt": 1, "decision": "rejected"}
        files = {}
        for name in queue.TERMINAL_NAMES:
            value = result if name.endswith("result") else {"state": "completed", "decision": "rejected"}
            path = staging / (name + ".json")
            self.controller.atomic_json(path, value)
            files[name] = site.receipt(path)
        observed = []
        original_rename = Path.rename
        def rename(path, destination):
            """Observe both synced lease states at the exact final queue transition."""
            if path.parent == self.run / "queue/running":
                observed.append((site.read(attempt / "lease.json")["state"],
                    site.read(self.run / "queue/leases" / self.specification.name)["state"]))
            return original_rename(path, destination)
        with patch.object(queue, "validate_execution", return_value=({"terminal_files": files}, result)), patch.object(Path, "rename", rename):
            self.assertEqual(queue.finish_claim(self.run, case, lease, 0), "completed")
        self.assertEqual(observed, [("completed", "completed")])
        for name, path in queue.terminal_paths(unit, attempt).items():
            self.assertEqual(path.read_bytes(), Path(files[name]["path"]).read_bytes())

    def test_interrupted_claim_stays_recoverable_without_consuming_pending_attempt(self):
        """A failed first lease write leaves the original running entry without creating attempt data."""
        a, b, c, d, e = self.claim_context()
        with a, b, c, d, e, patch.object(self.controller, "atomic_json", side_effect=OSError("interrupted lease write")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                queue.claim_next(self.policy_path, self.identity, self.allocation, 5000)
        self.assertFalse(self.pending.exists())
        self.assertTrue((self.run / "queue/running" / self.specification.name).exists())
        self.assertFalse((self.run / "data" / self.identifier).exists())
        self.assertEqual(self.controller.recover_unstarted_claims(self.run), 1)
        self.assertTrue(self.pending.exists())
        self.assertEqual(self.claim()[1]["attempt"], 1)

    def test_changed_stage_under_lock_is_rejected_before_any_canonical_write(self):
        """The lock-protected byte check catches a file changed after outside validation."""
        case, lease, unit, attempt = self.prepare_running()
        staging = attempt / "slurm_terminal_staging"
        staging.mkdir()
        files = {}
        for name in queue.TERMINAL_NAMES:
            path = staging / (name + ".json")
            self.controller.atomic_json(path, {"case_completed": True, "attempt": 1})
            files[name] = site.receipt(path)
        def checked(*args):
            """Change a staged result after its successful proof was returned."""
            Path(files["unit_result"]["path"]).write_text('{"case_completed": false}\n')
            return {"terminal_files": files}, {"decision": "rejected"}
        with patch.object(queue, "validate_execution", side_effect=checked):
            with self.assertRaisesRegex(ValueError, "changed before publication"):
                queue.finish_claim(self.run, case, lease, 0)
        self.assertFalse((unit / "result.json").exists())
        self.assertFalse((attempt / "result.json").exists())

    def test_partial_publication_resumes_only_identical_intent_and_survives_terminal_lease(self):
        """A recorded partial publication can continue without accepting unrelated old output."""
        case, lease, unit, attempt = self.prepare_running()
        staging = attempt / "slurm_terminal_staging"
        staging.mkdir()
        files = {}
        for name in queue.TERMINAL_NAMES:
            path = staging / (name + ".json")
            self.controller.atomic_json(path, {"case_completed": True, "attempt": 1, "decision": "rejected"})
            files[name] = site.receipt(path)
        original = queue.publish_bytes
        calls = []
        def interrupted(path, contents):
            """Interrupt once after the first exact canonical result became durable."""
            calls.append(path)
            if len(calls) == 2:
                raise OSError("publication interrupted")
            original(path, contents)
        with patch.object(queue, "validate_execution", return_value=({"terminal_files": files}, {"decision": "rejected"})), \
                patch.object(queue, "publish_bytes", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "interrupted"):
                queue.finish_claim(self.run, case, lease, 0)
        self.assertTrue((attempt / "result.json").exists())
        self.assertTrue((attempt / "slurm_publication_intent.json").exists())
        with patch.object(queue, "validate_execution", return_value=({"terminal_files": files}, {"decision": "rejected"})):
            self.assertEqual(queue.finish_claim(self.run, case, lease, 0), "completed")
            completed_lease = site.read(attempt / "lease.json")
            self.assertEqual(queue.finish_claim(self.run, case, completed_lease, 0), "completed")

    def test_completed_leases_before_interrupted_final_rename_can_finish_exact_case(self):
        """A completed durable lease pair cannot leave an otherwise valid publication permanently running."""
        case, lease, unit, attempt = self.prepare_running()
        staging = attempt / "slurm_terminal_staging"
        staging.mkdir()
        files = {}
        for name in queue.TERMINAL_NAMES:
            path = staging / (name + ".json")
            self.controller.atomic_json(path, {"case_completed": True, "attempt": 1, "decision": "rejected"})
            files[name] = site.receipt(path)
        original_rename = Path.rename
        def interrupted(path, destination):
            """Interrupt the final queue rename after both terminal leases are already durable."""
            if path.parent == self.run / "queue/running":
                self.assertEqual(site.read(attempt / "lease.json")["state"], "completed")
                self.assertEqual(site.read(self.run / "queue/leases" / self.specification.name)["state"], "completed")
                raise OSError("final queue rename interrupted")
            return original_rename(path, destination)
        with patch.object(queue, "validate_execution", return_value=({"terminal_files": files}, {"decision": "rejected"})), \
                patch.object(Path, "rename", interrupted):
            with self.assertRaisesRegex(OSError, "final queue rename"):
                queue.finish_claim(self.run, case, lease, 0)
        self.assertTrue((self.run / "queue/running" / self.specification.name).exists())
        with patch.object(queue, "validate_execution", return_value=({"terminal_files": files}, {"decision": "rejected"})):
            self.assertEqual(queue.finish_claim(self.run, case, site.read(attempt / "lease.json"), 0), "completed")
        self.assertFalse((self.run / "queue/running" / self.specification.name).exists())
        self.assertTrue((self.run / "queue/completed" / self.specification.name).exists())

    def test_first_birth_or_owner_write_failure_stops_only_started_child(self):
        """A waiting Popen-owned process is stopped before the card-lock context can end."""
        case, lease, unit, attempt = self.prepare_running()
        for kind in ("birth", "owner"):
            with self.subTest(kind=kind):
                child = Mock(pid=111)
                fake_site = Mock()
                fake_site.write_once.side_effect = [None, RuntimeError("owner write")] if kind == "owner" else None
                fake_site.process_start_time.side_effect = RuntimeError("birth read") if kind == "birth" else None
                fake_site.process_start_time.return_value = "222"
                def loader(name):
                    return self.controller if name == "controller" else fake_site
                with patch.object(queue, "load", side_effect=loader), patch.object(queue.subprocess, "Popen", return_value=child):
                    with self.assertRaises(RuntimeError):
                        queue.run_child(self.run, case, lease, self.policy_path, self.run / "allocation", 3)
                fake_site.stop_child.assert_called_once_with(child, None if kind == "birth" else "222")
                (attempt / "runner.log").unlink()

    def test_full_execution_receipt_survives_terminal_leases_and_rejects_changed_proof(self):
        """Validate raw child/publication receipts through the historical API before and after lease finalization."""
        case, lease, unit, attempt = self.prepare_running()
        history = queue.load("h100_nvl_slurm_evidence")
        post = queue.load("post_comparison_validation")
        self.controller.atomic_json(attempt / "case.json", case)
        allocation_directory = self.run / "allocation"
        allocation_directory.mkdir()
        self.controller.atomic_json(allocation_directory / "result.json", {"passed": True})
        lease["allocation_validation"] = site.receipt(allocation_directory / "result.json")
        self.controller.atomic_json(attempt / "lease.json", lease)
        self.controller.atomic_json(self.run / "queue/leases" / self.specification.name, lease)
        self.controller.atomic_json(attempt / "slurm_released_lease.json", lease)
        owner = {"pid": 987, "process_start_time": "654", "parent_pid": 123, "parent_start_time": "456", "job_id": "333"}
        self.controller.atomic_json(attempt / "slurm_child_owner.json", owner)
        (attempt / "runner.log").write_text("original unchanged benchmark output\n")
        terminal = {"exit_code": 0, "owner": site.receipt(attempt / "slurm_child_owner.json"), "log": site.receipt(attempt / "runner.log"),
            "absent_files": [str(attempt / "failure.json")]}
        self.controller.atomic_json(attempt / "slurm_child_terminal.json", terminal)
        installed = {"packages": {"jax": "captured version"}}
        assets = {"tree_digest": "captured assets"}
        for name, value in (("installed", installed), ("assets", assets)):
            self.controller.atomic_json(self.run / (name + ".json"), value)
        # Full site/Git/backend semantics are covered by the separate historical suite; retain real child and publication bytes here.
        plan = {name: {} for name in ("campaign_run_dir", "numerical_source", "reference", "packages", "asset_record", "jaxtari_source_root",
            "python", "cleanrl_reference", "configuration", "runtime_environment", "admission")}
        plan["schema_version"] = 2
        plan["numerical_source"] = {"source_root": self.source}
        plan["reference"] = {"shared_installed_backend": site.receipt(self.run / "installed.json"),
            "shared_asset_receipt": site.receipt(self.run / "assets.json"), "backend": {}}
        hardware = {"pid": 987, "process_start_time": "654", "gpu_uuid": "GPU-owned", "slurm_job_id": "333", "hostname": "serval06",
            "source_root": self.source, "source_commit": site.SOURCE_COMMIT,
            "gpu_query": "GPU-owned, NVIDIA H100 NVL, 95830 MiB, " + site.DRIVER}
        self.controller.atomic_json(attempt / "hardware.json", hardware)
        staging = attempt / "slurm_terminal_staging"
        staging.mkdir()
        result = {"case_completed": True, "measurement_valid": False, "decision": "rejected", "failure_stage": "correctness",
            "correctness_attempted": True, "correctness_passed": False, "error": "synthetic receipt test mismatch",
            "specification_digest": case["specification_digest"],
            "attempt": 1, "unit_id": self.identifier, "source_commit": site.SOURCE_COMMIT, "hardware_class": "h100",
            "idea": case["hypothesis"], "copies": 1}
        progress = {"state": "completed", "case_completed": True, "attempt": 1, "unit_id": self.identifier, "pid": 987, "decision": "rejected"}
        files = {}
        for name in queue.TERMINAL_NAMES:
            path = staging / (name + ".json")
            self.controller.atomic_json(path, result if name.endswith("result") else progress)
            files[name] = site.receipt(path)
        record = {"kind": queue.EXECUTION_KIND, "schema_version": 1, "placement_policy": site.receipt(self.policy_path),
            "allocation_validation": lease["allocation_validation"], "lease_token": lease["token"], "attempt": 1,
            "specification_digest": case["specification_digest"], "source_commit": site.SOURCE_COMMIT, "pid": 987,
            "process_start_time": "654", "gpu_uuid": "GPU-owned", "job_id": "333", "runtime_environment": site.RUNTIME,
            "cpu_affinity": list(range(8)), "installed_backend": installed, "asset_receipt": assets,
            "loaded_libraries": {"pid": 987}, "module_paths": {}, "jaxtari_import": {"captured": "installed import identity"}, "terminal_files": files,
            "released_lease": site.receipt(attempt / "slurm_released_lease.json"), "hardware": site.receipt(attempt / "hardware.json")}
        self.controller.atomic_json(attempt / "slurm_execution.json", record)
        scope = {"hostname": "serval06", "gpu_uuid": "GPU-owned", "job_id": "333"}
        def queue_loader(name):
            """Use the real historical validator while substituting no child or terminal metadata."""
            return history if name == "h100_nvl_slurm_evidence" else site if name == "h100_nvl_slurm_site_validation" else self.controller
        with patch.object(queue, "load", side_effect=queue_loader), \
                patch.object(history, "load", side_effect=lambda name: post if name == "post_comparison_validation" else site), \
                patch.object(history, "policy_evidence", return_value=(self.policy, plan)), \
                patch.object(history, "site_evidence", return_value=(plan, {"cpu_affinity": list(range(8))}, scope)), \
                patch.object(history, "backend"), patch.object(history, "module_paths"), patch.object(history, "scientific_result"):
            self.assertEqual(queue.validate_execution(attempt, case, lease)[1], result)
            self.assertEqual(queue.finish_claim(self.run, case, lease, 0), "completed")
            completed_lease = site.read(attempt / "lease.json")
            self.assertEqual(queue.validate_execution(attempt, case, completed_lease, require_completed=True)[1], result)
            with self.assertRaisesRegex(ValueError, "matching original ordinary lease"):
                queue.validate_execution(attempt, case, {**completed_lease, "runner_start_time": "wrong"})
            Path(files["unit_result"]["path"]).write_text('{"case_completed":false}')
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                queue.validate_execution(attempt, case, completed_lease)


class OriginalWriterTests(unittest.TestCase):
    """Spy on the exact original writer without loading a model or opening a GPU."""

    def setUp(self):
        """Create four canonical paths and an original serializer fixture."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.unit = Path(self.temporary.name).resolve()
        self.attempt = self.unit / "attempts/attempt-0001"
        self.attempt.mkdir(parents=True)
        self.writes = []
        def original(path, value):
            """Serialize exactly like original atomic_json while recording all actual destinations."""
            self.writes.append(Path(path))
            Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        self.runner = types.SimpleNamespace(atomic_json=original)

    def test_only_four_terminal_writes_are_staged_before_any_completed_publication(self):
        """Intermediates remain visible while all original result/completed records remain private until proof."""
        original = self.runner.atomic_json
        with case_module.terminal_writer(self.runner, self.unit, self.attempt, queue) as records:
            self.runner.atomic_json(self.attempt / "progress.json", {"state": "initializing"})
            self.runner.atomic_json(self.unit / "progress.json", {"state": "initializing"})
            for name, path in queue.terminal_paths(self.unit, self.attempt).items():
                value = {"case_completed": True}
                if name.endswith("progress"):
                    value["state"] = "completed"
                self.runner.atomic_json(path, value)
            self.assertEqual(set(records), set(queue.TERMINAL_NAMES))
            self.assertFalse((self.attempt / "result.json").exists())
            self.assertFalse((self.unit / "result.json").exists())
            self.assertEqual(site.read(self.unit / "progress.json")["state"], "initializing")
        self.assertIs(self.runner.atomic_json, original)
        self.assertEqual(len([path for path in self.writes if path.parent == self.attempt / "slurm_terminal_staging"]), 4)

    def test_unknown_terminal_path_state_or_duplicate_is_rejected(self):
        """A later frozen-source change cannot quietly publish an extra or differently named terminal record."""
        with case_module.terminal_writer(self.runner, self.unit, self.attempt, queue):
            for path, value in ((self.unit / "other/result.json", {"case_completed": True}),
                (self.unit / "progress.json", {"state": "finished"}),
                (self.unit / "progress.json", {"state": "running", "case_completed": True})):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    self.runner.atomic_json(path, value)
            self.runner.atomic_json(self.unit / "result.json", {"case_completed": True})
            with self.assertRaisesRegex(ValueError, "twice"):
                self.runner.atomic_json(self.unit / "result.json", {"case_completed": True})

    def test_real_fresh_metadata_import_boundary_and_alias_resolution(self):
        """A fresh Python process loads the staged metadata names without importing controller or arrays."""
        program = """import importlib.util,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); spec=importlib.util.spec_from_file_location('literal_case_alias',root/'h100_nvl_slurm_case.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
queue=module.load_queue(); site=queue.load('h100_nvl_slurm_site_validation'); training=site.load('train_optimized')
assert pathlib.Path(queue.__file__).resolve()==root/'h100_nvl_slurm_queue.py'
assert pathlib.Path(site.__file__).resolve()==root/'h100_nvl_slurm_site_validation.py'
module.require_clean_imports()
print(json.dumps({'clean':True}))
"""
        result = subprocess.run([sys.executable, "-I", "-c", program, str(SCRIPTS)], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout), {"clean": True})


if __name__ == "__main__":
    unittest.main()
