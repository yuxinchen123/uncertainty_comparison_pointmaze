"""Check exact retry approvals, claims and original-result proofs using only standard-library fixtures."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import infrastructure_retry as retry
import infrastructure_retry_worker as worker
import run_infrastructure_retry as runner
import controller
import post_comparison_validation as validation


class RetryTests(unittest.TestCase):
    """Keep every test free of numerical imports, GPU queries and existing campaign writes."""

    def setUp(self):
        """Create an original held failure and independently stored fourth-attempt evidence."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name).resolve()
        self.source, self.helpers = self.run / "frozen", self.run / "helpers"
        self.unit = "approved-original-case"
        controller.layout(self.run)
        self.case_path = self.run / "queue/specifications" / (self.unit + ".json")
        self.case = {"unit_id": self.unit, "hypothesis_position": 18, "hypothesis": "batch_rollout_intrinsic_rewards",
            "source_root": str(self.source), "source_commit": retry.SOURCE_COMMIT,
            "campaign_run_dir": str(self.run), "hardware_class": "jaguar03_rtx_a4500",
            "configuration": {"n_copies": 2}, "candidate": {"learner": {"hoist_rollout_bonus": True}, "environment": {}, "execution": {}},
            "rounds": 7, "iterations": 1, "seed_start": 1, "maximum_attempts": 3}
        self.case["specification_digest"] = controller.content_digest(self.case)
        self.write(self.case_path, self.case)
        self.original_bytes = self.case_path.read_bytes()
        approved = {18: (self.case["hypothesis"], self.case["candidate"]["learner"], retry.digest(self.original_bytes))}
        self.addCleanup(patch.stopall)
        patch.object(retry, "APPROVED", approved).start()
        patch.object(retry, "validate_git_receipt").start()
        self.previous = self.run / "data" / self.unit / "attempts/attempt-0003"
        self.prior_lease = {"unit_id": self.unit, "attempt": 3, "state": "failed", "token": "original-token"}
        self.write(self.previous / "lease.json", self.prior_lease)
        self.write(self.run / "queue/leases" / (self.unit + ".json"), self.prior_lease)
        self.write(self.previous / "failure.json", {"case_completed": False, "retryable": True, "attempt": 3,
            "source_commit": retry.SOURCE_COMMIT, "error": "RESOURCE_EXHAUSTED: Out of memory in jit_iteration"})
        self.write(self.run / "queue/failed" / (self.unit + ".json"), self.case)
        self.region = self.region_record()
        memory = {"gpu_uuid": "GPU-five", "other_gpu_processes": [], "sampled_process_peak_bytes": 1,
                  "allocator": {"peak_bytes_in_use": 1}}
        row = {"iterations": 1, "copies": 2, "environments_per_copy": 128, "rollout_steps": 128,
               "training_transitions": 32768, "seconds": 1., "memory": memory}
        measured = {"measurement_valid": True, "contaminated": False, "copies": 2, "hardware_class": "jaguar03_rtx_a4500",
            "copy_seeds": [1, 2], "correctness_passed": True, "correctness_attempted": True,
            "correctness": {"passed": True, "complete_iteration": {"leaves": 2},
                "supplied_inputs": {name: {"leaves": 2} for name in ("preparation", "fixed_batch", "fixed_batch_update")}},
            "rounds": [{"order": ["reference", "candidate"] if i % 2 == 0 else ["candidate", "reference"],
                        "reference": row, "candidate": row} for i in range(7)]}
        diagnosis = {"completed": True, "kind": "allocator_region_history_diagnosis", "source_commit": retry.SOURCE_COMMIT,
            "eligible_for_formal_campaign_admission": False, "hypothesis_count_contribution": 0, "runtime_environment": retry.RUNTIME,
            "paired_measurement": measured, "hardware": {"gpu_uuid": "GPU-five"}, "region_preparation": self.region,
            "helpers": {"source_root": str(self.helpers), "source_commit": "a" * 40, "files": {}}}
        self.diagnosis_path = self.run / "diagnosis/result.json"
        self.write(self.diagnosis_path, diagnosis)
        patch.object(retry, "DIAGNOSIS_SHA256", retry.digest(self.diagnosis_path.read_bytes())).start()
        self.approval_path = self.run / retry.APPROVALS_PATH / (self.unit + ".json")
        self.approval = {"schema_version": 1, "kind": "allocator_region_infrastructure_retry", "unit_id": self.unit,
            "campaign_run_dir": str(self.run), "approved_attempt": 4, "previous_attempt": 3,
            "case": self.receipt(self.case_path), "previous_lease": self.receipt(self.previous / "lease.json"),
            "previous_failure": self.receipt(self.previous / "failure.json"), "diagnosis": self.receipt(self.diagnosis_path),
            "helpers": {"source_root": str(self.helpers), "source_commit": "b" * 40, "files": {retry.PREFIX + n: "c" * 64 for n in retry.HELPERS}},
            "numerical_source": {"source_root": str(self.source), "source_commit": retry.SOURCE_COMMIT,
                "files": {retry.PREFIX + n: "d" * 64 for n in ("run_case.py", "controller.py", "ideas.py", "paired_benchmark.py", "worker_environment.sh")}},
            "runtime_environment": retry.RUNTIME, "allocator": retry.ALLOCATOR, "preparation": retry.PREPARATION,
            "asset_record": {}, "jaxtari_source_root": str(self.run / "jaxtari"), "python": sys.executable}
        patch.object(retry, "numerical_source_paths", return_value=sorted(self.approval["numerical_source"]["files"])).start()
        self.old_approval_path = self.run / retry.ORIGINAL_APPROVALS_PATH / (self.unit + ".json")
        old = copy.deepcopy(self.approval)
        old["helpers"].update(source_root=str(self.run / "original_helpers"), source_commit="e" * 40)
        self.write(self.old_approval_path, old)
        self.old_approval_bytes = self.old_approval_path.read_bytes()
        failure_log = self.run / "preflight/failure.log"
        cpu_log = self.run / "preflight/cpu.log"
        failure_log.parent.mkdir(parents=True)
        failure_log.write_text("original failure before a comparison claim\n")
        cpu_log.write_text("original metadata CPU observation\n")
        preflight = {"kind": "allocator_retry_preflight_failure", "comparison_attempts_consumed": 0,
            "failed_job_id": "6778001", "metadata_diagnosis_job_id": "6778002", "source_commit": retry.SOURCE_COMMIT,
            "helper_commit": "e" * 40, "failure_log": self.receipt(failure_log), "cpu_diagnosis_log": self.receipt(cpu_log),
            "unclaimed_original_approvals": [{**self.receipt(self.old_approval_path), "unit_id": self.unit, "current_attempt": 3,
                "no_fourth_attempt_directory_observed": True, "no_retry_claim_observed": True}],
            "cpu_observation": {"hostname": "jaguar03", "environment": self.resolved_cpu_environment(),
                "affinity": list(range(56, 76)) + list(range(168, 188))}}
        preflight_path = self.write(self.run / "preflight/receipt.json", preflight)
        patch.object(retry, "PREFLIGHT_FAILURE_SHA256", retry.digest(preflight_path.read_bytes())).start()
        self.approval.update(supersedes_approval=self.receipt(self.old_approval_path),
            preflight_failure=self.receipt(preflight_path), operational_revision=retry.OPERATIONAL_REVISION)
        self.write(self.approval_path, self.approval)
        self.assignment_path = self.run / "retry-node/assignment.json"
        self.assignment = self.assignment_record()
        self.assignment["approvals"] = [self.receipt(self.approval_path)]
        self.write(self.assignment_path, self.assignment)

    def write(self, path, value):
        """Store isolated synthetic artifacts without touching existing run outputs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(retry.json_bytes(value))
        return path

    def receipt(self, path):
        """Hash exact fixture bytes through the production reader."""
        return retry.file_record(path, retry.LocalEvidence())

    def region_record(self):
        """Represent distinct actual-style memory observations before, during and after one allocation."""
        result = {"allocation_count": 1, "array_nbytes": retry.REGION_BYTES, "array_dtype": "uint8",
            "array_shape": [9, 1024, 1024, 1024], "allocator_pool_release_assumed": False}
        for index, name in enumerate(("before", "allocated", "released")):
            result[name] = {"time_seconds": 101. + index, "device_memory": {"gpu_uuid": "GPU-five", "total_bytes": 20 * 2 ** 30,
                "used_bytes": 16 * 2 ** 30, "free_bytes": 4 * 2 ** 30, "processes": []},
                "allocator": {"bytes_limit": 17 * 2 ** 30, "bytes_in_use": retry.REGION_BYTES if name == "allocated" else 0,
                    "largest_free_block_bytes": 16 * 2 ** 30}}
        return result

    def resolved_cpu_environment(self):
        """Retain the exact Slurm fields observed in the owned forty-CPU metadata diagnosis."""
        mask = "0x000000000FFFFF00000000000000000000000FFFFF00000000000000"
        return {"SLURM_CPUS_PER_TASK": "40", "SLURM_CPU_BIND_TYPE": "mask_cpu:",
            "SLURM_CPU_BIND_LIST": mask, "SLURM_CPU_BIND": "quiet,mask_cpu:" + mask}

    def assignment_record(self):
        """Model two physical cards with remapped visible indices and unchanged Slurm minor allocations."""
        xml = "<nvidia_smi_log>" + "".join(f"<gpu><uuid>GPU-{word}</uuid><product_name>NVIDIA RTX A4500</product_name><minor_number>{minor}</minor_number></gpu>"
            for word, minor in (("five", 5), ("seven", 7))) + "</nvidia_smi_log>"
        roster = ["GPU-five", "GPU-seven"]
        return {"hostname": "jaguar03", "parent_pid": 123, "parent_process_start_time": "321", "gpu_uuids": roster,
            "expected_gpu_count": 2, "cpu_allocation": retry.cpu_allocation({"SLURM_CPUS_PER_TASK": "16", "SLURM_CPU_BIND_TYPE": "threads"}, roster, range(16)),
            "allocation": {"hostname": "jaguar03", "slurm_job_id": "6779999", "slurm_allocations": {"SLURM_JOB_GPUS": "5,7", "SLURM_STEP_GPUS": "5,7"},
                "cuda_visible_devices_original": "0,1", "gpu_csv": "0, GPU-five, NVIDIA RTX A4500\n1, GPU-seven, NVIDIA RTX A4500\n", "gpu_xml": xml,
                "cards": [{"visible_index": str(index), "gpu_uuid": "GPU-" + word, "physical_minor": str(minor), "name": "NVIDIA RTX A4500"}
                          for index, (word, minor) in enumerate((("five", 5), ("seven", 7)))]}}

    def completed_attempt(self):
        """Build a result-bound receipt with distinct preserved lease, claim and process metadata."""
        attempt = self.run / "data" / self.unit / "attempts/attempt-0004"
        lease = {"unit_id": self.unit, "attempt": 4, "state": "running", "token": "fourth-token", "worker_pid": 12,
            "worker_start_time": "21", "runner_pid": 34, "runner_start_time": "43", "gpu_uuid": "GPU-five", "slurm_job_id": "6779999",
            "cpu_affinity": list(range(8)), "thread_environment": retry.THREAD_ENVIRONMENT,
            "infrastructure_retry_approval": self.receipt(self.approval_path), "assignment": self.receipt(self.assignment_path)}
        self.write(attempt / "lease.json", lease)
        self.write(self.run / "queue/infrastructure_retries/claims" / (self.unit + ".json"), {"approval": self.receipt(self.approval_path), "lease": lease})
        self.write(attempt / "hardware.json", {"pid": 34, "process_start_time": "43", "worker_pid": 12, "worker_start_time": "21",
            "gpu_uuid": "GPU-five", "slurm_job_id": "6779999"})
        result_path = self.write(attempt / "result.json", {"case_completed": True, "measurement_valid": True, "attempt": 4,
            "unit_id": self.unit, "idea": self.case["hypothesis"], "specification_digest": self.case["specification_digest"],
            "source_commit": retry.SOURCE_COMMIT, "hardware_class": "jaguar03_rtx_a4500", "copies": 2})
        self.write(attempt / "region_preparation.json", self.region)
        sidecar = {"schema_version": 1, "completed": True, "unit_id": self.unit, "attempt": 4,
            "approval": self.receipt(self.approval_path), "lease_token": "fourth-token", "helpers": self.approval["helpers"],
            "numerical_source": self.approval["numerical_source"], "runner": str(self.helpers / retry.PREFIX / "run_infrastructure_retry.py"),
            "original_result": self.receipt(result_path), "region_preparation": self.receipt(attempt / "region_preparation.json"),
            "assignment": self.receipt(self.assignment_path), "pid": 34, "process_start_time": "43", "gpu_uuid": "GPU-five",
            "cuda_visible_devices": "GPU-five", "slurm_allocations": self.assignment["allocation"]["slurm_allocations"],
            "runtime_environment": {**retry.RUNTIME, **retry.ALLOCATOR, "JAX_ENABLE_COMPILATION_CACHE": "false",
                "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "", "JAXTARI_SKIP_SPRITE_UPDATE": "1", **retry.THREAD_ENVIRONMENT},
            "cpu_affinity": list(range(8)), "thread_environment": retry.THREAD_ENVIRONMENT,
            "frozen_modules": {name: str(self.source / retry.PREFIX / (name + ".py")) for name in ("run_case", "controller", "ideas", "paired_benchmark")},
            "started_at_seconds": 100., "finished_at_seconds": 110.}
        self.write(attempt / "infrastructure_retry_execution.json", sidecar)
        return result_path, lease, sidecar

    def test_approval_accepts_only_the_exact_original_case_and_diagnosis(self):
        """Case edits or a copied Boolean-only diagnostic cannot enlarge the approved scope."""
        approved, case = retry.validate_approval(self.approval_path)
        self.assertEqual(case["maximum_attempts"], 3)
        self.assertEqual(approved["approved_attempt"], 4)
        self.case_path.write_bytes(self.original_bytes + b" ")
        with self.assertRaisesRegex(ValueError, "original case bytes"):
            retry.validate_approval(self.approval_path)
        self.case_path.write_bytes(self.original_bytes)
        altered = copy.deepcopy(self.approval)
        altered["approved_attempt"] = 5
        self.write(self.approval_path, altered)
        with self.assertRaisesRegex(ValueError, "one-shot attempt"):
            retry.validate_approval(self.approval_path)

    def test_real_diagnosis_reports_are_required_not_seven_empty_rounds(self):
        """Even a matching diagnosis byte hash requires actual original correctness and measured work."""
        diagnosis = json.loads(self.diagnosis_path.read_bytes())
        diagnosis["paired_measurement"]["rounds"] = [{}] * 7
        self.write(self.diagnosis_path, diagnosis)
        receipt = self.receipt(self.diagnosis_path)
        with patch.object(retry, "DIAGNOSIS_SHA256", receipt["sha256"]):
            with self.assertRaisesRegex(ValueError, "alternate"):
                retry.validate_diagnosis(receipt, retry.LocalEvidence())

    def test_claim_consumes_only_one_approval_without_changing_case_limit(self):
        """The next attempt is four; unchanged ordinary pending and old attempts remain preserved."""
        original_failure = (self.previous / "failure.json").read_bytes()
        ready = {"source_root": str(self.source), "source_commit": retry.SOURCE_COMMIT}
        with patch.object(controller, "ready_specification", return_value=ready), patch.object(worker, "verify_worker_source"):
            claimed = worker.claim_retry(self.run, {"worker_pid": 12, "worker_start_time": "21"}, self.assignment_path)
            self.assertEqual(claimed[1]["attempt"], 4)
            self.assertEqual(worker.claim_retry(self.run, {}, self.assignment_path), None)
        self.assertEqual(self.case_path.read_bytes(), self.original_bytes)
        self.assertEqual((self.previous / "failure.json").read_bytes(), original_failure)
        self.assertEqual(list((self.run / "queue/pending").glob("*.json")), [])
        self.assertTrue((self.run / "queue/running" / (self.unit + ".json")).exists())
        self.assertEqual(json.loads(self.case_path.read_bytes())["maximum_attempts"], 3)
        self.assertEqual(self.old_approval_path.read_bytes(), self.old_approval_bytes)
        self.assertTrue((self.run / retry.CLAIMS_PATH / (self.unit + ".json")).exists())
        self.assertFalse((self.approval_path.parent.parent / "claims").exists())

    def test_new_worker_does_not_consume_old_approval_namespace(self):
        """An old unclaimed approval remains historical metadata rather than implicit new authorization."""
        self.approval_path.unlink()
        ready = {"source_root": str(self.source), "source_commit": retry.SOURCE_COMMIT}
        with patch.object(controller, "ready_specification", return_value=ready):
            self.assertIsNone(worker.claim_retry(self.run, {}, self.assignment_path))
        self.assertEqual(self.old_approval_path.read_bytes(), self.old_approval_bytes)
        self.assertTrue((self.run / "queue/failed" / (self.unit + ".json")).exists())

    def test_replacement_is_bound_to_the_old_approval_and_preflight_bytes(self):
        """Neither a changed earlier approval nor a changed preflight record can grant the operational replacement."""
        retry.validate_approval(self.approval_path)
        self.old_approval_path.write_bytes(self.old_approval_bytes + b" ")
        with self.assertRaisesRegex(ValueError, "evidence bytes changed"):
            retry.validate_approval(self.approval_path)
        self.old_approval_path.write_bytes(self.old_approval_bytes)
        preflight = Path(self.approval["preflight_failure"]["path"])
        preflight.write_bytes(preflight.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "evidence bytes changed"):
            retry.validate_approval(self.approval_path)

    def test_valid_sidecar_binds_raw_bytes_and_keeps_original_result_unchanged(self):
        """A separately approved runtime receipt can count the exact original result without rewriting it."""
        path, lease, _ = self.completed_attempt()
        original = path.read_bytes()
        proof = retry.validate_execution(path, self.case, lease)
        self.assertEqual(proof["scope"], "actual_infrastructure_retry_runtime")
        self.assertEqual(path.read_bytes(), original)
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "exact approved"):
            retry.validate_execution(path, self.case, lease)

    def test_missing_or_changed_process_or_runtime_proof_is_rejected(self):
        """A raw numerical success alone, another PID, or an altered allocator cannot complete attempt four."""
        path, lease, sidecar = self.completed_attempt()
        sidecar_path = path.parent / "infrastructure_retry_execution.json"
        for field, value in (("pid", 999), ("runtime_environment", {**sidecar["runtime_environment"], "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9"})):
            changed = {**sidecar, field: value}
            self.write(sidecar_path, changed)
            with self.assertRaises(ValueError):
                retry.validate_execution(path, self.case, lease)
        sidecar_path.unlink()
        with self.assertRaises(FileNotFoundError):
            controller.validate_attempt_completion(self.run, self.case, lease, path)

    def test_full_slurm_allocation_is_preserved_with_physical_minor_mapping(self):
        """Visible zero can be physical minor five; the child must retain both five and seven."""
        retry.validate_assignment(self.assignment, "GPU-five")
        changed = copy.deepcopy(self.assignment)
        changed["allocation"]["slurm_allocations"]["SLURM_JOB_GPUS"] = "5"
        with self.assertRaisesRegex(ValueError, "full Slurm"):
            retry.validate_assignment(changed, "GPU-five")
        with self.assertRaisesRegex(ValueError, "physical GPU roster"):
            retry.validate_assignment(self.assignment, "GPU-outside")

    def test_prepare_region_has_one_allocation_and_no_cache_change(self):
        """Preparation finishes before original learner work and never calls clear_caches."""
        events, snapshots = [], iter(self.region[name] for name in ("before", "allocated", "released"))
        array = types.SimpleNamespace(nbytes=retry.REGION_BYTES, dtype="uint8", delete=lambda: events.append("delete"))
        arrays = types.SimpleNamespace(uint8="uint8", zeros=Mock(side_effect=lambda *a, **k: events.append("allocate") or array))
        runtime = types.SimpleNamespace(block_until_ready=lambda x: events.append("synchronize"), clear_caches=Mock())
        with patch.object(runner.gc, "collect", return_value=0):
            result = runner.prepare_region(runtime, arrays, lambda: next(snapshots))
        self.assertEqual(events, ["allocate", "synchronize", "delete"])
        self.assertEqual(result["array_nbytes"], retry.REGION_BYTES)
        runtime.clear_caches.assert_not_called()

    def test_actual_device_reserved_memory_is_preserved_in_region_proof(self):
        """The reviewed card reports a real remainder; that memory is never counted as free."""
        region = copy.deepcopy(self.region)
        for phase in ("before", "allocated", "released"):
            used, free = (207618048, 20808990720) if phase == "before" else (17393778688, 3622830080)
            region[phase]["device_memory"].update(total_bytes=21464350720, used_bytes=used, free_bytes=free)
        retry.validate_region(region, "GPU-five")
        self.assertEqual(region["released"]["device_memory"]["free_bytes"], 3622830080)
        self.assertEqual(21464350720 - 17393778688 - 3622830080, 447741952)
        for bad in (-1, True, 3622830080.0):
            changed = copy.deepcopy(region)
            changed["released"]["device_memory"]["free_bytes"] = bad
            with self.assertRaisesRegex(ValueError, "actual selected-device"):
                retry.validate_region(changed, "GPU-five")

    def test_current_controller_shadowing_is_rejected_before_original_import(self):
        """A fresh numerical child cannot inherit the metadata worker's current controller module."""
        with patch.dict(sys.modules, {"controller": types.SimpleNamespace(__file__="/changed/controller.py")}):
            with self.assertRaisesRegex(ValueError, "before current"):
                runner.load_original_runner(self.case)

    def test_fresh_standard_library_process_imports_original_modules(self):
        """The runner's real fresh-process import order resolves D28 files before any current helpers."""
        directory = self.source / retry.PREFIX
        directory.mkdir(parents=True)
        (directory / "controller.py").write_text("identity = 'original controller'\n")
        (directory / "ideas.py").write_text("identity = 'original ideas'\n")
        (directory / "run_case.py").write_text("import controller, ideas\n")
        code = "import sys,json;sys.path.insert(0,sys.argv[1]);import run_infrastructure_retry as r;r.load_original_runner(json.loads(sys.argv[2]));assert 'jax' not in sys.modules;print(sys.modules['controller'].identity)"
        outcome = subprocess.run([sys.executable, "-I", "-c", code, str(SCRIPTS), json.dumps(self.case)], capture_output=True, text=True, check=True)
        self.assertEqual(outcome.stdout.strip(), "original controller")

    def test_fourth_attempt_failure_cannot_be_retried_under_the_original_limit(self):
        """The approved receipt grants no fifth attempt or silent maximum-attempt increase."""
        self.write(self.run / "queue/leases" / (self.unit + ".json"), {"attempt": 4})
        self.write(self.run / "data" / self.unit / "failure.json", {"retryable": True})
        self.assertFalse(controller.retry_failed(self.run, self.unit))

    def test_immutable_publication_never_overwrites_existing_bytes(self):
        """Approval and sidecar publishing cannot replace completed proof with different contents."""
        path = self.run / "immutable.json"
        retry.write_once(path, b"original\n")
        retry.write_once(path, b"original\n")
        with self.assertRaisesRegex(ValueError, "already differs"):
            retry.write_once(path, b"changed\n")
        self.assertEqual(path.read_bytes(), b"original\n")

    def test_git_receipt_rejects_nested_archive_instead_of_inheriting_parent_identity(self):
        """An actual detached root is required even if an archive sits under that root."""
        root = self.run / "actual-git"
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        (root / "file.py").write_text("pass\n")
        subprocess.run(["git", "-C", str(root), "add", "file.py"], check=True)
        subprocess.run(["git", "-C", str(root), "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        subprocess.run(["git", "-C", str(root), "checkout", "-q", "--detach", commit], check=True)
        receipt = retry.git_receipt(root, commit, ["file.py"])
        self.assertEqual(receipt["source_commit"], commit)
        nested = root / "nested-archive"
        nested.mkdir()
        (nested / "file.py").write_text("pass\n")
        with self.assertRaisesRegex(ValueError, "own exact detached"):
            retry.git_receipt(nested, commit, ["file.py"])

    def test_final_raw_validator_uses_actual_retry_proof_instead_of_legacy_runtime_inference(self):
        """The completed retry is accepted only through its result-bound operational sidecar."""
        path, lease, sidecar = self.completed_attempt()
        lease["state"] = "completed"
        self.write(path.parent / "lease.json", lease)
        hardware = json.loads((path.parent / "hardware.json").read_bytes())
        hardware.update(source_root=str(self.source), source_commit=retry.SOURCE_COMMIT, hardware_class="jaguar03_rtx_a4500")
        self.write(path.parent / "hardware.json", hardware)
        self.write(path.parent / "case.json", self.case)
        record = json.loads(path.read_bytes())
        measured = json.loads(self.diagnosis_path.read_bytes())["paired_measurement"]
        record.update({name: measured[name] for name in ("correctness", "correctness_passed", "correctness_attempted", "contaminated", "rounds")})
        reference = {"learner": self.case["configuration"], "environment": {}, "execution": {}}
        record.update(reference=reference, candidate={**reference, "learner": {**reference["learner"], "hoist_rollout_bonus": True}},
            implementation_distinct=True)
        self.write(path, record)
        sidecar["original_result"] = self.receipt(path)
        self.write(path.parent / "infrastructure_retry_execution.json", sidecar)
        with patch.object(validation, "case_source_defaults", return_value=({}, {}, "unused")), \
                patch.object(validation, "configuration", side_effect=lambda value: value), \
                patch.object(validation, "retry_validation_module", return_value=retry), \
                patch.object(validation, "cohort_runtime_evidence", side_effect=AssertionError("legacy inference must not be used")):
            proof = validation.validate_cohort_case(path, self.case["hypothesis"], "jaguar03_rtx_a4500", 2,
                retry.SOURCE_COMMIT, validation.Evidence())
            self.assertEqual(proof["scope"], "actual_infrastructure_retry_runtime")
            (path.parent / "infrastructure_retry_execution.json").unlink()
            with self.assertRaises(FileNotFoundError):
                validation.validate_cohort_case(path, self.case["hypothesis"], "jaguar03_rtx_a4500", 2,
                    retry.SOURCE_COMMIT, validation.Evidence())

    def test_successful_recovery_publishes_both_terminal_leases_before_index_and_final_validation(self):
        """A valid dead runner remains complete through recovery, registry indexing and final raw validation."""
        path, lease, sidecar = self.completed_attempt()
        lease["hostname"] = worker.socket.gethostname()
        self.write(path.parent / "lease.json", lease)
        self.write(self.run / "queue/leases" / (self.unit + ".json"), lease)
        (self.run / "queue/failed" / (self.unit + ".json")).rename(self.run / "queue/running" / (self.unit + ".json"))
        original_prior = (self.previous / "lease.json").read_bytes()
        self.write(path.parent / "case.json", self.case)
        hardware = json.loads((path.parent / "hardware.json").read_bytes())
        hardware.update(source_root=str(self.source), source_commit=retry.SOURCE_COMMIT, hardware_class="jaguar03_rtx_a4500")
        self.write(path.parent / "hardware.json", hardware)
        record = json.loads(path.read_bytes())
        measured = json.loads(self.diagnosis_path.read_bytes())["paired_measurement"]
        record.update({name: measured[name] for name in ("correctness", "correctness_passed", "correctness_attempted", "contaminated", "rounds")})
        reference = {"learner": self.case["configuration"], "environment": {}, "execution": {}}
        record.update(reference=reference, candidate={**reference, "learner": {**reference["learner"], "hoist_rollout_bonus": True}},
            implementation_distinct=True, decision="accepted", paired_rounds=7)
        self.write(path, record)
        sidecar["original_result"] = self.receipt(path)
        self.write(path.parent / "infrastructure_retry_execution.json", sidecar)
        original_result = path.read_bytes()
        original_rename = Path.rename
        def check_publish_order(source, destination):
            """At the visible completion transition, both durable lease copies must already be terminal."""
            if source == self.run / "queue/running" / (self.unit + ".json"):
                preserved = json.loads((path.parent / "lease.json").read_bytes())
                current = json.loads((self.run / "queue/leases" / (self.unit + ".json")).read_bytes())
                self.assertEqual(preserved, current)
                self.assertEqual(current["state"], "completed")
            return original_rename(source, destination)
        def actual_retry_proof(run, case, current, result_path):
            """Delegate to the production proof validator with the fixture's explicitly patched case identities."""
            return retry.validate_execution(result_path, case, current)
        registry = {"initial_ideas": [{"name": self.case["hypothesis"], "results": {}}], "adaptive_batches": []}
        with patch.object(controller, "process_matches", return_value=False), \
                patch.object(controller, "validate_attempt_completion", side_effect=actual_retry_proof), \
                patch.object(Path, "rename", check_publish_order):
            controller.recover_local_claim(self.run, self.unit, lease["token"])
            self.assertEqual(controller.index_completed(self.run, registry), 1)
        self.assertEqual(json.loads((path.parent / "lease.json").read_bytes())["state"], "completed")
        self.assertEqual((self.previous / "lease.json").read_bytes(), original_prior)
        self.assertEqual(path.read_bytes(), original_result)
        with patch.object(validation, "case_source_defaults", return_value=({}, {}, "unused")), \
                patch.object(validation, "configuration", side_effect=lambda value: value), \
                patch.object(validation, "retry_validation_module", return_value=retry):
            proof = validation.validate_cohort_case(path, self.case["hypothesis"], "jaguar03_rtx_a4500", 2,
                retry.SOURCE_COMMIT, validation.Evidence())
        self.assertEqual(proof["scope"], "actual_infrastructure_retry_runtime")

    def test_cpu_assignment_requires_eight_allocated_cpus_per_gpu(self):
        """An unbound or underallocated node parent cannot claim any approved retries."""
        roster = self.assignment["gpu_uuids"]
        for environment, available in (({"SLURM_CPUS_PER_TASK": "8", "SLURM_CPU_BIND_TYPE": "threads"}, range(16)),
                ({"SLURM_CPUS_PER_TASK": "16", "SLURM_CPU_BIND_TYPE": "threads"}, range(8)),
                ({"SLURM_CPUS_PER_TASK": "16", "SLURM_CPU_BIND_TYPE": "cores"}, range(16)),
                ({"SLURM_CPUS_PER_TASK": "16", "SLURM_CPU_BIND_TYPE": "threads"}, range(32))):
            with self.assertRaisesRegex(ValueError, "eight CPUs per GPU"):
                retry.cpu_allocation(environment, roster, available)

    def test_actual_resolved_slurm_mask_matches_all_forty_available_cpus(self):
        """The actual threads request is accepted through its exact resolved mask, not a label guess."""
        roster = [f"GPU-{index}" for index in range(5)]
        available = list(range(56, 76)) + list(range(168, 188))
        environment = self.resolved_cpu_environment()
        proof = retry.cpu_allocation(environment, roster, available)
        self.assertEqual(proof["parent_available_cpus"], available)
        self.assertEqual(proof["slurm_cpu_bind_list"], environment["SLURM_CPU_BIND_LIST"])
        self.assertEqual(proof["slurm_cpu_bind"], environment["SLURM_CPU_BIND"])
        self.assertEqual([proof["gpu_cpu_affinities"][gpu] for gpu in roster], [available[i*8:(i+1)*8] for i in range(5)])
        with self.assertRaisesRegex(ValueError, "mask differs"):
            retry.cpu_allocation(environment, roster, list(range(40)))
        changed = {**environment, "SLURM_CPU_BIND": "quiet,mask_cpu:0xffff"}
        with self.assertRaisesRegex(ValueError, "mask fields disagree"):
            retry.cpu_allocation(changed, roster, available)

    def test_overlapping_or_substituted_cpu_masks_are_rejected(self):
        """Another worker's CPUs cannot be substituted even if the replacement contains eight integers."""
        for replacement in (list(range(8)), list(range(1, 9))):
            changed = copy.deepcopy(self.assignment)
            changed["cpu_allocation"]["gpu_cpu_affinities"]["GPU-seven"] = replacement
            with self.assertRaisesRegex(ValueError, "overlaps or differs"):
                retry.validate_assignment(changed, "GPU-seven")

    def test_worker_cpu_binding_is_inherited_by_the_original_numerical_runner(self):
        """Only the worker sets affinity; the original numerical runner must already inherit the exact mask."""
        environment = {"SLURM_CPUS_PER_TASK": "16", "SLURM_CPU_BIND_TYPE": "threads",
            "SLURM_JOB_GPUS": "5,7", "SLURM_STEP_GPUS": "5,7", "OMP_NUM_THREADS": "16"}
        with patch.object(worker.os, "sched_setaffinity") as bind, patch.object(worker.os, "sched_getaffinity", return_value=set(range(8))):
            selected = worker.bind_worker_cpus(self.assignment, "GPU-five", environment)
            bind.assert_called_once_with(0, set(range(8)))
            lease = {"gpu_uuid": "GPU-five", "cpu_affinity": selected, "thread_environment": retry.THREAD_ENVIRONMENT}
            with patch.dict(runner.os.environ, environment, clear=True):
                runner.verify_cpu_inheritance(self.assignment, lease)
            self.assertEqual(bind.call_count, 1)
        self.assertEqual(environment["OMP_NUM_THREADS"], "8")
        self.assertEqual(environment["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(environment["MKL_NUM_THREADS"], "1")
        self.assertEqual(environment["SLURM_JOB_GPUS"], "5,7")
        self.assertEqual(environment["SLURM_STEP_GPUS"], "5,7")
        with patch.dict(runner.os.environ, environment, clear=True), \
                patch.object(runner.os, "sched_getaffinity", return_value=set(range(8, 16))):
            with self.assertRaisesRegex(ValueError, "did not inherit"):
                runner.verify_cpu_inheritance(self.assignment, lease)

    def test_recovery_keeps_raw_success_with_missing_retry_proof_failed(self):
        """A dead runner's raw success cannot pass recovery without the completed execution receipt."""
        path, lease, _ = self.completed_attempt()
        lease["hostname"] = worker.socket.gethostname()
        self.write(self.run / "queue/leases" / (self.unit + ".json"), lease)
        self.write(path.parent / "lease.json", lease)
        (self.run / "queue/failed" / (self.unit + ".json")).rename(self.run / "queue/running" / (self.unit + ".json"))
        (path.parent / "infrastructure_retry_execution.json").unlink()
        with patch.object(controller, "process_matches", return_value=False):
            controller.recover_local_claim(self.run, self.unit, lease["token"])
        self.assertTrue((self.run / "queue/failed" / (self.unit + ".json")).exists())
        failure = json.loads((path.parent / "failure.json").read_bytes())
        self.assertEqual(failure["failure_stage"], "infrastructure_retry_proof")
        self.assertFalse(failure["retryable"])
        self.assertTrue(json.loads(path.read_bytes())["case_completed"])

    def test_indexing_rejects_an_unproved_fourth_attempt_before_registry_changes(self):
        """Neither a completed queue filename nor a raw success substitutes for execution proof."""
        path, lease, _ = self.completed_attempt()
        self.write(self.run / "queue/leases" / (self.unit + ".json"), lease)
        self.write(self.run / "queue/completed" / (self.unit + ".json"), self.case)
        (path.parent / "infrastructure_retry_execution.json").unlink()
        with patch.object(controller, "record_result") as record:
            with self.assertRaises(FileNotFoundError):
                controller.index_completed(self.run, {"initial_ideas": [], "adaptive_batches": []})
            record.assert_not_called()
        self.assertEqual(list((self.run / "queue/indexed_results").glob("*.json")), [])

    def test_signal_cleanup_restores_handlers_and_stops_only_exact_owned_children(self):
        """A termination signal passes through owned-child cleanup before the worker exits."""
        previous = signal.getsignal(signal.SIGTERM)
        process = Mock(pid=999)
        process.poll.return_value = None
        text = "999 (python) " + " ".join(["S", str(os.getpid()), *["0"] * 17, "1234"])
        with patch.object(Path, "read_text", return_value=text):
            with self.assertRaises(InterruptedError):
                with worker.interruption_cleanup():
                    try:
                        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                    except InterruptedError:
                        worker.stop_owned([{"process": process, "process_start_time": "1234"}])
                        raise
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=15)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        with patch.object(Path, "read_text", return_value=text):
            with self.assertRaisesRegex(RuntimeError, "exact parent/birth"):
                worker.stop_owned([{"process": process, "process_start_time": "different-birth"}])

    def test_parent_receipt_failure_stops_its_child_before_closing_output(self):
        """A child started just before a metadata write failure is tracked and cleaned up."""
        for index in range(4):
            self.write(self.approval_path.with_name(f"extra-{index}.json"), self.approval)
        arguments = types.SimpleNamespace(run_dir=self.run, output_dir=self.run / "new-node-output", python=Path(sys.executable), gpu_count=2)
        process = Mock(pid=999)
        real_write = retry.write_once
        def publish(path, contents):
            """Fail exactly after Popen, when the child's durable process receipt would be published."""
            if str(path).endswith(".process.json"):
                raise OSError("synthetic receipt failure")
            return real_write(path, contents)
        with patch.object(retry, "validate_approval", return_value=(self.approval, self.case)), \
                patch.object(worker, "verify_worker_source"), \
                patch.object(worker.subprocess, "check_output", return_value="GPU-five\nGPU-seven\n"), \
                patch.object(worker.scaling, "allocation_proof", return_value=self.assignment["allocation"]), \
                patch.object(worker.subprocess, "Popen", return_value=process), \
                patch.object(controller, "process_start_time", return_value="1234"), \
                patch.dict(worker.os.environ, {"SLURM_CPUS_PER_TASK": "16", "SLURM_CPU_BIND_TYPE": "threads", "SLURM_GPUS_ON_NODE": "2"}), \
                patch.object(worker.os, "sched_getaffinity", return_value=set(range(16))), \
                patch.object(retry, "write_once", side_effect=publish), patch.object(worker, "stop_owned") as stop:
            with self.assertRaisesRegex(OSError, "synthetic receipt"):
                worker.run_parent(arguments)
        stop.assert_called_once_with([{"process": process, "process_start_time": "1234"}])
        self.assertTrue((arguments.output_dir / "assignment.json").exists())

    def test_parent_rejects_eight_visible_devices_for_a_five_gpu_request_before_outputs(self):
        """An extra NVML-visible card never becomes an implicit member of the allocation."""
        for index in range(4):
            self.write(self.approval_path.with_name(f"extra-{index}.json"), self.approval)
        arguments = types.SimpleNamespace(run_dir=self.run, output_dir=self.run / "wrong-roster", python=Path(sys.executable), gpu_count=5)
        with patch.object(retry, "validate_approval", return_value=(self.approval, self.case)), \
                patch.object(worker, "verify_worker_source"), \
                patch.object(worker.subprocess, "check_output", return_value="\n".join(f"GPU-{index}" for index in range(8))), \
                patch.dict(worker.os.environ, {"SLURM_GPUS_ON_NODE": "5"}), \
                patch.object(worker.scaling, "allocation_proof") as allocated:
            with self.assertRaisesRegex(ValueError, "exactly the requested"):
                worker.run_parent(arguments)
            allocated.assert_not_called()
        self.assertFalse(arguments.output_dir.exists())

    def test_five_gpu_parent_preserves_full_allocation_and_disjoint_cpu_assignments(self):
        """The five-card parent keeps every JOB/STEP physical minor and assigns forty actual CPUs once."""
        for index in range(4):
            self.write(self.approval_path.with_name(f"extra-{index}.json"), self.approval)
        roster = [f"GPU-{index}" for index in range(5)]
        proof = {"hostname": "jaguar03", "slurm_job_id": "6779999",
            "slurm_allocations": {"SLURM_JOB_GPUS": "3,4,5,6,7", "SLURM_STEP_GPUS": "3,4,5,6,7"},
            "cuda_visible_devices_original": "0,1,2,3,4",
            "gpu_csv": "\n".join(f"{i}, {gpu}, NVIDIA RTX A4500" for i, gpu in enumerate(roster)),
            "gpu_xml": "<nvidia_smi_log>" + "".join(f"<gpu><uuid>{gpu}</uuid><product_name>NVIDIA RTX A4500</product_name><minor_number>{i+3}</minor_number></gpu>"
                for i, gpu in enumerate(roster)) + "</nvidia_smi_log>",
            "cards": [{"visible_index": str(i), "gpu_uuid": gpu, "physical_minor": str(i+3), "name": "NVIDIA RTX A4500"} for i, gpu in enumerate(roster)]}
        arguments = types.SimpleNamespace(run_dir=self.run, output_dir=self.run / "five-gpu-parent", python=Path(sys.executable), gpu_count=5)
        process = Mock(pid=999)
        real_write = retry.write_once
        def publish(path, contents):
            """Stop after the first child receipt boundary while retaining the actual parent assignment."""
            if str(path).endswith(".process.json"):
                raise OSError("synthetic receipt failure")
            return real_write(path, contents)
        environment = {"SLURM_GPUS_ON_NODE": "5", "SLURM_CPUS_PER_TASK": "40", "SLURM_CPU_BIND_TYPE": "threads",
            "SLURM_JOB_GPUS": "3,4,5,6,7", "SLURM_STEP_GPUS": "3,4,5,6,7"}
        with patch.object(retry, "validate_approval", return_value=(self.approval, self.case)), \
                patch.object(worker, "verify_worker_source"), patch.object(worker.subprocess, "check_output", return_value="\n".join(roster)), \
                patch.object(worker.scaling, "allocation_proof", return_value=proof), \
                patch.object(worker.subprocess, "Popen", return_value=process) as start, \
                patch.object(controller, "process_start_time", return_value="1234"), \
                patch.dict(worker.os.environ, environment), patch.object(worker.os, "sched_getaffinity", return_value=set(range(40))), \
                patch.object(retry, "write_once", side_effect=publish), patch.object(worker, "stop_owned"):
            with self.assertRaisesRegex(OSError, "synthetic receipt"):
                worker.run_parent(arguments)
        assignment = json.loads((arguments.output_dir / "assignment.json").read_bytes())
        self.assertEqual(assignment["expected_gpu_count"], 5)
        self.assertEqual(assignment["allocation"]["slurm_allocations"], proof["slurm_allocations"])
        masks = assignment["cpu_allocation"]["gpu_cpu_affinities"]
        self.assertEqual([masks[gpu] for gpu in roster], [list(range(i*8, (i+1)*8)) for i in range(5)])
        child_environment = start.call_args.kwargs["env"]
        self.assertEqual(child_environment["SLURM_JOB_GPUS"], "3,4,5,6,7")
        self.assertEqual(child_environment["SLURM_STEP_GPUS"], "3,4,5,6,7")
        self.assertEqual(child_environment["CUDA_VISIBLE_DEVICES"], "GPU-0")


if __name__ == "__main__":
    unittest.main()
