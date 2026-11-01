"""Check the case-only fifth attempt with real metadata validation and no numerical imports."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import pregather_async_retry as retry
import run_pregather_async_retry as runner
import controller
import post_comparison_validation as validation


class PregatherRetryTests(unittest.TestCase):
    """Keep original attempts and fresh proof records separate in a temporary campaign."""

    def setUp(self):
        """Build a complete synthetic diagnostic, original failure, and proposed fifth approval."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name).resolve()
        controller.layout(self.run)
        self.unit = "original-pregather-case"
        self.source, self.helpers = self.run / "source", self.run / "helpers"
        self.case = {"unit_id": self.unit, "hypothesis_position": 23, "hypothesis": "gather_epoch_minibatches_in_advance",
            "source_root": str(self.source), "source_commit": retry.SOURCE_COMMIT, "campaign_run_dir": str(self.run),
            "hardware_class": "jaguar03_rtx_a4500", "configuration": {"n_copies": 2},
            "candidate": {"learner": {"pre_gather_minibatches": True}, "environment": {}, "execution": {}},
            "rounds": 7, "iterations": 1, "seed_start": 1, "maximum_attempts": 3}
        self.case["specification_digest"] = controller.content_digest(self.case)
        self.case_path = self.write(self.run / "queue/specifications" / (self.unit + ".json"), self.case)
        self.case_bytes = self.case_path.read_bytes()
        self.addCleanup(patch.stopall)
        patch.object(retry.infra, "APPROVED", {23: (self.case["hypothesis"], self.case["candidate"]["learner"], retry.infra.digest(self.case_bytes))}).start()
        patch.object(retry.infra, "validate_git_receipt").start()
        patch.object(retry, "validate_core").start()
        self.manifest = self.write(self.run / "asset_manifest.json", {"source_commit": "a" * 40, "files": {"sprite": "b" * 64}})
        self.asset_record = {"root": str(self.run / "assets"), "manifest": str(self.manifest), "sha256": self.record(self.manifest)["sha256"]}
        self.assets = {"root": self.asset_record["root"], "manifest_sha256": self.asset_record["sha256"],
            "source_commit": "a" * 40, "file_count": 1,
            "tree_digest": hashlib.sha256(json.dumps({"sprite": "b" * 64}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        self.prior = self.run / "data" / self.unit / "attempts/attempt-0004"
        self.old_path = self.run / retry.infra.APPROVALS_PATH / (self.unit + ".json")
        self.old = {"helpers": {"source_commit": "c" * 40}, "asset_record": self.asset_record,
            "numerical_source": {"source_root": str(self.source), "source_commit": retry.SOURCE_COMMIT, "files": {}},
            "jaxtari_source_root": str(self.run / "public"), "python": sys.executable}
        self.write(self.old_path, self.old)
        patch.object(retry.infra, "validate_approval", return_value=(self.old, self.case)).start()
        self.prior_lease = {"attempt": 4, "state": "failed", "token": "fourth-token", "runner_pid": 34, "runner_start_time": "43",
            "infrastructure_retry_approval": self.record(self.old_path)}
        self.write(self.prior / "lease.json", self.prior_lease)
        self.write(self.prior / "case.json", self.case)
        self.write(self.prior / "failure.json", {"attempt": 4, "case_completed": False, "source_commit": retry.SOURCE_COMMIT,
            "pid": 34, "error": "RESOURCE_EXHAUSTED: Out of memory in jit_update"})
        self.write(self.prior / "hardware.json", {"pid": 34, "process_start_time": "43"})
        self.old_claim = self.write(self.run / retry.infra.CLAIMS_PATH / (self.unit + ".json"),
            {"approval": self.record(self.old_path), "lease": self.prior_lease})
        self.write(self.run / "queue/leases" / (self.unit + ".json"), self.prior_lease)
        self.write(self.run / "queue/failed" / (self.unit + ".json"), self.case)
        self.diagnostic = self.run / "diagnostic"
        self.configuration = {"learner": self.case["configuration"], "environment": {}, "execution": {}}
        self.gpu = "GPU-selected"
        self.parser_source = self.run / "xla_client.py"
        self.parser_source.write_text("# parser bytes are checked without importing them\n")
        self.parser = {"source": {"path": str(self.parser_source), "sha256": self.record(self.parser_source)["sha256"],
            "function": "generate_pjrt_gpu_plugin_options", "allowed_allocators": ["default", "cuda_async"]},
            "parsed_options": {"allocator": "cuda_async", "memory_fraction": .85, "preallocate": False},
            "environment": retry.ALLOCATOR, "memory_preparation": None}
        self.runtime = {**retry.infra.RUNTIME, **retry.ALLOCATOR, "JAX_ENABLE_COMPILATION_CACHE": "false",
            "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "", "JAXTARI_SKIP_SPRITE_UPDATE": "1"}
        self.hardware = {"hostname": "jaguar03", "gpu_uuid": self.gpu,
            "gpu_query": self.gpu + ", NVIDIA RTX A4500, 20470 MiB, 580.159.04", "pid": 50, "process_start_time": "55",
            "slurm_job_id": "6778016", "cuda_visible_devices": self.gpu, "runtime_environment": self.runtime,
            "cpu_affinity": list(range(8)), "thread_environment": retry.infra.THREAD_ENVIRONMENT}
        self.write(self.diagnostic / "hardware.json", self.hardware)
        self.write(self.diagnostic / "actual_allocator.json", self.parser)
        library = self.run / "libcudart.so.12"
        library.write_bytes(b"real synthetic shared-image bytes")
        images = {str(library): {"sha256": self.record(library)["sha256"], "version": "12.9"}}
        self.installed = {"jaxtari_source": {"source_commit": "a" * 40}, "gpu_libraries": images}
        self.loaded = {"pid": 50, "platform": "gpu", "platform_version": "CUDA 12.9", "cudnnGetVersion": 92400,
            "cudaRuntimeGetVersion": 12090, "cudnn_library": str(library), "cuda_runtime_library": str(library),
            "loaded_shared_libraries": [str(library)], "verified_package_libraries": images}
        self.write(self.diagnostic / "installed_backend.json", self.installed)
        self.write(self.diagnostic / "asset_receipt.json", self.assets)
        self.plan = {"case": self.case, "configuration": self.configuration, "source_root": str(self.source),
            "source_commit": retry.SOURCE_COMMIT, "rounds": 7, "iterations": 1, "copy_seeds": [1, 2], "latency_calls": 3,
            "allocator": retry.ALLOCATOR, "memory_preparation": None, "allocator_parser": self.parser["source"],
            "gpu_uuid": self.gpu, "core": {}, "paired_source": {}, "helpers": {}, "asset_record": self.asset_record}
        runtime_input = self.write(self.run / "runtime.json", retry.infra.RUNTIME)
        asset_input = self.write(self.run / "asset_input.json", self.asset_record)
        reference_input = self.write(self.run / "reference.json", {"reference": self.configuration})
        self.plan["inputs"] = {"case": self.record(self.prior / "case.json"), "original_failure": self.record(self.prior / "failure.json"),
            "runtime_environment": self.record(runtime_input), "asset_record": self.record(asset_input), "reference": self.record(reference_input)}
        self.raw = self.raw_record()
        self.write(self.diagnostic / "original_paired_result.json", self.raw)
        phases = ("check_fixed_inputs_entered", "check_fixed_inputs_completed", "correctness_checks_entered", "correctness_checks_completed", "original_benchmark_completed")
        (self.diagnostic / "memory_snapshots.jsonl").write_text("".join(json.dumps({"phase": phase, "pid": 50,
            "device_memory": {"gpu_uuid": self.gpu}}) + "\n" for phase in phases))
        self.diagnostic_result = {"completed": True, "kind": "pregather_async_allocator_diagnosis",
            "eligible_for_formal_campaign_admission": False, "hypothesis_count_contribution": 0, "source_commit": retry.SOURCE_COMMIT,
            "hardware": self.hardware, "helpers": {}, "allocator": retry.ALLOCATOR,
            "original_result": self.record(self.diagnostic / "original_paired_result.json"), "loaded_gpu_libraries": self.loaded}
        self.refresh_diagnostic()
        self.approval_path = self.run / retry.APPROVALS_PATH / (self.unit + ".json")
        previous = {key: self.record(self.prior / filename) for key, filename in
            (("case", "case.json"), ("lease", "lease.json"), ("failure", "failure.json"), ("hardware", "hardware.json"))}
        previous.update(approval=self.record(self.old_path), claim=self.record(self.old_claim))
        self.approval = {"schema_version": 1, "kind": "pregather_cuda_async_formal_retry", "approved_attempt": 5,
            "previous_attempt": 4, "unit_id": self.unit, "campaign_run_dir": str(self.run), "case": self.record(self.case_path),
            "previous": previous, "diagnostic": self.record(self.diagnostic / "result.json"),
            "helpers": {"source_root": str(self.helpers), "source_commit": "d" * 40,
                "files": {retry.infra.PREFIX + name: "e" * 64 for name in retry.HELPERS}},
            "numerical_source": self.old["numerical_source"], "runtime_environment": retry.infra.RUNTIME,
            "allocator": retry.ALLOCATOR, "thread_environment": retry.infra.THREAD_ENVIRONMENT, "memory_preparation": None,
            **{key: self.old[key] for key in ("asset_record", "jaxtari_source_root", "python")}}
        self.write(self.approval_path, self.approval)

    def write(self, path, value):
        """Store one synthetic immutable-document version in the test's own directory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(retry.infra.json_bytes(value))
        return path

    def record(self, path):
        """Hash the exact fixture bytes through the production reader."""
        return retry.infra.file_record(path, retry.LocalEvidence())

    def raw_record(self):
        """Supply actual report shapes and matched complete training work, rather than Boolean-only proof."""
        row = {"iterations": 1, "copies": 2, "environments_per_copy": 128, "rollout_steps": 128,
            "training_transitions": 32768, "seconds": 2., "memory": {"gpu_uuid": self.gpu, "other_gpu_processes": [],
                "sampled_process_peak_bytes": 100, "allocator": {"peak_bytes_in_use": 100}}}
        return {"measurement_valid": True, "contaminated": False, "source_commit": retry.SOURCE_COMMIT,
            "hardware_class": "jaguar03_rtx_a4500", "copies": 2, "copy_seeds": [1, 2], "reference": self.configuration,
            "candidate": {**self.configuration, "learner": {**self.configuration["learner"], "pre_gather_minibatches": True}},
            "correctness_passed": True, "correctness_attempted": True, "correctness": {"passed": True,
                "complete_iteration": {"leaves": 10}, "supplied_inputs": {name: {"leaves": 10}
                for name in ("preparation", "fixed_batch", "fixed_batch_update")}},
            "rounds": [{"order": ["reference", "candidate"] if index % 2 == 0 else ["candidate", "reference"],
                "reference": copy.deepcopy(row), "candidate": copy.deepcopy(row)} for index in range(7)],
            "synchronized_latency": {arm: {"calls": 3} for arm in ("reference", "candidate")}}

    def refresh_diagnostic(self):
        """Rebind reviewed fixture hashes while leaving semantic production checks enabled."""
        plan_path = self.write(self.diagnostic / "measurement_plan.json", self.plan)
        self.write(self.diagnostic / "measurement_prepared.json", {"plan_sha256": self.record(plan_path)["sha256"]})
        self.diagnostic_result["original_result"] = self.record(self.diagnostic / "original_paired_result.json")
        self.write(self.diagnostic / "result.json", self.diagnostic_result)
        for name, filename in (("DIAGNOSTIC_SHA256", "result.json"), ("DIAGNOSTIC_RAW_SHA256", "original_paired_result.json"),
                ("DIAGNOSTIC_PLAN_SHA256", "measurement_plan.json"), ("DIAGNOSTIC_PARSER_SHA256", "actual_allocator.json")):
            patch.object(retry, name, self.record(self.diagnostic / filename)["sha256"]).start()

    def completed(self):
        """Construct a fresh exact process/allocator/owned-allocation sidecar around an original result."""
        directory = self.run / "data" / self.unit / "attempts/attempt-0005"
        gpu = self.gpu
        allocations = {"SLURM_JOB_GPUS": "5", "SLURM_STEP_GPUS": "5"}
        assignment = {"hostname": "jaguar03", "parent_pid": 12, "parent_process_start_time": "21", "gpu_uuids": [gpu], "expected_gpu_count": 1,
            "allocation": {"hostname": "jaguar03", "slurm_job_id": "6779999", "slurm_allocations": allocations,
                "cuda_visible_devices_original": "0", "gpu_csv": f"0, {gpu}, NVIDIA RTX A4500\n",
                "gpu_xml": f"<nvidia_smi_log><gpu><uuid>{gpu}</uuid><product_name>NVIDIA RTX A4500</product_name><minor_number>5</minor_number></gpu></nvidia_smi_log>",
                "cards": [{"visible_index": "0", "gpu_uuid": gpu, "physical_minor": "5", "name": "NVIDIA RTX A4500"}]},
            "cpu_allocation": retry.infra.cpu_allocation({"SLURM_CPUS_PER_TASK": "8", "SLURM_CPU_BIND_TYPE": "threads"}, [gpu], range(8)),
            "approvals": [self.record(self.approval_path)]}
        assignment_path = self.write(self.run / "operational/assignment.json", assignment)
        lease = {"attempt": 5, "state": "completed", "token": "fifth-token", "runner_pid": 70, "runner_start_time": "77", "gpu_uuid": gpu,
            "worker_pid": 12, "worker_start_time": "21", "pregather_async_retry_approval": self.record(self.approval_path),
            "assignment": self.record(assignment_path), "cpu_affinity": list(range(8)), "thread_environment": retry.infra.THREAD_ENVIRONMENT}
        self.write(directory / "lease.json", lease)
        self.write(self.run / retry.CLAIMS_PATH / (self.unit + ".json"), {"approval": self.record(self.approval_path), "lease": lease})
        hardware = {**self.hardware, "pid": 70, "process_start_time": "77", "slurm_job_id": "6779999"}
        self.write(directory / "hardware.json", hardware)
        ledger = self.run / "slurm/submitted_jobids_uva.txt"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text("6779999\n")
        owned = {"pid": 70, "process_start_time": "77", "hostname": "jaguar03", "gpu_uuid": gpu, "job_id": "6779999",
            "cgroup": "0::/slurm/uid_1/job_6779999/step_0\n", "ledger_path": str(ledger), "ledger_snapshot": ledger.read_text()}
        owned_path = self.write(directory / "owned_allocation.json", owned)
        initial = {**self.runtime, **retry.infra.THREAD_ENVIRONMENT, "CUDA_VISIBLE_DEVICES": gpu, "SLURM_JOB_ID": "6779999", **allocations}
        initial_path = self.write(directory / "process_environment.json", {"source": "/proc/self/environ allowlist", "values": initial,
            "pid": 70, "process_start_time": "77", "gpu_uuid": gpu})
        parser = {**self.parser, "pid": 70, "process_start_time": "77", "gpu_uuid": gpu}
        parser_path = self.write(directory / "actual_allocator.json", parser)
        self.write(directory / "installed_backend.json", self.installed)
        self.write(directory / "asset_receipt.json", self.assets)
        result_path = self.write(directory / "result.json", {**self.raw, "case_completed": True, "attempt": 5, "unit_id": self.unit,
            "idea": self.case["hypothesis"], "specification_digest": self.case["specification_digest"], "decision": "rejected",
            "implementation_distinct": True})
        sidecar = {"schema_version": 1, "unit_id": self.unit, "completed": True, "attempt": 5, "approval": self.record(self.approval_path), "original_result": self.record(result_path),
            "helpers": self.approval["helpers"], "numerical_source": self.approval["numerical_source"], "memory_preparation": None,
            "lease_token": lease["token"], "pid": 70, "process_start_time": "77", "gpu_uuid": gpu, "assignment": self.record(assignment_path),
            "cpu_affinity": list(range(8)), "thread_environment": retry.infra.THREAD_ENVIRONMENT, "owned_allocation": self.record(owned_path),
            "process_environment": initial, "process_environment_record": self.record(initial_path), "runtime_environment": self.runtime,
            "actual_allocator": self.record(parser_path), "loaded_gpu_libraries": {**self.loaded, "pid": 70},
            "runner": str(self.helpers / retry.infra.PREFIX / "run_pregather_async_retry.py"), "started_at_seconds": 100., "finished_at_seconds": 120.,
            "frozen_modules": {name: str(self.source / retry.infra.PREFIX / (name + ".py")) for name in ("run_case", "controller", "ideas", "paired_benchmark")}}
        self.write(directory / "pregather_async_retry_execution.json", sidecar)
        return result_path, lease, sidecar

    def test_approval_preserves_original_limit_and_every_prior_byte(self):
        """The sole new allowance does not rewrite the original case or old approvals/claim."""
        before = {path: path.read_bytes() for path in (self.case_path, self.old_path, self.old_claim, self.prior / "failure.json")}
        approval, case = retry.validate_approval(self.approval_path)
        self.assertEqual((approval["approved_attempt"], case["maximum_attempts"]), (5, 3))
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.approval["approved_attempt"] = 6
        self.write(self.approval_path, self.approval)
        with self.assertRaisesRegex(ValueError, "one exact case"):
            retry.validate_approval(self.approval_path)

    def test_diagnostic_requires_all_original_checks_and_seven_real_pairs(self):
        """A completed flag and matching hashes cannot replace fixed-input reports or measured work."""
        retry.diagnostic_proof(self.approval["diagnostic"], self.case, retry.LocalEvidence())
        self.raw["rounds"] = [{}] * 7
        self.write(self.diagnostic / "original_paired_result.json", self.raw)
        self.refresh_diagnostic()
        with self.assertRaisesRegex(ValueError, "alternating pairs"):
            retry.diagnostic_proof(self.record(self.diagnostic / "result.json"), self.case, retry.LocalEvidence())

    def test_diagnostic_rejects_incomplete_result_wrong_parser_and_process(self):
        """Each diagnostic prerequisite is rebuilt from the original completed unit."""
        self.diagnostic_result["completed"] = False
        self.refresh_diagnostic()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            retry.diagnostic_proof(self.record(self.diagnostic / "result.json"), self.case, retry.LocalEvidence())
        self.diagnostic_result["completed"] = True
        self.parser["parsed_options"]["preallocate"] = True
        self.write(self.diagnostic / "actual_allocator.json", self.parser)
        self.refresh_diagnostic()
        with self.assertRaisesRegex(ValueError, "async parser"):
            retry.diagnostic_proof(self.record(self.diagnostic / "result.json"), self.case, retry.LocalEvidence())

    def test_original_failure_changes_block_fifth_approval(self):
        """A changed preserved failure cannot be covered by the old fourth approval."""
        self.write(self.prior / "failure.json", {"attempt": 4, "error": "another failure"})
        with self.assertRaisesRegex(ValueError, "changed"):
            retry.validate_approval(self.approval_path)

    def test_rehashing_changed_prior_failure_cannot_replace_diagnostic_input(self):
        """Even unchanged JSON values with different raw bytes remain a different historical failure."""
        path = self.prior / "failure.json"
        path.write_bytes(path.read_bytes() + b" ")
        self.approval["previous"]["failure"] = self.record(path)
        self.write(self.approval_path, self.approval)
        with self.assertRaisesRegex(ValueError, "changed"):
            retry.validate_approval(self.approval_path)

    def test_full_execution_requires_fresh_original_result_and_actual_allocator(self):
        """The final proof binds a fresh complete result, process, job, allocation, parser and libraries."""
        path, lease, sidecar = self.completed()
        proof = retry.validate_execution(path, self.case)
        self.assertEqual(proof["scope"], "actual_pregather_async_fifth_attempt_runtime")
        sidecar["memory_preparation"] = {"bytes": 9 * 2 ** 30}
        self.write(path.with_name("pregather_async_retry_execution.json"), sidecar)
        with self.assertRaisesRegex(ValueError, "fresh exact result"):
            retry.validate_execution(path, self.case)

    def test_missing_or_changed_same_process_proofs_are_rejected(self):
        """Changing the actual allocator or process receipt cannot preserve a valid original-result sidecar."""
        path, lease, sidecar = self.completed()
        sidecar["pid"] = 999
        self.write(path.with_name("pregather_async_retry_execution.json"), sidecar)
        with self.assertRaisesRegex(ValueError, "another process"):
            retry.validate_execution(path, self.case)
        sidecar["pid"] = 70
        self.write(path.with_name("pregather_async_retry_execution.json"), sidecar)
        parser = json.loads(path.with_name("actual_allocator.json").read_bytes())
        parser["parsed_options"]["allocator"] = "default"
        self.write(path.with_name("actual_allocator.json"), parser)
        sidecar["actual_allocator"] = self.record(path.with_name("actual_allocator.json"))
        self.write(path.with_name("pregather_async_retry_execution.json"), sidecar)
        with self.assertRaisesRegex(ValueError, "async parser"):
            retry.validate_execution(path, self.case)

    def test_append_only_ledger_and_terminal_lease_are_required(self):
        """Later job IDs may append, while replaced ledgers and unfinished attempts cannot pass final proof."""
        path, lease, sidecar = self.completed()
        ledger = self.run / "slurm/submitted_jobids_uva.txt"
        ledger.write_text("6779999\n6780000\n")
        retry.validate_execution(path, self.case)
        ledger.write_text("6780000\n")
        with self.assertRaisesRegex(ValueError, "append-only"):
            retry.validate_execution(path, self.case)
        ledger.write_text("6779999\n")
        lease["state"] = "running"
        self.write(path.with_name("lease.json"), lease)
        retry.validate_execution(path, self.case, lease)
        with self.assertRaisesRegex(ValueError, "queue transition"):
            retry.validate_execution(path, self.case)

    def test_one_shot_claim_preserves_global_fourth_claim_and_never_grants_six(self):
        """The separate fifth claim is unique without consuming or replacing the old global fourth claim."""
        old_bytes = self.old_claim.read_bytes()
        assignment = self.write(self.run / "assignment.json", {})
        identity = {"worker_pid": 12, "worker_start_time": "21", "gpu_uuid": self.gpu}
        ready = {"source_root": str(self.source), "source_commit": retry.SOURCE_COMMIT}
        with patch.object(controller, "ready_specification", return_value=ready):
            first = retry.claim(self.run, self.approval_path, identity, assignment)
            second = retry.claim(self.run, self.approval_path, identity, assignment)
        self.assertEqual(first[2]["attempt"], 5)
        self.assertIsNone(second)
        self.assertEqual(self.old_claim.read_bytes(), old_bytes)
        self.assertEqual(self.case_path.read_bytes(), self.case_bytes)
        self.assertFalse((self.prior.parent / "attempt-0006").exists())

    def test_process_environment_records_only_the_explicit_allowlist(self):
        """Credentials and unrelated inherited settings are omitted even when present in proc bytes."""
        raw = b"SECRET_ACCESS_TOKEN=do-not-record\0XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async\0SLURM_JOB_ID=6779999\0"
        with patch.object(Path, "read_bytes", return_value=raw):
            recorded = runner.initial_environment()
        self.assertEqual(recorded, {"XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async", "SLURM_JOB_ID": "6779999"})
        self.assertNotIn("SECRET", json.dumps(recorded))

    def test_runtime_projection_drops_unrelated_prefix_secrets_and_rejects_aliases(self):
        """Runtime metadata uses named controls, and competing allocator names remain explicit failures."""
        projected = retry.runtime_environment({**self.runtime, "CUDA_SECRET_TOKEN": "do-not-record", "NVIDIA_TOKEN": "do-not-record"})
        self.assertEqual(projected, self.runtime)
        with self.assertRaisesRegex(ValueError, "runtime"):
            retry.validate_runtime({**self.runtime, "TF_GPU_ALLOCATOR": "bfc"})

    def test_complete_parser_options_must_equal_the_completed_diagnostic(self):
        """Unapproved additional parsed allocator options cannot hide behind the three main options."""
        path, lease, sidecar = self.completed()
        parser = json.loads(path.with_name("actual_allocator.json").read_bytes())
        parser["parsed_options"]["use_tf_runtime"] = True
        self.write(path.with_name("actual_allocator.json"), parser)
        sidecar["actual_allocator"] = self.record(path.with_name("actual_allocator.json"))
        self.write(path.with_name("pregather_async_retry_execution.json"), sidecar)
        with self.assertRaisesRegex(ValueError, "another installed allocator parser"):
            retry.validate_execution(path, self.case)

    def test_recovered_fifth_result_remains_valid_through_index_and_final_cohort(self):
        """Recovery publishes both terminal leases before indexing and later raw-proof rebuilding."""
        path, lease, sidecar = self.completed()
        lease.update(state="running", hostname="jaguar03")
        self.write(path.with_name("lease.json"), lease)
        self.write(self.run / "queue/leases" / (self.unit + ".json"), lease)
        self.write(self.run / "queue/running" / (self.unit + ".json"), self.case)
        (self.run / "queue/failed" / (self.unit + ".json")).unlink()
        self.write(path.with_name("case.json"), self.case)
        hardware = json.loads(path.with_name("hardware.json").read_bytes())
        hardware.update(source_root=str(self.source), source_commit=retry.SOURCE_COMMIT, hardware_class="jaguar03_rtx_a4500")
        self.write(path.with_name("hardware.json"), hardware)
        registry = {"initial_ideas": [{"name": self.case["hypothesis"], "results": {}}], "adaptive_batches": []}
        def validate_current(run, case, live, result_path):
            return retry.validate_execution(result_path, case, live)
        with patch.object(controller.socket, "gethostname", return_value="jaguar03"), \
                patch.object(controller, "process_matches", return_value=False), \
                patch.object(controller, "validate_attempt_completion", side_effect=validate_current) as proof:
            controller.recover_local_claim(self.run, self.unit, lease["token"])
            self.assertEqual(controller.index_completed(self.run, registry), 1)
            self.assertEqual(proof.call_count, 2)
        self.assertEqual(json.loads(path.with_name("lease.json").read_bytes())["state"], "completed")
        self.assertTrue((self.run / "queue/completed" / (self.unit + ".json")).exists())
        with patch.object(validation, "case_source_defaults", return_value=({}, {}, b"")), \
                patch.object(validation, "configuration", side_effect=lambda value: value), \
                patch.object(validation, "pregather_retry_validation_module", return_value=retry):
            final = validation.validate_cohort_case(path, self.case["hypothesis"], "jaguar03_rtx_a4500", 2,
                retry.SOURCE_COMMIT, validation.Evidence())
        self.assertEqual(final["allocator"], "cuda_async")

    def test_child_calls_original_full_case_after_parser_and_placement_without_preparation(self):
        """Mock numerical libraries while exercising the child control path and the original call arguments."""
        path, lease, sidecar = self.completed()
        for filename in ("result.json", "pregather_async_retry_execution.json"):
            path.with_name(filename).unlink()
        approval = copy.deepcopy(self.approval)
        approval["helpers"]["source_root"] = str(SCRIPTS.parents[2])
        calls = []
        def full_case(run, case, attempt, token):
            calls.append("original_case")
            self.assertEqual((run, case, attempt, token), (self.run, self.case, 5, lease["token"]))
            return self.write_result(path)
        original_runner = types.SimpleNamespace(attempt_paths=lambda *args: (path.parent.parent.parent, path.parent),
            wait_for_recorded_release=lambda *args: {**lease, "pregather_async_retry_approval": self.record(self.approval_path)},
            process_start_time=lambda pid: "99", atomic_json=lambda target, value: self.write(target, value), run_case=full_case)
        training = types.SimpleNamespace(enforce_runtime=Mock(), verify_assets=lambda *args: self.assets,
            jaxtari_python_receipt=lambda *args: self.installed["jaxtari_source"],
            installed_gpu_library_receipts=lambda: self.installed["gpu_libraries"], actual_gpu_library_receipt=lambda *args: self.loaded)
        original = types.SimpleNamespace(retry=types.SimpleNamespace(__file__=str(SCRIPTS / "infrastructure_retry.py")),
            load_original_runner=lambda case: (calls.append("original_imports") or original_runner),
            verify_placement=lambda *args: calls.append("placement"), load_training_helper=lambda *args: training,
            verify_cpu_inheritance=Mock(), verify_original_modules=lambda *args: sidecar["frozen_modules"])
        parsed = types.SimpleNamespace(__file__=str(self.parser_source), generate_pjrt_gpu_plugin_options=lambda:
            (calls.append("actual_parser") or self.parser["parsed_options"]))
        fake_jax = types.SimpleNamespace(config=types.SimpleNamespace(update=Mock()), devices=lambda:
            [types.SimpleNamespace(platform="gpu", device_kind="NVIDIA RTX A4500", client=object())])
        initial = {**self.runtime, **retry.infra.THREAD_ENVIRONMENT, "CUDA_VISIBLE_DEVICES": self.gpu}
        with patch.object(retry, "validate_approval", return_value=(approval, self.case)), \
                patch.object(retry, "adjacent", return_value=original), patch.object(retry, "owned_allocation", return_value={}), \
                patch.object(retry, "diagnostic_proof", return_value={"parser": self.parser, "installed": self.installed}), \
                patch.object(retry, "validate_execution"), patch.object(runner, "initial_environment", return_value=initial), \
                patch.dict(runner.os.environ, initial, clear=True), patch.dict(sys.modules, {"jax": fake_jax,
                    "jaxtari": types.SimpleNamespace(__file__=str(self.run / "installed/jaxtari/__init__.py"))}), \
                patch.object(runner.metadata, "distribution", return_value=types.SimpleNamespace(locate_file=lambda name: self.run / "installed" / name)), \
                patch.object(runner.importlib, "import_module", return_value=parsed):
            result = runner.execute(self.approval_path, self.run / "operational/assignment.json", lease["token"])
        self.assertEqual(result["attempt"], 5)
        self.assertEqual(calls, ["original_imports", "placement", "actual_parser", "original_case"])
        self.assertIsNone(json.loads(path.with_name("pregather_async_retry_execution.json").read_bytes())["memory_preparation"])

    def test_unapproved_jax_controls_fail_before_projection_without_recording_values(self):
        """A second execution change cannot disappear from the small safe runtime receipt."""
        with self.assertRaisesRegex(ValueError, "JAX_DISABLE_JIT") as error:
            retry.runtime_environment({**self.runtime, "JAX_DISABLE_JIT": "secret-value-not-to-report"})
        self.assertNotIn("secret-value", str(error.exception))
        projected = retry.runtime_environment({**self.runtime, "JAX_COMPILATION_CACHE_DIR": ""})
        self.assertEqual(projected, self.runtime)

    def test_wrong_global_infrastructure_helper_is_rejected_before_frozen_imports(self):
        """The unchanged runtime helper's ordinary import must still point to its approved adjacent file."""
        approval = copy.deepcopy(self.approval)
        approval["helpers"]["source_root"] = str(SCRIPTS.parents[2])
        wrong = types.SimpleNamespace(__file__="/unapproved/infrastructure_retry.py")
        with patch.object(retry, "validate_approval", return_value=(approval, self.case)), \
                patch.dict(sys.modules, {"infrastructure_retry": wrong}), \
                patch.dict(runner.os.environ, self.runtime, clear=True):
            with self.assertRaisesRegex(ValueError, "another infrastructure retry helper"):
                runner.execute(self.approval_path, self.run / "unused-assignment.json", "unreleased-token")

    def test_worker_background_records_standard_sections_and_its_exact_command(self):
        """Operational metadata includes the required sections and a separate reproducible worker CLI."""
        import shlex
        directory = self.run / "operational-background"
        directory.mkdir()
        arguments = types.SimpleNamespace(run_dir=self.run, approval=self.approval_path, gpu_uuid=self.gpu)
        retry.worker_background(directory, arguments, self.approval, self.case)
        background = (directory / "experiment_background.md").read_text()
        for title in ("Purpose", "Key hyperparameters", "Code and config changes", "Git state"):
            self.assertIn("## " + title + "\n", background)
        command = shlex.split((directory / "command.txt").read_text())
        self.assertEqual(command, [sys.executable, str(SCRIPTS / "pregather_async_retry.py"), "worker", "--run-dir", str(self.run),
            "--approval", str(self.approval_path), "--output-dir", str(directory), "--gpu-uuid", self.gpu])

    def write_result(self, path):
        """Publish the original-runner result in the mocked numerical control-path test."""
        result = {"attempt": 5, "unit_id": self.unit, "decision": "rejected"}
        self.write(path, result)
        return result


if __name__ == "__main__":
    unittest.main()
