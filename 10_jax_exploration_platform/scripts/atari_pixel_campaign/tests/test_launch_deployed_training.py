"""Check explicit ACES queue/ownership/stop metadata without arrays, SSH or real submission."""

from contextlib import contextmanager
import copy
import fcntl
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launch_deployed_training as launch


class DeployedLauncherTests(unittest.TestCase):
    """Keep immutable reserved identities and exact ACES job evidence around the shared runner."""

    def setUp(self):
        """Create a real detached synthetic helper Git tree plus already-computed admission records."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.queue = self.run / "final_training_queue"
        self.write("slurm/submitted_jobids_aces.txt", "500\n")
        for name in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt"):
            self.write(name, "synthetic recorded campaign\n")
        self.root = self.run / "detached_helpers"
        scripts = self.root / launch.metadata.PREFIX
        scripts.mkdir(parents=True)
        names = set(launch.train_deployed.HELPERS) | {"validate_deployed_training_completion.py"}
        for name in names:
            (scripts / name).write_text("# synthetic immutable helper\n")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Metadata test", "-c", "user.email=test@example.invalid",
                        "commit", "-qm", "Create synthetic helper bytes"], check=True)
        self.commit = subprocess.check_output(["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True).strip()
        subprocess.run(["git", "-C", str(self.root), "checkout", "-q", "--detach", self.commit], check=True)
        self.helpers = {"source_root": str(self.root), "source_commit": self.commit,
            "files": {launch.metadata.PREFIX + name: launch.metadata.file_digest(scripts / name) for name in launch.train_deployed.HELPERS}}
        self.configuration = {"learner": {"n_copies": 1, "n_envs": 128, "num_steps": 128, "update_epochs": 4,
                                          "num_minibatches": 4, "total_timesteps": 2_000_000_000},
                              "environment": {"num_envs": 128}, "execution": {"iterations_per_call": 1}}
        self.runtime = {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                        "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}
        packing = self.write("evidence/packing_result.json", {"completed": True, "source_commit": "a" * 40,
            "configuration": self.configuration, "hardware": {"slurm_job_id": "500"}})
        self.deployment = {"target": {"server": "aces", "hostnames": ["ac049"], "gpu_model": "NVIDIA H100 PCIe"},
            "backend": {"packages": {"jax": "synthetic"}}, "actual_runtime_environment": self.runtime,
            "executions": [{"path": str(packing), "mode": "packing-point"}]}
        dep = self.write("evidence/deployment.json", self.deployment)
        self.specification = {"unit_id": "explicit_h100_policy_seed_101", "source_root": "/logical/source", "source_commit": "a" * 40,
            "hardware_class": "h100", "copy_seeds": [101], "configuration": self.configuration,
            "runtime_environment": self.runtime, "deployment_verification": str(dep)}
        self.spec = self.write("evidence/final_specification.json", self.specification)
        self.admitted = {**self.specification, "validation": {"kind": "deployed_training_admission",
            "specification_digest": launch.metadata.digest(self.specification),
            "deployment_digest": launch.metadata.digest(self.deployment),
            "admission_records": [launch.receipt(self.spec)], "evidence": [launch.receipt(dep), launch.receipt(packing)]}}
        admitted = self.write("evidence/admitted.json", self.admitted)
        self.unit = {"schema_version": 1, "kind": "final_training_unit", "assigned_server": "aces", "hardware_class": "h100",
            "unit_id": self.specification["unit_id"], "numerical_source": {"root": "/logical/source", "commit": "a" * 40},
            "copy_seeds": [101], "configuration_digest": launch.metadata.digest(self.configuration),
            "runtime_environment_digest": launch.metadata.digest(self.runtime), "checkpoint_seconds": 1200,
            "helpers": self.helpers, "python": sys.executable, "final_specification": launch.receipt(self.spec)}
        descriptor = self.write("reserved/global_descriptor.json", self.unit)
        self.binding = {"global_descriptor": launch.receipt(descriptor), "admission_receipt": launch.receipt(admitted), "evidence_manifest": None}
        self.shape = {"account": "158590754315", "partition": "gpu", "qos": "normal", "gres": "gpu:h100:1", "cpus_per_task": 8,
                      "host_memory_gib": 32, "walltime_seconds": 172800, "checkpoint_reserve_seconds": 1800}
        template = self.write("resource_evidence/submitted.sbatch", "\n".join("#SBATCH --" + name + "=" + str(value) for name, value in (
            ("account", self.shape["account"]), ("partition", "gpu"), ("qos", "normal"), ("gres", "gpu:h100:1"), ("cpus-per-task", 8))))
        usage = self.write("resource_evidence/usage.txt", "500.batch|5242880K\n")
        self.proposal = {"schema_version": 1, "bindings": [self.binding], "resources": self.shape, "evidence_root": str(self.run),
            "resource_proof": {"scheduler_template": launch.receipt(template), "usage_record": launch.receipt(usage), "measurement_records": [launch.receipt(packing)]},
            "cpu_monitor": {"partition": "cpu", "qos": "normal", "cpus": 2, "memory_gib": 2, "walltime_seconds": 259200}}
        self.input = self.write("proposal.json", self.proposal)
        self.policy = launch.prepare_local(self.run, self.input)

    def write(self, relative, value):
        """Write one explicit synthetic JSON/text receipt under the isolated test run."""
        path = self.run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n" if not isinstance(value, str) else value)
        return path

    def test_prepare_is_idempotent_without_creating_training_outputs_or_new_seeds(self):
        """Existing descriptor/Python/seed/source identities are copied byte-for-byte into local metadata."""
        original = (self.queue / launch.POLICY).read_bytes()
        self.assertEqual(launch.prepare_local(self.run, self.input), self.policy)
        self.assertEqual((self.queue / launch.POLICY).read_bytes(), original)
        self.assertEqual(launch.metadata.read(self.queue / "units" / (self.unit["unit_id"] + ".json")), self.unit)
        self.assertFalse((self.run / "data").exists())
        self.assertFalse(any(name in sys.modules for name in ("jax", "numpy", "torch")))

    def test_old_deployed_helper_receipt_does_not_gain_a_dependency_its_callers_never_used(self):
        """An old checkpoint helper identity remains valid under the current metadata launcher."""
        unit = copy.deepcopy(self.unit)
        unit["helpers"]["files"].pop(launch.metadata.PREFIX + "pregather_async_retry.py")
        descriptor = self.write("reserved/old_helper_descriptor.json", unit)
        binding = {**self.binding, "global_descriptor": launch.receipt(descriptor)}
        admitted, _, _ = launch.verify_binding(self.run, binding)
        self.assertEqual(admitted["helpers"], unit["helpers"])
        self.assertEqual(admitted["copy_seeds"], self.unit["copy_seeds"])

    def test_changed_binding_helper_or_resource_proof_fails_before_submission(self):
        """Pinned actual bytes cannot be replaced while keeping a saved admission or policy."""
        path = Path(self.binding["global_descriptor"]["path"])
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            launch.verify_binding(self.run, self.binding)
        with self.assertRaisesRegex(ValueError, "recorded one-H100"):
            launch.resource_shape(self.run, {**self.policy, "resources": {**self.shape, "qos": "other"}})

    def test_selected_packing_memory_floor_allows_a_larger_explicit_request(self):
        """Recorded MaxRSS plus 25 percent is a minimum; explicit additional memory is allowed."""
        with self.assertRaisesRegex(ValueError, "25 percent"):
            launch.resource_shape(self.run, {**self.policy, "resources": {**self.shape, "host_memory_gib": 5}})
        shape = launch.resource_shape(self.run, {**self.policy, "resources": {**self.shape, "host_memory_gib": 64}})
        self.assertEqual(shape["host_memory_gib"], 64)
        with self.assertRaisesRegex(ValueError, "recorded one-H100"):
            launch.resource_shape(self.run, {**self.policy, "resources": {**self.shape, "host_memory_gib": True}})

    def test_maxrss_cannot_come_from_an_unrelated_owned_one_policy_canary(self):
        """Final resource sizing must use the selected source/configuration/copy count's actual packing job."""
        proof = copy.deepcopy(self.policy)
        proof["resource_proof"]["measurement_records"] = []
        with self.assertRaisesRegex(ValueError, "selected target packing"):
            launch.resource_shape(self.run, proof)
        self.write("slurm/submitted_jobids_aces.txt", "500\n501\n")
        usage = self.write("resource_evidence/unrelated_usage.txt", "501.batch|5242880K\n")
        proof = copy.deepcopy(self.policy)
        proof["resource_proof"]["usage_record"] = launch.receipt(usage)
        with self.assertRaisesRegex(ValueError, "exact selected target packing jobs"):
            launch.resource_shape(self.run, proof)

    def test_duplicate_seed_or_changed_policy_does_not_add_local_units(self):
        """Invalid preparation does not publish a partially enlarged registered roster."""
        before = sorted(path.name for path in (self.queue / "units").iterdir())
        proposal = copy.deepcopy(self.proposal)
        proposal["bindings"].append(self.binding)
        self.write(self.input, proposal)
        with self.assertRaisesRegex(ValueError, "listed twice"):
            launch.prepare_local(self.run, self.input)
        self.assertEqual(sorted(path.name for path in (self.queue / "units").iterdir()), before)
        self.write(self.input, {**self.proposal, "resources": {**self.shape, "walltime_seconds": 86400}})
        with self.assertRaisesRegex(ValueError, "immutable"):
            launch.prepare_local(self.run, self.input)

    def test_accounting_queries_only_exact_aces_ids_and_rejects_unknown_rows(self):
        """An equal numeric ID in UVA is not ACES ownership."""
        self.write("slurm/submitted_jobids_uva.txt", "777\n")
        with patch.object(launch.subprocess, "run") as query:
            with self.assertRaisesRegex(ValueError, "exact ACES"):
                launch.accounting(self.run, ["777"])
            query.assert_not_called()
        with patch.object(launch.subprocess, "run", return_value=SimpleNamespace(stdout="500|COMPLETED|0:0|2026-10-31T00:00:00|\n", returncode=0)) as query:
            rows, proof = launch.accounting(self.run, ["500"])
            self.assertEqual(rows["500"]["state"], "COMPLETED")
            self.assertEqual(query.call_args.args[0][4], "--format=" + launch.TERMINAL_COLUMNS)
        with patch.object(launch.subprocess, "run", return_value=SimpleNamespace(stdout="", returncode=0)):
            with self.assertRaisesRegex(ValueError, "unknown"):
                launch.accounting(self.run, ["500"])

    def test_scheduler_deadline_and_accounting_use_utc_with_a_non_utc_client(self):
        """The allocation stop and checkpoint reserve do not shift with the client timezone or daylight saving time."""
        expected = datetime(2026, 11, 1, 6, 30, tzinfo=timezone.utc).timestamp()
        raw = "500|COMPLETED|0:0|2026-11-01T06:30:00|\n"
        try:
            with patch.dict(launch.os.environ, {"TZ": "America/Los_Angeles"}), \
                    patch.object(launch.subprocess, "check_output", return_value="JobId=500 EndTime=2026-11-01T06:30:00 ") as control, \
                    patch.object(launch.subprocess, "run", return_value=SimpleNamespace(stdout=raw, returncode=0)) as accounting:
                launch.time.tzset()
                self.assertEqual(launch.scheduler_end("500"), expected)
                self.assertEqual(control.call_args.kwargs["env"]["TZ"], "UTC")
                rows, proof = launch.accounting(self.run, ["500"])
                self.assertEqual(accounting.call_args.kwargs["env"]["TZ"], "UTC")
                self.assertEqual(proof["stdout"], raw)
                self.assertEqual(proof["command"], ["sacct", "-X", "-j", "500", "--format=" + launch.TERMINAL_COLUMNS, "-P", "-n"])
        finally:
            launch.time.tzset()

    def test_owned_compute_rejects_login_and_foreign_cluster_job(self):
        """Job environment alone cannot authorize compute or cross-cluster validation."""
        real_read = Path.read_text
        def cgroup(path, *args, **kwargs):
            """Supply only the synthetic kernel cgroup while retaining actual metadata reads."""
            return "0::/slurm/job_500/step_batch\n" if path == Path("/proc/self/cgroup") else real_read(path, *args, **kwargs)
        with patch.dict(launch.os.environ, {"SLURM_JOB_ID": "500"}, clear=True), patch.object(Path, "read_text", cgroup), \
                patch.object(launch.socket, "gethostname", return_value="ac049"):
            self.assertEqual(launch.owned_compute(self.run), "500")
        with patch.dict(launch.os.environ, {"SLURM_JOB_ID": "777"}, clear=True), patch.object(Path, "read_text", cgroup), \
                patch.object(launch.socket, "gethostname", return_value="ac049"), self.assertRaises(ValueError):
            launch.owned_compute(self.run)

    def test_validator_cli_uses_original_pinned_helper_and_explicit_proof_selector(self):
        """A metadata upgrade cannot replace the helper used to admit this unit's continuation."""
        with patch.object(launch.subprocess, "run", return_value=SimpleNamespace(stdout='{"completed":false}')) as validate:
            self.assertEqual(launch.validation_command(self.unit, self.binding, self.run, checkpoint=True,
                terminal=self.run / "terminal.json", proof_attempt=2), {"completed": False})
        command = validate.call_args.args[0]
        self.assertEqual(command[0], self.unit["python"])
        self.assertEqual(command[1], str(self.root / launch.metadata.PREFIX / "validate_deployed_training_completion.py"))
        self.assertIn("--checkpoint", command)
        self.assertEqual(command[-2:], ["--checkpoint-proof-attempt", "2"])

    def test_portable_terminal_cli_never_combines_terminal_and_graceful_snapshot_inputs(self):
        """The real strict validator selects one ownership/presence proof protocol for each invocation."""
        manifest = self.write("nonempty_base_manifest.json", {"schema_version": 1, "kind": "portable_raw_evidence",
            "files": [{"original_path": "/logical/proof.json", "relative_path": "files/proof", "sha256": "a" * 64}], "git_roots": []})
        binding = {**self.binding, "evidence_manifest": launch.receipt(manifest)}
        transport = {"extended_manifest": {"path": str(self.run / "extended_manifest.json"), "sha256": "b" * 64},
                     "unit_snapshot": {"path": str(self.run / "output_snapshot.json"), "sha256": "c" * 64}}
        for terminal in (None, self.run / "terminal.json"):
            with self.subTest(terminal=terminal), patch.object(launch, "validation_transport", return_value=transport), \
                    patch.object(launch.subprocess, "run", return_value=SimpleNamespace(stdout='{"completed":false}')) as validate:
                launch.validation_command(self.unit, binding, self.run, checkpoint=True, terminal=terminal)
                command = validate.call_args.args[0]
                self.assertEqual("--unit-snapshot" in command, terminal is None)
                self.assertEqual("--terminal-receipt" in command, terminal is not None)
                self.assertIn("--evidence-manifest", command)

    def test_terminal_snapshot_contains_complete_owned_files_and_accounting(self):
        """Infrastructure recovery preserves failures/history rather than omitting inconvenient files."""
        directory = self.run / "data" / self.unit["unit_id"]
        self.write(directory / "progress.json", {"attempt": 2, "state": "running"})
        self.write(directory / "failure.json", {"attempt": 1, "error": "historical failure"})
        self.write(directory / "checkpoints/latest.pkl", "synthetic checkpoint bytes")
        self.write("final_training_queue/claims/" + self.unit["unit_id"] + "/aces_job-500.json", {"job_id": "500"})
        rows = {"500": {"state": "TIMEOUT"}}
        raw = {"command": ["sacct", "-X", "-j", "500", "--format=" + launch.TERMINAL_COLUMNS, "-P", "-n"],
               "returncode": 0, "stdout": "500|TIMEOUT|0:0|2026-10-31T00:00:00|\n"}
        with patch.object(launch, "accounting", return_value=(rows, raw)):
            proof = launch.metadata.read(launch.terminal_receipt(self.run, self.unit, "500"))
        snapshot = launch.metadata.read(proof["terminal_unit_snapshot"]["path"])
        self.assertIn("failure.json", snapshot["files"])
        self.assertIn("checkpoints/latest.pkl", snapshot["files"])
        self.assertIn(str(directory / "attempts/attempt-0002/failure.json"), snapshot["absent"])
        self.assertEqual(proof["accounting"]["command"], raw["command"])
        for state in ("FAILED", "OUT_OF_MEMORY", "COMPLETED"):
            with patch.object(launch, "accounting", return_value=({"500": {"state": state}}, raw)), self.assertRaises(ValueError):
                launch.terminal_receipt(self.run, self.unit, "500")

    def test_refresh_holds_application_failure_without_calling_checkpoint_validator(self):
        """Failed/OOM jobs never automatically become eligible for another application attempt."""
        status = self.queue / "status" / (self.unit["unit_id"] + ".json")
        self.write(status, {"state": "running", "unit_id": self.unit["unit_id"], "last_job_cluster": "aces", "last_job_id": "500"})
        self.write("final_training_queue/jobs/aces_job-500.json", {"job_id": "500", "kind": "aces_final_h100_worker"})
        with patch.object(launch, "owned_compute"), patch.object(launch, "accounting", return_value=({"500": {"state": "OUT_OF_MEMORY"}}, None)), \
                patch.object(launch, "validation_command") as validate:
            result = launch.refresh(self.run, self.policy)
            self.assertEqual(result["counts"]["held"], 1)
            validate.assert_not_called()
            self.assertEqual(launch.refresh(self.run, self.policy)["counts"]["held"], 1)
            validate.assert_not_called()

    def test_tick_revalidates_raw_admission_before_one_submission_and_holds_failures(self):
        """Full source/evidence is rebuilt on the owned CPU monitor before allocating an H100."""
        snapshot = {"units": [{"state": "ready", "unit_id": self.unit["unit_id"]}], "jobs": {},
                    "counts": {"ready": 1, "running": 0, "checkpointed": 0, "completed": 0, "held": 0}}
        with patch.object(launch, "owned_compute"), patch.object(launch, "refresh", return_value=snapshot), \
                patch.object(launch, "validation_command", return_value=self.admitted) as validate, \
                patch.object(launch, "submit", return_value={"job_id": "9001"}) as submit:
            self.assertEqual(launch.tick(self.run)["state"], "submitted")
            validate.assert_called_once()
            submit.assert_called_once()
            validate.return_value = {"changed": True}
            self.assertEqual(launch.tick(self.run)["state"], "held")
            self.assertEqual(submit.call_count, 1)

    @contextmanager
    def submission_environment(self, identifier="9001"):
        """Mock only scheduler/platform edges while retaining real intents, plans and durable ledgers."""
        with patch.object(launch.socket, "gethostname", return_value="alogin1"), \
                patch.object(launch, "verify_binding", return_value=(self.unit, self.admitted, self.deployment)), \
                patch.object(launch, "accounting", return_value=({}, None)), \
                patch.object(launch, "owned_compute"), patch.object(launch, "live_monitor"), \
                patch.object(launch, "metadata_source", return_value=self.helpers), \
                patch.object(launch.subprocess, "run", return_value=SimpleNamespace(stdout=identifier, returncode=0)) as sbatch:
            yield sbatch

    def test_one_gpu_submission_records_exact_id_before_worker_release_and_uses_reserved_resources(self):
        """One generic H100 job, one node and one exact ID retain all selected source/seed identities."""
        original_atomic = launch.metadata.atomic
        released = []
        def observe(path, value, **options):
            """The release record cannot appear before every owned exact-ID ledger is durable."""
            if Path(path).name == "aces_job-9001.json":
                for name in ("submitted_jobids_aces.txt", "submitted_jobids.txt", "deployed_training_jobids_aces.txt"):
                    self.assertIn("9001", (self.run / "slurm" / name).read_text().splitlines())
                released.append(True)
            return original_atomic(path, value, **options)
        with self.submission_environment() as sbatch, patch.object(launch.metadata, "atomic", side_effect=observe):
            record = launch.submit(self.run, self.policy)
        self.assertEqual(released, [True])
        command = sbatch.call_args.args[0]
        for argument in ("--nodes=1", "--ntasks=1", "--gres=gpu:h100:1", "--account=158590754315", "--qos=normal",
                         "--cpus-per-task=8", "--mem=32G", "--nodelist=ac049", "--signal=B:USR1@1800"):
            self.assertIn(argument, command)
        self.assertEqual(launch.metadata.read(record["job_plan"]["path"])["helpers"], self.helpers)
        self.assertFalse((self.run / "data").exists())

    def test_cpu_monitor_is_separate_and_unresolved_returned_id_blocks_repeated_sbatch(self):
        """A first CPU allocation contains no GPU request and failed publication retains its returned ID."""
        with self.submission_environment() as sbatch:
            launch.submit(self.run, self.policy, monitor=True)
        command = sbatch.call_args.args[0]
        self.assertIn("--partition=cpu", command)
        self.assertFalse(any(value.startswith("--gres") for value in command))
        with self.submission_environment("9002") as sbatch, patch.object(launch, "record_submission", side_effect=OSError("ledger unavailable")):
            with self.assertRaisesRegex(OSError, "ledger"):
                launch.submit(self.run, self.policy)
        intents = [launch.metadata.read(path) for path in (self.queue / "aces_submissions").glob("*.json")]
        self.assertTrue(any(row.get("state") == "submitting" and row.get("returned_job_id") == "9002" for row in intents))
        with self.submission_environment("9003") as sbatch:
            with self.assertRaisesRegex(ValueError, "unresolved"):
                launch.submit(self.run, self.policy)
            sbatch.assert_not_called()

    def test_submission_cannot_use_a_changed_valid_policy_object(self):
        """Scheduler arguments and the worker's hashed policy always describe the same explicit shape."""
        changed = copy.deepcopy(self.policy)
        changed["resources"]["walltime_seconds"] = 86400
        with self.submission_environment() as sbatch, self.assertRaisesRegex(ValueError, "on-disk policy"):
            launch.submit(self.run, changed)
        sbatch.assert_not_called()

    def test_active_job_and_missing_live_monitor_block_a_second_gpu_allocation(self):
        """Existing pending work and a missing CPU supervisor both prevent new GPU requests."""
        self.write("final_training_queue/jobs/aces_job-500.json", {"job_id": "500", "kind": "aces_final_h100_worker"})
        with self.submission_environment() as sbatch, patch.object(launch, "accounting", return_value=({"500": {"state": "PENDING"}}, None)):
            with self.assertRaisesRegex(ValueError, "active or pending"):
                launch.submit(self.run, self.policy)
            sbatch.assert_not_called()
        with self.submission_environment() as sbatch, patch.object(launch, "live_monitor", side_effect=ValueError("no recorded live CPU monitor")):
            with self.assertRaisesRegex(ValueError, "live CPU"):
                launch.submit(self.run, self.policy)
            sbatch.assert_not_called()

    def test_monitor_publishes_heartbeat_before_first_tick_and_finishes_without_sleep(self):
        """The first GPU submission can verify the actual already-recorded CPU monitor heartbeat."""
        plan = self.run / "monitor_plan.json"
        def finished(run):
            """Check initialization order without submitting or querying any real job."""
            self.assertEqual(launch.metadata.read(self.queue / "aces_monitor_heartbeat.json")["job_id"], "500")
            return {"state": "finished"}
        with patch.object(launch, "admitted_job", return_value=(self.run, "500", {}, self.policy, plan)), \
                patch.object(launch, "scheduler_end", return_value=launch.time.time() + 7200), \
                patch.object(launch.metadata, "process_birth", return_value="1234"), \
                patch.object(launch, "tick", side_effect=finished), patch.object(launch.time, "sleep") as sleep:
            launch.run_monitor(plan)
            sleep.assert_not_called()

    def test_extended_transport_uses_immutable_byte_copies_and_preserves_base_identities(self):
        """A later checkpoint write cannot change the evidence already handed to the strict validator."""
        base = self.run / "base_capsule"
        helper_clone = base / "repositories/helpers"
        helper_clone.parent.mkdir(parents=True)
        subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(self.root), str(helper_clone)], check=True)
        subprocess.run(["git", "-C", str(helper_clone), "checkout", "-q", "--detach", self.commit], check=True)
        global_bytes = b'{"global":"original raw bytes"}\n'
        expected = hashlib.sha256(global_bytes).hexdigest()
        self.write(base / "files" / expected, global_bytes.decode())
        manifest = {"schema_version": 1, "kind": "portable_raw_evidence", "files": [{"original_path": "/logical/global.json",
            "relative_path": "files/" + expected, "sha256": expected}], "git_roots": [{"original_root": str(self.root),
            "relative_path": "repositories/helpers", "commit": self.commit}]}
        path = self.write(base / "manifest.json", manifest)
        binding = {**self.binding, "evidence_manifest": launch.receipt(path)}
        original = path.read_bytes()
        directory = self.run / "data" / self.unit["unit_id"]
        self.write(directory / "progress.json", {"attempt": 1})
        checkpoint = self.write(directory / "checkpoints/latest.pkl", "first checkpoint bytes")
        transport = launch.validation_transport(self.run, self.unit, binding)
        resolver = launch.portable_evidence.ExactPathResolver(transport["extended_manifest"]["path"], transport["extended_manifest"]["sha256"])
        self.assertEqual(resolver.read_bytes("/logical/global.json"), global_bytes)
        self.assertEqual(resolver.read_bytes(checkpoint), b"first checkpoint bytes")
        self.assertNotEqual(resolver.file_path(checkpoint).stat().st_ino, checkpoint.stat().st_ino)
        checkpoint.write_text("new checkpoint bytes")
        self.assertEqual(resolver.read_bytes(checkpoint), b"first checkpoint bytes")
        self.assertEqual(path.read_bytes(), original)
        snapshot = launch.metadata.read(transport["unit_snapshot"]["path"])
        self.assertIn(str(directory / "attempts/attempt-0001/failure.json"), snapshot["absent"])

    def test_extended_transport_rejects_base_git_object_storage_outside_common_root(self):
        """Every translated base Git mapping has the same independence checks as a newly added helper root."""
        base = self.run / "base_capsule"
        cloned = base / "repositories/helpers"
        cloned.parent.mkdir(parents=True)
        subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(self.root), str(cloned)], check=True)
        subprocess.run(["git", "-C", str(cloned), "checkout", "-q", "--detach", self.commit], check=True)
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        objects = cloned / ".git/objects"
        external = Path(outside.name) / "objects"
        objects.rename(external)
        objects.symlink_to(external, target_is_directory=True)
        manifest = self.write(base / "manifest.json", {"schema_version": 1, "kind": "portable_raw_evidence", "files": [],
            "git_roots": [{"original_root": str(self.root), "relative_path": "repositories/helpers", "commit": self.commit}]})
        binding = {**self.binding, "evidence_manifest": launch.receipt(manifest)}
        with self.assertRaisesRegex(ValueError, "symlink|outside"):
            launch.validation_transport(self.run, self.unit, binding)
        self.assertFalse(list(self.run.glob("live_output_manifest_*.json")))

    @contextmanager
    def worker_environment(self, proof=None):
        """Replace external child/admission edges while exercising actual queue claims and metadata publication."""
        plan_path = self.write("worker_plan.json", {"helpers": self.helpers})
        record_path = self.write("final_training_queue/jobs/aces_job-500.json", {
            "schema_version": 1, "cluster": "aces", "job_id": "500", "kind": "aces_final_h100_worker", "job_plan": launch.receipt(plan_path)})
        process = unittest.mock.Mock(pid=777)
        process.wait.return_value = 0
        process.poll.return_value = 0
        def validate(*args, **kwargs):
            """The strict validator is independently tested; this fixture checks only its exact placement in orchestration."""
            return self.admitted if kwargs.get("admission_only") else proof or {"completed": True, "checkpoint_proof_attempt": 1}
        with patch.object(launch, "admitted_job", return_value=(self.run, "500", {}, self.policy, record_path)), \
                patch.object(launch, "scheduler_end", return_value=launch.time.time() + 10000), \
                patch.object(launch, "verify_binding", return_value=(self.unit, self.admitted, self.deployment)), \
                patch.object(launch, "validation_command", side_effect=validate) as validator, \
                patch.object(launch, "checkpoint_selector", return_value=None), \
                patch.object(launch.metadata, "process_birth", return_value="1234"), \
                patch.object(launch.signal, "signal"), \
                patch.object(launch.subprocess, "Popen", return_value=process) as start, \
                patch.dict(launch.os.environ, {}, clear=True):
            yield plan_path, process, start, validator

    def test_worker_uses_pinned_runner_one_seed_block_and_validates_actual_child_completion(self):
        """No numerical loop is implemented by the launcher; the existing deployed runner receives its original identity."""
        with self.worker_environment() as (plan, process, start, validator):
            launch.run_job(plan)
        command = start.call_args.args[0]
        self.assertEqual(command[:2], [self.unit["python"], str(self.root / launch.metadata.PREFIX / "train_deployed.py")])
        self.assertIn(self.unit["final_specification"]["path"], command)
        self.assertEqual(command[command.index("--checkpoint-seconds") + 1], "1200")
        self.assertGreater(float(command[command.index("--maximum-seconds") + 1]), 1800)
        self.assertEqual(start.call_args.kwargs["env"]["JAX_ENABLE_X64"], "true")
        self.assertEqual(start.call_args.kwargs["env"]["PYTHONNOUSERSITE"], "1")
        self.assertEqual(validator.call_count, 2)
        status = launch.metadata.read(self.queue / "status" / (self.unit["unit_id"] + ".json"))
        self.assertEqual(status["state"], "completed")
        claim = launch.metadata.read(self.queue / "claims" / self.unit["unit_id"] / "aces_job-500.json")
        self.assertEqual(claim["training_specification"], self.unit["final_specification"])
        self.assertEqual(claim["helpers"], self.helpers)

    def test_worker_publication_or_wait_failure_reaps_its_owned_child_before_holding_unit(self):
        """Metadata and wait exceptions cannot leave this process's training child running after supervisor failure."""
        for boundary in ("publication", "wait"):
            self.write("final_training_queue/status/" + self.unit["unit_id"] + ".json", {
                "state": "ready", "unit_id": self.unit["unit_id"], "last_job_id": None, "last_job_cluster": "aces"})
            claim = self.queue / "claims" / self.unit["unit_id"] / "aces_job-500.json"
            claim.unlink(missing_ok=True)
            with self.subTest(boundary=boundary), self.worker_environment() as (plan, process, start, validator):
                process.poll.return_value = None
                if boundary == "wait":
                    process.wait.side_effect = RuntimeError("synthetic wait failure")
                original = launch.metadata.atomic
                failed, cleaned = [False], []
                def clean(children, deadline):
                    """Only the exact Popen-owned child is stopped and reaped at this boundary."""
                    self.assertEqual(children, [process])
                    cleaned.append(True)
                    process.poll.return_value = 0
                def publish(path, value, **options):
                    """Inject one real metadata boundary and require cleanup before publishing its hold."""
                    if value.get("state") == "held":
                        self.assertEqual(cleaned, [True])
                    if boundary == "publication" and value.get("state") == "running" and "child_pid" in value and not failed[0]:
                        failed[0] = True
                        raise OSError("synthetic publication failure")
                    return original(path, value, **options)
                with patch.object(launch.metadata, "cleanup_children", side_effect=clean), patch.object(launch.metadata, "atomic", side_effect=publish):
                    with self.assertRaises((RuntimeError, OSError)):
                        launch.run_job(plan)
                self.assertEqual(cleaned, [True])
                self.assertEqual(launch.metadata.read(self.queue / "status" / (self.unit["unit_id"] + ".json"))["state"], "held")

    def test_signal_during_admission_restores_unstarted_unit_without_creating_child(self):
        """A queued stop request cannot be lost merely because the numerical child does not yet exist."""
        handlers = {}
        with self.worker_environment() as (plan, process, start, validator), \
                patch.object(launch.signal, "signal", side_effect=lambda number, handler: handlers.update({number: handler})):
            def stop_during_admission(*args, **kwargs):
                """Fire the scheduler stop while the supervisor is still rebuilding raw source evidence."""
                handlers[launch.signal.SIGTERM](launch.signal.SIGTERM, None)
                return self.admitted
            validator.side_effect = stop_during_admission
            launch.run_job(plan)
            start.assert_not_called()
            self.assertEqual(launch.metadata.read(self.queue / "status" / (self.unit["unit_id"] + ".json"))["state"], "ready")

    def test_signal_in_popen_assignment_gap_is_forwarded_after_child_ownership_publication(self):
        """A child created as the handler runs still receives the stored original checkpoint request."""
        handlers = {}
        with self.worker_environment({"completed": False, "checkpoint_proof_attempt": 1}) as (plan, process, start, validator), \
                patch.object(launch.signal, "signal", side_effect=lambda number, handler: handlers.update({number: handler})):
            process.poll.return_value = None
            def start_during_signal(*args, **kwargs):
                """Popen has created its process while the caller has not assigned the returned handle."""
                handlers[launch.signal.SIGUSR1](launch.signal.SIGUSR1, None)
                return process
            start.side_effect = start_during_signal
            launch.run_job(plan)
            process.send_signal.assert_called_once_with(launch.signal.SIGUSR1)
            self.assertEqual(launch.metadata.read(self.queue / "status" / (self.unit["unit_id"] + ".json"))["state"], "checkpointed")

    def test_absolute_stop_accounts_for_child_startup_and_requests_a_complete_checkpoint(self):
        """The parent's allocation deadline covers admission/startup outside the shared trainer's elapsed timer."""
        with self.worker_environment({"completed": False, "checkpoint_proof_attempt": 1}) as (plan, process, start, validator):
            process.wait.side_effect = [subprocess.TimeoutExpired("owned runner", 1), 0]
            process.poll.return_value = None
            launch.run_job(plan)
            process.send_signal.assert_called_once_with(launch.signal.SIGUSR1)
            self.assertEqual(process.wait.call_count, 2)
            self.assertGreater(process.wait.call_args_list[0].kwargs["timeout"], 1800)
        self.assertEqual(launch.metadata.read(self.queue / "status" / (self.unit["unit_id"] + ".json"))["state"], "checkpointed")

    def test_nonzero_application_exit_is_held_without_checkpoint_revalidation_or_repeat(self):
        """A true application failure cannot be renamed an infrastructure continuation."""
        with self.worker_environment() as (plan, process, start, validator):
            process.wait.return_value = 1
            process.poll.return_value = 1
            with self.assertRaisesRegex(ValueError, "application failed"):
                launch.run_job(plan)
            self.assertEqual(validator.call_count, 1)
        self.assertEqual(launch.metadata.read(self.queue / "status" / (self.unit["unit_id"] + ".json"))["state"], "held")

    def test_checkpoint_receipt_rechecks_actual_pickle_bytes_before_resume(self):
        """A saved proof cannot authorize different live checkpoint bytes at a later allocation."""
        directory = self.run / "data" / self.unit["unit_id"]
        identity = self.write(directory / "training_identity.json", {"synthetic": "complete identity"})
        checkpoint = self.write(directory / "checkpoints/latest.pkl", "old verified checkpoint")
        proof = {"unit_id": self.unit["unit_id"], "source_commit": self.unit["numerical_source"]["commit"],
            "configuration_digest": self.unit["configuration_digest"], "copy_seeds": self.unit["copy_seeds"],
            "identity_digest": launch.metadata.digest(launch.metadata.read(identity)), "checkpoint_sha256": launch.metadata.file_digest(checkpoint),
            "completed": False, "evidence": [launch.receipt(identity), launch.receipt(checkpoint)]}
        path = self.write("final_training_queue/validation/prior.json", proof)
        status = {"checkpoint_validation": launch.receipt(path)}
        self.assertEqual(launch.verify_checkpoint_receipt(self.run, self.unit, status), proof)
        checkpoint.write_text("changed live checkpoint")
        with self.assertRaisesRegex(ValueError, "changed before reuse"):
            launch.verify_checkpoint_receipt(self.run, self.unit, status)

    def test_initial_existing_outputs_cannot_be_adopted_without_original_queue_history(self):
        """Preparation cannot silently continue an unregistered old training directory."""
        (self.queue / "status" / (self.unit["unit_id"] + ".json")).unlink()
        self.write("data/" + self.unit["unit_id"] + "/progress.json", {"state": "running"})
        with self.assertRaisesRegex(ValueError, "original registered"):
            launch.prepare_local(self.run, self.input)

    def test_terminal_checkpoint_refresh_uses_shared_validator_before_release(self):
        """A graceful terminal job and an abrupt infrastructure job each need the shared raw checkpoint validator."""
        for state in ("COMPLETED", "TIMEOUT"):
            self.write("final_training_queue/status/" + self.unit["unit_id"] + ".json", {
                "state": "checkpointed" if state == "COMPLETED" else "running", "unit_id": self.unit["unit_id"],
                "last_job_cluster": "aces", "last_job_id": "500"})
            self.write("final_training_queue/jobs/aces_job-500.json", {"job_id": "500", "kind": "aces_final_h100_worker"})
            with self.subTest(state=state), patch.object(launch, "owned_compute"), \
                    patch.object(launch, "accounting", return_value=({"500": {"state": state}}, None)), \
                    patch.object(launch, "terminal_receipt", return_value=self.run / "owned_terminal.json") as terminal, \
                    patch.object(launch, "checkpoint_selector", return_value=1), \
                    patch.object(launch, "validation_command", return_value={"completed": False, "checkpoint_proof_attempt": 1}) as validate:
                result = launch.refresh(self.run, self.policy)
                self.assertEqual(result["counts"]["ready"], 1)
                self.assertEqual(validate.call_args.kwargs["proof_attempt"], 1)
                self.assertTrue(validate.call_args.kwargs["checkpoint"])
                self.assertEqual(terminal.call_count, int(state == "TIMEOUT"))
                self.assertEqual(launch.metadata.read(self.queue / "status" / (self.unit["unit_id"] + ".json"))["checkpoint_proof_attempt"], 1)

    def test_ready_unit_foreign_cluster_marker_is_held_before_any_accounting(self):
        """Numeric overlap with an owned ACES ID does not erase a prior foreign-cluster identity."""
        self.write("final_training_queue/status/" + self.unit["unit_id"] + ".json", {
            "state": "ready", "unit_id": self.unit["unit_id"], "last_job_id": "500", "last_job_cluster": "uva"})
        with patch.object(launch, "owned_compute"), patch.object(launch, "accounting") as query, patch.object(launch, "submit") as submit:
            result = launch.tick(self.run)
            self.assertEqual(result["state"], "held")
            self.assertIn("ACES unit/cluster", result["reason"])
            query.assert_not_called()
            submit.assert_not_called()

    def test_overlapping_tick_waits_without_scheduler_or_submission(self):
        """The shared decision lock makes a concurrent tick harmless to the already-running CPU monitor."""
        with (self.queue / "aces_monitor.lock").open("a+") as lock, patch.object(launch, "owned_compute"), \
                patch.object(launch, "accounting") as query, patch.object(launch, "submit") as submit:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(launch.tick(self.run)["state"], "waiting")
            query.assert_not_called()
            submit.assert_not_called()

    def test_malformed_policy_is_reported_held_without_ending_monitor(self):
        """Bad metadata never produces a false zero-unit completion or a new job."""
        (self.queue / launch.POLICY).write_text("{bad json")
        with patch.object(launch, "owned_compute"), patch.object(launch, "submit") as submit:
            result = launch.tick(self.run)
            self.assertEqual(result["state"], "held")
            submit.assert_not_called()
            report = (self.run / "20_mins_monitoring/aces_latest_report.md").read_text()
            for section in ("TRAINING", "WAITING", "HELD", "FINISHED", "PLANNED"):
                self.assertIn(section, report)

    def test_disabled_tick_has_no_scheduler_or_directory_side_effects(self):
        """An earlier campaign phase cannot implicitly prepare remote final policies."""
        with tempfile.TemporaryDirectory() as temporary, patch.object(launch, "owned_compute") as compute:
            self.assertEqual(launch.tick(Path(temporary))["state"], "disabled")
            compute.assert_not_called()
            self.assertFalse((Path(temporary) / "final_training_queue").exists())


if __name__ == "__main__":
    unittest.main()
