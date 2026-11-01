"""Validate full deployed completion from synthetic detached Git and checkpoint bytes only."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import validate_deployed_training_completion as completion


class DeployedCompletionTests(unittest.TestCase):
    """Preserve target-specific admission and reject incomplete or replaced completion records."""

    def setUp(self):
        """Build full raw global/target proofs, then create an explicit completed-process metadata fixture."""
        definition = importlib.util.spec_from_file_location("deployed_completion_admission_fixture", SCRIPTS / "tests/test_deployed_training_admission.py")
        fixtures = importlib.util.module_from_spec(definition)
        definition.loader.exec_module(fixtures)
        self.fixture = fixtures.DeployedTrainingAdmissionTests("test_target_selected_count_is_compatible_with_training_specification_validation")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.raw = self.fixture.fixture
        self.path = self.fixture.path
        self.admitted = completion.admission.validate_deployed_training_admission(self.path)
        self.receipt = self.fixture.deployment
        self.directory = self.fixture.directory / "training_campaign/data" / self.admitted["unit_id"]
        self.directory.mkdir(parents=True)
        completion.training.prepare_unit(self.directory, completion.runner.immutable_specification(self.admitted))
        self.number, self.attempt = completion.training.new_attempt(self.directory)
        self.hardware = {"hostname": "ac049", "gpu_uuid": "GPU-target", "gpu_query": "GPU-target, NVIDIA H100 PCIe, 1000 MiB, synthetic-driver",
            "pid": 9000, "process_start_time": "90001", "runtime_environment": self.raw.actual_runtime, "slurm_job_id": "12345"}
        self.assets = {"root": str(self.fixture.operational_assets), **self.receipt["assets"]}
        hooks = completion.runner.DeploymentMetadata(self.admitted, self.receipt, "aces", self.hardware, self.assets, self.directory, self.attempt)
        hooks.before_numerical_imports(self.raw.installed)
        self.backend = {"device_kind": "NVIDIA H100 PCIe", "driver_version": "synthetic-driver", "packages": self.raw.installed["packages"],
            "jaxtari_source": self.raw.installed["jaxtari_source"], "installed_gpu_libraries": self.raw.installed["gpu_libraries"]}
        metadata = hooks.identity_metadata(self.raw.actual_runtime, self.backend, self.assets)
        helpers_root = self.fixture.directory / "runner_helpers"
        for name in completion.runner.HELPERS:
            target = helpers_root / completion.deployment.PREFIX / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SCRIPTS / name, target)
        helper_commit = self.raw.commit_source(helpers_root)
        helpers = completion.training.source_receipt(helpers_root, helper_commit,
            [completion.deployment.PREFIX + name for name in completion.runner.HELPERS])
        self.identity = completion.training.checkpoint_identity(self.admitted, helpers, metadata["runtime_environment"], metadata["backend"], metadata["assets"])
        self.identity["deployment"] = metadata["deployment"]
        self.loaded = {**self.raw.loaded, "pid": self.hardware["pid"]}
        hooks.after_first_update(self.loaded)
        self.write("loaded_gpu_libraries.json", self.loaded)
        self.write("loaded_gpu_libraries.json", self.loaded, attempt=True)
        self.target = 122070
        self.steps = self.target * 128 * 128
        self.result = {"completed": True, "update": self.target, "target_updates": self.target, "requested_steps_per_copy": 2_000_000_000,
            "steps_per_copy": self.steps, "copy_seeds": self.admitted["copy_seeds"], "copies": 2,
            "identity_digest": completion.training.canonical_digest(self.identity), "pending_report_updates": 0,
            "last_report_iteration": 0, "stop_reason": "completed", "process_elapsed_seconds": 11.,
            "recorded_elapsed_seconds_total": 12., "attempt": self.number, "time": "synthetic recorded finish"}
        checkpoint = self.directory / "checkpoints/latest.pkl"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"synthetic full-state pickle bytes; never unpickle this metadata fixture")
        self.checkpoint = {"schema_version": 1, "kind": "optimized_training_checkpoint", "identity_digest": self.result["identity_digest"],
            "configuration_digest": completion.training.canonical_digest(self.admitted["configuration"]), "source_commit": self.admitted["source_commit"],
            "iteration": self.target, "target_updates": self.target, "steps_per_copy": self.steps, "copy_seeds": self.admitted["copy_seeds"],
            "pending_report_updates": 0, "elapsed_seconds": 11.9, "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "loaded_gpu_libraries": self.loaded}
        self.write("training_identity.json", self.identity)
        self.write("checkpoints/latest.metadata.json", self.checkpoint)
        self.write("final_specification.json", self.admitted)
        self.write("validation_receipt.json", self.admitted["validation"])
        self.write("hardware.json", {"hostname": "ac049", "pid": self.hardware["pid"], "device_kind": self.backend["device_kind"],
                                     "runtime_environment": self.raw.actual_runtime})
        self.publish_result()

    def write(self, relative, value, attempt=False):
        """Publish one explicit raw metadata change without touching scientific records from other tests."""
        path = (self.attempt if attempt else self.directory) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def publish_result(self):
        """Retain exactly matching root/attempt result and progress files as the actual trainer does."""
        for attempt in (False, True):
            self.write("result.json", self.result, attempt)
            self.write("progress.json", {**self.result, "state": "completed" if self.result["completed"] else "checkpointed", "pid": self.hardware["pid"]}, attempt)

    def boundary(self, iteration):
        """Describe a real unpadded graceful checkpoint without claiming full-budget completion."""
        self.result.update(completed=False, update=iteration, steps_per_copy=iteration * 128 * 128, stop_reason="SIGUSR1")
        self.checkpoint.update(iteration=iteration, steps_per_copy=iteration * 128 * 128)
        if iteration == 0:
            self.checkpoint["loaded_gpu_libraries"] = None
        self.write("checkpoints/latest.metadata.json", self.checkpoint)
        self.publish_result()

    def terminal(self, state="TIMEOUT", no_result=True):
        """Produce a fully bound owned-job receipt and complete frozen unit snapshot outside the unit."""
        campaign = self.directory.parent.parent
        queue = campaign / "final_training_queue"
        queue.mkdir(exist_ok=True)

        def save(name, value):
            """Pin a synthetic external metadata file by its actual original bytes."""
            path = queue / name
            path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
            return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        if no_result:
            for path in (self.directory / "result.json", self.attempt / "result.json"):
                path.unlink(missing_ok=True)
            for attempt in (False, True):
                self.write("progress.json", {"state": "running", "attempt": self.number, "pid": self.hardware["pid"]}, attempt)
        ledger = campaign / "slurm/submitted_jobids_aces.txt"
        ledger.parent.mkdir(exist_ok=True)
        ledger.write_text("12345\n")
        plan = save("plan.json", {"helpers": self.identity["helpers"]})
        generic = save("job.json", {"schema_version": 1, "cluster": "aces", "job_id": "12345", "job_plan": plan})
        claim = save("claim.json", {"schema_version": 1, "cluster": "aces", "job_id": "12345", "unit_id": self.admitted["unit_id"],
            "training_specification": {"path": str(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()},
            "helpers": self.identity["helpers"], "job_plan": plan, "generic_submission": generic,
            "supervisor_pid": 8000, "supervisor_process_start_time": "80001"})
        files = {str(path.relative_to(self.directory)): {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                 for path in self.directory.rglob("*") if path.is_file()}
        absent = [str(path) for path in (self.directory / "result.json", self.directory / "failure.json",
                  self.attempt / "result.json", self.attempt / "failure.json") if not path.exists()]
        observed = "2026-10-31T12:00:00+00:00"
        snapshot = save("snapshot.json", {"schema_version": 1, "kind": "deployed_training_terminal_unit_snapshot",
            "unit_directory": str(self.directory), "observed_at": observed, "attempt": self.number, "files": files, "absent": absent})
        receipt = {"schema_version": 1, "kind": "owned_slurm_terminal_receipt", "cluster": "aces", "campaign_run_dir": str(campaign),
            "job_id": "12345", "observed_at": observed, "ledger_path": str(ledger), "ledger_snapshot": save("ledger.txt", ledger.read_bytes()),
            "submission_record": claim, "accounting": {"command": ["sacct", "-X", "-j", "12345", "--format=JobIDRaw,State,ExitCode,End", "-P", "-n"],
                "returncode": 0, "stdout": save("accounting.txt", ("12345|" + state + "|0:0|2026-10-31T11:59:00|\n").encode())},
            "terminal_unit_snapshot": snapshot}
        return Path(save("terminal.json", receipt)["path"])

    def validate(self):
        """Run full deployed admission and raw completion validation, never ordinary NVL admission."""
        return completion.validate_deployed_training_completion(self.path, self.directory)

    def test_exact_completed_schedule_and_target_count_are_validated_without_arrays(self):
        """The PCIe-selected count is two even though the global NVL proof selected one."""
        with mock.patch.object(completion.admission.validation, "validate_training_admission", side_effect=AssertionError("ordinary NVL admission used")), \
                mock.patch("pickle.load", side_effect=AssertionError("a raw completion validator must not unpickle")), \
                mock.patch.object(Path, "rglob", side_effect=AssertionError("a raw completion validator must not walk assets")):
            record = self.validate()
        self.assertEqual(record["kind"], "deployed_training_completion")
        self.assertTrue(record["completed"])
        self.assertEqual(record["updates"], 122070)
        self.assertEqual(record["steps_per_copy"], 1_999_994_880)
        self.assertEqual(record["copy_seeds"], [1001, 1002])
        self.assertEqual(record["deployment_digest"], self.admitted["validation"]["deployment_digest"])
        self.assertFalse(any(name in sys.modules for name in ("jax", "numpy", "torch")))

    def test_partial_run_extra_update_or_wrong_requested_budget_fails(self):
        """A stopped run, one omitted/extra update and another budget cannot be presented as 2B completion."""
        original = copy.deepcopy(self.result)
        for name, value in (("completed", False), ("update", 122069), ("update", 122071), ("requested_steps_per_copy", 1_000_000_000),
                            ("steps_per_copy", 2_000_000_000), ("copies", 1), ("copy_seeds", [1, 2]), ("pending_report_updates", 1)):
            self.result = {**original, name: value}
            self.publish_result()
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "update/report schedule|graceful checkpoint"):
                self.validate()

    def test_replaced_checkpoint_bytes_or_metadata_identity_fails(self):
        """The raw full-state bytes and readable sidecar must retain the admitted identity and exact schedule."""
        path = self.directory / "checkpoints/latest.pkl"
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "recorded bytes"):
            self.validate()
        path.write_bytes(original)
        for name, value in (("identity_digest", "0" * 64), ("iteration", 122069), ("copy_seeds", [1, 2]),
                            ("source_commit", "0" * 40), ("checkpoint_path", "/another/checkpoint.pkl")):
            self.write("checkpoints/latest.metadata.json", {**self.checkpoint, name: value})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "checkpoint"):
                self.validate()

    def test_identity_source_seeds_deployment_runtime_assets_and_helpers_are_bound(self):
        """Refreshing an identity digest cannot make an altered scientific or backend identity valid."""
        for field in ("source_root", "source_commit", "configuration", "copy_seeds", "runtime_environment", "deployment", "assets", "helpers"):
            changed = copy.deepcopy(self.identity)
            if field in ("source_root", "source_commit"):
                changed[field] = "/another/source" if field == "source_root" else "0" * 40
            elif field == "copy_seeds":
                changed[field] = [1003, 1004]
            elif field == "configuration":
                changed[field]["execution"]["iterations_per_call"] = 5
            elif field == "runtime_environment":
                changed[field]["JAX_ENABLE_COMPILATION_CACHE"] = "true"
            elif field == "deployment":
                changed[field]["deployment_digest"] = "0" * 64
            elif field == "assets":
                changed[field]["tree_digest"] = "0" * 64
            else:
                changed[field]["files"].pop(completion.deployment.PREFIX + "train_deployed.py")
            self.write("training_identity.json", changed)
            with self.subTest(field=field), self.assertRaises((ValueError, FileNotFoundError)):
                self.validate()

    def test_actual_completed_attempt_cannot_reuse_another_process_or_model(self):
        """Completion requires the actual target backend proof from the recorded finished process."""
        changes = [("placement_hardware.json", {**self.hardware, "pid": 9001}),
                   ("placement_hardware.json", {**self.hardware, "gpu_query": "GPU-target, NVIDIA H100 NVL, 1000 MiB, synthetic-driver"}),
                   ("loaded_gpu_libraries.json", {**self.loaded, "pid": 9001}),
                   ("asset_receipt.json", {**self.assets, "tree_digest": "0" * 64})]
        originals = {name: (self.attempt / name).read_bytes() for name, _ in changes}
        for name, value in changes:
            self.write(name, value, attempt=True)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate()
            (self.attempt / name).write_bytes(originals[name])
        (self.attempt / "deployment_backend_match.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()

    def test_replaced_original_proof_or_validation_receipt_fails(self):
        """A saved completion cannot replace the raw global/target deployment or admission proof."""
        validation = copy.deepcopy(self.admitted["validation"])
        validation["deployment_digest"] = "0" * 64
        self.write("validation_receipt.json", validation)
        with self.assertRaisesRegex(ValueError, "specification/admission"):
            self.validate()
        self.write("validation_receipt.json", self.admitted["validation"])
        confirmation = Path(self.raw.specification["confirmation_record"])
        confirmation.write_bytes(confirmation.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "evidence"):
            self.validate()

    def test_graceful_partial_and_initial_checkpoints_do_not_claim_completion(self):
        """An exact graceful boundary is resumable; the zero-update state requires installed admission only."""
        for iteration in (125, 0):
            self.boundary(iteration)
            record = completion.validate_deployed_training_checkpoint(self.path, self.directory)
            self.assertFalse(record["completed"])
            self.assertEqual(record["updates"], iteration)
            self.assertEqual(record["checkpoint_proof_attempt"], 1 if iteration else None)
            with self.assertRaisesRegex(ValueError, "not completed"):
                self.validate()
        with self.assertRaisesRegex(ValueError, "zero-update"):
            completion.validate_deployed_training_checkpoint(self.path, self.directory, checkpoint_proof_attempt=1)

    def test_periodic_checkpoint_requires_owned_terminal_job_proof_and_exact_snapshot(self):
        """TIMEOUT can retain a periodic checkpoint; ledger appends do not invalidate its exact owned row."""
        self.boundary(125)
        terminal = self.terminal()
        record = completion.validate_deployed_training_checkpoint(self.path, self.directory, terminal_receipt=terminal)
        self.assertFalse(record["completed"])
        self.assertEqual(record["terminal"]["state"], "TIMEOUT")
        ledger = self.directory.parent.parent / "slurm/submitted_jobids_aces.txt"
        with ledger.open("a") as stream:
            stream.write("12346\n")
        self.assertFalse(completion.validate_deployed_training_checkpoint(self.path, self.directory, terminal_receipt=terminal)["completed"])
        ledger.write_text("12346\n")
        with self.assertRaisesRegex(ValueError, "owned append-only ledger"):
            completion.validate_deployed_training_checkpoint(self.path, self.directory, terminal_receipt=terminal)

    def test_application_failure_oom_unknown_or_omitted_unit_file_never_auto_resume(self):
        """An owned allocation is insufficient when it failed in the application or its snapshot omitted a file."""
        self.boundary(125)
        for state in ("OUT_OF_MEMORY", "FAILED", "UNKNOWN", "COMPLETED"):
            terminal = self.terminal(state)
            with self.subTest(state=state), self.assertRaises(ValueError):
                completion.validate_deployed_training_checkpoint(self.path, self.directory, terminal_receipt=terminal)
        terminal = self.terminal()
        self.write("failure.json", {"attempt": 1, "error": "application error"}, attempt=True)
        with self.assertRaisesRegex(ValueError, "snapshot omitted|include all"):
            completion.validate_deployed_training_checkpoint(self.path, self.directory, terminal_receipt=terminal)
        terminal = self.terminal()
        with self.assertRaisesRegex(ValueError, "application-failed"):
            completion.validate_deployed_training_checkpoint(self.path, self.directory, terminal_receipt=terminal)

    def test_full_checkpoint_terminal_failure_then_completion_uses_explicit_prior_proof(self):
        """A full-budget saved state can be finalized without adding updates or inventing a new library observation."""
        terminal = self.terminal()
        incomplete = completion.validate_deployed_training_checkpoint(self.path, self.directory, terminal_receipt=terminal)
        self.assertFalse(incomplete["completed"])
        self.assertEqual(incomplete["updates"], 122070)
        # Before: a terminal process saved the final state but no result. After: an installed-admitted process restores it.
        self.number, self.attempt = completion.training.new_attempt(self.directory)
        self.hardware = {**self.hardware, "pid": 9002, "process_start_time": "90002", "slurm_job_id": "12346"}
        hooks = completion.runner.DeploymentMetadata(self.admitted, self.receipt, "aces", self.hardware, self.assets, self.directory, self.attempt)
        hooks.before_numerical_imports(self.raw.installed)
        self.result["attempt"] = 2
        self.write("hardware.json", {"hostname": "ac049", "pid": 9002, "device_kind": self.backend["device_kind"],
                                     "runtime_environment": self.raw.actual_runtime})
        self.publish_result()
        with self.assertRaises(FileNotFoundError):
            self.validate()
        finished = completion.validate_deployed_training_completion(self.path, self.directory, checkpoint_proof_attempt=1)
        self.assertTrue(finished["completed"])
        self.assertEqual(finished["updates"], 122070)
        self.assertEqual(finished["checkpoint_proof_attempt"], 1)
        self.assertEqual(finished["process"]["pid"], 9002)
        self.assertEqual(finished["process"]["checkpoint_proof_process"]["pid"], 9000)
        prior = self.directory / "attempts/attempt-0001/loaded_gpu_libraries.json"
        changed = json.loads(prior.read_text())
        changed["cudnnGetVersion"] = 1
        prior.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "another process/checkpoint"):
            completion.validate_deployed_training_completion(self.path, self.directory, checkpoint_proof_attempt=1)

    def relocate(self):
        """Move exact byte copies and real detached Git roots, then remove every original evidence directory."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        staging = Path(temporary.name)
        manifest = {"schema_version": 1, "kind": "portable_raw_evidence", "files": [], "git_roots": []}
        roots = [(self.raw.source, "numerical", self.raw.commit), (self.raw.public, "public", completion.deployment.JAXTARI_COMMIT),
                 (Path(self.identity["helpers"]["source_root"]), "helpers", self.identity["helpers"]["source_commit"])]
        for root, name, commit in roots:
            target = staging / "repositories" / name
            subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(root), str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "-q", "--detach", commit], check=True)
            manifest["git_roots"].append({"original_root": str(root), "relative_path": "repositories/" + name, "commit": commit})
        for root in (self.fixture.directory, self.raw.fixture.directory):
            for path in sorted(root.rglob("*")):
                if not path.is_file() or ".git" in path.relative_to(root).parts:
                    continue
                contents = path.read_bytes()
                digest = hashlib.sha256(contents).hexdigest()
                target = staging / "files" / digest
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(contents)
                manifest["files"].append({"original_path": str(path), "relative_path": "files/" + digest, "sha256": digest})
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        resolver = completion.portable_evidence.ExactPathResolver(manifest_path)
        shutil.rmtree(self.fixture.directory)
        shutil.rmtree(self.raw.fixture.directory)
        return resolver

    def test_portable_completion_preserves_original_identity_after_original_trees_are_removed(self):
        """All completion files, libraries and real Git roots resolve exactly without rewriting raw evidence."""
        original = self.validate()
        resolver = self.relocate()
        self.assertEqual(completion.validate_deployed_training_completion(self.path, self.directory, resolver=resolver), original)

    def test_portable_graceful_boundary_requires_explicit_complete_unit_snapshot(self):
        """A mapped absent failure needs a byte-bound snapshot; missing resolver entries never prove absence."""
        self.boundary(125)
        terminal_path = self.terminal(no_result=False)
        terminal = json.loads(terminal_path.read_text())
        snapshot = json.loads(Path(terminal["terminal_unit_snapshot"]["path"]).read_text())
        snapshot["kind"] = "deployed_training_unit_snapshot"
        snapshot_path = terminal_path.parent / "graceful_snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot))
        original = completion.validate_deployed_training_checkpoint(self.path, self.directory, unit_snapshot=snapshot_path)
        resolver = self.relocate()
        with self.assertRaisesRegex(ValueError, "no exact file entry"):
            completion.validate_deployed_training_checkpoint(self.path, self.directory, resolver=resolver)
        moved = completion.validate_deployed_training_checkpoint(self.path, self.directory, resolver=resolver, unit_snapshot=snapshot_path)
        self.assertEqual(moved, original)
        self.assertFalse(moved["completed"])
        omitted = copy.deepcopy(snapshot)
        omitted["files"].pop("progress.json")
        with self.assertRaisesRegex(ValueError, "all mapped unit files"):
            completion.checked_unit_snapshot(omitted, self.directory, completion.deployment.DeploymentEvidence(resolver),
                                             "deployed_training_unit_snapshot")
        # Before: the snapshot says the current failure was absent. After: the exact manifest explicitly maps that failure.
        manifest = json.loads(resolver.manifest_path.read_text())
        failure = b'{"attempt":1,"error":"recorded application failure"}'
        digest = hashlib.sha256(failure).hexdigest()
        target = resolver.manifest_path.parent / "files" / digest
        target.write_bytes(failure)
        manifest["files"].append({"original_path": str(self.attempt / "failure.json"), "relative_path": "files/" + digest, "sha256": digest})
        changed = resolver.manifest_path.parent / "manifest_with_current_failure.json"
        changed.write_text(json.dumps(manifest))
        conflicting = completion.portable_evidence.ExactPathResolver(changed)
        with self.assertRaisesRegex(ValueError, "explicitly mapped file absent"):
            completion.validate_deployed_training_checkpoint(self.path, self.directory, resolver=conflicting, unit_snapshot=snapshot_path)


if __name__ == "__main__":
    unittest.main()
