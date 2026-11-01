"""Exercise UVA final continuation using synthetic receipts and mocked scheduler/GPU processes."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import diagnose_gpu_execution as diagnostic
import launch_final_training as final
import train_optimized as training


class FinalLauncherTests(unittest.TestCase):
    """Keep science, seed ownership, queue claims and exact scheduler history explicit."""

    def setUp(self):
        """Build already-computed synthetic admission artifacts without running their numerical code."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.queue = self.run / "final_training_queue"
        self.commit = "a" * 40
        self.helpers = {"source_root": "/synthetic/detached/helpers", "source_commit": "b" * 40,
                        "files": {final.PREFIX + name: "c" * 64 for name in final.TRAIN_HELPERS}}
        self.runtime = {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                        "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}
        defaults = diagnostic.default_configuration(SCRIPTS.parents[2])
        self.configuration = json.loads(json.dumps({"learner": defaults["learner"], "environment": defaults["environment"],
            "execution": {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
                          "report_readout": "none", "report_every_iterations": 10}}))
        self.installed = {"jaxtari_source": {"tree_digest": "synthetic"}, "gpu_libraries": {"/synthetic/libcudnn.so": {"sha256": "d" * 64}},
                          "packages": {name: "synthetic version" for name in training.PACKAGE_NAMES}}
        self.backend = {"gpu_name": "NVIDIA RTX A4500", "total_memory_bytes": 20470 * 2 ** 20,
            "driver_version": "580.159", "platform": "gpu", "platform_version": "synthetic CUDA platform",
            "installed_backend": self.installed}
        self.write("study_registry.json", {"synthetic": "already validated fifty-mechanism registry"})
        for name in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt"):
            self.write(name, "synthetic original campaign launch record\n")
        self.write("slurm/submitted_jobids_uva.txt", "")
        self.python = self.run / "venv/bin/python"
        self.python.parent.mkdir(parents=True)
        self.python.symlink_to(sys.executable)
        self.asset_root = self.run / "assets/original"
        self.asset_root.mkdir(parents=True)
        self.asset_files = {"sprites/synthetic.png": "e" * 64}
        manifest = self.write("asset_metadata/original_manifest.json", {"schema_version": 1,
            "source_commit": training.JAXTARI_COMMIT, "files": self.asset_files})
        self.asset_record = {"root": str(self.asset_root), "manifest": str(manifest), "sha256": final.file_digest(manifest)}
        self.asset_input = self.write("asset_metadata/asset_record.json", self.asset_record)
        self.producer_files = {name: b"# synthetic committed asset producer\n" for name in (
            "measure_selected.py", "diagnose_gpu_execution.py", "post_comparison_validation.py", "infrastructure_retry.py", "pregather_async_retry.py", "train_optimized.py")}
        self.producer_helpers = {"source_root": "/synthetic/producer", "source_commit": "f" * 40,
            "files": {final.PREFIX + name: hashlib.sha256(contents).hexdigest()
                      for name, contents in self.producer_files.items()}}
        context = {"confirmation_record": str(self.evidence("confirmation", 1)),
            "packing_records": [str(self.evidence("packing", copies)) for copies in (1, 2)],
            "packing_confirmation_records": [str(self.evidence("repeat", copies)) for copies in (1, 2)]}
        spec = {"study_registry": str(self.run / "study_registry.json"), "source_commit": self.commit,
                "contexts": {"jaguar03_rtx_a4500": context}}
        self.verified = {"kind": "final_verification", "source_commit": self.commit, "specification": spec,
            "specification_digest": final.digest(spec), "selected": {"jaguar03_rtx_a4500": {"configuration": self.configuration}},
            **{name: True for name in ("final_combination_correctness_passed", "final_confirmation_passed",
                                      "h100_packing_complete", "a4500_packing_complete", "eight_gpu_scaling_complete")}}
        self.verification = self.write("final_verification.json", self.verified)
        self.proposals = [self.proposal(index) for index in range(1, 9)]
        self.proposal_path = self.write("final_unit_inputs.json", {"schema_version": 1, "units": self.proposals})
        mock.patch.object(final, "frozen_helpers", return_value=self.helpers).start()
        mock.patch.object(final, "module", return_value=training).start()
        self.addCleanup(mock.patch.stopall)
        for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            self.addCleanup(signal.signal, signum, signal.getsignal(signum))

    def write(self, path, value):
        """Write only a temporary JSON/text artifact for a metadata fixture."""
        path = self.run / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value, indent=2) + "\n")
        return path

    def evidence(self, kind, copies):
        """Give selected confirmation/packing the same actual-style complete backend identity."""
        directory = self.run / "evidence" / f"{kind}_copies_{copies}"
        configuration = copy.deepcopy(self.configuration)
        configuration["learner"]["n_copies"] = copies
        self.write(directory / "hardware.json", {"hostname": "jaguar03", "gpu_uuid": "GPU-synthetic",
            "gpu_query": "GPU-synthetic, NVIDIA RTX A4500, 20470 MiB, 580.159"})
        self.write(directory / "installed_backend.json", self.installed)
        plan = {"kind": "selected_configuration_measurement", "mode": "confirmation" if kind == "confirmation" else "packing-point",
            "hardware_class": "jaguar03_rtx_a4500", "source_root": "/synthetic/numerical/source", "source_commit": self.commit,
            "configuration": configuration, "asset_record": self.asset_record, "helpers": self.producer_helpers,
            "inputs": {"asset_record": {"path": str(self.asset_input), "sha256": final.file_digest(self.asset_input)}}}
        plan_path = self.write(directory / "measurement_plan.json", plan)
        self.write(directory / "measurement_prepared.json", {"plan_sha256": final.file_digest(plan_path)})
        self.write(directory / "asset_receipt.json", {"root": str(self.asset_root), "manifest_sha256": self.asset_record["sha256"],
            "source_commit": training.JAXTARI_COMMIT, "file_count": len(self.asset_files), "tree_digest": final.digest(self.asset_files)})
        for name, contents in self.producer_files.items():
            self.write(directory / "code" / name, contents.decode())
        return self.write(directory / "result.json", {"configuration": configuration, "candidate": configuration,
            "helpers": self.producer_helpers,
            "hardware": final.read(directory / "hardware.json"),
            "installed_backend": self.installed, "loaded_gpu_libraries": {
                "platform": "gpu", "platform_version": "synthetic CUDA platform"}})

    def proposal(self, seed, asset_record=None):
        """Provide an explicit unit and seed instead of allowing a launcher to generate repeats."""
        spec = {"schema_version": 1, "source_root": "/synthetic/numerical/source", "source_commit": self.commit,
            "hardware_class": "jaguar03_rtx_a4500", "configuration": self.configuration, "copy_seeds": [seed],
            "runtime_environment": self.runtime, "final_verification": str(self.verification),
            "asset_record": self.asset_record if asset_record is None else asset_record}
        path = self.write(f"specifications/seed-{seed}.json", spec)
        admitted = {**spec, "validation": {"kind": "training_admission", "specification_digest": final.digest(spec),
            "source_commit": self.commit, "configuration_digest": final.digest(self.configuration), "selected_copies": 1,
            "admission_records": [{"path": str(path), "sha256": final.file_digest(path)},
                                  {"path": str(self.verification), "sha256": final.file_digest(self.verification)}],
            "evidence": [{"path": str(path), "sha256": final.file_digest(path)}
                         for path in sorted((self.run / "evidence").glob("*/*.json"))]}}
        receipt = self.write(f"admission/seed-{seed}.json", admitted)
        return {"unit_id": "optimized_pixel_ppo_random_network_distillation_seed-" + str(seed),
            "final_specification": str(path), "training_admission_receipt": str(receipt),
            "helper_root": self.helpers["source_root"], "helper_commit": self.helpers["source_commit"],
            "python": str(self.python), "checkpoint_seconds": 1200}

    def prepare(self, count=8):
        """Stage only metadata descriptors outside the model's output directories."""
        self.write(self.proposal_path, {"schema_version": 1, "units": self.proposals[:count]})
        return final.prepare(self.run, self.proposal_path)

    def submit(self, identifier="9001", gpus=8):
        """Mock sbatch while retaining real immediate ledger/intent publication."""
        files = {"launch_final_training.py": b"# synthetic committed launcher\n", "launch_queue.py": b"# synthetic committed resolver\n"}
        with mock.patch.object(final, "tool_files", return_value=(files, "f" * 40)), \
                mock.patch.object(final.launch_queue, "current_reservation", return_value=None), \
                mock.patch.object(final.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, identifier + ";uva\n", "")) as call:
            record = final.submit(self.run, gpus=gpus, walltime_seconds=43200, host_memory_gib_per_gpu=64)
        return record, call

    def checkpoint(self, unit, *, completed=False, job="9001", pid=555):
        """Represent actual result/identity/sidecar bytes without creating or unpickling arrays."""
        directory = self.run / "data" / unit["unit_id"]
        iteration = final.TARGET_UPDATES if completed else 100
        identity = {"source_root": unit["numerical_source"]["root"], "source_commit": self.commit,
            "hardware_class": "jaguar03_rtx_a4500", "helpers": self.helpers, "configuration": self.configuration,
            "copy_seeds": unit["copy_seeds"], "runtime_environment": self.runtime,
            "backend": {"driver_version": self.backend["driver_version"], "packages": self.installed["packages"],
                "installed_gpu_libraries": self.installed["gpu_libraries"], "jaxtari_source": self.installed["jaxtari_source"]}}
        self.write(directory / "training_identity.json", identity)
        payload = self.write(directory / "checkpoints/latest.pkl", "synthetic complete checkpoint bytes")
        metadata = {"kind": "optimized_training_checkpoint", "source_commit": self.commit,
            "checkpoint_path": str(payload), "checkpoint_sha256": final.file_digest(payload),
            "configuration_digest": unit["configuration_digest"], "identity_digest": final.digest(identity),
            "copy_seeds": unit["copy_seeds"], "iteration": iteration, "target_updates": final.TARGET_UPDATES,
            "steps_per_copy": iteration * 128 * 128}
        self.write(directory / "checkpoints/latest.metadata.json", metadata)
        result = {"completed": completed, "update": iteration, "target_updates": final.TARGET_UPDATES,
            "requested_steps_per_copy": 2_000_000_000, "steps_per_copy": iteration * 128 * 128,
            "copies": 1, "copy_seeds": unit["copy_seeds"], "identity_digest": final.digest(identity),
            "attempt": 1, "stop_reason": "completed" if completed else "maximum_seconds"}
        self.write(directory / "result.json", result)
        self.write(directory / "progress.json", {"attempt": 1, "state": "completed" if completed else "checkpointed"})
        self.write(directory / "hardware.json", {"slurm_job_id": job, "pid": pid, "cuda_visible_devices": "GPU-synthetic"})
        return metadata

    def completion(self, unit, metadata):
        """Represent the completion validator's explicit full-schedule receipt."""
        return {"kind": "training_completion", "completed": True, "source_commit": self.commit,
            "hardware_class": "jaguar03_rtx_a4500", "copy_seeds": unit["copy_seeds"], "updates": final.TARGET_UPDATES,
            "configuration_digest": unit["configuration_digest"], "checkpoint_sha256": metadata["checkpoint_sha256"], "evidence": []}

    def test_prepare_preserves_explicit_venv_and_does_not_touch_training_outputs(self):
        """Keep the venv symlink path and original helper/seed identity on an identical retry."""
        units = self.prepare()
        self.assertEqual(self.prepare(), units)
        self.assertEqual(units[0]["python"], str(self.python))
        self.assertNotEqual(units[0]["python"], str(self.python.resolve()))
        self.assertFalse((self.run / "data").exists())
        manifest = final.read(self.queue / "seed_manifest.json")
        self.assertEqual(len(manifest["units"]), 8)
        self.assertEqual(units[0]["target_backend"], self.backend)
        self.assertFalse(any(name in sys.modules for name in ("jax", "numpy", "torch")))

    def test_prepared_monitor_policy_allows_only_identical_existing_a4500_units(self):
        """Registration cannot add unlisted units while generic workers use a frozen policy roster."""
        units = self.prepare(count=1)
        paths = {unit["unit_id"]: self.queue / "units" / (unit["unit_id"] + ".json") for unit in units}
        policy = {"kind": "uva_final_training_monitor_policy", "units": {
            name: {"path": str(path), "sha256": final.file_digest(path)} for name, path in paths.items()}}
        self.write(self.queue / "monitor_policy.json", policy)
        before = (self.queue / "seed_manifest.json").read_bytes()
        self.assertEqual(self.prepare(count=1), units)
        with self.assertRaisesRegex(ValueError, "only identical existing"):
            self.prepare(count=2)
        self.assertEqual((self.queue / "seed_manifest.json").read_bytes(), before)
        self.assertEqual(len(list((self.queue / "units").glob("*.json"))), 1)
        original = dict(self.proposals[0])
        self.proposals[0] = {**original, "checkpoint_seconds": 600}
        with self.assertRaisesRegex(ValueError, "only identical existing"):
            self.prepare(count=1)
        self.proposals[0] = original
        path = paths[units[0]["unit_id"]]
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "only identical existing"):
            self.prepare(count=1)

    def test_a4500_monitor_policy_does_not_block_a_separately_prepared_h100_seed_block(self):
        """The shared registration layer accepts a new disjoint H100 unit without revising the A4500 policy."""
        units = self.prepare(count=1)
        unit = units[0]
        path = self.queue / "units" / (unit["unit_id"] + ".json")
        policy = {"kind": "uva_final_training_monitor_policy", "units": {
            unit["unit_id"]: {"path": str(path), "sha256": final.file_digest(path)}}}
        policy_path = self.write(self.queue / "monitor_policy.json", policy)
        original_policy = policy_path.read_bytes()
        h100 = {**unit, "unit_id": "explicit_h100_policy_seed_9001", "hardware_class": "h100", "copy_seeds": [9001]}
        proposal = self.write("prepared_h100_registration.json", {"schema_version": 1, "units": [{"unit_id": h100["unit_id"]}]})
        # The H100 producer owns its admission; this test exercises only shared registration and the policy lock.
        with mock.patch.object(final, "prepared_unit", return_value=h100):
            self.assertEqual(final.prepare(self.run, proposal), [h100])
            self.assertEqual(final.prepare(self.run, proposal), [h100])
        self.assertEqual(policy_path.read_bytes(), original_policy)
        seeds = final.read(self.queue / "seed_manifest.json")["units"]
        self.assertEqual(seeds[h100["unit_id"]]["copy_seeds"], [9001])
        self.assertEqual(seeds[unit["unit_id"]]["copy_seeds"], [1])
        with self.assertRaisesRegex(ValueError, "only identical existing"):
            self.prepare(count=2)
        changed = {**h100, "checkpoint_seconds": 600}
        with mock.patch.object(final, "prepared_unit", return_value=changed), self.assertRaisesRegex(ValueError, "cannot change"):
            final.prepare(self.run, proposal)

    def test_assets_allow_a_different_operational_root_only_with_the_complete_same_manifest(self):
        """Metadata staging accepts an exact copied manifest without reading sprite files on login."""
        copied = self.run / "asset_metadata/copied_manifest.json"
        shutil.copyfile(self.asset_record["manifest"], copied)
        record = {"root": str(self.run / "assets/another_operational_root"), "manifest": str(copied),
                  "sha256": final.file_digest(copied)}
        unit = final.prepared_unit(self.run, self.proposal(1, record))
        self.assertEqual(unit["assets"]["identity"]["file_count"], 1)
        self.assertEqual(unit["assets"]["identity"]["tree_digest"], final.digest(self.asset_files))
        self.assertFalse((self.asset_root / "sprites/synthetic.png").exists())
        with self.assertRaisesRegex(ValueError, "outside their operational asset roots"):
            final.prepared_unit(self.run, self.proposal(1, {**record, "root": str(copied.parent)}))

    def test_changed_asset_manifest_source_tree_or_actual_receipt_is_rejected(self):
        """A new self-consistent asset record cannot replace the selected actual game assets."""
        for change in ("source", "tree", "file_count"):
            manifest = {"schema_version": 1, "source_commit": training.JAXTARI_COMMIT, "files": dict(self.asset_files)}
            if change == "source":
                manifest["source_commit"] = "f" * 40
            elif change == "tree":
                manifest["files"]["sprites/synthetic.png"] = "f" * 64
            else:
                manifest["files"]["sprites/additional.png"] = "f" * 64
            path = self.write("asset_metadata/changed_" + change + ".json", manifest)
            record = {"root": str(self.asset_root), "manifest": str(path), "sha256": final.file_digest(path)}
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "asset.*source|assets differ"):
                final.prepared_unit(self.run, self.proposal(1, record))
        actual = Path(self.verified["specification"]["contexts"]["jaguar03_rtx_a4500"]["confirmation_record"]).parent / "asset_receipt.json"
        receipt = final.read(actual)
        receipt["tree_digest"] = "f" * 64
        self.write(actual, receipt)
        with self.assertRaisesRegex(ValueError, "actual manifest/source/tree/file-count"):
            final.prepared_unit(self.run, self.proposal(1))

    def test_asset_proof_hashes_are_rechecked_on_continuation(self):
        """Preserve the selected plan/actual receipt/manifest bytes after a unit is registered."""
        unit = self.prepare(1)[0]
        path = Path(self.verified["specification"]["contexts"]["jaguar03_rtx_a4500"]["confirmation_record"]).parent / "asset_receipt.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "admission/evidence file changed"):
            final.verify_unit(unit)

    def test_global_manifest_rejects_h100_overlap_and_changed_unit_identity(self):
        """Eventual remote training must reserve a disjoint global block rather than restart at seed one."""
        units = self.prepare()
        h100 = {"unit_id": "optimized_h100_explicit_seed_block", "hardware_class": "h100",
                "copy_seeds": [1, 2], "configuration_digest": "a" * 64}
        with self.assertRaisesRegex(ValueError, "cannot repeat"):
            final.reserve_seed_blocks(self.queue, [h100])
        h100["copy_seeds"] = [9, 10]
        final.reserve_seed_blocks(self.queue, [h100])
        self.assertEqual(len(final.read(self.queue / "seed_manifest.json")["units"]), 9)
        changed = {**units[0], "copy_seeds": [20]}
        with self.assertRaisesRegex(ValueError, "cannot change"):
            final.reserve_seed_blocks(self.queue, [changed])

    def test_shared_queue_h100_states_and_job_ids_are_not_queried_as_uva_work(self):
        """A foreign H100 job can coexist in the manifest without a local query or state change."""
        self.prepare()
        h100 = {"unit_id": "explicit_h100_seed-9", "hardware_class": "h100", "copy_seeds": [9],
                "configuration_digest": "d" * 64}
        final.reserve_seed_blocks(self.queue, [h100])
        self.write(self.queue / "units" / (h100["unit_id"] + ".json"), h100)
        foreign = {"state": "running", "unit_id": h100["unit_id"], "last_job_id": "123", "last_job_cluster": "aces"}
        self.write(self.queue / "status" / (h100["unit_id"] + ".json"), foreign)
        self.write(self.queue / "jobs/aces_job-123.json", {"cluster": "aces", "job_id": "123"})
        with mock.patch.object(final.subprocess, "check_output") as query:
            snapshot = final.status(self.run)
        query.assert_not_called()
        self.assertEqual(snapshot["counts"]["ready"], 8)
        self.assertEqual(final.read(self.queue / "status" / (h100["unit_id"] + ".json")), foreign)

    def test_backend_proof_requires_actual_memory_packages_and_consistency(self):
        """A hardware class label cannot substitute for the selected model/memory/backend evidence."""
        path = Path(self.verified["specification"]["contexts"]["jaguar03_rtx_a4500"]["confirmation_record"])
        hardware = final.read(path.parent / "hardware.json")
        hardware["gpu_query"] = "GPU-synthetic, NVIDIA RTX A4500, 10240 MiB, 580.159"
        self.write(path.parent / "hardware.json", hardware)
        with self.assertRaisesRegex(ValueError, "hardware differs"):
            final.expected_backend(self.verified)
        record = final.read(path)
        record["hardware"] = hardware
        self.write(path, record)
        with self.assertRaisesRegex(ValueError, "different hardware/backends"):
            final.expected_backend(self.verified)
        result = final.read(path)
        result.pop("installed_backend")
        self.write(path, result)
        with self.assertRaisesRegex(ValueError, "complete selected hardware/backend"):
            final.expected_backend(self.verified)

    def test_eight_gpus_use_one_generic_job_and_ids_are_durable_before_release(self):
        """One node job launches generic slots and records the exact job before its release record exists."""
        self.prepare()
        original = final.atomic
        def publish(path, value, **kwargs):
            """Inspect the primary ledger at the exact durable worker-release boundary."""
            if Path(path).parent == self.queue / "jobs":
                self.assertIn("9001", (self.run / "slurm/submitted_jobids_uva.txt").read_text().splitlines())
                self.assertIn("9001\tfinal-training-worker\t", (self.run / "slurm/submissions.tsv").read_text())
            return original(path, value, **kwargs)
        with mock.patch.object(final, "atomic", side_effect=publish):
            record, call = self.submit()
        self.assertEqual(call.call_count, 1)
        arguments = call.call_args.args[0]
        for value in ("--nodes=1", "--ntasks=1", "--gres=gpu:nvidia_rtx_a4500:8", "--mem=512G",
                      "--cpus-per-task=64", "--signal=B:USR1@1800", "--no-requeue"):
            self.assertIn(value, arguments)
        self.assertNotIn(self.proposals[0]["unit_id"], " ".join(arguments))
        self.assertEqual(final.read(record["plan"])["gpus"], 8)

    def test_status_is_metadata_only_and_changed_raw_bytes_block_submission(self):
        """Refresh ready/held counts on login and refuse stale raw evidence before requesting any GPU."""
        self.prepare()
        self.assertEqual(final.status(self.run)["counts"]["ready"], 8)
        path = Path(self.verified["specification"]["contexts"]["jaguar03_rtx_a4500"]["confirmation_record"])
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "original admission/evidence file changed"):
            self.submit()

    def test_active_unknown_or_unresolved_job_prevents_duplicate_submission(self):
        """No additional generic job is submitted while an earlier one may still execute."""
        self.prepare()
        self.submit()
        for state in ("RUNNING", "PENDING"):
            with mock.patch.object(final, "job_states", return_value={"9001": state}):
                with self.assertRaisesRegex(ValueError, "active or pending"):
                    self.submit("9002")
        with mock.patch.object(final, "job_states", side_effect=ValueError("unknown exact job")):
            with self.assertRaisesRegex(ValueError, "unknown exact job"):
                self.submit("9002")
        self.write(self.queue / "submissions/unresolved.json", {"state": "submitting", "cluster": "uva"})
        with self.assertRaisesRegex(ValueError, "unresolved submission"):
            self.submit("9002")

    def test_resume_requires_terminal_job_actual_checkpoint_and_unchanged_helpers(self):
        """Only a finished scheduler attempt with a valid complete checkpoint can become ready again."""
        unit = self.prepare(1)[0]
        self.submit(gpus=1)
        self.checkpoint(unit)
        status = self.queue / "status" / (unit["unit_id"] + ".json")
        final.atomic(status, {"state": "checkpointed", "unit_id": unit["unit_id"], "last_job_id": "9001", "last_job_cluster": "uva"})
        final.refresh(self.run, self.queue, {"9001": "RUNNING"})
        self.assertEqual(final.read(status)["state"], "checkpointed")
        final.refresh(self.run, self.queue, {"9001": "COMPLETED"})
        self.assertEqual(final.read(status)["state"], "ready")
        with mock.patch.object(final, "frozen_helpers", return_value={**self.helpers, "source_commit": "e" * 40}):
            with self.assertRaisesRegex(ValueError, "helper identity changed"):
                final.verify_unit(unit)

    def test_abrupt_scheduler_stop_resumes_verified_periodic_checkpoint_without_a_graceful_result(self):
        """Only explicit interruption states can resume a real saved state with still-running progress."""
        unit = self.prepare(1)[0]
        self.checkpoint(unit)
        directory = self.run / "data" / unit["unit_id"]
        (directory / "result.json").unlink()
        self.write(directory / "progress.json", {"attempt": 1, "state": "running", "update": 101})
        path = self.queue / "status" / (unit["unit_id"] + ".json")
        original = {"state": "running", "unit_id": unit["unit_id"], "last_job_id": "9001", "last_job_cluster": "uva"}
        for state in ("TIMEOUT", "PREEMPTED", "NODE_FAIL"):
            final.atomic(path, original)
            final.refresh(self.run, self.queue, {"9001": state})
            self.assertEqual(final.read(path)["state"], "ready")
            self.assertIn(state, final.read(path)["recovery_scope"])
            self.assertFalse((self.queue / "completion" / path.name).exists())
        final.atomic(path, original)
        final.refresh(self.run, self.queue, {"9001": "FAILED"})
        self.assertEqual(final.read(path)["state"], "held")
        self.write(directory / "attempts/attempt-0001/failure.json", {"attempt": 1})
        final.atomic(path, original)
        final.refresh(self.run, self.queue, {"9001": "PREEMPTED"})
        self.assertEqual(final.read(path)["state"], "held")

    def test_failed_unit_and_changed_checkpoint_are_held_for_diagnosis(self):
        """A checkpoint file alone cannot erase an application failure or changed state bytes."""
        unit = self.prepare(1)[0]
        self.checkpoint(unit)
        status = self.queue / "status" / (unit["unit_id"] + ".json")
        final.atomic(status, {"state": "checkpointed", "unit_id": unit["unit_id"], "last_job_id": "9001", "last_job_cluster": "uva"})
        (self.run / "data" / unit["unit_id"] / "checkpoints/latest.pkl").write_text("changed")
        final.refresh(self.run, self.queue, {"9001": "TIMEOUT"})
        self.assertEqual(final.read(status)["state"], "held")
        self.assertIn("bytes changed", final.read(status)["reason"])
        self.checkpoint(unit)
        self.write(self.run / "data" / unit["unit_id"] / "failure.json", {"attempt": 1, "error": "synthetic application error"})
        with self.assertRaisesRegex(ValueError, "requires diagnosis"):
            final.checkpoint_record(self.run, unit)
        self.write(self.run / "data" / unit["unit_id"] / "progress.json", {"attempt": 2, "state": "checkpointed"})
        result_path = self.run / "data" / unit["unit_id"] / "result.json"
        result = final.read(result_path)
        result["attempt"] = 2
        self.write(result_path, result)
        final.checkpoint_record(self.run, unit)
        final.refresh(self.run, self.queue, {"9001": "COMPLETED"})
        self.assertEqual(final.read(status)["state"], "held")
        self.assertTrue((self.run / "data" / unit["unit_id"] / "failure.json").exists())
        self.write(self.run / "data" / unit["unit_id"] / "attempts/attempt-0002/failure.json", {"attempt": 2})
        with self.assertRaisesRegex(ValueError, "current final attempt failed"):
            final.checkpoint_record(self.run, unit)

    def test_completed_boolean_requires_compute_completion_receipt(self):
        """Never infer completion from result.completed without validated final checkpoint evidence."""
        unit = self.prepare(1)[0]
        metadata = self.checkpoint(unit, completed=True)
        status = self.queue / "status" / (unit["unit_id"] + ".json")
        final.atomic(status, {"state": "running", "unit_id": unit["unit_id"], "last_job_id": "9001", "last_job_cluster": "uva"})
        final.refresh(self.run, self.queue, {"9001": "COMPLETED"})
        self.assertEqual(final.read(status)["state"], "held")
        final.atomic(status, {"state": "running", "unit_id": unit["unit_id"], "last_job_id": "9001", "last_job_cluster": "uva"})
        self.write(self.queue / "completion" / status.name, self.completion(unit, metadata))
        final.refresh(self.run, self.queue, {"9001": "COMPLETED"})
        self.assertEqual(final.read(status)["state"], "completed")

    def test_scheduler_queries_are_only_exact_uva_ids(self):
        """Foreign cluster ledgers or missing state cannot authorize a continuation."""
        self.write("slurm/submitted_jobids_aces.txt", "123\n")
        with mock.patch.object(final.subprocess, "check_output") as call:
            with self.assertRaisesRegex(ValueError, "exact UVA"):
                final.job_states(self.run, ["123"])
            call.assert_not_called()
        self.write("slurm/submitted_jobids_uva.txt", "123\n")
        with mock.patch.object(final.subprocess, "check_output", return_value="123|COMPLETED|\n") as call, \
                mock.patch.object(final, "active_owned_jobs", return_value=({}, None)):
            self.assertEqual(final.job_states(self.run, ["123"]), {"123": "COMPLETED"})
            self.assertEqual(call.call_args.args[0][3], "123")
            self.assertNotIn("-u", call.call_args.args[0])
        with mock.patch.object(final.subprocess, "check_output", return_value=""), \
                mock.patch.object(final, "active_owned_jobs", return_value=({}, None)):
            with self.assertRaisesRegex(ValueError, "unknown"):
                final.job_states(self.run, ["123"])

    def test_backend_probe_compares_full_selected_identity_without_running_jax_in_tests(self):
        """A different memory size or installed package map rejects before numerical training."""
        unit = self.prepare(1)[0]
        observed = {"device_kind": "NVIDIA RTX A4500", **{key: self.backend[key] for key in ("platform", "platform_version", "installed_backend")}}
        with mock.patch.object(final.subprocess, "check_output", side_effect=[json.dumps(observed), "GPU-synthetic, NVIDIA RTX A4500, 20470, 580.159\n"]):
            final.probe(unit, "GPU-synthetic")
        with mock.patch.object(final.subprocess, "check_output", side_effect=[json.dumps(observed), "GPU-synthetic, NVIDIA RTX A4500, 10240, 580.159\n"]):
            with self.assertRaisesRegex(ValueError, "model/memory/driver/backend"):
                final.probe(unit, "GPU-synthetic")

    def test_worker_runs_full_admission_and_completion_inside_owned_compute(self):
        """Mock only the process/GPU boundaries while retaining actual checkpoint and queue checks."""
        unit = self.prepare(1)[0]
        self.write("slurm/submitted_jobids_uva.txt", "9001\n")
        deadline = time.time() + 10000
        self.write(self.queue / "allocations/uva_job-9001.json", {"job_id": "9001", "gpu_uuids": ["GPU-synthetic"], "deadline": deadline})
        plan = self.write("synthetic_job_plan.json", {"campaign_run_dir": str(self.run), "target_backend": self.backend,
            "checkpoint_reserve_seconds": 1800, "cpus_per_gpu": 8})
        admitted = final.read(unit["training_admission_receipt"]["path"])
        metadata = self.checkpoint(unit, completed=True)
        validator = types.SimpleNamespace(validate_training_admission=mock.Mock(return_value=admitted),
            validate_training_completion=mock.Mock(return_value=self.completion(unit, metadata)))
        process = types.SimpleNamespace(pid=555, wait=lambda: 0, poll=lambda: 0)
        def probe(unit, gpu):
            """Check environment isolation before either pinned-Python child could start."""
            self.assertEqual(os.environ["PYTHONNOUSERSITE"], "1")
            self.assertEqual(os.environ["PYTHONUNBUFFERED"], "1")
        lock = Path("/tmp/sl5nw_atari_pixel_benchmark_GPU-synthetic.lock")
        self.addCleanup(lambda: lock.unlink(missing_ok=True))
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-synthetic"}), \
                mock.patch.object(final, "owned_compute", return_value="9001") as placement, \
                mock.patch.object(final, "module", return_value=validator), mock.patch.object(final, "probe", side_effect=probe), \
                mock.patch.object(final, "process_birth", return_value="synthetic birth"), \
                mock.patch.object(final.subprocess, "check_output", return_value=""), \
                mock.patch.object(final.subprocess, "Popen", return_value=process) as spawn:
            final.worker(plan, "GPU-synthetic", deadline)
        placement.assert_called_once_with(self.run)
        validator.validate_training_admission.assert_called_once_with(unit["final_specification"]["path"])
        validator.validate_training_completion.assert_called_once()
        self.assertEqual(final.read(self.queue / "status" / (unit["unit_id"] + ".json"))["state"], "completed")
        command = spawn.call_args.args[0]
        self.assertEqual(command[0], str(self.python))
        self.assertEqual(command[command.index("--checkpoint-seconds") + 1], "1200")
        self.assertLess(float(command[command.index("--maximum-seconds") + 1]), 8200)

    def test_worker_metadata_failure_reaps_owned_child_before_card_lock_release(self):
        """A failed child-identity write cannot leave numerical training running outside its worker."""
        import fcntl
        unit = self.prepare(1)[0]
        deadline = time.time() + 10000
        self.write(self.queue / "allocations/uva_job-9001.json", {"job_id": "9001", "gpu_uuids": ["GPU-cleanup"], "deadline": deadline})
        plan = self.write("cleanup_job_plan.json", {"campaign_run_dir": str(self.run), "target_backend": self.backend,
            "checkpoint_reserve_seconds": 1800, "cpus_per_gpu": 8})
        validator = types.SimpleNamespace(validate_training_admission=mock.Mock(return_value=final.read(unit["training_admission_receipt"]["path"])))
        events, active = [], {"running": True}
        lock = Path("/tmp/sl5nw_atari_pixel_benchmark_GPU-cleanup.lock")
        self.addCleanup(lambda: lock.unlink(missing_ok=True))
        def wait(**kwargs):
            """Verify graceful cleanup finishes while the original physical-card lock is still held."""
            with lock.open("a+") as other:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            events.append("wait")
            active["running"] = False
            return 0
        child = types.SimpleNamespace(pid=555, poll=lambda: None if active["running"] else 0,
            send_signal=lambda signum: events.append(("signal", signum)), wait=wait)
        original = final.atomic
        def publish(path, value, **kwargs):
            """Fail exactly after Popen succeeds but before the worker can reach child.wait."""
            if value.get("state") == "running" and "child_pid" in value:
                raise OSError("synthetic child metadata publication failure")
            if value.get("state") == "held":
                events.append("held")
            return original(path, value, **kwargs)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-cleanup"}), \
                mock.patch.object(final, "owned_compute", return_value="9001"), mock.patch.object(final, "module", return_value=validator), \
                mock.patch.object(final, "probe"), mock.patch.object(final, "process_birth", return_value="birth"), \
                mock.patch.object(final.subprocess, "check_output", return_value=""), \
                mock.patch.object(final.subprocess, "Popen", return_value=child), mock.patch.object(final, "atomic", side_effect=publish):
            with self.assertRaisesRegex(OSError, "publication failure"):
                final.worker(plan, "GPU-cleanup", deadline)
        self.assertEqual(events, [("signal", signal.SIGTERM), "wait", "held"])
        self.assertFalse(active["running"])

    def test_worker_wait_exception_stops_and_reaps_its_original_child(self):
        """An exception from the first wait must still clean the live numerical process before returning."""
        unit = self.prepare(1)[0]
        deadline = time.time() + 10000
        self.write(self.queue / "allocations/uva_job-9001.json", {"job_id": "9001", "gpu_uuids": ["GPU-wait-cleanup"], "deadline": deadline})
        plan = self.write("wait_cleanup_plan.json", {"campaign_run_dir": str(self.run), "target_backend": self.backend,
            "checkpoint_reserve_seconds": 1800, "cpus_per_gpu": 8})
        validator = types.SimpleNamespace(validate_training_admission=mock.Mock(return_value=final.read(unit["training_admission_receipt"]["path"])))
        events, active = [], {"running": True}
        def wait(**kwargs):
            """Fail the foreground wait once, then allow graceful cleanup to reap this exact child."""
            if not kwargs:
                events.append("failed wait")
                raise OSError("synthetic wait failure")
            events.append("cleanup wait")
            active["running"] = False
            return 0
        child = types.SimpleNamespace(pid=555, poll=lambda: None if active["running"] else 0,
            send_signal=lambda signum: events.append(("signal", signum)), wait=wait)
        lock = Path("/tmp/sl5nw_atari_pixel_benchmark_GPU-wait-cleanup.lock")
        self.addCleanup(lambda: lock.unlink(missing_ok=True))
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-wait-cleanup"}), \
                mock.patch.object(final, "owned_compute", return_value="9001"), mock.patch.object(final, "module", return_value=validator), \
                mock.patch.object(final, "probe"), mock.patch.object(final, "process_birth", return_value="birth"), \
                mock.patch.object(final.subprocess, "check_output", return_value=""), \
                mock.patch.object(final.subprocess, "Popen", return_value=child):
            with self.assertRaisesRegex(OSError, "wait failure"):
                final.worker(plan, "GPU-wait-cleanup", deadline)
        self.assertEqual(events, ["failed wait", ("signal", signal.SIGTERM), "cleanup wait"])
        self.assertFalse(active["running"])

    def test_partial_node_worker_startup_cleans_already_started_workers(self):
        """If the second GPU worker cannot start, the first original child is stopped and reaped."""
        self.prepare()
        record, _ = self.submit(gpus=2)
        events, active = [], {"running": True}
        def wait(**kwargs):
            """Finish only this exact synthetic Popen-owned worker."""
            events.append("wait")
            active["running"] = False
            return 0
        child = types.SimpleNamespace(poll=lambda: None if active["running"] else 0,
            send_signal=lambda signum: events.append(("signal", signum)), wait=wait)
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "9001"}), \
                mock.patch.object(final, "owned_compute", return_value="9001"), \
                mock.patch.object(final, "allocated_gpus", return_value=["GPU-first", "GPU-second"]), \
                mock.patch.object(final, "end_time", return_value=time.time() + 10000), \
                mock.patch.object(final.subprocess, "Popen", side_effect=[child, OSError("synthetic second startup failure")]):
            with self.assertRaisesRegex(OSError, "second startup failure"):
                final.run_job(record["plan"])
        self.assertEqual(events, [("signal", signal.SIGTERM), "wait"])

    def test_job_does_not_check_allocation_or_start_before_durable_release(self):
        """An immediately started Slurm job waits for its captured ID and complete immutable plan record."""
        path = self.write("unreleased_plan.json", {"campaign_run_dir": str(self.run)})
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "9001"}), \
                mock.patch.object(final.time, "monotonic", side_effect=[0., 0., 121.]), \
                mock.patch.object(final.time, "sleep"), mock.patch.object(final, "owned_compute") as placement:
            with self.assertRaisesRegex(ValueError, "durably recorded"):
                final.run_job(path)
        placement.assert_not_called()


class AllocatedGpuTests(unittest.TestCase):
    """Check safe physical allocation for a complete multi-GPU node job."""

    def test_multi_gpu_minor_mapping_rejects_conflicting_or_ambiguous_sets(self):
        """Visible indices zero/one must still match physical Slurm device minors five/six."""
        rows = "0, GPU-five, NVIDIA RTX A4500\n1, GPU-six, NVIDIA RTX A4500\n"
        xml = "<nvidia_smi_log>" + "".join(f"<gpu><uuid>GPU-{name}</uuid><minor_number>{minor}</minor_number></gpu>"
                                         for name, minor in (("five", 5), ("six", 6))) + "</nvidia_smi_log>"
        with mock.patch.dict(os.environ, {"SLURM_JOB_GPUS": "5,6", "SLURM_STEP_GPUS": "5,6", "CUDA_VISIBLE_DEVICES": "0,1"}), \
                mock.patch.object(final.subprocess, "check_output", side_effect=[rows, xml]):
            self.assertEqual(final.allocated_gpus(2), ["GPU-five", "GPU-six"])
        with mock.patch.dict(os.environ, {"SLURM_JOB_GPUS": "0,1", "SLURM_STEP_GPUS": "0,1", "CUDA_VISIBLE_DEVICES": "0,1"}), \
                mock.patch.object(final.subprocess, "check_output", side_effect=[rows, xml]):
            with self.assertRaisesRegex(ValueError, "physical allocation differs"):
                final.allocated_gpus(2)
        with mock.patch.object(final.launch_queue, "allocated_gpu_uuid", return_value="GPU-five"):
            self.assertEqual(final.allocated_gpus(1), ["GPU-five"])

    def test_owned_compute_rejects_foreign_cluster_only_job_id(self):
        """Matching numeric IDs in an ACES ledger cannot authorize a UVA GPU process."""
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "slurm").mkdir()
            (run / "slurm/submitted_jobids_uva.txt").write_text("")
            (run / "slurm/submitted_jobids_aces.txt").write_text("123\n")
            original = Path.read_text
            def read(path, *args, **kwargs):
                """Provide a real-shaped owned cgroup while preserving exact cluster ledger reads."""
                return "0::/slurm/job_123/step_batch\n" if path == Path("/proc/self/cgroup") else original(path, *args, **kwargs)
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "123"}), mock.patch.object(Path, "read_text", read), \
                    mock.patch.object(final.socket, "gethostname", return_value="jaguar03"):
                with self.assertRaisesRegex(ValueError, "owned jaguar03 UVA"):
                    final.owned_compute(run)
                (run / "slurm/submitted_jobids_uva.txt").write_text("123\n")
                self.assertEqual(final.owned_compute(run), "123")

    def test_group_cleanup_last_resort_targets_only_an_owned_unreaped_process_group(self):
        """Allow checkpoint time, then stop only the new session created for this original worker."""
        child = mock.Mock(pid=123)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("synthetic owned worker", 1200), 0]
        with mock.patch.object(final.os, "killpg") as kill:
            final.cleanup_children([child], time.time() + 10000, process_groups=True)
        child.send_signal.assert_called_once_with(signal.SIGTERM)
        kill.assert_called_once_with(123, signal.SIGKILL)
        self.assertEqual(child.wait.call_args_list[-1].kwargs, {"timeout": 30})


class FrozenHelperTests(unittest.TestCase):
    """Require a real detached helper root instead of accepting an archive under a Git ancestor."""

    def test_old_pinned_callers_keep_their_original_helper_identity(self):
        """Adding a future lazy dependency does not rewrite receipts for callers that never used it."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            h100_names = {"h100_nvl_slurm_queue.py", "h100_nvl_slurm_case.py", "h100_nvl_slurm_evidence.py", "h100_nvl_slurm_site_validation.py"}
            qualification_names = {"h100_nvl_slurm_qualification.py", "h100_nvl_allocation_observation.py"}
            old_names = tuple(name for name in final.TRAIN_HELPERS if name != "pregather_async_retry.py" and name not in h100_names | qualification_names)
            for name in old_names:
                path = root / final.PREFIX / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# old pinned metadata helper\n")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            def commit():
                """Record the two source generations without importing either fixture."""
                subprocess.run(["git", "-C", str(root), "add", "."], check=True)
                subprocess.run(["git", "-C", str(root), "-c", "user.name=Metadata dependency test",
                    "-c", "user.email=test@example.invalid", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Record helper dependency source"], check=True)
                value = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
                subprocess.run(["git", "-C", str(root), "checkout", "-q", "--detach", value], check=True)
                return value
            old_commit = commit()
            expected = {"source_root": str(root), "source_commit": old_commit,
                "files": {final.PREFIX + name: final.file_digest(root / final.PREFIX / name) for name in old_names}}
            self.assertEqual(final.frozen_helpers(root, old_commit), expected)
            (root / final.PREFIX / "post_comparison_validation.py").write_text("PREGATHER_ASYNC_RETRY_MODULE = 'pregather_async_retry.py'\n")
            (root / final.PREFIX / "pregather_async_retry.py").write_text("# future adjacent helper\n")
            current = final.frozen_helpers(root, commit())
            self.assertEqual(set(current["files"]), {final.PREFIX + name for name in final.TRAIN_HELPERS if name not in h100_names | qualification_names})
            # A new H100 caller must pin all four adjacent readers rather than reusing the older receipt.
            caller = root / final.PREFIX / "post_comparison_validation.py"
            caller.write_text(caller.read_text() + "H100_NVL_SLURM_QUEUE_MODULE = 'h100_nvl_slurm_queue.py'\n")
            missing = sorted(h100_names)[-1]
            for name in h100_names - {missing}:
                (root / final.PREFIX / name).write_text("# committed H100 metadata reader\n")
            with self.assertRaises((ValueError, subprocess.CalledProcessError, FileNotFoundError)):
                final.frozen_helpers(root, commit())
            (root / final.PREFIX / missing).write_text("# final required H100 metadata reader\n")
            complete = final.frozen_helpers(root, commit())
            self.assertEqual(set(complete["files"]), {final.PREFIX + name for name in final.TRAIN_HELPERS if name not in qualification_names})
            # A declared qualification profile adds both adjacent files only for the new pinned caller.
            queue = root / final.PREFIX / "h100_nvl_slurm_queue.py"
            queue.write_text(queue.read_text() + "H100_NVL_SLURM_QUALIFICATION_MODULE = 'h100_nvl_slurm_qualification.py'\n")
            with self.assertRaises((ValueError, subprocess.CalledProcessError, FileNotFoundError)):
                final.frozen_helpers(root, commit())
            (root / final.PREFIX / "h100_nvl_slurm_qualification.py").write_text("# committed qualification metadata\n")
            with self.assertRaises((ValueError, subprocess.CalledProcessError, FileNotFoundError)):
                final.frozen_helpers(root, commit())
            (root / final.PREFIX / "h100_nvl_allocation_observation.py").write_text("# committed hardware observation metadata\n")
            qualified = final.frozen_helpers(root, commit())
            self.assertEqual(set(qualified["files"]), {final.PREFIX + name for name in final.TRAIN_HELPERS})

    def test_exact_git_top_is_required_for_helpers(self):
        """A nested ignored copy cannot borrow the containing checkout's committed identity."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".gitignore").write_text("archive/\n")
            for name in final.TRAIN_HELPERS:
                path = root / final.PREFIX / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# synthetic helper bytes\n")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Synthetic final launcher test",
                "-c", "user.email=synthetic@example.invalid", "-c", "core.hooksPath=/dev/null",
                "commit", "-qm", "Create synthetic helper source"], check=True)
            commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            subprocess.run(["git", "-C", str(root), "checkout", "-q", "--detach"], check=True)
            self.assertEqual(final.frozen_helpers(root, commit)["source_root"], str(root))
            archive = root / "archive/tree"
            for name in final.TRAIN_HELPERS:
                target = archive / final.PREFIX / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / final.PREFIX / name, target)
            with self.assertRaisesRegex(ValueError, "exact clean detached commit"):
                final.frozen_helpers(archive, commit)


if __name__ == "__main__":
    unittest.main()
