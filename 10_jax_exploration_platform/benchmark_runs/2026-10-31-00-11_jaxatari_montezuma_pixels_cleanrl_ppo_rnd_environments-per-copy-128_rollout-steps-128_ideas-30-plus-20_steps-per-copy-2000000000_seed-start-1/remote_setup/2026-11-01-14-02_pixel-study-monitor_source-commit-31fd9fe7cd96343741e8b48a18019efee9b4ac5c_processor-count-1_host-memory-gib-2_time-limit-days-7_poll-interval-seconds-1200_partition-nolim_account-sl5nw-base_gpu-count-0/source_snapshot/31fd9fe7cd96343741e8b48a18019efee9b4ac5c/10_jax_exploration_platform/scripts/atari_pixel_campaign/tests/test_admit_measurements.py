"""Exercise source admission with standard-library fixtures and no numerical imports."""

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
SPECIFICATION = importlib.util.spec_from_file_location("pixel_admission", SCRIPTS / "admit_measurements.py")
ADMISSION = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(ADMISSION)


class AdmissionTests(unittest.TestCase):
    """Keep evidence complete and immutable without creating jobs or importing JAX."""

    def setUp(self):
        """Create the exact recorded snapshot and complete source-check metadata."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = Path(self.temporary.name)
        self.commit = "a" * 40
        self.older_commit = "b" * 40
        self.source = self.run_dir / "frozen"
        self.source.mkdir()
        self.configuration = {**ADMISSION.REFERENCE_ALGORITHM, "n_copies": 1,
                              "per_copy_scalar_settings": [], "network_layout": "nchw"}
        self.reference = self.write("reference.json", self.configuration)
        self.snapshot = self.make_snapshot(self.commit)
        checks = {}
        for context in ADMISSION.CONTEXTS:
            for kind in ("learner", "renderer"):
                checks[context, kind] = self.make_check(context, kind)
        self.arguments = argparse.Namespace(run_dir=self.run_dir, source_root=self.source,
            source_commit=self.commit, reference_configuration_source=self.reference,
            matched_copy_counts=[1], capacity_record=[], seed_start=1,
            h100_learner_record=checks["h100", "learner"],
            h100_renderer_record=checks["h100", "renderer"],
            a4500_learner_record=checks["jaguar03_rtx_a4500", "learner"],
            a4500_renderer_record=checks["jaguar03_rtx_a4500", "renderer"])

    def write(self, relative, contents):
        """Write one independent JSON evidence fixture."""
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(contents) + "\n")
        return path

    def make_snapshot(self, commit):
        """Represent an archived source tree with the three permitted check targets."""
        root = self.run_dir / "data/code_snapshots" / commit / "10_jax_exploration_platform"
        (root / "tests/atari_pixels").mkdir(parents=True)
        (root / "tests/test_atari_pixel_environment.py").touch()
        (root / "scripts/atari_pixel_campaign").mkdir(parents=True)
        (root / "scripts/atari_pixel_campaign/check_rendering.py").touch()
        (root / "scripts/atari_pixel_campaign/run_unit.py").touch()
        return root

    def hardware(self, context):
        """Record a single visible GPU with deterministic full-precision runtime metadata."""
        return {"hostname": "serval02" if context == "h100" else "jaguar03",
                "gpu_query": "GPU-test, NVIDIA H100 NVL" if context == "h100" else "GPU-test, NVIDIA RTX A4500",
                "slurm_job_id": "", "runtime_environment": {
                    "JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                    "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}}

    def make_check(self, context, kind):
        """Produce the exact full learner suite or all-candidate renderer command."""
        command = (["/venv/bin/python", "-m", "pytest", "-q", str(self.snapshot / "tests/atari_pixels"),
                    str(self.snapshot / "tests/test_atari_pixel_environment.py")] if kind == "learner"
                   else ["/venv/bin/python", str(self.snapshot / "scripts/atari_pixel_campaign/check_rendering.py"),
                         "--candidate", "all"])
        return self.write(f"checks/{context}_{kind}.json", {"source_commit": self.commit, "command": command,
            "passed": True, "exit_code": 0, "hardware": self.hardware(context)})

    def make_capacity(self, context, copies=2):
        """Record a completed full-size benchmark and its original older-source launch command."""
        old_source = self.run_dir / "data/code_snapshots" / self.older_commit / "10_jax_exploration_platform"
        if not old_source.exists():
            self.make_snapshot(self.older_commit)
        unit = f"{context}_reference_copies_{copies}"
        directory = self.run_dir / "data" / unit
        self.write(directory / "hardware.json", self.hardware(context))
        self.write(directory / "config_resolved.json", {**self.configuration, "n_copies": copies})
        self.write(directory / "arguments.json", {"mode": "benchmark", "variant": "reference", "copies": copies,
            "envs_per_copy": 128, "rollout_steps": 128, "iterations": 10, "warmup_rollouts": 50,
            "epochs": 4, "minibatches": 4, "rounds": 7, "unit_id": unit, "run_dir": str(self.run_dir),
            "configuration": None, "environment_configuration": None})
        rounds = [{"record": "benchmark", "round": index, "copies": copies, "variant": "reference",
                   "transitions": copies * 128 * 128 * 10, "seconds": 1.0} for index in range(7)]
        result = self.write(directory / "result.json", {"completed": True, "mode": "benchmark",
                            "variant": "reference", "copies": copies, "rounds": rounds})
        script = self.run_dir / f"slurm/{unit}.sh"
        script.parent.mkdir(exist_ok=True)
        script.write_text(f"/venv/bin/python {old_source}/scripts/atari_pixel_campaign/run_unit.py "
                          f"--run-dir {self.run_dir} --unit-id {unit} --mode benchmark --copies {copies}\n")
        processes = self.run_dir / "processes.json"
        records = ADMISSION.read_json(processes) if processes.exists() else []
        self.write(processes, records + [{"unit": unit, "source_commit": self.older_commit, "script": str(script)}])
        return result

    def run_admission(self):
        """Mock only Git/controller work while exercising all real evidence and publication checks."""
        with patch.object(ADMISSION, "verify_source"), patch.object(ADMISSION, "ready_specification", return_value={}), \
                patch.object(ADMISSION, "tick", return_value={}), redirect_stdout(io.StringIO()):
            ADMISSION.admit(self.arguments)

    def test_full_commands_pass_and_selectors_or_other_sources_fail(self):
        """Reject collection-only, deselected or differently sourced tests even when their result says passed."""
        path = self.arguments.h100_learner_record
        baseline = ADMISSION.read_json(path)
        ADMISSION.verified_check(path, self.commit, "h100", "learner", self.run_dir, self.source)
        alternate = self.make_snapshot("c" * 40)
        mutations = [baseline["command"] + ["--collect-only"], baseline["command"] + ["--deselect=x"],
                     [*baseline["command"][:4], str(alternate / "tests/atari_pixels"), baseline["command"][5]]]
        for command in mutations:
            with self.subTest(command=command):
                self.write(path, {**baseline, "command": command})
                with self.assertRaises(ValueError):
                    ADMISSION.verified_check(path, self.commit, "h100", "learner", self.run_dir, self.source)
        renderer = self.arguments.h100_renderer_record
        record = ADMISSION.read_json(renderer)
        record["command"][1] = str(alternate / "scripts/atari_pixel_campaign/check_rendering.py")
        self.write(renderer, record)
        with self.assertRaisesRegex(ValueError, "source commit"):
            ADMISSION.verified_check(renderer, self.commit, "h100", "renderer", self.run_dir, self.source)

    def test_failed_late_evidence_publishes_nothing(self):
        """A failed second-device renderer cannot leave first-device admission files behind."""
        path = self.arguments.a4500_renderer_record
        self.write(path, {**ADMISSION.read_json(path), "passed": False})
        with self.assertRaisesRegex(ValueError, "has not passed"):
            self.run_admission()
        self.assertFalse((self.run_dir / "data/source_checks").exists())
        self.assertFalse((self.run_dir / ADMISSION.SENTINEL_NAME).exists())

    def test_frozen_checkout_requires_unchanged_test_files(self):
        """The optional checkout path must verify tests in addition to the learner implementation."""
        target = self.source / "10_jax_exploration_platform/tests/atari_pixels"
        target.mkdir(parents=True)
        changed = subprocess.CompletedProcess([], 0, " M 10_jax_exploration_platform/tests/atari_pixels/test_loss.py\n", "")
        with patch.object(ADMISSION, "verify_source"), \
                patch.object(ADMISSION.subprocess, "run", return_value=changed) as status:
            with self.assertRaisesRegex(ValueError, "test files differ"):
                ADMISSION.source_path(str(target), "tests/atari_pixels", self.commit, self.run_dir, self.source)
        self.assertIn("10_jax_exploration_platform/tests/atari_pixels", status.call_args.args[0])

    def test_existing_collision_changes_no_files_and_identical_retry_preserves_bytes(self):
        """Refuse a changed sentinel before creating evidence; exact retries retain existing bytes."""
        sentinel = self.write(ADMISSION.SENTINEL_NAME, {"existing": "authorization"})
        contents = sentinel.read_bytes()
        with self.assertRaisesRegex(ValueError, "immutable admission"):
            self.run_admission()
        self.assertEqual(sentinel.read_bytes(), contents)
        self.assertFalse((self.run_dir / "data/source_checks").exists())
        sentinel.unlink()
        self.run_admission()
        published = list((self.run_dir / "data/source_checks").rglob("*.json")) + [sentinel]
        snapshots = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in published}
        self.run_admission()
        self.assertEqual({path: (path.read_bytes(), path.stat().st_mtime_ns) for path in published}, snapshots)
        reference = self.run_dir / "data/source_checks" / self.commit / "reference_configuration.json"
        self.write(reference, {"changed": True})
        before = {path: path.read_bytes() for path in published}
        with self.assertRaisesRegex(ValueError, "immutable admission"):
            self.run_admission()
        self.assertEqual({path: path.read_bytes() for path in published}, before)

    def test_both_capacity_records_required_and_original_source_preserved(self):
        """A requested count needs both devices; completed records keep their actual older source hash."""
        self.arguments.matched_copy_counts = [1, 2]
        self.arguments.capacity_record = [self.make_capacity("h100")]
        with self.assertRaisesRegex(ValueError, "each GPU context"):
            self.run_admission()
        self.assertFalse((self.run_dir / "data/source_checks").exists())
        self.arguments.capacity_record.append(self.make_capacity("jaguar03_rtx_a4500"))
        self.run_admission()
        gate = ADMISSION.read_json(self.run_dir / "data/source_checks" / self.commit / "h100.json")
        capacity = gate["capacity_checks"][0]
        self.assertEqual(capacity["source_commit"], self.older_commit)
        self.assertEqual(capacity["source_root"], str(self.run_dir / "data/code_snapshots" / self.older_commit))
        self.assertEqual(capacity["configuration"]["n_copies"], 2)
        self.assertEqual(capacity["result"]["path"], str(self.arguments.capacity_record[0]))
        self.assertEqual(len(capacity["configuration_record"]["sha256"]), 64)

    def test_capacity_rejects_partial_wrong_shape_and_runtime(self):
        """Seven rounds alone cannot admit a small, incomplete or differently configured measurement."""
        result_path = self.make_capacity("h100")
        directory = result_path.parent
        cases = [(result_path, "rounds", []), (result_path, "completed", False),
                 (directory / "config_resolved.json", "n_envs", 8),
                 (directory / "arguments.json", "warmup_rollouts", 1)]
        for path, key, value in cases:
            original = ADMISSION.read_json(path)
            self.write(path, {**original, key: value})
            with self.subTest(path=path.name, key=key), self.assertRaises(ValueError):
                ADMISSION.verified_capacity(result_path, self.run_dir, self.configuration, [1, 2])
            self.write(path, original)
        hardware = ADMISSION.read_json(directory / "hardware.json")
        hardware["runtime_environment"]["XLA_FLAGS"] = "--xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_exclude_nondeterministic_ops=false"
        self.write(directory / "hardware.json", hardware)
        with self.assertRaisesRegex(ValueError, "deterministic full-precision"):
            ADMISSION.verified_capacity(result_path, self.run_dir, self.configuration, [1, 2])


if __name__ == "__main__":
    unittest.main()
