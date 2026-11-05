"""Test the exact diagnostic cache boundary without importing a numerical library."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
SPECIFICATION = importlib.util.spec_from_file_location("temporary_correctness_cache_diagnosis", SCRIPTS / "diagnose_correctness_cache.py")
DIAGNOSIS = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(DIAGNOSIS)


class CorrectnessCacheDiagnosisTests(unittest.TestCase):
    """Require successful original host comparisons before the sole diagnostic intervention."""

    def test_original_checks_finish_before_memory_and_cache_intervention(self):
        """The exact original reports reach subsequent whole-iteration checks unchanged."""
        events, reports = [], {name: {"leaves": 12} for name in ("preparation", "fixed_batch", "fixed_batch_update")}
        def original(reference, candidate, state):
            """Represent all original comparisons returning their actual report object."""
            events.append("fixed input host comparisons completed")
            return reports
        def snapshot(reference, candidate):
            """Distinguish the actual observations around the intervention."""
            events.append("memory snapshot")
            return {"memory_observation": len(events)}
        paired = types.SimpleNamespace(check_fixed_inputs=original)
        runtime = types.SimpleNamespace(clear_caches=lambda: events.append("clear JAX caches"))
        published = []
        with patch.object(DIAGNOSIS.gc, "collect", side_effect=lambda: events.append("collect unreachable Python objects") or 2):
            with DIAGNOSIS.release_after_fixed_inputs(paired, runtime, snapshot, published.append) as calls:
                returned = paired.check_fixed_inputs("reference", "candidate", "host state")
                self.assertIs(returned, reports)
                events.append("original whole iteration checks begin")
            self.assertIs(paired.check_fixed_inputs, original)
        self.assertEqual(events, ["fixed input host comparisons completed", "memory snapshot", "clear JAX caches",
                                  "collect unreachable Python objects", "memory snapshot", "original whole iteration checks begin"])
        self.assertEqual(calls, published)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["fixed_input_comparisons_completed"])
        self.assertFalse(calls[0]["allocator_pool_release_assumed"])

    def test_original_numerical_rejection_is_not_cleared_or_weakened(self):
        """A failed original comparison propagates before snapshots, cache clearing or continuation."""
        rejection = ValueError("original fixed-input mismatch")
        original = Mock(side_effect=rejection)
        paired = types.SimpleNamespace(check_fixed_inputs=original)
        runtime = types.SimpleNamespace(clear_caches=Mock())
        snapshot, publish = Mock(), Mock()
        with self.assertRaisesRegex(ValueError, "original fixed-input mismatch"):
            with DIAGNOSIS.release_after_fixed_inputs(paired, runtime, snapshot, publish) as calls:
                paired.check_fixed_inputs(None, None, None)
        self.assertIs(paired.check_fixed_inputs, original)
        self.assertEqual(calls, [])
        runtime.clear_caches.assert_not_called()
        snapshot.assert_not_called()
        publish.assert_not_called()

    def test_original_binding_restored_after_later_failure(self):
        """An OOM after cache release cannot leave the frozen module patched in the diagnostic process."""
        original = Mock(return_value={"preparation": {"leaves": 1}})
        paired = types.SimpleNamespace(check_fixed_inputs=original)
        runtime = types.SimpleNamespace(clear_caches=Mock())
        with patch.object(DIAGNOSIS.gc, "collect", return_value=0):
            with self.assertRaisesRegex(RuntimeError, "later whole iteration OOM"):
                with DIAGNOSIS.release_after_fixed_inputs(paired, runtime, Mock(return_value={}), Mock()):
                    paired.check_fixed_inputs(None, None, None)
                    raise RuntimeError("later whole iteration OOM")
        self.assertIs(paired.check_fixed_inputs, original)
        runtime.clear_caches.assert_called_once()

    def setUp(self):
        """Create one actual-shape case fixture and a complete reference configuration."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = SCRIPTS.parents[2]
        defaults = DIAGNOSIS.diagnostic.default_configuration(self.source)
        reference = {name: defaults[name] for name in ("learner", "environment")}
        reference["learner"]["n_copies"] = 2
        reference["execution"] = {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
                                   "report_readout": "none", "report_every_iterations": 10}
        reference = json.loads(json.dumps(reference))
        case = {"hypothesis": "batch_rollout_intrinsic_rewards", "source_root": str(self.source),
            "source_commit": DIAGNOSIS.SOURCE_COMMIT, "hardware_class": "jaguar03_rtx_a4500", "seed_start": 1,
            "reference_environment": None, "reference_execution": None, "configuration": reference["learner"],
            "candidate": {"learner": {"hoist_rollout_bonus": True}, "environment": {}, "execution": {}},
            "rounds": 7, "iterations": 1}
        documents = {"case": case, "reference": {"case_completed": True, "source_commit": DIAGNOSIS.SOURCE_COMMIT,
            "hardware_class": "jaguar03_rtx_a4500", "copies": 2, "reference": reference},
            "runtime": {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                        "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"},
            "assets": {"root": "/assets", "manifest": "/assets_manifest.json", "sha256": "a" * 64}}
        self.files = {}
        for name, document in documents.items():
            self.files[name] = self.directory / (name + ".json")
            self.files[name].write_text(json.dumps(document) + "\n")

    def arguments(self):
        """Use metadata preparation with no numerical-runtime overrides."""
        return DIAGNOSIS.parse_arguments(["--prepare-only", "--source-root", str(self.source), "--run-dir", str(self.directory),
            "--case", str(self.files["case"]), "--reference-record", str(self.files["reference"]),
            "--runtime-environment", str(self.files["runtime"]), "--asset-record", str(self.files["assets"]),
            "--unit-id", "temporary_correctness_cache_diagnosis_copies-2_seed-start-1", "--gpu-uuid", "GPU-fixture"])

    def prepare(self):
        """Mock only immutable Git receipts, retaining all configuration/protocol/staging checks."""
        helpers = {"source_commit": "b" * 40, "files": {
            "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name:
                hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest() for name in DIAGNOSIS.HELPERS}}
        with patch.object(DIAGNOSIS.diagnostic, "core_identity", return_value={"source_commit": DIAGNOSIS.SOURCE_COMMIT}), \
                patch.object(DIAGNOSIS.training, "source_receipt", return_value={"files": {}}), \
                patch.object(DIAGNOSIS, "helper_receipt", return_value=helpers), \
                patch.object(DIAGNOSIS.measurement, "preflight") as gpu, patch.object(DIAGNOSIS, "compute") as compute:
            directory = DIAGNOSIS.prepare(self.arguments())
            gpu.assert_not_called()
            compute.assert_not_called()
        return directory

    def test_staged_diagnosis_preserves_original_bytes_and_one_update_rounds(self):
        """The diagnostic retains seven original one-update rounds rather than silently switching to packing windows."""
        original = self.files["case"].read_bytes()
        directory = self.prepare()
        plan = json.loads((directory / "measurement_plan.json").read_text())
        self.assertEqual((plan["rounds"], plan["iterations"], plan["latency_calls"]), (7, 1, 3))
        self.assertEqual((directory / "inputs/original_case.json").read_bytes(), original)
        self.assertEqual(plan["original_case_bytes_sha256"], hashlib.sha256(original).hexdigest())
        self.assertFalse(plan["eligible_for_formal_campaign_admission"])
        self.assertEqual(plan["allocator"], {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85"})
        for name in ("manifest.yaml", "config_resolved.yaml", "experiment_background.md", "command.txt", "measurement_prepared.json"):
            self.assertTrue((directory / name).is_file())
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_changed_original_candidate_and_mutable_execution_are_rejected(self):
        """No alternate learner change or working-tree command can run as this diagnostic."""
        directory = self.prepare()
        with patch.object(DIAGNOSIS, "compute") as compute:
            with self.assertRaisesRegex(ValueError, "unchanged staged diagnostic"):
                DIAGNOSIS.run_prepared(directory)
            compute.assert_not_called()
        case = json.loads(self.files["case"].read_text())
        case["candidate"]["learner"]["learning_rate"] = .01
        self.files["case"].write_text(json.dumps(case))
        with self.assertRaisesRegex(ValueError, "original D28"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
