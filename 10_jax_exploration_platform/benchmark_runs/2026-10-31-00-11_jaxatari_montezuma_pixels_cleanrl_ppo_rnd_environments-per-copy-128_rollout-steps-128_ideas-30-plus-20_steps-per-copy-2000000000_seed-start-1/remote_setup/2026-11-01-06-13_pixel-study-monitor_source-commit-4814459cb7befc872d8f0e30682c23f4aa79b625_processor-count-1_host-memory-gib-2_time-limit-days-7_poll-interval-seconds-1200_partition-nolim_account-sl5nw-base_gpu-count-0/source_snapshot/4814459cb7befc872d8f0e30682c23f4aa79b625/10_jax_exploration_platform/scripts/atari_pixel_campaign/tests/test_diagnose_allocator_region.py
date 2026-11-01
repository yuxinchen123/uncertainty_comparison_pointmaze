"""Check the explicit allocator-region diagnostic without allocating any real arrays."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import test_diagnose_correctness_cache as input_fixtures

SPECIFICATION = importlib.util.spec_from_file_location("allocator_region_history_diagnosis", SCRIPTS / "diagnose_allocator_region.py")
DIAGNOSIS = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(DIAGNOSIS)


class AllocatorRegionTests(unittest.TestCase):
    """Keep the diagnostic's one temporary array separate from every learner/game operation."""

    def setUp(self):
        """Reuse only the complete original-case metadata fixture, not its old diagnostic preparation."""
        self.fixture = input_fixtures.CorrectnessCacheDiagnosisTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def arguments(self):
        """Name a fresh independent allocator-region unit with explicit original inputs."""
        fixture = self.fixture
        return DIAGNOSIS.parse_arguments(["--prepare-only", "--source-root", str(fixture.source),
            "--run-dir", str(fixture.directory), "--case", str(fixture.files["case"]),
            "--reference-record", str(fixture.files["reference"]), "--runtime-environment", str(fixture.files["runtime"]),
            "--asset-record", str(fixture.files["assets"]),
            "--unit-id", "allocator_region_before_initialization_bytes-9663676416_copies-2_seed-start-1",
            "--gpu-uuid", "GPU-fixture"])

    def prepare(self):
        """Mock only committed Git receipts; all real metadata/byte preservation checks remain active."""
        helpers = {"source_commit": "b" * 40, "files": {
            "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name:
                hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest() for name in DIAGNOSIS.HELPERS}}
        with patch.object(DIAGNOSIS.diagnostic, "core_identity", return_value={"source_commit": DIAGNOSIS.SOURCE_COMMIT}), \
                patch.object(DIAGNOSIS.training, "source_receipt", return_value={"files": {}}), \
                patch.object(DIAGNOSIS, "helper_receipt", return_value=helpers), \
                patch.object(DIAGNOSIS.measurement, "preflight") as gpu, patch.object(DIAGNOSIS, "compute") as numeric:
            directory = DIAGNOSIS.prepare(self.arguments())
            gpu.assert_not_called()
            numeric.assert_not_called()
        return directory

    def test_one_exact_byte_array_is_synchronized_and_deleted_before_learner_work(self):
        """The original learner cannot see or use the separate nine-GiB preparation array."""
        events = []
        region = types.SimpleNamespace(nbytes=9 * 2 ** 30, dtype="uint8", delete=lambda: events.append("delete array"))
        arrays = types.SimpleNamespace(uint8="uint8", zeros=Mock(side_effect=lambda shape, dtype: events.append("allocate array") or region))
        runtime = types.SimpleNamespace(block_until_ready=Mock(side_effect=lambda value: events.append("synchronize array")), clear_caches=Mock())
        def snapshot():
            """Return distinguishable real-memory observation positions without device operations."""
            events.append("memory snapshot")
            return {"position": len(events)}
        published = []
        with patch.object(DIAGNOSIS.gc, "collect", side_effect=lambda: events.append("collect unreachable objects") or 0):
            result = DIAGNOSIS.prepare_region(runtime, arrays, snapshot, published.append)
            events.append("original learner initialization")
        arrays.zeros.assert_called_once_with((9, 1024, 1024, 1024), dtype="uint8")
        runtime.block_until_ready.assert_called_once_with(region)
        runtime.clear_caches.assert_not_called()
        self.assertEqual(events, ["memory snapshot", "allocate array", "synchronize array", "memory snapshot",
                                  "delete array", "collect unreachable objects", "memory snapshot", "original learner initialization"])
        self.assertEqual(result["array_nbytes"], 9663676416)
        self.assertEqual(result["allocation_count"], 1)
        self.assertFalse(result["allocator_pool_release_assumed"])
        self.assertEqual(published, [result])

    def test_wrong_array_size_is_deleted_and_not_published(self):
        """A failed memory preparation cannot be silently treated as the requested nine-GiB intervention."""
        region = types.SimpleNamespace(nbytes=8 * 2 ** 30, dtype="uint8", delete=Mock())
        runtime = types.SimpleNamespace(block_until_ready=Mock())
        arrays = types.SimpleNamespace(uint8="uint8", zeros=Mock(return_value=region))
        publish = Mock()
        with self.assertRaisesRegex(ValueError, "exact nine-GiB"):
            DIAGNOSIS.prepare_region(runtime, arrays, Mock(return_value={}), publish)
        region.delete.assert_called_once()
        publish.assert_not_called()

    def test_staging_keeps_case_protocol_bytes_and_allocator_unchanged(self):
        """A distinct helper/unit records the one memory intervention while preserving all original inputs."""
        original = self.fixture.files["case"].read_bytes()
        directory = self.prepare()
        plan = json.loads((directory / "measurement_plan.json").read_text())
        self.assertEqual(plan["kind"], "allocator_region_history_diagnosis")
        self.assertFalse(plan["eligible_for_formal_campaign_admission"])
        self.assertEqual((plan["rounds"], plan["iterations"], plan["latency_calls"]), (7, 1, 3))
        self.assertEqual(plan["intervention"]["bytes"], 9 * 2 ** 30)
        self.assertEqual(plan["allocator"], {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85"})
        self.assertEqual((directory / "inputs/original_case.json").read_bytes(), original)
        self.assertEqual(plan["original_case_bytes_sha256"], hashlib.sha256(original).hexdigest())
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_original_benchmark_follows_region_preparation_without_changing_checks(self):
        """The numerical entrypoint calls the frozen original benchmark with its original one-update protocol."""
        directory = self.prepare()
        plan = json.loads((directory / "measurement_plan.json").read_text())
        events = []
        runtime = types.ModuleType("jax")
        arrays = types.ModuleType("jax.numpy")
        runtime.numpy = arrays
        runtime.config = types.SimpleNamespace(update=Mock())
        runtime.devices = lambda: [types.SimpleNamespace(platform="gpu", device_kind="NVIDIA RTX A4500", client=None)]
        runtime.clear_caches = Mock()
        config_module = types.ModuleType("exploration_platform.atari_pixels.config")
        config_module.PixelPPOConfig = Mock(side_effect=lambda **values: events.append("construct learner config") or values)
        original_check = Mock()
        paired = types.SimpleNamespace(check_fixed_inputs=original_check,
            run_paired_benchmark=Mock(side_effect=lambda *args, **kwargs: events.append("original paired benchmark") or {"original": True}))
        region = {"array_nbytes": DIAGNOSIS.REGION_BYTES}
        with patch.dict(sys.modules, {"jax": runtime, "jax.numpy": arrays, "jaxtari": types.ModuleType("jaxtari"),
                "exploration_platform.atari_pixels.config": config_module}), \
                patch.object(DIAGNOSIS, "prepare_region", side_effect=lambda *args: events.append("complete region preparation") or region), \
                patch.object(DIAGNOSIS.measurement, "MemoryObservation"), \
                patch.object(DIAGNOSIS.measurement, "verify_imported_sources"), \
                patch.object(DIAGNOSIS.importlib, "import_module", return_value=paired), \
                patch.object(DIAGNOSIS.training, "actual_gpu_library_receipt", return_value={}):
            DIAGNOSIS.compute(directory, plan, {"gpu_uuid": "GPU-fixture"}, {})
        self.assertEqual(events, ["complete region preparation", "construct learner config", "original paired benchmark"])
        self.assertIs(paired.check_fixed_inputs, original_check)
        runtime.clear_caches.assert_not_called()
        call = paired.run_paired_benchmark.call_args
        self.assertEqual((call.kwargs["rounds"], call.kwargs["iterations"], call.kwargs["seed_start"], call.kwargs["latency_calls"]), (7, 1, 1, 3))
        self.assertEqual(call.args[1], {"hoist_rollout_bonus": True})
        self.assertNotIn("jax", sys.modules)

    def test_mutable_working_entrypoint_cannot_run_prepared_region(self):
        """A staged diagnostic cannot be rerouted through changing integration code."""
        directory = self.prepare()
        with patch.object(DIAGNOSIS, "compute") as numeric:
            with self.assertRaisesRegex(ValueError, "unchanged staged allocator-region"):
                DIAGNOSIS.run_prepared(directory)
            numeric.assert_not_called()


if __name__ == "__main__":
    unittest.main()
