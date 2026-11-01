"""Real GPU component checks for the future trainer; never part of metadata discovery.

Run only in an allocated or authorized compute context, after freezing helpers:
RND_RUN_GPU_COMPONENT_TESTS=1 <pixel-python> -m pytest -v <this-file>

Archive launches also set RND_COMPONENT_SOURCE_COMMIT to the full commit; the
snapshot directory itself must have that exact name. Without this variable,
the fixture accepts only a real Git checkout rooted exactly at PROJECT_ROOT.

The tiny shapes and short budgets below are component fixtures. They cannot
pass final-training admission and are not performance measurements. The checks
keep exact game/integer/pixel comparison and the existing strict FP32/FP64
tolerances; they do not change the frozen learner or its correctness rules.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock


if os.environ.get("RND_RUN_GPU_COMPONENT_TESTS") != "1":
    raise unittest.SkipTest("GPU component checks require an explicit compute-only invocation")

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPT_ROOT.parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "10_jax_exploration_platform/src"))
import train_optimized as training

training.training_placement(os.environ, socket.gethostname())
COMPONENT_SOURCE_COMMIT = training.component_source_commit(PROJECT_ROOT, os.environ)
training.enforce_runtime({"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                          "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}, os.environ)
import jax
import jax.numpy as jnp
import numpy as np
import paired_benchmark as paired
from exploration_platform.atari_pixels.config import PixelPPOConfig
from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
from exploration_platform.atari_pixels.training import PixelPPORND

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class TrainingGPUComponentTests(unittest.TestCase):
    """Compare real continuation with the existing complete-update Executor."""

    @classmethod
    def setUpClass(cls):
        """Create a tiny real game/learner fixture on one visible GPU."""
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise RuntimeError("the component fixture needs exactly one visible GPU")
        # Seven two-transition updates make a nondivisible tail for groups of three.
        cls.config = PixelPPOConfig(n_copies=1, n_envs=1, num_steps=2, total_timesteps=14,
                                    update_epochs=1, num_minibatches=1,
                                    observation_warmup_iterations=0, donate_state=True)
        cls.environment = AtariPixelEnvironment(num_envs=1, cache_room_backgrounds=True)
        cls.trainer = PixelPPORND(cls.config, cls.environment)
        cls.initial_host = paired.host_copy(cls.trainer.warmup(cls.trainer.initialize((11,))))
        cls.source_commit = COMPONENT_SOURCE_COMMIT
        cls.jaxtari_source = training.jaxtari_python_receipt()
        cls.installed_gpu_libraries = training.installed_gpu_library_receipts()

    def identity(self, options):
        """Identify tiny component state without weakening the production admission boundary."""
        specification = {"source_root": str(PROJECT_ROOT), "source_commit": self.source_commit,
                         "hardware_class": "component_fixture", "copy_seeds": [11],
                         "configuration": {"learner": asdict(self.config), "environment": {"num_envs": 1},
                                           "execution": options}}
        return training.checkpoint_identity(specification, {"source_commit": self.source_commit},
            {name: os.environ[name] for name in training.REQUIRED_RUNTIME},
            {"device_kind": jax.devices()[0].device_kind}, {"fixture": "existing staged pixel assets"})

    def execution(self, options, report=None, trainer=None):
        """Use the same frozen Executor for uninterrupted and resumed component calls."""
        return training.TrainingExecution(paired.Executor(self.trainer if trainer is None else trainer, options),
                                          paired, jax, np, jnp, report)

    def restarted_trainer(self):
        """Rebuild static objects so continuation cannot depend on the preceding trainer instance."""
        environment = AtariPixelEnvironment(num_envs=1, cache_room_backgrounds=True)
        return PixelPPORND(self.config, environment)

    def assert_state_agreement(self, expected, actual, executor):
        """Require exact discrete/game/pixel values and existing strict floating comparisons."""
        expected = paired.canonical_training_state(executor, expected)
        actual = paired.canonical_training_state(executor, actual)
        checked = paired.agreement(expected, actual, exact_environment=True)
        self.assertGreater(checked["leaves"], 0)
        self.assertEqual(int(jax.device_get(actual.iteration)), self.config.total_iterations)

    def test_group_checkpoint_resume_and_unpadded_tail_match_original_updates(self):
        """A saved group resumes with both full-group/tail and single-update/full-group orders."""
        options = {**paired.DEFAULT_EXECUTION, "iterations_per_call": 3}
        identity = self.identity(options)
        # The reference performs seven original updates with their real annealing positions.
        uninterrupted = paired.device_copy(self.initial_host)
        reference_metrics = None
        for _ in range(self.config.total_iterations):
            uninterrupted, reference_metrics = self.trainer.train_iteration(uninterrupted)
        jax.block_until_ready(uninterrupted)

        state = paired.device_copy(self.initial_host)
        execution = self.execution(options)
        state, _, _ = execution.advance(state, 3, 0)
        jax.block_until_ready(state)
        self.assertEqual(int(jax.device_get(state.iteration)), 3)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "latest.pkl"
            training.save_checkpoint(checkpoint, jax.device_get(state), jax.device_get(execution.report),
                                     identity, 3, 1.0)
            restored = training.load_checkpoint(checkpoint, identity)
            # Each order receives independent restored buffers and static objects;
            # [1, 3] exercises an original single update before the next full group.
            for order in ((3, 1), (1, 3)):
                with self.subTest(remaining_call_order=order):
                    state = paired.device_copy(restored["state"])
                    execution = self.execution(options, dict(restored["report_state"]), self.restarted_trainer())
                    calls = []
                    iteration = int(jax.device_get(state.iteration))
                    for count in order:
                        state, metrics, _ = execution.advance(state, count, iteration)
                        jax.block_until_ready(state)
                        actual = int(jax.device_get(state.iteration))
                        self.assertEqual(actual, iteration + count)
                        calls.append(count)
                        iteration = actual
                    self.assertEqual(calls, list(order))
                    self.assertEqual(iteration, self.config.total_iterations)
                    training.save_checkpoint(checkpoint, jax.device_get(state), jax.device_get(execution.report),
                                             identity, iteration, 2.0)
                    self.assertEqual(training.load_checkpoint(checkpoint, identity)["iteration"], 7)
                    self.assert_state_agreement(uninterrupted, state, execution.executor)
                    paired.agreement(jax.device_get(reference_metrics), jax.device_get(metrics))
        receipt = training.actual_gpu_library_receipt(jax.devices()[0].client, self.installed_gpu_libraries)
        training.validate_gpu_library_receipt(receipt, {"installed_gpu_libraries": self.installed_gpu_libraries})
        self.assertEqual(self.jaxtari_source["source_commit"], training.JAXTARI_COMMIT)

    def test_scheduled_pending_report_resume_matches_uninterrupted_executor(self):
        """Recover interrupted update-three publication and preserve the next four-update report row."""
        options = {**paired.DEFAULT_EXECUTION, "report_readout": "scheduled", "report_every_iterations": 4}
        identity = self.identity(options)
        original = paired.Executor(self.trainer, options)
        reference, reference_metrics = original.run(paired.device_copy(self.initial_host), 7, initial_iteration=0)
        jax.block_until_ready(reference)
        reference_reports = original.last_reports

        # An uninterrupted wrapper retains the final incomplete reporting window as well.
        continuous = self.execution(options)
        continuous_state = paired.device_copy(self.initial_host)
        continuous_reports = []
        for iteration in range(7):
            continuous_state, _, rows = continuous.advance(continuous_state, 1, iteration)
            continuous_reports.extend(rows)
        jax.block_until_ready(continuous_state)

        interrupted = self.execution(options)
        state = paired.device_copy(self.initial_host)
        for iteration in range(2):
            state, _, rows = interrupted.advance(state, 1, iteration)
            self.assertEqual(rows, [])
        jax.block_until_ready(state)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "latest.pkl"
            training.save_checkpoint(checkpoint, jax.device_get(state), jax.device_get(interrupted.report),
                                     identity, 2, 1.0)
            old_payload_sha256 = training.file_sha256(checkpoint)
            old_metadata = checkpoint.with_name("latest.metadata.json").read_bytes()
            state, _, rows = interrupted.advance(state, 1, 2)
            self.assertEqual(rows, [])
            jax.block_until_ready(state)
            original_sync = training.checkpoint_sync_directory

            def interrupt_after_payload(paths):
                """Stop after the new payload is durable while the previous sidecar remains canonical."""
                original_sync(paths)
                marker = json.loads(paths["marker"].read_text()) if paths["marker"].exists() else None
                if (marker is not None and marker["phase"] == "prepared"
                        and training.file_sha256(paths["payload"]) == marker["next"]["checkpoint_sha256"]
                        and paths["metadata"].read_bytes() == old_metadata):
                    raise InterruptedError("component checkpoint publication interruption")

            with mock.patch.object(training, "checkpoint_sync_directory", side_effect=interrupt_after_payload):
                with self.assertRaisesRegex(InterruptedError, "component checkpoint"):
                    training.save_checkpoint(checkpoint, jax.device_get(state), jax.device_get(interrupted.report),
                                             identity, 3, 1.5)
            self.assertEqual(checkpoint.with_name("latest.metadata.json").read_bytes(), old_metadata)
            self.assertEqual(training.file_sha256(checkpoint.with_name("latest.previous.pkl")), old_payload_sha256)
            with mock.patch.object(training.pickle, "load", side_effect=AssertionError("metadata recovery cannot unpickle")):
                recovered = training.recover_checkpoint(checkpoint, identity)
            self.assertEqual(recovered["action"], "finished_prepared_generation")
            restored = training.load_checkpoint(checkpoint, identity)
            self.assertEqual(restored["iteration"], 3)
            self.assertEqual(restored["report_state"]["pending_count"], 3)
            restored["report_state"]["pending_sums"] = paired.device_copy(restored["report_state"]["pending_sums"])
            resumed = self.execution(options, restored["report_state"], self.restarted_trainer())
            state = paired.device_copy(restored["state"])
            reports = []
            for iteration in range(3, 7):
                state, metrics, rows = resumed.advance(state, 1, iteration)
                reports.extend(rows)
            jax.block_until_ready(state)
            self.assertEqual(resumed.report["pending_count"], 3)
            self.assertEqual(resumed.report["last_report_iteration"], 4)
            training.save_checkpoint(checkpoint, jax.device_get(state), jax.device_get(resumed.report), identity, 7, 2.0)
            self.assertEqual(training.load_checkpoint(checkpoint, identity)["report_state"]["pending_count"], 3)
        self.assertEqual([row["iteration"] for row in reports], [4])
        self.assertEqual([row["window_updates"] for row in reports], [4])
        self.assertEqual(len(reference_reports), len(reports))
        self.assertEqual(len(continuous_reports), len(reports))
        paired.agreement(reference_reports[0]["metrics"], reports[0]["metrics"])
        paired.agreement(continuous_reports[0]["metrics"], reports[0]["metrics"])
        paired.agreement(jax.device_get(continuous.report["pending_sums"]), jax.device_get(resumed.report["pending_sums"]))
        paired.agreement(jax.device_get(reference_metrics), jax.device_get(metrics))
        self.assert_state_agreement(reference, state, resumed.executor)
        self.assert_state_agreement(continuous_state, state, resumed.executor)


if __name__ == "__main__":
    unittest.main()
